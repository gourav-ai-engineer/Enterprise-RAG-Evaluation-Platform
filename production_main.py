from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from pathlib import Path

from fastapi import File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pypdf import PdfReader

import gemini_main as base
import main as core

app = base.app
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", APP_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "documents"
CACHE_DIR = DATA_DIR / "cache"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "25"))
logger = logging.getLogger(__name__)

app.routes[:] = [
    route for route in app.routes
    if getattr(route, "path", None) not in {"/", "/api/documents/upload"}
]


def ensure_schema() -> None:
    conn = core.db()
    columns = {row[1] for row in conn.execute("PRAGMA table_info(documents)").fetchall()}
    if "sha256" not in columns:
        conn.execute("ALTER TABLE documents ADD COLUMN sha256 TEXT")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_documents_sha256 ON documents(sha256) WHERE sha256 IS NOT NULL")
    conn.commit()
    conn.close()


def document_row(document_id: str) -> dict | None:
    conn = core.db()
    row = conn.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def find_by_sha(sha256: str) -> dict | None:
    conn = core.db()
    row = conn.execute("SELECT * FROM documents WHERE sha256=?", (sha256,)).fetchone()
    conn.close()
    return dict(row) if row else None


def cache_path(sha256: str) -> Path:
    return CACHE_DIR / f"{sha256}.json"


def cached_chunks(sha256: str) -> list[tuple[int, int, str]] | None:
    path = cache_path(sha256)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("sha256") != sha256:
            return None
        if payload.get("chunker_version") != core.CHUNKER_VERSION:
            return None
        chunks = payload.get("chunks")
        if not isinstance(chunks, list):
            return None
        return [(int(x[0]), int(x[1]), str(x[2])) for x in chunks]
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return None


def save_chunk_cache(sha256: str, filename: str, pages: int, chunks: list[tuple[int, int, str]]) -> None:
    payload = {
        "sha256": sha256,
        "chunker_version": core.CHUNKER_VERSION,
        "filename": filename,
        "pages": pages,
        "chunks": [[start, end, text] for start, end, text in chunks],
    }
    cache_path(sha256).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


@app.on_event("startup")
def startup() -> None:
    ensure_schema()


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    filename = Path(file.filename or "document.pdf").name
    if Path(filename).suffix.lower() != ".pdf" and file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF files are supported.")
    content = await file.read()
    if not content:
        raise HTTPException(400, "The uploaded PDF is empty.")
    if len(content) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"PDF exceeds {MAX_FILE_MB} MB limit.")
    if not content.startswith(b"%PDF"):
        raise HTTPException(400, "The uploaded file is not a valid PDF.")

    sha256 = hashlib.sha256(content).hexdigest()
    existing = find_by_sha(sha256)
    if existing:
        return {
            "id": existing["id"], "document_id": existing["id"], "filename": existing["filename"],
            "pages": existing["pages"], "chunks": existing["chunks"], "sha256": sha256,
            "duplicate": True, "cache_hit": True,
            "message": "This PDF is already indexed; the existing document was reused.",
        }

    stored = UPLOAD_DIR / f"{sha256}.pdf"
    if not stored.exists():
        stored.write_bytes(content)

    chunks = cached_chunks(sha256)
    cache_hit = chunks is not None
    if chunks is None:
        try:
            reader = PdfReader(str(stored))
            pages = [core.clean_page_text(page.extract_text() or "") for page in reader.pages]
            chunks = core.chunk_pages(pages)
        except Exception as exc:
            stored.unlink(missing_ok=True)
            raise HTTPException(400, f"Could not read PDF: {exc}") from exc
        save_chunk_cache(sha256, filename, len(pages), chunks)
    else:
        pages = [""] * max((end for _, end, _ in chunks), default=0)

    document_id = uuid.uuid4().hex
    chunk_rows = [
        {
            "id": uuid.uuid4().hex,
            "document_id": document_id,
            "filename": filename,
            "page_start": start,
            "page_end": end,
            "text": text,
        }
        for start, end, text in chunks
    ]
    conn = core.db()
    try:
        conn.execute(
            "INSERT INTO documents(id,filename,stored_path,pages,chunks,sha256) VALUES(?,?,?,?,?,?)",
            (document_id, filename, str(stored), len(pages), len(chunk_rows), sha256),
        )
        conn.executemany(
            "INSERT INTO chunks(id,document_id,filename,page_start,page_end,text) VALUES(?,?,?,?,?,?)",
            [(
                item["id"], item["document_id"], item["filename"],
                item["page_start"], item["page_end"], item["text"],
            ) for item in chunk_rows],
        )
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise
    conn.close()

    # Warm/index dense vectors during ingestion so the first user query does not
    # have to embed every passage. A transient model-download failure is exposed
    # in the upload response; retrieval will retry indexing before answering.
    dense_indexed = True
    dense_index_error = None
    try:
        core.index_dense_embeddings(chunk_rows)
    except Exception:
        dense_indexed = False
        dense_index_error = (
            "Dense indexing could not complete during upload; the first query will retry. "
            "Check model download access and the persistent FastEmbed cache."
        )
        logger.exception("Dense embedding indexing failed for uploaded document %s", document_id)

    _retrieve_cache.clear()
    result = {
        "id": document_id, "document_id": document_id, "filename": filename,
        "pages": len(pages), "chunks": len(chunk_rows), "sha256": sha256,
        "duplicate": False, "cache_hit": cache_hit, "dense_indexed": dense_indexed,
        "message": "PDF indexed successfully.",
    }
    if dense_index_error:
        result["dense_index_error"] = dense_index_error
    return result


_retrieve_cache: dict[tuple, list[dict]] = {}
_retrieve_cache_hits = 0
_retrieve_cache_misses = 0
_ORIGINAL_HYBRID_RETRIEVE = core.hybrid_retrieve


def cached_hybrid_retrieve(
    query: str,
    rows,
    k: int,
    diversify_documents: bool = False,
):
    global _retrieve_cache_hits, _retrieve_cache_misses
    chunk_ids = tuple(row["id"] for row in rows)
    # The flag changes the returned candidates, so it must be part of the cache key.
    key = (
        query.strip().lower(), chunk_ids, int(k), bool(diversify_documents),
        core.dense_retrieval.model_name(),
        os.getenv("HYBRID_BM25_WEIGHT", "0.35"),
        os.getenv("HYBRID_TFIDF_WEIGHT", "0.15"),
        os.getenv("HYBRID_DENSE_WEIGHT", "0.50"),
        os.getenv("HYBRID_RRF_K", "60"),
    )
    hit = _retrieve_cache.get(key)
    if hit is not None:
        _retrieve_cache_hits += 1
        core._last_retrieval_cache_hit = True
        return [dict(item) for item in hit]
    _retrieve_cache_misses += 1
    core._last_retrieval_cache_hit = False
    result = _ORIGINAL_HYBRID_RETRIEVE(
        query, rows, k, diversify_documents=diversify_documents
    )
    _retrieve_cache[key] = [dict(item) for item in result]
    return result


core._last_retrieval_cache_hit = False
core.hybrid_retrieve = cached_hybrid_retrieve


@base.app.get("/")
async def root() -> HTMLResponse:
    html = base.HTML
    old = "docs.unshift(d);selected.add(d.document_id);renderDocs();$('provider').textContent='Indexed · '+d.filename;$('file').value=''"
    new = "if(d.duplicate){const idx=docs.findIndex(x=>x.id===d.id);if(idx>=0)docs.splice(idx,1);docs.unshift(d);selected.add(d.document_id);renderDocs();$('provider').textContent='Already indexed · '+d.filename}else{docs.unshift(d);selected.add(d.document_id);renderDocs();$('provider').textContent='Indexed · '+d.filename}$('file').value=''"
    if old in html:
        html = html.replace(old, new)
    return HTMLResponse(html)


@app.get("/api/cache/stats")
def cache_stats() -> dict:
    files = list(CACHE_DIR.glob("*.json"))
    model_name = core.dense_retrieval.model_name()
    conn = core.db()
    try:
        total_chunks = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        dense_chunks = int(conn.execute(
            "SELECT COUNT(DISTINCT chunk_id) FROM chunk_embeddings WHERE model_name=?",
            (model_name,),
        ).fetchone()[0])
    finally:
        conn.close()
    return {
        "retrieval_cache_entries": len(_retrieve_cache),
        "retrieval_cache_hits": _retrieve_cache_hits,
        "retrieval_cache_misses": _retrieve_cache_misses,
        "ingestion_cache_entries": len(files),
        "cache_directory": str(CACHE_DIR),
        "dense_embedding_model": model_name,
        "dense_indexed_chunks": dense_chunks,
        "chunks_missing_dense_embeddings": max(0, total_chunks - dense_chunks),
    }


ensure_schema()
