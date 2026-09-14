from __future__ import annotations

import hashlib
import json
import os
import shutil
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

# Remove the original upload and root routes. The Gemini query route is retained.
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
        chunks = payload.get("chunks")
        if not isinstance(chunks, list):
            return None
        return [(int(x[0]), int(x[1]), str(x[2])) for x in chunks]
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return None


def save_chunk_cache(sha256: str, filename: str, pages: int, chunks: list[tuple[int, int, str]]) -> None:
    payload = {
        "sha256": sha256,
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
            "id": existing["id"],
            "document_id": existing["id"],
            "filename": existing["filename"],
            "pages": existing["pages"],
            "chunks": existing["chunks"],
            "sha256": sha256,
            "duplicate": True,
            "cache_hit": True,
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
            pages = [core.clean_text(page.extract_text() or "") for page in reader.pages]
            chunks = core.chunk_pages(pages)
        except Exception as exc:
            stored.unlink(missing_ok=True)
            raise HTTPException(400, f"Could not read PDF: {exc}") from exc
        save_chunk_cache(sha256, filename, len(pages), chunks)
    else:
        pages = [""] * max((end for _, end, _ in chunks), default=0)

    document_id = uuid.uuid4().hex
    conn = core.db()
    try:
        conn.execute(
            "INSERT INTO documents(id,filename,stored_path,pages,chunks,sha256) VALUES(?,?,?,?,?,?)",
            (document_id, filename, str(stored), len(pages), len(chunks), sha256),
        )
        conn.executemany(
            "INSERT INTO chunks(id,document_id,filename,page_start,page_end,text) VALUES(?,?,?,?,?,?)",
            [(uuid.uuid4().hex, document_id, filename, start, end, text) for start, end, text in chunks],
        )
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise
    conn.close()

    # New indexed material invalidates retrieval-cache entries.
    _retrieve_cache.clear()

    return {
        "id": document_id,
        "document_id": document_id,
        "filename": filename,
        "pages": len(pages),
        "chunks": len(chunks),
        "sha256": sha256,
        "duplicate": False,
        "cache_hit": cache_hit,
        "message": "PDF indexed successfully.",
    }


# Small process-local retrieval cache. The document IDs and chunk IDs are part of
# the key, so answers cannot accidentally cross document scopes.
_retrieve_cache: dict[tuple, list[dict]] = {}
_ORIGINAL_HYBRID_RETRIEVE = core.hybrid_retrieve


def cached_hybrid_retrieve(query: str, rows, k: int):
    chunk_ids = tuple(row["id"] for row in rows)
    key = (query.strip().lower(), chunk_ids, int(k))
    hit = _retrieve_cache.get(key)
    if hit is not None:
        return [dict(item) for item in hit]
    result = _ORIGINAL_HYBRID_RETRIEVE(query, rows, k)
    _retrieve_cache[key] = [dict(item) for item in result]
    return result


core.hybrid_retrieve = cached_hybrid_retrieve


@base.app.get("/")
async def root() -> HTMLResponse:
    # Reuse the production Gemini UI, but make duplicate uploads update the
    # existing browser-side document list instead of adding the same SHA twice.
    html = base.HTML
    old = "docs.unshift(d);selected.add(d.document_id);renderDocs();$('provider').textContent='Indexed · '+d.filename;$('file').value=''"
    new = "if(d.duplicate){const idx=docs.findIndex(x=>x.id===d.id);if(idx>=0)docs.splice(idx,1);docs.unshift(d);selected.add(d.document_id);renderDocs();$('provider').textContent='Already indexed · '+d.filename}else{docs.unshift(d);selected.add(d.document_id);renderDocs();$('provider').textContent='Indexed · '+d.filename}$('file').value=''"
    if old in html:
        html = html.replace(old, new)
    return HTMLResponse(html)


@app.get("/api/cache/stats")
def cache_stats() -> dict:
    files = list(CACHE_DIR.glob("*.json"))
    return {
        "retrieval_cache_entries": len(_retrieve_cache),
        "ingestion_cache_entries": len(files),
        "cache_directory": str(CACHE_DIR),
    }


ensure_schema()
