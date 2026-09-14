from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import HTMLResponse

import gemini_main as ui
import production_main as base
import main as core

app = base.app
VERSION = "4.2.0"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def migrate() -> None:
    conn = core.db()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(documents)").fetchall()}
    for name, definition in {
        "sha256": "TEXT",
        "status": "TEXT DEFAULT 'indexed'",
        "source_type": "TEXT DEFAULT 'upload'",
        "collection_id": "TEXT",
        "version": "INTEGER DEFAULT 1",
        "quality_score": "REAL",
        "updated_at": "TEXT",
    }.items():
        if name not in cols:
            conn.execute(f"ALTER TABLE documents ADD COLUMN {name} {definition}")

    conn.execute("DROP INDEX IF EXISTS ux_documents_sha256")

    conn.execute("""CREATE TABLE IF NOT EXISTS collections(
        id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, description TEXT DEFAULT '',
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS feedback(
        id TEXT PRIMARY KEY, question TEXT NOT NULL, answer TEXT NOT NULL,
        rating INTEGER NOT NULL CHECK(rating IN (-1,1)), document_ids TEXT NOT NULL,
        mode TEXT, created_at TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS query_traces(
        id TEXT PRIMARY KEY, question TEXT NOT NULL, document_ids TEXT NOT NULL,
        mode TEXT, model TEXT, answerable INTEGER NOT NULL, evidence_coverage REAL NOT NULL,
        retrieval_ms REAL NOT NULL, rerank_ms REAL NOT NULL, generation_ms REAL NOT NULL,
        ttft_ms REAL NOT NULL, total_ms REAL NOT NULL, candidate_count INTEGER NOT NULL,
        context_chunks INTEGER NOT NULL, cache_hit INTEGER NOT NULL, created_at TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS audit_logs(
        id TEXT PRIMARY KEY, action TEXT NOT NULL, resource_type TEXT NOT NULL,
        resource_id TEXT, metadata TEXT NOT NULL, created_at TEXT NOT NULL)""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trace_time ON query_traces(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_logs(created_at)")

    legacy = conn.execute(
        "SELECT id, stored_path, created_at FROM documents WHERE sha256 IS NULL ORDER BY created_at DESC"
    ).fetchall()
    seen_sha: dict[str, str] = {}
    for item in legacy:
        path = Path(item["stored_path"])
        if not path.exists():
            continue
        try:
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
        if sha in seen_sha:
            conn.execute("DELETE FROM chunks WHERE document_id=?", (item["id"],))
            conn.execute("DELETE FROM documents WHERE id=?", (item["id"],))
            path.unlink(missing_ok=True)
            continue
        seen_sha[sha] = item["id"]
        conn.execute("UPDATE documents SET sha256=? WHERE id=?", (sha, item["id"]))

    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_documents_sha256 ON documents(sha256) WHERE sha256 IS NOT NULL")
    conn.execute("UPDATE documents SET status=COALESCE(status,'indexed'),version=COALESCE(version,1),updated_at=COALESCE(updated_at,created_at)")
    conn.commit()
    conn.close()


def audit(action: str, resource: str, resource_id: str | None = None, meta: dict[str, Any] | None = None) -> None:
    conn = core.db()
    conn.execute(
        "INSERT INTO audit_logs VALUES(?,?,?,?,?,?)",
        (uuid.uuid4().hex, action, resource, resource_id, json.dumps(meta or {}, ensure_ascii=False), now()),
    )
    conn.commit()
    conn.close()


def quality(row: sqlite3.Row) -> float:
    pages = max(1, int(row["pages"] or 0))
    chunks = int(row["chunks"] or 0)
    density = min(1.0, chunks / max(1.0, pages * 1.5))
    return round((0.7 * density + 0.3) * 100, 1)


# Remove lower-layer endpoints that must be owned by the enterprise entrypoint.
app.routes[:] = [
    route for route in app.routes
    if getattr(route, "path", None) not in {"/", "/health"}
]


@app.on_event("startup")
def enterprise_startup() -> None:
    migrate()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "enterprise-rag-evaluation-platform", "version": VERSION}


@app.get("/api/enterprise/overview")
def overview() -> dict[str, Any]:
    conn = core.db()
    d = conn.execute("SELECT COUNT(*) total, SUM(status='indexed') indexed FROM documents").fetchone()
    c = conn.execute("SELECT COUNT(*) FROM collections").fetchone()[0]
    t = conn.execute("""SELECT COUNT(*) n, AVG(total_ms) avg_ms, AVG(evidence_coverage) coverage,
                               SUM(answerable) answerable FROM query_traces""").fetchone()
    f = conn.execute("SELECT COUNT(*) n, SUM(rating=1) positive FROM feedback").fetchone()
    conn.close()
    n = int(t["n"] or 0)
    fn = int(f["n"] or 0)
    return {
        "version": VERSION,
        "documents": int(d["total"] or 0),
        "indexed_documents": int(d["indexed"] or 0),
        "collections": int(c),
        "queries": n,
        "answerable_rate": round((t["answerable"] or 0) / n, 4) if n else 0,
        "avg_latency_ms": round(float(t["avg_ms"] or 0), 1),
        "avg_evidence_coverage": round(float(t["coverage"] or 0), 4),
        "positive_feedback_rate": round((f["positive"] or 0) / fn, 4) if fn else 0,
        "retrieval_cache_entries": len(base._retrieve_cache),
        "ingestion_cache_entries": len(list(base.CACHE_DIR.glob("*.json"))),
    }


@app.get("/api/collections")
def collections() -> dict[str, Any]:
    conn = core.db()
    rows = conn.execute("""SELECT c.*, COUNT(d.id) document_count
        FROM collections c LEFT JOIN documents d ON d.collection_id=c.id
        GROUP BY c.id ORDER BY c.name""").fetchall()
    conn.close()
    return {"count": len(rows), "collections": [dict(r) for r in rows]}


@app.post("/api/collections")
async def create_collection(payload: dict[str, Any]) -> dict[str, Any]:
    name = str(payload.get("name", "")).strip()
    if not 2 <= len(name) <= 100:
        raise HTTPException(400, "Collection name must be 2-100 characters")
    cid, stamp = uuid.uuid4().hex, now()
    conn = core.db()
    try:
        conn.execute(
            "INSERT INTO collections VALUES(?,?,?,?,?)",
            (cid, name, str(payload.get("description", "")).strip(), stamp, stamp),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.rollback()
        raise HTTPException(409, "Collection already exists")
    finally:
        conn.close()
    audit("collection.create", "collection", cid, {"name": name})
    return {"id": cid, "name": name}


@app.patch("/api/documents/{document_id}/collection")
async def set_collection(document_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    cid = payload.get("collection_id")
    conn = core.db()
    if not conn.execute("SELECT 1 FROM documents WHERE id=?", (document_id,)).fetchone():
        conn.close()
        raise HTTPException(404, "Document not found")
    if cid and not conn.execute("SELECT 1 FROM collections WHERE id=?", (cid,)).fetchone():
        conn.close()
        raise HTTPException(404, "Collection not found")
    conn.execute("UPDATE documents SET collection_id=?,updated_at=? WHERE id=?", (cid, now(), document_id))
    conn.commit()
    conn.close()
    base._retrieve_cache.clear()
    audit("document.collection_assign", "document", document_id, {"collection_id": cid})
    return {"document_id": document_id, "collection_id": cid}


@app.get("/api/documents/{document_id}")
def document_detail(document_id: str) -> dict[str, Any]:
    conn = core.db()
    row = conn.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Document not found")
    item = dict(row)
    item["quality_score"] = quality(row)
    item["stored_file_exists"] = Path(row["stored_path"]).exists()
    return item


@app.post("/api/documents/{document_id}/reprocess")
def reprocess(document_id: str) -> dict[str, Any]:
    conn = core.db()
    row = conn.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Document not found")
    path = Path(row["stored_path"])
    if not path.exists():
        raise HTTPException(404, "Stored PDF is missing")
    from pypdf import PdfReader
    try:
        pages = [core.clean_text(p.extract_text() or "") for p in PdfReader(str(path)).pages]
        chunks = core.chunk_pages(pages)
    except Exception as exc:
        raise HTTPException(400, f"Could not reprocess PDF: {exc}") from exc
    conn = core.db()
    conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
    conn.executemany(
        "INSERT INTO chunks(id,document_id,filename,page_start,page_end,text) VALUES(?,?,?,?,?,?)",
        [(uuid.uuid4().hex, document_id, row["filename"], s, e, t) for s, e, t in chunks],
    )
    new_version = int(row["version"] or 1) + 1
    conn.execute(
        "UPDATE documents SET pages=?,chunks=?,version=?,status='indexed',updated_at=? WHERE id=?",
        (len(pages), len(chunks), new_version, now(), document_id),
    )
    conn.commit()
    conn.close()
    if row["sha256"]:
        base.save_chunk_cache(row["sha256"], row["filename"], len(pages), chunks)
    base._retrieve_cache.clear()
    audit("document.reprocess", "document", document_id, {"pages": len(pages), "chunks": len(chunks), "version": new_version})
    return {"document_id": document_id, "pages": len(pages), "chunks": len(chunks), "version": new_version}


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, Any]:
    conn = core.db()
    row = conn.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Document not found")
    conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
    conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
    conn.commit()
    conn.close()
    Path(row["stored_path"]).unlink(missing_ok=True)
    if row["sha256"]:
        base.cache_path(row["sha256"]).unlink(missing_ok=True)
    base._retrieve_cache.clear()
    audit("document.delete", "document", document_id, {"filename": row["filename"]})
    return {"deleted": True, "document_id": document_id}


@app.post("/api/feedback")
async def feedback(payload: dict[str, Any]) -> dict[str, Any]:
    q = str(payload.get("question", "")).strip()
    a = str(payload.get("answer", "")).strip()
    try:
        rating = int(payload.get("rating", 0))
    except (TypeError, ValueError):
        rating = 0
    if not q or not a or rating not in (-1, 1):
        raise HTTPException(400, "question, answer and rating (-1 or 1) are required")
    fid = uuid.uuid4().hex
    conn = core.db()
    conn.execute(
        "INSERT INTO feedback VALUES(?,?,?,?,?,?,?)",
        (fid, q, a, rating, json.dumps(payload.get("document_ids") or []), payload.get("mode"), now()),
    )
    conn.commit()
    conn.close()
    audit("feedback.submit", "answer", fid, {"rating": rating})
    return {"id": fid, "saved": True}


@app.post("/api/traces")
async def trace(payload: dict[str, Any]) -> dict[str, Any]:
    lat = payload.get("latency") or {}
    tid = uuid.uuid4().hex
    conn = core.db()
    conn.execute(
        "INSERT INTO query_traces VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            tid,
            str(payload.get("question", ""))[:2000],
            json.dumps(payload.get("scope_document_ids") or []),
            payload.get("mode"),
            payload.get("model"),
            int(bool(payload.get("answerable"))),
            float(payload.get("evidence_coverage") or 0),
            float(lat.get("retrieval_ms") or 0),
            float(lat.get("rerank_ms") or 0),
            float(lat.get("generation_ms") or 0),
            float(lat.get("ttft_ms") or 0),
            float(lat.get("total_ms") or 0),
            int(lat.get("candidate_count") or 0),
            int(lat.get("context_chunks") or 0),
            int(bool(payload.get("cache_hit"))),
            now(),
        ),
    )
    conn.commit()
    conn.close()
    return {"id": tid, "saved": True}


@app.get("/api/admin/audit-logs")
def audit_logs(limit: int = 50) -> dict[str, Any]:
    limit = max(1, min(200, int(limit)))
    conn = core.db()
    rows = conn.execute("SELECT * FROM audit_logs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return {"count": len(rows), "logs": [dict(r) for r in rows]}


migrate()

# Reuse the working Gemini UI. The previous implementation referenced
# production_main.HTML, but the HTML is owned by gemini_main.
ENTERPRISE_HTML = ui.HTML.replace(
    '<main class="main">',
    '<main class="main"><section id="enterpriseBar" style="margin-bottom:14px;border:1px solid #223049;background:#0d1624;border-radius:12px;padding:12px 14px;display:flex;align-items:center;gap:10px;flex-wrap:wrap">'
    '<b style="font-size:12px">Enterprise telemetry</b>'
    '<span id="eDocs" style="color:#8b99ad;font-size:11px">Docs —</span>'
    '<span id="eQueries" style="color:#8b99ad;font-size:11px">Queries —</span>'
    '<span id="eCoverage" style="color:#8b99ad;font-size:11px">Evidence —</span>'
    '<span id="eLatency" style="color:#8b99ad;font-size:11px">Latency —</span>'
    '<span id="eCache" style="color:#8b99ad;font-size:11px">Cache —</span>'
    '<a href="/docs" target="_blank" style="margin-left:auto;border:1px solid #223049;background:#101b2b;color:#d4deed;padding:6px 9px;border-radius:7px;text-decoration:none;font-size:11px">API</a>'
    '</section>'
)
ENTERPRISE_HTML = ENTERPRISE_HTML.replace(
    '</body></html>',
    '<script>async function enterpriseTelemetry(){try{const r=await fetch("/api/enterprise/overview");if(!r.ok)return;const d=await r.json();'
    'document.getElementById("eDocs").textContent=`Docs ${d.indexed_documents}/${d.documents}`;'
    'document.getElementById("eQueries").textContent=`Queries ${d.queries}`;'
    'document.getElementById("eCoverage").textContent=`Evidence ${(d.avg_evidence_coverage*100).toFixed(0)}%`;'
    'document.getElementById("eLatency").textContent=`Avg ${d.avg_latency_ms.toFixed(0)} ms`;'
    'document.getElementById("eCache").textContent=`Cache ${d.retrieval_cache_entries}R · ${d.ingestion_cache_entries}I`;}catch{}}'
    'enterpriseTelemetry();setInterval(enterpriseTelemetry,15000);</script></body></html>'
)

app.routes[:] = [r for r in app.routes if getattr(r, "path", None) != "/"]


@app.get("/")
async def enterprise_root() -> Any:
    return HTMLResponse(ENTERPRISE_HTML)
