from __future__ import annotations

import math
import os
import re
import sqlite3
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", APP_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "documents"
DB_PATH = DATA_DIR / "rag.db"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Enterprise RAG Evaluation Platform",
    version="2.0.1",
    description="Dynamic PDF ingestion, hybrid retrieval, citations and offline evaluation.",
)

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "25"))
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "180"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))

# Common function words are poor evidence for answerability. Keeping them out of
# retrieval/grounding checks prevents queries such as "what email addresses are
# used in it" from matching an unrelated chunk simply because it contains "in".
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "could", "did", "do", "does", "for", "from", "how", "i", "if", "in",
    "into", "is", "it", "its", "me", "of", "on", "or", "our", "please",
    "should", "that", "the", "their", "them", "there", "these", "this",
    "to", "was", "were", "what", "when", "where", "which", "who", "why",
    "with", "would", "you", "your",
}
ABSTAIN_MESSAGE = "I could not find enough relevant evidence in the indexed documents to answer that question."


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents(
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            pages INTEGER NOT NULL,
            chunks INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks(
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            page_start INTEGER NOT NULL,
            page_end INTEGER NOT NULL,
            text TEXT NOT NULL,
            FOREIGN KEY(document_id) REFERENCES documents(id)
        )
    """)
    conn.commit()
    return conn


def tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def content_tokens(text: str) -> list[str]:
    return [token for token in tokens(text) if token not in STOPWORDS and len(token) > 1]


def chunk_pages(pages: list[str]) -> list[tuple[int, int, str]]:
    chunks: list[tuple[int, int, str]] = []
    current: list[str] = []
    start_page = 1
    current_words = 0
    for page_no, page_text in enumerate(pages, start=1):
        words = page_text.split()
        while words:
            remaining = CHUNK_WORDS - current_words
            take = max(1, min(remaining, len(words)))
            current.extend(words[:take])
            words = words[take:]
            current_words += take
            if current_words >= CHUNK_WORDS:
                chunks.append((start_page, page_no, " ".join(current)))
                overlap = current[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else []
                current = overlap
                current_words = len(current)
                start_page = page_no
    if current:
        chunks.append((start_page, len(pages), " ".join(current)))
    return chunks


def load_chunks(document_id: str | None = None) -> list[sqlite3.Row]:
    conn = db()
    if document_id:
        rows = conn.execute("SELECT * FROM chunks WHERE document_id=?", (document_id,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM chunks ORDER BY rowid").fetchall()
    conn.close()
    return rows


def bm25_scores(query: str, docs: list[str]) -> list[float]:
    q = content_tokens(query)
    if not docs or not q:
        return [0.0] * len(docs)
    term_df = Counter()
    frequencies = []
    lengths = []
    for doc in docs:
        tf = Counter(content_tokens(doc))
        frequencies.append(tf)
        lengths.append(sum(tf.values()))
        term_df.update(tf.keys())
    avgdl = sum(lengths) / max(1, len(lengths))
    n = len(docs)
    k1, b = 1.5, 0.75
    out = []
    for tf, dl in zip(frequencies, lengths):
        score = 0.0
        for term in q:
            if term not in tf:
                continue
            df = term_df[term]
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            score += idf * ((tf[term] * (k1 + 1)) / (tf[term] + k1 * (1 - b + b * dl / max(1, avgdl))))
        out.append(score)
    return out


def hybrid_retrieve(query: str, rows: list[sqlite3.Row], k: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    texts = [row["text"] for row in rows]
    lexical = bm25_scores(query, texts)
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    qvec = vectorizer.transform([query])
    dense = cosine_similarity(qvec, matrix)[0].tolist()

    def normalize(values: list[float]) -> list[float]:
        mx = max(values) if values else 0.0
        return [v / mx if mx else 0.0 for v in values]

    lex_n = normalize(lexical)
    dense_n = normalize(dense)
    scored = []
    for row, b, d in zip(rows, lex_n, dense_n):
        score = 0.5 * b + 0.5 * d
        scored.append((score, row, b, d))
    scored.sort(key=lambda x: (-x[0], x[1]["id"]))
    return [
        {
            "chunk_id": row["id"],
            "document_id": row["document_id"],
            "filename": row["filename"],
            "page_start": row["page_start"],
            "page_end": row["page_end"],
            "score": round(score, 4),
            "bm25": round(b, 4),
            "tfidf": round(d, 4),
            "text": row["text"],
        }
        for score, row, b, d in scored[:k]
    ]


def evidence_coverage(question: str, results: list[dict[str, Any]]) -> float:
    """Measure meaningful question-term coverage in retrieved evidence."""
    qterms = set(content_tokens(question))
    if not qterms or not results:
        return 0.0
    context_terms = set(content_tokens(" ".join(r["text"] for r in results)))
    return len(qterms & context_terms) / len(qterms)


def has_sufficient_evidence(question: str, results: list[dict[str, Any]]) -> bool:
    """Return True only when retrieval contains enough meaningful query evidence."""
    qterms = set(content_tokens(question))
    if not qterms or not results:
        return False
    coverage = evidence_coverage(question, results)
    required_terms = max(1, math.ceil(len(qterms) * 0.34))
    matched_terms = math.floor(coverage * len(qterms))
    return matched_terms >= required_terms


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)


class EvaluateRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    answer: str = Field(min_length=1, max_length=10000)
    document_id: str | None = None
    reference_answer: str | None = Field(default=None, max_length=10000)
    top_k: int = Field(default=5, ge=1, le=10)


def extractive_answer(question: str, results: list[dict[str, Any]]) -> str:
    if not has_sufficient_evidence(question, results):
        return ABSTAIN_MESSAGE
    sentences: list[tuple[int, str]] = []
    qset = set(content_tokens(question))
    for result in results:
        for sentence in re.split(r"(?<=[.!?])\s+", result["text"]):
            overlap = len(qset & set(content_tokens(sentence)))
            if overlap:
                sentences.append((overlap, sentence.strip()))
    best = [s for _, s in sorted(sentences, key=lambda x: -x[0])[:4]]
    if not best:
        return ABSTAIN_MESSAGE
    return " ".join(best)


def llm_answer(question: str, results: list[dict[str, Any]]) -> tuple[str, str]:
    if not has_sufficient_evidence(question, results):
        return ABSTAIN_MESSAGE, "abstain"
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return extractive_answer(question, results), "extractive"
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    context = "\n\n".join(
        f"[Source {i}] {r['filename']} pages {r['page_start']}-{r['page_end']}\n{r['text']}"
        for i, r in enumerate(results, start=1)
    )
    prompt = (
        "Answer only from the supplied context. Cite claims using [Source N]. "
        "If the context does not contain the answer, say you cannot find it.\n\n"
        f"Question: {question}\n\nContext:\n{context}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a grounded enterprise RAG assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    try:
        response = httpx.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return content, "llm"
    except Exception:
        return extractive_answer(question, results), "extractive_fallback"


def offline_metrics(question: str, answer: str, results: list[dict[str, Any]], reference: str | None) -> dict[str, float | None]:
    context = " ".join(r["text"] for r in results)
    a = set(tokens(answer))
    c = set(tokens(context))
    q = set(tokens(question))
    groundedness = len(a & c) / max(1, len(a))
    relevance = len(q & a) / max(1, len(q))
    reference_score = None
    if reference:
        reference_score = len(set(tokens(reference)) & a) / max(1, len(set(tokens(reference))))
    values = [groundedness, relevance] + ([] if reference_score is None else [reference_score])
    return {
        "groundedness": round(groundedness, 4),
        "answer_relevance": round(relevance, 4),
        "reference_overlap": None if reference_score is None else round(reference_score, 4),
        "overall": round(sum(values) / len(values), 4),
    }


@app.on_event("startup")
def startup() -> None:
    db().close()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "enterprise-rag-evaluation-platform"}


@app.get("/api/documents")
def documents() -> dict[str, Any]:
    conn = db()
    rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
    conn.close()
    return {"count": len(rows), "documents": [dict(r) for r in rows]}


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF files are supported.")
    content = await file.read()
    if len(content) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"PDF exceeds {MAX_FILE_MB} MB limit.")
    document_id = uuid.uuid4().hex
    stored = UPLOAD_DIR / f"{document_id}.pdf"
    stored.write_bytes(content)
    try:
        reader = PdfReader(str(stored))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:
        stored.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not read PDF: {exc}") from exc
    chunks = chunk_pages(pages)
    conn = db()
    conn.execute(
        "INSERT INTO documents(id,filename,stored_path,pages,chunks) VALUES(?,?,?,?,?)",
        (document_id, file.filename or "document.pdf", str(stored), len(pages), len(chunks)),
    )
    conn.executemany(
        "INSERT INTO chunks(id,document_id,filename,page_start,page_end,text) VALUES(?,?,?,?,?,?)",
        [
            (uuid.uuid4().hex, document_id, file.filename or "document.pdf", start, end, text)
            for start, end, text in chunks
        ],
    )
    conn.commit()
    conn.close()
    return {"document_id": document_id, "filename": file.filename, "pages": len(pages), "chunks": len(chunks)}


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, str]:
    conn = db()
    row = conn.execute("SELECT stored_path FROM documents WHERE id=?", (document_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Document not found")
    conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
    conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
    conn.commit()
    conn.close()
    Path(row["stored_path"]).unlink(missing_ok=True)
    return {"status": "deleted"}


@app.get("/api/retrieve")
def retrieve(
    q: str = Query(min_length=3, max_length=2000),
    document_id: str | None = None,
    k: int = Query(default=5, ge=1, le=10),
) -> dict[str, Any]:
    results = hybrid_retrieve(q, load_chunks(document_id), k)
    return {
        "query": q,
        "evidence_coverage": round(evidence_coverage(q, results), 4),
        "answerable": has_sufficient_evidence(q, results),
        "results": results,
    }


@app.post("/api/ask")
def ask(body: AskRequest) -> dict[str, Any]:
    results = hybrid_retrieve(body.question, load_chunks(body.document_id), body.top_k)
    answer, mode = llm_answer(body.question, results)
    answerable = has_sufficient_evidence(body.question, results)
    return {
        "question": body.question,
        "answer": answer,
        "mode": mode,
        "answerable": answerable,
        "evidence_coverage": round(evidence_coverage(body.question, results), 4),
        "citations": [
            {
                "source": r["filename"],
                "pages": f"{r['page_start']}-{r['page_end']}",
                "score": r["score"],
            }
            for r in results
        ] if answerable else [],
        "retrieved": results,
    }


@app.post("/api/evaluate")
def evaluate(body: EvaluateRequest) -> dict[str, Any]:
    results = hybrid_retrieve(body.question, load_chunks(body.document_id), body.top_k)
    answer_metrics = offline_metrics(body.question, body.answer, results, body.reference_answer)
    return {"metrics": answer_metrics, "retrieved": results}


HTML = """
<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Enterprise RAG Evaluation Platform</title>
<style>
:root{color-scheme:dark}body{font-family:Inter,system-ui,-apple-system,sans-serif;margin:0;background:#0b1020;color:#eef2ff}main{max-width:1180px;margin:auto;padding:32px 20px 60px}.hero{margin-bottom:22px}.muted{color:#98a4bd}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{background:#151d33;border:1px solid #2a385a;border-radius:16px;padding:20px;margin:16px 0}.wide{grid-column:1/-1}input,textarea,button{width:100%;box-sizing:border-box;margin-top:10px;padding:12px;border-radius:10px;border:1px solid #364563;background:#0f1629;color:#fff;font:inherit}button{cursor:pointer}textarea{min-height:110px}.row{display:flex;gap:10px}.row>*{flex:1}pre{white-space:pre-wrap;word-break:break-word;background:#0f1629;border-radius:10px;padding:14px;max-height:420px;overflow:auto}.doc{padding:10px 0;border-bottom:1px solid #25314e}.pill{display:inline-block;padding:4px 8px;border-radius:999px;background:#202c46;color:#bdc8dc;font-size:12px;margin-right:6px}.score{font-size:22px;font-weight:700}.status{margin-top:10px;color:#b9c4d8}.warning{color:#f2c46d}@media(max-width:800px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}}
</style></head><body><main>
<section class='hero'><span class='pill'>LIVE RAG WORKBENCH</span><h1>Enterprise RAG Evaluation Platform</h1><p class='muted'>Upload PDFs, build a persistent index, retrieve with hybrid BM25 + TF-IDF search, ask grounded questions, inspect citations, and evaluate answers.</p></section>
<div class='grid'>
<div class='card'><h2>1. Upload PDF</h2><input id='file' type='file' accept='application/pdf'><button onclick='upload()'>Upload and index</button><div id='uploadStatus' class='status'></div></div>
<div class='card'><h2>2. Indexed documents</h2><button onclick='loadDocs()'>Refresh documents</button><div id='docs'></div></div>
<div class='card wide'><h2>3. Ask your documents</h2><div class='row'><input id='question' value='What is the main conclusion of this document?'><input id='docid' placeholder='Optional document ID'></div><button onclick='ask()'>Ask question</button><h3>Answer</h3><pre id='answer'>Upload a PDF and ask a question.</pre><h3>Sources</h3><pre id='sources'>—</pre></div>
<div class='card'><h2>4. Inspect retrieval</h2><input id='rq' value='summary and conclusion'><button onclick='retrieveDocs()'>Run hybrid retrieval</button><pre id='retrieval'>—</pre></div>
<div class='card'><h2>5. Evaluate an answer</h2><textarea id='evalAnswer'>The document explains its central findings and supporting evidence.</textarea><button onclick='evaluateAnswer()'>Evaluate</button><pre id='metrics'>—</pre></div>
</div>
<script>
async function loadDocs(){const r=await fetch('/api/documents');const d=await r.json();document.getElementById('docs').innerHTML=d.documents.map(x=>`<div class='doc'><b>${x.filename}</b><br><span class='pill'>${x.pages} pages</span><span class='pill'>${x.chunks} chunks</span><button onclick="document.getElementById('docid').value='${x.id}'">Use</button></div>`).join('')||'<div class="status">No PDFs indexed yet.</div>'}
async function upload(){const f=document.getElementById('file').files[0];if(!f)return;const fd=new FormData();fd.append('file',f);document.getElementById('uploadStatus').textContent='Indexing…';const r=await fetch('/api/documents/upload',{method:'POST',body:fd});const d=await r.json();document.getElementById('uploadStatus').textContent=r.ok?`Indexed ${d.filename}: ${d.pages} pages, ${d.chunks} chunks.`:(d.detail||'Upload failed');loadDocs()}
async function ask(){const body={question:document.getElementById('question').value,document_id:document.getElementById('docid').value||null,top_k:5};const r=await fetch('/api/ask',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});const d=await r.json();document.getElementById('answer').textContent=d.answer||d.detail;document.getElementById('sources').textContent=JSON.stringify({mode:d.mode,answerable:d.answerable,evidence_coverage:d.evidence_coverage,citations:d.citations},null,2)}
async function retrieveDocs(){const q=document.getElementById('rq').value;const id=document.getElementById('docid').value;const u='/api/retrieve?q='+encodeURIComponent(q)+(id?'&document_id='+encodeURIComponent(id):'');const r=await fetch(u);document.getElementById('retrieval').textContent=JSON.stringify(await r.json(),null,2)}
async function evaluateAnswer(){const body={question:document.getElementById('question').value,answer:document.getElementById('evalAnswer').value,document_id:document.getElementById('docid').value||null,top_k:5};const r=await fetch('/api/evaluate',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});document.getElementById('metrics').textContent=JSON.stringify(await r.json(),null,2)}
loadDocs()
</script></main></body></html>
"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return HTML
