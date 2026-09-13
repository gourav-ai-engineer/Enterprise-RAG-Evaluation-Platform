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
    version="3.0.0",
    description="Dynamic PDF ingestion, hybrid retrieval, grounded concise answers, citations and evaluation.",
)
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "25"))
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "160"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "35"))
STOPWORDS = set("a an and are as at be been but by can could did do does for from how i if in into is it its me of on or our please should that the their them there these this to was were what when where which who why with would you your".split())
ABSTAIN_MESSAGE = "I couldn't find enough relevant evidence in the indexed documents to answer that question."


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, filename TEXT NOT NULL, stored_path TEXT NOT NULL, pages INTEGER NOT NULL, chunks INTEGER NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
    conn.execute("CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY, document_id TEXT NOT NULL, filename TEXT NOT NULL, page_start INTEGER NOT NULL, page_end INTEGER NOT NULL, text TEXT NOT NULL)")
    conn.commit()
    return conn


def tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOPWORDS and len(t) > 1]


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return re.sub(r"(?<=\w)-\s+(?=\w)", "", text)


def chunk_pages(pages: list[str]) -> list[tuple[int, int, str]]:
    chunks: list[tuple[int, int, str]] = []
    current: list[str] = []
    start_page = 1
    current_words = 0
    for page_no, page_text in enumerate(pages, 1):
        words = clean_text(page_text).split()
        while words:
            take = min(CHUNK_WORDS - current_words, len(words))
            current.extend(words[:take])
            words = words[take:]
            current_words += take
            if current_words >= CHUNK_WORDS:
                chunks.append((start_page, page_no, " ".join(current)))
                current = current[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else []
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
    terms = content_tokens(query)
    if not docs or not terms:
        return [0.0] * len(docs)
    dfs: Counter[str] = Counter()
    frequencies: list[Counter[str]] = []
    lengths: list[int] = []
    for doc in docs:
        freq = Counter(content_tokens(doc))
        frequencies.append(freq)
        lengths.append(sum(freq.values()))
        dfs.update(freq.keys())
    avg_len = sum(lengths) / max(1, len(lengths))
    scores: list[float] = []
    for freq, length in zip(frequencies, lengths):
        score = 0.0
        for term in terms:
            if term not in freq:
                continue
            df = dfs[term]
            idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
            score += idf * freq[term] * 2.5 / (freq[term] + 1.5 * (0.25 + 0.75 * length / max(avg_len, 1)))
        scores.append(score)
    return scores


def hybrid_retrieve(query: str, rows: list[sqlite3.Row], k: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    texts = [row["text"] for row in rows]
    lexical = bm25_scores(query, texts)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    tfidf = cosine_similarity(vectorizer.transform([query]), matrix)[0].tolist()

    def normalize(values: list[float]) -> list[float]:
        maximum = max(values) if values else 0.0
        return [value / maximum if maximum else 0.0 for value in values]

    lexical_norm = normalize(lexical)
    tfidf_norm = normalize(tfidf)
    ranked = []
    for row, bm25, tf in zip(rows, lexical_norm, tfidf_norm):
        ranked.append((0.55 * bm25 + 0.45 * tf, row, bm25, tf))
    ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
    return [
        {
            "chunk_id": row["id"],
            "document_id": row["document_id"],
            "filename": row["filename"],
            "page_start": row["page_start"],
            "page_end": row["page_end"],
            "score": round(score, 4),
            "bm25": round(bm25, 4),
            "tfidf": round(tf, 4),
            "text": row["text"],
        }
        for score, row, bm25, tf in ranked[:k]
    ]


def evidence_coverage(question: str, results: list[dict[str, Any]]) -> float:
    question_terms = set(content_tokens(question))
    if not question_terms or not results:
        return 0.0
    context_terms = set(content_tokens(" ".join(result["text"] for result in results)))
    return len(question_terms & context_terms) / len(question_terms)


def has_sufficient_evidence(question: str, results: list[dict[str, Any]]) -> bool:
    question_terms = set(content_tokens(question))
    if not question_terms or not results:
        return False
    context_terms = set(content_tokens(" ".join(result["text"] for result in results)))
    matched = len(question_terms & context_terms)
    return matched >= max(1, math.ceil(len(question_terms) * 0.34))


def sentence_score(question: str, sentence: str) -> float:
    question_terms = set(content_tokens(question))
    sentence_terms = set(content_tokens(sentence))
    overlap = len(question_terms & sentence_terms)
    if not overlap:
        return -1.0
    score = overlap * 4.0
    lower = sentence.lower()
    if re.search(r"\b(is|are|refers to|denotes|defined as|we present|we propose|combines|combining)\b", lower):
        score += 5.0
    if len(sentence.split()) < 8:
        score -= 2.0
    digit_ratio = sum(character.isdigit() for character in sentence) / max(1, len(sentence))
    if digit_ratio > 0.08:
        score -= 4.0
    if "computation time" in lower or "memory" in lower or "table" in lower:
        score -= 5.0
    return score


def extractive_answer(question: str, results: list[dict[str, Any]]) -> str:
    if not has_sufficient_evidence(question, results):
        return ABSTAIN_MESSAGE
    candidates: list[tuple[float, int, str]] = []
    for rank, result in enumerate(results):
        for sentence in re.split(r"(?<=[.!?])\s+", result["text"]):
            sentence = clean_text(sentence)
            score = sentence_score(question, sentence)
            if score >= 0:
                candidates.append((score, -rank, sentence))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if not candidates:
        return ABSTAIN_MESSAGE
    selected: list[str] = []
    seen: set[str] = set()
    for _, _, sentence in candidates:
        key = " ".join(content_tokens(sentence))
        if key in seen:
            continue
        selected.append(sentence)
        seen.add(key)
        if len(selected) == 2:
            break
    return " ".join(selected)


def llm_answer(question: str, results: list[dict[str, Any]]) -> tuple[str, str]:
    if not has_sufficient_evidence(question, results):
        return ABSTAIN_MESSAGE, "abstain"
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return extractive_answer(question, results), "extractive"
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    context = "\n\n".join(
        f"[Source {index}] {result['filename']} pages {result['page_start']}-{result['page_end']}\n{result['text']}"
        for index, result in enumerate(results[:5], 1)
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a grounded enterprise document assistant. Answer concisely and only from the supplied context. Never invent missing facts. Return 1-3 sentences."},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
        ],
        "temperature": 0,
    }
    try:
        response = httpx.post(f"{base_url}/chat/completions", headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip(), "llm"
    except Exception:
        return extractive_answer(question, results), "extractive_fallback"


def compact_citations(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations = []
    seen = set()
    for result in results:
        key = (result["filename"], result["page_start"], result["page_end"])
        if key in seen:
            continue
        seen.add(key)
        citations.append({"source": result["filename"], "pages": f"{result['page_start']}-{result['page_end']}", "score": result["score"]})
        if len(citations) >= 3:
            break
    return citations


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


def offline_metrics(question: str, answer: str, results: list[dict[str, Any]], reference: str | None) -> dict[str, float | None]:
    context = set(tokens(" ".join(result["text"] for result in results)))
    answer_terms = set(tokens(answer))
    question_terms = set(tokens(question))
    groundedness = len(answer_terms & context) / max(1, len(answer_terms))
    relevance = len(question_terms & answer_terms) / max(1, len(question_terms))
    reference_overlap = None
    if reference:
        reference_terms = set(tokens(reference))
        reference_overlap = len(reference_terms & answer_terms) / max(1, len(reference_terms))
    values = [groundedness, relevance] + ([] if reference_overlap is None else [reference_overlap])
    return {
        "groundedness": round(groundedness, 4),
        "answer_relevance": round(relevance, 4),
        "reference_overlap": None if reference_overlap is None else round(reference_overlap, 4),
        "overall": round(sum(values) / len(values), 4),
    }


@app.on_event("startup")
def startup() -> None:
    db().close()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "enterprise-rag-evaluation-platform", "version": "3.0.0"}


@app.get("/api/documents")
def documents() -> dict[str, Any]:
    conn = db()
    rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
    conn.close()
    return {"count": len(rows), "documents": [dict(row) for row in rows]}


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
        pages = [clean_text(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:
        stored.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not read PDF: {exc}") from exc
    chunks = chunk_pages(pages)
    filename = file.filename or "document.pdf"
    conn = db()
    conn.execute("INSERT INTO documents(id,filename,stored_path,pages,chunks) VALUES(?,?,?,?,?)", (document_id, filename, str(stored), len(pages), len(chunks)))
    conn.executemany("INSERT INTO chunks(id,document_id,filename,page_start,page_end,text) VALUES(?,?,?,?,?,?)", [(uuid.uuid4().hex, document_id, filename, start, end, text) for start, end, text in chunks])
    conn.commit()
    conn.close()
    return {"document_id": document_id, "filename": filename, "pages": len(pages), "chunks": len(chunks)}


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
def retrieve(q: str = Query(min_length=3, max_length=2000), document_id: str | None = None, k: int = Query(default=5, ge=1, le=10)) -> dict[str, Any]:
    results = hybrid_retrieve(q, load_chunks(document_id), k)
    return {"query": q, "evidence_coverage": round(evidence_coverage(q, results), 4), "answerable": has_sufficient_evidence(q, results), "results": results}


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
        "citations": compact_citations(results) if answerable else [],
        "retrieved": results,
    }


@app.post("/api/evaluate")
def evaluate(body: EvaluateRequest) -> dict[str, Any]:
    results = hybrid_retrieve(body.question, load_chunks(body.document_id), body.top_k)
    return {"metrics": offline_metrics(body.question, body.answer, results, body.reference_answer), "retrieved": results}


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Enterprise RAG</title><style>
:root{color-scheme:dark;--bg:#080d18;--panel:#101827;--line:#243248;--text:#f3f6fb;--muted:#8d9ab0;--accent:#7c9cff;--good:#46d39a;--warn:#f2c66d}*{box-sizing:border-box}body{margin:0;background:radial-gradient(900px 500px at 15% -10%,#17264a,transparent 55%),var(--bg);font-family:Inter,system-ui,sans-serif;color:var(--text)}button,textarea{font:inherit}button{cursor:pointer}.top{height:68px;border-bottom:1px solid var(--line);display:flex;align-items:center;padding:0 28px}.brand{font-weight:750}.badge{margin-left:auto;font-size:12px;color:#aebbd0;border:1px solid var(--line);padding:6px 10px;border-radius:999px}.layout{display:grid;grid-template-columns:280px minmax(0,1fr);max-width:1440px;margin:auto;min-height:calc(100vh - 68px)}.side{border-right:1px solid var(--line);padding:26px 18px}.nav{padding:11px 12px;color:#9eabc0;border-radius:10px}.nav.active{background:#18233a;color:#fff}.note{margin-top:25px;padding:14px;border:1px solid var(--line);border-radius:12px;color:var(--muted);font-size:12px;line-height:1.55}.main{padding:34px}.hero{display:flex;justify-content:space-between;align-items:end;margin-bottom:24px}.hero h1{font-size:30px;margin:7px 0}.hero p,.meta,.hint{color:var(--muted);font-size:13px}.primary,.send{border:0;background:linear-gradient(135deg,#718cff,#8a73e8);color:#fff;border-radius:10px;font-weight:700}.primary{padding:11px 16px}.grid{display:grid;grid-template-columns:330px minmax(0,1fr);gap:18px}.card{border:1px solid var(--line);background:#101827e8;border-radius:16px;overflow:hidden}.head{padding:18px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between}.body{padding:18px}.drop{border:1px dashed #40516f;background:#0b1422;border-radius:12px;padding:24px;text-align:center}.drop p{color:var(--muted);font-size:12px}.choose,.mini{display:inline-block;border:1px solid var(--line);background:#0d1626;color:#c4cee0;border-radius:8px;padding:8px 11px;font-size:11px}.drop input{display:none}.docs{margin-top:14px}.doc{padding:12px 0;border-bottom:1px solid #1d293c}.docname{font-size:13px;font-weight:650}.actions{margin-top:8px;display:flex;gap:7px}.chat{min-height:620px;display:flex;flex-direction:column}.chatbody{padding:25px;display:flex;flex-direction:column;flex:1}.welcome{text-align:center;max-width:620px;margin:60px auto}.welcome p{color:var(--muted);line-height:1.6;font-size:13px}.answer{display:none;background:#0c1422;border:1px solid #2b3b56;border-radius:14px;padding:20px;line-height:1.7;font-size:14px;white-space:pre-wrap}.answer.show{display:block}.source-title{margin-top:18px;color:#aebbd0;font-size:12px}.sources{display:flex;gap:9px;flex-wrap:wrap;margin-top:9px}.source{border:1px solid var(--line);background:#0c1422;border-radius:10px;padding:10px 12px}.source b{font-size:11px;display:block;max-width:250px;overflow:hidden;text-overflow:ellipsis}.source span{font-size:10px;color:#78869d}.composer{margin-top:auto;padding-top:20px}.box{border:1px solid #31425f;background:#0b1422;border-radius:13px;padding:10px;display:flex;gap:10px}.box textarea{flex:1;resize:none;border:0;outline:0;background:transparent;color:#fff;min-height:44px;padding:7px}.send{width:42px}.advanced{margin-top:18px}.advanced summary{padding:15px 18px;color:#9eabc0;cursor:pointer}.metrics{padding:0 18px 18px;display:grid;grid-template-columns:repeat(3,1fr);gap:9px}.metric{background:#0c1422;border:1px solid var(--line);border-radius:10px;padding:12px}.metric span{display:block;color:#72819a;font-size:10px}.metric b{display:block;margin-top:5px}@media(max-width:1050px){.layout{grid-template-columns:1fr}.side{display:none}.grid{grid-template-columns:1fr}.main{padding:22px}.hero{align-items:flex-start;flex-direction:column}.metrics{grid-template-columns:1fr 1fr}}</style></head><body>
<header class="top"><div class="brand">✦ Enterprise RAG</div><div class="badge">● Grounded knowledge assistant</div></header><div class="layout"><aside class="side"><div class="nav active">⌂ Knowledge assistant</div><div class="nav">▣ Documents</div><div class="nav">✓ Evaluation</div><div class="nav">↳ Hybrid retrieval</div><div class="nav">⊙ Grounding guard</div><div class="note"><b>Verifiable answers</b><br>Answers are generated from indexed evidence. Weak or unsupported questions are rejected instead of guessed.</div></aside><main class="main"><div class="hero"><div><div class="meta">KNOWLEDGE WORKBENCH</div><h1>Ask your enterprise documents</h1><p>Upload PDFs and receive concise, traceable answers with page-level citations.</p></div><button class="primary" onclick="document.getElementById('file').click()">+ Add document</button></div><div class="grid"><section class="card"><div class="head"><b>Knowledge base</b><span id="count" class="meta">0 documents</span></div><div class="body"><div class="drop"><b>Drop a PDF here</b><p>PDF files up to 25 MB</p><label class="choose" for="file">Choose file</label><input id="file" type="file" accept="application/pdf"></div><div id="status" class="meta"></div><div id="docs" class="docs"></div></div></section><section class="card chat"><div class="head"><div><b>Document assistant</b><div class="meta">Concise answers with source attribution</div></div><button class="mini" onclick="clearChat()">Clear</button></div><div class="chatbody"><div id="welcome" class="welcome"><h2>What would you like to know?</h2><p>Select a document, then ask a question. The assistant prioritizes answer-focused evidence and abstains when the document does not support an answer.</p></div><div id="answer" class="answer"></div><div id="sourceWrap" style="display:none"><div class="source-title">Sources</div><div id="sources" class="sources"></div></div><div class="composer"><div class="box"><textarea id="question" placeholder="Ask a question about your documents…"></textarea><button class="send" onclick="ask()">↑</button></div><div class="hint">Enter to send · Shift + Enter for a new line</div></div></div></section></div><details class="card advanced"><summary>Quality & evaluation</summary><div class="metrics"><div class="metric"><span>Answerability</span><b id="mA">—</b></div><div class="metric"><span>Evidence coverage</span><b id="mC">—</b></div><div class="metric"><span>Answer mode</span><b id="mM">—</b></div></div></details></main></div><script>
let selectedDoc=null;const $=id=>document.getElementById(id);const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));async function loadDocs(){const r=await fetch('/api/documents'),d=await r.json();$('count').textContent=d.count+' document'+(d.count===1?'':'s');$('docs').innerHTML=d.documents.map(x=>`<div class="doc"><div class="docname">${esc(x.filename)}</div><div class="meta">${x.pages} pages · ${x.chunks} chunks</div><div class="actions"><button class="mini" onclick="useDoc('${x.id}')">Use document</button><button class="mini" onclick="removeDoc('${x.id}')">Remove</button></div></div>`).join('')||'<div class="meta">No documents indexed yet.</div>'}function useDoc(id){selectedDoc=id}async function removeDoc(id){if(!confirm('Remove this document?'))return;await fetch('/api/documents/'+id,{method:'DELETE'});if(selectedDoc===id)selectedDoc=null;loadDocs()}$('file').addEventListener('change',async e=>{const f=e.target.files[0];if(!f)return;const fd=new FormData();fd.append('file',f);$('status').textContent='Indexing…';const r=await fetch('/api/documents/upload',{method:'POST',body:fd});const d=await r.json();if(!r.ok){$('status').textContent=d.detail||'Upload failed';return}selectedDoc=d.document_id;$('status').textContent=`Indexed ${d.filename} · ${d.pages} pages`;loadDocs()});$('question').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();ask()}});async function ask(){const q=$('question').value.trim();if(!q)return;$('welcome').style.display='none';$('answer').classList.remove('show');$('sourceWrap').style.display='none';const r=await fetch('/api/ask',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({question:q,document_id:selectedDoc,top_k:5})});const d=await r.json();$('answer').textContent=d.answer;$('answer').classList.add('show');$('mA').textContent=d.answerable?'Yes':'No';$('mA').style.color=d.answerable?'var(--good)':'var(--warn)';$('mC').textContent=Math.round((d.evidence_coverage||0)*100)+'%';$('mM').textContent=d.mode||'—';if(d.citations?.length){$('sources').innerHTML=d.citations.map((c,i)=>`<div class="source"><b>📄 Source ${i+1} · ${esc(c.source)}</b><span>Pages ${esc(c.pages)}</span></div>`).join('');$('sourceWrap').style.display='block'}}function clearChat(){$('question').value='';$('answer').classList.remove('show');$('sourceWrap').style.display='none';$('welcome').style.display='block';['mA','mC','mM'].forEach(x=>$(x).textContent='—')}loadDocs();</script></body></html>'''


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return HTML
