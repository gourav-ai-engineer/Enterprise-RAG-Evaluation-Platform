from __future__ import annotations

import math
import os
import re
import sqlite3
import time
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

ROOT = Path(__file__).resolve().parent
DATA = Path(os.getenv("DATA_DIR", ROOT / "data"))
UPLOADS = DATA / "documents"
DB = DATA / "rag.db"
UPLOADS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Enterprise RAG Evaluation Platform", version="3.1.0", description="Hybrid retrieval → reranking → grounded LLM generation with evaluation telemetry.")
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
STOP = set("a an and are as at be been but by can could did do does for from how i if in into is it its me of on or our please should that the their them there these this to was were what when where which who why with would you your".split())
ABSTAIN = "I couldn't find enough relevant evidence in the indexed documents to answer that question."
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "180"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "20"))


def connect():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, filename TEXT NOT NULL, stored_path TEXT NOT NULL, pages INTEGER NOT NULL, chunks INTEGER NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
    c.execute("CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY, document_id TEXT NOT NULL, filename TEXT NOT NULL, page_start INTEGER NOT NULL, page_end INTEGER NOT NULL, text TEXT NOT NULL)")
    c.commit()
    return c


def tokens(text: str) -> list[str]:
    return [x.lower() for x in TOKEN_RE.findall(text or "")]


def content_tokens(text: str) -> list[str]:
    return [x for x in tokens(text) if x not in STOP and len(x) > 1]


def clean(text: str) -> str:
    return re.sub(r"(?<=\w)-\s+(?=\w)", "", re.sub(r"\s+", " ", text or "").strip())


def chunk_pages(pages: list[str]) -> list[tuple[int, int, str]]:
    chunks: list[tuple[int, int, str]] = []
    for page, raw in enumerate(pages, 1):
        paragraphs = [clean(p) for p in re.split(r"\n\s*\n+", raw or "") if clean(p)]
        if not paragraphs and clean(raw):
            paragraphs = [clean(raw)]
        words: list[str] = []
        for paragraph in paragraphs:
            pw = paragraph.split()
            while pw:
                room = CHUNK_WORDS - len(words)
                words.extend(pw[:room])
                pw = pw[room:]
                if len(words) >= CHUNK_WORDS:
                    chunks.append((page, page, " ".join(words)))
                    words = words[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else []
            if len(words) >= CHUNK_WORDS:
                chunks.append((page, page, " ".join(words)))
                words = words[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else []
        if words:
            chunks.append((page, page, " ".join(words)))
    return chunks


def load_chunks(document_id: str | None = None):
    c = connect()
    if document_id:
        rows = c.execute("SELECT * FROM chunks WHERE document_id=? ORDER BY rowid", (document_id,)).fetchall()
    else:
        rows = c.execute("SELECT * FROM chunks ORDER BY rowid").fetchall()
    c.close()
    return rows


def bm25(query: str, docs: list[str]) -> list[float]:
    terms = content_tokens(query)
    frequencies = [Counter(content_tokens(d)) for d in docs]
    dfs = Counter()
    lengths = []
    for f in frequencies:
        dfs.update(f.keys()); lengths.append(sum(f.values()))
    avg = sum(lengths) / max(1, len(lengths)); out = []
    for f, dl in zip(frequencies, lengths):
        score = 0.0
        for term in terms:
            if term not in f: continue
            idf = math.log(1 + (len(docs) - dfs[term] + 0.5) / (dfs[term] + 0.5))
            score += idf * (f[term] * 2.5) / (f[term] + 1.5 * (0.25 + 0.75 * dl / max(avg, 1)))
        out.append(score)
    return out


def normalize(values: list[float]) -> list[float]:
    maximum = max(values) if values else 0.0
    return [v / maximum if maximum else 0.0 for v in values]


def intent(question: str) -> str:
    q = question.lower()
    if re.search(r"\b(what is|what are|define|defined|meaning|refers to|denotes)\b", q): return "definition"
    if re.search(r"\b(how is|how are|how does|calculate|calculated|formula|equation)\b", q): return "method"
    if re.search(r"\b(why|advantage|benefit|purpose|problem)\b", q): return "purpose"
    if re.search(r"\b(compare|difference|differ|versus|vs)\b", q): return "comparison"
    return "general"


def retrieve(query: str, rows, k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows: return [], {"retrieval_ms": 0.0, "rerank_ms": 0.0, "candidate_count": 0, "reranker": "hybrid_cross_signal"}
    t0 = time.perf_counter(); texts = [r["text"] for r in rows]
    b = normalize(bm25(query, texts))
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vec.fit_transform(texts); tf = normalize(cosine_similarity(vec.transform([query]), matrix)[0].tolist())
    candidates = []
    for row, bb, tt in zip(rows, b, tf): candidates.append((0.55 * bb + 0.45 * tt, row, bb, tt))
    candidates.sort(key=lambda x: (-x[0], x[1]["id"]))
    retrieval_ms = (time.perf_counter() - t0) * 1000
    candidates = candidates[:min(CANDIDATE_K, max(k * 4, k), len(candidates))]
    t1 = time.perf_counter(); qterms = set(content_tokens(query)); qintent = intent(query); scored = []
    for base, row, bb, tt in candidates:
        text = row["text"]; low = text.lower(); terms = set(content_tokens(text))
        overlap = len(qterms & terms) / max(1, len(qterms))
        phrase = sum(1 for x in qterms if x in low) / max(1, len(qterms))
        score = 0.45 * base + 0.35 * overlap + 0.20 * phrase
        if qintent == "definition" and re.search(r"\b(is|are|refers to|defined as|denotes|we propose|we present)\b", low): score += 0.18
        if qintent == "method" and re.search(r"\b(compute|computed|calculate|formula|equation|using|defined)\b", low): score += 0.12
        if qintent == "purpose" and re.search(r"\b(problem|purpose|advantage|benefit|limitation|address)\b", low): score += 0.10
        if sum(c.isdigit() for c in text) / max(1, len(text)) > 0.08: score -= 0.06
        scored.append((score, row, bb, tt))
    scored.sort(key=lambda x: (-x[0], x[1]["id"]))
    out = []; seen = set()
    for score, row, bb, tt in scored:
        key = (row["document_id"], row["page_start"], row["page_end"], row["text"][:100])
        if key in seen: continue
        seen.add(key)
        out.append({"chunk_id": row["id"], "document_id": row["document_id"], "filename": row["filename"], "page_start": row["page_start"], "page_end": row["page_end"], "score": round(score, 4), "hybrid_score": round(0.55 * bb + 0.45 * tt, 4), "bm25": round(bb, 4), "tfidf": round(tt, 4), "text": row["text"]})
        if len(out) >= k: break
    return out, {"retrieval_ms": round(retrieval_ms, 3), "rerank_ms": round((time.perf_counter() - t1) * 1000, 3), "candidate_count": len(candidates), "reranker": "hybrid_cross_signal"}


def coverage(question: str, results: list[dict[str, Any]]) -> float:
    q = set(content_tokens(question)); c = set(content_tokens(" ".join(r["text"] for r in results)))
    return len(q & c) / max(1, len(q))


def answerable(question: str, results: list[dict[str, Any]]) -> bool:
    return bool(results) and coverage(question, results) >= 0.30 and results[0]["score"] >= 0.15


def extractive(question: str, results: list[dict[str, Any]]) -> str:
    if not answerable(question, results): return ABSTAIN
    qterms = set(content_tokens(question)); candidates = []
    for rank, result in enumerate(results):
        for sentence in re.split(r"(?<=[.!?])\s+", result["text"]):
            sentence = clean(sentence); overlap = len(qterms & set(content_tokens(sentence)))
            if not overlap: continue
            score = overlap * 4 - rank
            if re.search(r"\b(is|are|refers to|defined as|denotes|we propose|we present|combines|using|formula)\b", sentence.lower()): score += 5
            if len(sentence.split()) < 8: score -= 2
            candidates.append((score, sentence))
    candidates.sort(reverse=True); selected=[]; seen=set()
    for _, s in candidates:
        key=" ".join(content_tokens(s))
        if key in seen: continue
        seen.add(key); selected.append(s)
        if len(selected) == 2: break
    return " ".join(selected) if selected else ABSTAIN


def context(results):
    return "\n\n".join(f"[Source {i}] {r['filename']} pages {r['page_start']}-{r['page_end']}\n{r['text']}" for i, r in enumerate(results[:5], 1))


def generate(question: str, results: list[dict[str, Any]]) -> tuple[str, str, float]:
    if not answerable(question, results): return ABSTAIN, "abstain", 0.0
    key = os.getenv("OPENAI_API_KEY")
    if not key: return extractive(question, results), "extractive_fallback", 0.0
    start = time.perf_counter(); base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"); model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    payload = {"model": model, "temperature": 0, "messages": [{"role": "system", "content": "You are an enterprise RAG assistant. Answer ONLY from the reranked evidence. Do not use outside knowledge or invent facts. Answer in 1-4 concise sentences. Cite factual claims using [Source N]. If evidence is insufficient, abstain."}, {"role": "user", "content": f"Question: {question}\n\nReranked evidence:\n{context(results)}"}]}
    try:
        response = httpx.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=60); response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip(), "llm", round((time.perf_counter() - start) * 1000, 3)
    except Exception:
        return extractive(question, results), "extractive_fallback", round((time.perf_counter() - start) * 1000, 3)


def citation_groups(results):
    out=[]; seen=set()
    for r in results:
        key=(r["filename"],r["page_start"],r["page_end"])
        if key in seen: continue
        seen.add(key); out.append({"source":r["filename"],"pages":f"{r['page_start']}-{r['page_end']}"})
    return out[:5]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)


@app.on_event("startup")
def startup(): connect().close()

@app.get("/health")
def health(): return {"status":"ok","service":"enterprise-rag-evaluation-platform","version":"3.1.0"}

@app.get("/api/documents")
def documents():
    c=connect(); rows=c.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall(); c.close(); return {"count":len(rows),"documents":[dict(r) for r in rows]}

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf": raise HTTPException(400,"Only PDF files are supported.")
    data=await file.read(); limit=int(os.getenv("MAX_FILE_MB","25"))
    if len(data)>limit*1024*1024: raise HTTPException(413,f"PDF exceeds {limit} MB limit.")
    docid=uuid.uuid4().hex; path=UPLOADS/f"{docid}.pdf"; path.write_bytes(data)
    try: pages=[p.extract_text() or "" for p in PdfReader(str(path)).pages]
    except Exception as exc: path.unlink(missing_ok=True); raise HTTPException(400,f"Could not read PDF: {exc}") from exc
    chunks=chunk_pages(pages); name=file.filename or "document.pdf"; c=connect()
    c.execute("INSERT INTO documents(id,filename,stored_path,pages,chunks) VALUES(?,?,?,?,?)",(docid,name,str(path),len(pages),len(chunks)))
    c.executemany("INSERT INTO chunks(id,document_id,filename,page_start,page_end,text) VALUES(?,?,?,?,?,?)",[(uuid.uuid4().hex,docid,name,a,b,t) for a,b,t in chunks]); c.commit(); c.close()
    return {"document_id":docid,"filename":name,"pages":len(pages),"chunks":len(chunks)}

@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str):
    c=connect(); row=c.execute("SELECT stored_path FROM documents WHERE id=?",(document_id,)).fetchone()
    if not row: c.close(); raise HTTPException(404,"Document not found")
    c.execute("DELETE FROM chunks WHERE document_id=?",(document_id,)); c.execute("DELETE FROM documents WHERE id=?",(document_id,)); c.commit(); c.close(); Path(row["stored_path"]).unlink(missing_ok=True); return {"status":"deleted"}

@app.get("/api/retrieve")
def retrieve_api(q: str = Query(min_length=3,max_length=2000), document_id: str | None = None, k: int = Query(default=5,ge=1,le=10)):
    results,lat=retrieve(q,load_chunks(document_id),k); return {"query":q,"answerable":answerable(q,results),"evidence_coverage":round(coverage(q,results),4),"latency":lat,"results":results}

@app.post("/api/ask")
def ask(body: AskRequest):
    start=time.perf_counter(); results,lat=retrieve(body.question,load_chunks(body.document_id),body.top_k); answer,mode,generation_ms=generate(body.question,results); lat["generation_ms"]=generation_ms; lat["context_chunks"]=min(5,len(results)); lat["total_ms"]=round((time.perf_counter()-start)*1000,3); ok=answerable(body.question,results)
    return {"question":body.question,"answer":answer,"mode":mode,"answerable":ok,"evidence_coverage":round(coverage(body.question,results),4),"citations":citation_groups(results) if ok else [],"latency":lat,"retrieved":results}


HTML=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Enterprise RAG</title><style>:root{color-scheme:dark;--bg:#080d18;--panel:#101827;--line:#293852;--text:#f5f7fb;--muted:#8997ad;--accent:#7c86f7}*{box-sizing:border-box}body{margin:0;background:radial-gradient(900px 500px at 10% -10%,#18284b,transparent 55%),var(--bg);color:var(--text);font:14px Inter,system-ui,sans-serif}.top{height:70px;border-bottom:1px solid var(--line);display:flex;align-items:center;padding:0 30px}.brand{font-weight:800;font-size:16px}.pill{margin-left:auto;border:1px solid var(--line);border-radius:999px;padding:6px 10px;color:var(--muted);font-size:11px}.layout{max-width:1440px;margin:auto;display:grid;grid-template-columns:290px 1fr;min-height:calc(100vh - 70px)}aside{border-right:1px solid var(--line);padding:25px 18px}.nav{padding:12px;border-radius:10px;color:var(--muted);margin-bottom:4px}.nav.active{background:#18233a;color:#fff}.note{margin-top:25px;border:1px solid var(--line);border-radius:12px;padding:14px;color:var(--muted);font-size:12px;line-height:1.6}.main{padding:30px}.hero{display:flex;justify-content:space-between;align-items:end;margin-bottom:22px}.hero h1{margin:6px 0;font-size:29px}.muted{color:var(--muted);font-size:12px}.grid{display:grid;grid-template-columns:340px 1fr;gap:18px}.card{background:#101827e8;border:1px solid var(--line);border-radius:16px;overflow:hidden}.head{padding:17px 18px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between}.body{padding:18px}.drop{border:1px dashed #425372;background:#0b1422;border-radius:12px;padding:23px;text-align:center}.drop input{margin-top:12px;max-width:100%}.docs{margin-top:12px}.doc{padding:13px 0;border-bottom:1px solid #202e44}.doc b{font-size:12px}.btn{margin-top:8px;border:1px solid var(--line);background:#0d1626;color:#dce5f4;border-radius:8px;padding:7px 9px;cursor:pointer;font-size:11px}.chat{min-height:650px}.chatbody{padding:22px;display:flex;flex-direction:column;min-height:580px}.answer{margin-top:18px;background:#0b1422;border:1px solid #31415c;border-radius:14px;padding:19px;line-height:1.7;white-space:pre-wrap}.sources{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}.source{border:1px solid var(--line);border-radius:10px;background:#0c1422;padding:10px;font-size:11px}.composer{margin-top:auto;display:flex;gap:9px}.composer textarea{flex:1;min-height:58px;resize:none;background:#0b1422;border:1px solid #31415c;border-radius:12px;color:#fff;padding:13px}.send{width:48px;border:0;border-radius:10px;background:linear-gradient(135deg,#718cff,#8b70e8);color:white;font-weight:800}.metrics{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:15px}.metric{background:#0c1422;border:1px solid var(--line);border-radius:10px;padding:10px}.metric span{display:block;color:#718099;font-size:10px}.metric b{display:block;margin-top:4px;font-size:12px}@media(max-width:1000px){.layout{grid-template-columns:1fr}aside{display:none}.grid{grid-template-columns:1fr}.hero{align-items:flex-start;flex-direction:column}.metrics{grid-template-columns:1fr 1fr}}</style></head><body><header class="top"><div class="brand">✦ Enterprise RAG</div><div class="pill">Hybrid → Rerank → Grounded LLM</div></header><div class="layout"><aside><div class="nav active">⌂ Knowledge assistant</div><div class="nav">▣ Documents</div><div class="nav">✓ Evaluation</div><div class="nav">↳ Hybrid retrieval</div><div class="nav">⊙ Grounding guard</div><div class="note"><b>Production pipeline</b><br>Queries search the selected document or the full knowledge base, rerank candidate chunks, then pass only the best evidence to the generator.</div></aside><main class="main"><div class="hero"><div><div class="muted">KNOWLEDGE WORKBENCH</div><h1>Ask your enterprise documents</h1><div class="muted">Multi-document retrieval, reranking, grounded generation and latency telemetry.</div></div></div><div class="grid"><section class="card"><div class="head"><b>Knowledge base</b><span id="count" class="muted">0</span></div><div class="body"><div class="drop"><b>Add PDF</b><div class="muted">Paragraph-aware chunks · overlap · page tracking</div><input id="file" type="file" accept="application/pdf"></div><div id="docs" class="docs"></div></div></section><section class="card chat"><div class="head"><div><b>Document assistant</b><div class="muted">All documents by default</div></div><button class="btn" onclick="clearChat()">Clear</button></div><div class="chatbody"><div id="answer"></div><div id="sources" class="sources"></div><div id="metrics"></div><div class="composer"><textarea id="q" placeholder="Ask a question…"></textarea><button class="send" onclick="ask()">↑</button></div></div></section></div></main></div><script>let selected=null;const $=x=>document.getElementById(x),esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));async function load(){const d=await(await fetch('/api/documents')).json();$('count').textContent=d.count+' document'+(d.count===1?'':'s');$('docs').innerHTML=d.documents.map(x=>`<div class="doc"><b>${esc(x.filename)}</b><div class="muted">${x.pages} pages · ${x.chunks} chunks</div><button class="btn" onclick="selected='${x.id}'">Use document</button> <button class="btn" onclick="removeDoc('${x.id}')">Remove</button></div>`).join('')||'<div class="muted">No documents indexed.</div>'}async function removeDoc(id){await fetch('/api/documents/'+id,{method:'DELETE'});if(selected===id)selected=null;load()}$('file').onchange=async()=>{const f=$('file').files[0];if(!f)return;const fd=new FormData();fd.append('file',f);await fetch('/api/documents/upload',{method:'POST',body:fd});load()};$('q').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();ask()}});async function ask(){const q=$('q').value.trim();if(!q)return;$('answer').innerHTML='<div class="answer">Retrieving candidates → hybrid ranking → reranking → grounded generation…</div>';$('sources').innerHTML='';const d=await(await fetch('/api/ask',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({question:q,document_id:selected,top_k:5})})).json();$('answer').innerHTML=`<div class="answer">${esc(d.answer)}</div>`;if(d.citations?.length)$('sources').innerHTML=d.citations.map((x,i)=>`<div class="source"><b>Source ${i+1}</b><br>${esc(x.source)}<br><span class="muted">Pages ${esc(x.pages)}</span></div>`).join('');const l=d.latency||{};$('metrics').innerHTML=`<div class="metrics"><div class="metric"><span>Retrieval</span><b>${l.retrieval_ms??'-'} ms</b></div><div class="metric"><span>Rerank</span><b>${l.rerank_ms??'-'} ms</b></div><div class="metric"><span>Generation</span><b>${l.generation_ms??'-'} ms</b></div><div class="metric"><span>Total</span><b>${l.total_ms??'-'} ms</b></div><div class="metric"><span>Answerable</span><b>${d.answerable?'Yes':'No'}</b></div></div>`}function clearChat(){$('q').value='';$('answer').innerHTML='';$('sources').innerHTML='';$('metrics').innerHTML=''}load();</script></body></html>'''

@app.get("/", response_class=HTMLResponse)
def home(): return HTML
