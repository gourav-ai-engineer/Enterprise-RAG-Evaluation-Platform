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

app = FastAPI(title="Enterprise RAG Evaluation Platform", version="2.1.0", description="Dynamic PDF ingestion, hybrid retrieval, citations, abstention and offline evaluation.")
TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "25"))
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "180"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))
STOPWORDS = {"a","an","and","are","as","at","be","been","but","by","can","could","did","do","does","for","from","how","i","if","in","into","is","it","its","me","of","on","or","our","please","should","that","the","their","them","there","these","this","to","was","were","what","when","where","which","who","why","with","would","you","your"}
ABSTAIN_MESSAGE = "I couldn't find enough relevant evidence in the indexed documents to answer that question."


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, filename TEXT NOT NULL, stored_path TEXT NOT NULL, pages INTEGER NOT NULL, chunks INTEGER NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
    conn.execute("CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY, document_id TEXT NOT NULL, filename TEXT NOT NULL, page_start INTEGER NOT NULL, page_end INTEGER NOT NULL, text TEXT NOT NULL, FOREIGN KEY(document_id) REFERENCES documents(id))")
    conn.commit()
    return conn


def tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOPWORDS and len(t) > 1]


def chunk_pages(pages: list[str]) -> list[tuple[int, int, str]]:
    chunks=[]; current=[]; start_page=1; current_words=0
    for page_no,page_text in enumerate(pages,1):
        words=page_text.split()
        while words:
            take=max(1,min(CHUNK_WORDS-current_words,len(words))); current.extend(words[:take]); words=words[take:]; current_words+=take
            if current_words>=CHUNK_WORDS:
                chunks.append((start_page,page_no," ".join(current))); current=current[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else []; current_words=len(current); start_page=page_no
    if current: chunks.append((start_page,len(pages)," ".join(current)))
    return chunks


def load_chunks(document_id: str | None=None) -> list[sqlite3.Row]:
    conn=db()
    rows=conn.execute("SELECT * FROM chunks WHERE document_id=?",(document_id,)).fetchall() if document_id else conn.execute("SELECT * FROM chunks ORDER BY rowid").fetchall()
    conn.close(); return rows


def bm25_scores(query: str, docs: list[str]) -> list[float]:
    q=content_tokens(query)
    if not docs or not q: return [0.0]*len(docs)
    term_df=Counter(); frequencies=[]; lengths=[]
    for doc in docs:
        tf=Counter(content_tokens(doc)); frequencies.append(tf); lengths.append(sum(tf.values())); term_df.update(tf.keys())
    avgdl=sum(lengths)/max(1,len(lengths)); n=len(docs); k1,b=1.5,0.75; scores=[]
    for tf,dl in zip(frequencies,lengths):
        score=0.0
        for term in q:
            if term not in tf: continue
            df=term_df[term]; idf=math.log(1+(n-df+0.5)/(df+0.5)); score+=idf*((tf[term]*(k1+1))/(tf[term]+k1*(1-b+b*dl/max(1,avgdl))))
        scores.append(score)
    return scores


def hybrid_retrieve(query: str, rows: list[sqlite3.Row], k: int) -> list[dict[str,Any]]:
    if not rows: return []
    texts=[r["text"] for r in rows]; lexical=bm25_scores(query,texts)
    vectorizer=TfidfVectorizer(lowercase=True,stop_words="english",ngram_range=(1,2),min_df=1); matrix=vectorizer.fit_transform(texts); dense=cosine_similarity(vectorizer.transform([query]),matrix)[0].tolist()
    def normalize(values):
        mx=max(values) if values else 0.0; return [v/mx if mx else 0.0 for v in values]
    lex_n,dense_n=normalize(lexical),normalize(dense); scored=[]
    for row,b,d in zip(rows,lex_n,dense_n): scored.append((0.5*b+0.5*d,row,b,d))
    scored.sort(key=lambda x:(-x[0],x[1]["id"]))
    return [{"chunk_id":row["id"],"document_id":row["document_id"],"filename":row["filename"],"page_start":row["page_start"],"page_end":row["page_end"],"score":round(score,4),"bm25":round(b,4),"tfidf":round(d,4),"text":row["text"]} for score,row,b,d in scored[:k]]


def evidence_coverage(question: str, results: list[dict[str,Any]]) -> float:
    q=set(content_tokens(question))
    if not q or not results: return 0.0
    c=set(content_tokens(" ".join(r["text"] for r in results))); return len(q&c)/len(q)


def has_sufficient_evidence(question: str, results: list[dict[str,Any]]) -> bool:
    q=set(content_tokens(question))
    if not q or not results: return False
    coverage=evidence_coverage(question,results); return math.floor(coverage*len(q))>=max(1,math.ceil(len(q)*0.34))


class AskRequest(BaseModel):
    question: str = Field(min_length=3,max_length=2000)
    document_id: str | None = None
    top_k: int = Field(default=5,ge=1,le=10)


class EvaluateRequest(BaseModel):
    question: str = Field(min_length=3,max_length=2000)
    answer: str = Field(min_length=1,max_length=10000)
    document_id: str | None = None
    reference_answer: str | None = Field(default=None,max_length=10000)
    top_k: int = Field(default=5,ge=1,le=10)


def extractive_answer(question: str, results: list[dict[str,Any]]) -> str:
    if not has_sufficient_evidence(question,results): return ABSTAIN_MESSAGE
    q=set(content_tokens(question)); ranked=[]
    for result in results:
        for sentence in re.split(r"(?<=[.!?])\s+",result["text"]):
            overlap=len(q&set(content_tokens(sentence)))
            if overlap: ranked.append((overlap,sentence.strip()))
    best=[s for _,s in sorted(ranked,key=lambda x:-x[0])[:4]]; return " ".join(best) if best else ABSTAIN_MESSAGE


def llm_answer(question: str, results: list[dict[str,Any]]) -> tuple[str,str]:
    if not has_sufficient_evidence(question,results): return ABSTAIN_MESSAGE,"abstain"
    api_key=os.getenv("OPENAI_API_KEY")
    if not api_key: return extractive_answer(question,results),"extractive"
    base=os.getenv("OPENAI_BASE_URL","https://api.openai.com/v1").rstrip("/"); model=os.getenv("LLM_MODEL","gpt-4o-mini")
    context="\n\n".join(f"[Source {i}] {r['filename']} pages {r['page_start']}-{r['page_end']}\n{r['text']}" for i,r in enumerate(results,1))
    payload={"model":model,"messages":[{"role":"system","content":"You are a grounded enterprise RAG assistant. Never invent facts."},{"role":"user","content":"Answer only from the supplied context. Cite claims using [Source N]. If the context does not contain the answer, say you cannot find it.\n\nQuestion: "+question+"\n\nContext:\n"+context}],"temperature":0}
    try:
        response=httpx.post(f"{base}/chat/completions",headers={"Authorization":f"Bearer {api_key}"},json=payload,timeout=60); response.raise_for_status(); return response.json()["choices"][0]["message"]["content"],"llm"
    except Exception: return extractive_answer(question,results),"extractive_fallback"


def offline_metrics(question: str, answer: str, results: list[dict[str,Any]], reference: str|None) -> dict[str,float|None]:
    context=" ".join(r["text"] for r in results); a,c,q=set(tokens(answer)),set(tokens(context)),set(tokens(question)); grounded=len(a&c)/max(1,len(a)); relevance=len(q&a)/max(1,len(q)); ref_score=None
    if reference:
        ref=set(tokens(reference)); ref_score=len(ref&a)/max(1,len(ref))
    values=[grounded,relevance]+([] if ref_score is None else [ref_score])
    return {"groundedness":round(grounded,4),"answer_relevance":round(relevance,4),"reference_overlap":None if ref_score is None else round(ref_score,4),"overall":round(sum(values)/len(values),4)}


@app.on_event("startup")
def startup(): db().close()


@app.get("/health")
def health(): return {"status":"ok","service":"enterprise-rag-evaluation-platform"}


@app.get("/api/documents")
def documents():
    conn=db(); rows=conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall(); conn.close(); return {"count":len(rows),"documents":[dict(r) for r in rows]}


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile=File(...)):
    if file.content_type!="application/pdf": raise HTTPException(400,"Only PDF files are supported.")
    content=await file.read()
    if len(content)>MAX_FILE_MB*1024*1024: raise HTTPException(413,f"PDF exceeds {MAX_FILE_MB} MB limit.")
    document_id=uuid.uuid4().hex; stored=UPLOAD_DIR/f"{document_id}.pdf"; stored.write_bytes(content)
    try:
        reader=PdfReader(str(stored)); pages=[page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        stored.unlink(missing_ok=True); raise HTTPException(400,f"Could not read PDF: {exc}") from exc
    chunks=chunk_pages(pages); conn=db(); filename=file.filename or "document.pdf"
    conn.execute("INSERT INTO documents(id,filename,stored_path,pages,chunks) VALUES(?,?,?,?,?)",(document_id,filename,str(stored),len(pages),len(chunks)))
    conn.executemany("INSERT INTO chunks(id,document_id,filename,page_start,page_end,text) VALUES(?,?,?,?,?,?)",[(uuid.uuid4().hex,document_id,filename,s,e,t) for s,e,t in chunks]); conn.commit(); conn.close()
    return {"document_id":document_id,"filename":filename,"pages":len(pages),"chunks":len(chunks)}


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str):
    conn=db(); row=conn.execute("SELECT stored_path FROM documents WHERE id=?",(document_id,)).fetchone()
    if not row: conn.close(); raise HTTPException(404,"Document not found")
    conn.execute("DELETE FROM chunks WHERE document_id=?",(document_id,)); conn.execute("DELETE FROM documents WHERE id=?",(document_id,)); conn.commit(); conn.close(); Path(row["stored_path"]).unlink(missing_ok=True); return {"status":"deleted"}


@app.get("/api/retrieve")
def retrieve(q: str=Query(min_length=3,max_length=2000),document_id: str|None=None,k: int=Query(default=5,ge=1,le=10)):
    results=hybrid_retrieve(q,load_chunks(document_id),k); return {"query":q,"evidence_coverage":round(evidence_coverage(q,results),4),"answerable":has_sufficient_evidence(q,results),"results":results}


@app.post("/api/ask")
def ask(body: AskRequest):
    results=hybrid_retrieve(body.question,load_chunks(body.document_id),body.top_k); answer,mode=llm_answer(body.question,results); answerable=has_sufficient_evidence(body.question,results)
    return {"question":body.question,"answer":answer,"mode":mode,"answerable":answerable,"evidence_coverage":round(evidence_coverage(body.question,results),4),"citations":[{"source":r["filename"],"pages":f"{r['page_start']}-{r['page_end']}","score":r["score"]} for r in results] if answerable else [],"retrieved":results}


@app.post("/api/evaluate")
def evaluate(body: EvaluateRequest):
    results=hybrid_retrieve(body.question,load_chunks(body.document_id),body.top_k); return {"metrics":offline_metrics(body.question,body.answer,results,body.reference_answer),"retrieved":results}


HTML=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Enterprise RAG | Knowledge Workbench</title><style>
:root{color-scheme:dark;--bg:#080d18;--panel:#101827;--panel2:#0c1422;--line:#243248;--text:#f3f6fb;--muted:#8d9ab0;--accent:#7c9cff;--accent2:#9a7cff;--good:#46d39a;--warn:#f2c66d}*{box-sizing:border-box}body{margin:0;background:radial-gradient(1000px 500px at 15% -10%,#17264a 0,transparent 55%),var(--bg);font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--text)}button,input,textarea{font:inherit}button{cursor:pointer}.topbar{height:68px;border-bottom:1px solid var(--line);background:#080d18dd;backdrop-filter:blur(14px);display:flex;align-items:center;padding:0 32px;position:sticky;top:0;z-index:10}.brand{display:flex;align-items:center;gap:12px;font-weight:750}.logo{width:32px;height:32px;border-radius:9px;background:linear-gradient(135deg,var(--accent),var(--accent2));display:grid;place-items:center}.topmeta{margin-left:auto;display:flex;gap:10px}.badge{font-size:12px;color:#aebbd0;border:1px solid var(--line);background:#0d1524;padding:6px 10px;border-radius:999px}.dot{width:7px;height:7px;background:var(--good);border-radius:50%;display:inline-block;margin-right:6px}.layout{display:grid;grid-template-columns:280px minmax(0,1fr);max-width:1440px;margin:auto;min-height:calc(100vh - 68px)}.sidebar{border-right:1px solid var(--line);padding:26px 18px}.side-label{font-size:11px;color:#66758d;text-transform:uppercase;letter-spacing:.12em;margin:8px 10px 10px}.nav{display:flex;gap:10px;align-items:center;padding:11px 12px;border-radius:10px;color:#9eabc0;font-size:14px;margin-bottom:4px}.nav.active{background:#18233a;color:#fff}.side-note{margin-top:28px;padding:14px;border:1px solid var(--line);background:#0c1422;border-radius:12px}.side-note b{font-size:12px}.side-note p{color:var(--muted);font-size:12px;line-height:1.55;margin:8px 0 0}.content{padding:34px;min-width:0}.hero{display:flex;justify-content:space-between;gap:20px;align-items:flex-end;margin-bottom:26px}.eyebrow{font-size:11px;color:#8291aa;text-transform:uppercase;letter-spacing:.14em}.hero h1{font-size:30px;line-height:1.1;margin:8px 0;letter-spacing:-.04em}.hero p{color:var(--muted);margin:0;max-width:720px;font-size:14px}.primary{border:0;background:linear-gradient(135deg,#718cff,#8a73e8);color:white;padding:11px 16px;border-radius:10px;font-weight:700}.grid{display:grid;grid-template-columns:330px minmax(0,1fr);gap:18px;align-items:start}.card{border:1px solid var(--line);background:#101827e6;border-radius:16px;overflow:hidden}.card-head{padding:18px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center}.card-head h2{font-size:14px;margin:0}.sub{font-size:12px;color:var(--muted);margin-top:4px}.card-body{padding:18px}.drop{border:1px dashed #40516f;background:#0b1422;border-radius:12px;padding:24px 16px;text-align:center}.drop.drag{border-color:var(--accent);background:#101c32}.upload-icon{width:42px;height:42px;border-radius:12px;background:#17243c;margin:0 auto 12px;display:grid;place-items:center;color:#aebcff}.drop strong{font-size:13px}.drop p{font-size:12px;color:var(--muted);margin:7px 0 14px}.drop input{display:none}.choose{display:inline-flex;padding:9px 13px;border:1px solid var(--line);border-radius:9px;background:#111c2d;font-size:12px}.status{font-size:12px;color:var(--muted);margin:12px 0}.doc-list{max-height:390px;overflow:auto}.doc{padding:13px 0;border-bottom:1px solid #1d293c}.doc-name{font-size:13px;font-weight:650;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.doc-meta{font-size:11px;color:#718099;margin-top:5px}.doc-actions{display:flex;gap:7px;margin-top:9px}.mini{border:1px solid var(--line);background:#0d1626;color:#aebbd0;border-radius:7px;padding:6px 8px;font-size:11px}.mini.use{color:#dce4ff}.empty{color:var(--muted);font-size:12px;padding:14px 0}.chat-card{min-height:620px;display:flex;flex-direction:column}.chat-head{display:flex;align-items:center;justify-content:space-between;padding:18px;border-bottom:1px solid var(--line)}.chat-title{display:flex;gap:11px;align-items:center}.avatar{width:34px;height:34px;border-radius:10px;background:#172540;display:grid;place-items:center;color:#aabaff}.chat-title b{font-size:13px}.chat-title span{display:block;color:#718099;font-size:11px;margin-top:3px}.chat-body{padding:26px;display:flex;flex-direction:column;flex:1}.welcome{max-width:650px;margin:30px auto;text-align:center}.big-icon{width:54px;height:54px;border-radius:15px;background:#17243b;display:grid;place-items:center;margin:0 auto 15px}.welcome h3{font-size:20px;margin:0 0 8px}.welcome p{color:var(--muted);font-size:13px;line-height:1.6}.answer{background:#0c1422;border:1px solid #24334c;border-radius:14px;padding:18px;line-height:1.7;font-size:14px;white-space:pre-wrap;display:none}.answer.show{display:block}.sources{margin-top:18px;display:none}.sources.show{display:block}.sources h4{font-size:12px;margin:0 0 9px;color:#aebbd0}.source-list{display:flex;gap:9px;flex-wrap:wrap}.source{border:1px solid var(--line);background:#0c1422;border-radius:10px;padding:10px 12px;min-width:170px}.source b{font-size:11px;display:block;max-width:230px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.source span{font-size:10px;color:#78869d;display:block;margin-top:4px}.composer{margin-top:auto;padding-top:20px}.composer-box{border:1px solid #31425f;background:#0b1422;border-radius:13px;padding:10px;display:flex;align-items:flex-end;gap:10px}.composer textarea{flex:1;resize:none;border:0;outline:0;background:transparent;color:#fff;min-height:44px;max-height:140px;padding:7px;font-size:13px}.send{width:42px;height:42px;border:0;border-radius:10px;background:linear-gradient(135deg,#718cff,#8a73e8);color:#fff}.hint{font-size:10px;color:#66758d;margin-top:7px}.advanced{margin-top:18px}.advanced summary{cursor:pointer;list-style:none;padding:15px 18px;font-size:12px;color:#9eabc0}.advanced summary::-webkit-details-marker{display:none}.advanced-body{padding:0 18px 18px}.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:9px}.metric{border:1px solid var(--line);background:#0c1422;border-radius:10px;padding:12px}.metric span{display:block;color:#72819a;font-size:10px;text-transform:uppercase;letter-spacing:.08em}.metric b{display:block;margin-top:6px;font-size:17px}.toast{position:fixed;right:22px;bottom:22px;padding:12px 15px;border:1px solid var(--line);background:#101827;border-radius:10px;font-size:12px;box-shadow:0 18px 50px #0007;transform:translateY(20px);opacity:0;pointer-events:none;transition:.2s}.toast.show{transform:none;opacity:1}.toast.error{border-color:#633c3c;color:#ffc1c1}@media(max-width:1050px){.layout{grid-template-columns:1fr}.sidebar{display:none}.grid{grid-template-columns:1fr}.content{padding:24px 18px}.hero{align-items:flex-start;flex-direction:column}.metric-grid{grid-template-columns:repeat(2,1fr)}}
</style></head><body><header class="topbar"><div class="brand"><div class="logo">R</div><span>Enterprise RAG</span></div><div class="topmeta"><span class="badge"><span class="dot"></span>System healthy</span><span class="badge">Local workspace</span></div></header><div class="layout"><aside class="sidebar"><div class="side-label">Workspace</div><div class="nav active">⌂&nbsp; Knowledge assistant</div><div class="nav">▣&nbsp; Documents</div><div class="nav">✓&nbsp; Evaluation</div><div class="side-label" style="margin-top:26px">Pipeline</div><div class="nav">↳&nbsp; Hybrid retrieval</div><div class="nav">⊙&nbsp; Grounding guard</div><div class="side-note"><b>Built for verifiable answers</b><p>Answers are grounded in indexed documents and expose page-level sources. When evidence is weak, the assistant abstains instead of guessing.</p></div></aside><main class="content"><div class="hero"><div><div class="eyebrow">Knowledge workbench</div><h1>Ask your enterprise documents</h1><p>Upload PDFs, search your knowledge base, and get concise answers with traceable sources.</p></div><button class="primary" onclick="document.getElementById('file').click()">+ Add document</button></div><div class="grid"><section class="card"><div class="card-head"><div><h2>Knowledge base</h2><div class="sub" id="docCount">0 documents</div></div><button class="mini" onclick="loadDocs()">Refresh</button></div><div class="card-body"><div class="drop" id="drop"><div class="upload-icon">↑</div><strong>Drop a PDF here</strong><p>PDF files up to 25 MB</p><label class="choose" for="file">Choose file</label><input id="file" type="file" accept="application/pdf"></div><div id="uploadStatus" class="status"></div><div class="doc-list" id="docs"><div class="empty">No documents indexed yet.</div></div></div></section><section class="card chat-card"><div class="chat-head"><div class="chat-title"><div class="avatar">✦</div><div><b>Document assistant</b><span>Grounded answers with source attribution</span></div></div><button class="mini" onclick="clearChat()">Clear</button></div><div class="chat-body"><div class="welcome" id="welcome"><div class="big-icon">✦</div><h3>What would you like to know?</h3><p>Ask a question about the selected document. The assistant will only answer when indexed evidence is strong enough.</p></div><div class="answer" id="answer"></div><div class="sources" id="sources"><h4>Sources</h4><div class="source-list" id="sourceList"></div></div><div class="composer"><div class="composer-box"><textarea id="question" rows="1" placeholder="Ask a question about your documents…"></textarea><button class="send" onclick="ask()">↑</button></div><div class="hint">Enter to send · Shift + Enter for a new line</div></div></div></section></div><details class="card advanced"><summary>Quality & evaluation</summary><div class="advanced-body"><div class="metric-grid"><div class="metric"><span>Answerability</span><b id="mAnswerable">—</b></div><div class="metric"><span>Evidence coverage</span><b id="mCoverage">—</b></div><div class="metric"><span>Answer mode</span><b id="mMode">—</b></div><div class="metric"><span>Overall eval</span><b id="mOverall">—</b></div></div></div></details></main></div><div class="toast" id="toast"></div><script>
const $=id=>document.getElementById(id);let selectedDoc=null;function toast(msg,error=false){const t=$("toast");t.textContent=msg;t.className="toast show"+(error?" error":"");setTimeout(()=>t.className="toast",2600)}function esc(s){return String(s).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#039;")}function short(s){return s.length>34?s.slice(0,31)+"…":s}
async function loadDocs(){try{const r=await fetch("/api/documents"),d=await r.json();$("docCount").textContent=`${d.count} document${d.count===1?"":"s"}`;$("docs").innerHTML=d.documents.length?d.documents.map(x=>`<div class="doc"><div class="doc-name" title="${esc(x.filename)}">${esc(short(x.filename))}</div><div class="doc-meta">${x.pages} pages · indexed</div><div class="doc-actions"><button class="mini use" onclick="useDoc('${x.id}')">Use document</button><button class="mini" onclick="removeDoc('${x.id}')">Remove</button></div></div>`).join(""):"<div class=\"empty\">No documents indexed yet.</div>"}catch(e){toast("Could not load documents",true)}}
function useDoc(id){selectedDoc=id;toast("Document selected")}
async function removeDoc(id){if(!confirm("Remove this document from the knowledge base?"))return;const r=await fetch("/api/documents/"+id,{method:"DELETE"});if(r.ok){if(selectedDoc===id)selectedDoc=null;await loadDocs();toast("Document removed")}else toast("Could not remove document",true)}
async function upload(file){if(!file)return;if(file.type!=="application/pdf"){toast("Only PDF files are supported",true);return}const fd=new FormData();fd.append("file",file);$("uploadStatus").textContent="Indexing document…";try{const r=await fetch("/api/documents/upload",{method:"POST",body:fd}),d=await r.json();if(!r.ok)throw new Error(d.detail||"Upload failed");selectedDoc=d.document_id;$("uploadStatus").textContent=`Indexed ${d.filename} · ${d.pages} pages`;await loadDocs();toast("Document indexed successfully")}catch(e){$("uploadStatus").textContent="";toast(e.message,true)}}
$("file").addEventListener("change",e=>upload(e.target.files[0]));const drop=$("drop");["dragenter","dragover"].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add("drag")}));["dragleave","drop"].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove("drag")}));drop.addEventListener("drop",e=>upload(e.dataTransfer.files[0]));$("question").addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();ask()}});
async function ask(){const q=$("question").value.trim();if(!q){toast("Enter a question first",true);return}$("answer").classList.remove("show");$("sources").classList.remove("show");$("welcome").style.display="block";$("welcome").innerHTML="<div class=\"big-icon\">…</div><h3>Searching your knowledge base</h3><p>Retrieving relevant evidence…</p>";try{const r=await fetch("/api/ask",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({question:q,document_id:selectedDoc,top_k:5})}),d=await r.json();if(!r.ok)throw new Error(d.detail||"Request failed");$("welcome").style.display="none";$("answer").textContent=d.answer;$("answer").classList.add("show");$("mAnswerable").textContent=d.answerable?"Yes":"No";$("mAnswerable").style.color=d.answerable?"var(--good)":"var(--warn)";$("mCoverage").textContent=Math.round((d.evidence_coverage||0)*100)+"%";$("mMode").textContent=d.mode||"—";$("mOverall").textContent="—";$("sourceList").innerHTML=(d.citations||[]).map((c,i)=>`<div class="source"><b>Source ${i+1} · ${esc(c.source)}</b><span>Pages ${esc(c.pages)}</span></div>`).join("");if(d.citations?.length)$("sources").classList.add("show")}catch(e){$("welcome").style.display="block";$("welcome").innerHTML="<div class=\"big-icon\">!</div><h3>Request failed</h3><p>Could not reach the RAG service.</p>";toast(e.message,true)}}
function clearChat(){$("question").value="";$("answer").classList.remove("show");$("sources").classList.remove("show");$("welcome").style.display="block";$("welcome").innerHTML="<div class=\"big-icon\">✦</div><h3>What would you like to know?</h3><p>Ask a question about the selected document.</p>";["mAnswerable","mCoverage","mMode","mOverall"].forEach(id=>$(id).textContent="—")}
loadDocs();
</script></body></html>'''


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return HTML
