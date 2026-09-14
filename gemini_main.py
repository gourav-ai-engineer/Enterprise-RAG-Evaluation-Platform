from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from typing import AsyncIterator

import httpx
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

import main as core
import fast_main as fast

app = fast.app
app.routes[:] = [
    route for route in app.routes
    if getattr(route, "path", None) not in {"/", "/api/runtime", "/api/ask/stream"}
]

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_MODELS = [
    "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.6-flash",
    "gemini-3.8-flash", "gemini-2.5-flash-lite", "gemini-2.5-flash",
]


class StreamAskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_ids: list[str] = Field(default_factory=list, max_length=50)
    document_id: str | None = None
    top_k: int = Field(default=3, ge=1, le=10)
    model: str | None = None


def sse(kind: str, payload: dict) -> str:
    return f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def gemini_key() -> str:
    return os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY", "")


def gemini_url(model: str) -> str:
    return "https://generativelanguage.googleapis.com/v1beta/" f"models/{model}:streamGenerateContent?alt=sse"


def _page_label(pages: list[int]) -> str:
    pages = sorted(set(pages))
    if not pages:
        return ""
    ranges = []
    start = prev = pages[0]
    for page in pages[1:]:
        if page == prev + 1:
            prev = page
            continue
        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = page
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ", ".join(ranges)


def evidence_groups(results: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        grouped[result["filename"]].append(result)
    groups = []
    for filename, items in grouped.items():
        pages = []
        for item in items:
            pages.extend(range(int(item["page_start"]), int(item["page_end"]) + 1))
        groups.append({
            "source_id": len(groups) + 1,
            "filename": filename,
            "pages": _page_label(pages),
            "chunks": len(items),
            "evidence": [
                {
                    "chunk_id": item["chunk_id"],
                    "pages": f"{item['page_start']}-{item['page_end']}",
                    "text": item["text"],
                }
                for item in items[:3]
            ],
        })
    return groups


def build_prompt(question: str, results: list[dict]) -> tuple[str, str]:
    groups = evidence_groups(results)
    context = "\n\n".join(
        f"[Source {group['source_id']}] {group['filename']} pages {group['pages']}\n" +
        "\n".join(item["text"] for item in group["evidence"])
        for group in groups
    )
    system = (
        "You are the generation layer of a grounded enterprise RAG system. "
        "Answer ONLY from the supplied evidence. Do not use outside knowledge. "
        "Return only the final answer, never reasoning or analysis. "
        "Do not say 'let me analyze', 'from the evidence', or describe your process. "
        "Be concise: normally 1-3 sentences. Preserve important equations or terms when needed. "
        "Cite factual claims using [Source N], where N is the source number provided. "
        "If the evidence does not support the answer, state that the evidence is insufficient."
    )
    return system, f"Question: {question}\n\nRetrieved evidence:\n{context}"


def compact_citations(results: list[dict]) -> list[dict]:
    return [
        {
            "source": group["filename"],
            "pages": group["pages"],
            "chunks": group["chunks"],
            "source_id": group["source_id"],
        }
        for group in evidence_groups(results)
    ]


async def gemini_stream_generate(model: str, question: str, results: list[dict]) -> tuple[str, float, float, int]:
    key = gemini_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    system, user = build_prompt(question, results)
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "maxOutputTokens": 128,
            "thinkingConfig": {"thinkingLevel": "minimal"},
        },
    }
    started = time.perf_counter()
    ttft_ms = 0.0
    chunks = 0
    pieces: list[str] = []
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", gemini_url(model),
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json=payload,
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise RuntimeError(f"Gemini HTTP {response.status_code}: {body[:600]}")
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw:
                    continue
                data = json.loads(raw)
                candidates = data.get("candidates") or []
                if not candidates:
                    continue
                content = candidates[0].get("content") or {}
                for part in content.get("parts") or []:
                    text = part.get("text") or ""
                    if not text:
                        continue
                    if ttft_ms == 0.0:
                        ttft_ms = (time.perf_counter() - started) * 1000
                    pieces.append(text)
                    chunks += 1
    generation_ms = (time.perf_counter() - started) * 1000
    answer = "".join(pieces).strip()
    if not answer:
        raise RuntimeError("Gemini returned an empty answer")
    return answer, ttft_ms, generation_ms, chunks


@app.get("/api/runtime")
async def runtime():
    return {"provider": "gemini", "models": GEMINI_MODELS, "default_model": DEFAULT_GEMINI_MODEL, "streaming": True}


@app.post("/api/ask/stream")
async def ask_stream(body: StreamAskRequest):
    async def events() -> AsyncIterator[str]:
        total = time.perf_counter()
        selected = list(dict.fromkeys(body.document_ids))
        scope = selected or ([body.document_id] if body.document_id else None)
        retrieval_ms = rerank_ms = generation_ms = ttft_ms = 0.0
        candidates: list[dict] = []
        results: list[dict] = []
        answerable = False
        model = body.model or DEFAULT_GEMINI_MODEL
        yield sse("stage", {"id": "scope", "label": "Preparing scope", "detail": "Loading selected documents"})
        try:
            rows = core.load_chunks(document_ids=scope) if scope else core.load_chunks()
            yield sse("stage", {"id": "retrieval", "label": "Hybrid retrieval", "detail": f"Searching {len(rows)} indexed chunks with BM25 + TF-IDF", "active": True})
            t = time.perf_counter()
            candidates = core.hybrid_retrieve(body.question, rows, max(body.top_k, core.RETRIEVAL_CANDIDATES))
            retrieval_ms = (time.perf_counter() - t) * 1000
            yield sse("stage", {"id": "retrieval", "label": "Hybrid retrieval", "detail": f"Found {len(candidates)} candidates · {retrieval_ms:.0f} ms", "done": True})
            yield sse("stage", {"id": "rerank", "label": "Reranking evidence", "detail": "Scoring top candidates", "active": True})
            t = time.perf_counter()
            results = core.rerank(body.question, candidates, body.top_k)
            rerank_ms = (time.perf_counter() - t) * 1000
            answerable = core.has_sufficient_evidence(body.question, results)
            yield sse("stage", {"id": "rerank", "label": "Reranking evidence", "detail": f"Selected {len(results)} evidence chunks · {rerank_ms:.1f} ms", "done": True})
            if not answerable:
                answer = core.ABSTAIN_MESSAGE
                mode = "abstain"
                yield sse("stage", {"id": "generation", "label": "Grounding guard", "detail": "Evidence threshold not met · Gemini generation skipped", "done": True})
            else:
                yield sse("stage", {"id": "generation", "label": "Generating grounded answer", "detail": f"Gemini · {model} · streaming response", "active": True})
                gen_started = time.perf_counter()
                try:
                    answer, ttft_ms, generation_ms, chunks = await gemini_stream_generate(model, body.question, results)
                    mode = "gemini"
                    yield sse("stage", {"id": "generation", "label": "Generating grounded answer", "detail": f"Completed · TTFT {ttft_ms:.0f} ms · {generation_ms:.0f} ms total · {chunks} chunks", "done": True})
                except Exception as exc:
                    answer = f"Gemini generation failed: {str(exc)}"
                    mode = "gemini_error"
                    generation_ms = (time.perf_counter() - gen_started) * 1000
                    yield sse("stage", {"id": "generation", "label": "Gemini generation failed", "detail": str(exc)[:300], "done": True, "error": True})
            total_ms = (time.perf_counter() - total) * 1000
            payload = {
                "question": body.question,
                "answer": answer,
                "mode": mode,
                "provider": "gemini",
                "model": model if answerable else None,
                "answerable": answerable,
                "scope_document_ids": scope or [],
                "evidence_coverage": round(core.evidence_coverage(body.question, results), 4),
                "citations": compact_citations(results) if answerable else [],
                "evidence_groups": evidence_groups(results) if answerable else [],
                "retrieved": results,
                "latency": {
                    "retrieval_ms": round(retrieval_ms, 2),
                    "rerank_ms": round(rerank_ms, 2),
                    "ttft_ms": round(ttft_ms, 2),
                    "generation_ms": round(generation_ms, 2),
                    "total_ms": round(total_ms, 2),
                    "candidate_count": len(candidates),
                    "context_chunks": len(results),
                },
            }
            yield sse("result", payload)
        except Exception as exc:
            yield sse("error", {"message": str(exc)})
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Enterprise RAG · Workbench</title><style>
:root{color-scheme:dark;--bg:#070c16;--panel:#0d1626;--panel2:#0a1321;--line:#263650;--line2:#344867;--text:#f4f7fb;--muted:#8d9bb2;--accent:#8290ff;--ok:#50d8a2}*{box-sizing:border-box}body{margin:0;background:radial-gradient(900px 500px at 15% -10%,#17264a,transparent 55%),var(--bg);color:var(--text);font:14px Inter,system-ui,sans-serif}.top{height:66px;border-bottom:1px solid var(--line);display:flex;align-items:center;padding:0 28px}.brand{font-weight:800}.pill{margin-left:auto;border:1px solid var(--line);padding:7px 11px;border-radius:999px;color:#b8c4d8;font-size:12px}.main{max-width:1240px;margin:auto;padding:28px}.head{display:flex;justify-content:space-between;align-items:end;margin-bottom:18px}.head h1{margin:4px 0;font-size:28px}.muted{color:var(--muted);font-size:12px}.panel{background:#0d1626e8;border:1px solid var(--line);border-radius:16px;overflow:hidden}.bar{padding:16px 18px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:10px}.scope{margin-left:auto;border:1px solid var(--line2);background:#101b2d;color:#dce5f3;border-radius:9px;padding:9px 12px;cursor:pointer}.chat{min-height:700px;display:flex;flex-direction:column}.body{padding:22px;flex:1;display:flex;flex-direction:column}.answer{background:var(--panel2);border:1px solid #30415f;border-radius:14px;padding:18px;line-height:1.7;white-space:pre-wrap;font-size:15px}.sources{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:9px;margin-top:14px}.source{border:1px solid var(--line);padding:12px;border-radius:10px;background:var(--panel2);cursor:pointer;transition:.15s}.source:hover{border-color:#52698e;transform:translateY(-1px)}.source b{display:block;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.source .pages{font-size:10px;color:var(--muted);margin-top:5px}.source .view{font-size:10px;color:#aebaff;margin-top:9px}.composer{margin-top:auto;padding-top:18px}.box{display:flex;gap:10px;border:1px solid var(--line2);background:var(--panel2);border-radius:13px;padding:10px}.box textarea{flex:1;background:transparent;border:0;outline:0;color:white;resize:none;min-height:45px}.send{width:44px;border:0;border-radius:10px;background:linear-gradient(135deg,#718cff,#8b73e8);color:white;font-size:20px;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin-top:14px}.metric{padding:12px;border:1px solid var(--line);border-radius:10px;background:var(--panel2)}.metric span{font-size:10px;color:#71819b}.metric b{display:block;margin-top:5px}.overlay,.modal{position:fixed;inset:0;background:rgba(3,7,14,.74);backdrop-filter:blur(7px);display:none;align-items:center;justify-content:center;z-index:20}.loader,.modalbox{width:min(650px,94vw);background:var(--panel);border:1px solid #334663;border-radius:18px;box-shadow:0 30px 90px #0008}.loader{padding:25px}.spinner{width:30px;height:30px;border:3px solid #33425c;border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite;margin-bottom:18px}@keyframes spin{to{transform:rotate(360deg)}}.stage{display:flex;gap:12px;padding:12px;border:1px solid transparent;border-radius:10px;color:#7f8ca4}.stage.active{border-color:#30415f;background:#101b2d;color:#eaf0f9}.stage.done{color:var(--ok)}.dot{width:9px;height:9px;border-radius:50%;background:#52627c;margin-top:5px}.active .dot{background:var(--accent);box-shadow:0 0 12px var(--accent)}.done .dot{background:var(--ok)}.modalbox{max-height:82vh;overflow:auto}.modalhead{padding:18px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between}.evidence{padding:16px 18px;border-bottom:1px solid #1e2a3f}.evidencehead{display:flex;justify-content:space-between;gap:10px}.evidence pre{white-space:pre-wrap;font:12px/1.6 Inter,system-ui,sans-serif;color:#b9c5d8;background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:12px;margin:10px 0 0}.docrow{display:flex;align-items:center;gap:12px;padding:14px 18px;border-bottom:1px solid #1e2a3f}.docrow input{width:17px;height:17px}.docrow .info{flex:1}.modalfoot{padding:14px 18px;display:flex;justify-content:space-between;gap:8px;position:sticky;bottom:0;background:var(--panel);border-top:1px solid var(--line)}button{font:inherit}.mini{border:1px solid var(--line);background:#101b2d;color:#d6deeb;padding:8px 11px;border-radius:8px;cursor:pointer}.primary{border:0;background:linear-gradient(135deg,#718cff,#8b73e8);color:#fff;padding:9px 13px;border-radius:9px;cursor:pointer}@media(max-width:700px){.main{padding:14px}.metrics{grid-template-columns:1fr 1fr}.head{align-items:flex-start;gap:10px;flex-direction:column}.scope{margin-left:0}}
</style></head><body><header class="top"><div class="brand">✦ Enterprise RAG · Workbench</div><div id="provider" class="pill">Connecting Gemini…</div></header><main class="main"><div class="head"><div><div class="muted">GROUNDED DOCUMENT INTELLIGENCE</div><h1>Ask your documents</h1><div class="muted">Hybrid retrieval → reranking → grounded generation</div></div><div><input id="file" type="file" accept="application/pdf" style="display:none"><button class="primary" onclick="$("file").click()">+ Add PDF</button></div></div><section class="panel chat"><div class="bar"><div><b>Document assistant</b><div id="scopeText" class="muted">Using all documents</div></div><button class="scope" onclick="openScope()"><span id="scopeBtn">All documents</span> ▾</button><button class="mini" onclick="clearChat()">Clear</button></div><div class="body"><div id="answer" class="answer" style="display:none"></div><div id="sources" class="sources"></div><div id="metrics" class="metrics" style="display:none"><div class="metric"><span>Retrieval</span><b id="mR">—</b></div><div class="metric"><span>Rerank</span><b id="mK">—</b></div><div class="metric"><span>Generation</span><b id="mG">—</b></div><div class="metric"><span>Total</span><b id="mT">—</b></div></div><div class="composer"><div class="box"><textarea id="q" placeholder="Ask a question about the selected documents…"></textarea><button class="send" onclick="ask()">↑</button></div><div class="muted">Enter to send · Shift + Enter for a new line</div></div></div></section></main><div id="overlay" class="overlay"><div class="loader"><div class="spinner"></div><h3 id="loadTitle">Preparing retrieval…</h3><div id="loadDetail" class="muted" style="margin-bottom:14px">Please wait</div><div id="stages"></div></div></div><div id="modal" class="modal"><div class="modalbox"><div class="modalhead"><div><b>Retrieval scope</b><div class="muted">Choose exactly which documents can be searched</div></div><button class="mini" onclick="closeScope()">Close</button></div><div style="padding:14px 18px;display:flex;gap:8px"><button class="mini" onclick="selectAll()">Select all</button><button class="mini" onclick="clearSelect()">Clear</button></div><div id="docs"></div><div class="modalfoot"><div id="selCount" class="muted">All documents</div><button class="primary" onclick="applyScope()">Apply scope</button></div></div></div><div id="evidenceModal" class="modal"><div class="modalbox"><div class="modalhead"><div><b id="evidenceTitle">Evidence</b><div id="evidencePages" class="muted"></div></div><button class="mini" onclick="closeEvidence()">Close</button></div><div id="evidenceBody"></div></div></div>
<script>
const $=id=>document.getElementById(id);let docs=[],selected=new Set(JSON.parse(localStorage.getItem('rag_scope')||'[]')),lastGroups=[];
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
async function loadDocs(){const r=await fetch('/api/documents');const d=await r.json();docs=d.documents||[];if(!selected.size)docs.forEach(x=>selected.add(x.id));renderDocs();updateScope()}
async function runtime(){try{const r=await fetch('/api/runtime');const d=await r.json();$("provider").textContent=`Gemini · ${d.default_model} · streaming`}catch{$("provider").textContent='Gemini · unavailable'}}
function renderDocs(){const box=$("docs");box.innerHTML=docs.length?docs.map(d=>`<label class="docrow"><input type="checkbox" data-id="${esc(d.id)}" ${selected.has(d.id)?'checked':''} onchange="syncCount()"><div class="info"><b>${esc(d.filename)}</b><div class="muted">${d.pages} pages · ${d.chunks} chunks</div></div></label>`).join(''):`<div style="padding:25px" class="muted">No PDFs indexed yet. Add a PDF to begin.</div>`;syncCount()}
function syncCount(){const checked=[...document.querySelectorAll('#docs input:checked')];$("selCount").textContent=checked.length?`${checked.length} document${checked.length>1?'s':''} selected`:'No documents selected'}
function selectAll(){document.querySelectorAll('#docs input').forEach(x=>x.checked=true);syncCount()}
function clearSelect(){document.querySelectorAll('#docs input').forEach(x=>x.checked=false);syncCount()}
function applyScope(){selected=new Set([...document.querySelectorAll('#docs input:checked')].map(x=>x.dataset.id));localStorage.setItem('rag_scope',JSON.stringify([...selected]));updateScope();closeScope()}
function updateScope(){const n=selected.size;const all=docs.length&&n===docs.length;$("scopeBtn").textContent=all?'All documents':`${n} selected`;$("scopeText").textContent=all?'Using all documents':`Using ${n} selected document${n===1?'':'s'}`}
function openScope(){renderDocs();$("modal").style.display='flex'}function closeScope(){$("modal").style.display='none'}
function showEvidence(i){const g=lastGroups[i];if(!g)return;$("evidenceTitle").textContent=`Source ${g.source_id} · ${g.filename}`;$("evidencePages").textContent=`Pages ${g.pages} · ${g.chunks} retrieved chunk${g.chunks>1?'s':''}`;$("evidenceBody").innerHTML=g.evidence.map((e,j)=>`<div class="evidence"><div class="evidencehead"><b>Evidence passage ${j+1}</b><span class="muted">Pages ${esc(e.pages)}</span></div><pre>${esc(e.text)}</pre></div>`).join('');$("evidenceModal").style.display='flex'}
function closeEvidence(){$("evidenceModal").style.display='none'}
function renderSources(groups){lastGroups=groups||[];$("sources").innerHTML=lastGroups.map((g,i)=>`<button class="source" onclick="showEvidence(${i})"><b>📄 Source ${g.source_id} · ${esc(g.filename)}</b><div class="pages">Pages ${esc(g.pages)} · ${g.chunks} chunk${g.chunks>1?'s':''}</div><div class="view">View exact evidence →</div></button>`).join('')}
function setStage(e){const id=e.id;let el=document.querySelector(`[data-stage="${id}"]`);if(!el){el=document.createElement('div');el.className='stage';el.dataset.stage=id;$("stages").appendChild(el)}el.className='stage '+(e.done?'done':e.active?'active':'');el.innerHTML=`<div class="dot"></div><div><b>${esc(e.label)}</b><div class="muted">${esc(e.detail||'')}</div></div>`;$("loadTitle").textContent=e.label;$("loadDetail").textContent=e.detail||''}
async function ask(){const question=$("q").value.trim();if(!question)return;const ids=[...selected];$("overlay").style.display='flex';$("stages").innerHTML='';$("answer").style.display='none';$("sources").innerHTML='';$("metrics").style.display='none';const body={question,document_ids:ids,top_k:3,model:null};try{const res=await fetch('/api/ask/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});if(!res.ok)throw new Error(await res.text());const reader=res.body.getReader(),dec=new TextDecoder();let buf='';while(true){const {value,done}=await reader.read();if(done)break;buf+=dec.decode(value,{stream:true});const blocks=buf.split('\n\n');buf=blocks.pop()||'';for(const block of blocks){const lines=block.split('\n');const type=(lines.find(x=>x.startsWith('event:'))||'').slice(6).trim();const raw=(lines.find(x=>x.startsWith('data:'))||'').slice(5).trim();if(!raw)continue;const data=JSON.parse(raw);if(type==='stage')setStage(data);if(type==='result')renderResult(data);if(type==='error')throw new Error(data.message)}}}catch(e){$("answer").style.display='block';$("answer").textContent='Request failed: '+e.message}finally{$("overlay").style.display='none'}}
function renderResult(d){$("answer").style.display='block';$("answer").textContent=d.answer;renderSources(d.evidence_groups||[]);const l=d.latency||{};$("metrics").style.display='grid';$("mR").textContent=(l.retrieval_ms??0).toFixed(0)+' ms';$("mK").textContent=(l.rerank_ms??0).toFixed(1)+' ms';$("mG").textContent=(l.generation_ms??0).toFixed(0)+' ms';$("mT").textContent=(l.total_ms??0).toFixed(0)+' ms'}
function clearChat(){$("answer").style.display='none';$("sources").innerHTML='';$("metrics").style.display='none';lastGroups=[]}
$("q").addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();ask()}});$("file").addEventListener('change',async e=>{const f=e.target.files[0];if(!f)return;const fd=new FormData();fd.append('file',f);$("provider").textContent='Indexing PDF…';try{const r=await fetch('/api/documents/upload',{method:'POST',body:fd});if(!r.ok)throw new Error(await r.text());await loadDocs();$("provider").textContent='PDF indexed · Gemini ready'}catch(err){alert('Upload failed: '+err.message)}e.target.value='' });loadDocs();runtime();
</script></body></html>'''


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(HTML)
