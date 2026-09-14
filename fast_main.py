from __future__ import annotations
import json, os, time
from typing import AsyncIterator
import httpx
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
import main as core

app = core.app

class StreamAskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_ids: list[str] = Field(default_factory=list, max_length=50)
    document_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)
    model: str | None = None

def sse(kind: str, payload: dict) -> str:
    return f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

async def ollama_models():
    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
    native = base[:-3] if base.endswith("/v1") else base
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            r = await client.get(f"{native}/api/tags")
            r.raise_for_status()
            return [m.get("name") for m in r.json().get("models", []) if m.get("name")]
    except Exception:
        return []

async def generate(model: str, question: str, results: list[dict]) -> tuple[str, float, str]:
    api_key = os.getenv("OPENAI_API_KEY", "")
    base = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
    if not base:
        base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
    context = "\n\n".join(f"[Source {i}] {r['filename']} pages {r['page_start']}-{r['page_end']}\n{r['text']}" for i,r in enumerate(results,1))
    system = "You are a grounded enterprise document assistant. Answer ONLY from the supplied evidence. Do not use outside knowledge. Give a concise 1-4 sentence answer. Cite factual claims as [Source N]. If evidence is insufficient, say so. Do not explain your reasoning."
    t=time.perf_counter()
    if "11434" in base:
        native = base[:-3] if base.endswith("/v1") else base
        payload={"model":model,"messages":[{"role":"system","content":system},{"role":"user","content":f"Question: {question}\n\nReranked evidence:\n{context}"}],"stream":False,"think":False,"keep_alive":"10m","options":{"temperature":0,"num_predict":256,"num_ctx":4096}}
        async with httpx.AsyncClient(timeout=120) as client:
            r=await client.post(f"{native}/api/chat",json=payload); r.raise_for_status(); data=r.json(); answer=(data.get("message",{}).get("content") or "").strip()
        return answer,(time.perf_counter()-t)*1000,"ollama"
    headers={"Content-Type":"application/json"}
    if api_key: headers["Authorization"] = f"Bearer {api_key}"
    payload={"model":model,"messages":[{"role":"system","content":system},{"role":"user","content":f"Question: {question}\n\nReranked evidence:\n{context}"}],"temperature":0,"max_tokens":256}
    async with httpx.AsyncClient(timeout=120) as client:
        r=await client.post(f"{base}/chat/completions",headers=headers,json=payload); r.raise_for_status(); data=r.json(); answer=(data["choices"][0]["message"]["content"] or "").strip()
    return answer,(time.perf_counter()-t)*1000,"llm"

@app.get("/api/runtime")
async def runtime():
    models=await ollama_models()
    default=os.getenv("OLLAMA_MODEL", models[0] if models else "qwen3:8b")
    return {"provider":"ollama" if models else "unavailable","models":models,"default_model":default}

@app.post("/api/ask/stream")
async def ask_stream(body: StreamAskRequest):
    async def events() -> AsyncIterator[str]:
        total=time.perf_counter()
        selected=list(dict.fromkeys(body.document_ids))
        scope=selected or ([body.document_id] if body.document_id else None)
        yield sse("stage",{"id":"scope","label":"Preparing scope","detail":"Loading selected documents"})
        await __import__('asyncio').sleep(0)
        try:
            rows=core.load_chunks(document_ids=scope) if scope else core.load_chunks()
            yield sse("stage",{"id":"retrieval","label":"Hybrid retrieval","detail":f"Searching {len(rows)} indexed chunks with BM25 + TF-IDF","active":True})
            t=time.perf_counter(); candidates=core.hybrid_retrieve(body.question,rows,max(body.top_k,core.RETRIEVAL_CANDIDATES)); retrieval_ms=(time.perf_counter()-t)*1000
            yield sse("stage",{"id":"retrieval","label":"Hybrid retrieval","detail":f"Found {len(candidates)} candidates · {retrieval_ms:.0f} ms","done":True})
            yield sse("stage",{"id":"rerank","label":"Reranking evidence","detail":"Scoring top candidates","active":True})
            t=time.perf_counter(); results=core.rerank(body.question,candidates,body.top_k); rerank_ms=(time.perf_counter()-t)*1000
            answerable=core.has_sufficient_evidence(body.question,results)
            yield sse("stage",{"id":"rerank","label":"Reranking evidence","detail":f"Selected {len(results)} evidence chunks · {rerank_ms:.1f} ms","done":True})
            if not answerable:
                answer=core.ABSTAIN_MESSAGE; mode="abstain"; generation_ms=0.0
                yield sse("stage",{"id":"generation","label":"Grounding guard","detail":"Evidence threshold not met · generation skipped","done":True})
            else:
                models=await ollama_models()
                model=body.model or os.getenv("OLLAMA_MODEL", "qwen3:8b")
                if model not in models and models:
                    model=models[0]
                yield sse("stage",{"id":"generation","label":"Generating grounded answer","detail":f"{model} · context locked to reranked chunks","active":True})
                try:
                    answer,generation_ms,mode=await generate(model,body.question,results)
                    yield sse("stage",{"id":"generation","label":"Generating grounded answer","detail":f"Completed · {generation_ms:.0f} ms","done":True})
                except Exception as exc:
                    answer=core.extractive_answer(body.question,results); generation_ms=0.0; mode="extractive_fallback"
                    yield sse("stage",{"id":"generation","label":"Generation fallback","detail":"LLM unavailable; returned grounded extractive answer","done":True,"error":str(exc)[:180]})
            total_ms=(time.perf_counter()-total)*1000
            payload={"question":body.question,"answer":answer,"mode":mode,"answerable":answerable,"scope_document_ids":scope or [],"evidence_coverage":round(core.evidence_coverage(body.question,results),4),"citations":core.compact_citations(results) if answerable else [],"retrieved":results,"latency":{"retrieval_ms":round(retrieval_ms,2),"rerank_ms":round(rerank_ms,2),"generation_ms":round(generation_ms,2),"total_ms":round(total_ms,2),"candidate_count":len(candidates),"context_chunks":len(results)},"model":(body.model or os.getenv("OLLAMA_MODEL","qwen3:8b")) if answerable else None}
            yield sse("result",payload)
        except Exception as exc:
            yield sse("error",{"message":str(exc)})
    return StreamingResponse(events(),media_type="text/event-stream",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

NEW_HTML=r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Enterprise RAG — Fast Workbench</title><style>:root{color-scheme:dark;--bg:#070c16;--panel:#0d1626;--line:#263650;--text:#f4f7fb;--muted:#8d9bb2}*{box-sizing:border-box}body{margin:0;background:radial-gradient(900px 500px at 15% -10%,#17264a,transparent 55%),var(--bg);color:var(--text);font:14px Inter,system-ui,sans-serif}.top{height:66px;border-bottom:1px solid var(--line);display:flex;align-items:center;padding:0 28px}.brand{font-weight:800}.pill{margin-left:auto;border:1px solid var(--line);padding:7px 11px;border-radius:999px;color:#b8c4d8;font-size:12px}.main{max-width:1240px;margin:auto;padding:28px}.head{display:flex;justify-content:space-between;align-items:end;margin-bottom:18px}.head h1{margin:4px 0;font-size:28px}.muted{color:var(--muted);font-size:12px}.panel{background:#0d1626e8;border:1px solid var(--line);border-radius:16px;overflow:hidden}.bar{padding:16px 18px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:10px}.scope{margin-left:auto;border:1px solid #3a4b68;background:#101b2d;color:#dce5f3;border-radius:9px;padding:9px 12px;cursor:pointer}.chat{min-height:700px;display:flex;flex-direction:column}.body{padding:22px;flex:1;display:flex;flex-direction:column}.answer{background:#0a1321;border:1px solid #30415f;border-radius:14px;padding:18px;line-height:1.7;white-space:pre-wrap}.sources{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}.source{border:1px solid var(--line);padding:10px 12px;border-radius:10px;background:#0a1321}.source b{display:block;font-size:11px}.source span{font-size:10px;color:var(--muted)}.composer{margin-top:auto;padding-top:18px}.box{display:flex;gap:10px;border:1px solid #344867;background:#0a1321;border-radius:13px;padding:10px}.box textarea{flex:1;background:transparent;border:0;outline:0;color:white;resize:none;min-height:45px}.send{width:44px;border:0;border-radius:10px;background:linear-gradient(135deg,#718cff,#8b73e8);color:white;font-size:20px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin-top:14px}.metric{padding:12px;border:1px solid var(--line);border-radius:10px;background:#0a1321}.metric span{font-size:10px;color:#71819b}.metric b{display:block;margin-top:5px}.overlay{position:fixed;inset:0;background:rgba(3,7,14,.72);backdrop-filter:blur(7px);display:none;align-items:center;justify-content:center;z-index:20}.loader{width:min(560px,92vw);background:#0d1626;border:1px solid #334663;border-radius:18px;box-shadow:0 30px 90px #0008;padding:25px}.loader h3{margin:0 0 5px}.spinner{width:30px;height:30px;border:3px solid #33425c;border-top-color:#8791ff;border-radius:50%;animation:spin .8s linear infinite;margin-bottom:18px}@keyframes spin{to{transform:rotate(360deg)}}.stage{display:flex;gap:12px;padding:12px;border:1px solid transparent;border-radius:10px;color:#7f8ca4}.stage.active{border-color:#30415f;background:#101b2d;color:#eaf0f9}.stage.done{color:#50d8a2}.dot{width:9px;height:9px;border-radius:50%;background:#52627c;margin-top:5px}.active .dot{background:#8290ff;box-shadow:0 0 12px #8290ff}.done .dot{background:#4ad59e}.modal{position:fixed;inset:0;background:#03070dbb;backdrop-filter:blur(5px);display:none;align-items:center;justify-content:center;z-index:15}.modalbox{width:min(680px,94vw);max-height:82vh;overflow:auto;background:#0d1626;border:1px solid #344766;border-radius:16px}.modalhead{padding:18px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between}.docrow{display:flex;align-items:center;gap:12px;padding:14px 18px;border-bottom:1px solid #1e2a3f}.docrow input{width:17px;height:17px}.docrow .info{flex:1}.modalfoot{padding:14px 18px;display:flex;justify-content:space-between;gap:8px;position:sticky;bottom:0;background:#0d1626;border-top:1px solid var(--line)}button{font:inherit}.mini{border:1px solid var(--line);background:#101b2d;color:#d6deeb;padding:8px 11px;border-radius:8px;cursor:pointer}.primary{border:0;background:linear-gradient(135deg,#718cff,#8b73e8);color:#fff;padding:9px 13px;border-radius:9px;cursor:pointer}@media(max-width:700px){.main{padding:14px}.metrics{grid-template-columns:1fr 1fr}.head{align-items:flex-start;gap:10px;flex-direction:column}.scope{margin-left:0}}</style></head><body><header class="top"><div class="brand">✦ Enterprise RAG · Fast Workbench</div><div id="provider" class="pill">Checking runtime…</div></header><main class="main"><div class="head"><div><div class="muted">GROUNDED DOCUMENT INTELLIGENCE</div><h1>Ask your documents</h1><div class="muted">Hybrid retrieval → reranking → grounded generation</div></div><div><input id="file" type="file" accept="application/pdf" style="display:none"><button class="primary" onclick="$('file').click()">+ Add PDF</button></div></div><section class="panel chat"><div class="bar"><div><b>Document assistant</b><div id="scopeText" class="muted">Using all documents</div></div><button class="scope" onclick="openScope()"><span id="scopeBtn">All documents</span> ▾</button><button class="mini" onclick="clearChat()">Clear</button></div><div class="body"><div id="answer" class="answer" style="display:none"></div><div id="sources" class="sources"></div><div id="metrics" class="metrics" style="display:none"><div class="metric"><span>Retrieval</span><b id="mR">—</b></div><div class="metric"><span>Rerank</span><b id="mK">—</b></div><div class="metric"><span>Generation</span><b id="mG">—</b></div><div class="metric"><span>Total</span><b id="mT">—</b></div></div><div class="composer"><div class="box"><textarea id="q" placeholder="Ask a question about the selected documents…"></textarea><button class="send" onclick="ask()">↑</button></div><div class="muted">Enter to send · Shift + Enter for a new line</div></div></div></section></main><div id="overlay" class="overlay"><div class="loader"><div class="spinner"></div><h3 id="loadTitle">Preparing retrieval…</h3><div id="loadDetail" class="muted" style="margin-bottom:14px">Please wait</div><div id="stages"></div></div></div><div id="modal" class="modal"><div class="modalbox"><div class="modalhead"><div><b>Retrieval scope</b><div class="muted">Choose exactly which documents can be searched</div></div><button class="mini" onclick="closeScope()">Close</button></div><div style="padding:14px 18px;display:flex;gap:8px"><button class="mini" onclick="selectAll()">Select all</button><button class="mini" onclick="clearSelect()">Clear</button></div><div id="docs"></div><div class="modalfoot"><div id="selCount" class="muted">All documents</div><button class="primary" onclick="applyScope()">Apply scope</button></div></div></div><script>let docs=[],selected=new Set(),models=[],activeModel='';const $=x=>document.getElementById(x);function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]))}async function init(){const [d,r]=await Promise.all([fetch('/api/documents').then(x=>x.json()),fetch('/api/runtime').then(x=>x.json())]);docs=d.documents||[];models=r.models||[];activeModel=r.default_model||models[0]||'';$('provider').textContent=models.length?'Ollama · '+models.length+' model'+(models.length>1?'s':''):'LLM unavailable';renderDocs()}async function upload(){const f=$('file').files[0];if(!f)return;const fd=new FormData();fd.append('file',f);$('provider').textContent='Indexing PDF…';const r=await fetch('/api/documents/upload',{method:'POST',body:fd});const d=await r.json();if(!r.ok){alert(d.detail||'Upload failed');return}docs.unshift(d);selected.add(d.document_id);renderDocs();$('provider').textContent='Indexed · '+d.filename;$('file').value=''}function renderDocs(){$('docs').innerHTML=docs.map(d=>`<label class="docrow"><input type="checkbox" ${selected.has(d.id)?'checked':''} onchange="toggle('${d.id}',this.checked)"><div class="info"><b>${esc(d.filename)}</b><div class="muted">${d.pages} pages · ${d.chunks} chunks</div></div></label>`).join('')||'<div style="padding:20px" class="muted">No documents indexed.</div>';updateCount()}function toggle(id,on){on?selected.add(id):selected.delete(id);updateCount()}function updateCount(){const n=selected.size;$('selCount').textContent=n?`${n} document${n>1?'s':''} selected`:'All documents';$('scopeBtn').textContent=n?`${n} document${n>1?'s':''}`:'All documents';$('scopeText').textContent=n?`Using ${n} selected document${n>1?'s':''}`:'Using all documents'}function selectAll(){docs.forEach(d=>selected.add(d.id));renderDocs()}function clearSelect(){selected.clear();renderDocs()}function openScope(){renderDocs();$('modal').style.display='flex'}function closeScope(){$('modal').style.display='none'}function applyScope(){closeScope()}function showLoader(){const ids=['scope','retrieval','rerank','generation'];$('stages').innerHTML=ids.map(id=>`<div id="stage-${id}" class="stage"><span class="dot"></span><div><b>${({scope:'Preparing scope',retrieval:'Hybrid retrieval',rerank:'Reranking evidence',generation:'Generating grounded answer'})[id]}</b><div class="muted" id="detail-${id}">Waiting…</div></div></div>`).join('');$('overlay').style.display='flex'}function stage(e){const p=JSON.parse(e.data),el=$('stage-'+p.id);if(el){el.className='stage '+(p.done?'done':'active');$('detail-'+p.id).textContent=p.detail||''}$('loadTitle').textContent=p.label||'Working…';$('loadDetail').textContent=p.detail||''}async function ask(){const question=$('q').value.trim();if(!question)return;showLoader();$('answer').style.display='none';$('sources').innerHTML='';$('metrics').style.display='none';const payload={question,document_ids:[...selected],top_k:5,model:activeModel};try{const res=await fetch('/api/ask/stream',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(payload)});if(!res.ok)throw new Error(await res.text());const reader=res.body.getReader(),decoder=new TextDecoder();let buf='',result=null;while(true){const {value,done}=await reader.read();if(done)break;buf+=decoder.decode(value,{stream:true});const parts=buf.split('\n\n');buf=parts.pop();for(const block of parts){const em=block.match(/event: (.+)\ndata: (.+)/);if(!em)continue;const type=em[1],data=em[2];if(type==='stage')stage({data});else if(type==='result')result=JSON.parse(data);else if(type==='error')throw new Error(JSON.parse(data).message)}}if(!result)throw new Error('No result returned');renderResult(result)}catch(err){$('answer').textContent='Request failed: '+err.message;$('answer').style.display='block'}finally{$('overlay').style.display='none'}}function renderResult(d){$('answer').textContent=d.answer;$('answer').style.display='block';$('sources').innerHTML=(d.citations||[]).map((c,i)=>`<div class="source"><b>📄 Source ${i+1} · ${esc(c.source)}</b><span>Pages ${esc(c.pages)}</span></div>`).join('');$('metrics').style.display='grid';$('mR').textContent=d.latency.retrieval_ms+' ms';$('mK').textContent=d.latency.rerank_ms+' ms';$('mG').textContent=d.latency.generation_ms+' ms';$('mT').textContent=d.latency.total_ms+' ms'}function clearChat(){$('q').value='';$('answer').style.display='none';$('sources').innerHTML='';$('metrics').style.display='none'}$('file').addEventListener('change',upload);$('q').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();ask()}});init();</script></body></html>'''

for route in list(app.router.routes):
    if getattr(route, 'path', None) == '/':
        app.router.routes.remove(route)
        break

@app.get('/', response_class=HTMLResponse)
def fast_home():
    return NEW_HTML
