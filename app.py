from __future__ import annotations

import math
import re
from dataclasses import dataclass
from statistics import mean
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Enterprise RAG Evaluation Platform", version="1.0.0")


@dataclass
class Chunk:
    id: str
    text: str
    source: str


CORPUS = [
    Chunk("c1", "Refunds are processed within 5-7 business days after approval.", "refund_policy.md"),
    Chunk("c2", "Enterprise plans include SSO, audit logs, and role based access control.", "enterprise.md"),
    Chunk("c3", "API rate limits are 100 requests per minute for standard tenants.", "api_limits.md"),
    Chunk("c4", "Critical incidents should be escalated to the on-call engineer immediately.", "incident_response.md"),
]


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def retrieve(query: str, k: int = 3) -> list[Chunk]:
    q = tokenize(query)
    scored = []
    for c in CORPUS:
        t = tokenize(c.text)
        score = len(q & t) / max(1, math.sqrt(len(q) * len(t)))
        scored.append((score, c))
    return [c for _, c in sorted(scored, reverse=True, key=lambda x: x[0])[:k]]


class EvalRequest(BaseModel):
    query: str = Field(min_length=3)
    answer: str = Field(min_length=1)
    expected: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/retrieve")
def api_retrieve(q: str, k: int = 3) -> dict[str, Any]:
    docs = retrieve(q, max(1, min(k, 10)))
    return {"query": q, "results": [{"id": d.id, "source": d.source, "text": d.text} for d in docs]}


@app.post("/api/evaluate")
def evaluate(body: EvalRequest) -> dict[str, Any]:
    docs = retrieve(body.query)
    context = " ".join(d.text for d in docs).lower()
    answer_tokens = tokenize(body.answer)
    context_tokens = tokenize(context)
    groundedness = len(answer_tokens & context_tokens) / max(1, len(answer_tokens))
    relevance = len(tokenize(body.query) & answer_tokens) / max(1, len(tokenize(body.query)))
    exact = 1.0 if body.expected and tokenize(body.expected) <= answer_tokens else None
    scores = [groundedness, relevance] + ([] if exact is None else [exact])
    return {
        "groundedness": round(groundedness, 3),
        "relevance": round(relevance, 3),
        "exact_match": None if exact is None else round(exact, 3),
        "overall": round(mean(scores), 3),
        "retrieved": [d.source for d in docs],
    }


HTML = """
<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Enterprise RAG Evaluation Platform</title><style>
body{font-family:Inter,system-ui;margin:0;background:#0b1020;color:#edf2f7}main{max-width:1000px;margin:auto;padding:40px}
.card{background:#151d33;border:1px solid #293453;border-radius:16px;padding:24px;margin:18px 0}input,textarea,button{width:100%;box-sizing:border-box;margin-top:10px;padding:12px;border-radius:10px;border:1px solid #34405f;background:#0f1629;color:white}button{cursor:pointer}pre{white-space:pre-wrap}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:700px){.grid{grid-template-columns:1fr}}
</style></head><body><main>
<h1>Enterprise RAG Evaluation Platform</h1><p>Retrieval, groundedness, relevance and evaluation in one inspectable demo.</p>
<div class='grid'><div class='card'><h2>Retrieve</h2><input id='q' value='How long do refunds take?'><button onclick='retrieve()'>Run retrieval</button><pre id='r'></pre></div>
<div class='card'><h2>Evaluate answer</h2><input id='eq' value='How long do refunds take?'><textarea id='a'>Refunds are processed within 5-7 business days after approval.</textarea><button onclick='evaluate()'>Evaluate</button><pre id='e'></pre></div></div>
</main><script>
async function retrieve(){let q=document.getElementById('q').value;let r=await fetch('/api/retrieve?q='+encodeURIComponent(q));document.getElementById('r').textContent=JSON.stringify(await r.json(),null,2)}
async function evaluate(){let b={query:document.getElementById('eq').value,answer:document.getElementById('a').value};let r=await fetch('/api/evaluate',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(b)});document.getElementById('e').textContent=JSON.stringify(await r.json(),null,2)}
retrieve();evaluate();</script></body></html>
"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return HTML
