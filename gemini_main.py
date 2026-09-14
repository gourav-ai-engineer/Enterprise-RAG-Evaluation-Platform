from __future__ import annotations

import json
import os
import time
from typing import AsyncIterator

import httpx
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import main as core
import fast_main as fast

# Reuse the polished Fast Workbench UI from fast_main.py, but replace its
# Ollama-oriented generation/runtime routes with an explicit Gemini backend.
app = fast.app

# Remove the old same-path routes so the Gemini routes below are selected.
app.routes[:] = [
    route
    for route in app.routes
    if getattr(route, "path", None) not in {"/api/runtime", "/api/ask/stream"}
]

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
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
    # Gemini REST streaming uses models/{model}:streamGenerateContent.
    return (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{model}:streamGenerateContent?alt=sse"
    )


def build_prompt(question: str, results: list[dict]) -> tuple[str, str]:
    context = "\n\n".join(
        f"[Source {i}] {r['filename']} pages {r['page_start']}-{r['page_end']}\n{r['text']}"
        for i, r in enumerate(results, 1)
    )
    system = (
        "You are the generation layer of a grounded enterprise RAG system. "
        "Answer ONLY from the supplied evidence. Do not use outside knowledge. "
        "Return only the final answer, never reasoning or analysis. "
        "Do not say 'let me analyze', 'from the evidence', or describe your process. "
        "Be concise: normally 1-3 sentences. Preserve important equations or terms when needed. "
        "Cite factual claims using [Source N]. If the evidence does not support the answer, "
        "state that the evidence is insufficient."
    )
    user = f"Question: {question}\n\nRetrieved evidence:\n{context}"
    return system, user


async def gemini_stream_generate(
    model: str, question: str, results: list[dict]
) -> tuple[str, float, float, int]:
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
            "POST",
            gemini_url(model),
            headers={
                "x-goog-api-key": key,
                "Content-Type": "application/json",
            },
            json=payload,
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Gemini HTTP {response.status_code}: {body[:600]}"
                )

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
    return {
        "provider": "gemini",
        "models": GEMINI_MODELS,
        "default_model": DEFAULT_GEMINI_MODEL,
        "streaming": True,
    }


@app.post("/api/ask/stream")
async def ask_stream(body: StreamAskRequest):
    async def events() -> AsyncIterator[str]:
        total = time.perf_counter()
        selected = list(dict.fromkeys(body.document_ids))
        scope = selected or ([body.document_id] if body.document_id else None)
        retrieval_ms = 0.0
        rerank_ms = 0.0
        generation_ms = 0.0
        ttft_ms = 0.0
        candidates: list[dict] = []
        results: list[dict] = []
        answerable = False
        model = body.model or DEFAULT_GEMINI_MODEL

        yield sse(
            "stage",
            {
                "id": "scope",
                "label": "Preparing scope",
                "detail": "Loading selected documents",
            },
        )

        try:
            rows = core.load_chunks(document_ids=scope) if scope else core.load_chunks()
            yield sse(
                "stage",
                {
                    "id": "retrieval",
                    "label": "Hybrid retrieval",
                    "detail": f"Searching {len(rows)} indexed chunks with BM25 + TF-IDF",
                    "active": True,
                },
            )

            t = time.perf_counter()
            candidates = core.hybrid_retrieve(
                body.question,
                rows,
                max(body.top_k, core.RETRIEVAL_CANDIDATES),
            )
            retrieval_ms = (time.perf_counter() - t) * 1000
            yield sse(
                "stage",
                {
                    "id": "retrieval",
                    "label": "Hybrid retrieval",
                    "detail": f"Found {len(candidates)} candidates · {retrieval_ms:.0f} ms",
                    "done": True,
                },
            )

            yield sse(
                "stage",
                {
                    "id": "rerank",
                    "label": "Reranking evidence",
                    "detail": "Scoring top candidates",
                    "active": True,
                },
            )
            t = time.perf_counter()
            results = core.rerank(body.question, candidates, body.top_k)
            rerank_ms = (time.perf_counter() - t) * 1000
            answerable = core.has_sufficient_evidence(body.question, results)
            yield sse(
                "stage",
                {
                    "id": "rerank",
                    "label": "Reranking evidence",
                    "detail": f"Selected {len(results)} evidence chunks · {rerank_ms:.1f} ms",
                    "done": True,
                },
            )

            if not answerable:
                answer = core.ABSTAIN_MESSAGE
                mode = "abstain"
                yield sse(
                    "stage",
                    {
                        "id": "generation",
                        "label": "Grounding guard",
                        "detail": "Evidence threshold not met · Gemini generation skipped",
                        "done": True,
                    },
                )
            else:
                yield sse(
                    "stage",
                    {
                        "id": "generation",
                        "label": "Generating grounded answer",
                        "detail": f"Gemini · {model} · streaming response",
                        "active": True,
                    },
                )
                try:
                    answer, ttft_ms, generation_ms, chunks = await gemini_stream_generate(
                        model, body.question, results
                    )
                    mode = "gemini"
                    yield sse(
                        "stage",
                        {
                            "id": "generation",
                            "label": "Generating grounded answer",
                            "detail": (
                                f"Completed · TTFT {ttft_ms:.0f} ms · "
                                f"{generation_ms:.0f} ms total · {chunks} chunks"
                            ),
                            "done": True,
                        },
                    )
                except Exception as exc:
                    # Never silently convert a provider failure into a successful-looking
                    # benchmark. Return the failure explicitly so latency measurements stay honest.
                    answer = f"Gemini generation failed: {str(exc)}"
                    mode = "gemini_error"
                    generation_ms = (time.perf_counter() - t) * 1000
                    yield sse(
                        "stage",
                        {
                            "id": "generation",
                            "label": "Gemini generation failed",
                            "detail": str(exc)[:300],
                            "done": True,
                            "error": True,
                        },
                    )

            total_ms = (time.perf_counter() - total) * 1000
            payload = {
                "question": body.question,
                "answer": answer,
                "mode": mode,
                "provider": "gemini",
                "model": model if answerable else None,
                "answerable": answerable,
                "scope_document_ids": scope or [],
                "evidence_coverage": round(
                    core.evidence_coverage(body.question, results), 4
                ),
                "citations": core.compact_citations(results) if answerable else [],
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

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
