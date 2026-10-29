from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import defaultdict
from typing import AsyncIterator, Awaitable, Callable

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

# Prefer the low-latency model for short, evidence-grounded document questions.
# Fallback order favours response speed first, then higher-capability models.
GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
]


def resolve_gemini_model(configured: str | None = None) -> str:
    """Use a supported Gemini model, falling back from stale environment settings."""
    requested = configured if configured is not None else os.getenv("GEMINI_MODEL")
    return requested if requested in GEMINI_MODELS else "gemini-3.5-flash-lite"


DEFAULT_GEMINI_MODEL = resolve_gemini_model()


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
        # Group by document identity, not filename: two distinct PDFs may share a filename.
        key = str(result.get("document_id") or result["filename"])
        grouped[key].append(result)
    groups = []
    for document_id, items in grouped.items():
        filename = items[0]["filename"]
        pages = []
        for item in items:
            pages.extend(range(int(item["page_start"]), int(item["page_end"]) + 1))
        groups.append({
            "source_id": len(groups) + 1,
            "document_id": document_id,
            "filename": filename,
            "pages": _page_label(pages),
            "chunks": len(items),
            "evidence": [
                {
                    "chunk_id": item["chunk_id"],
                    "pages": f"{item['page_start']}-{item['page_end']}",
                    "text": item["text"],
                }
                for item in items[:32]
            ],
        })
    return groups



def candidate_matches_question(
    question: str,
    item: dict,
    allow_semantic_for_relevant_document: bool = False,
) -> bool:
    """Accept wording matches or a strong dense match within a relevant document."""
    if float(item.get("score", 0.0)) <= 0:
        return False
    if core.query_term_match_score(question, str(item.get("text", ""))) > 0:
        return True
    # For profile queries, a source must first pass document-level attribute and
    # entity checks. Within that already-relevant source, dense similarity may
    # select a paraphrased passage without repeating the exact topic word.
    is_profile_query = core.profile_query_intent(question) is not None
    if is_profile_query and not allow_semantic_for_relevant_document:
        return False
    query_terms = set(core.content_tokens(question))
    return (
        len(query_terms) >= 2
        and float(item.get("dense", 0.0)) >= float(
            os.getenv("DENSE_SOURCE_RELEVANCE_THRESHOLD", "0.38")
        )
        and item.get("dense_rank") is not None
        and int(item["dense_rank"]) <= 10
    )


def diversify_reranked_results(
    question: str,
    ranked_results: list[dict],
    top_k: int,
    all_rows: list | None = None,
) -> list[dict]:
    """Select topic-relevant passages from documents matching the query as a whole."""
    source_rows = all_rows if all_rows is not None else ranked_results
    relevant_docs = core.relevant_document_ids(question, source_rows)
    semantic_docs = {
        str(item.get("document_id") or item.get("filename", ""))
        for item in ranked_results
        if candidate_matches_question(question, item)
        and core.query_term_match_score(question, str(item.get("text", ""))) == 0
    }
    if core.profile_query_intent(question) is None:
        relevant_docs.update(semantic_docs)
    if not relevant_docs:
        return []

    eligible = [
        item for item in ranked_results
        if str(item.get("document_id") or item.get("filename", "")) in relevant_docs
        and candidate_matches_question(
            question,
            item,
            allow_semantic_for_relevant_document=(
                str(item.get("document_id") or item.get("filename", "")) in relevant_docs
            ),
        )
    ]
    best_by_document: dict[str, dict] = {}
    for item in eligible:
        document_id = str(item.get("document_id") or item.get("filename", ""))
        current = best_by_document.get(document_id)
        key = (
            core.query_term_match_score(question, str(item.get("text", ""))),
            float(item.get("rerank_score", 0.0)),
            float(item.get("score", 0.0)),
        )
        current_key = (
            core.query_term_match_score(question, str(current.get("text", ""))),
            float(current.get("rerank_score", 0.0)),
            float(current.get("score", 0.0)),
        ) if current is not None else None
        if current is None or key > current_key:
            best_by_document[document_id] = item

    document_winners = sorted(
        best_by_document.values(),
        key=lambda item: (
            -core.query_term_match_score(question, str(item.get("text", ""))),
            -float(item.get("rerank_score", 0.0)),
            -float(item.get("score", 0.0)),
        ),
    )
    effective_k = min(10, max(top_k, len(document_winners)))
    selected: list[dict] = []
    selected_ids: set[str] = set()

    for item in document_winners:
        if len(selected) >= effective_k:
            break
        chunk_id = str(item.get("chunk_id", ""))
        selected.append(item)
        selected_ids.add(chunk_id)

    # Fill only from source documents that independently match the full query.
    for item in eligible:
        if len(selected) >= effective_k:
            break
        chunk_id = str(item.get("chunk_id", ""))
        if chunk_id not in selected_ids:
            selected.append(item)
            selected_ids.add(chunk_id)

    return selected


SUMMARY_INTENT_RE = re.compile(
    r"\b(?:summarize|summarise|summarizing|summarising|summary|overview|key points|main ideas)\b",
    re.IGNORECASE,
)


def is_summary_request(question: str) -> bool:
    """Detect broad summaries that have no meaningful lexical query terms."""
    return bool(SUMMARY_INTENT_RE.search(question))


METHOD_INTENT_RE = re.compile(
    r"\b(?:explain|describe|discuss|outline)\b.{0,100}"
    r"\b(?:main|primary|proposed|key)?\s*"
    r"(?:method(?:ology)?|approach|technique|framework|process|algorithm)\b"
    r"|\b(?:main|primary|proposed|key)\s+"
    r"(?:method(?:ology)?|approach|technique|framework|process|algorithm)\b",
    re.IGNORECASE,
)


def is_method_explanation_request(question: str) -> bool:
    """Recognize requests to explain a document's method even without topic keywords."""
    return bool(METHOD_INTENT_RE.search(question))


def select_summary_evidence(
    rows: list,
    max_total_chunks: int = 32,
    max_chunks_per_document: int = 20,
) -> list[dict]:
    """Sample chunks across each selected PDF instead of ranking generic words like 'PDF'."""
    grouped: dict[str, list] = {}
    for row in rows:
        document_id = str(row["document_id"])
        grouped.setdefault(document_id, []).append(row)
    if not grouped:
        return []

    slots_per_document = min(
        max_chunks_per_document,
        max(1, max_total_chunks // len(grouped)),
    )
    selected: list[dict] = []
    for document_rows in grouped.values():
        ordered = sorted(
            document_rows,
            key=lambda row: (int(row["page_start"]), int(row["page_end"]), str(row["id"])),
        )
        count = min(slots_per_document, len(ordered))
        if count == 0:
            continue
        if count == len(ordered):
            indices = list(range(count))
        elif count == 1:
            indices = [0]
        else:
            indices = sorted({
                round(index * (len(ordered) - 1) / (count - 1))
                for index in range(count)
            })
        for index in indices:
            row = ordered[index]
            selected.append({
                "chunk_id": row["id"],
                "document_id": row["document_id"],
                "filename": row["filename"],
                "page_start": row["page_start"],
                "page_end": row["page_end"],
                "score": 1.0,
                "bm25": 0.0,
                "tfidf": 0.0,
                "rerank_score": 1.0,
                "text": row["text"],
            })
    return selected[:max_total_chunks]


def build_prompt(question: str, results: list[dict]) -> tuple[str, str]:
    groups = evidence_groups(results)
    context = "\n\n".join(
        f"[Source {group['source_id']}] {group['filename']} pages {group['pages']}\n" +
        "\n".join(item["text"] for item in group["evidence"])
        for group in groups
    )
    if is_summary_request(question):
        system = (
            "You are the final answer layer of a grounded enterprise RAG system. "
            "The user requested a document summary. Summarize each supplied source separately, "
            "using its filename as a heading and 2-4 concise bullet points with key ideas, "
            "goals, methods, findings, and important details when present in the evidence. "
            "Cite every bullet with [Source N], using only source numbers supplied below. "
            "Use only the provided passages. Do not invent details or imply unseen sections "
            "were inspected; describe the answer as a concise overview of the available passages. "
            "Return only the summary, never reasoning or process commentary. "
            "Aim for a clear, useful summary under 250 words total. "
            "For mathematical notation, use readable Unicode/plain text and never raw LaTeX."
        )
    elif is_method_explanation_request(question):
        system = (
            "You are the final answer layer of a grounded enterprise RAG system. "
            "The user asks about the method or approach described in selected documents. "
            "Explain the method for each supplied source separately, if it contains one. "
            "For each source, identify the problem or goal, the main steps or components, "
            "how the method works, and any stated results or limitations. Use concise bullets "
            "or a short structured explanation, with citations such as [Source N] on each claim. "
            "Do not confuse a document's subject/background with its proposed method. "
            "If a source does not describe a method in the supplied passages, say that the "
            "available passages do not describe one rather than inventing a method. "
            "Use only the supplied evidence and do not imply unseen pages were inspected. "
            "Return only the explanation, normally under 250 words total, and never raw LaTeX."
        )
    else:
        system = (
            "You are the final answer layer of a grounded enterprise RAG system. "
            "Answer ONLY from the supplied evidence. Never use outside knowledge. "
            "Return only the final answer, never reasoning, analysis, planning, or process commentary. "
            "Answer the exact question directly in 1-3 COMPLETE sentences, normally under 90 words. "
            "Finish the final sentence; never stop at an unfinished clause. "
            "Cite factual claims with [Source N], using only source numbers present below. "
            "For mathematical notation, use readable Unicode/plain text: α, β, γ, ρᵢ, ρ̄. "
            "NEVER output LaTeX delimiters ($, $$, \\\\(, \\\\), \\\\[ or \\\\]) or raw commands such as \\\\alpha. "
            "If an equation is needed, write it plainly, e.g. QIS = αI(ρᵢ) + βQJSD(ρᵢ) − γF(ρᵢ, ρ̄). "
            "If the supplied evidence is insufficient, say so clearly instead of guessing."
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


def clean_answer(answer: str) -> str:
    """Normalize common LaTeX leakage from model output for the UI."""
    text = answer.strip()
    text = text.replace("$$", "").replace("$", "")
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace("\\[", "").replace("\\]", "")
    replacements = {
        r"\\alpha": "α", r"\\beta": "β", r"\\gamma": "γ",
        r"\\delta": "δ", r"\\epsilon": "ε", r"\\theta": "θ",
        r"\\lambda": "λ", r"\\mu": "μ", r"\\rho": "ρ",
        r"\\sigma": "σ", r"\\tau": "τ", r"\\phi": "φ",
        r"\\psi": "ψ", r"\\omega": "ω",
        r"\\infty": "∞", r"\\pm": "±", r"\\times": "×",
        r"\\cdot": "·", r"\\leq": "≤", r"\\geq": "≥",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\\(?:text|mathrm|mathbf|mathit)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\bar\{([^{}]*)\}", r"\1̄", text)
    text = re.sub(r"\\hat\{([^{}]*)\}", r"\1̂", text)
    # Convert common underscore subscripts into Unicode for simple identifiers.
    subs = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")
    text = re.sub(r"([A-Za-zα-ωΑ-Ω])_([0-9+\-=()]+)", lambda m: m.group(1) + m.group(2).translate(subs), text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text.strip()


async def gemini_stream_generate(
    model: str,
    question: str,
    results: list[dict],
    on_text: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[str, float, float, int]:
    key = gemini_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    system, user = build_prompt(question, results)
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "maxOutputTokens": 480 if (is_summary_request(question) or is_method_explanation_request(question)) else 160,
            # Simple factual RAG questions do not need deep reasoning.
            "thinkingConfig": {
                "thinkingLevel": "minimal" if model in {
                    "gemini-3.5-flash-lite", "gemini-3.6-flash"
                } else "low"
            },
        },
    }
    started = time.perf_counter()
    ttft_ms = 0.0
    chunks = 0
    pieces: list[str] = []
    finish_reason = None
    timeout = httpx.Timeout(connect=8.0, read=35.0, write=8.0, pool=8.0)
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
                candidate = candidates[0]
                finish_reason = candidate.get("finishReason") or finish_reason
                content = candidate.get("content") or {}
                for part in content.get("parts") or []:
                    text = part.get("text") or ""
                    if not text:
                        continue
                    if ttft_ms == 0.0:
                        ttft_ms = (time.perf_counter() - started) * 1000
                    pieces.append(text)
                    chunks += 1
                    if on_text is not None:
                        await on_text(text)
    generation_ms = (time.perf_counter() - started) * 1000
    answer = clean_answer("".join(pieces))
    if not answer:
        raise RuntimeError("Gemini returned an empty answer")
    if finish_reason == "MAX_TOKENS":
        raise RuntimeError("Gemini reached the output limit; answer was incomplete")
    return answer, ttft_ms, generation_ms, chunks


async def gemini_stream_generate_with_fallback(
    model: str,
    question: str,
    results: list[dict],
    on_text: Callable[[str], Awaitable[None]] | None = None,
    on_reset: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[str, float, float, int, str]:
    """Allow normal model startup time, then fail over with a bounded total budget."""
    requested = resolve_gemini_model(model)
    models = [requested] + [item for item in GEMINI_MODELS if item != requested]
    failures: list[str] = []
    generation_started = time.perf_counter()
    first_token_ms: float | None = None

    budget_seconds = max(15.0, float(os.getenv("GEMINI_GENERATION_BUDGET_SECONDS", "40")))
    request_timeout = max(2.0, float(os.getenv("GEMINI_REQUEST_TIMEOUT_SECONDS", "28")))
    first_token_timeout = max(0.5, float(os.getenv("GEMINI_FIRST_TOKEN_TIMEOUT_SECONDS", "12")))
    fallback_request_timeout = max(
        2.0, float(os.getenv("GEMINI_FALLBACK_REQUEST_TIMEOUT_SECONDS", "15"))
    )
    fallback_first_token_timeout = max(
        0.5, float(os.getenv("GEMINI_FALLBACK_FIRST_TOKEN_TIMEOUT_SECONDS", "6"))
    )
    deadline = time.monotonic() + budget_seconds

    for model_index, candidate_model in enumerate(models):
        # Retry the primary model only when it promptly returns a transient 5xx/429.
        attempts = 2 if model_index == 0 else 1
        model_request_timeout = request_timeout if model_index == 0 else fallback_request_timeout
        model_first_token_timeout = first_token_timeout if model_index == 0 else fallback_first_token_timeout
        for attempt in range(attempts):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if (model_index > 0 or attempt > 0) and on_reset is not None:
                await on_reset(candidate_model)

            first_text = asyncio.Event()

            async def relay_text(piece: str) -> None:
                nonlocal first_token_ms
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - generation_started) * 1000
                first_text.set()
                if on_text is not None:
                    await on_text(piece)

            attempt_budget = min(model_request_timeout, remaining)
            attempt_deadline = time.monotonic() + attempt_budget
            generation_task = asyncio.create_task(
                gemini_stream_generate(
                    candidate_model, question, results, on_text=relay_text
                )
            )
            first_token_waiter = asyncio.create_task(first_text.wait())
            try:
                done, _ = await asyncio.wait(
                    {generation_task, first_token_waiter},
                    timeout=min(model_first_token_timeout, attempt_budget),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if generation_task in done:
                    answer, _, _, chunks = await generation_task
                    elapsed_ms = (time.perf_counter() - generation_started) * 1000
                    return answer, first_token_ms or elapsed_ms, elapsed_ms, chunks, candidate_model

                if first_token_waiter not in done:
                    generation_task.cancel()
                    await asyncio.gather(generation_task, return_exceptions=True)
                    raise asyncio.TimeoutError(
                        f"{candidate_model} did not send a first token within "
                        f"{model_first_token_timeout:.1f} seconds"
                    )

                remaining_attempt = attempt_deadline - time.monotonic()
                if remaining_attempt <= 0:
                    raise asyncio.TimeoutError(f"{candidate_model} request timed out")
                answer, _, _, chunks = await asyncio.wait_for(
                    generation_task, timeout=remaining_attempt
                )
                elapsed_ms = (time.perf_counter() - generation_started) * 1000
                return answer, first_token_ms or elapsed_ms, elapsed_ms, chunks, candidate_model
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                message = str(exc) or type(exc).__name__
                match = re.search(r"Gemini HTTP (\d{3})", message)
                status = int(match.group(1)) if match else None
                is_timeout = isinstance(exc, (asyncio.TimeoutError, httpx.TimeoutException))
                is_network_error = isinstance(exc, httpx.NetworkError)
                transient_status = status in {408, 429, 500, 502, 503, 504}
                unavailable_model = status == 404
                if not (is_timeout or is_network_error or transient_status or unavailable_model):
                    raise

                failures.append(f"{candidate_model}: {message[:160]}")
                # Avoid another full attempt when the model fails to start. For actual
                # 429/5xx responses, one short retry precedes model fallback.
                if transient_status and attempt == 0:
                    await asyncio.sleep(0.25)
                    continue
                break
            finally:
                if not generation_task.done():
                    generation_task.cancel()
                    await asyncio.gather(generation_task, return_exceptions=True)
                if not first_token_waiter.done():
                    first_token_waiter.cancel()
                    await asyncio.gather(first_token_waiter, return_exceptions=True)

        if time.monotonic() >= deadline:
            break

    detail = " | ".join(failures[-3:])
    elapsed = time.perf_counter() - generation_started
    raise RuntimeError(
        f"Gemini could not complete the answer within {budget_seconds:.0f} seconds "
        f"(elapsed {elapsed:.1f}s). The primary model gets a "
        f"{first_token_timeout:.0f}s first-token window and fallbacks get "
        f"{fallback_first_token_timeout:.0f}s. The service may be overloaded; "
        "please retry shortly. " + detail
    )


@app.get("/api/runtime")
async def runtime():
    return {
        "provider": "gemini",
        "models": GEMINI_MODELS,
        "default_model": DEFAULT_GEMINI_MODEL,
        "streaming": True,
        "retrieval": {
            "sparse": ["BM25", "TF-IDF"],
            "dense_model": core.dense_retrieval.model_name(),
            "fusion": "weighted reciprocal rank fusion",
            "weights": {
                "bm25": float(os.getenv("HYBRID_BM25_WEIGHT", "0.35")),
                "tfidf": float(os.getenv("HYBRID_TFIDF_WEIGHT", "0.15")),
                "dense": float(os.getenv("HYBRID_DENSE_WEIGHT", "0.50")),
            },
        },
    }


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
        model = resolve_gemini_model(body.model or DEFAULT_GEMINI_MODEL)
        model_used = model
        yield sse("stage", {"id": "scope", "label": "Preparing scope", "detail": "Loading selected documents"})
        try:
            rows = core.load_chunks(document_ids=scope) if scope else core.load_chunks()
            scoped_document_ids = sorted({str(row["document_id"]) for row in rows})
            multi_document = len(scoped_document_ids) > 1
            corpus_texts = [str(row["text"]) for row in rows]
            corrected_question = core.correct_query_terms(body.question, corpus_texts)
            search_question = core.expand_profile_query(corrected_question, corpus_texts)
            normalization_notes = []
            if corrected_question.casefold() != body.question.casefold():
                normalization_notes.append(f"spelling normalized: {corrected_question}")
            if search_question.casefold() != corrected_question.casefold():
                intent = core.profile_query_intent(corrected_question)
                if intent:
                    normalization_notes.append(f"{intent} topic expansion enabled")
            correction_note = (" · " + " · ".join(normalization_notes)) if normalization_notes else ""
            summary_request = is_summary_request(body.question)
            method_request = is_method_explanation_request(body.question)
            overview_request = summary_request or method_request
            t = time.perf_counter()
            if overview_request:
                results = select_summary_evidence(rows)
                candidates = list(results)
                retrieval_ms = (time.perf_counter() - t) * 1000
                answerable = bool(results)
                yield sse("stage", {
                    "id": "retrieval",
                    "label": "Document summary retrieval" if summary_request else "Method explanation retrieval",
                    "detail": (
                        f"Selected {len(results)} representative passages across "
                        f"{len({item['document_id'] for item in results})} selected documents"
                    ),
                    "done": True,
                })
                rerank_ms = 0.0
                yield sse("stage", {
                    "id": "rerank",
                    "label": "Preparing summary context" if summary_request else "Preparing method explanation context",
                    "detail": "Balanced page coverage across selected documents",
                    "done": True,
                })
            else:
                yield sse("stage", {
                    "id": "retrieval",
                    "label": "Hybrid retrieval",
                    "detail": (
                        f"Searching {len(rows)} indexed chunks across {len(scoped_document_ids)} "
                        f"documents with BM25 + TF-IDF + dense semantics (weighted RRF){correction_note}"
                    ),
                    "active": True,
                })
                candidates = core.hybrid_retrieve(
                    search_question,
                    rows,
                    max(body.top_k, core.RETRIEVAL_CANDIDATES),
                    diversify_documents=multi_document,
                )
                retrieval_ms = (time.perf_counter() - t) * 1000
                yield sse("stage", {
                    "id": "retrieval",
                    "label": "Hybrid retrieval",
                    "detail": (
                        f"Found {len(candidates)} candidates across "
                        f"{len({item['document_id'] for item in candidates})} sources · "
                        f"{retrieval_ms:.0f} ms"
                    ),
                    "done": True,
                })
                yield sse("stage", {
                    "id": "rerank",
                    "label": "Reranking evidence",
                    "detail": "Ranking relevant passages while preserving source diversity",
                    "active": True,
                })
                t = time.perf_counter()
                if multi_document:
                    reranked = core.rerank(search_question, candidates, len(candidates))
                    results = diversify_reranked_results(
                        search_question, reranked, body.top_k, all_rows=rows
                    )
                else:
                    results = core.rerank(search_question, candidates, body.top_k)
                rerank_ms = (time.perf_counter() - t) * 1000
                answerable = core.has_sufficient_evidence(search_question, results)
                yield sse("stage", {
                    "id": "rerank",
                    "label": "Reranking evidence",
                    "detail": f"Selected {len(results)} evidence chunks · {rerank_ms:.1f} ms",
                    "done": True,
                })
            if not answerable:
                answer = core.ABSTAIN_MESSAGE
                mode = "abstain"
                yield sse("stage", {"id": "generation", "label": "Grounding guard", "detail": "Evidence threshold not met · Gemini generation skipped", "done": True})
            else:
                yield sse("stage", {"id": "generation", "label": "Generating grounded answer", "detail": f"Gemini · {model} · streaming response", "active": True})
                gen_started = time.perf_counter()
                token_events: asyncio.Queue[str] = asyncio.Queue()

                async def enqueue_text(piece: str) -> None:
                    await token_events.put(sse("token", {"text": piece}))

                async def reset_partial_answer(next_model: str) -> None:
                    await token_events.put(sse("generation_reset", {"model": next_model}))
                    await token_events.put(sse("stage", {
                        "id": "generation",
                        "label": "Trying fallback model",
                        "detail": f"Switching to {next_model}",
                        "active": True,
                    }))

                generation_task = asyncio.create_task(
                    gemini_stream_generate_with_fallback(
                        model,
                        body.question,
                        results,
                        on_text=enqueue_text,
                        on_reset=reset_partial_answer,
                    )
                )
                try:
                    while not generation_task.done() or not token_events.empty():
                        try:
                            event = await asyncio.wait_for(token_events.get(), timeout=0.1)
                            yield event
                        except asyncio.TimeoutError:
                            continue
                    answer, ttft_ms, generation_ms, chunks, model_used = await generation_task
                    mode = "gemini"
                    fallback_note = f" · model {model_used}" if model_used != model else ""
                    yield sse("stage", {"id": "generation", "label": "Answer ready", "detail": f"Completed{fallback_note} · first token {ttft_ms:.0f} ms · generation {generation_ms:.0f} ms · {chunks} chunks", "done": True})
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
                "provider": "gemini" if mode in {"gemini", "gemini_error"} else "local",
                "model": model_used if mode == "gemini" else None,
                "answerable": answerable,
                "scope_document_ids": scoped_document_ids,
                "evidence_coverage": round(core.evidence_coverage(search_question, results), 4),
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
:root{color-scheme:dark;--bg:#080d17;--surface:#0e1727;--surface-2:#101c2e;--surface-3:#0a1220;--border:#24344d;--border-strong:#354967;--text:#f3f6fc;--muted:#96a5bc;--subtle:#6f829e;--accent:#8b9bff;--accent-2:#7869e8;--success:#5ed7ab;--danger:#ff9d9d;--shadow:0 24px 80px rgba(0,0,0,.3)}
*{box-sizing:border-box}html{min-height:100%;background:var(--bg);scroll-behavior:smooth}
body{min-height:100vh;margin:0;background:radial-gradient(1100px 600px at 5% -18%,rgba(74,94,174,.2),transparent 62%),var(--bg);color:var(--text);font:14px/1.5 Inter,"Segoe UI",system-ui,-apple-system,sans-serif;-webkit-font-smoothing:antialiased}
button,input,textarea{font:inherit}button{color:inherit}button:focus-visible,input:focus-visible,textarea:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
.top{min-height:72px;display:flex;align-items:center;gap:18px;padding:12px clamp(18px,4vw,58px);border-bottom:1px solid rgba(53,73,103,.58);background:rgba(8,13,23,.82);backdrop-filter:blur(16px)}
.brand{display:flex;align-items:center;gap:12px;font-weight:750;letter-spacing:-.02em}
.brand-icon{width:36px;height:36px;display:grid;place-items:center;flex:0 0 36px;border:1px solid rgba(161,173,255,.3);border-radius:12px;background:linear-gradient(145deg,rgba(139,155,255,.22),rgba(120,105,232,.08));color:#c3cbff;font-size:20px}
.brand-copy{display:flex;flex-direction:column;gap:1px}.brand-copy strong{font-size:14px}.brand-copy small{color:var(--subtle);font-size:9px;font-weight:650;letter-spacing:.16em}
.pill{margin-left:auto;max-width:min(48vw,360px);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;display:flex;align-items:center;gap:8px;border:1px solid var(--border);padding:7px 11px;border-radius:999px;color:#c1cce0;background:rgba(16,28,46,.7);font-size:11px}
.pill:before{content:"";width:7px;height:7px;border-radius:50%;background:var(--success);box-shadow:0 0 0 3px rgba(94,215,171,.12);flex:0 0 7px}.pill[data-state="error"]:before{background:var(--danger);box-shadow:0 0 0 3px rgba(255,157,157,.12)}
.main{width:min(calc(100% - 40px),1240px);margin:0 auto;padding:34px 0 40px}
.head{display:flex;justify-content:space-between;align-items:center;gap:24px;margin-bottom:22px}.head-copy{min-width:0}.eyebrow{display:flex;align-items:center;gap:8px;color:#aab7ff;font-size:10px;font-weight:750;letter-spacing:.16em}.eyebrow:before{content:"";width:18px;height:1px;background:var(--accent)}
.head h1{margin:8px 0 5px;font-size:clamp(28px,3.2vw,38px);line-height:1.12;letter-spacing:-.045em;font-weight:720}.head-subtitle{max-width:650px;color:var(--muted);font-size:13px}
.head-actions{display:flex;align-items:center;gap:10px;flex-shrink:0}.primary,.mini,.scope,.send,.query-chip{display:inline-flex;align-items:center;justify-content:center;gap:8px;cursor:pointer;transition:background .16s,border-color .16s,transform .16s,opacity .16s}
.primary{border:1px solid rgba(177,186,255,.25);background:linear-gradient(135deg,#788bff,#8570e8);color:#fff;padding:11px 16px;border-radius:11px;font-size:12px;font-weight:700;box-shadow:0 7px 20px rgba(107,111,231,.16)}
.primary:hover{transform:translateY(-1px);filter:brightness(1.07)}.primary:active,.send:active{transform:translateY(0)}.button-icon{font-size:17px;line-height:1;font-weight:400}button:disabled{opacity:.55;cursor:not-allowed;transform:none!important}
.panel{background:linear-gradient(155deg,rgba(15,24,41,.98),rgba(11,19,33,.98));border:1px solid var(--border);border-radius:18px;overflow:hidden;box-shadow:0 15px 55px rgba(0,0,0,.12)}
.chat{min-height:clamp(560px,calc(100vh - 220px),820px);display:flex;flex-direction:column}
.bar{min-height:76px;padding:17px 22px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px;background:rgba(17,28,46,.45)}
.workspace-title{display:flex;align-items:center;gap:12px;min-width:0}.workspace-mark{width:36px;height:36px;display:grid;place-items:center;flex:0 0 36px;border-radius:11px;border:1px solid var(--border);background:#0b1524;color:#bac5ff;font-size:17px}
.workspace-title b{display:block;font-size:13px;letter-spacing:-.01em}.muted{color:var(--muted);font-size:11px}
.scope{margin-left:auto;border:1px solid var(--border-strong);background:#101b2c;color:#dfe7f4;border-radius:10px;padding:9px 12px;font-size:12px;white-space:nowrap}.scope:hover,.mini:hover,.query-chip:hover{border-color:#576d92;background:#15233a}
.mini{border:1px solid var(--border);background:transparent;color:#c4cfe1;padding:9px 12px;border-radius:10px;font-size:12px}
.body{padding:22px;flex:1;display:flex;flex-direction:column;gap:15px;min-width:0}
.empty-state{flex:1;min-height:270px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:28px 16px 35px}
.empty-icon{width:58px;height:58px;display:grid;place-items:center;margin-bottom:20px;border-radius:18px;border:1px solid #35456a;background:linear-gradient(145deg,rgba(139,155,255,.16),rgba(15,26,43,.86));color:#c3cbff;box-shadow:0 12px 38px rgba(0,0,0,.16)}
.empty-icon svg{width:27px;height:27px}.empty-kicker{color:#aab7ff;font-size:9px;font-weight:750;letter-spacing:.16em}
.empty-state h2{margin:8px 0 7px;font-size:22px;letter-spacing:-.035em;font-weight:700}.empty-state p{max-width:470px;margin:0;color:var(--muted);font-size:12px;line-height:1.7}
.prompt-examples{display:flex;justify-content:center;flex-wrap:wrap;gap:8px;margin-top:20px}
.query-chip{border:1px solid var(--border);background:rgba(16,28,46,.7);color:#c7d1e4;border-radius:999px;padding:8px 11px;font-size:11px}.query-chip span{color:#9eabff}
.progress-status{display:flex;align-items:center;gap:9px;min-height:34px;padding:8px 11px;border:1px solid rgba(139,155,255,.17);border-radius:10px;background:rgba(139,155,255,.055);color:#c9d2fa;font-size:11px}
.progress-dot{width:7px;height:7px;border-radius:50%;background:var(--accent);flex:0 0 7px;box-shadow:0 0 0 4px rgba(139,155,255,.1)}
.answer{background:linear-gradient(145deg,#0c1625,#0a1320);border:1px solid #30435f;border-radius:15px;padding:20px 21px;line-height:1.8;white-space:pre-wrap;overflow-wrap:anywhere;font-size:15px;color:#e8eef8}
.sources{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));gap:10px}
.source{width:100%;text-align:left;display:block;border:1px solid var(--border);padding:14px;border-radius:12px;background:rgba(10,19,33,.82);cursor:pointer;transition:border-color .16s,background .16s,transform .16s}
.source:hover{border-color:#576b91;background:#111f33;transform:translateY(-1px)}.source:focus-visible{outline:2px solid var(--accent);outline-offset:2px}.source b{display:flex;align-items:center;gap:7px;font-size:11px;font-weight:680;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.source .pages{font-size:10px;color:var(--muted);margin-top:6px}.source .view{font-size:10px;color:#aebaff;margin-top:11px}
.metrics{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:9px}.metric{min-width:0;padding:12px 13px;border:1px solid var(--border);border-radius:11px;background:rgba(8,17,30,.68)}.metric span{font-size:10px;color:var(--subtle);letter-spacing:.01em}.metric b{display:block;margin-top:5px;font-size:14px;letter-spacing:-.015em;font-variant-numeric:tabular-nums}
.composer{margin-top:auto;padding-top:5px}.box{display:flex;align-items:flex-end;gap:12px;border:1px solid var(--border-strong);background:#091321;border-radius:15px;padding:10px 11px 10px 15px;transition:border-color .16s,box-shadow .16s}.box:focus-within{border-color:#7786d7;box-shadow:0 0 0 3px rgba(139,155,255,.09)}
.box textarea{flex:1;width:100%;min-width:0;background:transparent;border:0;outline:0;color:var(--text);resize:vertical;min-height:43px;max-height:180px;padding:10px 0;line-height:1.55;font-size:13px}.box textarea::placeholder{color:#72829b}
.send{width:42px;height:42px;flex:0 0 42px;border:0;border-radius:12px;background:linear-gradient(135deg,#8194ff,#8170e9);color:white;font-size:20px;box-shadow:0 6px 18px rgba(104,111,231,.18)}.send:hover{filter:brightness(1.08);transform:translateY(-1px)}
.composer-hint{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:9px;color:var(--subtle);font-size:10px}kbd{font:inherit;font-size:9px;border:1px solid var(--border);border-radius:5px;padding:2px 5px;color:#9aa9c1;background:#101b2c}
.overlay,.modal{position:fixed;inset:0;background:rgba(3,7,14,.74);backdrop-filter:blur(9px);display:none;align-items:center;justify-content:center;z-index:20;padding:18px}
.loader,.modalbox{width:min(620px,100%);background:var(--surface);border:1px solid #334663;border-radius:18px;box-shadow:var(--shadow)}.loader{padding:25px}
.loader-header{display:flex;align-items:center;gap:14px;margin-bottom:19px}.spinner{width:28px;height:28px;flex:0 0 28px;border:3px solid #33425c;border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}
.loader h3{margin:0 0 3px;font-size:17px;letter-spacing:-.02em}.loader .muted{font-size:11px}@keyframes spin{to{transform:rotate(360deg)}}
.stage{display:flex;gap:11px;padding:11px;border:1px solid transparent;border-radius:10px;color:#8493ab;font-size:11px}.stage b{font-size:11px;font-weight:680}.stage.active{border-color:#30415f;background:#101b2d;color:#eaf0f9}.stage.done{color:var(--success)}
.dot{width:8px;height:8px;flex:0 0 8px;border-radius:50%;background:#52627c;margin-top:5px}.active .dot{background:var(--accent);box-shadow:0 0 0 4px rgba(139,155,255,.1)}.done .dot{background:var(--success)}
.modalbox{max-height:min(82vh,760px);overflow:auto}.modalhead{padding:18px 20px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;gap:14px}.modalhead b{font-size:13px}
.evidence{padding:16px 20px;border-bottom:1px solid rgba(36,52,77,.7)}.evidence:last-child{border-bottom:0}.evidencehead{display:flex;justify-content:space-between;gap:10px;font-size:11px}.evidence pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.7 Inter,"Segoe UI",system-ui,sans-serif;color:#c3cfe1;background:var(--surface-3);border:1px solid var(--border);border-radius:10px;padding:13px;margin:10px 0 0}
.docrow{display:flex;align-items:center;gap:12px;padding:14px 20px;border-bottom:1px solid rgba(36,52,77,.7);cursor:pointer}.docrow:hover{background:rgba(139,155,255,.04)}.docrow input{width:16px;height:16px;accent-color:var(--accent)}.docrow .info{flex:1;min-width:0}.docrow .info b{font-size:12px;overflow-wrap:anywhere}
.modalfoot{padding:14px 20px;display:flex;justify-content:space-between;align-items:center;gap:8px;position:sticky;bottom:0;background:var(--surface);border-top:1px solid var(--border)}
.toast{position:fixed;bottom:22px;right:22px;z-index:40;max-width:min(420px,calc(100vw - 32px));padding:12px 15px;border:1px solid var(--border-strong);border-radius:12px;background:#111d30;color:#e5edf8;box-shadow:var(--shadow);font-size:12px;opacity:0;transform:translateY(8px);pointer-events:none;transition:opacity .18s,transform .18s}
.toast.visible{opacity:1;transform:translateY(0)}.toast[data-kind="success"]{border-color:rgba(94,215,171,.5)}.toast[data-kind="error"]{border-color:rgba(255,157,157,.55)}
@media(max-width:900px){.main{width:min(calc(100% - 32px),1240px);padding-top:24px}.body{padding:16px}.metrics{grid-template-columns:repeat(3,minmax(0,1fr))}.chat{min-height:580px}}
@media(max-width:620px){.top{min-height:64px;padding:10px 16px;gap:10px}.brand-icon{width:32px;height:32px;flex-basis:32px;border-radius:10px}.brand-copy small{font-size:8px}.pill{max-width:42vw;font-size:10px;padding:6px 9px}.main{width:calc(100% - 24px);padding:23px 0 26px}.head{align-items:flex-start;flex-direction:column;gap:16px;margin-bottom:17px}.head h1{font-size:29px}.head-actions{width:100%}.primary{padding:10px 13px}.bar{padding:14px;gap:8px;flex-wrap:wrap}.workspace-title{flex:1 1 100%;margin-bottom:4px}.scope{margin-left:0;flex:1}.body{padding:12px;gap:12px}.chat{min-height:calc(100svh - 185px);min-height:570px}.empty-state{min-height:260px;padding:24px 5px}.empty-state h2{font-size:20px}.empty-state p{font-size:11px}.prompt-examples{gap:6px;margin-top:15px}.query-chip{font-size:10px;padding:7px 9px}.answer{padding:15px;font-size:14px}.sources{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,minmax(0,1fr));gap:7px}.metric:last-child{grid-column:span 2}.box{padding:8px 9px 8px 12px;gap:8px}.box textarea{font-size:13px}.send{width:39px;height:39px;flex-basis:39px}.composer-hint{font-size:9px}.loader{padding:19px}.modalhead,.evidence{padding:15px}.modalfoot{padding:12px 15px}.composer-hint{align-items:flex-start;flex-direction:column;gap:4px}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{scroll-behavior:auto!important;animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}}
</style></head><body><header class="top"><div class="brand"><span class="brand-icon" aria-hidden="true">✦</span><span class="brand-copy"><strong>RAG Studio</strong><small>DOCUMENT INTELLIGENCE</small></span></div><div id="provider" class="pill" role="status" aria-live="polite">Connecting Gemini…</div></header><main class="main"><div class="head"><div class="head-copy"><div class="eyebrow">GROUNDED DOCUMENT INTELLIGENCE</div><h1>Ask your documents</h1><div class="head-subtitle">Find answers across your PDFs, with evidence you can inspect and verify.</div></div><div class="head-actions"><input id="file" type="file" accept="application/pdf" aria-label="Choose a PDF to upload" style="display:none"><button id="addPdfButton" class="primary" type="button"><span class="button-icon" aria-hidden="true">＋</span><span>Add PDF</span></button></div></div><section class="panel chat" aria-label="Document question workspace"><div class="bar"><div class="workspace-title"><div class="workspace-mark" aria-hidden="true">↗</div><div><b>Document assistant</b><div id="scopeText" class="muted">Using all documents</div></div></div><button class="scope" type="button" aria-controls="modal" onclick="openScope()"><span id="scopeBtn">All documents</span><span aria-hidden="true">⌄</span></button><button class="mini" type="button" onclick="clearChat()">Clear answer</button></div><div class="body"><section id="emptyState" class="empty-state"><div class="empty-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M7 3.75h7l4.25 4.5v11A1.75 1.75 0 0 1 16.5 21h-9A1.75 1.75 0 0 1 5.75 19.25v-13.75A1.75 1.75 0 0 1 7.5 3.75Z"/><path d="M14 4v5h4.25M8.5 13h7M8.5 16.5h5"/></svg></div><div class="empty-kicker" id="emptyKicker">YOUR WORKSPACE</div><h2 id="emptyTitle">Your documents, ready to explore</h2><p id="emptyDescription">Add a PDF, ask a specific question, and inspect the cited passages before trusting an answer.</p><div class="prompt-examples" id="promptExamples"><button class="query-chip" type="button" onclick="setPrompt('Summarize the main ideas in these documents.')">Summarize key ideas <span aria-hidden="true">↗</span></button><button class="query-chip" type="button" onclick="setPrompt('Explain the main method described in these documents.')">Explain a method <span aria-hidden="true">↗</span></button><button class="query-chip" type="button" onclick="setPrompt('What evidence supports the main conclusion?')">Find supporting evidence <span aria-hidden="true">↗</span></button></div></section><div id="progressStatus" class="progress-status" role="status" aria-live="polite" style="display:none"><span class="progress-dot" aria-hidden="true"></span><span id="progressText">Preparing answer…</span></div><div id="answer" class="answer" aria-live="polite" style="display:none"></div><div id="sources" class="sources"></div><div id="metrics" class="metrics" style="display:none"><div class="metric"><span>Retrieval</span><b id="mR">—</b></div><div class="metric"><span>Rerank</span><b id="mK">—</b></div><div class="metric"><span>Generation</span><b id="mG">—</b></div><div class="metric"><span>First token</span><b id="mF">—</b></div><div class="metric"><span>Total</span><b id="mT">—</b></div></div><div class="composer"><div class="box"><textarea id="q" rows="1" aria-label="Ask a question about selected documents" placeholder="Ask a question about your PDFs…"></textarea><button id="sendBtn" class="send" type="button" onclick="ask()" aria-label="Send question" title="Send question">↑</button></div><div class="composer-hint"><span>Answers are grounded in retrieved evidence</span><span><kbd>Enter</kbd> send · <kbd>Shift</kbd> + <kbd>Enter</kbd> new line</span></div></div></div></section></main><div id="overlay" class="overlay" aria-live="polite" aria-busy="true"><div class="loader"><div class="loader-header"><div class="spinner" aria-hidden="true"></div><div><h3 id="loadTitle">Preparing retrieval…</h3><div id="loadDetail" class="muted">Please wait</div></div></div><div id="stages"></div></div></div><div id="modal" class="modal" role="dialog" aria-modal="true" aria-labelledby="scopeTitle"><div class="modalbox"><div class="modalhead"><div><b id="scopeTitle">Retrieval scope</b><div class="muted">Choose exactly which documents can be searched</div></div><button class="mini" onclick="closeScope()">Close</button></div><div style="padding:14px 18px;display:flex;gap:8px"><button class="mini" onclick="selectAll()">Select all</button><button class="mini" onclick="clearSelect()">Clear</button></div><div id="docs"></div><div class="modalfoot"><div id="selCount" class="muted">All documents</div><button class="primary" onclick="applyScope()">Apply scope</button></div></div></div><div id="evidenceModal" class="modal" role="dialog" aria-modal="true" aria-labelledby="evidenceTitle"><div class="modalbox"><div class="modalhead"><div><b id="evidenceTitle">Evidence</b><div id="evidencePages" class="muted"></div></div><button class="mini" onclick="closeEvidence()">Close</button></div><div id="evidenceBody"></div></div></div><div id="toast" class="toast" role="status" aria-live="polite"></div>
<script>
const $=id=>document.getElementById(id);let docs=[],selected=new Set(JSON.parse(localStorage.getItem('rag_scope')||'[]')),lastGroups=[],isAsking=false,toastTimer=null;
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function showToast(message,kind='info'){const el=$("toast");if(!el)return;el.textContent=message;el.dataset.kind=kind;el.classList.add('visible');if(toastTimer)clearTimeout(toastTimer);toastTimer=setTimeout(()=>el.classList.remove('visible'),3600)}
async function loadDocs(){try{const r=await fetch('/api/documents');if(!r.ok)throw new Error('Document request failed');const d=await r.json();docs=d.documents||[];const valid=new Set(docs.map(x=>x.id));selected=new Set([...selected].filter(id=>valid.has(id)));if(!selected.size&&docs.length)docs.forEach(x=>selected.add(x.id));localStorage.setItem('rag_scope',JSON.stringify([...selected]));renderDocs();updateScope();updateEmptyState()}catch(e){showToast('Could not load the document list. Check that the API is running.','error')}}
function updateEmptyState(){const hasDocs=docs.length>0;$("emptyKicker").textContent=hasDocs?'READY TO ASK':'GET STARTED';$("emptyTitle").textContent=hasDocs?'Your documents are ready':'Start with a PDF';$("emptyDescription").textContent=hasDocs?'Ask a specific question, then inspect the source passages to verify the answer.':'Upload a PDF to index its contents, then ask a question about what it says.';$("promptExamples").style.display=hasDocs?'flex':'none'}
function setPrompt(text){$("q").value=text;$("q").focus();$("q").dispatchEvent(new Event('input'))}
async function runtime(){try{const r=await fetch('/api/runtime');const d=await r.json();$("provider").textContent=`Gemini · ${d.default_model} · streaming`}catch{$("provider").textContent='Gemini · unavailable'}}
function renderDocs(){const box=$("docs");box.innerHTML=docs.length?docs.map(d=>`<label class="docrow"><input type="checkbox" data-id="${esc(d.id)}" ${selected.has(d.id)?'checked':''} onchange="syncCount()"><div class="info"><b>${esc(d.filename)}</b><div class="muted">${d.pages} pages · ${d.chunks} chunks</div></div></label>`).join(''):`<div style="padding:25px" class="muted">No PDFs indexed yet. Add a PDF to begin.</div>`;syncCount()}
function syncCount(){const checked=[...document.querySelectorAll('#docs input:checked')];$("selCount").textContent=checked.length?`${checked.length} document${checked.length>1?'s':''} selected`:'No documents selected'}
function selectAll(){document.querySelectorAll('#docs input').forEach(x=>x.checked=true);syncCount()}
function clearSelect(){document.querySelectorAll('#docs input').forEach(x=>x.checked=false);syncCount()}
function applyScope(){selected=new Set([...document.querySelectorAll('#docs input:checked')].map(x=>x.dataset.id));localStorage.setItem('rag_scope',JSON.stringify([...selected]));updateScope();closeScope()}
function updateScope(){const n=selected.size;const all=!n||n===docs.length;$("scopeBtn").textContent=all?'All documents':`${n} selected`;$("scopeText").textContent=!docs.length?'No documents indexed':all?'Searching all indexed documents':`Searching ${n} selected document${n===1?'':'s'}`}
function openScope(){renderDocs();$("modal").style.display='flex'}function closeScope(){$("modal").style.display='none'}
function showEvidence(i){const g=lastGroups[i];if(!g)return;$("evidenceTitle").textContent=`Source ${g.source_id} · ${g.filename}`;$("evidencePages").textContent=`Pages ${g.pages} · ${g.chunks} retrieved chunk${g.chunks>1?'s':''}`;$("evidenceBody").innerHTML=g.evidence.map((e,j)=>`<div class="evidence"><div class="evidencehead"><b>Evidence passage ${j+1}</b><span class="muted">Pages ${esc(e.pages)}</span></div><pre>${esc(e.text)}</pre></div>`).join('');$("evidenceModal").style.display='flex'}
function closeEvidence(){$("evidenceModal").style.display='none'}
function renderSources(groups){lastGroups=groups||[];$("sources").innerHTML=lastGroups.map((g,i)=>`<button class="source" onclick="showEvidence(${i})"><b>📄 Source ${g.source_id} · ${esc(g.filename)}</b><div class="pages">Pages ${esc(g.pages)} · ${g.chunks} chunk${g.chunks>1?'s':''}</div><div class="view">View exact evidence →</div></button>`).join('')}
function setStage(e){const id=e.id;let el=document.querySelector(`[data-stage="${id}"]`);if(!el){el=document.createElement('div');el.className='stage';el.dataset.stage=id;$("stages").appendChild(el)}el.className='stage '+(e.done?'done':e.active?'active':'');el.innerHTML=`<div class="dot"></div><div><b>${esc(e.label)}</b><div class="muted">${esc(e.detail||'')}</div></div>`;$("loadTitle").textContent=e.label;$("loadDetail").textContent=e.detail||'';$("progressStatus").style.display='flex';$("progressText").textContent=e.label+(e.detail?' · '+e.detail:'')}
async function ask(){
 const question=$("q").value.trim();
 if(!question||isAsking)return;
 if(!docs.length){showToast('Add a PDF before asking a question.','error');return}
 isAsking=true;
 const send=$("sendBtn");send.disabled=true;
 const ids=[...selected];
 $("overlay").style.display='flex';$("overlay").setAttribute('aria-busy','true');
 $("stages").innerHTML='';$("emptyState").style.display='none';
 $("answer").style.display='none';$("answer").textContent='';
 $("sources").innerHTML='';$("metrics").style.display='none';
 $("progressStatus").style.display='flex';$("progressText").textContent='Searching your documents…';
 $("provider").textContent='Preparing answer…';
 const body={question,document_ids:ids,top_k:3,model:null};
 try{
  const res=await fetch('/api/ask/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!res.ok)throw new Error(await res.text());
  const reader=res.body.getReader(),dec=new TextDecoder();let buf='';
  while(true){
   const {value,done}=await reader.read();if(done)break;
   buf+=dec.decode(value,{stream:true});const blocks=buf.split('\n\n');buf=blocks.pop()||'';
   for(const block of blocks){
    const lines=block.split('\n');const type=(lines.find(x=>x.startsWith('event:'))||'').slice(6).trim();
    const raw=(lines.find(x=>x.startsWith('data:'))||'').slice(5).trim();if(!raw)continue;
    const data=JSON.parse(raw);
    if(type==='stage'){
     setStage(data);
     if(data.id==='generation')$("provider").textContent=data.label+(data.detail?' · '+data.detail:'');
    }
    if(type==='token'){
     $("overlay").style.display='none';$("answer").style.display='block';$("answer").textContent+=data.text;
     $("progressText").textContent='Receiving answer from Gemini…';
    }
    if(type==='generation_reset'){
     $("answer").textContent='';$("answer").style.display='none';$("overlay").style.display='flex';
     $("provider").textContent='Trying '+(data.model||'alternate model')+'…';
     $("progressText").textContent='Switching to '+(data.model||'another model')+'…';
    }
    if(type==='result'){
     renderResult(data);
     const failed=data.mode==='gemini_error';
     $("provider").textContent=failed?'Generation unavailable':'Ready · '+(data.model||'Gemini');
     $("progressText").textContent=failed?'Generation failed · review the message above':'Answer ready · sources below';
     if(failed)showToast('Gemini could not complete the answer. Check the model/service error above.','error');
    }
    if(type==='error')throw new Error(data.message);
   }
  }
 }catch(e){
  $("overlay").style.display='none';$("answer").style.display='block';
  $("answer").textContent='Request failed: '+e.message;$("provider").textContent='Request failed';
  $("progressStatus").style.display='flex';$("progressText").textContent='Request failed';
  showToast('The request failed. See the answer area for details.','error');
 }finally{
  $("overlay").style.display='none';$("overlay").setAttribute('aria-busy','false');
  send.disabled=false;isAsking=false;
 }
}
function renderResult(d){$("emptyState").style.display='none';$("answer").style.display='block';$("answer").textContent=d.answer;renderSources(d.evidence_groups||[]);const l=d.latency||{};$("metrics").style.display='grid';$("mR").textContent=(l.retrieval_ms??0).toFixed(0)+' ms';$("mK").textContent=(l.rerank_ms??0).toFixed(1)+' ms';$("mG").textContent=(l.generation_ms??0).toFixed(0)+' ms';$("mF").textContent=(l.ttft_ms??0).toFixed(0)+' ms';$("mT").textContent=(l.total_ms??0).toFixed(0)+' ms';$("progressStatus").style.display='flex'}
function clearChat(){$("answer").style.display='none';$("answer").textContent='';$("sources").innerHTML='';$("metrics").style.display='none';$("progressStatus").style.display='none';$("stages").innerHTML='';$("emptyState").style.display='flex';lastGroups=[];updateEmptyState()}
$("q").addEventListener('input',()=>{const el=$("q");el.style.height='auto';el.style.height=Math.min(el.scrollHeight,180)+'px'});
$("q").addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();ask()}});
$("addPdfButton").addEventListener('click',()=> $("file").click());
$("file").addEventListener('change',async e=>{
 const f=e.target.files[0];if(!f)return;
 const button=$("addPdfButton"),previous=button.innerHTML,fd=new FormData();fd.append('file',f);
 button.disabled=true;button.innerHTML='<span class="button-icon">…</span><span>Adding PDF</span>';
 $("provider").textContent='Uploading and indexing PDF…';
 try{
  const r=await fetch('/api/documents/upload',{method:'POST',body:fd});
  if(!r.ok)throw new Error(await r.text());
  await loadDocs();showToast('Indexed '+f.name+' successfully.','success');
  $("provider").textContent='PDF indexed · Gemini ready';
 }catch(err){
  showToast('Upload failed: '+err.message,'error');$("provider").textContent='Upload failed';
 }finally{
  button.disabled=false;button.innerHTML=previous;e.target.value='';
 }
});
loadDocs();runtime();
</script></body></html>'''


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(HTML)
