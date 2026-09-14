from __future__ import annotations

import json
import re
import time

import httpx

import gemini_main as base

# Keep the existing production UI, retrieval, citations, and document scope.
app = base.app


def build_prompt(question: str, results: list[dict]) -> tuple[str, str]:
    groups = base.evidence_groups(results)
    context = "\n\n".join(
        f"[Source {group['source_id']}] {group['filename']} pages {group['pages']}\n"
        + "\n".join(item["text"] for item in group["evidence"])
        for group in groups
    )
    system = (
        "You are the final answer layer of a grounded enterprise RAG system. "
        "Answer ONLY from the supplied evidence. Never use outside knowledge. "
        "Return only the final answer; never output reasoning, analysis, planning, or process commentary. "
        "Do not say 'let me analyze', 'from the evidence', or describe your search. "
        "Answer the exact question directly in 1-3 COMPLETE sentences, normally under 90 words. "
        "Finish the final sentence before stopping. Do not end with an unfinished clause. "
        "Cite factual claims with [Source N], using only source numbers present below. "
        "For mathematical notation, use readable Unicode/plain text such as α, β, γ, ρᵢ and ρ̄. "
        "Never emit LaTeX delimiters ($, $$, \\(, \\), \\[ or \\]) or raw commands such as \\alpha. "
        "If an equation is needed, write it plainly, for example: QIS = αI(ρᵢ) + βQJSD(ρᵢ) − γF(ρᵢ, ρ̄). "
        "If the supplied evidence is insufficient, say that clearly instead of guessing."
    )
    return system, f"Question: {question}\n\nRetrieved evidence:\n{context}"


def clean_answer(answer: str) -> str:
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
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\\(?:text|mathrm)\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\bar\{([^{}]+)\}", r"\1̄", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text.strip()


async def improved_generate(
    model: str, question: str, results: list[dict]
) -> tuple[str, float, float, int]:
    key = base.gemini_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    system, user = build_prompt(question, results)
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            # 128 was too small for grounded answers containing equations.
            # Keep minimal thinking for latency while allowing a complete answer.
            "maxOutputTokens": 256,
            "thinkingConfig": {"thinkingLevel": "minimal"},
        },
    }

    started = time.perf_counter()
    ttft_ms = 0.0
    chunks = 0
    pieces: list[str] = []
    finish_reason = None
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            base.gemini_url(model),
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

    generation_ms = (time.perf_counter() - started) * 1000
    answer = clean_answer("".join(pieces))
    if not answer:
        raise RuntimeError("Gemini returned an empty answer")
    if finish_reason == "MAX_TOKENS":
        raise RuntimeError("Gemini reached the output limit; answer was incomplete")
    return answer, ttft_ms, generation_ms, chunks


# FastAPI's existing /api/ask/stream endpoint resolves this global from
# gemini_main at request time, so patching the module keeps the existing UI intact.
base.build_prompt = build_prompt
base.gemini_stream_generate = improved_generate
