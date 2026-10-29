import asyncio
import json

import gemini_main
import main as core


def make_chunk(chunk_id: str, document_id: str, filename: str, page: int, text: str) -> dict:
    return {
        "id": chunk_id,
        "document_id": document_id,
        "filename": filename,
        "page_start": page,
        "page_end": page,
        "text": text,
    }


def test_summary_intent_detects_broad_multi_document_requests() -> None:
    assert gemini_main.is_summary_request("summarize both pdf")
    assert gemini_main.is_summary_request("Give me an overview of these documents")
    assert not gemini_main.is_summary_request("What are Gourav's achievements?")


def test_summary_evidence_samples_pages_from_every_selected_document() -> None:
    rows = [
        make_chunk(f"seminar-{page}", "seminar", "QFC_Seminar.pdf", page, f"Seminar key point {page}.")
        for page in range(1, 15)
    ]
    rows.extend([
        make_chunk("resume-1", "resume", "Gourav_Resume.pdf", 1, "Education and experience."),
        make_chunk("resume-2", "resume", "Gourav_Resume.pdf", 2, "Projects and achievements."),
    ])

    evidence = gemini_main.select_summary_evidence(rows)

    by_doc: dict[str, list[dict]] = {}
    for item in evidence:
        by_doc.setdefault(item["document_id"], []).append(item)

    assert set(by_doc) == {"seminar", "resume"}
    assert len(by_doc["seminar"]) == 14
    assert len(by_doc["resume"]) == 2
    assert min(item["page_start"] for item in by_doc["seminar"]) == 1
    assert max(item["page_end"] for item in by_doc["seminar"]) == 14


def test_summary_prompt_instructs_model_to_summarize_each_source() -> None:
    results = [
        {
            "chunk_id": "a-1",
            "document_id": "doc-a",
            "filename": "first.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "The first document explains research motivation.",
            "score": 1.0,
        },
        {
            "chunk_id": "b-1",
            "document_id": "doc-b",
            "filename": "second.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "The second document lists project achievements.",
            "score": 1.0,
        },
    ]

    system, user = gemini_main.build_prompt("summarize both pdf", results)

    assert "Summarize each supplied source separately" in system
    assert "first.pdf" in user
    assert "second.pdf" in user


def test_broad_summary_reaches_gemini_without_keyword_overlap_abstention(monkeypatch) -> None:
    question = "summarize both pdf"
    rows = [
        make_chunk(
            "seminar-1",
            "seminar",
            "QFC_Seminar.pdf",
            1,
            "The seminar studies transformer attention heads and proposes a fidelity-based pruning framework.",
        ),
        make_chunk(
            "resume-1",
            "resume",
            "Gourav_Resume.pdf",
            1,
            "Gourav Kumar is an AI/ML engineer with computer vision and deep learning project experience.",
        ),
    ]
    loaded_scopes: list[list[str] | None] = []
    generated: list[list[dict]] = []

    def fake_load_chunks(document_ids=None, document_id=None):
        loaded_scopes.append(document_ids)
        return rows

    async def fake_generation(model, q, results, on_text=None, on_reset=None):
        generated.append(results)
        answer = (
            "QFC_Seminar.pdf: The seminar presents a fidelity-based framework for selecting "
            "Transformer attention heads [Source 1]. "
            "Gourav_Resume.pdf: The resume highlights AI/ML engineering, computer vision, "
            "and deep learning projects [Source 2]."
        )
        if on_text:
            await on_text(answer)
        return answer, 5.0, 15.0, 2, "test-model"

    monkeypatch.setattr(core, "load_chunks", fake_load_chunks)
    monkeypatch.setattr(gemini_main, "gemini_stream_generate_with_fallback", fake_generation)

    async def call_endpoint():
        response = await gemini_main.ask_stream(gemini_main.StreamAskRequest(
            question=question,
            document_ids=["seminar", "resume"],
            top_k=3,
        ))
        chunks = []
        async for part in response.body_iterator:
            chunks.append(part.decode() if isinstance(part, bytes) else part)
        return "".join(chunks)

    stream = asyncio.run(call_endpoint())
    result_block = next(block for block in stream.split("\n\n") if block.startswith("event: result"))
    result = json.loads(result_block.split("data: ", 1)[1])

    assert loaded_scopes == [["seminar", "resume"]]
    assert len(generated) == 1
    assert result["answerable"] is True
    assert result["mode"] == "gemini"
    assert {group["document_id"] for group in result["evidence_groups"]} == {"seminar", "resume"}
