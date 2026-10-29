import asyncio
import json

import gemini_main
import main as core


def make_chunk(chunk_id: str, document_id: str, filename: str, text: str) -> dict:
    return {
        "id": chunk_id,
        "document_id": document_id,
        "filename": filename,
        "page_start": 1,
        "page_end": 1,
        "text": text,
    }


def test_hybrid_candidates_preserve_relevant_documents() -> None:
    question = "retrieval augmented generation evidence grounding"
    rows = [
        make_chunk(
            f"a-{index:02d}",
            "doc-a",
            "dense-topic.pdf",
            "Retrieval augmented generation retrieves evidence from relevant documents "
            "before a language model generates a grounded answer.",
        )
        for index in range(30)
    ]
    rows.append(
        make_chunk(
            "b-01",
            "doc-b",
            "other-topic.pdf",
            "Evidence grounding combines retrieval augmented generation with cited "
            "source passages from multiple documents.",
        )
    )

    candidates = core.hybrid_retrieve(question, rows, 10, diversify_documents=True)

    assert len(candidates) == 10
    assert candidates[0]["document_id"] == "doc-a"
    assert candidates[1]["document_id"] == "doc-b"


def test_reranking_reserves_a_relevant_passage_per_document() -> None:
    question = "retrieval augmented generation evidence grounding"
    ranked = [
        {
            "chunk_id": f"a-{index}",
            "document_id": "doc-a",
            "filename": "dense-topic.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "Retrieval augmented generation retrieves evidence from relevant documents.",
            "score": 0.9 - index * 0.01,
            "rerank_score": 12 - index,
        }
        for index in range(5)
    ]
    ranked.append({
        "chunk_id": "b-01",
        "document_id": "doc-b",
        "filename": "other-topic.pdf",
        "page_start": 1,
        "page_end": 1,
        "text": "Evidence grounding combines retrieval augmented generation with cited source passages.",
        "score": 0.4,
        "rerank_score": 7.0,
    })

    results = gemini_main.diversify_reranked_results(question, ranked, top_k=3)

    assert len(results) == 3
    assert {item["document_id"] for item in results} == {"doc-a", "doc-b"}


def test_evidence_groups_keep_same_named_documents_separate() -> None:
    results = [
        {
            "chunk_id": "a-01",
            "document_id": "doc-a",
            "filename": "notes.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "First document evidence.",
        },
        {
            "chunk_id": "b-01",
            "document_id": "doc-b",
            "filename": "notes.pdf",
            "page_start": 2,
            "page_end": 2,
            "text": "Second document evidence.",
        },
    ]

    groups = gemini_main.evidence_groups(results)

    assert len(groups) == 2
    assert [group["document_id"] for group in groups] == ["doc-a", "doc-b"]


def test_multi_document_ask_uses_all_selected_ids_and_returns_both_sources(monkeypatch) -> None:
    question = "What is retrieval augmented generation evidence grounding?"
    rows = [
        make_chunk(
            "a-01",
            "doc-a",
            "retrieval.pdf",
            "Retrieval augmented generation retrieves evidence from relevant documents "
            "before a language model generates a grounded answer.",
        ),
        make_chunk(
            "b-01",
            "doc-b",
            "grounding.pdf",
            "Evidence grounding combines retrieval augmented generation with cited "
            "source passages from multiple documents.",
        ),
    ]
    loaded_scopes: list[list[str] | None] = []

    def fake_load_chunks(document_ids=None, document_id=None):
        loaded_scopes.append(document_ids)
        return rows

    async def fake_generation(model, q, results, on_text=None, on_reset=None):
        assert {item["document_id"] for item in results} == {"doc-a", "doc-b"}
        answer = "The selected sources describe retrieval augmented generation and evidence grounding [Source 1] [Source 2]."
        if on_text:
            await on_text(answer)
        return answer, 4.0, 9.0, 1, "test-model"

    monkeypatch.setattr(core, "load_chunks", fake_load_chunks)
    monkeypatch.setattr(gemini_main, "gemini_stream_generate_with_fallback", fake_generation)

    async def call_endpoint():
        response = await gemini_main.ask_stream(gemini_main.StreamAskRequest(
            question=question,
            document_ids=["doc-a", "doc-b"],
            top_k=3,
        ))
        blocks = []
        async for part in response.body_iterator:
            blocks.append(part.decode() if isinstance(part, bytes) else part)
        return "".join(blocks)

    stream = asyncio.run(call_endpoint())
    result_block = next(block for block in stream.split("\n\n") if block.startswith("event: result"))
    result = json.loads(result_block.split("data: ", 1)[1])

    assert loaded_scopes == [["doc-a", "doc-b"]]
    assert result["scope_document_ids"] == ["doc-a", "doc-b"]
    assert {group["document_id"] for group in result["evidence_groups"]} == {"doc-a", "doc-b"}
