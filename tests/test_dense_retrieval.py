from __future__ import annotations

import numpy as np

import dense_retrieval
import main as core

_REAL_DENSE_SIMILARITY_SCORES = core.dense_similarity_scores


def _row(chunk_id: str, text: str, document_id: str | None = None) -> dict:
    return {
        "id": chunk_id,
        "document_id": document_id or chunk_id,
        "filename": f"{document_id or chunk_id}.pdf",
        "page_start": 1,
        "page_end": 1,
        "text": text,
    }


def test_dense_semantic_rank_can_retrieve_a_paraphrase_without_keyword_overlap(
    monkeypatch,
) -> None:
    query = "What is a practical way to commute?"
    rows = [
        _row("technical-paper", "Transformer self-attention is used to prune redundant head weights."),
        _row("transport-note", "People travel between cities in automobiles."),
        _row("physics-note", "Quantum circuits encode computational states."),
    ]
    # This controlled score fixture simulates a dense model matching a paraphrase
    # that has no important query words in common with the indexed passage.
    monkeypatch.setattr(
        core,
        "dense_similarity_scores",
        lambda query, rows: [0.12, 0.91, 0.08],
    )

    results = core.hybrid_retrieve(query, rows, k=3)

    assert results
    assert results[0]["chunk_id"] == "transport-note"
    assert results[0]["dense_rank"] == 1
    assert results[0]["dense"] == 0.91
    assert results[0]["rrf_score"] > 0


def test_weighted_rrf_keeps_exact_wording_and_semantic_ranks_in_result_metadata(
    monkeypatch,
) -> None:
    query = "What are the project qualifications?"
    rows = [
        _row("exact", "Project qualifications include a degree and GATE qualification."),
        _row("semantic", "Academic credentials include a postgraduate engineering degree."),
        _row("irrelevant", "A recipe uses flour, water, and salt."),
    ]
    monkeypatch.setattr(
        core,
        "dense_similarity_scores",
        lambda query, rows: [0.45, 0.94, 0.05],
    )

    results = core.hybrid_retrieve(query, rows, k=3)
    by_id = {item["chunk_id"]: item for item in results}

    assert "exact" in by_id
    assert "semantic" in by_id
    assert by_id["semantic"]["dense_rank"] == 1
    assert by_id["exact"]["bm25_rank"] == 1
    assert by_id["exact"]["tfidf_rank"] == 1
    assert all(0 < item["score"] <= 1 for item in results)


def test_dense_vectors_are_cached_by_chunk_and_model_in_sqlite(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(core, "DB_PATH", tmp_path / "dense-test.sqlite3")
    monkeypatch.setenv("DENSE_EMBEDDING_MODEL", "test/deterministic-model")
    monkeypatch.setattr(
        dense_retrieval,
        "embed_passages",
        lambda texts, name=None: [
            np.asarray([1.0, 0.0], dtype=np.float32)
            if "alpha" in text
            else np.asarray([0.0, 1.0], dtype=np.float32)
            for text in texts
        ],
    )
    monkeypatch.setattr(
        dense_retrieval,
        "embed_query",
        lambda query, name=None: np.asarray([1.0, 0.0], dtype=np.float32),
    )
    monkeypatch.setattr(core, "dense_similarity_scores", _REAL_DENSE_SIMILARITY_SCORES)

    rows = [
        _row("dense-test-alpha", "alpha passage"),
        _row("dense-test-beta", "beta passage"),
    ]
    indexed = core.index_dense_embeddings(rows)
    scores = core.dense_similarity_scores("query", rows)

    conn = core.db()
    stored = conn.execute(
        "SELECT COUNT(*) FROM chunk_embeddings WHERE model_name=?",
        ("test/deterministic-model",),
    ).fetchone()[0]
    conn.close()

    assert indexed == 2
    assert stored == 2
    assert scores[0] == 1.0
    assert scores[1] == 0.0


def test_dense_cosine_helper_handles_zero_vectors() -> None:
    scores = dense_retrieval.cosine_similarity_scores(
        np.asarray([1.0, 0.0], dtype=np.float32),
        [
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 1.0], dtype=np.float32),
            np.asarray([0.0, 0.0], dtype=np.float32),
        ],
    )

    assert np.allclose(scores, [1.0, 0.0, 0.0])


def test_dense_relevance_can_pass_evidence_gate_for_a_paraphrase() -> None:
    result = [{
        "chunk_id": "semantic-1",
        "document_id": "transport",
        "filename": "transport.pdf",
        "page_start": 1,
        "page_end": 1,
        "text": "Automobiles provide road-based personal mobility for commuters.",
        "score": 1.0,
        "dense": 0.91,
        "dense_rank": 1,
    }]

    assert not (set(core.content_tokens("What is a practical way to commute?"))
                & set(core.content_tokens(result[0]["text"])))
    assert core.has_sufficient_evidence("What is a practical way to commute?", result)


def test_dense_relevance_alone_does_not_support_a_name_only_query() -> None:
    result = [{
        "chunk_id": "footer-1",
        "document_id": "seminar",
        "filename": "QFC_Seminar.pdf",
        "page_start": 2,
        "page_end": 2,
        "text": "An unrelated presentation discusses a mathematical method.",
        "score": 1.0,
        "dense": 0.99,
        "dense_rank": 1,
    }]

    assert not core.has_sufficient_evidence("Gourav", result)
