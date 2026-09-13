from app import ABSTAIN_MESSAGE, extractive_answer, has_sufficient_evidence


def _result(text: str) -> dict:
    return {
        "chunk_id": "1",
        "document_id": "doc",
        "filename": "paper.pdf",
        "page_start": 1,
        "page_end": 1,
        "score": 1.0,
        "bm25": 1.0,
        "tfidf": 1.0,
        "text": text,
    }


def test_unanswerable_question_abstains() -> None:
    results = [_result("Quantum Information Score combines Von Neumann Entropy and Quantum Fidelity.")]
    assert has_sufficient_evidence("what email addresses are used", results) is False
    assert extractive_answer("what email addresses are used", results) == ABSTAIN_MESSAGE


def test_answerable_question_returns_evidence() -> None:
    results = [_result(
        "The proposed QIS combines Von Neumann Entropy, QJSD, and Quantum Fidelity for attention-head pruning."
    )]
    assert has_sufficient_evidence("what does QIS combine", results) is True
    answer = extractive_answer("what does QIS combine", results)
    assert "QIS" in answer
    assert "Quantum Fidelity" in answer
