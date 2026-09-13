from main import ABSTAIN, answerable, extractive


def result(text: str):
    return [{
        "document_id": "paper",
        "filename": "paper.pdf",
        "page_start": 5,
        "page_end": 6,
        "score": 0.9,
        "text": text,
    }]


def test_qis_question_prefers_definition_over_table_noise():
    results = result(
        "Scoring Method Computation Time Memory Shannon Entropy 0.0517 0.02. "
        "We presented QIS, a quantum information-theoretic framework for Transformer attention-head pruning. "
        "By combining Von Neumann Entropy, Quantum Jensen-Shannon Divergence, and Quantum Fidelity, QIS captures spectral complexity."
    )
    answer = extractive("What is QIS?", results)
    assert "quantum information-theoretic framework" in answer.lower()
    assert "computation time" not in answer.lower()


def test_unanswerable_question_abstains():
    results = result("QIS combines Von Neumann Entropy, QJSD, and Quantum Fidelity.")
    question = "What email addresses are used?"
    assert not answerable(question, results)
    assert extractive(question, results) == ABSTAIN


def test_qis_combination_is_concise():
    results = result(
        "QIS combines Von Neumann Entropy, Quantum Jensen-Shannon Divergence, and Quantum Fidelity. "
        "Table V reports scoring overhead and model inference speedup."
    )
    answer = extractive("What does QIS combine?", results)
    assert "Von Neumann Entropy" in answer
    assert "Quantum Fidelity" in answer
    assert len(answer) < 500
