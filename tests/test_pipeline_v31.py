from main import answerable, coverage, intent, retrieve


def rows(items):
    return [dict(id=f"c{i}", document_id=doc, filename=f"{doc}.pdf", page_start=1, page_end=1, text=text) for i,(doc,text) in enumerate(items)]


def test_global_retrieval_searches_multiple_documents():
    data = rows([
        ("transformer", "The Transformer is based solely on attention mechanisms."),
        ("bert", "BERT is designed for language representation using bidirectional pre-training."),
        ("lora", "LoRA reduces trainable parameters through low-rank adaptation."),
    ])
    results, _ = retrieve("What is BERT?", data, 3)
    assert results[0]["document_id"] == "bert"


def test_reranker_prefers_definition_sentence():
    data = rows([
        ("a", "Table 4 memory 512 1024 2048. Computation time was measured."),
        ("b", "BERT is a method for pre-training bidirectional representations of language."),
    ])
    results, _ = retrieve("What is BERT?", data, 2)
    assert results[0]["document_id"] == "b"


def test_unanswerable_question_is_not_supported():
    data = rows([("qis", "Quantum Information Score combines entropy, QJSD, and fidelity.")])
    results, _ = retrieve("What is Gourav's contribution to the Transformer paper?", data, 3)
    assert not answerable("What is Gourav's contribution to the Transformer paper?", results)
    assert coverage("What is Gourav's contribution to the Transformer paper?", results) < 0.30


def test_query_intent_classification():
    assert intent("What is BERT?") == "definition"
    assert intent("How is Quantum Fidelity calculated?") == "method"
    assert intent("Why does RAG help?") == "purpose"
    assert intent("How does BERT differ from the Transformer?") == "comparison"
