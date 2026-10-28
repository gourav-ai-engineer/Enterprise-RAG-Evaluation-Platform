import gemini_main


def test_simple_who_question_uses_local_evidence_and_citation() -> None:
    results = [
        {
            "chunk_id": "chunk-profile-1",
            "document_id": "doc-profile",
            "chunk_id": "chunk-profile-2",
            "document_id": "doc-profile",
            "filename": "profile.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": (
                "Gourav Kumar is an AI/ML engineer and M.Tech candidate at NIT Silchar. "
                "He works on retrieval-augmented generation systems."
            ),
        }
    ]

    answer = gemini_main.fast_extractive_answer("Who is Gourav?", results)

    assert answer == "Gourav Kumar is an AI/ML engineer and M.Tech candidate at NIT Silchar. [Source 1]"


def test_explanatory_question_uses_gemini_path() -> None:
    results = [
        {
            "chunk_id": "chunk-rag-1",
            "document_id": "doc-rag",
            "filename": "rag.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "Hybrid retrieval combines BM25 and TF-IDF to retrieve relevant passages.",
        }
    ]

    assert gemini_main.fast_extractive_answer(
        "Explain how hybrid retrieval works", results
    ) is None


def test_fast_answer_declines_when_question_has_no_evidence() -> None:
    results = [
        {
            "filename": "profile.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "Gourav is an AI/ML engineer.",
        }
    ]
    # A question unsupported by the evidence must remain on the model/abstention path.
    assert gemini_main.fast_extractive_answer("Who is Maria?", results) is None


def test_comparisons_are_not_answered_extractive() -> None:
    results = [
        {
            "filename": "rag.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "BM25 is a lexical retrieval algorithm. TF-IDF uses term frequency and inverse document frequency.",
        }
    ]
    assert gemini_main.fast_extractive_answer(
        "Which is better, BM25 or TF-IDF?", results
    ) is None
