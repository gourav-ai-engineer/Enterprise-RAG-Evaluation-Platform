import gemini_main


def test_simple_who_question_uses_local_evidence_and_citation() -> None:
    results = [
        {
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
            "filename": "rag.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "Hybrid retrieval combines BM25 and TF-IDF to retrieve relevant passages.",
        }
    ]

    assert gemini_main.fast_extractive_answer(
        "Explain how hybrid retrieval works", results
    ) is None


def test_fast_answer_falls_back_if_sentence_cannot_be_cited() -> None:
    results = [
        {
            "filename": "profile.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "Gourav is an AI/ML engineer.",
        }
    ]
    # The helper may only return sentences that can be traced back to retrieved text.
    answer = gemini_main.fast_extractive_answer("Who is Gourav?", results)
    assert answer == "Gourav is an AI/ML engineer. [Source 1]"
