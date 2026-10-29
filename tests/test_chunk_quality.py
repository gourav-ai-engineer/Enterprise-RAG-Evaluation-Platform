from __future__ import annotations

import enterprise_main
import main as core


def test_chunks_never_cross_page_boundaries() -> None:
    pages = [
        "PAGEONE " + "alpha " * 200,
        "PAGETWO " + "beta " * 200,
    ]

    chunks = core.chunk_pages(pages)

    assert chunks
    assert all(start_page == end_page for start_page, end_page, _ in chunks)
    assert all(not ("PAGEONE" in text and "PAGETWO" in text) for _, _, text in chunks)


def test_repeated_slides_footer_is_removed_before_chunking() -> None:
    pages = [
        f"Unique body content for slide {page_no}.\nGourav — 2522105 — NIT Silchar {page_no} / 14"
        for page_no in range(1, 15)
    ]

    chunks = core.chunk_pages(pages)

    assert chunks
    assert all("2522105" not in text for _, _, text in chunks)
    assert all("NIT Silchar" not in text for _, _, text in chunks)


def test_pdf_page_cleaner_preserves_lines_for_boilerplate_detection() -> None:
    text = "Page heading\nBody line one\nBody line two"

    cleaned = core.clean_page_text(text)

    assert "\n" in cleaned
    assert "Page heading" in cleaned
    assert "Body line one" in cleaned


def test_query_correction_fixes_single_character_typo_from_corpus() -> None:
    normalized = core.correct_query_terms(
        "achivements of Gourav",
        ["Achievements and certifications. Gourav qualified GATE in 2024 and 2025."],
    )

    assert "achievements" in normalized.casefold()
    assert "gourav" in normalized.casefold()


def test_repeated_name_alone_is_not_enough_for_achievements_query() -> None:
    footer_only = [{
        "text": "Gourav — 2522105 — NIT Silchar",
        "score": 0.9,
        "chunk_id": "footer-1",
        "document_id": "seminar",
        "filename": "QFC_Seminar.pdf",
        "page_start": 4,
        "page_end": 8,
    }]

    assert not core.has_sufficient_evidence("achivements of Gourav", footer_only)


def test_bulk_reprocess_endpoint_rebuilds_all_indexed_documents(monkeypatch) -> None:
    class FakeCursor:
        def fetchall(self):
            return [{"id": "doc-a"}, {"id": "doc-b"}]

    class FakeConnection:
        def execute(self, query):
            assert "SELECT id FROM documents" in query
            return FakeCursor()

        def close(self):
            pass

    processed = []
    monkeypatch.setattr(enterprise_main.core, "db", lambda: FakeConnection())

    def fake_reprocess(document_id: str):
        processed.append(document_id)
        return {"document_id": document_id, "chunks": 3, "version": 2}

    monkeypatch.setattr(enterprise_main, "reprocess", fake_reprocess)

    result = enterprise_main.reprocess_all()

    assert processed == ["doc-a", "doc-b"]
    assert result["total"] == 2
    assert result["reprocessed"] == 2
    assert result["failed"] == 0



def test_misspelled_topic_selects_achievement_chunk_not_name_footer() -> None:
    question = "achivements of Gourav"
    rows = [
        {
            "id": "resume-name",
            "document_id": "resume",
            "filename": "Gourav_M.tech_CSE(AI).pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "Gourav Kumar is an M.Tech candidate at NIT Silchar.",
        },
        {
            "id": "resume-achievements",
            "document_id": "resume",
            "filename": "Gourav_M.tech_CSE(AI).pdf",
            "page_start": 2,
            "page_end": 2,
            "text": "Achievements and Certifications. GATE qualified in 2024 and 2025. Solved 200 plus LeetCode problems.",
        },
        {
            "id": "seminar-footer",
            "document_id": "seminar",
            "filename": "QFC_Seminar.pdf",
            "page_start": 4,
            "page_end": 4,
            "text": "Gourav — 2522105 — NIT Silchar",
        },
    ]
    normalized = core.correct_query_terms(question, [row["text"] for row in rows])

    candidates = core.hybrid_retrieve(normalized, rows, 10, diversify_documents=True)

    assert "achievements" in normalized.casefold()
    assert candidates
    assert {item["document_id"] for item in candidates} == {"resume"}
    assert candidates[0]["chunk_id"] == "resume-achievements"
