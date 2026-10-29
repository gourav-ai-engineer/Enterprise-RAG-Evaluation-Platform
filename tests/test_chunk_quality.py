from __future__ import annotations

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
