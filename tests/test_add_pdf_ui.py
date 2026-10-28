from pathlib import Path


def test_add_pdf_button_has_a_valid_click_listener() -> None:
    source_path = Path(__file__).resolve().parents[1] / "gemini_main.py"
    source = source_path.read_text(encoding="utf-8")

    assert '<button id="addPdfButton" class="primary" type="button">+ Add PDF</button>' in source
    assert '$("addPdfButton").addEventListener(\'click\',()=>{$("file").click()})' in source
    assert r"document.getElementById(\'file\').click()" not in source
