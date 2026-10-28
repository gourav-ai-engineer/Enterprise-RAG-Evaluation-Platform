from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "gemini_main.py"


def test_workbench_has_clear_responsive_empty_state_and_progress() -> None:
    source = SOURCE.read_text(encoding="utf-8")

    assert "RAG Studio" in source
    assert 'id="emptyState"' in source
    assert 'id="promptExamples"' in source
    assert 'id="progressStatus"' in source
    assert "aria-live=\"polite\"" in source
    assert "@media(max-width:620px)" in source
    assert "prefers-reduced-motion:reduce" in source


def test_upload_uses_inline_feedback_instead_of_browser_alert() -> None:
    source = SOURCE.read_text(encoding="utf-8")

    assert "function showToast(message" in source
    assert "Indexed '+f.name+' successfully." in source
    assert "showToast('Upload failed: '+err.message,'error')" in source
    assert "alert('Upload failed" not in source


def test_dialogs_and_question_controls_have_accessible_labels() -> None:
    source = SOURCE.read_text(encoding="utf-8")

    assert 'aria-modal="true" aria-labelledby="scopeTitle"' in source
    assert 'aria-modal="true" aria-labelledby="evidenceTitle"' in source
    assert 'aria-label="Ask a question about selected documents"' in source
    assert 'aria-label="Send question"' in source


def test_embedded_workbench_javascript_parses() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is not installed; skipping inline JavaScript syntax check")

    source = SOURCE.read_text(encoding="utf-8")
    match = re.search(r"<script>(.*?)</script>", source, re.DOTALL)
    assert match, "Expected the main workbench script"

    with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8", delete=False) as handle:
        handle.write(match.group(1))
        script_path = handle.name

    try:
        result = subprocess.run(
            [node, "--check", script_path],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr or result.stdout
    finally:
        Path(script_path).unlink(missing_ok=True)
