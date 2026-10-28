from pathlib import Path

import gemini_main


def test_default_model_is_current_supported_flash(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert gemini_main.resolve_gemini_model() == "gemini-3.5-flash-lite"


def test_retired_model_environment_setting_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    assert gemini_main.resolve_gemini_model() == "gemini-3.5-flash-lite"


def test_available_model_choices_exclude_gemini_25() -> None:
    assert "gemini-3.8-flash" in gemini_main.GEMINI_MODELS
    assert all(not model.startswith("gemini-2.5-") for model in gemini_main.GEMINI_MODELS)


def test_generation_uses_supported_thinking_level() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("gemini_main.py", "gemini_quality.py"):
        source = (root / name).read_text(encoding="utf-8")
        if name == "gemini_main.py":
            assert '"thinkingLevel": "minimal" if model in {' in source
            assert '"thinkingLevel": "low"' in source
        else:
            assert '"thinkingLevel": "low"' in source
