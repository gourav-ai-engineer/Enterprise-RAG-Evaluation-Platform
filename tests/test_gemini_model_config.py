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
            assert 'else "low"' in source
        else:
            assert '"thinkingLevel": "low"' in source


def test_frontend_handles_partial_answer_events() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "gemini_main.py").read_text(encoding="utf-8")
    assert 'sse("token", {"text": piece})' in source
    assert "if(type==='token')" in source
    assert "generation_reset" in source


def test_default_model_order_prefers_low_latency() -> None:
    assert gemini_main.GEMINI_MODELS[0] == "gemini-3.5-flash-lite"



def test_answerable_questions_do_not_use_local_extractive_generation() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "gemini_main.py").read_text(encoding="utf-8")
    assert "def fast_extractive_answer" not in source
    assert "extractive_fast" not in source
    assert "gemini_stream_generate_with_fallback(" in source



def test_generation_timeouts_are_explicitly_bounded() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "gemini_main.py").read_text(encoding="utf-8")
    assert 'GEMINI_GENERATION_BUDGET_SECONDS", "40"' in source
    assert 'GEMINI_REQUEST_TIMEOUT_SECONDS", "28"' in source
    assert 'GEMINI_FIRST_TOKEN_TIMEOUT_SECONDS", "12"' in source
    assert 'GEMINI_FALLBACK_REQUEST_TIMEOUT_SECONDS", "15"' in source
    assert 'GEMINI_FALLBACK_FIRST_TOKEN_TIMEOUT_SECONDS", "6"' in source



def test_primary_and_fallback_have_different_latency_windows() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "gemini_main.py").read_text(encoding="utf-8")
    assert "model_first_token_timeout = first_token_timeout if model_index == 0 else fallback_first_token_timeout" in source
    assert "model_request_timeout = request_timeout if model_index == 0 else fallback_request_timeout" in source
