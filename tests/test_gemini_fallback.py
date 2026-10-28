import asyncio

import pytest

import gemini_main


def test_gemini_overload_retries_then_falls_back(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_sleep(_seconds: float) -> None:
        return None

    async def fake_generate(model: str, question: str, results: list[dict]):
        calls.append(model)
        if model == "gemini-3.8-flash":
            raise RuntimeError("Gemini HTTP 503: temporarily overloaded")
        return "grounded answer", 12.0, 30.0, 2

    monkeypatch.setattr(gemini_main.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(gemini_main, "gemini_stream_generate", fake_generate)

    result = asyncio.run(
        gemini_main.gemini_stream_generate_with_fallback(
            "gemini-3.8-flash", "Who is Gourav?", [{"text": "Gourav"}]
        )
    )

    assert result == ("grounded answer", 12.0, 30.0, 2, "gemini-3.6-flash")
    assert calls == ["gemini-3.8-flash", "gemini-3.8-flash", "gemini-3.6-flash"]


def test_non_transient_gemini_error_does_not_retry(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_generate(model: str, question: str, results: list[dict]):
        calls.append(model)
        raise RuntimeError("Gemini HTTP 403: permission denied")

    monkeypatch.setattr(gemini_main, "gemini_stream_generate", fake_generate)

    with pytest.raises(RuntimeError, match="Gemini HTTP 403"):
        asyncio.run(
            gemini_main.gemini_stream_generate_with_fallback(
                "gemini-3.8-flash", "Who is Gourav?", [{"text": "Gourav"}]
            )
        )

    assert calls == ["gemini-3.8-flash"]
