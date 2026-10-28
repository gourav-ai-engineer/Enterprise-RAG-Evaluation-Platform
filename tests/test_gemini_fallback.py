import asyncio

import pytest

import gemini_main


def test_gemini_overload_retries_then_falls_back(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_sleep(_seconds: float) -> None:
        return None

    async def fake_generate(model: str, question: str, results: list[dict], on_text=None):
        calls.append(model)
        if model == "gemini-3.8-flash":
            raise RuntimeError("Gemini HTTP 503: temporarily overloaded")
        if on_text is not None:
            await on_text("grounded answer")
        return "grounded answer", 12.0, 30.0, 2

    monkeypatch.setattr(gemini_main.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(gemini_main, "gemini_stream_generate", fake_generate)

    result = asyncio.run(
        gemini_main.gemini_stream_generate_with_fallback(
            "gemini-3.8-flash", "Who is Gourav?", [{"text": "Gourav"}]
        )
    )

    assert result[0] == "grounded answer"
    assert result[3] == 2
    assert result[4] == "gemini-3.5-flash-lite"
    assert calls == ["gemini-3.8-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite"]


def test_non_transient_gemini_error_does_not_retry(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_generate(model: str, question: str, results: list[dict], on_text=None):
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


def test_streaming_callback_receives_partial_answer(monkeypatch) -> None:
    chunks: list[str] = []

    async def fake_generate(model: str, question: str, results: list[dict], on_text=None):
        if on_text is not None:
            await on_text("Gourav is ")
            await on_text("an AI/ML engineer.")
        return "Gourav is an AI/ML engineer.", 15.0, 40.0, 2

    async def collect(piece: str) -> None:
        chunks.append(piece)

    monkeypatch.setattr(gemini_main, "gemini_stream_generate", fake_generate)
    result = asyncio.run(
        gemini_main.gemini_stream_generate_with_fallback(
            "gemini-3.5-flash-lite",
            "Who is Gourav?",
            [{"text": "Gourav is an AI/ML engineer."}],
            on_text=collect,
        )
    )

    assert result[0] == "Gourav is an AI/ML engineer."
    assert chunks == ["Gourav is ", "an AI/ML engineer."]



def test_slow_first_token_fails_over_without_retrying_same_model(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setenv("GEMINI_FIRST_TOKEN_TIMEOUT_SECONDS", "0.5")
    monkeypatch.setenv("GEMINI_REQUEST_TIMEOUT_SECONDS", "2")
    monkeypatch.setenv("GEMINI_GENERATION_BUDGET_SECONDS", "8")

    async def fake_generate(model: str, question: str, results: list[dict], on_text=None):
        calls.append(model)
        if model == "gemini-3.5-flash-lite":
            await asyncio.sleep(2)
        if on_text is not None:
            await on_text("A grounded answer [Source 1].")
        return "A grounded answer [Source 1].", 1.0, 2.0, 1

    monkeypatch.setattr(gemini_main, "gemini_stream_generate", fake_generate)
    result = asyncio.run(
        gemini_main.gemini_stream_generate_with_fallback(
            "gemini-3.5-flash-lite",
            "Who is Gourav?",
            [{"text": "Gourav is described in the source."}],
        )
    )

    assert result[0] == "A grounded answer [Source 1]."
    assert result[4] == "gemini-3.6-flash"
    assert calls == ["gemini-3.5-flash-lite", "gemini-3.6-flash"]
    # The reported generation time includes the slow attempt, not only the winning model.
    assert result[2] >= 450
    assert result[1] >= 450
