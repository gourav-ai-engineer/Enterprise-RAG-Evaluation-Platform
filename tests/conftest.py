from __future__ import annotations

import pytest

import main as core


@pytest.fixture(autouse=True)
def avoid_network_model_download_in_unit_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep generic unit tests deterministic; dense integration is tested with fake vectors."""
    monkeypatch.setattr(
        core,
        "dense_similarity_scores",
        lambda query, rows: [0.0 for _ in rows],
    )
