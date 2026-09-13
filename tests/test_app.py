from fastapi.testclient import TestClient

from app import app, retrieve

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_retrieve_returns_refund_policy() -> None:
    response = client.get("/api/retrieve", params={"q": "How long do refunds take?"})
    assert response.status_code == 200
    data = response.json()
    assert data["results"]
    assert data["results"][0]["source"] == "refund_policy.md"


def test_evaluate_grounded_answer_scores_well() -> None:
    response = client.post(
        "/api/evaluate",
        json={
            "query": "How long do refunds take?",
            "answer": "Refunds are processed within 5-7 business days after approval.",
        },
    )
    assert response.status_code == 200
    assert response.json()["groundedness"] > 0.8
    assert response.json()["relevance"] > 0.3


def test_validation_rejects_short_query() -> None:
    response = client.get("/api/retrieve", params={"q": "x"})
    assert response.status_code == 422


def test_retrieval_is_deterministic() -> None:
    first = retrieve("enterprise SSO", 3)
    second = retrieve("enterprise SSO", 3)
    assert first == second
