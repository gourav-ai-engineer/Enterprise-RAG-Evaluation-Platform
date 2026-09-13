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
    assert response.json()["scores"]["groundedness"] > 0.8


def test_batch_evaluation() -> None:
    response = client.post(
        "/api/evaluate/batch",
        json={
            "items": [
                {"query": "What is the API rate limit?", "answer": "100 requests per minute."},
                {"query": "What is required for deployment?", "answer": "A successful test suite and approved change record."},
            ]
        },
    )
    assert response.status_code == 200
    assert response.json()["count"] == 2


def test_retrieval_is_deterministic() -> None:
    first = retrieve("enterprise SSO", 3)
    second = retrieve("enterprise SSO", 3)
    assert first == second
