from fastapi.testclient import TestClient

from enterprise_main import app, VERSION


def test_enterprise_overview_and_health():
    with TestClient(app) as client:
        overview = client.get("/api/enterprise/overview")
        assert overview.status_code == 200
        payload = overview.json()
        assert payload["version"] == VERSION
        assert "documents" in payload
        assert "retrieval_cache_entries" in payload
        assert "retrieval_cache_hits" in payload

        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["version"] == VERSION


def test_enterprise_root_exposes_telemetry_and_feedback_hook():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Enterprise telemetry" in response.text
        assert "/api/traces" in response.text
        assert "/api/feedback" in response.text


def test_collection_feedback_and_trace_lifecycle():
    with TestClient(app) as client:
        name = "CI Collection"
        created = client.post("/api/collections", json={"name": name, "description": "CI test"})
        assert created.status_code in (200, 409)
        if created.status_code == 200:
            collection_id = created.json()["id"]
            listed = client.get("/api/collections")
            assert listed.status_code == 200
            assert any(item["id"] == collection_id for item in listed.json()["collections"])

        feedback = client.post("/api/feedback", json={
            "question": "What is RAG?",
            "answer": "A grounded retrieval pipeline.",
            "rating": 1,
            "document_ids": [],
            "mode": "test",
        })
        assert feedback.status_code == 200
        assert feedback.json()["saved"] is True

        trace = client.post("/api/traces", json={
            "question": "What is RAG?",
            "answerable": True,
            "evidence_coverage": 0.9,
            "mode": "test",
            "model": "test-model",
            "latency": {
                "retrieval_ms": 10,
                "rerank_ms": 2,
                "generation_ms": 30,
                "ttft_ms": 15,
                "total_ms": 42,
                "candidate_count": 5,
                "context_chunks": 3,
            },
            "cache_hit": False,
        })
        assert trace.status_code == 200
        assert trace.json()["saved"] is True

        overview = client.get("/api/enterprise/overview").json()
        assert overview["queries"] >= 1
        assert overview["positive_feedback_rate"] >= 0
