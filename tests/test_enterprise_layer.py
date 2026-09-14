from fastapi.testclient import TestClient

from enterprise_main import app


def test_enterprise_overview_and_health():
    with TestClient(app) as client:
        overview = client.get("/api/enterprise/overview")
        assert overview.status_code == 200
        payload = overview.json()
        assert "documents" in payload
        assert "retrieval_cache_entries" in payload

        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["version"] == "4.1.0"


def test_enterprise_root_exposes_telemetry():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Enterprise telemetry" in response.text
