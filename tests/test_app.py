from io import BytesIO

from fastapi.testclient import TestClient

from app import app, hybrid_retrieve, load_chunks

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_pdf_upload_rejects_invalid_pdf() -> None:
    pdf = b"%PDF-1.4\n% invalid fixture\n"
    response = client.post(
        "/api/documents/upload",
        files={"file": ("fixture.pdf", BytesIO(pdf), "application/pdf")},
    )
    assert response.status_code == 400


def test_non_pdf_upload_is_rejected() -> None:
    response = client.post(
        "/api/documents/upload",
        files={"file": ("notes.txt", BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 400


def test_hybrid_retrieval_empty_index_returns_empty() -> None:
    assert hybrid_retrieve("test question", [], 5) == []


def test_health_does_not_require_documents() -> None:
    assert load_chunks() is not None
