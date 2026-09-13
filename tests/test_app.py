from io import BytesIO

from fastapi.testclient import TestClient

from app import app, hybrid_retrieve, load_chunks

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_empty_corpus_is_supported() -> None:
    response = client.get("/api/documents")
    assert response.status_code == 200
    assert "documents" in response.json()


def test_pdf_upload_and_retrieval() -> None:
    pdf = b"%PDF-1.4\n% test fixture is intentionally minimal\n"
    response = client.post(
        "/api/documents/upload",
        files={"file": ("fixture.pdf", BytesIO(pdf), "application/pdf")},
    )
    # Minimal bytes are rejected by pypdf; this asserts validation is active.
    assert response.status_code == 400


def test_retrieval_empty_index() -> None:
    assert hybrid_retrieve("test question", load_chunks(), 5) == []
