# Enterprise RAG Evaluation Platform

Production-oriented Retrieval-Augmented Generation workbench for **PDF upload, document indexing, hybrid retrieval, grounded Q&A, citations, and reproducible retrieval benchmarks**.

## What it does

```text
PDF upload
   ↓
PyPDF extraction
   ↓
page-aware chunking
   ↓
SQLite document/index store
   ↓
BM25 + TF-IDF hybrid retrieval
   ↓
source/page citations
   ↓
LLM generation (optional) / extractive fallback
   ↓
offline answer-quality evaluation
```

The corpus is **not hard-coded**. Users upload their own PDFs through the web UI or `POST /api/documents/upload`.

## Features

- PDF upload with file-size and MIME validation
- Page-aware text extraction and overlapping chunks
- Document registry and local SQLite index
- Hybrid BM25 + TF-IDF retrieval
- Retrieval inspection with component scores
- Q&A restricted to retrieved evidence
- Source and page citations in the answer payload
- Optional OpenAI-compatible LLM endpoint via environment variables
- Deterministic extractive fallback when no LLM key is configured
- Offline groundedness, answer-relevance and reference-overlap evaluation
- Benchmark harness for Recall@K, Hit@K, MRR, nDCG@K and latency
- Dockerized deployment and CI tests

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`.

## API

### Upload

`POST /api/documents/upload` with multipart field `file`.

### List documents

`GET /api/documents`

### Retrieve

`GET /api/retrieve?q=<question>&document_id=<optional>&k=5`

### Ask

`POST /api/ask`

```json
{
  "question": "What is the main conclusion?",
  "document_id": "optional-id",
  "top_k": 5
}
```

### Evaluate

`POST /api/evaluate`

```json
{
  "question": "What is the main conclusion?",
  "answer": "...",
  "document_id": "optional-id",
  "reference_answer": "optional ground truth",
  "top_k": 5
}
```

## LLM configuration

The application remains usable without an LLM key. For generative answers, configure an OpenAI-compatible endpoint:

```env
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=your-model
```

Any compatible provider can be used by changing `OPENAI_BASE_URL` and `LLM_MODEL`.

## Benchmark

The benchmark is intentionally reproducible rather than relying on marketing numbers. See [`benchmark/README.md`](benchmark/README.md) and run:

```bash
python benchmark/benchmark.py
```

It produces JSON and Markdown reports with retrieval quality and latency. Use the generated measurements—not invented numbers—in a resume, portfolio, or project write-up.

## Deployment

The project is containerized for Railway/Render/any Docker host. For production persistence, mount `/app/data` (or set `DATA_DIR`) to a persistent volume; the demo deployment can also run with ephemeral storage.

## Research basis

The architecture follows patterns used in public production-oriented RAG projects: hybrid lexical+dense retrieval, reranking/traceability, PDF ingestion, and explicit retrieval evaluation. The benchmark layer uses standard retrieval metrics such as Hit@K, Recall@K, MRR and nDCG.
