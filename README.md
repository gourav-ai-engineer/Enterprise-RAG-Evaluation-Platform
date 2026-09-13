# Enterprise RAG Evaluation Platform

A production-minded, inspectable Retrieval-Augmented Generation evaluation service. The project demonstrates retrieval ranking plus transparent answer-quality metrics through a FastAPI API and a browser UI.

## What it demonstrates

- Deterministic lexical retrieval with normalized similarity scores
- Source inspection through a retrieval API
- Groundedness scoring against retrieved context
- Query relevance scoring against the answer
- Optional exact-match scoring against a reference answer
- Batch evaluation endpoint for small evaluation sets
- OpenAPI documentation through FastAPI
- Dockerized deployment
- Automated tests with GitHub Actions

## Architecture

```text
Browser UI
    |
    v
FastAPI application
    +--> /api/retrieve ------> Retriever ------> In-memory corpus
    |
    +--> /api/evaluate ------> Retriever + Evaluation metrics
    |
    +--> /api/evaluate/batch -> Multiple evaluation runs
    |
    +--> /health
```

The retrieval layer is deliberately dependency-light so the demo is reproducible. In a production extension, the same interface can be backed by a vector database, BM25 index, hybrid search, reranker, and model-based evaluator.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open `http://localhost:8000` for the UI and `http://localhost:8000/docs` for interactive API documentation.

## Test

```bash
pip install pytest
pytest -q
```

## Docker

```bash
docker build -t enterprise-rag-eval .
docker run --rm -p 8000:8000 enterprise-rag-eval
```

## API examples

Retrieve evidence:

```bash
curl "http://localhost:8000/api/retrieve?q=How%20long%20do%20refunds%20take&k=3"
```

Evaluate an answer:

```bash
curl -X POST "http://localhost:8000/api/evaluate" \\
  -H "content-type: application/json" \\
  -d '{"query":"How long do refunds take?","answer":"Refunds are processed within 5-7 business days after approval."}'
```

Batch evaluation:

```bash
curl -X POST "http://localhost:8000/api/evaluate/batch" \\
  -H "content-type: application/json" \\
  -d '{"items":[{"query":"What is the API rate limit?","answer":"100 requests per minute."}]}'
```

## Evaluation note

The demo metrics are intentionally interpretable heuristics rather than claims of human-quality LLM judging. They are useful for regression testing retrieval and answer behavior, while a production evaluator would typically combine retrieval metrics, reference-based metrics, and model-based judging.

## Repository

Built as Project 2 in Gourav's AI systems portfolio.
