# Enterprise RAG Evaluation Platform

Production-oriented Retrieval-Augmented Generation workbench for **dynamic PDF ingestion, hybrid retrieval, grounded concise Q&A, abstention, citations, evaluation, and reproducible retrieval benchmarks**.

## What it does

```text
PDF upload
   ↓
page-aware extraction + text cleanup
   ↓
overlapping chunks + SQLite index
   ↓
BM25 + TF-IDF hybrid retrieval
   ↓
answerability / evidence gate
   ↓
answer-focused sentence selection or optional LLM
   ↓
concise answer + deduplicated page citations
   ↓
offline quality evaluation + benchmark
```

The corpus is **not hard-coded**. Users upload their own PDFs through the web UI or `POST /api/documents/upload`.

## Key quality safeguards

- **Grounded answerability gate:** unsupported questions abstain instead of returning the highest-scoring unrelated chunk.
- **Answer-focused extraction:** definition and explanatory sentences are preferred over table-like numeric text.
- **Noise suppression:** computation-time/memory/table-heavy sentences are down-ranked for normal Q&A.
- **Concise answers:** the deterministic fallback returns at most two high-quality evidence sentences.
- **Citation deduplication:** the user-facing UI shows at most three distinct source/page groups.
- **Advanced retrieval data remains available:** `/api/retrieve` and the API payload still expose retrieval details for evaluation.

## Features

- PDF upload with file-size and MIME validation
- Page-aware text extraction and overlapping chunks
- Document registry and local SQLite index
- Hybrid BM25 + TF-IDF retrieval
- Retrieval inspection with component scores
- Grounded Q&A restricted to retrieved evidence
- Safe abstention for unsupported questions
- Source and page citations
- Optional OpenAI-compatible LLM endpoint
- Deterministic extractive fallback when no LLM key is configured
- Offline groundedness, answer-relevance and reference-overlap evaluation
- Benchmark harness for Recall@K, Hit@K, MRR, nDCG@K and latency
- Dockerized deployment and regression tests

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

The response includes `answerable`, `evidence_coverage`, `mode`, `citations`, and the retrieved evidence.

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

## Regression coverage

`tests/test_answer_quality.py` covers the failure mode found during manual testing: a question such as **“What is QIS?”** must prefer the explanatory definition instead of a noisy scoring table, while an unsupported question such as **“What email addresses are used?”** must abstain.

## Deployment

The project is containerized for Railway/Render/any Docker host. For production persistence, mount `/app/data` (or set `DATA_DIR`) to a persistent volume; ephemeral storage is suitable for demos only.
