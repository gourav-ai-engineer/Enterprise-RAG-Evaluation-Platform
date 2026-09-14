# Enterprise RAG Evaluation Platform

Production-oriented Retrieval-Augmented Generation workbench for **dynamic PDF ingestion, SHA-256 deduplication, cached ingestion, multi-document hybrid retrieval, reranking, grounded Gemini/Ollama generation, abstention, citations, document lifecycle management, collections, feedback, audit logs, telemetry, evaluation, and reproducible benchmarks**.

## Current architecture

```text
PDF upload
   ↓
Validate + SHA-256 content identity
   ↓
Duplicate check ──→ reuse existing indexed document
   ↓
Content-addressed PDF + ingestion cache
   ↓
Page-aware extraction + overlapping chunks
   ↓
BM25 + TF-IDF hybrid retrieval
   ↓
Reranking + evidence/answerability gate
   ↓
Grounded Gemini streaming answer
   ↓
Citations + latency trace + user feedback
```

The corpus is **not hard-coded**. Users upload PDFs through the web UI or `POST /api/documents/upload`.

## Enterprise capabilities

### Ingestion and data integrity
- PDF MIME/extension validation, size limit and PDF magic-byte validation.
- SHA-256 content identity means the same PDF is detected as a duplicate even when uploaded under a different filename.
- Content-addressed storage at `data/documents/<sha256>.pdf`.
- Ingestion cache at `data/cache/<sha256>.json` so parsing/chunking can be reused.
- Reprocessing endpoint with document version increment.
- Safe document deletion and cache invalidation.
- Retrieval cache is scoped by query, chunk set and top-k so document scope cannot leak across requests.

### Retrieval and generation
- Hybrid BM25 + TF-IDF bigram retrieval.
- Candidate expansion followed by intent-aware reranking.
- Definition/method/purpose signals and table/numeric noise suppression.
- Evidence/answerability gate with safe abstention.
- Gemini streaming generation with strict evidence-only prompting and source IDs.
- Optional Ollama/OpenAI-compatible workflow remains available through the underlying application modules.
- Grouped document/page citations and exact evidence inspection.

### Knowledge operations
- Collections with unique names and document assignment.
- Document status, source type, version and update metadata.
- Document quality signal based on indexed text density.
- Reprocess/delete lifecycle actions.

### Evaluation-driven operations
- Query traces persisted to SQLite with retrieval, rerank, TTFT, generation and total latency.
- Evidence coverage and answerability tracking.
- 👍/👎 feedback persisted for later evaluation-set improvement.
- Audit log for collection, document lifecycle and feedback events.
- Live enterprise telemetry strip in the workbench.
- `/api/enterprise/overview` exposes real operational counters; no metrics are hard-coded.

## API

### Upload
`POST /api/documents/upload` with multipart field `file`.

Response includes `sha256`, `duplicate`, `cache_hit`, page count and chunk count.

### Documents
- `GET /api/documents`
- `POST /api/documents/{id}/reprocess`
- `DELETE /api/documents/{id}`
- `PATCH /api/documents/{id}/collection`

### Collections
- `GET /api/collections`
- `POST /api/collections`

### Enterprise telemetry
- `GET /api/enterprise/overview`
- `GET /api/admin/audit-logs`
- `POST /api/traces`
- `POST /api/feedback`
- `GET /health`

### Retrieve / Ask / Evaluate
The underlying APIs remain available from `main.py`/`gemini_main.py`, including document-scoped retrieval, grounded Q&A and evaluation.

## Gemini configuration

```powershell
$env:GEMINI_API_KEY="your-key"
$env:GEMINI_MODEL="gemini-3.5-flash-lite"
```

Run the complete enterprise application locally:

```powershell
.\.venv\Scripts\python.exe -m uvicorn enterprise_main:app --reload
```

Open `http://127.0.0.1:8000`.

## Cache diagnostics

```text
GET /api/cache/stats
GET /api/enterprise/overview
```

The cache is intentionally outside Git and should live under persistent storage in production.

## Benchmark

Run the reproducible benchmark after the current retrieval/chunking implementation has been verified:

```bash
python benchmark/benchmark.py
```

Benchmark results must be regenerated after retrieval, chunking, reranking or answerability changes. **Do not copy old benchmark numbers into the resume.**

## Tests

```bash
pytest -q
```

## Docker

The production image now starts `enterprise_main:app` and includes the enterprise lifecycle/telemetry layer. Mount `/app/data` (or set `DATA_DIR`) to persistent storage in production because SQLite, uploaded PDFs and ingestion caches are stateful.

## Roadmap aligned to the enterprise PR

The current release establishes the production foundation first: integrity, caching, document lifecycle, collections, feedback, auditability and telemetry. Next layers can be added without replacing the working retrieval core: PostgreSQL/Qdrant/OpenSearch adapters, asynchronous workers, OCR, authentication/RBAC, policy-aware retrieval, RAGAS/DeepEval evaluation, Prometheus/OpenTelemetry and multi-tenant isolation.
