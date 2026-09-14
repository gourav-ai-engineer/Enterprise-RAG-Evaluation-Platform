# Enterprise RAG Evaluation Platform

Production-oriented Retrieval-Augmented Generation workbench for **dynamic document ingestion, SHA-256 deduplication, cached ingestion, multi-document hybrid retrieval, reranking, grounded Gemini/Ollama generation, abstention, citations, document lifecycle management, collections, feedback, audit logs, telemetry, evaluation, and reproducible benchmarks**.

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
- Legacy SHA migration removes duplicate legacy rows before rebuilding the unique SHA index.
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
- Document detail endpoint with status, source type, version, update metadata and quality signal.
- Reprocess/delete lifecycle actions.
- Stored-file existence diagnostics.

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
- `GET /api/documents/{id}`
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
- `GET /api/cache/stats`
- `GET /health`

### Retrieve / Ask / Evaluate
The underlying retrieval, grounded Q&A and evaluation APIs remain available through the composed application stack.

## Gemini configuration

```powershell
$env:GEMINI_API_KEY="your-key"
$env:GEMINI_MODEL="gemini-3.5-flash-lite"
```

Run the complete enterprise application locally:

```powershell
.\.venv\Scripts\python.exe -m py_compile enterprise_main.py
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

Run the reproducible benchmark only after the current upload, deduplication, cache and query path has been verified:

```bash
python benchmark/benchmark.py
```

Benchmark results must be regenerated after retrieval, chunking, reranking, caching or answerability changes. **Do not copy old benchmark numbers into the resume.**

## Tests

```bash
pytest -q
```

## Docker

The production image starts `enterprise_main:app` and includes the enterprise lifecycle/telemetry layer. Mount `/app/data` (or set `DATA_DIR`) to persistent storage in production because SQLite, uploaded PDFs and ingestion caches are stateful.

## Roadmap aligned to the enterprise PR

The current release establishes a tested production foundation first: content integrity, caching, document lifecycle, collections, feedback, auditability and telemetry. The next implementation layers should be delivered incrementally with integration tests:

1. semantic embeddings + Qdrant/pgvector adapter
2. OpenSearch BM25 adapter for large corpora
3. asynchronous Redis/Celery ingestion
4. Docling/Tesseract OCR pipeline
5. MinIO object-storage adapter
6. Keycloak/OIDC authentication and RBAC
7. policy-aware retrieval and tenant isolation
8. RAGAS/DeepEval regression suite
9. OpenTelemetry + Prometheus/Grafana
10. multi-model routing and production load testing

These components are **not claimed as implemented until they are actually integrated and tested**.
