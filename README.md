# Enterprise RAG Evaluation Platform

Production-oriented Retrieval-Augmented Generation workbench for **dynamic PDF ingestion, multi-document hybrid retrieval, reranking, grounded LLM generation, abstention, citations, latency telemetry, evaluation, and reproducible benchmarks**.

## Production query workflow

```text
User question
    ↓
Document scope: selected document OR entire knowledge base
    ↓
Page-aware extraction + paragraph-aware overlapping chunks
    ↓
BM25 + TF-IDF hybrid candidate retrieval
    ↓
Cross-signal reranking of top candidates
    ↓
Evidence / answerability gate
    ↓
Best reranked chunks → grounded LLM
    ↓
Concise answer + source/page attribution
    ↓
Retrieval + rerank + generation + end-to-end latency telemetry
```

The corpus is **not hard-coded**. Users upload PDFs through the web UI or `POST /api/documents/upload`. By default a question searches the whole indexed knowledge base; clicking **Use document** scopes the query to one document.

## Retrieval and generation

- **Hybrid retrieval:** BM25 + TF-IDF bigram retrieval provides complementary lexical signals.
- **Candidate expansion:** retrieves a larger candidate pool before final ranking.
- **Reranking:** combines hybrid score, query-term coverage, phrase overlap, and query-intent signals to select the best evidence chunks.
- **Chunking:** paragraph-aware page chunks with configurable `CHUNK_WORDS` and `CHUNK_OVERLAP` preserve page attribution while reducing oversized context.
- **Grounded generation:** when `OPENAI_API_KEY` is configured, only reranked chunks are sent to the LLM with an explicit no-invention instruction and `[Source N]` citation format.
- **Safe fallback:** without an LLM key, the same reranked evidence is used for a deterministic extractive answer; unsupported questions abstain.
- **Document-aware citations:** duplicate chunks are consolidated into distinct document/page source groups.

## Quality safeguards

- **Answerability gate:** weak or unsupported questions abstain instead of returning an unrelated top chunk.
- **Intent-aware extraction/reranking:** definition, method, purpose and comparison questions receive different evidence preferences.
- **Noise suppression:** numeric/table-heavy evidence is down-ranked for normal explanatory questions.
- **Concise answers:** fallback extraction returns at most two high-quality evidence sentences.
- **Multi-document isolation:** explicit document selection scopes retrieval; otherwise all indexed chunks are candidates.
- **Telemetry:** `/api/ask` exposes retrieval, reranking, generation and total latency plus candidate/context counts.

## API

### Upload

`POST /api/documents/upload` with multipart field `file`.

### List documents

`GET /api/documents`

### Retrieve and rerank

`GET /api/retrieve?q=<question>&document_id=<optional>&k=5`

The response includes `answerable`, `evidence_coverage`, `latency`, the reranker name, and ranked evidence chunks.

### Ask

`POST /api/ask`

```json
{
  "question": "How does the Transformer differ from BERT?",
  "document_id": "optional-id",
  "top_k": 5
}
```

The response includes `answerable`, `evidence_coverage`, `mode`, grouped `citations`, `latency`, and the reranked evidence used to construct the answer.

## LLM configuration

For the full retrieval → rerank → LLM workflow, configure an OpenAI-compatible endpoint:

```env
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=your-model
```

Any compatible provider can be used by changing `OPENAI_BASE_URL` and `LLM_MODEL`.

## Benchmark

Run the reproducible benchmark with:

```bash
python benchmark/benchmark.py
```

The benchmark compares **Hybrid** with **Hybrid + Rerank** using Hit@K, Recall@K, MRR, nDCG@K, mean latency and p95 latency. It also reports separate retrieval and reranking latency. Results are generated at runtime and must be rerun after retrieval/chunking changes; no resume metric is hard-coded.

## Regression tests

```bash
pytest -q
```

`tests/test_pipeline_v31.py` covers multi-document retrieval, definition-aware reranking, abstention, and query-intent classification. Existing answer-quality tests cover the QIS definition/noise and unsupported-email failure modes.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000`.

## Deployment

Docker now starts `main:app`. The application is suitable for Railway/Render/any Docker host. Mount `/app/data` (or set `DATA_DIR`) to persistent storage in production because uploaded PDFs and SQLite data are otherwise ephemeral.
