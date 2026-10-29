# Retrieval Benchmark

The benchmark now calls the **production retrieval implementation** in `main.py`, rather than maintaining a separate lexical-only copy of the algorithms.

Compared methods:

- **BM25 baseline** — sparse keyword ranking.
- **BM25 + TF-IDF + Dense (weighted RRF)** — current hybrid retriever.
- **Hybrid RRF + heuristic reranker** — the current hybrid retriever followed by the production query-aware reranker.

Dense retrieval uses the configured FastEmbed model (default `BAAI/bge-small-en-v1.5`). Passage vectors are indexed before timing. First-time model download and vector creation are setup costs, not query-latency measurements.

Metrics:

- Hit@K
- Recall@K
- MRR
- nDCG@K
- Mean and p95 latency
- Retrieval-stage and rerank-stage latency

## Run

From the repository root:

```bash
python benchmark/benchmark.py
```

The first run downloads the small ONNX embedding model unless it is already cached. Model artifacts are stored under `data/models/fastembed` by default. In Docker, persist the `/app/data` volume so model files and indexed vectors survive restarts.

Outputs:

- `benchmark/results/benchmark_results.json`
- `benchmark/results/benchmark_report.md`

The benchmark stores temporary vectors using unique IDs and removes them when finished. It uses the same `hybrid_retrieve` and `rerank` functions as the running application.

## Chunking and retrieval evaluation

Chunk size, overlap, sparse/dense weights, RRF constant, and semantic answerability thresholds are experimental variables. Evaluate them on representative labeled queries, including paraphrases, multi-document questions and unanswerable questions. Do not assume a universal optimal chunk size or weight setting.

A useful evaluation should compare the current hybrid to BM25-only and measure both ranking quality and latency. Use a larger domain-specific set before reporting external resume metrics; the bundled 15-query set is suitable for smoke/regression evaluation, not a claim of broad production accuracy.

## Resume-safe reporting

Do not quote old benchmark numbers after the retrieval implementation changes. Run the current benchmark and report generated metrics with the query/document count and baseline named explicitly.

Example:

> Built an enterprise RAG platform with page-local PDF chunking, hybrid BM25/TF-IDF/dense semantic retrieval, weighted Reciprocal Rank Fusion, query-aware reranking, grounded generation, and source/page citations; measured **[metric]** against BM25-only across **[N]** labeled queries.

## Extending the benchmark

Add queries to `dataset.json` and label all relevant document IDs. Include exact wording, synonyms/paraphrases, multi-hop or multi-document questions, negative/unanswerable questions, OCR-heavy files, and cases where unrelated documents repeat the same entity name.
