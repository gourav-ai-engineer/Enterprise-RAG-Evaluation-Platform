# Retrieval Benchmark

This benchmark compares three retrievers on a fixed, version-controlled question set:

- **BM25** — lexical ranking
- **TF-IDF cosine** — sparse semantic proxy
- **Hybrid-RRF** — Reciprocal Rank Fusion of both rankings

Metrics: Hit@K, Recall@K, MRR, nDCG@K, mean latency and p95 latency.

## Run

From the repository root:

```bash
python benchmark/benchmark.py
```

Outputs:

- `benchmark/results/benchmark_results.json`
- `benchmark/results/benchmark_report.md`

## Resume-safe reporting

Do **not** copy invented metrics into a resume. Run the benchmark on your machine or CI and use the generated measurements.

Example resume format:

> Built an enterprise RAG evaluation platform with PDF ingestion, hybrid BM25 + TF-IDF retrieval, page-level citations, and reproducible retrieval benchmarking; improved **[metric]** by **[measured delta]** versus **[baseline]** across **[N]** evaluation queries.

Example achievement format:

> Benchmarked BM25, TF-IDF and hybrid RRF retrieval using Recall@K, MRR, nDCG@K and p95 latency; achieved **[measured result]** on the version-controlled benchmark suite.

## Extending the benchmark

Add more documents and questions to `dataset.json`. Each question should identify one or more relevant document IDs. Keep the dataset version-controlled and report dataset size with every result so experiments remain comparable.
