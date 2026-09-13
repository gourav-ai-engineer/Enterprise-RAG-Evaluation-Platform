# Retrieval Benchmark

This benchmark compares three retrievers on a fixed, version-controlled question set:

- **BM25** — lexical ranking
- **TF-IDF cosine** — sparse semantic proxy
- **Hybrid-RRF** — Reciprocal Rank Fusion of both rankings

Metrics follow common retrieval-evaluation practice: Hit@K, Recall@K, MRR, nDCG@K, mean latency and p95 latency. Public RAG evaluation projects also use these metrics to compare retrieval quality and efficiency. citeturn546053search3turn546053search7

## Run

From the repository root:

```bash
python benchmark/benchmark.py
```

Outputs:

- `benchmark/results/benchmark_results.json`
- `benchmark/results/benchmark_report.md`

## Resume-safe reporting

Do **not** copy invented metrics into a resume. Run the benchmark on your machine/CI and use the generated measurements.

A strong resume format is:

> Built an enterprise RAG evaluation platform with PDF ingestion, hybrid BM25 + TF-IDF retrieval, page-level citations, and reproducible retrieval benchmarking; improved **[metric]** by **[measured delta]** versus **[baseline]** across **[N]** evaluation queries.

A second achievement can report:

> Benchmarked BM25, TF-IDF and hybrid RRF retrieval using Recall@K, MRR, nDCG@K and p95 latency; achieved **[measured result]** on the version-controlled benchmark suite.

## Extending the benchmark

Add more documents and questions to `dataset.json`. Each question should identify one or more relevant document IDs. Keep the dataset version-controlled and report dataset size with every result so experiments remain comparable.
