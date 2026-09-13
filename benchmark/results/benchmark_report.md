# Retrieval Benchmark Results

Reference run on the 15-document / 15-query versioned benchmark dataset.

| Retriever | Hit@1 | Hit@3 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| BM25 | 93.33% | 100.00% | 100.00% | 95.56% | 96.67% |
| TF-IDF | 93.33% | 93.33% | 100.00% | 95.00% | 96.20% |
| Hybrid-RRF | 93.33% | 100.00% | 100.00% | 95.56% | 96.67% |

Reference latency from the benchmark runner's execution environment:

| Retriever | Mean latency | p95 latency |
|---|---:|---:|
| BM25 | 1.249 ms | 1.608 ms |
| TF-IDF | 5.646 ms | 6.889 ms |
| Hybrid-RRF | 6.597 ms | 7.460 ms |

These latency values are hardware/runtime specific. Regenerate them with:

```bash
python benchmark/benchmark.py
```

Do not represent the reference latency as a production SLO. The quality metrics are the version-controlled retrieval benchmark results for this dataset.
