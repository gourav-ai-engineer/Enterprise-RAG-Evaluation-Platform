# Retrieval Benchmark

The benchmark compares the committed retrieval pipeline before and after reranking:

- **Hybrid** — BM25 + TF-IDF candidate ranking
- **Hybrid + Rerank** — hybrid candidates followed by the query-aware cross-signal reranker used by `main.py`

Metrics:

- Hit@K
- Recall@K
- MRR
- nDCG@K
- mean latency
- p95 latency
- retrieval-stage latency
- rerank-stage latency

## Run

From the repository root:

```bash
python benchmark/benchmark.py
```

Outputs:

- `benchmark/results/benchmark_results.json`
- `benchmark/results/benchmark_report.md`

## Chunking evaluation

The production index uses paragraph-aware, page-preserving chunks with configurable `CHUNK_WORDS` and `CHUNK_OVERLAP`. Chunking should be treated as an experimental variable, not a permanently assumed optimum. Research shows that chunk size and structure interact with the corpus and task; recent studies report different optima for concise fact retrieval versus broader contextual questions. citeturn0academia15turn0search0

For this project, the next benchmark stage should evaluate the same query set across several chunk profiles and select the configuration by measured retrieval quality plus latency, rather than claiming a universal "best" chunk size.

## Resume-safe reporting

Do **not** copy invented metrics into a resume. Run the benchmark on the current code and report the generated measurements.

Example:

> Built an enterprise RAG evaluation platform with paragraph-aware PDF chunking, hybrid BM25 + TF-IDF retrieval, query-aware reranking, grounded LLM generation, page citations, and reproducible latency benchmarking; improved **[measured metric]** by **[measured delta]** versus **[baseline]** across **[N]** queries.

## Extending the benchmark

Add more documents and questions to `dataset.json`. Each question should identify one or more relevant document IDs. For a realistic production benchmark, include multi-document questions, negative/unanswerable questions, page-level evidence, and questions spanning tables as well as prose.
