# Resume / Achievement Metrics

## Project bullet

> Built a production-oriented Enterprise RAG platform supporting user-uploaded PDFs, page-aware chunking, hybrid BM25 + TF-IDF retrieval, source/page citations, optional OpenAI-compatible generation, and automated answer evaluation.

## Benchmark bullet

> Benchmarked BM25, TF-IDF and Hybrid-RRF retrieval on a version-controlled 15-document / 15-query suite, reaching **93.33% Hit@1**, **100% Hit@3**, **100% Recall@5**, **95.56% MRR** and **96.67% nDCG@5** for BM25/Hybrid-RRF in the reference run.

## Important wording

The above quality metrics are the current reference benchmark results for the bundled dataset. Latency is environment-specific. For an external resume/interview claim, rerun the benchmark in CI or on the target hardware and update the numbers.

## Stronger future achievement

After adding a larger real-world evaluation set, replace the reference sentence with an improvement claim such as:

> Improved **Recall@5 by X%** and reduced **p95 retrieval latency by Y%** versus the baseline after introducing hybrid retrieval/RRF.
