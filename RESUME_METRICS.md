# Resume / Achievement Metrics

## Project bullet

> Built a production-oriented Enterprise RAG platform supporting user-uploaded PDFs, page-local overlapping chunks, hybrid BM25 + TF-IDF + dense semantic retrieval, weighted Reciprocal Rank Fusion, heuristic reranking, source/page citations, grounded Gemini generation, and automated retrieval benchmarking.

## Benchmark reporting

The benchmark implementation has changed to call the actual production retriever and compare BM25-only against sparse-dense RRF and RRF plus reranking. **Previous reference numbers from the lexical-only benchmark are not evidence for the current dense hybrid.** Run `python benchmark/benchmark.py` on the current commit and replace this note only with the generated results.

Report the model, dataset size, relevant-query labels, baseline, metric and hardware/runtime context. Latency depends on CPU, model-cache state and corpus size. The first model download and vector indexing are intentionally excluded from query-latency timing.

## Suggested measured bullet

> Improved **[Hit@K/Recall@K/MRR/nDCG]** by **[measured delta]** over a BM25-only baseline across **[N]** labeled queries using dense semantic embeddings and weighted Reciprocal Rank Fusion.

Do not invent or copy previously measured metrics into an external resume. Expand the bundled dataset beyond the current small smoke/regression set before making broad accuracy claims.
