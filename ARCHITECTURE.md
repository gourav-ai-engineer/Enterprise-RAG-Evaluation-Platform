# Architecture

## Scope

This repository is a self-contained RAG evaluation reference implementation. It keeps the retrieval and scoring paths deterministic and inspectable so behavior can be regression-tested without a paid model or external database.

## Request flow

1. A user submits a natural-language query.
2. The retriever tokenizes the query and corpus chunks.
3. A normalized lexical similarity score ranks chunks.
4. The evaluation endpoint retrieves evidence using the same retriever.
5. Groundedness measures answer-token coverage in the retrieved context.
6. Relevance measures query-token coverage in the answer.
7. When a reference answer is supplied, exact-match is also calculated.

## Production extension path

The current interfaces are intentionally small. A production version can replace each implementation independently:

- `CORPUS` -> Postgres/object storage ingestion pipeline
- `retrieve()` -> hybrid BM25 + vector retrieval
- lexical similarity -> cross-encoder reranking
- heuristic metrics -> reference metrics + LLM-as-judge evaluator
- in-memory state -> persistent evaluation runs and experiment tracking
- single process -> horizontally scaled API workers

This separation keeps the demo useful as both an educational project and a foundation for a larger evaluation platform.
