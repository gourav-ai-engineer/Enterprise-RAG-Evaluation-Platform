# Architecture

## Current application entry point

The Docker image starts `enterprise_main:app`. The application is assembled in layers:

- `enterprise_main.py`: schema migrations, collections, document lifecycle, feedback, audit logs, and telemetry.
- `production_main.py`: validated PDF upload, SHA-256 deduplication, content-addressed storage, chunk cache, dense indexing on upload, and cached retrieval wrapper.
- `gemini_main.py`: RAG Studio UI, request intent routing, evidence selection, source grouping, Gemini streaming, and bounded model fallback.
- `fast_main.py`: the underlying streaming API and optional OpenAI-compatible/Ollama path.
- `main.py`: SQLite schema, PDF chunking utilities, sparse scoring, dense vector persistence, RRF fusion, reranking, evidence sufficiency checks, and evaluation helpers.
- `dense_retrieval.py`: local FastEmbed model lifecycle, query/passage embeddings, and cosine similarity.
- `benchmark/benchmark.py`: ranking metrics and timing around the actual production retrieval/reranking functions.

## Ingestion flow

1. Validate the PDF extension/content and enforce the file-size limit.
2. Calculate SHA-256; reuse an already indexed identical document.
3. Extract PDF text with `pypdf` page by page.
4. Normalize text, remove repeated short page headers/footers, and create page-local chunks (180 words by default, 40 words of overlap).
5. Store the PDF, document/chunk metadata, and chunk embeddings. Dense embeddings use the configured FastEmbed model and are persisted in SQLite by chunk ID and model name.
6. Save the chunk cache with the chunker version; reprocessing invalidates old chunk embeddings and regenerates the index.

Scanned image-only PDFs need OCR; `pypdf` alone does not provide OCR.

## Query and retrieval flow

1. Load chunks only for selected document IDs when supplied.
2. Correct supported one-character spelling mistakes and, for profile-attribute questions, expand the retrieval query using vocabulary found in the selected corpus.
3. Calculate three ranking signals:
   - **BM25**: sparse wording relevance using term frequency, inverse document frequency, and chunk-length normalization.
   - **TF-IDF cosine similarity**: sparse unigram/bigram overlap.
   - **Dense cosine similarity**: semantic similarity between FastEmbed query and passage vectors.
4. Fuse rankings using weighted Reciprocal Rank Fusion (RRF), defaults BM25 0.35, TF-IDF 0.15, dense 0.50, with RRF constant 60.
5. Rerank candidates with a heuristic score that considers fused relevance, lexical coverage, explanatory/method wording, dense similarity, and a few noise penalties.
6. For multi-document questions, preserve topical evidence across relevant sources. Summary and method-explanation intents use balanced representative page-local passages rather than literal keyword search.
7. Apply the evidence sufficiency gate. Lexical coverage is preferred; for multi-concept queries a high dense similarity may support answerability even when the wording differs.
8. Send the original question and selected evidence to Gemini; stream tokens and source groups to the UI.

## Dense embedding storage

- Default model: `BAAI/bge-small-en-v1.5` (384 dimensions).
- Inference: FastEmbed/ONNX on CPU; it does not require PyTorch, a GPU, or a separate vector database.
- Dense vectors are stored in the `chunk_embeddings` SQLite table, keyed by `(chunk_id, model_name)`.
- Model artifacts live under `data/models/fastembed` by default. Persist the data volume across container restarts.
- Old or previously indexed chunks are embedded on demand when retrieval encounters missing vectors.
- The current implementation computes exact cosine similarity over the selected chunk vectors. For very large corpora, replace this scan with a vector index such as Qdrant or pgvector.

## Testing and benchmarking

- Unit tests mock the embedding scores or embedding model where possible, avoiding model downloads in CI.
- Dense retrieval tests cover semantic candidate ranking, RRF metadata, vector persistence, and cosine similarity.
- `python benchmark/benchmark.py` compares a BM25 baseline, the current BM25 + TF-IDF + dense RRF retriever, and RRF plus the heuristic reranker. It uses the production ranking code.
- The benchmark pre-indexes the small test corpus before timing, so the first model download and vector indexing are not counted as query latency.
- The benchmark set is small; report quality and latency only after running it on the current code and expanding it to representative real queries.

## Trade-offs and next steps

The current implementation favors a simple, inspectable local stack. Its reranker is heuristic, semantic similarity thresholds require calibration, SQLite vector scanning is linear in the selected corpus size, and OCR/authentication/tenant isolation are not implemented. Next improvements include a vector index for larger corpora, a learned cross-encoder reranker, OCR/layout-aware extraction, and evaluation on a larger labeled dataset.
