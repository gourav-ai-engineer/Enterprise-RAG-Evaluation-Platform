"""Local dense embeddings for semantic PDF retrieval.

FastEmbed runs a small quantized ONNX embedding model on CPU. Vectors are
persisted by the caller so the model is only needed when indexing new chunks
or embedding a new query.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", APP_DIR / "data"))
MODEL_CACHE_DIR = Path(
    os.getenv("FASTEMBED_CACHE_PATH", str(DATA_DIR / "models" / "fastembed"))
)


def model_name() -> str:
    return os.getenv("DENSE_EMBEDDING_MODEL", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME


@lru_cache(maxsize=2)
def _load_model(name: str, cache_dir: str, threads: int):
    # Import lazily. Health checks and application startup should not download
    # a model; the first indexing/query operation initializes it on demand.
    from fastembed import TextEmbedding

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return TextEmbedding(
        model_name=name,
        cache_dir=cache_dir,
        threads=threads,
        cuda=False,
    )


def get_model(name: str | None = None):
    configured_name = name or model_name()
    threads = max(1, int(os.getenv("DENSE_EMBEDDING_THREADS", "2")))
    return _load_model(configured_name, str(MODEL_CACHE_DIR), threads)


def embed_passages(texts: Iterable[str], name: str | None = None) -> list[np.ndarray]:
    """Encode document chunks in batches and return float32 vectors."""
    passage_texts = [str(text or "") for text in texts]
    if not passage_texts:
        return []
    model = get_model(name)
    vectors = list(model.embed(passage_texts, batch_size=32, parallel=None))
    if len(vectors) != len(passage_texts):
        raise RuntimeError(
            f"Dense embedder returned {len(vectors)} vectors for "
            f"{len(passage_texts)} passages"
        )
    return [np.asarray(vector, dtype=np.float32).reshape(-1) for vector in vectors]


def embed_query(query: str, name: str | None = None) -> np.ndarray:
    """Encode one query with the embedding model's query-specific path."""
    model = get_model(name)
    vectors = list(model.query_embed([str(query or "")]))
    if len(vectors) != 1:
        raise RuntimeError(f"Dense embedder returned {len(vectors)} vectors for one query")
    return np.asarray(vectors[0], dtype=np.float32).reshape(-1)


def cosine_similarity_scores(query_vector: np.ndarray, vectors: list[np.ndarray]) -> list[float]:
    """Cosine similarity with safe handling of empty/zero vectors."""
    if not vectors:
        return []
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    matrix = np.vstack([np.asarray(vector, dtype=np.float32).reshape(-1) for vector in vectors])
    if matrix.shape[1] != query.shape[0]:
        raise ValueError(
            f"Embedding dimension mismatch: query={query.shape[0]}, passages={matrix.shape[1]}"
        )
    query_norm = float(np.linalg.norm(query))
    matrix_norms = np.linalg.norm(matrix, axis=1)
    if query_norm == 0:
        return [0.0] * len(vectors)
    denominator = matrix_norms * query_norm
    similarities = np.divide(
        matrix @ query,
        denominator,
        out=np.zeros(len(vectors), dtype=np.float32),
        where=denominator > 0,
    )
    return [float(value) for value in similarities]
