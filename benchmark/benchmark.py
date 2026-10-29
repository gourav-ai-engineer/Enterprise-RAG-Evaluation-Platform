from __future__ import annotations

import json
import math
import os
import statistics
import time
import uuid
from pathlib import Path

import main as core

DATASET = Path(__file__).with_name("dataset.json")
RESULTS = Path(__file__).with_name("results")
RESULTS.mkdir(exist_ok=True)


def hit(ret: list[str], relevant: set[str], k: int) -> float:
    return float(bool(set(ret[:k]) & relevant))


def recall(ret: list[str], relevant: set[str], k: int) -> float:
    return len(set(ret[:k]) & relevant) / max(1, len(relevant))


def mrr(ret: list[str], relevant: set[str], k: int) -> float:
    for rank, document_id in enumerate(ret[:k], start=1):
        if document_id in relevant:
            return 1 / rank
    return 0.0


def ndcg(ret: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, document_id in enumerate(ret[:k], start=1)
        if document_id in relevant
    )
    ideal_count = min(k, len(relevant))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal if ideal else 0.0


def evaluate(
    name: str,
    ranker,
    queries: list[dict],
    k_values: tuple[int, ...] = (1, 3, 5),
) -> dict:
    metrics = {
        f"Hit@{k}": [] for k in k_values
    }
    metrics.update({f"Recall@{k}": [] for k in k_values})
    metrics.update({f"NDCG@{k}": [] for k in k_values})
    metrics["MRR"] = []
    metrics["latency_ms"] = []

    for query in queries:
        started = time.perf_counter()
        ranking = ranker(query["query"])
        elapsed_ms = (time.perf_counter() - started) * 1000
        relevant = set(query["relevant"])
        metrics["latency_ms"].append(elapsed_ms)
        metrics["MRR"].append(mrr(ranking, relevant, max(k_values)))
        for k in k_values:
            metrics[f"Hit@{k}"].append(hit(ranking, relevant, k))
            metrics[f"Recall@{k}"].append(recall(ranking, relevant, k))
            metrics[f"NDCG@{k}"].append(ndcg(ranking, relevant, k))

    result = {"retriever": name, "queries": len(queries)}
    for metric, values in metrics.items():
        if metric == "latency_ms":
            result["latency_mean_ms"] = round(statistics.mean(values), 4) if values else 0.0
            ordered = sorted(values)
            p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
            result["latency_p95_ms"] = round(ordered[p95_index], 4) if ordered else 0.0
        else:
            result[metric] = round(statistics.mean(values), 4) if values else 0.0
    return result


def main() -> None:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    documents = data["documents"]
    queries = data["queries"]
    run_id = uuid.uuid4().hex
    rows = [
        {
            "id": f"benchmark-{run_id}-{document['id']}",
            "document_id": document["id"],
            "filename": f"{document['id']}.txt",
            "page_start": 1,
            "page_end": 1,
            "text": document["text"],
        }
        for document in documents
    ]
    texts = [row["text"] for row in rows]
    chunk_ids = [row["id"] for row in rows]

    try:
        # Warm the embedding model and pre-index benchmark passage vectors before
        # measuring query latency. First-time model download is setup, not retrieval.
        core.index_dense_embeddings(rows)

        def bm25_only(query: str) -> list[str]:
            scores = core.bm25_scores(query, texts)
            ranked_indices = sorted(
                range(len(rows)),
                key=lambda index: (-scores[index], rows[index]["document_id"]),
            )
            return [rows[index]["document_id"] for index in ranked_indices]

        def hybrid_rrf(query: str) -> list[str]:
            results = core.hybrid_retrieve(query, rows, len(rows))
            return [item["document_id"] for item in results]

        def hybrid_rrf_reranked(query: str) -> list[str]:
            candidates = core.hybrid_retrieve(query, rows, len(rows))
            results = core.rerank(query, candidates, len(candidates))
            return [item["document_id"] for item in results]

        results = [
            evaluate("BM25 baseline", bm25_only, queries),
            evaluate("BM25 + TF-IDF + Dense (weighted RRF)", hybrid_rrf, queries),
            evaluate("Hybrid RRF + heuristic reranker", hybrid_rrf_reranked, queries),
        ]

        retrieval_times: list[float] = []
        rerank_times: list[float] = []
        for query in queries:
            for _ in range(3):
                started = time.perf_counter()
                candidates = core.hybrid_retrieve(query["query"], rows, len(rows))
                retrieval_times.append((time.perf_counter() - started) * 1000)

                started = time.perf_counter()
                core.rerank(query["query"], candidates, len(candidates))
                rerank_times.append((time.perf_counter() - started) * 1000)

        latency = {
            "retrieval_mean_ms": round(statistics.mean(retrieval_times), 4),
            "retrieval_p95_ms": round(
                sorted(retrieval_times)[max(0, math.ceil(len(retrieval_times) * 0.95) - 1)],
                4,
            ),
            "rerank_mean_ms": round(statistics.mean(rerank_times), 4),
            "rerank_p95_ms": round(
                sorted(rerank_times)[max(0, math.ceil(len(rerank_times) * 0.95) - 1)],
                4,
            ),
        }
        output = {
            "dataset": str(DATASET),
            "embedding_model": core.dense_retrieval.model_name(),
            "fusion": {
                "method": "weighted reciprocal rank fusion",
                "rrf_k": int(os.getenv("HYBRID_RRF_K", "60")),
                "weights": {
                    "bm25": float(os.getenv("HYBRID_BM25_WEIGHT", "0.35")),
                    "tfidf": float(os.getenv("HYBRID_TFIDF_WEIGHT", "0.15")),
                    "dense": float(os.getenv("HYBRID_DENSE_WEIGHT", "0.50")),
                },
            },
            "results": results,
            "pipeline_latency": latency,
            "note": (
                "Uses the production retrieval and reranking functions. Dense vectors are "
                "pre-indexed before timing. Rerun on the current code before quoting any metrics."
            ),
        }
        (RESULTS / "benchmark_results.json").write_text(
            json.dumps(output, indent=2), encoding="utf-8"
        )

        lines = [
            "# Retrieval Benchmark Results",
            "",
            "Generated by `python benchmark/benchmark.py` using the production retriever.",
            "",
            "| Retriever | Hit@1 | Hit@3 | Recall@5 | MRR | NDCG@5 | Mean latency (ms) | p95 latency (ms) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for result in results:
            lines.append(
                f"| {result['retriever']} | {result['Hit@1']:.3f} | "
                f"{result['Hit@3']:.3f} | {result['Recall@5']:.3f} | "
                f"{result['MRR']:.3f} | {result['NDCG@5']:.3f} | "
                f"{result['latency_mean_ms']:.3f} | {result['latency_p95_ms']:.3f} |"
            )
        lines += [
            "",
            "## Pipeline latency",
            "",
            f"- Retrieval mean: {latency['retrieval_mean_ms']:.3f} ms",
            f"- Retrieval p95: {latency['retrieval_p95_ms']:.3f} ms",
            f"- Rerank mean: {latency['rerank_mean_ms']:.3f} ms",
            f"- Rerank p95: {latency['rerank_p95_ms']:.3f} ms",
            "",
            f"- Dense model: `{core.dense_retrieval.model_name()}`",
            "- Dense model download and passage indexing are excluded from query-latency measurements.",
        ]
        (RESULTS / "benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
    finally:
        # Benchmark vectors are temporary: keep the document corpus clean.
        conn = core.db()
        try:
            conn.executemany(
                "DELETE FROM chunk_embeddings WHERE chunk_id=?",
                [(chunk_id,) for chunk_id in chunk_ids],
            )
            conn.commit()
        finally:
            conn.close()


if __name__ == "__main__":
    main()
