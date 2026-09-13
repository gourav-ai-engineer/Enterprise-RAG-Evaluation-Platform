from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = Path(__file__).with_name("dataset.json")
RESULTS_DIR = Path(__file__).with_name("results")
RESULTS_DIR.mkdir(exist_ok=True)


def tokens(text: str) -> list[str]:
    return [x.lower() for x in __import__("re").findall(r"[a-zA-Z0-9_]+", text)]


def bm25_scores(query: str, docs: list[str]) -> list[float]:
    q = tokens(query)
    n = len(docs)
    freqs = []
    dfs: dict[str, int] = {}
    lengths = []
    for doc in docs:
        tf: dict[str, int] = {}
        for term in tokens(doc):
            tf[term] = tf.get(term, 0) + 1
        freqs.append(tf)
        lengths.append(sum(tf.values()))
        for term in tf:
            dfs[term] = dfs.get(term, 0) + 1
    avgdl = sum(lengths) / max(1, n)
    k1, b = 1.5, 0.75
    scores = []
    for tf, dl in zip(freqs, lengths):
        score = 0.0
        for term in q:
            f = tf.get(term, 0)
            if not f:
                continue
            df = dfs[term]
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / max(1, avgdl)))
        scores.append(score)
    return scores


def ranks(values: list[float], reverse: bool = True) -> list[int]:
    return sorted(range(len(values)), key=lambda i: (-values[i], i) if reverse else (values[i], i))


def reciprocal_rank_fusion(*rankings: list[int], k: int = 60) -> list[int]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking, start=1):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (k + rank)
    return [idx for idx, _ in sorted(fused.items(), key=lambda x: (-x[1], x[0]))]


def hit_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    return float(bool(set(retrieved[:k]) & relevant))


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / max(1, len(relevant))


def mrr(retrieved: list[int], relevant: set[int], k: int) -> float:
    for rank, idx in enumerate(retrieved[:k], start=1):
        if idx in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    dcg = 0.0
    for rank, idx in enumerate(retrieved[:k], start=1):
        if idx in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_len = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_len + 1))
    return dcg / idcg if idcg else 0.0


def evaluate(name: str, ranker, documents, queries, k_values=(1, 3, 5)) -> dict:
    values = {f"Hit@{k}": [] for k in k_values}
    values.update({f"Recall@{k}": [] for k in k_values})
    values.update({f"NDCG@{k}": [] for k in k_values})
    values["RR"] = []
    values["latency_ms"] = []
    for item in queries:
        relevant = {next(i for i, d in enumerate(documents) if d["id"] == rid) for rid in item["relevant"]}
        t0 = time.perf_counter()
        retrieved = ranker(item["query"])
        elapsed = (time.perf_counter() - t0) * 1000
        values["latency_ms"].append(elapsed)
        values["RR"].append(mrr(retrieved, relevant, max(k_values)))
        for k in k_values:
            values[f"Hit@{k}"].append(hit_at_k(retrieved, relevant, k))
            values[f"Recall@{k}"].append(recall_at_k(retrieved, relevant, k))
            values[f"NDCG@{k}"].append(ndcg_at_k(retrieved, relevant, k))
    summary = {"retriever": name, "queries": len(queries)}
    for metric, series in values.items():
        summary[metric] = round(statistics.mean(series), 4)
    summary["latency_p95_ms"] = round(sorted(values["latency_ms"])[max(0, math.ceil(len(values["latency_ms"]) * 0.95) - 1)], 4)
    return summary


def main() -> int:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    documents = dataset["documents"]
    queries = dataset["queries"]
    texts = [d["text"] for d in documents]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    matrix = vectorizer.fit_transform(texts)

    def bm25(query: str) -> list[int]:
        return ranks(bm25_scores(query, texts))

    def tfidf(query: str) -> list[int]:
        q = vectorizer.transform([query])
        return ranks(cosine_similarity(q, matrix)[0].tolist())

    def hybrid(query: str) -> list[int]:
        return reciprocal_rank_fusion(bm25(query), tfidf(query))

    summaries = [
        evaluate("BM25", bm25, documents, queries),
        evaluate("TF-IDF", tfidf, documents, queries),
        evaluate("Hybrid-RRF", hybrid, documents, queries),
    ]
    output = {"dataset": str(DATASET_PATH), "results": summaries}
    (RESULTS_DIR / "benchmark_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    lines = ["# Retrieval Benchmark Results", "", "Generated by `python benchmark/benchmark.py`.", "", "| Retriever | Hit@1 | Hit@3 | Recall@5 | MRR | NDCG@5 | Mean latency (ms) | p95 latency (ms) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['retriever']} | {r['Hit@1']:.3f} | {r['Hit@3']:.3f} | {r['Recall@5']:.3f} | {r['RR']:.3f} | {r['NDCG@5']:.3f} | {r['latency_ms']:.3f} | {r['latency_p95_ms']:.3f} |")
    (RESULTS_DIR / "benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
