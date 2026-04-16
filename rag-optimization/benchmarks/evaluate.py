"""
Retrieval quality evaluation — Recall@K and MRR.

Uses MS MARCO relevance judgments to verify that optimizations
don't degrade retrieval quality.
"""
import json
import numpy as np
from typing import Optional

import config


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int = config.TOP_K,
) -> float:
    """
    Recall@K: fraction of relevant documents found in the top-K results.
    """
    if not relevant_ids:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    return len(retrieved_set & relevant_ids) / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """
    Reciprocal Rank: 1/rank of the first relevant document.
    Returns 0 if no relevant document is found.
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retriever(
    retriever,
    queries: list[dict],
    chunk_lookup: Optional[dict] = None,
    k: int = config.TOP_K,
    use_batch_embedding: bool = False,
) -> dict:
    """
    Evaluate a retriever on a set of queries with relevance judgments.

    Args:
        retriever: Retriever instance with .retrieve(query) method
        queries: list of dicts with "text" and "relevant_passages" keys
        chunk_lookup: dict mapping chunk_id -> source passage text
                      (needed to map chunk results back to passage-level relevance)
        k: cutoff for Recall@K

    Returns:
        dict with mean Recall@K, MRR, and per-query details
    """
    recalls = []
    mrrs = []
    latencies = []
    details = []

    if use_batch_embedding:
        query_texts = [q["text"] for q in queries]
        batch_results = retriever.retrieve_batch(query_texts, use_batch_embedding=True)
        iterator = enumerate(zip(queries, batch_results))
    else:
        iterator = enumerate((q, retriever.retrieve(q["text"])) for q in queries)

    for i, (q, result) in iterator:

        retrieved_ids = [doc_id for doc_id, _ in result["results"]]

        # If using chunk-level retrieval, map back to passage text for matching
        if chunk_lookup:
            retrieved_texts = {chunk_lookup.get(did, "") for did in retrieved_ids}
        else:
            retrieved_texts = set(retrieved_ids)

        relevant_set = set(q.get("relevant_passages", []))

        # For text-based matching (when IDs don't align)
        if chunk_lookup:
            r_at_k = len(retrieved_texts & relevant_set) / max(len(relevant_set), 1)
            rr = 0.0
            for rank, doc_id in enumerate(retrieved_ids, start=1):
                text = chunk_lookup.get(doc_id, "")
                if text in relevant_set:
                    rr = 1.0 / rank
                    break
        else:
            r_at_k = recall_at_k(retrieved_ids, relevant_set, k)
            rr = reciprocal_rank(retrieved_ids, relevant_set)

        recalls.append(r_at_k)
        mrrs.append(rr)
        latencies.append(result["timings"]["total_ms"])

        details.append({
            "query": q["text"],
            "recall@k": r_at_k,
            "mrr": rr,
            "latency_ms": result["timings"]["total_ms"],
        })

        if (i + 1) % 100 == 0:
            print(f"  Evaluated {i+1}/{len(queries)} queries")

    return {
        "mean_recall@k": float(np.mean(recalls)),
        "mean_mrr": float(np.mean(mrrs)),
        "mean_latency_ms": float(np.mean(latencies)),
        "std_latency_ms": float(np.std(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "num_queries": len(queries),
        "k": k,
        "details": details,
    }


def compare_results(baseline: dict, optimized: dict) -> str:
    """Pretty-print comparison between baseline and optimized evaluation."""
    lines = [
        "",
        "=" * 60,
        "QUALITY & LATENCY COMPARISON",
        "=" * 60,
        f"{'Metric':<25s} {'Baseline':>12s} {'Optimized':>12s} {'Delta':>12s}",
        "-" * 60,
    ]

    metrics = [
        ("Recall@K", "mean_recall@k", "{:.4f}"),
        ("MRR", "mean_mrr", "{:.4f}"),
        ("Mean Latency (ms)", "mean_latency_ms", "{:.2f}"),
        ("P95 Latency (ms)", "p95_latency_ms", "{:.2f}"),
    ]

    for label, key, fmt in metrics:
        b_val = baseline[key]
        o_val = optimized[key]

        if "latency" in key.lower():
            delta = f"{(o_val - b_val) / b_val * 100:+.1f}%"
        else:
            delta = f"{o_val - b_val:+.4f}"

        lines.append(
            f"{label:<25s} {fmt.format(b_val):>12s} {fmt.format(o_val):>12s} {delta:>12s}"
        )

    lines.append("=" * 60)
    report = "\n".join(lines)
    print(report)
    return report


# ──────────────────────────────────────────────
# Quick self-test with mock data
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Mock test
    retrieved = ["doc_3", "doc_7", "doc_1", "doc_5", "doc_9"]
    relevant = {"doc_1", "doc_5"}

    print(f"Recall@5: {recall_at_k(retrieved, relevant, k=5):.2f}")
    print(f"Recall@3: {recall_at_k(retrieved, relevant, k=3):.2f}")
    print(f"RR:       {reciprocal_rank(retrieved, relevant):.4f}")
