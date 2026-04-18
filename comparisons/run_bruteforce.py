"""
Configuration 1: Pure BruteForce, ZERO optimizations.

Purpose: the "unoptimized reference" -- what the system looks like straight
out of the box. Every query scans the ENTIRE corpus with plain NumPy cosine
similarity, no caching, no concurrency, no streaming.

Pipeline per query:
  query text -> LocalEmbedder (CPU)
             -> BruteForceIndex.search() with cosine_sim_numpy
             -> BaselineGenerator.generate() (non-streaming)
             -> done

No overlap between queries. Everything serial. This is the slow end of the
spectrum -- if you can't beat this, your optimization isn't working.
"""
from __future__ import annotations

# Path + CPU-only setup MUST come before torch/numba imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    setup_cpu_only, load_knowledge_base, get_test_queries,
    build_chunk_to_passage_text, make_kong_generator,
    compute_recall, mean, save_result, print_summary, warmup_embedder,
)
setup_cpu_only(verbose=True)

import time
import numpy as np


def run(dataset: str = "medium", n_queries: int = 8, k: int = 3,
        max_tokens: int = 128) -> dict:
    # Heavy imports AFTER setup_cpu_only
    from components.embedder import LocalEmbedder
    from components.vector_index import BruteForceIndex
    from components.similarity import cosine_sim_numpy

    print(f"\n[Config 1: BruteForce, no optim] dataset={dataset}, n_queries={n_queries}")

    vectors, chunks, queries, data_dir = load_knowledge_base(dataset)
    test_queries     = get_test_queries(queries, n=n_queries)
    chunk_to_passage = build_chunk_to_passage_text(chunks, data_dir)
    chunk_by_id      = {c["id"]: c["text"] for c in chunks}

    # --- Build index (timed) ---
    print("Building BruteForceIndex...")
    t0 = time.perf_counter()
    bf = BruteForceIndex()
    bf.build(vectors.astype(np.float32), [c["id"] for c in chunks])
    build_ms = (time.perf_counter() - t0) * 1000

    # --- Set up services (not timed) ---
    embedder = LocalEmbedder(device="cpu")
    warmup_embedder(embedder)     # trigger lazy model load BEFORE timing
    gen = make_kong_generator(max_tokens=max_tokens)

    # --- Run queries serially, time each stage ---
    print(f"\nRunning {n_queries} queries end-to-end (serial, non-streaming)...")
    per_query = []
    batch_t0 = time.perf_counter()
    for i, q in enumerate(test_queries):
        # Stage 1: embed
        t0 = time.perf_counter()
        qv = embedder.embed_query(q["text"])
        embed_ms = (time.perf_counter() - t0) * 1000

        # Stage 2: search (norm cache EXPLICITLY disabled for the honest baseline)
        t0 = time.perf_counter()
        results = bf.search(qv, k=k, sim_fn=cosine_sim_numpy,
                            use_precomputed_norms=False)
        search_ms = (time.perf_counter() - t0) * 1000

        retrieved_ids = [doc_id for doc_id, _ in results]
        recall = compute_recall(retrieved_ids, q["relevant_passages"], chunk_to_passage)

        # Stage 3: gen (non-streaming)
        t0 = time.perf_counter()
        contexts = [chunk_by_id.get(did, "") for did, _ in results]
        _ = gen.generate(q["text"], contexts)
        gen_ms = (time.perf_counter() - t0) * 1000

        total_ms = embed_ms + search_ms + gen_ms
        per_query.append({
            "query":     q["text"][:60],
            "embed_ms":  embed_ms,
            "search_ms": search_ms,
            "gen_ms":    gen_ms,
            "total_ms":  total_ms,
            "recall@k":  recall,
        })
        print(f"  [{i+1}/{n_queries}] '{q['text'][:40]}...' "
              f"embed={embed_ms:5.0f} search={search_ms:5.1f} "
              f"gen={gen_ms:5.0f} total={total_ms:5.0f}  recall={recall:.3f}")
    batch_ms = (time.perf_counter() - batch_t0) * 1000

    result = {
        "name":          "Config 1: BruteForce (no optim)",
        "description":   "Pure BruteForce scan, no norm cache, serial execution, non-streaming gen.",
        "config": {
            "index":              "BruteForceIndex",
            "sim_fn":             "cosine_sim_numpy",
            "use_precomputed_norms": False,
            "generator":          "BaselineGenerator (non-streaming)",
            "architecture":       "serial per-query",
            "n_retrieved":        k,
            "max_tokens":         max_tokens,
        },
        "build_time_ms":   build_ms,
        "per_query":       per_query,
        "per_query_mean":  {
            "embed_ms":  mean([p["embed_ms"]  for p in per_query]),
            "search_ms": mean([p["search_ms"] for p in per_query]),
            "gen_ms":    mean([p["gen_ms"]    for p in per_query]),
            "total_ms":  mean([p["total_ms"]  for p in per_query]),
            "recall@k":  mean([p["recall@k"]  for p in per_query]),
        },
        "batch_total_ms":  batch_ms,
        "n_queries":       n_queries,
    }

    print_summary(result)
    return result


if __name__ == "__main__":
    result = run()
    save_result(result, "01_bruteforce")