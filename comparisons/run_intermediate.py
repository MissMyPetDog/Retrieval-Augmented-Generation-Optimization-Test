"""
Configuration 2: IVF index, but no deep optimizations.

Purpose: represent what a reasonable engineer would write after realizing
BruteForce is slow -- switch to IVF with simple defaults, but don't know
(or don't use) Numba, norm caching, streaming, etc.

Pipeline per query:
  query text -> LocalEmbedder (CPU)
             -> IVFIndex(n_clusters=32, n_probes=4).search() with plain NumPy
             -> BaselineGenerator.generate() (non-streaming)
             -> done

IVF itself is a structural optimization (skip most vectors), but we hold
everything else at baseline:
  - random K-Means init (no K-Means++)
  - pure-NumPy K-Means (no Numba)
  - norm cache explicitly DISABLED
  - np-gather explicitly DISABLED
  - non-streaming generation
  - serial per-query execution
"""
from __future__ import annotations

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
        n_clusters: int = 32, n_probes: int = 4,
        max_tokens: int = 128) -> dict:
    from components.embedder import LocalEmbedder
    from components.vector_index import IVFIndex
    from components.similarity import cosine_sim_numpy

    print(f"\n[Config 2: IVF default, no deep optim] "
          f"dataset={dataset}, n_clusters={n_clusters}, n_probes={n_probes}")

    vectors, chunks, queries, data_dir = load_knowledge_base(dataset)
    test_queries     = get_test_queries(queries, n=n_queries)
    chunk_to_passage = build_chunk_to_passage_text(chunks, data_dir)
    chunk_by_id      = {c["id"]: c["text"] for c in chunks}

    # --- Build IVF (timed) ---
    # Use the baseline IVFIndex (NOT the Numba subclass), with default random init.
    print(f"Building IVFIndex({n_clusters} clusters)...")
    t0 = time.perf_counter()
    ivf = IVFIndex(n_clusters=n_clusters, n_probes=n_probes, kmeans_iters=20)
    ivf.build(vectors.astype(np.float32), [c["id"] for c in chunks])
    build_ms = (time.perf_counter() - t0) * 1000

    embedder = LocalEmbedder(device="cpu")
    warmup_embedder(embedder)     # trigger lazy model load BEFORE timing
    gen = make_kong_generator(max_tokens=max_tokens)

    # --- Run queries serially ---
    print(f"\nRunning {n_queries} queries end-to-end (serial, non-streaming, "
          f"norm cache OFF, np gather OFF)...")
    per_query = []
    batch_t0 = time.perf_counter()
    for i, q in enumerate(test_queries):
        # Embed
        t0 = time.perf_counter()
        qv = embedder.embed_query(q["text"])
        embed_ms = (time.perf_counter() - t0) * 1000

        # Search -- explicitly disable friend's query-path optimizations
        t0 = time.perf_counter()
        results = ivf.search(
            qv, k=k, n_probes=n_probes, sim_fn=cosine_sim_numpy,
            use_precomputed_norms=False,
            use_numpy_candidate_gather=False,
        )
        search_ms = (time.perf_counter() - t0) * 1000

        retrieved_ids = [doc_id for doc_id, _ in results]
        recall = compute_recall(retrieved_ids, q["relevant_passages"], chunk_to_passage)

        # Generate (non-streaming)
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
            "recall@k": recall,
        })
        print(f"  [{i+1}/{n_queries}] '{q['text'][:40]}...' "
              f"embed={embed_ms:5.0f} search={search_ms:5.1f} "
              f"gen={gen_ms:5.0f} total={total_ms:5.0f}  recall={recall:.3f}")
    batch_ms = (time.perf_counter() - batch_t0) * 1000

    result = {
        "name":          f"Config 2: IVF({n_clusters}, np={n_probes}) default",
        "description":   ("IVF with default parameters (random K-Means init, pure NumPy), "
                          "no norm cache, no np gather, non-streaming serial gen."),
        "config": {
            "index":                       f"IVFIndex(n_clusters={n_clusters})",
            "n_probes":                    n_probes,
            "sim_fn":                      "cosine_sim_numpy",
            "kmeans_impl":                 "baseline (random init, pure NumPy)",
            "use_precomputed_norms":       False,
            "use_numpy_candidate_gather":  False,
            "generator":                   "BaselineGenerator (non-streaming)",
            "architecture":                "serial per-query",
            "n_retrieved":                 k,
            "max_tokens":                  max_tokens,
        },
        "build_time_ms":   build_ms,
        "per_query":       per_query,
        "per_query_mean":  {
            "embed_ms":  mean([p["embed_ms"]  for p in per_query]),
            "search_ms": mean([p["search_ms"] for p in per_query]),
            "gen_ms":    mean([p["gen_ms"]    for p in per_query]),
            "total_ms":  mean([p["total_ms"]  for p in per_query]),
            "recall@k": mean([p["recall@k"] for p in per_query]),
        },
        "batch_total_ms":  batch_ms,
        "n_queries":       n_queries,
    }

    print_summary(result)
    return result


if __name__ == "__main__":
    result = run()
    save_result(result, "02_intermediate")