"""
Configuration 3: Fully optimized end-to-end pipeline.

Everything from Steps 1-6:
  [Step 1] IVF parameter tuning   : n_clusters=64, n_probes=8 (sweet spot)
  [Step 2] Numba K-Means          : @njit accumulate_sums, cached v_norms
  [Step 3] K-Means++ init         : distance-weighted seeding
  [Step 6] Friend's query-path    : norm cache ON, np gather ON (variant C)
  [Step 4] LLM streaming          : generate_stream() with TTFT measurement
  [Step 5] Pipelined RAG          : two thread pools (retrieval || generation)

Pipeline for N queries:
  Retrieval pool (4 workers)         Generation pool (8 workers)
  ---------------------------        ---------------------------
    embed + search query i  --submit-->  generate_stream for query i
    embed + search query i+1                (streaming, overlaps
    ...                                      with retrieval of i+2)

Reports both per-query means (amortized) and true batch total time.
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
from concurrent.futures import ThreadPoolExecutor


def run(dataset: str = "medium", n_queries: int = 8, k: int = 3,
        n_clusters: int = 64, n_probes: int = 8,
        n_embed_workers: int = 4, n_gen_workers: int = 8,
        max_tokens: int = 128) -> dict:
    from components.embedder import LocalEmbedder
    from components.similarity import cosine_sim_numpy
    from optimized.kmeans_numba import IVFIndexNumbaPP, warmup_kmeans_numba

    print(f"\n[Config 3: fully optimized] "
          f"n_clusters={n_clusters}, n_probes={n_probes}, "
          f"embed_workers={n_embed_workers}, gen_workers={n_gen_workers}")

    vectors, chunks, queries, data_dir = load_knowledge_base(dataset)
    test_queries     = get_test_queries(queries, n=n_queries)
    chunk_to_passage = build_chunk_to_passage_text(chunks, data_dir)
    chunk_by_id      = {c["id"]: c["text"] for c in chunks}

    # --- Build index with Numba K-Means + K-Means++ ---
    print("Warming up Numba JIT...")
    warmup_kmeans_numba()
    print(f"Building IVFIndexNumbaPP({n_clusters} clusters)...")
    t0 = time.perf_counter()
    ivf = IVFIndexNumbaPP(n_clusters=n_clusters, n_probes=n_probes, kmeans_iters=20)
    ivf.build(vectors.astype(np.float32), [c["id"] for c in chunks])
    build_ms = (time.perf_counter() - t0) * 1000

    embedder = LocalEmbedder(device="cpu")
    warmup_embedder(embedder)     # trigger lazy model load BEFORE the timed pipelined run
    gen = make_kong_generator(max_tokens=max_tokens)

    # --- Pipelined run ---
    # Each retrieval worker submits its result directly to the gen pool
    # so retrieval of query i+1 overlaps with generation of query i.
    print(f"\nRunning {n_queries} queries PIPELINED (streaming, norm cache ON, np gather ON)...")
    per_query_timings = []

    def retrieve_and_submit(q_obj, gen_pool):
        # Runs on retrieval worker
        t_embed_start = time.perf_counter()
        qv = embedder.embed_query(q_obj["text"])
        t_embed_done = time.perf_counter()

        results = ivf.search(
            qv, k=k, n_probes=n_probes, sim_fn=cosine_sim_numpy,
            use_precomputed_norms=True,
            use_numpy_candidate_gather=True,
        )
        t_search_done = time.perf_counter()

        retrieved_ids = [doc_id for doc_id, _ in results]
        recall = compute_recall(retrieved_ids, q_obj["relevant_passages"], chunk_to_passage)
        contexts = [chunk_by_id.get(did, "") for did, _ in results]

        # Hand off to gen pool; return (timings, gen_future)
        gen_future = gen_pool.submit(gen.generate_stream, q_obj["text"], contexts)
        timings = {
            "query":     q_obj["text"][:60],
            "embed_ms":  (t_embed_done  - t_embed_start) * 1000,
            "search_ms": (t_search_done - t_embed_done)  * 1000,
            "recall@k": recall,
        }
        return timings, gen_future

    batch_t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_embed_workers) as retrieve_pool, \
         ThreadPoolExecutor(max_workers=n_gen_workers) as gen_pool:
        retrieve_futures = [
            retrieve_pool.submit(retrieve_and_submit, q, gen_pool)
            for q in test_queries
        ]
        # Each retrieve_future resolves to (timings_dict, gen_future)
        timings_and_gen_futures = [rf.result() for rf in retrieve_futures]

        # Wait for gens, record gen timing per query
        for t, gen_fut in timings_and_gen_futures:
            t_gen_start = time.perf_counter()
            gen_result = gen_fut.result()
            t["gen_ms"]   = gen_result["total_ms"]
            t["ttft_ms"]  = gen_result["ttft_ms"]
            t["total_ms"] = t["embed_ms"] + t["search_ms"] + t["gen_ms"]
            per_query_timings.append(t)
    batch_ms = (time.perf_counter() - batch_t0) * 1000

    # Print per query
    for i, t in enumerate(per_query_timings):
        print(f"  [{i+1}/{n_queries}] '{t['query'][:40]}...' "
              f"embed={t['embed_ms']:5.0f} search={t['search_ms']:5.2f} "
              f"TTFT={t['ttft_ms']:5.0f} gen_total={t['gen_ms']:5.0f}  "
              f"recall={t['recall@k']:.3f}")

    # Summary means (per-query metrics are "isolated" per call,
    # but batch_total_ms reflects the overlap)
    result = {
        "name":          f"Config 3: Fully optimized (IVF({n_clusters}, np={n_probes}) + pipelined stream)",
        "description":   ("All 6 optimization steps: Numba K-Means + K-Means++ + norm cache "
                          "+ np gather + LLM streaming + dual-pool pipelined serving."),
        "config": {
            "index":                       f"IVFIndexNumbaPP(n_clusters={n_clusters})",
            "n_probes":                    n_probes,
            "sim_fn":                      "cosine_sim_numpy",
            "kmeans_impl":                 "Numba + K-Means++ init",
            "use_precomputed_norms":       True,
            "use_numpy_candidate_gather":  True,
            "generator":                   "BaselineGenerator.generate_stream (streaming)",
            "architecture":                f"pipelined (retrieval={n_embed_workers} || gen={n_gen_workers})",
            "n_retrieved":                 k,
            "max_tokens":                  max_tokens,
        },
        "build_time_ms":   build_ms,
        "per_query":       per_query_timings,
        "per_query_mean":  {
            "embed_ms":  mean([p["embed_ms"]  for p in per_query_timings]),
            "search_ms": mean([p["search_ms"] for p in per_query_timings]),
            "gen_ms":    mean([p["gen_ms"]    for p in per_query_timings]),
            "ttft_ms":   mean([p["ttft_ms"]   for p in per_query_timings]),
            "total_ms":  mean([p["total_ms"]  for p in per_query_timings]),
            "recall@k": mean([p["recall@k"] for p in per_query_timings]),
        },
        "batch_total_ms":  batch_ms,
        "n_queries":       n_queries,
    }

    print_summary(result)
    if "ttft_ms" in result["per_query_mean"]:
        print(f"  Mean TTFT:       {result['per_query_mean']['ttft_ms']:8.1f} ms")
    print(f"  Batch vs per-q*N: {batch_ms:.0f} vs {result['per_query_mean']['total_ms']*n_queries:.0f} ms "
          f"-> pipeline savings: {1 - batch_ms/(result['per_query_mean']['total_ms']*n_queries):.0%}")

    return result


if __name__ == "__main__":
    result = run()
    save_result(result, "03_optimized")