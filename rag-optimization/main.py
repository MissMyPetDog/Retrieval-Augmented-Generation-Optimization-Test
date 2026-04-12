"""
Main RAG pipeline — runs baseline or optimized versions end-to-end.

Usage:
    python main.py --mode baseline      # Run with pure Python/NumPy
    python main.py --mode optimized     # Run with all optimizations
    python main.py --mode compare       # Run both and compare
    python main.py --mode demo          # Interactive query demo
"""
import os
import sys
import time
import argparse
import json
import numpy as np

import config


def build_pipeline(mode: str = "baseline"):
    """
    Build the full RAG pipeline in either baseline or optimized mode.
    Returns (retriever, chunks, build_timings).
    """
    from components.preprocessor import load_passages, process_passages
    from components.embedder import BaselineEmbedder
    from components.vector_index import BruteForceIndex, IVFIndex
    from components.similarity import cosine_sim_python, cosine_sim_numpy
    from components.retriever import Retriever

    timings = {}

    # ── Step 1: Load and preprocess ──
    print("\n[1/4] Loading and preprocessing passages...")
    t0 = time.perf_counter()
    passages = load_passages()
    chunks = process_passages(passages)
    timings["preprocess_s"] = time.perf_counter() - t0
    print(f"  {len(passages)} passages -> {len(chunks)} chunks "
          f"({timings['preprocess_s']:.2f}s)")

    texts = [c["text"] for c in chunks]
    chunk_ids = [c["id"] for c in chunks]

    # ── Step 2: Generate embeddings ──
    print("\n[2/4] Generating embeddings...")
    t0 = time.perf_counter()

    if mode == "optimized":
        try:
            from optimized.async_embedder import ConcurrentEmbedder
            embedder = ConcurrentEmbedder(n_workers=4)
            vectors = embedder.embed_texts(texts)
            embedder_for_retriever = BaselineEmbedder()  # For single query encoding
        except ImportError:
            print("  [FALLBACK] Concurrent embedder not available, using baseline")
            embedder_for_retriever = BaselineEmbedder()
            vectors = embedder_for_retriever.embed_texts(texts)
    else:
        embedder_for_retriever = BaselineEmbedder()
        vectors = embedder_for_retriever.embed_texts(texts)

    timings["embedding_s"] = time.perf_counter() - t0
    print(f"  Embeddings: {vectors.shape} ({timings['embedding_s']:.2f}s)")

    # ── Step 3: Build index ──
    print("\n[3/4] Building vector index...")
    t0 = time.perf_counter()

    if mode == "optimized":
        # Use IVF with parallel building
        index = IVFIndex(
            n_clusters=config.IVF_NUM_CLUSTERS,
            n_probes=config.IVF_NUM_PROBES,
        )
        try:
            from optimized.parallel_indexer import ParallelIVFBuilder
            builder = ParallelIVFBuilder(n_clusters=config.IVF_NUM_CLUSTERS)
            builder.build_parallel(vectors, chunk_ids, index)
        except ImportError:
            print("  [FALLBACK] Parallel builder not available, using sequential")
            index.build(vectors, chunk_ids)
    else:
        # Baseline: brute-force index
        index = BruteForceIndex()
        index.build(vectors, chunk_ids)

    timings["index_build_s"] = time.perf_counter() - t0
    print(f"  Index built ({timings['index_build_s']:.2f}s)")

    # ── Step 4: Create retriever ──
    print("\n[4/4] Assembling retriever...")

    if mode == "optimized":
        # Try to use the fastest available similarity function
        sim_fn = cosine_sim_numpy  # default fallback
        try:
            from optimized.similarity_numba import cosine_sim_numba_parallel, warmup_numba
            warmup_numba()
            sim_fn = cosine_sim_numba_parallel
            print("  Using: Numba parallel similarity")
        except ImportError:
            print("  Using: NumPy similarity (Numba not available)")
    else:
        sim_fn = cosine_sim_numpy
        print("  Using: NumPy similarity (baseline)")

    retriever = Retriever(
        index=index,
        embedder=embedder_for_retriever,
        sim_fn=sim_fn,
    )

    return retriever, chunks, timings


def run_evaluation(retriever, chunks, label: str = ""):
    """Run evaluation on the query set."""
    from components.preprocessor import load_queries
    from benchmarks.evaluate import evaluate_retriever

    queries = load_queries()
    if not queries:
        print("No queries found. Run data/download_data.py first.")
        return None

    # Build chunk_id -> text lookup for passage-level matching
    chunk_lookup = {c["id"]: c["text"] for c in chunks}

    print(f"\nEvaluating {label} on {len(queries)} queries...")
    results = evaluate_retriever(retriever, queries, chunk_lookup=chunk_lookup)

    print(f"\n{'─'*40}")
    print(f"  {label} Results:")
    print(f"  Recall@{results['k']}: {results['mean_recall@k']:.4f}")
    print(f"  MRR:        {results['mean_mrr']:.4f}")
    print(f"  Latency:    {results['mean_latency_ms']:.1f}ms (mean), "
          f"{results['p95_latency_ms']:.1f}ms (p95)")
    print(f"{'─'*40}")

    return results


def interactive_demo(retriever, chunks):
    """Interactive query loop for demo."""
    from components.generator import generate_answer

    chunk_lookup = {c["id"]: c["text"] for c in chunks}

    print("\n" + "=" * 60)
    print("RAG Interactive Demo")
    print("Type a question, or 'quit' to exit.")
    print("=" * 60)

    while True:
        query = input("\nQuery> ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        result = retriever.retrieve(query)

        print(f"\n  Search time: {result['timings']['search_ms']:.1f}ms")
        print(f"  Top {len(result['results'])} results:")
        for i, (doc_id, score) in enumerate(result["results"][:5]):
            text = chunk_lookup.get(doc_id, "[text not found]")
            preview = text[:120] + "..." if len(text) > 120 else text
            print(f"    [{i+1}] (score={score:.4f}) {preview}")

        # Generate answer
        contexts = [chunk_lookup.get(did, "") for did, _ in result["results"][:3]]
        answer = generate_answer(query, contexts)
        print(f"\n  Answer: {answer}")


def main():
    parser = argparse.ArgumentParser(description="RAG Optimization Pipeline")
    parser.add_argument(
        "--mode",
        choices=["baseline", "optimized", "compare", "demo"],
        default="baseline",
        help="Pipeline mode",
    )
    args = parser.parse_args()

    if args.mode == "compare":
        # Run both and compare
        print("\n" + "=" * 60)
        print("BASELINE PIPELINE")
        print("=" * 60)
        ret_base, chunks_base, t_base = build_pipeline("baseline")
        eval_base = run_evaluation(ret_base, chunks_base, "Baseline")

        print("\n" + "=" * 60)
        print("OPTIMIZED PIPELINE")
        print("=" * 60)
        ret_opt, chunks_opt, t_opt = build_pipeline("optimized")
        eval_opt = run_evaluation(ret_opt, chunks_opt, "Optimized")

        # Compare build timings
        print("\n" + "=" * 60)
        print("BUILD TIME COMPARISON")
        print("=" * 60)
        for key in t_base:
            b = t_base[key]
            o = t_opt[key]
            speedup = b / o if o > 0 else float("inf")
            print(f"  {key:<20s}: {b:8.2f}s -> {o:8.2f}s (speedup: {speedup:.2f}x)")

        if eval_base and eval_opt:
            from benchmarks.evaluate import compare_results
            compare_results(eval_base, eval_opt)

    elif args.mode == "demo":
        retriever, chunks, _ = build_pipeline("optimized")
        interactive_demo(retriever, chunks)

    else:
        retriever, chunks, timings = build_pipeline(args.mode)
        run_evaluation(retriever, chunks, args.mode.capitalize())

        # Save timings
        out_path = f"timings_{args.mode}.json"
        with open(out_path, "w") as f:
            json.dump(timings, f, indent=2)
        print(f"\nTimings saved to {out_path}")


if __name__ == "__main__":
    main()
