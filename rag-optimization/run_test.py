"""
Run all benchmarks with EXPLICIT device control.

The --device flag controls where EVERYTHING runs:
  --device cpu   → All computation on CPU. No GPU involved anywhere.
                   Use this to measure pure code optimization effects.
  --device cuda  → Embedding on GPU, similarity has both CPU and GPU versions tested.
                   Use this to see full performance including hardware acceleration.

This separation ensures you can tell whether speedups come from
YOUR CODE OPTIMIZATION or from GPU hardware.

Usage:
    python run_test.py --data_dir data/small --device cpu     # Pure CPU test
    python run_test.py --data_dir data/small --device cuda    # With GPU
    python run_test.py --data_dir data/small --device both    # Run both, compare

Output: prints benchmark results + saves to data_dir/test_results_{device}.json
"""
import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def force_cpu():
    """Force all PyTorch operations to CPU by hiding CUDA."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    assert not torch.cuda.is_available(), "Failed to disable CUDA"


def bench(fn, args, warmup=1, repeats=5):
    """Measure function, return mean time in ms."""
    for _ in range(warmup):
        fn(*args)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)


def run_similarity_tests(vectors, device, skip_pure_python=False):
    """
    Test all similarity implementations.

    IMPORTANT: Similarity computation is ALWAYS on CPU except for the
    explicit GPU (CuPy) version. NumPy, Cython, Numba all run on CPU
    regardless of the --device flag. The --device flag only affects
    whether we ALSO test the GPU version.
    """
    print(f"\n{'='*65}")
    print(f"SIMILARITY BENCHMARK ({len(vectors):,} vectors, D={vectors.shape[1]})")
    print(f"  All methods below run on CPU unless marked [GPU]")
    print(f"{'='*65}")

    query = vectors[0]
    corpus = vectors[1:]
    results = {}

    # ── Pure Python [CPU] ──
    if not skip_pure_python:
        from components.similarity import cosine_sim_python
        n_test = len(corpus)
        # If too slow (>30s estimated), limit to subset
        if n_test > 5000:
            small = corpus[:1000]
            mean, _ = bench(cosine_sim_python, (query, small), warmup=0, repeats=1)
            estimated = mean * (n_test / 1000)
            if estimated > 30000:
                print(f"  Pure Python [CPU]:              {estimated/1000:.0f}s estimated (skipping full run)")
                results["pure_python_cpu"] = {"ms": estimated, "device": "cpu", "note": "extrapolated"}
            else:
                print(f"  Running Pure Python on full {n_test:,} vectors...")
                mean, std = bench(cosine_sim_python, (query, corpus), warmup=0, repeats=1)
                results["pure_python_cpu"] = {"ms": mean, "device": "cpu"}
                print(f"  Pure Python [CPU]:              {mean:.2f}ms")
        else:
            mean, std = bench(cosine_sim_python, (query, corpus), warmup=0, repeats=3)
            results["pure_python_cpu"] = {"ms": mean, "device": "cpu"}
            print(f"  Pure Python [CPU]:              {mean:.2f}ms (±{std:.2f})")

    # ── NumPy [CPU] ──
    from components.similarity import cosine_sim_numpy
    mean, std = bench(cosine_sim_numpy, (query, corpus), repeats=10)
    results["numpy_cpu"] = {"ms": mean, "device": "cpu"}
    print(f"  NumPy [CPU]:                    {mean:.2f}ms (±{std:.2f})")

    # ── Cython [CPU] ──
    try:
        from optimized.similarity_cython import cosine_sim_cython
        mean, std = bench(cosine_sim_cython, (query, corpus), repeats=10)
        results["cython_cpu"] = {"ms": mean, "device": "cpu"}
        print(f"  Cython [CPU]:                   {mean:.2f}ms (±{std:.2f})")
    except ImportError:
        print("  Cython [CPU]:                   SKIPPED (not compiled)")

    # ── Numba single [CPU] ──
    try:
        from optimized.similarity_numba import (
            cosine_sim_numba, cosine_sim_numba_parallel, warmup_numba,
        )
        warmup_numba()

        mean, std = bench(cosine_sim_numba, (query, corpus), repeats=10)
        results["numba_single_cpu"] = {"ms": mean, "device": "cpu"}
        print(f"  Numba single-thread [CPU]:      {mean:.2f}ms (±{std:.2f})")

        mean, std = bench(cosine_sim_numba_parallel, (query, corpus), repeats=10)
        results["numba_parallel_cpu"] = {"ms": mean, "device": "cpu"}
        print(f"  Numba parallel [CPU]:           {mean:.2f}ms (±{std:.2f})")
    except ImportError:
        print("  Numba [CPU]:                    SKIPPED (not installed)")

    # ── GPU [GPU] — only if device allows ──
    if device in ("cuda", "both"):
        try:
            from optimized.similarity_gpu import GPUSimilarityEngine
            engine = GPUSimilarityEngine()
            engine.load_corpus(corpus)
            _ = engine.cosine_sim_gpu(query)
            mean, std = bench(engine.cosine_sim_gpu, (query,), repeats=20)
            results["gpu_cupy"] = {"ms": mean, "device": "gpu"}
            print(f"  CuPy [GPU]:                     {mean:.2f}ms (±{std:.2f})")
            engine.free()
        except (ImportError, RuntimeError) as e:
            print(f"  CuPy [GPU]:                     SKIPPED ({e})")
    else:
        print("  CuPy [GPU]:                     SKIPPED (--device cpu)")

    # ── Summary ──
    cpu_results = {k: v for k, v in results.items() if v["device"] == "cpu"}
    if cpu_results:
        baseline_ms = results.get("numpy_cpu", {}).get("ms", 1)
        print(f"\n  --- CPU-only comparison (optimization effect) ---")
        print(f"  {'Method':<30s} {'Time':>10s} {'vs NumPy':>10s}")
        print(f"  {'-'*52}")
        for name, data in results.items():
            speedup = baseline_ms / data["ms"] if data["ms"] > 0 else 0
            tag = " [GPU]" if data["device"] == "gpu" else " [CPU]"
            print(f"  {name:<30s} {data['ms']:>9.2f}ms {speedup:>9.1f}x{tag}")

    return results


def run_retrieval_tests(vectors, doc_ids, chunk_lookup, queries, device):
    """
    Test retrieval with explicit device tracking.

    CRITICAL: The retrieval latency includes BOTH embedding time and search time.
    - Embedding: runs on whatever device sentence-transformers picks (GPU if available)
    - Search: runs on CPU (NumPy/Numba) or GPU (CuPy)

    To get a FAIR comparison of search optimization, we report search_ms separately.
    """
    print(f"\n{'='*65}")
    print(f"RETRIEVAL BENCHMARK ({len(vectors):,} vectors, {len(queries)} queries)")
    print(f"  Device for embedding: {device}")
    print(f"{'='*65}")

    from components.vector_index import BruteForceIndex, IVFIndex
    from components.similarity import cosine_sim_numpy
    from components.embedder import BaselineEmbedder
    from components.retriever import Retriever
    from benchmarks.evaluate import evaluate_retriever

    # Force embedding device
    if device == "cpu":
        embedder = BaselineEmbedder()
        # Override model device to CPU
        _ = embedder.model  # trigger lazy load
        embedder.model.to("cpu")
        print("  Embedding model forced to CPU")
    else:
        embedder = BaselineEmbedder()
        print(f"  Embedding model on: {embedder.model.device}")

    # Auto cluster count
    n = len(vectors)
    n_clusters = 32 if n <= 10_000 else (64 if n <= 100_000 else 128)

    results = {}

    # ── BruteForce + NumPy [CPU search] ──
    print(f"\n  === BruteForce + NumPy [CPU search] ===")
    bf = BruteForceIndex()
    bf.build(vectors, doc_ids)
    retriever = Retriever(index=bf, embedder=embedder, sim_fn=cosine_sim_numpy)

    bf.use_precomputed_norms = False
    result_bf_legacy = evaluate_retriever(retriever, queries, chunk_lookup=chunk_lookup)
    results["BruteForce + NumPy (no norm cache)"] = {
        "recall@10": result_bf_legacy["mean_recall@k"],
        "mrr": result_bf_legacy["mean_mrr"],
        "mean_latency_ms": result_bf_legacy["mean_latency_ms"],
        "p95_latency_ms": result_bf_legacy["p95_latency_ms"],
        "search_device": "cpu",
        "embed_device": device,
    }

    bf.use_precomputed_norms = True
    result_bf_cached = evaluate_retriever(retriever, queries, chunk_lookup=chunk_lookup)
    results["BruteForce + NumPy (norm cache)"] = {
        "recall@10": result_bf_cached["mean_recall@k"],
        "mrr": result_bf_cached["mean_mrr"],
        "mean_latency_ms": result_bf_cached["mean_latency_ms"],
        "p95_latency_ms": result_bf_cached["p95_latency_ms"],
        "search_device": "cpu",
        "embed_device": device,
    }
    print(f"    Recall@10: {result_bf_cached['mean_recall@k']:.4f}")
    print(f"    MRR:       {result_bf_cached['mean_mrr']:.4f}")
    print(
        f"    Latency:   {result_bf_legacy['mean_latency_ms']:.1f}ms -> "
        f"{result_bf_cached['mean_latency_ms']:.1f}ms "
        f"({result_bf_legacy['mean_latency_ms']/result_bf_cached['mean_latency_ms']:.2f}x)"
    )

    # ── BruteForce + Numba parallel [CPU search] ──
    try:
        from optimized.similarity_numba import (
            cosine_sim_numba_parallel,
            cosine_sim_numba_parallel_precomputed,
            warmup_numba,
        )
        warmup_numba()
        print(f"\n  === BruteForce + Numba parallel [CPU search] ===")
        retriever = Retriever(index=bf, embedder=embedder, sim_fn=cosine_sim_numba_parallel)
        result = evaluate_retriever(retriever, queries, chunk_lookup=chunk_lookup)
        results["BruteForce + Numba parallel"] = {
            "recall@10": result["mean_recall@k"],
            "mrr": result["mean_mrr"],
            "mean_latency_ms": result["mean_latency_ms"],
            "p95_latency_ms": result["p95_latency_ms"],
            "search_device": "cpu",
            "embed_device": device,
        }
        print(f"    Recall@10: {result['mean_recall@k']:.4f}")
        print(f"    Latency:   {result['mean_latency_ms']:.1f}ms mean")
    except ImportError:
        pass

    # ── BruteForce + GPU [GPU search] — only if allowed ──
    if device in ("cuda", "both"):
        try:
            from optimized.similarity_gpu import GPUSimilarityEngine
            engine = GPUSimilarityEngine()
            engine.load_corpus(vectors)

            def gpu_sim_fn(query_vec, corpus_matrix):
                return engine.cosine_sim_gpu(query_vec)

            print(f"\n  === BruteForce + GPU [GPU search] ===")
            retriever = Retriever(index=bf, embedder=embedder, sim_fn=gpu_sim_fn)
            result = evaluate_retriever(retriever, queries, chunk_lookup=chunk_lookup)
            results["BruteForce + GPU"] = {
                "recall@10": result["mean_recall@k"],
                "mrr": result["mean_mrr"],
                "mean_latency_ms": result["mean_latency_ms"],
                "p95_latency_ms": result["p95_latency_ms"],
                "search_device": "gpu",
                "embed_device": device,
            }
            print(f"    Recall@10: {result['mean_recall@k']:.4f}")
            print(f"    Latency:   {result['mean_latency_ms']:.1f}ms mean")
            engine.free()
        except (ImportError, RuntimeError):
            pass

    # ── IVF + NumPy [CPU search] ──
    print(f"\n  === IVF({n_clusters},8) + NumPy [CPU search] ===")
    ivf = IVFIndex(n_clusters=n_clusters, n_probes=8)
    ivf.build(vectors, doc_ids)
    retriever = Retriever(index=ivf, embedder=embedder, sim_fn=cosine_sim_numpy)

    ivf.use_precomputed_norms = True
    ivf.use_numpy_candidate_gather = False
    result_ivf_cached_py = evaluate_retriever(retriever, queries, chunk_lookup=chunk_lookup)
    results[f"IVF({n_clusters},8) + NumPy (norm cache, py gather)"] = {
        "recall@10": result_ivf_cached_py["mean_recall@k"],
        "mrr": result_ivf_cached_py["mean_mrr"],
        "mean_latency_ms": result_ivf_cached_py["mean_latency_ms"],
        "p95_latency_ms": result_ivf_cached_py["p95_latency_ms"],
        "search_device": "cpu",
        "embed_device": device,
    }

    ivf.use_numpy_candidate_gather = True
    result_ivf_cached_np = evaluate_retriever(retriever, queries, chunk_lookup=chunk_lookup)
    results[f"IVF({n_clusters},8) + NumPy (norm cache, np gather)"] = {
        "recall@10": result_ivf_cached_np["mean_recall@k"],
        "mrr": result_ivf_cached_np["mean_mrr"],
        "mean_latency_ms": result_ivf_cached_np["mean_latency_ms"],
        "p95_latency_ms": result_ivf_cached_np["p95_latency_ms"],
        "search_device": "cpu",
        "embed_device": device,
    }
    print(f"    Recall@10: {result_ivf_cached_np['mean_recall@k']:.4f}")
    print(
        f"    Gather latency: {result_ivf_cached_py['mean_latency_ms']:.1f}ms -> "
        f"{result_ivf_cached_np['mean_latency_ms']:.1f}ms "
        f"({result_ivf_cached_py['mean_latency_ms']/result_ivf_cached_np['mean_latency_ms']:.2f}x)"
    )

    # ── IVF + NumPy + batch query embedding (throughput optimization) ──
    ivf.use_numpy_candidate_gather = True
    result_ivf_batch = evaluate_retriever(
        retriever,
        queries,
        chunk_lookup=chunk_lookup,
        use_batch_embedding=True,
    )
    results[f"IVF({n_clusters},8) + NumPy (norm cache, np gather, batch embed)"] = {
        "recall@10": result_ivf_batch["mean_recall@k"],
        "mrr": result_ivf_batch["mean_mrr"],
        "mean_latency_ms": result_ivf_batch["mean_latency_ms"],
        "p95_latency_ms": result_ivf_batch["p95_latency_ms"],
        "search_device": "cpu",
        "embed_device": device,
    }
    print(
        f"    Batch embed latency: "
        f"{result_ivf_cached_py['mean_latency_ms']:.1f}ms -> "
        f"{result_ivf_batch['mean_latency_ms']:.1f}ms "
        f"({result_ivf_cached_py['mean_latency_ms']/result_ivf_batch['mean_latency_ms']:.2f}x)"
    )

    # ── IVF + Numba parallel [CPU search] ──
    try:
        from optimized.similarity_numba import cosine_sim_numba_parallel, warmup_numba
        warmup_numba()

        ivf.use_numpy_candidate_gather = True
        ivf.use_precomputed_norms = False
        retriever_numba_ivf = Retriever(
            index=ivf,
            embedder=embedder,
            sim_fn=cosine_sim_numba_parallel,
        )

        result_ivf_numba = evaluate_retriever(
            retriever_numba_ivf,
            queries,
            chunk_lookup=chunk_lookup,
        )
        results[f"IVF({n_clusters},8) + Numba parallel (np gather, no norm cache)"] = {
            "recall@10": result_ivf_numba["mean_recall@k"],
            "mrr": result_ivf_numba["mean_mrr"],
            "mean_latency_ms": result_ivf_numba["mean_latency_ms"],
            "p95_latency_ms": result_ivf_numba["p95_latency_ms"],
            "search_device": "cpu",
            "embed_device": device,
        }

        ivf.use_precomputed_norms = True
        retriever_numba_ivf_cached = Retriever(
            index=ivf,
            embedder=embedder,
            sim_fn=cosine_sim_numba_parallel_precomputed,
        )
        result_ivf_numba_cached = evaluate_retriever(
            retriever_numba_ivf_cached,
            queries,
            chunk_lookup=chunk_lookup,
        )
        results[f"IVF({n_clusters},8) + Numba parallel (np gather, norm cache)"] = {
            "recall@10": result_ivf_numba_cached["mean_recall@k"],
            "mrr": result_ivf_numba_cached["mean_mrr"],
            "mean_latency_ms": result_ivf_numba_cached["mean_latency_ms"],
            "p95_latency_ms": result_ivf_numba_cached["p95_latency_ms"],
            "search_device": "cpu",
            "embed_device": device,
        }

        result_ivf_numba_batch = evaluate_retriever(
            retriever_numba_ivf_cached,
            queries,
            chunk_lookup=chunk_lookup,
            use_batch_embedding=True,
        )
        results[f"IVF({n_clusters},8) + Numba parallel (np gather, norm cache, batch embed)"] = {
            "recall@10": result_ivf_numba_batch["mean_recall@k"],
            "mrr": result_ivf_numba_batch["mean_mrr"],
            "mean_latency_ms": result_ivf_numba_batch["mean_latency_ms"],
            "p95_latency_ms": result_ivf_numba_batch["p95_latency_ms"],
            "search_device": "cpu",
            "embed_device": device,
        }
        print(
            f"    Numba norm-cache latency: {result_ivf_numba['mean_latency_ms']:.1f}ms -> "
            f"{result_ivf_numba_cached['mean_latency_ms']:.1f}ms "
            f"({result_ivf_numba['mean_latency_ms']/result_ivf_numba_cached['mean_latency_ms']:.2f}x)"
        )
        print(
            f"    Numba+batch latency: {result_ivf_numba_cached['mean_latency_ms']:.1f}ms -> "
            f"{result_ivf_numba_batch['mean_latency_ms']:.1f}ms "
            f"({result_ivf_numba_cached['mean_latency_ms']/result_ivf_numba_batch['mean_latency_ms']:.2f}x)"
        )
    except ImportError:
        pass

    # ── Summary ──
    config_width = max(len("Config"), *(len(name) for name in results))
    header = (
        f"  {'Config':<{config_width}s} {'Recall':>8s} {'MRR':>8s} "
        f"{'Latency':>10s} {'Search':>8s}"
    )
    print(f"\n{header}")
    print(f"  {'-' * (len(header) - 2)}")
    for name, r in results.items():
        print(
            f"  {name:<{config_width}s} {r['recall@10']:>8.4f} {r['mrr']:>8.4f} "
            f"{r['mean_latency_ms']:>9.1f}ms {r['search_device']:>8s}"
        )

    return results


def run_all(data_dir, device, skip_pure_python=False, similarity_only=False):
    """Run all tests for a given device setting."""
    # Load data
    vectors = np.load(os.path.join(data_dir, "vectors.npy"))
    with open(os.path.join(data_dir, "passages.jsonl"), encoding="utf-8") as f:
        passages = [json.loads(line) for line in f]
    with open(os.path.join(data_dir, "queries.jsonl"), encoding="utf-8") as f:
        queries = [json.loads(line) for line in f]

    doc_ids = [p["id"] for p in passages]
    chunk_lookup = {p["id"]: p["text"] for p in passages}

    print(f"\n{'#'*65}")
    print(f"# RUNNING TESTS — device: {device.upper()}")
    print(f"# Vectors: {vectors.shape}, Queries: {len(queries)}")
    print(f"{'#'*65}")

    all_results = {"device": device}

    # Similarity
    all_results["similarity"] = run_similarity_tests(
        vectors, device, skip_pure_python=skip_pure_python
    )

    # Retrieval
    if not similarity_only:
        all_results["retrieval"] = run_retrieval_tests(
            vectors, doc_ids, chunk_lookup, queries, device
        )

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run RAG benchmarks")
    parser.add_argument("--data_dir", default="data",
                        help="Data directory with built knowledge base")
    parser.add_argument("--device", choices=["cpu", "cuda", "both"], default="cpu",
                        help="cpu=pure CPU test, cuda=with GPU, both=run both and compare")
    parser.add_argument("--skip_pure_python", action="store_true",
                        help="Skip the slow Pure Python baseline")
    parser.add_argument("--similarity_only", action="store_true",
                        help="Only run similarity benchmarks")
    args = parser.parse_args()

    data_dir = args.data_dir

    # Verify data exists
    for fname in ["vectors.npy", "passages.jsonl", "queries.jsonl"]:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"ERROR: {path} not found.")
            print(f"Run: python build_knowledge_base.py --download --data_dir {data_dir}")
            sys.exit(1)

    if args.device == "both":
        # Run CPU first
        print("\n" + "=" * 65)
        print("PHASE 1: CPU-ONLY (measures pure code optimization)")
        print("=" * 65)
        force_cpu()
        cpu_results = run_all(data_dir, "cpu", args.skip_pure_python, args.similarity_only)

        # Save CPU results
        cpu_path = os.path.join(data_dir, "test_results_cpu.json")
        with open(cpu_path, "w", encoding="utf-8") as f:
            json.dump(cpu_results, f, indent=2, default=float)
        print(f"\nCPU results saved → {cpu_path}")

        # Note: can't re-enable CUDA in same process after disabling
        print("\n" + "=" * 65)
        print("PHASE 2: To run GPU tests, run separately:")
        print(f"  python run_test.py --data_dir {data_dir} --device cuda")
        print("=" * 65)

    elif args.device == "cpu":
        force_cpu()
        results = run_all(data_dir, "cpu", args.skip_pure_python, args.similarity_only)
        output_path = os.path.join(data_dir, "test_results_cpu.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\nResults saved → {output_path}")

    else:  # cuda
        results = run_all(data_dir, "cuda", args.skip_pure_python, args.similarity_only)
        output_path = os.path.join(data_dir, "test_results_cuda.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\nResults saved → {output_path}")


if __name__ == "__main__":
    main()
