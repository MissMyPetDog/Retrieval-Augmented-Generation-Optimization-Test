"""
Benchmark runner — measures and compares all optimization stages.

Produces a structured report:
  - Per-component latency (embed, search, index build)
  - Before/after comparison table
  - Speedup curves

Usage:
    python -m benchmarks.benchmark_runner
"""
import time
import json
import sys
import os
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


@dataclass
class BenchmarkResult:
    name: str
    component: str         # "similarity", "index_build", "search", "embedding"
    n_vectors: int
    dimension: int
    times_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return np.mean(self.times_ms)

    @property
    def std_ms(self) -> float:
        return np.std(self.times_ms)

    @property
    def median_ms(self) -> float:
        return np.median(self.times_ms)

    @property
    def min_ms(self) -> float:
        return np.min(self.times_ms)

    def summary(self) -> str:
        return (f"{self.name:30s} | "
                f"mean={self.mean_ms:10.2f}ms | "
                f"std={self.std_ms:8.2f}ms | "
                f"min={self.min_ms:10.2f}ms | "
                f"N={self.n_vectors:,}")


def benchmark_function(
    fn: Callable,
    args: tuple,
    name: str,
    component: str,
    n_vectors: int,
    dimension: int,
    warmup: int = config.BENCHMARK_WARMUP,
    repeats: int = config.BENCHMARK_REPEATS,
) -> BenchmarkResult:
    """
    Generic benchmark wrapper.
    Runs warmup rounds, then measures `repeats` timed calls.
    """
    # Warmup
    for _ in range(warmup):
        fn(*args)

    # Timed runs
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        times.append(elapsed)

    result = BenchmarkResult(
        name=name,
        component=component,
        n_vectors=n_vectors,
        dimension=dimension,
        times_ms=times,
    )
    return result


# ──────────────────────────────────────────────
# Similarity benchmarks
# ──────────────────────────────────────────────

def run_similarity_benchmarks(
    n_vectors: int = 100_000,
    dimension: int = config.EMBEDDING_DIM,
) -> list[BenchmarkResult]:
    """Benchmark all similarity implementations."""
    print(f"\n{'='*60}")
    print(f"SIMILARITY BENCHMARKS (N={n_vectors:,}, D={dimension})")
    print(f"{'='*60}")

    np.random.seed(42)
    query = np.random.randn(dimension).astype(np.float32)
    corpus = np.random.randn(n_vectors, dimension).astype(np.float32)
    results = []

    # ── Pure Python (on small subset) ──
    from components.similarity import cosine_sim_python
    small_corpus = corpus[:1000]
    r = benchmark_function(
        cosine_sim_python, (query, small_corpus),
        name="pure_python",
        component="similarity",
        n_vectors=1000,
        dimension=dimension,
        repeats=3,
    )
    print(r.summary())
    # Extrapolate to full size
    print(f"  (estimated for {n_vectors:,}: {r.mean_ms * (n_vectors / 1000):.0f}ms)")
    results.append(r)

    # ── NumPy ──
    from components.similarity import cosine_sim_numpy
    r = benchmark_function(
        cosine_sim_numpy, (query, corpus),
        name="numpy",
        component="similarity",
        n_vectors=n_vectors,
        dimension=dimension,
    )
    print(r.summary())
    results.append(r)

    # ── Numba (if available) ──
    try:
        from optimized.similarity_numba import (
            cosine_sim_numba, cosine_sim_numba_parallel, warmup_numba,
        )
        warmup_numba()

        r = benchmark_function(
            cosine_sim_numba, (query, corpus),
            name="numba_single",
            component="similarity",
            n_vectors=n_vectors,
            dimension=dimension,
        )
        print(r.summary())
        results.append(r)

        r = benchmark_function(
            cosine_sim_numba_parallel, (query, corpus),
            name="numba_parallel",
            component="similarity",
            n_vectors=n_vectors,
            dimension=dimension,
        )
        print(r.summary())
        results.append(r)
    except ImportError:
        print("  [SKIP] Numba not installed")

    # ── Cython (if compiled) ──
    try:
        from optimized.similarity_cython import cosine_sim_cython
        r = benchmark_function(
            cosine_sim_cython, (query, corpus),
            name="cython",
            component="similarity",
            n_vectors=n_vectors,
            dimension=dimension,
        )
        print(r.summary())
        results.append(r)
    except ImportError:
        print("  [SKIP] Cython extension not compiled")

    # ── GPU (if available) ──
    try:
        from optimized.similarity_gpu import GPUSimilarityEngine
        engine = GPUSimilarityEngine()
        engine.load_corpus(corpus)
        # Warmup
        _ = engine.cosine_sim_gpu(query)

        r = benchmark_function(
            engine.cosine_sim_gpu, (query,),
            name="gpu_cupy",
            component="similarity",
            n_vectors=n_vectors,
            dimension=dimension,
        )
        print(r.summary())
        results.append(r)
        engine.free()
    except (ImportError, RuntimeError):
        print("  [SKIP] CuPy / GPU not available")

    return results


# ──────────────────────────────────────────────
# Search benchmarks (end-to-end retrieval)
# ──────────────────────────────────────────────

def run_search_benchmarks(
    n_vectors: int = 100_000,
    dimension: int = config.EMBEDDING_DIM,
) -> list[BenchmarkResult]:
    """Benchmark brute-force vs IVF search."""
    print(f"\n{'='*60}")
    print(f"SEARCH BENCHMARKS (N={n_vectors:,}, D={dimension})")
    print(f"{'='*60}")

    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    doc_ids = [f"doc_{i}" for i in range(n_vectors)]
    query = np.random.randn(dimension).astype(np.float32)
    results = []

    from components.vector_index import BruteForceIndex, IVFIndex
    from components.similarity import cosine_sim_numpy

    # ── BruteForce ──
    bf = BruteForceIndex()
    bf.build(vectors, doc_ids)
    r = benchmark_function(
        bf.search, (query, 10, cosine_sim_numpy),
        name="bruteforce_numpy",
        component="search",
        n_vectors=n_vectors,
        dimension=dimension,
    )
    print(r.summary())
    results.append(r)

    # ── IVF ──
    ivf = IVFIndex(n_clusters=config.IVF_NUM_CLUSTERS, n_probes=config.IVF_NUM_PROBES)
    ivf.build(vectors, doc_ids)
    r = benchmark_function(
        ivf.search, (query, 10, None, cosine_sim_numpy),
        name=f"ivf_numpy_probes={config.IVF_NUM_PROBES}",
        component="search",
        n_vectors=n_vectors,
        dimension=dimension,
    )
    print(r.summary())
    results.append(r)

    return results


# ──────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────

def generate_report(all_results: list[BenchmarkResult], output_path: str = "benchmark_report.json"):
    """Save all results to JSON for later visualization."""
    data = []
    for r in all_results:
        d = {
            "name": r.name,
            "component": r.component,
            "n_vectors": r.n_vectors,
            "dimension": r.dimension,
            "mean_ms": r.mean_ms,
            "std_ms": r.std_ms,
            "median_ms": r.median_ms,
            "min_ms": r.min_ms,
        }
        data.append(d)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nReport saved to {output_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    all_results = []
    all_results.extend(run_similarity_benchmarks(n_vectors=100_000))
    all_results.extend(run_search_benchmarks(n_vectors=100_000))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        print(r.summary())

    generate_report(all_results)
