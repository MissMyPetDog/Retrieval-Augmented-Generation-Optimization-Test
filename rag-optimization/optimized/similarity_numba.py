"""
Numba JIT-accelerated cosine similarity — Week 6 optimization.

Key techniques demonstrated:
  - @njit for nopython mode compilation
  - @njit(parallel=True) + prange for auto-parallelization
  - Avoiding Python object overhead in hot loops
  - First-call compilation cost vs subsequent call speed
"""
import numpy as np
from numba import njit, prange


# ──────────────────────────────────────────────
# V1: Numba single-threaded
# ──────────────────────────────────────────────

@njit(cache=True)
def cosine_sim_numba(query_vec, corpus_matrix):
    """
    Numba JIT cosine similarity — single-threaded.
    """
    n = corpus_matrix.shape[0]
    d = corpus_matrix.shape[1]
    scores = np.empty(n, dtype=np.float64)

    # Query norm
    q_norm = 0.0
    for j in range(d):
        q_norm += query_vec[j] * query_vec[j]
    q_norm = np.sqrt(q_norm)

    if q_norm == 0.0:
        return np.zeros(n, dtype=np.float64)

    for i in range(n):
        dot = 0.0
        c_norm = 0.0
        for j in range(d):
            val = corpus_matrix[i, j]
            dot += query_vec[j] * val
            c_norm += val * val
        c_norm = np.sqrt(c_norm)

        if c_norm == 0.0:
            scores[i] = 0.0
        else:
            scores[i] = dot / (q_norm * c_norm)

    return scores


# ──────────────────────────────────────────────
# V2: Numba parallel (multi-core)
# ──────────────────────────────────────────────

@njit(parallel=True, cache=True)
def cosine_sim_numba_parallel(query_vec, corpus_matrix):
    """
    Numba JIT cosine similarity — parallel across CPU cores.
    Uses prange to distribute the outer loop.
    """
    n = corpus_matrix.shape[0]
    d = corpus_matrix.shape[1]
    scores = np.empty(n, dtype=np.float64)

    # Query norm
    q_norm = 0.0
    for j in range(d):
        q_norm += query_vec[j] * query_vec[j]
    q_norm = np.sqrt(q_norm)

    if q_norm == 0.0:
        return np.zeros(n, dtype=np.float64)

    for i in prange(n):     # <-- prange for parallel execution
        dot = 0.0
        c_norm = 0.0
        for j in range(d):
            val = corpus_matrix[i, j]
            dot += query_vec[j] * val
            c_norm += val * val
        c_norm = np.sqrt(c_norm)

        if c_norm == 0.0:
            scores[i] = 0.0
        else:
            scores[i] = dot / (q_norm * c_norm)

    return scores


@njit(parallel=True, cache=True)
def cosine_sim_numba_parallel_precomputed(query_vec, corpus_matrix, corpus_norms):
    """
    Numba parallel cosine similarity using precomputed corpus norms.
    """
    n = corpus_matrix.shape[0]
    d = corpus_matrix.shape[1]
    scores = np.empty(n, dtype=np.float64)

    q_norm = 0.0
    for j in range(d):
        q_norm += query_vec[j] * query_vec[j]
    q_norm = np.sqrt(q_norm)

    if q_norm == 0.0:
        return np.zeros(n, dtype=np.float64)

    for i in prange(n):
        dot = 0.0
        for j in range(d):
            dot += query_vec[j] * corpus_matrix[i, j]

        c_norm = corpus_norms[i]
        if c_norm == 0.0:
            scores[i] = 0.0
        else:
            scores[i] = dot / (q_norm * c_norm)

    return scores


# Hint for index.search(): this function accepts precomputed norms.
cosine_sim_numba_parallel_precomputed.uses_precomputed_norms = True


# ──────────────────────────────────────────────
# Warmup helper
# ──────────────────────────────────────────────

def warmup_numba():
    """
    Trigger JIT compilation before benchmarking.
    First call is slow (compilation); subsequent calls are fast.
    """
    dummy_q = np.random.randn(384).astype(np.float32)
    dummy_c = np.random.randn(10, 384).astype(np.float32)
    dummy_norms = np.linalg.norm(dummy_c, axis=1)
    cosine_sim_numba(dummy_q, dummy_c)
    cosine_sim_numba_parallel(dummy_q, dummy_c)
    cosine_sim_numba_parallel_precomputed(dummy_q, dummy_c, dummy_norms)
    print("Numba JIT warmup complete.")


# ──────────────────────────────────────────────
# Quick benchmark
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import time

    np.random.seed(42)
    N, D = 100_000, 384
    query = np.random.randn(D).astype(np.float32)
    corpus = np.random.randn(N, D).astype(np.float32)

    # Warmup
    warmup_numba()

    # Benchmark
    for name, fn in [
        ("numba_single", cosine_sim_numba),
        ("numba_parallel", cosine_sim_numba_parallel),
    ]:
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            scores = fn(query, corpus)
            times.append(time.perf_counter() - t0)
        avg = np.mean(times)
        print(f"{name:20s}: {avg*1000:.2f}ms (N={N:,})")
