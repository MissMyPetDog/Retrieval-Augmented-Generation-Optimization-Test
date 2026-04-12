"""
Cosine similarity computation — baseline implementations.

This is the PRIMARY optimization target for the project.
Each function has the same interface so they can be swapped freely.

Versions:
  - cosine_sim_python()       : Pure Python loops         (baseline)
  - cosine_sim_numpy()        : NumPy vectorized          (baseline+)
  - cosine_sim_cython()       : Cython typed memoryview    (Week 5)   → optimized/
  - cosine_sim_numba()        : Numba @jit                (Week 6)   → optimized/
  - cosine_sim_gpu()          : CuPy on GPU               (Week 12)  → optimized/

All functions follow:
    f(query_vec: np.ndarray, corpus_matrix: np.ndarray) -> np.ndarray
    - query_vec:     shape (D,)
    - corpus_matrix: shape (N, D)
    - returns:       shape (N,)  similarity scores
"""
import math
import time
import numpy as np


# ──────────────────────────────────────────────
# V0: Pure Python (intentionally slow baseline)
# ──────────────────────────────────────────────

def cosine_sim_python(query_vec: np.ndarray, corpus_matrix: np.ndarray) -> np.ndarray:
    """
    Pure Python cosine similarity. No NumPy ops in the hot loop.
    This is the slowest version — exists only to show optimization gains.
    """
    query = query_vec.tolist()
    n, d = corpus_matrix.shape
    scores = []

    # Precompute query norm
    q_norm = math.sqrt(sum(x * x for x in query))
    if q_norm == 0:
        return np.zeros(n)

    for i in range(n):
        dot = 0.0
        c_norm = 0.0
        for j in range(d):
            val = corpus_matrix[i, j]
            dot += query[j] * val
            c_norm += val * val
        c_norm = math.sqrt(c_norm)
        if c_norm == 0:
            scores.append(0.0)
        else:
            scores.append(dot / (q_norm * c_norm))

    return np.array(scores)


# ──────────────────────────────────────────────
# V0.5: NumPy vectorized (strong baseline)
# ──────────────────────────────────────────────

def cosine_sim_numpy(query_vec: np.ndarray, corpus_matrix: np.ndarray) -> np.ndarray:
    """
    NumPy-vectorized cosine similarity. Already fast due to BLAS,
    but still runs on a single CPU core.
    """
    # dot products: (N, D) @ (D,) -> (N,)
    dots = corpus_matrix @ query_vec

    # norms
    q_norm = np.linalg.norm(query_vec)
    c_norms = np.linalg.norm(corpus_matrix, axis=1)

    # avoid division by zero
    denom = q_norm * c_norms
    denom = np.where(denom == 0, 1e-10, denom)

    return dots / denom


# ──────────────────────────────────────────────
# Top-K selection
# ──────────────────────────────────────────────

def top_k_python(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    """
    Pure Python top-K selection using a min-heap.
    Returns list of (index, score) sorted descending.
    """
    import heapq
    # heapq.nlargest is O(N log K), better than full sort for small K
    top = heapq.nlargest(k, range(len(scores)), key=lambda i: scores[i])
    return [(i, float(scores[i])) for i in top]


def top_k_numpy(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    """
    NumPy top-K using argpartition — O(N) average.
    """
    if k >= len(scores):
        indices = np.argsort(scores)[::-1]
    else:
        # argpartition is O(N), then sort only the top K
        indices = np.argpartition(scores, -k)[-k:]
        indices = indices[np.argsort(scores[indices])[::-1]]

    return [(int(i), float(scores[i])) for i in indices]


# ──────────────────────────────────────────────
# Quick benchmark
# ──────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    D = 384
    N = 50_000

    query = np.random.randn(D).astype(np.float32)
    corpus = np.random.randn(N, D).astype(np.float32)

    # Benchmark NumPy version
    t0 = time.perf_counter()
    scores_np = cosine_sim_numpy(query, corpus)
    t_numpy = time.perf_counter() - t0
    print(f"NumPy   ({N:,} vectors, D={D}): {t_numpy:.4f}s")

    # Benchmark pure Python (only on small subset)
    small_corpus = corpus[:1000]
    t0 = time.perf_counter()
    scores_py = cosine_sim_python(query, small_corpus)
    t_python = time.perf_counter() - t0
    print(f"Python  ({1000} vectors, D={D}): {t_python:.4f}s")
    print(f"Estimated Python for {N:,}: {t_python * (N / 1000):.1f}s")
    print(f"Speedup (NumPy vs Python): ~{t_python * (N / 1000) / t_numpy:.0f}x")

    # Verify correctness
    diff = np.max(np.abs(scores_np[:1000] - scores_py))
    print(f"Max difference: {diff:.2e}")

    # Top-K
    top = top_k_numpy(scores_np, 10)
    print(f"\nTop-10 results: {[(i, f'{s:.4f}') for i, s in top]}")
