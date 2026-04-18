"""
Numba-accelerated K-Means for IVF index building.

Week 6 optimization (Numba JIT).

Key differences vs baseline `components/vector_index.py::IVFIndex._kmeans`:

  1. Vector norms are hoisted OUTSIDE the iteration loop (they don't change).
     Baseline recomputes np.linalg.norm(vectors) every iteration -- wasteful.

  2. The update step is rewritten as a Numba-compiled single pass over vectors,
     accumulating per-cluster sums + counts. Baseline uses a Python loop over
     clusters with boolean indexing (vectors[mask].mean()) per cluster -- each
     iteration allocates a new array and Python dispatches 64 sub-operations.

Same inputs, same outputs, same random seed -> produces the same centroids as
the baseline (modulo floating-point ordering), just faster.
"""
import time

import numpy as np
from numba import njit

from components.vector_index import IVFIndex


# =====================================================================
# Numba kernel: per-cluster sum + count in a single pass over vectors
# =====================================================================

@njit(cache=True)
def _accumulate_sums(vectors, assignments, n_clusters):
    """
    Single pass through `vectors`, accumulating:
      sums[k, :]  = sum of all vectors assigned to cluster k
      counts[k]   = number of vectors assigned to cluster k

    This replaces the baseline's Python loop:
      for k in range(n_clusters):
          mask = assignments == k
          new_centroids[k] = vectors[mask].mean(axis=0)
    which is O(K*N) and allocates per cluster. Ours is O(N), single alloc.
    """
    n, d = vectors.shape
    sums   = np.zeros((n_clusters, d), dtype=vectors.dtype)
    counts = np.zeros(n_clusters, dtype=np.int64)

    for i in range(n):
        c = assignments[i]
        counts[c] += 1
        for j in range(d):
            sums[c, j] += vectors[i, j]
    return sums, counts


# =====================================================================
# Public K-Means function (drop-in compatible with IVFIndex._kmeans)
# =====================================================================

def kmeans_pp_init(vectors: np.ndarray, n_clusters: int, seed: int = 42) -> np.ndarray:
    """
    K-Means++ seeding. Instead of picking `n_clusters` centroids uniformly at
    random, this picks them one at a time with probability proportional to
    the squared distance from each point to its nearest already-chosen
    centroid.

    Effect: initial centroids are spread out across the embedding space,
    which usually cuts the number of K-Means iterations needed to converge
    and tends to avoid bad local minima.

    Implementation note: we expand ||v - c||^2 = ||v||^2 - 2 v.c + ||c||^2
    so each iteration is a single BLAS GEMV `vectors @ c` plus some O(N)
    arithmetic -- NO (N, D) temporary array allocation. This is ~10x
    faster than the naive `vectors - c` subtraction approach for large N.
    """
    vectors = vectors.astype(np.float32)
    n, d = vectors.shape
    rng = np.random.default_rng(seed)

    centroids = np.empty((n_clusters, d), dtype=vectors.dtype)

    # Precompute ||v||^2 for every vector -- stays constant across iterations.
    v_sq = np.einsum("ij,ij->i", vectors, vectors)

    def dist_sq_to(c: np.ndarray) -> np.ndarray:
        # ||v - c||^2 = ||v||^2 - 2 v.c + ||c||^2
        dots = vectors @ c                         # (N,)  BLAS GEMV
        c_sq = float(np.dot(c, c))
        out = v_sq - 2.0 * dots + c_sq
        np.maximum(out, 0.0, out=out)              # numerical safety
        return out

    # 1. First centroid: uniform random
    first_idx = int(rng.integers(n))
    centroids[0] = vectors[first_idx]
    dist_sq = dist_sq_to(centroids[0])

    # 2. Remaining centroids: sample with probability proportional to dist_sq
    for k in range(1, n_clusters):
        total = float(dist_sq.sum())
        if total <= 0.0:
            next_idx = int(rng.integers(n))
        else:
            probs = dist_sq / total
            next_idx = int(rng.choice(n, p=probs))
        centroids[k] = vectors[next_idx]

        new_dist = dist_sq_to(centroids[k])
        np.minimum(dist_sq, new_dist, out=dist_sq)

    return centroids


def kmeans_numba(vectors: np.ndarray,
                 n_clusters: int,
                 kmeans_iters: int = 20,
                 seed: int = 42,
                 tol: float = 1e-6,
                 verbose: bool = True,
                 init: str = "random") -> np.ndarray:
    """
    Numba-accelerated K-Means (cosine distance).

    init: "random" (uniform sample) or "kmeans++" (distance-weighted seeding).

    Returns centroids of shape (n_clusters, D).
    """
    vectors = vectors.astype(np.float32)
    n, d = vectors.shape

    rng = np.random.default_rng(seed)
    if init == "random":
        indices = rng.choice(n, size=n_clusters, replace=False)
        centroids = vectors[indices].copy()
    elif init == "kmeans++":
        if verbose:
            print(f"    [Numba] running K-Means++ init for {n_clusters} centroids...")
        t0 = time.perf_counter()
        centroids = kmeans_pp_init(vectors, n_clusters, seed=seed)
        if verbose:
            print(f"    [Numba] K-Means++ init done in {(time.perf_counter()-t0)*1000:.0f} ms")
    else:
        raise ValueError(f"Unknown init={init!r}. Use 'random' or 'kmeans++'.")

    # HOIST: vector norms don't change across iterations -- compute once.
    v_norms = np.linalg.norm(vectors, axis=1, keepdims=True)   # (N, 1)

    for iteration in range(kmeans_iters):
        # ---- Assignment: NumPy GEMM (BLAS-friendly, don't rewrite) ----
        dots    = vectors @ centroids.T                        # (N, K)
        c_norms = np.linalg.norm(centroids, axis=1)            # (K,)
        cos_sim = dots / (v_norms * c_norms[np.newaxis, :] + 1e-10)
        assignments = np.argmax(cos_sim, axis=1).astype(np.int64)

        # ---- Update: single Numba pass instead of per-cluster Python loop ----
        sums, counts = _accumulate_sums(vectors, assignments, n_clusters)

        new_centroids = np.empty_like(centroids)
        for k in range(n_clusters):
            if counts[k] > 0:
                new_centroids[k] = sums[k] / counts[k]
            else:
                # Dead cluster: reseed from a random vector
                new_centroids[k] = vectors[rng.integers(n)]

        # ---- Convergence ----
        shift = float(np.linalg.norm(new_centroids - centroids))
        centroids = new_centroids

        if verbose and (iteration + 1) % 5 == 0:
            print(f"    [Numba] K-Means iter {iteration + 1}/{kmeans_iters}, shift={shift:.6f}")

        if shift < tol:
            if verbose:
                print(f"    [Numba] K-Means converged at iteration {iteration + 1}")
            break

    return centroids


def warmup_kmeans_numba() -> None:
    """Trigger Numba JIT compilation so the first real call is not paying that cost."""
    dummy_vecs = np.random.randn(100, 32).astype(np.float32)
    dummy_assign = np.zeros(100, dtype=np.int64)
    _accumulate_sums(dummy_vecs, dummy_assign, 4)
    print("  Numba K-Means warmup complete.")


# =====================================================================
# IVFIndex subclass that uses Numba K-Means
# =====================================================================

class IVFIndexNumba(IVFIndex):
    """
    Drop-in replacement for IVFIndex. Only _kmeans is overridden.
    Uses Numba-accelerated K-Means with RANDOM initialization.
    (Step 2 behavior.)
    """

    def _kmeans(self, vectors: np.ndarray) -> np.ndarray:
        return kmeans_numba(
            vectors,
            n_clusters=self.n_clusters,
            kmeans_iters=self.kmeans_iters,
            init="random",
            verbose=True,
        )


class IVFIndexNumbaPP(IVFIndex):
    """
    Same as IVFIndexNumba but seeds centroids with K-Means++.
    Expected effect: fewer iterations to converge, slightly better local optimum.
    (Step 3 behavior.)
    """

    def _kmeans(self, vectors: np.ndarray) -> np.ndarray:
        return kmeans_numba(
            vectors,
            n_clusters=self.n_clusters,
            kmeans_iters=self.kmeans_iters,
            init="kmeans++",
            verbose=True,
        )
