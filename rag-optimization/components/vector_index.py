"""
Vector index implementations — self-built, no FAISS.

Two index types:
  1. BruteForceIndex : Linear scan over all vectors (baseline)
  2. IVFIndex        : Inverted File Index with K-Means clustering

Optimization targets:
  - Week 5:  Cython for inner-loop distance computation
  - Week 6:  Numba JIT for search kernel
  - Week 8:  Optimized K-Means (scipy.optimize)
  - Week 10: Multiprocessing for parallel index building
  - Week 12: GPU-accelerated search (CuPy matrix multiply)
"""
import time
import pickle
import numpy as np
from typing import Optional

import config
from components.similarity import (
    cosine_sim_numpy,
    top_k_numpy,
)


class BruteForceIndex:
    """
    Linear scan index. Simple but O(N*D) per query.
    Serves as the accuracy reference — IVF results are compared against this.
    """

    def __init__(self):
        self.vectors: Optional[np.ndarray] = None   # (N, D)
        self.vector_norms: Optional[np.ndarray] = None  # (N,)
        self.use_precomputed_norms: bool = True
        self.doc_ids: list[str] = []
        self.build_time: float = 0.0

    def build(self, vectors: np.ndarray, doc_ids: list[str]):
        """Store all vectors. 'Building' is just a memcopy here."""
        t0 = time.perf_counter()
        self.vectors = vectors.astype(np.float32)
        self.vector_norms = np.linalg.norm(self.vectors, axis=1).astype(np.float32)
        self.doc_ids = doc_ids
        self.build_time = time.perf_counter() - t0
        print(f"BruteForceIndex built: {len(doc_ids)} vectors in {self.build_time:.3f}s")

    @staticmethod
    def _cosine_sim_with_precomputed_norms(
        query_vec: np.ndarray,
        corpus_matrix: np.ndarray,
        corpus_norms: np.ndarray,
    ) -> np.ndarray:
        """Cosine similarity using cached corpus norms."""
        dots = corpus_matrix @ query_vec
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            return np.zeros(corpus_matrix.shape[0], dtype=np.float32)

        denom = q_norm * corpus_norms
        denom = np.where(denom == 0, 1e-10, denom)
        return dots / denom

    def search(
        self,
        query_vec: np.ndarray,
        k: int = config.TOP_K,
        sim_fn=cosine_sim_numpy,
        use_precomputed_norms: Optional[bool] = None,
    ) -> list[tuple[str, float]]:
        """
        Search for top-K nearest neighbors.
        sim_fn is swappable: pass cosine_sim_cython, cosine_sim_numba, etc.
        """
        if use_precomputed_norms is None:
            use_precomputed_norms = self.use_precomputed_norms

        if use_precomputed_norms and sim_fn is cosine_sim_numpy and self.vector_norms is not None:
            scores = self._cosine_sim_with_precomputed_norms(
                query_vec, self.vectors, self.vector_norms
            )
        else:
            scores = sim_fn(query_vec, self.vectors)
        top = top_k_numpy(scores, k)
        return [(self.doc_ids[idx], score) for idx, score in top]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vectors": self.vectors,
                    "vector_norms": self.vector_norms,
                    "doc_ids": self.doc_ids,
                },
                f,
            )

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectors = data["vectors"]
        self.vector_norms = data.get("vector_norms")
        if self.vector_norms is None and self.vectors is not None:
            # Backward-compatible loading for older pickle files.
            self.vector_norms = np.linalg.norm(self.vectors, axis=1).astype(np.float32)
        self.doc_ids = data["doc_ids"]


class IVFIndex:
    """
    Inverted File Index: partition vectors into clusters via K-Means,
    then only search the nearest clusters at query time.

    Trade-off: build is slower (K-Means), but search is O(n_probe * N/K * D)
    instead of O(N * D).
    """

    def __init__(
        self,
        n_clusters: int = config.IVF_NUM_CLUSTERS,
        n_probes: int = config.IVF_NUM_PROBES,
        kmeans_iters: int = 20,
    ):
        self.n_clusters = n_clusters
        self.n_probes = n_probes
        self.kmeans_iters = kmeans_iters

        self.centroids: Optional[np.ndarray] = None   # (K, D)
        self.centroid_norms: Optional[np.ndarray] = None  # (K,)
        self.inverted_lists: dict[int, list[int]] = {} # cluster_id -> [vec indices]
        self.vectors: Optional[np.ndarray] = None
        self.vector_norms: Optional[np.ndarray] = None  # (N,)
        self.use_precomputed_norms: bool = True
        self.doc_ids: list[str] = []
        self.build_time: float = 0.0

    def _kmeans(self, vectors: np.ndarray) -> np.ndarray:
        """
        Simple K-Means implementation (baseline: pure NumPy).
        Returns centroids of shape (n_clusters, D).

        Optimization target (Week 8): scipy.optimize, better init strategies.
        """
        n, d = vectors.shape
        # Random initialization
        rng = np.random.default_rng(42)
        indices = rng.choice(n, size=self.n_clusters, replace=False)
        centroids = vectors[indices].copy()

        for iteration in range(self.kmeans_iters):
            # Assignment step: find nearest centroid for each vector
            # (N, D) @ (D, K) -> (N, K) distance matrix
            dots = vectors @ centroids.T
            c_norms = np.linalg.norm(centroids, axis=1)  # (K,)
            v_norms = np.linalg.norm(vectors, axis=1, keepdims=True)  # (N, 1)
            cos_sim = dots / (v_norms * c_norms[np.newaxis, :] + 1e-10)
            assignments = np.argmax(cos_sim, axis=1)  # (N,)

            # Update step: recompute centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = assignments == k
                if mask.any():
                    new_centroids[k] = vectors[mask].mean(axis=0)
                else:
                    # Dead cluster: reinitialize randomly
                    new_centroids[k] = vectors[rng.integers(n)]

            # Convergence check
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids

            if (iteration + 1) % 5 == 0:
                print(f"    K-Means iter {iteration + 1}/{self.kmeans_iters}, shift={shift:.6f}")

            if shift < 1e-6:
                print(f"    K-Means converged at iteration {iteration + 1}")
                break

        return centroids

    def build(self, vectors: np.ndarray, doc_ids: list[str]):
        """Build the IVF index: run K-Means + build inverted lists."""
        t0 = time.perf_counter()

        self.vectors = vectors.astype(np.float32)
        self.vector_norms = np.linalg.norm(self.vectors, axis=1).astype(np.float32)
        self.doc_ids = doc_ids
        n = len(vectors)

        print(f"Building IVF index: {n} vectors, {self.n_clusters} clusters...")

        # Step 1: K-Means clustering
        self.centroids = self._kmeans(self.vectors)
        self.centroid_norms = np.linalg.norm(self.centroids, axis=1).astype(np.float32)

        # Step 2: Assign each vector to its nearest cluster
        dots = self.vectors @ self.centroids.T
        v_norms = self.vector_norms[:, np.newaxis]
        c_norms = self.centroid_norms[np.newaxis, :]
        cos_sim = dots / (v_norms * c_norms + 1e-10)
        assignments = np.argmax(cos_sim, axis=1)

        # Step 3: Build inverted lists
        self.inverted_lists = {k: [] for k in range(self.n_clusters)}
        for i, cluster_id in enumerate(assignments):
            self.inverted_lists[int(cluster_id)].append(i)

        self.build_time = time.perf_counter() - t0

        # Stats
        sizes = [len(v) for v in self.inverted_lists.values()]
        print(f"IVF index built in {self.build_time:.2f}s")
        print(f"  Cluster sizes: min={min(sizes)}, max={max(sizes)}, "
              f"mean={np.mean(sizes):.0f}, std={np.std(sizes):.0f}")

    def search(
        self,
        query_vec: np.ndarray,
        k: int = config.TOP_K,
        n_probes: Optional[int] = None,
        sim_fn=cosine_sim_numpy,
        use_precomputed_norms: Optional[bool] = None,
    ) -> list[tuple[str, float]]:
        """
        Search: find nearest clusters, then search within them.
        """
        if n_probes is None:
            n_probes = self.n_probes
        if use_precomputed_norms is None:
            use_precomputed_norms = self.use_precomputed_norms

        # Step 1: Find the closest clusters to query
        if use_precomputed_norms and sim_fn is cosine_sim_numpy and self.centroid_norms is not None:
            centroid_scores = BruteForceIndex._cosine_sim_with_precomputed_norms(
                query_vec, self.centroids, self.centroid_norms
            )
        else:
            centroid_scores = sim_fn(query_vec, self.centroids)
        top_clusters = np.argpartition(centroid_scores, -n_probes)[-n_probes:]

        # Step 2: Gather candidate vectors from those clusters
        candidate_indices = []
        for cluster_id in top_clusters:
            candidate_indices.extend(self.inverted_lists[int(cluster_id)])

        if not candidate_indices:
            return []

        candidate_indices = np.array(candidate_indices)
        candidate_vectors = self.vectors[candidate_indices]
        candidate_norms = (
            self.vector_norms[candidate_indices]
            if self.vector_norms is not None
            else None
        )

        # Step 3: Score candidates
        if use_precomputed_norms and sim_fn is cosine_sim_numpy and candidate_norms is not None:
            scores = BruteForceIndex._cosine_sim_with_precomputed_norms(
                query_vec, candidate_vectors, candidate_norms
            )
        else:
            scores = sim_fn(query_vec, candidate_vectors)
        top = top_k_numpy(scores, k)

        return [
            (self.doc_ids[candidate_indices[idx]], score)
            for idx, score in top
        ]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "centroids": self.centroids,
                "centroid_norms": self.centroid_norms,
                "inverted_lists": self.inverted_lists,
                "vectors": self.vectors,
                "vector_norms": self.vector_norms,
                "doc_ids": self.doc_ids,
                "n_clusters": self.n_clusters,
                "n_probes": self.n_probes,
            }, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.centroids = data["centroids"]
        self.centroid_norms = data.get("centroid_norms")
        if self.centroid_norms is None and self.centroids is not None:
            self.centroid_norms = np.linalg.norm(self.centroids, axis=1).astype(np.float32)
        self.inverted_lists = data["inverted_lists"]
        self.vectors = data["vectors"]
        self.vector_norms = data.get("vector_norms")
        if self.vector_norms is None and self.vectors is not None:
            # Backward-compatible loading for older pickle files.
            self.vector_norms = np.linalg.norm(self.vectors, axis=1).astype(np.float32)
        self.doc_ids = data["doc_ids"]
        self.n_clusters = data["n_clusters"]
        self.n_probes = data["n_probes"]


# ──────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    N, D = 10_000, 384
    vectors = np.random.randn(N, D).astype(np.float32)
    doc_ids = [f"doc_{i}" for i in range(N)]
    query = np.random.randn(D).astype(np.float32)

    # Test BruteForce
    bf = BruteForceIndex()
    bf.build(vectors, doc_ids)

    t0 = time.perf_counter()
    bf_results = bf.search(query, k=10)
    t_bf = time.perf_counter() - t0
    print(f"\nBruteForce search: {t_bf*1000:.2f}ms")
    print(f"  Top-3: {bf_results[:3]}")

    # Test IVF
    ivf = IVFIndex(n_clusters=32, n_probes=4)
    ivf.build(vectors, doc_ids)

    t0 = time.perf_counter()
    ivf_results = ivf.search(query, k=10)
    t_ivf = time.perf_counter() - t0
    print(f"\nIVF search: {t_ivf*1000:.2f}ms")
    print(f"  Top-3: {ivf_results[:3]}")

    # Check recall: how many of BF's top-10 are in IVF's top-10
    bf_set = set(doc_id for doc_id, _ in bf_results)
    ivf_set = set(doc_id for doc_id, _ in ivf_results)
    recall = len(bf_set & ivf_set) / len(bf_set)
    print(f"\nIVF Recall@10 vs BruteForce: {recall:.1%}")
    print(f"Search speedup: {t_bf / t_ivf:.1f}x")
