"""
Parallel index construction — Week 10-11 optimization.

Key techniques demonstrated:
  - multiprocessing.Pool for data-parallel index building
  - Shared memory (multiprocessing.shared_memory) to avoid data copying
  - Process pool vs thread pool comparison
  - Speedup analysis: linear scaling, Amdahl's law
"""
import time
import numpy as np
from multiprocessing import Pool, cpu_count, shared_memory
from functools import partial

import config
from components.vector_index import IVFIndex
from components.similarity import cosine_sim_numpy


def _assign_chunk(args):
    """
    Worker function: assign a chunk of vectors to their nearest centroids.
    Runs in a separate process.
    """
    start, end, shm_name, shm_shape, shm_dtype, centroids = args

    # Attach to shared memory
    shm = shared_memory.SharedMemory(name=shm_name)
    vectors = np.ndarray(shm_shape, dtype=shm_dtype, buffer=shm.buf)

    chunk = vectors[start:end]

    # Compute assignments for this chunk
    dots = chunk @ centroids.T
    c_norms = np.linalg.norm(centroids, axis=1)
    v_norms = np.linalg.norm(chunk, axis=1, keepdims=True)
    cos_sim = dots / (v_norms * c_norms[np.newaxis, :] + 1e-10)
    assignments = np.argmax(cos_sim, axis=1)

    shm.close()
    return start, assignments


class ParallelIVFBuilder:
    """
    Builds an IVF index using multiprocessing for the assignment step.
    K-Means itself is still sequential (optimization left for Week 8).
    """

    def __init__(
        self,
        n_clusters: int = config.IVF_NUM_CLUSTERS,
        n_workers: int = None,
    ):
        self.n_clusters = n_clusters
        self.n_workers = n_workers or cpu_count()

    def build_parallel(
        self,
        vectors: np.ndarray,
        doc_ids: list[str],
        base_index: IVFIndex,
    ) -> float:
        """
        Build IVF index with parallel cluster assignment.
        Uses shared memory to avoid copying the full vector matrix.

        Returns build time in seconds.
        """
        t0 = time.perf_counter()
        n = len(vectors)

        # Step 1: K-Means (reuse from base_index for now)
        print(f"Running K-Means with {self.n_clusters} clusters...")
        centroids = base_index._kmeans(vectors)

        # Step 2: Parallel assignment using shared memory
        print(f"Parallel assignment with {self.n_workers} workers...")

        # Create shared memory block for vectors
        shm = shared_memory.SharedMemory(create=True, size=vectors.nbytes)
        shared_vectors = np.ndarray(vectors.shape, dtype=vectors.dtype, buffer=shm.buf)
        np.copyto(shared_vectors, vectors)

        # Split work into chunks
        chunk_size = (n + self.n_workers - 1) // self.n_workers
        tasks = []
        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            tasks.append((i, end, shm.name, vectors.shape, vectors.dtype, centroids))

        # Run parallel assignment
        all_assignments = np.empty(n, dtype=np.int32)
        with Pool(self.n_workers) as pool:
            results = pool.map(_assign_chunk, tasks)

        for start, chunk_assignments in results:
            end = start + len(chunk_assignments)
            all_assignments[start:end] = chunk_assignments

        # Cleanup shared memory
        shm.close()
        shm.unlink()

        # Step 3: Build inverted lists
        base_index.centroids = centroids
        base_index.vectors = vectors.astype(np.float32)
        base_index.doc_ids = doc_ids
        base_index.inverted_lists = {k: [] for k in range(self.n_clusters)}
        for i, cluster_id in enumerate(all_assignments):
            base_index.inverted_lists[int(cluster_id)].append(i)

        build_time = time.perf_counter() - t0
        base_index.build_time = build_time
        print(f"Parallel IVF build complete: {build_time:.2f}s "
              f"({self.n_workers} workers)")

        return build_time


# ──────────────────────────────────────────────
# Benchmark: sequential vs parallel build
# ──────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    N, D = 100_000, 384
    vectors = np.random.randn(N, D).astype(np.float32)
    doc_ids = [f"doc_{i}" for i in range(N)]

    n_clusters = 64

    # Sequential build
    seq_index = IVFIndex(n_clusters=n_clusters)
    seq_index.build(vectors, doc_ids)
    t_seq = seq_index.build_time

    # Parallel build for different worker counts
    for n_workers in [2, 4, 8]:
        par_index = IVFIndex(n_clusters=n_clusters)
        builder = ParallelIVFBuilder(n_clusters=n_clusters, n_workers=n_workers)
        t_par = builder.build_parallel(vectors, doc_ids, par_index)

        print(f"\nWorkers={n_workers}: {t_par:.2f}s (speedup: {t_seq/t_par:.2f}x)")
