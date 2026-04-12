"""
GPU-accelerated cosine similarity using CuPy — Week 12 optimization.

Key techniques demonstrated:
  - CuPy arrays on GPU (zero-copy from NumPy)
  - Matrix multiplication on GPU (cuBLAS under the hood)
  - GPU vs CPU comparison
  - Memory management (GPU memory is limited)

Requires: pip install cupy-cuda12x  (match your CUDA version)
"""
import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available — GPU similarity disabled.")


class GPUSimilarityEngine:
    """
    Manages GPU memory and provides cosine similarity on GPU.
    Pre-loads corpus to GPU once; queries are transferred per-call.
    """

    def __init__(self):
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy is required for GPU similarity")
        self.corpus_gpu = None
        self.corpus_norms_gpu = None

    def load_corpus(self, corpus_matrix: np.ndarray):
        """Transfer corpus to GPU and precompute norms."""
        self.corpus_gpu = cp.asarray(corpus_matrix, dtype=cp.float32)
        self.corpus_norms_gpu = cp.linalg.norm(self.corpus_gpu, axis=1)
        # Replace zeros to avoid division errors
        self.corpus_norms_gpu = cp.where(
            self.corpus_norms_gpu == 0, 1e-10, self.corpus_norms_gpu
        )
        print(f"Loaded {corpus_matrix.shape[0]:,} vectors to GPU "
              f"({corpus_matrix.nbytes / 1e6:.0f} MB)")

    def cosine_sim_gpu(self, query_vec: np.ndarray, corpus_matrix=None) -> np.ndarray:
        """
        Compute cosine similarity on GPU.
        If corpus_matrix is None, uses pre-loaded corpus.
        """
        query_gpu = cp.asarray(query_vec, dtype=cp.float32)
        q_norm = cp.linalg.norm(query_gpu)

        if q_norm == 0:
            n = self.corpus_gpu.shape[0] if corpus_matrix is None else len(corpus_matrix)
            return np.zeros(n, dtype=np.float32)

        if corpus_matrix is not None:
            c_gpu = cp.asarray(corpus_matrix, dtype=cp.float32)
            c_norms = cp.linalg.norm(c_gpu, axis=1)
            c_norms = cp.where(c_norms == 0, 1e-10, c_norms)
        else:
            c_gpu = self.corpus_gpu
            c_norms = self.corpus_norms_gpu

        # Core computation: matrix-vector multiply on GPU
        dots = c_gpu @ query_gpu      # (N,)
        scores = dots / (q_norm * c_norms)

        # Transfer result back to CPU
        return cp.asnumpy(scores)

    def top_k_gpu(self, query_vec: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        """Full Top-K on GPU including argpartition."""
        query_gpu = cp.asarray(query_vec, dtype=cp.float32)
        q_norm = cp.linalg.norm(query_gpu)

        dots = self.corpus_gpu @ query_gpu
        scores = dots / (q_norm * self.corpus_norms_gpu)

        # argpartition on GPU
        if k >= len(scores):
            indices = cp.argsort(scores)[::-1]
        else:
            indices = cp.argpartition(scores, -k)[-k:]
            indices = indices[cp.argsort(scores[indices])[::-1]]

        # Transfer only the top-K back to CPU
        indices_cpu = cp.asnumpy(indices)
        scores_cpu = cp.asnumpy(scores[indices])

        return [(int(i), float(s)) for i, s in zip(indices_cpu, scores_cpu)]

    def free(self):
        """Release GPU memory."""
        self.corpus_gpu = None
        self.corpus_norms_gpu = None
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()


# ──────────────────────────────────────────────
# Standalone function (for sim_fn interface compatibility)
# ──────────────────────────────────────────────

_engine = None

def cosine_sim_gpu(query_vec: np.ndarray, corpus_matrix: np.ndarray) -> np.ndarray:
    """
    Drop-in replacement matching the sim_fn interface.
    Note: This transfers corpus to GPU each call — for benchmarking only.
    Use GPUSimilarityEngine.load_corpus() for production use.
    """
    global _engine
    if _engine is None:
        _engine = GPUSimilarityEngine()
    return _engine.cosine_sim_gpu(query_vec, corpus_matrix)


# ──────────────────────────────────────────────
# Quick benchmark
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import time

    if not GPU_AVAILABLE:
        print("Skipping GPU benchmark — CuPy not installed")
        exit()

    np.random.seed(42)
    N, D = 500_000, 384
    query = np.random.randn(D).astype(np.float32)
    corpus = np.random.randn(N, D).astype(np.float32)

    engine = GPUSimilarityEngine()
    engine.load_corpus(corpus)

    # Warmup
    _ = engine.cosine_sim_gpu(query)

    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        scores = engine.cosine_sim_gpu(query)
        times.append(time.perf_counter() - t0)

    avg = np.mean(times) * 1000
    print(f"GPU cosine sim ({N:,} vectors): {avg:.2f}ms avg")

    # Compare with NumPy
    from components.similarity import cosine_sim_numpy
    t0 = time.perf_counter()
    _ = cosine_sim_numpy(query, corpus)
    t_np = (time.perf_counter() - t0) * 1000
    print(f"NumPy cosine sim ({N:,} vectors): {t_np:.2f}ms")
    print(f"GPU speedup: {t_np / avg:.1f}x")

    engine.free()
