"""
Concurrent embedding generation — Week 9 optimization.

Key techniques demonstrated:
  - concurrent.futures.ProcessPoolExecutor for CPU-bound batch encoding
  - concurrent.futures.ThreadPoolExecutor comparison
  - Batch-level parallelism (each worker processes a batch)
  - Speedup measurement and analysis
"""
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

import config


def _encode_batch_worker(args):
    """
    Worker function for process pool.
    Each worker loads its own model instance (process-level isolation).
    """
    batch_texts, model_name = args

    # Each process needs its own model (can't share across processes)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        batch_texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


class ConcurrentEmbedder:
    """
    Generates embeddings using multiple processes.
    Each process loads its own model and handles a subset of batches.
    """

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL,
        batch_size: int = config.EMBEDDING_BATCH_SIZE,
        n_workers: int = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.n_workers = n_workers or min(4, cpu_count())

    def embed_texts(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode texts using multiple processes.
        Splits data into n_workers chunks, each processed independently.
        """
        n = len(texts)
        chunk_size = (n + self.n_workers - 1) // self.n_workers

        # Split texts into worker-level chunks
        chunks = []
        for i in range(0, n, chunk_size):
            chunks.append(texts[i : i + chunk_size])

        if show_progress:
            print(f"Encoding {n} texts with {self.n_workers} workers "
                  f"({len(chunks)} chunks)...")

        # Submit to process pool
        all_embeddings = [None] * len(chunks)
        tasks = [(chunk, self.model_name) for chunk in chunks]

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(_encode_batch_worker, task): idx
                for idx, task in enumerate(tasks)
            }

            for future in as_completed(futures):
                idx = futures[future]
                all_embeddings[idx] = future.result()
                if show_progress:
                    print(f"  Worker {idx + 1}/{len(chunks)} done")

        return np.vstack(all_embeddings)


# ──────────────────────────────────────────────
# Benchmark: sequential vs concurrent
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from components.embedder import BaselineEmbedder

    # Generate dummy texts
    texts = [f"This is test sentence number {i} for embedding." for i in range(2000)]

    # Sequential baseline
    seq = BaselineEmbedder()
    t0 = time.perf_counter()
    vecs_seq = seq.embed_texts(texts, show_progress=False)
    t_seq = time.perf_counter() - t0
    print(f"Sequential: {t_seq:.2f}s")

    # Concurrent
    for n_workers in [2, 4]:
        con = ConcurrentEmbedder(n_workers=n_workers)
        t0 = time.perf_counter()
        vecs_con = con.embed_texts(texts, show_progress=False)
        t_con = time.perf_counter() - t0
        print(f"Concurrent (workers={n_workers}): {t_con:.2f}s, "
              f"speedup: {t_seq/t_con:.2f}x")

        # Verify correctness
        diff = np.max(np.abs(vecs_seq - vecs_con))
        print(f"  Max embedding diff: {diff:.6f}")
