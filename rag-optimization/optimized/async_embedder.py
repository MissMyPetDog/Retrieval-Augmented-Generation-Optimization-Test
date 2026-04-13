"""
Optimized embedding generation — Week 9 (Concurrency) + Week 10 (Parallel).

Three optimization strategies:

  1. ThreadPoolExecutor — for API-based embedding (IO-bound)
     Threads release GIL during network wait → true parallelism for IO.

  2. asyncio + aiohttp — for API-based embedding (IO-bound)
     Event loop handles thousands of concurrent requests with minimal overhead.

  3. ProcessPoolExecutor — for local model embedding (CPU-bound)
     Each process loads its own model, processes a data chunk independently.

Key insight: The RIGHT concurrency tool depends on the BOTTLENECK TYPE:
  - IO-bound (API calls, network wait) → threading or asyncio
  - CPU-bound (local model inference)  → multiprocessing (avoids GIL)
"""
import time
import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import config


# ══════════════════════════════════════════════
# STRATEGY 1: ThreadPoolExecutor (API, IO-bound)
# ══════════════════════════════════════════════

class ThreadedAPIEmbedder:
    """
    Concurrent API embedder using threads.

    Why threads work here (but not for CPU tasks):
      Python's GIL prevents parallel CPU computation in threads.
      BUT when a thread is waiting for network IO, it RELEASES the GIL.
      So multiple threads can wait for API responses simultaneously.

    Baseline:  16 batches × 100ms each = 1,600ms (sequential)
    Threaded:  16 batches in parallel   =   ~100ms (all waiting at once)
    """

    def __init__(
        self,
        api_embedder,
        n_workers: int = 8,
    ):
        self.api_embedder = api_embedder
        self.n_workers = n_workers
        self.batch_size = api_embedder.batch_size

    def embed_texts(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """Embed all texts with concurrent API calls."""
        # Split into batches
        batches = []
        for i in range(0, len(texts), self.batch_size):
            batches.append(texts[i : i + self.batch_size])

        if show_progress:
            print(f"  Sending {len(batches)} batches with {self.n_workers} threads...")

        # Submit all batches to thread pool
        all_embeddings = [None] * len(batches)

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(self.api_embedder._call_api_single_batch, batch): idx
                for idx, batch in enumerate(batches)
            }
            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                all_embeddings[idx] = future.result()
                completed += 1
                if show_progress and completed % 10 == 0:
                    print(f"    Completed {completed}/{len(batches)} batches")

        return np.vstack(all_embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Single query — no benefit from threading, just pass through."""
        return self.api_embedder.embed_query(query)


# ══════════════════════════════════════════════
# STRATEGY 2: asyncio (API, IO-bound)
# ══════════════════════════════════════════════

class AsyncAPIEmbedder:
    """
    Concurrent API embedder using asyncio.

    asyncio vs ThreadPool:
      - asyncio: single thread, event loop switches between tasks during IO wait.
        Lower memory overhead, better for thousands of concurrent requests.
      - ThreadPool: multiple OS threads, each blocked during IO.
        Simpler code, but more memory per concurrent request.

    Both achieve similar speedup for our use case (tens of requests).
    asyncio shines when you have hundreds/thousands of concurrent calls.
    """

    def __init__(
        self,
        api_embedder,
        max_concurrent: int = 16,
    ):
        self.api_embedder = api_embedder
        self.max_concurrent = max_concurrent
        self.batch_size = api_embedder.batch_size

    async def _async_call_batch(self, batch: list[str], semaphore: asyncio.Semaphore) -> np.ndarray:
        """Call API with concurrency control via semaphore."""
        async with semaphore:
            # Run the synchronous API call in a thread to not block the event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.api_embedder._call_api_single_batch,
                batch,
            )
            return result

    async def _embed_all_async(self, texts: list[str], show_progress: bool) -> np.ndarray:
        """Core async logic."""
        batches = []
        for i in range(0, len(texts), self.batch_size):
            batches.append(texts[i : i + self.batch_size])

        semaphore = asyncio.Semaphore(self.max_concurrent)

        if show_progress:
            print(f"  Sending {len(batches)} batches async (max {self.max_concurrent} concurrent)...")

        tasks = [
            self._async_call_batch(batch, semaphore)
            for batch in batches
        ]

        results = await asyncio.gather(*tasks)
        return np.vstack(results)

    def embed_texts(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """Public sync interface — creates event loop internally."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # In Jupyter/Colab, use nest_asyncio
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self._embed_all_async(texts, show_progress))
            else:
                return loop.run_until_complete(self._embed_all_async(texts, show_progress))
        except RuntimeError:
            return asyncio.run(self._embed_all_async(texts, show_progress))

    def embed_query(self, query: str) -> np.ndarray:
        return self.api_embedder.embed_query(query)


# ══════════════════════════════════════════════
# STRATEGY 3: ProcessPoolExecutor (Local, CPU-bound)
# ══════════════════════════════════════════════

def _local_encode_worker(args):
    """
    Worker for process pool. Each process loads its own model.
    Must be a top-level function (not method) for pickling.
    """
    texts, model_name, device = args
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


class ParallelLocalEmbedder:
    """
    Parallel local embedder using multiple processes.

    Why processes (not threads) for local model:
      sentence-transformers encoding is CPU-bound (or GPU-bound).
      Python's GIL prevents threads from using multiple CPU cores.
      Processes bypass GIL — each gets its own Python interpreter.

    Trade-off: each process loads its own copy of the model (~500MB RAM each).
    """

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL,
        n_workers: int = None,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.n_workers = n_workers or min(4, cpu_count())
        self.device = device

    def embed_texts(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """Split texts across processes, each loads own model."""
        n = len(texts)
        chunk_size = (n + self.n_workers - 1) // self.n_workers
        chunks = [texts[i:i + chunk_size] for i in range(0, n, chunk_size)]
        tasks = [(chunk, self.model_name, self.device) for chunk in chunks]

        if show_progress:
            print(f"  Encoding {n} texts with {self.n_workers} processes...")

        all_embeddings = [None] * len(chunks)

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(_local_encode_worker, task): idx
                for idx, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                all_embeddings[idx] = future.result()
                if show_progress:
                    print(f"    Process {idx + 1}/{len(chunks)} done")

        return np.vstack(all_embeddings)


# ══════════════════════════════════════════════
# Benchmark: all strategies compared
# ══════════════════════════════════════════════

if __name__ == "__main__":
    from components.embedder import APIEmbedder

    texts = [f"This is test sentence number {i} for embedding." for i in range(640)]
    print(f"Benchmark: {len(texts)} texts, simulated 100ms API latency\n")

    # Create base API embedder
    api = APIEmbedder(api_provider="simulated", simulated_latency_ms=100, batch_size=64)
    n_batches = (len(texts) + api.batch_size - 1) // api.batch_size

    # Sequential baseline
    t0 = time.perf_counter()
    vecs_seq = api.embed_texts(texts, show_progress=False)
    t_seq = time.perf_counter() - t0
    print(f"Sequential:     {t_seq:.2f}s ({n_batches} batches × 100ms)")

    # Threaded
    api2 = APIEmbedder(api_provider="simulated", simulated_latency_ms=100, batch_size=64)
    threaded = ThreadedAPIEmbedder(api2, n_workers=n_batches)
    t0 = time.perf_counter()
    vecs_thr = threaded.embed_texts(texts, show_progress=False)
    t_thr = time.perf_counter() - t0
    print(f"Threaded:       {t_thr:.2f}s  (speedup: {t_seq/t_thr:.1f}x)")

    # Async
    api3 = APIEmbedder(api_provider="simulated", simulated_latency_ms=100, batch_size=64)
    async_emb = AsyncAPIEmbedder(api3, max_concurrent=n_batches)
    t0 = time.perf_counter()
    vecs_async = async_emb.embed_texts(texts, show_progress=False)
    t_async = time.perf_counter() - t0
    print(f"Async:          {t_async:.2f}s  (speedup: {t_seq/t_async:.1f}x)")

    # Verify consistency
    diff = np.max(np.abs(vecs_seq - vecs_thr))
    print(f"\nMax diff (seq vs threaded): {diff:.6f}")
