"""
Optimized LLM generation — Week 9 (Concurrency).

Two optimization strategies for the generation (answer-making) step:

  1. ThreadPoolExecutor — send multiple LLM requests in parallel
  2. asyncio — same idea with async/await

Both work because LLM API calls are IO-bound: the CPU is idle
while waiting for the remote model to generate tokens.

Baseline:  10 queries × 500ms each = 5,000ms
Optimized: 10 queries in parallel   =   ~500ms
Speedup:   ~10x
"""
import time
import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from components.generator import BaselineGenerator


# ══════════════════════════════════════════════
# STRATEGY 1: ThreadPoolExecutor
# ══════════════════════════════════════════════

class ThreadedGenerator:
    """
    Concurrent LLM generator using threads.

    Sends multiple generation requests simultaneously.
    Since each request involves ~500ms of network wait,
    threading overlaps all the waiting time.
    """

    def __init__(self, base_generator: BaselineGenerator, n_workers: int = 8):
        self.generator = base_generator
        self.n_workers = n_workers

    def generate(self, query: str, contexts: list[str]) -> dict:
        """Single query — pass through to base generator."""
        return self.generator.generate(query, contexts)

    def generate_batch(self, items: list[tuple[str, list[str]]]) -> list[dict]:
        """
        Generate answers for multiple queries CONCURRENTLY.

        items: list of (query, contexts) tuples
        """
        results = [None] * len(items)

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(self.generator.generate, query, contexts): idx
                for idx, (query, contexts) in enumerate(items)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return results


# ══════════════════════════════════════════════
# STRATEGY 2: asyncio
# ══════════════════════════════════════════════

class AsyncGenerator:
    """
    Concurrent LLM generator using asyncio.

    Lower overhead than threads for many concurrent requests.
    Uses a semaphore to limit concurrency (respect API rate limits).
    """

    def __init__(self, base_generator: BaselineGenerator, max_concurrent: int = 16):
        self.generator = base_generator
        self.max_concurrent = max_concurrent

    async def _async_generate(self, query: str, contexts: list[str], semaphore: asyncio.Semaphore) -> dict:
        async with semaphore:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.generator.generate,
                query,
                contexts,
            )
            return result

    async def _generate_all(self, items: list[tuple[str, list[str]]]) -> list[dict]:
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [
            self._async_generate(query, contexts, semaphore)
            for query, contexts in items
        ]
        return await asyncio.gather(*tasks)

    def generate_batch(self, items: list[tuple[str, list[str]]]) -> list[dict]:
        """Public sync interface."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self._generate_all(items))
            else:
                return loop.run_until_complete(self._generate_all(items))
        except RuntimeError:
            return asyncio.run(self._generate_all(items))


# ══════════════════════════════════════════════
# Benchmark
# ══════════════════════════════════════════════

if __name__ == "__main__":
    n_queries = 20
    items = [
        (f"What is question {i}?", [f"Context for question {i}"])
        for i in range(n_queries)
    ]
    print(f"Benchmark: {n_queries} queries, simulated 500ms LLM latency\n")

    # Sequential baseline
    gen = BaselineGenerator(api_provider="simulated", simulated_latency_ms=500)
    t0 = time.perf_counter()
    results_seq = gen.generate_batch(items)
    t_seq = time.perf_counter() - t0
    print(f"Sequential:  {t_seq:.2f}s ({n_queries} × 500ms)")

    # Threaded
    gen2 = BaselineGenerator(api_provider="simulated", simulated_latency_ms=500)
    threaded = ThreadedGenerator(gen2, n_workers=n_queries)
    t0 = time.perf_counter()
    results_thr = threaded.generate_batch(items)
    t_thr = time.perf_counter() - t0
    print(f"Threaded:    {t_thr:.2f}s  (speedup: {t_seq/t_thr:.1f}x)")

    # Async
    gen3 = BaselineGenerator(api_provider="simulated", simulated_latency_ms=500)
    async_gen = AsyncGenerator(gen3, max_concurrent=n_queries)
    t0 = time.perf_counter()
    results_async = async_gen.generate_batch(items)
    t_async = time.perf_counter() - t0
    print(f"Async:       {t_async:.2f}s  (speedup: {t_seq/t_async:.1f}x)")

    print(f"\nExpected: sequential ~{n_queries * 0.5:.0f}s, "
          f"concurrent ~0.5s, speedup ~{n_queries}x")
