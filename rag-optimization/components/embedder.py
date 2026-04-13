"""
Embedding generation — baseline implementations.

Two modes:
  1. LOCAL mode:  Uses sentence-transformers model locally (default)
  2. API mode:    Calls a remote embedding API (e.g. OpenAI, Cohere)

Optimization targets:
  - Week 3  (itertools):     Batch iterator with itertools.islice for memory efficiency
  - Week 9  (Concurrency):   asyncio for concurrent API calls (IO-bound)
  - Week 9  (Concurrency):   ThreadPoolExecutor for parallel API requests
  - Week 10 (Parallel):      ProcessPoolExecutor for local model on multiple CPU cores
  - Week 12 (GPU):           GPU-accelerated local encoding

The key insight: LOCAL embedding is CPU/GPU-bound → optimize with parallelism and hardware.
                 API embedding is IO-bound → optimize with async/threading.
"""
import time
import numpy as np
from typing import Optional

import config


# ══════════════════════════════════════════════
# LOCAL EMBEDDER (sentence-transformers)
# ══════════════════════════════════════════════

class LocalEmbedder:
    """
    Baseline local embedder. Loads model into memory, encodes sequentially.
    """

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL,
        batch_size: int = config.EMBEDDING_BATCH_SIZE,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    @property
    def model(self):
        """Lazy model loading."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch

            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device

            print(f"Loading embedding model: {self.model_name} (device={device})...")
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    def embed_texts(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        Baseline: sequential batch processing within a single process.

        ┌─────────────────────────────────────────────────────────────┐
        │ OPTIMIZATION OPPORTUNITIES:                                  │
        │                                                              │
        │ 1. [Week 10] ProcessPoolExecutor:                            │
        │    Split texts across N workers, each loads own model.       │
        │    Good for CPU encoding on multi-core machines.             │
        │                                                              │
        │ 2. [Week 3] itertools.islice:                                │
        │    Instead of loading all texts into memory, use a generator │
        │    + islice to process in streaming batches.                 │
        │    Reduces peak memory from O(N) to O(batch_size).           │
        │                                                              │
        │ 3. [Week 12] GPU batching:                                   │
        │    Larger batch sizes better utilize GPU parallelism.        │
        │    Tune batch_size based on GPU memory.                      │
        └─────────────────────────────────────────────────────────────┘
        """
        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1

            if show_progress and batch_num % 10 == 0:
                print(f"  Embedding batch {batch_num}/{total_batches}")

            embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query string."""
        embedding = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding[0].astype(np.float32)


# ══════════════════════════════════════════════
# API EMBEDDER (remote service)
# ══════════════════════════════════════════════

class APIEmbedder:
    """
    Baseline API embedder. Calls a remote embedding service SEQUENTIALLY.
    Each call involves network latency (typically 50-200ms per request).

    This is intentionally slow — every batch waits for the previous one
    to complete before sending the next request.

    Supports:
      - Real API: set api_provider="openai" and provide api_key
      - Simulated: set api_provider="simulated" (default, no key needed)
    """

    def __init__(
        self,
        api_provider: str = "simulated",
        api_key: str = None,
        model_name: str = "text-embedding-3-small",
        batch_size: int = 64,
        simulated_latency_ms: float = 100.0,
        embedding_dim: int = config.EMBEDDING_DIM,
    ):
        self.api_provider = api_provider
        self.api_key = api_key
        self.model_name = model_name
        self.batch_size = batch_size
        self.simulated_latency_ms = simulated_latency_ms
        self.embedding_dim = embedding_dim

        # Stats tracking
        self.total_requests = 0
        self.total_latency_ms = 0.0

    def _call_api_single_batch(self, texts: list[str]) -> np.ndarray:
        """
        Send one batch to the embedding API and wait for response.
        This is the function that optimization should target.

        ┌─────────────────────────────────────────────────────────────┐
        │ THIS IS THE BOTTLENECK FOR API-BASED EMBEDDING.             │
        │                                                              │
        │ Baseline: call sequentially, wait for each response.         │
        │ Each call takes ~100ms of network latency.                   │
        │ 1000 texts / 64 per batch = 16 calls × 100ms = 1.6 seconds │
        │                                                              │
        │ With async (16 calls in parallel): ~100ms total              │
        │ Speedup: 16x                                                 │
        └─────────────────────────────────────────────────────────────┘
        """
        t0 = time.perf_counter()

        if self.api_provider == "simulated":
            # Simulate network latency + return random embeddings
            time.sleep(self.simulated_latency_ms / 1000.0)
            # Use deterministic "embeddings" based on text hash for consistency
            embeddings = np.array([
                self._deterministic_embedding(t) for t in texts
            ], dtype=np.float32)

        elif self.api_provider == "openai":
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.embeddings.create(
                model=self.model_name,
                input=texts,
            )
            embeddings = np.array(
                [item.embedding for item in response.data],
                dtype=np.float32,
            )

        else:
            raise ValueError(f"Unknown API provider: {self.api_provider}")

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.total_requests += 1
        self.total_latency_ms += elapsed_ms

        return embeddings

    def _deterministic_embedding(self, text: str) -> np.ndarray:
        """Generate a deterministic fake embedding from text (for simulation)."""
        import hashlib
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.embedding_dim).astype(np.float32)
        vec /= np.linalg.norm(vec)  # normalize
        return vec

    def embed_texts(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """
        Embed all texts by calling API SEQUENTIALLY, one batch at a time.

        ┌─────────────────────────────────────────────────────────────┐
        │ OPTIMIZATION OPPORTUNITIES:                                  │
        │                                                              │
        │ 1. [Week 9] asyncio + aiohttp:                               │
        │    Send ALL batches concurrently using async event loop.     │
        │    Since API calls are IO-bound (waiting for network),       │
        │    async can overlap all the waiting time.                   │
        │    Expected speedup: ~N_batches x (near-linear)              │
        │                                                              │
        │ 2. [Week 9] ThreadPoolExecutor:                               │
        │    Submit batches to a thread pool. Threads release GIL      │
        │    during IO wait, so this works well for network calls.     │
        │    Simpler than asyncio, nearly as effective.                │
        │                                                              │
        │ 3. [Week 2] Batch size tuning:                                │
        │    Larger batches = fewer API calls = less total latency.    │
        │    But APIs often have limits (e.g. 2048 tokens per batch).  │
        └─────────────────────────────────────────────────────────────┘
        """
        self.total_requests = 0
        self.total_latency_ms = 0.0

        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1

            if show_progress:
                print(f"  API batch {batch_num}/{total_batches} "
                      f"({len(batch)} texts)...", end="", flush=True)

            embeddings = self._call_api_single_batch(batch)
            all_embeddings.append(embeddings)

            if show_progress:
                print(f" done ({self.total_latency_ms / self.total_requests:.0f}ms avg)")

        result = np.vstack(all_embeddings)

        if show_progress:
            print(f"  Total: {self.total_requests} API calls, "
                  f"{self.total_latency_ms:.0f}ms total latency")

        return result

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query via API."""
        return self._call_api_single_batch([query])[0]


# ══════════════════════════════════════════════
# Backward-compatible alias
# ══════════════════════════════════════════════

BaselineEmbedder = LocalEmbedder


# ══════════════════════════════════════════════
# Quick self-test
# ══════════════════════════════════════════════

if __name__ == "__main__":
    # Test local embedder
    print("=== Local Embedder ===")
    local = LocalEmbedder(device="cpu")
    texts = ["The capital of France is Paris.", "Python is a programming language."]
    vecs = local.embed_texts(texts, show_progress=False)
    print(f"  Shape: {vecs.shape}")

    # Test API embedder (simulated)
    print("\n=== API Embedder (simulated, 100ms latency) ===")
    api = APIEmbedder(api_provider="simulated", simulated_latency_ms=100)
    texts_batch = [f"Test sentence number {i}" for i in range(320)]

    t0 = time.perf_counter()
    vecs = api.embed_texts(texts_batch, show_progress=True)
    elapsed = time.perf_counter() - t0

    print(f"\n  Total wall time: {elapsed:.2f}s")
    print(f"  {api.total_requests} sequential requests × 100ms = {api.total_requests * 0.1:.1f}s expected")
    print(f"  Shape: {vecs.shape}")
    print(f"\n  → With async optimization, this could be ~0.1s instead of {elapsed:.1f}s")
