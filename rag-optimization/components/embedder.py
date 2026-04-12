"""
Embedding generation — baseline sequential implementation.

Optimization targets:
  - Week 9:  asyncio / concurrent.futures for parallel batch encoding
  - Week 10: multiprocessing to distribute across CPU cores
  - Week 12: GPU-accelerated encoding

Uses sentence-transformers for the actual model inference.
Our optimization focus is on the *orchestration* (batching, parallelism),
not the model internals.
"""
import time
import numpy as np
from typing import Optional

import config


class BaselineEmbedder:
    """
    Sequential embedding generator.
    Loads model once, encodes texts in batches sequentially.
    """

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL,
        batch_size: int = config.EMBEDDING_BATCH_SIZE,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        """Lazy model loading."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        Baseline: sequential batch processing.

        Returns: np.ndarray of shape (len(texts), EMBEDDING_DIM), dtype float32
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
                normalize_embeddings=True,  # Pre-normalize for cosine sim
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


# ──────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    embedder = BaselineEmbedder()

    texts = [
        "The capital of France is Paris.",
        "Python is a programming language.",
        "Machine learning uses data to learn patterns.",
        "Paris is known for the Eiffel Tower.",
    ]

    t0 = time.perf_counter()
    vecs = embedder.embed_texts(texts, show_progress=False)
    elapsed = time.perf_counter() - t0

    print(f"Embedded {len(texts)} texts in {elapsed:.3f}s")
    print(f"Shape: {vecs.shape}, dtype: {vecs.dtype}")

    # Check similarity: "capital of France" should be close to "Paris + Eiffel"
    from components.similarity import cosine_sim_numpy
    scores = cosine_sim_numpy(vecs[0], vecs)
    for i, (text, score) in enumerate(zip(texts, scores)):
        print(f"  [{i}] {score:.4f} — {text}")
