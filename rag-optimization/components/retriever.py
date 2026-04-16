"""
Retriever: orchestrates query embedding + index search.
"""
import time
import numpy as np

import config
from components.embedder import BaselineEmbedder
from components.vector_index import BruteForceIndex, IVFIndex
from components.similarity import cosine_sim_python, cosine_sim_numpy


class Retriever:
    """
    Orchestrates the retrieval pipeline:
      query text -> embedding -> index search -> ranked results
    """

    def __init__(
        self,
        index,
        embedder: BaselineEmbedder,
        sim_fn=cosine_sim_numpy,
        top_k: int = config.TOP_K,
    ):
        self.index = index
        self.embedder = embedder
        self.sim_fn = sim_fn
        self.top_k = top_k

    def retrieve(self, query: str) -> dict:
        """
        Full retrieval for a single query.
        Returns dict with results and timing breakdown.
        """
        timings = {}

        # Step 1: Embed query
        t0 = time.perf_counter()
        query_vec = self.embedder.embed_query(query)
        timings["embed_ms"] = (time.perf_counter() - t0) * 1000

        # Step 2: Search index
        t0 = time.perf_counter()
        results = self.index.search(query_vec, k=self.top_k, sim_fn=self.sim_fn)
        timings["search_ms"] = (time.perf_counter() - t0) * 1000

        timings["total_ms"] = timings["embed_ms"] + timings["search_ms"]

        return {
            "query": query,
            "results": results,       # list of (doc_id, score)
            "timings": timings,
        }

    def retrieve_batch(
        self,
        queries: list[str],
        use_batch_embedding: bool = False,
    ) -> list[dict]:
        """
        Retrieve for a batch of queries.
        If use_batch_embedding=True, embed queries in one batch call.
        """
        if not use_batch_embedding:
            return [self.retrieve(q) for q in queries]

        if not hasattr(self.embedder, "embed_texts"):
            return [self.retrieve(q) for q in queries]

        t0 = time.perf_counter()
        query_vectors = self.embedder.embed_texts(queries, show_progress=False)
        total_embed_ms = (time.perf_counter() - t0) * 1000
        per_query_embed_ms = total_embed_ms / max(len(queries), 1)

        outputs = []
        for query, query_vec in zip(queries, query_vectors):
            t0 = time.perf_counter()
            results = self.index.search(query_vec, k=self.top_k, sim_fn=self.sim_fn)
            search_ms = (time.perf_counter() - t0) * 1000
            outputs.append(
                {
                    "query": query,
                    "results": results,
                    "timings": {
                        "embed_ms": per_query_embed_ms,
                        "search_ms": search_ms,
                        "total_ms": per_query_embed_ms + search_ms,
                    },
                }
            )
        return outputs
