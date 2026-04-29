"""
Shared utilities for the 3-way pipeline comparison scripts.

Every script in this folder:
  1. Calls setup_cpu_only() BEFORE importing torch / numba / components
  2. Loads the same knowledge base + same N test queries
  3. Reports results in the same JSON schema (see make_result_schema)
  4. Writes output to results/comparisons/<script_name>.json
"""
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------
# Path setup: add rag-optimization to sys.path for component imports
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_DIR = PROJECT_ROOT / "rag-optimization"
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))


# ---------------------------------------------------------------------
# CPU-only setup (must be called before heavy imports)
# ---------------------------------------------------------------------
def setup_cpu_only(verbose: bool = False) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NUMBA_DISABLE_CUDA"] = "1"
    if sys.platform == "win32":
        import tempfile
        short_cache = Path(tempfile.gettempdir()) / "numba_cache"
        short_cache.mkdir(parents=True, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = str(short_cache)
        if verbose:
            print(f"[common] Numba cache -> {short_cache}")


# ---------------------------------------------------------------------
# Data loading (same corpus + queries for every config)
# ---------------------------------------------------------------------
def load_knowledge_base(dataset: str = "medium"):
    data_dir = RAG_DIR / "data" / dataset
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset}' not found at {data_dir}. "
            f"Build it first: python build_knowledge_base.py --download --size {dataset} --data_dir data/{dataset}"
        )
    vectors = np.load(data_dir / "vectors.npy")
    with open(data_dir / "chunks.jsonl", "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    with open(data_dir / "queries.jsonl", "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f]
    return vectors, chunks, queries, data_dir


def get_test_queries(queries: list, n: int = 8) -> list:
    """First N queries that have relevance judgments. Deterministic."""
    good = [q for q in queries if q.get("relevant_passages")]
    return good[:n]


def build_chunk_to_passage_text(chunks: list, data_dir: Path) -> dict:
    """chunk_id -> source passage text, for recall computation."""
    passages = {}
    with open(data_dir / "passages.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            passages[p["id"]] = p["text"]
    return {c["id"]: passages.get(c["source_id"], c["text"]) for c in chunks}


# ---------------------------------------------------------------------
# ChatGPT-4o generator (real API)
# ---------------------------------------------------------------------
def make_chatgpt_generator(max_tokens: int = 128):
    from components.generator import BaselineGenerator
    api_key = os.environ["OPENAI_API_KEY"]
    return BaselineGenerator(
        api_provider="openai",
        api_key=api_key,
        model_name="gpt-4o",
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def compute_recall(retrieved_chunk_ids: list, relevant_passages: list,
                   chunk_to_passage: dict) -> float:
    if not relevant_passages:
        return 0.0
    retrieved_texts = {chunk_to_passage.get(did, "") for did in retrieved_chunk_ids}
    relevant_set = set(relevant_passages)
    return len(retrieved_texts & relevant_set) / len(relevant_set)


def mean(values):
    return float(np.mean(values)) if values else 0.0


# ---------------------------------------------------------------------
# Result output
# ---------------------------------------------------------------------
def save_result(result: dict, script_name: str) -> Path:
    out_dir = PROJECT_ROOT / "results" / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{script_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(result), f, indent=2)
    print(f"\n[common] Saved: {out_path}")
    return out_path


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, Path):
        return str(o)
    return o


def print_summary(result: dict) -> None:
    m = result["per_query_mean"]
    k = result.get("config", {}).get("n_retrieved", "?")
    print(f"\n--- {result['name']} ---")
    print(f"  Build:           {result['build_time_ms']:8.1f} ms")
    print(f"  Mean embed:      {m['embed_ms']:8.1f} ms")
    print(f"  Mean search:     {m['search_ms']:8.2f} ms")
    print(f"  Mean gen:        {m['gen_ms']:8.1f} ms")
    print(f"  Mean total/q:    {m['total_ms']:8.1f} ms")
    print(f"  Batch total:     {result['batch_total_ms']:8.1f} ms ({result['n_queries']} queries)")
    print(f"  Mean Recall@{k}:  {m['recall@k']:8.4f}")


def warmup_embedder(embedder, verbose: bool = True) -> None:
    """Do one throwaway embed call to trigger lazy model load before timing."""
    if verbose:
        print("Warming up embedder (loading MiniLM model)...")
    t0 = time.perf_counter()
    _ = embedder.embed_query("warmup")
    if verbose:
        print(f"  Model loaded in {(time.perf_counter()-t0)*1000:.0f} ms")