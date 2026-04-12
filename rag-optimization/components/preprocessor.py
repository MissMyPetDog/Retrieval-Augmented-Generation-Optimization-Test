"""
Baseline document preprocessor: loading, cleaning, chunking.

Optimization targets (later):
  - Week 2-3: itertools + generators for memory-efficient streaming
  - Week 9:   multiprocessing for parallel chunk generation
  - Week 13:  PySpark for distributed preprocessing
"""
import json
import re
import time
from typing import Generator

import config


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

def load_passages(path: str = config.PASSAGE_FILE) -> list[dict]:
    """Load all passages into memory (baseline: eager loading)."""
    passages = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            passages.append(json.loads(line))
    return passages


def load_passages_lazy(path: str = config.PASSAGE_FILE) -> Generator[dict, None, None]:
    """Generator-based lazy loading (optimized version)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def load_queries(path: str = config.QUERY_FILE) -> list[dict]:
    """Load evaluation queries."""
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


# ──────────────────────────────────────────────
# Text Cleaning
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)               # Collapse whitespace
    text = re.sub(r"[^\w\s.,;:!?'\"-]", "", text)  # Remove odd chars
    return text


# ──────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────

def chunk_text_baseline(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping word-level chunks.
    Baseline: plain Python loop.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def process_passages(
    passages: list[dict],
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[dict]:
    """
    Full preprocessing pipeline: clean + chunk all passages.
    Returns list of {id, text, source_id}.
    """
    chunks = []
    chunk_id = 0

    for passage in passages:
        cleaned = clean_text(passage["text"])
        text_chunks = chunk_text_baseline(cleaned, chunk_size, overlap)

        for chunk_text in text_chunks:
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": chunk_text,
                "source_id": passage["id"],
            })
            chunk_id += 1

    return chunks


# ──────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    sample = "This is a sample passage. " * 100
    t0 = time.perf_counter()
    result = chunk_text_baseline(sample, chunk_size=50, overlap=10)
    elapsed = time.perf_counter() - t0
    print(f"Chunked into {len(result)} pieces in {elapsed:.4f}s")
    print(f"First chunk preview: {result[0][:80]}...")
