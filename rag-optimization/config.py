"""
Global configuration for the RAG optimization project.
"""
import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INDEX_DIR = os.path.join(PROJECT_ROOT, "indexes")

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
DATASET_NAME = "ms_marco"
DATASET_SUBSET = "v1.1"
NUM_PASSAGES = 100_000        # Start with 100K for dev; scale up later
PASSAGE_FILE = os.path.join(DATA_DIR, "passages.jsonl")
QUERY_FILE = os.path.join(DATA_DIR, "queries.jsonl")

# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────
CHUNK_SIZE = 256              # Tokens per chunk
CHUNK_OVERLAP = 32            # Overlap between chunks

# ──────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_BATCH_SIZE = 256

# ──────────────────────────────────────────────
# Vector Index
# ──────────────────────────────────────────────
IVF_NUM_CLUSTERS = 256        # Number of IVF clusters
IVF_NUM_PROBES = 8            # Clusters to search at query time
TOP_K = 10                    # Number of results to retrieve

# ──────────────────────────────────────────────
# Benchmark
# ──────────────────────────────────────────────
BENCHMARK_NUM_QUERIES = 1000
BENCHMARK_WARMUP = 5
BENCHMARK_REPEATS = 50
