# RAG System Optimization — Complete Project Guide

---

## 1. Project Description

This project builds a Retrieval-Augmented Generation (RAG) system entirely from scratch. No existing RAG frameworks are used — no LangChain, no LlamaIndex, no FAISS. Every core component is implemented manually in Python.

The purpose is not to build a production RAG system. The purpose is to use RAG as a real-world workload that contains multiple types of computational bottlenecks, each of which can be optimized using a different advanced Python tool from the course curriculum.

The system answers questions by:
1. Converting the question into a numerical vector (embedding)
2. Comparing that vector against hundreds of thousands of document vectors to find the most relevant ones
3. Feeding the relevant documents to a language model to generate an answer

Each of these steps involves a different type of computation — CPU-bound math, IO-bound API calls, memory-intensive data processing — and therefore benefits from a different optimization strategy.

---

## 2. Dataset

**MS MARCO (Microsoft Machine Reading Comprehension)**

- Source: `microsoft/ms_marco` on HuggingFace, version 1.1, train split
- Content: Real search queries from Bing with human-annotated relevant passages
- Why this dataset: It provides ground-truth relevance labels, allowing us to verify that speed optimizations do not degrade retrieval quality
- Download is deterministic: fixed ordering, no shuffling, SHA256 hash verification ensures every team member has identical data

---

## 3. System Architecture

The system has two phases:

### 3.1 Offline Phase — Build Knowledge Base

This runs once. It processes raw documents into a searchable vector index.

```
Raw Passages
    │
    ▼
[Preprocess] ── clean text, split into chunks
    │
    ▼
[Embed] ── convert each chunk to a 384-dimensional vector
    │
    ▼
[Build Index] ── organize vectors for fast search
    │
    ▼
Searchable Knowledge Base (vectors.npy + index files)
```

### 3.2 Online Phase — Answer Queries

This runs for every user question.

```
User Query
    │
    ▼
[Embed Query] ── convert question to vector (same model as offline)
    │
    ▼
[Search Index] ── compute similarity between query vector and all document vectors
    │                find the Top-K most similar documents
    │
    ▼
[Generate Answer] ── send query + retrieved documents to LLM
    │
    ▼
Answer
```

### 3.3 Where the Time Goes

| Step | Type | Typical Latency | Bottleneck |
|------|------|----------------|------------|
| Embed Query | CPU or GPU compute | 2-10ms | Model inference |
| Search Index | CPU or GPU compute | 1-120ms depending on method | **Primary optimization target** — matrix math over N vectors |
| Generate Answer | Network IO | 500-2000ms | Waiting for LLM API response |
| Preprocess (offline) | CPU | seconds to minutes | Text processing over entire corpus |
| Embed All Docs (offline) | CPU or GPU | seconds to minutes | Model inference over entire corpus |
| Build Index (offline) | CPU | seconds | K-Means clustering |

---

## 4. Code Structure

```
rag-optimization/
│
├── build_knowledge_base.py          # One-command: download + preprocess + embed + index
├── run_test.py                      # One-command: benchmark all implementations
├── config.py                        # All parameters in one place
│
├── components/                      # BASELINE implementations
│   ├── preprocessor.py              #   Text loading, cleaning, chunking
│   ├── embedder.py                  #   Embedding: local model + API-based
│   ├── similarity.py                #   Cosine similarity: Pure Python + NumPy
│   ├── vector_index.py              #   BruteForce index + IVF index
│   ├── retriever.py                 #   Orchestrates: embed query → search → rank
│   └── generator.py                 #   LLM answer generation: simulated + real API
│
├── optimized/                       # OPTIMIZED implementations
│   ├── similarity_cython.pyx        #   Cython cosine similarity
│   ├── similarity_numba.py          #   Numba JIT cosine similarity (single + parallel)
│   ├── similarity_gpu.py            #   CuPy GPU cosine similarity
│   ├── parallel_indexer.py          #   Multiprocessing IVF index builder
│   ├── async_embedder.py            #   Concurrent embedding (threading + asyncio + multiprocessing)
│   └── async_generator.py           #   Concurrent LLM generation (threading + asyncio)
│
├── benchmarks/
│   ├── benchmark_runner.py          #   Automated benchmark harness
│   └── evaluate.py                  #   Recall@K, MRR quality metrics
│
├── data/
│   └── download_data.py             #   Deterministic data downloader + hash verification
│
├── TEAMMATE_GUIDE.md
├── README.md
└── requirements.txt
```

### 4.1 The Relationship Between `components/` and `optimized/`

`components/` contains baseline code that works correctly but is not optimized. Every function uses the simplest possible implementation.

`optimized/` contains faster versions of the same functions. Each optimized file corresponds to a specific course tool (Cython, Numba, GPU, etc.).

The critical design rule: **optimized functions have the exact same interface as baseline functions.** Same inputs, same outputs. This means you can swap any baseline function with its optimized version without changing any other code.

Example:
```python
# Baseline
from components.similarity import cosine_sim_numpy
retriever = Retriever(index=bf, sim_fn=cosine_sim_numpy)

# Swap to optimized — only this one line changes
from optimized.similarity_numba import cosine_sim_numba_parallel
retriever = Retriever(index=bf, sim_fn=cosine_sim_numba_parallel)
```

---

## 5. Detailed Optimization Map

### 5.1 Similarity Computation — `components/similarity.py`

**What it does:** Given 1 query vector (D dimensions) and N document vectors (N × D matrix), compute cosine similarity between the query and every document. Return N similarity scores.

**Why it matters:** This runs for every query. At 500K documents, naive code takes 37 seconds per query.

**Baseline implementations in `components/similarity.py`:**

| Function | How it works |
|----------|-------------|
| `cosine_sim_python()` | Two nested for-loops. Iterates element by element. Pure Python, no libraries. |
| `cosine_sim_numpy()` | Single line: `corpus @ query`. NumPy delegates to BLAS (optimized C/Fortran library). |

**Optimized implementations in `optimized/`:**

| File | Function | Course Week | What it does |
|------|----------|-------------|-------------|
| `similarity_cython.pyx` | `cosine_sim_cython()` | Week 5 (Cython) | Same loop as Pure Python but compiled to C. Uses typed memoryviews, disables bounds checking, calls C-level math. |
| `similarity_numba.py` | `cosine_sim_numba()` | Week 6 (Numba) | Same loop with `@njit` decorator. Numba compiles it to machine code at runtime. |
| `similarity_numba.py` | `cosine_sim_numba_parallel()` | Week 6 + 10 (Numba + Parallel) | Same as above but outer loop uses `prange` — Numba distributes iterations across CPU cores. |
| `similarity_gpu.py` | `cosine_sim_gpu()` via `GPUSimilarityEngine` | Week 12 (GPU) | Transfers vectors to GPU memory, computes matrix-vector multiply using CUDA cores via CuPy. |

**How to add your own optimization:** Create a new file in `optimized/`, write a function with this exact signature:
```python
def your_cosine_sim(query_vec: np.ndarray, corpus_matrix: np.ndarray) -> np.ndarray:
    # query_vec: shape (D,)
    # corpus_matrix: shape (N, D)
    # return: shape (N,) — similarity scores
```

### 5.2 Vector Index — `components/vector_index.py`

**What it does:** Organizes vectors so that search can be done faster than comparing against every single one.

**Baseline implementations:**

| Class | How it works | Complexity |
|-------|-------------|-----------|
| `BruteForceIndex` | Compares query against ALL N vectors. Guaranteed to find the best results. | O(N × D) per query |
| `IVFIndex` | First clusters vectors into K groups using K-Means. At query time, only searches the closest `n_probes` clusters. | O(n_probes × N/K × D) per query |

**Optimization opportunities:**

| Component | File to modify | Course Week | What to optimize |
|-----------|---------------|-------------|-----------------|
| K-Means clustering | `components/vector_index.py` → `_kmeans()` | Week 6 (Numba) / Week 8 (Optimization) | The update step has a Python loop over clusters. Rewrite with Numba. Try K-Means++ initialization for faster convergence. |
| Cluster assignment | `optimized/parallel_indexer.py` | Week 10 (Parallel) | Already implemented with multiprocessing. Currently slower than sequential due to Amdahl's Law (K-Means is the real bottleneck). Fixing K-Means speed makes this worthwhile. |
| IVF search logic | `components/vector_index.py` → `search()` | Week 5 (Cython) | The search method has Python-level list operations (collecting candidates from multiple clusters). Rewrite as a single Cython function to eliminate Python overhead. |

### 5.3 Embedding Generation — `components/embedder.py`

**What it does:** Converts text into numerical vectors using a neural network model.

**Baseline implementations:**

| Class | How it works |
|-------|-------------|
| `LocalEmbedder` | Loads `all-MiniLM-L6-v2` model, encodes texts in sequential batches. Device can be CPU or GPU (auto-detected). |
| `APIEmbedder` | Sends text to a remote embedding API. Each batch waits for the previous one to finish. Default: simulated with 100ms latency. Also supports OpenAI API. |

**Optimized implementations in `optimized/async_embedder.py`:**

| Class | Course Week | What it does |
|-------|-------------|-------------|
| `ThreadedAPIEmbedder` | Week 9 (Concurrency) | Sends all API batches via ThreadPoolExecutor. Threads release GIL during network wait, so all batches run in parallel. |
| `AsyncAPIEmbedder` | Week 9 (Concurrency) | Same idea using asyncio event loop. Lower overhead than threads for many concurrent requests. |
| `ParallelLocalEmbedder` | Week 10 (Parallel) | Splits texts across processes, each loads own model copy. For CPU-bound local encoding. |

**Key insight for the report:** API embedding is IO-bound (waiting for network). Local embedding is CPU/GPU-bound (running model inference). The optimal concurrency tool is different for each:
- IO-bound → threading or asyncio (GIL not a problem because CPU is idle during wait)
- CPU-bound → multiprocessing (GIL prevents threads from using multiple cores)

### 5.4 Answer Generation — `components/generator.py`

**What it does:** Takes the user's query plus retrieved documents, sends them to an LLM, returns the answer.

**Baseline implementation:**

| Class | How it works |
|-------|-------------|
| `BaselineGenerator` | Calls LLM API sequentially. Each query blocks until the full response is received. Default: simulated with 500ms latency. Also supports Anthropic and OpenAI APIs. |

**Optimized implementations in `optimized/async_generator.py`:**

| Class | Course Week | What it does |
|-------|-------------|-------------|
| `ThreadedGenerator` | Week 9 (Concurrency) | Sends N generation requests to ThreadPoolExecutor simultaneously. |
| `AsyncGenerator` | Week 9 (Concurrency) | Same with asyncio. Uses semaphore to control concurrency (respect API rate limits). |

**Additional optimization opportunity (not yet implemented):**

| Optimization | Course Week | Description |
|-------------|-------------|-------------|
| Streaming responses | Week 9 | Instead of waiting for the full LLM response, stream tokens as they arrive. Reduces perceived latency. |
| Prompt optimization | Week 2 (Performance Tips) | Shorter prompts = fewer tokens = faster and cheaper. Remove redundant context, truncate intelligently. |

### 5.5 Preprocessing — `components/preprocessor.py`

**What it does:** Loads raw passages, cleans text, splits into overlapping chunks.

**Baseline implementation:** Eager loading (reads all passages into memory), simple Python loop for chunking.

**Optimization opportunities (not yet implemented):**

| Optimization | Course Week | Description |
|-------------|-------------|-------------|
| Generator-based streaming | Week 3 (itertools) | Replace eager loading with generators + `itertools.islice`. Process documents in streaming fashion. Reduces peak memory from O(N) to O(batch_size). |
| Parallel chunking | Week 10 (Parallel) | Split documents across processes, chunk in parallel. |
| PySpark preprocessing | Week 13 (PySpark) | For very large datasets (millions of docs), distribute preprocessing across a Spark cluster. |

---

## 6. Setup Instructions

### 6.1 Install Dependencies

```bash
git clone https://github.com/MissMyPetDog/Retrieval-Augmented-Generation-Optimization-Test.git
cd Retrieval-Augmented-Generation-Optimization-Test/rag-optimization
pip install -r requirements.txt
```

For GPU support (optional, requires NVIDIA GPU + CUDA):
```bash
pip install cupy-cuda12x
```

For Cython compilation:
```bash
python setup_cython.py build_ext --inplace
```

### 6.2 Build Knowledge Base

```bash
# Small (5K docs) — for development and debugging
python build_knowledge_base.py --download --size small --data_dir data/small

# Medium (100K docs) — for standard benchmarks
python build_knowledge_base.py --download --size medium --data_dir data/medium

# Large (500K docs) — for final benchmarks
python build_knowledge_base.py --download --size large --data_dir data/large
```

Add `--device cpu` to force CPU-only embedding, or `--device cuda` for GPU.

If HuggingFace download is slow, add your token:
```bash
python build_knowledge_base.py --download --size small --data_dir data/small --hf_token YOUR_TOKEN
```

### 6.3 Verify Data Consistency

After building, verify your data matches teammates:
```bash
python build_knowledge_base.py --verify --data_dir data/small
```

This checks all files exist and SHA256 hashes match `metadata.json`.

---

## 7. How to Run Tests

### 7.1 Device Control

The `--device` flag isolates GPU effects from code optimization effects:

```bash
# CPU only — measures pure code optimization, no GPU anywhere
python run_test.py --data_dir data/small --device cpu

# With GPU — includes CuPy tests, embedding on GPU
python run_test.py --data_dir data/small --device cuda
```

When `--device cpu` is set:
- `CUDA_VISIBLE_DEVICES` is set to empty, completely hiding the GPU
- Embedding model is forced to CPU
- CuPy tests are skipped
- Every result reflects code optimization only

When `--device cuda` is set:
- GPU is available for embedding and CuPy similarity
- Each result is tagged `[CPU]` or `[GPU]` so you know which hardware ran it

### 7.2 Run Similarity Benchmark Only (Fast)

```bash
python run_test.py --data_dir data/small --device cpu --similarity_only
```

This tests only the `cosine_sim_*` functions without running the full retrieval pipeline. Useful for quick iteration when developing a new similarity optimization.

### 7.3 Run Full Benchmark

```bash
python run_test.py --data_dir data/small --device cpu
```

This tests similarity computation AND end-to-end retrieval (quality + latency).

Results are saved to `data/small/test_results_cpu.json`.

### 7.4 Test API Concurrency Optimizations

These use simulated API latency — no real API key needed:

```bash
# Test embedding API optimization (sequential vs threaded vs async)
python optimized/async_embedder.py

# Test LLM generation optimization (sequential vs threaded vs async)
python optimized/async_generator.py
```

To use real APIs, change the initialization in the scripts:
```python
# Simulated (default)
api = APIEmbedder(api_provider="simulated")

# Real OpenAI
api = APIEmbedder(api_provider="openai", api_key="sk-xxx")
```

```python
# Simulated (default)
gen = BaselineGenerator(api_provider="simulated")

# Real Anthropic
gen = BaselineGenerator(api_provider="anthropic", api_key="sk-ant-xxx")

# Real OpenAI
gen = BaselineGenerator(api_provider="openai", api_key="sk-xxx")
```

### 7.5 Test on Larger Data

```bash
python build_knowledge_base.py --download --size medium --data_dir data/medium
python run_test.py --data_dir data/medium --device cpu
```

Larger datasets amplify optimization effects — the gap between fast and slow implementations becomes more visible.

---

## 8. How to Add Your Own Optimization

### 8.1 The Process

1. Pick a baseline function from `components/`
2. Create a new file in `optimized/` (or modify an existing one)
3. Write your optimized version with the **same function signature**
4. Run `run_test.py` to compare

### 8.2 Quick Test Template

Save this as a Python script or paste into a notebook to test any optimization:

```python
import sys
sys.path.insert(0, ".")
import numpy as np
import time

# Load data
vectors = np.load("data/small/vectors.npy")
query = vectors[0]
corpus = vectors[1:]

# Baseline
from components.similarity import cosine_sim_numpy
t0 = time.perf_counter()
for _ in range(10):
    baseline_result = cosine_sim_numpy(query, corpus)
baseline_ms = (time.perf_counter() - t0) / 10 * 1000

# Your optimization
from optimized.your_file import your_function
t0 = time.perf_counter()
for _ in range(10):
    optimized_result = your_function(query, corpus)
optimized_ms = (time.perf_counter() - t0) / 10 * 1000

# Results
print(f"Baseline:  {baseline_ms:.2f}ms")
print(f"Optimized: {optimized_ms:.2f}ms")
print(f"Speedup:   {baseline_ms / optimized_ms:.1f}x")

# Correctness check — MUST pass
diff = np.max(np.abs(baseline_result - optimized_result))
print(f"Max diff:  {diff:.2e}")
assert diff < 1e-4, "RESULTS DO NOT MATCH — optimization is incorrect"
```

### 8.3 Optimization Targets Reference

| What to optimize | Baseline file | Where to put optimization | Course tool | Bottleneck type |
|-----------------|--------------|--------------------------|------------|----------------|
| Cosine similarity | `components/similarity.py` | `optimized/` | Cython, Numba, GPU | CPU-bound |
| K-Means in IVF build | `components/vector_index.py` `_kmeans()` | `optimized/` | Numba, scipy.optimize | CPU-bound |
| IVF search logic | `components/vector_index.py` `search()` | `optimized/` | Cython | CPU-bound (irregular) |
| Text preprocessing | `components/preprocessor.py` | `optimized/` | itertools, multiprocessing | CPU-bound / Memory |
| Embedding (local model) | `components/embedder.py` `LocalEmbedder` | `optimized/async_embedder.py` | multiprocessing, GPU | CPU/GPU-bound |
| Embedding (API) | `components/embedder.py` `APIEmbedder` | `optimized/async_embedder.py` | asyncio, threading | IO-bound |
| LLM generation | `components/generator.py` `BaselineGenerator` | `optimized/async_generator.py` | asyncio, threading | IO-bound |
| Parallel index build | `optimized/parallel_indexer.py` | same file | multiprocessing, shared_memory | CPU-bound |

---

## 9. Summary of Commands

| Task | Command |
|------|---------|
| Install dependencies | `pip install -r requirements.txt` |
| Build knowledge base (small) | `python build_knowledge_base.py --download --size small --data_dir data/small` |
| Build knowledge base (medium) | `python build_knowledge_base.py --download --size medium --data_dir data/medium` |
| Build knowledge base (large) | `python build_knowledge_base.py --download --size large --data_dir data/large` |
| Verify data | `python build_knowledge_base.py --verify --data_dir data/small` |
| Benchmark (CPU only) | `python run_test.py --data_dir data/small --device cpu` |
| Benchmark (with GPU) | `python run_test.py --data_dir data/small --device cuda` |
| Benchmark (similarity only) | `python run_test.py --data_dir data/small --device cpu --similarity_only` |
| Test async embedding | `python optimized/async_embedder.py` |
| Test async generation | `python optimized/async_generator.py` |
| Compile Cython | `python setup_cython.py build_ext --inplace` |
