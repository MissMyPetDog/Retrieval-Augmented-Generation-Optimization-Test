# RAG Pipeline — Optimization Map

A stage-by-stage view of the full RAG pipeline, with each optimization point clearly labeled. Under every stage you can see its initial settings and the concrete changes applied at each step.

**Dataset for all numbers:** `medium` — 99,999 vectors × 384 dims, 500 queries. CPU only.

---

## Full Pipeline (Offline + Online)

```
============================================================================
  OFFLINE PHASE  (runs once, produces vectors.npy + index files on disk)
============================================================================

      Raw MS MARCO passages
              |
              v
   +----------------------+
   | [P1] Preprocess      |   -- clean text, chunk, tokenize
   |      (eager loading) |
   +----------------------+
              |
              v
   +----------------------+
   | [P2] Embed corpus    |   -- sentence-transformers all-MiniLM-L6-v2
   |                      |   -- batched, CPU (or GPU if available)
   +----------------------+
              |
              v
   +----------------------+   <<< STEP 1: n_clusters tuning
   | [P3] Build index     |   <<< STEP 2: Numba K-Means
   |      (K-Means + IVF) |   <<< STEP 3: K-Means++ init
   +----------------------+
              |
              v
       vectors.npy + index_bruteforce.pkl + index_ivf.pkl


============================================================================
  ONLINE PHASE  (runs per user query)
============================================================================

       User query text
              |
              v
   +----------------------+
   | [Q1] Embed query     |   -- same model, text -> 384-dim vector
   |      (CPU, ~300 ms)  |
   +----------------------+                           +-----------------+
              |                                       |                 |
              v                                       |   STEP 5:       |
   +----------------------+   <<< STEP 1: probes   <  |   pipelined     |
   | [Q2] Search index    |       per-config          |   RAG overlaps  |
   |      (BF or IVF)     |                           |   Q1/Q2/Q3      |
   |                      |   Section 2: similarity   |   stages across |
   |                      |     (NumPy/Numba/Numba-   |   multiple      |
   |                      |      parallel)            |   queries       |
   +----------------------+                           |                 |
              |                                       |                 |
              v                                       |                 |
   +----------------------+                           |                 |
   | [Q3] Generate answer |   <<< Section 7: async <  |                 |
   |      (call Kong      |       concurrent calls    |                 |
   |      gpt-4o)         |   <<< STEP 4: streaming   |                 |
   +----------------------+                           +-----------------+
              |
              v
         Answer to user
```

Legend of what each step touches:

| Step | Stage(s) | Course Week |
|------|----------|:-----------:|
| **1** `1_ivf_tuned` | P3 (build), Q2 (search) | tuning |
| **2** `2_kmeans_numba` | P3 (build) | **Week 6** |
| **3** `3_kmeans_pp_init` | P3 (build) | **Week 8** |
| **4** `4_llm_streaming` | Q3 (generate) | **Week 9** |
| **5** `5_pipeline` | Q1 + Q2 + Q3 overlap | **Week 10/11** |
| (Section 2) | Q2 kernel | Week 6 |
| (Section 6) | P2 (if using API) | Week 9 |
| (Section 7) | Q3 concurrency | Week 9 |

---

## [P1] Preprocess — Document Cleaning & Chunking

### What happens here

Load raw MS MARCO passages, strip HTML / extra whitespace, split into overlapping chunks.

### Initial state (Step 0)

| Parameter | Value |
|-----------|-------|
| Loading | eager (`load_passages`) — reads everything into RAM |
| Chunking | sequential Python loop |
| `chunk_size`, `chunk_overlap` | from `config.py` |

### Steps that touch this stage

**None in this project.** All 5 steps leave preprocessing alone. Future optimization opportunity: `itertools`-based generator streaming (Week 3).

### Current result

Preprocess time isn't on the critical path for medium (~0.2 s total) and not benchmarked here.

---

## [P2] Embed Corpus — Dense Vector Representation (Offline)

### What happens here

Run the embedding model over every chunk, producing an `(N, D)` float32 matrix saved as `vectors.npy`.

### Initial state (Step 0)

| Parameter | Value |
|-----------|-------|
| Model | `all-MiniLM-L6-v2` |
| Device | CPU (forced by `setup_cpu_only()`) |
| `EMBEDDING_BATCH_SIZE` | from `config.py` |
| API vs local | local (`LocalEmbedder`) |
| Concurrency | sequential |

### Steps that touch this stage

**None in the core 5 steps.** BUT Section 6 of the notebook benchmarks **API-based** concurrent embedding (`ThreadedAPIEmbedder` / `AsyncAPIEmbedder` — simulated 100 ms/batch). That is a Week 9 (Concurrency) demo. It's included as part of the project's coverage.

### Section 6 result (simulated 10 batches, 100 ms each)

| Mode | Total | Speedup |
|------|-------|---------|
| Sequential | 1,113 ms | 1.0x |
| Threaded (n=10) | 232 ms | **4.80x** |
| Async (max=16) | 227 ms | **4.90x** |

---

## [P3] Build Index — K-Means + IVF (Offline)

This is where **Steps 1, 2, 3** all concentrate. Three distinct changes, each on top of the previous.

### What happens here

1. Run K-Means clustering over the N vectors to get K centroids
2. Assign each vector to its nearest centroid (inverted lists)
3. Save `index_ivf.pkl` with centroids + inverted lists

### Initial state (Step 0)

| Parameter | Value |
|-----------|-------|
| Index type | `IVFIndex` from `components/vector_index.py` |
| **`n_clusters`** | **32** |
| **`n_probes`** (default) | 4 |
| **`kmeans_iters`** (max) | 20 |
| **`tol`** (convergence) | 1e-6 |
| **K-Means init** | **random** (`rng.choice(n, size=32)`) |
| K-Means update step | pure Python loop: `for k in range(32): vectors[mask].mean(0)` |
| `v_norms` caching | no — recomputed every iteration |
| Parallel build | `ParallelIVFBuilder` (multiprocessing) available but unused |

### Step 0 result

| Metric | Value |
|--------|-------|
| IVF(32) sequential build | **3,405 ms** |
| IVF(32) parallel build (4 workers) | 3,706 ms (0.92x — **slower**; Amdahl dominates) |

### Changes across steps

```
Step 1: n_clusters 32 -> 64
Step 2:                        + Numba K-Means + v_norms cache
Step 3:                                                      + K-Means++ init
```

Each layer is cumulative (code paths preserved; switched via `kmeans_impl="..."` argument).

#### Step 1 (`1_ivf_tuned`) — Parameter tuning

| Parameter | Before | After |
|-----------|:------:|:-----:|
| `n_clusters` | 32 | **64** |
| `probes` (test range) | (1,2,4,8) | **(2,4,8,16)** |

New numbers: sequential build **3,852 ms** (+13% vs Step 0, expected cost of more centroids).

#### Step 2 (`2_kmeans_numba`) — Numba K-Means + norm caching

New code: `rag-optimization/optimized/kmeans_numba.py`

- `_accumulate_sums()` `@njit` kernel: single-pass per-cluster sum + count
- Vector norms hoisted to loop entry (computed once)
- `IVFIndexNumba(IVFIndex)` subclass overrides only `_kmeans`

| Parameter | Before | After |
|-----------|:------:|:-----:|
| K-Means update | Python loop | **Numba `@njit`** |
| `v_norms` | recomputed | **cached** |

New numbers: sequential build **1,741 ms** (-54.8% vs Step 1; -48.9% vs Step 0).

#### Step 3 (`3_kmeans_pp_init`) — K-Means++ seeding

Extension of the same file.

| Parameter | Before | After |
|-----------|:------:|:-----:|
| K-Means init | random | **kmeans++** (distance-weighted) |

New numbers: sequential build **2,080 ms** (+19% vs Step 2; tol=1e-6 prevents early convergence, so init cost isn't recouped by fewer iterations).

### Build time progression

```
Step 0 (32, random):    3,405 ms     +------+
Step 1 (64, random):    3,852 ms     +----------+       (+13%)
Step 2 (64, numba):     1,741 ms     +---+                   (-55% vs Step 1)
Step 3 (64, ++):        2,080 ms     +----+                  (+19% vs Step 2, trade for recall)
```

---

## [Q1] Embed Query — Per-Query Vectorization (Online)

### What happens here

Convert the user's query text into a 384-dim vector using the same model that embedded the corpus.

### Initial state (Step 0)

| Parameter | Value |
|-----------|-------|
| Method | `LocalEmbedder.embed_query()` |
| Device | CPU |
| Batch? | no (single query) |

### Steps that touch this stage

**Step 5** uses `embed_query()` in parallel via 4 retrieval workers. The method itself isn't rewritten — it's just dispatched concurrently via `ThreadPoolExecutor`.

### Result

Per-query CPU embedding: ~300 ms.

---

## [Q2] Search Index — Cosine Similarity Over Vectors (Online)

This stage has two orthogonal axes of optimization.

### What happens here

Given one query vector:
- **BruteForce**: score against all N corpus vectors, return top-K
- **IVF**: find `n_probes` nearest centroids → score query against only those clusters' vectors → return top-K

### Initial state (Step 0)

| Parameter | Value |
|-----------|-------|
| Indexes | BruteForce, IVF(32) |
| Similarity kernel | `cosine_sim_numpy` (BLAS GEMV) |
| Top-K selection | `np.argpartition` |

### Step 0 result (per query)

| Config | Latency |
|--------|:-------:|
| BruteForce (NumPy) | 61.17 ms |
| IVF n_probes=1 | 4.37 ms (14.0x vs BF) |
| IVF n_probes=4 | — (not shown in step 0 diff) |
| IVF n_probes=8 | — |

### Axis 1: Similarity kernel (Section 2 benchmarks, independent of index)

Full-corpus single-query scan (N=99,999, D=384):

| Kernel | Latency |
|--------|:-------:|
| Pure Python (extrapolated) | 14,251 ms |
| NumPy (BLAS) | 55.78 ms |
| Numba single-thread | 39.12 ms |
| **Numba parallel** (`prange`) | **5.84 ms** (9.55x vs NumPy) |

This is measured independently and **not wired into IVF search** by default — it's a demo of what the `sim_fn` abstraction can accept.

### Axis 2: IVF parameters — Step 1

| Parameter | Before | After |
|-----------|:------:|:-----:|
| `n_clusters` | 32 | 64 (each cluster is smaller, finer partitions) |
| `probes` tested | (1,2,4,8) | (2,4,8,16) |

### Step 1 result (per query)

| Config | Before (Step 0) | After (Step 1) | Change |
|--------|:---------------:|:--------------:|:------:|
| Fastest IVF config | 4.37 ms (np=1) | **3.55 ms (np=2)** | -18.7% |
| 12.5% corpus coverage | 15.36 ms (np=4 @32) | 7.73 ms (np=8 @64) | **-50%** |

### After all steps (current state, per query)

| Config | Latency |
|--------|:-------:|
| BruteForce (NumPy) | 57.80 ms |
| IVF(64) n_probes=2 | 3.57 ms (16.21x) |
| IVF(64) n_probes=4 | 7.73 ms (8.52x) |
| IVF(64) n_probes=8 | 12.99 ms (4.45x) |
| IVF(64) n_probes=16 | 28.94 ms (2.15x) |

### Recall (Q2 quality, verified by Section 5)

| Config | Recall@10 | % of BF ceiling |
|--------|:---------:|:---------------:|
| BruteForce (ceiling) | 0.8908 | 100.0% |
| IVF(64) n_probes=2 | 0.7442 | 83.5% |
| IVF(64) n_probes=4 | 0.8178 | 91.8% |
| IVF(64) n_probes=8 | 0.8508 | 95.5% |
| IVF(64) n_probes=16 | 0.8728 | 98.0% |

### Currently NOT done at Q2 (friend's territory)

These would squeeze out more ms/query if desired:
- `norm_cache` (pre-compute corpus vector norms at build time)
- `np_gather` (NumPy fancy-index candidate collection in IVF.search)
- `batch_embed` (M query GEMM instead of M sequential GEMVs)

Not doing these is intentional — they're friend's optimization axis.

---

## [Q3] Generate Answer — LLM Call (Online)

This is the **biggest absolute time saver** in the whole pipeline (gen dominates at ~1.5 s/call).

### What happens here

Format a prompt with the query and retrieved chunks → call gpt-4o through NYU's Kong proxy → return the answer.

### Initial state (Step 0)

| Parameter | Value |
|-----------|-------|
| Generator | `BaselineGenerator(api_provider="openai", base_url=Kong, ...)` |
| `model_name` | `gpt-4o` |
| `max_tokens` | 128 |
| `stream` | **off** |
| Concurrency pattern | sequential (Section 7 baseline) |

### Section 7 benchmark (still Step 0 state; part of the "initial toolkit")

| Mode | Batch total | Speedup |
|------|:-----------:|:-------:|
| Sequential | 12,735 ms | 1.0x |
| Threaded (n=8) | 2,093 ms | 5.30x |
| Async (max=8) | 2,217 ms | 5.00x |

Concurrent threaded/async was already available at Step 0 — it's not one of the 5 numbered steps.

### Step 4 (`4_llm_streaming`) — add streaming

New code in `components/generator.py`:

| Parameter | Before | After |
|-----------|:------:|:-----:|
| `stream=` | False | **True** (new `generate_stream()` method) |
| Measured metric | `total_ms` only | **`ttft_ms` + `total_ms`** |

### Step 4 result (8 real Kong gpt-4o calls)

| Mode | Batch total | Mean TTFT | Perceived-latency change |
|------|:-----------:|:---------:|:-----------------------:|
| Sequential NON-stream | 13,908 ms | **1,738 ms** | baseline |
| Sequential STREAM | 10,933 ms | **961 ms** | **-45%** |
| Concurrent STREAM (n=8) | **2,421 ms** | 1,158 ms | batch 5.7x + TTFT -33% |

Full-answer latency isn't much changed (Kong buffers chunks), but the moment the user *sees the first token* drops from 1.7 s to 0.96 s.

---

## [CROSS-STAGE] End-to-End Pipeline Architecture

### Initial architecture (Step 0 through Step 4)

```
[for each batch of queries]

   | batch-embed all queries |   (fast, CPU batch GEMV)
              |
              v
   | search each (fast, parallel over queries)
              |
              v
   | fire all gens concurrently, wait  |
```

This is a **staged** architecture: each stage fully completes before the next begins.

### Step 5 architecture

```
Retrieval pool (4 workers)              Generation pool (8 workers)
    |                                         ^
    |  embed + search                         |
    |   for query i                           |
    |                                         |
    |  when done,                             |
    +-- submit (query_i, ctx) to gen pool ----+
    |                                         |
    |  meanwhile query i+1                    |
    |  starts on another                      |
    |  retrieval worker                       |
```

Retrieval of query i+1 **overlaps** with generation of query i. CPU-bound retrieval and IO-bound generation saturate at the same time.

### Step 5 result (8 queries end-to-end, real Kong gpt-4o)

| Mode | Total | Per-query amortized |
|------|:-----:|:------------------:|
| Sequential naive | 14,456 ms | 1,807 ms/q |
| **Pipelined** | **2,817 ms** | **352 ms/q** |
| **Speedup** | **5.13x** | |

---

## Global Before/After

| Stage | Step 0 | Latest step | Change |
|-------|--------|-------------|--------|
| [P3] IVF build (sequential) | 3,405 ms | 2,080 ms | **-38.9%** |
| [P3] IVF build (parallel) | 3,706 ms | 2,436 ms | -34.3% |
| [Q2] Fastest IVF query | 4.37 ms | 3.57 ms | -18.3% |
| [Q2] Recall @ 3.1% scan | 0.5897 | **0.7442** | **+26.2%** |
| [Q2] Recall @ 25% scan | 0.8588 | 0.8728 | +1.6% |
| [Q3] TTFT (perceived latency) | 1,738 ms | **961 ms** | **-45%** |
| Cross: end-to-end 8-query batch | 14,456 ms | **2,817 ms** | **5.13x** |

---

## Stage-to-Step Cross-Reference (one-page map)

| Pipeline stage | Initial setting | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 |
|----------------|:---------------:|:------:|:------:|:------:|:------:|:------:|
| [P1] Preprocess | eager load | -- | -- | -- | -- | -- |
| [P2] Corpus embed | CPU sequential | -- | -- | -- | -- | -- |
| [P3] Build index | K=32, random init, Python K-Means | **K=64** | **Numba K-Means** | **K-Means++** | -- | -- |
| [Q1] Embed query | CPU `embed_query` | -- | -- | -- | -- | **4 parallel workers** |
| [Q2] Search | IVF probes=(1,2,4,8), NumPy | **probes=(2,4,8,16)** | -- | -- | -- | -- |
| [Q3] Generate | non-streaming, threaded pool | -- | -- | -- | **stream=True** | uses Step 4's streaming |
| Pipeline composition | staged | staged | staged | staged | staged | **dual-pool overlap** |

---

## How to read this document

- `[PX]` stages are **offline** — they run when building the knowledge base
- `[QX]` stages are **online** — they run for every user query
- The 5 "Steps" are cumulative, each saved as `results/experiments/medium_<step>.json`
- Every optimization is additive — baseline code is untouched; new variants live in `optimized/`
- Every parameter change is traceable: check the `STEP_DESC` in each saved experiment

For full benchmark numbers per step, see [OPTIMIZATION_JOURNEY.md](OPTIMIZATION_JOURNEY.md).
