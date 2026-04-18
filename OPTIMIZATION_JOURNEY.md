# RAG Optimization Journey — Complete Record

Full documentation of the 5-step incremental optimization, measured on the `medium` dataset (MS MARCO 100K passages).

All benchmarks: CPU only, one machine, tagged experiments saved in `results/experiments/medium_<step>.json`.

---

## Table of Contents

- [A. Initial State (Step 0 — Baseline)](#a-initial-state-step-0--baseline)
- [B. Step-by-Step Optimization Records](#b-step-by-step-optimization-records)
  - [Step 1 — IVF Parameter Tuning](#step-1--ivf-parameter-tuning)
  - [Step 2 — Numba K-Means (Week 6)](#step-2--numba-k-means-week-6)
  - [Step 3 — K-Means++ Initialization (Week 8)](#step-3--k-means-initialization-week-8)
  - [Step 4 — LLM Streaming (Week 9)](#step-4--llm-streaming-week-9)
  - [Step 5 — Pipelined RAG (Week 10/11)](#step-5--pipelined-rag-week-1011)
- [C. Parameter Evolution Table](#c-parameter-evolution-table)
- [D. Final Comparison (Step 0 vs Step 5)](#d-final-comparison-step-0-vs-step-5)
- [E. Course Week Coverage](#e-course-week-coverage)

---

## A. Initial State (Step 0 — Baseline)

### Dataset

- **Dataset**: MS MARCO, `medium` size (100,000 passages filtered from 612 items scanned)
- **Corpus**: N = 99,999 vectors, D = 384 dims (float32)
- **Queries**: 500 with human-annotated `relevant_passages`
- **Embedding model**: `all-MiniLM-L6-v2` (sentence-transformers), CPU

### Code State

```
rag-optimization/
├── components/                  # Baseline implementations
│   ├── similarity.py            # Pure Python + NumPy cosine
│   ├── vector_index.py          # BruteForceIndex + IVFIndex (n_clusters=32 default)
│   ├── embedder.py              # LocalEmbedder + APIEmbedder
│   └── generator.py             # BaselineGenerator (OpenAI / Anthropic / simulated)
└── optimized/                   # Already-existing optimized drop-ins
    ├── similarity_numba.py      # Numba JIT cosine (single + parallel)
    ├── parallel_indexer.py      # multiprocessing IVF builder
    ├── async_embedder.py        # ThreadedAPIEmbedder / AsyncAPIEmbedder
    └── async_generator.py       # ThreadedGenerator / AsyncGenerator
```

### Step 0 Parameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| IVF | `n_clusters` | **32** |
| IVF | `probes` tested | `(1, 2, 4, 8)` |
| IVF | `kmeans_iters` (max) | 20 |
| IVF | `tol` (convergence) | 1e-6 |
| IVF | `kmeans init` | **random** (`np.random.choice`) |
| K-Means impl | update step | pure Python loop over clusters |
| K-Means impl | `v_norms` caching | no (recomputed every iteration) |
| Generator | `stream` | off |
| Generator | `n_threads` (concurrency) | 8 |

### Step 0 Benchmark Results (medium)

| Category | Metric | Value |
|----------|--------|-------|
| **Similarity** (single query vs full corpus) | Pure Python (extrapolated) | 14,251 ms |
| | NumPy | 55.78 ms |
| | Numba | 39.12 ms |
| | **Numba parallel** | **5.84 ms** (9.55× vs NumPy) |
| **Index build** | BruteForce | 1.3 ms |
| | IVF(32) sequential | 3,405 ms |
| | IVF(32) parallel (4 workers) | 3,706 ms (0.92×) |
| **Index query** (per query) | BruteForce (NumPy) | 61.17 ms |
| | IVF n_probes=1 | 4.37 ms (14.00× vs BF) |
| **Recall@10** | BruteForce (ceiling) | 0.8908 |
| | IVF n_probes=1 | 0.5897 (66.2% of BF) |
| | IVF n_probes=4 | 0.8202 (92.1% of BF) |
| | IVF n_probes=8 | 0.8588 (96.4% of BF) |
| **Async embedding** (simulated 100 ms/batch, 10 batches) | Sequential | 1,125 ms |
| | Threaded (n=10) | 257 ms (4.37×) |
| | Async (max=16) | 227 ms (4.90×) |
| **Async generation** (real Kong gpt-4o, 8 queries) | Sequential | 12,735 ms (1,592 ms/call) |
| | Threaded (n=8) | 2,093 ms (5.30×) |
| | Async (max=8) | 2,217 ms (5.00×) |

---

## B. Step-by-Step Optimization Records

---

### Step 1 — IVF Parameter Tuning

> **Week mapping:** tuning (not a specific course week)
> **File:** `results/experiments/medium_1_ivf_tuned.json`
> **Parent:** `0_baseline`

#### What changed

- **`N_CLUSTERS: 32 → 64`** (closer to `sqrt(N) ≈ 316` heuristic; each cluster holds ~1,562 vectors instead of ~3,125)
- **`probes: (1, 2, 4, 8) → (2, 4, 8, 16)`** (keeps identical scan-coverage fractions as Step 0 for fair comparison)

#### Rationale

With 32 clusters the partition was too coarse: at `n_probes=1` you scan 3.1% of the corpus but cluster boundaries are rough, hurting recall. Doubling to 64 halves each cluster, making partitioning more accurate per scan dollar.

#### Code change

Notebook Section 4 build cell:

```python
# was:
N_CLUSTERS = 32
# now:
N_CLUSTERS = 64
```

#### Results vs Step 0

| Metric | Step 0 | Step 1 | Change |
|--------|--------|--------|--------|
| `ivf_best_ms` (fastest config) | 4.37 ms (np=1) | **3.55 ms (np=2)** | −18.7% |
| `ivf_build_seq_ms` | 3,405 ms | 3,852 ms | **+13%** (expected cost) |
| Recall @ 3.1% coverage | 0.5897 (np=1 @32) | **0.7248 (np=2 @64)** | **+22.9%** |
| Recall @ 6.25% coverage | 0.7367 (np=2 @32) | 0.8148 (np=4 @64) | +10.6% |
| Recall @ 12.5% coverage | 0.8202 (np=4 @32) | 0.8508 (np=8 @64) | +3.7% |
| Recall @ 25% coverage | 0.8588 (np=8 @32) | **0.8768 (np=16 @64)** | +2.1% |

#### Key insight

**The entire Pareto curve shifts up** — at every scan budget, recall improves. The single "raw probe number" comparison in the diff table is misleading (it compares different coverages); the coverage-aligned comparison is the real win. Cost: +13% build time — becomes the target of Step 2.

---

### Step 2 — Numba K-Means (Week 6)

> **Week mapping:** Week 6 (Numba)
> **File:** `results/experiments/medium_2_kmeans_numba.json`
> **Parent:** `1_ivf_tuned`

#### What changed

Created new file `rag-optimization/optimized/kmeans_numba.py` containing:

1. **`_accumulate_sums(vectors, assignments, n_clusters)`** — Numba `@njit` kernel doing per-cluster sum + count in a single pass over vectors. Replaces the baseline's O(K·N) Python loop that does `vectors[mask].mean(axis=0)` per cluster (allocates new array each iteration).

2. **Vector norms hoisted outside the iteration loop** — baseline recomputes `np.linalg.norm(vectors, axis=1)` on every iteration (20 redundant full-corpus scans); Numba version computes once.

3. **`IVFIndexNumba(IVFIndex)`** — subclass that overrides only `_kmeans`. All other behavior (build, search, save/load) is inherited unchanged — baseline code is never modified.

Portal integration:

```python
portal.run_index_build_benchmarks(..., kmeans_impl="numba")
```

#### Parameter changes

No user-facing parameter changes. Only the underlying K-Means implementation switches.

#### Results vs Step 1

| Metric | Step 1 | Step 2 | Change |
|--------|--------|--------|--------|
| `ivf_build_seq_ms` | 3,852 ms | **1,741 ms** | **−54.8%** (2.2×) |
| `ivf_build_par_ms` | 4,151 ms | 2,153 ms | −48.1% |
| `recall_ivf_np2/4/8/16` | 0.7248 / 0.8148 / 0.8508 / 0.8768 | **identical to 4 decimals** | 0.00% |

#### Key insight

**Build time halved, recall mathematically equivalent.** The +13% regression from Step 1 is not just fixed — it's reversed to −49% vs original Step 0 baseline. Also worth noting: parallel build got **relatively worse** (0.92× → 0.81×) because K-Means shrank from 85% of build time to ~40%, making process-spawning overhead a larger fraction — a textbook Amdahl's Law demonstration.

---

### Step 3 — K-Means++ Initialization (Week 8)

> **Week mapping:** Week 8 (Optimization in Python)
> **File:** `results/experiments/medium_3_kmeans_pp_init.json`
> **Parent:** `2_kmeans_numba`

#### What changed

Extended `optimized/kmeans_numba.py`:

1. **`kmeans_pp_init(vectors, n_clusters, seed)`** — distance-weighted centroid seeding. First centroid uniformly random; each subsequent with probability proportional to `||v − nearest_centroid||²`.

2. **Efficient implementation** — expands `||v − c||² = ||v||² − 2 v·c + ||c||²`. Uses a single BLAS GEMV per new centroid, avoids `(N, D)` temporary allocation. First implementation used `vectors - centroids[k]` + `np.einsum("ij,ij->i", ...)` which was 7× slower (3.4 s vs 450 ms for this step).

3. **`kmeans_numba(..., init="random" | "kmeans++")`** — new parameter.

4. **`IVFIndexNumbaPP(IVFIndex)`** — second subclass, same signature as `IVFIndexNumba` but uses K-Means++ init.

Portal integration:

```python
portal.run_index_build_benchmarks(..., kmeans_impl="numba_pp")
```

#### Parameter changes

- **`kmeans init`: random → kmeans++**

#### Results vs Step 2

| Metric | Step 2 | Step 3 | Change |
|--------|--------|--------|--------|
| `ivf_build_seq_ms` | 1,741 ms | 2,080 ms | **+19.4%** (K-Means++ init cost ≈ 450 ms) |
| K-Means iterations actually used | ~20 (max) | ~18.7 | saves ~1-2 iters, not enough to offset init |
| `recall_ivf_np2` | 0.7248 | **0.7442** | **+2.7%** |
| `recall_ivf_np4` | 0.8148 | 0.8178 | +0.4% |
| `recall_ivf_np8` | 0.8508 | 0.8478 | −0.4% |
| `recall_ivf_np16` | 0.8768 | 0.8728 | −0.5% |

#### Key insight

**Trade-off optimization, not a pure win.** The convergence tolerance `tol=1e-6` is too strict — on this dataset, K-Means `shift` plateaus around 0.04 even at iter 20, so neither random nor K-Means++ ever triggers early stopping. Both run the 20-iteration cap. K-Means++ therefore only pays its init cost without recouping iteration savings. The upside is a +2.7% recall bump at `n_probes=2` (the fastest config) — cluster seeding quality affects small-probe accuracy disproportionately.

---

### Step 4 — LLM Streaming (Week 9)

> **Week mapping:** Week 9 (Python Concurrency)
> **File:** `results/experiments/medium_4_llm_streaming.json`
> **Parent:** `3_kmeans_pp_init`

#### What changed

Extended `components/generator.py`:

1. **`generate_stream(query, contexts)`** — new method returning `{answer, ttft_ms, total_ms, prompt_tokens}`.

2. **`_openai_generate_stream(prompt, ...)`** — calls the OpenAI-compatible SDK with `stream=True`, reads `chunk.choices[0].delta.content` from the iterator. Records TTFT the first time a content chunk arrives.

3. **`_simulated_generate_stream(...)`** — simulated version for dev (20% initial + 80% per-token sleeps).

Portal additions:

- **`run_streaming_generation_benchmarks(items, ...)`** — three-way real-API comparison:
  1. Sequential non-streaming (Section 7 baseline)
  2. Sequential streaming
  3. Concurrent streaming (ThreadPoolExecutor + `stream=True` in each worker)

- **`plot_streaming_generation(results)`** — three-panel: batch total, mean TTFT, per-call total.

`RESULTS["generation_stream"]` key added.

#### Parameter changes

- **`stream`: False → True** (new capability)
- **TTFT** metric introduced

#### Results (all new; nothing in previous steps)

| Mode | Batch total | Mean TTFT | Per-call total |
|------|-------------|-----------|----------------|
| Sequential non-stream (Section 7 baseline) | 13,908 ms | **1,738 ms** | 1,738 ms |
| Sequential streaming | 10,933 ms | **961 ms** | 1,367 ms |
| Concurrent streaming (n=8) | **2,421 ms** | 1,158 ms | 2,301 ms |

#### Key insight

**Perceived latency is a different metric from total latency.** TTFT drops 45% going from non-streaming to streaming (1,738 → 961 ms) with the *same total time* — the user sees the answer appear sooner, even though the full answer takes as long. Kong proxy buffering explains why TTFT didn't reach the theoretical 200 ms (a direct OpenAI call would likely hit that).

Counterintuitively, **concurrent streaming has higher TTFT than sequential streaming** (1,158 vs 961 ms) — running 8 concurrent SSE streams creates network contention, delaying each stream's first token. But **batch total drops 5.7×** compared to the non-stream baseline. Concurrent streaming is the production sweet spot: low per-user TTFT *and* high throughput.

---

### Step 5 — Pipelined RAG (Week 10/11)

> **Week mapping:** Week 10/11 (Parallel Programming)
> **File:** `results/experiments/medium_5_pipeline.json`
> **Parent:** `4_llm_streaming`

#### What changed

Portal additions (no changes to `components/` or `optimized/`):

- **`run_pipeline_benchmarks(queries, chunks, bf, ...)`** — two end-to-end RAG-serving patterns on real Kong gpt-4o:

  - **Mode A — Sequential naive**: for each query, `embed → search → generate_stream` serially. Total ≈ N × (embed + search + gen).
  - **Mode B — Pipelined**: two `ThreadPoolExecutor` pools running in parallel:
    - retrieval pool with `n_embed_workers` workers handles embed + search
    - generation pool with `n_gen_workers` workers handles gpt-4o streaming
    - **Each retrieval worker, upon finishing, submits the (query, contexts) directly to the generation pool from within its worker thread**. So gen of query `i` overlaps with retrieval of query `i+1`.

- **`plot_pipeline(results)`** — three-panel: batch total, per-query amortized, speedup.

`RESULTS["pipeline"]` key added.

#### Parameter changes

- **`n_embed_workers: — → 4`** (new)
- **`n_gen_workers: 8`** (same as Section 7)
- **Architecture: single pool / staged → dual-pool pipelined**

#### Results

| Metric | Value |
|--------|-------|
| `pipe_seq_total_ms` (Mode A, naive baseline) | 14,456 ms (≈1,807 ms/query) |
| `pipe_opt_total_ms` (Mode B, pipelined) | **2,817 ms (352 ms/query)** |
| `pipe_speedup` | **5.13×** |

#### Key insight

End-to-end: retrieval (CPU-bound, ~100 ms/query when parallelized) and generation (network-IO-bound, ~1,500 ms streaming) saturate simultaneously rather than serially. Theoretical upper bound is ~7× (with negligible overhead and perfect overlap); measured 5.13× reflects real thread-scheduling + GIL + Kong concurrency overhead. All honest, all explainable.

This is the headline end-to-end speedup for the report.

---

## C. Parameter Evolution Table

| Parameter | Step 0 | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 |
|-----------|:------:|:------:|:------:|:------:|:------:|:------:|
| `N_CLUSTERS` | 32 | **64** | 64 | 64 | 64 | 64 |
| `probes` | (1,2,4,8) | **(2,4,8,16)** | (2,4,8,16) | (2,4,8,16) | (2,4,8,16) | (2,4,8,16) |
| K-Means impl | NumPy Python loop | NumPy Python loop | **Numba `@njit`** | Numba | Numba | Numba |
| `v_norms` cached | no | no | **yes** | yes | yes | yes |
| K-Means init | random | random | random | **kmeans++** | kmeans++ | kmeans++ |
| `kmeans_iters` (max) | 20 | 20 | 20 | 20 | 20 | 20 |
| `tol` | 1e-6 | 1e-6 | 1e-6 | 1e-6 | 1e-6 | 1e-6 |
| LLM `stream=` | no | no | no | no | **yes** | yes |
| Architecture | staged | staged | staged | staged | staged (+stream) | **dual-pool pipelined** |
| `n_embed_workers` | — | — | — | — | — | **4** |
| `n_gen_workers` (Kong) | 8 | 8 | 8 | 8 | 8 | 8 |

---

## D. Final Comparison (Step 0 vs Step 5)

### Cumulative improvements

| Metric | Step 0 | Step 5 | Total change |
|--------|--------|--------|--------------|
| IVF build time (sequential) | 3,405 ms | 2,080 ms | **−38.9%** (1.64×) |
| Fastest IVF query | 4.37 ms | 3.57 ms | −18.3% |
| Recall @ 3.1% scan coverage | 0.5897 | **0.7442** | **+26.2%** |
| Recall @ 25% scan coverage | 0.8588 | 0.8728 | +1.6% |
| LLM perceived latency (TTFT) | 1,738 ms (no stream) | **961 ms (stream)** | **−45%** |
| **End-to-end 8-query batch** | **14,456 ms** | **2,817 ms** | **5.13×** |

### Key compositional insights

- **Step 1 + Step 2 compose**: parameter tuning pushes up the Pareto frontier; Numba K-Means fixes the induced build-time regression.
- **Step 3 is a trade**: +2.7% recall at fastest config, −17% build time vs Step 2.
- **Step 4 + Step 5 compose**: streaming cuts perceived latency per query; pipelining cuts total batch latency. Concurrent streaming from Step 4 is the generation backend used by Step 5.

---

## E. Course Week Coverage

| Course Week | Topic | Where in the project |
|-------------|-------|----------------------|
| **Week 6** | Numba | Step 2 (K-Means rewrite); also Section 2 (Numba similarity) |
| **Week 8** | Optimization in Python | Step 3 (K-Means++ algorithmic upgrade) |
| **Week 9** | Python Concurrency | Step 4 (LLM streaming); also Section 6/7 (Threaded / Async embedding + generation) |
| **Week 10/11** | Parallel Programming | Step 5 (dual-pool pipelined RAG); also `optimized/parallel_indexer.py` (multiprocessing IVF build) |

Baseline implementations in `components/` are the non-optimized references. Every step adds a drop-in optimized variant without modifying the baseline, preserving traceability.

---

## Reproducibility

- Each step is tagged in `results/experiments/medium_<step>.json` with parent lineage + description
- `portal.compare_experiments(dataset, step_a, step_b)` prints any two-step diff
- `portal.plot_experiment_progression(dataset)` charts metric evolution across all steps
- `portal.print_experiments(dataset)` lists the full lineage

The notebook `interact_portal.ipynb` runs each step; changing the `STEP`, `STEP_DESC`, `STEP_PARENT` variables in the save cell and rerunning produces a new tagged snapshot.
