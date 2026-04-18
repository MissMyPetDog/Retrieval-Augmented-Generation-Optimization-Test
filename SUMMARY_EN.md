# RAG Pipeline Optimization — Complete Project Summary

**Project:** Advanced Python (Spring 2026) — Retrieval-Augmented Generation optimization
**Dataset:** MS MARCO `medium` (99,999 passages × 384 dims, 500 queries with relevance judgments)
**Constraint:** CPU only (all GPUs hidden; every speedup reflects pure code-level optimization)
**Course weeks covered:** Week 6 (Numba), Week 8 (Optimization), Week 9 (Concurrency), Week 10/11 (Parallel Programming)

> *Figures referenced below are stored under `results/figures/` (export from the Jupyter notebooks when needed).*

---

## Table of Contents

1. [RAG Workflow](#1-rag-workflow)
2. [Pure Baseline (BruteForce)](#2-pure-baseline-bruteforce)
3. [Optimization Journey](#3-optimization-journey)
   - [Step 1 — IVF Parameter Tuning](#step-1--ivf-parameter-tuning)
   - [Step 2 — Numba K-Means (Week 6)](#step-2--numba-k-means-week-6)
   - [Step 3 — K-Means++ Initialization (Week 8)](#step-3--k-means-initialization-week-8)
   - [Step 4 — LLM Streaming (Week 9)](#step-4--llm-streaming-week-9)
   - [Step 5 — Pipelined RAG (Week 10/11)](#step-5--pipelined-rag-week-1011)
   - [Step 6 — Peer's Query-Path Optimizations](#step-6--peers-query-path-optimizations)
4. [Final 3-Way Comparison](#4-final-3-way-comparison)
5. [User-Facing Impact](#5-user-facing-impact)
6. [Conclusions & Key Insights](#6-conclusions--key-insights)

---

## 1. RAG Workflow

A Retrieval-Augmented Generation pipeline splits into two phases.

### Offline (build once)

```
Raw MS MARCO passages
     │
     ▼
[Preprocess]  clean text, split into overlapping chunks
     │
     ▼
[Embed corpus]  MiniLM-L6-v2 -> (N, 384) float32 matrix
     │
     ▼
[Build index]  K-Means clustering + IVF inverted lists
     │
     ▼
vectors.npy + index_ivf.pkl on disk
```

### Online (per user query)

```
User query text
     │
     ▼
[Embed query]  same model -> (384,) vector       ~300 ms CPU
     │
     ▼
[Search index]  cosine similarity -> top-K chunks  ~5-60 ms
     │
     ▼
[Generate answer]  query + chunks -> gpt-4o API   ~1500 ms (network)
     │
     ▼
Answer
```

Each stage is a different optimization surface: the search stage is CPU-bound matrix math; the generation stage is IO-bound network wait. The project targets each with a different course tool.

---

## 2. Pure Baseline (BruteForce)

The "zero-optimization" reference. No IVF, no caching, no concurrency, no streaming.

### Configuration

| Component | Setting |
|-----------|---------|
| Index | `BruteForceIndex` — linear scan over all 100K vectors |
| Similarity | `cosine_sim_numpy` with on-the-fly norm computation |
| `use_precomputed_norms` | `False` (explicitly disabled) |
| Generator | `BaselineGenerator` with `stream=False` |
| Architecture | Serial per-query: embed → search → generate → next |

### Result (8 real queries on Kong gpt-4o, K=5 retrieved per query)

| Metric | Value |
|--------|:-----:|
| Index build | 130 ms |
| Embed per query | 8.1 ms |
| Search per query | **62.9 ms** |
| Generate per query | 1754 ms |
| Total per query | 1825 ms |
| **Batch total (8 queries)** | **14.6 s** |
| Recall@5 | 0.75 |

### Why this baseline matters

BruteForce guarantees 100% retrieval recall — it scans every document vector. Its value is as the **correctness ceiling** and as the "if you do nothing, this is the cost" reference.

---

## 3. Optimization Journey

Each step is an **additive** change on top of the previous step. Baseline code is never modified — new variants live under `optimized/` and are selected via parameters.

---

### Step 1 — IVF Parameter Tuning

**What it does:** switch from linear BruteForce to inverted-file indexing. Cluster the corpus with K-Means into K partitions; at query time scan only the `n_probes` nearest partitions instead of all 100K vectors.

**Key changes:**
- `N_CLUSTERS: 32 → 64` (closer to `sqrt(N) ≈ 316` heuristic, gives finer partitions)
- `probes: (1, 2, 4, 8) → (2, 4, 8, 16)` — test at matching scan-coverage ratios for fair comparison

**Course alignment:** Tuning (not a specific course week) — informed by algorithm design.

**Result (500-query quality eval, K=10 retrieved):**

| Config | Recall@10 | Query latency |
|--------|:---:|:---:|
| IVF(64) n_probes=2 | 0.7442 | 3.5 ms |
| IVF(64) n_probes=4 | 0.8178 | 6.8 ms |
| **IVF(64) n_probes=8** | **0.8508** | **13.0 ms** |
| IVF(64) n_probes=16 | 0.8728 | 26.9 ms |
| BruteForce (ceiling) | 0.8908 | 63 ms |

**Same-coverage comparison to the original 32-cluster version:**

| Scan coverage | Step 0 (32 clusters) | Step 1 (64 clusters) | Change |
|:---:|:---:|:---:|:---:|
| 3.1% | 0.590 | **0.724** | **+22.9%** |
| 6.25% | 0.737 | 0.815 | +10.6% |
| 12.5% | 0.820 | 0.851 | +3.7% |
| 25% | 0.859 | 0.873 | +1.6% |

**Key insight:** the whole Pareto frontier shifts up. At any scan budget, recall is strictly better than before. Cost: build time +13% due to more centroids.

---

### Step 2 — Numba K-Means (Week 6)

**What it does:** rewrite the K-Means clustering kernel with Numba JIT compilation. Fixes two inefficiencies in the baseline:

1. **Vector-norm recomputation every iteration** — the baseline recomputes `np.linalg.norm(corpus)` on each of 20 K-Means iterations. We cache it once.
2. **Python `for k in range(K):` update loop** with per-cluster boolean indexing — allocates a new array per cluster. We replace it with a Numba `@njit` single-pass accumulator.

**Code location:** new file `rag-optimization/optimized/kmeans_numba.py` defining `IVFIndexNumba` as a subclass of `IVFIndex` that only overrides `_kmeans`. Baseline code untouched.

**Course alignment:** **Week 6 — Numba JIT compilation**.

**Result:**

| Metric | Step 1 (NumPy K-Means) | Step 2 (Numba) | Change |
|--------|:---:|:---:|:---:|
| IVF build (sequential) | 3852 ms | **1741 ms** | **−54.8%** |
| Recall@10 at each n_probes | unchanged | unchanged | **0.00%** |

**Key insight:** 2.2× faster index build with **mathematically identical** clustering output (verified to 4 decimal places). Net vs the original Step 0 baseline: **−49%**.

---

### Step 3 — K-Means++ Initialization (Week 8)

**What it does:** replace random K-Means initialization with K-Means++. Instead of picking K centroids uniformly at random, pick them one at a time with probability proportional to squared distance from the nearest already-chosen centroid. Result: initial centroids are spread out, giving better local optima.

**Implementation note:** first version used `vectors - centroids[k]` + `np.einsum` which was 7× slower due to large temporary allocations. Rewritten with the expanded distance formula `||v−c||² = ||v||² − 2v·c + ||c||²` — each iteration is a single BLAS GEMV, no temporary array allocation.

**Course alignment:** **Week 8 — Optimization in Python** (algorithmic improvement).

**Result — this is a trade, not a pure win:**

| Metric | Step 2 (random init) | Step 3 (K-Means++) | Change |
|--------|:---:|:---:|:---:|
| IVF build | 1741 ms | 2080 ms | +19% |
| Recall@10 at n_probes=2 | 0.7248 | **0.7442** | **+2.7%** |
| Recall@10 at n_probes=4-16 | — | essentially identical | ±0.5% |

**Honest reflection:** the tolerance threshold `tol=1e-6` is too strict for this dataset, so both random init and K-Means++ run the full 20 iterations without triggering early stopping. K-Means++ pays an extra 450 ms init cost without recouping it through iteration savings. The upside is a measurable recall improvement at the fastest query configuration (n_probes=2).

---

### Step 4 — LLM Streaming (Week 9)

**What it does:** enable `stream=True` on the gpt-4o API so tokens are returned as generated rather than all at once. Introduces a new metric: **TTFT (Time To First Token)** — when the user sees the first word of the answer.

**Code location:** `components/generator.py::generate_stream()` method using OpenAI SDK's streaming iterator.

**Course alignment:** **Week 9 — Python Concurrency** (iterator-based streaming).

**Result (8 real Kong gpt-4o calls):**

| Mode | Batch total | Mean TTFT | Perceived latency change |
|------|:---:|:---:|:---:|
| Sequential non-streaming | 13,908 ms | **1,738 ms** | (baseline) |
| Sequential streaming | 10,933 ms | **961 ms** | **−45%** |
| Concurrent streaming (n=8 workers) | **2,421 ms** | 1,158 ms | Batch 5.7× + TTFT −33% |

**Key insight:** the full generation time is unchanged (gpt-4o generates at its own pace), but the user sees the answer *start appearing* much sooner. For a chat UI this is the metric that matters.

**Unexpected finding:** concurrent streaming has *higher* TTFT than sequential streaming (1,158 vs 961 ms). Running 8 simultaneous SSE streams creates network contention through the Kong proxy, delaying each stream's first token. This is an honest real-world tradeoff between per-user latency and batch throughput.

---

### Step 5 — Pipelined RAG (Week 10/11)

**What it does:** the **headline architectural optimization**. Instead of running stages serially (retrieve all, then generate all), two thread pools run simultaneously:

- **Retrieval pool** (4 workers) handles CPU-bound embed + search
- **Generation pool** (8 workers) handles IO-bound gpt-4o streaming

Each retrieval worker, upon completion, *submits the result directly into the generation pool from within its own worker thread*. This means retrieval of query `i+1` overlaps with generation of query `i` — CPU and network saturate simultaneously.

```
Retrieval pool (4 workers)              Generation pool (8 workers)
─────────────────────────               ─────────────────────────
  embed + search query i  ──submit──→  gpt-4o streaming for i
  embed + search query i+1             (concurrent with others)
                                        gpt-4o streaming for i+1
  ...                                   ...
```

**Course alignment:** **Week 10/11 — Parallel Programming** (multi-pool thread coordination).

**Result (8 real queries, end-to-end, includes embed + search + full generation):**

| Mode | Batch total | Per-query amortized | Speedup |
|------|:---:|:---:|:---:|
| Sequential naive | 14,456 ms | 1,807 ms | 1.0× |
| **Pipelined (4 retrieve + 8 gen)** | **2,817 ms** | **352 ms** | **5.13×** |

**Theoretical ceiling ~7×;** measured 5.13× reflects real thread-scheduling overhead, Python GIL transitions, and Kong proxy concurrency limits. Honestly measured, honestly explained.

**Why threading (not multiprocessing) is correct here:** both stages release the GIL — gpt-4o network waits naturally do, and PyTorch embedding inference does internally. Multiprocessing would add process-spawn overhead (~400 ms on Windows) with no added parallelism. Indeed, the project's `ParallelIVFBuilder` (multiprocessing-based) runs *slower* than its sequential counterpart — a negative result that validates the choice of threads here.

---

### Step 6 — Peer's Query-Path Optimizations

A teammate on a shared repository contributed three query-path micro-optimizations that were merged and integrated without modifying the baseline code.

**What was added:**
1. **`use_precomputed_norms`** — L2 norms of corpus vectors computed once at build time and cached, instead of recomputed on every query
2. **`use_numpy_candidate_gather`** — IVF candidate collection via `np.concatenate` instead of Python `list.extend`
3. **`cosine_sim_numba_parallel_precomputed`** — Numba parallel similarity kernel that accepts precomputed norms as a third argument

**Integration:** merged via `git merge origin/main` into a separate `integrate-friend` branch. All three changes live in existing files (`components/vector_index.py`, `optimized/similarity_numba.py`) as **opt-in flags**, preserving backward compatibility with Steps 1-5 code.

**Ablation study (IVF n_probes=2, medium dataset):**

| Variant | sim_fn | norm cache | np gather | Latency | vs A |
|---------|--------|:---:|:---:|:---:|:---:|
| **A. flags off** | NumPy | ✗ | ✗ | 3.91 ms | 1.0× |
| **B. + norm cache** | NumPy | ✓ | ✗ | **1.78 ms** | **2.19×** |
| **C. + np gather** | NumPy | ✓ | ✓ | **1.77 ms** | **2.21×** |
| D. + Numba parallel precomputed | Numba par | ✓ | ✓ | 2.06 ms | 1.90× ❌ |

**Key findings:**
- **Norm cache alone captures 95% of the gain** (3.91 → 1.78 ms). The single-line optimization of caching immutable corpus norms.
- **np gather is marginal** (+0.5%) at this scan size (~3000 candidates). Python list overhead is small relative to the work.
- **Variant D is a negative result** — Numba parallel with precomputed norms is *slower* than NumPy on IVF's reduced scan size. BLAS GEMV on ~3000 vectors is hard to beat with hand-rolled `prange`. This confirms a central project thesis: **the right tool depends on operation shape**. Numba parallel wins on large full-corpus scans (Section 2: 10× vs NumPy on 100K vectors) but loses on IVF-filtered small subsets.

**Unexpected bonus — BruteForce became 10× faster:**

Because `use_precomputed_norms` defaults to `True` after the merge, `BruteForceIndex.search` automatically benefits. The post-merge BF latency drops from **57.80 ms to 5.85 ms** — a 10× speedup from a merge, not from our own optimization effort. Honest to report as a "free win from integration."

---

## 4. Final 3-Way Comparison

Three self-contained Python scripts under `comparisons/` run the full 8-query RAG pipeline end-to-end and report identical metrics.

### Configurations

| Config | Index | sim_fn | Norm cache | Np gather | Generator | Architecture |
|--------|-------|--------|:---:|:---:|-----------|--------------|
| 1. BruteForce | `BruteForceIndex` | `cosine_sim_numpy` | ✗ | — | non-stream | serial |
| 2. IVF default | `IVFIndex(32, np=4)` | `cosine_sim_numpy` | ✗ | ✗ | non-stream | serial |
| 3. Fully optimized | `IVFIndexNumbaPP(64, np=8)` | `cosine_sim_numpy` | ✓ | ✓ | **streaming** | **pipelined (4+8)** |

### Results (real Kong gpt-4o, 8 queries, K=5 retrieved)

| Metric | Config 1: BruteForce | Config 2: IVF default | Config 3: Fully optimized | Speedup 3 vs 1 |
|--------|:---:|:---:|:---:|:---:|
| Build time | 130 ms | 3,530 ms | 1,972 ms | 0.07× |
| Embed ms/query | 8.1 | 7.6 | 37.2 | — |
| **Search ms/query** | 62.9 | 16.7 | **14.5** | **4.35×** |
| Gen ms/query | 1,754 | 1,722 | 1,598 | — |
| Total ms/query | 1,825 | 1,747 | 1,650 | 1.11× |
| **Batch total** | 14,601 ms | 13,974 ms | **3,770 ms** | **3.87×** |
| Recall@5 (8 queries) | 0.75 | 0.50 | 0.625 | — |

*The Recall@5 row has low resolution because the sample size is only 8 queries (each is binary found/not-found). For robust recall see Step 1 numbers based on 500 queries.*

![comparison figure](results/figures/three_way_comparison.png)

### The Three Critical Findings

**1. BruteForce → IVF default = only 4% faster batch (14.60 s → 13.97 s).**
Swapping the algorithm alone barely helps. Generation dominates batch time at ~1.7 s/query × 8 = 13.6 s. IVF's 4× search speedup (63 → 17 ms) saves only ~0.4 s total, **invisible against the generation wall**.

**2. IVF default → Fully optimized = 3.7× faster batch.**
The real speedup comes from **changing the architecture**, not the algorithm. Pipelined streaming lets 8 gen calls run simultaneously while retrieval overlaps them.

**3. The algorithmic win is still real at the search layer (4.35×).**
It just doesn't show up in end-to-end wall-clock for an LLM-dominated pipeline. For a non-LLM retrieval system (like classic search), it would be the dominant metric.

**Central insight:** *when one stage dominates total latency, optimizing the fast stages has no end-to-end effect. Only rearranging the architecture to overlap stages provides user-visible speedup.*

---

## 5. User-Facing Impact

Translating backend metrics into user experience.

### Scenario A: Single user, single question

| Stage | Baseline | Fully optimized | User impact |
|-------|:---:|:---:|:---:|
| Query embed (CPU) | 300 ms | 300 ms | — |
| Search | 63 ms | 10 ms | — |
| Generation to first word (TTFT) | **1,738 ms** | **961 ms** | **−45%** |
| **Time until answer starts appearing** | ~2,100 ms | **~1,270 ms** | **32% shorter wait** |
| Time to full answer | 2,100 ms | 1,750 ms | 17% shorter |

A user perceives the response as "1.3 s" instead of "2.1 s" — the difference between "snappy" and "laggy" in chat UIs.

### Scenario B: Burst of 8 queries (multi-turn chat or batch)

| Pattern | Total wait |
|---------|:---:|
| All queries serial (original) | **14.6 s** |
| Pipelined + streaming (optimized) | **2.8 s** |
| **Speedup** | **5.13×** |

A user asking 8 questions in quick succession waits 3 seconds instead of 15.

### Scenario C: Multi-user serving (10 concurrent users, one pipeline)

| Pattern | Last user's wait |
|---------|:---:|
| Original (serial dispatch) | ~18 s |
| Optimized (pipelined concurrent) | ~3 s |
| **Reduction** | **~83%** |

The same hardware handles roughly 5× more concurrent load without the tail user waiting minutes.

### Quality is preserved

| Metric | Baseline | Optimized | Change |
|--------|:---:|:---:|:---:|
| Recall@10 (500 queries) | 0.8908 | 0.8508 (IVF np=8) | −4.5% |
| MRR | 0.4937 | 0.4754 | −3.7% |

A ~4% relative recall drop for a 5× batch speedup is a favorable trade. Choosing `n_probes=16` instead would give **Recall@10 = 0.8728 (98% of BF)** at roughly double the search latency — still well under the generation-dominated total.

---

## 6. Conclusions & Key Insights

### Numbers worth remembering

| Dimension | Best achievement |
|-----------|:---:|
| Index build time | 3.41 s → 2.08 s (**−39%**, via Numba K-Means) |
| Per-query search | 63 ms → 14 ms (**4.35×**, IVF + norm cache + np gather) |
| Perceived latency (TTFT) | 1,738 ms → 961 ms (**−45%**, via streaming) |
| End-to-end batch | 14,601 ms → 3,770 ms (**3.87×**, via pipelined architecture) |
| Retrieval quality | 0.8908 → 0.8508 (**−4.5%**, acceptable trade) |

### Optimization dimensions exercised

This project deliberately covers **six distinct dimensions** of optimization rather than drilling deep on one. Each maps to a different course week:

| Dimension | Technique | Course Week |
|-----------|-----------|:---:|
| Algorithmic structure | IVF replaces BruteForce | (design) |
| Algorithmic refinement | K-Means++ seeding | **Week 8** |
| Code execution | Numba JIT-compiled kernel | **Week 6** |
| Data access patterns | Pre-computed cached norms | (merged from peer) |
| Concurrency | Streaming + concurrent generation | **Week 9** |
| System architecture | Dual-pool pipelined serving | **Week 10/11** |

### Project thesis

> **Speed does not live in one layer. Real systems have bottlenecks at every layer — CPU math, data access patterns, IO waiting, and stage composition — and each needs its own tool.**

The most counterintuitive result in this project is the 3-way comparison showing that **IVF alone gives only 4% end-to-end speedup** because gpt-4o generation dominates wall-clock time. The dramatic 3.87× batch speedup is entirely attributable to architectural rearrangement (pipelining + streaming), not algorithmic optimization. This validates the broad-coverage approach: understanding which tool applies where matters more than pushing one tool to the limit.

---

## Reproducibility

- Every optimization step is a tagged JSON snapshot under `results/experiments/medium_*.json`
- The 3-way comparison artifacts live in `results/comparisons/` and can be regenerated with `compare_pipelines.ipynb`
- Baseline code in `rag-optimization/components/` is never modified; every optimization lives in `optimized/` as a drop-in variant
- The integration branch `integrate-friend` preserves both the author's and the peer's commit history without overwriting either

For full per-step detail see [OPTIMIZATION_JOURNEY.md](OPTIMIZATION_JOURNEY.md) and [PIPELINE_MAP.md](PIPELINE_MAP.md).