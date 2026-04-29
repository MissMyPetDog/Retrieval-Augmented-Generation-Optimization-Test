# RAG Pipeline Optimization

A Retrieval-Augmented Generation system built from scratch (no LangChain / LlamaIndex / FAISS), then systematically optimized end-to-end on CPU only.

Dataset: **MS MARCO `medium`** — 100,000 passages, 500 queries with relevance judgments. LLM endpoint: **ChatGPT-4o**.

Most numbers below are loaded from [`rag-optimization/data/medium/test_results_cpu.json`](rag-optimization/data/medium/test_results_cpu.json) (produced by `run_test.py`); the end-to-end comparison comes from `results/5_configs.json` (produced by `run_5_configs.py`).

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r rag-optimization/requirements.txt
# or:  conda env create -f rag-optimization/environment.yaml

# 2. Build the knowledge base (one-time, ~15 min for medium on CPU)
python rag-optimization/build_knowledge_base.py --download --size medium --data_dir rag-optimization/data/medium

# 3. Component-level benchmarks (similarity kernels, K-Means variants, retrieval ablation)
python rag-optimization/run_test.py --data_dir rag-optimization/data/medium --device cpu --with_build

# Optional: explore IVF n-probe trade-off
python rag-optimization/benchmarks/nprobe_tradeoff.py --data_dir rag-optimization/data/medium --probes 1 2 4 8 16 32 64

# 4. Add real LLM benchmarks (requires ChatGPT_API_KEY env var, ~$0.15)
python rag-optimization/run_test.py --data_dir rag-optimization/data/medium --device cpu --with_build --with_llm

# 5. End-to-end 5-config comparison: full pipeline timed for each component combo
python run_5_configs.py -n 8                    # default: batch embed, 8 queries (~$0.22)
python run_5_configs.py -n 8 --no_batch_embed   # per-query embed variant
python run_5_configs.py -n 100 -w 16            # probe higher LLM concurrency (~$2.80)
```

Results land in `rag-optimization/data/medium/test_results_cpu.json` and `results/5_configs.json`.

### Interactive notebooks

```bash
jupyter lab interact_portal.ipynb       # Step-by-step benchmark (6 optimization steps)
jupyter lab compare_pipelines.ipynb     # 3-way final comparison (BruteForce vs IVF default vs Fully optimized)
```

### CLI flags

`run_test.py`:

| **Flag**            | **Effect**                                   |
| :------------------ | :------------------------------------------- |
| `--similarity_only` | only similarity kernels                      |
| `--build_only`      | only K-Means variants                        |
| `--llm_only`        | only LLM tests (requires `ChatGPT_API_KEY`)  |
| `--with_build`      | adds K-Means build comparison to default run |
| `--with_llm`        | adds LLM tests to default run                |

`run_5_configs.py`:

| **Flag**             | **Effect**                                                                   |
| :------------------- | :--------------------------------------------------------------------------- |
| `-n, --n_queries`    | queries per config (default 8). Total real Kong calls = `n_queries * 7`      |
| `-w, --n_async_workers` | LLM concurrency for configs 2-5 (default 8). Try 16/32/64 to probe Kong's cap |
| `--no_batch_embed`   | use per-query embed for configs 2-5 (hides Embed/q + batch columns in table) |
| `--max_tokens`       | LLM max_tokens (default 128)                                                 |

---

## RAG Workflow

```
OFFLINE (built once)
  passages → preprocess → embed (MiniLM) → K-Means + IVF index → disk

ONLINE (per query)
  query → embed (~300 ms) → search index (~5-60 ms) → gpt-4o generate (~1.5 s) → answer
```

Each stage has a different bottleneck type — CPU math for search, IO wait for generation — and is optimized with different techniques.

---

## Results

### Retrieval (500 queries, Recall@10 + mean latency)

| Configuration                                         | Recall@10  | MRR        | Latency     | Search speedup |
| :---------------------------------------------------- | :--------- | :--------- | :---------- | :------------- |
| BruteForce + NumPy -> Baseline                        | **0.8908** | 0.4937     | 68.1 ms     | 1.00×          |
| BruteForce + NumPy (norm cache)                       | 0.8908     | 0.4937     | 12.5 ms     | 5.43×          |
| BruteForce + Numba parallel                           | 0.8908     | 0.4937     | 19.1 ms     | 3.56×          |
| IVF(64, 8) + NumPy                                    | 0.8508     | 0.4740     | 21.6 ms     | 3.15×          |
| IVF(64, 8) + NumPy (norm cache)                       | 0.8508     | 0.4740     | 13.9 ms     | 4.90×          |
| **IVF(64, 8) + NumPy (norm cache, batch embed)**      | **0.8508** | **0.4740** | **7.27 ms** | **9.37×**      |
| IVF(64, 8) + Numba parallel                           | 0.8508     | 0.4740     | 19.3 ms     | 3.53×          |
| IVF(64, 8) + Numba parallel (norm cache, batch embed) | 0.8508     | 0.4740     | 9.91 ms     | 6.87×          |

IVF trades ~4.5% recall for 5-9× faster queries. The "norm cache + batch embed" combination is the clear winner on CPU.


### LLM: concurrency (batch throughput, 8 real gpt-4o calls)

| Mode              | Batch total  | Speedup   |
| :---------------- | :----------- | :-------- |
| Sequential        | 13,099 ms    | 1.00×     |
| Threaded (n=8)    | 6,688 ms     | 1.96×     |
| **Async (max=8)** | **4,920 ms** | **2.66×** |

### End-to-end pipeline comparison (8 queries, real ChatGPT-4o)

5 fully-stacked pipelines, each timed embed + search + gen. Per-query embed (no batch). Speedup is vs Config 1 (zero-optimization baseline). Numbers from `results/5_configs.json`, produced by `python run_5_configs.py -n 8 --no_batch_embed`.

| #   | Index       | Sim fn                                  | cache | LLM mode    | Search/q             | Gen/call             | E2E                    |
| :-- | :---------- | :-------------------------------------- | :---: | :---------- | :------------------- | :------------------- | :--------------------- |
| 1   | BruteForce  | `cosine_sim_numpy`                      | OFF   | sequential  | 72.79 ms (1.00×)     | 1,637 ms (1.00×)     | 13.74 s (1.00×)        |
| **2** | **IVF(64,8)** | **`cosine_sim_numpy`**                 | **ON** | **async** | **7.37 ms (9.88×)** | **262 ms (6.24×)**  | **2.22 s (6.19×)**     |
| 3   | IVF(64,8)   | `cosine_sim_numpy`                      | ON    | threaded    | 7.71 ms (9.44×)      | 374 ms (4.37×)       | 3.12 s (4.40×)         |
| 4   | IVF(64,8)   | `cosine_sim_numba_parallel_precomputed` | ON    | async       | 8.87 ms (8.21×)      | 286 ms (5.72×)       | 2.43 s (5.64×)         |
| 5   | IVF(64,8)   | `cosine_sim_numba_parallel_precomputed` | ON    | threaded    | 8.82 ms (8.26×)      | 346 ms (4.73×)       | 2.91 s (4.72×)         |

**Best E2E: ~6.2× faster than baseline.** Roughly half the win is structural retrieval (IVF + norm cache → 9.4-9.9× search speedup); the other half is LLM concurrency (async/threaded → 4-6× per-call gen speedup amortized over 8 calls). At this candidate-set size (n_probes=8 over 64 clusters), Numba parallel doesn't outpace BLAS NumPy — the corpus slice is small enough that thread-launch overhead eats the kernel speedup.

---

## Repository Layout

```
.
├── README.md                    ← this file
├── portal.py                    ← main benchmarking module (shared by both notebooks)
├── interact_portal.ipynb        ← step-by-step optimization notebook (Steps 1-6)
├── compare_pipelines.ipynb      ← 3-way final comparison notebook
├── run_5_configs.py             ← end-to-end 5-config comparison CLI (table above)
│
├── comparisons/                 ← self-contained Python scripts for 3-way comparison
│   ├── common.py                ← shared setup and data loading
│   ├── run_bruteforce.py        ← Config 1: zero optimizations
│   ├── run_intermediate.py      ← Config 2: IVF default
│   └── run_optimized.py         ← Config 3: full stack
│
├── rag-optimization/            ← core project
│   ├── build_knowledge_base.py  ← offline pipeline: download + preprocess + embed + index
│   ├── run_test.py              ← CLI benchmark (similarity + retrieval + build + LLM)
│   ├── components/              ← baseline implementations (never modified)
│   │   ├── preprocessor.py, embedder.py, similarity.py
│   │   ├── vector_index.py      ← BruteForceIndex + IVFIndex (with peer's opt flags)
│   │   ├── retriever.py, generator.py (with generate_stream for Step 4)
│   ├── optimized/               ← optimized drop-in replacements
│   │   ├── similarity_numba.py  ← Numba JIT cosine (+ peer's precomputed variant)
│   │   ├── kmeans_numba.py      ← Numba K-Means + K-Means++ seeding
│   │   ├── async_embedder.py, async_generator.py, parallel_indexer.py
│   └── data/medium/             ← built knowledge base + test_results_cpu.json
│
└── results/experiments/         ← tagged JSON snapshots per optimization step
    ├── medium_0_baseline.json
    ├── medium_1_ivf_tuned.json
    ├── medium_2_kmeans_numba.json
    ├── medium_3_kmeans_pp_init.json
    ├── medium_4_llm_streaming.json
    ├── medium_5_pipeline.json
    └── medium_6_friend_optimizations.json
```

---

## Reproducibility

- Every optimization step is tagged under `results/experiments/medium_<step>.json` with lineage + timestamp
- Use `portal.compare_experiments("medium", <step_a>, <step_b>)` to diff any two steps
- CPU is explicitly enforced at setup: `CUDA_VISIBLE_DEVICES=""`, `NUMBA_DISABLE_CUDA=1`, verified via `torch.cuda.is_available() == False`
- Knowledge base files include SHA256 hashes in `metadata.json`; verify with `python build_knowledge_base.py --verify --data_dir ...`

This repo represents a **collaborative study** — optimizations from four contributors (breadth vs. depth) integrated on the `integrate-friend` branch without either side's work being overwritten.
