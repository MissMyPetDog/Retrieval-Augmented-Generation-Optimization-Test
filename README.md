# RAG Pipeline Optimization

A Retrieval-Augmented Generation system built from scratch (no LangChain / LlamaIndex / FAISS), then systematically optimized end-to-end inference latency on CPU only.

Dataset: **MS MARCO `medium`** — 100,000 passages, 500 queries with relevance judgments. LLM endpoint: **ChatGPT-4o**.

Most numbers below are loaded from [`rag-optimization/data/medium/test_results_cpu.json`](rag-optimization/data/medium/test_results_cpu.json) (produced by `run_test.py`); the end-to-end inference comparison comes from `results/5_configs.json` (produced by `run_5_configs.py`).

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

# 5. End-to-end inference 5-config comparison: full pipeline timed for each component combo
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

| **Flag**                | **Effect**                                                                                   |
| :---------------------- | :------------------------------------------------------------------------------------------- |
| `-n, --n_queries`       | queries per config (default 8). Total real ChatGPT calls = `n_queries * 7`                   |
| `-w, --n_async_workers` | LLM concurrency for configs 2-5 (default 8). Try 16/32/64 to probe ChatGPT's concurrency cap |
| `--no_batch_embed`      | use per-query embed for configs 2-5 (hides Embed/q + batch columns in table)                 |
| `--max_tokens`          | LLM max_tokens (default 128)                                                                 |

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
| BruteForce + NumPy                                    | **0.8908** | 0.4937     | 68.1 ms     | 1.00×          |
| BruteForce + NumPy (norm cache)                       | 0.8908     | 0.4937     | 12.5 ms     | 5.43×          |
| BruteForce + Numba parallel                           | 0.8908     | 0.4937     | 19.1 ms     | 3.56×          |
| IVF(64, 8) + NumPy                                    | 0.8908     | 0.4740     | 21.6 ms     | 3.15×          |
| IVF(64, 8) + NumPy (norm cache)                       | 0.8508     | 0.4740     | 13.9 ms     | 4.90×          |
| **IVF(64, 8) + NumPy (norm cache, batch embed)**      | **0.8508** | **0.4740** | **7.27 ms** | **9.37×**      |
| IVF(64, 8) + Numba parallel                           | 0.8508     | 0.4740     | 19.3 ms     | 3.53×          |
| IVF(64, 8) + Numba parallel (norm cache, batch embed) | 0.8508     | 0.4740     | 9.91 ms     | 6.87×          |

IVF trades ~4.5% recall for 5-9× faster queries. The "norm cache + batch embed" combination is the clear winner on CPU.


### LLM: concurrency (batch throughput, 8 gpt-4o calls)

| Mode              | Batch total  | Speedup   |
| :---------------- | :----------- | :-------- |
| Sequential        | 13,099 ms    | 1.00×     |
| Threaded (n=8)    | 6,688 ms     | 1.96×     |
| **Async (max=8)** | **4,920 ms** | **2.66×** |

### End-to-end inference latency comparison (8 queries, gpt-4o)

5 fully-stacked pipelines, each timed embed + search + gen. Speedup is vs the first row (zero-optimization baseline). Numbers from `results/5_configs.json`, produced by `python run_5_configs.py -n 8`.

| Index         | Cosine similarity kenrel | norm cache | batch embed | LLM mode     | Embed/q latency    | Search/q latency    | Gen/call latency   | E2E inference latency (total) |
| :------------ | :----------------------- | :--------- | :---------- | :----------- | :----------------- | :------------------ | :----------------- | :---------------------------- |
| BruteForce    | NumPy                    | OFF        | OFF         | sequential   | 7.3 ms (1.00x)     | 69.04 ms (1.00x)    | 1996 ms (1.00x)    | 16.58 s (1.00x)               |
| IVF(64,8)     | NumPy                    | ON         | ON          | async        | 2.5 ms (2.93x)     | 6.37 ms (10.83x)    | 433 ms (4.61x)     | 3.54 s (4.69x)                |
| IVF(64,8)     | NumPy                    | ON         | ON          | threaded     | 2.3 ms (3.13x)     | 7.16 ms (9.64x)     | 465 ms (4.29x)     | 3.79 s (4.37x)                |
| IVF(64,8)     | Numba parallel           | ON         | ON          | async        | 2.5 ms (2.93x)     | 8.82 ms (7.83x)     | 353 ms (5.65x)     | 2.92 s (5.68x)                |
| **IVF(64,8)** | **Numba parallel**       | **ON**     | **ON**      | **threaded** | **2.3 ms (3.13x)** | **7.88 ms (8.76x)** | **266 ms (7.52x)** | **2.21 s (7.52x)**            |


<!-- | Retrieval Config                                          | Gen Mode     | Total Search latency (ms) | Total Gen latency (s) | Total E2E Inference Latency (s) | E2E Speedup |
| :-------------------------------------------------------- | :----------- | :------------------------ | :-------------------- | :------------------------------ | :---------- |
| BruteForce + NumPy                                        | sequential   | 552.32                    | 15.97                 | 16.58                           | 1.00x       |
| IVF(64, 8) + NumPy (norm cache, batch embed)              | async        | 50.96                     | 3.46                  | 3.54                            | 4.69x       |
| IVF(64, 8) + NumPy (norm cache, batch embed)              | threaded     | 57.28                     | 3.72                  | 3.79                            | 4.37x       |
| IVF(64, 8) + Numba parallel (norm cache, batch embed)     | async        | 70.56                     | 2.82                  | 2.92                            | 5.68x       |
| **IVF(64, 8) + Numba parallel (norm cache, batch embed)** | **threaded** | **63.04**                 | **2.13**              | **2.21**                        | **7.52x**   | -->


**Best E2E in this run: row 5 (~7.5× vs baseline).** IVF + norm cache + batched query embedding cuts search to ~8–11× faster than BruteForce; overlapping LLM calls (async/threaded) amortizes generation. Here Numba + threaded edges out the NumPy + async row on end-to-end latency (API variance also affects gen/call).

---

## Repository Layout

```
.
├── README.md                    ← this file
├── portal.py                    ← main benchmarking module (shared by both notebooks)
├── interact_portal.ipynb        ← step-by-step optimization notebook (Steps 1-6)
├── compare_pipelines.ipynb      ← 3-way final comparison notebook
├── run_5_configs.py             ← end-to-end inference 5-config comparison CLI (table above)
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
│   ├── benchmarks/              ← benchmark scripts for IVF and retrieval optimization
│   │   ├── benchmark_runner.py   ← benchmark orchestration and reporting
│   │   ├── evaluate.py           ← evaluation helpers for retrieval experiments
│   │   └── nprobe_tradeoff.py    ← IVF n-probe tradeoff exploration
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
