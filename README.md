# RAG Pipeline Optimization

A Retrieval-Augmented Generation system built from scratch (no LangChain / LlamaIndex / FAISS), then systematically optimized end-to-end on CPU only.

Dataset: **MS MARCO `medium`** — 100,000 passages, 500 queries with relevance judgments. LLM endpoint: **ChatGPT-4o**.

All numbers below are loaded verbatim from [`rag-optimization/data/medium/test_results_cpu.json`](rag-optimization/data/medium/test_results_cpu.json), produced by a single `run_test.py` invocation.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r rag-optimization/requirements.txt
# or:  conda env create -f rag-optimization/environment.yaml

# 2. Build the knowledge base (one-time, ~15 min for medium on CPU)
python rag-optimization/build_knowledge_base.py --download --size medium --data_dir rag-optimization/data/medium

# 3. Run the full CPU benchmark (similarity + retrieval + K-Means variants)
python rag-optimization/run_test.py --data_dir rag-optimization/data/medium --device cpu --with_build

# Optional: explore IVF n-probe trade-off
python rag-optimization/benchmarks/nprobe_tradeoff.py --data_dir rag-optimization/data/medium --probes 1 2 4 8 16 32 64

# 4. Add real LLM benchmarks (requires ChatGPT_API_KEY env var, ~$0.15)
python rag-optimization/run_test.py --data_dir rag-optimization/data/medium --device cpu --with_build --with_llm

# Results saved to rag-optimization/data/medium/test_results_cpu.json
```

### Interactive notebooks

```bash
jupyter lab interact_portal.ipynb       # Step-by-step benchmark (6 optimization steps)
jupyter lab compare_pipelines.ipynb     # 3-way final comparison (BruteForce vs IVF default vs Fully optimized)
```

### CLI flags

| **Flag**            | **Effect**                                   |
| :------------------ | :------------------------------------------- |
| `--similarity_only` | only similarity kernels                      |
| `--build_only`      | only K-Means variants                        |
| `--llm_only`        | only LLM tests (requires `ChatGPT_API_KEY`)  |
| `--with_build`      | adds K-Means build comparison to default run |
| `--with_llm`        | adds LLM tests to default run                |

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

### LLM: streaming (perceived latency, TTFT)

| Mode                       | TTFT         | Total per call | TTFT reduction |
| :------------------------- | :----------- | :------------- | :------------- |
| Sequential non-streaming   | 1,435 ms     | 1,435 ms       | baseline       |
| **Sequential streaming**   | **1,112 ms** | 1,153 ms       | **−22.5%**     |
| Concurrent streaming (n=8) | 1,229 ms     | 1,267 ms       | −14.4%         |

Streaming makes the user see the first word sooner even though total generation time is unchanged.

### LLM: pipelined RAG (end-to-end, embed + search + gen)

| Mode                               | Batch total  | Per-query amortized | Speedup   |
| :--------------------------------- | :----------- | :------------------ | :-------- |
| Sequential naive                   | 11,727 ms    | 1,466 ms            | 1.00×     |
| **Pipelined (4 retrieve + 8 gen)** | **6,724 ms** | **841 ms**          | **1.74×** |

The pipelined architecture overlaps CPU-bound retrieval with IO-bound generation through two parallel thread pools. In lighter ChatGPT-load runs this reaches **5.1×** (see `results/experiments/medium_5_pipeline.json`); the 1.74× shown here reflects real-world proxy rate-limiting under back-to-back concurrent sections.

---

## Repository Layout

```
.
├── README.md                    ← this file
├── portal.py                    ← main benchmarking module (shared by both notebooks)
├── interact_portal.ipynb        ← step-by-step optimization notebook (Steps 1-6)
├── compare_pipelines.ipynb      ← 3-way final comparison notebook
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
