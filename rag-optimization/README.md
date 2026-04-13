# RAG System Optimization with Advanced Python

A Retrieval-Augmented Generation system built from scratch, systematically optimized
using advanced Python tools: Cython, Numba, multiprocessing, GPU (CuPy), and PySpark.

## Quick Start (For Team Members)

### Option A: One command — download data + build everything

```bash
pip install numpy sentence-transformers datasets numba cython cupy-cuda12x matplotlib scipy
python build_knowledge_base.py --download --size medium
```

This will:
1. Download 100K passages from MS MARCO (deterministic — everyone gets the same data)
2. Preprocess and chunk the text
3. Generate embeddings (auto-detects GPU)
4. Build BruteForce and IVF indexes
5. Save everything to `data/`
6. Output a build report with SHA256 hashes for verification

### Option B: Step by step

```bash
# 1. Download data
python data/download_data.py --size medium

# 2. Build knowledge base
python build_knowledge_base.py

# 3. Verify your data matches teammates'
python data/download_data.py --verify
python build_knowledge_base.py --verify
```

### Size options

| Size | Passages | Queries | Vector File | Build Time (GPU) |
|------|----------|---------|-------------|-----------------|
| small | 5,000 | 100 | ~8 MB | ~10s |
| medium | 100,000 | 500 | ~154 MB | ~30s |
| large | 500,000 | 1,000 | ~768 MB | ~2min |

### Verify data consistency

After building, share `data/metadata.json` with your team. Others can run:

```bash
python data/download_data.py --verify
```

If hashes match, your data is identical.

## Project Structure

```
rag-optimization/
├── build_knowledge_base.py    # ★ One-click knowledge base builder
├── main.py                    # Main pipeline entry point
├── config.py                  # Global configuration
├── data/
│   ├── download_data.py       # Deterministic data downloader
│   ├── passages.jsonl         # (generated) raw passages
│   ├── queries.jsonl          # (generated) evaluation queries
│   ├── chunks.jsonl           # (generated) processed chunks
│   ├── vectors.npy            # (generated) embedding matrix
│   └── metadata.json          # (generated) hashes for verification
├── components/                # Baseline implementations
│   ├── preprocessor.py        # Document loading & chunking
│   ├── embedder.py            # Embedding generation
│   ├── similarity.py          # Cosine similarity (Pure Python + NumPy)
│   ├── vector_index.py        # BruteForce & IVF index
│   ├── retriever.py           # Query orchestrator
│   └── generator.py           # LLM answer generation
├── optimized/                 # Optimized versions
│   ├── similarity_cython.pyx  # Cython (Week 5)
│   ├── similarity_numba.py    # Numba JIT (Week 6)
│   ├── similarity_gpu.py      # CuPy GPU (Week 12)
│   ├── parallel_indexer.py    # Multiprocessing (Week 10-11)
│   └── async_embedder.py      # Concurrent embedding (Week 9)
└── benchmarks/
    ├── benchmark_runner.py    # Automated benchmark harness
    └── evaluate.py            # Recall@K, MRR evaluation
```

## Running Benchmarks

```bash
# Run on Google Colab: upload RAG_Optimization_Final.ipynb
# Or locally:
python -m benchmarks.benchmark_runner
python main.py --mode compare
```
