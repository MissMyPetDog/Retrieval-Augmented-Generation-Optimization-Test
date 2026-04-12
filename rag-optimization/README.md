# RAG System Optimization with Advanced Python

A Retrieval-Augmented Generation system built from scratch, systematically optimized
using advanced Python tools: Cython, Numba, multiprocessing, GPU (CuPy), and PySpark.

## Project Structure

```
rag-optimization/
├── config.py                 # Global configuration
├── main.py                   # Main pipeline entry point
├── data/
│   └── download_data.py      # MS MARCO dataset downloader
├── components/               # Baseline (pure Python) implementations
│   ├── preprocessor.py       # Document loading & chunking
│   ├── embedder.py           # Embedding generation
│   ├── similarity.py         # Cosine similarity (pure Python + NumPy)
│   ├── vector_index.py       # Brute-force & IVF index
│   ├── retriever.py          # Top-K retrieval orchestrator
│   └── generator.py          # LLM answer generation
├── optimized/                # Optimized versions of components
│   ├── similarity_cython.pyx # Cython-accelerated similarity
│   ├── similarity_numba.py   # Numba JIT similarity
│   ├── similarity_gpu.py     # CuPy GPU similarity
│   ├── parallel_indexer.py   # Multiprocessing index builder
│   └── async_embedder.py     # Async/concurrent embedding
├── benchmarks/
│   ├── benchmark_runner.py   # Micro-benchmark harness
│   └── evaluate.py           # Recall@K, MRR evaluation
```

## Quick Start

```bash
pip install numpy sentence-transformers datasets tqdm
python data/download_data.py          # Download MS MARCO subset
python main.py --mode baseline        # Run baseline pipeline
python main.py --mode optimized       # Run optimized pipeline
python -m benchmarks.benchmark_runner # Run all benchmarks
```
