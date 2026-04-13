# Teammate Guide: How to Set Up & Test

You don't need to understand vector databases. Just follow this guide.

---

## Step 1: Set Up Environment

```bash
git clone https://github.com/MissMyPetDog/Retrieval-Augmented-Generation-Optimization-Test.git
cd Retrieval-Augmented-Generation-Optimization-Test/rag-optimization
pip install numpy sentence-transformers datasets numba cython scipy matplotlib
```

If you have an NVIDIA GPU, also install:
```bash
pip install cupy-cuda12x
```

---

## Step 2: Build the Knowledge Base (One Command)

```bash
# Small (5K docs, ~1 min on CPU, good for testing)
python build_knowledge_base.py --download --size small --data_dir data/small

# Medium (100K docs, ~5 min on CPU / ~30s on GPU)
python build_knowledge_base.py --download --size medium --data_dir data/medium

# Large (500K docs, ~25 min on CPU / ~2 min on GPU)
python build_knowledge_base.py --download --size large --data_dir data/large
```

If you have a GPU, add `--device cuda` to speed up embedding generation:
```bash
python build_knowledge_base.py --download --size medium --data_dir data/medium --device cuda
```

When it finishes you will see:
```
✓ Knowledge base ready. Run notebooks or main.py to query.
```

Verify your data matches everyone else's:
```bash
python build_knowledge_base.py --verify --data_dir data/medium
```

---

## Step 3: Run the Baseline Test

This tests the UNOPTIMIZED system so you have a "before" number to compare against.

```bash
python run_test.py --data_dir data/small
```

This will print something like:
```
=== Similarity Benchmark (5,000 vectors) ===
  Pure Python : 370.12ms
  NumPy       :   1.14ms   (325x faster)

=== Retrieval Test (100 queries) ===
  BruteForce + NumPy : Recall@10=0.35, Latency=17.1ms
```

These are your BASELINE numbers. Write them down.

---

## Step 4: Make Your Optimization

All the code you can optimize is in two folders:

```
components/     ← baseline (slow) versions
optimized/      ← optimized (fast) versions
```

### What can you optimize?

**Option A: Improve an existing optimized file**

Open any file in `optimized/` and try to make it faster. For example:
- `optimized/similarity_numba.py` — can you write a better Numba kernel?
- `optimized/parallel_indexer.py` — can you parallelize K-Means itself?

**Option B: Write a new optimization**

Pick a function from `components/` and write a faster version. The key rule:
**your function must have the same input/output as the original.**

Example — say you want to optimize the chunking in `components/preprocessor.py`:

1. Open `components/preprocessor.py`, find `chunk_text_baseline()`
2. Create a new file `optimized/preprocessor_fast.py`
3. Write your optimized version with the SAME interface:

```python
# optimized/preprocessor_fast.py
from numba import njit
# or: import cython, multiprocessing, cupy, etc.

def chunk_text_optimized(text, chunk_size=256, overlap=32):
    """Same input/output as chunk_text_baseline, but faster."""
    # your optimized code here
    words = text.split()
    # ...
    return chunks  # must return list[str], same as original
```

4. Test it (see Step 5)

### What tools can you use?

| Tool | Import | Best for |
|------|--------|----------|
| NumPy | `import numpy as np` | Vectorized math |
| Cython | `.pyx` file + compile | C-speed loops |
| Numba | `from numba import njit` | JIT-compiled loops, parallel |
| multiprocessing | `from multiprocessing import Pool` | CPU parallelism |
| CuPy | `import cupy as cp` | GPU acceleration |
| itertools | `import itertools` | Memory-efficient iteration |

---

## Step 5: Test Your Optimization

### Quick test — compare your function directly

```python
import sys
sys.path.insert(0, ".")

import numpy as np
import time

# Load data
vectors = np.load("data/small/vectors.npy")
query = vectors[0]
corpus = vectors[1:]

# Import baseline
from components.similarity import cosine_sim_numpy

# Import YOUR optimization
from optimized.similarity_numba import cosine_sim_numba_parallel  # or your own file

# Time baseline
t0 = time.perf_counter()
for _ in range(10):
    result_baseline = cosine_sim_numpy(query, corpus)
t_baseline = (time.perf_counter() - t0) / 10 * 1000

# Time your version
t0 = time.perf_counter()
for _ in range(10):
    result_optimized = cosine_sim_numba_parallel(query, corpus)
t_optimized = (time.perf_counter() - t0) / 10 * 1000

# Compare
print(f"Baseline:  {t_baseline:.2f}ms")
print(f"Optimized: {t_optimized:.2f}ms")
print(f"Speedup:   {t_baseline / t_optimized:.1f}x")

# IMPORTANT: verify correctness (results must be the same)
diff = np.max(np.abs(result_baseline - result_optimized))
print(f"Max difference: {diff:.2e}")  # should be < 1e-5
```

### Full test — run the automated benchmark

```bash
python run_test.py --data_dir data/small
```

Compare the numbers with your Step 3 baseline. If your optimization works,
the numbers should be faster with the same or similar Recall@10.

---

## Step 6: Test on Larger Data

Once your optimization works on small data, test on medium or large:

```bash
# Build larger knowledge base (if not done already)
python build_knowledge_base.py --download --size medium --data_dir data/medium

# Test on it
python run_test.py --data_dir data/medium
```

The speedup difference between baseline and optimized should be MORE visible
on larger data.

---

## Common Issues

**"No module named 'config'"**
→ Make sure you are in the `rag-optimization/` directory, not the parent.

**"UnicodeDecodeError: gbk"**
→ You're on Windows. Add `encoding="utf-8"` to any `open()` call that fails.

**"CUDA not available"**
→ You don't have an NVIDIA GPU or CuPy is not installed. Skip GPU tests,
  everything else works on CPU.

**"Numba compilation slow on first run"**
→ Normal. Numba compiles on first call, subsequent calls are fast.
  Always do a warmup call before timing.

---

## File Reference

| File | What it does | Can I optimize it? |
|------|-------------|-------------------|
| `components/similarity.py` | Cosine similarity (Pure Python + NumPy) | YES — this is the main target |
| `components/vector_index.py` | BruteForce + IVF index | YES — K-Means, search logic |
| `components/preprocessor.py` | Text cleaning + chunking | YES — itertools, multiprocessing |
| `components/embedder.py` | Embedding generation | YES — batch/parallel strategies |
| `optimized/similarity_numba.py` | Numba JIT version | Already done, can improve |
| `optimized/similarity_cython.pyx` | Cython version | Already done, can improve |
| `optimized/similarity_gpu.py` | CuPy GPU version | Already done, can improve |
| `optimized/parallel_indexer.py` | Parallel index build | Already done, currently slower than sequential |
| `components/retriever.py` | Orchestrator | Not much to optimize |
| `components/generator.py` | LLM wrapper | Not an optimization target |
