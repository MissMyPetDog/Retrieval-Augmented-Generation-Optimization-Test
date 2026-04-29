"""
Testing & benchmarking logic for interact_portal.ipynb.

All heavy lifting lives here so the notebook stays readable. Edit this file;
with %autoreload 2 in the notebook, changes propagate automatically.

NOTE: kept ASCII-only on purpose. On Chinese Windows the default encoding is
GBK, and IPython's deduperreload opens files without encoding="utf-8", so any
non-ASCII character breaks autoreload.
"""
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =====================================================================
# Path setup
# =====================================================================
PROJECT_ROOT = Path(__file__).parent
RAG_DIR = PROJECT_ROOT / "rag-optimization"
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))


# =====================================================================
# Section 1 -- Setup
# =====================================================================

def setup_cpu_only(verbose: bool = True) -> None:
    """Hide CUDA from torch / numba / cupy so every timing reflects CPU only."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NUMBA_DISABLE_CUDA"] = "1"
    # Windows MAX_PATH workaround: the default Numba cache lives next to the
    # .py file in __pycache__/, and long project paths + long function names
    # (e.g. cosine_sim_numba_parallel_precomputed) can exceed the 260-char limit.
    # Redirect to the OS temp dir which is much shorter.
    if sys.platform == "win32":
        import tempfile
        short_cache = Path(tempfile.gettempdir()) / "numba_cache"
        short_cache.mkdir(parents=True, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = str(short_cache)
        if verbose:
            print(f"Numba cache redirected to: {short_cache}")
    if verbose:
        try:
            import torch
            print(f"torch sees CUDA? {torch.cuda.is_available()}  (expect False)")
        except ImportError:
            print("torch not imported (fine)")


def load_knowledge_base(data_subdir: str = "small"):
    """Load prebuilt KB. Returns (vectors, chunks, queries, data_dir, dataset_name)."""
    data_dir = RAG_DIR / "data" / data_subdir
    assert data_dir.exists(), f"Data dir not found: {data_dir}"

    vectors = np.load(data_dir / "vectors.npy")
    with open(data_dir / "chunks.jsonl", "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    with open(data_dir / "queries.jsonl", "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f]

    print(f"Data dir:   {data_dir}")
    print(f"Dataset:    {data_subdir}")
    print(f"vectors:    {vectors.shape}  dtype={vectors.dtype}")
    print(f"chunks:     {len(chunks)}")
    print(f"queries:    {len(queries)}")
    print(f"1st query:  {queries[0]['text'][:80]}")
    return vectors, chunks, queries, data_dir, data_subdir


# =====================================================================
# Timing helper
# =====================================================================

def bench(fn, args, warmup: int = 2, repeats: int = 10):
    """Run fn(*args) `repeats` times after warmup. Returns (mean_ms, std_ms, last_output)."""
    for _ in range(warmup):
        out = fn(*args)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times)), out


# =====================================================================
# Section 2 -- Similarity benchmarks
# =====================================================================

def run_similarity_benchmarks(vectors: np.ndarray, subset: int = 500) -> dict:
    """
    Benchmark Pure Python / NumPy / Numba / Numba parallel cosine similarity.
    Returns {name: {mean_ms, std_ms, speedup_vs_numpy, [note]}}.
    """
    from components.similarity import cosine_sim_python, cosine_sim_numpy
    from optimized.similarity_numba import (
        cosine_sim_numba,
        cosine_sim_numba_parallel,
        warmup_numba,
    )

    query = vectors[0]
    corpus = vectors[1:]
    N, D = corpus.shape
    print(f"Corpus: N={N}, D={D}")
    print("Warming up Numba JIT...")
    warmup_numba()

    numpy_m, numpy_s, numpy_out = bench(cosine_sim_numpy, (query, corpus))
    print(f"NumPy            : {numpy_m:8.3f} +/- {numpy_s:.3f} ms")

    numba_m, numba_s, numba_out = bench(cosine_sim_numba, (query, corpus))
    print(f"Numba            : {numba_m:8.3f} +/- {numba_s:.3f} ms")

    par_m, par_s, par_out = bench(cosine_sim_numba_parallel, (query, corpus))
    print(f"Numba (parallel) : {par_m:8.3f} +/- {par_s:.3f} ms")

    sub = corpus[:subset]
    py_m, py_s, py_out = bench(cosine_sim_python, (query, sub), warmup=1, repeats=3)
    py_full = py_m * (N / subset)
    print(f"Pure Python      : {py_m:8.3f} ms on {subset} vecs  ->  ~{py_full:.1f} ms "
          f"extrapolated to {N}")

    print("\nCorrectness vs NumPy:")
    for label, arr in [
        ("Numba", numba_out),
        ("Numba (parallel)", par_out),
        ("Pure Python (subset)", py_out),
    ]:
        ref = numpy_out[:subset] if label.startswith("Pure") else numpy_out
        diff = float(np.max(np.abs(ref - arr)))
        ok = diff < 1e-4
        print(f"  {label:20s} max|diff| = {diff:.2e}  {'OK' if ok else 'FAIL'}")

    results = {
        "Pure Python (extrapolated)": {
            "mean_ms": py_full, "std_ms": py_s * (N / subset),
            "note": f"timed on {subset}, scaled to {N}",
        },
        "NumPy":            {"mean_ms": numpy_m, "std_ms": numpy_s},
        "Numba":            {"mean_ms": numba_m, "std_ms": numba_s},
        "Numba (parallel)": {"mean_ms": par_m,   "std_ms": par_s},
    }
    for d in results.values():
        d["speedup_vs_numpy"] = numpy_m / d["mean_ms"]

    print("\nSummary (speedup vs NumPy):")
    for name, d in results.items():
        print(f"  {name:30s} {d['mean_ms']:10.3f} ms   {d['speedup_vs_numpy']:6.2f}x")
    return results


# =====================================================================
# Section 3 -- Similarity visualization
# =====================================================================

def plot_similarity(results: dict, N: int, D: int) -> None:
    names    = list(results.keys())
    means    = [results[n]["mean_ms"] for n in names]
    stds     = [results[n].get("std_ms", 0.0) for n in names]
    speedups = [results[n]["speedup_vs_numpy"] for n in names]

    colors = ["#e74c3c", "#3498db", "#f39c12", "#27ae60"][:len(names)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bars1 = ax1.bar(names, means, yerr=stds, color=colors, capsize=4, edgecolor="black")
    ax1.set_yscale("log")
    ax1.set_ylabel("Latency per query (ms, log scale)")
    ax1.set_title(f"Cosine similarity latency - N={N}, D={D} (CPU only)")
    ax1.tick_params(axis="x", rotation=15)
    for bar, m in zip(bars1, means):
        label = f"{m:.2f} ms" if m >= 1 else f"{m*1000:.0f} us"
        ax1.text(bar.get_x() + bar.get_width()/2, m * 1.15, label,
                 ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(names, speedups, color=colors, edgecolor="black")
    ax2.axhline(y=1.0, color="black", linestyle="--", linewidth=1, label="NumPy baseline")
    ax2.set_ylabel("Speedup vs NumPy (x)")
    ax2.set_title("Speedup relative to NumPy")
    ax2.tick_params(axis="x", rotation=15)
    ax2.legend()
    for bar, s in zip(bars2, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2, s, f"{s:.2f}x",
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()

    py_ms = results["Pure Python (extrapolated)"]["mean_ms"]
    print("\nSpeedup vs Pure Python (the interpreted-Python baseline):")
    for name, d in results.items():
        print(f"  {name:30s} {py_ms / d['mean_ms']:8.1f}x")


# =====================================================================
# Section 4 -- Index benchmarks
# =====================================================================

def run_index_build_benchmarks(vectors: np.ndarray, chunks: list,
                               n_clusters: int = 32,
                               kmeans_impl: str = "baseline"):
    """
    Build BruteForce, sequential IVF, and (if possible) parallel IVF.
    Returns (bf, ivf_sequential, ivf_parallel_or_None, results_dict).

    kmeans_impl:
      - "baseline": original pure-NumPy K-Means (Python loop over clusters)
      - "numba":    Numba JIT K-Means with hoisted norms + single-pass update
    """
    from components.vector_index import BruteForceIndex, IVFIndex
    from optimized.parallel_indexer import ParallelIVFBuilder

    if kmeans_impl == "numba":
        from optimized.kmeans_numba import IVFIndexNumba, warmup_kmeans_numba
        IVF_CLASS = IVFIndexNumba
        print("Warming up Numba K-Means JIT...")
        warmup_kmeans_numba()
    elif kmeans_impl == "numba_pp":
        from optimized.kmeans_numba import IVFIndexNumbaPP, warmup_kmeans_numba
        IVF_CLASS = IVFIndexNumbaPP
        print("Warming up Numba K-Means JIT...")
        warmup_kmeans_numba()
    elif kmeans_impl == "baseline":
        IVF_CLASS = IVFIndex
    else:
        raise ValueError(
            f"Unknown kmeans_impl={kmeans_impl!r}. "
            f"Use 'baseline', 'numba', or 'numba_pp'."
        )

    doc_ids = [c["id"] for c in chunks]
    full = vectors.astype(np.float32)
    print(f"Corpus: N={len(full)}, D={full.shape[1]}, clusters={n_clusters}, "
          f"kmeans={kmeans_impl}")

    bf = BruteForceIndex()
    t0 = time.perf_counter()
    bf.build(full, doc_ids)
    bf_ms = (time.perf_counter() - t0) * 1000

    ivf_seq = IVF_CLASS(n_clusters=n_clusters, n_probes=4, kmeans_iters=20)
    t0 = time.perf_counter()
    ivf_seq.build(full, doc_ids)
    ivf_seq_ms = (time.perf_counter() - t0) * 1000

    ivf_par = IVF_CLASS(n_clusters=n_clusters, n_probes=4, kmeans_iters=20)
    ivf_par_ms = None
    try:
        builder = ParallelIVFBuilder(n_clusters=n_clusters, n_workers=4)
        t0 = time.perf_counter()
        builder.build_parallel(full, doc_ids, ivf_par)
        ivf_par_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        print(f"Parallel build failed ({type(e).__name__}): {e}")
        print("(Often happens on Windows/Jupyter -- sequential result still reported.)")
        ivf_par = None

    results = {
        "BruteForce":       {"ms": bf_ms},
        "IVF (sequential)": {"ms": ivf_seq_ms},
    }
    if ivf_par_ms is not None:
        results["IVF (parallel)"] = {"ms": ivf_par_ms}
    results["_kmeans_impl"] = kmeans_impl

    print("\n--- Build times ---")
    print(f"  BruteForce          : {bf_ms:8.1f} ms")
    print(f"  IVF (sequential)    : {ivf_seq_ms:8.1f} ms   (kmeans={kmeans_impl})")
    if ivf_par_ms is not None:
        print(f"  IVF (parallel, 4 w) : {ivf_par_ms:8.1f} ms   "
              f"({ivf_seq_ms/ivf_par_ms:.2f}x vs sequential)")
    return bf, ivf_seq, ivf_par, results


def run_index_query_benchmarks(bf, ivf, vectors: np.ndarray, k: int = 10,
                               probes: tuple = (1, 2, 4, 8),
                               n_queries: int = 20,
                               warmup: int = 2, repeats: int = 5) -> dict:
    """Benchmark search latency per query for BruteForce + IVF at several n_probes."""
    from components.similarity import cosine_sim_numpy

    query_vecs = vectors[:n_queries].astype(np.float32)

    def bench_idx(search_fn):
        for q in query_vecs[:warmup]:
            _ = search_fn(q)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for q in query_vecs:
                _ = search_fn(q)
            times.append((time.perf_counter() - t0) * 1000 / len(query_vecs))
        return float(np.mean(times)), float(np.std(times))

    bf_m, bf_s = bench_idx(lambda q: bf.search(q, k=k, sim_fn=cosine_sim_numpy))
    print(f"BruteForce (NumPy)       : {bf_m:7.3f} +/- {bf_s:.3f} ms/query")

    results = {"BruteForce (NumPy)": {"mean_ms": bf_m, "std_ms": bf_s}}
    for n_probes in probes:
        m, s = bench_idx(
            lambda q, p=n_probes: ivf.search(q, k=k, n_probes=p, sim_fn=cosine_sim_numpy)
        )
        results[f"IVF n_probes={n_probes}"] = {"mean_ms": m, "std_ms": s}
        print(f"IVF n_probes={n_probes:<2}         : {m:7.3f} +/- {s:.3f} ms/query   "
              f"({bf_m/m:5.2f}x vs BruteForce)")

    for d in results.values():
        d["speedup_vs_bruteforce"] = bf_m / d["mean_ms"]
    return results


def plot_index(build_results: dict, query_results: dict, N: int, n_clusters: int) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Skip metadata keys like "_kmeans_impl"
    b_names = [k for k in build_results.keys() if not k.startswith("_")]
    b_ms    = [build_results[n]["ms"] for n in b_names]
    b_colors = ["#95a5a6", "#3498db", "#27ae60"][:len(b_names)]
    bars1 = ax1.bar(b_names, b_ms, color=b_colors, edgecolor="black")
    ax1.set_ylabel("Build time (ms)")
    ax1.set_title(f"Index build time - N={N}, K={n_clusters}")
    ax1.tick_params(axis="x", rotation=15)
    for bar, m in zip(bars1, b_ms):
        ax1.text(bar.get_x() + bar.get_width()/2, m, f"{m:.0f} ms",
                 ha="center", va="bottom", fontsize=9)

    q_names  = list(query_results.keys())
    q_means  = [query_results[n]["mean_ms"] for n in q_names]
    q_stds   = [query_results[n].get("std_ms", 0.0) for n in q_names]
    q_speed  = [query_results[n]["speedup_vs_bruteforce"] for n in q_names]
    q_colors = ["#e74c3c"] + ["#27ae60"] * (len(q_names) - 1)

    bars2 = ax2.bar(q_names, q_means, yerr=q_stds, color=q_colors,
                    edgecolor="black", capsize=4)
    ax2.set_ylabel("Query latency (ms)")
    ax2.set_title("Query latency vs BruteForce (lower = faster)")
    ax2.tick_params(axis="x", rotation=20)
    for bar, m, s in zip(bars2, q_means, q_speed):
        ax2.text(bar.get_x() + bar.get_width()/2, m, f"{m:.2f} ms\n({s:.1f}x)",
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()
    print("\nHigher n_probes -> slower query but higher recall (verified in Section 5).")


# =====================================================================
# Section 5 -- Retrieval quality (Recall@K, MRR)
# =====================================================================

def _load_chunk_to_passage_text(data_dir: Path, chunks: list) -> dict:
    """
    Build {chunk_id -> source passage text}. MS MARCO relevance is at the
    passage level, so we map retrieved chunks back to their source passage
    before comparing with relevant_passages.
    """
    passages = {}
    with open(data_dir / "passages.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            passages[p["id"]] = p["text"]
    return {c["id"]: passages.get(c["source_id"], c["text"]) for c in chunks}


def run_retrieval_quality(
    vectors: np.ndarray,
    chunks: list,
    queries: list,
    data_dir: Path,
    bf,
    ivf,
    k: int = 10,
    probes: tuple = (1, 2, 4, 8),
    n_queries=None,
) -> dict:
    """
    For each (index, config): embed real queries, search, compute Recall@K + MRR
    against MS MARCO relevance judgments.
    Returns {config_name: {recall@k, mrr, mean_latency_ms}}.
    """
    from components.embedder import LocalEmbedder
    from components.similarity import cosine_sim_numpy

    chunk_to_passage_text = _load_chunk_to_passage_text(data_dir, chunks)

    good = [q for q in queries if q.get("relevant_passages")]
    if n_queries is not None:
        good = good[:n_queries]
    print(f"Evaluating on {len(good)} queries with relevance judgments.")

    print("Embedding query texts (CPU)...")
    embedder = LocalEmbedder(device="cpu")
    t0 = time.perf_counter()
    query_vecs = embedder.embed_texts([q["text"] for q in good], show_progress=False)
    print(f"  embedded {len(good)} queries in {(time.perf_counter()-t0)*1000:.0f} ms")

    def evaluate(search_fn) -> dict:
        recalls, mrrs, lats = [], [], []
        for q, qv in zip(good, query_vecs):
            relevant = set(q["relevant_passages"])
            t0 = time.perf_counter()
            results = search_fn(qv)
            lats.append((time.perf_counter() - t0) * 1000)

            retrieved_ids = [doc_id for doc_id, _ in results]
            retrieved_texts = {chunk_to_passage_text.get(did, "") for did in retrieved_ids}
            recalls.append(len(retrieved_texts & relevant) / len(relevant))

            rr = 0.0
            for rank, did in enumerate(retrieved_ids, start=1):
                if chunk_to_passage_text.get(did, "") in relevant:
                    rr = 1.0 / rank
                    break
            mrrs.append(rr)

        return {
            "recall@k":        float(np.mean(recalls)),
            "mrr":             float(np.mean(mrrs)),
            "mean_latency_ms": float(np.mean(lats)),
        }

    results: dict = {}
    print("Running BruteForce (ground truth)...")
    results["BruteForce"] = evaluate(
        lambda qv: bf.search(qv, k=k, sim_fn=cosine_sim_numpy)
    )
    for n_probes in probes:
        label = f"IVF n_probes={n_probes}"
        print(f"Running {label}...")
        results[label] = evaluate(
            lambda qv, p=n_probes: ivf.search(qv, k=k, n_probes=p, sim_fn=cosine_sim_numpy)
        )

    bf_recall = results["BruteForce"]["recall@k"]
    print(f"\n{'Config':<22s} {'Recall@K':>10s} {'MRR':>10s} {'Lat (ms)':>10s} {'% of BF':>10s}")
    print("-" * 70)
    for name, d in results.items():
        pct = d["recall@k"] / bf_recall * 100 if bf_recall else 0
        print(f"{name:<22s} {d['recall@k']:>10.4f} {d['mrr']:>10.4f} "
              f"{d['mean_latency_ms']:>10.3f} {pct:>9.1f}%")
    return results


def plot_retrieval_quality(results: dict, k: int = 10) -> None:
    names   = list(results.keys())
    recalls = [results[n]["recall@k"] for n in names]
    mrrs    = [results[n]["mrr"] for n in names]
    lats    = [results[n]["mean_latency_ms"] for n in names]

    colors = ["#e74c3c"] + ["#27ae60"] * (len(names) - 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    bars = ax1.bar(names, recalls, color=colors, edgecolor="black")
    ax1.set_ylabel(f"Recall@{k}")
    ax1.set_title(f"Recall@{k} - IVF vs ground truth")
    ax1.tick_params(axis="x", rotation=25)
    ax1.set_ylim(0, max(1.0, max(recalls) * 1.1))
    for bar, r in zip(bars, recalls):
        ax1.text(bar.get_x() + bar.get_width()/2, r, f"{r:.3f}",
                 ha="center", va="bottom", fontsize=9)

    bars = ax2.bar(names, mrrs, color=colors, edgecolor="black")
    ax2.set_ylabel("MRR")
    ax2.set_title("Mean Reciprocal Rank")
    ax2.tick_params(axis="x", rotation=25)
    ax2.set_ylim(0, max(1.0, max(mrrs) * 1.1))
    for bar, m in zip(bars, mrrs):
        ax2.text(bar.get_x() + bar.get_width()/2, m, f"{m:.3f}",
                 ha="center", va="bottom", fontsize=9)

    ax3.scatter(lats, recalls, c=colors, s=120, edgecolor="black", zorder=3)
    for n, l, r in zip(names, lats, recalls):
        ax3.annotate(n, (l, r), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax3.set_xlabel("Mean query latency (ms)")
    ax3.set_ylabel(f"Recall@{k}")
    ax3.set_title("Speed vs Quality tradeoff")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =====================================================================
# Section 6 -- Async embedding (simulated IO-bound API)
# =====================================================================

def run_async_embedding_benchmarks(
    chunks: list,
    n_texts: int = 640,
    batch_size: int = 64,
    simulated_latency_ms: float = 100.0,
    n_threads: int = 8,
    max_async: int = 16,
) -> dict:
    """
    Compare three strategies for embedding N texts through an API with
    `simulated_latency_ms` per batch:
      1. Sequential APIEmbedder (baseline)
      2. ThreadedAPIEmbedder (ThreadPoolExecutor)
      3. AsyncAPIEmbedder (asyncio + run_in_executor)
    """
    from components.embedder import APIEmbedder
    from optimized.async_embedder import ThreadedAPIEmbedder, AsyncAPIEmbedder

    texts = [c["text"] for c in chunks[:n_texts]]
    n_batches = (len(texts) + batch_size - 1) // batch_size
    ideal_seq_ms = n_batches * simulated_latency_ms
    ideal_conc_ms = simulated_latency_ms  # everything in parallel
    print(f"Embedding {len(texts)} texts  |  batch_size={batch_size}  "
          f"|  {n_batches} batches  |  {simulated_latency_ms:.0f} ms/batch")
    print(f"Ideal sequential: {ideal_seq_ms:.0f} ms")
    print(f"Ideal concurrent: ~{ideal_conc_ms:.0f} ms (all batches in parallel)\n")

    results: dict = {}

    # ---- 1. Sequential ----
    api = APIEmbedder(api_provider="simulated",
                      batch_size=batch_size,
                      simulated_latency_ms=simulated_latency_ms)
    t0 = time.perf_counter()
    _ = api.embed_texts(texts, show_progress=False)
    seq_ms = (time.perf_counter() - t0) * 1000
    results["Sequential"] = {"total_ms": seq_ms}
    print(f"Sequential          : {seq_ms:8.1f} ms")

    # ---- 2. Threaded ----
    threaded = ThreadedAPIEmbedder(api, n_workers=n_threads)
    t0 = time.perf_counter()
    _ = threaded.embed_texts(texts, show_progress=False)
    thr_ms = (time.perf_counter() - t0) * 1000
    results[f"Threaded (n={n_threads})"] = {"total_ms": thr_ms}
    print(f"Threaded (n={n_threads:<2})      : {thr_ms:8.1f} ms   "
          f"({seq_ms/thr_ms:5.2f}x vs sequential)")

    # ---- 3. Async ----
    async_emb = AsyncAPIEmbedder(api, max_concurrent=max_async)
    t0 = time.perf_counter()
    _ = async_emb.embed_texts(texts, show_progress=False)
    async_ms = (time.perf_counter() - t0) * 1000
    results[f"Async (max={max_async})"] = {"total_ms": async_ms}
    print(f"Async (max={max_async:<2})       : {async_ms:8.1f} ms   "
          f"({seq_ms/async_ms:5.2f}x vs sequential)")

    for d in results.values():
        d["speedup_vs_sequential"] = seq_ms / d["total_ms"]

    # Annotate metadata
    results["_meta"] = {
        "n_texts": len(texts),
        "n_batches": n_batches,
        "batch_size": batch_size,
        "latency_ms_per_batch": simulated_latency_ms,
        "ideal_concurrent_ms": ideal_conc_ms,
    }
    return results


def plot_async_embedding(results: dict) -> None:
    meta = results.get("_meta", {})
    named = {k: v for k, v in results.items() if not k.startswith("_")}
    names   = list(named.keys())
    totals  = [named[n]["total_ms"] for n in names]
    speedups = [named[n]["speedup_vs_sequential"] for n in names]

    colors = ["#e74c3c", "#3498db", "#27ae60"][:len(names)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bars1 = ax1.bar(names, totals, color=colors, edgecolor="black")
    ax1.set_ylabel("Total time (ms)")
    title = f"API embedding -- {meta.get('n_texts','?')} texts, " \
            f"{meta.get('n_batches','?')} batches @ {meta.get('latency_ms_per_batch','?')}ms/batch"
    ax1.set_title(title)
    ax1.tick_params(axis="x", rotation=15)
    if "ideal_concurrent_ms" in meta:
        ax1.axhline(y=meta["ideal_concurrent_ms"], linestyle="--",
                    color="gray", label=f"Ideal concurrent ({meta['ideal_concurrent_ms']:.0f} ms)")
        ax1.legend()
    for bar, t in zip(bars1, totals):
        ax1.text(bar.get_x() + bar.get_width()/2, t, f"{t:.0f} ms",
                 ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(names, speedups, color=colors, edgecolor="black")
    ax2.axhline(y=1.0, linestyle="--", color="black", linewidth=1, label="Sequential baseline")
    ax2.set_ylabel("Speedup vs sequential (x)")
    ax2.set_title("Concurrency speedup")
    ax2.tick_params(axis="x", rotation=15)
    ax2.legend()
    for bar, s in zip(bars2, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2, s, f"{s:.2f}x",
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()


# =====================================================================
# Section 7 -- Async generation (REAL ChatGPT-4o)
# =====================================================================

CHATGPT_BASE_URL = "https://kong-api.prod1.nyumc.org/gpt-4o/v1.3.0"


def make_chatgpt_generator(model: str = "gpt-4o", max_tokens: int = 128):
    """
    Build a BaselineGenerator wired to NYU's ChatGPT-4o proxy.
    Reads KONG_API_KEY from env. Returns the generator.
    """
    from components.generator import BaselineGenerator
    api_key = os.environ["KONG_API_KEY"]
    return BaselineGenerator(
        api_provider="openai",
        api_key=api_key,
        model_name=model,
        base_url=CHATGPT_BASE_URL,
        extra_headers={"api-key": api_key},
        max_tokens=max_tokens,
    )


def prepare_generation_items(
    queries: list,
    chunks: list,
    bf,
    n_items: int = 8,
    k: int = 3,
) -> list:
    """
    Build (query_text, contexts) tuples for real generation benchmarks.

    For each of the first `n_items` queries, embed the text, retrieve top-k
    chunks via BruteForce, and collect their text as context.
    """
    from components.embedder import LocalEmbedder
    from components.similarity import cosine_sim_numpy

    selected = queries[:n_items]
    chunk_by_id = {c["id"]: c["text"] for c in chunks}

    print(f"Embedding {len(selected)} queries for retrieval (CPU)...")
    embedder = LocalEmbedder(device="cpu")
    qvecs = embedder.embed_texts([q["text"] for q in selected], show_progress=False)

    items = []
    for q, qv in zip(selected, qvecs):
        results = bf.search(qv, k=k, sim_fn=cosine_sim_numpy)
        contexts = [chunk_by_id.get(doc_id, "") for doc_id, _ in results]
        items.append((q["text"], contexts))

    print(f"Prepared {len(items)} (query, contexts) pairs. Sample query: "
          f"{items[0][0][:70]!r}")
    return items


def run_async_generation_benchmarks(
    items: list,
    n_threads: int = 8,
    max_async: int = 8,
    model: str = "gpt-4o",
    max_tokens: int = 128,
    verbose: bool = True,
) -> dict:
    """
    Run the same batch of (query, contexts) pairs through the ChatGPT-4o API
    in three modes:
      1. Sequential (BaselineGenerator.generate_batch)
      2. Threaded   (ThreadedGenerator, n_workers)
      3. Async      (AsyncGenerator, max_concurrent)

    Each method hits the REAL endpoint. Expect ~500-2000 ms per call.
    """
    from optimized.async_generator import ThreadedGenerator, AsyncGenerator

    n = len(items)
    results: dict = {}
    sample_answer = None

    # ---- 1. Sequential ----
    gen = make_chatgpt_generator(model=model, max_tokens=max_tokens)
    if verbose:
        print(f"\n[1/3] Sequential : sending {n} real API calls one-by-one...")
    t0 = time.perf_counter()
    seq_out = gen.generate_batch(items)
    seq_ms = (time.perf_counter() - t0) * 1000
    results["Sequential"] = {"total_ms": seq_ms, "n": n}
    sample_answer = seq_out[0]["answer"] if seq_out else None
    if verbose:
        print(f"      total: {seq_ms:8.1f} ms   "
              f"(mean per call: {seq_ms/n:6.1f} ms)")

    # ---- 2. Threaded ----
    gen = make_chatgpt_generator(model=model, max_tokens=max_tokens)
    threaded = ThreadedGenerator(gen, n_workers=n_threads)
    if verbose:
        print(f"[2/3] Threaded   : firing {n} calls with {n_threads} threads...")
    t0 = time.perf_counter()
    _ = threaded.generate_batch(items)
    thr_ms = (time.perf_counter() - t0) * 1000
    results[f"Threaded (n={n_threads})"] = {"total_ms": thr_ms, "n": n}
    if verbose:
        print(f"      total: {thr_ms:8.1f} ms   "
              f"({seq_ms/thr_ms:5.2f}x vs sequential)")

    # ---- 3. Async ----
    gen = make_chatgpt_generator(model=model, max_tokens=max_tokens)
    async_gen = AsyncGenerator(gen, max_concurrent=max_async)
    if verbose:
        print(f"[3/3] Async      : firing {n} calls with max_concurrent={max_async}...")
    t0 = time.perf_counter()
    _ = async_gen.generate_batch(items)
    async_ms = (time.perf_counter() - t0) * 1000
    results[f"Async (max={max_async})"] = {"total_ms": async_ms, "n": n}
    if verbose:
        print(f"      total: {async_ms:8.1f} ms   "
              f"({seq_ms/async_ms:5.2f}x vs sequential)")

    for d in results.values():
        d["speedup_vs_sequential"] = seq_ms / d["total_ms"]
        d["mean_per_call_ms"] = d["total_ms"] / d["n"]

    results["_meta"] = {
        "n_items": n,
        "model": model,
        "max_tokens": max_tokens,
        "sample_query": items[0][0] if items else None,
        "sample_answer": sample_answer,
    }

    if verbose and sample_answer:
        print(f"\nSample answer (first query):")
        print(f"  Q: {items[0][0][:80]}")
        print(f"  A: {sample_answer[:200]}")
    return results


def plot_async_generation(results: dict) -> None:
    meta = results.get("_meta", {})
    named = {k: v for k, v in results.items() if not k.startswith("_")}
    names    = list(named.keys())
    totals   = [named[n]["total_ms"] for n in names]
    speedups = [named[n]["speedup_vs_sequential"] for n in names]

    colors = ["#e74c3c", "#3498db", "#27ae60"][:len(names)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bars1 = ax1.bar(names, totals, color=colors, edgecolor="black")
    ax1.set_ylabel("Total time (ms)")
    ax1.set_title(f"ChatGPT {meta.get('model','gpt-4o')} generation -- "
                  f"{meta.get('n_items','?')} real API calls")
    ax1.tick_params(axis="x", rotation=15)
    for bar, t in zip(bars1, totals):
        ax1.text(bar.get_x() + bar.get_width()/2, t, f"{t:.0f} ms",
                 ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(names, speedups, color=colors, edgecolor="black")
    ax2.axhline(y=1.0, linestyle="--", color="black", linewidth=1, label="Sequential baseline")
    ax2.set_ylabel("Speedup vs sequential (x)")
    ax2.set_title("Concurrency speedup on real LLM API")
    ax2.tick_params(axis="x", rotation=15)
    ax2.legend()
    for bar, s in zip(bars2, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2, s, f"{s:.2f}x",
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()


# =====================================================================
# Step 4 -- Streaming generation (TTFT metric, Week 9)
# =====================================================================

def run_streaming_generation_benchmarks(
    items: list,
    model: str = "gpt-4o",
    max_tokens: int = 128,
    concurrent_workers: int = 8,
    verbose: bool = True,
) -> dict:
    """
    Three-way comparison of real ChatGPT-4o generation:
      1. Sequential, NON-streaming        (Section 7 baseline -- perceived = total)
      2. Sequential, streaming            (TTFT drops sharply, total unchanged)
      3. Concurrent, streaming (threaded) (best of both: low TTFT AND low total)

    TTFT (Time To First Token) is the user-perceived latency in a chat UI --
    the moment the answer starts appearing on screen.
    """
    from concurrent.futures import ThreadPoolExecutor
    n = len(items)
    results: dict = {}
    sample_answer = None

    def _summarize(name: str, latencies_total, latencies_ttft, batch_total_ms):
        results[name] = {
            "batch_total_ms": float(batch_total_ms),
            "mean_ttft_ms":   float(np.mean(latencies_ttft)),
            "p95_ttft_ms":    float(np.percentile(latencies_ttft, 95)),
            "mean_total_ms":  float(np.mean(latencies_total)),
            "p95_total_ms":   float(np.percentile(latencies_total, 95)),
            "n":              len(latencies_total),
        }

    # ---- 1. Sequential non-streaming (matches Section 7 baseline) ----
    gen = make_chatgpt_generator(model=model, max_tokens=max_tokens)
    if verbose:
        print(f"\n[1/3] Sequential NON-streaming : {n} real calls, one by one...")
    t0 = time.perf_counter()
    outs = [gen.generate(q, c) for q, c in items]
    seq_total_ms = (time.perf_counter() - t0) * 1000
    # For non-streaming, TTFT == total latency of that call
    seq_lats   = [o["latency_ms"] for o in outs]
    _summarize("Sequential (non-stream)", seq_lats, seq_lats, seq_total_ms)
    sample_answer = outs[0]["answer"] if outs else None
    if verbose:
        r = results["Sequential (non-stream)"]
        print(f"      batch total: {r['batch_total_ms']:.0f} ms | "
              f"mean TTFT: {r['mean_ttft_ms']:.0f} ms")

    # ---- 2. Sequential streaming ----
    gen = make_chatgpt_generator(model=model, max_tokens=max_tokens)
    if verbose:
        print(f"[2/3] Sequential STREAMING     : same {n} calls, stream=True...")
    t0 = time.perf_counter()
    stream_outs = [gen.generate_stream(q, c) for q, c in items]
    seq_stream_total_ms = (time.perf_counter() - t0) * 1000
    stream_totals = [o["total_ms"] for o in stream_outs]
    stream_ttfts  = [o["ttft_ms"]  for o in stream_outs]
    _summarize("Sequential (streaming)", stream_totals, stream_ttfts, seq_stream_total_ms)
    if verbose:
        r = results["Sequential (streaming)"]
        print(f"      batch total: {r['batch_total_ms']:.0f} ms | "
              f"mean TTFT: {r['mean_ttft_ms']:.0f} ms  "
              f"(TTFT -{(1 - r['mean_ttft_ms']/results['Sequential (non-stream)']['mean_ttft_ms'])*100:.0f}%)")

    # ---- 3. Concurrent streaming (ThreadPool + per-call streaming) ----
    gen = make_chatgpt_generator(model=model, max_tokens=max_tokens)
    if verbose:
        print(f"[3/3] CONCURRENT streaming     : {n} calls, {concurrent_workers} workers, stream=True...")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrent_workers) as pool:
        futures = [pool.submit(gen.generate_stream, q, c) for q, c in items]
        conc_outs = [f.result() for f in futures]
    conc_total_ms = (time.perf_counter() - t0) * 1000
    conc_totals = [o["total_ms"] for o in conc_outs]
    conc_ttfts  = [o["ttft_ms"]  for o in conc_outs]
    _summarize(f"Concurrent streaming (n={concurrent_workers})",
               conc_totals, conc_ttfts, conc_total_ms)
    if verbose:
        r = results[f"Concurrent streaming (n={concurrent_workers})"]
        print(f"      batch total: {r['batch_total_ms']:.0f} ms | "
              f"mean TTFT: {r['mean_ttft_ms']:.0f} ms")

    # ---- Compute speedups vs the Sequential non-streaming baseline ----
    base = results["Sequential (non-stream)"]
    for d in results.values():
        d["batch_speedup_vs_baseline"] = base["batch_total_ms"] / d["batch_total_ms"]
        d["ttft_speedup_vs_baseline"]  = base["mean_ttft_ms"]   / d["mean_ttft_ms"]

    results["_meta"] = {
        "n_items": n,
        "model": model,
        "max_tokens": max_tokens,
        "sample_answer": sample_answer,
    }

    if verbose and sample_answer:
        print(f"\nSample non-streaming answer (first query):")
        print(f"  A: {sample_answer[:200]}")
    return results


def plot_streaming_generation(results: dict) -> None:
    meta  = results.get("_meta", {})
    named = {k: v for k, v in results.items() if not k.startswith("_")}
    names = list(named.keys())

    totals      = [named[n]["batch_total_ms"] for n in names]
    mean_ttfts  = [named[n]["mean_ttft_ms"]   for n in names]
    mean_totals = [named[n]["mean_total_ms"]  for n in names]

    colors = ["#e74c3c", "#f39c12", "#27ae60"][:len(names)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: Batch total time ---
    ax = axes[0]
    bars = ax.bar(names, totals, color=colors, edgecolor="black")
    ax.set_ylabel("Batch total time (ms)")
    ax.set_title(f"Batch total time -- {meta.get('n_items','?')} calls")
    ax.tick_params(axis="x", rotation=18)
    for bar, t in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width()/2, t, f"{t:.0f} ms",
                ha="center", va="bottom", fontsize=9)

    # --- Panel 2: Mean per-call TTFT (perceived latency) ---
    ax = axes[1]
    bars = ax.bar(names, mean_ttfts, color=colors, edgecolor="black")
    ax.set_ylabel("Mean time-to-first-token (ms)")
    ax.set_title("Perceived latency (TTFT)")
    ax.tick_params(axis="x", rotation=18)
    for bar, t in zip(bars, mean_ttfts):
        ax.text(bar.get_x() + bar.get_width()/2, t, f"{t:.0f} ms",
                ha="center", va="bottom", fontsize=9)

    # --- Panel 3: Per-call total time ---
    ax = axes[2]
    bars = ax.bar(names, mean_totals, color=colors, edgecolor="black")
    ax.set_ylabel("Mean per-call total (ms)")
    ax.set_title("Per-call total time (full response)")
    ax.tick_params(axis="x", rotation=18)
    for bar, t in zip(bars, mean_totals):
        ax.text(bar.get_x() + bar.get_width()/2, t, f"{t:.0f} ms",
                ha="center", va="bottom", fontsize=9)

    fig.suptitle("Step 4 -- Streaming & concurrency on real gpt-4o calls",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


# =====================================================================
# Step 6 -- Friend's query-path optimizations (integration branch)
# =====================================================================

def warmup_friend_numba() -> None:
    """JIT-compile friend's precomputed-norms Numba kernel."""
    from optimized.similarity_numba import cosine_sim_numba_parallel_precomputed
    dummy_q = np.random.randn(384).astype(np.float32)
    dummy_c = np.random.randn(10, 384).astype(np.float32)
    dummy_norms = np.linalg.norm(dummy_c, axis=1)
    cosine_sim_numba_parallel_precomputed(dummy_q, dummy_c, dummy_norms)
    print("  Friend's Numba precomputed-norms kernel warmed up.")


def run_friend_query_benchmarks(
    ivf,
    vectors: np.ndarray,
    k: int = 10,
    probes: tuple = (2, 4, 8, 16),
    n_queries: int = 20,
    warmup: int = 2,
    repeats: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Ablation study of friend's three query-path optimizations, progressively
    enabled on TOP of the same IVF index instance (e.g. IVFIndexNumbaPP from
    Step 5 -- no rebuild needed, just flag toggles).

    Variants:
      A. flags off                    -- pre-merge Step 5 behavior
      B. + norm cache                 -- friend commit a578d95
      C. + np candidate gather        -- friend commit 06f6ed1
      D. + Numba parallel w/ norms    -- friend commit dcec67c
    """
    from components.similarity import cosine_sim_numpy
    from optimized.similarity_numba import cosine_sim_numba_parallel_precomputed

    if verbose:
        print("Warming up friend's Numba kernel...")
    warmup_friend_numba()

    query_vecs = vectors[:n_queries].astype(np.float32)

    def bench(search_fn):
        for q in query_vecs[:warmup]:
            _ = search_fn(q)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for q in query_vecs:
                _ = search_fn(q)
            times.append((time.perf_counter() - t0) * 1000 / len(query_vecs))
        return float(np.mean(times)), float(np.std(times))

    # (label, sim_fn, use_precomputed_norms, use_numpy_candidate_gather)
    variants = [
        ("A_flags_off",               cosine_sim_numpy,                        False, False),
        ("B_norm_cache",              cosine_sim_numpy,                        True,  False),
        ("C_norm_cache+np_gather",    cosine_sim_numpy,                        True,  True),
        ("D_numba_par_precomp",       cosine_sim_numba_parallel_precomputed,   True,  True),
    ]

    results: dict = {}
    header = f"\n{'Variant':<28s} " + " ".join(f"np={p:<4d}" for p in probes)
    if verbose:
        print(header)
        print("-" * len(header))

    for label, sim_fn, use_norms, use_gather in variants:
        by_probe = {}
        for n_probes in probes:
            m, s = bench(
                lambda q, p=n_probes, sfn=sim_fn, un=use_norms, ug=use_gather:
                    ivf.search(q, k=k, n_probes=p, sim_fn=sfn,
                               use_precomputed_norms=un,
                               use_numpy_candidate_gather=ug)
            )
            by_probe[f"np={n_probes}"] = {"mean_ms": m, "std_ms": s}
        results[label] = by_probe

        if verbose:
            row = f"{label:<28s} "
            for p in probes:
                ms = by_probe[f"np={p}"]["mean_ms"]
                row += f"{ms:7.2f} "
            print(row)

    # Compute per-probe speedup relative to variant A (flags off)
    base = results["A_flags_off"]
    for label, by_probe in results.items():
        for p_key, d in by_probe.items():
            d["speedup_vs_A"] = base[p_key]["mean_ms"] / d["mean_ms"]

    if verbose:
        print(f"\nSpeedup vs 'flags off' baseline (A):")
        for label in [v[0] for v in variants]:
            row = f"{label:<28s} "
            for p in probes:
                s = results[label][f"np={p}"]["speedup_vs_A"]
                row += f"{s:6.2f}x "
            print(row)

    results["_meta"] = {
        "n_queries_tested": n_queries,
        "probes":           list(probes),
        "k":                k,
        "index_class":      type(ivf).__name__,
    }
    return results


def plot_friend_benchmarks(results: dict) -> None:
    """Grouped bar chart: one group per n_probes, 4 bars per group (variants)."""
    meta = results.get("_meta", {})
    probes = meta.get("probes", [2, 4, 8, 16])

    variant_order = [
        "A_flags_off",
        "B_norm_cache",
        "C_norm_cache+np_gather",
        "D_numba_par_precomp",
    ]
    colors = ["#e74c3c", "#f39c12", "#3498db", "#27ae60"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # --- Panel 1: absolute latency ---
    x = np.arange(len(probes))
    width = 0.2
    for i, label in enumerate(variant_order):
        if label not in results:
            continue
        values = [results[label][f"np={p}"]["mean_ms"] for p in probes]
        ax1.bar(x + i * width, values, width, label=label, color=colors[i], edgecolor="black")
    ax1.set_xlabel("n_probes")
    ax1.set_ylabel("Latency (ms/query)")
    ax1.set_title("Friend's query-path optimizations: per-probe latency")
    ax1.set_xticks(x + 1.5 * width)
    ax1.set_xticklabels([str(p) for p in probes])
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # --- Panel 2: speedup vs A ---
    for i, label in enumerate(variant_order):
        if label not in results:
            continue
        values = [results[label][f"np={p}"]["speedup_vs_A"] for p in probes]
        ax2.bar(x + i * width, values, width, label=label, color=colors[i], edgecolor="black")
    ax2.axhline(y=1.0, linestyle="--", color="black", linewidth=1)
    ax2.set_xlabel("n_probes")
    ax2.set_ylabel("Speedup vs variant A (x)")
    ax2.set_title("Cumulative speedup by enabling each flag")
    ax2.set_xticks(x + 1.5 * width)
    ax2.set_xticklabels([str(p) for p in probes])
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()


# =====================================================================
# End-to-end COMBO comparison (stacks components, real ChatGPT-4o)
# =====================================================================

def run_endtoend_combos(
    queries: list,
    chunks: list,
    bf,
    ivf,
    data_dir,
    n_items: int = 8,
    k: int = 3,
    n_probes: int = 8,
    n_async_workers: int = 8,
    max_tokens: int = 128,
    llm_mode: str = "async",         # "async" | "threaded" | "sequential"
    embed_mode: str = "batch",       # "batch" | "per_query"
    verbose: bool = True,
) -> dict:
    """
    Stack different retrieval/similarity components together with the SAME async
    LLM generation, then time the FULL end-to-end pipeline for N queries.

    Combos compared (each is one full RAG pipeline):
      A) BruteForce + cosine_sim_numpy (no optimizations) + Async LLM
      B) IVF + cosine_sim_numpy + norm_cache + np_gather   + Async LLM
      C) IVF + cosine_sim_numba_parallel_precomputed + cache + gather + Async LLM

    Per combo, measures: Recall@K, retrieval latency (per-query and total),
    generation latency (per-call and total), and end-to-end batch time.
    """
    from components.embedder import LocalEmbedder
    from components.similarity import cosine_sim_numpy
    from optimized.similarity_numba import (
        cosine_sim_numba_parallel_precomputed,
    )
    from optimized.async_generator import AsyncGenerator, ThreadedGenerator

    if llm_mode not in ("async", "threaded", "sequential"):
        raise ValueError(f"llm_mode must be 'async' | 'threaded' | 'sequential'; got {llm_mode!r}")
    if embed_mode not in ("batch", "per_query"):
        raise ValueError(f"embed_mode must be 'batch' | 'per_query'; got {embed_mode!r}")

    chunk_to_passage = _load_chunk_to_passage_text(data_dir, chunks)
    chunk_by_id = {c["id"]: c["text"] for c in chunks}

    good = [q for q in queries if q.get("relevant_passages")][:n_items]
    if verbose:
        mode_str = {
            "async": f"async (max={n_async_workers})",
            "threaded": f"threaded (n={n_async_workers})",
            "sequential": "sequential",
        }[llm_mode]
        print(f"\n{'='*72}")
        print(f"END-TO-END COMBO COMPARISON ({len(good)} queries, K={k}, LLM={mode_str})")
        print(f"{'='*72}")

    # Pre-embed all queries once (shared across combos)
    embedder = LocalEmbedder(device="cpu")
    if verbose:
        print("Warming up embedder + Numba JIT...")
    _ = embedder.embed_query("warmup")
    warmup_friend_numba()

    # Batch-embed all queries up front IF requested. In per-query mode each combo
    # embeds inside its own loop, paying per-query model overhead, and reports its
    # own embed timing.
    if embed_mode == "batch":
        t_embed = time.perf_counter()
        q_vecs = embedder.embed_texts([q["text"] for q in good], show_progress=False)
        embed_total_ms_batch = (time.perf_counter() - t_embed) * 1000
    else:
        q_vecs = None
        embed_total_ms_batch = 0.0

    ivf_n_clusters = getattr(ivf, "n_clusters", "?")
    ivf_label = f"IVF({ivf_n_clusters},{n_probes})"

    combos = [
        {
            "label": "A. BF + NumPy (no opts)",
            "index": bf,
            "sim_fn": cosine_sim_numpy,
            "search_kwargs": {"use_precomputed_norms": False},
            "components": {
                "index":      "BruteForce",
                "sim_fn":     "cosine_sim_numpy",
                "norm_cache": "OFF",
                "np_gather":  "N/A",
            },
        },
        {
            "label": "B. IVF + NumPy + cache + gather",
            "index": ivf,
            "sim_fn": cosine_sim_numpy,
            "search_kwargs": {
                "n_probes": n_probes,
                "use_precomputed_norms": True,
                "use_numpy_candidate_gather": True,
            },
            "components": {
                "index":      ivf_label,
                "sim_fn":     "cosine_sim_numpy",
                "norm_cache": "ON",
                "np_gather":  "ON",
            },
        },
        {
            "label": "C. IVF + Numba par + cache + gather",
            "index": ivf,
            "sim_fn": cosine_sim_numba_parallel_precomputed,
            "search_kwargs": {
                "n_probes": n_probes,
                "use_precomputed_norms": True,
                "use_numpy_candidate_gather": True,
            },
            "components": {
                "index":      ivf_label,
                "sim_fn":     "cosine_sim_numba_parallel_precomputed",
                "norm_cache": "ON",
                "np_gather":  "ON",
            },
        },
    ]

    results: dict = {}
    for combo in combos:
        label = combo["label"]
        if verbose:
            print(f"\n--- {label} ---")

        # ---- Stage 1: retrieval (embed if per-query, search, gather contexts, compute recall) ----
        recalls, items = [], []
        embed_ms_combo = 0.0
        search_ms_combo = 0.0
        for i, q in enumerate(good):
            # 1) Embed
            if embed_mode == "batch":
                qv = q_vecs[i]   # already embedded outside, no per-query cost
            else:
                t_e = time.perf_counter()
                qv = embedder.embed_query(q["text"])
                embed_ms_combo += (time.perf_counter() - t_e) * 1000

            # 2) Search (timed alone)
            t_s = time.perf_counter()
            r = combo["index"].search(qv, k=k, sim_fn=combo["sim_fn"], **combo["search_kwargs"])
            search_ms_combo += (time.perf_counter() - t_s) * 1000

            # 3) Untimed bookkeeping
            retrieved_ids = [doc_id for doc_id, _ in r]
            retrieved_texts = {chunk_to_passage.get(did, "") for did in retrieved_ids}
            relevant = set(q["relevant_passages"])
            rec = len(retrieved_texts & relevant) / len(relevant) if relevant else 0.0
            recalls.append(rec)
            contexts = [chunk_by_id.get(did, "") for did, _ in r]
            items.append((q["text"], contexts))

        embed_total_ms    = embed_total_ms_batch if embed_mode == "batch" else embed_ms_combo
        retrieve_total_ms = search_ms_combo

        # ---- Stage 2: LLM generation on all N prepared items (mode-dependent) ----
        gen = make_chatgpt_generator(model="gpt-4o", max_tokens=max_tokens)
        if llm_mode == "async":
            wrapped = AsyncGenerator(gen, max_concurrent=n_async_workers)
        elif llm_mode == "threaded":
            wrapped = ThreadedGenerator(gen, n_workers=n_async_workers)
        else:  # sequential
            wrapped = gen   # BaselineGenerator.generate_batch is serial
        t0 = time.perf_counter()
        _ = wrapped.generate_batch(items)
        gen_total_ms = (time.perf_counter() - t0) * 1000

        end_to_end_ms = embed_total_ms + retrieve_total_ms + gen_total_ms

        results[label] = {
            "components":            combo["components"],
            "recall@k":              float(np.mean(recalls)),
            "embed_total_ms":        embed_total_ms,
            "embed_per_query_ms":    embed_total_ms / len(good),
            "retrieve_total_ms":     retrieve_total_ms,
            "retrieve_per_query_ms": retrieve_total_ms / len(good),
            "gen_total_ms":          gen_total_ms,
            "gen_per_call_ms":       gen_total_ms / len(good),
            "end_to_end_ms":         end_to_end_ms,
            "n":                     len(good),
        }

        if verbose:
            r = results[label]
            print(f"  Recall@{k}        : {r['recall@k']:.4f}")
            print(f"  Retrieve total   : {r['retrieve_total_ms']:7.1f} ms "
                  f"({r['retrieve_per_query_ms']:.2f} ms/query)")
            print(f"  Gen total        : {r['gen_total_ms']:7.1f} ms "
                  f"({r['gen_per_call_ms']:.0f} ms/call amortized)")
            print(f"  End-to-end batch : {end_to_end_ms:7.1f} ms")

    # Compute speedups vs config A
    base_e2e = results[combos[0]["label"]]["end_to_end_ms"]
    base_retrieve = results[combos[0]["label"]]["retrieve_per_query_ms"]
    for lbl, d in results.items():
        d["end_to_end_speedup_vs_A"] = base_e2e / d["end_to_end_ms"]
        d["retrieve_speedup_vs_A"]   = base_retrieve / d["retrieve_per_query_ms"]

    if verbose:
        print(f"\n{'='*120}")
        print(f"  SUMMARY ({len(good)} queries, K={k}, LLM={mode_str}, "
              f"non-streaming, real ChatGPT-4o)")
        print(f"{'='*120}")
        print(f"  {'Index':<11s} {'Sim fn':<37s} {'norm':>5s} {'gather':>7s}  "
              f"{'Recall':>7s} {'Retrieve/q':>11s} {'Gen/call':>10s} {'E2E':>9s} {'E2E spd':>9s}")
        print(f"  {'-'*118}")
        for label, r in results.items():
            c = r["components"]
            print(f"  {c['index']:<11s} {c['sim_fn']:<37s} "
                  f"{c['norm_cache']:>5s} {c['np_gather']:>7s}  "
                  f"{r['recall@k']:>7.4f} "
                  f"{r['retrieve_per_query_ms']:>8.2f} ms "
                  f"{r['gen_per_call_ms']:>7.0f} ms "
                  f"{r['end_to_end_ms']/1000:>7.2f} s "
                  f"{r['end_to_end_speedup_vs_A']:>7.2f}x")

    results["_meta"] = {
        "n_items":         len(good),
        "k":               k,
        "n_probes":        n_probes,
        "n_async_workers": n_async_workers,
        "max_tokens":      max_tokens,
        "llm_mode":        llm_mode,
    }
    return results


def run_endtoend_combos_grid(
    queries: list,
    chunks: list,
    bf,
    ivf,
    data_dir,
    n_items: int = 8,
    k: int = 3,
    n_probes: int = 8,
    n_async_workers: int = 8,
    max_tokens: int = 128,
    llm_modes: tuple = ("async", "threaded"),
    embed_mode: str = "batch",         # "batch" | "per_query"
    verbose: bool = True,
) -> dict:
    """
    Run run_endtoend_combos for EVERY LLM mode in `llm_modes` (default: async + threaded),
    producing a 3 retrieval combos x N llm modes grid. At the end prints a single
    unified table so you can see the full grid in one shot.

    Total real ChatGPT calls = len(combos in run_endtoend_combos) * len(llm_modes) * n_items
    Default: 3 * 2 * 8 = 48 calls, ~$0.20.
    """
    grid: dict = {}
    for mode in llm_modes:
        if verbose:
            print(f"\n{'#'*72}")
            print(f"#  GRID: LLM mode = {mode}")
            print(f"{'#'*72}")
        grid[mode] = run_endtoend_combos(
            queries, chunks, bf, ivf, data_dir,
            n_items=n_items, k=k, n_probes=n_probes,
            n_async_workers=n_async_workers, max_tokens=max_tokens,
            llm_mode=mode, embed_mode=embed_mode, verbose=verbose,
        )

    # Unified grid summary
    if verbose:
        # Pull n_clusters off the IVF object for the constants header
        ivf_n_clusters = getattr(ivf, "n_clusters", "?")
        n_combos = sum(
            1 for label in next(iter(grid.values())).keys()
            if not label.startswith("_")
        ) if grid else 0

        print(f"\n{'='*128}")
        print(f"  COMPONENTS HELD CONSTANT (same for every row below)")
        print(f"  {'-'*124}")
        print(f"  Embedder      : LocalEmbedder (all-MiniLM-L6-v2, CPU; "
              f"queries pre-embedded once and shared across combos)")
        print(f"  LLM model     : gpt-4o via ChatGPT proxy")
        print(f"  Generation    : non-streaming (full response per call)")
        print(f"  Max tokens    : {max_tokens}")
        print(f"  N queries     : {n_items}")
        print(f"  Top-K         : {k}")
        print(f"  IVF setup     : {ivf_n_clusters} clusters, n_probes={n_probes}")
        print(f"  LLM workers   : {n_async_workers}  (used by 'async' max_concurrent / 'threaded' n_workers)")
        print(f"{'='*128}")
        print(f"  GRID: {n_combos} retrieval combos x {len(grid)} LLM modes "
              f"= {n_combos*len(grid)} pipelines (each timed end-to-end)")
        print(f"{'='*128}")
        print(f"  {'LLM mode':<11s} {'Index':<11s} {'Sim fn':<37s} "
              f"{'norm':>5s} {'gather':>7s}  "
              f"{'Recall':>7s} {'Retrieve/q':>11s} {'Gen/call':>10s} {'E2E':>9s}")
        print(f"  {'-'*126}")
        for mode, results in grid.items():
            for label, r in results.items():
                if label.startswith("_"):
                    continue
                c = r["components"]
                print(f"  {mode:<11s} {c['index']:<11s} {c['sim_fn']:<37s} "
                      f"{c['norm_cache']:>5s} {c['np_gather']:>7s}  "
                      f"{r['recall@k']:>7.4f} "
                      f"{r['retrieve_per_query_ms']:>8.2f} ms "
                      f"{r['gen_per_call_ms']:>7.0f} ms "
                      f"{r['end_to_end_ms']/1000:>7.2f} s")
        print(f"  {'='*126}")

        # Find the global best E2E and report which full pipeline won
        best_label, best_mode, best_e2e, best_comp = None, None, float("inf"), None
        for mode, results in grid.items():
            for label, r in results.items():
                if label.startswith("_"):
                    continue
                if r["end_to_end_ms"] < best_e2e:
                    best_e2e   = r["end_to_end_ms"]
                    best_label = label
                    best_mode  = mode
                    best_comp  = r["components"]
        print(f"\n  Best end-to-end pipeline:")
        print(f"    LLM mode    : {best_mode}")
        print(f"    Index       : {best_comp['index']}")
        print(f"    Sim fn      : {best_comp['sim_fn']}")
        print(f"    norm cache  : {best_comp['norm_cache']}")
        print(f"    np gather   : {best_comp['np_gather']}")
        print(f"    -> {best_e2e/1000:.2f} s for {n_items} queries end-to-end")

    grid["_meta"] = {
        "n_items":   n_items,
        "k":         k,
        "n_probes":  n_probes,
        "llm_modes": list(llm_modes),
    }
    return grid


# =====================================================================
# Step 5 -- Pipeline parallelism (Week 10/11)
# =====================================================================

def run_pipeline_benchmarks(
    queries: list,
    chunks: list,
    bf,
    n_items: int = 8,
    k: int = 3,
    n_embed_workers: int = 4,
    n_gen_workers: int = 8,
    model: str = "gpt-4o",
    max_tokens: int = 128,
    verbose: bool = True,
) -> dict:
    """
    Two end-to-end RAG-serving patterns, real ChatGPT-4o calls:

      A) SEQUENTIAL (naive baseline):
         For each query in order: embed -> search -> gen (streaming).
         No overlap between any stages.
         Total  = N * (embed + search + gen)

      B) PIPELINED (Week 10/11):
         Two thread pools running in parallel:
           - retrieval pool (`n_embed_workers` workers) handles embed+search
           - generation pool (`n_gen_workers` workers) handles gpt-4o streaming
         As each retrieval future finishes, it *submits* its (query, contexts)
         to the generation pool from WITHIN the retrieval worker, so gen of
         query i overlaps with retrieval of query i+1.
         Total  ~=  max(retrieval_total, gen_concurrent) + small overhead

    Use this to demonstrate the pipelined multi-stage concurrency pattern.
    """
    from concurrent.futures import ThreadPoolExecutor
    from components.embedder import LocalEmbedder
    from components.similarity import cosine_sim_numpy

    chunk_by_id = {c["id"]: c["text"] for c in chunks}
    selected   = queries[:n_items]
    embedder   = LocalEmbedder(device="cpu")

    results: dict = {}

    # ---- A. Sequential naive baseline ----
    gen = make_chatgpt_generator(model=model, max_tokens=max_tokens)
    if verbose:
        print(f"\n[1/2] Sequential (naive): {n_items} queries, one at a time "
              f"(embed -> search -> gen_stream)...")
    t0 = time.perf_counter()
    for q in selected:
        qv = embedder.embed_query(q["text"])
        r  = bf.search(qv, k=k, sim_fn=cosine_sim_numpy)
        ctx = [chunk_by_id.get(did, "") for did, _ in r]
        _ = gen.generate_stream(q["text"], ctx)
    seq_ms = (time.perf_counter() - t0) * 1000
    results["Sequential (naive)"] = {
        "total_ms": seq_ms, "n": n_items, "mean_ms_per_query": seq_ms / n_items,
    }
    if verbose:
        print(f"      total: {seq_ms:.0f} ms   ({seq_ms/n_items:.0f} ms/query)")

    # ---- B. Pipelined: retrieval pool -> gen pool, overlapping ----
    gen = make_chatgpt_generator(model=model, max_tokens=max_tokens)
    if verbose:
        print(f"[2/2] Pipelined: retrieve pool (n={n_embed_workers}) + "
              f"gen pool (n={n_gen_workers}), overlapping stages...")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_embed_workers) as retrieve_pool, \
         ThreadPoolExecutor(max_workers=n_gen_workers) as gen_pool:

        def retrieve_and_submit(q_obj):
            # Runs on a retrieval worker. As soon as retrieval finishes, we
            # hand off to the gen pool -- gen of this query overlaps with
            # retrieval of subsequent queries (which are still on retrieval pool).
            qv = embedder.embed_query(q_obj["text"])
            r  = bf.search(qv, k=k, sim_fn=cosine_sim_numpy)
            ctx = [chunk_by_id.get(did, "") for did, _ in r]
            return gen_pool.submit(gen.generate_stream, q_obj["text"], ctx)

        # Every retrieval future's .result() is ITSELF a gen future
        retrieve_futures = [retrieve_pool.submit(retrieve_and_submit, q)
                            for q in selected]
        gen_futures = [rf.result() for rf in retrieve_futures]
        _ = [gf.result() for gf in gen_futures]
    pipe_ms = (time.perf_counter() - t0) * 1000
    label = f"Pipelined (embed={n_embed_workers}, gen={n_gen_workers})"
    results[label] = {
        "total_ms": pipe_ms, "n": n_items, "mean_ms_per_query": pipe_ms / n_items,
    }
    if verbose:
        print(f"      total: {pipe_ms:.0f} ms   ({pipe_ms/n_items:.0f} ms/query)   "
              f"{seq_ms/pipe_ms:.2f}x vs sequential")

    for d in results.values():
        d["speedup_vs_sequential"] = seq_ms / d["total_ms"]

    results["_meta"] = {
        "n_items": n_items,
        "n_embed_workers": n_embed_workers,
        "n_gen_workers": n_gen_workers,
        "model": model,
        "max_tokens": max_tokens,
    }
    return results


def plot_pipeline(results: dict) -> None:
    meta  = results.get("_meta", {})
    named = {k: v for k, v in results.items() if not k.startswith("_")}
    names = list(named.keys())
    totals   = [named[n]["total_ms"] / 1000 for n in names]          # seconds
    per_q    = [named[n]["mean_ms_per_query"] for n in names]        # ms/query
    speedups = [named[n]["speedup_vs_sequential"] for n in names]

    colors = ["#e74c3c", "#27ae60"][:len(names)]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    bars = ax1.bar(names, totals, color=colors, edgecolor="black")
    ax1.set_ylabel("Total batch time (s)")
    ax1.set_title(f"End-to-end total ({meta.get('n_items','?')} real queries)")
    ax1.tick_params(axis="x", rotation=10)
    for bar, t in zip(bars, totals):
        ax1.text(bar.get_x() + bar.get_width()/2, t, f"{t:.2f}s",
                 ha="center", va="bottom", fontsize=9)

    bars = ax2.bar(names, per_q, color=colors, edgecolor="black")
    ax2.set_ylabel("Mean time per query (ms)")
    ax2.set_title("Amortized per-query time")
    ax2.tick_params(axis="x", rotation=10)
    for bar, t in zip(bars, per_q):
        ax2.text(bar.get_x() + bar.get_width()/2, t, f"{t:.0f} ms",
                 ha="center", va="bottom", fontsize=9)

    bars = ax3.bar(names, speedups, color=colors, edgecolor="black")
    ax3.axhline(y=1.0, linestyle="--", color="black", linewidth=1,
                label="Sequential baseline")
    ax3.set_ylabel("Speedup vs sequential (x)")
    ax3.set_title("Pipeline speedup")
    ax3.tick_params(axis="x", rotation=10)
    ax3.legend()
    for bar, s in zip(bars, speedups):
        ax3.text(bar.get_x() + bar.get_width()/2, s, f"{s:.2f}x",
                 ha="center", va="bottom", fontsize=9)

    fig.suptitle("Step 5 -- Pipelined RAG: retrieval pool || generation pool",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


# =====================================================================
# Section 8 -- Summary dashboard
# =====================================================================

def _best(results: dict, metric: str, higher_is_better: bool = True):
    """Pick the best (name, value) pair from a results dict for a given metric."""
    named = {k: v for k, v in results.items() if not k.startswith("_")}
    if not named:
        return None, None
    key_fn = (lambda kv: kv[1][metric]) if higher_is_better else (lambda kv: -kv[1][metric])
    name, d = max(named.items(), key=key_fn)
    return name, d[metric]


def print_summary(RESULTS: dict) -> None:
    """Executive summary across all sections."""
    dataset = RESULTS.get("dataset", "?")
    n_vec = RESULTS.get("n_vectors", "?")
    dim   = RESULTS.get("dim", "?")
    print("=" * 72)
    print(f"  RAG Optimization Benchmark Summary -- dataset={dataset}, "
          f"N={n_vec}, D={dim} (CPU only)")
    print("=" * 72)

    # Similarity
    sim = RESULTS.get("similarity", {})
    if sim:
        py_ms = sim["Pure Python (extrapolated)"]["mean_ms"]
        np_ms = sim["NumPy"]["mean_ms"]
        best_name, best_speedup = _best(sim, "speedup_vs_numpy")
        best_ms = sim[best_name]["mean_ms"]
        n_vec = RESULTS.get("n_vectors", "?")
        dim   = RESULTS.get("dim", "?")
        dataset = RESULTS.get("dataset", "?")
        print(f"\n[Section 2] Cosine similarity (per query, dataset={dataset}, "
              f"N={n_vec}, D={dim})")
        print(f"  Pure Python baseline:  {py_ms:10.2f} ms")
        print(f"  NumPy baseline:        {np_ms:10.2f} ms   ({py_ms/np_ms:6.1f}x vs Python)")
        print(f"  Best: {best_name:25s} {best_ms:10.2f} ms   ({py_ms/best_ms:6.1f}x vs Python, "
              f"{best_speedup:.2f}x vs NumPy)")

    # Index query
    q = RESULTS.get("index", {}).get("query", {})
    if q:
        bf_ms = q["BruteForce (NumPy)"]["mean_ms"]
        best_name, best_speedup = _best(q, "speedup_vs_bruteforce")
        best_ms = q[best_name]["mean_ms"]
        print("\n[Section 4] Index search (per query)")
        print(f"  BruteForce:            {bf_ms:10.3f} ms")
        print(f"  Best: {best_name:25s} {best_ms:10.3f} ms   ({best_speedup:.2f}x vs BruteForce)")

    # Quality
    qu = RESULTS.get("quality", {})
    if qu:
        bf_r = qu["BruteForce"]["recall@k"]
        print("\n[Section 5] Retrieval quality (Recall@10)")
        print(f"  BruteForce (ceiling):  {bf_r:.4f}")
        for name, d in qu.items():
            if name == "BruteForce":
                continue
            pct = d["recall@k"] / bf_r * 100 if bf_r else 0
            print(f"  {name:25s} {d['recall@k']:.4f}   ({pct:.1f}% of BF ceiling, "
                  f"{d['mean_latency_ms']:.2f} ms/query)")

    # Async embedding
    em = RESULTS.get("embedding", {})
    if em:
        seq_ms = em["Sequential"]["total_ms"]
        best_name, best_speedup = _best(em, "speedup_vs_sequential")
        best_ms = em[best_name]["total_ms"]
        meta = em.get("_meta", {})
        n = meta.get("n_batches", "?")
        print(f"\n[Section 6] API embedding ({n} simulated batches @ "
              f"{meta.get('latency_ms_per_batch','?')} ms/batch)")
        print(f"  Sequential:            {seq_ms:10.1f} ms")
        print(f"  Best: {best_name:25s} {best_ms:10.1f} ms   ({best_speedup:.2f}x)")

    # Async generation
    gen = RESULTS.get("generation", {})
    if gen:
        seq_ms = gen["Sequential"]["total_ms"]
        best_name, best_speedup = _best(gen, "speedup_vs_sequential")
        best_ms = gen[best_name]["total_ms"]
        meta = gen.get("_meta", {})
        n = meta.get("n_items", "?")
        print(f"\n[Section 7] LLM generation ({n} real gpt-4o calls via ChatGPT)")
        print(f"  Sequential:            {seq_ms:10.1f} ms  "
              f"(mean {seq_ms/n:.0f} ms/call)")
        print(f"  Best: {best_name:25s} {best_ms:10.1f} ms   ({best_speedup:.2f}x)")

    # End-to-end scenario
    print("\n" + "=" * 72)
    print("  End-to-end scenario: batch of 8 RAG queries")
    print("=" * 72)
    if sim and q and gen and em:
        # Baseline: sequential embed + BF + sequential generate
        baseline_per_query_ms = q["BruteForce (NumPy)"]["mean_ms"]
        baseline_gen_ms = gen["Sequential"]["total_ms"]
        baseline_total = 8 * baseline_per_query_ms + baseline_gen_ms

        # Optimized: best search + async generation (embedding already done)
        best_q_name, _ = _best(q, "speedup_vs_bruteforce")
        opt_per_query = q[best_q_name]["mean_ms"]
        best_gen_name, _ = _best(gen, "speedup_vs_sequential")
        opt_gen_ms = gen[best_gen_name]["total_ms"]
        opt_total = 8 * opt_per_query + opt_gen_ms

        print(f"  Baseline total  : {baseline_total/1000:6.2f} s  "
              f"(search {baseline_per_query_ms:.2f}ms x 8 + sequential gen {baseline_gen_ms/1000:.2f}s)")
        print(f"  Optimized total : {opt_total/1000:6.2f} s  "
              f"({best_q_name} + {best_gen_name})")
        print(f"  Speedup         : {baseline_total/opt_total:6.2f}x")
    print("=" * 72)


def plot_summary(RESULTS: dict) -> None:
    """One 2x3 dashboard panel of headline results."""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.3)

    red, blue, orange, green = "#e74c3c", "#3498db", "#f39c12", "#27ae60"

    # --- Panel 1: Similarity (Section 2) ---
    ax1 = fig.add_subplot(gs[0, 0])
    sim = RESULTS.get("similarity", {})
    if sim:
        names  = list(sim.keys())
        means  = [sim[n]["mean_ms"] for n in names]
        colors = [red, blue, orange, green][:len(names)]
        bars = ax1.bar(names, means, color=colors, edgecolor="black")
        ax1.set_yscale("log")
        ax1.set_ylabel("Latency (ms, log)")
        ax1.set_title("Section 2: Cosine similarity")
        ax1.tick_params(axis="x", rotation=20)
        for bar, m in zip(bars, means):
            lab = f"{m:.1f}" if m >= 1 else f"{m*1000:.0f}us"
            ax1.text(bar.get_x() + bar.get_width()/2, m * 1.15, lab,
                     ha="center", va="bottom", fontsize=8)

    # --- Panel 2: Index query (Section 4) ---
    ax2 = fig.add_subplot(gs[0, 1])
    q = RESULTS.get("index", {}).get("query", {})
    if q:
        names = list(q.keys())
        means = [q[n]["mean_ms"] for n in names]
        colors = [red] + [green] * (len(names) - 1)
        bars = ax2.bar(names, means, color=colors, edgecolor="black")
        ax2.set_ylabel("Latency (ms)")
        ax2.set_title("Section 4: Index query")
        ax2.tick_params(axis="x", rotation=25)
        for bar, m in zip(bars, means):
            ax2.text(bar.get_x() + bar.get_width()/2, m, f"{m:.2f}",
                     ha="center", va="bottom", fontsize=8)

    # --- Panel 3: Recall vs Latency (Section 5) ---
    ax3 = fig.add_subplot(gs[0, 2])
    qu = RESULTS.get("quality", {})
    if qu:
        names   = list(qu.keys())
        lats    = [qu[n]["mean_latency_ms"] for n in names]
        recalls = [qu[n]["recall@k"] for n in names]
        colors  = [red] + [green] * (len(names) - 1)
        ax3.scatter(lats, recalls, c=colors, s=120, edgecolor="black", zorder=3)
        for n, l, r in zip(names, lats, recalls):
            ax3.annotate(n, (l, r), xytext=(5, 5), textcoords="offset points",
                         fontsize=8)
        ax3.set_xlabel("Latency (ms/query)")
        ax3.set_ylabel("Recall@10")
        ax3.set_title("Section 5: Speed vs Quality")
        ax3.grid(True, alpha=0.3)

    # --- Panel 4: Async embedding (Section 6) ---
    ax4 = fig.add_subplot(gs[1, 0])
    em = RESULTS.get("embedding", {})
    if em:
        named = {k: v for k, v in em.items() if not k.startswith("_")}
        names  = list(named.keys())
        totals = [named[n]["total_ms"] for n in names]
        colors = [red, blue, green][:len(names)]
        bars = ax4.bar(names, totals, color=colors, edgecolor="black")
        ax4.set_ylabel("Total time (ms)")
        ax4.set_title("Section 6: API embedding (simulated)")
        ax4.tick_params(axis="x", rotation=15)
        for bar, t in zip(bars, totals):
            ax4.text(bar.get_x() + bar.get_width()/2, t, f"{t:.0f}",
                     ha="center", va="bottom", fontsize=8)

    # --- Panel 5: Async generation (Section 7) ---
    ax5 = fig.add_subplot(gs[1, 1])
    gen = RESULTS.get("generation", {})
    if gen:
        named = {k: v for k, v in gen.items() if not k.startswith("_")}
        names  = list(named.keys())
        totals = [named[n]["total_ms"] / 1000 for n in names]
        colors = [red, blue, green][:len(names)]
        bars = ax5.bar(names, totals, color=colors, edgecolor="black")
        ax5.set_ylabel("Total time (s)")
        ax5.set_title("Section 7: gpt-4o generation (real)")
        ax5.tick_params(axis="x", rotation=15)
        for bar, t in zip(bars, totals):
            ax5.text(bar.get_x() + bar.get_width()/2, t, f"{t:.2f}s",
                     ha="center", va="bottom", fontsize=8)

    # --- Panel 6: End-to-end story ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    lines = ["End-to-end: 8 RAG queries\n"]
    if q and gen:
        bf_ms  = q["BruteForce (NumPy)"]["mean_ms"]
        base_gen = gen["Sequential"]["total_ms"] / 1000
        base_total = (8 * bf_ms / 1000) + base_gen

        best_q_name, _ = _best(q, "speedup_vs_bruteforce")
        opt_q = q[best_q_name]["mean_ms"]
        best_gen_name, _ = _best(gen, "speedup_vs_sequential")
        opt_gen = gen[best_gen_name]["total_ms"] / 1000
        opt_total = (8 * opt_q / 1000) + opt_gen

        lines += [
            f"Baseline stack",
            f"  search (BruteForce) x 8 : {8*bf_ms/1000:6.3f} s",
            f"  generation (seq)        : {base_gen:6.3f} s",
            f"  TOTAL                   : {base_total:6.3f} s",
            "",
            f"Optimized stack",
            f"  search ({best_q_name}) x 8:",
            f"                            {8*opt_q/1000:6.3f} s",
            f"  generation ({best_gen_name}):",
            f"                            {opt_gen:6.3f} s",
            f"  TOTAL                   : {opt_total:6.3f} s",
            "",
            f"  SPEEDUP: {base_total/opt_total:.2f}x",
        ]
    ax6.text(0.0, 1.0, "\n".join(lines), fontsize=11, family="monospace",
             va="top", ha="left")
    ax6.set_title("End-to-end scenario")

    title_dataset = RESULTS.get("dataset", "unknown")
    fig.suptitle(f"RAG Optimization Dashboard -- dataset={title_dataset} (CPU only)",
                 fontsize=16, y=1.00)
    plt.tight_layout()
    plt.show()


# =====================================================================
# Result persistence (per-dataset JSON files)
# =====================================================================

RESULTS_DIR = PROJECT_ROOT / "results"


def _json_safe(o):
    """Recursively convert numpy scalars/arrays to plain Python for JSON."""
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return float(o) if isinstance(o, np.floating) else int(o)
    return o


def save_results(RESULTS: dict, dataset: str | None = None,
                 results_dir: Path | str = RESULTS_DIR) -> Path:
    """
    Save RESULTS to `results/<dataset>.json`.

    If `dataset` is None, reads RESULTS["dataset"]. Raises if neither is set.
    Overwrites the per-dataset file each run so the saved copy always reflects
    the most recent benchmark.
    """
    dataset = dataset or RESULTS.get("dataset")
    if not dataset:
        raise ValueError("No dataset name. Pass `dataset=` or set RESULTS['dataset'].")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{dataset}.json"

    payload = {
        "dataset":     dataset,
        "saved_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "results":     _json_safe(RESULTS),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {out_path.resolve()}  ({out_path.stat().st_size/1024:.1f} KB)")
    return out_path


def load_results(dataset: str, results_dir: Path | str = RESULTS_DIR) -> dict:
    """Load a previously-saved per-dataset results file. Returns RESULTS dict."""
    path = Path(results_dir) / f"{dataset}.json"
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    print(f"Loaded {path.name}  (saved at {payload.get('saved_at','?')})")
    return payload["results"]


def list_saved_results(results_dir: Path | str = RESULTS_DIR) -> list:
    """Return a list of (dataset_name, saved_at, path) for all saved runs."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []
    rows = []
    for p in sorted(results_dir.glob("*.json")):
        if p.is_dir():
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                meta = json.load(f)
            rows.append((meta.get("dataset", p.stem), meta.get("saved_at", "?"), p))
        except Exception as e:
            rows.append((p.stem, f"ERR: {e}", p))
    return rows


# =====================================================================
# Experiment tracking -- step-by-step optimization with lineage
# =====================================================================

EXPERIMENTS_DIR = RESULTS_DIR / "experiments"


def save_experiment(RESULTS: dict, step_name: str, description: str = "",
                    parent: str | None = None,
                    experiments_dir: Path | str = EXPERIMENTS_DIR) -> Path:
    """
    Save the current RESULTS as a TAGGED SNAPSHOT for an optimization step.

    Writes: results/experiments/<dataset>_<step_name>.json

    Never overwrites previous experiments (filename includes step_name). Use
    this at the end of each optimization iteration so you can reconstruct the
    progression later.

    Args:
        step_name:   short tag, e.g. "0_baseline", "1_ivf_tuned", "2_kmeans_numba"
        description: free text -- what you changed and why
        parent:      the step this was derived from (e.g. "0_baseline")
    """
    dataset = RESULTS.get("dataset")
    if not dataset:
        raise ValueError("RESULTS['dataset'] must be set before saving experiment.")

    experiments_dir = Path(experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    out_path = experiments_dir / f"{dataset}_{step_name}.json"

    payload = {
        "dataset":     dataset,
        "step":        step_name,
        "parent_step": parent,
        "description": description,
        "saved_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "results":     _json_safe(RESULTS),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved experiment: {out_path.name}  ({out_path.stat().st_size/1024:.1f} KB)")
    if parent:
        print(f"  parent: {parent}")
    if description:
        print(f"  description: {description}")
    return out_path


def load_experiment(dataset: str, step_name: str,
                    experiments_dir: Path | str = EXPERIMENTS_DIR) -> dict:
    """Load one experiment payload. Returns the full payload dict (not just results)."""
    path = Path(experiments_dir) / f"{dataset}_{step_name}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_experiments(dataset: str | None = None,
                     experiments_dir: Path | str = EXPERIMENTS_DIR) -> list:
    """List experiments sorted by step_name. Optionally filter by dataset."""
    experiments_dir = Path(experiments_dir)
    if not experiments_dir.exists():
        return []
    payloads = []
    for p in sorted(experiments_dir.glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        if dataset and meta.get("dataset") != dataset:
            continue
        payloads.append(meta)
    return payloads


def print_experiments(dataset: str | None = None) -> None:
    """Pretty-print the experiment lineage."""
    exps = list_experiments(dataset)
    if not exps:
        print(f"No experiments found for dataset={dataset}")
        return
    print(f"{'Step':<25s} {'Parent':<15s} {'Saved at':<20s} Description")
    print("-" * 100)
    for e in exps:
        step = e.get("step", "?")
        parent = e.get("parent_step") or "-"
        at = e.get("saved_at", "?")
        desc = e.get("description", "")[:60]
        print(f"{step:<25s} {parent:<15s} {at:<20s} {desc}")


def _extract_key_metrics(R: dict) -> dict:
    """Flatten RESULTS into a small dict of scalar metrics for cross-step diff."""
    m = {}
    sim = R.get("similarity", {}) or {}
    if sim.get("NumPy"):
        m["sim_numpy_ms"] = sim["NumPy"]["mean_ms"]
    if sim.get("Numba (parallel)"):
        m["sim_numba_par_ms"] = sim["Numba (parallel)"]["mean_ms"]

    b = (R.get("index", {}) or {}).get("build", {}) or {}
    if b.get("IVF (sequential)"):
        m["ivf_build_seq_ms"] = b["IVF (sequential)"]["ms"]
    if b.get("IVF (parallel)"):
        m["ivf_build_par_ms"] = b["IVF (parallel)"]["ms"]

    q = (R.get("index", {}) or {}).get("query", {}) or {}
    if q.get("BruteForce (NumPy)"):
        m["bf_query_ms"] = q["BruteForce (NumPy)"]["mean_ms"]
    ivf_q = {k: v for k, v in q.items() if k.startswith("IVF")}
    if ivf_q:
        best = min(ivf_q.items(), key=lambda kv: kv[1]["mean_ms"])
        m["ivf_best_ms"] = best[1]["mean_ms"]
        m["ivf_best_cfg"] = best[0]

    qu = R.get("quality", {}) or {}
    if qu.get("BruteForce"):
        m["recall_bf"] = qu["BruteForce"]["recall@k"]
    # Accept any n_probes value that exists in the current run
    for name in list(qu.keys()):
        if not name.startswith("IVF n_probes="):
            continue
        short = name.replace("IVF n_probes=", "recall_ivf_np")
        m[short] = qu[name]["recall@k"]

    em = R.get("embedding", {}) or {}
    if em.get("Sequential"):
        m["emb_seq_ms"] = em["Sequential"]["total_ms"]
    em_best = {k: v for k, v in em.items()
               if not k.startswith("_") and k != "Sequential"}
    if em_best:
        best = min(em_best.items(), key=lambda kv: kv[1]["total_ms"])
        m["emb_best_ms"] = best[1]["total_ms"]

    gen = R.get("generation", {}) or {}
    if gen.get("Sequential"):
        m["gen_seq_ms"] = gen["Sequential"]["total_ms"]
    gen_best = {k: v for k, v in gen.items()
                if not k.startswith("_") and k != "Sequential"}
    if gen_best:
        best = min(gen_best.items(), key=lambda kv: kv[1]["total_ms"])
        m["gen_best_ms"] = best[1]["total_ms"]

    # Step 4: streaming generation
    gs = R.get("generation_stream", {}) or {}
    if gs.get("Sequential (non-stream)"):
        m["stream_base_total_ms"] = gs["Sequential (non-stream)"]["batch_total_ms"]
        m["stream_base_ttft_ms"]  = gs["Sequential (non-stream)"]["mean_ttft_ms"]
    if gs.get("Sequential (streaming)"):
        m["stream_seq_total_ms"] = gs["Sequential (streaming)"]["batch_total_ms"]
        m["stream_seq_ttft_ms"]  = gs["Sequential (streaming)"]["mean_ttft_ms"]
    # Concurrent streaming key name varies by worker count -- match prefix
    for k, v in gs.items():
        if k.startswith("Concurrent streaming"):
            m["stream_conc_total_ms"] = v["batch_total_ms"]
            m["stream_conc_ttft_ms"]  = v["mean_ttft_ms"]

    # Step 5: pipelined RAG
    pp = R.get("pipeline", {}) or {}
    if pp.get("Sequential (naive)"):
        m["pipe_seq_total_ms"] = pp["Sequential (naive)"]["total_ms"]
    for k, v in pp.items():
        if k.startswith("Pipelined"):
            m["pipe_opt_total_ms"] = v["total_ms"]
            m["pipe_opt_per_q_ms"] = v["mean_ms_per_query"]
            m["pipe_speedup"]      = v["speedup_vs_sequential"]

    # Step 6: friend's query-path optimizations (ablation at n_probes=2, the fastest config)
    fo = R.get("friend_opts", {}) or {}
    fastest_probe = "np=2"
    for variant in ("A_flags_off", "B_norm_cache",
                    "C_norm_cache+np_gather", "D_numba_par_precomp"):
        if variant in fo and fastest_probe in fo[variant]:
            short = variant.split("_", 1)[0].lower()   # A/B/C/D
            m[f"friend_{short}_np2_ms"] = fo[variant][fastest_probe]["mean_ms"]
    return m


def compare_experiments(dataset: str, step_a: str, step_b: str) -> None:
    """Print a side-by-side diff of two experiments' key metrics."""
    a = load_experiment(dataset, step_a)
    b = load_experiment(dataset, step_b)
    ma = _extract_key_metrics(a["results"])
    mb = _extract_key_metrics(b["results"])

    print("=" * 84)
    print(f"  Experiment diff  ({dataset})   {step_a}  -->  {step_b}")
    if a.get("description"):
        print(f"    {step_a}: {a['description']}")
    if b.get("description"):
        print(f"    {step_b}: {b['description']}")
    print("=" * 84)
    print(f"{'Metric':<22s} {step_a:>18s} {step_b:>18s} {'Delta':>12s} {'%':>8s}")
    print("-" * 84)

    all_keys = sorted(set(ma.keys()) | set(mb.keys()))
    for k in all_keys:
        va, vb = ma.get(k), mb.get(k)
        if isinstance(va, str) or isinstance(vb, str):
            print(f"{k:<22s} {str(va):>18s} {str(vb):>18s}")
            continue
        if va is None or vb is None:
            print(f"{k:<22s} {str(va):>18s} {str(vb):>18s}")
            continue
        delta = vb - va
        pct = (delta / va * 100) if va else 0
        arrow = "v" if delta < 0 else "^" if delta > 0 else "="
        # For latency metrics, lower is better; for recall, higher is better.
        print(f"{k:<22s} {va:>18.4f} {vb:>18.4f} {delta:>+12.4f} {pct:>+6.1f}% {arrow}")
    print("=" * 84)


def plot_experiment_progression(dataset: str,
                                metrics: tuple = ("bf_query_ms", "ivf_best_ms",
                                                  "ivf_build_seq_ms",
                                                  "emb_best_ms", "gen_best_ms"),
                                 experiments_dir: Path | str = EXPERIMENTS_DIR) -> None:
    """
    Plot selected metrics across all experiment steps for a dataset.
    Each subplot shows one metric's evolution across steps (lower = better).
    """
    exps = list_experiments(dataset, experiments_dir)
    if not exps:
        print(f"No experiments for dataset={dataset}")
        return

    step_names = [e["step"] for e in exps]
    all_metrics = [_extract_key_metrics(e["results"]) for e in exps]

    n_cols = min(len(metrics), 3)
    n_rows = (len(metrics) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, metric in zip(axes, metrics):
        values = [m.get(metric) for m in all_metrics]
        valid = [(s, v) for s, v in zip(step_names, values) if v is not None]
        if not valid:
            ax.set_title(f"{metric} (no data)")
            ax.axis("off")
            continue
        xs, ys = zip(*valid)
        ax.plot(xs, ys, marker="o", markersize=8, linewidth=2, color="#2c7fb8")
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(metrics):]:
        ax.axis("off")

    fig.suptitle(f"Optimization progression -- {dataset}", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()
