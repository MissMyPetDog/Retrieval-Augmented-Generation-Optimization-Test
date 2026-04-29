"""
Run the end-to-end COMBO GRID and print one table:
  3 retrieval combos x N LLM modes, each timed full-pipeline (embed + search + gen).

Each row of the output names every component used (Index, Sim fn, norm_cache,
np_gather, LLM mode), then reports Recall, Retrieve/q, Gen/call, and E2E batch time.

Default: 3 retrieval combos x 2 LLM modes x 8 queries = 48 real ChatGPT calls (~$0.20).

Usage:
    python run_combo_grid.py
    python run_combo_grid.py --modes async threaded sequential   # 9 rows / ~$0.30
    python run_combo_grid.py --n_items 4 --modes async           # cheap smoke test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import portal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",         default="medium")
    ap.add_argument("--n_items",         type=int, default=8)
    ap.add_argument("--k",               type=int, default=3)
    ap.add_argument("--n_clusters",      type=int, default=64)
    ap.add_argument("--n_probes",        type=int, default=8)
    ap.add_argument("--n_async_workers", type=int, default=8)
    ap.add_argument("--max_tokens",      type=int, default=128)
    ap.add_argument("--modes",           nargs="+", default=["async", "threaded"],
                    choices=["sequential", "async", "threaded"])
    ap.add_argument("--save",            default="results/combo_grid.json")
    args = ap.parse_args()

    portal.setup_cpu_only()
    vectors, chunks, queries, data_dir, _ = portal.load_knowledge_base(args.dataset)

    # Build BF + IVFIndexNumbaPP with K-Means++ (same index used by Step 6 / Config 3).
    bf, ivf_seq, _ivf_par, _build = portal.run_index_build_benchmarks(
        vectors, chunks, n_clusters=args.n_clusters, kmeans_impl="numba_pp",
    )

    t0 = time.perf_counter()
    grid = portal.run_endtoend_combos_grid(
        queries, chunks, bf=bf, ivf=ivf_seq, data_dir=data_dir,
        n_items=args.n_items, k=args.k, n_probes=args.n_probes,
        n_async_workers=args.n_async_workers, max_tokens=args.max_tokens,
        llm_modes=tuple(args.modes),
    )
    wall_ms = (time.perf_counter() - t0) * 1000

    out_path = Path(args.save)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset":    args.dataset,
        "n_items":    args.n_items,
        "k":          args.k,
        "n_clusters": args.n_clusters,
        "n_probes":   args.n_probes,
        "modes":      list(args.modes),
        "wall_ms":    wall_ms,
        "grid":       grid,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\nSaved -> {out_path}  (wall {wall_ms/1000:.1f}s)")


if __name__ == "__main__":
    main()
