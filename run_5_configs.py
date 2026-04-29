"""
Run 5 specific RAG pipeline configurations and report per-component timings + E2E.

  1. BruteForce + numpy           (no norm cache, no batch embed) + sequential
  2. IVF(64,8) + numpy            (norm cache, batch embed)        + async
  3. IVF(64,8) + numpy            (norm cache, batch embed)        + threaded
  4. IVF(64,8) + numba parallel   (norm cache, batch embed)        + async
  5. IVF(64,8) + numba parallel   (norm cache, batch embed)        + threaded

No recall / MRR. Only timings.

Reuses:
  - comparisons/run_bruteforce.py (config 1, exact match)
  - portal.run_endtoend_combos_grid (configs 2-5, B & C combos x 2 LLM modes)

Real ChatGPT gpt-4o calls: 8 (config 1) + 6*8 (full grid, 2 BF cells discarded) = 56 calls (~$0.22).
"""
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "comparisons"))

import portal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n_queries", type=int, default=8,
                    help="number of queries per config (default: 8). "
                         "Total real ChatGPT calls = n_queries * 7 (1 sequential BF + "
                         "3 retrieval combos x 2 LLM modes).")
    ap.add_argument("--dataset",         default="medium")
    ap.add_argument("--k",               type=int, default=3)
    ap.add_argument("--max_tokens",      type=int, default=128)
    ap.add_argument("-w", "--n_async_workers", type=int, default=8,
                    help="LLM concurrency for configs 2-5 (async max_concurrent / "
                         "threaded n_workers). Default 8. Try 16/32/64 to probe ChatGPT's "
                         "concurrency cap.")
    ap.add_argument("--no_batch_embed", action="store_true",
                    help="Configs 2-5 use per-query embed instead of batch embed. "
                         "Embed column is hidden from the printed table since it "
                         "becomes a constant (~7-8 ms/q) across all rows.")
    args = ap.parse_args()

    n             = args.n_queries
    embed_mode    = "per_query" if args.no_batch_embed else "batch"
    show_embed    = not args.no_batch_embed   # hide embed column under --no_batch_embed
    show_batch    = not args.no_batch_embed   # batch column is constant (OFF) -- also hide
    cost_calls    = n * 7
    print(f"\nConfig: n_queries={n}, k={args.k}, max_tokens={args.max_tokens}, "
          f"dataset={args.dataset}, n_async_workers={args.n_async_workers}, "
          f"embed_mode={embed_mode}")
    print(f"Real ChatGPT calls planned: {cost_calls}  (~${cost_calls * 0.004:.2f} at ~$0.004/call)")

    portal.setup_cpu_only()
    vectors, chunks, queries, data_dir, _ = portal.load_knowledge_base(args.dataset)

    # Build BF + IVFIndexNumbaPP(64). IVF will be reused for configs 2-5.
    bf, ivf, _ivf_par, _build = portal.run_index_build_benchmarks(
        vectors, chunks, n_clusters=64, kmeans_impl="numba_pp",
    )

    # ---- Config 1: BF + numpy, no cache, no batch, sequential ----
    from run_bruteforce import run as run_bf
    print(f"\n{'#'*72}")
    print(f"#  CONFIG 1: BruteForce + numpy (no cache, no batch) + sequential")
    print(f"{'#'*72}")
    bf_result = run_bf(dataset=args.dataset, n_queries=n, k=args.k, max_tokens=args.max_tokens)

    # ---- Configs 2-5: IVF(B,C) x (async, threaded) ----
    print(f"\n{'#'*72}")
    print(f"#  CONFIGS 2-5: IVF(64,8) + (numpy | numba parallel) x (async | threaded)")
    print(f"{'#'*72}")
    grid = portal.run_endtoend_combos_grid(
        queries, chunks, bf=bf, ivf=ivf, data_dir=data_dir,
        n_items=n, k=args.k, n_probes=8, n_async_workers=args.n_async_workers,
        max_tokens=args.max_tokens,
        llm_modes=("async", "threaded"),
        embed_mode=embed_mode,
        verbose=False,
    )

    # ---- Assemble 5-row table ----
    rows: list[dict[str, Any]] = []

    m1 = bf_result["per_query_mean"]
    rows.append({
        "n":         1,
        "index":     "BruteForce",
        "sim_fn":    "cosine_sim_numpy",
        "cache":     "OFF",
        "batch":     "OFF",
        "llm_mode":  "sequential",
        "embed_ms":  m1["embed_ms"],
        "search_ms": m1["search_ms"],
        "gen_ms":    m1["gen_ms"],
        "e2e_s":     bf_result["batch_total_ms"] / 1000,
    })

    label_b = "B. IVF + NumPy + cache + gather"
    label_c = "C. IVF + Numba par + cache + gather"
    plan = [
        (2, "async",    label_b, "cosine_sim_numpy"),
        (3, "threaded", label_b, "cosine_sim_numpy"),
        (4, "async",    label_c, "cosine_sim_numba_parallel_precomputed"),
        (5, "threaded", label_c, "cosine_sim_numba_parallel_precomputed"),
    ]
    for row_n, mode, label, sim_fn in plan:
        r = grid[mode][label]
        rows.append({
            "n":         row_n,
            "index":     "IVF(64,8)",
            "sim_fn":    sim_fn,
            "cache":     "ON",
            "batch":     "ON" if not args.no_batch_embed else "OFF",
            "llm_mode":  mode,
            "embed_ms":  r["embed_per_query_ms"],
            "search_ms": r["retrieve_per_query_ms"],
            "gen_ms":    r["gen_per_call_ms"],
            "e2e_s":     r["end_to_end_ms"] / 1000,
        })

    # ---- Compute speedup vs Config 1 (baseline) ----
    base = rows[0]
    for r in rows:
        r["embed_speedup"]  = base["embed_ms"]  / r["embed_ms"]  if r["embed_ms"]  else 0.0
        r["search_speedup"] = base["search_ms"] / r["search_ms"] if r["search_ms"] else 0.0
        r["gen_speedup"]    = base["gen_ms"]    / r["gen_ms"]    if r["gen_ms"]    else 0.0
        r["e2e_speedup"]    = base["e2e_s"]     / r["e2e_s"]     if r["e2e_s"]     else 0.0

    def cell(value: float, speedup: float, fmt: str, unit: str) -> str:
        return f"{value:{fmt}} {unit} ({speedup:5.2f}x)"

    # ---- Print unified 5-row table ----
    width = 168 if show_embed else 142
    print(f"\n\n{'='*width}")
    print(f"  5-CONFIG END-TO-END COMPARISON ({n} queries, real Chat-GPT-4o; "
          f"embed_mode={embed_mode}; speedup is vs Config 1 baseline)")
    print(f"{'='*width}")
    header_left = (f"  {'#':>2s}  {'Index':<11s} {'Sim fn':<37s} "
                   f"{'cache':>5s} ")
    if show_batch:
        header_left += f"{'batch':>5s} "
    header_left += f" {'LLM mode':<11s} "
    header_cols = ""
    if show_embed:
        header_cols += f"{'Embed/q':>18s}  "
    header_cols += f"{'Search/q':>18s}  {'Gen/call':>19s}  {'E2E':>18s}"
    print(header_left + header_cols)
    print(f"  {'-'*(width-4)}")
    for r in rows:
        line = (f"  {r['n']:>2d}  {r['index']:<11s} {r['sim_fn']:<37s} "
                f"{r['cache']:>5s} ")
        if show_batch:
            line += f"{r['batch']:>5s} "
        line += f" {r['llm_mode']:<11s} "
        if show_embed:
            line += f"{cell(r['embed_ms'],  r['embed_speedup'],  '7.1f', 'ms'):>18s}  "
        line += (f"{cell(r['search_ms'], r['search_speedup'], '7.2f', 'ms'):>18s}  "
                 f"{cell(r['gen_ms'],    r['gen_speedup'],    '8.0f', 'ms'):>19s}  "
                 f"{cell(r['e2e_s'],     r['e2e_speedup'],    '7.2f', 's'):>18s}")
        print(line)
    print(f"{'='*width}")

    # ---- Save JSON ----
    out_path = PROJECT_ROOT / "results" / "5_configs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_queries":       n,
        "k":               args.k,
        "max_tokens":      args.max_tokens,
        "dataset":         args.dataset,
        "n_async_workers": args.n_async_workers,
        "embed_mode":      embed_mode,
        "n_probes":        8,
        "n_clusters":      64,
        "rows":            rows,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
