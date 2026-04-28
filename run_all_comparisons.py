"""
Run all 3 RAG pipeline configurations end-to-end and print a unified comparison.

  Config 1: BruteForce baseline    (zero optimizations)
  Config 2: IVF default            (basic structural optimization)
  Config 3: Fully optimized        (all 6 steps stacked, pipelined streaming)

Real Kong gpt-4o calls: ~24 (8 queries x 3 configs), ~$0.10.
Requires KONG_API_KEY env var.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "comparisons"))

from run_bruteforce   import run as run_bf
from run_intermediate import run as run_inter
from run_optimized    import run as run_opt
from common           import save_result


if __name__ == "__main__":
    print("\n" + "#" * 72 + "\n# CONFIG 1: BruteForce baseline\n" + "#" * 72)
    r1 = run_bf();    save_result(r1, "01_bruteforce")

    print("\n" + "#" * 72 + "\n# CONFIG 2: IVF default\n" + "#" * 72)
    r2 = run_inter(); save_result(r2, "02_intermediate")

    print("\n" + "#" * 72 + "\n# CONFIG 3: Fully optimized\n" + "#" * 72)
    r3 = run_opt();   save_result(r3, "03_optimized")

    print("\n" + "=" * 80)
    print("  3-WAY PIPELINE COMPARISON (8 queries, real gpt-4o)")
    print("=" * 80)
    print(f"  {'Config':<48s} {'Recall':>7s} {'Per-q':>10s} {'Batch':>10s}")
    print(f"  {'-' * 76}")
    for r in (r1, r2, r3):
        m = r["per_query_mean"]
        print(f"  {r['name'][:47]:<48s} {m['recall@k']:>7.4f} "
              f"{m['total_ms']:>7.0f} ms {r['batch_total_ms']/1000:>7.2f} s")
    print(f"\n  Pipeline speedup (Config 3 vs Config 1): "
          f"{r1['batch_total_ms'] / r3['batch_total_ms']:.2f}x")
    print("=" * 80)
