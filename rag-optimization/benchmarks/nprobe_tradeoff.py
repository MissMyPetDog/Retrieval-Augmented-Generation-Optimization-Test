"""
Plot IVF n_probes recall-latency trade-off.

This experiment fixes the IVF index and varies only n_probes. It measures
search latency, Recall@10, and MRR for each setting, then saves:
  - nprobe_tradeoff.json
  - nprobe_tradeoff.png

Example:
    python rag-optimization/benchmarks/nprobe_tradeoff.py \
        --data_dir rag-optimization/data/medium \
        --probes 1 2 4 8 16 32 64
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


RAG_DIR = Path(__file__).resolve().parents[1]
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))


def force_cpu() -> None:
    """Keep this benchmark CPU-only for reproducible software comparisons."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NUMBA_DISABLE_CUDA"] = "1"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_chunk_to_passage_text(chunks: list[dict], passages: list[dict]) -> dict[str, str]:
    """Map retrieved chunk IDs back to original passage text for MS MARCO labels."""
    passage_by_id = {p["id"]: p["text"] for p in passages}
    return {
        c["id"]: passage_by_id.get(c.get("source_id"), c.get("text", ""))
        for c in chunks
    }


def recall_and_mrr(retrieved_ids: list[str], relevant_passages: list[str],
                   chunk_to_passage: dict[str, str], k: int) -> tuple[float, float]:
    relevant = set(relevant_passages)
    if not relevant:
        return 0.0, 0.0

    retrieved_texts = [chunk_to_passage.get(doc_id, "") for doc_id in retrieved_ids[:k]]
    recall = len(set(retrieved_texts) & relevant) / len(relevant)

    mrr = 0.0
    for rank, text in enumerate(retrieved_texts, start=1):
        if text in relevant:
            mrr = 1.0 / rank
            break

    return recall, mrr


def get_or_build_query_vectors(data_dir: Path, queries: list[dict], device: str) -> np.ndarray:
    """Embed all evaluation queries once so n_probes timing isolates search only."""
    cache_path = data_dir / "query_vectors.npy"
    if cache_path.exists():
        query_vectors = np.load(cache_path)
        if len(query_vectors) == len(queries):
            print(f"Loaded cached query vectors: {cache_path}")
            return query_vectors.astype(np.float32)

    from components.embedder import LocalEmbedder

    print("Embedding queries once for the sweep...")
    embedder = LocalEmbedder(device=device)
    query_vectors = embedder.embed_texts(
        [q["text"] for q in queries],
        show_progress=True,
    ).astype(np.float32)
    np.save(cache_path, query_vectors)
    print(f"Saved query vectors: {cache_path}")
    return query_vectors


def run_sweep(
    data_dir: Path,
    probes: list[int],
    k: int = 10,
    device: str = "cpu",
    use_norm_cache: bool = True,
    use_np_gather: bool = True,
    repeats: int = 3,
) -> dict:
    from components.similarity import cosine_sim_numpy
    from components.vector_index import IVFIndex

    passages = load_jsonl(data_dir / "passages.jsonl")
    chunks = load_jsonl(data_dir / "chunks.jsonl")
    queries = load_jsonl(data_dir / "queries.jsonl")
    queries = [q for q in queries if q.get("relevant_passages")]
    query_vectors = get_or_build_query_vectors(data_dir, queries, device=device)
    chunk_to_passage = build_chunk_to_passage_text(chunks, passages)

    index_path = data_dir / "index_ivf.pkl"
    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} not found. Build it first with build_knowledge_base.py."
        )

    ivf = IVFIndex()
    ivf.load(str(index_path))
    ivf.use_precomputed_norms = use_norm_cache
    ivf.use_numpy_candidate_gather = use_np_gather

    print(
        f"Loaded IVF index: clusters={ivf.n_clusters}, "
        f"queries={len(queries)}, k={k}, repeats={repeats}"
    )
    print(
        f"Flags: norm_cache={use_norm_cache}, np_gather={use_np_gather}"
    )

    rows = []
    for n_probes in probes:
        recalls = []
        mrrs = []
        latencies = []

        # Warmup.
        for qv in query_vectors[: min(10, len(query_vectors))]:
            ivf.search(
                qv,
                k=k,
                n_probes=n_probes,
                sim_fn=cosine_sim_numpy,
                use_precomputed_norms=use_norm_cache,
                use_numpy_candidate_gather=use_np_gather,
            )

        for _ in range(repeats):
            for q, qv in zip(queries, query_vectors):
                t0 = time.perf_counter()
                results = ivf.search(
                    qv,
                    k=k,
                    n_probes=n_probes,
                    sim_fn=cosine_sim_numpy,
                    use_precomputed_norms=use_norm_cache,
                    use_numpy_candidate_gather=use_np_gather,
                )
                search_ms = (time.perf_counter() - t0) * 1000
                latencies.append(search_ms)

                retrieved_ids = [doc_id for doc_id, _ in results]
                recall, mrr = recall_and_mrr(
                    retrieved_ids,
                    q["relevant_passages"],
                    chunk_to_passage,
                    k,
                )
                recalls.append(recall)
                mrrs.append(mrr)

        row = {
            "n_probes": int(n_probes),
            "recall_at_10": float(np.mean(recalls)),
            "mrr": float(np.mean(mrrs)),
            "mean_search_ms": float(np.mean(latencies)),
            "p95_search_ms": float(np.percentile(latencies, 95)),
            "num_queries": len(queries),
        }
        rows.append(row)
        print(
            f"n_probes={n_probes:<3d} "
            f"Recall@10={row['recall_at_10']:.4f} "
            f"MRR={row['mrr']:.4f} "
            f"mean={row['mean_search_ms']:.3f} ms "
            f"p95={row['p95_search_ms']:.3f} ms"
        )

    return {
        "dataset": str(data_dir),
        "index": "IVFIndex",
        "n_clusters": int(ivf.n_clusters),
        "k": int(k),
        "use_norm_cache": bool(use_norm_cache),
        "use_np_gather": bool(use_np_gather),
        "repeats": int(repeats),
        "results": rows,
    }


def save_plot(payload: dict, out_path: Path, highlight_probe: int = 8) -> None:
    import matplotlib.pyplot as plt

    rows = payload["results"]
    xs = [r["mean_search_ms"] for r in rows]
    ys = [r["recall_at_10"] for r in rows]
    probes = [r["n_probes"] for r in rows]

    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, ys, marker="o", linewidth=2)

    label_offsets = {
        1: (8, 4),
        2: (8, 4),
        4: (8, 0),
        8: (0, 8),
        16: (8, 8),
        32: (8, 8),
        64: (0, 8),
    }

    for x, y, p in zip(xs, ys, probes):
        label = f"n={p}"
        if p == highlight_probe:
            plt.scatter([x], [y], s=120, color="crimson", zorder=3)
            # label += " (chosen)"
        offset = label_offsets.get(p, (8, 6))
        ha = "right" if offset[0] < 0 else "left"
        plt.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=offset,
            ha=ha,
            va="bottom",
        )

    plt.title("IVF n_probes Recall-Latency Trade-off")
    plt.xlabel("Mean search latency per query (ms)")
    plt.ylabel("Recall@10")
    plt.xlim(min(xs) - 0.05 * (max(xs) - min(xs)), max(xs) + 0.08 * (max(xs) - min(xs)))
    plt.ylim(min(ys) - 0.05 * (max(ys) - min(ys)), max(ys) + 0.10 * (max(ys) - min(ys)))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep IVF n_probes and plot recall vs runtime.")
    parser.add_argument("--data_dir", type=Path, default=RAG_DIR / "data" / "medium")
    parser.add_argument("--probes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "auto"], default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no_norm_cache", action="store_true")
    parser.add_argument("--no_np_gather", action="store_true")
    parser.add_argument("--highlight_probe", type=int, default=8)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    force_cpu()

    out_dir = args.out_dir or args.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = run_sweep(
        data_dir=args.data_dir,
        probes=args.probes,
        k=args.k,
        device=args.device,
        use_norm_cache=not args.no_norm_cache,
        use_np_gather=not args.no_np_gather,
        repeats=args.repeats,
    )

    json_path = out_dir / "nprobe_tradeoff.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results: {json_path}")

    save_plot(payload, out_dir / "nprobe_tradeoff.png", highlight_probe=args.highlight_probe)


if __name__ == "__main__":
    main()
