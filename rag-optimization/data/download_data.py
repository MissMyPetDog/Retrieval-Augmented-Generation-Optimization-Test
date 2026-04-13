"""
Deterministic MS MARCO data downloader.

Guarantees every team member gets the EXACT same passages and queries
by using fixed ordering and deterministic selection.

Usage:
    python data/download_data.py                           # Default: 100K passages
    python data/download_data.py --size small              # 5K passages (dev/debug)
    python data/download_data.py --size medium             # 100K passages
    python data/download_data.py --size large              # 500K passages
    python data/download_data.py --verify                  # Verify data matches teammates'

Output:
    data/passages.jsonl     — one JSON per line: {"id": "passage_0", "text": "..."}
    data/queries.jsonl      — one JSON per line: {"id": "query_0", "text": "...", "relevant_passages": [...]}
    data/metadata.json      — records exact parameters + SHA256 hashes for verification
"""
import os
import sys
import json
import hashlib
import argparse
from datetime import datetime

# ── Size presets ──
PRESETS = {
    "small":  {"num_passages": 5_000,   "num_queries": 100},
    "medium": {"num_passages": 100_000, "num_queries": 500},
    "large":  {"num_passages": 500_000, "num_queries": 1_000},
}

DATASET_NAME = "microsoft/ms_marco"
DATASET_CONFIG = "v1.1"
DATASET_SPLIT = "train"


def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file for verification."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download(num_passages: int, num_queries: int, data_dir: str, hf_token: str = None):
    """
    Download MS MARCO with deterministic ordering.

    Key guarantee: the dataset is iterated in its original order (no shuffling),
    and we take the FIRST num_passages unique passages and FIRST num_queries
    queries with relevant passages. Since the dataset order is fixed on
    HuggingFace, every team member gets identical data.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    os.makedirs(data_dir, exist_ok=True)

    passage_path = os.path.join(data_dir, "passages.jsonl")
    query_path = os.path.join(data_dir, "queries.jsonl")
    metadata_path = os.path.join(data_dir, "metadata.json")

    # ── Stream dataset (no full download needed) ──
    print(f"Streaming {DATASET_NAME} ({DATASET_CONFIG}, split={DATASET_SPLIT})...")
    print(f"Target: {num_passages:,} passages, {num_queries:,} queries\n")

    dataset = load_dataset(
        DATASET_NAME, DATASET_CONFIG,
        split=DATASET_SPLIT,
        streaming=True,
    )

    passages = []
    queries = []
    seen_texts = set()
    passage_count = 0

    for item_idx, item in enumerate(dataset):
        # ── Extract unique passages (in order) ──
        for text in item["passages"]["passage_text"]:
            cleaned = text.strip()
            if cleaned and cleaned not in seen_texts and passage_count < num_passages:
                seen_texts.add(cleaned)
                passages.append({
                    "id": f"passage_{passage_count}",
                    "text": cleaned,
                })
                passage_count += 1

        # ── Extract queries with relevance labels ──
        if len(queries) < num_queries:
            relevant = []
            for t, selected in zip(
                item["passages"]["passage_text"],
                item["passages"]["is_selected"],
            ):
                if selected == 1 and t.strip() in seen_texts:
                    relevant.append(t.strip())
            if relevant:
                queries.append({
                    "id": f"query_{len(queries)}",
                    "text": item["query"].strip(),
                    "query_type": item.get("query_type", ""),
                    "relevant_passages": relevant,
                })

        # ── Progress ──
        if item_idx % 5000 == 0 and item_idx > 0:
            print(f"  Processed {item_idx:,} items → "
                  f"{passage_count:,} passages, {len(queries)} queries")

        # ── Done? ──
        if passage_count >= num_passages and len(queries) >= num_queries:
            break

    # ── Save passages ──
    with open(passage_path, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(passages):,} passages → {passage_path}")

    # ── Save queries ──
    with open(query_path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Saved {len(queries):,} queries  → {query_path}")

    # ── Save metadata for reproducibility ──
    passage_hash = compute_file_hash(passage_path)
    query_hash = compute_file_hash(query_path)

    metadata = {
        "dataset": DATASET_NAME,
        "config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "num_passages": len(passages),
        "num_queries": len(queries),
        "items_scanned": item_idx + 1,
        "passage_file_sha256": passage_hash,
        "query_file_sha256": query_hash,
        "created_at": datetime.now().isoformat(),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved → {metadata_path}")
    print(f"  Passage SHA256: {passage_hash[:16]}...")
    print(f"  Query   SHA256: {query_hash[:16]}...")
    print(f"\n✓ Share metadata.json with teammates to verify identical data.")


def verify(data_dir: str):
    """Verify local data matches expected hashes."""
    metadata_path = os.path.join(data_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print("No metadata.json found. Run download first.")
        return False

    with open(metadata_path) as f:
        meta = json.load(f)

    passage_path = os.path.join(data_dir, "passages.jsonl")
    query_path = os.path.join(data_dir, "queries.jsonl")

    print(f"Verifying against metadata ({meta['num_passages']:,} passages, {meta['num_queries']} queries)...\n")

    ok = True
    for filepath, key in [(passage_path, "passage_file_sha256"), (query_path, "query_file_sha256")]:
        if not os.path.exists(filepath):
            print(f"  ✗ Missing: {filepath}")
            ok = False
            continue
        actual = compute_file_hash(filepath)
        expected = meta[key]
        if actual == expected:
            print(f"  ✓ {os.path.basename(filepath)}: hash matches")
        else:
            print(f"  ✗ {os.path.basename(filepath)}: HASH MISMATCH")
            print(f"    expected: {expected[:16]}...")
            print(f"    actual:   {actual[:16]}...")
            ok = False

    if ok:
        print("\n✓ Data verified. Your data is identical to teammates'.")
    else:
        print("\n✗ Verification failed. Re-run download.")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MS MARCO data")
    parser.add_argument("--size", choices=["small", "medium", "large"], default="medium",
                        help="Dataset size preset")
    parser.add_argument("--num_passages", type=int, default=None,
                        help="Custom number of passages (overrides --size)")
    parser.add_argument("--num_queries", type=int, default=None,
                        help="Custom number of queries (overrides --size)")
    parser.add_argument("--data_dir", default=None,
                        help="Output directory (default: data/)")
    parser.add_argument("--hf_token", default=None,
                        help="HuggingFace token for faster downloads")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing data against metadata hashes")
    args = parser.parse_args()

    # Resolve data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or script_dir

    if args.verify:
        verify(data_dir)
    else:
        preset = PRESETS[args.size]
        num_passages = args.num_passages or preset["num_passages"]
        num_queries = args.num_queries or preset["num_queries"]
        download(num_passages, num_queries, data_dir, args.hf_token)
