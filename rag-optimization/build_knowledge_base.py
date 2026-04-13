"""
Build Vector Knowledge Base — complete offline pipeline.

This script takes raw passages and produces a searchable vector index.
Any team member can run this to build an identical knowledge base.

Pipeline:
    passages.jsonl → [Preprocess] → [Embed] → [Build Index] → vectors.npy + index files

Usage:
    python build_knowledge_base.py                              # Default: use existing data
    python build_knowledge_base.py --download --size medium     # Download + build
    python build_knowledge_base.py --download --size large      # 500K version
    python build_knowledge_base.py --device cpu                 # Force CPU for embedding
    python build_knowledge_base.py --device cuda                # Force GPU for embedding
    python build_knowledge_base.py --verify                     # Verify data + index integrity

Output (all saved to data/):
    vectors.npy         — embedding matrix, shape (N, 384), float32
    chunks.jsonl        — processed text chunks with IDs
    index_bruteforce.pkl — serialized BruteForce index
    index_ivf.pkl       — serialized IVF index
    build_report.json   — timing breakdown + checksums
"""
import os
import sys
import json
import time
import hashlib
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from components.preprocessor import load_passages, process_passages, clean_text, chunk_text_baseline
from components.vector_index import BruteForceIndex, IVFIndex


def compute_file_hash(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def step_download(args):
    """Step 0: Download data if requested."""
    print("\n" + "=" * 60)
    print("STEP 0: Download Data")
    print("=" * 60)

    from data.download_data import download, PRESETS
    preset = PRESETS[args.size]
    num_passages = args.num_passages or preset["num_passages"]
    num_queries = args.num_queries or preset["num_queries"]
    download(num_passages, num_queries, args.data_dir, args.hf_token)


def step_preprocess(data_dir: str) -> tuple[list[dict], float]:
    """Step 1: Load and preprocess passages into chunks."""
    print("\n" + "=" * 60)
    print("STEP 1: Preprocess")
    print("=" * 60)

    passage_path = os.path.join(data_dir, "passages.jsonl")
    if not os.path.exists(passage_path):
        print(f"ERROR: {passage_path} not found.")
        print("Run with --download first, or place passages.jsonl in data/")
        sys.exit(1)

    t0 = time.perf_counter()

    # Load passages
    passages = load_passages(passage_path)
    print(f"  Loaded {len(passages):,} passages")

    # Preprocess: clean + chunk
    chunks = process_passages(passages)
    elapsed = time.perf_counter() - t0

    print(f"  Produced {len(chunks):,} chunks")
    print(f"  Time: {elapsed:.2f}s")

    # Save chunks
    chunk_path = os.path.join(data_dir, "chunks.jsonl")
    with open(chunk_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"  Saved → {chunk_path}")

    return chunks, elapsed


def step_embed(chunks: list[dict], data_dir: str, device: str = "auto") -> tuple[np.ndarray, float]:
    """Step 2: Generate embeddings for all chunks."""
    print("\n" + "=" * 60)
    print("STEP 2: Generate Embeddings")
    print("=" * 60)

    from sentence_transformers import SentenceTransformer
    import torch

    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    print(f"  Model:  {config.EMBEDDING_MODEL}")

    model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
    texts = [c["text"] for c in chunks]

    print(f"  Encoding {len(texts):,} chunks...")
    t0 = time.perf_counter()
    vectors = model.encode(
        texts,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    vectors = np.array(vectors, dtype=np.float32)
    elapsed = time.perf_counter() - t0

    print(f"  Shape: {vectors.shape}")
    print(f"  Time: {elapsed:.2f}s ({len(texts) / elapsed:.0f} texts/sec)")

    # Save
    vector_path = os.path.join(data_dir, "vectors.npy")
    np.save(vector_path, vectors)
    print(f"  Saved → {vector_path} ({vectors.nbytes / 1e6:.0f} MB)")

    return vectors, elapsed


def step_build_index(
    vectors: np.ndarray,
    doc_ids: list[str],
    data_dir: str,
    n_clusters: int = None,
) -> dict:
    """Step 3: Build BruteForce and IVF indexes."""
    print("\n" + "=" * 60)
    print("STEP 3: Build Indexes")
    print("=" * 60)

    n = len(vectors)
    # Auto-select cluster count based on data size
    if n_clusters is None:
        if n <= 10_000:
            n_clusters = 32
        elif n <= 100_000:
            n_clusters = 64
        elif n <= 500_000:
            n_clusters = 128
        else:
            n_clusters = 256

    timings = {}

    # BruteForce
    print(f"\n  Building BruteForce index...")
    bf = BruteForceIndex()
    t0 = time.perf_counter()
    bf.build(vectors, doc_ids)
    timings["bruteforce_s"] = time.perf_counter() - t0

    bf_path = os.path.join(data_dir, "index_bruteforce.pkl")
    bf.save(bf_path)
    print(f"  Saved → {bf_path}")

    # IVF
    print(f"\n  Building IVF index ({n_clusters} clusters)...")
    ivf = IVFIndex(n_clusters=n_clusters, n_probes=8)
    t0 = time.perf_counter()
    ivf.build(vectors, doc_ids)
    timings["ivf_s"] = time.perf_counter() - t0

    ivf_path = os.path.join(data_dir, "index_ivf.pkl")
    ivf.save(ivf_path)
    print(f"  Saved → {ivf_path}")

    timings["n_clusters"] = n_clusters

    return timings


def step_verify(data_dir: str):
    """Verify all knowledge base files exist and are valid."""
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    required_files = {
        "passages.jsonl": "Raw passages",
        "queries.jsonl": "Evaluation queries",
        "chunks.jsonl": "Processed chunks",
        "vectors.npy": "Embedding vectors",
        "index_bruteforce.pkl": "BruteForce index",
        "index_ivf.pkl": "IVF index",
    }

    all_ok = True
    for filename, description in required_files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            if size > 1e9:
                size_str = f"{size / 1e9:.1f} GB"
            elif size > 1e6:
                size_str = f"{size / 1e6:.0f} MB"
            else:
                size_str = f"{size / 1e3:.0f} KB"
            print(f"  ✓ {filename:<25s} {size_str:>10s}  ({description})")
        else:
            print(f"  ✗ {filename:<25s} MISSING     ({description})")
            all_ok = False

    # Verify vector shape
    vector_path = os.path.join(data_dir, "vectors.npy")
    if os.path.exists(vector_path):
        v = np.load(vector_path)
        chunk_path = os.path.join(data_dir, "chunks.jsonl")
        if os.path.exists(chunk_path):
            with open(chunk_path,encoding="utf-8") as f:
                n_chunks = sum(1 for _ in f)
            if v.shape[0] == n_chunks:
                print(f"\n  ✓ Vector count ({v.shape[0]:,}) matches chunk count")
            else:
                print(f"\n  ✗ Vector count ({v.shape[0]:,}) != chunk count ({n_chunks:,})")
                all_ok = False
        print(f"  ✓ Vector shape: {v.shape}, dtype: {v.dtype}")

    # Verify data hashes if metadata exists
    metadata_path = os.path.join(data_dir, "metadata.json")
    if os.path.exists(metadata_path):
        from data.download_data import verify
        print()
        verify(data_dir)

    if all_ok:
        print("\n✓ Knowledge base is complete and ready to use.")
    else:
        print("\n✗ Some files are missing. Run build_knowledge_base.py to fix.")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Build Vector Knowledge Base")
    parser.add_argument("--download", action="store_true",
                        help="Download data before building")
    parser.add_argument("--size", choices=["small", "medium", "large"], default="medium",
                        help="Dataset size preset (only used with --download)")
    parser.add_argument("--num_passages", type=int, default=None)
    parser.add_argument("--num_queries", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Device for embedding generation")
    parser.add_argument("--n_clusters", type=int, default=None,
                        help="Number of IVF clusters (auto-selected if not set)")
    parser.add_argument("--data_dir", default=None,
                        help="Data directory (default: data/)")
    parser.add_argument("--hf_token", default=None,
                        help="HuggingFace token")
    parser.add_argument("--verify", action="store_true",
                        help="Only verify existing knowledge base")
    args = parser.parse_args()

    # Resolve paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.join(project_root, "data")
    args.data_dir = data_dir

    print("=" * 60)
    print("RAG Knowledge Base Builder")
    print("=" * 60)
    print(f"Data directory: {data_dir}")

    # Verify only mode
    if args.verify:
        step_verify(data_dir)
        return

    total_t0 = time.perf_counter()
    timings = {}

    # Step 0: Download (optional)
    if args.download:
        step_download(args)

    # Step 1: Preprocess
    chunks, t_preprocess = step_preprocess(data_dir)
    timings["preprocess_s"] = t_preprocess

    # Step 2: Embed
    vectors, t_embed = step_embed(chunks, data_dir, device=args.device)
    timings["embed_s"] = t_embed

    # Step 3: Build index
    doc_ids = [c["id"] for c in chunks]
    index_timings = step_build_index(vectors, doc_ids, data_dir, args.n_clusters)
    timings.update(index_timings)

    # Total time
    total_time = time.perf_counter() - total_t0
    timings["total_s"] = total_time

    # Save build report
    report = {
        "num_passages": len(chunks),
        "embedding_dim": vectors.shape[1],
        "device": args.device,
        "model": config.EMBEDDING_MODEL,
        "timings": timings,
        "files": {},
    }

    # Add file hashes to report
    for filename in ["passages.jsonl", "queries.jsonl", "chunks.jsonl", "vectors.npy"]:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            report["files"][filename] = {
                "sha256": compute_file_hash(filepath),
                "size_bytes": os.path.getsize(filepath),
            }

    report_path = os.path.join(data_dir, "build_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"  Passages:    {len(chunks):,}")
    print(f"  Vectors:     {vectors.shape}")
    print(f"  Preprocess:  {timings['preprocess_s']:.2f}s")
    print(f"  Embedding:   {timings['embed_s']:.2f}s")
    print(f"  BF Index:    {timings['bruteforce_s']:.2f}s")
    print(f"  IVF Index:   {timings['ivf_s']:.2f}s")
    print(f"  Total:       {total_time:.2f}s")
    print(f"\n  Report → {report_path}")
    print(f"\n✓ Knowledge base ready. Run notebooks or main.py to query.")

    # Verify
    step_verify(data_dir)


if __name__ == "__main__":
    main()
