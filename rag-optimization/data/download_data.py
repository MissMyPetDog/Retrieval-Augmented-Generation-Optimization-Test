"""
Download and prepare a subset of MS MARCO passages for the RAG project.

Usage:
    python data/download_data.py [--num_passages 100000]
"""
import json
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def download_msmarco(num_passages: int = config.NUM_PASSAGES):
    """Download MS MARCO passages and queries from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets: pip install datasets")
        sys.exit(1)

    os.makedirs(config.DATA_DIR, exist_ok=True)

    # ── Download passages ──
    print(f"Downloading MS MARCO passages (first {num_passages})...")
    dataset = load_dataset(
        "microsoft/ms_marco", config.DATASET_SUBSET,
        split="train",
        trust_remote_code=True,
    )

    passages = []
    seen = set()
    count = 0

    for item in dataset:
        for passage_text, is_selected in zip(
            item["passages"]["passage_text"],
            item["passages"]["is_selected"],
        ):
            if passage_text not in seen:
                seen.add(passage_text)
                passages.append({
                    "id": f"passage_{count}",
                    "text": passage_text.strip(),
                })
                count += 1
                if count >= num_passages:
                    break
        if count >= num_passages:
            break

    with open(config.PASSAGE_FILE, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(passages)} passages -> {config.PASSAGE_FILE}")

    # ── Download queries with relevance labels ──
    print("Extracting queries with relevance judgments...")
    queries = []
    for i, item in enumerate(dataset):
        if i >= config.BENCHMARK_NUM_QUERIES:
            break

        relevant_passages = []
        for j, (text, selected) in enumerate(zip(
            item["passages"]["passage_text"],
            item["passages"]["is_selected"],
        )):
            if selected == 1 and text.strip() in seen:
                relevant_passages.append(text.strip())

        if relevant_passages:
            queries.append({
                "id": f"query_{i}",
                "text": item["query"].strip(),
                "query_type": item.get("query_type", ""),
                "relevant_passages": relevant_passages,
            })

    with open(config.QUERY_FILE, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Saved {len(queries)} queries -> {config.QUERY_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_passages", type=int, default=config.NUM_PASSAGES)
    args = parser.parse_args()
    download_msmarco(args.num_passages)
