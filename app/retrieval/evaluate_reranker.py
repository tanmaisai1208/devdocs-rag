
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

from reranker import CrossEncoderReranker


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "dataset"
    / "pilot_v2"
    / "candidate_labels_dev.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "dataset"
    / "pilot_v2"
    / "candidate_labels_dev_with_reranker.csv"
)

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.jsonl"
)

LATENCY_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "pilot_v2"
    / "reranker_latency_per_query.csv"
)

HYBRID_CANDIDATE_K = 10


def load_chunks():
    """Load chunk content, indexed by stable chunk ID."""
    chunks_by_id = {}

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            chunk = json.loads(line)
            chunks_by_id[chunk["chunk_id"]] = chunk

    if not chunks_by_id:
        raise ValueError("No processed chunks were loaded.")

    return chunks_by_id


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation CSV not found: {INPUT_PATH}"
        )

    with INPUT_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        original_columns = reader.fieldnames or []
        rows = list(reader)

    required = {
        "query_id",
        "query",
        "chunk_id",
        "hybrid_rank",
        "relevant",
    }

    missing = required - set(original_columns)

    if missing:
        raise ValueError(
            f"Input CSV is missing columns: {sorted(missing)}"
        )

    if not rows:
        raise ValueError("Input CSV contains no rows.")

    chunks_by_id = load_chunks()

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["query_id"]].append(row)

    # Add a separate rank column without overwriting existing ranks.
    for row in rows:
        row["reranker_rank"] = ""

    reranker = CrossEncoderReranker()
    latency_rows = []

    print(f"Loaded {len(rows)} labeled candidate rows.")
    print(f"Evaluation queries: {len(grouped)}")
    print("Cross-encoder loaded. Starting reranking...")

    for query_id, query_rows in sorted(grouped.items()):
        query_text = query_rows[0]["query"]

        if any(row["query"] != query_text for row in query_rows):
            raise ValueError(
                f"Inconsistent query text for {query_id}."
            )

        # Rerank only candidates in the existing hybrid top 10.
        hybrid_candidates = []

        for row in query_rows:
            raw_rank = row["hybrid_rank"].strip()

            if not raw_rank:
                continue

            hybrid_rank = int(raw_rank)

            if hybrid_rank > HYBRID_CANDIDATE_K:
                continue

            chunk_id = row["chunk_id"]

            if chunk_id not in chunks_by_id:
                raise ValueError(
                    f"Chunk {chunk_id} not found in chunks.jsonl."
                )

            chunk = chunks_by_id[chunk_id]
            metadata = chunk.get("metadata", {})

            hybrid_candidates.append(
                {
                    "chunk_id": chunk_id,
                    "content": chunk["content"],
                    "source": metadata.get("source", "Unknown"),
                    "section": metadata.get("section", "Unknown"),
                }
            )

        if not hybrid_candidates:
            raise ValueError(
                f"No hybrid top-{HYBRID_CANDIDATE_K} candidates "
                f"found for {query_id}."
            )

        # Model loading is excluded from this timing measurement.
        start_time = time.perf_counter()

        reranked = reranker.rerank(
            query=query_text,
            candidates=hybrid_candidates,
            top_k=HYBRID_CANDIDATE_K,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        reranker_ranks = {
            candidate["chunk_id"]: str(rank)
            for rank, candidate in enumerate(reranked, start=1)
        }

        for row in query_rows:
            row["reranker_rank"] = reranker_ranks.get(
                row["chunk_id"], ""
            )

        latency_rows.append(
            {
                "query_id": query_id,
                "candidate_count": len(hybrid_candidates),
                "reranking_latency_ms": round(elapsed_ms, 3),
            }
        )

        print(
            f"{query_id}: reranked {len(hybrid_candidates)} "
            f"candidates in {elapsed_ms:.2f} ms"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATENCY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    output_columns = [
        *original_columns,
        "reranker_rank",
    ]

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(rows)

    with LATENCY_OUTPUT.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "query_id",
                "candidate_count",
                "reranking_latency_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(latency_rows)

    print("\nReranking completed.")
    print(f"Updated evaluation data: {OUTPUT_PATH}")
    print(f"Reranking latency data: {LATENCY_OUTPUT}")


if __name__ == "__main__":
    main()
