
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LABELS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "dataset"
    / "candidate_labels_with_reranker.csv"
)

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.jsonl"
)

# Start with the two cases where reranking showed problems.
QUERIES_TO_INSPECT = {"q09", "q12"}
MAX_CHARS = 900


def main():
    chunks_by_id = {}

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            chunk = json.loads(line)
            chunks_by_id[chunk["chunk_id"]] = chunk

    with LABELS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    for query_id in sorted(QUERIES_TO_INSPECT):
        query_rows = [
            row for row in rows
            if row["query_id"] == query_id
        ]

        if not query_rows:
            print(f"\nNo rows found for {query_id}")
            continue

        print("\n" + "=" * 90)
        print(f"QUERY: {query_id}")
        print(f"Question: {query_rows[0]['query']}")

        candidates = [
            row for row in query_rows
            if row["hybrid_rank"].strip()
            and int(row["hybrid_rank"]) <= 10
        ]

        candidates.sort(
            key=lambda row: int(row["hybrid_rank"])
        )

        for row in candidates:
            chunk_id = row["chunk_id"]
            chunk = chunks_by_id.get(chunk_id)

            print("\n" + "-" * 90)
            print(f"Chunk ID: {chunk_id}")
            print(f"Relevant label: {row['relevant']}")
            print(f"Hybrid rank: {row['hybrid_rank']}")
            print(
                "Reranker rank: "
                f"{row['reranker_rank'].strip() or 'Not ranked'}"
            )

            if chunk is None:
                print("Chunk content not found.")
                continue

            metadata = chunk.get("metadata", {})
            print(f"Source: {metadata.get('source', 'Unknown')}")
            print(f"Section: {metadata.get('section', 'Unknown')}")
            print("\nPassage:")
            print(chunk["content"][:MAX_CHARS])

    print("\nInspection complete.")


if __name__ == "__main__":
    main()
