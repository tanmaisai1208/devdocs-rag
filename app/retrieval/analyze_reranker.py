
import csv
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "dataset"
    / "candidate_labels_with_reranker.csv"
)

TOP_K = 5


def main():
    grouped = defaultdict(list)

    with INPUT_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        for row in csv.DictReader(file):
            grouped[row["query_id"]].append(row)

    print("\n--- HYBRID vs RERANKER: PER-QUERY DIAGNOSTICS ---")

    for query_id, rows in sorted(grouped.items()):
        query = rows[0]["query"]

        relevant = [
            row for row in rows
            if row["relevant"].strip() == "1"
        ]


        hybrid_top5 = sum(
            1
            for row in rows
            if row["relevant"].strip() == "1"
            and row["hybrid_rank"].strip() != ""
            and int(row["hybrid_rank"]) <= TOP_K
        )

        reranker_top5 = sum(
            1
            for row in rows
            if row["relevant"].strip() == "1"
            and row["reranker_rank"].strip() != ""
            and int(row["reranker_rank"]) <= TOP_K
        )


        hybrid_first = [
            int(row["hybrid_rank"])
            for row in relevant
            if row["hybrid_rank"].strip()
        ]

        reranker_first = [
            int(row["reranker_rank"])
            for row in relevant
            if row["reranker_rank"].strip()
        ]

        hybrid_mrr = (
            1 / min(hybrid_first) if hybrid_first else 0.0
        )
        reranker_mrr = (
            1 / min(reranker_first) if reranker_first else 0.0
        )

        print("\n" + "-" * 75)
        print(f"Query: {query_id}")
        print(f"Question: {query}")
        print(f"Relevant passages in labeled pool: {len(relevant)}")
        print(f"Hybrid relevant in top 5: {hybrid_top5}")
        print(f"Reranker relevant in top 5: {reranker_top5}")
        print(f"Hybrid first-relevant reciprocal rank: {hybrid_mrr:.3f}")
        print(f"Reranker first-relevant reciprocal rank: {reranker_mrr:.3f}")

        print("\nRelevant passage rank changes:")

        for row in relevant:
            hybrid_rank = row["hybrid_rank"].strip() or "Not ranked"
            rerank_rank = row["reranker_rank"].strip() or "Not ranked"

            print(
                f"  Chunk {row['chunk_id']}: "
                f"hybrid={hybrid_rank}, reranker={rerank_rank}"
            )


if __name__ == "__main__":
    main()
