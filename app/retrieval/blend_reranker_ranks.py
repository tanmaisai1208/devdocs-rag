
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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "dataset"
    / "candidate_labels_with_blended_reranker.csv"
)

HYBRID_WEIGHT = 0.75
RERANKER_WEIGHT = 0.25


def main():
    with INPUT_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not rows:
        raise ValueError(f"No candidate rows found in {INPUT_PATH}")

    required_columns = {
        "query_id",
        "hybrid_rank",
        "reranker_rank",
    }
    missing = required_columns - set(fieldnames)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if "blended_rank" not in fieldnames:
        fieldnames.append("blended_rank")

    grouped_rows = defaultdict(list)

    for row in rows:
        row["blended_rank"] = ""

        hybrid_rank = row["hybrid_rank"].strip()
        reranker_rank = row["reranker_rank"].strip()

        # Only blend candidates that were ranked by both methods.
        if not hybrid_rank or not reranker_rank:
            continue

        h_rank = int(hybrid_rank)
        c_rank = int(reranker_rank)

        if h_rank <= 0 or c_rank <= 0:
            raise ValueError(
                f"Invalid rank for query {row['query_id']}: "
                f"hybrid={h_rank}, reranker={c_rank}"
            )

        score = (
            HYBRID_WEIGHT / h_rank
            + RERANKER_WEIGHT / c_rank
        )

        grouped_rows[row["query_id"]].append(
            (score, h_rank, c_rank, row)
        )

    for query_id, candidates in grouped_rows.items():
        # Higher blended score is better.
        # Hybrid rank is the first tie-breaker.
        candidates.sort(
            key=lambda item: (
                -item[0],
                item[1],
                item[2],
                item[3].get("chunk_id", ""),
            )
        )

        for rank, (_, _, _, row) in enumerate(candidates, start=1):
            row["blended_rank"] = str(rank)

    with OUTPUT_PATH.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Input rows: {len(rows)}")
    print(f"Queries processed: {len(grouped_rows)}")
    print(f"Output saved to: {OUTPUT_PATH}")
    print("Rank blending complete.")


if __name__ == "__main__":
    main()
