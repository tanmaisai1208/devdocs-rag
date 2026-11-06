
import csv
import math
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "dataset"
    / "pilot_v2"
    / "candidate_labels_dev_final.csv"
)

LABEL_COLUMN = "final_relevant"

RESULTS_DIR = PROJECT_ROOT / "results" / "pilot_v2_final_labels"
PER_QUERY_OUTPUT = RESULTS_DIR / "retrieval_metrics_per_query.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "retrieval_metrics_summary.csv"

RETRIEVERS = {
    "dense": "dense_rank",
    "bm25": "bm25_rank",
    "hybrid": "hybrid_rank",
}

K_VALUES = (5, 10)


def load_candidates(path):
    """Load and validate the labeled candidate CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation CSV not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        required_columns = {
            "query_id",
            "query",
            "chunk_id",
            LABEL_COLUMN,
            *RETRIEVERS.values(),
        }

        fieldnames = set(reader.fieldnames or [])
        missing = required_columns - fieldnames

        if missing:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing)}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError("The evaluation CSV contains no rows.")

    seen_pairs = set()

    for line_number, row in enumerate(rows, start=2):
        query_id = row["query_id"].strip()
        chunk_id = row["chunk_id"].strip()
        label = row[LABEL_COLUMN].strip()

        if not query_id or not chunk_id:
            raise ValueError(
                f"Missing query_id or chunk_id on CSV line {line_number}."
            )

        if label not in {"0", "1"}:
            raise ValueError(
                f"Invalid relevance label on line {line_number}: {label!r}. "
                "Every label must be 0 or 1."
            )

        pair = (query_id, chunk_id)
        if pair in seen_pairs:
            raise ValueError(
                f"Duplicate query/chunk pair on line {line_number}: {pair}"
            )
        seen_pairs.add(pair)

        row["relevant"] = int(label)

        for rank_column in RETRIEVERS.values():
            raw_rank = row[rank_column].strip()

            if not raw_rank:
                row[rank_column] = None
                continue

            try:
                rank = float(raw_rank)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid rank in {rank_column} on line {line_number}: "
                    f"{raw_rank!r}"
                ) from exc

            if not math.isfinite(rank) or rank < 1 or not rank.is_integer():
                raise ValueError(
                    f"Rank must be a positive integer in {rank_column} "
                    f"on line {line_number}: {raw_rank!r}"
                )

            row[rank_column] = int(rank)

    return rows


def dcg_at_k(ranked_rows, k):
    """Calculate DCG@k using binary relevance."""
    dcg = 0.0

    for row in ranked_rows:
        rank = row["rank"]

        if rank > k:
            continue

        relevance = row["relevant"]
        dcg += relevance / math.log2(rank + 1)

    return dcg


def evaluate_query(rows, rank_column):
    """Calculate retrieval metrics for one query and one retriever."""
    total_relevant = sum(row["relevant"] for row in rows)

    if total_relevant == 0:
        raise ValueError(
            "A query has no relevant labeled candidates. "
            "Review its labels before calculating metrics."
        )

    ranked_rows = [
        {
            "rank": row[rank_column],
            "relevant": row["relevant"],
        }
        for row in rows
        if row[rank_column] is not None
    ]

    ranked_rows.sort(key=lambda row: row["rank"])

    relevant_ranks = [
        row["rank"]
        for row in ranked_rows
        if row["relevant"] == 1
    ]

    reciprocal_rank = (
        1.0 / min(relevant_ranks)
        if relevant_ranks
        else 0.0
    )

    metrics = {
        "relevant_in_pool": total_relevant,
        "first_relevant_rank": (
            min(relevant_ranks) if relevant_ranks else ""
        ),
        "mrr": reciprocal_rank,
    }

    for k in K_VALUES:
        found_relevant = sum(
            row["relevant"] == 1 and row["rank"] <= k
            for row in ranked_rows
        )

        metrics[f"recall_at_{k}"] = found_relevant / total_relevant

        dcg = dcg_at_k(ranked_rows, k)

        # Ideal ranking puts all known relevant candidates first.
        ideal_relevant_count = min(total_relevant, k)
        ideal_dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_relevant_count + 1)
        )

        metrics[f"ndcg_at_{k}"] = (
            dcg / ideal_dcg if ideal_dcg else 0.0
        )

    return metrics


def write_csv(path, rows, fieldnames):
    """Write results to a CSV file."""
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = load_candidates(INPUT_PATH)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["query_id"]].append(row)

    if len(grouped) < 2:
        raise ValueError(
            "At least two queries are required for a useful comparison."
        )

    print(f"Input file: {INPUT_PATH}")
    print(f"Candidate rows: {len(rows)}")
    print(f"Evaluation queries: {len(grouped)}")

    per_query_results = []

    for query_id, query_rows in sorted(grouped.items()):
        query_text = query_rows[0]["query"]

        if any(row["query"] != query_text for row in query_rows):
            raise ValueError(
                f"Inconsistent query text for query_id {query_id}."
            )

        for retriever_name, rank_column in RETRIEVERS.items():
            metrics = evaluate_query(query_rows, rank_column)

            per_query_results.append({
                "query_id": query_id,
                "retriever": retriever_name,
                "relevant_in_pool": metrics["relevant_in_pool"],
                "first_relevant_rank": metrics["first_relevant_rank"],
                "recall_at_5": metrics["recall_at_5"],
                "recall_at_10": metrics["recall_at_10"],
                "mrr": metrics["mrr"],
                "ndcg_at_5": metrics["ndcg_at_5"],
                "ndcg_at_10": metrics["ndcg_at_10"],
            })

    metric_columns = [
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "ndcg_at_5",
        "ndcg_at_10",
    ]

    summary_results = []

    for retriever_name in RETRIEVERS:
        method_rows = [
            row for row in per_query_results
            if row["retriever"] == retriever_name
        ]

        summary = {
            "retriever": retriever_name,
            "queries_evaluated": len(method_rows),
        }

        for metric in metric_columns:
            summary[metric] = sum(
                row[metric] for row in method_rows
            ) / len(method_rows)

        summary_results.append(summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(
        PER_QUERY_OUTPUT,
        per_query_results,
        [
            "query_id",
            "retriever",
            "relevant_in_pool",
            "first_relevant_rank",
            "recall_at_5",
            "recall_at_10",
            "mrr",
            "ndcg_at_5",
            "ndcg_at_10",
        ],
    )

    write_csv(
        SUMMARY_OUTPUT,
        summary_results,
        [
            "retriever",
            "queries_evaluated",
            "recall_at_5",
            "recall_at_10",
            "mrr",
            "ndcg_at_5",
            "ndcg_at_10",
        ],
    )

    print("\n--- RETRIEVAL EVALUATION SUMMARY ---")
    for result in summary_results:
        print(f"\n{result['retriever'].upper()}")
        for metric in metric_columns:
            print(f"  {metric}: {result[metric]:.4f}")

    print(f"\nPer-query results: {PER_QUERY_OUTPUT}")
    print(f"Summary results:   {SUMMARY_OUTPUT}")
    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
