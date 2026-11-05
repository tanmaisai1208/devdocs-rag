import csv
import math
from collections import defaultdict
from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "dataset"
    / "pilot_v2"
    / "candidate_labels_dev_with_reranker.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "pilot_v2"

PER_QUERY_OUTPUT = (
    RESULTS_DIR / "reranker_comparison_per_query.csv"
)

SUMMARY_OUTPUT = (
    RESULTS_DIR / "reranker_comparison_summary.csv"
)

TOP_K = 10

METHODS = {
    "hybrid_top10": "hybrid_rank",
    "hybrid_reranker": "reranker_rank",
}

K_VALUES = (5, 10)


# --------------------------------------------------
# Data loading and validation
# --------------------------------------------------

def load_candidates():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_PATH}"
        )

    with INPUT_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "query_id",
            "query",
            "chunk_id",
            "relevant",
            "hybrid_rank",
            "reranker_rank",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Missing CSV columns: {sorted(missing)}"
            )

        rows = list(reader)

    grouped = defaultdict(list)

    for line_number, row in enumerate(rows, start=2):
        raw_hybrid_rank = row["hybrid_rank"].strip()

        # Keep only the original hybrid top 10 candidates.
        if not raw_hybrid_rank:
            continue

        try:
            hybrid_rank = float(raw_hybrid_rank)
            relevance = int(row["relevant"].strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid rank or relevance label on line "
                f"{line_number}."
            ) from exc

        if (
            not math.isfinite(hybrid_rank)
            or not hybrid_rank.is_integer()
            or not 1 <= hybrid_rank <= TOP_K
        ):
            continue

        if relevance not in {0, 1}:
            raise ValueError(
                f"Invalid relevance label on line {line_number}."
            )

        raw_reranker_rank = row["reranker_rank"].strip()

        if not raw_reranker_rank:
            raise ValueError(
                f"Missing reranker rank on line {line_number}."
            )

        try:
            reranker_rank = float(raw_reranker_rank)
        except ValueError as exc:
            raise ValueError(
                f"Invalid reranker rank on line {line_number}."
            ) from exc

        if (
            not math.isfinite(reranker_rank)
            or not reranker_rank.is_integer()
            or not 1 <= reranker_rank <= TOP_K
        ):
            raise ValueError(
                f"Invalid reranker rank on line {line_number}."
            )

        grouped[row["query_id"]].append({
            "query": row["query"],
            "chunk_id": row["chunk_id"],
            "relevant": relevance,
            "hybrid_rank": int(hybrid_rank),
            "reranker_rank": int(reranker_rank),
        })

    if not grouped:
        raise ValueError("No hybrid top-10 candidates found.")

    expected_ranks = set(range(1, TOP_K + 1))

    for query_id, candidates in grouped.items():
        if len(candidates) != TOP_K:
            raise ValueError(
                f"{query_id} has {len(candidates)} candidates; "
                f"expected {TOP_K}."
            )

        hybrid_ranks = {
            candidate["hybrid_rank"]
            for candidate in candidates
        }

        reranker_ranks = {
            candidate["reranker_rank"]
            for candidate in candidates
        }

        if hybrid_ranks != expected_ranks:
            raise ValueError(
                f"Invalid hybrid ranks for {query_id}."
            )

        if reranker_ranks != expected_ranks:
            raise ValueError(
                f"Invalid reranker ranks for {query_id}."
            )

        query_texts = {
            candidate["query"] for candidate in candidates
        }

        if len(query_texts) != 1:
            raise ValueError(
                f"Inconsistent query text for {query_id}."
            )

        chunk_ids = {
            candidate["chunk_id"] for candidate in candidates
        }

        if len(chunk_ids) != TOP_K:
            raise ValueError(
                f"Duplicate candidate chunk IDs for {query_id}."
            )

    return grouped


# --------------------------------------------------
# Ranking metrics
# --------------------------------------------------

def calculate_metrics(candidates, rank_column):
    total_relevant = sum(
        candidate["relevant"]
        for candidate in candidates
    )

    if total_relevant == 0:
        return None

    ranked = sorted(
        candidates,
        key=lambda candidate: candidate[rank_column],
    )

    relevant_ranks = [
        candidate[rank_column]
        for candidate in ranked
        if candidate["relevant"] == 1
    ]

    first_relevant_rank = min(relevant_ranks)

    metrics = {
        "first_relevant_rank": first_relevant_rank,
        "mrr": 1.0 / first_relevant_rank,
    }

    for k in K_VALUES:
        found_relevant = sum(
            candidate["relevant"] == 1
            and candidate[rank_column] <= k
            for candidate in ranked
        )

        metrics[f"recall_at_{k}"] = (
            found_relevant / total_relevant
        )

        dcg = sum(
            candidate["relevant"]
            / math.log2(candidate[rank_column] + 1)
            for candidate in ranked
            if candidate[rank_column] <= k
        )

        ideal_count = min(total_relevant, k)

        ideal_dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_count + 1)
        )

        metrics[f"ndcg_at_{k}"] = (
            dcg / ideal_dcg if ideal_dcg else 0.0
        )

    return metrics


# --------------------------------------------------
# CSV output
# --------------------------------------------------

def write_csv(path, rows, fieldnames):
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------

def main():
    grouped = load_candidates()

    total_queries = len(grouped)

    queries_with_relevant_candidates = sum(
        any(candidate["relevant"] == 1 for candidate in candidates)
        for candidates in grouped.values()
    )

    candidate_hit_rate = (
        queries_with_relevant_candidates / total_queries
    )

    print(f"Input: {INPUT_PATH}")
    print(f"Evaluation queries: {total_queries}")
    print(
        "Queries with relevant candidates in hybrid top 10: "
        f"{queries_with_relevant_candidates}"
    )
    print(f"Hybrid top-10 candidate hit rate: {candidate_hit_rate:.4f}")

    per_query_results = []

    for query_id, candidates in sorted(grouped.items()):
        relevant_count = sum(
            candidate["relevant"]
            for candidate in candidates
        )

        for method_name, rank_column in METHODS.items():
            metrics = calculate_metrics(
                candidates,
                rank_column,
            )

            result = {
                "query_id": query_id,
                "retriever": method_name,
                "relevant_in_candidates": relevant_count,
                "candidate_hit": int(relevant_count > 0),
            }

            if metrics is None:
                result.update({
                    "first_relevant_rank": "",
                    "recall_at_5": "",
                    "mrr": "",
                    "ndcg_at_5": "",
                    "ndcg_at_10": "",
                })
            else:
                result.update(metrics)

            per_query_results.append(result)

    metric_columns = [
        "recall_at_5",
        "mrr",
        "ndcg_at_5",
        "ndcg_at_10",
    ]

    summary_results = []

    for method_name in METHODS:
        method_rows = [
            row for row in per_query_results
            if row["retriever"] == method_name
            and row["candidate_hit"] == 1
        ]

        summary = {
            "retriever": method_name,
            "queries_total": total_queries,
            "queries_with_relevant_candidates": (
                queries_with_relevant_candidates
            ),
            "candidate_hit_rate": candidate_hit_rate,
            "queries_scored": len(method_rows),
        }

        for metric in metric_columns:
            summary[metric] = (
                sum(row[metric] for row in method_rows)
                / len(method_rows)
                if method_rows else 0.0
            )

        summary_results.append(summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(
        PER_QUERY_OUTPUT,
        per_query_results,
        [
            "query_id",
            "retriever",
            "relevant_in_candidates",
            "candidate_hit",
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
            "queries_total",
            "queries_with_relevant_candidates",
            "candidate_hit_rate",
            "queries_scored",
            "recall_at_5",
            "recall_at_10",
            "mrr",
            "ndcg_at_5",
            "ndcg_at_10",
        ],
    )

    print("\n--- HYBRID vs. RERANKER SUMMARY ---")

    for result in summary_results:
        print(f"\n{result['retriever'].upper()}")

        for metric in metric_columns:
            print(f"  {metric}: {result[metric]:.4f}")

    print(f"\nPer-query results: {PER_QUERY_OUTPUT}")
    print(f"Summary results:   {SUMMARY_OUTPUT}")
    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()