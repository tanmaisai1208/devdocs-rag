"""Compare the normal a08 query against focused parameter queries.

Run from the project root:
    python scripts/compare_a08_query_expansion.py

This is an experiment only: it does not change the retriever or Qdrant.
It writes two CSV reports under results/answer_eval_runs/.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.retrieval.hybrid_retriever import HybridRetriever

QUERY = (
    "Which PostgreSQL connection parameters specify the server hostname, "
    "port, database, and user?"
)

# The suffixes identify the specific PostgreSQL parameter-definition chunks,
# rather than URI examples that merely mention these terms.
TARGET_SECTIONS = {
    "host": "parameter key words > host",
    "port": "parameter key words > port",
    "dbname": "parameter key words > dbname",
    "user": "parameter key words > user",
}

FOCUSED_QUERIES = {
    "host": "PostgreSQL libpq connection parameter host hostname server address",
    "port": "PostgreSQL libpq connection parameter port number to connect to server",
    "dbname": "PostgreSQL libpq parameter dbname database name",
    "user": "PostgreSQL libpq connection parameter user name to connect as",
}

OUTPUT_DIR = Path("results/answer_eval_runs")
RRF_K = 60
CANDIDATE_K = 20
PER_QUERY_TOP_K = 20


def get_target_parameter(section: Any) -> str | None:
    normalized = str(section or "").strip().lower()
    for parameter, suffix in TARGET_SECTIONS.items():
        if normalized.endswith(suffix):
            return parameter
    return None


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    retriever = HybridRetriever(candidate_k=CANDIDATE_K, rrf_k=RRF_K)

    try:
        query_specs = [("original", QUERY)] + [
            (parameter, focused_query)
            for parameter, focused_query in FOCUSED_QUERIES.items()
        ]

        per_query_results: dict[str, list[dict[str, Any]]] = {}
        for label, query in query_specs:
            print(f"Running {label} query...")
            per_query_results[label] = retriever.retrieve(
                query=query,
                top_k=PER_QUERY_TOP_K,
            )
            print(f"  returned {len(per_query_results[label])} passages")

        baseline = per_query_results["original"]
        baseline_ranks: dict[str, int] = {}
        for rank, passage in enumerate(baseline, start=1):
            parameter = get_target_parameter(passage.get("section"))
            if parameter and parameter not in baseline_ranks:
                baseline_ranks[parameter] = rank

        # RRF across the original query and four focused queries. Each query's
        # own hybrid-ranked list contributes one reciprocal-rank score.
        fused_scores: defaultdict[str, float] = defaultdict(float)
        passage_by_id: dict[str, dict[str, Any]] = {}
        best_focused_rank: dict[str, int] = {}
        best_focused_query: dict[str, str] = {}

        for label, passages in per_query_results.items():
            for rank, passage in enumerate(passages, start=1):
                chunk_id = str(passage.get("chunk_id", ""))
                if not chunk_id:
                    continue
                fused_scores[chunk_id] += 1.0 / (RRF_K + rank)
                passage_by_id.setdefault(chunk_id, passage)
                parameter = get_target_parameter(passage.get("section"))
                if parameter and label != "original":
                    if parameter not in best_focused_rank or rank < best_focused_rank[parameter]:
                        best_focused_rank[parameter] = rank
                        best_focused_query[parameter] = label

        fused_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
        fused_rank_by_parameter: dict[str, int] = {}
        for rank, chunk_id in enumerate(fused_ids, start=1):
            passage = passage_by_id[chunk_id]
            parameter = get_target_parameter(passage.get("section"))
            if parameter and parameter not in fused_rank_by_parameter:
                fused_rank_by_parameter[parameter] = rank

        comparison_rows = []
        for parameter, suffix in TARGET_SECTIONS.items():
            target_passage = next(
                (
                    passage
                    for passage in passage_by_id.values()
                    if get_target_parameter(passage.get("section")) == parameter
                ),
                {},
            )
            comparison_rows.append(
                {
                    "parameter": parameter,
                    "target_section_suffix": suffix,
                    "original_query_rank_with_candidate_k_20": baseline_ranks.get(parameter, "NOT_IN_TOP20"),
                    "best_focused_query": best_focused_query.get(parameter, "NOT_FOUND"),
                    "best_focused_query_rank": best_focused_rank.get(parameter, "NOT_FOUND"),
                    "multi_query_rrf_rank": fused_rank_by_parameter.get(parameter, "NOT_FOUND"),
                    "target_chunk_id": target_passage.get("chunk_id", ""),
                    "target_section": target_passage.get("section", ""),
                    "target_content": target_passage.get("content", ""),
                }
            )

        comparison_path = OUTPUT_DIR / "a08_query_expansion_comparison.csv"
        with comparison_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
            writer.writeheader()
            writer.writerows(comparison_rows)

        top_rows = []
        for rank, chunk_id in enumerate(fused_ids[:30], start=1):
            passage = passage_by_id[chunk_id]
            top_rows.append(
                {
                    "multi_query_rrf_rank": rank,
                    "rrf_score": fused_scores[chunk_id],
                    "chunk_id": chunk_id,
                    "section": passage.get("section", ""),
                    "source": passage.get("source", ""),
                    "dense_rank_in_recorded_query": passage.get("dense_rank", ""),
                    "bm25_rank_in_recorded_query": passage.get("bm25_rank", ""),
                    "content": passage.get("content", ""),
                }
            )
        top_path = OUTPUT_DIR / "a08_query_expansion_top30.csv"
        with top_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(top_rows[0].keys()))
            writer.writeheader()
            writer.writerows(top_rows)

        print("\nTARGET PARAMETER CHUNKS")
        for row in comparison_rows:
            print(
                f"{row['parameter']}: original={row['original_query_rank_with_candidate_k_20']}; "
                f"best focused={row['best_focused_query']} "
                f"(rank {row['best_focused_query_rank']}); "
                f"multi-query RRF rank={row['multi_query_rrf_rank']}"
            )

        print("\nTOP 13 OF MULTI-QUERY RRF")
        for rank, chunk_id in enumerate(fused_ids[:13], start=1):
            passage = passage_by_id[chunk_id]
            print(
                f"{rank}. {passage.get('section')} "
                f"(RRF={fused_scores[chunk_id]:.6f})"
            )

        print(f"\nComparison CSV saved to: {comparison_path.resolve()}")
        print(f"Top-30 CSV saved to: {top_path.resolve()}")
        print("\nThis is a diagnostic experiment; production retrieval was not changed.")
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
