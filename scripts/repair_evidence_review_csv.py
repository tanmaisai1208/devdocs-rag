"""Repair a completed top-1 evidence review CSV without losing review annotations.

Run from the project root:
    python scripts/repair_evidence_review_csv.py

Reads the original retrieval-only JSONL as the source of truth for facts and
passage text, then merges the three review columns from the manually reviewed CSV.
Writes a NEW CSV and leaves both input files unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

DEFAULT_JSONL = Path("results/answer_eval_runs/retrieval_only_after_api_tests.jsonl")
DEFAULT_REVIEW_CSV = Path("results/answer_eval_runs/top1_evidence_review.csv")
DEFAULT_OUTPUT = Path("results/answer_eval_runs/top1_evidence_review_repaired.csv")
REVIEW_FIELDS = (
    "review_directly_supports_all_facts",
    "review_number_of_facts_supported",
    "review_notes",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc
    return records


def read_review_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No review rows found in {path}")
    return {str(row.get("query_id", "")).strip(): row for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input_jsonl.exists():
        parser.error(f"Input JSONL not found: {args.input_jsonl}")
    if not args.review_csv.exists():
        parser.error(f"Reviewed CSV not found: {args.review_csv}")

    source_records = read_jsonl(args.input_jsonl)
    reviews = read_review_csv(args.review_csv)
    output_rows: list[dict[str, Any]] = []
    missing_reviews: list[str] = []

    for record in source_records:
        if record.get("should_abstain") is not False:
            continue

        query_id = str(record.get("query_id", "")).strip()
        review = reviews.get(query_id, {})
        if not review:
            missing_reviews.append(query_id)

        passages = record.get("retrieved_passages") or []
        top1 = passages[0] if passages else {}
        facts = record.get("expected_facts") or []

        # Use plain text instead of bullet-prefixed multiline facts so Excel is
        # less likely to interpret strings beginning with '-' as formulas.
        expected_facts_text = " | ".join(str(fact).strip() for fact in facts)
        direct_support = str(review.get(REVIEW_FIELDS[0], "")).strip().lower()
        if direct_support not in {"yes", "no"}:
            direct_support = ""

        fact_count = str(review.get(REVIEW_FIELDS[1], "")).strip()
        if fact_count and fact_count.isdigit():
            fact_count = int(fact_count)
        else:
            fact_count = ""

        row = {
            "query_id": query_id,
            "query": record.get("query", ""),
            "expected_facts": expected_facts_text,
            "expected_fact_count": len(facts),
            "top1_chunk_id": top1.get("chunk_id", ""),
            "top1_hybrid_rank": top1.get("hybrid_rank", ""),
            "top1_dense_rank": top1.get("dense_rank", ""),
            "top1_bm25_rank": top1.get("bm25_rank", ""),
            "top1_dense_score": top1.get("dense_score", ""),
            "top1_source": top1.get("source", ""),
            "top1_title": top1.get("title", ""),
            "top1_section": top1.get("section", ""),
            "top1_content": top1.get("content", ""),
            "expected_source_urls": " | ".join(record.get("expected_source_urls") or []),
            "expected_sections": " | ".join(record.get("expected_sections") or []),
            "review_directly_supports_all_facts": direct_support,
            "review_number_of_facts_supported": fact_count,
            "review_notes": str(review.get(REVIEW_FIELDS[2], "")).strip(),
        }
        output_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Source JSONL records: {len(source_records)}")
    print(f"Reviewed rows merged: {len(output_rows) - len(missing_reviews)}")
    print(f"Missing review rows: {len(missing_reviews)}")
    if missing_reviews:
        print("Missing query IDs: " + ", ".join(missing_reviews))
    print(f"Repaired CSV saved to: {args.output.resolve()}")
    print("Original JSONL and reviewed CSV were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
