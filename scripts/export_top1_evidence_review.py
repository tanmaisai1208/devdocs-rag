"""Export top-1 passages for manual evidence-support review.

Run from the project root:
    python scripts/export_top1_evidence_review.py

Reads retrieval-only JSONL results and creates a CSV with one row per
answerable query. It does not call an LLM or modify the input file.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path("results/answer_eval_runs/retrieval_only_after_api_tests.jsonl")
DEFAULT_OUTPUT = Path("results/answer_eval_runs/top1_evidence_review.csv")


def read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {line_no}: {exc}") from exc
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}")

    records = read_records(args.input)
    if not records:
        raise SystemExit(f"No JSONL records found in {args.input}")

    rows: list[dict[str, Any]] = []
    skipped = 0
    for record in records:
        # Only audit answerable questions; out-of-scope questions have no expected facts.
        if record.get("should_abstain") is not False:
            skipped += 1
            continue

        passages = record.get("retrieved_passages") or []
        if not passages:
            top1: dict[str, Any] = {}
        else:
            top1 = passages[0]

        expected_facts = record.get("expected_facts") or []
        rows.append({
            "query_id": record.get("query_id", ""),
            "query": record.get("query", ""),
            "expected_facts": "\n".join(f"- {fact}" for fact in expected_facts),
            "expected_fact_count": len(expected_facts),
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
            "review_directly_supports_all_facts": "",
            "review_number_of_facts_supported": "",
            "review_notes": "",
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["query_id", "query"]
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Input records: {len(records)}")
    print(f"Answerable queries exported: {len(rows)}")
    print(f"Out-of-scope / abstention queries skipped: {skipped}")
    print(f"CSV saved to: {args.output.resolve()}")
    print("\nFor each row, review the top-1 passage against the expected facts.")
    print("Fill in review_directly_supports_all_facts with yes/no, and")
    print("review_number_of_facts_supported with a count (for example, 0, 1, or 2).")
    print("Use review_notes to record missing details or text-extraction problems.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
