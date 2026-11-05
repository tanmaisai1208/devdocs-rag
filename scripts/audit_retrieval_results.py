"""Audit source and section coverage in retrieval-only JSONL results.

This is a lightweight diagnostic, not a semantic relevance evaluator. Exact source
matches can be too generous when an entire documentation page contains many sections.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path(
    "results/answer_eval_runs/retrieval_only_after_api_tests.jsonl"
)
DEFAULT_OUTPUT = Path("results/answer_eval_runs/retrieval_audit_summary.csv")


def normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def get_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def source_matches(passage: dict[str, Any], expected_urls: list[str]) -> bool:
    source = str(passage.get("source", "")).strip()
    return bool(source) and source in expected_urls


def section_matches(passage: dict[str, Any], expected_sections: list[str]) -> bool:
    actual = normalize(passage.get("section", ""))
    if not actual:
        return False
    for expected in expected_sections:
        target = normalize(expected)
        if target and (target in actual or actual in target):
            return True
    return False


def first_match_rank(passages: list[dict[str, Any]], matcher, expected: list[str]) -> str:
    if not expected:
        return "N/A"
    for rank, passage in enumerate(passages, start=1):
        if matcher(passage, expected):
            return str(rank)
    return "not found"


def coverage_at_k(passages: list[dict[str, Any]], matcher, expected: list[str], k: int) -> str:
    if not expected:
        return "N/A"
    return "yes" if any(matcher(p, expected) for p in passages[:k]) else "no"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input JSONL file not found: {args.input}")

    records: list[dict[str, Any]] = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {line_number}: {exc}") from exc

    if not records:
        raise SystemExit(f"No records found in {args.input}")

    rows: list[dict[str, Any]] = []
    for record in records:
        passages = record.get("retrieved_passages") or []
        expected_urls = get_list(record.get("expected_source_urls"))
        expected_sections = get_list(record.get("expected_sections"))
        top_sections = [str(p.get("section", "")) for p in passages[:5]]

        row = {
            "query_id": record.get("query_id", ""),
            "category": record.get("category", ""),
            "should_abstain": record.get("should_abstain", ""),
            "query": record.get("query", ""),
            "expected_sections": " | ".join(expected_sections),
            "expected_source_urls": " | ".join(expected_urls),
            "retrieved_count": len(passages),
            "first_expected_source_rank": first_match_rank(passages, source_matches, expected_urls),
            "expected_source_hit_top1": coverage_at_k(passages, source_matches, expected_urls, 1),
            "expected_source_hit_top5": coverage_at_k(passages, source_matches, expected_urls, 5),
            "expected_source_hit_top13": coverage_at_k(passages, source_matches, expected_urls, 13),
            "first_expected_section_rank": first_match_rank(passages, section_matches, expected_sections),
            "expected_section_hit_top1": coverage_at_k(passages, section_matches, expected_sections, 1),
            "expected_section_hit_top5": coverage_at_k(passages, section_matches, expected_sections, 5),
            "expected_section_hit_top13": coverage_at_k(passages, section_matches, expected_sections, 13),
            "top1_section": str(passages[0].get("section", "")) if passages else "",
            "top5_sections": " || ".join(top_sections),
            "evidence_gate_sufficient": (record.get("evidence_gate") or {}).get("sufficient", ""),
            "max_dense_score": (record.get("evidence_gate") or {}).get("max_dense_score", ""),
            "retrieval_latency_ms": record.get("retrieval_latency_ms", ""),
        }
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    print(f"Input records: {total}")
    print(f"CSV saved to: {args.output.resolve()}")
    print("\nCoverage summary (source matches are diagnostic; section matches are approximate):")
    for label, field in [
        ("Expected source found in top 1", "expected_source_hit_top1"),
        ("Expected source found in top 5", "expected_source_hit_top5"),
        ("Expected source found in top 13", "expected_source_hit_top13"),
        ("Expected section match in top 1", "expected_section_hit_top1"),
        ("Expected section match in top 5", "expected_section_hit_top5"),
        ("Expected section match in top 13", "expected_section_hit_top13"),
    ]:
        eligible = [r for r in rows if r[field] != "N/A"]
        hits = sum(r[field] == "yes" for r in eligible)
        print(f"  {label}: {hits}/{len(eligible)}")

    print("\nPer-query first expected-section rank:")
    for row in rows:
        print(
            f"  {row['query_id']}: section rank={row['first_expected_section_rank']}; "
            f"source rank={row['first_expected_source_rank']}; "
            f"gate={row['evidence_gate_sufficient']}"
        )

    print("\nNote: lexical section-name matching can miss valid evidence or count a broad heading.")
    print("Review the CSV rows and passage text before treating these figures as quality scores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
