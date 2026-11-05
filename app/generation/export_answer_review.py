"""Export generated RAG answers into a CSV for structured human review.

Example:
    python app/generation/export_answer_review.py \
        --input results/answer_eval_runs/generation_top13_full.jsonl \
        --output evaluation/answer_review_top13.csv

The script does not call any model APIs. It preserves expected facts and includes
citation metadata plus excerpts from the exact retrieved passages when available.
The reviewer columns are intentionally blank for manual scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REVIEW_COLUMNS = [
    "review_fact_coverage",        # 2=all, 1=some, 0=none, NA=not applicable
    "review_factual_correctness",  # 2=correct, 1=partially correct, 0=incorrect
    "review_citation_support",     # 2=all key claims supported, 1=partially, 0=unsupported
    "review_abstention_quality",   # correct_abstention / correct_answer / incorrect
    "review_unsupported_claims",
    "review_missing_facts",
    "review_notes",
]

BASE_COLUMNS = [
    "query_id",
    "category",
    "query",
    "expected_facts",
    "expected_sections",
    "expected_source_urls",
    "should_abstain",
    "actual_abstained",
    "run_status",
    "generation_status",
    "evidence_gate_sufficient",
    "evidence_gate_reason",
    "citation_status_validator",
    "invalid_citations",
    "citation_evidence",
    "answer",
    "retrieval_latency_ms",
    "generation_latency_ms",
    "input_tokens",
    "output_tokens",
    "total_tokens",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected each JSONL line to be an object; line {line_number} is not."
                )
            records.append(value)
    return records


def json_cell(value: Any) -> str:
    """Serialize structured values as readable, spreadsheet-safe JSON text."""
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, indent=2)


def list_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(f"{index}. {item}" for index, item in enumerate(value, start=1))
    if isinstance(value, dict):
        return json_cell(value)
    return str(value)


def make_citation_evidence(record: dict[str, Any], excerpt_chars: int) -> str:
    citations = record.get("citations") or []
    passages = record.get("retrieved_passages") or []
    passages_by_id = {
        str(passage.get("chunk_id")): passage
        for passage in passages
        if passage.get("chunk_id") is not None
    }

    blocks: list[str] = []
    seen: set[tuple[str, str]] = set()

    for citation in citations:
        if not isinstance(citation, dict):
            continue

        citation_id = str(citation.get("citation_id", "?"))
        chunk_id = str(citation.get("chunk_id", ""))
        dedupe_key = (citation_id, chunk_id)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        passage = passages_by_id.get(chunk_id, {})
        section = citation.get("section") or passage.get("section") or "(section unavailable)"
        source = citation.get("source") or passage.get("source") or "(source unavailable)"
        content = passage.get("content") or passage.get("page_content") or ""
        if content and len(content) > excerpt_chars:
            content = content[:excerpt_chars].rstrip() + " … [excerpt truncated]"
        if not content:
            content = "(matching retrieved passage content unavailable)"

        blocks.append(
            f"[{citation_id}] {section}\nSource: {source}\n"
            f"Chunk ID: {chunk_id or '(missing)'}\nEvidence excerpt:\n{content}"
        )

    return "\n\n---\n\n".join(blocks)


def make_row(record: dict[str, Any], excerpt_chars: int) -> dict[str, Any]:
    evidence_gate = record.get("evidence_gate") or {}
    token_usage = record.get("token_usage") or {}

    row: dict[str, Any] = {
        "query_id": record.get("query_id", ""),
        "category": record.get("category", ""),
        "query": record.get("query", ""),
        "expected_facts": list_cell(record.get("expected_facts")),
        "expected_sections": list_cell(record.get("expected_sections")),
        "expected_source_urls": list_cell(record.get("expected_source_urls")),
        "should_abstain": record.get("should_abstain", ""),
        "actual_abstained": record.get("abstained", ""),
        "run_status": record.get("run_status", ""),
        "generation_status": record.get("generation_status", ""),
        "evidence_gate_sufficient": evidence_gate.get("sufficient", ""),
        "evidence_gate_reason": evidence_gate.get("reason", ""),
        "citation_status_validator": record.get("citation_status", ""),
        "invalid_citations": json_cell(record.get("invalid_citations")),
        "citation_evidence": make_citation_evidence(record, excerpt_chars),
        "answer": record.get("answer", ""),
        "retrieval_latency_ms": record.get("retrieval_latency_ms", ""),
        "generation_latency_ms": record.get("generation_latency_ms", ""),
        "input_tokens": token_usage.get("input_tokens", ""),
        "output_tokens": token_usage.get("output_tokens", ""),
        "total_tokens": token_usage.get("total_tokens", ""),
    }
    for column in REVIEW_COLUMNS:
        row[column] = ""
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export RAG answer results to a human-review CSV. No model calls are made."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/answer_eval_runs/generation_top13_full.jsonl"),
        help="Input JSONL produced by run_answer_evaluation.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/answer_review_top13.csv"),
        help="Output CSV review template",
    )
    parser.add_argument(
        "--excerpt-chars",
        type=int,
        default=900,
        help="Maximum characters of each cited retrieved passage included in the CSV (default: 900)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.excerpt_chars < 100:
        raise SystemExit("--excerpt-chars must be at least 100.")
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}\nRun the generation evaluation first or pass --input.")

    records = read_jsonl(args.input)
    if not records:
        raise SystemExit(f"No JSONL records found in: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = BASE_COLUMNS + REVIEW_COLUMNS
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(make_row(record, args.excerpt_chars))

    print(f"Input records: {len(records)}")
    print(f"Review CSV saved to: {args.output.resolve()}")
    print("The reviewer scoring columns are blank. Fill them in manually; no model API was called.")
    print("Scoring guide:")
    print("  review_fact_coverage: 2=all expected facts, 1=some, 0=none, NA=not applicable")
    print("  review_factual_correctness: 2=correct, 1=partially correct, 0=incorrect")
    print("  review_citation_support: 2=all key claims supported, 1=partially, 0=unsupported")
    print("  review_abstention_quality: correct_abstention / correct_answer / incorrect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
