#!/usr/bin/env python3
"""Summarize manually reviewed RAG answer-evaluation CSV results.

Uses only the Python standard library. Review scores are summarized as supplied;
this script does not determine whether answers or citations are actually correct.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

SCORE_COLUMNS = {
    "Expected-fact coverage": "review_fact_coverage",
    "Factual correctness": "review_factual_correctness",
    "Citation support": "review_citation_support",
}


def parse_bool(value: Any) -> bool | None:
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def parse_number(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"na", "n/a", "nan", "none", "null"}:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # filter NaN


def metric_summary(values: list[float]) -> str:
    if not values:
        return "No numeric scores"
    total = sum(values)
    max_total = 2 * len(values)
    return (
        f"{total:g}/{max_total} points ({total / max_total * 100:.1f}%), "
        f"mean {statistics.mean(values):.2f}/2 across {len(values)} reviewed answers"
    )


def latency_summary(values: list[float]) -> str:
    if not values:
        return "No numeric values"
    return (
        f"median {statistics.median(values):,.1f} ms; "
        f"mean {statistics.mean(values):,.1f} ms; n={len(values)}"
    )


def token_summary(values: list[float]) -> str:
    if not values:
        return "No numeric values"
    return (
        f"total {sum(values):,.0f}; mean {statistics.mean(values):,.1f}; "
        f"median {statistics.median(values):,.0f}; n={len(values)}"
    )


def generate_report(input_path: Path) -> str:
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"No data rows found in {input_path}")

    required = {"query_id", *SCORE_COLUMNS.values(), "should_abstain", "actual_abstained"}
    missing = sorted(required - set(rows[0].keys()))
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))

    answerable_rows = [row for row in rows if parse_bool(row.get("should_abstain")) is False]
    expected_abstain_rows = [row for row in rows if parse_bool(row.get("should_abstain")) is True]
    unknown_scope_rows = [row for row in rows if parse_bool(row.get("should_abstain")) is None]

    output: list[str] = [
        "# RAG answer-review summary",
        "",
        f"- **Input file:** `{input_path}`",
        f"- **Questions in CSV:** {len(rows)}",
        f"- **Expected-answerable questions:** {len(answerable_rows)}",
        f"- **Expected-abstention questions:** {len(expected_abstain_rows)}",
        f"- **Rows with unknown `should_abstain`:** {len(unknown_scope_rows)}",
        "",
        "> **Important:** This report calculates summaries from the CSV's review labels. It does not independently validate those labels. If the CSV contains assistant-generated first-pass reviews, have a human check them before reporting these values as benchmark results.",
        "",
        "## Answer-review scores (provisional)",
        "",
        "Scores use a 0–2 scale and are summarized on expected-answerable questions only. Blank values are excluded.",
        "",
        "| Metric | Summary |",
        "|---|---|",
    ]

    for label, column in SCORE_COLUMNS.items():
        values = [
            score
            for row in answerable_rows
            if (score := parse_number(row.get(column))) is not None
        ]
        output.append(f"| {label} | {metric_summary(values)} |")

    abstention_counts = Counter(str(row.get("review_abstention_quality") or "(blank)").strip() for row in rows)
    actual_matches = 0
    actual_known = 0
    for row in rows:
        expected = parse_bool(row.get("should_abstain"))
        actual = parse_bool(row.get("actual_abstained"))
        if expected is not None and actual is not None:
            actual_known += 1
            actual_matches += int(expected == actual)

    output.extend([
        "",
        "## Abstention and execution checks",
        "",
        f"- **Expected vs. actual abstention agreement:** {actual_matches}/{actual_known} questions" if actual_known else "- **Expected vs. actual abstention agreement:** not available",
        f"- **Review abstention labels:** {dict(abstention_counts)}",
    ])

    latency_values = [v for row in rows if (v := parse_number(row.get("retrieval_latency_ms"))) is not None]
    generation_values = [v for row in answerable_rows if (v := parse_number(row.get("generation_latency_ms"))) is not None and v > 0]
    token_values = [v for row in answerable_rows if (v := parse_number(row.get("total_tokens"))) is not None]

    output.extend([
        "",
        "## Performance",
        "",
        f"- **Retrieval latency:** {latency_summary(latency_values)}",
        f"- **Generation latency (answerable questions with a recorded generation):** {latency_summary(generation_values)}",
        f"- **Tokens per generated answer:** {token_summary(token_values)}",
        "",
        "## Per-question review",
        "",
        "| ID | Should abstain | Did abstain | Fact coverage | Factual correctness | Citation support | Abstention review |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])

    for row in rows:
        cells = [
            str(row.get("query_id", "")),
            str(parse_bool(row.get("should_abstain"))),
            str(parse_bool(row.get("actual_abstained"))),
            str(row.get("review_fact_coverage") or "—"),
            str(row.get("review_factual_correctness") or "—"),
            str(row.get("review_citation_support") or "—"),
            str(row.get("review_abstention_quality") or "—"),
        ]
        cells = [cell.replace("|", "\\|").replace("\n", " ") for cell in cells]
        output.append("| " + " | ".join(cells) + " |")

    output.extend([
        "",
        "## Recommended interpretation",
        "",
        "Treat these numbers as a snapshot of the current review sheet. Keep the original generation JSONL and this CSV as separate artifacts. After manually verifying the review labels, rerun this script to regenerate the summary.",
        "",
    ])
    return "\n".join(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="evaluation/answer_review_top13.csv",
        help="Reviewed CSV path (default: evaluation/answer_review_top13.csv)",
    )
    parser.add_argument(
        "--output",
        default="results/answer_eval_runs/answer_review_summary_top13.md",
        help="Markdown report path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_file():
        parser.error(f"Input CSV does not exist: {input_path}")

    report = generate_report(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Review summary saved to: {output_path.resolve()}")
    print("\n".join(report.splitlines()[:30]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
