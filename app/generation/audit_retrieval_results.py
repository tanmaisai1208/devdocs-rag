
import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUNS_DIR = PROJECT_ROOT / "results" / "answer_eval_runs"

OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "retrieval_evidence_audit.csv"
)

def normalize_text(text: str) -> str:
    return (
        text.lower()
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )

def normalize_url(url: str) -> str:
    return url.strip().rstrip("/")


def find_latest_run() -> Path:
    runs = sorted(
        RUNS_DIR.glob("retrieval-only_*.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not runs:
        raise FileNotFoundError(
            f"No retrieval-only JSONL files found in {RUNS_DIR}"
        )

    return runs[0]


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

            records.append(record)

    return records


def audit_record(record: dict[str, Any]) -> dict[str, Any]:
    expected_sources = record["expected_source_urls"]
    expected_terms = record["expected_sections"]
    passages = record.get("retrieved_passages", [])

    normalized_expected_sources = {
        normalize_url(url) for url in expected_sources
    }

    matching_source_passages = [
        passage
        for passage in passages
        if normalize_url(passage.get("source", ""))
        in normalized_expected_sources
    ]

    found_sources = sorted({
        normalize_url(passage.get("source", ""))
        for passage in matching_source_passages
    })

    # Search expected section/concept terms only within passages
    # from the expected documentation sources.

    source_evidence = normalize_text(
        " ".join(
            " ".join([
                str(passage.get("section", "")),
                str(passage.get("title", "")),
                str(passage.get("content", "")),
            ])
            for passage in matching_source_passages
        )
    )

    found_terms = [
        term
        for term in expected_terms
        if normalize_text(term) in source_evidence
    ]

    missing_terms = [
        term
        for term in expected_terms
        if normalize_text(term) not in source_evidence
    ]


    expected_abstention = record["should_abstain"]

    if expected_abstention:
        source_status = "not_applicable"
        term_coverage = "not_applicable"
    else:
        source_status = (
            "found" if matching_source_passages else "missing"
        )

        term_coverage = (
            f"{len(found_terms)}/{len(expected_terms)}"
            if expected_terms
            else "not_applicable"
        )

    top_passage = passages[0] if passages else {}

    return {
        "query_id": record["query_id"],
        "category": record["category"],
        "should_abstain": expected_abstention,
        "run_status": record.get("run_status", ""),
        "retrieved_count": len(passages),
        "expected_source_status": source_status,
        "matching_source_passages": len(matching_source_passages),
        "expected_terms_covered": term_coverage,
        "found_terms": "; ".join(found_terms),
        "missing_terms": "; ".join(missing_terms),
        "top1_source": top_passage.get("source", ""),
        "top1_section": top_passage.get("section", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit evidence retrieved for answer evaluation."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional JSONL run file; defaults to the most recent run.",
    )
    args = parser.parse_args()

    input_path = (
        Path(args.input)
        if args.input
        else find_latest_run()
    )

    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    records = load_records(input_path)

    if not records:
        raise ValueError("The selected retrieval run is empty.")

    audited = [
        audit_record(record)
        for record in records
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(audited[0].keys()),
        )
        writer.writeheader()
        writer.writerows(audited)

    answerable = [
        row for row in audited
        if not row["should_abstain"]
    ]

    source_hits = sum(
        row["expected_source_status"] == "found"
        for row in answerable
    )

    print(f"Input run: {input_path}")
    print(f"Questions audited: {len(audited)}")
    print(f"Answerable questions: {len(answerable)}")
    print(
        f"Expected source found: {source_hits}/{len(answerable)}"
    )
    print(f"Audit CSV: {OUTPUT_PATH}")
    print(
        "Note: term matching is a text-based heuristic, "
        "not proof of semantic relevance or answer correctness."
    )


if __name__ == "__main__":
    main()
