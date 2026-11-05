
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "dataset"
    / "answer_eval.jsonl"
)

REQUIRED_FIELDS = {
    "query_id",
    "query",
    "should_abstain",
    "expected_facts",
    "expected_source_urls",
    "expected_sections",
    "category",
}


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    records = []
    seen_ids = set()

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} must contain a JSON object."
                )

            missing = REQUIRED_FIELDS - set(record)
            if missing:
                raise ValueError(
                    f"Line {line_number} is missing fields: "
                    f"{sorted(missing)}"
                )

            query_id = record["query_id"]

            if not isinstance(query_id, str) or not query_id.strip():
                raise ValueError(
                    f"Invalid query_id on line {line_number}."
                )

            if query_id in seen_ids:
                raise ValueError(
                    f"Duplicate query_id: {query_id}"
                )

            seen_ids.add(query_id)

            if (
                not isinstance(record["query"], str)
                or not record["query"].strip()
            ):
                raise ValueError(
                    f"Empty query on line {line_number}."
                )

            if not isinstance(record["should_abstain"], bool):
                raise ValueError(
                    f"should_abstain must be boolean on line "
                    f"{line_number}."
                )

            for field in (
                "expected_facts",
                "expected_source_urls",
                "expected_sections",
            ):
                if not isinstance(record[field], list):
                    raise ValueError(
                        f"{field} must be a list on line "
                        f"{line_number}."
                    )

                if not all(
                    isinstance(item, str) and item.strip()
                    for item in record[field]
                ):
                    raise ValueError(
                        f"{field} contains an invalid item on "
                        f"line {line_number}."
                    )

            if record["should_abstain"]:
                if (
                    record["expected_facts"]
                    or record["expected_source_urls"]
                    or record["expected_sections"]
                ):
                    raise ValueError(
                        f"Abstention case {query_id} should not "
                        "declare required answer facts or sources."
                    )
            else:
                if (
                    not record["expected_facts"]
                    or not record["expected_source_urls"]
                    or not record["expected_sections"]
                ):
                    raise ValueError(
                        f"Answerable case {query_id} requires "
                        "expected facts, sources, and sections."
                    )

            records.append(record)

    if not records:
        raise ValueError("The evaluation dataset is empty.")

    answerable_count = sum(
        not record["should_abstain"]
        for record in records
    )

    abstention_count = sum(
        record["should_abstain"]
        for record in records
    )

    print("Dataset validation passed.")
    print(f"Total questions: {len(records)}")
    print(f"Answerable questions: {answerable_count}")
    print(f"Expected abstentions: {abstention_count}")
    print(f"Dataset: {DATASET_PATH}")


if __name__ == "__main__":
    main()
