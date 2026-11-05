
import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from dotenv import parser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"


if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from retrieval.evidence_gate import assess_evidence_sufficiency


DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "dataset"
    / "answer_eval.jsonl"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "answer_eval_runs"

DEFAULT_DENSE_THRESHOLD = 0.47

REQUIRED_FIELDS = {
    "query_id",
    "query",
    "should_abstain",
    "expected_facts",
    "expected_source_urls",
    "expected_sections",
    "category",
}


def load_eval_dataset(
    path: Path = DATASET_PATH,
) -> list[dict[str, Any]]:
    """Load and perform basic structural validation of the dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    records = []
    seen_ids = set()

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
                raise ValueError(f"Duplicate query_id: {query_id}")

            if (
                not isinstance(record["query"], str)
                or not record["query"].strip()
            ):
                raise ValueError(f"Empty query on line {line_number}.")

            if not isinstance(record["should_abstain"], bool):
                raise ValueError(
                    f"should_abstain must be boolean on line {line_number}."
                )

            for field in (
                "expected_facts",
                "expected_source_urls",
                "expected_sections",
            ):
                if not isinstance(record[field], list):
                    raise ValueError(
                        f"{field} must be a list on line {line_number}."
                    )

            seen_ids.add(query_id)
            records.append(record)

    if not records:
        raise ValueError("The evaluation dataset is empty.")

    return records


def select_queries(
    records: list[dict[str, Any]],
    start_index: int = 0,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Select a consecutive, zero-indexed subset of evaluation questions."""
    if start_index < 0:
        raise ValueError("start_index cannot be negative.")

    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1.")

    selected = records[start_index:]

    if limit is not None:
        selected = selected[:limit]

    return selected


def normalize_passage(
    passage: dict[str, Any],
) -> dict[str, Any]:
    """Keep useful evidence and ranking details in evaluation output."""
    metadata = passage.get("metadata", {})

    if not isinstance(metadata, dict):
        metadata = {}

    def get_value(field: str) -> Any:
        value = passage.get(field)

        if value is None or value == "":
            value = metadata.get(field)

        return value

    content = passage.get("content")

    if not isinstance(content, str) or not content.strip():
        content = passage.get("page_content", "")

    return {
        "chunk_id": str(get_value("chunk_id") or ""),
        "content": content if isinstance(content, str) else "",
        "source": get_value("source") or "",
        "title": get_value("title") or "",
        "section": get_value("section") or "",
        "hybrid_rank": passage.get("hybrid_rank"),
        "dense_rank": passage.get("dense_rank"),
        "bm25_rank": passage.get("bm25_rank"),
        "rrf_score": passage.get("rrf_score"),
        "dense_score": passage.get("dense_score"),
        "bm25_score": passage.get("bm25_score"),
        "max_dense_candidate_score": passage.get(
            "max_dense_candidate_score"
        ),
    }


def build_result_record(
    query_record: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    """Prepare a consistent result schema for one evaluation question."""
    return {
        "query_id": query_record["query_id"],
        "query": query_record["query"],
        "category": query_record["category"],
        "should_abstain": query_record["should_abstain"],
        "expected_facts": query_record["expected_facts"],
        "expected_source_urls": query_record["expected_source_urls"],
        "expected_sections": query_record["expected_sections"],
        "run_mode": mode,
        "run_status": "pending",
        "retrieved_passages": [],
        "retrieval_latency_ms": None,
        "generation_latency_ms": None,
        "generation_status": None,
        "evidence_gate": None,
        "dense_score_threshold": None,
        "answer": None,
        "citations": [],
        "invalid_citations": [],
        "citation_status": None,
        "abstained": None,
        "model": None,
        "token_usage": None,
        "error_type": None,
        "error": None,
    }


def write_record(
    output_file: TextIO,
    record: dict[str, Any],
) -> None:
    """Write one complete record, then flush so progress is preserved."""
    output_file.write(
        json.dumps(record, ensure_ascii=False, default=str) + "\n"
    )
    output_file.flush()


def make_output_path(mode: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"{mode}_{timestamp}.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the engineering-documentation answer evaluation."
    )

    parser.add_argument(
        "--mode",
        choices=["dry-run", "retrieval-only", "generate"],
        default="dry-run",
        help="dry-run makes no model calls; retrieval-only skips Gemini.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DATASET_PATH),
        help="Evaluation JSONL dataset path, relative to the project root or absolute.",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based position in answer_eval.jsonl.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of questions to process.",
    )

    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
        help="Seconds to wait before each Gemini generation request.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path, relative to the project root or absolute.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of final passages to retrieve per question.",
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=10,
        help="Number of dense and BM25 candidates before RRF fusion.",
    )

    parser.add_argument(
        "--dense-threshold",
        type=float,
        default=DEFAULT_DENSE_THRESHOLD,
        help=(
            "Minimum maximum dense-candidate similarity required "
            "for generation."
        ),
    )

    args = parser.parse_args()

    # Validate CLI arguments before loading models.
    if args.top_k < 1:
        print("Argument error: --top-k must be at least 1.")
        return 2

    if args.candidate_k < 1:
        print("Argument error: --candidate-k must be at least 1.")
        return 2

    if not math.isfinite(args.request_delay) or args.request_delay < 0:
        print("Argument error: --request-delay must be non-negative.")
        return 2

    if not math.isfinite(args.dense_threshold):
        print("Argument error: --dense-threshold must be a finite number.")
        return 2

    if args.dense_threshold < -1 or args.dense_threshold > 1:
        print(
            "Argument error: --dense-threshold must be between -1 and 1 "
            "for cosine similarity."
        )
        return 2

    dataset_path = Path(args.dataset)

    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    try:
        records = load_eval_dataset(dataset_path)
        selected = select_queries(
            records,
            start_index=args.start_index,
            limit=args.limit,
        )
    except (ValueError, FileNotFoundError) as error:
        print(f"Dataset error: {error}")
        return 1

    print(f"Dataset questions: {len(records)}")
    print(f"Dataset file: {dataset_path}")
    print(f"Selected questions: {len(selected)}")
    print(f"Mode: {args.mode}")

    if not selected:
        print("No questions selected.")
        return 0

    print(
        "Questions: "
        + ", ".join(record["query_id"] for record in selected)
    )

    if args.mode == "dry-run":
        print("Dry run complete. No retrieval or model calls were made.")
        return 0

    # Require an explicit batch size to prevent accidental API usage.
    if args.mode == "generate" and args.limit is None:
        print(
            "Generation requires --limit to control API usage. "
            "Start with --limit 1."
        )
        return 2

    output_path = (
        Path(args.output)
        if args.output
        else make_output_path(args.mode)
    )

    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Import retrieval only after dry-run handling.
    from retrieval.hybrid_retriever import HybridRetriever

    retriever = None

    try:
        retriever = HybridRetriever(
            candidate_k=args.candidate_k,
        )

        generator = None

        if args.mode == "generate":
            from generation.answer_generator import AnswerGenerator

            generator = AnswerGenerator()

        with output_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as output_file:
            for index, query_record in enumerate(selected, start=1):
                record = build_result_record(query_record, args.mode)
                record["retrieval_top_k"] = args.top_k
                record["retrieval_candidate_k"] = args.candidate_k
                record["dense_score_threshold"] = args.dense_threshold

                print(
                    f"\n[{index}/{len(selected)}] "
                    f"{query_record['query_id']}: "
                    f"{query_record['query']}"
                )

                try:
                    start = time.perf_counter()

                    passages = retriever.retrieve(
                        query=query_record["query"],
                        top_k=args.top_k,
                    )

                    record["retrieval_latency_ms"] = round(
                        (time.perf_counter() - start) * 1000,
                        2,
                    )

                    # Normalize the passages once, then use the same
                    # normalized records for output and gate evaluation.
                    normalized_passages = [
                        normalize_passage(passage)
                        for passage in passages
                    ]

                    record["retrieved_passages"] = normalized_passages

                    gate_result = assess_evidence_sufficiency(
                        normalized_passages,
                        threshold=args.dense_threshold,
                    )

                    record["evidence_gate"] = gate_result

                except Exception as error:
                    record["run_status"] = "retrieval_error"
                    record["error_type"] = type(error).__name__
                    record["error"] = str(error)
                    write_record(output_file, record)

                    print(
                        f"Retrieval or evidence-gate evaluation failed: "
                        f"{type(error).__name__}: {error}"
                    )
                    break

                score = gate_result["max_dense_score"]
                score_text = (
                    f"{score:.4f}" if score is not None else "unavailable"
                )

                gate_status = (
                    "PASS"
                    if gate_result["sufficient"]
                    else "REJECT"
                )

                print(
                    f"Evidence gate: {gate_status} "
                    f"(maximum dense score: {score_text}; "
                    f"threshold: {args.dense_threshold:.4f})."
                )

                if args.mode == "retrieval-only":
                    record["run_status"] = "retrieval_only"
                    record["generation_status"] = "skipped"

                else:
                    try:
                        # If the gate rejects the evidence, pass an empty
                        # list to AnswerGenerator. Its existing empty-input
                        # path returns the abstention response without
                        # making a Gemini API call.
                        passages_for_generation = (
                            passages
                            if gate_result["sufficient"]
                            else []
                        )

                        if gate_result["sufficient"] and args.request_delay > 0:
                            time.sleep(args.request_delay)

                        result = generator.generate_answer(
                            question=query_record["query"],
                            retrieved_passages=passages_for_generation,
                        )

                        generation_status = (
                            "success"
                            if gate_result["sufficient"]
                            else "skipped_insufficient_evidence"
                        )

                        record.update({
                            "run_status": "complete",
                            "generation_status": generation_status,
                            "answer": result["answer"],
                            "citations": result["citations"],
                            "invalid_citations": result["invalid_citations"],
                            "citation_status": result["citation_status"],
                            "abstained": result["abstained"],
                            "model": result["model"],
                            "generation_latency_ms": result["latency_ms"],
                            "token_usage": result["token_usage"],
                        })

                    except Exception as error:
                        record["run_status"] = "generation_error"
                        record["generation_status"] = "error"
                        record["error_type"] = type(error).__name__
                        record["error"] = str(error)
                        write_record(output_file, record)

                        print(
                            f"Generation failed: "
                            f"{type(error).__name__}: {error}"
                        )
                        print(
                            "Stopping after the first generation error "
                            "to avoid repeated API requests."
                        )
                        break

                write_record(output_file, record)

                print(
                    f"Retrieved {len(record['retrieved_passages'])} passages "
                    f"in {record['retrieval_latency_ms']} ms."
                )

                if args.mode == "generate":
                    if (
                        record["generation_status"]
                        == "skipped_insufficient_evidence"
                    ):
                        print(
                            "Generation skipped: evidence was insufficient. "
                            "Gemini was not called."
                        )
                    else:
                        print(
                            f"Generation latency: "
                            f"{record['generation_latency_ms']} ms."
                        )

                    print(
                        f"Abstained: {record['abstained']}; "
                        f"citation status: {record['citation_status']}."
                    )

    except Exception as error:
        print(
            f"Evaluation setup failed: "
            f"{type(error).__name__}: {error}"
        )
        return 1

    finally:
        if retriever is not None:
            retriever.close()

    print(f"\nResults saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
