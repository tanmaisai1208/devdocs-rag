"""Compare baseline hybrid retrieval with query-driven parameter-definition boosting.

Run from the project root:
    python scripts/compare_parameter_boost_all_queries.py

This is an experiment only. It does not modify the production retriever.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from app.retrieval.hybrid_retriever import HybridRetriever

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
QUERIES_PATH = ROOT / "evaluation" / "dataset" / "queries.jsonl"
OUTPUT_PATH = ROOT / "results" / "answer_eval_runs" / "parameter_boost_all_queries.csv"
TOP_K = 13
CANDIDATE_K = 20

# Small synonym map for common PostgreSQL libpq connection parameter wording.
# Parameter names themselves are discovered from the processed-document sections.
ALIASES: dict[str, tuple[str, ...]] = {
    "host": ("hostname", "host name", "server host", "server hostname"),
    "port": ("port number",),
    "dbname": ("database", "database name", "db name", "database name parameter"),
    "user": ("username", "user name", "database user"),
}
POSTGRES_CONTEXT = re.compile(
    r"\b(postgresql|postgres|libpq)\b|connection\s+(?:parameters?|strings?|settings?)",
    re.IGNORECASE,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_metadata(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("metadata", {})
    return value if isinstance(value, dict) else {}


def get_section(item: dict[str, Any]) -> str:
    return str(item.get("section") or get_metadata(item).get("section") or "")


def get_chunk_id(item: dict[str, Any]) -> str:
    return str(item.get("chunk_id") or get_metadata(item).get("chunk_id") or "")


def get_question(record: dict[str, Any]) -> str:
    for key in ("query", "question", "text", "user_query", "prompt"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"No query/question text field found in record: {record}")


def get_query_id(record: dict[str, Any], index: int) -> str:
    for key in ("query_id", "id", "qid", "question_id"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return f"q{index:02d}"


def parameter_chunks(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        section = get_section(chunk).strip()
        match = re.search(r"parameter key words\s*>\s*([^>]+)\s*$", section, re.IGNORECASE)
        if match:
            parameter = match.group(1).strip().lower()
            found.setdefault(parameter, chunk)
    return found


def matched_parameters(question: str, available: dict[str, dict[str, Any]]) -> list[tuple[int, str]]:
    # Avoid boosting generic words such as "user" in unrelated questions.
    if not POSTGRES_CONTEXT.search(question):
        # Explicit parameter names can still establish the domain.
        known = [name for name in available if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", question, re.I)]
        if not known:
            return []

    matches: list[tuple[int, str]] = []
    for parameter in available:
        terms = (parameter.replace("_", " "),) + ALIASES.get(parameter, ())
        positions: list[int] = []
        for term in terms:
            pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
            match = pattern.search(question)
            if match:
                positions.append(match.start())
        if positions:
            matches.append((min(positions), parameter))

    # A synonym like "database" should locate libpq's `dbname` definition.
    # Deduplicate by parameter and order by appearance in the user's question.
    return sorted(matches, key=lambda item: (item[0], item[1]))


def main() -> None:
    for path in (CHUNKS_PATH, QUERIES_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required project file not found: {path}")

    chunks = load_jsonl(CHUNKS_PATH)
    queries = load_jsonl(QUERIES_PATH)
    definitions = parameter_chunks(chunks)
    print(f"Processed chunks: {len(chunks)}")
    print(f"Queries loaded: {len(queries)}")
    print(f"Parameter-definition sections found: {len(definitions)}")

    retriever = HybridRetriever(candidate_k=CANDIDATE_K)
    rows: list[dict[str, Any]] = []
    try:
        for index, record in enumerate(queries, start=1):
            query_id = get_query_id(record, index)
            question = get_question(record)
            baseline = retriever.retrieve(question, top_k=TOP_K)
            baseline_ids = [get_chunk_id(p) for p in baseline]
            baseline_rank = {cid: rank for rank, cid in enumerate(baseline_ids, start=1) if cid}

            matched = matched_parameters(question, definitions)
            injected: list[dict[str, Any]] = []
            seen: set[str] = set()
            for _, parameter in matched:
                chunk = definitions[parameter]
                cid = get_chunk_id(chunk)
                if cid and cid not in seen:
                    injected.append(chunk)
                    seen.add(cid)

            targeted = injected + [p for p in baseline if get_chunk_id(p) not in seen]
            targeted = targeted[:TOP_K]
            targeted_rank = {
                get_chunk_id(p): rank
                for rank, p in enumerate(targeted, start=1)
                if get_chunk_id(p)
            }

            if matched:
                for _, parameter in matched:
                    chunk = definitions[parameter]
                    cid = get_chunk_id(chunk)
                    rows.append({
                        "query_id": query_id,
                        "query": question,
                        "matched_parameter": parameter,
                        "section": get_section(chunk),
                        "baseline_rank_top13": baseline_rank.get(cid, ""),
                        "parameter_boost_rank_top13": targeted_rank.get(cid, ""),
                        "baseline_in_top13": cid in baseline_rank,
                        "boosted_in_top13": cid in targeted_rank,
                        "baseline_result_count": len(baseline),
                        "boosted_result_count": len(targeted),
                    })
            else:
                rows.append({
                    "query_id": query_id,
                    "query": question,
                    "matched_parameter": "(none; baseline unchanged)",
                    "section": "",
                    "baseline_rank_top13": "",
                    "parameter_boost_rank_top13": "",
                    "baseline_in_top13": "",
                    "boosted_in_top13": "",
                    "baseline_result_count": len(baseline),
                    "boosted_result_count": len(targeted),
                })

            print(f"\n{query_id}: {question}")
            print("  matched parameter definitions:", ", ".join(p for _, p in matched) or "none; baseline fallback")
            if matched:
                for _, parameter in matched:
                    cid = get_chunk_id(definitions[parameter])
                    print(
                        f"  {parameter}: baseline rank {baseline_rank.get(cid, 'outside top 13')}; "
                        f"boosted rank {targeted_rank.get(cid, 'outside top 13')}"
                    )

    finally:
        retriever.close()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id", "query", "matched_parameter", "section",
        "baseline_rank_top13", "parameter_boost_rank_top13",
        "baseline_in_top13", "boosted_in_top13",
        "baseline_result_count", "boosted_result_count",
    ]
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved to: {OUTPUT_PATH}")
    print(
        "\nInterpretation: this is a diagnostic experiment, not a benchmark win. "
        "It boosts metadata-matched PostgreSQL parameter definitions only when "
        "parameter terms are detected in a connection-related query. It does not "
        "change the production retriever or calculate relevance-label metrics."
    )


if __name__ == "__main__":
    main()
