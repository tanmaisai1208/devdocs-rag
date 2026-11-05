"""Audit expected-fact similarity across the top-5 and top-13 retrieved passages.

This is an automated screening aid, not an entailment/factual-support evaluator.
It embeds each expected fact and short snippets from retrieved passages using the
project's local sentence-transformer model, then reports best cosine similarities.
No Gemini/API calls are made.

Run from the project root:
    python scripts/audit_fact_coverage.py

Optional paths/model/threshold:
    python scripts/audit_fact_coverage.py --input results/answer_eval_runs/retrieval_only_after_api_tests.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_INPUT = Path("results/answer_eval_runs/retrieval_only_after_api_tests.jsonl")
DEFAULT_REVIEW_CSV = Path("results/answer_eval_runs/top1_evidence_review.csv")
DEFAULT_DETAIL_OUTPUT = Path("results/answer_eval_runs/fact_similarity_audit.csv")
DEFAULT_SUMMARY_OUTPUT = Path("results/answer_eval_runs/fact_similarity_audit_summary.csv")
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60)


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


def read_reviews(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            str(row.get("query_id", "")).strip(): row
            for row in csv.DictReader(handle)
            if str(row.get("query_id", "")).strip()
        }


def safe_int(value: Any) -> int | str:
    text = str(value or "").strip()
    return int(text) if text.isdigit() else ""


def split_sentences(text: str) -> list[str]:
    """Split extracted documentation into usable sentence-sized text snippets."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    # Sentence boundaries are intentionally conservative; technical abbreviations may remain joined.
    pieces = re.split(r"(?<=[.!?])\s+(?=(?:[A-Z0-9`\"'(]|When\b|If\b|The\b|This\b|It\b|For\b|An\b|A\b))", text)
    return [p.strip() for p in pieces if p.strip()]


def make_snippets(content: str, max_chars: int = 520) -> list[str]:
    """Create overlapping one-, two-, and three-sentence windows for comparison."""
    sentences = split_sentences(content)
    if not sentences and str(content or "").strip():
        sentences = [re.sub(r"\s+", " ", str(content)).strip()]
    snippets: list[str] = []
    seen: set[str] = set()
    for window_size in (1, 2, 3):
        for start in range(len(sentences)):
            window = " ".join(sentences[start : start + window_size]).strip()
            if not window:
                continue
            # Keep unusually long extracted blocks bounded for predictable model input.
            if len(window) > max_chars:
                window = window[:max_chars].rsplit(" ", 1)[0]
            key = window.lower()
            if key not in seen:
                seen.add(key)
                snippets.append(window)
    return snippets


def get_passage_rank(passage: dict[str, Any], fallback: int) -> int:
    value = passage.get("hybrid_rank")
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def best_match_for_fact(
    fact_vec: np.ndarray,
    snippets: list[dict[str, Any]],
    vectors: np.ndarray,
    max_rank: int,
) -> dict[str, Any]:
    eligible = [i for i, item in enumerate(snippets) if item["rank"] <= max_rank]
    if not eligible:
        return {"score": "", "rank": "", "source": "", "section": "", "snippet": ""}
    scores = vectors[eligible] @ fact_vec
    best_local = int(np.argmax(scores))
    best_idx = eligible[best_local]
    item = snippets[best_idx]
    return {
        "score": float(scores[best_local]),
        "rank": item["rank"],
        "source": item["source"],
        "section": item["section"],
        "snippet": item["snippet"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--detail-output", type=Path, default=DEFAULT_DETAIL_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input JSONL not found: {args.input}")

    records = read_jsonl(args.input)
    if not records:
        raise SystemExit(f"No records found in {args.input}")
    reviews = read_reviews(args.review_csv)

    jobs: list[dict[str, Any]] = []
    all_snippet_texts: list[str] = []
    all_fact_texts: list[str] = []
    snippets_by_query: dict[str, list[dict[str, Any]]] = {}

    for record in records:
        # The out-of-scope records deliberately have no expected facts.
        if record.get("should_abstain") is not False:
            continue
        query_id = str(record.get("query_id", "")).strip()
        facts = record.get("expected_facts") or []
        passages = record.get("retrieved_passages") or []
        snippets: list[dict[str, Any]] = []
        for fallback_rank, passage in enumerate(passages, start=1):
            rank = get_passage_rank(passage, fallback_rank)
            if rank > 13:
                continue
            for snippet in make_snippets(passage.get("content", "")):
                if len(snippet.split()) < 3:
                    continue
                snippets.append({
                    "query_id": query_id,
                    "rank": rank,
                    "source": str(passage.get("source", "")),
                    "section": str(passage.get("section", "")),
                    "snippet": snippet,
                })
                all_snippet_texts.append(snippet)
        snippets_by_query[query_id] = snippets
        for fact_index, fact in enumerate(facts, start=1):
            fact_text = str(fact).strip()
            if not fact_text:
                continue
            all_fact_texts.append(fact_text)
            jobs.append({
                "record": record,
                "query_id": query_id,
                "fact_index": fact_index,
                "fact": fact_text,
                "fact_vector_index": len(all_fact_texts) - 1,
            })

    if not jobs or not all_snippet_texts:
        raise SystemExit("No answerable facts or retrieved passage text found to compare.")

    print(f"Answerable queries: {len(snippets_by_query)}")
    print(f"Expected facts: {len(all_fact_texts)}")
    print(f"Passage snippets: {len(all_snippet_texts)}")
    print(f"Loading local embedding model: {args.model}")
    model = SentenceTransformer(args.model)
    fact_vectors = model.encode(
        all_fact_texts, normalize_embeddings=True, batch_size=64,
        convert_to_numpy=True, show_progress_bar=False,
    )
    snippet_vectors = model.encode(
        all_snippet_texts, normalize_embeddings=True, batch_size=64,
        convert_to_numpy=True, show_progress_bar=True,
    )

    # Map each per-query snippet to its global vector row.
    snippet_cursor = 0
    for query_id, snippets in snippets_by_query.items():
        for item in snippets:
            item["vector_index"] = snippet_cursor
            snippet_cursor += 1
    # The global list was built in the same record/snippet order as the map above.
    assert snippet_cursor == len(all_snippet_texts)

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    fact_scores_by_query: dict[str, list[dict[str, Any]]] = {}

    for job in jobs:
        record = job["record"]
        query_id = job["query_id"]
        query_snippets = snippets_by_query.get(query_id, [])
        if query_snippets:
            local_vectors = snippet_vectors[[item["vector_index"] for item in query_snippets]]
        else:
            local_vectors = np.empty((0, fact_vectors.shape[1]), dtype=np.float32)
        fact_vec = fact_vectors[job["fact_vector_index"]]
        top5 = best_match_for_fact(fact_vec, query_snippets, local_vectors, 5)
        top13 = best_match_for_fact(fact_vec, query_snippets, local_vectors, 13)
        detail = {
            "query_id": query_id,
            "query": record.get("query", ""),
            "fact_index": job["fact_index"],
            "expected_fact": job["fact"],
            "expected_fact_count": len(record.get("expected_facts") or []),
            "top5_best_similarity": top5["score"],
            "top5_best_passage_rank": top5["rank"],
            "top5_best_source": top5["source"],
            "top5_best_section": top5["section"],
            "top5_best_snippet": top5["snippet"],
            "top13_best_similarity": top13["score"],
            "top13_best_passage_rank": top13["rank"],
            "top13_best_source": top13["source"],
            "top13_best_section": top13["section"],
            "top13_best_snippet": top13["snippet"],
        }
        for threshold in THRESHOLDS:
            suffix = f"{threshold:.2f}".replace(".", "_")
            detail[f"top5_candidate_at_{suffix}"] = (
                "yes" if isinstance(top5["score"], (int, float)) and top5["score"] >= threshold else "no"
            )
            detail[f"top13_candidate_at_{suffix}"] = (
                "yes" if isinstance(top13["score"], (int, float)) and top13["score"] >= threshold else "no"
            )
        detail_rows.append(detail)
        fact_scores_by_query.setdefault(query_id, []).append({"top5": top5, "top13": top13})

    for record in records:
        if record.get("should_abstain") is not False:
            continue
        query_id = str(record.get("query_id", "")).strip()
        facts = record.get("expected_facts") or []
        facts_scores = fact_scores_by_query.get(query_id, [])
        review = reviews.get(query_id, {})
        top1_supported = safe_int(review.get("review_number_of_facts_supported", ""))
        human_all = str(review.get("review_directly_supports_all_facts", "")).strip().lower()
        if human_all not in {"yes", "no"}:
            human_all = ""
        summary = {
            "query_id": query_id,
            "query": record.get("query", ""),
            "expected_fact_count": len(facts),
            "top1_human_review_supported_fact_count": top1_supported,
            "top1_human_review_supports_all_facts": human_all,
        }
        for k_name in ("top5", "top13"):
            scores = [item[k_name]["score"] for item in facts_scores]
            for threshold in THRESHOLDS:
                suffix = f"{threshold:.2f}".replace(".", "_")
                num = sum(isinstance(score, (int, float)) and score >= threshold for score in scores)
                summary[f"{k_name}_candidate_facts_at_{suffix}"] = num
                summary[f"{k_name}_candidate_fraction_at_{suffix}"] = (
                    round(num / len(facts), 3) if facts else ""
                )
        summary_rows.append(summary)

    write_csv(args.detail_output, detail_rows)
    write_csv(args.summary_output, summary_rows)

    print(f"\nFact-level details saved to: {args.detail_output.resolve()}")
    print(f"Per-query summary saved to: {args.summary_output.resolve()}")
    print("\nSimilarity candidates across all expected facts (NOT verified factual support):")
    total_facts = len(detail_rows)
    for threshold in THRESHOLDS:
        suffix = f"{threshold:.2f}".replace(".", "_")
        top5_count = sum(row[f"top5_candidate_at_{suffix}"] == "yes" for row in detail_rows)
        top13_count = sum(row[f"top13_candidate_at_{suffix}"] == "yes" for row in detail_rows)
        print(
            f"  threshold {threshold:.2f}: top-5 {top5_count}/{total_facts}; "
            f"top-13 {top13_count}/{total_facts}"
        )
    print("\nInterpret scores as candidate evidence matches only. Similarity does not prove that a passage entails a fact.")
    print("No answer-generation API was called; only the local embedding model was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
