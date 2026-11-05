"""Fairly compare live hybrid retrieval with the experimental parameter boost.

Run from the project root:
    python scripts/benchmark_parameter_boost.py

This script does not modify production retrieval code and does not call Gemini.
If the current retrieval output contains chunks absent from the existing relevance
labels, it writes them to parameter_boost_extra_labels.csv and pauses for manual
labeling. Fill relevant with 1 (directly useful evidence) or 0 (not directly useful),
then rerun this script. Do not guess labels; inspect each passage against its query.
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.retrieval.hybrid_retriever import HybridRetriever

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
LABELS_PATH = ROOT / "evaluation" / "dataset" / "candidate_labels_with_blended_reranker.csv"
OUT_DIR = ROOT / "results" / "answer_eval_runs"
EXTRA_LABELS_PATH = OUT_DIR / "parameter_boost_extra_labels.csv"
PER_QUERY_PATH = OUT_DIR / "parameter_boost_benchmark_per_query.csv"
SUMMARY_PATH = OUT_DIR / "parameter_boost_benchmark_summary.csv"
TOP_K = 13
CANDIDATE_K = 20
K_VALUES = (5, 10)

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
EXTRA_FIELDS = [
    "query_id", "query", "chunk_id", "content", "source", "title", "section",
    "relevant", "baseline_hybrid_rank_top13", "parameter_boost_rank_top13",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metadata_of(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("metadata", {})
    return value if isinstance(value, dict) else {}


def get_value(item: dict[str, Any], key: str) -> str:
    value = item.get(key)
    if value is None or value == "":
        value = metadata_of(item).get(key, "")
    return str(value or "")


def chunk_id_of(item: dict[str, Any]) -> str:
    return get_value(item, "chunk_id")


def section_of(item: dict[str, Any]) -> str:
    return get_value(item, "section")


def parameter_chunks(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        match = re.search(
            r"parameter key words\s*>\s*([^>]+)\s*$",
            section_of(chunk).strip(),
            re.IGNORECASE,
        )
        if match:
            found.setdefault(match.group(1).strip().lower(), chunk)
    return found


def matched_parameters(question: str, available: dict[str, dict[str, Any]]) -> list[tuple[int, str]]:
    if not POSTGRES_CONTEXT.search(question):
        explicit = [
            name for name in available
            if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", question, re.IGNORECASE)
        ]
        if not explicit:
            return []

    matches: list[tuple[int, str]] = []
    for parameter in available:
        terms = (parameter.replace("_", " "),) + ALIASES.get(parameter, ())
        positions: list[int] = []
        for term in terms:
            match = re.search(rf"(?<!\w){re.escape(term)}(?!\w)", question, re.IGNORECASE)
            if match:
                positions.append(match.start())
        if positions:
            matches.append((min(positions), parameter))
    return sorted(matches, key=lambda x: (x[0], x[1]))


def passage_text(item: dict[str, Any]) -> str:
    return str(item.get("content") or item.get("page_content") or "")


def build_boosted(baseline: list[dict[str, Any]], matched: list[tuple[int, str]], definitions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    injected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, name in matched:
        chunk = definitions.get(name)
        if chunk is None:
            continue
        cid = chunk_id_of(chunk)
        if cid and cid not in seen:
            injected.append(chunk)
            seen.add(cid)
    combined = injected + [p for p in baseline if chunk_id_of(p) not in seen]
    return combined[:TOP_K]


def rank_map(passages: list[dict[str, Any]]) -> dict[str, int]:
    return {
        chunk_id_of(p): rank
        for rank, p in enumerate(passages, start=1)
        if chunk_id_of(p)
    }


def evaluate_query(rows: list[dict[str, Any]], rank_key: str) -> dict[str, float | int | str]:
    total_relevant = sum(int(row["relevant"]) for row in rows)
    if total_relevant == 0:
        raise ValueError(
            f"Query {rows[0]['query_id']} has no relevant labeled chunks; fix labels before evaluating."
        )
    ranked = [
        {"rank": row[rank_key], "relevant": int(row["relevant"])}
        for row in rows if row[rank_key] is not None
    ]
    ranked.sort(key=lambda x: x["rank"])
    relevant_ranks = [x["rank"] for x in ranked if x["relevant"] == 1]
    metrics: dict[str, float | int | str] = {
        "relevant_labeled_candidates": total_relevant,
        "first_relevant_rank": min(relevant_ranks) if relevant_ranks else "",
        "mrr": 1.0 / min(relevant_ranks) if relevant_ranks else 0.0,
    }
    for k in K_VALUES:
        found = sum(x["relevant"] == 1 and x["rank"] <= k for x in ranked)
        metrics[f"recall_at_{k}"] = found / total_relevant
        dcg = sum(
            x["relevant"] / math.log2(x["rank"] + 1)
            for x in ranked if x["rank"] <= k
        )
        ideal_count = min(total_relevant, k)
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        metrics[f"ndcg_at_{k}"] = dcg / ideal_dcg if ideal_dcg else 0.0
    return metrics


def normalized_label(value: str, location: str) -> int:
    value = str(value or "").strip()
    if value not in {"0", "1"}:
        raise ValueError(f"Missing/invalid relevance label {value!r} at {location}; expected 0 or 1.")
    return int(value)


def main() -> None:
    for path in (CHUNKS_PATH, LABELS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    chunks = load_jsonl(CHUNKS_PATH)
    label_rows = load_csv(LABELS_PATH)
    required = {"query_id", "query", "chunk_id", "relevant"}
    if not label_rows or not required.issubset(label_rows[0]):
        raise ValueError(f"Label file must contain columns: {sorted(required)}")

    primary: dict[tuple[str, str], dict[str, Any]] = {}
    queries: dict[str, str] = {}
    for line, row in enumerate(label_rows, start=2):
        qid = row["query_id"].strip()
        cid = row["chunk_id"].strip()
        question = row["query"].strip()
        if not qid or not cid or not question:
            raise ValueError(f"Empty query_id/query/chunk_id at label CSV line {line}.")
        if qid in queries and queries[qid] != question:
            raise ValueError(f"Inconsistent query text for query_id {qid}.")
        queries[qid] = question
        key = (qid, cid)
        if key in primary:
            raise ValueError(f"Duplicate query/chunk label in primary label file: {key}")
        record = dict(row)
        record["relevant"] = normalized_label(row["relevant"], f"primary CSV line {line}")
        primary[key] = record

    old_extra: dict[tuple[str, str], dict[str, Any]] = {}
    if EXTRA_LABELS_PATH.exists():
        for line, row in enumerate(load_csv(EXTRA_LABELS_PATH), start=2):
            qid, cid = row.get("query_id", "").strip(), row.get("chunk_id", "").strip()
            if not qid or not cid:
                continue
            # Preserve user-entered relevant labels, including blanks awaiting review.
            old_extra[(qid, cid)] = {field: row.get(field, "") for field in EXTRA_FIELDS}

    definitions = parameter_chunks(chunks)
    print(f"Primary relevance labels: {len(primary)} query/chunk pairs")
    print(f"Queries: {len(queries)} | processed chunks: {len(chunks)}")
    print(f"Parameter definitions found: {len(definitions)}")
    print(f"Existing supplemental labels: {len(old_extra)}")
    print("Running current hybrid retrieval (no Gemini calls)...")

    retriever = HybridRetriever(candidate_k=CANDIDATE_K)
    results_by_query: dict[str, dict[str, Any]] = {}
    retrieved_passage_info: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        for qid, question in sorted(queries.items()):
            baseline = retriever.retrieve(question, top_k=TOP_K)
            matched = matched_parameters(question, definitions)
            boosted = build_boosted(baseline, matched, definitions)
            base_ranks, boost_ranks = rank_map(baseline), rank_map(boosted)
            results_by_query[qid] = {
                "query": question,
                "matched_parameters": [name for _, name in matched],
                "baseline_ranks": base_ranks,
                "boosted_ranks": boost_ranks,
                "baseline_count": len(baseline),
                "boosted_count": len(boosted),
            }
            for passage in baseline:
                cid = chunk_id_of(passage)
                if cid:
                    retrieved_passage_info[(qid, cid)] = passage
            for passage in boosted:
                cid = chunk_id_of(passage)
                if cid:
                    retrieved_passage_info[(qid, cid)] = passage
            print(f"  {qid}: matched {', '.join(name for _, name in matched) or 'none'}")
    finally:
        retriever.close()

    # Prepare supplemental human labels for any top-13 candidate absent from the
    # original judged pool. Unknown chunks must not be silently treated as 0.
    extra_candidates: dict[tuple[str, str], dict[str, Any]] = dict(old_extra)
    for key, passage in retrieved_passage_info.items():
        if key in primary or key in extra_candidates:
            continue
        qid, cid = key
        res = results_by_query[qid]
        meta = metadata_of(passage)
        extra_candidates[key] = {
            "query_id": qid,
            "query": res["query"],
            "chunk_id": cid,
            "content": passage_text(passage),
            "source": get_value(passage, "source"),
            "title": get_value(passage, "title"),
            "section": section_of(passage),
            "relevant": "",
            "baseline_hybrid_rank_top13": res["baseline_ranks"].get(cid, ""),
            "parameter_boost_rank_top13": res["boosted_ranks"].get(cid, ""),
        }

    # Refresh ranking columns while preserving labels manually entered in earlier runs.
    for key, row in extra_candidates.items():
        qid, cid = key
        res = results_by_query.get(qid)
        if res:
            row["query"] = res["query"]
            row["baseline_hybrid_rank_top13"] = res["baseline_ranks"].get(cid, "")
            row["parameter_boost_rank_top13"] = res["boosted_ranks"].get(cid, "")
        passage = retrieved_passage_info.get(key)
        if passage:
            row["content"] = row.get("content") or passage_text(passage)
            row["source"] = row.get("source") or get_value(passage, "source")
            row["title"] = row.get("title") or get_value(passage, "title")
            row["section"] = row.get("section") or section_of(passage)
        # Keep manual labels only if the field is exactly 0 or 1; otherwise pending.
        row["relevant"] = str(row.get("relevant", "")).strip()

    extra_rows = [extra_candidates[k] for k in sorted(extra_candidates)]
    write_csv(EXTRA_LABELS_PATH, extra_rows, EXTRA_FIELDS)
    pending = [row for row in extra_rows if str(row.get("relevant", "")).strip() not in {"0", "1"}]
    if pending:
        print(f"\nNEEDS MANUAL LABELING: {len(pending)} supplemental query/chunk pairs")
        print(f"Review this file: {EXTRA_LABELS_PATH}")
        print("For each row, read the query and passage. Set relevant=1 only if the passage directly helps answer that query; otherwise set relevant=0.")
        print("Save the CSV and rerun this same command. The script preserves labels you enter and will calculate metrics once all rows are labeled.")
        print("No benchmark metrics were calculated because unjudged passages must not be treated as irrelevant.")
        return

    # Merge the primary and supplemental human labels, then attach the live result ranks.
    all_labels: dict[tuple[str, str], dict[str, Any]] = dict(primary)
    for row in extra_rows:
        key = (row["query_id"], row["chunk_id"])
        if key in all_labels:
            raise ValueError(f"Supplemental label duplicates primary label: {key}")
        all_labels[key] = {**row, "relevant": normalized_label(row["relevant"], f"supplemental label {key}")}

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (qid, cid), row in all_labels.items():
        result = results_by_query.get(qid)
        if result is None:
            raise ValueError(f"Label contains query ID {qid} that has no query text in the primary file.")
        copy = dict(row)
        copy["query_id"] = qid
        copy["query"] = result["query"]
        copy["relevant"] = int(row["relevant"])
        copy["baseline_live_rank"] = result["baseline_ranks"].get(cid)
        copy["parameter_boost_rank"] = result["boosted_ranks"].get(cid)
        grouped[qid].append(copy)

    per_query: list[dict[str, Any]] = []
    summary_metrics: dict[str, dict[str, list[float]]] = {
        "hybrid_baseline": defaultdict(list),
        "parameter_boost": defaultdict(list),
    }
    metric_names = ["mrr", "recall_at_5", "recall_at_10", "ndcg_at_5", "ndcg_at_10"]
    for qid in sorted(grouped):
        rows = grouped[qid]
        for variant, rank_key in (
            ("hybrid_baseline", "baseline_live_rank"),
            ("parameter_boost", "parameter_boost_rank"),
        ):
            metrics = evaluate_query(rows, rank_key)
            result_row = {
                "query_id": qid,
                "query": results_by_query[qid]["query"],
                "variant": variant,
                "matched_parameters": ";".join(results_by_query[qid]["matched_parameters"]),
                **metrics,
            }
            per_query.append(result_row)
            for name in metric_names:
                summary_metrics[variant][name].append(float(metrics[name]))

    summary: list[dict[str, Any]] = []
    averages: dict[str, dict[str, float]] = {}
    for variant in ("hybrid_baseline", "parameter_boost"):
        averages[variant] = {
            name: sum(values) / len(values) if values else 0.0
            for name, values in summary_metrics[variant].items()
        }
        summary.append({
            "variant": variant,
            "queries_evaluated": len(grouped),
            "labeled_query_chunk_pairs": len(all_labels),
            **averages[variant],
        })
    summary.append({
        "variant": "delta_boost_minus_baseline",
        "queries_evaluated": len(grouped),
        "labeled_query_chunk_pairs": len(all_labels),
        **{name: averages["parameter_boost"][name] - averages["hybrid_baseline"][name] for name in metric_names},
    })

    write_csv(PER_QUERY_PATH, per_query, [
        "query_id", "query", "variant", "matched_parameters", "relevant_labeled_candidates",
        "first_relevant_rank", "mrr", "recall_at_5", "recall_at_10", "ndcg_at_5", "ndcg_at_10",
    ])
    write_csv(SUMMARY_PATH, summary, [
        "variant", "queries_evaluated", "labeled_query_chunk_pairs", "mrr",
        "recall_at_5", "recall_at_10", "ndcg_at_5", "ndcg_at_10",
    ])

    print("\nBENCHMARK SUMMARY (macro-average across queries)")
    print("variant                     MRR     R@5     R@10    nDCG@5  nDCG@10")
    for row in summary:
        print(
            f"{row['variant']:27} "
            f"{row['mrr']:.4f}  {row['recall_at_5']:.4f}  {row['recall_at_10']:.4f}  "
            f"{row['ndcg_at_5']:.4f}  {row['ndcg_at_10']:.4f}"
        )
    print(f"\nPer-query metrics saved: {PER_QUERY_PATH}")
    print(f"Summary metrics saved:   {SUMMARY_PATH}")
    print("Note: this compares a live top-13 baseline with the simulated parameter-boost ordering; production code is unchanged.")


if __name__ == "__main__":
    main()
