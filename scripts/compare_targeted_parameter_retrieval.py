import csv
import json
from pathlib import Path

from app.retrieval.hybrid_retriever import HybridRetriever

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
OUTPUT_PATH = ROOT / "results" / "answer_eval_runs" / "a08_targeted_retrieval_comparison.csv"

QUERY = (
    "Which PostgreSQL connection parameters specify the server hostname, "
    "port, database, and user?"
)
PARAMETERS = ("host", "port", "dbname", "user")


def load_chunks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def metadata_of(chunk: dict) -> dict:
    metadata = chunk.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def section_of(chunk: dict) -> str:
    return str(chunk.get("section") or metadata_of(chunk).get("section") or "")


def chunk_id_of(chunk: dict) -> str:
    return str(chunk.get("chunk_id") or metadata_of(chunk).get("chunk_id") or "")


def find_parameter_chunks(chunks: list[dict]) -> dict[str, dict]:
    """Find exact parameter-definition sections in the processed corpus."""
    found = {}
    for parameter in PARAMETERS:
        matches = [
            chunk for chunk in chunks
            if section_of(chunk).rstrip().lower().endswith(
                f"parameter key words > {parameter}"
            )
        ]
        if matches:
            found[parameter] = matches[0]
    return found


def main() -> None:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Processed chunks not found at {CHUNKS_PATH}. "
            "Run this script from the project's scripts folder after placing it there."
        )

    chunks = load_chunks(CHUNKS_PATH)
    direct_chunks = find_parameter_chunks(chunks)

    print(f"Processed chunks loaded: {len(chunks)}")
    print(f"Exact parameter sections found: {len(direct_chunks)}/{len(PARAMETERS)}")

    for parameter in PARAMETERS:
        if parameter in direct_chunks:
            print(f"  {parameter}: {section_of(direct_chunks[parameter])}")
        else:
            print(f"  {parameter}: NOT FOUND")

    retriever = HybridRetriever(candidate_k=20)
    try:
        baseline = retriever.retrieve(QUERY, top_k=13)
    finally:
        retriever.close()

    baseline_rank = {
        chunk_id_of(passage): rank
        for rank, passage in enumerate(baseline, start=1)
    }

    # Experimental strategy: prepend exact, metadata-matched parameter chunks,
    # then retain the baseline hybrid results in their original order.
    targeted_order = []
    seen = set()

    for parameter in PARAMETERS:
        chunk = direct_chunks.get(parameter)
        if chunk is None:
            continue
        chunk_id = chunk_id_of(chunk)
        if chunk_id and chunk_id not in seen:
            targeted_order.append((parameter, chunk))
            seen.add(chunk_id)

    for passage in baseline:
        chunk_id = chunk_id_of(passage)
        if chunk_id and chunk_id not in seen:
            targeted_order.append(("hybrid", passage))
            seen.add(chunk_id)

    targeted_rank = {
        chunk_id_of(passage): rank
        for rank, (_, passage) in enumerate(targeted_order, start=1)
    }

    rows = []
    print("\nPARAMETER CHUNK RANK COMPARISON")
    print("parameter | baseline hybrid rank | targeted rank | found")
    for parameter in PARAMETERS:
        chunk = direct_chunks.get(parameter)
        chunk_id = chunk_id_of(chunk) if chunk else ""
        b_rank = baseline_rank.get(chunk_id) if chunk_id else None
        t_rank = targeted_rank.get(chunk_id) if chunk_id else None
        print(
            f"{parameter:9} | {str(b_rank or 'not in top 13'):20} | "
            f"{str(t_rank or 'not found'):13} | {'yes' if chunk else 'no'}"
        )
        rows.append({
            "query_id": "a08",
            "query": QUERY,
            "parameter": parameter,
            "chunk_id": chunk_id,
            "section": section_of(chunk) if chunk else "",
            "baseline_hybrid_rank_top13": b_rank or "",
            "targeted_rank": t_rank or "",
            "exact_section_found": bool(chunk),
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved to: {OUTPUT_PATH}")
    print(
        "\nNote: this is a targeted-lookup diagnostic, not proof of an overall "
        "retrieval improvement. The parameter chunks are deliberately placed "
        "before baseline results; evaluate all answerable queries before adoption."
    )


if __name__ == "__main__":
    main()
