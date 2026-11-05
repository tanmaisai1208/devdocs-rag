
import csv
import json
import re
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

QUERIES_PATH = PROJECT_ROOT / "evaluation" / "dataset" / "queries_pilot_v2_dev.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "dataset" / "pilot_v2" / "candidate_labels_dev.csv"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"

COLLECTION_NAME = "engineering_docs_v1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CANDIDATE_K = 10
RRF_K = 60


# --------------------------------------------------
# Loading data
# --------------------------------------------------

def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}"
                ) from exc

    return records


def tokenize(text):
    """Preserve technical identifiers such as pool_size and __cause__."""
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def validate_data(queries, chunks):
    if not queries:
        raise ValueError("No evaluation queries were found.")

    if not chunks:
        raise ValueError("No document chunks were found.")

    query_ids = [query.get("query_id") for query in queries]

    if any(not query_id for query_id in query_ids):
        raise ValueError("Every query must have a query_id.")

    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Duplicate query_id values were found.")

    for query in queries:
        if not query.get("query", "").strip():
            raise ValueError(
                f"Empty query found: {query.get('query_id')}"
            )

    for chunk in chunks:
        if not chunk.get("chunk_id") or not chunk.get("content", "").strip():
            raise ValueError("A chunk is missing chunk_id or content.")


# --------------------------------------------------
# Retrieval helpers
# --------------------------------------------------

def get_ranked_results(results, score_field):
    """Convert results into a chunk_id -> rank/score mapping."""
    ranked = {}

    for rank, result in enumerate(results, start=1):
        chunk_id = result["chunk_id"]

        ranked[chunk_id] = {
            "rank": rank,
            "score": result[score_field],
        }

    return ranked


def reciprocal_rank_fusion(dense_results, bm25_results):
    """
    Combine rankings using RRF.

    RRF score = sum(1 / (RRF_K + rank))
    """
    fused_scores = {}

    for results in (dense_results, bm25_results):
        for rank, result in enumerate(results, start=1):
            chunk_id = result["chunk_id"]

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + 1.0 / (RRF_K + rank)
            )

    return sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )


# --------------------------------------------------
# Main evaluation candidate export
# --------------------------------------------------

def main():
    queries = load_jsonl(QUERIES_PATH)
    chunks = load_jsonl(CHUNKS_PATH)

    validate_data(queries, chunks)

    print(f"Loaded {len(queries)} queries.")
    print(f"Loaded {len(chunks)} document chunks.")

    # Build the BM25 index once.
    tokenized_documents = [
        tokenize(chunk["content"]) for chunk in chunks
    ]
    bm25 = BM25Okapi(tokenized_documents)

    # Map chunk IDs to their original document records.
    chunks_by_id = {
        chunk["chunk_id"]: chunk for chunk in chunks
    }

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    client = QdrantClient(path=str(QDRANT_PATH))

    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        indexed_count = collection_info.points_count

        if indexed_count != len(chunks):
            raise ValueError(
                f"Qdrant has {indexed_count} points, but the chunks "
                f"file contains {len(chunks)} records. "
                "Check that both represent the same dataset."
            )

        rows = []

        for query_number, query_record in enumerate(queries, start=1):
            query_id = query_record["query_id"]
            query_text = query_record["query"].strip()

            print(
                f"[{query_number}/{len(queries)}] "
                f"Evaluating {query_id}: {query_text}"
            )

            # 1. Dense retrieval
            query_vector = embeddings.embed_query(query_text)

            dense_points = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=CANDIDATE_K,
                with_payload=True,
            ).points

            dense_results = []

            for point in dense_points:
                payload = point.payload or {}
                chunk_id = payload.get("chunk_id")

                if chunk_id not in chunks_by_id:
                    continue

                dense_results.append({
                    "chunk_id": chunk_id,
                    "dense_score": float(point.score),
                })

            # 2. BM25 retrieval
            query_tokens = tokenize(query_text)
            bm25_scores = bm25.get_scores(query_tokens)

            bm25_indices = sorted(
                range(len(bm25_scores)),
                key=lambda index: bm25_scores[index],
                reverse=True,
            )

            bm25_results = [
                {
                    "chunk_id": chunks[index]["chunk_id"],
                    "bm25_score": float(bm25_scores[index]),
                }
                for index in bm25_indices
                if bm25_scores[index] > 0
            ][:CANDIDATE_K]

            # 3. Rank lookup for dense and BM25.
            dense_ranked = get_ranked_results(
                dense_results, "dense_score"
            )
            bm25_ranked = get_ranked_results(
                bm25_results, "bm25_score"
            )

            # 4. Hybrid retrieval using RRF.
            fused_results = reciprocal_rank_fusion(
                dense_results, bm25_results
            )

            hybrid_ranked = {
                chunk_id: {
                    "rank": rank,
                    "score": score,
                }
                for rank, (chunk_id, score) in enumerate(
                    fused_results, start=1
                )
                if rank <= CANDIDATE_K
            }

            # Export the union of dense top 10, BM25 top 10,
            # and hybrid top 10. A chunk appears only once per query.
            candidate_ids = set(dense_ranked)
            candidate_ids.update(bm25_ranked)
            candidate_ids.update(hybrid_ranked)

            for chunk_id in candidate_ids:
                chunk = chunks_by_id[chunk_id]
                metadata = chunk.get("metadata", {})

                dense_info = dense_ranked.get(chunk_id, {})
                bm25_info = bm25_ranked.get(chunk_id, {})
                hybrid_info = hybrid_ranked.get(chunk_id, {})

                rows.append({
                    "query_id": query_id,
                    "query": query_text,
                    "chunk_id": chunk_id,
                    "content": chunk["content"],
                    "source": metadata.get("source", ""),
                    "title": metadata.get("title", ""),
                    "section": metadata.get("section", ""),
                    "dense_rank": dense_info.get("rank", ""),
                    "dense_score": dense_info.get("score", ""),
                    "bm25_rank": bm25_info.get("rank", ""),
                    "bm25_score": bm25_info.get("score", ""),
                    "hybrid_rank": hybrid_info.get("rank", ""),
                    "rrf_score": hybrid_info.get("score", ""),
                    "relevant": "",
                })

    finally:
        client.close()

    # Stable, easy-to-review ordering in the CSV.
    rows.sort(
        key=lambda row: (
            row["query_id"],
            min(
                row["dense_rank"] or 999,
                row["bm25_rank"] or 999,
                row["hybrid_rank"] or 999,
            ),
        )
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "query_id",
        "query",
        "chunk_id",
        "content",
        "source",
        "title",
        "section",
        "dense_rank",
        "dense_score",
        "bm25_rank",
        "bm25_score",
        "hybrid_rank",
        "rrf_score",
        "relevant",
    ]

    with OUTPUT_PATH.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n--- CANDIDATE EXPORT COMPLETE ---")
    print(f"Queries processed: {len(queries)}")
    print(f"Candidate rows exported: {len(rows)}")
    print(f"Output file: {OUTPUT_PATH}")
    print("\nThe 'relevant' column is intentionally blank.")
    print("Review the candidates and label them manually.")


if __name__ == "__main__":
    main()
