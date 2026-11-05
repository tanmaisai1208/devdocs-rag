
import json
import re
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"

COLLECTION_NAME = "engineering_docs_v1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5
CANDIDATE_K = 10
RRF_K = 60


def tokenize(text: str) -> list[str]:
    """Tokenize text while preserving technical identifiers."""
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def load_chunks() -> list[dict]:
    """Load processed chunks using the existing JSONL schema."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunk file not found: {CHUNKS_PATH}")

    chunks = []

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from exc

            content = record.get("content")
            chunk_id = record.get("chunk_id")
            metadata = record.get("metadata", {})

            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"Missing content on line {line_number}"
                )

            if not chunk_id:
                raise ValueError(
                    f"Missing chunk_id on line {line_number}"
                )

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "content": content,
                    "source": metadata.get("source", "Unknown"),
                    "section": metadata.get("section", "Unknown"),
                }
            )

    if not chunks:
        raise ValueError("No chunks were loaded.")

    return chunks


def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
) -> list[dict]:
    """Fuse ranked lists using reciprocal rank fusion."""
    fused = {}

    for results in (dense_results, bm25_results):
        for rank, result in enumerate(results, start=1):
            chunk_id = result["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    **result,
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                }

            fused[chunk_id]["rrf_score"] += (
                1.0 / (RRF_K + rank)
            )

    # Record each retriever's rank for diagnostic output.
    for rank, result in enumerate(dense_results, start=1):
        fused[result["chunk_id"]]["dense_rank"] = rank

    for rank, result in enumerate(bm25_results, start=1):
        fused[result["chunk_id"]]["bm25_rank"] = rank

    return sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )


def main():
    chunks = load_chunks()
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

    print(f"Loaded {len(chunks)} chunks.")

    tokenized_corpus = [
        tokenize(chunk["content"]) for chunk in chunks
    ]
    bm25 = BM25Okapi(tokenized_corpus)

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
                f"Chunk/index count mismatch: {len(chunks)} JSONL "
                f"chunks but {indexed_count} Qdrant points."
            )

        print("Dense and BM25 retrievers are ready.")

        while True:
            query = input(
                "\nEnter a troubleshooting question "
                "(or type 'exit'): "
            ).strip()

            if query.lower() == "exit":
                break

            if not query:
                print("Please enter a non-empty question.")
                continue

            query_tokens = tokenize(query)

            if not query_tokens:
                print("The query contains no searchable tokens.")
                continue

            # 1. Dense retrieval
            query_vector = embeddings.embed_query(query)

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

                chunk = chunks_by_id[chunk_id]
                dense_results.append(
                    {
                        **chunk,
                        "dense_score": point.score,
                    }
                )

            # 2. BM25 keyword retrieval
            bm25_scores = bm25.get_scores(query_tokens)

            bm25_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True,
            )

            bm25_results = [
                {
                    **chunks[index],
                    "bm25_score": float(bm25_scores[index]),
                }
                for index in bm25_indices[:CANDIDATE_K]
                if bm25_scores[index] > 0
            ]

            # 3. Fuse the two ranked lists
            fused_results = reciprocal_rank_fusion(
                dense_results,
                bm25_results,
            )

            if not fused_results:
                print("No matching chunks found.")
                continue

            print(
                f"\nTop {min(TOP_K, len(fused_results))} "
                "hybrid results"
            )

            for rank, result in enumerate(
                fused_results[:TOP_K], start=1
            ):
                dense_rank = result["dense_rank"]
                bm25_rank = result["bm25_rank"]

                print("\n" + "=" * 70)
                print(f"Hybrid rank: {rank}")
                print(f"RRF score: {result['rrf_score']:.6f}")
                print(f"Dense rank: {dense_rank or 'Not in top candidates'}")
                print(f"BM25 rank: {bm25_rank or 'Not in top candidates'}")
                print(f"Source: {result['source']}")
                print(f"Section: {result['section']}")
                print(f"Chunk ID: {result['chunk_id']}")
                print("\nContent:")
                print(result["content"])

    finally:
        client.close()


if __name__ == "__main__":
    main()
