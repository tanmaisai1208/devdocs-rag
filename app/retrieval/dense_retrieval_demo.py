
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"

COLLECTION_NAME = "engineering_docs_v1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


def main():
    query = input("\nEnter a troubleshooting question: ").strip()

    if not query:
        print("Please enter a non-empty question.")
        return

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    client = QdrantClient(path=str(QDRANT_PATH))

    try:
        query_vector = embeddings.embed_query(query)

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=TOP_K,
            with_payload=True,
        ).points

        if not results:
            print("No matching chunks found.")
            return

        print(f"\nTop {len(results)} retrieved chunks")

        for rank, point in enumerate(results, start=1):
            payload = point.payload or {}

            print("\n" + "=" * 70)
            print(f"Rank: {rank}")
            print(f"Similarity score: {point.score:.4f}")
            print(f"Source: {payload.get('source', 'Unknown')}")
            print(f"Section: {payload.get('section', 'Unknown')}")
            print(f"Chunk ID: {payload.get('chunk_id', 'Unknown')}")
            print("\nContent:")
            print(payload.get("content", ""))

    finally:
        client.close()


if __name__ == "__main__":
    main()
