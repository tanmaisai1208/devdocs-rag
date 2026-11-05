
import json
import uuid
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models


# Resolve paths from the project root, not the terminal's current directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"

COLLECTION_NAME = "engineering_docs_v1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384
BATCH_SIZE = 32

# A fixed namespace makes point IDs reproducible between runs.
POINT_NAMESPACE = uuid.UUID("a38c4f25-cf14-4c81-bf3d-bd24d13c72ab")


def load_chunks():
    """Load and validate the existing JSONL chunk records."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {CHUNKS_PATH}\n"
            "Run scripts/ingest_docs.py first."
        )

    chunks = []

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

            content = record.get("content")
            metadata = record.get("metadata")
            chunk_id = record.get("chunk_id")

            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"Missing content on line {line_number}"
                )

            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Invalid metadata on line {line_number}"
                )

            for field in ("source", "title", "section"):
                if not str(metadata.get(field, "")).strip():
                    raise ValueError(
                        f"Missing metadata.{field} on line {line_number}"
                    )

            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(
                    f"Missing chunk_id on line {line_number}"
                )

            chunks.append(record)

    if not chunks:
        raise ValueError("The chunk file contains no usable records.")

    # Prevent accidental collisions in the source identifiers.
    chunk_ids = [record["chunk_id"] for record in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs found in the JSONL file.")

    return chunks


def make_point_id(chunk_id):
    """Convert the existing SHA-256 chunk ID into a stable Qdrant UUID."""
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} validated chunks.")

    QDRANT_PATH.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Confirm that the model produces the expected vector dimensions.
    test_vector = embeddings.embed_query(
        "Test vector dimensions for Qdrant."
    )

    if len(test_vector) != VECTOR_SIZE:
        raise ValueError(
            f"Expected {VECTOR_SIZE} dimensions, "
            f"but received {len(test_vector)}."
        )

    # Local persistent storage: no server or API key required.
    client = QdrantClient(path=str(QDRANT_PATH))

    try:
        if client.collection_exists(COLLECTION_NAME):
            collection_info = client.get_collection(COLLECTION_NAME)
            vector_config = collection_info.config.params.vectors

            if not isinstance(vector_config, dict):
                raise ValueError(
                    "The existing collection uses an unexpected "
                    "vector configuration. Inspect it before proceeding."
                )

            dense_config = vector_config.get("")

            if dense_config is None:
                raise ValueError(
                    "Could not verify the existing collection's "
                    "unnamed dense vector configuration."
                )

            if dense_config.size != VECTOR_SIZE:
                raise ValueError(
                    f"Existing collection dimension is "
                    f"{dense_config.size}; expected {VECTOR_SIZE}. "
                    "Do not delete it automatically."
                )

            if dense_config.distance != models.Distance.COSINE:
                raise ValueError(
                    "Existing collection does not use cosine distance. "
                    "Inspect the collection before proceeding."
                )

            print(
                f"Reusing existing collection: {COLLECTION_NAME}"
            )
        else:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )
            print(f"Created collection: {COLLECTION_NAME}")

        # Upsert in batches. Stable point IDs make reruns idempotent
        # for the same chunk IDs.
        total = len(chunks)

        for start in range(0, total, BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            texts = [record["content"] for record in batch]

            vectors = embeddings.embed_documents(texts)

            points = []

            for record, vector in zip(batch, vectors):
                metadata = record["metadata"]

                payload = {
                    "chunk_id": record["chunk_id"],
                    "content": record["content"],
                    "source": metadata["source"],
                    "title": metadata["title"],
                    "section": metadata["section"],
                    "start_index": metadata.get("start_index", 0),
                }

                points.append(
                    models.PointStruct(
                        id=make_point_id(record["chunk_id"]),
                        vector=vector,
                        payload=payload,
                    )
                )

            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True,
            )

            print(
                f"Indexed {min(start + len(batch), total)}/{total} chunks"
            )

        # Verify the persisted collection count.
        info = client.get_collection(COLLECTION_NAME)

        print("\n--- INDEXING COMPLETE ---")
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Expected points: {total}")
        print(f"Points in collection: {info.points_count}")
        print(f"Vector dimensions: {VECTOR_SIZE}")
        print(f"Distance metric: cosine")
        print(f"Persistent storage: {QDRANT_PATH}")

        if info.points_count != total:
            raise RuntimeError(
                "Point count does not match the input chunk count. "
                "Investigate before testing retrieval."
            )

    finally:
        client.close()


if __name__ == "__main__":
    main()
