
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"

TOP_K = 5


def tokenize(text: str) -> list[str]:
    """Tokenize text while preserving technical identifiers and underscores."""
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def load_chunks() -> list[dict]:
    """Load and validate the processed document chunks."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunk file not found: {CHUNKS_PATH}")

    chunks = []

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from exc

            content = chunk.get("content")
            metadata = chunk.get("metadata", {})

            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"Missing or empty content on line {line_number}"
                )

            chunks.append(
                {
                    "chunk_id": chunk.get("chunk_id", "Unknown"),
                    "content": content,
                    "source": metadata.get("source", "Unknown"),
                    "section": metadata.get("section", "Unknown"),
                }
            )

    if not chunks:
        raise ValueError("No valid chunks found.")

    return chunks


def main():
    chunks = load_chunks()

    # Tokenize every chunk once when building the BM25 index.
    tokenized_corpus = [
        tokenize(chunk["content"]) for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    print(f"Loaded {len(chunks)} chunks.")
    print("BM25 keyword index is ready.")

    while True:
        query = input(
            "\nEnter a troubleshooting question (or type 'exit'): "
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

        scores = bm25.get_scores(query_tokens)

        # Sort chunk indices by descending BM25 score.
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        # BM25 scores are not probabilities. A non-positive
        # maximum means this query found no useful positive match.
        if not ranked_indices or scores[ranked_indices[0]] <= 0:
            print("No meaningful lexical matches found.")
            continue

        top_indices = ranked_indices[:TOP_K]

        print(f"\nTop {len(top_indices)} retrieved chunks")

        for rank, index in enumerate(top_indices, start=1):
            chunk = chunks[index]

            print("\n" + "=" * 70)
            print(f"Rank: {rank}")
            print(f"BM25 score: {scores[index]:.4f}")
            print(f"Source: {chunk['source']}")
            print(f"Section: {chunk['section']}")
            print(f"Chunk ID: {chunk['chunk_id']}")
            print("\nContent:")
            print(chunk["content"])


if __name__ == "__main__":
    main()
