
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
)

QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"

COLLECTION_NAME = "engineering_docs_v1"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CANDIDATE_K = 10
RRF_K = 60


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Tokenize text while preserving identifiers containing underscores."""
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def load_chunks() -> list[dict[str, Any]]:
    """Load processed chunks and expose their metadata consistently."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Processed chunks not found: {CHUNKS_PATH}"
        )

    chunks = []

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            chunk = json.loads(line)
            metadata = chunk.get("metadata", {})

            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Invalid metadata on line {line_number}"
                )

            chunk_id = chunk.get("chunk_id")
            content = chunk.get("content")

            if not chunk_id or not isinstance(content, str):
                raise ValueError(
                    f"Missing chunk_id/content on line {line_number}"
                )

            normalized = {
                "chunk_id": str(chunk_id),
                "content": content,
                "metadata": metadata,
                "source": metadata.get(
                    "source", chunk.get("source", "")
                ),
                "title": metadata.get(
                    "title", chunk.get("title", "")
                ),
                "section": metadata.get(
                    "section", chunk.get("section", "")
                ),
            }

            chunks.append(normalized)

    if not chunks:
        raise ValueError("No processed chunks were loaded.")

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs found in chunks.jsonl.")

    return chunks


# --------------------------------------------------
# Hybrid retriever
# --------------------------------------------------

class HybridRetriever:
    """
    Retrieve passages using dense search + BM25 + RRF fusion.

    The returned passages can be passed directly to AnswerGenerator.
    """

    def __init__(
        self,
        candidate_k: int = CANDIDATE_K,
        rrf_k: int = RRF_K,
    ) -> None:
        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1.")

        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

        print("Loading processed documentation chunks...")

        self.chunks = load_chunks()

        self.chunks_by_id = {
            chunk["chunk_id"]: chunk
            for chunk in self.chunks
        }

        # Used only as a fallback if a Qdrant payload lacks chunk_id.
        self.content_to_id = {}

        for chunk in self.chunks:
            self.content_to_id.setdefault(
                chunk["content"],
                chunk["chunk_id"],
            )

        print("Preparing BM25 index...")

        tokenized_corpus = [
            tokenize(chunk["content"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

        print("Loading embedding model...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        print("Opening local Qdrant collection...")

        self.client = QdrantClient(
            path=str(QDRANT_PATH)
        )

        collection_info = self.client.get_collection(
            collection_name=COLLECTION_NAME
        )

        indexed_count = collection_info.points_count

        if indexed_count != len(self.chunks):
            self.client.close()
            raise RuntimeError(
                "Qdrant index and processed chunks have different "
                f"sizes: {indexed_count} indexed points versus "
                f"{len(self.chunks)} processed chunks. "
                "Check whether the index needs to be rebuilt."
            )

        print(
            f"Retriever ready: {len(self.chunks)} chunks indexed."
        )


    def _dense_search(self, query: str) -> list[tuple[str, float]]:
        """Return chunk IDs and dense similarity scores, ranked highest first."""
        query_vector = self.embeddings.embed_query(query)

        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=self.candidate_k,
            with_payload=True,
        )

        results = []
        seen = set()

        for point in response.points:
            payload = point.payload or {}
            metadata = payload.get("metadata", {})

            if not isinstance(metadata, dict):
                metadata = {}

            chunk_id = (
                payload.get("chunk_id")
                or metadata.get("chunk_id")
            )

            if not chunk_id:
                content = (
                    payload.get("content")
                    or payload.get("page_content")
                )

                if isinstance(content, str):
                    chunk_id = self.content_to_id.get(content)

            if not chunk_id:
                raise RuntimeError(
                    "A Qdrant result could not be mapped to a chunk ID."
                )

            chunk_id = str(chunk_id)

            if chunk_id not in self.chunks_by_id:
                raise RuntimeError(
                    f"Qdrant returned an unknown chunk ID: {chunk_id}"
                )

            if chunk_id not in seen:
                results.append((chunk_id, float(point.score)))
                seen.add(chunk_id)

        return results


    def _bm25_search(self, query: str) -> list[tuple[str, float]]:
        """Return chunk IDs and BM25 scores, ranked highest first."""
        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for index in ranked_indices:
            score = float(scores[index])

            if score <= 0:
                continue

            results.append(
                (self.chunks[index]["chunk_id"], score)
            )

            if len(results) >= self.candidate_k:
                break

        return results


    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve passages using dense search, BM25, and RRF."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query = query.strip()

        dense_results = self._dense_search(query)
        bm25_results = self._bm25_search(query)

        max_dense_candidate_score = max(
            (score for _, score in dense_results),
            default=None,
        )

        scores = defaultdict(float)
        dense_ranks = {}
        bm25_ranks = {}
        dense_scores = {}
        bm25_scores = {}

        for rank, (chunk_id, dense_score) in enumerate(
            dense_results, start=1
        ):
            scores[chunk_id] += 1.0 / (self.rrf_k + rank)
            dense_ranks[chunk_id] = rank
            dense_scores[chunk_id] = dense_score

        for rank, (chunk_id, bm25_score) in enumerate(
            bm25_results, start=1
        ):
            scores[chunk_id] += 1.0 / (self.rrf_k + rank)
            bm25_ranks[chunk_id] = rank
            bm25_scores[chunk_id] = bm25_score

        ranked_ids = sorted(
            scores,
            key=lambda chunk_id: (
                -scores[chunk_id],
                min(
                    dense_ranks.get(
                        chunk_id, self.candidate_k + 1
                    ),
                    bm25_ranks.get(
                        chunk_id, self.candidate_k + 1
                    ),
                ),
                chunk_id,
            ),
        )

        results = []

        for rank, chunk_id in enumerate(
            ranked_ids[:top_k], start=1
        ):
            passage = dict(self.chunks_by_id[chunk_id])

            passage["hybrid_rank"] = rank
            passage["dense_rank"] = dense_ranks.get(chunk_id)
            passage["bm25_rank"] = bm25_ranks.get(chunk_id)
            passage["dense_score"] = dense_scores.get(chunk_id)
            passage["bm25_score"] = bm25_scores.get(chunk_id)
            passage["rrf_score"] = scores[chunk_id]
            passage["max_dense_candidate_score"] = max_dense_candidate_score
            results.append(passage)

        return results


    def close(self) -> None:
        """Release the local Qdrant client."""
        self.client.close()
