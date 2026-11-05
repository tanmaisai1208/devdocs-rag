
from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = CrossEncoder(
            model_name,
            device="cpu",
        )

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 10,
    ) -> list[dict]:
        """Score and reorder query-passage pairs."""
        if not candidates:
            return []

        pairs = [
            (query, candidate["content"])
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs,
            batch_size=16,
            show_progress_bar=False,
        )

        results = []

        for candidate, score in zip(candidates, scores):
            result = candidate.copy()
            result["rerank_score"] = float(score)
            results.append(result)

        results.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return results[:top_k]
