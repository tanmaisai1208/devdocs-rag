
from math import isfinite
from typing import Any, Iterable


DEFAULT_DENSE_SCORE_THRESHOLD = 0.47


def assess_evidence_sufficiency(
    passages: Iterable[dict[str, Any]],
    threshold: float = DEFAULT_DENSE_SCORE_THRESHOLD,
) -> dict[str, Any]:
    """
    Estimate evidence sufficiency using the highest dense score
    among the retrieved passages.

    This is a provisional heuristic, not a guarantee that a
    passage contains the facts needed to answer a question.
    """
    if not isfinite(threshold):
        raise ValueError("Threshold must be a finite number.")

    scores = []

    for passage in passages:
        score = passage.get("max_dense_candidate_score")

        # Support older records and existing unit tests.
        if score is None:
            score = passage.get("dense_score")

        if isinstance(score, (int, float)) and isfinite(float(score)):
            scores.append(float(score))

    if not scores:
        return {
            "sufficient": False,
            "max_dense_score": None,
            "threshold": threshold,
            "reason": "missing_dense_scores",
        }

    max_score = max(scores)
    sufficient = max_score >= threshold

    return {
        "sufficient": sufficient,
        "max_dense_score": max_score,
        "threshold": threshold,
        "reason": (
            "score_meets_threshold"
            if sufficient
            else "score_below_threshold"
        ),
    }
