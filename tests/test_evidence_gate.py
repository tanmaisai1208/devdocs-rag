
from app.retrieval.evidence_gate import (
    DEFAULT_DENSE_SCORE_THRESHOLD,
    assess_evidence_sufficiency,
)


def test_answerable_example_passes():
    passages = [{"dense_score": 0.5905}]

    result = assess_evidence_sufficiency(passages)

    assert result["sufficient"] is True
    assert result["reason"] == "score_meets_threshold"


def test_out_of_scope_example_fails():
    passages = [
        {"dense_score": 0.2003},
        {"dense_score": 0.3551},
    ]

    result = assess_evidence_sufficiency(passages)

    assert result["sufficient"] is False
    assert result["reason"] == "score_below_threshold"


def test_missing_scores_fail_safely():
    passages = [{"content": "Some text without a score"}]

    result = assess_evidence_sufficiency(passages)

    assert result["sufficient"] is False
    assert result["reason"] == "missing_dense_scores"


def test_empty_passages_fail_safely():
    result = assess_evidence_sufficiency([])

    assert result["sufficient"] is False
    assert result["max_dense_score"] is None


def test_threshold_boundary_passes():
    passages = [
        {"dense_score": DEFAULT_DENSE_SCORE_THRESHOLD}
    ]

    result = assess_evidence_sufficiency(passages)

    assert result["sufficient"] is True
