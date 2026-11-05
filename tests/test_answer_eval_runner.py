
import json

from app.run_answer_evaluation import (
    build_result_record,
    load_eval_dataset,
    normalize_passage,
    select_queries,
    write_record,
)


def test_load_evaluation_dataset():
    records = load_eval_dataset()

    assert len(records) == 15
    assert sum(not row["should_abstain"] for row in records) == 12
    assert sum(row["should_abstain"] for row in records) == 3


def test_select_queries():
    records = load_eval_dataset()

    selected = select_queries(
        records,
        start_index=12,
        limit=2,
    )

    assert [row["query_id"] for row in selected] == ["a13", "a14"]


def test_normalize_passage():
    passage = {
        "chunk_id": "chunk-123",
        "content": "Example technical documentation.",
        "metadata": {
            "source": "https://example.com/docs",
            "title": "Example Documentation",
            "section": "Example section",
        },
        "hybrid_rank": 1,
        "dense_rank": 2,
        "bm25_rank": 1,
    }

    result = normalize_passage(passage)

    assert result["chunk_id"] == "chunk-123"
    assert result["content"] == "Example technical documentation."
    assert result["source"] == "https://example.com/docs"
    assert result["hybrid_rank"] == 1


def test_result_record_preserves_expected_facts():
    query = load_eval_dataset()[0]

    record = build_result_record(query, "retrieval-only")

    assert record["query_id"] == query["query_id"]
    assert record["expected_facts"] == query["expected_facts"]
    assert record["retrieved_passages"] == []
    assert record["run_status"] == "pending"


def test_jsonl_writer():
    query = load_eval_dataset()[0]
    record = build_result_record(query, "retrieval-only")

    import io

    output = io.StringIO()
    write_record(output, record)

    saved = json.loads(output.getvalue())

    assert saved["query_id"] == query["query_id"]
    assert saved["run_mode"] == "retrieval-only"
