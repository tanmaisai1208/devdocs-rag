from app.generation.answer_generator import (
    AnswerGenerator,
    ABSTENTION_MESSAGE,
)


class FakeResponse:
    def __init__(self, content):
        self.content = content
        self.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }


class FakeLLM:
    """Simulates Gemini without making an API request."""

    def __init__(self, response_text):
        self.response_text = response_text
        self.called = False

    def invoke(self, prompt):
        self.called = True
        return FakeResponse(self.response_text)


def make_generator(response_text):
    # Construct the object without initializing a real Gemini client.
    generator = AnswerGenerator.__new__(AnswerGenerator)
    generator.llm = FakeLLM(response_text)
    generator.model_name = "test-model"
    return generator


def sample_passage(chunk_id="chunk-1"):
    return {
        "chunk_id": chunk_id,
        "content": "The service parameter refers to a service name in pg_service.conf.",
        "source": "https://www.postgresql.org/docs/current/libpq-connect.html",
        "title": "PostgreSQL Connection Documentation",
        "section": "Parameter Key Words > service",
    }


def test_valid_single_citation():
    generator = make_generator(
        "The service parameter uses pg_service.conf [S1]."
    )

    result = generator.generate_answer(
        "How does the service parameter work?",
        [sample_passage()],
    )

    assert result["citation_status"] == "valid"
    assert result["abstained"] is False
    assert len(result["citations"]) == 1
    assert result["citations"][0]["citation_id"] == "S1"


def test_grouped_citations():
    generator = make_generator(
        "The connection uses these settings [S1, S2]."
    )

    passages = [
        sample_passage("chunk-1"),
        {
            **sample_passage("chunk-2"),
            "section": "Connection Strings",
        },
    ]

    result = generator.generate_answer(
        "How are connection parameters specified?",
        passages,
    )

    assert result["citation_status"] == "valid"
    assert [
        citation["citation_id"]
        for citation in result["citations"]
    ] == ["S1", "S2"]


def test_invalid_citation_is_flagged():
    generator = make_generator(
        "The service parameter uses a config file [S9]."
    )

    result = generator.generate_answer(
        "How does the service parameter work?",
        [sample_passage()],
    )

    assert result["citation_status"] == "invalid"
    assert result["invalid_citations"] == ["S9"]
    assert "[unverified citation]" in result["answer"]


def test_no_passages_abstains_without_model_call():
    generator = make_generator("This should never be returned.")

    result = generator.generate_answer(
        "How do I configure MongoDB sharding?",
        [],
    )

    assert result["abstained"] is True
    assert result["citation_status"] == "abstained"
    assert result["citations"] == []
    assert result["answer"] == ABSTENTION_MESSAGE
    assert generator.llm.called is False


def test_model_abstention_is_detected():
    generator = make_generator(ABSTENTION_MESSAGE)

    result = generator.generate_answer(
        "How do I configure MongoDB sharding?",
        [sample_passage()],
    )

    assert result["abstained"] is True
    assert result["citation_status"] == "abstained"
    assert result["citations"] == []


def test_answer_without_citations_is_flagged():
    generator = make_generator(
        "The service parameter refers to a configured service."
    )

    result = generator.generate_answer(
        "How does the service parameter work?",
        [sample_passage()],
    )

    assert result["citation_status"] == "missing"
    assert result["citations"] == []