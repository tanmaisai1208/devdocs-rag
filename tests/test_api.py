from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import main as api_main


class FakeService:
    def __init__(self) -> None:
        self.closed = False

    def ask(self, question: str) -> dict:
        return {
            "question": question,
            "answer": "Use the documented configuration option [S1].",
            "citations": [
                {
                    "citation_id": "S1",
                    "title": "Example documentation",
                    "section": "Configuration",
                    "source": "https://example.com/docs",
                }
            ],
            "abstained": False,
            "evidence_gate": {
                "sufficient": True,
                "reason": "Relevant evidence found.",
                "max_dense_score": 0.8,
                "threshold": 0.47,
            },
            "retrieved_passages": 5,
            "retrieval_latency_ms": 22.1,
            "generation_latency_ms": 850.0,
            "model": "mock-model",
            "token_usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            "citation_status": "valid",
            "invalid_citations": [],
        }

    def close(self) -> None:
        self.closed = True


class QuotaService(FakeService):
    def ask(self, question: str) -> dict:
        raise RuntimeError("RESOURCE_EXHAUSTED: quota exceeded")


def test_health_and_ask_endpoint() -> None:
    service = FakeService()
    app = api_main.create_app(service_factory=lambda: service)

    with TestClient(app) as client:
        health = client.get("/health")
        response = client.post(
            "/ask",
            json={"question": "  How do I configure this?  "},
        )

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "How do I configure this?"
    assert body["citations"][0]["citation_id"] == "S1"
    assert body["evidence_gate"]["sufficient"] is True
    assert body["abstained"] is False
    assert service.closed is True


def test_blank_question_is_rejected() -> None:
    app = api_main.create_app(service_factory=FakeService)

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 422


def test_quota_error_returns_503_without_leaking_exception() -> None:
    app = api_main.create_app(service_factory=QuotaService)

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "Explain this error."})

    assert response.status_code == 503
    assert "quota" in response.json()["detail"].lower()
    assert "RESOURCE_EXHAUSTED" not in response.text


def test_evidence_gate_rejection_skips_real_generator_call(monkeypatch) -> None:
    class FakeRetriever:
        def __init__(self) -> None:
            self.closed = False

        def retrieve(self, query: str, top_k: int) -> list[dict]:
            return [
                {
                    "chunk_id": "irrelevant-1",
                    "content": "Unrelated documentation passage.",
                    "dense_score": 0.1,
                    "source": "https://example.com",
                    "title": "Unrelated document",
                    "section": "Overview",
                }
            ]

        def close(self) -> None:
            self.closed = True

    class FakeGenerator:
        def __init__(self) -> None:
            self.calls: list[list[dict]] = []

        def generate_answer(self, question: str, retrieved_passages: list[dict]) -> dict:
            self.calls.append(retrieved_passages)
            return {
                "answer": "I don't have enough relevant evidence to answer that question.",
                "citations": [],
                "abstained": True,
                "citation_status": "valid",
                "invalid_citations": [],
                "latency_ms": 0,
                "model": None,
                "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }

    monkeypatch.setattr(api_main, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(api_main, "AnswerGenerator", FakeGenerator)

    service = api_main.RAGService()
    try:
        response = service.ask("How do I configure an unrelated database?")
        assert response["evidence_gate"]["sufficient"] is False
        assert response["abstained"] is True
        assert response["citations"] == []
        assert response["generation_latency_ms"] is None
        # The fake generator only receives empty evidence; its implementation
        # is the same no-provider-call branch used by the current CLI pipeline.
        assert service.generator.calls == [[]]
    finally:
        service.close()
