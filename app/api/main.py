from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

# The existing project scripts import modules from app/ as top-level packages
# (for example, "retrieval.hybrid_retriever"). Add app/ to the import path so
# this module works with both `uvicorn api.main:app --app-dir app` and tests
# that import it as `app.api.main`.
APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from retrieval.evidence_gate import (  # noqa: E402
    assess_evidence_sufficiency,
    DEFAULT_DENSE_SCORE_THRESHOLD,
)
from generation.answer_generator import AnswerGenerator  # noqa: E402

logger = logging.getLogger(__name__)

DENSE_SCORE_THRESHOLD = DEFAULT_DENSE_SCORE_THRESHOLD
TOP_K = 5


class AskRequest(BaseModel):
    """Request body for the technical-question endpoint."""

    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def strip_and_validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be empty or whitespace only.")
        return value


class AskResponse(BaseModel):
    """Answer, evidence-gate decision, citations, and basic timings."""

    question: str
    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    abstained: bool
    evidence_gate: dict[str, Any]
    retrieved_passages: int
    retrieved_passage_details: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_latency_ms: float
    generation_latency_ms: float | None = None
    model: str | None = None
    token_usage: dict[str, Any] | None = None
    citation_status: str
    invalid_citations: list[str] = Field(default_factory=list)


class RAGService:
    """Own one retriever and one answer generator for the API process."""

    def __init__(self) -> None:
        self.retriever: HybridRetriever | None = None
        try:
            self.retriever = HybridRetriever()
            self.generator = AnswerGenerator()
        except Exception:
            if self.retriever is not None:
                self.retriever.close()
            raise

    def close(self) -> None:
        if self.retriever is not None:
            self.retriever.close()
            self.retriever = None

    def ask(self, question: str) -> dict[str, Any]:
        if self.retriever is None:
            raise RuntimeError("The retrieval service is not initialized.")

        retrieval_start = time.perf_counter()
        passages = self.retriever.retrieve(query=question, top_k=TOP_K)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

        gate_result = assess_evidence_sufficiency(
            passages,
            threshold=DENSE_SCORE_THRESHOLD,
        )
        sufficient = bool(gate_result.get("sufficient", False))

        # The generator's empty-evidence path returns the standard abstention
        # without making a Gemini request. This lets the API preserve the same
        # abstention wording as the CLI pipeline.
        generation_result = self.generator.generate_answer(
            question=question,
            retrieved_passages=passages if sufficient else [],
        )

        evidence_gate = {
            "sufficient": sufficient,
            "reason": gate_result.get("reason", "No reason provided."),
            "max_dense_score": gate_result.get("max_dense_score"),
            "threshold": DENSE_SCORE_THRESHOLD,
        }

        if sufficient:
            citations = generation_result.get("citations") or []
            token_usage = generation_result.get("token_usage")
            generation_latency_ms = generation_result.get("latency_ms")
            model = generation_result.get("model")
            abstained = bool(generation_result.get("abstained", False))
        else:
            # Do not expose citations for a response that abstains due to
            # insufficient evidence, even if a future generator change adds any.
            citations = []
            token_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }
            generation_latency_ms = None
            model = None
            abstained = True

        return {
            "question": question,
            "answer": generation_result.get("answer", "I don't have enough evidence to answer that question."),
            "citations": citations,
            "abstained": abstained,
            "evidence_gate": evidence_gate,
            "retrieved_passages": len(passages),
            "retrieved_passage_details": [
                {
                    "chunk_id": passage.get("chunk_id"),
                    "content": passage.get("content"),
                    "source": passage.get("source"),
                    "title": passage.get("title"),
                    "section": passage.get("section"),
                    "hybrid_rank": passage.get("hybrid_rank"),
                    "dense_rank": passage.get("dense_rank"),
                    "bm25_rank": passage.get("bm25_rank"),
                    "dense_score": passage.get("dense_score"),
                    "bm25_score": passage.get("bm25_score"),
                    "rrf_score": passage.get("rrf_score"),
                }
                for passage in passages
            ],
            "retrieval_latency_ms": round(retrieval_latency_ms, 2),
            "generation_latency_ms": (
                round(float(generation_latency_ms), 2)
                if generation_latency_ms is not None
                else None
            ),
            "model": model,
            "token_usage": token_usage,
            "citation_status": generation_result.get(
                "citation_status", "not_checked"
            ),
            "invalid_citations": generation_result.get("invalid_citations") or [],
        }


def is_quota_error(error: Exception) -> bool:
    message = f"{type(error).__name__}: {error}".lower()
    return any(
        marker in message
        for marker in (
            "resource_exhausted",
            "quota exceeded",
            "quota has been exceeded",
            "rate limit",
            "429",
        )
    )


def create_app(service_factory: Callable[[], Any] | None = None) -> FastAPI:
    """Application factory; the factory parameter also enables offline tests."""

    factory = service_factory or RAGService

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        service = factory()
        application.state.rag_service = service
        try:
            yield
        finally:
            close = getattr(service, "close", None)
            if callable(close):
                close()

    application = FastAPI(
        title="Engineering Knowledge Intelligence Platform",
        description=(
            "Evidence-grounded technical documentation Q&A with citations "
            "and abstention when retrieved evidence is insufficient."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/health", tags=["operations"])
    def health(request: Request) -> dict[str, str]:
        service_ready = getattr(request.app.state, "rag_service", None) is not None
        return {"status": "ok" if service_ready else "not_ready"}

    @application.post(
        "/ask",
        response_model=AskResponse,
        tags=["question answering"],
    )
    def ask_question(payload: AskRequest, request: Request) -> AskResponse:
        service = getattr(request.app.state, "rag_service", None)
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The RAG service is not ready.",
            )

        try:
            return AskResponse(**service.ask(payload.question))
        except Exception as error:
            if is_quota_error(error):
                logger.warning("Gemini quota/rate limit reached: %s", error)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=(
                        "The answer-generation quota is currently unavailable. "
                        "Try again after the provider quota resets."
                    ),
                ) from error

            logger.exception("Failed to process a question through the RAG API.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process the question. Check the API server logs.",
            ) from error

    return application


app = create_app()
