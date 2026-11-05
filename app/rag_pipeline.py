
import time

from retrieval.hybrid_retriever import HybridRetriever
from retrieval.evidence_gate import (
    assess_evidence_sufficiency,
    DEFAULT_DENSE_SCORE_THRESHOLD,
)
from generation.answer_generator import AnswerGenerator


# Provisional threshold based on the current 15-question benchmark.
DENSE_SCORE_THRESHOLD = DEFAULT_DENSE_SCORE_THRESHOLD

TOP_K = 5


def main() -> None:
    retriever = None

    try:
        # Initialize models and indexes once for the entire session.
        retriever = HybridRetriever()
        generator = AnswerGenerator()

        print("\n" + "=" * 70)
        print("ENGINEERING KNOWLEDGE INTELLIGENCE PLATFORM")
        print("Ask a question about the indexed technical documentation.")
        print("Type 'exit' or 'quit' to stop.")
        print(f"Evidence gate threshold: {DENSE_SCORE_THRESHOLD:.2f}")
        print("=" * 70)

        while True:
            try:
                question = input("\nQuestion: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

            if question.lower() in {"exit", "quit"}:
                print("Exiting...")
                break

            if not question:
                print("Please enter a question.")
                continue

            try:
                # --------------------------------------------------
                # Stage 1: Retrieval
                # --------------------------------------------------
                retrieval_start = time.perf_counter()

                passages = retriever.retrieve(
                    query=question,
                    top_k=TOP_K,
                )

                retrieval_latency_ms = (
                    time.perf_counter() - retrieval_start
                ) * 1000

                # --------------------------------------------------
                # Stage 2: Evidence sufficiency check
                # --------------------------------------------------
                gate_result = assess_evidence_sufficiency(
                    passages,
                    threshold=DENSE_SCORE_THRESHOLD,
                )

                max_dense_score = gate_result["max_dense_score"]

                if max_dense_score is None:
                    score_text = "unavailable"
                else:
                    score_text = f"{max_dense_score:.4f}"

                if gate_result["sufficient"]:
                    print(
                        f"\nEvidence gate: PASS "
                        f"(maximum dense score: {score_text}; "
                        f"threshold: {DENSE_SCORE_THRESHOLD:.4f})"
                    )
                else:
                    print(
                        f"\nEvidence gate: REJECT "
                        f"(maximum dense score: {score_text}; "
                        f"threshold: {DENSE_SCORE_THRESHOLD:.4f})"
                    )
                    print(
                        "The retrieved evidence appears insufficient. "
                        "The system will abstain without calling Gemini."
                    )

                # --------------------------------------------------
                # Stage 3: Answer generation or abstention
                # --------------------------------------------------
                generation_skipped = not gate_result["sufficient"]

                if generation_skipped:
                    # AnswerGenerator already handles an empty passage
                    # list by returning the standard abstention response
                    # without invoking the Gemini model.
                    result = generator.generate_answer(
                        question=question,
                        retrieved_passages=[],
                    )
                else:
                    result = generator.generate_answer(
                        question=question,
                        retrieved_passages=passages,
                    )

                # --------------------------------------------------
                # Stage 4: Display answer and citations
                # --------------------------------------------------
                print("\n" + "-" * 70)
                print("ANSWER")
                print(result["answer"])

                print("\nSOURCES")

                if result["citations"]:
                    for citation in result["citations"]:
                        print(
                            f"\n[{citation['citation_id']}] "
                            f"{citation.get('title') or 'Technical documentation'}"
                        )

                        if citation.get("section"):
                            print(f"Section: {citation['section']}")

                        if citation.get("source"):
                            print(f"URL: {citation['source']}")
                else:
                    print("No source citations were returned.")

                # --------------------------------------------------
                # Stage 5: Pipeline details
                # --------------------------------------------------
                print("\nPIPELINE DETAILS")
                print(f"Retrieved passages: {len(passages)}")
                print(
                    f"Retrieval latency: "
                    f"{retrieval_latency_ms:.2f} ms"
                )
                print(
                    f"Evidence sufficient: "
                    f"{gate_result['sufficient']}"
                )
                print(
                    f"Evidence gate reason: "
                    f"{gate_result['reason']}"
                )

                if generation_skipped:
                    print("Generation: skipped; Gemini was not called.")
                    print("Model: not called")
                    print("Token usage: 0 (no Gemini request)")
                else:
                    print(
                        f"Generation latency: "
                        f"{result['latency_ms']:.2f} ms"
                    )
                    print(f"Model: {result['model']}")
                    print(
                        f"Token usage: {result['token_usage']}"
                    )

                print(
                    f"Citation status: "
                    f"{result['citation_status']}"
                )
                print(f"Abstained: {result['abstained']}")

                if result["invalid_citations"]:
                    print(
                        "Invalid citation IDs: "
                        f"{result['invalid_citations']}"
                    )

            except Exception as error:
                error_text = str(error)

                print(
                    f"\nError processing this question: "
                    f"{type(error).__name__}: {error}"
                )

                if (
                    "RESOURCE_EXHAUSTED" in error_text
                    or "quota exceeded" in error_text.lower()
                ):
                    print(
                        "\nGemini quota is exhausted. "
                        "Stopping the interactive session to avoid "
                        "repeated failed API requests. Try again after "
                        "the quota resets."
                    )
                    break

    finally:
        if retriever is not None:
            retriever.close()


if __name__ == "__main__":
    main()
