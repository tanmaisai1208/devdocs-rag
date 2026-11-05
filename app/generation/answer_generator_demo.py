from answer_generator import AnswerGenerator


def main():
    question = (
        "How does the PostgreSQL service connection "
        "parameter work?"
    )

    # A representative passage from the official documentation.
    # The real retriever will supply passages in the next step.
    retrieved_passages = [
        {
            "chunk_id": "postgresql-service-example",
            "source": (
                "https://www.postgresql.org/docs/current/"
                "libpq-connect.html"
            ),
            "title": "PostgreSQL libpq connection documentation",
            "section": (
                "32.1.2. Parameter Key Words > service"
            ),
            "content": (
                "Service name to use for additional parameters. "
                "It specifies a service name in pg_service.conf "
                "that holds additional connection parameters. "
                "This allows applications to specify only a "
                "service name so connection parameters can be "
                "centrally maintained."
            ),
        }
    ]

    generator = AnswerGenerator()

    result = generator.generate_answer(
        question=question,
        retrieved_passages=retrieved_passages,
    )

    print("\n" + "=" * 70)
    print("QUESTION")
    print(question)

    print("\nANSWER")
    print(result["answer"])

    print("\nCITATION CHECK")
    print(result["citation_status"])

    for citation in result["citations"]:
        print(f"\n[{citation['citation_id']}]")
        print(f"Source: {citation['source']}")
        print(f"Section: {citation['section']}")

    print("\nOTHER DETAILS")
    print(f"Model: {result['model']}")
    print(f"Latency: {result['latency_ms']} ms")
    print(f"Token usage: {result['token_usage']}")
    print(f"Abstained: {result['abstained']}")

    if result["invalid_citations"]:
        print(
            "Invalid citation IDs:",
            result["invalid_citations"],
        )


if __name__ == "__main__":
    main()