
import re

from hybrid_retriever import HybridRetriever


QUERY = (
    "How can PostgreSQL clients configure TLS or SSL "
    "for a connection?"
)

PARAMETERS = [
    "sslmode",
    "sslrootcert",
    "sslcert",
    "sslkey",
    "sslcertmode",
    "verify-full",
    "verify-ca",
]


def main():
    retriever = HybridRetriever(candidate_k=20)

    try:
        results = retriever.retrieve(QUERY, top_k=15)

        print(f"\nQuery: {QUERY}")
        print(f"Retrieved passages: {len(results)}")

        for passage in results:
            content = passage["content"]
            section = passage.get("section", "")
            searchable_text = f"{section}\n{content}"

            # Match complete parameter names rather than substrings.
            matched = [
                term
                for term in PARAMETERS
                if re.search(
                    rf"(?<![A-Za-z0-9_])"
                    rf"{re.escape(term)}"
                    rf"(?![A-Za-z0-9_])",
                    searchable_text,
                    flags=re.IGNORECASE,
                )
            ]

            print("\n" + "=" * 80)
            print(f"Hybrid rank: {passage['hybrid_rank']}")
            print(f"Dense rank: {passage['dense_rank']}")
            print(f"BM25 rank: {passage['bm25_rank']}")
            print(f"Section: {section}")
            print(f"Matched terms: {', '.join(matched) or 'None'}")
            print(f"Source: {passage.get('source', '')}")
            print("\nPassage:")
            print(content[:700])

    finally:
        retriever.close()


if __name__ == "__main__":
    main()
