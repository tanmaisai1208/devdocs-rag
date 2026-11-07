from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

DEFAULT_API_URL = "http://127.0.0.1:8000"
API_BASE_URL = os.getenv("RAG_API_URL", DEFAULT_API_URL).rstrip("/")
HEALTH_TIMEOUT_SECONDS = 3
ASK_TIMEOUT_SECONDS = 90

st.set_page_config(
    page_title="Engineering Knowledge Intelligence Platform",
    page_icon="📚",
    layout="wide",
)

st.title("Engineering Knowledge Intelligence Platform")
st.caption(
    "Ask technical questions and get evidence-grounded answers from the indexed "
    "Python, SQLAlchemy, and PostgreSQL documentation."
)

with st.sidebar:
    st.subheader("API connection")
    st.code(API_BASE_URL, language=None)
    if st.button("Check API health", use_container_width=True):
        try:
            health_response = requests.get(
                f"{API_BASE_URL}/health", timeout=HEALTH_TIMEOUT_SECONDS
            )
            if health_response.ok:
                st.success(f"API is available: {health_response.json().get('status', 'ok')}")
            else:
                st.error(f"API returned HTTP {health_response.status_code}.")
        except requests.RequestException as error:
            st.error(f"Could not reach the API: {error}")

    st.divider()
    st.markdown("**Example questions**")
    st.markdown(
        "- Why can a SQLAlchemy connection pool become exhausted?\n"
        "- What is the difference between `__cause__` and `__context__`?\n"
        "- How do I configure PostgreSQL TLS/SSL?"
    )
    st.caption(
        "Answerable questions may call Gemini. If your quota is exhausted, "
        "the API will report that generation is temporarily unavailable."
    )

with st.form("question_form", clear_on_submit=False):
    question = st.text_area(
        "Your technical question",
        placeholder="E.g. Why can SQLAlchemy report that the connection pool is exhausted?",
        height=110,
        max_chars=2000,
    )
    submitted = st.form_submit_button("Ask question", type="primary", use_container_width=True)

if submitted:
    normalized_question = question.strip()
    if not normalized_question:
        st.warning("Enter a question before submitting.")
    else:
        with st.spinner("Retrieving relevant documentation and checking evidence…"):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/ask",
                    json={"question": normalized_question},
                    timeout=ASK_TIMEOUT_SECONDS,
                )
                if response.ok:
                    st.session_state["last_rag_response"] = response.json()
                    st.session_state["last_rag_http_status"] = response.status_code
                    st.session_state.pop("last_rag_error", None)
                else:
                    try:
                        details = response.json().get("detail", response.text)
                    except ValueError:
                        details = response.text
                    st.session_state["last_rag_error"] = (
                        response.status_code,
                        str(details),
                    )
                    st.session_state.pop("last_rag_response", None)
            except requests.Timeout:
                st.session_state["last_rag_error"] = (
                    "timeout",
                    "The API request timed out. Check that the API server is running, "
                    "then try again.",
                )
                st.session_state.pop("last_rag_response", None)
            except requests.RequestException as error:
                st.session_state["last_rag_error"] = (
                    "connection",
                    f"Could not reach the API at {API_BASE_URL}. Start the FastAPI server first. Details: {error}",
                )
                st.session_state.pop("last_rag_response", None)

if "last_rag_error" in st.session_state:
    status_code, detail = st.session_state["last_rag_error"]
    if status_code == 503:
        st.warning(f"Service temporarily unavailable: {detail}")
    elif status_code == 422:
        st.error(f"The question was not accepted: {detail}")
    else:
        st.error(f"API request failed ({status_code}): {detail}")

result: dict[str, Any] | None = st.session_state.get("last_rag_response")
if result:
    st.divider()
    gate = result.get("evidence_gate") or {}
    abstained = bool(result.get("abstained", False))

    if abstained:
        st.warning("The system abstained because it could not provide a sufficiently supported answer.")
    else:
        st.success("Answer generated")

    st.subheader("Answer")
    st.markdown(result.get("answer") or "No answer was returned.")

    citations = result.get("citations") or []
    st.subheader(f"Sources ({len(citations)})")
    if not citations:
        st.caption("No citations were returned for this response.")
    else:
        for citation in citations:
            citation_id = citation.get("citation_id", "Source")
            title = citation.get("title") or "Technical documentation"
            section = citation.get("section")
            source_url = citation.get("source")
            with st.expander(f"[{citation_id}] {title}", expanded=True):
                if section:
                    st.markdown(f"**Section:** {section}")
                if source_url:
                    st.markdown(f"**URL:** [{source_url}]({source_url})")

    with st.expander("Pipeline details"):
        left, middle, right = st.columns(3)
        left.metric("Retrieved passages", result.get("retrieved_passages", 0))
        middle.metric("Retrieval latency", f"{result.get('retrieval_latency_ms', 0):.2f} ms")
        gen_latency = result.get("generation_latency_ms")
        right.metric("Generation latency", "Skipped" if gen_latency is None else f"{gen_latency:.0f} ms")

        st.markdown("**Retrieved passages**")

        passage_details = result.get("retrieved_passage_details") or []

        if not passage_details:
            st.caption("No retrieved passage details available.")
        else:
            for passage in passage_details:
                rank = passage.get("hybrid_rank", "?")
                title = passage.get("title") or "Technical documentation"
                section = passage.get("section") or "Unknown section"

                with st.expander(
                    f"Passage {rank}: {title} — {section}"
                ):
                    st.write(f"**Chunk ID:** {passage.get('chunk_id', 'N/A')}")
                    st.write(f"**Source:** {passage.get('source') or 'N/A'}")
                    st.write(f"**Section:** {section}")

                    score_col1, score_col2, score_col3 = st.columns(3)

                    dense_score = passage.get("dense_score")
                    bm25_score = passage.get("bm25_score")
                    rrf_score = passage.get("rrf_score")

                    score_col1.metric(
                        "Dense score",
                        f"{dense_score:.4f}" if dense_score is not None else "N/A"
                    )
                    score_col2.metric(
                        "BM25 score",
                        f"{bm25_score:.4f}" if bm25_score is not None else "N/A"
                    )
                    score_col3.metric(
                        "RRF score",
                        f"{rrf_score:.4f}" if rrf_score is not None else "N/A"
                    )

                    st.write(
                        f"**Dense rank:** {passage.get('dense_rank') or 'Not ranked'}"
                    )
                    st.write(
                        f"**BM25 rank:** {passage.get('bm25_rank') or 'Not ranked'}"
                    )

                    st.markdown("**Passage content**")
                    st.text(passage.get("content") or "No content available.")

        st.markdown("**Evidence gate**")
        gate_status = "PASS" if gate.get("sufficient") else "REJECT"
        st.write(f"Decision: **{gate_status}**")
        if gate.get("reason"):
            st.write(f"Reason: `{gate['reason']}`")
        if gate.get("max_dense_score") is not None:
            st.write(f"Maximum dense score: `{gate['max_dense_score']:.4f}`")
        if gate.get("threshold") is not None:
            st.write(f"Threshold: `{gate['threshold']:.4f}`")

        st.markdown("**Generation metadata**")
        st.write(f"Model: `{result.get('model') or 'not called'}`")
        st.write(f"Citation status: `{result.get('citation_status', 'unknown')}`")
        token_usage = result.get("token_usage") or {}
        st.write(
            "Token usage: "
            f"{token_usage.get('input_tokens', 0)} input, "
            f"{token_usage.get('output_tokens', 0)} output, "
            f"{token_usage.get('total_tokens', 0)} total"
        )
        invalid_citations = result.get("invalid_citations") or []
        if invalid_citations:
            st.error(f"Invalid citation IDs: {', '.join(invalid_citations)}")

st.caption("This tool answers only from its indexed documentation. An abstention is preferable to an unsupported answer.")
