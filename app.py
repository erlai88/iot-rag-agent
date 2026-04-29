"""
Streamlit application for the IoT RAG demo.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

# Set this before importing modules that pull in chromadb/opentelemetry.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import streamlit as st
from dotenv import load_dotenv

from chain import ask
from ingest import DATA_DIR, ingest_pdfs, list_knowledge_base_documents
from logger import build_bad_case_report, update_feedback


load_dotenv()


def init_state() -> None:
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def apply_secret_overrides() -> None:
    """Mirror Streamlit secrets into process env for runtime config."""
    secret_keys = [
        "EMBEDDING_PROVIDER",
        "EMBEDDING_API_KEY",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_MODEL",
        "OPENAI_EMBEDDING_MODEL",
        "ZHIPUAI_API_KEY",
        "ZHIPUAI_EMBEDDING_MODEL",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "DEEPSEEK_API_KEY",
    ]
    for key in secret_keys:
        if key in st.secrets and str(st.secrets[key]).strip():
            os.environ[key] = str(st.secrets[key]).strip()


def resolve_embedding_api_key(manual_api_key: str) -> str:
    """Resolve the embedding API key."""
    if manual_api_key.strip():
        return manual_api_key.strip()

    if "EMBEDDING_API_KEY" in st.secrets:
        return str(st.secrets["EMBEDDING_API_KEY"]).strip()

    if "ZHIPUAI_API_KEY" in st.secrets:
        return str(st.secrets["ZHIPUAI_API_KEY"]).strip()

    if "OPENAI_API_KEY" in st.secrets:
        return str(st.secrets["OPENAI_API_KEY"]).strip()

    return (
        os.getenv("EMBEDDING_API_KEY")
        or os.getenv("ZHIPUAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    ).strip()


def has_secret_embedding_key() -> bool:
    """Return whether an embedding key is already configured via secrets or env."""
    return bool(
        ("EMBEDDING_API_KEY" in st.secrets and str(st.secrets["EMBEDDING_API_KEY"]).strip())
        or ("ZHIPUAI_API_KEY" in st.secrets and str(st.secrets["ZHIPUAI_API_KEY"]).strip())
        or ("OPENAI_API_KEY" in st.secrets and str(st.secrets["OPENAI_API_KEY"]).strip())
        or os.getenv("EMBEDDING_API_KEY")
        or os.getenv("ZHIPUAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )


def resolve_llm_api_key() -> str:
    """Resolve the LLM API key from secrets or env."""
    if "LLM_API_KEY" in st.secrets:
        return str(st.secrets["LLM_API_KEY"]).strip()

    if "DEEPSEEK_API_KEY" in st.secrets:
        return str(st.secrets["DEEPSEEK_API_KEY"]).strip()

    if "OPENAI_API_KEY" in st.secrets:
        return str(st.secrets["OPENAI_API_KEY"]).strip()

    return (
        os.getenv("LLM_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    ).strip()


def save_uploaded_files(uploaded_files: list) -> list[Path]:
    """Save uploaded PDFs into the local data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for uploaded_file in uploaded_files:
        target_path = DATA_DIR / uploaded_file.name
        target_path.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(target_path)

    return saved_paths


def render_sidebar(embedding_api_key: str) -> None:
    """Render the sidebar controls."""
    with st.sidebar:
        st.header("Knowledge Base")

        uploaded_files = st.file_uploader(
            "Upload PDF manuals",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if st.button("Upload and ingest", use_container_width=True):
            if not uploaded_files:
                st.warning("Please select at least one PDF file.")
            elif not embedding_api_key:
                st.warning("Please provide an embedding API key.")
            else:
                try:
                    saved_paths = save_uploaded_files(uploaded_files)
                    with st.spinner("Parsing and ingesting PDFs..."):
                        total_chunks = ingest_pdfs(saved_paths, api_key=embedding_api_key)
                    st.success(f"Ingest complete. Added {total_chunks} chunks.")
                except Exception as exc:
                    st.error(f"Ingest failed: {exc}")

        kb_docs = list_knowledge_base_documents()
        total_chunks = sum(int(item["chunk_count"]) for item in kb_docs)

        st.subheader("Current Documents")
        st.caption(f"Documents: {len(kb_docs)} | Total chunks: {total_chunks}")
        if kb_docs:
            for item in kb_docs:
                st.write(f"- {item['source']} ({item['chunk_count']} chunks)")
        else:
            st.info("No knowledge base documents yet.")

        st.divider()
        st.subheader("Bad Case")
        report_content = build_bad_case_report()
        st.download_button(
            "Export bad case report",
            data=report_content,
            file_name=f"bad_case_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True,
        )


def render_sources(sources: list[dict]) -> None:
    """Render source documents for one answer."""
    with st.expander("Source documents"):
        if not sources:
            st.write("None")
            return

        for source in sources:
            st.write(f"- {source.get('source')} | page {source.get('page')}")


def render_feedback_controls(message: dict) -> None:
    """Render good/bad feedback buttons."""
    interaction_id = message.get("interaction_id")
    if not interaction_id:
        return

    current_feedback = message.get("feedback")
    cols = st.columns([1, 1, 6])

    if cols[0].button("👍", key=f"good_{interaction_id}"):
        if update_feedback(interaction_id, "good"):
            message["feedback"] = "good"
            st.rerun()

    if cols[1].button("👎", key=f"bad_{interaction_id}"):
        if update_feedback(interaction_id, "bad"):
            message["feedback"] = "bad"
            st.rerun()

    if current_feedback:
        cols[2].caption(f"Current label: {current_feedback}")


def render_chat_history() -> None:
    """Render historical chat messages."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
                continue

            st.write(message["answer"])

            if message.get("used_tool"):
                device_id = (message.get("tool_result") or {}).get("device_id")
                label = "device status queried"
                if device_id:
                    label = f"{label} | {device_id}"
                st.markdown(f"`{label}`")

            render_sources(message.get("sources") or [])
            render_feedback_controls(message)


def handle_user_input(embedding_api_key: str, llm_api_key: str) -> None:
    """Handle a new user question."""
    user_query = st.chat_input("Ask a question about the uploaded manuals")
    if not user_query:
        return

    if not embedding_api_key:
        st.error("Please provide an embedding API key.")
        return

    if not llm_api_key:
        st.error("Please provide an LLM API key.")
        return

    st.session_state.messages.append({"role": "user", "content": user_query})

    try:
        with st.spinner("Searching documents and generating an answer..."):
            result = ask(
                user_query,
                embedding_api_key=embedding_api_key,
                llm_api_key=llm_api_key,
            )
    except Exception as exc:
        st.error(f"Question answering failed: {exc}")
        return

    st.session_state.messages.append(
        {
            "role": "assistant",
            "answer": result["answer"],
            "sources": result["sources"],
            "used_tool": result["used_tool"],
            "tool_result": result["tool_result"],
            "interaction_id": result.get("interaction_id"),
            "feedback": None,
        }
    )
    st.rerun()


def main() -> None:
    """Render the application."""
    st.set_page_config(page_title="IoT RAG Agent", page_icon="📘", layout="wide")
    apply_secret_overrides()
    init_state()

    st.title("IoT RAG Agent")
    st.caption("Upload PDF manuals, build a knowledge base, and chat with retrieval + tools.")

    if has_secret_embedding_key():
        manual_embedding_api_key = ""
        st.sidebar.caption("Embedding key is configured in deployment settings.")
    else:
        manual_embedding_api_key = st.sidebar.text_input("Embedding API Key", type="password")
    embedding_api_key = resolve_embedding_api_key(manual_embedding_api_key)
    llm_api_key = resolve_llm_api_key()
    if llm_api_key:
        st.sidebar.caption("LLM key detected from deployment config.")

    render_sidebar(embedding_api_key)
    render_chat_history()
    handle_user_input(embedding_api_key, llm_api_key)


if __name__ == "__main__":
    main()
