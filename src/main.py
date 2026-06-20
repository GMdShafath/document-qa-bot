"""
main.py
-------
Streamlit web UI for the Document Q&A Bot.

Run:
    streamlit run src/main.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.config import DATA_DIR, DB_DIR, GEMINI_API_KEY
from src.ingest import run_ingestion
from src.query import query_rag_pipeline, is_db_ready, reset_clients

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Document Q&A Bot",
    page_icon="📚",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .answer-box {
        background: #f0f4ff;
        border-left: 4px solid #4361ee;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        font-size: 0.97rem;
        line-height: 1.65;
    }
    .citation-tag {
        display: inline-block;
        background: #e8f0fe;
        color: #1a73e8;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.82rem;
        margin: 2px 4px 2px 0;
    }
    .context-block {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        color: #444;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">📚 Document Q&A Bot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Ask questions about your documents — powered by RAG + Gemini</div>',
    unsafe_allow_html=True,
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Setup")

    # API Key check
    if not GEMINI_API_KEY:
        st.error("❌ GEMINI_API_KEY not found. Please set it in your `.env` file.")
    else:
        st.success("✅ API key loaded")

    st.divider()

    # Document stats
    st.subheader("📂 Documents")
    if os.path.isdir(DATA_DIR):
        docs = [f for f in os.listdir(DATA_DIR) if not f.startswith(".")]
        if docs:
            for d in docs:
                st.markdown(f"- `{d}`")
        else:
            st.info("No files in data/ yet. Add PDFs, DOCXs, or TXTs.")
    else:
        st.warning(f"`{DATA_DIR}` folder not found.")

    st.divider()

    # Ingest button
    st.subheader("🔄 Index Documents")
    st.caption("Run this whenever you add or update files in the `data/` folder.")

    if st.button("▶ Run Ingestion", use_container_width=True):
        if not GEMINI_API_KEY:
            st.error("Set your GEMINI_API_KEY first.")
        else:
            with st.spinner("Ingesting documents…"):
                try:
                    count = run_ingestion(DATA_DIR, DB_DIR)
                    reset_clients()
                    if count:
                        st.success(f"✅ Indexed {count} chunks successfully!")
                    else:
                        st.warning("No chunks were indexed. Check that `data/` contains documents.")
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

    st.divider()

    # DB status
    st.subheader("📊 Database Status")
    if is_db_ready():
        st.success("Vector DB is ready")
    else:
        st.warning("DB not found — run ingestion first")

    st.divider()
    st.caption("Built with Streamlit · ChromaDB · Gemini")

# ── Chat history ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(
                f'<div class="answer-box">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            if msg.get("citations"):
                st.markdown("**📎 Sources:**")
                citation_html = " ".join(
                    f'<span class="citation-tag">📄 {c}</span>' for c in msg["citations"]
                )
                st.markdown(citation_html, unsafe_allow_html=True)

            if msg.get("context") and st.toggle(
                "Show retrieved context", key=f"ctx_{msg['id']}"
            ):
                for i, chunk in enumerate(msg["context"], 1):
                    with st.expander(f"Chunk {i}"):
                        st.markdown(
                            f'<div class="context-block">{chunk}</div>',
                            unsafe_allow_html=True,
                        )
        else:
            st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask a question about your documents…"):
    # Guard rails
    if not GEMINI_API_KEY:
        st.error("Please set GEMINI_API_KEY in your .env file before querying.")
        st.stop()

    if not is_db_ready():
        st.warning("⚠️ The vector database is empty. Click **Run Ingestion** in the sidebar first.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer…"):
            try:
                result = query_rag_pipeline(prompt)
                answer = result["answer"]
                citations = result["citations"]
                raw_context = result["raw_context"]
            except Exception as exc:
                answer = f"⚠️ An error occurred: {exc}"
                citations = []
                raw_context = []

        # Display answer
        st.markdown(
            f'<div class="answer-box">{answer}</div>',
            unsafe_allow_html=True,
        )

        # Display citations
        if citations:
            st.markdown("**📎 Sources:**")
            citation_html = " ".join(
                f'<span class="citation-tag">📄 {c}</span>' for c in citations
            )
            st.markdown(citation_html, unsafe_allow_html=True)

        # Expandable raw context
        if raw_context:
            with st.expander("🔍 View retrieved context chunks"):
                for i, chunk in enumerate(raw_context, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.markdown(
                        f'<div class="context-block">{chunk}</div>',
                        unsafe_allow_html=True,
                    )

    # Persist to session state
    msg_id = len(st.session_state.messages)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "context": raw_context,
        "id": msg_id,
    })
