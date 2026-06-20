"""
query.py
--------
Query pipeline:
  1. Load the pre-built ChromaDB collection from disk.
  2. Embed the user's question with the same model used at ingestion.
  3. Retrieve the top-k most relevant chunks.
  4. Build a grounded prompt with citations.
  5. Call Gemini to generate a factual, sourced answer.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DB_DIR, COLLECTION_NAME, GENERATION_MODEL,
    EMBEDDING_MODEL, GEMINI_API_KEY, TOP_K,
)


# ── Lazy singletons ───────────────────────────────────────────────────────────
# Kept as module-level variables so Streamlit doesn't recreate them on every run.

_collection = None
_gen_model = None


def _get_collection():
    """Load (or return cached) ChromaDB collection."""
    global _collection
    if _collection is not None:
        return _collection

    import chromadb
    from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction

    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Please add it to your .env file."
        )

    client = chromadb.PersistentClient(path=DB_DIR)
    embedding_fn = GoogleGenerativeAiEmbeddingFunction(
        api_key=GEMINI_API_KEY,
        model_name=EMBEDDING_MODEL,
    )

    _collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )
    return _collection


def _get_gen_model():
    """Load (or return cached) Gemini generative model."""
    global _gen_model
    if _gen_model is not None:
        return _gen_model

    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    _gen_model = genai.GenerativeModel(GENERATION_MODEL)
    return _gen_model


def reset_clients() -> None:
    """Force re-initialisation of DB and model clients (call after re-ingestion)."""
    global _collection, _gen_model
    _collection = None
    _gen_model = None


# ── Core pipeline ─────────────────────────────────────────────────────────────

def is_db_ready() -> bool:
    """Return True if the vector DB exists and has at least one document."""
    try:
        col = _get_collection()
        return col.count() > 0
    except Exception:
        return False


def query_rag_pipeline(user_query: str, k: int = TOP_K) -> dict:
    """
    Full RAG pipeline for a single user question.

    Returns
    -------
    dict with keys:
      - answer      : str  — LLM-generated, grounded answer
      - citations   : list[str] — formatted citation strings
      - raw_context : list[str] — raw chunk texts used as context
    """
    collection = _get_collection()
    model = _get_gen_model()

    # ── Retrieve top-k chunks ──────────────────────────────────────────────
    results = collection.query(
        query_texts=[user_query],
        n_results=min(k, collection.count()),
    )

    documents = results["documents"][0]   # list of chunk texts
    metadatas = results["metadatas"][0]   # list of metadata dicts

    if not documents:
        return {
            "answer": "I am sorry, but no relevant documents were found in the database.",
            "citations": [],
            "raw_context": [],
        }

    # ── Build context payload with citations ───────────────────────────────
    context_blocks: list[str] = []
    citations: list[str] = []

    for doc, meta in zip(documents, metadatas):
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        citation = f"{source}, Page {page}"
        citations.append(citation)
        context_blocks.append(f"[Source: {citation}]\n{doc}")

    context_payload = "\n\n---\n\n".join(context_blocks)

    # ── Grounded prompt ────────────────────────────────────────────────────
    system_prompt = (
        "You are a professional, precise document Q&A assistant. "
        "Answer the user's question using ONLY the provided document context. "
        "For every fact you state, cite the source inline in parentheses, e.g. (annual_report.pdf, Page 3). "
        "If the answer cannot be found in the context, respond with exactly: "
        "'I am sorry, but the provided documents do not contain the answer to your question.' "
        "Do NOT use any external knowledge or make up information."
    )

    full_prompt = (
        f"{system_prompt}\n\n"
        f"DOCUMENT CONTEXT:\n{context_payload}\n\n"
        f"USER QUESTION: {user_query}\n\n"
        f"GROUNDED ANSWER:"
    )

    # ── Generate answer ────────────────────────────────────────────────────
    response = model.generate_content(full_prompt)

    return {
        "answer": response.text,
        "citations": list(dict.fromkeys(citations)),  # deduplicated, order-preserved
        "raw_context": documents,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not is_db_ready():
        print("[ERROR] Vector database not found. Run `python src/ingest.py` first.")
        sys.exit(1)

    print("=" * 55)
    print("  Document Q&A Bot — Interactive Query Mode")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 55)

    while True:
        try:
            user_input = input("\n❓  Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        result = query_rag_pipeline(user_input)

        print(f"\n💬  Answer:\n{result['answer']}")
        if result["citations"]:
            print("\n📚  Sources used:")
            for c in result["citations"]:
                print(f"    • {c}")
