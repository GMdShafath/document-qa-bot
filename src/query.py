import os
import sys
import json
import faiss
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DB_DIR, GEMINI_API_KEY, EMBEDDING_MODEL, GENERATION_MODEL, TOP_K
import google.generativeai as genai

genai.configure(api_key=GEMINI_API_KEY)

_index = None
_chunks = None
_gen_model = None

def _load_db():
    global _index, _chunks
    if _index is not None:
        return _index, _chunks
    index_path = os.path.join(DB_DIR, "index.faiss")
    chunks_path = os.path.join(DB_DIR, "chunks.json")
    _index = faiss.read_index(index_path)
    with open(chunks_path, "r") as f:
        _chunks = json.load(f)
    return _index, _chunks

def _get_gen_model():
    global _gen_model
    if _gen_model is None:
        _gen_model = genai.GenerativeModel(GENERATION_MODEL)
    return _gen_model

def reset_clients():
    global _index, _chunks, _gen_model
    _index = None
    _chunks = None
    _gen_model = None

def is_db_ready():
    try:
        index_path = os.path.join(DB_DIR, "index.faiss")
        chunks_path = os.path.join(DB_DIR, "chunks.json")
        return os.path.exists(index_path) and os.path.exists(chunks_path)
    except:
        return False

def get_embedding(text):
    result = genai.embed_content(model=EMBEDDING_MODEL, content=text)
    return result['embedding']

def query_rag_pipeline(user_query, k=TOP_K):
    index, chunks = _load_db()
    model = _get_gen_model()

    query_emb = np.array([get_embedding(user_query)]).astype("float32")
    faiss.normalize_L2(query_emb)
    scores, indices = index.search(query_emb, min(k, len(chunks)))

    context_blocks = []
    citations = []

    for idx in indices[0]:
        if idx == -1:
            continue
        chunk = chunks[idx]
        source = chunk["metadata"].get("source", "unknown")
        page = chunk["metadata"].get("page", "?")
        citation = f"{source}, Page {page}"
        citations.append(citation)
        context_blocks.append(f"[Source: {citation}]\n{chunk['text']}")

    context_payload = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are a professional document Q&A assistant. "
        "Answer using ONLY the provided context. "
        "Cite sources inline. "
        "If answer not found, say: 'I am sorry, but the provided documents do not contain the answer.' "
        "Do not use external knowledge."
    )

    prompt = f"{system_prompt}\n\nCONTEXT:\n{context_payload}\n\nQUESTION: {user_query}\n\nANSWER:"

    response = model.generate_content(prompt)

    return {
        "answer": response.text,
        "citations": list(dict.fromkeys(citations)),
        "raw_context": [c["text"] for c in [chunks[i] for i in indices[0] if i != -1]]
    }