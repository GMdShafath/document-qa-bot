import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DB_DIR, GEMINI_API_KEY, EMBEDDING_MODEL, GENERATION_MODEL, TOP_K
import google.generativeai as genai

genai.configure(api_key=GEMINI_API_KEY)

_embeddings = None
_chunks = None
_gen_model = None

def _load_db():
    global _embeddings, _chunks
    if _embeddings is not None:
        return _embeddings, _chunks
    emb_path = os.path.join(DB_DIR, "embeddings.npy")
    chunks_path = os.path.join(DB_DIR, "chunks.json")
    _embeddings = np.load(emb_path)
    with open(chunks_path, "r") as f:
        _chunks = json.load(f)
    return _embeddings, _chunks

def _get_gen_model():
    global _gen_model
    if _gen_model is None:
        _gen_model = genai.GenerativeModel(GENERATION_MODEL)
    return _gen_model

def reset_clients():
    global _embeddings, _chunks, _gen_model
    _embeddings = None
    _chunks = None
    _gen_model = None

def is_db_ready():
    emb_path = os.path.join(DB_DIR, "embeddings.npy")
    chunks_path = os.path.join(DB_DIR, "chunks.json")
    return os.path.exists(emb_path) and os.path.exists(chunks_path)

def get_embedding(text):
    result = genai.embed_content(model=EMBEDDING_MODEL, content=text)
    return np.array(result['embedding']).astype("float32")

def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(b, a)

def query_rag_pipeline(user_query, k=TOP_K):
    embeddings, chunks = _load_db()
    model = _get_gen_model()
    query_emb = get_embedding(user_query)
    scores = cosine_similarity(query_emb, embeddings)
    top_indices = np.argsort(scores)[::-1][:k]
    context_blocks = []
    citations = []
    for idx in top_indices:
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
        "raw_context": [chunks[i]["text"] for i in top_indices]
    }