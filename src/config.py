"""
config.py
---------
Central configuration and constants for the Document Q&A Bot.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API ───────────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# ── Models ────────────────────────────────────────────────────────────────────
GENERATION_MODEL: str = "gemini-2.5-flash"
EMBEDDING_MODEL: str = "models/gemini-embedding-001"

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
DB_DIR: str = os.path.join(BASE_DIR, "db")

# ── ChromaDB ──────────────────────────────────────────────────────────────────
COLLECTION_NAME: str = "document_knowledge_base"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 1000       # characters per chunk
CHUNK_OVERLAP: int = 200     # character overlap between consecutive chunks

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K: int = 4               # number of chunks to retrieve per query

# ── Supported file types ──────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS: tuple = (".pdf", ".docx", ".txt")
