"""
ingest.py
---------
Ingestion pipeline:
  1. Scan the data/ directory for supported documents.
  2. Extract text page-by-page, preserving metadata.
  3. Split text into overlapping chunks.
  4. Embed chunks via Gemini text-embedding-004.
  5. Persist vectors in a local ChromaDB database.

Run directly:
    python src/ingest.py
"""

import os
import sys
from tqdm import tqdm

# ── allow `python src/ingest.py` to find sibling modules ──────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DATA_DIR, DB_DIR, COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP,
    SUPPORTED_EXTENSIONS, GEMINI_API_KEY, EMBEDDING_MODEL,
)


# ── 1. Document extraction ────────────────────────────────────────────────────

def extract_pdf_pages(file_path: str) -> list[dict]:
    """Extract text page-by-page from a PDF file."""
    from pypdf import PdfReader

    extracted = []
    file_name = os.path.basename(file_path)

    try:
        reader = PdfReader(file_path)
        for idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                clean_text = " ".join(text.split())
                extracted.append({
                    "text": clean_text,
                    "metadata": {"source": file_name, "page": idx + 1},
                })
    except Exception as exc:
        print(f"[ERROR] Could not read PDF '{file_name}': {exc}")

    return extracted


def extract_docx_pages(file_path: str) -> list[dict]:
    """Extract text paragraph-by-paragraph from a .docx file.
    Groups paragraphs into ~page-sized blocks to keep metadata consistent."""
    from docx import Document

    extracted = []
    file_name = os.path.basename(file_path)

    try:
        doc = Document(file_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        # Group into pseudo-pages of 30 paragraphs each
        page_size = 30
        for page_idx in range(0, len(paragraphs), page_size):
            block = paragraphs[page_idx: page_idx + page_size]
            text = " ".join(block)
            if text:
                extracted.append({
                    "text": text,
                    "metadata": {
                        "source": file_name,
                        "page": (page_idx // page_size) + 1,
                    },
                })
    except Exception as exc:
        print(f"[ERROR] Could not read DOCX '{file_name}': {exc}")

    return extracted


def extract_txt_pages(file_path: str) -> list[dict]:
    """Extract text from a plain .txt file, split into ~page-sized chunks."""
    extracted = []
    file_name = os.path.basename(file_path)

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Split into ~2000-character pseudo-pages
        page_size = 2000
        for page_idx in range(0, len(content), page_size):
            block = content[page_idx: page_idx + page_size].strip()
            if block:
                extracted.append({
                    "text": block,
                    "metadata": {
                        "source": file_name,
                        "page": (page_idx // page_size) + 1,
                    },
                })
    except Exception as exc:
        print(f"[ERROR] Could not read TXT '{file_name}': {exc}")

    return extracted


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Scan data_dir and extract text from all supported files."""
    all_pages: list[dict] = []

    if not os.path.isdir(data_dir):
        print(f"[WARN] Data directory not found: {data_dir}")
        return all_pages

    files = [
        f for f in os.listdir(data_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        print(f"[WARN] No supported documents found in {data_dir}")
        return all_pages

    print(f"\n📂  Found {len(files)} document(s) in '{data_dir}'")

    for file_name in tqdm(files, desc="Extracting documents"):
        full_path = os.path.join(data_dir, file_name)
        ext = os.path.splitext(file_name)[1].lower()

        if ext == ".pdf":
            pages = extract_pdf_pages(full_path)
        elif ext == ".docx":
            pages = extract_docx_pages(full_path)
        elif ext == ".txt":
            pages = extract_txt_pages(full_path)
        else:
            continue

        print(f"  ✅  {file_name}: {len(pages)} page(s) extracted")
        all_pages.extend(pages)

    return all_pages


# ── 2. Text chunking ──────────────────────────────────────────────────────────

def chunk_pages(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split page-level docs into overlapping, fixed-size character chunks."""
    chunks: list[dict] = []

    for page in pages:
        text = page["text"]
        metadata = page["metadata"]
        text_length = len(text)
        start = 0

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_text = text[start:end]

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "chunk_range": f"{start}-{end}",
                },
            })

            if end == text_length:
                break
            start += chunk_size - chunk_overlap

    return chunks


# ── 3. Vector database ────────────────────────────────────────────────────────

def save_to_vector_db(chunks: list[dict], db_path: str = DB_DIR) -> None:
    """Embed chunks and persist them in a local ChromaDB collection."""
    import chromadb
    from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction

    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Please add it to your .env file."
        )

    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)

    embedding_fn = GoogleGenerativeAiEmbeddingFunction(
        api_key=GEMINI_API_KEY,
        model_name=EMBEDDING_MODEL,
    )

    # Drop and recreate to allow re-ingestion
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB has a batch-size limit; upload in batches of 100
    batch_size = 100
    total = len(chunks)

    print(f"\n🔢  Embedding and indexing {total} chunk(s)…")

    for batch_start in tqdm(range(0, total, batch_size), desc="Uploading batches"):
        batch = chunks[batch_start: batch_start + batch_size]
        ids = [f"id_{batch_start + i}" for i in range(len(batch))]
        documents = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"\n✅  Successfully indexed {total} chunks into ChromaDB at '{db_path}'")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_ingestion(data_dir: str = DATA_DIR, db_path: str = DB_DIR) -> int:
    """Full ingestion pipeline. Returns number of chunks indexed."""
    print("=" * 55)
    print("  Document Q&A Bot — Ingestion Pipeline")
    print("=" * 55)

    pages = load_documents(data_dir)
    if not pages:
        print("[ERROR] No pages extracted. Aborting ingestion.")
        return 0

    chunks = chunk_pages(pages)
    print(f"\n✂️   Created {len(chunks)} chunk(s) from {len(pages)} page(s)")

    save_to_vector_db(chunks, db_path)
    return len(chunks)


if __name__ == "__main__":
    run_ingestion()
