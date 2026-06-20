import os
import sys
import json
import faiss
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR, DB_DIR, GEMINI_API_KEY, EMBEDDING_MODEL
import google.generativeai as genai

genai.configure(api_key=GEMINI_API_KEY)

def get_embedding(text):
    result = genai.embed_content(model=EMBEDDING_MODEL, content=text)
    return result['embedding']

def extract_pdf_pages(file_path):
    from pypdf import PdfReader
    extracted = []
    file_name = os.path.basename(file_path)
    try:
        reader = PdfReader(file_path)
        for idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                extracted.append({"text": " ".join(text.split()), "metadata": {"source": file_name, "page": idx + 1}})
    except Exception as e:
        print(f"Error: {e}")
    return extracted

def extract_docx_pages(file_path):
    from docx import Document
    extracted = []
    file_name = os.path.basename(file_path)
    try:
        doc = Document(file_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for i in range(0, len(paragraphs), 30):
            text = " ".join(paragraphs[i:i+30])
            if text:
                extracted.append({"text": text, "metadata": {"source": file_name, "page": (i//30)+1}})
    except Exception as e:
        print(f"Error: {e}")
    return extracted

def extract_txt_pages(file_path):
    extracted = []
    file_name = os.path.basename(file_path)
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        for i in range(0, len(content), 2000):
            block = content[i:i+2000].strip()
            if block:
                extracted.append({"text": block, "metadata": {"source": file_name, "page": (i//2000)+1}})
    except Exception as e:
        print(f"Error: {e}")
    return extracted

def load_documents(data_dir=DATA_DIR):
    all_pages = []
    if not os.path.isdir(data_dir):
        return all_pages
    files = [f for f in os.listdir(data_dir) if os.path.splitext(f)[1].lower() in (".pdf", ".docx", ".txt")]
    for file_name in tqdm(files, desc="Extracting"):
        full_path = os.path.join(data_dir, file_name)
        ext = os.path.splitext(file_name)[1].lower()
        if ext == ".pdf":
            pages = extract_pdf_pages(full_path)
        elif ext == ".docx":
            pages = extract_docx_pages(full_path)
        else:
            pages = extract_txt_pages(full_path)
        print(f"  {file_name}: {len(pages)} pages")
        all_pages.extend(pages)
    return all_pages

def chunk_pages(pages, chunk_size=1000, chunk_overlap=200):
    chunks = []
    for page in pages:
        text = page["text"]
        metadata = page["metadata"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append({"text": text[start:end], "metadata": metadata})
            if end == len(text):
                break
            start += chunk_size - chunk_overlap
    return chunks

def run_ingestion(data_dir=DATA_DIR, db_path=DB_DIR):
    print("Starting ingestion...")
    pages = load_documents(data_dir)
    if not pages:
        return 0
    chunks = chunk_pages(pages)
    print(f"Created {len(chunks)} chunks")

    os.makedirs(db_path, exist_ok=True)
    embeddings = []
    for chunk in tqdm(chunks, desc="Embedding"):
        emb = get_embedding(chunk["text"])
        embeddings.append(emb)

    emb_array = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatIP(emb_array.shape[1])
    faiss.normalize_L2(emb_array)
    index.add(emb_array)

    faiss.write_index(index, os.path.join(db_path, "index.faiss"))
    with open(os.path.join(db_path, "chunks.json"), "w") as f:
        json.dump(chunks, f)

    print(f"Indexed {len(chunks)} chunks!")
    return len(chunks)

if __name__ == "__main__":
    run_ingestion()
