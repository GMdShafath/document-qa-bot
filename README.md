# 📚 Document Q&A Bot — RAG System

A production-quality Retrieval-Augmented Generation (RAG) application that lets you ask natural language questions about your own documents (PDFs, DOCX, TXT files). Built with **Python**, **ChromaDB**, **Google Gemini**, and **Streamlit**.

---

## Table of Contents

- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Design Decisions](#design-decisions)

---

## Architecture

```
User Query
    │
    ▼
Embed Query (text-embedding-004)
    │
    ▼
Semantic Search ──► ChromaDB (local, disk-persistent)
    │                   ▲
    │              Indexed Chunks
    │              + Embeddings
    │                   │
    │           Ingest Pipeline:
    │           PDF/DOCX/TXT → Extract → Chunk → Embed → Store
    │
    ▼
Top-K Retrieved Chunks + Metadata (source, page)
    │
    ▼
Grounded Prompt Builder
    │
    ▼
Gemini (gemini-2.5-flash-preview-09-2025)
    │
    ▼
Grounded Answer + Inline Citations
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Embeddings | Google `text-embedding-004` |
| Generation | Google `gemini-2.5-flash-preview-09-2025` |
| Vector DB | ChromaDB (local, disk-persistent) |
| PDF parsing | pypdf |
| DOCX parsing | python-docx |
| UI | Streamlit |
| Config | python-dotenv |

---

## Project Structure

```
document-qa-bot/
├── .env                  # Your API key (never commit this)
├── .env.example          # Template for .env
├── .gitignore
├── README.md
├── requirements.txt
├── data/                 # Put your documents here (PDF, DOCX, TXT)
├── db/                   # ChromaDB stores vectors here (auto-created)
└── src/
    ├── __init__.py
    ├── config.py         # All constants (models, paths, chunk sizes, etc.)
    ├── ingest.py         # Ingestion pipeline (extract → chunk → embed → store)
    ├── query.py          # Query pipeline (embed → retrieve → generate)
    └── main.py           # Streamlit web UI
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd document-qa-bot
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

```bash
cp .env.example .env
```

Open `.env` and add your Google Gemini API key:

```
GEMINI_API_KEY=your_key_here
```

Get a free key at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).

### 5. Add your documents

Copy your PDF, DOCX, or TXT files into the `data/` folder:

```
data/
├── business_report.pdf
├── science_paper.pdf
└── factsheet.docx
```

---

## Usage

### Option A — Streamlit Web UI (recommended)

```bash
streamlit run src/main.py
```

1. Open [http://localhost:8501](http://localhost:8501) in your browser.
2. Click **▶ Run Ingestion** in the sidebar to index your documents.
3. Type any question in the chat box.

### Option B — Command Line

**Step 1: Ingest documents**

```bash
python src/ingest.py
```

**Step 2: Query interactively**

```bash
python src/query.py
```

---

## How It Works

### Ingestion (`ingest.py`)

1. **Extract** — Each file in `data/` is parsed with the appropriate library (`pypdf` for PDFs, `python-docx` for DOCX, built-in `open()` for TXT). Text is extracted page-by-page and paired with metadata: `{source, page}`.

2. **Chunk** — Long pages are split into overlapping character windows (`chunk_size=1000`, `chunk_overlap=200`). Overlap ensures that facts sitting near a boundary appear in at least one complete chunk.

3. **Embed & Store** — Each chunk is embedded with `text-embedding-004` and stored in a local ChromaDB collection using cosine distance. The DB is persisted in `db/` so re-ingestion only happens when documents change.

### Querying (`query.py`)

1. **Embed the query** — The user's question is converted to a vector using the same embedding model.

2. **Retrieve top-k chunks** — ChromaDB performs an approximate nearest-neighbour search and returns the `k=4` most semantically similar chunks.

3. **Build grounded prompt** — Each chunk is labelled with its source citation and assembled into a context payload. The system prompt strictly instructs the LLM to use only this context.

4. **Generate answer** — Gemini produces a factual answer with inline citations. If it cannot find the answer in the context, it says so explicitly.

---

## Design Decisions

**Why ChromaDB over FAISS?**
ChromaDB is disk-persistent out of the box and supports metadata filtering without extra configuration, making it ideal for a project where the document set changes infrequently.

**Why character-level chunking over sentence splitting?**
Sentence tokenisation adds a dependency (e.g. NLTK/spaCy) and can fail on poorly formatted PDFs. Fixed-size character windows with overlap are reliable across all document types.

**Why separate `ingest.py` from `query.py`?**
Embedding documents costs API tokens and time. By separating concerns, the heavy lifting happens once at ingestion; every subsequent query hits only the lightweight retrieval and generation path.

**Hallucination prevention**
The system prompt explicitly forbids the model from using external knowledge. If no relevant context is retrieved, the model returns a canned "I cannot find the answer" message rather than guessing.
