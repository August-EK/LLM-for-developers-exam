# US Declaration of Independence — RAG Document Assistant

A Flask web application that lets you ask questions about the US Declaration of Independence (1776). The system uses Retrieval-Augmented Generation (RAG) to fetch relevant passages from the document before generating an answer with a local LLM via Ollama.

---

## Architecture Overview

```
User question
     │
     ▼
Sentence Transformer (all-MiniLM-L6-v2)
     │ encodes question into vector
     ▼
ChromaDB (vector search)
     │ returns top-3 relevant passages from document.pdf
     ▼
Ollama (tinyllama)
     │ generates answer based on retrieved context
     ▼
Flask → HTML response to user
```

### Components

| Component | Role |
|---|---|
| `ingest.py` | Reads `document.pdf`, splits by page, embeds with sentence-transformers, stores in ChromaDB |
| `app.py` | Flask server — handles `/` (UI) and `/ask` (RAG + LLM pipeline) |
| `templates/index.html` | Single-page frontend with question input and answer display |
| `chroma_db/` | Persistent vector store — generated once by `ingest.py`, reused on every query |
| Ollama (tinyllama) | Local LLM — runs entirely on your machine, no internet required for inference |
| `all-MiniLM-L6-v2` | Hugging Face embedding model — converts text to vectors for semantic search |

---

## RAG Implementation

1. **Ingestion** (`ingest.py`): The PDF is read page by page. Each page is encoded into a vector using `all-MiniLM-L6-v2` and stored in ChromaDB with a persistent client.
2. **Retrieval** (`app.py`): When a question arrives, it is encoded with the same model. ChromaDB performs a cosine similarity search and returns the 3 most relevant pages.
3. **Generation** (`app.py`): The retrieved passages are injected into the prompt as context. The local LLM generates an answer strictly based on that context.

---

## 4T's Prompt Engineering

The system prompt in `app.py` is structured using the 4T's framework:

| T | Value | Prompt line |
|---|---|---|
| **Traits** | Knowledgeable and neutral historian | `You are a knowledgeable and neutral historian (Traits)` |
| **Task** | Answer questions based only on retrieved excerpts | `Your task is to answer questions strictly based on the provided excerpts (Task)` |
| **Tone** | Clear, academic, informative | `Use a clear, academic and informative tone (Tone)` |
| **Target** | Curious students and researchers | `Your answers are aimed at curious students and researchers (Target)` |

The 4T's are visible directly in the system prompt in `app.py` and constrain the LLM to act as a document-grounded historian rather than a general-purpose assistant.

---

## MVP Definition

The MVP is a single-page web application that:
- accepts a natural language question about the Declaration of Independence
- retrieves the 3 most relevant passages from the document using RAG
- generates a grounded answer using a local LLM
- displays both the answer and the retrieved source context

What is **not** included in the MVP:
- support for multiple documents
- user authentication
- persistent conversation history
- production deployment or scalability

---

## Prerequisites

- Python 3.10 or newer
- [Ollama](https://ollama.com) installed and running
- `tinyllama` model pulled in Ollama:
  ```powershell
  ollama pull tinyllama
  ```
- The file `document.pdf` placed in the project root (US Declaration of Independence)

---

## Installation

```powershell
# 1. Clone or download the project
cd rag-doc-assistant

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Configuration

No configuration file is required. The model name and ChromaDB path are set directly in `app.py`:

| Setting | Value | Location |
|---|---|---|
| LLM model | `tinyllama` | `app.py` line 31 |
| ChromaDB path | `./chroma_db` | `app.py` line 7 |
| Embedding model | `all-MiniLM-L6-v2` | `app.py` line 9 |
| Retrieved chunks | 3 | `app.py` line 27 |
| Max tokens | 200 | `app.py` line 36 |

---

## Loading the RAG Source Material

Run `ingest.py` once before starting the app. This reads `document.pdf`, generates embeddings, and stores them in `chroma_db/`:

```powershell
python ingest.py
```

Expected output:
```
Ingested 7 pages into ChromaDB.
```

You only need to run this again if you change `document.pdf`.

---

## Starting the System

```powershell
python app.py
```

Open `http://localhost:5000` in your browser.

---

## How to Use

1. Type a question in the text field, e.g. *"What does the document say about equality?"*
2. Click **Ask**
3. Wait for the answer (first request takes ~2 minutes while the model loads into memory; subsequent requests take ~30 seconds)
4. Click **Show retrieved context** to see which passages from the document were used

---

## Known Limitations

- **Speed**: Running on CPU only — first response takes ~2 minutes as the model loads into RAM. Subsequent responses are ~30 seconds.
- **Chunk granularity**: The document is split by page, not by paragraph. A very long page may exceed the model's context window.
- **Single document**: The system is designed for one document only. Adding more documents would require re-running `ingest.py` with modifications.
- **tinyllama quality**: The model is small and fast but may produce shorter or less precise answers than larger models.
- **No conversation history**: Each question is answered independently with no memory of previous questions.
