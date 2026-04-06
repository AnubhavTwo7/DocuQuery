# 🧠 RAG System — Universal Document Q&A

A production-ready **Retrieval-Augmented Generation (RAG)** system that lets you upload any document (PDF, TXT, DOCX) and ask questions about it. Powered by a hybrid FAISS + BM25 retrieval engine and streaming LLM responses via [OpenRouter](https://openrouter.ai/).

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-CDN-06B6D4?style=flat&logo=tailwindcss&logoColor=white)
![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-7C3AED?style=flat)

---

## ✨ Features

- 📄 **Universal Document Ingestion** — Upload PDF, TXT, or DOCX files and index them instantly
- 🔍 **Hybrid Retrieval** — Combines dense FAISS vector search with sparse BM25 keyword search for superior context recall
- 🤖 **Streaming AI Responses** — Live token-by-token streaming via Server-Sent Events (SSE)
- 🧮 **LaTeX Math Rendering** — Inline and display math rendered via MathJax in real-time as the response streams
- 📝 **Universal Markdown Formatting** — Full support for tables, blockquotes, code blocks, headings, lists, and links
- 📖 **Source Viewer Panel** — Click any source citation to view the exact document chunk in a simulated PDF viewer
- 🗑️ **Document Management** — List and delete uploaded documents from the knowledge base
- 🧬 **Semantic Deduplication** — Retrieval engine deduplicates near-identical chunks for cleaner context

---

## 🏗️ Architecture

```
rag_system/
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── document.py      # Upload, list, delete endpoints
│   │       └── query.py         # SSE streaming Q&A endpoint
│   ├── core/
│   │   └── config.py            # Environment & settings
│   ├── models/
│   │   ├── api_models.py        # Pydantic request/response schemas
│   │   └── domain_models.py     # Internal data structures
│   ├── services/
│   │   ├── ingestion.py         # PDF/TXT/DOCX parsing & chunking
│   │   ├── retrieval.py         # HybridRetriever (FAISS + BM25) + OpenRouter Embeddings
│   │   ├── generation.py        # LLM streaming via LangChain + OpenRouter
│   │   └── evaluation.py        # LLM-as-a-Judge evaluation utilities
│   └── main.py                  # FastAPI app entry point
├── static/
│   └── index.html               # Full-stack frontend (TailwindCSS + Vanilla JS)
├── data/                        # Persisted FAISS index, BM25 corpus & metadata
├── .env                         # API keys (not committed to git)
├── requirements.txt
└── README.md
```

### Retrieval Pipeline

```
User Query
    │
    ▼
Query Rewriting (LLM)  <-- ACTIVE
    │
    ├──► FAISS Dense Search (OpenRouter Embeddings)
    │
    └──► BM25 Sparse Search
         │
         ▼
    Reciprocal Rank Fusion + Semantic Deduplication
         │
         ▼
    Top-K Chunks → LLM (Streaming) → SSE Response to Browser
```


---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- An [OpenRouter](https://openrouter.ai/) API key (free tier available)

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd rag_system

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. Run the Server

```bash
uvicorn app.main:app --reload
```

Open your browser at **http://localhost:8000**

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the frontend UI |
| `POST` | `/api/upload` | Upload a document (multipart/form-data) |
| `GET` | `/api/documents` | List all indexed documents |
| `DELETE` | `/api/documents/{filename}` | Remove a document from the index |
| `POST` | `/api/query` | Stream a RAG response (SSE) |

### Query Request Body

```json
{
  "query": "What is the Ginzburg-Landau order parameter?",
  "top_k": 5
}
```

### SSE Stream Event Types

```
data: {"type": "sources", "data": [...]}   // Source chunks (sent first)
data: {"type": "content", "data": "..."}   // LLM token chunks
data: [DONE]                               // Stream complete
```

---

## 🤖 Models Used

| Role | Model | Provider |
|------|-------|----------|
| **LLM (Generation)** | `qwen/qwen3.6-plus-preview:free` | OpenRouter |
| **Embeddings** | `nvidia/llama-nemotron-embed-vl-1b-v2:free` | OpenRouter |

> Both models are available on OpenRouter's **free tier**. You can swap them out in `app/services/generation.py` and `app/services/retrieval.py`.

---

## 💡 Design Decisions

- **Soft Deletion**: When a document is deleted, metadata and BM25 entries are removed but FAISS embeddings are left as "ghost" vectors. This avoids expensive full-index rebuilds when deletion is infrequent.
- **SSE over WebSockets**: The query pipeline uses standard HTTP streaming (Server-Sent Events) rather than WebSockets, keeping the backend stateless and simpler to deploy.
- **Math Protection Layer**: Before passing LLM output to the Markdown parser, LaTeX blocks are regex-extracted, stored in a vault, and re-injected after parsing — preventing Markdown from corrupting math syntax.

---

## 📦 Dependencies

```
fastapi        — Web framework
uvicorn        — ASGI server
langchain      — LLM orchestration
langchain-openai — OpenAI-compatible API client
faiss-cpu      — Dense vector index
rank_bm25      — Sparse keyword retrieval
pypdf          — PDF parsing
python-multipart — File upload support
python-dotenv  — Environment variable loading
```

---

## 📝 License

MIT License — feel free to use, modify, and distribute.
