# 🧠 Personal Knowledge Base Agent

> **An AI-native knowledge management system** — capture, organize, and converse with your knowledge using semantic search, retrieval-augmented generation, and adaptive tutoring intelligence.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-1.5%20Flash-4285F4?style=flat-square&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)](https://langchain.com)
[![Gradio](https://img.shields.io/badge/Gradio-4.36%2B-FF7C00?style=flat-square&logo=gradio&logoColor=white)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Overview

The **Personal Knowledge Base Agent** is a full-stack AI application that transforms how you interact with your own knowledge. Rather than simply storing information, it makes your notes **queryable**, **searchable by meaning**, and **conversationally accessible** through a Retrieval-Augmented Generation (RAG) pipeline powered by Google Gemini.

Built for learners, researchers, and developers who need more than a note-taking app — this system understands context, detects intent, clusters related ideas, and adapts its responses based on how you're thinking.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                             │
│          Vanilla JS UI (index.html)  /  Gradio (app.py)        │
└────────────────────────────┬────────────────────────────────────┘
                             │  HTTP / REST
┌────────────────────────────▼────────────────────────────────────┐
│                     FastAPI Server (main.py)                    │
│         Routes: /notes · /search · /ask · /agent               │
│                 /upload · /cluster · /stats · /fetch            │
└────────────────────────────┬────────────────────────────────────┘
                             │
          ┌──────────────────┴──────────────────┐
          │                                     │
┌─────────▼──────────┐               ┌──────────▼─────────┐
│   PKBAgent         │               │  LangChain Agent   │
│   (agent.py)       │               │ (agent_langchain.py)│
│                    │               │                    │
│  ┌──────────────┐  │               │  ┌──────────────┐  │
│  │ CRUD Engine  │  │               │  │ Wikipedia    │  │
│  │ TF-IDF Index │  │               │  │ Tool         │  │
│  │ KMeans       │  │               │  │ Study Advisor│  │
│  │ Clustering   │  │               │  │ Tool         │  │
│  │ Student Mode │  │               │  └──────────────┘  │
│  │ Detector     │  │               │                    │
│  └──────────────┘  │               └──────────▲─────────┘
└─────────┬──────────┘                          │
          │                                     │
          └──────────────────┬──────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                  Google Gemini 1.5 Flash API                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    pkb_data.json  (Local Store)                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### Intelligent Knowledge Management
- Full **CRUD operations** on notes with title, content, tags, word count, and timestamp metadata
- **File ingestion** — import `.txt` and `.md` files directly into the knowledge base
- **Auto web-fetch** — when notes are sparse, the system automatically retrieves content from Wikipedia or generates it via Gemini

### Semantic Search (No Vector DB Required)
- Custom **TF-IDF vectorization** built from scratch in pure Python — zero external ML dependencies
- **Cosine similarity ranking** across the entire knowledge base
- Real-time index rebuilding on every query for freshness

### RAG Question Answering with Adaptive Tutoring
- **Student mindset detection** classifies queries into four modes:

  | Mode | Trigger Keywords | Response Style |
  |------|-----------------|----------------|
  | `exam` | exam, revision, bullet, quick, summary | Concise bullet points, key facts only |
  | `deep` | why, how, explain, detail, understand | In-depth with examples and analogies |
  | `simple` | confused, beginner, easy, difficult | Plain-language, first-principles explanation |
  | `normal` | *(default)* | Clear, helpful, balanced |

- Gemini answers **strictly from your notes** — no hallucinated external facts

### Topic Discovery via KMeans Clustering
- Pure-Python **KMeans implementation** — no scikit-learn dependency
- Automatically groups notes into thematic clusters
- Extracts top keywords per cluster for human-readable labels

### Dual-Agent Backend
- **PKBAgent** — core RAG pipeline, local knowledge operations
- **LangChain Agent** — broader knowledge via Wikipedia tool + smart router that bypasses agent overhead for simple queries

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| AI Model | Google Gemini 1.5 Flash | RAG generation, content synthesis |
| Agent Framework | LangChain 0.2+ | Tool-augmented reasoning |
| Web Framework | FastAPI | Async REST API server |
| UI (local) | Vanilla JS + HTML | Zero-dependency frontend |
| UI (cloud) | Gradio 4.36+ | HuggingFace Spaces deployment |
| Search | Custom TF-IDF | Semantic note retrieval |
| Clustering | Custom KMeans | Topic discovery |
| Storage | JSON (flat-file) | Portable, zero-config persistence |

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- A Google Gemini API key ([get one free](https://aistudio.google.com/app/apikey))

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/personal-knowledge-base-agent.git
cd personal-knowledge-base-agent
```

**2. Create and activate a virtual environment**
```bash
# macOS / Linux
python3 -m venv venv && source venv/bin/activate

# Windows
python -m venv venv && venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure environment variables**
```bash
cp .env.example .env
```

Open `.env` and set your key:
```env
GEMINI_API_KEY=AIzaSy...your_key_here
```

> **Note:** The free tier provides 15 requests/min and 1,000,000 tokens/day — sufficient for personal use.

**5. Start the server**
```bash
python main.py
# or
uvicorn main:app --reload --port 8000
```

**6. Open the application**

Navigate to [http://localhost:8000](http://localhost:8000)

---

## API Reference

All endpoints accept and return JSON. Interactive documentation is available at `/docs` (Swagger UI) and `/redoc`.

### Notes

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/notes` | Create a new note |
| `GET` | `/notes` | List all notes |
| `GET` | `/notes/{id}` | Retrieve a specific note |
| `PUT` | `/notes/{id}` | Update a note |
| `DELETE` | `/notes/{id}` | Delete a note |

### Intelligence

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `POST` | `/ask` | `{ "question": "..." }` | RAG Q&A from your notes |
| `POST` | `/search` | `{ "query": "..." }` | Semantic TF-IDF search |
| `POST` | `/agent` | `{ "question": "..." }` | LangChain agent with tools |
| `POST` | `/fetch` | `{ "topic": "..." }` | Fetch & save from web |
| `POST` | `/cluster` | — | Discover topic clusters |

### Utilities

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload a `.txt` or `.md` file |
| `GET` | `/stats` | Dashboard statistics |
| `GET` | `/health` | Server health check |

### Example Requests

```bash
# Add a note
curl -X POST http://localhost:8000/notes \
  -H "Content-Type: application/json" \
  -d '{"title":"Binary Search","content":"Binary search runs in O(log n) time by halving the search space each iteration. Requires a sorted array.","tags":["algorithms","dsa"]}'

# Semantic search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"efficient search algorithms"}'

# RAG Q&A (auto-detects exam mode)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Give me a quick exam summary of binary search"}'

# Discover topic clusters
curl -X POST http://localhost:8000/cluster

# Stats overview
curl http://localhost:8000/stats
```

---

## Project Structure

```
pkb_agent/
│
├── agent.py                  # Core PKBAgent — CRUD, TF-IDF, KMeans, RAG, student mode
├── agent_langchain.py        # LangChain agent — Wikipedia tool, smart router, LLM caching
├── main.py                   # FastAPI server — all REST routes, file upload, CORS
├── app.py                    # Gradio UI — HuggingFace Spaces entry point
├── index.html                # Vanilla JS web UI — dark theme, 8 functional pages
│
├── pkb_data.json             # Auto-created knowledge store (excluded from Git)
├── requirements.txt          # Python dependencies
├── .env                      # Local secrets (never committed)
├── .env.example              # Template for environment setup
├── .gitignore
└── README.md
```

---

## Deployment

### HuggingFace Spaces

This project is Gradio-ready for one-click deployment on HuggingFace Spaces:

1. Push the repository to HuggingFace
2. Set `GEMINI_API_KEY` in **Space Secrets** (Settings → Repository secrets)
3. The Space will use `app.py` as the entry point automatically

Space metadata (top of `README.md`):
```yaml
---
title: Personal Knowledge Base Agent
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
---
```

### Local with VS Code Debugger

The project includes a complete VS Code debug configuration. Press `F5` to launch the server with full debugger support — set breakpoints anywhere in `agent.py` or `main.py` to inspect prompts, search vectors, or clustering inputs at runtime.

---

## How It Works

### TF-IDF Semantic Search

Notes are tokenized and converted to TF-IDF weighted vectors entirely in pure Python. Query vectors are computed the same way, and notes are ranked by **cosine similarity** — giving meaning-based results without any external vector database or embedding model.

### Retrieval-Augmented Generation

1. The query is vectorized and matched against all notes by cosine similarity
2. The top-5 semantically relevant excerpts (up to 800 characters each) are extracted
3. The student mode is detected from the query's vocabulary
4. A structured prompt combining the context and mode-specific instructions is sent to Gemini
5. Gemini responds **strictly from the provided context** — preventing hallucination

### KMeans Topic Clustering

All notes are vectorized over a shared vocabulary. A pure-Python KMeans implementation groups them into thematic clusters. The top high-frequency terms per cluster become human-readable topic labels.

### LangChain Agent with Smart Routing

The LangChain agent includes a router that classifies incoming queries. Simple questions (greetings, study tips, short phrases) bypass the agent entirely and go directly to Gemini — reducing latency from ~15s to ~4s. Only factual or research queries invoke the Wikipedia tool.

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | *(required)* | Google Gemini API key |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |

---

## Troubleshooting

| Symptom | Resolution |
|---------|-----------|
| `EnvironmentError: GEMINI_API_KEY not set` | Verify `.env` exists in the project root with a valid key |
| `ModuleNotFoundError` on startup | Run `pip install -r requirements.txt` with the venv activated |
| `401 API_KEY_INVALID` from Gemini | Regenerate your key at [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| Port 8000 already in use | Set `PORT=8001` in `.env` or terminate the conflicting process |
| Clustering returns a single cluster | Add at least 4–5 notes on distinct topics before clustering |
| Notes not persisting | Verify write permission in the project directory — `pkb_data.json` is auto-created |

---

## Roadmap

- [ ] Persistent vector embeddings with FAISS or ChromaDB
- [ ] Multi-user support with authentication
- [ ] Note versioning and change history
- [ ] Spaced repetition scheduler for review reminders
- [ ] Export to Notion, Obsidian, or Markdown vault
- [ ] AI-generated quizzes and flashcards from notes
- [ ] Voice input via Whisper integration
- [ ] Semantic deduplication across notes

---

## License

This project is released under the [MIT License](LICENSE). You are free to use, modify, and distribute it for personal or commercial purposes.

---

## Author

Built with precision and purpose for learners, researchers, and developers who believe **knowledge compounds** — and should be queryable.

> *"The value of a note is not in writing it. It's in finding it when you need it."*

---

<div align="center">

**[Documentation](#api-reference) · [Architecture](#architecture) · [Getting Started](#getting-started) · [Roadmap](#roadmap)**

</div>
