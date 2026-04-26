"""
Personal Knowledge Base Agent  ·  main.py
──────────────────────────────────────────
FastAPI server — serves the REST API + the web UI from index.html

Run in VS Code:  Press F5  (uses .vscode/launch.json)
Run in terminal: uvicorn main:app --reload --port 8000
"""

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # must happen before importing agent
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from agent_langchain import ask_agent
from agent import PKBAgent

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Personal Knowledge Base Agent",
    description="RAG-powered note-taking with Gemini 2.5 Flash",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = PKBAgent()

# ── Models ─────────────────────────────────────────────────────────────────────

class NoteCreate(BaseModel):
    title:   str
    content: str
    tags:    list[str] = []

class QuestionRequest(BaseModel):
    question: str

class SearchRequest(BaseModel):
    query: str

# ── Serve the UI ───────────────────────────────────────────────────────────────

UI_PATH = Path(__file__).parent / "index.html"

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    if not UI_PATH.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return HTMLResponse(UI_PATH.read_text(encoding="utf-8"))

# ── Notes ──────────────────────────────────────────────────────────────────────

@app.post("/notes", status_code=201)
async def add_note(note: NoteCreate):
    if not note.title.strip():
        raise HTTPException(400, "Title cannot be empty")
    if not note.content.strip():
        raise HTTPException(400, "Content cannot be empty")
    return agent.add_note(note.title, note.content, note.tags)

@app.get("/notes")
async def list_notes():
    return agent.list_notes()

@app.get("/notes/{note_id}")
async def get_note(note_id: str):
    note = agent.get_note(note_id)
    if not note:
        raise HTTPException(404, f"Note '{note_id}' not found")
    return note

@app.delete("/notes/{note_id}")
async def delete_note(note_id: str):
    if not agent.delete_note(note_id):
        raise HTTPException(404, f"Note '{note_id}' not found")
    return {"deleted": note_id, "status": "ok"}

# ── Search ─────────────────────────────────────────────────────────────────────

@app.post("/search")
async def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    return agent.search(req.query)

# ── RAG Q&A ───────────────────────────────────────────────────────────────────

@app.post("/ask")
async def ask(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    answer = agent.answer_question(req.question)
    return {"question": req.question, "answer": answer}

# ── File Upload ────────────────────────────────────────────────────────────────

ALLOWED_TYPES = {".txt", ".md"}

@app.post("/upload", status_code=201)
async def upload_file(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_TYPES:
        raise HTTPException(400, f"Only {ALLOWED_TYPES} files are supported")
    raw = await file.read()
    text = raw.decode("utf-8", errors="ignore").strip()
    if not text:
        raise HTTPException(400, "Uploaded file is empty")
    return agent.add_note(
        title   = Path(file.filename).stem.replace("-", " ").replace("_", " ").title(),
        content = text,
        tags    = ["uploaded", ext.lstrip(".")],
    )

# ── Clusters ───────────────────────────────────────────────────────────────────

@app.post("/cluster")
async def cluster():
    return agent.cluster_topics()

# ── Stats ──────────────────────────────────────────────────────────────────────

@app.get("/stats")
async def stats():
    return agent.get_stats()

# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "notes": len(agent.notes), "model": "gemini-2.5-flash"}

class FetchRequest(BaseModel):
    topic: str

@app.post("/fetch")
async def fetch(req: FetchRequest):
    if not req.topic.strip():
        raise HTTPException(400, "Topic cannot be empty")

    return agent.fetch_from_web(req.topic)
# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    print(f"\n🧠  PKB Agent running → http://localhost:{port}\n")
    uvicorn.run("main:app", host=host, port=port, reload=True)

class LangRequest(BaseModel):
    question: str

@app.post("/agent")
async def lang_agent(req: LangRequest):
    return {"answer": ask_agent(req.question)}