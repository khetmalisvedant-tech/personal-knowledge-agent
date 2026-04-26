"""
Personal Knowledge Base Agent  ·  agent.py
───────────────────────────────────────────
Enhanced Version (Student-Aware AI)

Features:
  • CRUD notes (pkb_data.json)
  • Semantic search (TF-IDF)
  • RAG Q&A (Gemini)
  • Topic clustering (KMeans)
  • 🧠 Student mindset detection
  • 🎯 Adaptive tutor responses
  • 🌐 Auto web-fetch when notes are empty
"""

import os, json, uuid, math, re, random
from datetime import datetime
from collections import defaultdict, Counter
import requests
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# ── Load ENV ─────────────────────────────────────────────────
_api_key = os.environ.get("GEMINI_API_KEY", "")
if not _api_key:
    raise EnvironmentError("❌ GEMINI_API_KEY not set in .env")

genai.configure(api_key=_api_key)
_model = genai.GenerativeModel("gemini-1.5-flash-latest")

# ── TOKENIZATION ─────────────────────────────────────────────

def _tokenize(text: str):
    return re.findall(r"[a-z]+", text.lower())

def _compute_idf(docs):
    N = len(docs) or 1
    df = defaultdict(int)
    for doc in docs:
        for t in set(doc):
            df[t] += 1
    return {t: math.log((N + 1) / (c + 1)) + 1 for t, c in df.items()}

def _tfidf_vector(tokens, idf):
    tf = Counter(tokens)
    total = len(tokens) or 1
    return {t: (c / total) * idf.get(t, 0) for t, c in tf.items()}

def _cosine(a, b):
    keys = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0

# ── KMEANS ───────────────────────────────────────────────────

def _kmeans(vectors, k, iters=25):
    if len(vectors) <= k:
        return list(range(len(vectors)))

    dim = len(vectors[0])
    centroids = [list(v) for v in random.sample(vectors, k)]
    assignments = [0] * len(vectors)

    def dist(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b))

    for _ in range(iters):
        for i, v in enumerate(vectors):
            assignments[i] = min(range(k), key=lambda c: dist(v, centroids[c]))
        for c in range(k):
            members = [vectors[i] for i, a in enumerate(assignments) if a == c]
            if members:
                centroids[c] = [sum(x[d] for x in members) / len(members) for d in range(dim)]

    return assignments

# ── DB ───────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(__file__), "pkb_data.json")

def _load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []

def _save_db(notes):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2)

# ── GEMINI ───────────────────────────────────────────────────

def _ask_gemini(prompt: str) -> str:
    try:
        res = _model.generate_content(prompt)
        return res.text.strip()
    except Exception as e:
        return f"⚠️ Gemini error: {e}"

# ── STUDENT MODE DETECTION ───────────────────────────────────

def detect_student_mode(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["exam", "important", "short", "quick", "revision", "bullet", "summary"]):
        return "exam"
    elif any(w in q for w in ["why", "how", "deep", "understand", "explain", "detail"]):
        return "deep"
    elif any(w in q for w in ["confused", "hard", "difficult", "simple", "beginner", "easy"]):
        return "simple"
    return "normal"

# ── AGENT ────────────────────────────────────────────────────

class PKBAgent:

    def __init__(self):
        self.notes = _load_db()

    # ── INDEX ───────────────────────────────────────────────

    def _rebuild_index(self):
        self.tokens = [_tokenize(n["title"] + " " + n["content"]) for n in self.notes]
        self.idf = _compute_idf(self.tokens)
        self.vecs = [_tfidf_vector(t, self.idf) for t in self.tokens]

    # ── CRUD ────────────────────────────────────────────────

    def add_note(self, title: str, content: str, tags: list = None) -> dict:
        note = {
            "id": str(uuid.uuid4())[:8],
            "title": title,
            "content": content,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "word_count": len(content.split()),
        }
        self.notes.append(note)
        _save_db(self.notes)
        return note

    def update_note(self, note_id: str, title: str = None, content: str = None, tags: list = None) -> dict | None:
        note = self.get_note(note_id)
        if not note:
            return None
        if title is not None:
            note["title"] = title
        if content is not None:
            note["content"] = content
            note["word_count"] = len(content.split())
        if tags is not None:
            note["tags"] = tags
        _save_db(self.notes)
        return note

    def list_notes(self) -> list:
        return self.notes

    def get_note(self, note_id: str) -> dict | None:
        return next((n for n in self.notes if n["id"] == note_id), None)

    def delete_note(self, note_id: str) -> bool:
        before = len(self.notes)
        self.notes = [n for n in self.notes if n["id"] != note_id]
        if len(self.notes) < before:
            _save_db(self.notes)
            return True
        return False

    # ── SEARCH ──────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list:
        if not self.notes:
            return []

        self._rebuild_index()
        q_vec = _tfidf_vector(_tokenize(query), self.idf)

        scored = sorted(
            ((i, _cosine(q_vec, v)) for i, v in enumerate(self.vecs)),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            {**self.notes[i], "score": round(score, 4)}
            for i, score in scored[:top_k]
            if score > 0
        ]

    # ── RAG Q&A (SMART TUTOR) ───────────────────────────────

    def answer_question(self, question: str) -> str:
        # If no notes exist, auto-fetch from web first
        if not self.notes:
            web_data = self.fetch_from_web(question)
            if "error" in web_data:
                return "No notes found and web fetch failed. Please add some notes first."

        mode = detect_student_mode(question)
        relevant = self.search(question) or self.notes[:3]

        context = "\n\n---\n\n".join(
            f"[{n['title']}]\n{n['content'][:800]}"
            for n in relevant
        )

        instructions = {
            "exam":   "Give a concise bullet-point answer for exam revision. Include only key facts.",
            "deep":   "Explain in depth with examples, analogies, and follow-up insights.",
            "simple": "Explain as simply as possible. Use plain language like you are teaching a complete beginner.",
            "normal": "Answer clearly and helpfully.",
        }

        prompt = f"""You are a helpful AI tutor assisting a student.

{instructions[mode]}

Use ONLY the following knowledge base context to answer. Do not make up information.

=== KNOWLEDGE BASE ===
{context}
=== END CONTEXT ===

Question: {question}

Answer:"""

        return _ask_gemini(prompt)

    # ── CLUSTER ────────────────────────────────────────────

    def cluster_topics(self, n_clusters: int = 3) -> list:
        if len(self.notes) < 2:
            return []

        self._rebuild_index()
        vocab = sorted(self.idf.keys())

        def vec(v):
            return [v.get(t, 0) for t in vocab]

        matrix = [vec(v) for v in self.vecs]
        labels = _kmeans(matrix, min(n_clusters, len(self.notes)))

        clusters = defaultdict(list)
        for i, l in enumerate(labels):
            clusters[l].append(self.notes[i])

        result = []
        for cid, notes in clusters.items():
            words = []
            for n in notes:
                words += _tokenize(n["title"] + " " + n["content"])

            top = [w for w, _ in Counter(words).most_common(8)
                   if len(w) > 3][:5]  # skip short stop-words

            result.append({
                "topic": " / ".join(top[:2]) if top else f"Cluster {cid + 1}",
                "keywords": top,
                "count": len(notes),
                "notes": [{"id": n["id"], "title": n["title"]} for n in notes],
            })

        return result

    # ── STATS ──────────────────────────────────────────────

    def get_stats(self) -> dict:
        all_tags = [tag for n in self.notes for tag in n.get("tags", [])]
        unique_tags = list(set(all_tags))
        tag_counts = Counter(all_tags)
        recent = sorted(self.notes, key=lambda n: n.get("created_at", ""), reverse=True)[:5]

        return {
            "total_notes": len(self.notes),
            "total_words": sum(n.get("word_count", 0) for n in self.notes),
            "unique_tags": len(unique_tags),
            "top_tags": tag_counts.most_common(10),
            "recent_notes": [
                {"id": n["id"], "title": n["title"], "created_at": n.get("created_at", "")}
                for n in recent
            ],
        }

    # ── SUGGEST ACTIONS ────────────────────────────────────

    def suggest_actions(self, query: str) -> dict:
        if not self.notes:
            return {
                "type": "suggestion",
                "message": "Start by adding notes on your subjects.",
            }

        q = query.lower()

        if "study" in q or "learn" in q:
            return {
                "type": "plan",
                "message": "Break the topic into subtopics and study step-by-step.",
                "actions": ["Search related notes", "Revise concepts", "Practice questions"],
            }
        if "exam" in q:
            return {
                "type": "exam_strategy",
                "message": "Focus on revision and key topics.",
                "actions": ["Review notes", "Revise weak areas", "Practice questions"],
            }
        if "confused" in q:
            return {
                "type": "help",
                "message": "Try simplifying the topic.",
                "actions": ["Break into smaller parts", "Search simpler explanations"],
            }

        return {
            "type": "insight",
            "message": "Explore related topics for better understanding.",
            "actions": ["Search related notes", "Cluster topics", "Ask follow-up questions"],
        }

    # ── WEB FETCH ──────────────────────────────────────────

    def fetch_from_web(self, topic: str) -> dict:
        # Clean topic string
        clean = topic.split("about")[-1].strip()
        if not clean:
            clean = topic.strip()

        # 1️⃣ Try Wikipedia first
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(clean)}"
            res = requests.get(url, timeout=8)
            data = res.json()
            text = data.get("extract", "").strip()

            if text and len(text) > 100:
                note = self.add_note(
                    title=data.get("title", clean.title()),
                    content=text,
                    tags=["web", "wikipedia"],
                )
                return {"message": "Fetched from Wikipedia", "note": note}
        except Exception:
            pass

        # 2️⃣ Fallback → Gemini
        try:
            prompt = (
                f"Give a clear and detailed explanation of '{clean}' "
                f"for a student preparing for an exam. "
                f"Include key concepts, examples, and important facts."
            )
            response = _ask_gemini(prompt)
            note = self.add_note(
                title=clean.title(),
                content=response,
                tags=["ai-generated"],
            )
            return {"message": "Generated using Gemini AI", "note": note}

        except Exception as e:
            return {"error": str(e)}