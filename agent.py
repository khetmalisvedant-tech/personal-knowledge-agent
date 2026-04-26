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
"""

import os, json, uuid, math, re, random
from datetime import datetime
from collections import defaultdict, Counter
import requests
from dotenv import load_dotenv
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
# ── Load ENV ─────────────────────────────────────────────────
load_dotenv()

_api_key = os.environ.get("GEMINI_API_KEY", "")
if not _api_key:
    raise EnvironmentError("❌ GEMINI_API_KEY not set in .env")

genai.configure(api_key=_api_key)
_model = genai.GenerativeModel("gemini-2.5-flash")

# ── TOKENIZATION ─────────────────────────────────────────────

def _tokenize(text: str):
    return re.findall(r"[a-z]+", text.lower())

def _compute_idf(docs):
    N = len(docs) or 1
    df = defaultdict(int)
    for doc in docs:
        for t in set(doc):
            df[t] += 1
    return {t: math.log((N + 1)/(c + 1)) + 1 for t, c in df.items()}

def _tfidf_vector(tokens, idf):
    tf = Counter(tokens)
    total = len(tokens) or 1
    return {t: (c/total)*idf.get(t, 0) for t, c in tf.items()}

def _cosine(a, b):
    keys = set(a) & set(b)
    dot = sum(a[k]*b[k] for k in keys)
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    return dot/(na*nb) if na and nb else 0

# ── KMEANS ───────────────────────────────────────────────────

def _kmeans(vectors, k, iters=25):
    if len(vectors) <= k:
        return list(range(len(vectors)))

    dim = len(vectors[0])
    centroids = [list(v) for v in random.sample(vectors, k)]
    assignments = [0]*len(vectors)

    def dist(a,b):
        return sum((x-y)**2 for x,y in zip(a,b))

    for _ in range(iters):
        for i,v in enumerate(vectors):
            assignments[i] = min(range(k), key=lambda c: dist(v, centroids[c]))

        for c in range(k):
            members = [vectors[i] for i,a in enumerate(assignments) if a==c]
            if members:
                centroids[c] = [sum(x[d] for x in members)/len(members) for d in range(dim)]

    return assignments

# ── DB ───────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(__file__), "pkb_data.json")

def _load_db():
    if os.path.exists(DB_PATH):
        return json.load(open(DB_PATH, encoding="utf-8"))
    return []

def _save_db(notes):
    json.dump(notes, open(DB_PATH,"w",encoding="utf-8"), indent=2)

# ── GEMINI ───────────────────────────────────────────────────

def _ask_gemini(prompt):
    res = _model.generate_content(prompt)
    return res.text.strip()

# ── STUDENT MODE DETECTION 🔥 ────────────────────────────────

def detect_student_mode(query):
    q = query.lower()

    if any(w in q for w in ["exam","important","short","quick","revision"]):
        return "exam"
    elif any(w in q for w in ["why","how","deep","understand","explain"]):
        return "deep"
    elif any(w in q for w in ["confused","hard","difficult"]):
        return "simple"
    return "normal"

# ── AGENT ────────────────────────────────────────────────────

class PKBAgent:

    def __init__(self):
        self.notes = _load_db()

    # ── INDEX ───────────────────────────────────────────────

    def _rebuild_index(self):
        self.tokens = [_tokenize(n["title"]+" "+n["content"]) for n in self.notes]
        self.idf = _compute_idf(self.tokens)
        self.vecs = [_tfidf_vector(t, self.idf) for t in self.tokens]

    # ── CRUD ────────────────────────────────────────────────

    def add_note(self, title, content, tags=None):
        note = {
            "id": str(uuid.uuid4())[:8],
            "title": title,
            "content": content,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "word_count": len(content.split())
        }
        self.notes.append(note)
        _save_db(self.notes)
        return note

    def list_notes(self):
        return self.notes

    def get_note(self, note_id):
        return next((n for n in self.notes if n["id"]==note_id), None)

    def delete_note(self, note_id):
        self.notes = [n for n in self.notes if n["id"] != note_id]
        _save_db(self.notes)
        return True

    # 🌐 If no notes found → fetch from internet
        # ── SEARCH ──────────────────────────────────────────────

    def search(self, query, top_k=5):
        if not self.notes:
            return []

        self._rebuild_index()
        q_vec = _tfidf_vector(_tokenize(query), self.idf)

        scored = sorted(
            ((i,_cosine(q_vec,v)) for i,v in enumerate(self.vecs)),
            key=lambda x:x[1], reverse=True
        )

        return [
            {**self.notes[i], "score": round(score,4)}
            for i,score in scored[:top_k] if score>0
        ]

    # ── 🔥 UPDATED RAG (SMART TUTOR) ────────────────────────

    def answer_question(self, question: str) -> str:
        if not self.notes:
            web_data = self.fetch_from_web(question)
            if "error" not in web_data:
                return web_data["note"]["content"]
        
        mode = detect_student_mode(question)

        relevant = self.search(question) or self.notes[:3]

        context = "\n\n---\n\n".join(
            f"[{n['title']}]\n{n['content'][:800]}"
            for n in relevant
        )

        if mode == "exam":
            instruction = "Give short bullet point answer for revision."
        elif mode == "deep":
            instruction = "Explain deeply with examples."
        elif mode == "simple":
            instruction = "Explain simply like beginner."
        else:
            instruction = "Answer clearly."

        prompt = f"""
            You are a helpful AI tutor.

            {instruction}

            Use ONLY this data:

            {context}

            Question: {question}
            """

        try:
            return _ask_gemini(prompt)
        except Exception as e:
            return f"Error: {e}"

    # ── CLUSTER ────────────────────────────────────────────

    def cluster_topics(self, n_clusters=3):
        if len(self.notes)<2:
            return []

        self._rebuild_index()
        vocab = sorted(self.idf.keys())

        def vec(v):
            return [v.get(t,0) for t in vocab]

        matrix = [vec(v) for v in self.vecs]
        labels = _kmeans(matrix, n_clusters)

        clusters = defaultdict(list)
        for i,l in enumerate(labels):
            clusters[l].append(self.notes[i])

        result = []
        for cid, notes in clusters.items():
            words = []
            for n in notes:
                words += _tokenize(n["content"])

            top = [w for w,_ in Counter(words).most_common(5)]

            result.append({
                "topic": " / ".join(top[:2]),
                "count": len(notes),
                "notes": [{"id":n["id"],"title":n["title"]} for n in notes]
            })

        return result

    # ── STATS ──────────────────────────────────────────────

    def get_stats(self):
        return {
            "total_notes": len(self.notes),
            "total_words": sum(n["word_count"] for n in self.notes)
        }
    def suggest_actions(self, query: str) -> dict:
        """
        Agent decides what user should do next
        """

        if not self.notes:
            return {
                "type": "suggestion",
                "message": "Start by adding notes on your subjects."
            }

        q = query.lower()

        if "study" in q or "learn" in q:
            return {
                "type": "plan",
                "message": "Break topic into subtopics and study step-by-step.",
                "actions": [
                    "Search related notes",
                    "Revise concepts",
                    "Practice questions"
                ]
            }

        if "exam" in q:
            return {
                "type": "exam_strategy",
                "message": "Focus on revision and key topics.",
                "actions": [
                    "Review notes",
                    "Revise weak areas",
                    "Practice questions"
                ]
            }

        if "confused" in q:
            return {
                "type": "help",
                "message": "Try simplifying the topic.",
                "actions": [
                    "Break into smaller parts",
                    "Search simpler explanations"
                ]
            }

        return {
            "type": "insight",
            "message": "Explore related topics for better understanding.",
            "actions": [
                "Search related notes",
                "Cluster topics",
                "Ask follow-up questions"
            ]
        }

    def fetch_from_web(self, topic: str) -> dict:
        topic = topic.split("about")[-1].strip()

    # 1️⃣ Try Wikipedia first
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
            res = requests.get(url).json()
            text = res.get("extract", "")

            if text:
                note = self.add_note(
                    title=topic.title(),
                    content=text,
                    tags=["web", "wiki"]
                )
                return {"message": "Fetched from Wikipedia", "note": note}
        except:
            pass

        # 2️⃣ Fallback → Gemini
        try:
            prompt = f"Explain {topic} clearly for students who is studying for their exam and want the detailed information."
            response = _ask_gemini(prompt)

            note = self.add_note(
                title=topic.title(),
                content=response,
                tags=["ai-generated"]
            )
            return {"message": "Generated using Gemini", "note": note}

        except Exception as e:
            return {"error": str(e)}