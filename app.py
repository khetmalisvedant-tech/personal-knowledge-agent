"""
app.py  ·  HuggingFace Spaces Entry Point
──────────────────────────────────────────
Full-featured Gradio interface wrapping PKBAgent.
All features preserved: CRUD, search, RAG Q&A, web fetch, clusters.

Deploy: Push this repo to HF Spaces (SDK: Gradio)
        Set GEMINI_API_KEY in Space Secrets.
"""

import os
import json
from dotenv import load_dotenv
load_dotenv()

from agent import PKBAgent
from agent_langchain import ask_agent, get_llm

# ── Pre-warm ──────────────────────────────────────────────────
agent = PKBAgent()
try:
    get_llm()
    print("✅ LLM warmed up")
except Exception as e:
    print(f"⚠️  Warmup skipped: {e}")

# ── Helpers ───────────────────────────────────────────────────

def _note_to_str(note: dict) -> str:
    tags = ", ".join(note.get("tags", [])) or "none"
    return (
        f"📝 [{note['id']}] **{note['title']}**\n"
        f"Tags: {tags} · {note.get('word_count', 0)} words\n"
        f"{note['content'][:300]}{'...' if len(note['content']) > 300 else ''}"
    )

# ── Tab: Add Note ─────────────────────────────────────────────

def add_note_fn(title, content, tags_raw):
    title = title.strip()
    content = content.strip()
    if not title or not content:
        return "❌ Title and content are required."
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
    note = agent.add_note(title, content, tags)
    return f"✅ Note saved! ID: `{note['id']}` · {note['word_count']} words"

# ── Tab: List Notes ───────────────────────────────────────────

def list_notes_fn():
    notes = agent.list_notes()
    if not notes:
        return "📭 No notes yet. Add your first note!"
    return "\n\n---\n\n".join(_note_to_str(n) for n in reversed(notes))

# ── Tab: Delete Note ──────────────────────────────────────────

def delete_note_fn(note_id):
    note_id = note_id.strip()
    if not note_id:
        return "❌ Enter a note ID."
    ok = agent.delete_note(note_id)
    return f"✅ Note `{note_id}` deleted." if ok else f"❌ Note `{note_id}` not found."

# ── Tab: Search ───────────────────────────────────────────────

def search_fn(query):
    query = query.strip()
    if not query:
        return "❌ Enter a search query."
    results = agent.search(query)
    if not results:
        return "🔍 No matching notes found."
    lines = []
    for r in results:
        lines.append(f"🔎 **Score {r['score']:.3f}** · [{r['id']}] **{r['title']}**\n{r['content'][:200]}...")
    return "\n\n---\n\n".join(lines)

# ── Tab: Ask AI (RAG) ─────────────────────────────────────────

def ask_rag_fn(question):
    question = question.strip()
    if not question:
        return "❌ Enter a question."
    return agent.answer_question(question)

# ── Tab: LangChain Agent ─────────────────────────────────────

def ask_lc_fn(question):
    question = question.strip()
    if not question:
        return "❌ Enter a question."
    return ask_agent(question)

# ── Tab: Web Fetch ────────────────────────────────────────────

def fetch_web_fn(topic):
    topic = topic.strip()
    if not topic:
        return "❌ Enter a topic."
    result = agent.fetch_from_web(topic)
    if "error" in result:
        return f"❌ Error: {result['error']}"
    note = result["note"]
    return f"✅ {result['message']}\n\nSaved as: **{note['title']}** (ID: `{note['id']}`)\n{note['word_count']} words added."

# ── Tab: Topics / Clusters ────────────────────────────────────

def cluster_fn():
    clusters = agent.cluster_topics()
    if not clusters:
        return "❌ Need at least 2 notes to cluster topics."
    lines = []
    for c in clusters:
        kw = " · ".join(c.get("keywords", []))
        note_titles = ", ".join(n["title"] for n in c["notes"])
        lines.append(
            f"### 🗂 Topic: {c['topic'].title()}\n"
            f"**Keywords:** {kw}\n"
            f"**Notes ({c['count']}):** {note_titles}"
        )
    return "\n\n---\n\n".join(lines)

# ── Tab: Stats ────────────────────────────────────────────────

def stats_fn():
    s = agent.get_stats()
    top_tags = "\n".join(f"  • {tag}: {count}" for tag, count in s["top_tags"]) or "  (none)"
    recent = "\n".join(f"  • [{n['id']}] {n['title']}" for n in s["recent_notes"]) or "  (none)"
    return (
        f"📊 **Knowledge Base Stats**\n\n"
        f"**Total Notes:** {s['total_notes']}\n"
        f"**Total Words:** {s['total_words']}\n"
        f"**Unique Tags:** {s['unique_tags']}\n\n"
        f"**Top Tags:**\n{top_tags}\n\n"
        f"**Recent Notes:**\n{recent}"
    )

# ── Build Gradio UI ───────────────────────────────────────────

import gradio as gr

CSS = """
.gradio-container { max-width: 900px !important; margin: 0 auto; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="🧠 Personal Knowledge Base Agent",
    theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"),
    css=CSS,
) as demo:

    gr.Markdown(
        """
        # 🧠 Personal Knowledge Base Agent
        **RAG-powered AI tutor** · Gemini Flash · TF-IDF Search · KMeans Clustering
        > Add notes, search by meaning, ask questions grounded in your knowledge, fetch from the web.
        """
    )

    with gr.Tabs():

        # ── Add Note ──────────────────────────────────────────
        with gr.Tab("📝 Add Note"):
            with gr.Row():
                with gr.Column():
                    t_title   = gr.Textbox(label="Title", placeholder="e.g. Python List Comprehensions")
                    t_content = gr.Textbox(label="Content", lines=6, placeholder="Write your note here...")
                    t_tags    = gr.Textbox(label="Tags (comma-separated)", placeholder="python, tips, interview")
                    btn_add   = gr.Button("💾 Save Note", variant="primary")
                    out_add   = gr.Markdown()
            btn_add.click(add_note_fn, [t_title, t_content, t_tags], out_add)

        # ── Browse Notes ──────────────────────────────────────
        with gr.Tab("📚 All Notes"):
            btn_list = gr.Button("🔄 Refresh Notes", variant="secondary")
            out_list = gr.Markdown()
            btn_list.click(list_notes_fn, [], out_list)
            demo.load(list_notes_fn, [], out_list)

        # ── Search ────────────────────────────────────────────
        with gr.Tab("🔍 Search"):
            t_search  = gr.Textbox(label="Search Query", placeholder="e.g. recursion algorithms")
            btn_search = gr.Button("Search", variant="primary")
            out_search = gr.Markdown()
            btn_search.click(search_fn, [t_search], out_search)
            t_search.submit(search_fn, [t_search], out_search)

        # ── Ask AI (RAG) ──────────────────────────────────────
        with gr.Tab("🤖 Ask AI (RAG)"):
            gr.Markdown(
                "Ask questions answered using **your own notes**. "
                "Detects exam/deep/confused mode automatically."
            )
            with gr.Row():
                t_rag = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g. Explain binary search for exam / Why is O(log n) faster?",
                    scale=4,
                )
                btn_rag = gr.Button("Ask →", variant="primary", scale=1)
            out_rag = gr.Markdown()
            btn_rag.click(ask_rag_fn, [t_rag], out_rag)
            t_rag.submit(ask_rag_fn, [t_rag], out_rag)

            gr.Examples(
                examples=[
                    ["Explain binary search for my exam tomorrow"],
                    ["Why is quicksort faster than bubble sort?"],
                    ["I'm confused about recursion, explain simply"],
                    ["Give me a deep explanation of neural networks"],
                ],
                inputs=t_rag,
            )

        # ── LangChain Agent ───────────────────────────────────
        with gr.Tab("🦜 LangChain Agent"):
            gr.Markdown(
                "Uses **LangChain + Wikipedia tool** for broader knowledge queries."
            )
            with gr.Row():
                t_lc  = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g. Who invented the telephone? / Study tips for tomorrow",
                    scale=4,
                )
                btn_lc = gr.Button("Ask →", variant="primary", scale=1)
            out_lc = gr.Markdown()
            btn_lc.click(ask_lc_fn, [t_lc], out_lc)
            t_lc.submit(ask_lc_fn, [t_lc], out_lc)

            gr.Examples(
                examples=[
                    ["What is machine learning?"],
                    ["Study tips for my exam tomorrow"],
                    ["Who invented the telephone?"],
                    ["Explain neural networks simply"],
                ],
                inputs=t_lc,
            )

        # ── Web Fetch ─────────────────────────────────────────
        with gr.Tab("🌐 Web Fetch"):
            gr.Markdown("Fetch knowledge from **Wikipedia** or generate via **Gemini AI** and save it as a note.")
            t_web   = gr.Textbox(label="Topic", placeholder="e.g. Quantum Computing")
            btn_web = gr.Button("🌐 Fetch & Save", variant="primary")
            out_web = gr.Markdown()
            btn_web.click(fetch_web_fn, [t_web], out_web)

        # ── Topic Clusters ────────────────────────────────────
        with gr.Tab("🗂 Topics"):
            gr.Markdown("AI-discovered topic clusters across your knowledge base.")
            btn_cluster = gr.Button("Discover Topics", variant="primary")
            out_cluster = gr.Markdown()
            btn_cluster.click(cluster_fn, [], out_cluster)

        # ── Delete Note ───────────────────────────────────────
        with gr.Tab("🗑 Delete Note"):
            t_del   = gr.Textbox(label="Note ID", placeholder="e.g. a9c9a75c")
            btn_del = gr.Button("Delete", variant="stop")
            out_del = gr.Markdown()
            btn_del.click(delete_note_fn, [t_del], out_del)

        # ── Stats ─────────────────────────────────────────────
        with gr.Tab("📊 Stats"):
            btn_stats = gr.Button("Load Stats", variant="secondary")
            out_stats = gr.Markdown()
            btn_stats.click(stats_fn, [], out_stats)
            demo.load(stats_fn, [], out_stats)

demo.launch(server_name="0.0.0.0", server_port=7860)