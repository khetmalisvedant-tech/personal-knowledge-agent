"""
agent_langchain.py  ·  SPEED-OPTIMIZED  (Modern LangChain)
──────────────────────────────────────────────────────────
Compatible with: langchain>=0.2.0, langchain-google-genai>=1.0.0

Fixes from original:
  FIX 1 — LLM cached with lru_cache
  FIX 2 — Agent cached with lru_cache
  FIX 3 — WikipediaAPIWrapper: top_k=1, chars=600
  FIX 4 — max_iterations=2 with early stopping
  FIX 5 — Smart router bypasses agent for simple queries
  FIX 6 — Replaced deprecated .predict() with .invoke()
  FIX 7 — Updated to langchain-community for WikipediaAPIWrapper
  FIX 8 — Graceful fallback chain so it never crashes silently
"""

import os
import functools
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# ─────────────────────────────────────────────────────────────
# FIX 1: Cache the LLM — created ONCE, reused forever
# ─────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY / GEMINI_API_KEY is missing — add it to your .env or Space Secrets")

    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.3,
        google_api_key=api_key,
        max_output_tokens=512,
        timeout=30,
    )

# ─────────────────────────────────────────────────────────────
# FIX 3 + FIX 7: Wikipedia via langchain-community, slim results
# ─────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def get_wiki_tool():
    try:
        from langchain_community.utilities import WikipediaAPIWrapper
    except ImportError:
        from langchain.utilities import WikipediaAPIWrapper  # older fallback

    wiki = WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=600,
    )
    return Tool(
        name="Wikipedia",
        func=wiki.run,
        description="Search factual information about people, places, events, or concepts. Use only when needed.",
    )

# Study Advisor — instant, no API call
def _study_advisor(query: str) -> str:
    return (
        "📚 Study Tips:\n"
        "• Break the topic into smaller subtopics\n"
        "• Use active recall, not passive re-reading\n"
        "• Spaced repetition: revise after 1 day, 1 week, 1 month\n"
        "• Practice past questions under timed conditions\n"
        "• Focus extra time on weak areas\n"
        "• Teach the concept to someone else to test understanding\n"
    )

@functools.lru_cache(maxsize=1)
def get_study_tool():
    return Tool(
        name="Study Advisor",
        func=_study_advisor,
        description="Returns study strategies and revision tips for students.",
    )

# ─────────────────────────────────────────────────────────────
# FIX 2 + FIX 4: Cache the agent
# ─────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def get_agent():
    llm = get_llm()
    tools = [get_wiki_tool(), get_study_tool()]

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        max_iterations=2,
        early_stopping_method="generate",
        handle_parsing_errors=True,
    )

# ─────────────────────────────────────────────────────────────
# FIX 5: Smart router — bypass agent for fast-path queries
# ─────────────────────────────────────────────────────────────
_STUDY_KEYWORDS  = {"study", "exam", "revision", "revise", "tips", "memorize",
                    "marks", "prepare", "score", "test", "learn"}
_BYPASS_KEYWORDS = {"hi", "hello", "hey", "thanks", "thank", "bye", "help",
                    "what can", "who are you", "how are you"}

def _should_bypass_agent(question: str) -> bool:
    q     = question.strip().lower()
    words = set(q.split())

    if words & _BYPASS_KEYWORDS:
        return True
    if words & _STUDY_KEYWORDS:
        return True
    if len(words) <= 4:
        return True

    return False

# ─────────────────────────────────────────────────────────────
# FIX 6: Use .invoke() instead of deprecated .predict()
# ─────────────────────────────────────────────────────────────
def _llm_answer(question: str) -> str:
    """Direct LLM call — no agent overhead."""
    llm = get_llm()
    try:
        result = llm.invoke(question)
        # result is an AIMessage; extract text content
        if hasattr(result, "content"):
            return result.content
        return str(result)
    except Exception as e:
        return f"⚠️ LLM error: {e}"

# ─────────────────────────────────────────────────────────────
# PUBLIC FUNCTION
# ─────────────────────────────────────────────────────────────
def ask_agent(question: str) -> str:
    """
    Fast path  (bypasses agent):  ~3–8s   — 1 Gemini call
    Agent path (uses tools):      ~8–20s  — 1–3 Gemini calls
    """
    if not question.strip():
        return "Please enter a question."

    try:
        # ⚡ Fast path
        if _should_bypass_agent(question):
            return _llm_answer(question)

        # 🤖 Agent path — may use Wikipedia tool
        agent = get_agent()
        result = agent.invoke({"input": question})

        # agent.invoke returns a dict with "output" key
        if isinstance(result, dict):
            return result.get("output", str(result))
        return str(result)

    except Exception as e:
        # Final fallback: answer directly with LLM
        try:
            return _llm_answer(question)
        except Exception:
            return f"⚠️ Error: {e}"