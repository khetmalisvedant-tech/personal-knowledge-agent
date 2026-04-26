from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.utilities import WikipediaAPIWrapper

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.3
)

# Wikipedia Tool
wiki = WikipediaAPIWrapper()

wiki_tool = Tool(
    name="Wikipedia",
    func=wiki.run,
    description="Search information from Wikipedia"
)

# Study Tool
def study_advisor(query):
    return "Break topic into parts, revise notes, and practice questions."

advisor_tool = Tool(
    name="Study Advisor",
    func=study_advisor,
    description="Gives study advice"
)

# Agent
agent = initialize_agent(
    tools=[wiki_tool, advisor_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def ask_agent(question):
    return agent.run(question)