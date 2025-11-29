# ai_agent.py

import os
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# Validate API Keys
# -------------------------
def safe_env(var):
    value = os.getenv(var)
    if not value:
        raise Exception(f"{var} is missing in .env")
    return value

safe_env("GROQ_API_KEY")
safe_env("TAVILY_API_KEY")
safe_env("GOOGLE_API_KEY")

# -------------------------
# Imports
# -------------------------
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage
from langchain_core.messages import SystemMessage, HumanMessage


# -------------------------
# Load LLMs
# -------------------------
groq_llm = ChatGroq(model="llama-3.3-70b-versatile")
gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Default search tool
search_tool = TavilySearch(max_results=2)


# -------------------------
# Create AI Agent
# -------------------------
def get_response_from_ai_agent(query, allow_search, system_prompt, provider):

    # Select model provider
    if provider == "groq":
        llm = groq_llm
    elif provider == "google":
        llm = gemini_llm
    else:
        raise ValueError("Invalid model provider: must be 'groq' or 'google'")

    # Enable tools dynamically
    tools = [search_tool] if allow_search else []

    # Create ReAct agent
    agent = create_react_agent(model=llm, tools=tools)

    # Build message state
    state = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
    }

    # AI Output
    response = agent.invoke(state)
    messages = response.get("messages", [])

    ai_messages = [
        msg.content for msg in messages if isinstance(msg, AIMessage)
    ]

    return "\n".join(ai_messages)
