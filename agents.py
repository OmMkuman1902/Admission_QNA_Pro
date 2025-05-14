# agents.py
from crewai import Agent

from tools import scrape_college_info, speech_to_text, speak_text
from vector_db import store_in_vector_db, load_vector_db
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from main import GeminiLLM

# Gemini LLM
llm = GeminiLLM(model_name="models/gemini-1.5-pro-latest", temperature=0.1)

ScraperAgent = Agent(
    role="Scraper",
    goal="Extract college info and placements in real time",
    backstory="You are responsible for gathering latest data from a college website.",
    verbose=True,
    tools=[scrape_college_info],
)

IndexerAgent = Agent(
    role="Indexer",
    goal="Store college info in vector DB",
    backstory="You convert college info to embeddings and store in FAISS.",
    verbose=True,
    tools=[store_in_vector_db],
)

QueryAgent = Agent(
    role="Responder",
    goal="Answer user's question and speak the response",
    backstory="You answer queries by fetching vector DB info and using Gemini.",
    verbose=True,
    tools=[speech_to_text, speak_text]
)
