from langchain.agents import initialize_agent, AgentType, Tool
from tools import scrape_website_tool, create_vectorstore_tool, query_knowledgebase_tool
from langchain_core.language_models import BaseLanguageModel
from gemini_llm import GeminiLLM # type: ignore
from speech_to_text import speech_to_text
from text_to_speech import speak_text

# ✅ Step 1: Initialize Gemini LLM
llm = GeminiLLM(
    model_name="models/gemini-1.5-pro-latest",
    temperature=0.1,
)

# ✅ Step 2: Define Tools for the Agent
scrape_tool = Tool(
    name="ScrapeWebsite",
    func=scrape_website_tool,
    description="Use this tool to scrape the content from a college website. Input should be the URL."
)

store_tool = Tool(
    name="CreateVectorStore",
    func=create_vectorstore_tool,
    description="Use this tool to store the scraped data into a FAISS vectorstore. Input should be the document text."
)

query_tool = Tool(
    name="QueryKnowledgebase",
    func=query_knowledgebase_tool,
    description="Use this tool to query from the stored college data. Input should be a natural language question."
)

# ✅ Step 3: Initialize Agent with Tools
agent = initialize_agent(
    tools=[scrape_tool, store_tool, query_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ✅ Step 4: Run Voice-based Workflow
def main():
    
    query = speech_to_text()
    print(f"🧠 You asked: {query}")

    print("🤖 Processing...")
    response = agent.run(query)

    print(f"💬 Response: {response}")
    speak_text(response)

if __name__ == "__main__":
    main()
