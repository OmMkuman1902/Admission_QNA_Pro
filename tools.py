"""# tools.py
from langchain.tools import Tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

VDB_PATH = "faiss_index"

# Tool 1: Scrape college website
def scrape_college_website(url: str):
    loader = WebBaseLoader("https://iccs.ac.in/")
    documents = loader.load()
    return documents

# Tool 2: Save documents into FAISS
def store_to_vector_db(docs: list[Document]):
    vectordb = FAISS.from_documents(docs, embedding=embedding_model)
    vectordb.save_local(VDB_PATH)
    return "Stored into vector DB."

# Tool 3: Load vector DB for retrieval
def load_vector_db():
    return FAISS.load_local(
        folder_path=VDB_PATH,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

scrape_tool = Tool(
    name="ScrapeCollegeWebsite",
    func=scrape_college_website,
    description="Scrape college website content as documents. Input should be a URL."
)

store_tool = Tool(
    name="StoreToVectorDB",
    func=store_to_vector_db,
    description="Store a list of documents into FAISS vector database."
)

load_db_tool = Tool(
    name="LoadVectorDB",
    func=load_vector_db,
    description="Load the FAISS vector database."
)

__all__ = ["scrape_tool", "store_tool", "load_db_tool"]
"""