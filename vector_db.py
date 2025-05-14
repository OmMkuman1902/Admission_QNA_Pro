"""# vector_db.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb_path = "faiss_index"

def store_in_vector_db(text, metadata=None):
    doc = Document(page_content=text, metadata=metadata or {})
    vectordb = FAISS.from_documents([doc], embedding_model)
    vectordb.save_local(vectordb_path)

def load_vector_db():
    return FAISS.load_local(vectordb_path, embedding_model, allow_dangerous_deserialization=True)
"""