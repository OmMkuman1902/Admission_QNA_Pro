# ✅ NEW IMPORT for Gemini
import google.generativeai as genai

# ✅ Other imports as you had
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.language_models import LLM
from typing import Optional, List
from langchain_core.outputs import Generation, LLMResult
from pydantic import PrivateAttr
import os

from dotenv import load_dotenv
load_dotenv()

# ✅ Step 1: Setup Gemini API
api_key = os.environ['GOOGLE_API_KEY']  # Replace with your API key
genai.configure(api_key=api_key)

class GeminiLLM(LLM):
    model_name: str = "models/gemini-pro"
    temperature: float = 0.1
    api_key: Optional[str] = None

    # 👉 Private attribute for internal model
    _model: any = PrivateAttr()

    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.1, api_key: Optional[str] = None):
        super().__init__()
        if api_key:
            genai.configure(api_key=api_key)
        self.model_name = model_name or self.model_name
        self.temperature = temperature
        self.api_key = api_key
        self._model = genai.GenerativeModel(self.model_name)

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self._model.generate_content(
            prompt,
            generation_config={"temperature": self.temperature}
        )
        return response.text

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)
    
 
# ✅ Step 3: Initialize Gemini LLM
llm = GeminiLLM(
    model_name="models/gemini-1.5-pro-latest",
    temperature=0.1,
)

# ✅ Step 6: Create Embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(model_name=model_name)
vectordb_file_path = "faiss_index"


def create_vector_db():
    # ✅ Step 5: Load your data
    loader = CSVLoader(file_path='college_questions_answers.csv', source_column="Question")
    data = loader.load()

   
    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=embedding_model)

    vectordb.save_local(vectordb_file_path)

def get_qa_chain():

     # 🚨 Load the FAISS index correctly
    vectordb = FAISS.load_local(
        folder_path=vectordb_file_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True  # ✅ Added here

    )
    retriever = vectordb.as_retriever(score_threshold = 0.8)

    

    # ✅ Step 11: Prompt Template
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "Answer" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    # ✅ Step 12: Create RetrievalQA Chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )
    return chain


from text_to_speech import speak_text
from dotenv import load_dotenv
import os

# === Your own modules ===
from speech_to_text import speech_to_text          # Your function for voice input

# === Load API Keys ===



# === Main Application ===
def main():
    print("🎙️ Listening for your question...")

    # Step 1: Get spoken question from Whisper
    question = speech_to_text()
    print(f"🧠 You asked: {question}")

    # Step 2: Create or load vector database
    #create_vector_db()

    # Step 3: Load the LangChain QA chain
    chain = get_qa_chain()

    # Step 4: Ask the question to the chain
    response = chain.invoke({"query": question})
    answer = response["result"]
    print(f"🤖 Answer: {answer}")

    # Step 5: Speak the answer
    speak_text(answer)

if __name__ == "__main__":
    main()


