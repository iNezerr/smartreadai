import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")

TEMP_DIR = "temp"
CHUNKS_FILE = os.path.join(TEMP_DIR, "chunks.json")
FAISS_INDEX_DIR = os.path.join(TEMP_DIR, "faiss_index")  # Now correctly treated as a folder

def create_faiss_index():
    """Loads text chunks, generates embeddings, and stores them in FAISS."""
    if not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE}")

    # Load stored chunks
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        raise ValueError("Chunks file is empty. Cannot create FAISS index.")

    # Generate embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY, model="text-embedding-ada-002")

    # Convert chunks into Document objects for FAISS
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Create FAISS vector store
    vector_db = FAISS.from_documents(documents, embeddings)

    # Save FAISS index
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    vector_db.save_local(FAISS_INDEX_DIR)  # ✅ Ensure it's saved as a directory

    print(f"FAISS index stored at {FAISS_INDEX_DIR}")

def get_answer(question):
    """Retrieves the best answer from FAISS for a given question using an LLM."""
    if not os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss")):
        raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX_DIR}")

    # Load FAISS index
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY, model="text-embedding-ada-002")
    vector_db = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    # Set up retriever
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Initialize LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=OPENAI_KEY)

    # Use the provided system prompt
    system_prompt = (
    "If you're asked who you are:\n"
    "You are a helpful assistant that answers questions based on a book.\n"
    "Answer the question below using the provided book content.\n"
    "If the user asks for a summary of a chapter or section, summarize it based on the given book content.\n"
    "If the chapter or section is not fully available, summarize what is available.\n"
    "If you can't find any relevant information, say: 'I don't have enough information to answer.'\n\n"
    "Book Content:\n{context}\n\n"
)

    # Create a Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # Create a chain that retrieves documents and processes them with the LLM
    qa_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

    # Get AI-generated answer
    response = qa_chain.invoke({"input": question})["answer"]

    return response
