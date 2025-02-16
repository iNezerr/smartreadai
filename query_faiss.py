import faiss
import json
import openai
import numpy as np
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

FAISS_INDEX_FILE = "faiss_index.bin"
TEXTS_FILE = "faiss_texts.json"

def get_embedding(text):
    """Get OpenAI embedding for a given text."""
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response.data[0].embedding).astype("float32")

def search_faiss(query, top_k=3):
    """Search FAISS index for the most relevant text chunks."""
    # Load FAISS index
    index = faiss.read_index(FAISS_INDEX_FILE)

    # Load stored texts
    with open(TEXTS_FILE, "r", encoding="utf-8") as f:
        texts = json.load(f)

    # Convert query to embedding
    query_embedding = get_embedding(query).reshape(1, -1)

    # Search FAISS index
    distances, indices = index.search(query_embedding, top_k)

    # Return top results
    results = [texts[i] for i in indices[0] if i < len(texts)]
    return results

def generate_answer(query, retrieved_texts):
    """Use GPT to generate a proper answer from retrieved texts."""
    context = "\n\n".join(retrieved_texts)

    prompt = f"""
    You are a helpful assistant that answers questions based on a book.
    Answer the question below using the provided book content.
    If the book doesn't contain enough information, say you don't know.

    Book Content:
    {context}

    Question: {query}
    Answer:
    """

    response = openai.chat.completions.create(
        model="gpt-4",  # Or "gpt-3.5-turbo" if you want a cheaper option
        messages=[{"role": "system", "content": "You are a knowledgeable assistant."},
                  {"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

# Example Usage
if __name__ == "__main__":
    query = input("Enter your question: ")
    retrieved_texts = search_faiss(query)
    answer = generate_answer(query, retrieved_texts)
    print("\nAnswer:\n", answer)
