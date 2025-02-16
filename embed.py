import openai
import json
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def embed_chunks(input_file="chunks.txt", output_file="embeddings.json"):
    with open(input_file, "r", encoding="utf-8") as f:
        chunks = f.read().split("\n\n")

    embeddings = []
    for chunk in chunks:
        if chunk.strip():
            embedding = get_embedding(chunk)
            embeddings.append({"text": chunk, "embedding": embedding})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)

    print(f"Generated {len(embeddings)} embeddings and saved to {output_file}")

# Example Usage
if __name__ == "__main__":
    embed_chunks()
