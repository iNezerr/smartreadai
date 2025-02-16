import faiss
import json
import numpy as np

FAISS_INDEX_FILE = "faiss_index.bin"

def store_embeddings_faiss(embeddings_file="embeddings.json"):
    # Load embeddings from JSON file
    with open(embeddings_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [item["text"] for item in data]
    vectors = np.array([item["embedding"] for item in data]).astype("float32")

    # Create FAISS index
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    # Save index to file
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open("faiss_texts.json", "w", encoding="utf-8") as f:
        json.dump(texts, f)

    print(f"Stored {len(texts)} embeddings in FAISS and saved index to {FAISS_INDEX_FILE}")

# Example Usage
if __name__ == "__main__":
    store_embeddings_faiss()
