import nltk
import os
from nltk.tokenize import sent_tokenize

# Ensure NLTK uses the correct data directory
# nltk.data.path.append(os.path.expanduser("~/nltk_data"))

nltk.download("punkt_tab")

def split_text_into_chunks(input_file="extracted_text.txt", output_file="chunks.txt", max_words=200):
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = sent_tokenize(text)
    chunks = []
    chunk = []

    for sentence in sentences:
        chunk.append(sentence)
        if len(" ".join(chunk).split()) >= max_words:
            chunks.append(" ".join(chunk))
            chunk = []

    if chunk:
        chunks.append(" ".join(chunk))

    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n\n")

    print(f"Text split into {len(chunks)} chunks and saved to {output_file}")

# Example Usage
if __name__ == "__main__":
    split_text_into_chunks()
