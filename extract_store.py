import json
import os
from pypdf import PdfReader
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
import tiktoken  # Moved tokenizer inside here

TEMP_DIR = "temp"
CHUNKS_FILE = os.path.join(TEMP_DIR, "chunks.json")

def split_text(text, chunk_size=500, overlap=50):
    """Splits text into chunks with overlap, using OpenAI tokenizer."""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i : i + chunk_size]
        chunks.append(enc.decode(chunk))

    return chunks

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def extract_text_from_epub(epub_path):
    """Extracts text from an EPUB file."""
    text = ""
    book = epub.read_epub(epub_path)
    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text += soup.get_text() + "\n"
    return text.strip()

def extract_and_store_chunks(temp_file, file_type):
    """Extracts text, splits into chunks, and saves as JSON."""
    os.makedirs(TEMP_DIR, exist_ok=True)

    text = extract_text_from_pdf(temp_file) if file_type == "pdf" else extract_text_from_epub(temp_file)
    chunks = split_text(text)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)

    print(f"Chunks saved to {CHUNKS_FILE}")
    return CHUNKS_FILE
