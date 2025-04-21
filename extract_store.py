import json
import os
from pypdf import PdfReader
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
import tiktoken  # Tokenizer for chunk size calculation

TEMP_DIR = "temp"
CHUNKS_FILE = os.path.join(TEMP_DIR, "chunks.json")

def split_text(text, chunk_size=500, overlap=50, max_chunks=None):
    """
    Splits text into chunks with overlap, using OpenAI tokenizer.
    
    Args:
        text: The text to split
        chunk_size: Number of tokens per chunk
        overlap: Number of tokens to overlap between chunks
        max_chunks: Optional limit on the number of chunks to create
        
    Returns:
        List of text chunks
    """
    # Use cl100k_base for better compatibility with new models
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks = []
    chunk_count = 0
    
    print(f"Splitting text into chunks (chunk size: {chunk_size}, overlap: {overlap})")
    print(f"Total tokens in document: {len(tokens)}")
    
    for i in range(0, len(tokens), chunk_size - overlap):
        if max_chunks is not None and chunk_count >= max_chunks:
            print(f"Reached maximum chunk count limit of {max_chunks}")
            break
            
        chunk = tokens[i : i + chunk_size]
        chunks.append(enc.decode(chunk))
        chunk_count += 1
        
        # Print progress every 20 chunks
        if chunk_count % 20 == 0:
            print(f"  Created {chunk_count} chunks so far...")
    
    print(f"Created {len(chunks)} chunks in total")
    return chunks

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file with progress reporting."""
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            total_pages = len(reader.pages)
            print(f"Processing PDF with {total_pages} pages")
            
            for i, page in enumerate(reader.pages):
                # Report progress every 10 pages
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{total_pages} pages")
                text += page.extract_text() + "\n"
                
        print(f"Finished extracting text from PDF: {pdf_path}")
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        raise

def extract_text_from_epub(epub_path):
    """Extracts text from an EPUB file with progress reporting."""
    text = ""
    try:
        book = epub.read_epub(epub_path)
        items = list(book.get_items())
        document_items = [item for item in items if item.get_type() == ITEM_DOCUMENT]
        
        print(f"Processing EPUB with {len(document_items)} document items")
        
        for i, item in enumerate(document_items):
            # Report progress every 10 items
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(document_items)} document items")
                
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text += soup.get_text() + "\n"
            
        print(f"Finished extracting text from EPUB: {epub_path}")
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from EPUB: {str(e)}")
        raise

def extract_and_store_chunks(temp_file, file_type, chunk_size=500, overlap=50, max_chunks=None):
    """
    Extracts text, splits into chunks, and saves as JSON.
    
    Args:
        temp_file: Path to the temporary file
        file_type: Type of file ("pdf" or "epub")
        chunk_size: Number of tokens per chunk
        overlap: Number of tokens to overlap between chunks
        max_chunks: Optional limit on the number of chunks (to avoid timeouts)
    
    Returns:
        Path to the saved chunks file
    """
    os.makedirs(TEMP_DIR, exist_ok=True)

    print(f"Starting text extraction from {file_type.upper()} file: {temp_file}")
    
    # Extract text based on file type
    text = extract_text_from_pdf(temp_file) if file_type == "pdf" else extract_text_from_epub(temp_file)
    
    # Split text into chunks
    chunks = split_text(text, chunk_size=chunk_size, overlap=overlap, max_chunks=max_chunks)

    # Save chunks to file
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)

    print(f"Chunks saved to {CHUNKS_FILE}")
    return CHUNKS_FILE
