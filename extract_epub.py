from ebooklib import epub
from bs4 import BeautifulSoup

def extract_text_from_epub(epub_path, output_file="extracted_text.txt"):
    book = epub.read_epub(epub_path)
    text = ""

    for item in book.get_items():
        if item.get_type() == 9:  # EPUB text content
            soup = BeautifulSoup(item.get_body_content(), "html.parser")
            text += soup.get_text() + "\n"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text.strip())

    print(f"Text extracted and saved to {output_file}")

# Example Usage
if __name__ == "__main__":
    extract_text_from_epub("externals/book_1.epub")
