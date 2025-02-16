import pdfplumber

def extract_text_from_pdf(pdf_path, output_file="extracted_text.txt"):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text.strip())

    print(f"Text extracted and saved to {output_file}")

# Example Usage
if __name__ == "__main__":
    extract_text_from_pdf("externals/book_1.pdf")
