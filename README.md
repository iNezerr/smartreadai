# SmartReadAI

**SmartReadAI** is an AI-powered document search and question-answering system that allows users to extract, process, and query information from books, PDFs, and EPUBs. Instead of just searching for keywords, it intelligently understands and responds to queries based on the content of the documents.

## Features

- Extracts text from **PDFs** and **EPUBs**
- Splits text into manageable chunks for processing
- Embeds text using a vector-based model
- Stores embeddings in **FAISS** for efficient similarity search
- Answers questions based on document content using AI

## Project Structure

```

.

├── embed.py             # Generates embeddings for extracted text

├── extract_epub.py      # Extracts text from EPUB files

├── extract_pdf.py       # Extracts text from PDF files

├── split.py             # Splits extracted text into smaller chunks

├── store_faiss.py       # Stores embeddings in FAISS for retrieval

├── query_faiss.py       # Searches FAISS index to answer queries

├── externals/           # Folder containing input book files (PDF, EPUB)

├── extracted_text.txt   # Processed extracted text

├── faiss_index.bin      # FAISS index file for fast querying

├── faiss_texts.json     # JSON file storing text chunk mappings

├── nltk_data/           # NLTK tokenizer data

├── .env                 # Stores API keys and environment variables

├── .gitignore           # Git ignore file to exclude generated files

└── README.md            # Project documentation

```

## Setup & Installation

###**1. Clone the Repository**

```bash

gitclonehttps://github.com/iNezerr/smartreadai.git

cdsmartreadai

```

###**2. Install Dependencies**

Ensure you have Python 3.8+ installed, then install required packages:

```bash

pipinstall-rrequirements.txt

```

###**3. Set Up Environment Variables**

Create a `.env` file and add your API keys:

```

OPENAI_API_KEY=your_api_key_here

```

###**4. Download NLTK Data**

Run the following command to download necessary tokenizer data:

```python

import nltk

nltk.download('punkt')

```

###**5. Run the Pipeline**

####**Extract text from books:**

```bash

pythonextract_pdf.pyexternals/book_1.pdf

pythonextract_epub.pyexternals/book_1.epub

```

####**Split text into chunks:**

```bash

pythonsplit.py

```

####**Generate embeddings and store in FAISS:**

```bash

pythonembed.py

pythonstore_faiss.py

```

####**Query the system for answers:**

```bash

pythonquery_faiss.py"What is photosynthesis?"

```

## Usage Example

```

User: What is photosynthesis?

SmartReadAI: Photosynthesis is the process by which plants convert sunlight into energy, using carbon dioxide and water to produce oxygen and glucose.

```

## Roadmap

- [ ] Improve AI-generated responses for better understanding
- [ ] Web interface for interactive queries
- [ ] Support for more document formats

## Contributing

Contributions are welcome! Feel free to submit issues and pull requests.

## License

MIT License. See `LICENSE` for more details.
