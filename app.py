import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from extract_store import extract_and_store_chunks
from query_faiss import create_faiss_index, get_answer
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
TEMP_DIR = "temp"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# Default processing parameters
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50
MAX_CHUNKS_LIMIT = 1000  # Safety limit for very large books

# Ensure necessary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


@app.route("/")
def index():
    """Render the landing page."""
    return render_template("landing.html")


@app.route("/chat")
def chat():
    """Render the chat interface page."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handles file uploads and processes the document with configurable parameters."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in ["pdf", "epub"]:
        return jsonify({"error": "Unsupported file type"}), 400

    # Get processing parameters from the request or use defaults
    chunk_size = int(request.form.get("chunk_size", DEFAULT_CHUNK_SIZE))
    overlap = int(request.form.get("overlap", DEFAULT_OVERLAP))
    
    # Option to limit processing for very large books
    # If set to "auto", will use our default limit
    max_chunks_str = request.form.get("max_chunks", "auto")
    if max_chunks_str == "auto":
        max_chunks = MAX_CHUNKS_LIMIT
    elif max_chunks_str.lower() == "none":
        max_chunks = None
    else:
        try:
            max_chunks = int(max_chunks_str)
        except ValueError:
            max_chunks = MAX_CHUNKS_LIMIT

    # Save the uploaded file
    temp_file = os.path.join(TEMP_DIR, f"temp.{file_ext}")
    file.save(temp_file)

    try: 
        # Process with configured parameters
        print(f"Processing file: {file.filename} with parameters: chunk_size={chunk_size}, overlap={overlap}, max_chunks={max_chunks}")
        chunks_file = extract_and_store_chunks(
            temp_file, 
            file_ext,
            chunk_size=chunk_size,
            overlap=overlap,
            max_chunks=max_chunks
        )
        
        # Create FAISS index after processing the document
        collection_name = create_faiss_index()

        return jsonify({
            "message": "File processed and indexed successfully!", 
            "chunks_file": chunks_file,
            "collection": collection_name,
            "stats": {
                "chunk_size": chunk_size,
                "overlap": overlap,
                "max_chunks": max_chunks
            }
        }), 200
    except Exception as e:
        error_message = str(e)
        print(f"Error processing file: {error_message}")
        
        # Check for common error types and provide helpful messages
        if "rate limit" in error_message.lower():
            error_message = "OpenAI API rate limit exceeded. Try again in a few minutes or consider using smaller chunk sizes."
        elif "timeout" in error_message.lower():
            error_message = "Process timed out while generating embeddings. Try setting a smaller max_chunks value."
        
        return jsonify({"error": error_message}), 500


@app.route("/ask", methods=["POST"])
def ask_question():
    """Handles user queries and returns AI-generated answers."""
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        answer = get_answer(question)
        return jsonify({"answer": answer}), 200
    except Exception as e:
        error_message = str(e)
        print(f"Error answering question: {error_message}")
        return jsonify({"error": error_message}), 500


if __name__ == "__main__":
    app.run(debug=True)
