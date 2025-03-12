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

# Ensure necessary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


@app.route("/")
def index():
    """Render the main HTML page."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handles file uploads and processes the document."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in ["pdf", "epub"]:
        return jsonify({"error": "Unsupported file type"}), 400

    temp_file = os.path.join(TEMP_DIR, f"temp.{file_ext}")
    file.save(temp_file)

    try: 
        chunks_file = extract_and_store_chunks(temp_file, file_ext)
        
        # Create FAISS index after processing the document
        create_faiss_index()

        return jsonify({"message": "File processed and indexed successfully!", "chunks_file": chunks_file}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
