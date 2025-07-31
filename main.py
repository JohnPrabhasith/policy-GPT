from flask import Flask, request, jsonify
from flask_cors import CORS
from services.load_docs import load_document_from_url
from services.chunks import get_text_chunks
from services.astra_client import get_vector_store
from services.chain import get_conversational_chain
from dotenv import load_dotenv
from auth import verify_token
from schema import QueryRequest, QueryResponse
import logging
import os

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "LLM-Powered Intelligent Query-Retrieval System is running!"})

@app.route("/api/v1/hackrx/run", methods=["POST"])
def run_submission():
    """
    Processes a document from a URL and answers questions about it.
    """
    try:
        payload = request.get_json()
        doc_url = payload["documents"]
        questions = payload["questions"]

        # 1. Input Document Processing
        logger.info("Loading document from URL....")
        documents = load_document_from_url(doc_url)
        if not documents:
            return jsonify({"error": "Failed to load or process the document."}), 500

        # 2. Text Chunking
        logger.info("Splitting document into chunks...")
        text_chunks = get_text_chunks(documents)
        if not text_chunks:
            return jsonify({"error": "Failed to split the document into chunks."}), 500

        # 3. Embedding and Vector Store
        logger.info("Creating vector store and embeddings...")
        vector_store = get_vector_store(text_chunks)
        if not vector_store:
            return jsonify({"error": "Failed to create vector store."}), 500

        # 4. Initialize Conversational Chain
        logger.info("Initializing conversational QA chain...")
        chain = get_conversational_chain(vector_store)
        if not chain:
            return jsonify({"error": "Failed to initialize the conversational chain."}), 500

        answers = []
        logger.info("Processing questions...")
        for i, question in enumerate(questions):
            # 5. Clause Retrieval (Similarity Search)
            print(f"---------------question '{i+1}'----------")
            # 6. Logic Evaluation (LLM)
            response = chain.invoke({"input": question})
            answers.append(response["answer"])

        # 7. JSON Output
        return jsonify({"answers": answers})

    except Exception as e:
        logger.error(f"An internal error occurred: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
