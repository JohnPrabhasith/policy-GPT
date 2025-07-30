from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from services.load_docs import load_document_from_url
from services.chunks import get_text_chunks
from services.astra_client import get_vector_store
from services.chain import get_conversational_chain
from dotenv import load_dotenv
from auth import verify_token
from schema import QueryRequest, QueryResponse
import logging
import uvicorn
import os

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Intelligent Query Retrieval System",
    description="An LLM-powered system to process documents and answer questions.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "LLM-Powered Intelligent Query-Retrieval System is running!"}


@app.post(
    "/api/v1/hackrx/run",
    response_model=QueryResponse,
    tags=["Submissions"],
    summary="Run Document Query Submission",
    dependencies=[Depends(verify_token)]
)
async def run_submission(payload: QueryRequest):
    """
    Processes a document from a URL and answers questions about it.

    This endpoint performs the following steps:
    1.  Authenticates the request using a bearer token.
    2.  Downloads and parses the document from the provided URL.
    3.  Splits the document into manageable text chunks.
    4.  Generates embeddings and stores them in AstraDB.
    5.  For each question, retrieves relevant context and generates an answer using an LLM.
    """
    vector_store = None  # Initialize to None for safe cleanup
    try:
        doc_url = payload.documents
        questions = payload.questions
        # 1. Input Document Processing
        logger.info("Loading document from URL....")
        documents = load_document_from_url(doc_url)
        if not documents:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Failed to load or process the document.")

        # 2. Text Chunking
        logger.info("Splitting document into chunks...")
        text_chunks = get_text_chunks(documents)
        if not text_chunks:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Failed to split the document into chunks.")

        # 3. Embedding and Vector Store
        logger.info("Creating vector store and embeddings...")
        vector_store = get_vector_store(text_chunks)
        if not vector_store:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Failed to create vector store.")

        # 4. Initialize Conversational Chain
        logger.info("Initializing conversational QA chain...")
        chain = get_conversational_chain(vector_store)
        if not chain:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Failed to initialize the conversational chain.")

        answers = []
        logger.info("Processing questions...")
        for question in questions:
            # 5. Clause Retrieval (Similarity Search)
            context_docs = vector_store.similarity_search(question)
            print(f"Context documents retrieved: {len(context_docs)} for question '{question}'")
            # 6. Logic Evaluation (LLM)
            response = chain.invoke(
                {
                    "input": question
                }
            )
            answers.append(response["answer"])

        # 7. JSON Output
        return QueryResponse(answers=answers)

    except Exception as e:
        logger.error(f"An internal error occurred: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An internal server error occurred.")

    finally:
        # Safe cleanup - only delete if vector_store was successfully created
        if vector_store is not None:
            try:
                vector_store.delete_collection()
            except Exception as cleanup_error:
                logger.warning(
                    f"Failed to cleanup vector store: {cleanup_error}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
