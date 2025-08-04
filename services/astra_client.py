import os
import numpy as np
import logging
from dotenv import load_dotenv
from langchain_community.embeddings import JinaEmbeddings
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import JinaRerank
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import faiss

load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
JINA_API_KEY = os.getenv("JINA_API")

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Embedding model
embedding_model = JinaEmbeddings(
    jina_api_key=JINA_API_KEY,
    model_name="jina-embeddings-v2-base-en",
)

# LLM model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", temperature=0.3, api_key=GOOGLE_API_KEY
)

# Prompt
from prompts import initial_prompt
prompt = PromptTemplate(
    template=initial_prompt,
    input_variables=["context", "input"]
)

def build_faiss_hnsw(documents: list[Document], m: int = 32, ef_construction: int = 200):
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    embeddings = np.array([embedding_model.embed_query(text) for text in texts], dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, m)
    index.hnsw.efConstruction = ef_construction
    index.add(embeddings)

    logger.info(f"Indexed {len(documents)} documents with FAISS HNSW (dim={dim})")
    return index, documents

def search_faiss(index, documents, query: str, top_k: int = 5, ef_search: int = 64):
    index.hnsw.efSearch = ef_search
    query_vector = np.array([embedding_model.embed_query(query)], dtype=np.float32)
    _, indices = index.search(query_vector, top_k)
    return [documents[i] for i in indices[0] if i < len(documents)]

def get_conversational_chain(index, documents) -> create_retrieval_chain:
    logger.info("Initializing conversational QA chain with FAISS retriever...")

    class FAISSRetriever:
        def __init__(self, index, docs):
            self.index = index
            self.docs = docs

        def get_relevant_documents(self, query):
            return search_faiss(self.index, self.docs, query)

    retriever = FAISSRetriever(index, documents)
    # compressor = JinaRerank(jina_api_key=JINA_API_KEY, model="jina-reranker-v1-tiny-en")
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor,
    #     base_retriever=retriever
    # )

    combine_docs_chain = create_stuff_documents_chain(model, prompt)
    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain
    )

    if not chain:
        logger.error("Failed to initialize the conversational chain.")
        return None
    return chain
