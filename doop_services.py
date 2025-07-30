# In services.py

import os
import requests
import io
from langchain.document_loaders import PyPDFLoader

def load_document_from_url(url: str):
    """
    Loads a PDF document from a URL and returns its content as LangChain Document objects.
    
    Args:
        url (str): The public URL of the PDF document.

    Returns:
        list: A list of Document objects loaded from the PDF, or None on failure.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # Create a temporary file path to use with PyPDFLoader
        temp_pdf_path = "temp_document.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(response.content)

        # Load the PDF using the loader
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()
        
        # Clean up the temporary file
        os.remove(temp_pdf_path)
        
        return documents
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to download the document from the URL. {e}")
        return None
    except Exception as e:
        print(f"Error: Failed to process the document. {e}")
        return None
    


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def get_text_chunks(documents: list[Document]):
    """
    Splits the loaded documents into smaller chunks for processing.

    Args:
        documents (list[Document]): A list of Document objects.

    Returns:
        list: A list of smaller Document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, 
        chunk_overlap=1000
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


# In services.py

import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain.docstore.document import Document

def get_vector_store(text_chunks: list[Document]):
    """
    Creates embeddings and stores them in AstraDB.
    Returns a vector store object for similarity search.

    Args:
        text_chunks (list[Document]): A list of document chunks to embed and store.

    Returns:
        AstraDBVectorStore: A vector store object ready for querying.
    """
    # Initialize the Google embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Initialize the AstraDB vector store
    vector_store = AstraDBVectorStore(
        embedding=embeddings,
        collection_name=os.getenv("ASTRA_DB_COLLECTION_NAME"),
        api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
        token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    )
    
    # Add the document chunks to the vector store
    vector_store.add_documents(text_chunks)
    
    return vector_store


# In services.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.Youtubeing import load_qa_chain

def get_conversational_chain():
    """
    Creates and returns a conversational QA chain using Gemini Pro.
    The chain is configured with a specific prompt to guide the model's responses.
    
    Returns:
        A LangChain QA chain object.
    """
    prompt_template = """
    You are an expert at analyzing policy documents. Answer the user's question in detail
    based on the provided document context. Make sure to provide all relevant details and conditions. 
    If the answer is not available in the context, clearly state: 
    "The answer is not available in the provided document." Do not invent information.

    Context:
    {context}

    Question:
    {question}

    Detailed Answer:
    """
    
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3  
    )
    
    # Create the prompt from the template
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    # Load the question-answering chain
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain