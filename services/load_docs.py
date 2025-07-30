import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

def load_document_from_url(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif ext == ".docx":
        return Docx2txtLoader(file_path).load()
    elif ext == ".txt":
        return TextLoader(file_path).load()
    else:
        raise ValueError("Unsupported file type")