import os
import tempfile
import requests
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

def load_document_from_url(file_path: str):
    # Extract the extension correctly without breaking the URL
    base = file_path.split("?")[0]
    print(f"Base file path: {base}")
    ext = os.path.splitext(base)[1].lower()  # includes the dot, e.g., ".pdf"
    print(f"File extension: {ext}")
    # Download the remote file to a temporary file
    response = requests.get(file_path)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name

    # Now use appropriate loader
    if ext == ".pdf":
        return PyPDFLoader(tmp_path).load()
    elif ext == ".docx":
        return Docx2txtLoader(tmp_path).load()
    elif ext == ".txt":
        return TextLoader(tmp_path, encoding="utf-8").load()
    else:
        raise ValueError("Unsupported file type")
