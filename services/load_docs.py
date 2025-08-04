import os
import requests
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_document_from_url(file_path: str):
    # Extract the extension correctly without breaking the URL
    base = file_path.split("?")[0]
    print(f"Base file path: {base}")
    ext = os.path.splitext(base)[1].lower()  # includes the dot, e.g., ".pdf"
    print(f"File extension: {ext}")
    # Download the remote file to a temporary file
    response = requests.get(file_path)
    response.raise_for_status()
    tmp_path = f'document{ext}'
    with open(tmp_path, "wb") as f:
            f.write(response.content)

    # Now use appropriate loader
    if ext == ".pdf":
        documents = PyPDFLoader(tmp_path).load()
    elif ext == ".docx":
        documents = Docx2txtLoader(tmp_path).load()
    elif ext == ".txt":
        documents = TextLoader(tmp_path, encoding="utf-8").load()
    else:
        raise ValueError("Unsupported file type")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_documents(documents)

file_path = "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"
print(load_document_from_url(file_path)[0])
    