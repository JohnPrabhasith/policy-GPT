import os
import sys
import tempfile
import time
import io
import re
import requests
from typing import Any, Callable, List
from rich import print

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import List, Optional
from docx import Document as DocxDocument
import fitz

from auth import verify_token 

app = Flask(__name__)
CORS(app)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    sys.exit("Environment variable GROQ_API_KEY is required")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={"device": "cpu"},
                                   encode_kwargs={"normalize_embeddings": True})

folder = "./faiss_index"
if not os.path.exists(folder):
    os.makedirs(folder)

model = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=GROQ_API_KEY,
        temperature=0.0,
        max_tokens=None,
        timeout=(30, 60),
        max_retries=1,
    )

class CleanEmbeddings(HuggingFaceEmbeddings):
    def embed_documents(self, texts: List[Any]) -> List[List[float]]:
        clean_texts: List[str] = []
        for x in texts:
            if isinstance(x, Document):
                txt = x.page_content
            elif isinstance(x, dict):
                txt = x.get("page_content", "")
            else:
                txt = str(x)
            clean_texts.append(txt.replace("\n", " "))
        return super().embed_documents(clean_texts)

    def embed_query(self, q: Any) -> List[float]:
        return self.embed_documents([q])[0]


clean_embeddings = CleanEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs=embeddings.model_kwargs,
    encode_kwargs=embeddings.encode_kwargs,
)

def fetch_and_split(url: str, chunk_size: int = 1000, chunk_overlap: int = 150, use_recursive: bool = True) -> List[str]:

    base = url.split("?", 1)[0]
    ext = os.path.splitext(base)[1].lower()
    if ext not in (".pdf", ".docx", ".txt"):
        raise ValueError(f"Unsupported extension: '{ext}'")

    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.content

    if ext == ".pdf":
        if not fitz:
            raise RuntimeError("PyMuPDF (fitz) is required for PDF support")
        text = _extract_pdf_text_in_memory(data)
    elif ext == ".docx":
        if not DocxDocument:
            raise RuntimeError("python-docx is required for DOCX support")
        text = _extract_docx_text_in_memory(data)
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1", errors="ignore")

    text = text.strip()
    if not text:
        return []

    if use_recursive and len(text) > chunk_size * 2:
        return _recursive_chunk(text, max_length=chunk_size, overlap=chunk_overlap)
    else:
        return _sliding_window_chunk(text, window=chunk_size, overlap=chunk_overlap)


def _extract_pdf_text_in_memory(pdf_bytes: bytes) -> str:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages = [page.get_text() for page in doc]
    return "\n".join(pages)


def _extract_docx_text_in_memory(docx_bytes: bytes) -> str:
    file_like = io.BytesIO(docx_bytes)
    doc = DocxDocument(file_like)
    paras = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paras)


def _sliding_window_chunk(text: str, window: int, overlap: int) -> List[str]:

    chunks = []
    step = max(window - overlap, 1)
    for i in range(0, len(text), step):
        chunk = text[i : i + window].strip()
        if chunk:
            chunks.append(chunk)
        if i + window >= len(text):
            break
    return chunks


def _recursive_chunk(text: str, max_length: int, overlap: int) -> List[str]:

    if len(text) <= max_length:
        return [text]

    sep_order = ["\n\n", "\n", r"[\.!?]\s+", " "]
    for sep in sep_order:
        # recursive split
        parts = re.split(sep, text)
        if len(parts) == 1:
            continue
        chunks = []
        current = ""
        for part in parts:
            if not part.strip():
                continue
            segment = (current + sep + part).strip() if current else part
            if len(segment) <= max_length:
                current = segment
            else:
                if current:
                    chunks.extend(_sliding_window_chunk(current, max_length, overlap))
                current = part
        if current:
            chunks.extend(_recursive_chunk(current, max_length, overlap))

        if chunks and len(max(chunks, key=len)) <= max_length * 1.5:
            return chunks

    return _sliding_window_chunk(text, max_length, overlap)


def make_vector_store(docs: List[Document], folder: str = "./faiss_index") -> FAISS:
    if os.path.isdir(folder):
        return FAISS.load_local(folder, clean_embeddings, allow_dangerous_deserialization=True)

    vs = FAISS.from_documents(docs, clean_embeddings)
    vs.save_local(folder)
    return vs


def make_prompt() -> PromptTemplate:
    tpl = """
            You are a highly intelligent and experienced AI built to analyze complex policy, legal, and institutional content. Your PRIMARY and MOST IMPORTANT task is to extract precise and meaningful answers based strictly on the provided context text and user query.

            Instructions:
            - Return a clear and grammatically correct answer in paragraph format.
            - Each answer must be concise, limited to a maximum of 3 to 4 lines.
            - DO NOT start the answer with phrases like “The provided text states...” or “According to the context...”. Start directly with the answer.
            - DO NOT use bullet points, numbered lists, or any formatting characters.
            - DO NOT include asterisk (*) or newline (\\n) characters.
            - DO NOT hallucinate or fabricate information.

            Now, analyze the following context and answer the user’s query accurately.
            Context:
            {context}

            Question:
            {input}

            Detailed Answer:
            """
    return PromptTemplate(template=tpl, input_variables=["context", "input"])

prompt = make_prompt()

def format_docs_as_str(docs: List[Any]) -> str:
    body_lines = []
    for d in docs:
        content = d.page_content if hasattr(d, "page_content") else d.get("page_content", "")
        body_lines.append(str(content))
    return "\n\n".join(body_lines)


def make_chain(vs: FAISS) -> Callable:
    retr = vs.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.12}
    )

    mapping = {
        "context": retr | format_docs_as_str,
        "input": RunnablePassthrough(),
    }
    
    chain = mapping | prompt | model | StrOutputParser()

    return chain

@app.route('/', methods=['GET'])
def index():
    return {"status": "HackRx RAG API is running. Use /api/v1/hackrx/run to ask questions."}

@app.route("/api/v1/hackrx/run", methods=["POST"])
def ask():
    t_open = time.perf_counter()
    token_check = verify_token()
    if token_check is not True:
        return token_check 
    print("Received request to /api/v1/hackrx/run")
    print(f"Request body: {request.get_json()}")
    print(f"[blue]Token verified successfully in {time.perf_counter() - t_open:.2f} seconds[/blue]")
    try:
        data = request.json
        document = data["documents"]
        questions = data["questions"]

        t1 = time.perf_counter()
        docs = fetch_and_split(document)
        t2 = time.perf_counter()
        vs = make_vector_store(docs)
        t3 = time.perf_counter()
        chain = make_chain(vs)
        t4 = time.perf_counter()
        print(f"[green]Document fetch time: {t2 - t1:.2f}s[/green]")
        print(f"[green]Vector store creation time: {t3 - t2:.2f}s[/green]")
        print(f"[green]Chain creation time: {t4 - t3:.2f}s[/green]")
        results = []
        for q in questions:
            print(f"Processing question: {q}")
            t5 = time.perf_counter()
            try:
                ans = chain.invoke({"input": q})
            except Exception as e:
                ans = f"ERROR: {e}"
            results.append(ans)
            t6 = time.perf_counter()
            print(f"Question processing time: {t6 - t5:.2f}s")
        total_time = time.perf_counter() - t_open
        print(f"[green]Total processing time: {total_time:.2f}s[/green]")
        return jsonify({"answers": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
