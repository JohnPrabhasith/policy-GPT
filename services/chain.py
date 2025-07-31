from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import JinaRerank
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from prompts import initial_prompt
from dotenv import load_dotenv
import logging
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
JINA_API_KEY = os.getenv("JINA_API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, api_key=GOOGLE_API_KEY)

prompt = PromptTemplate(
    template=initial_prompt, 
    input_variables=["context", "input"]
)

def get_conversational_chain(vector_store: Document) -> create_retrieval_chain:
    
    logger.info("Initializing conversational QA chain...")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 10, "score_threshold": 0.5}
    )
    compressor = JinaRerank(jina_api_key=JINA_API_KEY, model= "jina-reranker-v1-base-en")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    combine_docs_chain = create_stuff_documents_chain(model, prompt)

    chain = create_retrieval_chain(
        retriever=compression_retriever,
        combine_docs_chain=combine_docs_chain
    )
    
    if not chain:
        logger.error("Failed to initialize the conversational chain.")
        return None
    
    return chain



# combine_chain = create_stuff_documents_chain(
#     llm=model,
#     prompt=prompt
# )

# refine_chain = RefineDocumentsChain.from_llm(
#     llm=model,
#     prompt_initial=init_prompt,
#     prompt_refine=ref_prompt
# )