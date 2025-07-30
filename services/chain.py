from langchain.chains import RefineDocumentsChain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
from prompts import initial_prompt, refined_prompt
import logging

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3, api_key=GOOGLE_API_KEY)

init_prompt = PromptTemplate(
    template=initial_prompt, 
    input_variables=["context", "input"]
)

ref_prompt = PromptTemplate(
    template=refined_prompt, 
    input_variables=["existing_answer", "context", "input"]
)

def get_conversational_chain(vector_store: Document) -> create_retrieval_chain:
    
    logger.info("Initializing conversational QA chain...")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.6}
    )

    # combine_chain = create_stuff_documents_chain(
    #     llm=model,
    #     prompt=prompt
    # )

    refine_chain = RefineDocumentsChain.from_llm(
        llm=model,
        prompt_initial=init_prompt,
        prompt_refine=ref_prompt
    )

    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=refine_chain
    )
    
    if not chain:
        logger.error("Failed to initialize the conversational chain.")
        return None
    
    return chain