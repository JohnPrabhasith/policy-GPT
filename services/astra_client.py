import random
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from dotenv import load_dotenv
import os
# from astrapy.info import VectorServiceOptions
from langchain_community.embeddings import JinaEmbeddings

load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_API_ENDPOINT = os.environ["ASTRA_DB_API_ENDPOINT"]
COLLECTION_NAME = os.environ["ASTRA_DB_COLLECTION"]
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# MODEL_ID = "models/gemini-embedding-001"

# embedding = GoogleGenerativeAIEmbeddings(
#     model=MODEL_ID,
#     google_api_key=GOOGLE_API_KEY
# )

JINA_API_KEY = os.environ['JINA_API']


embedding = JinaEmbeddings(
    jina_api_key=JINA_API_KEY,
    model_name="jina-embeddings-v2-base-en",
)

# vectorize_options = VectorServiceOptions(
#     provider="jinaAI",
#     model_name="jina-embeddings-v2-base-en",
#     authentication={
#         "providerKey": JINA_API_KEY,
#     }
# )
COLLECTION_NAME += f'_{random.randint(1000, 9999)}'
def get_vector_store(documents: list) -> AstraDBVectorStore:
    try:
        vstore = AstraDBVectorStore(
            collection_name=COLLECTION_NAME,
            token=ASTRA_DB_APPLICATION_TOKEN,
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            embedding=embedding
            # collection_vector_service_options=vectorize_options
        )
        vstore.add_documents(documents)
        return vstore
    except Exception as e:
        print(f"[red]Vector store setup error:[/red] {e}")
        return None


