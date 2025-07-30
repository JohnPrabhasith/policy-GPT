import os
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from astrapy.info import VectorServiceOptions

load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_API_ENDPOINT = os.environ["ASTRA_DB_API_ENDPOINT"]
COLLECTION_NAME = os.environ["ASTRA_DB_COLLECTION"]
JINA_API_KEY = os.environ['JINA_API_KEY']

vectorize_options = VectorServiceOptions(
    provider="jinaAI",
    model_name="jina-embeddings-v3",
    authentication={
        "providerKey": JINA_API_KEY,
    }
)

def get_vector_store(documents: list) -> AstraDBVectorStore:
    try:
        vstore = AstraDBVectorStore(
            collection_name=COLLECTION_NAME,
            token=ASTRA_DB_APPLICATION_TOKEN,
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            collection_vector_service_options =vectorize_options
        )
        vstore.add_documents(documents)
        return vstore
    except Exception as e:
        print(f"[red]Vector store setup error:[/red] {e}")
        return None
