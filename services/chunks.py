from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_text_chunks(documents : list) -> list:
    """
    Splits the document into manageable text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_documents(documents)