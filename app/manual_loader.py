import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List


def load_manual_chunks(file_path: str) -> List[Document]:
    """
    Loads a PDF manual and returns a list of chunked documents.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Manual not found: {file_path}")
    
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    
    return chunks 