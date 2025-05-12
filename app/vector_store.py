from typing import List
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

def create_vector_store(chunks: List[Document]) -> InMemoryVectorStore:
    """
    Creates an in-memory vector store from the provided document chunks.
    """
    embeddings = VertexAIEmbeddings(model="text-embedding-004")
    vector_store = InMemoryVectorStore.from_documents(documents=chunks, embedding=embeddings)
    
    return vector_store

def query_vector_store(vector_store: InMemoryVectorStore, query: str, k: int = 3) -> str:
    """
    Queries the vector store with a question and returns the top k relevant documents.
    """
    docs = vector_store.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context