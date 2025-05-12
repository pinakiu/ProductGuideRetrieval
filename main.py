import getpass
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import init_chat_model
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import HumanMessage

from vertexai import init
from dotenv import load_dotenv 

google_api_key = os.getenv("GOOGLE_API_KEY")
google_cloud_project = os.getenv("GOOGLE_CLOUD_PROJECT")
google_cloud_location = os.getenv("GOOGLE_CLOUD_LOCATION")

init(project=google_cloud_project, location=google_cloud_location) 
load_dotenv()

llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

embeddings = VertexAIEmbeddings(model="text-embedding-004")

file_path = "Ascent A2300 and A2500 Domestic Owners Manual.pdf"
loader = PyPDFLoader(file_path)
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(pages)


vector_store = InMemoryVectorStore.from_documents(documents=chunks, embedding=embeddings)
query = "My mixture in my blender is stuck. What do I do?"

results = vector_store.similarity_search(query, k=2)

context = "\n\n".join([doc.page_content for doc in results])

final_prompt = f"""
You are a helpful assistant answering questions about Ascent A2300 and A2500 Domestic Owners Manual.
Answer the question based on the context provided.
Context: {context}
Question: {query}
"""
response = llm.invoke([HumanMessage(content=final_prompt)])
print("\n--- Final Response ---")
print(response.content)

