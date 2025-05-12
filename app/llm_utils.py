import os
from vertexai import init
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel


def intialize_gemini() -> BaseChatModel:
    google_cloud_project = os.getenv("GOOGLE_CLOUD_PROJECT")
    google_cloud_location = os.getenv("GOOGLE_CLOUD_LOCATION")

    init(project=google_cloud_project, location=google_cloud_location)

    llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

    return llm

llm = intialize_gemini()

def ask_gemini(question: str, context: str) -> str:
    final_prompt = f"""
        You are a helpful assistant answering questions about the product manual.
        Answer the question based on the context provided.
        Context: {context}
        Question: {question}
        """
    response = llm.invoke([HumanMessage(content=final_prompt)])
    return response.content
