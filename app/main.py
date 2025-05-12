import getpass
import os

from fastapi import FastAPI, HTTPException, Query
from app.manual_loader import load_manual_chunks
from app.vector_store import create_vector_store, query_vector_store
from app.llm_utils import ask_gemini 

app = FastAPI()

manual_directory = os.path.join(os.path.dirname(__file__), "data/manuals")
manuals = [os.path.splitext(f)[0] for f in os.listdir(manual_directory) if f.endswith(".pdf")]
manual_chunks = {}
vector_stores = {}

for manual in manuals:
    chunks = load_manual_chunks(os.path.join(manual_directory, manual))
    manual_chunks[manual] = chunks 
    vector_stores[manual] = create_vector_store(chunks)

@app.get("/manuals")
def get_manuals():
    return {"manuals": manuals}

@app.get("/ask")
def ask_manual(question: str = Query(...), manual: str = Query(...)):
    if manual not in vector_stores:
        raise HTTPException(status_code=404, detail="Manual not found")
    
    context = query_vector_store(vector_stores[manual], question, k=3)
    response = ask_gemini(question, context)
    return {"response": response}



