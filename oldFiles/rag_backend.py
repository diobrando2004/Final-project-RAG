from getEmbeddingFunction import get_embedding_function
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from pydantic import BaseModel
import requests
from langchain_ollama import OllamaLLM
import argparse
import os
from langchain_core.documents import Document
import time
import subprocess
from fastapi import FastAPI
import re
from llama_cpp import Llama
app = FastAPI()
CHROMA_PATH = "chroma_index"
embedding_function = get_embedding_function()
LLAMA_API = "http://localhost:8000/v1/chat/completions"
local_model = "Qwen3-1.7B-GGUF/Qwen3-1.7B-Q6_K.gguf"
PROMPT_TEMPLATE = """
You are a factual assistant.
Only use information that appears in the provided context.
if there is no relevance detail say "I don't know""


Answer the user's question.

{context}

---

Answer the question based on the above context: {question}
"""
MIN_TOKENS = 2500

def parse_chunk_id(chunk_id):

    match = re.match(r"(.*):(\d+)$", chunk_id)
    if not match:
        return None

    filename, chunk = match.groups()
    return {
        "filename": filename,
        "chunk": int(chunk)
    }

def get_neighbors_from_id(db, chunk_id):
    parsed = parse_chunk_id(chunk_id)
    if not parsed:
        return []

    filename = parsed["filename"]
    chunk = parsed["chunk"]
    anchor = db.get(where={"id": chunk_id})
    if not anchor or len(anchor["documents"]) == 0:
        return []
    anchor_meta = anchor["metadatas"][0]
    anchor_title = anchor_meta.get("section_title")
    candidates = [
        f"{filename}:{chunk - 1}",
        f"{filename}:{chunk + 1}",
        f"{filename}:{chunk + 2}",
        f"{filename}:{chunk + 3}"
    ]

    neighbors = []

    for nid in candidates:
        if nid.endswith(":-1"):
            continue

        res = db.get(where={"id": nid})

        if res and len(res["documents"]) > 0:
            for text, meta in zip(res["documents"], res["metadatas"]):
                if meta.get("section_title") == anchor_title:
                    neighbors.append(Document(page_content=text, metadata=meta))

    return neighbors
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
all_docs = db.get(include=["documents", "metadatas"])
bm25_global = BM25Retriever.from_texts(
    all_docs["documents"],
    metadatas=all_docs["metadatas"]
)
semantic_retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 2, "score_threshold": 0.3}
)

hybrid = EnsembleRetriever(
    retrievers=[semantic_retriever, bm25_global],
    weights=[0.6, 0.4]
)

prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)

class QueryRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_rag(request: QueryRequest):

    query_text = request.question
    t_start = time.time()

    top_docs = hybrid.invoke(query_text)[:3]

    if not top_docs:
        return {"answer": "No relevant documents found.", "sources": []}

    expanded_docs = []

    for d in top_docs[:2]:
        neighbors = get_neighbors_from_id(db, d.metadata["id"])
        expanded_docs.append(d)
        expanded_docs.extend(neighbors)

    expanded_docs.append(top_docs[-1])

    # Remove duplicates
    seen = set()
    final_docs = []
    for d in expanded_docs:
        if d.metadata["id"] not in seen:
            seen.add(d.metadata["id"])
            final_docs.append(d)

    context_text = "\n\n---\n\n".join([d.page_content for d in final_docs])

    prompt = prompt_template.format(
        context=context_text,
        question=query_text
    )

    response = requests.post(
        LLAMA_API,
        json={
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Only provide the final answer. Do not show reasoning."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.1,
        "top_p": 0.9,
        "reasoning": False
    }
    )

    result = response.json()
    answer = result

    t_end = time.time()

    return {
        "answer": answer,
        "sources": [d.metadata.get("id") for d in final_docs],
        "time": round(t_end - t_start, 2)
    }