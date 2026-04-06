from getEmbeddingFunction import get_embedding_function
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
import argparse
import os
from langchain_core.documents import Document
import time
from llama_cpp import Llama
CHROMA_PATH = "chroma_index"
embedding_function = get_embedding_function()
#Do not infer emotions, intentions, or events that are not directly described.
#You will think step by step before answering. Be detailed, factual, and logical.
#Do not infer emotions, intentions, or events that are not directly described
#If you are not sure about something, say "I don't know.
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
import re

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

def query_rag2(query_text: str):
    t_start = time.time()
    model_path = os.path.join(local_model)
    llm = Llama(
    model_path=model_path,
    n_ctx=24096,         
    n_gpu_layers=0,   
    verbose=True
)
    all_docs = db.get(include=["documents", "metadatas"])
    all_texts = all_docs["documents"]
    all_metadatas = all_docs["metadatas"]
    bm25_global = BM25Retriever.from_texts(all_texts, metadatas=all_metadatas)

    semantic_retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 2, "score_threshold": 0.3}
    )

    hybrid = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_global],
        weights=[0.6, 0.4]
    )
    MAX_HYBRID_CHUNKS = 3
    top_docs = hybrid.invoke(query_text)
    top_docs = top_docs[:MAX_HYBRID_CHUNKS]
    if not top_docs:
        print("No results found by hybrid search.")
        return ""
    expanded_docs = []
    for d in top_docs[:2]:
        leng = 0
        neighbors = get_neighbors_from_id(db, d.metadata["id"])
        if neighbors:
            expanded_docs.append(neighbors[0])
            leng += len(neighbors[0].page_content)
        expanded_docs.append(d)
        leng += len(d.page_content)
        for neighbor in neighbors[1:]:
            if leng >= MIN_TOKENS:
                break
            expanded_docs.append(neighbor)
            leng += len(neighbor.page_content)

    expanded_docs.append(top_docs[2])
    # Remove duplicates
    seen = set()
    unique_docs = []
    for d in expanded_docs:
        if d.metadata["id"] not in seen:
            seen.add(d.metadata["id"])
            unique_docs.append(d)
    final_docs = unique_docs
    context_text = "\n\n---\n\n".join([d.page_content for d in final_docs])
    print(context_text)

    prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    response_text = llm(prompt,
    max_tokens=150,
    temperature=0.1,
    top_p=0.9,
    repeat_penalty=1.2,
    echo=False)

    t_end = time.time()

    sources = [d.metadata.get("id") for d in final_docs]
    print("---- Context Used ----")
    print("----------------------")
    answer = response_text["choices"][0]["text"]
    print(f"Response: {answer}\nSources: {sources}")
    print(f"Time: {t_end - t_start:.2f}s")

    return response_text





def main():
    while True:
        query_text = input("Ask a question (or type 'exit'): ")
        if query_text.lower() == "exit":
            break
        query_rag2(query_text)
        

if __name__ == "__main__":
    main()