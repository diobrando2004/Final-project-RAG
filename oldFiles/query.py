from getEmbeddingFunction import get_embedding_function
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
import argparse
from langchain_core.documents import Document
import time
CHROMA_PATH = "chroma_index"
embedding_function = get_embedding_function()
#Do not infer emotions, intentions, or events that are not directly described.
#You will think step by step before answering. Be detailed, factual, and logical.
#Do not infer emotions, intentions, or events that are not directly described
#If you are not sure about something, say "I don't know.
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
def query_rag1(query_text: str):


    t_start = time.time()


    top_hits = db.similarity_search_with_score(query_text, k=2)
    for doc, score in top_hits:
        print(f"[Semantic] Score={score:.4f} | ID={doc.metadata.get('id')}")
    main_docs = [doc for doc, _ in top_hits]

    expanded_docs = []
    for d in main_docs:
        expanded_docs.append(d)
        neighbors = get_neighbors_from_id(db, d.metadata["id"])
        expanded_docs.extend(neighbors)

    # Remove duplicates
    seen = set()
    unique_docs = []
    for d in expanded_docs:
        if d.metadata["id"] not in seen:
            seen.add(d.metadata["id"])
            unique_docs.append(d)

    #
    # 3. HYBRID RERANK (semantic + bm25)
    #
    texts = [d.page_content for d in unique_docs]
    metadatas = [d.metadata for d in unique_docs]

    bm25_retriever = BM25Retriever.from_texts(texts, metadatas=metadatas)

    semantic_retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.3}
    )

    hybrid = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )

    final_docs = hybrid.invoke(query_text)
    

    MAX_CONTEXT_CHUNKS = 6
    final_docs = final_docs[:MAX_CONTEXT_CHUNKS]


    context_text = "\n\n---\n\n".join([d.page_content for d in final_docs])
    print(context_text)

    prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    model = OllamaLLM(model="qwen3:4b", temperature=0.1)
    response_text = model.invoke(prompt)

    t_end = time.time()

    sources = [d.metadata.get("id") for d in final_docs]
    print("---- Context Used ----")
    print("----------------------")
    print(f"Response: {response_text}\nSources: {sources}")
    print(f"Time: {t_end - t_start:.2f}s")

    return response_text

def query_rag2(query_text: str):
    t_start = time.time()

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
    model = OllamaLLM(model="qwen3:1.7b", temperature=0.1)
    response_text = model.invoke(prompt)

    t_end = time.time()

    sources = [d.metadata.get("id") for d in final_docs]
    print("---- Context Used ----")
    print("----------------------")
    print(f"Response: {response_text}\nSources: {sources}")
    print(f"Time: {t_end - t_start:.2f}s")

    return response_text



def query_rag(query_text: str):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    #results = db.similarity_search_with_score(query_text, k=5)

    t2 = time.time()

    #context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    #prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
    #prompt = prompt_template.format(context=context_text, question=query_text)

   # model = OllamaLLM(model="phi4-mini", temperature=0.1)
    #response_text = model.invoke(prompt)
    #t3 = time.time()

    #sources = [doc.metadata.get("id", None) for doc, _score in results]
    #formatted_response = f"Response: {response_text}\nSources: {sources}"
    #print(formatted_response)
    #print(f" Model generation:    {t3 - t2:.2f}s")
    #return response_text
    semantic_retriever = db.as_retriever(search_type="similarity_score_threshold",search_kwargs={"k": 5,"score_threshold": 0.3})
    docs = [doc for doc, _ in db.similarity_search_with_score(query_text, k=20)]
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    bm25_retriever = BM25Retriever.from_texts(texts, metadatas=metadatas)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.6, 0.4]  
    )
    results = hybrid_retriever.invoke(query_text)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
    print(context_text)
    print("/n")
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = OllamaLLM(model="qwen3:4b", temperature=0.1)
    response_text = model.invoke(prompt)
    t3 = time.time()
    sources = [doc.metadata.get("id", None) for doc in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    print(f"Model generation:    {t3 - t2:.2f}s")
    return response_text



def main():
    while True:
        query_text = input("Ask a question (or type 'exit'): ")
        if query_text.lower() == "exit":
            break
        query_rag2(query_text)
        

if __name__ == "__main__":
    main()