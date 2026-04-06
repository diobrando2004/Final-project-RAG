from getEmbeddingFunction import get_embedding_function
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import time

CHROMA_PATH = "chroma_index"

# Path to your local phi-4-mini-instruct model folder
MODEL_PATH = "phi4-mini-instruct"
OFFLOAD_DIR = "offload_dir"

PROMPT_TEMPLATE = """
You are a factual assistant.
Only use information that explicitly appears in the provided context.
Do not infer emotions, intentions, or events that are not directly described.
If you are not sure about something, say "I don't know."

Answer the user's question.

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)
    t2 = time.time()

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map="auto",
        offload_folder=OFFLOAD_DIR,
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.2,
        do_sample=False
    )

    # === Generate the answer ===
    output = generator(prompt)[0]["generated_text"]

    # Extract only model's answer (cut off redundant prompt)
    if prompt in output:
        response_text = output[len(prompt):].strip()
    else:
        response_text = output.strip()

    t3 = time.time()

    # === Display response and sources ===
    sources = [doc.metadata.get("id", None) for doc, _ in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    print(f"💬 Model generation: {t3 - t2:.2f}s")

    return response_text


def main():
    while True:
        query_text = input("Ask a question (or type 'exit'): ")
        if query_text.lower() == "exit":
            break
        query_rag(query_text)


if __name__ == "__main__":
    main()
