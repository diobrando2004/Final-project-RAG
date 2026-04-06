from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os
from getEmbeddingFunction import get_embedding_function

DATA_PATH = "E:\project3\documents"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

CHROMA_PATH = "chroma_index"


def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    vectorstore = Chroma(
        persist_directory= CHROMA_PATH,
        embedding_function=embedding_function
    )

