from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
def get_sparse_embedding_function():
    return FastEmbedSparse(model_name="Qdrant/bm25")