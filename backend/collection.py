import config
from getEmbeddingFunction import get_embedding_function, get_sparse_embedding_function
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


class Collection:
    def __init__(self):
        self.__client = QdrantClient(path=config.QDRANT_DB_PATH)
        self.__sparse_embedding = get_sparse_embedding_function()
        self.__dense_embedding = get_embedding_function()
    def _create_collection_if_missing(self, collection_name: str):
        if not self.__client.collection_exists(collection_name):
            print(f"Creating collection: {collection_name}")
            self.__client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=len(self.__dense_embedding.embed_query("test")),
                    distance=qmodels.Distance.COSINE
                ),
                sparse_vectors_config={
                    config.SPARSE_VECTOR_NAME: qmodels.SparseVectorParams()
                }
            )
            print(f"Created collection: {collection_name}")
        else:
            print(f"Collection already exists: {collection_name}")
 
    def create_collection(self, collection_name: str):
        self._create_collection_if_missing(collection_name)
 
    def create_summary_collection(self, collection_name: str = "document_summaries"):
        self._create_collection_if_missing(collection_name)

    

    def get_collection(self, collection_name: str) -> QdrantVectorStore:
        try:
            return QdrantVectorStore(
                client=self.__client,
                collection_name=collection_name,
                embedding=self.__dense_embedding,
                sparse_embedding=self.__sparse_embedding,
                retrieval_mode=RetrievalMode.HYBRID,
                sparse_vector_name=config.SPARSE_VECTOR_NAME
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to get collection '{collection_name}': {e}"
            ) from e
    
    def get_unique_sources(self, collection_name: str) -> list[str]:
        points, _ = self.__client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=["metadata.source"],
            with_vectors=False
        )
        sources = {
            p.payload['metadata']['source']
            for p in points
            if 'metadata' in p.payload
        }
        return sorted(list(sources))