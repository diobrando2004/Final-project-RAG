import os
import glob
import config
from collection import Collection
from chunker import Chunker
from store_parents import ParentStore
from llama_cpp import Llama
from fastembed import TextEmbedding


class RAGSystem:
    def __init__(self, collection_name=config.CHILD_COLLECTION):
        self.collection_name = collection_name
        self.summary_collection_name = "document_summaries"
        self.vector_db = Collection()
        self.parent_store = ParentStore()
        self.chunker = Chunker()

    def initialize(self):
        gguf_files = glob.glob(os.path.join(config.MODELS_DIR, "*.gguf"))
        if not gguf_files:
            raise FileNotFoundError(
                f"No .gguf model found in {config.MODELS_DIR}. "
                "Please add your model file and restart."
            )

        self.embedder = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        n_gpu_layers = config.LLM_N_GPU_LAYERS
        print(gguf_files)
        if n_gpu_layers != 0:
            try:
                self.llm = Llama(
                    model_path=gguf_files[0],
                    n_ctx=config.LLM_N_CTX,
                    n_gpu_layers=n_gpu_layers,
                    n_threads=config.LLM_N_THREADS,
                    n_batch=config.LLM_N_BATCH,
                    temperature=config.LLM_TEMPERATURE,
                    top_p=0.9,
                    verbose=False,
                )
                print(f"LLM loaded with {n_gpu_layers} GPU layers.")
            except Exception as e:
                print(f"GPU init failed ({e}), falling back to CPU.")
                n_gpu_layers = 0
 
        if n_gpu_layers == 0:
            self.llm = Llama(
                model_path=gguf_files[0],
                n_ctx=config.LLM_N_CTX,
                n_gpu_layers=0,
                n_threads=config.LLM_N_THREADS,
                n_batch=config.LLM_N_BATCH,
                temperature=config.LLM_TEMPERATURE,
                top_p=0.9,
                verbose=False,
            )
            print("LLM loaded on CPU.")
 
        self.vector_db.create_collection(self.collection_name)
        self.vector_db.create_summary_collection(self.summary_collection_name)
