import glob
import os
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import config
from rag_pipe_line import RAGPipeline
 
# --- Load model ---
gguf_files = glob.glob(os.path.join(config.MODELS_DIR, "*.gguf"))
if not gguf_files:
    raise FileNotFoundError(f"No .gguf model found in {config.MODELS_DIR}")
 
model_path = gguf_files[0]
print(f"Loading model: {model_path}")
 
llm = Llama(
    model_path=model_path,
    n_ctx=config.LLM_N_CTX,
    n_gpu_layers=0,
    n_threads=config.LLM_N_THREADS,
    n_batch=config.LLM_N_BATCH,
    temperature=0,
    verbose=True,
)
 
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
 
pipeline = RAGPipeline(llm=llm, embedder=embedder)
pipeline.run()