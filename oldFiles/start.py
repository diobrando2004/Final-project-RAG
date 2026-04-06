import subprocess
import time
import webbrowser
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Absolute paths (IMPORTANT on Windows)
llama_exe = os.path.join(BASE_DIR, "llama.cpp", "llama-server.exe")
local_model = os.path.join(BASE_DIR, "Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q6_K.gguf")
gbnf = os.path.join(BASE_DIR, "llama.cpp", "ascii.gbnf")
llama_process = subprocess.Popen([
    llama_exe,
    "-m", local_model,
    "-c", "24096",
    "-ngl", "0",
    "--port", "8000",

    "--grammar-file", gbnf
    
])

print("Loading model...")
time.sleep(8)
if llama_process.poll() is not None:
    print("❌ llama-server crashed!")
    exit()
# Start FastAPI backend
backend_process = subprocess.Popen([
    "uvicorn",
    "rag_backend:app",
    "--port", "5000"
])
if backend_process.poll() is not None:
    print("❌ FastAPI crashed!")
    exit()
time.sleep(2)

# Open browser
webbrowser.open("http://localhost:5000/docs")
print("App is running...")
llama_process.wait()
backend_process.wait()