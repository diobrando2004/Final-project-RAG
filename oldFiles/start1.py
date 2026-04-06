import subprocess
import time
import webbrowser

# Start FastAPI backend
backend_process = subprocess.Popen([
    "uvicorn",
    "rag_backend:app",
    "--port", "5000"
])

print("Starting FastAPI...")

time.sleep(3)

if backend_process.poll() is not None:
    print("❌ FastAPI crashed!")
    exit()

webbrowser.open("http://localhost:5000/docs")
print("App is running...")

backend_process.wait()