# Use a slim Python image to save disk space
FROM python:3.11-slim

# Install essential build tools for llama-cpp-python if it needs to compile
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements first (better for Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files
COPY . .

# Expose Gradio's default port
EXPOSE 7860

# Run your main script
CMD ["python", "test_full_system.py"]