# syntax=docker/dockerfile:1.7

# Default to official NVIDIA PyTorch container with iGPU support for JetPack 6.2
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.09-py3-igpu
FROM ${BASE_IMAGE}

WORKDIR /app

# Install system dependencies if any
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Configure cache directory for Hugging Face models inside the container
ENV HF_HOME=/app/model_cache \
    TRANSFORMERS_CACHE=/app/model_cache \
    PYTHONUNBUFFERED=1

# Pre-download and cache ProsusAI/finbert model weights and tokenizer during build.
# We use huggingface_hub directly to avoid importing PyTorch/CUDA during docker build (which fails due to missing runtime GPU drivers).
RUN python3 -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="ProsusAI/finbert")'

# Copy application files
COPY app.py sentiment_service.py ./
COPY tests ./tests

# Expose port
EXPOSE 8000

# Run FastAPI app with uvicorn
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
