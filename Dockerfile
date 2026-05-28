# syntax=docker/dockerfile:1.7

# Default to NVIDIA L4T PyTorch image for Jetson Orin Nano (JetPack 6.0 / L4T r36.2)
ARG BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.2-py3
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
# This avoids downloading the model at runtime/startup, saving time and ensuring offline reliability.
RUN python3 -c 'from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained("ProsusAI/finbert"); AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")'

# Copy application files
COPY app.py sentiment_service.py ./
COPY tests ./tests

# Expose port
EXPOSE 8000

# Run FastAPI app with uvicorn
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
