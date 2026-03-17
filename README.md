# FinBERT Sentiment Analyser

This repository exposes the `ProsusAI/finbert` model as a small FastAPI service.
It accepts one text string or a list of text strings, splits long inputs into token-based chunks, runs FinBERT over those chunks, and returns weighted sentiment scores.

## How It Works

The runtime is built from two files:

- `app.py`
  Starts the FastAPI application, loads the model once at startup, and exposes:
  - `GET /health`
  - `POST /sentiment`
- `sentiment_service.py`
  Wraps Hugging Face `transformers` and PyTorch, handles token chunking, model inference, and score aggregation.

### Request flow

1. FastAPI starts and creates a single `FinBertSentimentService` instance.
2. The service loads:
   - tokenizer: `AutoTokenizer.from_pretrained("ProsusAI/finbert")`
   - model: `AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")`
3. Incoming text is converted into tokenizer token IDs.
4. Long inputs are split into overlapping chunks so they stay within the BERT token limit.
5. All chunks are scored in one model batch.
6. Chunk probabilities are combined using a token-count-weighted average.
7. The API returns:
   - dominant sentiment
   - confidence
   - per-label probabilities
   - chunk metadata
   - latency/device/model details

### API behavior

`POST /sentiment` accepts:

- `text`: a single string or a list of strings
- `max_length`: legacy compatibility field; chunking is the main length control now
- `chunk_tokens`: max content tokens per chunk
- `overlap_tokens`: chunk overlap
- `return_chunk_results`: include per-chunk outputs

Example request:

```json
{
  "text": "The company reported strong earnings but lowered guidance.",
  "chunk_tokens": 450,
  "overlap_tokens": 50,
  "return_chunk_results": false
}
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Sentiment request:

```bash
curl -X POST http://127.0.0.1:8000/sentiment \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "The company reported strong earnings but lowered guidance.",
    "chunk_tokens": 450,
    "overlap_tokens": 50
  }'
```

## Python Version

Use a Python `3.11` virtual environment.

Reasoning:

- the repo does not pin package versions
- `fastapi`, `transformers`, and `torch` are all commonly deployed on Python 3.11
- 3.11 is the safest default for broad wheel availability, especially for PyTorch

Python `3.12` may also work, but `3.11` is the conservative setup choice for this codebase.

## Virtual Environment Setup

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## Dependencies To Install

This repo does not currently include a `requirements.txt` or `pyproject.toml`, so install the runtime dependencies directly.

### CPU-only install

```bash
pip install fastapi "uvicorn[standard]" pydantic transformers safetensors torch
```

### NVIDIA GPU install

Install a CUDA-compatible PyTorch build first, then the remaining packages.

Example pattern:

```bash
pip install torch
pip install fastapi "uvicorn[standard]" pydantic transformers safetensors
```

If you need a specific CUDA build, use the PyTorch install command appropriate for that machine.

## Running Locally

Activate the virtual environment and start the API with Uvicorn:

```bash
source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000
```

Notes:

- on first startup, the model will be downloaded from Hugging Face
- startup time will be slower the first time because weights/tokenizer files must be cached
- if CUDA is available, the service will use GPU automatically

Interactive docs are then available at:

```text
http://127.0.0.1:8000/docs
```

## Running As A Dedicated systemd Service

This example runs the API as a dedicated service account named `finbert`.

### 1. Create a dedicated user

```bash
sudo useradd --system --home /opt/finbert --shell /usr/sbin/nologin finbert
```

### 2. Deploy the project

Copy the repository to a stable path, for example:

```text
/opt/finbert/finbert-sentiment-analyser
```

Then create the virtual environment there and install the dependencies into it.

Example:

```bash
sudo mkdir -p /opt/finbert
sudo chown -R finbert:finbert /opt/finbert
sudo -u finbert python3.11 -m venv /opt/finbert/finbert-sentiment-analyser/.venv
sudo -u finbert /opt/finbert/finbert-sentiment-analyser/.venv/bin/pip install --upgrade pip setuptools wheel
sudo -u finbert /opt/finbert/finbert-sentiment-analyser/.venv/bin/pip install fastapi "uvicorn[standard]" pydantic transformers safetensors torch
```

### 3. Create the unit file

Create `/etc/systemd/system/finbert.service`:

```ini
[Unit]
Description=FinBERT Sentiment API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=finbert
Group=finbert
WorkingDirectory=/opt/finbert/finbert-sentiment-analyser
Environment=PATH=/opt/finbert/finbert-sentiment-analyser/.venv/bin
ExecStart=/opt/finbert/finbert-sentiment-analyser/.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5

# Optional: keep Hugging Face cache in a predictable location
Environment=HF_HOME=/opt/finbert/.cache/huggingface
Environment=TRANSFORMERS_CACHE=/opt/finbert/.cache/huggingface

[Install]
WantedBy=multi-user.target
```

### 4. Enable and start the service

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now finbert.service
```

### 5. Check logs and status

```bash
sudo systemctl status finbert.service
sudo journalctl -u finbert.service -f
```

## Operational Notes

- Port: the examples use `8000`
- Health check: `GET /health`
- First-run network access: required to download `ProsusAI/finbert`
- Ongoing network access: not required after the model is cached, unless the cache is cleared
- GPU usage: automatic when `torch.cuda.is_available()` is true

## Known Code Observations

- There is no dependency lockfile yet, so installs are not fully reproducible.
- `tests/sentiment_test.py` calls `service.analyze(...)`, but the service currently exposes `analyze_one(...)` and `analyze_batch(...)`. That file is not aligned with the current implementation.
