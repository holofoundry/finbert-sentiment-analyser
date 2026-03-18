# FinBERT Sentiment Analyser

# Setup
## Clone the Repo

The project is hosted on GitHub https://github.com/holofoundry/finbert-sentiment-analyser

## Using the right version of Python

> Due to the wild west compatibility of dependencies on LLM frameworks, the best option is to start by installing `pyenv`, which is a python version manager. We're assuming an Ubuntu 24.04 system. 3.11 is the safest default for broad wheel availability, especially for PyTorch, so we're going with that.

First, install the build dependencies you’ll need to compile older Python versions:

```bash
sudo apt update  
sudo apt install -y \  
build-essential curl git \  
libssl-dev zlib1g-dev libbz2-dev \  
libreadline-dev libsqlite3-dev libncursesw5-dev \  
xz-utils tk-dev libxml2-dev libxmlsec1-dev \  
libffi-dev liblzma-dev ca-certificates
```

Then install `pyenv`:

```bash
curl https://pyenv.run | bash
```

Open the shell config:

```bash
nano ~/.bashrc
```

And add the following block to the end of the bash config:

```txt
# pyenv
export PYENV_ROOT="$HOME/.pyenv"  
export PATH="$PYENV_ROOT/bin:$PATH"  
eval "$(pyenv init - bash)"
```

Save it, then reload your shell properly:

```
source ~/.bashrc
```

And test it with the following command:

```bash
pyenv --version
```

If this works, then we've got `pyenv` up and running. Then, we want to install Python 3.11 to run FinBERT in:

```bash
pyenv install 3.11.15
```

## Create your virtual environment

Once you've cloned the project from GitHub, navigate to the root folder and make your virtual environment. We're going to call it `.venv` as this is our coding standard.

```bash
pyenv local 3.11.15
python -m venv .venv  
source .venv/bin/activate  
pip install --upgrade pip setuptools wheel
```

That works because `venv` uses the interpreter you ran it with, so once `pyenv local` makes `python` point at 3.10, your `.venv` is a Python 3.10 environment.

## Install the FinBERT stack

Then install the usual FinBERT stack. Hugging Face’s Transformers docs currently say Transformers is tested on Python 3.10+ and PyTorch 2.4+, so Python 3.10 is still a sensible target. The common FinBERT model on Hugging Face is `ProsusAI/finbert`, which is a finance-domain BERT model fine-tuned for sentiment classification.

```bash
pip install torch transformers scikit-learn pandas numpy fastapi
```

## Test it

Run the `sentiment_test.py` in the `tests` folder. On first run, this will pull down the required models from HuggingFace.

```bash
python tests/sentiment_test.py 
```

You should get an output that looks like:

```json
{'input_text': 'The firm warned of declining revenue and restructuring costs.', 'sentiment': 'negative', 'sentiment_code': -1, 'confidence': 0.97265625, 'summary': 'Sentiment: negative (confidence 0.97)', 'scores': {'positive': 0.00864410400390625, 'negative': 0.97265625, 'neutral': 0.01861572265625}, 'model': 'ProsusAI/finbert', 'device': 'cuda', 'latency_ms': 227.38, 'chunks_used': 1, 'total_tokens': 10, 'chunking': {'max_model_tokens': 512, 'chunk_tokens': 450, 'overlap_tokens': 50}}
```

> The short practical recipe is this: leave Ubuntu’s system Python alone, install 3.11 with `pyenv`, use `pyenv local 3.11.15` in your FinBERT project folder, then build `.venv` from that interpreter. That keeps your machine tidy and your model environment reproducible.

# Architecture
This project exposes the `ProsusAI/finbert` model as a small FastAPI service.

It accepts one text string or a list of text strings, splits long inputs into token-based chunks, runs FinBERT over those chunks, and returns weighted sentiment scores.

## How it works
The runtime is built from two files:
1. `app.py` starts the FastAPI application, loads the model once at startup, and exposes `GET /health` and  `POST /sentiment`
2. `sentiment_service.py` wraps Hugging Face `transformers` and PyTorch, handles token chunking, model inference, and score aggregation.

## Request flow
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

## API behaviour
### POST /sentiment

`POST /sentiment` accepts:
- `text`: a single string or a list of strings
- `max_length`: legacy compatibility field; chunking is the main length control now
- `chunk_tokens`: max content tokens per chunk
- `overlap_tokens`: chunk overlap
- `return_chunk_results`: include per-chunk outputs
#### Example request

```json
{
	"text": "The company reported strong earnings but lowered guidance.",
	"chunk_tokens": 450,
	"overlap_tokens": 50,
	"return_chunk_results": false
}
```

### GET /healthcheck

Checks to see if the service is still alive. Useful for monitoring.

```bash
curl http://127.0.0.1:8000/health
```

## Python Version
Use a Python `3.11` virtual environment as `fastapi`, `transformers`, and `torch` are all commonly deployed on Python 3.11. Ir's also the safest default for broad wheel availability, especially for PyTorch.

Python `3.12` may also work, but `3.11` is the conservative setup choice for this codebase.
# Running as a service
This example runs the API as a dedicated service account named `finbert`.

## Create a dedicated user

```bash
sudo useradd --system --home /opt/finbert --shell /usr/sbin/nologin finbert
```

## Deploy the project

Copy the repository to a stable path, for example:

```text
/opt/finbert/finbert-sentiment-analyser
```

Then create the virtual environment there and install the dependencies into it.
### Example

```bash
sudo mkdir -p /opt/finbert
sudo chown -R finbert:finbert /opt/finbert
sudo -u finbert python3.11 -m venv /opt/finbert/finbert-sentiment-analyser/.venv
sudo -u finbert /opt/finbert/finbert-sentiment-analyser/.venv/bin/pip install --upgrade pip setuptools wheel
sudo -u finbert /opt/finbert/finbert-sentiment-analyser/.venv/bin/pip install fastapi "uvicorn[standard]" pydantic transformers safetensors torch
```

## Create the unit file

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

## Enable and start the service

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now finbert.service
```

## Check logs and status

```bash
sudo systemctl status finbert.service
sudo journalctl -u finbert.service -f
```
