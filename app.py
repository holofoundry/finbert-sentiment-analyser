# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union, Dict, Any, Optional
# Diagnostics to surface underlying import errors hidden by transformers lazy loading
try:
    import traceback
    import sys
    print("=== RUNNING PRE-FLIGHT TRANSFORMERS IMPORT DIAGNOSTIC ===", flush=True)
    import transformers.models.auto.tokenization_auto
    print("Pre-flight diagnostic: AutoTokenizer module loaded successfully.", flush=True)
except Exception as e:
    print("=== DIAGNOSTIC IMPORT EXCEPTION CAUGHT ===", flush=True)
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()

from sentiment_service import FinBertSentimentService

app = FastAPI(title="FinBERT Sentiment API", version="1.1.0")

# Load once, keep warm
service: Optional[FinBertSentimentService] = None


class SentimentRequest(BaseModel):
    text: Union[str, List[str]] = Field(
        ...,
        description="A single text string or a list of text strings",
        examples=[
            "The company reported record profits.",
            ["Good earnings.", "Bad guidance."],
        ],
    )

    # Kept for backwards compatibility with your existing client calls.
    # Chunking now controls effective length; tokenizer truncation is only a safety net.
    max_length: int = Field(128, ge=16, le=512, description="Legacy tokenizer max_length (compat)")

    # New explicit chunking controls
    chunk_tokens: int = Field(
        450,
        ge=64,
        le=510,
        description="Token window size (content tokens) for chunking",
    )
    overlap_tokens: int = Field(
        50,
        ge=0,
        le=256,
        description="Token overlap between chunks",
    )
    return_chunk_results: bool = Field(
        False,
        description="Include per-chunk scores in the response",
    )


class SentimentResponse(BaseModel):
    results: List[Dict[str, Any]]


@app.on_event("startup")
def startup_event():
    global service
    service = FinBertSentimentService(fp16=True)


@app.get("/health")
def health():
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "device": service.device, "model": "ProsusAI/finbert"}


@app.post("/sentiment", response_model=SentimentResponse)
def sentiment(req: SentimentRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Normalize and validate input
    texts = req.text if isinstance(req.text, list) else [req.text]

    if any((t is None or not str(t).strip()) for t in texts):
        raise HTTPException(status_code=400, detail="All texts must be non-empty strings")

    if req.overlap_tokens >= req.chunk_tokens:
        raise HTTPException(status_code=400, detail="overlap_tokens must be less than chunk_tokens")

    # Single item path
    if len(texts) == 1:
        result = service.analyze_one(
            texts[0],
            max_length=req.max_length,
            chunk_tokens=req.chunk_tokens,
            overlap_tokens=req.overlap_tokens,
            return_chunk_results=req.return_chunk_results,
        )
        return {"results": [result]}

    # Batch path (service returns wrapper; keep your existing unwrap logic)
    batch_results = service.analyze_batch(
        texts,
        max_length=req.max_length,
        chunk_tokens=req.chunk_tokens,
        overlap_tokens=req.overlap_tokens,
        return_chunk_results=req.return_chunk_results,
    )

    if (
        isinstance(batch_results, list)
        and len(batch_results) == 1
        and isinstance(batch_results[0], dict)
        and "items" in batch_results[0]
        and isinstance(batch_results[0]["items"], list)
    ):
        return {"results": batch_results[0]["items"]}

    return {"results": batch_results}