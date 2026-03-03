from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union, Dict, Any, Optional

from sentiment_service import FinBertSentimentService

app = FastAPI(title="FinBERT Sentiment API", version="1.0.0")

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
    max_length: int = Field(128, ge=16, le=512, description="Tokenizer max_length")


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
    if isinstance(req.text, list):
        texts = req.text
    else:
        texts = [req.text]

    if any((t is None or not str(t).strip()) for t in texts):
        raise HTTPException(status_code=400, detail="All texts must be non-empty strings")

    # Use batch inference when multiple texts are provided
    if len(texts) == 1:
        result = service.analyze_one(texts[0], max_length=req.max_length)
        return {"results": [result]}

    # analyze_batch returns either a list of dicts or a wrapped batch payload,
    # depending on how you implemented it. The common/clean approach is a list.
    batch_results = service.analyze_batch(texts, max_length=req.max_length)

    # If your analyze_batch returns a wrapper like [{"batch_size":..., "items":[...]}],
    # unwrap it so the API response matches SentimentResponse (list of dicts).
    if (
        isinstance(batch_results, list)
        and len(batch_results) == 1
        and isinstance(batch_results[0], dict)
        and "items" in batch_results[0]
        and isinstance(batch_results[0]["items"], list)
    ):
        return {"results": batch_results[0]["items"]}

    return {"results": batch_results}