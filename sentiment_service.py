import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any, List
import time


class FinBertSentimentService:
    def __init__(self, model_name: str = "ProsusAI/finbert", fp16: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        dtype = torch.float16 if (fp16 and self.device == "cuda") else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(self.device).eval()

        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.model_name = model_name

    @torch.no_grad()
    def analyze_one(self, text: str, max_length: int = 128) -> Dict[str, Any]:
        start = time.time()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().float().cpu().tolist()

        scores = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}
        sentiment = max(scores, key=scores.get)
        confidence = scores[sentiment]

        sentiment_code_map = {"negative": -1, "neutral": 0, "positive": 1}
        sentiment_code = sentiment_code_map.get(sentiment, 0)

        summary = f"Sentiment: {sentiment} (confidence {confidence:.2f})"

        return {
            "input_text": text,
            "sentiment": sentiment,
            "sentiment_code": sentiment_code,
            "confidence": confidence,
            "summary": summary,
            "scores": scores,
            "model": self.model_name,
            "device": self.device,
            "latency_ms": round((time.time() - start) * 1000.0, 2),
        }

    @torch.no_grad()
    def analyze_batch(self, texts: List[str], max_length: int = 128) -> List[Dict[str, Any]]:
        start = time.time()

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).float().cpu()

        total_ms = (time.time() - start) * 1000.0

        results = []
        for text, row in zip(texts, probs):
            row = row.tolist()
            scores = {self.id2label[i]: float(row[i]) for i in range(len(row))}
            sentiment = max(scores, key=scores.get)
            confidence = scores[sentiment]

            sentiment_code_map = {"negative": -1, "neutral": 0, "positive": 1}
            sentiment_code = sentiment_code_map.get(sentiment, 0)

            summary = f"Sentiment: {sentiment} (confidence {confidence:.2f})"

            results.append({
                "input_text": text,
                "sentiment": sentiment,
                "sentiment_code": sentiment_code,
                "confidence": confidence,
                "summary": summary,
                "scores": scores,
                "model": self.model_name,
                "device": self.device,
            })

        # attach timing once, not per item, so it stays honest
        return [{
            "batch_size": len(texts),
            "total_latency_ms": round(total_ms, 2),
            "items": results,
        }]