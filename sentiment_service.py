import time
from typing import Dict, Any, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


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

        # BERT-family classifiers are typically limited to 512 total tokens including specials.
        # We'll leave room for [CLS]/[SEP] by treating "content tokens" as max 510.
        self.max_model_tokens = int(getattr(self.model.config, "max_position_embeddings", 512))

    def _sentiment_code(self, sentiment: str) -> int:
        sentiment_code_map = {"negative": -1, "neutral": 0, "positive": 1}
        return sentiment_code_map.get(sentiment, 0)

    def _chunk_text_by_tokens(
        self,
        text: str,
        *,
        chunk_tokens: int,
        overlap_tokens: int,
    ) -> List[Tuple[str, int]]:
        """
        Deterministically chunk a text by tokenizer token IDs.
        Returns list of (chunk_text, token_count) where token_count excludes special tokens.
        """
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be > 0")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens must be >= 0")
        if overlap_tokens >= chunk_tokens:
            raise ValueError("overlap_tokens must be < chunk_tokens")

        safe_content_limit = max(1, self.max_model_tokens - 2)  # leave room for specials
        chunk_tokens = min(chunk_tokens, safe_content_limit)

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        n = len(token_ids)

        if n == 0:
            return [("", 0)]

        step = chunk_tokens - overlap_tokens
        chunks: List[Tuple[str, int]] = []
        start = 0
        while start < n:
            end = min(start + chunk_tokens, n)
            window_ids = token_ids[start:end]
            chunk_str = self.tokenizer.decode(window_ids, skip_special_tokens=True)
            chunks.append((chunk_str, len(window_ids)))
            if end == n:
                break
            start += step

        return chunks

    def _scores_from_probs(self, probs: List[float]) -> Dict[str, float]:
        return {self.id2label[i]: float(probs[i]) for i in range(len(probs))}

    def _argmax_sentiment(self, scores: Dict[str, float]) -> Tuple[str, float]:
        sentiment = max(scores, key=scores.get)
        return sentiment, float(scores[sentiment])

    def _aggregate_scores_weighted(
        self,
        per_chunk_scores: List[Tuple[Dict[str, float], int]],
    ) -> Dict[str, Any]:
        """
        Weighted average of probabilities, weight = token_count (min 1).
        Returns {scores, sentiment, confidence}.
        """
        labels = list(self.id2label.values())

        total_weight = 0
        agg = {lbl: 0.0 for lbl in labels}

        for scores, tok_count in per_chunk_scores:
            w = max(1, int(tok_count))
            total_weight += w
            for lbl in labels:
                agg[lbl] += float(scores.get(lbl, 0.0)) * w

        if total_weight <= 0:
            total_weight = 1

        for lbl in labels:
            agg[lbl] /= total_weight

        sentiment, confidence = self._argmax_sentiment(agg)
        return {"scores": agg, "sentiment": sentiment, "confidence": confidence}

    @torch.no_grad()
    def analyze_one(
        self,
        text: str,
        # Keep signature for compatibility, but max_length is no longer the primary control.
        # We'll treat it as a legacy parameter and ignore it unless a caller relies on it.
        max_length: int = 128,
        *,
        chunk_tokens: int = 450,
        overlap_tokens: int = 50,
        return_chunk_results: bool = False,
    ) -> Dict[str, Any]:
        start = time.time()

        # Explicit chunking replaces "silent truncation".
        chunks = self._chunk_text_by_tokens(
            text,
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
        )

        # If it's already short, you still go through the same path, producing 1 chunk.
        chunk_texts = [c[0] for c in chunks]
        chunk_token_counts = [c[1] for c in chunks]

        # Run model on all chunks in one batch for speed.
        inputs = self.tokenizer(
            chunk_texts,
            return_tensors="pt",
            truncation=True,  # safety net; chunking should keep under limit
            max_length=self.max_model_tokens,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).float().cpu().tolist()  # List[List[float]]

        per_chunk_scores: List[Tuple[Dict[str, float], int]] = []
        chunk_results: List[Dict[str, Any]] = []

        for i, (p, tok_count) in enumerate(zip(probs, chunk_token_counts)):
            scores = self._scores_from_probs(p)
            sentiment, confidence = self._argmax_sentiment(scores)
            per_chunk_scores.append((scores, tok_count))

            if return_chunk_results:
                chunk_results.append(
                    {
                        "chunk_index": i,
                        "token_count": int(tok_count),
                        "sentiment": sentiment,
                        "confidence": float(confidence),
                        "scores": scores,
                    }
                )

        final = self._aggregate_scores_weighted(per_chunk_scores)
        sentiment = final["sentiment"]
        confidence = float(final["confidence"])
        scores = final["scores"]

        sentiment_code = self._sentiment_code(sentiment)

        if len(chunks) <= 1:
            summary = f"Sentiment: {sentiment} (confidence {confidence:.2f})"
        else:
            summary = f"Sentiment: {sentiment} (confidence {confidence:.2f}) using {len(chunks)} chunks"

        result: Dict[str, Any] = {
            "input_text": text,
            "sentiment": sentiment,
            "sentiment_code": sentiment_code,
            "confidence": confidence,
            "summary": summary,
            "scores": scores,
            "model": self.model_name,
            "device": self.device,
            "latency_ms": round((time.time() - start) * 1000.0, 2),
            "chunks_used": len(chunks),
            "total_tokens": int(sum(chunk_token_counts)),
            "chunking": {
                "max_model_tokens": int(self.max_model_tokens),
                "chunk_tokens": int(min(chunk_tokens, max(1, self.max_model_tokens - 2))),
                "overlap_tokens": int(overlap_tokens),
            },
        }

        if return_chunk_results:
            result["chunk_results"] = chunk_results

        return result

    @torch.no_grad()
    def analyze_batch(
        self,
        texts: List[str],
        max_length: int = 128,
        *,
        chunk_tokens: int = 450,
        overlap_tokens: int = 50,
        return_chunk_results: bool = False,
    ) -> List[Dict[str, Any]]:
        start = time.time()

        # Build a global chunk list so we can run one big batch through the model.
        # We'll keep an index so we can map chunk outputs back to their original input.
        text_chunk_map: List[Tuple[int, int, int]] = []
        all_chunk_texts: List[str] = []
        all_chunk_tok_counts: List[int] = []

        for text_index, text in enumerate(texts):
            chunks = self._chunk_text_by_tokens(
                text,
                chunk_tokens=chunk_tokens,
                overlap_tokens=overlap_tokens,
            )
            for chunk_index, (chunk_str, tok_count) in enumerate(chunks):
                text_chunk_map.append((text_index, chunk_index, tok_count))
                all_chunk_texts.append(chunk_str)
                all_chunk_tok_counts.append(int(tok_count))

        inputs = self.tokenizer(
            all_chunk_texts,
            return_tensors="pt",
            truncation=True,  # safety net
            max_length=self.max_model_tokens,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).float().cpu().tolist()

        # Accumulate per input text.
        per_text_scores: List[List[Tuple[Dict[str, float], int]]] = [[] for _ in texts]
        per_text_chunk_results: List[List[Dict[str, Any]]] = [[] for _ in texts]

        for (text_index, chunk_index, tok_count), p in zip(text_chunk_map, probs):
            scores = self._scores_from_probs(p)
            sentiment, confidence = self._argmax_sentiment(scores)

            per_text_scores[text_index].append((scores, tok_count))

            if return_chunk_results:
                per_text_chunk_results[text_index].append(
                    {
                        "chunk_index": int(chunk_index),
                        "token_count": int(tok_count),
                        "sentiment": sentiment,
                        "confidence": float(confidence),
                        "scores": scores,
                    }
                )

        total_ms = (time.time() - start) * 1000.0

        items: List[Dict[str, Any]] = []
        for text, chunk_score_list, chunk_res_list in zip(texts, per_text_scores, per_text_chunk_results):
            final = self._aggregate_scores_weighted(chunk_score_list)
            sentiment = final["sentiment"]
            confidence = float(final["confidence"])
            scores = final["scores"]

            sentiment_code = self._sentiment_code(sentiment)
            chunks_used = len(chunk_score_list)
            total_tokens = int(sum(tok for _, tok in chunk_score_list))

            if chunks_used <= 1:
                summary = f"Sentiment: {sentiment} (confidence {confidence:.2f})"
            else:
                summary = f"Sentiment: {sentiment} (confidence {confidence:.2f}) using {chunks_used} chunks"

            item: Dict[str, Any] = {
                "input_text": text,
                "sentiment": sentiment,
                "sentiment_code": sentiment_code,
                "confidence": confidence,
                "summary": summary,
                "scores": scores,
                "model": self.model_name,
                "device": self.device,
                "chunks_used": int(chunks_used),
                "total_tokens": total_tokens,
                "chunking": {
                    "max_model_tokens": int(self.max_model_tokens),
                    "chunk_tokens": int(min(chunk_tokens, max(1, self.max_model_tokens - 2))),
                    "overlap_tokens": int(overlap_tokens),
                },
            }

            if return_chunk_results:
                item["chunk_results"] = chunk_res_list

            items.append(item)

        # Keep your existing "batch wrapper" response structure.
        return [
            {
                "batch_size": len(texts),
                "total_latency_ms": round(total_ms, 2),
                "items": items,
            }
        ]