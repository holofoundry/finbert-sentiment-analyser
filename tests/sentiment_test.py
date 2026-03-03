from sentiment_service import FinBertSentimentService

service = FinBertSentimentService()

payload = service.analyze(
    "The firm warned of declining revenue and restructuring costs."
)

print(payload)