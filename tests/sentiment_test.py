import os
import sys

# Ensure the parent directory is in the path so we can import sentiment_service
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment_service import FinBertSentimentService

service = FinBertSentimentService()

payload = service.analyze_one(
    "The firm warned of declining revenue and restructuring costs."
)

print(payload)