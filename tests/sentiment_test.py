import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from sentiment_service import FinBertSentimentService

service = FinBertSentimentService()

payload = service.analyze_one(
    "The firm warned of declining revenue and restructuring costs."
)

print(payload)