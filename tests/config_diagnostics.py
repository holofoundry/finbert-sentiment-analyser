import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "ProsusAI/finbert"

print("python ok")
print("transformers:", transformers.__version__, transformers.__file__)
print("torch:", torch.__version__, torch.__file__)
print("cuda available:", torch.cuda.is_available())

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

print("model type:", type(model))
print("hasattr(config):", hasattr(model, "config"))

print(model.config)