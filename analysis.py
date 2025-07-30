import logging
from transformers import AutoTokenizer, DistilBertForSequenceClassification, pipeline
import torch

tokenizer = None
model = None
classifier = None

def init_models():
    global tokenizer, model, classifier
    logging.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    logging.info("Loading zero-shot classification pipeline...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    logging.info("Loading custom checkpoint...")
    state_dict = torch.load("checkpoints/best_goemotions_model.pt", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict, strict=False)

    logging.info("All models loaded successfully.")

def analyze_text(text):
    # Dummy response for testing
    return {"result": "analysis ok"}
