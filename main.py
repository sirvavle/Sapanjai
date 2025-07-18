import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from contextlib import asynccontextmanager

# Constants
SENTIMENT_MODEL_NAME = 'distilbert-base-uncased'
SENSITIVITY_MODEL_NAME = 'facebook/bart-large-mnli'
MODEL_SAVE_PATH = '/mnt/models/best_goemotions_model.pt'  # Adjust to your persistent disk path
MAX_LEN = 32

sensitive_labels = [
    "mental health", "depression", "stress", "suicide", "bullying", "eating disorder",
    "self-harm", "grief", "loss of loved one", "domestic violence",
    "personal preference", "food preference", "hobby", "fashion", "entertainment"
]

critical_sensitive = {
    "mental health", "depression", "stress", "suicide", "bullying",
    "eating disorder", "self-harm", "grief", "loss of loved one", "domestic violence"
}

emotions = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

positive_emotions = [
    'amusement', 'joy', 'approval', 'caring', 'curiosity', 'excitement',
    'gratitude', 'love', 'optimism', 'pride', 'realization', 'relief'
]
negative_emotions = [
    'anger', 'annoyance', 'disappointment', 'disapproval', 'disgust',
    'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness', 'surprise'
]

app = FastAPI()
model = None
tokenizer = None
zero_shot_classifier = None

class TextInput(BaseModel):
    text: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, zero_shot_classifier
    
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL_NAME,
        num_labels=len(emotions)
    )
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu"))
    model.eval()

    zero_shot_classifier = pipeline(
        "zero-shot-classification",
        model=SENSITIVITY_MODEL_NAME
    )
    
    yield

app.router.lifespan_context = lifespan

def predict_emotion(text: str):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    emotion_probs = [(emotions[i], float(probs[i])) for i in range(len(emotions))]
    emotion_probs.sort(key=lambda x: x[1], reverse=True)
    return emotion_probs

def analyze_sentiment(emotion_probs):
    positive_total = sum(prob for emo, prob in emotion_probs if emo in positive_emotions)
    negative_total = sum(prob for emo, prob in emotion_probs if emo in negative_emotions)
    final_score = positive_total - negative_total

    if final_score > 0.7:
        sentiment = "GREEN - Positive"
    elif 0 < final_score <= 0.7:
        sentiment = "YELLOW - Possibly Negative"
    elif -0.7 <= final_score < 0:
        sentiment = "ORANGE - Most likely Negative"
    else:
        sentiment = "RED - Negative"

    return sentiment, final_score

@app.post("/analyze")
async def analyze(input: TextInput):
    emotion_probs = predict_emotion(input.text)
    sentiment, final_score = analyze_sentiment(emotion_probs)

    sensitivity_output = zero_shot_classifier(input.text, candidate_labels=sensitive_labels)
    top_sensitive_label = sensitivity_output['labels'][0]

    if top_sensitive_label in critical_sensitive:
        trigger = 1
        sensitivity_warning = f"Most likely features sensitive topic: {top_sensitive_label}"
    else:
        trigger = 0
        sensitivity_warning = "Most likely features no sensitive topic"

    if trigger == 1 and final_score < 0:
        final_advice = f"RED: Message most likely to be sensitive (topics regarding {top_sensitive_label}). A rewrite is strongly suggested."
    elif trigger == 1:
        final_advice = f"ORANGE: Message can be sensitive (topics regarding {top_sensitive_label}). A rewrite is suggested."
    elif trigger == 0 and final_score < 0:
        final_advice = "YELLOW: Message likely to have a negative tone. A rewrite is suggested."
    else:
        final_advice = "GREEN: Message likely to be safe."

    return {
        "top_emotions": emotion_probs[:5],
        "sentiment_score": final_score,
        "sensitivity_warning": sensitivity_warning,
        "final_advice": final_advice
    }

@app.get("/")
async def healthcheck():
    return {"status": "Sapanjai AI is running 🎉"}
