from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
from contextlib import asynccontextmanager
import torch
import os

# --- Constants ---
MODEL_NAME = 'distilbert-base-uncased'
REPO_ID = 'Patzamangajuice/best_goemotions_mode'
FILENAME = 'best_goemotions_model.pt'
CACHE_DIR = '/tmp/models'
MAX_LEN = 10

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
    'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness',
    'surprise'
]

# --- Globals ---
model = None
tokenizer = None

# --- FastAPI lifespan for startup/shutdown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    HF_TOKEN = os.getenv("HF_TOKEN")

    print("📦 Downloading and loading model...")
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        token=HF_TOKEN,
        cache_dir=CACHE_DIR
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=28,
        cache_dir=CACHE_DIR
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    print("✅ Model and tokenizer loaded!")

    yield  # App runs

    print("👋 Shutting down Sapanjai API...")

# --- Initialize FastAPI with lifespan ---
app = FastAPI(lifespan=lifespan)

# --- Input Schema ---
class TextInput(BaseModel):
    text: str

# --- Core Logic ---
def predict_emotion(text):
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
        sentiment = "YELLOW - Most likely Positive"
    elif -0.7 <= final_score < 0:
        sentiment = "ORANGE - Most likely Negative"
    else:
        sentiment = "RED - Negative"

    return sentiment, final_score

# --- API Routes ---
@app.post("/predict")
def predict(input: TextInput):
    emotion_probs = predict_emotion(input.text)
    sentiment, final_score = analyze_sentiment(emotion_probs)

    return {
        "top_emotions": emotion_probs[:5],
        "sentiment_score": final_score,
        "sentiment_warning": sentiment
    }

@app.get("/")
def healthcheck():
    return {"status": "Sapanjai AI is running 🎉"}
