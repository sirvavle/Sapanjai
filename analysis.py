import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from huggingface_hub import hf_hub_download
import torch

# --- Constants ---
SENSITIVITY_MODEL_NAME = 'facebook/bart-large-mnli'
EMOTION_MODEL_NAME = 'distilbert-base-uncased'
REPO_ID = 'Patzamangajuice/best_goemotions_mode'
FILENAME = 'best_goemotions_model.pt'
MAX_LEN = 32

HF_TOKEN = os.getenv("HF_TOKEN")

sensitive_labels = [
    "mental health", "depression", "stress", "suicide", "bullying", "eating disorder",
    "self-harm", "grief", "loss of loved one", "domestic violence",
    "personal preference", "food preference", "hobby", "fashion", "entertainment"
]

emotions = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

positive_emotions = [
    'amusement', 'joy', 'approval', 'caring', 'curiosity', 'excitement', 'gratitude',
    'love', 'optimism', 'pride', 'realization', 'relief'
]
negative_emotions = [
    'anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'fear', 'grief', 'nervousness', 'remorse', 'sadness', 'surprise'
]
neutral_emotions = ['neutral']

# --- Init functions for FastAPI lifespan ---
def load_emotion_model():
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        token=HF_TOKEN,
        cache_dir="/tmp/models"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        EMOTION_MODEL_NAME, num_labels=28, cache_dir="/tmp/models"
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_NAME, cache_dir="/tmp/models")
    return model, tokenizer

def load_zero_shot_classifier():
    return pipeline(
        "zero-shot-classification",
        model=SENSITIVITY_MODEL_NAME,
        cache_dir="/tmp/models"
    )


# --- Core Logic ---
def predict_emotion(text: str, model, tokenizer):
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

def compute_sentiment_score(emotion_probs):
    positive_score = sum(prob for emotion, prob in emotion_probs if emotion in positive_emotions)
    negative_score = sum(prob for emotion, prob in emotion_probs if emotion in negative_emotions)
    return positive_score - negative_score

def get_sentiment_level(score):
    if score > 0.7:
        return "green"
    elif 0 < score <= 0.7:
        return "yellow"
    elif -0.7 <= score < 0:
        return "orange"
    else:
        return "red"

def detect_sensitivity(text: str, sentiment_score: float, zero_shot):
    result = zero_shot(text, candidate_labels=sensitive_labels)
    top_label = result['labels'][0]

    critical_labels = {
        "mental health", "depression", "stress", "suicide", "bullying",
        "eating disorder", "self-harm", "grief", "loss of loved one", "domestic violence"
    }

    if top_label in critical_labels or sentiment_score < -0.7:
        sensitivity_warning = f"Most likely features sensitive topic: {top_label}"
        trigger = 1
    else:
        sensitivity_warning = "Most likely features no sensitive topic"
        trigger = 0

    return top_label, sensitivity_warning, trigger

def analyze_text(text: str, model, tokenizer, zero_shot):
    print(f"🔍 Analyzing: {text}")
    emotion_probs = predict_emotion(text, model, tokenizer)
    sentiment_score = compute_sentiment_score(emotion_probs)
    sentiment_level = get_sentiment_level(sentiment_score)
    top_label, sensitivity_warning, trigger = detect_sensitivity(text, sentiment_score, zero_shot)

    if trigger == 1 and sentiment_score < 0:
        final = f"RED: Message most likely to be sensitive (topics regarding {top_label}). A rewrite is strongly suggested."
    elif trigger == 1:
        final = f"ORANGE: Message can be sensitive (topics regarding {top_label}). A rewrite is suggested."
    elif trigger == 0 and sentiment_score < 0:
        final = "YELLOW: Message likely to have a negative tone. A rewrite is suggested."
    else:
        final = "GREEN: Message likely to be safe."

    return {
        "sentiment_level": sentiment_level,
        "final_advice": final,
    }
