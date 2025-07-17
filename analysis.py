from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from huggingface_hub import hf_hub_download

SENSITIVITY_MODEL_NAME = 'facebook/bart-large-mnli'

zero_shot_classifier = pipeline("zero-shot-classification", model=SENSITIVITY_MODEL_NAME)

MODEL_NAME = 'distilbert-base-uncased'
REPO_ID = 'Patzamangajuice/best_goemotions_mode'  
FILENAME = 'best_goemotions_model.pt'
MAX_LEN = 32

# --- Download model file from Hugging Face Hub ---
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

# --- Load Emotion Classifier ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=28)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# --- Load Zero-Shot Classifier ---
zero_shot_classifier = pipeline("zero-shot-classification", model=SENSITIVITY_MODEL_NAME)

# --- Sensitivity Labels ---
sensitive_labels = [
    "mental health", "depression", "stress", "suicide", "bullying", "eating disorder",
    "self-harm", "grief", "loss of loved one", "domestic violence",
    "personal preference", "food preference", "hobby", "fashion", "entertainment"
]

# --- Emotion Labels ---
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

def compute_sentiment_score(emotion_probs):
    positive_score = sum(prob for emotion, prob in emotion_probs if emotion in positive_emotions)
    negative_score = sum(prob for emotion, prob in emotion_probs if emotion in negative_emotions)
    score = positive_score - negative_score
    return score

def get_sentiment_level(score):
    if score > 0.7:
        return "GREEN - Positive"
    elif 0 < score <= 0.7:
        return "YELLOW - Possibly Negative"
    elif -0.7 <= score < 0:
        return "ORANGE - Most likely Negative"
    else:
        return "RED - Negative"

def detect_sensitivity(text: str, sentiment_score: float):
    result = zero_shot_classifier(text, candidate_labels=sensitive_labels)
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

def analyze_text(text: str):
    emotion_probs = predict_emotion(text)
    sentiment_score = compute_sentiment_score(emotion_probs)
    sentiment_level = get_sentiment_level(sentiment_score)
    top_label, sensitivity_warning, trigger = detect_sensitivity(text, sentiment_score)

    # Final combined message
    if trigger == 1 and sentiment_score < 0:
        final = f"RED: Message most likely to be sensitive (topics regarding {top_label}). A rewrite is strongly suggested."
    elif trigger == 1:
        final = f"ORANGE: Message can be sensitive (topics regarding {top_label}). A rewrite is suggested."
    elif trigger == 0 and sentiment_score < 0:
        final = "YELLOW: Message likely to have a negative tone. A rewrite is suggested."
    else:
        final = "GREEN: Message likely to be safe."

    return {
        "top_emotions": emotion_probs[:5],
        "sentiment_score": sentiment_score,
        "sentiment_level": sentiment_level,
        "sensitive_label_match": top_label,
        "sensitivity_warning": sensitivity_warning,
        "final_advice": final,
    }

