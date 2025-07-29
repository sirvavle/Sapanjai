from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# Constants
SENTIMENT_MODEL_NAME = 'distilbert-base-uncased'
SENSITIVITY_MODEL_NAME = 'facebook/bart-large-mnli'
MODEL_SAVE_PATH = '/mnt/models/best_goemotions_model.pt'
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

# Globals
tokenizer = None
model = None
zero_shot_classifier = None

def init_models():
    global tokenizer, model, zero_shot_classifier

    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL_NAME, num_labels=len(emotions)
    )
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu"))
    model.eval()

    zero_shot_classifier = pipeline("zero-shot-classification", model=SENSITIVITY_MODEL_NAME)

def analyze_text(text: str):
    # Emotion prediction
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

    # Sentiment scoring
    positive_total = sum(prob for emo, prob in emotion_probs if emo in positive_emotions)
    negative_total = sum(prob for emo, prob in emotion_probs if emo in negative_emotions)
    final_score = positive_total - negative_total

    # Sensitivity check
    sensitivity_output = zero_shot_classifier(text, candidate_labels=sensitive_labels)
    top_sensitive_label = sensitivity_output['labels'][0]
    is_critical = top_sensitive_label in critical_sensitive

    # Final decision logic
    if is_critical and final_score < 0:
        sentiment_level = "red"
        final_advice = f"RED: Message most likely to be sensitive (topic: {top_sensitive_label}). A rewrite is strongly suggested."
    elif is_critical:
        sentiment_level = "orange"
        final_advice = f"ORANGE: Message can be sensitive (topic: {top_sensitive_label}). A rewrite is suggested."
    elif not is_critical and final_score < 0:
        sentiment_level = "yellow"
        final_advice = "YELLOW: Message likely to have a negative tone. A rewrite is suggested."
    else:
        sentiment_level = "green"
        final_advice = "GREEN: Message likely to be safe."

    return {
        "sentiment_level": sentiment_level,
        "final_advice": final_advice
    }
