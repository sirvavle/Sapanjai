from transformers import AutoTokenizer, BertForSequenceClassification, pipeline
import torch
import numpy as np

# Constants
MODEL_PATH = './best_goemotions_model.pt'  # เปลี่ยนเป็น path จริงของไฟล์ .pt ที่คุณมี
TOKENIZER_NAME = 'monologg/bert-base-cased-goemotions-original'
SENSITIVITY_MODEL_NAME = 'facebook/bart-large-mnli'
MAX_LEN = 32
NUM_LABELS = 28  # จำนวน label ของ GoEmotions

# Sensitive topics
sensitive_labels = [
    "mental health", "depression", "stress", "suicide", "bullying", "eating disorder",
    "self-harm", "grief", "loss of loved one", "domestic violence",
    "personal preference", "food preference", "hobby", "fashion", "entertainment"
]
critical_sensitive = {
    "mental health", "depression", "stress", "suicide", "bullying",
    "eating disorder", "self-harm", "grief", "loss of loved one", "domestic violence"
}

# Emotion labels (28 labels GoEmotions)
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

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# สร้างโมเดล architecture แล้วโหลดน้ำหนักจากไฟล์ .pt
model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=NUM_LABELS)
state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

zero_shot_classifier = pipeline("zero-shot-classification", model=SENSITIVITY_MODEL_NAME)
print("Models loaded successfully.")

def analyze_text(text: str):
    # Tokenize input text
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).squeeze().numpy()

    # Pair emotions with probabilities and sort descending
    emotion_probs = list(zip(emotions, probs))
    emotion_probs.sort(key=lambda x: x[1], reverse=True)

    # Calculate sentiment score
    positive_total = sum(prob for emo, prob in emotion_probs if emo in positive_emotions)
    negative_total = sum(prob for emo, prob in emotion_probs if emo in negative_emotions)
    final_score = positive_total - negative_total

    # Zero-shot classification for sensitive topic detection
    sensitivity_output = zero_shot_classifier(text, candidate_labels=sensitive_labels)
    top_sensitive_label = sensitivity_output['labels'][0]
    is_critical = top_sensitive_label in critical_sensitive

    # Decide sentiment level and advice
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
        "final_advice": final_advice,
        "top_emotions": emotion_probs[:5]  # top 5 emotions
    }

if __name__ == "__main__":
    sample_text = "I feel very sad and stressed about the situation."
    result = analyze_text(sample_text)
    print(result)
