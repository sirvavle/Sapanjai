from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# Constants
GOEMOTIONS_MODEL_NAME = 'Patzamangajuice/best_goemotions_mode'
SENSITIVITY_MODEL_NAME = 'facebook/bart-large-mnli'
MAX_LEN = 32

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

# Emotion labels
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

# Load models once at startup
print("Loading tokenizer and models...")
tokenizer = AutoTokenizer.from_pretrained(GOEMOTIONS_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(GOEMOTIONS_MODEL_NAME)
model.eval()
zero_shot_classifier = pipeline("zero-shot-classification", model=SENSITIVITY_MODEL_NAME)
print("Models loaded successfully.")

def analyze_text(text: str):
    # Tokenize input text
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    # Run model inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Pair emotions with probabilities and sort descending
    emotion_probs = [(emotions[i], float(probs[i])) for i in range(len(emotions))]
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

    # Return analysis result
    return {
        "sentiment_level": sentiment_level,
        "final_advice": final_advice,
        "top_emotions": emotion_probs[:5]  # top 5 emotions
    }

# Example usage
if __name__ == "__main__":
    sample_text = "I feel very sad and stressed about the situation."
    result = analyze_text(sample_text)
    print(result)
