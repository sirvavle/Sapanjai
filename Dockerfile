FROM python:3.11-slim

WORKDIR /app

# Install required system packages
RUN apt-get update && apt-get install -y git build-essential curl

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download models and checkpoint
RUN python -c "from transformers import AutoTokenizer, DistilBertForSequenceClassification; \
AutoTokenizer.from_pretrained('distilbert-base-uncased'); \
DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')"

RUN python -c "from transformers import pipeline; \
pipeline('zero-shot-classification', model='facebook/bart-large-mnli')"

RUN mkdir -p /app/checkpoints && \
    curl -L -o /app/checkpoints/best_goemotions_model.pt https://huggingface.co/Patzamangajuice/best_goemotions_mode/resolve/main/best_goemotions_model.pt

# Copy app files
COPY . .

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
