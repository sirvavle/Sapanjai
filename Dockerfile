FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu20.04

WORKDIR /app

# Install system packages
RUN apt-get update && apt-get install -y git curl python3 python3-pip

# Copy app files and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Preload HF models to cache
RUN python3 -c "from transformers import AutoTokenizer, DistilBertForSequenceClassification; \
AutoTokenizer.from_pretrained('distilbert-base-uncased'); \
DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')"

RUN python3 -c "from transformers import pipeline; \
pipeline('zero-shot-classification', model='facebook/bart-large-mnli')"

# Download checkpoint
RUN mkdir -p /app/checkpoints && \
    curl -L -o /app/checkpoints/best_goemotions_model.pt https://huggingface.co/Patzamangajuice/best_goemotions_mode/resolve/main/best_goemotions_model.pt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
