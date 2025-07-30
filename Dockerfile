FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure the model checkpoint file is copied into the image
# (make sure checkpoints/ folder exists and contains your model)
COPY checkpoints/best_goemotions_model.pt checkpoints/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
