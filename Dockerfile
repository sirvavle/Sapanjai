FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for torch & transformers
RUN apt-get update && apt-get install -y git build-essential

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
