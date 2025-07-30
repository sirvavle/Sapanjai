from fastapi import FastAPI
from pydantic import BaseModel
from analysis import analyze_text, init_models
import logging
import os

logging.basicConfig(level=logging.INFO)

app = FastAPI()

class TextRequest(BaseModel):
    text: str

model = None  # Optional: hold model globally if needed

@app.on_event("startup")
async def startup_event():
    logging.info("Starting up FastAPI app...")
    try:
        global model
        model = init_models()
        logging.info("Model initialization successful.")
    except Exception as e:
        logging.error("Model initialization failed.")
        logging.exception(e)

@app.post("/analyze")
async def analyze(text_request: TextRequest):
    if model is None:
        return {"error": "Model not loaded yet."}
    return analyze_text(text_request.text)

@app.get("/")
async def root():
    return {"status": "OK"}

@app.get("/routes")
async def get_routes():
    return [route.path for route in app.routes]
