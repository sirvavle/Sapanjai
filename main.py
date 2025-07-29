from fastapi import FastAPI
from pydantic import BaseModel
from Sapanjai.analysis import analyze_text, init_models

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.on_event("startup")
async def startup_event():
    init_models()

@app.post("/analyze")
async def analyze(text_request: TextRequest):
    return analyze_text(text_request.text)
