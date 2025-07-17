from fastapi import FastAPI
from pydantic import BaseModel
from analysis import analyze_text

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Sapanjai AI is up and running 🚀"}

@app.post("/analyze")
async def analyze(text_request: TextRequest):
    result = analyze_text(text_request.text)
    return result
