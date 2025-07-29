from fastapi import FastAPI
from pydantic import BaseModel
from analysis import analyze_text  # init_models ไม่จำเป็นแล้ว

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/analyze")
async def analyze(text_request: TextRequest):
    return analyze_text(text_request.text)

@app.get("/")
async def root():
    return {"status": "OK"}
