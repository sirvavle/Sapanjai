from fastapi import FastAPI
from pydantic import BaseModel
from analysis import analyze_text
import os
import uvicorn

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.get("/")
@app.head("/")
def root():
    return {"message": "Sapanjai AI is up and running 🚀"}

@app.post("/analyze")
async def analyze(text_request: TextRequest):
    result = analyze_text(text_request.text)
    return result

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use Render's assigned port or fallback
    uvicorn.run("main:app", host="0.0.0.0", port=port)