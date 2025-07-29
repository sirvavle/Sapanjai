from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from analysis import analyze_text, init_models

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.on_event("startup")
async def startup_event():
    print("Loading models...")
    init_models()
    print("Models loaded")

@app.post("/analyze")
async def analyze(request: Request):
    print("Received request")
    try:
        body = await request.json()
        print("Body:", body)

        text = body.get("text")
        if not text:
            raise HTTPException(status_code=400, detail="Missing 'text' in request body")

        result = analyze_text(text)
        print("Analysis result:", result)
        return result
    except Exception as e:
        print("Error during analysis:", e)
        raise HTTPException(status_code=500, detail="Analysis failed")
