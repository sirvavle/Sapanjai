from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from analysis import analyze_text, get_emotion_model, get_zero_shot_classifier
import time

app = FastAPI()

# CORS setup — allow all (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

# --- App startup ---
@app.on_event("startup")
async def startup_event():
    print("🚀 Application startup")
    # Preload models so first request isn't slow
    _ = get_emotion_model()
    _ = get_zero_shot_classifier()

# --- App shutdown ---
@app.on_event("shutdown")
async def shutdown_event():
    print("🛑 Application shutdown")

# --- Health Check ---
@app.get("/")
@app.head("/")
def root():
    return {"message": "Sapanjai AI is up and running 🚀"}

# --- Main route ---
@app.post("/analyze")
async def analyze(text_request: TextRequest):
    try:
        print(f"📩 Received: {text_request.text}")
        start = time.time()
        result = analyze_text(text_request.text)
        end = time.time()
        print(f"✅ Analysis result: {result}")
        print(f"⏱ Analyze took {end - start:.2f} seconds")
        return result
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return {"error": "Analysis failed", "details": str(e)}
