from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from analysis import analyze_text, load_emotion_model, load_zero_shot_classifier
from contextlib import asynccontextmanager
import time

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Application startup")
    # Preload and store models in app state
    emotion_model, emotion_tokenizer = load_emotion_model()
    zero_shot = load_zero_shot_classifier()
    app.state.emotion_model = emotion_model
    app.state.emotion_tokenizer = emotion_tokenizer
    app.state.zero_shot = zero_shot
    yield
    print("🛑 Application shutdown")

app = FastAPI(lifespan=lifespan)

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
        result = analyze_text(
            text=text_request.text,
            model=app.state.emotion_model,
            tokenizer=app.state.emotion_tokenizer,
            zero_shot=app.state.zero_shot,
        )
        end = time.time()
        print(f"✅ Analysis result: {result}")
        print(f"⏱ Analyze took {end - start:.2f} seconds")
        return result
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return {"error": "Analysis failed", "details": str(e)}
