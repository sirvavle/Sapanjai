"""from fastapi import FastAPI
from pydantic import BaseModel
from analysis import analyze_text  # ← your refactored logic
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS setup — allow everything for now (you can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

# --- Lifecycle logs ---
@app.on_event("startup")
async def startup_event():
    print("🚀 Application startup")

@app.on_event("shutdown")
async def shutdown_event():
    print("🛑 Application shutdown")

# --- Health check routes ---
@app.get("/")
@app.head("/")
def root():
    return {"message": "Sapanjai AI is up and running 🚀"}

# --- Main route ---
@app.post("/analyze")
async def analyze(text_request: TextRequest):
    try:
        print(f"📩 Received: {text_request.text}")
        result = analyze_text(text_request.text)
        print(f"✅ Analysis result: {result}")
        return result
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return {"error": "Analysis failed", "details": str(e)}
"""

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello, world!"}
