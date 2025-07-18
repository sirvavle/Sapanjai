from fastapi import FastAPI
from pydantic import BaseModel
from analysis import analyze_text
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow all origins (or restrict to specific ones if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["https://your-frontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

@app.get("/")
@app.head("/")
def root():
    return {"message": "Sapanjai AI is up and running 🚀"}

@app.post("/analyze")
async def analyze(text_request: TextRequest):
    result = analyze_text(text_request.text)
    print(f"Analyzing text: {text_request.text}")
    print(f"Result: {result}")

    return result
