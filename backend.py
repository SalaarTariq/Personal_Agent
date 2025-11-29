# backend.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from ai_agent import get_response_from_ai_agent

# These model names should stay in sync with the frontend and ai_agent.py
ALLOWED_MODEL_NAMES = [
    "gemini-1.5-flash",        # Google
    "llama-3.3-70b-versatile", # Groq
]

app = FastAPI(title="Personal Agent API")


# -------------------------
# Request Schema
# -------------------------
class RequestState(BaseModel):
    model_name: str
    model_provider: str   # "groq" or "google"
    system_prompt: str
    messages: str
    allow_search: bool


# -------------------------
# Home Route (Fix for 404)
# -------------------------
@app.get("/")
def home():
    return {"message": "Personal Agent API is running!"}


# -------------------------
# Chat Route
# -------------------------
@app.post("/chat")
def chat_endpoint(request: RequestState):

    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error": f"Model '{request.model_name}' is not allowed"}

    try:
        response = get_response_from_ai_agent(
            query=request.messages,
            allow_search=request.allow_search,
            system_prompt=request.system_prompt,
            provider=request.model_provider
        )
        return {"response": response}

    except Exception as e:
        return {"error": str(e)}


# -------------------------
# Run Server
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
