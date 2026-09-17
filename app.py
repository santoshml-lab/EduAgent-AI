from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import ask_agent


# ========================================
# FastAPI Application
# ========================================

app = FastAPI(
    title="EduAgent AI"
)


# ========================================
# CORS Configuration
# ========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========================================
# Request Model
# ========================================

class Question(BaseModel):
    question: str


# ========================================
# Health Check
# ========================================

@app.get("/")
def home():

    return {
        "message": "EduAgent AI is running 🚀"
    }


# ========================================
# Ask EduAgent
# ========================================

@app.post("/ask")
def ask_question(data: Question):

    result = ask_agent(
        data.question
    )

    return {
        "question": data.question,
        "answer": result["answer"],
        "tool_trace": result["tool_trace"],
        "sources": result.get("sources", [])
    }
