from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from agent import ask_agent
from evaluation import run_all_tests


app = FastAPI(title="EduAgent AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str
    session_id: Optional[str] = "default"


@app.get("/")
def home():
    return {"message": "EduAgent AI is running 🚀"}


@app.post("/ask")
def ask_question(data: Question):
    result = ask_agent(
        question=data.question,
        session_id=data.session_id
    )

    return {
        "question": data.question,
        "session_id": data.session_id,
        "answer": result["answer"],
        "tool_trace": result["tool_trace"],
        "sources": result.get("sources", [])
    }


@app.get("/evaluate")
def evaluate_agent():
    return run_all_tests()
