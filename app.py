from fastapi import FastAPI
from pydantic import BaseModel
from agent import ask_agent

app = FastAPI(title="EduAgent AI")


class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "EduAgent AI is running 🚀"
    }


@app.post("/ask")
def ask_question(data: Question):
    answer = ask_agent(data.question)

    return {
        "question": data.question,
        "answer": answer
    }
