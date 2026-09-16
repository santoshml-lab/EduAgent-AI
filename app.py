from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import ask_agent

app = FastAPI(title="EduAgent AI")


# Allow frontend to call the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "EduAgent AI is running 🚀"
    }


@app.post("/ask")
def ask_question(data: Question):

    result = ask_agent(
        data.question
    )

    return {
        "question": data.question,
        "answer": result["answer"],
        "tool_trace": result["tool_trace"]
    }
