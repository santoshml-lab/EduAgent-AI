from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def ask_agent(question: str):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are EduAgent AI, an intelligent education assistant. "
                    "Answer questions clearly, accurately, and step-by-step."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    return response.choices[0].message.content
