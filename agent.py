from groq import Groq
import os
import json
from dotenv import load_dotenv

from tools import calculator, TOOLS

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def ask_agent(question: str):

    messages = [
        {
            "role": "system",
            "content": (
                "You are EduAgent AI, an intelligent education assistant. "
                "Answer questions clearly and accurately. "
                "Use the calculator tool whenever mathematical calculation is required."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    # Step 1: Ask the LLM whether a tool is needed
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    message = response.choices[0].message

    # Step 2: If no tool is required, return normal answer
    if not message.tool_calls:
        return message.content

    # Step 3: Add the assistant's tool request
    messages.append(message)

    # Step 4: Execute requested tool
    for tool_call in message.tool_calls:

        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        if tool_name == "calculator":
            result = calculator(arguments["expression"])

        else:
            result = "Unknown tool."

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
        )

    # Step 5: Ask LLM to generate final answer using tool result
    final_response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
    )

    return final_response.choices[0].message.content
