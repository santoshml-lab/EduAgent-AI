from groq import Groq
import os
import json
from dotenv import load_dotenv

from tools import calculator, web_search, TOOLS

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def ask_agent(question: str):

    messages = [
        {
            "role": "system",
            "content": (
                "You are EduAgent AI, an intelligent education assistant. "
                "Answer questions clearly and accurately.\n\n"

                "TOOL RULES:\n"
                "1. Use the calculator tool whenever mathematical calculation "
                "is required.\n"
                "2. Use the web_search tool whenever current, recent, or "
                "up-to-date information is required.\n"
                "3. When using web search, base your answer on the retrieved "
                "search results.\n"
                "4. Do not invent facts, sources, URLs, or citations.\n"
                "5. For web-search answers, include a 'Sources' section "
                "with the relevant source titles and URLs."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    # First call: AI decides whether a tool is needed
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    message = response.choices[0].message

    # No tool needed
    if not message.tool_calls:
        return message.content

    # Add assistant tool-call message
    messages.append(
        {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in message.tool_calls
            ],
        }
    )

    # Execute requested tools
    for tool_call in message.tool_calls:

        tool_name = tool_call.function.name

        arguments = json.loads(
            tool_call.function.arguments
        )

        if tool_name == "calculator":

            result = calculator(
                arguments["expression"]
            )

        elif tool_name == "web_search":

            result = web_search(
                arguments["query"]
            )

        else:

            result = "Unknown tool."

        # Send tool result back to the AI
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
        )

    # Final call: AI uses tool results
    final_response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
    )

    return final_response.choices[0].message.content
