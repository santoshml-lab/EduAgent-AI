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
                "1. Use calculator whenever mathematical calculation "
                "is required.\n"
                "2. Use web_search whenever current, recent, or "
                "up-to-date information is required.\n"
                "3. You may use more than one tool when necessary.\n"
                "4. Use the result of one tool together with another "
                "tool when the question requires it.\n"
                "5. Base web-search answers only on retrieved results.\n"
                "6. Do not invent facts, sources, URLs, or citations.\n"
                "7. For web-search answers, refer to sources as "
                "[Source 1], [Source 2], etc.\n"
                "8. If the available information is insufficient, "
                "say so instead of guessing."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    web_sources = []

    # Agent loop
    while True:

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message

        # Agent has finished
        if not message.tool_calls:

            final_answer = message.content

            if web_sources:
                final_answer += "\n\n## Sources\n\n"

                for source in web_sources:
                    final_answer += (
                        f"[Source {source['source_id']}] "
                        f"{source['title']}\n"
                        f"{source['url']}\n\n"
                    )

            return final_answer

        # Add assistant tool calls
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

        # Execute every requested tool
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

                try:
                    parsed_result = json.loads(result)

                    if isinstance(parsed_result, list):
                        web_sources.extend(parsed_result)

                except Exception:
                    pass

            else:

                result = "Unknown tool."

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )
