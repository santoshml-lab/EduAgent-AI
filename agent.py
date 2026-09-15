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
                "3. When using web search, base your answer only on the "
                "retrieved search results.\n"
                "4. Do not invent facts, sources, URLs, or citations.\n"
                "5. For web-search answers, refer to sources as "
                "[Source 1], [Source 2], etc.\n"
                "6. Use only the source IDs provided by the web_search tool.\n"
                "7. If the retrieved sources do not provide enough evidence, "
                "say that the available sources are insufficient rather than guessing."
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

    # Store actual sources returned by web search
    web_sources = []

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

            # Save actual sources returned by Tavily
            try:
                parsed_result = json.loads(result)

                if isinstance(parsed_result, list):
                    web_sources.extend(parsed_result)

            except Exception:
                pass

        else:

            result = "Unknown tool."

        # Send tool result back to AI
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
        )

    # Final answer instruction
    messages.append(
        {
            "role": "system",
            "content": (
                "Now provide the final answer using only the tool results "
                "already provided. Do not call any tools. "
                "Do not invent facts, sources, or URLs. "
                "When referring to web-search evidence, use the source IDs "
                "provided by the tool results."
            ),
        }
    )

    # Final call
    final_response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        tool_choice="none",
    )

    final_answer = final_response.choices[0].message.content

    # Python adds the exact URLs returned by Tavily
    if web_sources:

        final_answer += "\n\n## Sources\n\n"

        for source in web_sources:

            source_id = source.get("source_id", "")
            title = source.get("title", "")
            url = source.get("url", "")

            if title and url:
                final_answer += (
                    f"[Source {source_id}] {title}\n"
                    f"{url}\n\n"
                )

    return final_answer
