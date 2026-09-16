from groq import Groq
import os
import json
from dotenv import load_dotenv

from tools import (
    calculator,
    web_search,
    education_router,
    quiz_generator,
    TOOLS
)

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


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
                "say so instead of guessing.\n"
                "9. For education-related requests, use education_router "
                "to identify the task type before answering.\n"
                "10. Supported education intents are: explanation, "
                "numerical, quiz, study_plan, current_information.\n"
                "11. When the education intent is quiz, use "
                "quiz_generator before generating the quiz."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    web_sources = []
    next_source_id = 1

    # ========================================
    # Agent Loop
    # ========================================
    while True:

        # ========================================
        # Groq API
        # ========================================
        try:

            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )

        except Exception as e:

            return (
                "EduAgent AI could not contact the AI service. "
                f"Error: {str(e)}"
            )

        message = response.choices[0].message

        # ========================================
        # Agent Finished
        # ========================================
        if not message.tool_calls:

            final_answer = message.content or (
                "I could not generate a final answer."
            )

            if web_sources:

                final_answer += "\n\n## Sources\n\n"

                for source in web_sources:

                    final_answer += (
                        f"[Source {source['source_id']}] "
                        f"{source['title']}\n"
                        f"{source['url']}\n\n"
                    )

            return final_answer

        # ========================================
        # Add Assistant Tool Calls
        # ========================================
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

        # ========================================
        # Execute Tools
        # ========================================
        for tool_call in message.tool_calls:

            tool_name = tool_call.function.name

            # ========================================
            # Parse Arguments
            # ========================================
            try:

                arguments = json.loads(
                    tool_call.function.arguments
                )

            except Exception:

                result = (
                    "Tool error: invalid JSON arguments "
                    "provided by the AI."
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

                continue

            # ========================================
            # Calculator
            # ========================================
            if tool_name == "calculator":

                try:

                    expression = arguments["expression"]

                    result = calculator(
                        expression
                    )

                except Exception as e:

                    result = (
                        f"Calculator tool error: {str(e)}"
                    )

            # ========================================
            # Web Search
            # ========================================
            elif tool_name == "web_search":

                try:

                    query = arguments["query"]

                    result = web_search(
                        query
                    )

                    try:

                        parsed_result = json.loads(
                            result
                        )

                        if isinstance(
                            parsed_result,
                            list
                        ):

                            for source in parsed_result:

                                source["source_id"] = (
                                    next_source_id
                                )

                                next_source_id += 1

                            result = json.dumps(
                                parsed_result,
                                ensure_ascii=False
                            )

                            web_sources.extend(
                                parsed_result
                            )

                    except Exception:

                        pass

                except Exception as e:

                    result = (
                        f"Web search tool error: {str(e)}"
                    )

            # ========================================
            # Education Router
            # ========================================
            elif tool_name == "education_router":

                try:

                    intent = arguments["intent"]

                    result = education_router(
                        intent
                    )

                except Exception as e:

                    result = (
                        f"Education router error: {str(e)}"
                    )

            # ========================================
            # Quiz Generator
            # ========================================
            elif tool_name == "quiz_generator":

                try:

                    subject = arguments["subject"]
                    topic = arguments["topic"]

                    number_of_questions = arguments.get(
                        "number_of_questions",
                        10
                    )

                    difficulty = arguments.get(
                        "difficulty",
                        "medium"
                    )

                    result = quiz_generator(
                        subject=subject,
                        topic=topic,
                        number_of_questions=number_of_questions,
                        difficulty=difficulty
                    )

                except Exception as e:

                    result = (
                        f"Quiz generator tool error: {str(e)}"
                    )

            # ========================================
            # Unknown Tool
            # ========================================
            else:

                result = (
                    f"Unknown tool requested: {tool_name}"
                )

            # ========================================
            # Send Tool Result Back
            # ========================================
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )
