from groq import Groq
import os
import json
from dotenv import load_dotenv

from tools import (
    calculator,
    web_search,
    education_router,
    quiz_generator,
    study_plan_generator,
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

                "IMPORTANT TOOL RULES:\n"

                "1. For every education-related question, call "
                "education_router FIRST.\n"

                "2. After education_router, identify ALL tasks contained "
                "in the user's question.\n"

                "3. If the question contains multiple tasks, use ALL "
                "required tools before generating the final answer.\n"

                "4. Numerical calculations MUST use calculator.\n"
                "Always convert natural-language percentages or calculations "
                "into a valid mathematical expression before calling "
                "calculator. Example: convert '25% of 2400' into "
                "'0.25 * 2400'.\n"

                "5. Current, latest, recent, today's, or 2026 information "
                "MUST use web_search.\n"

                "6. Quiz requests MUST use quiz_generator.\n"

                "7. Study-plan requests MUST use study_plan_generator.\n"

                "8. Explanation-only questions may be answered after "
                "education_router without another tool when no tool is "
                "genuinely required.\n"

                "9. For multi-task questions, do NOT stop after using only "
                "one tool if another tool is required for another part "
                "of the question.\n"

                "10. Use the results of all required tools when creating "
                "the final answer.\n"

                "11. For web-search answers, use only information returned "
                "by web_search. Do not invent facts, sources, URLs, or "
                "citations.\n"

                "12. If information is insufficient, clearly say so "
                "instead of guessing.\n"
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    web_sources = []
    next_source_id = 1
    tool_trace = []

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

            return {
                "answer": (
                    "EduAgent AI could not contact the AI service. "
                    f"Error: {str(e)}"
                ),
                "tool_trace": tool_trace,
                "sources": []
            }

        message = response.choices[0].message

        # ========================================
        # Agent Finished
        # ========================================

        if not message.tool_calls:

            final_answer = message.content or (
                "I could not generate a final answer."
            )

            return {
                "answer": final_answer,
                "tool_trace": tool_trace,
                "sources": web_sources
            }

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
            # Record Tool Execution
            # ========================================

            trace_entry = {
                "step": len(tool_trace) + 1,
                "tool": tool_name,
                "status": "running",
                "arguments": tool_call.function.arguments
            }

            tool_trace.append(trace_entry)

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

                tool_trace[-1]["status"] = "error"

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
            # Study Plan Generator
            # ========================================

            elif tool_name == "study_plan_generator":

                try:

                    subject = arguments["subject"]

                    days = arguments["days"]

                    hours_per_day = arguments[
                        "hours_per_day"
                    ]

                    topics = arguments.get(
                        "topics",
                        ""
                    )

                    result = study_plan_generator(
                        subject=subject,
                        days=days,
                        hours_per_day=hours_per_day,
                        topics=topics
                    )

                except Exception as e:

                    result = (
                        f"Study plan generator error: {str(e)}"
                    )

            # ========================================
            # Unknown Tool
            # ========================================

            else:

                result = (
                    f"Unknown tool requested: {tool_name}"
                )

            # ========================================
            # Update Tool Status
            # ========================================

            if result:

                if (
                    isinstance(result, str)
                    and (
                        "tool error" in result.lower()
                        or "error:" in result.lower()
                    )
                ):

                    tool_trace[-1]["status"] = "error"

                else:

                    tool_trace[-1]["status"] = "success"

            else:

                tool_trace[-1]["status"] = "success"

            # ========================================
            # Send Tool Result Back to LLM
            # ========================================

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )
