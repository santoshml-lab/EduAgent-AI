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

                "TOOL RULES:\n"
                "1. For every education-related question, education_router "
                "must be called first.\n"
                "2. Do not call calculator, web_search, quiz_generator, "
                "or study_plan_generator before education_router.\n"
                "3. For numerical questions, use calculator after routing.\n"
                "4. For quiz requests, use quiz_generator after routing.\n"
                "5. For study-plan requests, use study_plan_generator after routing.\n"
                "6. For current-information requests, use web_search after routing.\n"
                "7. For explanation requests, answer after routing without "
                "another tool unless one is genuinely required.\n"
                "8. You may use more than one tool when necessary.\n"
                "9. Use the result of one tool together with another tool "
                "when required.\n"
                "10. Base web-search answers only on retrieved results.\n"
                "11. Do not invent facts, sources, URLs, or citations.\n"
                "12. If available information is insufficient, say so "
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
                "tool_trace": tool_trace
            }

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

            return {
                "answer": final_answer,
                "tool_trace": tool_trace
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
            tool_trace.append({
            "step": len(tool_trace) + 1,
            "tool": tool_name,
            "status": "running",
            "arguments": tool_call.function.arguments
})
            
                
                    
                    
                
            

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
            # Study Plan Generator
            # ========================================
            elif tool_name == "study_plan_generator":

                try:

                    subject = arguments["subject"]
                    days = arguments["days"]
                    hours_per_day = arguments["hours_per_day"]

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
                        f"Study plan generator tool error: {str(e)}"
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
