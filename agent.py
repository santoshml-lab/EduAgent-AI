from groq import Groq
import os
import json
import re
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


# ============================================================
# Conversation Memory
# ============================================================

conversation_memory = {}

MAX_HISTORY = 6


# ============================================================
# Helper: Extract Calculation
# ============================================================

def extract_calculation(question: str):
    """
    Detect common mathematical expressions from
    natural-language questions.
    """

    text = question.lower().strip()

    # Percentage of a number
    percentage_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:of)\s*"
        r"(\d+(?:\.\d+)?)",
        text
    )

    if percentage_match:

        percentage = float(percentage_match.group(1))
        number = float(percentage_match.group(2))

        return f"({percentage} / 100) * {number}"

    # Total before percentage
    reverse_percentage_match = re.search(
        r"(\d+(?:\.\d+)?)\s*"
        r"(?:students?|people|persons?|children|items?|"
        r"candidates?|employees?|customers?|members?)?"
        r".{0,80}?"
        r"(\d+(?:\.\d+)?)\s*(?:%|percent)",
        text
    )

    if reverse_percentage_match:

        number = float(
            reverse_percentage_match.group(1)
        )

        percentage = float(
            reverse_percentage_match.group(2)
        )

        if percentage <= 100:

            return (
                f"({percentage} / 100) * {number}"
            )

    # Percentage with out of/from/among
    out_of_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:%|percent)"
        r".{0,30}?"
        r"(?:out of|from|among)\s*"
        r"(\d+(?:\.\d+)?)",
        text
    )

    if out_of_match:

        percentage = float(
            out_of_match.group(1)
        )

        number = float(
            out_of_match.group(2)
        )

        return (
            f"({percentage} / 100) * {number}"
        )

    # Basic arithmetic
    arithmetic_match = re.search(
        r"(\d+(?:\.\d+)?)\s*"
        r"([\+\-\*\/])\s*"
        r"(\d+(?:\.\d+)?)",
        text
    )

    if arithmetic_match:

        number1 = arithmetic_match.group(1)
        operator = arithmetic_match.group(2)
        number2 = arithmetic_match.group(3)

        return (
            f"{number1} {operator} {number2}"
        )

    return None


# ============================================================
# Helper: Extract Web Query
# ============================================================

def extract_web_query(question: str):

    return question.strip()


# ============================================================
# Helper: Execute Calculator
# ============================================================

def execute_calculator(question: str):

    expression = extract_calculation(question)

    if not expression:

        return {
            "success": False,
            "result": (
                "Calculator could not identify a mathematical "
                "expression from the question."
            )
        }

    try:

        result = calculator(expression)

        return {
            "success": True,
            "expression": expression,
            "result": result
        }

    except Exception as e:

        return {
            "success": False,
            "expression": expression,
            "result": f"Calculator error: {str(e)}"
        }


# ============================================================
# Helper: Execute Web Search
# ============================================================

def execute_web_search(question: str):

    query = extract_web_query(question)

    try:

        result = web_search(query)

        return {
            "success": True,
            "query": query,
            "result": result
        }

    except Exception as e:

        return {
            "success": False,
            "query": query,
            "result": f"Web search error: {str(e)}"
        }


# ============================================================
# Helper: Execute Quiz Generator
# ============================================================

def execute_quiz(question: str):

    extraction_prompt = [
        {
            "role": "system",
            "content": (
                "Extract quiz parameters from the user's request. "
                "Return ONLY valid JSON with these keys: "
                "subject, topic, number_of_questions, difficulty. "
                "number_of_questions must be an integer between 1 and 20. "
                "difficulty must be easy, medium, or hard. "
                "If not specified, use 10 and medium."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=extraction_prompt,
            temperature=0
        )

        content = response.choices[0].message.content or "{}"

        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

        params = json.loads(content)

        subject = params.get(
            "subject",
            "General"
        )

        topic = params.get(
            "topic",
            "General"
        )

        number_of_questions = int(
            params.get(
                "number_of_questions",
                10
            )
        )

        difficulty = params.get(
            "difficulty",
            "medium"
        )

        result = quiz_generator(
            subject=subject,
            topic=topic,
            number_of_questions=number_of_questions,
            difficulty=difficulty
        )

        return {
            "success": True,
            "result": result
        }

    except Exception as e:

        return {
            "success": False,
            "result": f"Quiz generator error: {str(e)}"
        }


# ============================================================
# Helper: Execute Study Plan
# ============================================================

def execute_study_plan(question: str):

    extraction_prompt = [
        {
            "role": "system",
            "content": (
                "Extract study-plan parameters from the user's request. "
                "Return ONLY valid JSON with these keys: "
                "subject, days, hours_per_day, topics. "
                "days must be an integer. "
                "hours_per_day must be a number. "
                "If topics are not specified, use an empty string."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=extraction_prompt,
            temperature=0
        )

        content = response.choices[0].message.content or "{}"

        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

        params = json.loads(content)

        subject = params.get(
            "subject",
            "General"
        )

        days = int(
            params.get(
                "days",
                7
            )
        )

        hours_per_day = float(
            params.get(
                "hours_per_day",
                2
            )
        )

        topics = params.get(
            "topics",
            ""
        )

        result = study_plan_generator(
            subject=subject,
            days=days,
            hours_per_day=hours_per_day,
            topics=topics
        )

        return {
            "success": True,
            "result": result
        }

    except Exception as e:

        return {
            "success": False,
            "result": f"Study plan generator error: {str(e)}"
        }


# ============================================================
# Helper: Resolve Conversation Context
# ============================================================

def resolve_context(question: str, history: list):

    if not history:
        return question

    history_text = "\n".join(
        [
            f"User: {item['question']}\n"
            f"Assistant: {item['answer']}"
            for item in history[-MAX_HISTORY:]
        ]
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a conversation context resolver for "
                "an education AI agent.\n\n"

                "Your job is to convert the CURRENT user question "
                "into a standalone question when it depends on "
                "previous conversation.\n\n"

                "Rules:\n"
                "1. Use previous conversation only when necessary.\n"
                "2. Resolve words like it, this, that, these, those, "
                "same subject, continue, change, modify, add, remove, "
                "make it easier, make it harder, etc.\n"
                "3. Preserve the user's actual requested change.\n"
                "4. If the current question is already standalone, "
                "return it unchanged.\n"
                "5. Return ONLY the resolved standalone question.\n"
                "6. Do not answer the question.\n\n"

                "Example:\n"
                "Previous: User created a Biology study plan.\n"
                "Current: Make it 2 hours per day.\n"
                "Resolved: Change the Biology study plan to 2 hours "
                "per day.\n"
            )
        },
        {
            "role": "user",
            "content": (
                "Previous conversation:\n\n"
                f"{history_text}\n\n"
                "Current user question:\n"
                f"{question}"
            )
        }
    ]

    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0
        )

        resolved = (
            response.choices[0].message.content
            or question
        )

        resolved = resolved.strip()

        return resolved if resolved else question

    except Exception:

        # If context resolution fails,
        # continue safely with the original question.
        return question


# ============================================================
# Main Agent
# ============================================================

def ask_agent(
    question: str,
    session_id: str = "default"
):

    question = question.strip()

    session_id = (
        session_id.strip()
        if session_id
        else "default"
    )

    if not question:

        return {
            "answer": "Please enter a question.",
            "tool_trace": [],
            "sources": []
        }

    # ========================================================
    # STEP 0 — LOAD MEMORY
    # ========================================================

    history = conversation_memory.get(
        session_id,
        []
    )

    # Resolve current question using previous conversation
    contextual_question = resolve_context(
        question,
        history
    )

    tool_trace = []
    web_sources = []

    # ========================================================
    # STEP 1 — EDUCATION ROUTER
    # ========================================================

    router_trace = {
        "step": 1,
        "tool": "education_router",
        "status": "running",
        "arguments": json.dumps(
            {
                "intents": []
            }
        )
    }

    tool_trace.append(router_trace)

    router_messages = [
        {
            "role": "system",
            "content": (
                "You are the education task router for EduAgent AI.\n\n"

                "Your ONLY job is to identify ALL applicable "
                "education intents in the user's question.\n\n"

                "Available intents:\n"
                "- explanation\n"
                "- numerical\n"
                "- quiz\n"
                "- study_plan\n"
                "- current_information\n\n"

                "IMPORTANT:\n"
                "If multiple tasks exist, return ALL applicable intents.\n\n"

                "Do not answer the user's question."
            )
        },
        {
            "role": "user",
            "content": contextual_question
        }
    ]

    try:

        router_response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=router_messages,
            tools=TOOLS,
            tool_choice={
                "type": "function",
                "function": {
                    "name": "education_router"
                }
            },
            temperature=0
        )

    except Exception as e:

        tool_trace[-1]["status"] = "error"

        return {
            "answer": (
                "EduAgent AI could not run the education router. "
                f"Error: {str(e)}"
            ),
            "tool_trace": tool_trace,
            "sources": []
        }

    router_message = (
        router_response.choices[0].message
    )

    intents = []

    if router_message.tool_calls:

        router_call = (
            router_message.tool_calls[0]
        )

        try:

            router_arguments = json.loads(
                router_call.function.arguments
            )

            intents = router_arguments.get(
                "intents",
                []
            )

        except Exception as e:

            tool_trace[-1]["status"] = "error"

            return {
                "answer": (
                    "Education router returned invalid arguments. "
                    f"Error: {str(e)}"
                ),
                "tool_trace": tool_trace,
                "sources": []
            }

    else:

        tool_trace[-1]["status"] = "error"

        return {
            "answer": (
                "Education router did not return "
                "a valid routing result."
            ),
            "tool_trace": tool_trace,
            "sources": []
        }

    intents = list(
        dict.fromkeys(intents)
    )

    tool_trace[-1]["arguments"] = json.dumps(
        {
            "intents": intents
        },
        ensure_ascii=False
    )

    tool_trace[-1]["status"] = "success"

    # ========================================================
    # STEP 2 — DETERMINISTIC TOOL EXECUTION
    # ========================================================

    tool_results = []

    # --------------------------------------------------------
    # Calculator
    # --------------------------------------------------------

    if "numerical" in intents:

        trace = {
            "step": len(tool_trace) + 1,
            "tool": "calculator",
            "status": "running",
            "arguments": "{}"
        }

        tool_trace.append(trace)

        calculation_result = execute_calculator(
            contextual_question
        )

        trace["arguments"] = json.dumps(
            {
                "expression": calculation_result.get(
                    "expression",
                    ""
                )
            },
            ensure_ascii=False
        )

        trace["status"] = (
            "success"
            if calculation_result["success"]
            else "error"
        )

        tool_results.append(
            {
                "tool": "calculator",
                "data": calculation_result
            }
        )

    # --------------------------------------------------------
    # Web Search
    # --------------------------------------------------------

    if "current_information" in intents:

        trace = {
            "step": len(tool_trace) + 1,
            "tool": "web_search",
            "status": "running",
            "arguments": json.dumps(
                {
                    "query": contextual_question
                },
                ensure_ascii=False
            )
        }

        tool_trace.append(trace)

        search_result = execute_web_search(
            contextual_question
        )

        if search_result["success"]:

            trace["status"] = "success"

            try:

                parsed_result = json.loads(
                    search_result["result"]
                )

                if isinstance(
                    parsed_result,
                    list
                ):

                    for source in parsed_result:

                        source["source_id"] = (
                            len(web_sources) + 1
                        )

                        web_sources.append(
                            source
                        )

            except Exception:

                pass

        else:

            trace["status"] = "error"

        tool_results.append(
            {
                "tool": "web_search",
                "data": search_result
            }
        )

    # --------------------------------------------------------
    # Quiz Generator
    # --------------------------------------------------------

    if "quiz" in intents:

        trace = {
            "step": len(tool_trace) + 1,
            "tool": "quiz_generator",
            "status": "running",
            "arguments": json.dumps(
                {
                    "question": contextual_question
                },
                ensure_ascii=False
            )
        }

        tool_trace.append(trace)

        quiz_result = execute_quiz(
            contextual_question
        )

        trace["status"] = (
            "success"
            if quiz_result["success"]
            else "error"
        )

        tool_results.append(
            {
                "tool": "quiz_generator",
                "data": quiz_result
            }
        )

    # --------------------------------------------------------
    # Study Plan Generator
    # --------------------------------------------------------

    if "study_plan" in intents:

        trace = {
            "step": len(tool_trace) + 1,
            "tool": "study_plan_generator",
            "status": "running",
            "arguments": json.dumps(
                {
                    "question": contextual_question
                },
                ensure_ascii=False
            )
        }

        tool_trace.append(trace)

        study_result = execute_study_plan(
            contextual_question
        )

        trace["status"] = (
            "success"
            if study_result["success"]
            else "error"
        )

        tool_results.append(
            {
                "tool": "study_plan_generator",
                "data": study_result
            }
        )

    # ========================================================
    # STEP 3 — FINAL AI RESPONSE
    # ========================================================

    final_context = {
        "user_question": question,
        "resolved_question": contextual_question,
        "detected_intents": intents,
        "tool_results": tool_results
    }

    final_messages = [
        {
            "role": "system",
            "content": (
                "You are EduAgent AI, an intelligent education "
                "assistant.\n\n"

                "Answer the user's question using the tool "
                "results provided below.\n\n"

                "IMPORTANT RULES:\n"
                "1. Use the calculator result for numerical answers.\n"
                "2. Do not recalculate numerical results yourself "
                "when a calculator result is available.\n"
                "3. For current information, use only information "
                "returned by web_search.\n"
                "4. Do not invent sources, URLs, facts, or citations.\n"
                "5. If a tool failed or information is insufficient, "
                "clearly say so.\n"
                "6. If multiple tasks exist, answer ALL of them.\n"
                "7. Give a clear, well-structured educational answer.\n"
                "8. Do not mention internal orchestration unless "
                "the user asks about it."
            )
        },
        {
            "role": "user",
            "content": json.dumps(
                final_context,
                ensure_ascii=False,
                indent=2
            )
        }
    ]

    try:

        final_response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=final_messages,
            temperature=0.2
        )

        final_answer = (
            final_response.choices[0].message.content
            or "I could not generate a final answer."
        )

    except Exception as e:

        final_answer = (
            "EduAgent AI completed the required tools, "
            "but could not generate the final response. "
            f"Error: {str(e)}"
        )

    # ========================================================
    # STEP 4 — SAVE CONVERSATION MEMORY
    # ========================================================

    conversation_memory.setdefault(
        session_id,
        []
    )

    conversation_memory[session_id].append(
        {
            "question": question,
            "answer": final_answer
        }
    )

    # Keep only recent conversations
    conversation_memory[session_id] = (
        conversation_memory[session_id][-MAX_HISTORY:]
    )

    # ========================================================
    # FINAL TRACE
    # ========================================================

    tool_trace.append(
        {
            "step": len(tool_trace) + 1,
            "tool": "final_response",
            "status": "success",
            "arguments": "{}"
        }
    )

    return {
        "answer": final_answer,
        "tool_trace": tool_trace,
        "sources": web_sources
    }

