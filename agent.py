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


def extract_calculation(question: str):
    """
    Detect common mathematical expressions from
    natural-language questions.
    """

    text = question.lower().strip()

    # --------------------------------------------------------
    # Percentage of a number
    # Examples:
    # 25% of 2400
    # 35 percent of 240 students
    # --------------------------------------------------------
    percentage_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:of)\s*"
        r"(\d+(?:\.\d+)?)",
        text
    )

    if percentage_match:
        percentage = float(percentage_match.group(1))
        number = float(percentage_match.group(2))

        return f"({percentage} / 100) * {number}"

    # --------------------------------------------------------
    # Percentage with total appearing BEFORE percentage
    #
    # Examples:
    # 240 students and 35% are girls
    # 500 people, 20% are children
    # 100 students have 30% girls
    # --------------------------------------------------------
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

        # Avoid treating a percentage followed by another
        # number as a reverse percentage calculation.
        if percentage <= 100:

            return (
                f"({percentage} / 100) * {number}"
            )

    # --------------------------------------------------------
    # Percentage with "out of"
    #
    # Example:
    # 35% out of 240 students
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Basic arithmetic expressions
    #
    # Examples:
    # 125 * 48
    # 500 + 250
    # 1000 / 25
    # --------------------------------------------------------
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
# Helper: Extract web-search query
# ============================================================

def extract_web_query(question: str):
    """
    For current/latest/recent questions, the original
    user question is a safe search query.
    """

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

    # Ask Groq to extract quiz parameters.
    # This is NOT the main tool orchestration.
    # The router has already identified the quiz intent.

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

        # Remove accidental markdown fences
        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

        params = json.loads(content)

        subject = params.get("subject", "General")
        topic = params.get("topic", "General")
        number_of_questions = int(
            params.get("number_of_questions", 10)
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
# Helper: Execute Study Plan Generator
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
            params.get("days", 7)
        )

        hours_per_day = float(
            params.get("hours_per_day", 2)
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
# Main Agent
# ============================================================

def ask_agent(question: str):

    question = question.strip()

    if not question:

        return {
            "answer": "Please enter a question.",
            "tool_trace": [],
            "sources": []
        }

    tool_trace = []
    web_sources = []

    # ========================================================
    # STEP 1 — EDUCATION ROUTER MUST RUN FIRST
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

                "Example:\n"
                "25% of 2400 and latest AI developments in 2026\n"
                "must return:\n"
                "[\"numerical\", \"current_information\"]\n\n"

                "Do not answer the user's question."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    try:

        # ----------------------------------------------------
        # FORCE router to run first
        # ----------------------------------------------------

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

    router_message = router_response.choices[0].message

    # ========================================================
    # STEP 2 — READ ROUTER RESULT
    # ========================================================

    intents = []

    if router_message.tool_calls:

        router_call = router_message.tool_calls[0]

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
                "Education router did not return a valid routing result."
            ),
            "tool_trace": tool_trace,
            "sources": []
        }

    # Remove duplicates while preserving order
    intents = list(dict.fromkeys(intents))

    tool_trace[-1]["arguments"] = json.dumps(
        {
            "intents": intents
        },
        ensure_ascii=False
    )

    tool_trace[-1]["status"] = "success"

    # ========================================================
    # STEP 3 — DETERMINISTIC TOOL EXECUTION
    # ========================================================

    tool_results = []

    # --------------------------------------------------------
    # NUMERICAL
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
            question
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

        if calculation_result["success"]:

            trace["status"] = "success"

        else:

            trace["status"] = "error"

        tool_results.append(
            {
                "tool": "calculator",
                "data": calculation_result
            }
        )

    # --------------------------------------------------------
    # CURRENT INFORMATION
    # --------------------------------------------------------

    if "current_information" in intents:

        trace = {
            "step": len(tool_trace) + 1,
            "tool": "web_search",
            "status": "running",
            "arguments": json.dumps(
                {
                    "query": question
                },
                ensure_ascii=False
            )
        }

        tool_trace.append(trace)

        search_result = execute_web_search(
            question
        )

        if search_result["success"]:

            trace["status"] = "success"

            try:

                parsed_result = json.loads(
                    search_result["result"]
                )

                if isinstance(parsed_result, list):

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
    # QUIZ
    # --------------------------------------------------------

    if "quiz" in intents:

        trace = {
            "step": len(tool_trace) + 1,
            "tool": "quiz_generator",
            "status": "running",
            "arguments": json.dumps(
                {
                    "question": question
                },
                ensure_ascii=False
            )
        }

        tool_trace.append(trace)

        quiz_result = execute_quiz(
            question
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
    # STUDY PLAN
    # --------------------------------------------------------

    if "study_plan" in intents:

        trace = {
            "step": len(tool_trace) + 1,
            "tool": "study_plan_generator",
            "status": "running",
            "arguments": json.dumps(
                {
                    "question": question
                },
                ensure_ascii=False
            )
        }

        tool_trace.append(trace)

        study_result = execute_study_plan(
            question
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
    # STEP 4 — FINAL AI RESPONSE
    # ========================================================

    final_context = {
        "user_question": question,
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
    # STEP 5 — FINAL RESPONSE TRACE
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

