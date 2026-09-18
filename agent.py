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
    weak_topic_detector,
    quiz_result_analyzer,
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
# Learning Progress Memory
# ============================================================

learning_progress = {}

MAX_PROGRESS_ITEMS = 20


# ============================================================
# Helper: Extract Calculation
# ============================================================

def extract_calculation(question: str):

    text = question.lower().strip()

    percentage_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:of)\s*"
        r"(\d+(?:\.\d+)?)",
        text
    )

    if percentage_match:

        percentage = float(percentage_match.group(1))
        number = float(percentage_match.group(2))

        return f"({percentage} / 100) * {number}"

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
# Helper: Extract Learning Progress
# ============================================================

def extract_learning_progress(question: str):

    extraction_prompt = [
        {
            "role": "system",
            "content": (
                "You extract ONLY explicit learning-performance "
                "information from a user's education-related message.\n\n"

                "Return ONLY valid JSON with these keys:\n"
                "subject, topic, score, score_type, note\n\n"

                "Rules:\n"
                "1. Only record information explicitly stated by the user.\n"
                "2. Do not guess or infer a score.\n"
                "3. score must be a number or null.\n"
                "4. score_type can be percentage, marks, or null.\n"
                "5. subject and topic can be empty strings if not stated.\n"
                "6. note should contain a short explicit performance note.\n"
                "7. If there is no learning-performance information, "
                "return score as null and note as an empty string.\n\n"

                "Example:\n"
                "User: I scored 60% in my photosynthesis quiz.\n"
                "JSON: {\n"
                "  \"subject\": \"\",\n"
                "  \"topic\": \"photosynthesis\",\n"
                "  \"score\": 60,\n"
                "  \"score_type\": \"percentage\",\n"
                "  \"note\": \"Scored 60% in a photosynthesis quiz.\"\n"
                "}"
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

        data = json.loads(content)

        score = data.get("score")

        if score is not None:

            try:
                score = float(score)
            except Exception:
                score = None

        return {
            "subject": data.get("subject", ""),
            "topic": data.get("topic", ""),
            "score": score,
            "score_type": data.get("score_type"),
            "note": data.get("note", "")
        }

    except Exception:

        return {
            "subject": "",
            "topic": "",
            "score": None,
            "score_type": None,
            "note": ""
        }

# ============================================================
# Helper: Save Learning Progress
# ============================================================

def save_learning_progress(
    session_id: str,
    progress: dict
):

    if progress.get("score") is None:
        return

    learning_progress.setdefault(
        session_id,
        []
    )

    new_record = {
        "subject": progress.get(
            "subject",
            ""
        ),
        "topic": progress.get(
            "topic",
            ""
        ),
        "score": progress.get(
            "score"
        ),
        "score_type": progress.get(
            "score_type"
        ),
        "note": progress.get(
            "note",
            ""
        )
    }

    topic = (
        new_record["topic"]
        .strip()
        .lower()
    )

    subject = (
        new_record["subject"]
        .strip()
        .lower()
    )

    updated = False

    # --------------------------------------------------------
    # Update existing topic record
    # --------------------------------------------------------

    for index, old_record in enumerate(
        learning_progress[session_id]
    ):

        old_topic = (
            old_record.get(
                "topic",
                ""
            )
            .strip()
            .lower()
        )

        old_subject = (
            old_record.get(
                "subject",
                ""
            )
            .strip()
            .lower()
        )

        if (
            topic
            and old_topic == topic
            and old_subject == subject
        ):

            learning_progress[session_id][index] = (
                new_record
            )

            updated = True

            break

    # --------------------------------------------------------
    # Add new topic record
    # --------------------------------------------------------

    if not updated:

        learning_progress[session_id].append(
            new_record
        )

    # --------------------------------------------------------
    # Keep memory within limit
    # --------------------------------------------------------

    learning_progress[session_id] = (
        learning_progress[session_id][
            -MAX_PROGRESS_ITEMS:
        ]
    )



        
            


# ============================================================
# Helper: Resolve Conversation Context
# ============================================================

def resolve_context(
    question: str,
    history: list,
    progress: list
):

    if not history and not progress:
        return question

    history_text = "\n".join(
        [
            f"User: {item['question']}\n"
            f"Assistant: {item['answer']}"
            for item in history[-MAX_HISTORY:]
        ]
    )

    progress_text = json.dumps(
        progress[-MAX_PROGRESS_ITEMS:],
        ensure_ascii=False,
        indent=2
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a conversation context resolver for "
                "an education AI agent.\n\n"

                "Your job is to convert the CURRENT user question "
                "into a standalone question when it depends on "
                "previous conversation or explicitly stored learning "
                "progress.\n\n"

                "Rules:\n"
                "1. Use previous conversation only when necessary.\n"
                "2. Use learning progress only when it is relevant.\n"
                "3. Resolve words like it, this, that, these, those, "
                "same subject, continue, change, modify, add, remove, "
                "make it easier, make it harder, etc.\n"
                "4. Preserve the user's actual requested change.\n"
                "5. If the current question is already standalone, "
                "return it unchanged.\n"
                "6. Do not invent learning scores or progress.\n"
                "7. Do not answer the question.\n"
                "8. Return ONLY the resolved standalone question.\n\n"

                "Example:\n"
                "Previous: User created a Biology study plan.\n"
                "Current: Make it 2 hours per day.\n"
                "Resolved: Change the Biology study plan to 2 hours "
                "per day.\n\n"

                "Learning progress example:\n"
                "Stored progress: User scored 60% in photosynthesis.\n"
                "Current: What should I revise next?\n"
                "Resolved: Based on my Biology learning progress, "
                "what should I revise next, considering my photosynthesis "
                "quiz score?"
            )
        },
        {
            "role": "user",
            "content": (
                "Previous conversation:\n\n"
                f"{history_text}\n\n"

                "Stored learning progress:\n\n"
                f"{progress_text}\n\n"

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

        return (
            resolved
            if resolved
            else question
        )

    except Exception:

        return question

def execute_quiz_result(question: str):

    try:

        # ----------------------------------------------------
        # Extract percentage score
        # Example:
        # "I scored 80% in my photosynthesis quiz."
        # ----------------------------------------------------

        score_match = re.search(
            r"(\d+(?:\.\d+)?)\s*%",
            question
        )

        if not score_match:

            return {
                "success": False,
                "result": (
                    "Quiz result analyzer error: "
                    "No percentage score found."
                )
            }

        score = float(
            score_match.group(1)
        )

        # ----------------------------------------------------
        # Extract topic
        # ----------------------------------------------------

        topic = ""

        topic_match = re.search(
            r"(?:in|on)\s+(?:my\s+)?(.+?)\s+quiz",
            question,
            re.IGNORECASE
        )

        if topic_match:

            topic = topic_match.group(1).strip()

        # ----------------------------------------------------
        # Clean topic
        # ----------------------------------------------------

        topic = re.sub(
            r"\s+",
            " ",
            topic
        )

        # ----------------------------------------------------
        # Run existing analyzer
        # ----------------------------------------------------

        result = quiz_result_analyzer(
            subject="",
            topic=topic,
            score=score,
            score_type="percentage",
            total_questions=0,
            correct_answers=0
        )

        return {
            "success": True,
            "result": result
        }

    except Exception as e:

        return {
            "success": False,
            "result": (
                f"Quiz result analyzer error: {str(e)}"
            )
        }

# ============================================================
# Helper: Generate Targeted Revision
# ============================================================

def generate_targeted_revision(
    progress: list
):

    weak_topics = weak_topic_detector(
        progress
    )

    try:

        weak_topics_data = json.loads(
            weak_topics
        )

    except Exception:

        return {
            "success": False,
            "result": "Could not read weak topics."
        }

    if not weak_topics_data:

        return {
            "success": True,
            "result": (
                "No weak topics are currently recorded."
            )
        }

    revision_items = []

    for item in weak_topics_data:

        revision_items.append(
            {
                "subject": item.get(
                    "subject",
                    ""
                ),
                "topic": item.get(
                    "topic",
                    ""
                ),
                "score": item.get(
                    "score"
                ),
                "revision_action": (
                    "Revise the core concepts and "
                    "practice targeted questions."
                )
            }
        )

    return {
        "success": True,
        "result": json.dumps(
            revision_items,
            ensure_ascii=False
        )
    }



    

        
            

            
                
                    
            







    

        

        





                
        
                
            


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

    progress = learning_progress.get(
        session_id,
        []
    )

    # ========================================================
    # STEP 0.5 — EXTRACT LEARNING PROGRESS
    # ========================================================

    progress_data = extract_learning_progress(
        question
    )

    save_learning_progress(
        session_id,
        progress_data
    )

    progress = learning_progress.get(
        session_id,
        []
    )

    print(
        "DEBUG LEARNING PROGRESS:",
        progress
    )

    print(
        "DEBUG SESSION ID:",
        session_id
    )

    # ========================================================
    # STEP 0.6 — RESOLVE CONTEXT
    # ========================================================

    contextual_question = resolve_context(
        question,
        history,
        progress
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
                "- current_information\n"
                "- weak_topic\n"
                "- quiz_result\n\n"

                "IMPORTANT ROUTING RULES:\n\n"

                "QUIZ RESULT RULE:\n"
                "If the user explicitly reports a quiz or test score, "
                "such as 'I scored 80%', 'I got 65%', "
                "'I scored 8 out of 10', or similar performance "
                "information, use quiz_result.\n\n"

                "If the user reports a score and asks to analyze, "
                "interpret, classify, evaluate, or understand "
                "their performance, use quiz_result.\n\n"

                "A quiz score must route to quiz_result regardless "
                "of whether the score is high, medium, or low.\n\n"

                "Do NOT use weak_topic merely because a score is "
                "mentioned.\n\n"

                "WEAK TOPIC RULE:\n"
                "Use weak_topic when the user asks about weak areas, "
                "what topic needs revision, what should be revised "
                "next, or which learning topics need more revision "
                "based on stored learning performance.\n\n"

                "STUDY PLAN RULE:\n"
                "Use study_plan when the user explicitly asks for "
                "a study schedule or study plan.\n\n"

                "QUIZ GENERATOR RULE:\n"
                "Use quiz when the user asks to create or generate "
                "a quiz, test, MCQs, or practice questions.\n\n"

                "EXPLANATION RULE:\n"
                "Use explanation when the user asks to explain or "
                "learn an educational concept.\n\n"

                "NUMERICAL RULE:\n"
                "Use numerical when the user asks to calculate or "
                "solve a mathematical/numerical problem.\n\n"

                "CURRENT INFORMATION RULE:\n"
                "Use current_information when the user asks for "
                "latest, current, recent, today's, or otherwise "
                "time-sensitive information.\n\n"

                "MULTIPLE INTENTS:\n"
                "If multiple tasks exist, return ALL applicable intents.\n\n"

                "PRIORITY:\n"
                "1. Explicit quiz/test score + performance analysis "
                "-> quiz_result\n"
                "2. Weak-topic/revision request based on stored progress "
                "-> weak_topic\n"
                "3. Explicit study schedule request -> study_plan\n"
                "4. Quiz creation request -> quiz\n"
                "5. Numerical calculation -> numerical\n"
                "6. Current/latest information -> current_information\n"
                "7. Concept learning/explanation -> explanation\n\n"

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
            tool_choice="required",
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

    # --------------------------------------------------------
    # Weak Topic Detector
    # --------------------------------------------------------

    if "weak_topic" in intents:

        trace = {
            "step": len(tool_trace) + 1,
            "tool": "weak_topic_detector",
            "status": "running",
            "arguments": json.dumps(
                {
                    "progress_items": len(progress)
                },
                ensure_ascii=False
            )
        }

        tool_trace.append(trace)

        try:

            weak_topic_result = weak_topic_detector(
                progress
            )

            trace["status"] = "success"

            tool_results.append(
                {
                    "tool": "weak_topic_detector",
                    "data": {
                        "success": True,
                        "result": weak_topic_result
                    }
                }
            )

        except Exception as e:

            trace["status"] = "error"

            tool_results.append(
                {
                    "tool": "weak_topic_detector",
                    "data": {
                        "success": False,
                        "result": (
                            f"Weak topic detector error: {str(e)}"
                        )
                    }
                }
            )

    # --------------------------------------------------------
    # Quiz Result Analyzer
    # --------------------------------------------------------

    if "quiz_result" in intents:

        trace = {
            "step": len(tool_trace) + 1,
            "tool": "quiz_result_analyzer",
            "status": "running",
            "arguments": json.dumps(
                {
                    "question": contextual_question
                },
                ensure_ascii=False
            )
        }

        tool_trace.append(trace)

        quiz_result_analysis = execute_quiz_result(
            contextual_question
        )

        trace["status"] = (
            "success"
            if quiz_result_analysis["success"]
            else "error"
        )
        trace["result"] = quiz_result_analysis.get(
           "result",
           ""
        )
            
            
        

        tool_results.append(
            {
                "tool": "quiz_result_analyzer",
                "data": quiz_result_analysis
            }
        )

    # ========================================================
    # STEP 3 — FINAL AI RESPONSE
    # ========================================================

    final_context = {
        "user_question": question,
        "resolved_question": contextual_question,
        "detected_intents": intents,
        "learning_progress": progress,
        "tool_results": tool_results
    }

    final_messages = [
        {
            "role": "system",
            "content": (
                "You are EduAgent AI, an intelligent education "
                "assistant.\n\n"

                "Answer the user's question using the tool "
                "results and learning progress provided below.\n\n"

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
                "7. Use stored learning progress only when relevant.\n"
                "8. Never invent or assume a user's score.\n"
                "9. When weak_topic_detector returns weak topics, "
                "clearly identify those topics and explain that they "
                "are flagged because their recorded percentage score "
                "is below 70%.\n"
                "10. If no weak topics are returned, do not invent any. "
                "Say that no recorded topic currently meets the "
                "weak-topic threshold.\n"
                "11. When quiz_result_analyzer returns a performance "
                "classification, clearly explain the classification "
                "and its recommendation.\n"
                "12. Give a clear, well-structured educational answer.\n"
                "13. Do not mention internal orchestration unless "
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
                




