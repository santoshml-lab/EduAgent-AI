import os
import json
import re

from dotenv import load_dotenv
from groq import Groq

from tools import (
    TOOLS,
    calculator,
    web_search,
    quiz_generator,
    study_plan_generator,
    weak_topic_detector,
    quiz_result_analyzer
)

load_dotenv()


# ============================================================
# Groq Client
# ============================================================

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# ============================================================
# Conversation Memory
# ============================================================

conversation_memory = {}

MAX_HISTORY = 6

def get_conversation_history(session_id: str):

    history = conversation_memory.get(
        session_id,
        []
    )

    messages = []

    for item in history:
        messages.append(
            {
                "role": "user",
                "content": item.get(
                    "user",
                    ""
                )
            }
        )

        messages.append(
            {
                "role": "assistant",
                "content": item.get(
                    "assistant",
                    ""
                )
            }
        )

    return messages


# ============================================================
# Learning Progress Memory
# ============================================================

learning_progress = {}

MAX_PROGRESS_ITEMS = 20


# ============================================================
# Helper: Extract Calculation
# ============================================================

def extract_calculation(question: str):

    patterns = [
        r"what is (.+?)(?:\?|$)",
        r"calculate (.+?)(?:\?|$)",
        r"solve (.+?)(?:\?|$)",
        r"find (.+?)(?:\?|$)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            question,
            re.IGNORECASE
        )

        if match:

            expression = match.group(1).strip()

            expression = re.sub(
                r"\b(of|students|people|items|units)\b.*$",
                "",
                expression,
                flags=re.IGNORECASE
            )

            return expression.strip()

    return None


# ============================================================
# Helper: Execute Calculator
# ============================================================

def execute_calculator(question: str):

    expression = extract_calculation(
        question
    )

    if not expression:

        return {
            "success": False,
            "result": "Could not extract calculation."
        }

    try:

        result = calculator(
            expression
        )

        return {
            "success": True,
            "result": result
        }

    except Exception as e:

        return {
            "success": False,
            "result": f"Calculator error: {str(e)}"
        }


# ============================================================
# Helper: Extract Web Query
# ============================================================

def extract_web_query(question: str):

    match = re.search(
        r"(?:latest|recent|current|today|news about|information about)\s+(.+)",
        question,
        re.IGNORECASE
    )

    if match:

        return match.group(1).strip()

    return question


# ============================================================
# Helper: Execute Web Search
# ============================================================

def execute_web_search(question: str):

    query = extract_web_query(
        question
    )

    try:

        result = web_search(
            query
        )

        return {
            "success": True,
            "result": result
        }

    except Exception as e:

        return {
            "success": False,
            "result": f"Web search error: {str(e)}"
        }
# ============================================================
# Helper: Validate Tool Result
# ============================================================

def validate_tool_result(
    question: str,
    tool_name: str,
    tool_result
):

    try:

        response = client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "system",
                    "content": """
You are the validation layer of EduAgent AI.

Check whether the tool result is useful and relevant
for answering the user's question.

Return ONLY valid JSON:

{
  "valid": true,
  "reason": "",
  "needs_retry": false
}

Rules:

- valid = true if the result is relevant and usable.
- valid = false if the result is clearly incorrect,
  empty, irrelevant, or unusable.
- needs_retry = true only when another tool attempt
  could reasonably fix the problem.
- Do not invent facts.
"""
                },
                {
                    "role": "user",
                    "content": f"""
User question:

{question}

Tool used:

{tool_name}

Tool result:

{json.dumps(tool_result, ensure_ascii=False)}

Validate this result.
"""
                }
            ],

            temperature=0
        )

        content = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        content = re.sub(
            r"^```json\s*",
            "",
            content,
            flags=re.IGNORECASE
        )

        content = re.sub(
            r"\s*```$",
            "",
            content
        )

        return json.loads(content)

    except Exception as e:

        return {
            "valid": True,
            "reason": f"Validation fallback: {str(e)}",
            "needs_retry": False
        }





def execute_quiz(question: str):

    # --------------------------------------------------------
    # Number of Questions
    # --------------------------------------------------------

    number_match = re.search(
        r"(\d+)\s*[-]?\s*questions?",
        question,
        re.IGNORECASE
    )

    if number_match:
        number_of_questions = int(number_match.group(1))
    else:
        number_of_questions = 5

    # Keep quiz size within allowed range
    number_of_questions = max(
        1,
        min(number_of_questions, 20)
    )

    # --------------------------------------------------------
    # Difficulty
    # --------------------------------------------------------

    if re.search(r"\bhard\b|\bdifficult\b|\badvanced\b", question, re.IGNORECASE):
        difficulty = "hard"

    elif re.search(r"\beasy\b|\bbasic\b", question, re.IGNORECASE):
        difficulty = "easy"

    else:
        difficulty = "medium"

    # --------------------------------------------------------
    # Subject
    # --------------------------------------------------------

    subjects = [
        "biology",
        "physics",
        "chemistry",
        "mathematics",
        "math",
        "computer",
        "computer science",
        "geography",
        "history",
        "civics",
        "english",
        "economics"
    ]

    subject = "General Studies"

    for item in subjects:
        if re.search(
            rf"\b{re.escape(item)}\b",
            question,
            re.IGNORECASE
        ):
            subject = item
            break

    # Normalize subject names
    subject_map = {
        "biology": "Biology",
        "physics": "Physics",
        "chemistry": "Chemistry",
        "mathematics": "Mathematics",
        "math": "Mathematics",
        "computer": "Computer Science",
        "computer science": "Computer Science",
        "geography": "Geography",
        "history": "History",
        "civics": "Civics",
        "english": "English",
        "economics": "Economics"
    }

    subject = subject_map.get(
        subject.lower(),
        subject
    )

    # --------------------------------------------------------
    # Topic
    # --------------------------------------------------------

    topic = subject

    # Case 1:
    # "quiz on Biology about Cell"
    topic_match = re.search(
        r"(?:on|about)\s+"
        r"(?:biology|physics|chemistry|mathematics|math|"
        r"computer\s+science|computer|geography|history|"
        r"civics|english|economics)"
        r"\s+(?:about|on)\s+"
        r"(.+?)(?:\s+at|\s+with|\s*$)",
        question,
        re.IGNORECASE
    )

    if topic_match:
        topic = topic_match.group(1).strip()

    else:

        # Case 2:
        # "quiz about Cell"
        topic_match = re.search(
            r"(?:about|on)\s+"
            r"(.+?)(?:\s+at|\s+with|\s*$)",
            question,
            re.IGNORECASE
        )

        if topic_match:

            extracted_topic = topic_match.group(1).strip()

            # If extracted topic starts with the subject,
            # remove the subject from it.
            subject_pattern = re.escape(subject)

            extracted_topic = re.sub(
                rf"^{subject_pattern}\s+(?:about|on)\s+",
                "",
                extracted_topic,
                flags=re.IGNORECASE
            ).strip()

            if extracted_topic:
                topic = extracted_topic

    # --------------------------------------------------------
    # Clean Topic
    # --------------------------------------------------------

    topic = topic.strip(" .,?!")

    if not topic:
        topic = subject

    # --------------------------------------------------------
    # Call Quiz Generator
    # --------------------------------------------------------

    result = quiz_generator(
        subject=subject,
        topic=topic,
        number_of_questions=number_of_questions,
        difficulty=difficulty
    )

    # --------------------------------------------------------
    # Return Result
    # --------------------------------------------------------

    return {
      "success": True,
      "subject": subject,
      "topic": topic,
      "number_of_questions": number_of_questions,
      "difficulty": difficulty,
      "instruction": (
        f"Generate {number_of_questions} "
        f"{difficulty}-difficulty questions "
        f"on {topic} in {subject}."
    ),
    "result": result
    }
        
        









    

        
            

        
        


# ============================================================
# Helper: Execute Study Plan
# ============================================================

def execute_study_plan(question: str):

    try:

        days_match = re.search(
            r"(\d+)\s*[-]?\s*day",
            question,
            re.IGNORECASE
        )

        hours_match = re.search(
            r"(\d+(?:\.\d+)?)\s*hours?",
            question,
            re.IGNORECASE
        )

        days = (
            int(days_match.group(1))
            if days_match
            else 7
        )

        hours_per_day = (
            float(hours_match.group(1))
            if hours_match
            else 2
        )

        subject_match = re.search(
            r"(?:for|of)\s+(?:Class\s+\d+\s+)?(.+?)(?:\s+with|\s+for|\s*$)",
            question,
            re.IGNORECASE
        )

        subject = (
            subject_match.group(1).strip()
            if subject_match
            else "General Studies"
        )

        result = study_plan_generator(
            subject=subject,
            days=days,
            hours_per_day=hours_per_day
        )

        return {
            "success": True,
            "result": result
        }

    except Exception as e:

        return {
            "success": False,
            "result": f"Study plan error: {str(e)}"
        }


# ============================================================
# Helper: Extract Learning Progress
# ============================================================

def extract_learning_progress(
    question: str
):

    try:

        response = client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "system",
                    "content": """
Extract learning progress from the user's message.

Return ONLY valid JSON.

Schema:

{
  "subject": "",
  "topic": "",
  "score": null,
  "score_type": "percentage",
  "note": ""
}

If the user did not provide a score,
return score as null.

Do not invent a score.
"""
                },
                {
                    "role": "user",
                    "content": question
                }
            ],

            temperature=0
        )

        content = response.choices[0].message.content

        content = content.strip()

        content = re.sub(
            r"^```json\s*",
            "",
            content,
            flags=re.IGNORECASE
        )

        content = re.sub(
            r"\s*```$",
            "",
            content
        )

        return json.loads(
            content
        )

    except Exception:

        return {
            "subject": "",
            "topic": "",
            "score": None,
            "score_type": "percentage",
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

    if not updated:

        learning_progress[session_id].append(
            new_record
        )

    learning_progress[session_id] = (
        learning_progress[session_id][
            -MAX_PROGRESS_ITEMS:
        ]
    )


# ============================================================
# Helper: Execute Quiz Result
# ============================================================

def execute_quiz_result(
    question: str
):

    try:

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

        topic = ""

        topic_match = re.search(
            r"(?:in|on)\s+(?:my\s+)?(.+?)\s+quiz",
            question,
            re.IGNORECASE
        )

        if topic_match:

            topic = topic_match.group(1).strip()

        topic = re.sub(
            r"\s+",
            " ",
            topic
        )

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

    # --------------------------------------------------------
    # User Question
    # --------------------------------------------------------

    standalone_question = question
    conversation_history = get_conversation_history(
    session_id
    )

    # --------------------------------------------------------
    # Direct Quiz Detection
    # --------------------------------------------------------

    direct_quiz_query = re.search(
        r"\b("
        r"give\s+(?:me\s+)?(?:a\s+)?(?:\d+\s+)?questions?\s+quiz|"
        r"make\s+(?:me\s+)?(?:a\s+)?(?:\d+\s+)?questions?\s+quiz|"
        r"create\s+(?:a\s+)?(?:\d+\s+)?questions?\s+quiz|"
        r"generate\s+(?:a\s+)?(?:\d+\s+)?questions?\s+quiz|"
        r"quiz\s+me"
        r")\b",
        question,
        re.IGNORECASE
    )

    if direct_quiz_query:

        quiz_result = execute_quiz(
            question
        )

        tool_trace = [
            {
                "step": 1,
                "tool": "quiz_generator",
                "status": (
                    "success"
                    if quiz_result["success"]
                    else "error"
                ),
                "arguments": json.dumps(
                    {
                        "question": question
                    },
                    ensure_ascii=False
                ),
                "result": quiz_result.get(
                    "result",
                    ""
                )
            }
        ]

        answer = quiz_result.get(
            "result",
            ""
        )

        tool_trace.append(
            {
                "step": 2,
                "tool": "final_response",
                "status": "success",
                "result": answer
            }
        )

        # ----------------------------------------------------
        # Save Conversation Memory
        # ----------------------------------------------------

        conversation_memory.setdefault(
            session_id,
            []
        )

        conversation_memory[session_id].append(
            {
                "user": question,
                "assistant": answer
            }
        )

        conversation_memory[session_id] = (
            conversation_memory[session_id][
                -MAX_HISTORY:
            ]
        )

        return {
            "answer": answer,
            "tool_trace": tool_trace,
            "sources": []
        }

    # --------------------------------------------------------
    # Learning Progress Extraction
    # --------------------------------------------------------

    progress_update = extract_learning_progress(
        question
    )

    save_learning_progress(
        session_id,
        progress_update
    )

    # --------------------------------------------------------
    # Current Progress
    # --------------------------------------------------------

    progress = learning_progress.get(
        session_id,
        []
    )

    # ========================================================
    # Direct Weak Topic / Revision Detection
    # ========================================================

    weak_topic_match = re.search(
        r"\b("
        r"what should i revise|"
        r"what should i study|"
        r"what do i need to revise|"
        r"what do i need to study|"
        r"which topic should i revise|"
        r"which topic should i study|"
        r"what should i work on|"
        r"where should i focus|"
        r"what are my weak topics|"
        r"show my weak topics|"
        r"find my weak topics"
        r")\b",
        standalone_question,
        re.IGNORECASE
    )

    if weak_topic_match:

        weak_topics_result = weak_topic_detector(
            progress
        )

        revision_result = generate_targeted_revision(
            progress
        )

        tool_trace = []

        # ----------------------------------------------------
        # Weak Topic Detector Trace
        # ----------------------------------------------------

        tool_trace.append(
            {
                "step": 1,
                "tool": "weak_topic_detector",
                "status": "success",
                "arguments": json.dumps(
                    {
                        "progress_items": len(progress)
                    },
                    ensure_ascii=False
                ),
                "result": weak_topics_result
            }
        )

        # ----------------------------------------------------
        # Targeted Revision Trace
        # ----------------------------------------------------

        tool_trace.append(
            {
                "step": 2,
                "tool": "targeted_revision",
                "status": (
                    "success"
                    if revision_result["success"]
                    else "error"
                ),
                "arguments": json.dumps(
                    {
                        "progress_items": len(progress)
                    },
                    ensure_ascii=False
                ),
                "result": revision_result.get(
                    "result",
                    ""
                )
            }
        )

        # ----------------------------------------------------
        # Generate Student-Friendly Answer
        # ----------------------------------------------------

        try:

            final_response = client.chat.completions.create(

                model="openai/gpt-oss-20b",

                messages=[
                    {
                        "role": "system",
                        "content": """
You are EduAgent AI.

Use the learning progress and revision results
to give the user a clear, student-friendly answer.

Identify the weak topic from the available data.

Give a practical revision recommendation.

Do not mention tools, routing, internal architecture,
or implementation details.
"""
                    },
                    {
                        "role": "user",
                        "content": f"""
User question:

{standalone_question}

Learning progress:

{json.dumps(progress, ensure_ascii=False)}

Weak topic result:

{weak_topics_result}

Targeted revision:

{revision_result.get("result", "")}

Write a concise and useful revision recommendation.
"""
                    }
                ],

                temperature=0.2
            )

            answer = (
                final_response
                .choices[0]
                .message
                .content
            )

        except Exception as e:

            answer = (
                f"Revision response error: {str(e)}"
            )

        # ----------------------------------------------------
        # Final Response Trace
        # ----------------------------------------------------

        tool_trace.append(
            {
                "step": 3,
                "tool": "final_response",
                "status": "success",
                "result": answer
            }
        )

        # ----------------------------------------------------
        # Save Conversation Memory
        # ----------------------------------------------------

        conversation_memory.setdefault(
            session_id,
            []
        )

        conversation_memory[session_id].append(
            {
                "user": question,
                "assistant": answer
            }
        )

        conversation_memory[session_id] = (
            conversation_memory[session_id][
                -MAX_HISTORY:
            ]
        )

        return {
            "answer": answer,
            "tool_trace": tool_trace,
            "sources": []
        }

    # ========================================================
    # Direct Quiz Result Detection
    # ========================================================

    quiz_result_match = re.search(
        r"\b(?:scored|got|achieved|received)\s+(\d+(?:\.\d+)?)\s*%",
        standalone_question,
        re.IGNORECASE
    )

    if quiz_result_match:

        quiz_result_analysis = execute_quiz_result(
            standalone_question
        )

        tool_trace = [
            {
                "step": 1,
                "tool": "quiz_result_analyzer",
                "status": (
                    "success"
                    if quiz_result_analysis["success"]
                    else "error"
                ),
                "arguments": json.dumps(
                    {
                        "question": standalone_question
                    },
                    ensure_ascii=False
                ),
                "result": quiz_result_analysis.get(
                    "result",
                    ""
                )
            }
        ]

        answer = quiz_result_analysis.get(
            "result",
            ""
        )

        # ----------------------------------------------------
        # Save Conversation Memory
        # ----------------------------------------------------

        conversation_memory.setdefault(
            session_id,
            []
        )

        conversation_memory[session_id].append(
            {
                "user": question,
                "assistant": answer
            }
        )

        conversation_memory[session_id] = (
            conversation_memory[session_id][
                -MAX_HISTORY:
            ]
        )

        return {
            "answer": answer,
            "tool_trace": tool_trace,
            "sources": []
        }

    # --------------------------------------------------------
    # Router Tools
    # --------------------------------------------------------

    router_tools = [
        tool
        for tool in TOOLS
        if tool["function"]["name"]
        != "quiz_result_analyzer"
    ]

    # --------------------------------------------------------
    # Router
    # --------------------------------------------------------

    router_messages = [

    {
        "role": "system",
        "content": """
You are the intent router for an educational AI agent.

Classify the user's request into one or more of these intents:

1. explanation
2. numerical
3. quiz
4. study_plan
5. current_information
6. weak_topic
7. quiz_result

IMPORTANT:

For quiz_result:
- Only classify the request as quiz_result.
- DO NOT call quiz_result_analyzer.
- Quiz result analysis is handled separately by the application.

Priority:

quiz_result
weak_topic
study_plan
quiz
numerical
current_information
explanation

Use the available tools only when appropriate.

Always choose the most relevant intent.
"""
    },

    *conversation_history,

    {
        "role": "user",
        "content": standalone_question
    }

    ]

    try:

        router_response = client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=router_messages,

            tools=router_tools,

            tool_choice="auto",

            temperature=0
        )

    except Exception as e:

        return {
            "answer": f"Router error: {str(e)}",
            "tool_trace": [],
            "sources": []
        }

    # --------------------------------------------------------
    # Read Router Tool Calls
    # --------------------------------------------------------

    tool_calls = (
        router_response
        .choices[0]
        .message
        .tool_calls
    )

    intents = []

    if tool_calls:

        for tool_call in tool_calls:

            try:

                arguments = json.loads(
                    tool_call.function.arguments
                )

            except Exception:

                arguments = {}

            intent = arguments.get(
                "intent"
            )

            if intent:

                intents.append(
                    intent
                )

    # --------------------------------------------------------
    # Tool Execution
    # --------------------------------------------------------

    tool_trace = []

    tool_results = []

    web_sources = []

    # --------------------------------------------------------
    # Numerical
    # --------------------------------------------------------

    if "numerical" in intents:

        trace = {
            "step": len(tool_trace) + 1,
            "tool": "calculator",
            "status": "running",
            "arguments": json.dumps(
                {
                    "question": standalone_question
                },
                ensure_ascii=False
            )
        }

        tool_trace.append(
            trace
        )

                calculator_result = execute_calculator(
            standalone_question
        )

        trace["status"] = (
            "success"
            if calculator_result["success"]
            else "error"
        )

        trace["result"] = calculator_result.get(
            "result",
            ""
        )

        # ----------------------------------------------------
        # Validate Calculator Result
        # ----------------------------------------------------

        validation_result = validate_tool_result(
            question=standalone_question,
            tool_name="calculator",
            tool_result=calculator_result
        )

        tool_trace.append(
            {
                "step": len(tool_trace) + 1,
                "tool": "result_validator",
                "status": (
                    "success"
                    if validation_result.get("valid", False)
                    else "rejected"
                ),
                "arguments": json.dumps(
                    {
                        "validated_tool": "calculator"
                    },
                    ensure_ascii=False
                ),
                "result": validation_result
            }
        )

        tool_results.append(
            {
                "tool": "calculator",
                "data": calculator_result,
                "validation": validation_result
            }
        )
            

    # --------------------------------------------------------
    # Current Information
    # --------------------------------------------------------

    if "current_information" in intents:

        trace = {
            "step": len(tool_trace) + 1,
            "tool": "web_search",
            "status": "running",
            "arguments": json.dumps(
                {
                    "question": standalone_question
                },
                ensure_ascii=False
            )
        }

        tool_trace.append(
            trace
        )

        web_result = execute_web_search(
            standalone_question
        )

        trace["status"] = (
            "success"
            if web_result["success"]
            else "error"
        )

        trace["result"] = web_result.get(
            "result",
            ""
        )

        tool_results.append(
            {
                "tool": "web_search",
                "data": web_result
            }
        )

        try:

            parsed_sources = json.loads(
                web_result.get(
                    "result",
                    "[]"
                )
            )

            if isinstance(
                parsed_sources,
                list
            ):

                web_sources = parsed_sources

        except Exception:

            web_sources = []

    # --------------------------------------------------------
    # Quiz
    # --------------------------------------------------------

    if "quiz" in intents:

        trace = {
            "step": len(tool_trace) + 1,
            "tool": "quiz_generator",
            "status": "running",
            "arguments": json.dumps(
                {
                    "question": standalone_question
                },
                ensure_ascii=False
            )
        }

        tool_trace.append(
            trace
        )

        quiz_result = execute_quiz(
            standalone_question
        )

        trace["status"] = (
            "success"
            if quiz_result["success"]
            else "error"
        )

        trace["result"] = quiz_result.get(
            "result",
            ""
        )

        tool_results.append(
            {
                "tool": "quiz_generator",
                "data": quiz_result
            }
        )

    # --------------------------------------------------------
    # Study Plan
    # --------------------------------------------------------

    if "study_plan" in intents:

        trace = {
            "step": len(tool_trace) + 1,
            "tool": "study_plan_generator",
            "status": "running",
            "arguments": json.dumps(
                {
                    "question": standalone_question
                },
                ensure_ascii=False
            )
        }

        tool_trace.append(
            trace
        )

        study_plan_result = execute_study_plan(
            standalone_question
        )

        trace["status"] = (
            "success"
            if study_plan_result["success"]
            else "error"
        )

        trace["result"] = study_plan_result.get(
            "result",
            ""
        )

        tool_results.append(
            {
                "tool": "study_plan_generator",
                "data": study_plan_result
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

        tool_trace.append(
            trace
        )

        weak_topics_result = weak_topic_detector(
            progress
        )

        trace["status"] = "success"

        trace["result"] = weak_topics_result

        tool_results.append(
            {
                "tool": "weak_topic_detector",
                "data": {
                    "success": True,
                    "result": weak_topics_result
                }
            }
        )

    # --------------------------------------------------------
    # Targeted Revision
    # --------------------------------------------------------

    if "weak_topic" in intents:

        trace = {
            "step": len(tool_trace) + 1,
            "tool": "targeted_revision",
            "status": "running",
            "arguments": json.dumps(
                {
                    "progress_items": len(progress)
                },
                ensure_ascii=False
            )
        }

        tool_trace.append(
            trace
        )

        revision_result = generate_targeted_revision(
            progress
        )

        trace["status"] = (
            "success"
            if revision_result["success"]
            else "error"
        )

        trace["result"] = revision_result.get(
            "result",
            ""
        )

        tool_results.append(
            {
                "tool": "targeted_revision",
                "data": revision_result
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
                    "question": standalone_question
                },
                ensure_ascii=False
            )
        }

        tool_trace.append(
            trace
        )

        quiz_result_analysis = execute_quiz_result(
            standalone_question
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

    # --------------------------------------------------------
    # Final Answer
    # --------------------------------------------------------

    if not tool_results:

         final_answer_messages = [

        {
            "role": "system",
            "content": """
You are EduAgent AI, an educational AI assistant.

Give a clear, accurate and student-friendly answer.

Use simple language.

If the question is educational,
explain the concept step by step when useful.
"""
        },

        *conversation_history,

        {
            "role": "user",
            "content": standalone_question
        }

    ]

    else:

        final_answer_messages = [

            {
                "role": "system",
                "content": """
You are EduAgent AI.

Generate the final answer using the tool results.

Do not mention internal routing,
tool execution,
agent architecture,
or hidden implementation details.

Give the user a clear and useful educational response.

If current information was retrieved,
use the retrieved information carefully.

If learning progress or weak topics are present,
give actionable revision guidance.
"""
            },
            *conversation_history,
            

            {
                "role": "user",
                "content": f"""
User question:

{standalone_question}

Learning progress:

{json.dumps(progress, ensure_ascii=False)}

Tool results:

{json.dumps(tool_results, ensure_ascii=False)}

Write the final answer.
"""
            }

        ]

    # --------------------------------------------------------
    # Generate Final Answer
    # --------------------------------------------------------

    try:

        final_response = client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=final_answer_messages,

            temperature=0.2
        )

        final_answer = (
            final_response
            .choices[0]
            .message
            .content
        )

    except Exception as e:

        final_answer = (
            f"Final response error: {str(e)}"
        )

    # --------------------------------------------------------
    # Add Final Response To Trace
    # --------------------------------------------------------

    tool_trace.append(
        {
            "step": len(tool_trace) + 1,
            "tool": "final_response",
            "status": "success",
            "result": final_answer
        }
    )

    # --------------------------------------------------------
    # Save Conversation Memory
    # --------------------------------------------------------

    conversation_memory.setdefault(
        session_id,
        []
    )

    conversation_memory[session_id].append(
        {
            "user": question,
            "assistant": final_answer
        }
    )

    conversation_memory[session_id] = (
        conversation_memory[session_id][
            -MAX_HISTORY:
        ]
    )

    # --------------------------------------------------------
    # Final Return
    # --------------------------------------------------------

    return {
        "answer": final_answer,
        "tool_trace": tool_trace,
        "sources": web_sources
        }
                




