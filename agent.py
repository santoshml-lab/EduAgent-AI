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
# Helper: Execute Calculator Retry
# ============================================================

def execute_calculator_retry(question: str):

    expression = extract_calculation(
        question
    )

    if not expression:

        return {
            "success": False,
            "result": "Could not extract calculation."
        }

    expression = (
        expression
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
    )

    try:

        result = calculator(
            expression
        )

        return {
            "success": True,
            "expression": expression,
            "result": result
        }

    except Exception as e:

        return {
            "success": False,
            "expression": expression,
            "result": f"Calculator retry error: {str(e)}"
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
# Helper: Execute Web Search Retry
# ============================================================

def execute_web_search_retry(question: str):

    query = (
        f"{question} latest stable release "
        f"official source 2026"
    )

    try:

        result = web_search(
            query
        )

        return {
            "success": True,
            "query": query,
            "result": result
        }

    except Exception as e:

        return {
            "success": False,
            "query": query,
            "result": f"Web search retry error: {str(e)}"
        }


# ============================================================
# Helper: Detect Version Conflicts
# ============================================================

def detect_version_conflict(
    question: str,
    tool_result
):

    if not re.search(
        r"\b(latest|newest|current)\b.*\b(version|release)\b"
        r"|\b(version|release)\b.*\b(latest|newest|current)\b",
        question,
        re.IGNORECASE
    ):

        return {
            "checked": False,
            "conflict": False,
            "reason": "Not a software version question."
        }

    result_text = json.dumps(
        tool_result,
        ensure_ascii=False
    )

    versions = re.findall(
        r"\b(\d+)\.(\d+)\.(\d+)\b",
        result_text
    )

    if not versions:

        return {
            "checked": True,
            "conflict": False,
            "reason": "No semantic version numbers detected."
        }

    version_tuples = [
        tuple(map(int, version))
        for version in versions
    ]

    highest_version = max(
        version_tuples
    )

    highest_version_text = ".".join(
        map(str, highest_version)
    )

    unique_versions = sorted(
        set(version_tuples),
        reverse=True
    )

    if len(unique_versions) > 1:

        return {
            "checked": True,
            "conflict": True,
            "reason": (
                "Multiple software versions were found in the "
                "retrieved result. The highest semantic version "
                f"detected is {highest_version_text}. "
                "The result requires careful validation."
            ),
            "highest_version": highest_version_text,
            "versions_found": [
                ".".join(map(str, version))
                for version in unique_versions
            ]
        }

    return {
        "checked": True,
        "conflict": False,
        "reason": (
            f"Only one semantic version was detected: "
            f"{highest_version_text}."
        ),
        "highest_version": highest_version_text,
        "versions_found": [
            highest_version_text
        ]
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

Check whether the tool result is useful, relevant,
accurate, and sufficient for answering the user's question.

Return ONLY valid JSON:

{
  "valid": true,
  "reason": "",
  "needs_retry": false,
  "retry_strategy": "none"
}

Rules:

1. Check relevance.

2. Check whether the result actually supports the
   user's question.

3. Never invent facts.

4. For current/latest questions, verify that the
   information represents the current state.

5. For software version questions, distinguish stable
   releases from beta, alpha, preview, nightly, and
   development versions.

6. If sources conflict and another search could resolve
   the conflict, use:
   "needs_retry": true
   "retry_strategy": "new_search"

7. For calculator errors, use:
   "retry_strategy": "recalculate"

8. Use "none" when retry is unnecessary.

9. Do not assume the first search result is correct.

10. Never invent missing information.
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

        return json.loads(
            content
        )

    except Exception as e:

        return {
            "valid": True,
            "reason": f"Validation fallback: {str(e)}",
            "needs_retry": False,
            "retry_strategy": "none"
        }


# ============================================================
# Helper: Create Agent Plan
# ============================================================

def create_agent_plan(question: str):

    try:

        response = client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "system",
                    "content": """
You are the planning layer of EduAgent AI.

Break the user's request into the minimum number of
logical steps required to answer it.

Return ONLY valid JSON.

Schema:

{
  "needs_planning": true,
  "goal": "",
  "subject": "",
  "days": 7,
  "hours_per_day": 1,
  "steps": [
    {
      "step": 1,
      "tool": "",
      "action": "",
      "reason": ""
    }
  ]
}

Allowed tools:

- weak_topic_detector
- study_plan_generator
- quiz_generator
- quiz_result_analyzer
- web_search
- calculator

Rules:

- Use only allowed tools.
- Never invent tool names.
- Simple questions should have needs_planning = false.
- Use multiple steps when one tool's result is required
  by a later step.
- Steps must be ordered logically.
- Do not answer the user's question.
- Extract subject, days, and hours only when explicitly
  provided by the user.
- Never invent a subject.
- Never invent a number of days.
- Never invent study hours.

If days are not explicitly provided:
use 7.

If hours per day are not explicitly provided:
use 1.

For study-plan requests return:
"subject": "",
"days": 7,
"hours_per_day": 1

The subject must come from the user's request or
available learning-progress data.

Do not create academic subtopics that the user
did not provide.

For a request such as:

"My Physics scores are:
Force 52%, Work Energy 64%, Light 85%.
Find my weak topics and make a 5-day revision plan
with 2 hours per day."

the correct plan is:

Step 1:
weak_topic_detector

Step 2:
study_plan_generator

The study plan must receive the weak-topic result
from Step 1.
"""
                },
                {
                    "role": "user",
                    "content": question
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

        return json.loads(
            content
        )

    except Exception as e:

        return {
            "needs_planning": False,
            "goal": "",
            "steps": [],
            "error": str(e)
        }

# ============================================================
# Agent Result Validator
# ============================================================

def validate_agent_step(question, step, tool_result):
    """
    Decide whether the executed tool result is sufficient
    or whether another agent step is required.
    """

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": """
You are the validation layer of EduAgent AI.

Your job is to inspect the result of ONE executed agent step.

Return ONLY valid JSON.

Schema:

{
  "valid": true,
  "needs_next_step": false,
  "reason": "",
  "next_action": ""
}

Rules:

1. valid = true when the tool result correctly addresses
   the executed step.

2. needs_next_step = true ONLY when another tool is genuinely
   required to complete the user's original request.

3. Do not invent missing information.

4. Do not create academic topics that the user did not provide.

5. If the result is sufficient for the current step but the
   original request still requires a later planned step,
   needs_next_step can be true.

6. If no additional tool is required, needs_next_step=false.

7. next_action must describe the required next action briefly.

8. Do not answer the user's original question.

9. Do not mention internal implementation details.
"""
                },
                {
                    "role": "user",
                    "content": f"""
Original user question:
{question}

Executed step:
{json.dumps(step, ensure_ascii=False)}

Tool result:
{json.dumps(tool_result, ensure_ascii=False)}
"""
                }
            ],
            temperature=0
        )

        raw = response.choices[0].message.content.strip()

        # Remove accidental markdown fences
        raw = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            raw,
            flags=re.IGNORECASE
        ).strip()

        result = json.loads(raw)

        return {
            "valid": bool(result.get("valid", False)),
            "needs_next_step": bool(
                result.get("needs_next_step", False)
            ),
            "reason": str(result.get("reason", "")),
            "next_action": str(
                result.get("next_action", "")
            )
        }

    except Exception as e:
        return {
            "valid": False,
            "needs_next_step": False,
            "reason": f"Validation error: {str(e)}",
            "next_action": ""
        }


# ============================================================
# Helper: Execute Quiz
# ============================================================

def execute_quiz(question: str):

    number_match = re.search(
        r"(\d+)\s*[-]?\s*questions?",
        question,
        re.IGNORECASE
    )

    if number_match:

        number_of_questions = int(
            number_match.group(1)
        )

    else:

        number_of_questions = 5

    number_of_questions = max(
        1,
        min(number_of_questions, 20)
    )

    if re.search(
        r"\bhard\b|\bdifficult\b|\badvanced\b",
        question,
        re.IGNORECASE
    ):

        difficulty = "hard"

    elif re.search(
        r"\beasy\b|\bbasic\b",
        question,
        re.IGNORECASE
    ):

        difficulty = "easy"

    else:

        difficulty = "medium"

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

    topic = subject

    topic_match = re.search(
        r"(?:about|on)\s+(.+?)(?:\s+at|\s+with|\s*$)",
        question,
        re.IGNORECASE
    )

    if topic_match:

        topic = topic_match.group(1).strip()

    topic = topic.strip(
        " .,?!"
    )

    if not topic:

        topic = subject

    result = quiz_generator(
        subject=subject,
        topic=topic,
        number_of_questions=number_of_questions,
        difficulty=difficulty
    )

    return {
        "success": True,
        "subject": subject,
        "topic": topic,
        "number_of_questions": number_of_questions,
        "difficulty": difficulty,
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
            r"(?:for|of)\s+"
            r"(?:Class\s+\d+\s+)?"
            r"(.+?)(?:\s+with|\s+for|\s*$)",
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

def extract_learning_progress(question: str):

    try:

        response = client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "system",
                    "content": """
Extract ALL learning progress records from the user's message.

Return ONLY valid JSON.

Schema:

{
  "records": [
    {
      "subject": "",
      "topic": "",
      "score": null,
      "score_type": "percentage",
      "note": ""
    }
  ]
}

Rules:

1. Extract every topic and score explicitly provided.

2. Create one record for each topic.

3. Never invent topics.

4. Never invent scores.

5. If no score is provided, return:

{
  "records": []
}
"""
                },
                {
                    "role": "user",
                    "content": question
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

        return json.loads(
            content
        )

    except Exception:

        return {
            "records": []
        }


# ============================================================
# Helper: Save Learning Progress
# ============================================================

def save_learning_progress(
    session_id: str,
    progress_data: dict
):

    records = progress_data.get(
        "records",
        []
    )

    if not isinstance(records, list):

        return

    learning_progress.setdefault(
        session_id,
        []
    )

    for progress in records:

        if progress.get("score") is None:

            continue

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

def execute_quiz_result(question: str):

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

def generate_targeted_revision(progress: list):

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
                    "Revise the topic and practice "
                    "targeted questions."
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
# Helper: Save Conversation
# ============================================================

def save_conversation(
    session_id: str,
    question: str,
    answer: str
):

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


# ============================================================
# Main Agent
# ============================================================
def ask_agent(
    question: str,
    session_id: str = "default"
):

    standalone_question = question.strip()

    # ============================================================
    # Direct Quiz Detection
    # ============================================================

    quiz_match = re.search(
        r"\b(quiz me|give me a quiz|test me|mcq|multiple choice)\b",
        standalone_question,
        re.IGNORECASE
    )

    if quiz_match:

        quiz_result = execute_quiz(
            standalone_question
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
                        "question":
                            standalone_question
                    },
                    ensure_ascii=False
                ),
                "result":
                    quiz_result.get(
                        "result",
                        ""
                    )
            }
        ]

        save_conversation(
            session_id,
            standalone_question,
            quiz_result.get(
                "result",
                ""
            )
        )

        return {
            "answer":
                quiz_result.get(
                    "result",
                    ""
                ),
            "tool_trace": tool_trace,
            "sources": []
        }

    # ============================================================
    # Learning Progress
    # ============================================================

    progress_data = extract_learning_progress(
        standalone_question
    )

    if progress_data:

        save_learning_progress(
            session_id,
            progress_data
        )

    # ============================================================
    # Direct Weak Topic Detection
    # ============================================================

    weak_topic_match = re.search(
        r"\b(weak topics?|weak areas?|where am i weak)\b",
        standalone_question,
        re.IGNORECASE
    )

    if (
        weak_topic_match
        and not re.search(
            r"\b(study plan|revision plan|study schedule|"
            r"timetable|schedule|plan for \d+\s*[-]?\s*day)\b",
            standalone_question,
            re.IGNORECASE
        )
    ):

        weak_result = execute_weak_topic(
            standalone_question
        )

        tool_trace = [
            {
                "step": 1,
                "tool": "weak_topic_detector",
                "status": (
                    "success"
                    if weak_result["success"]
                    else "error"
                ),
                "arguments": json.dumps(
                    {
                        "question":
                            standalone_question
                    },
                    ensure_ascii=False
                ),
                "result":
                    weak_result.get(
                        "result",
                        ""
                    )
            }
        ]

        return {
            "answer":
                weak_result.get(
                    "result",
                    ""
                ),
            "tool_trace": tool_trace,
            "sources": []
        }

    # ============================================================
    # Quiz Result Analysis
    # ============================================================

    quiz_result_match = re.search(
        r"\b(score|marks|result|quiz result|quiz score)\b",
        standalone_question,
        re.IGNORECASE
    )

    if quiz_result_match:

        quiz_analysis = execute_quiz_result(
            standalone_question
        )

        if quiz_analysis["success"]:

            tool_trace = [
                {
                    "step": 1,
                    "tool": "quiz_result_analyzer",
                    "status": "success",
                    "arguments": json.dumps(
                        {
                            "question":
                                standalone_question
                        },
                        ensure_ascii=False
                    ),
                    "result":
                        quiz_analysis.get(
                            "result",
                            ""
                        )
                }
            ]

            return {
                "answer":
                    quiz_analysis.get(
                        "result",
                        ""
                    ),
                "tool_trace": tool_trace,
                "sources": []
            }

    # ============================================================
    # Create Agent Plan
    # ============================================================

    plan = create_agent_plan(
        standalone_question
    )

    planned_tools = [
        step.get("tool")
        for step in plan.get(
            "steps",
            []
        )
    ]

    intents = plan.get(
        "intents",
        []
    )

    tool_trace = [
        {
            "step": 1,
            "tool": "agent_planner",
            "status": "success",
            "arguments": json.dumps(
                {
                    "question":
                        standalone_question
                },
                ensure_ascii=False
            ),
            "result": plan
        }
    ]

    tool_results = []

    web_sources = []

    # ============================================================
    # Planned Multi-Step Execution
    # ============================================================

    if plan.get(
        "needs_planning",
        False
    ):

        for index, step in enumerate(
            plan.get(
                "steps",
                []
            )
        ):

            tool_name = step.get(
                "tool"
            )

            # ----------------------------------------------------
            # Weak Topic Detector
            # ----------------------------------------------------

            if tool_name == "weak_topic_detector":

                result = execute_weak_topic(
                    standalone_question
                )

            # ----------------------------------------------------
            # Study Plan
            # ----------------------------------------------------

            elif tool_name == "study_plan_generator":

                previous_result = ""

                if tool_results:

                    previous_result = json.dumps(
                        tool_results[-1],
                        ensure_ascii=False
                    )

                result = execute_study_plan(
                    standalone_question,
                    previous_result
                )

            # ----------------------------------------------------
            # Web Search
            # ----------------------------------------------------

            elif tool_name == "web_search":

                result = execute_web_search(
                    standalone_question
                )

            else:

                result = {
                    "success": False,
                    "result":
                        f"Unsupported planned tool: {tool_name}"
                }

            # ----------------------------------------------------
            # Trace
            # ----------------------------------------------------

            tool_trace.append(
                {
                    "step":
                        len(tool_trace) + 1,
                    "tool":
                        tool_name,
                    "status": (
                        "success"
                        if result.get(
                            "success",
                            False
                        )
                        else "error"
                    ),
                    "arguments":
                        json.dumps(
                            {
                                "question":
                                    standalone_question
                            },
                            ensure_ascii=False
                        ),
                    "result":
                        result.get(
                            "result",
                            ""
                        )
                }
            )

            # ----------------------------------------------------
            # Validate Step
            # ----------------------------------------------------

            validation = validate_agent_step(
                question=
                    standalone_question,
                step=step,
                tool_result=result
            )

            # ----------------------------------------------------
            # Force planner continuation based on
            # remaining planned steps
            # ----------------------------------------------------

            if index < len(
                plan.get(
                    "steps",
                    []
                )
            ) - 1:

                validation[
                    "needs_next_step"
                ] = True

                validation[
                    "next_action"
                ] = (
                    "Execute next planned step."
                )

            else:

                validation[
                    "needs_next_step"
                ] = False

            tool_trace.append(
                {
                    "step":
                        len(tool_trace) + 1,
                    "tool":
                        "agent_validator",
                    "status": (
                        "success"
                        if validation.get(
                            "valid",
                            False
                        )
                        else "rejected"
                    ),
                    "arguments":
                        json.dumps(
                            {
                                "validated_tool":
                                    tool_name
                            },
                            ensure_ascii=False
                        ),
                    "result":
                        validation
                }
            )

            tool_results.append(
                {
                    "tool":
                        tool_name,
                    "data":
                        result,
                    "validation":
                        validation
                }
            )

            # ----------------------------------------------------
            # Collect Web Sources
            # ----------------------------------------------------

            if tool_name == "web_search":

                try:

                    parsed_sources = json.loads(
                        result.get(
                            "result",
                            "[]"
                        )
                    )

                    if isinstance(
                        parsed_sources,
                        list
                    ):

                        web_sources.extend(
                            parsed_sources
                        )

                except Exception:

                    pass

    # ============================================================
    # Router
    # ============================================================

    router_response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": """
You are the routing layer of EduAgent AI.

Return ONLY valid JSON.

Schema:

{
  "tool": "calculator",
  "intents": ["numerical"]
}

Allowed tools:

calculator
web_search
quiz_generator
study_plan_generator
weak_topic_detector
quiz_result_analyzer

Allowed intents:

numerical
current_information
quiz
study_plan
weak_topic
quiz_result
general

Choose the minimum required tool.
"""
            },
            {
                "role": "user",
                "content":
                    standalone_question
            }
        ],
        temperature=0
    )

    raw_router = (
        router_response
        .choices[0]
        .message
        .content
        .strip()
    )

    raw_router = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        raw_router,
        flags=re.IGNORECASE
    ).strip()

    try:

        router = json.loads(
            raw_router
        )

    except Exception:

        router = {
            "tool": None,
            "intents": ["general"]
        }

    router_tool = router.get(
        "tool"
    )

    intents = router.get(
        "intents",
        []
    )

    # ============================================================
    # Numerical
    # ============================================================

    if (
        "numerical" in intents
        and "calculator" not in planned_tools
    ):

        # ========================================================
        # First Calculator Attempt
        # ========================================================

        calculator_result = execute_calculator(
            standalone_question
        )

        tool_trace.append(
            {
                "step":
                    len(tool_trace) + 1,
                "tool":
                    "calculator",
                "status": (
                    "success"
                    if calculator_result[
                        "success"
                    ]
                    else "error"
                ),
                "arguments":
                    json.dumps(
                        {
                            "question":
                                standalone_question
                        },
                        ensure_ascii=False
                    ),
                "result":
                    calculator_result.get(
                        "result",
                        ""
                    )
            }
        )

        # ========================================================
        # Validate Calculator Result
        # ========================================================

        validation_result = (
            validate_tool_result(
                question=
                    standalone_question,
                tool_name=
                    "calculator",
                tool_result=
                    calculator_result
            )
        )

        tool_trace.append(
            {
                "step":
                    len(tool_trace) + 1,
                "tool":
                    "result_validator",
                "status": (
                    "success"
                    if validation_result.get(
                        "valid",
                        False
                    )
                    else "rejected"
                ),
                "arguments":
                    json.dumps(
                        {
                            "validated_tool":
                                "calculator"
                        },
                        ensure_ascii=False
                    ),
                "result":
                    validation_result
            }
        )

        # ========================================================
        # Calculator Recovery / Retry
        # ========================================================

        if (
            not validation_result.get(
                "valid",
                False
            )
            and validation_result.get(
                "needs_retry",
                False
            )
            and validation_result.get(
                "retry_strategy"
            ) == "recalculate"
        ):

            calculator_retry_result = (
                execute_calculator_retry(
                    standalone_question
                )
            )

            tool_trace.append(
                {
                    "step":
                        len(tool_trace) + 1,
                    "tool":
                        "calculator_retry",
                    "status": (
                        "success"
                        if calculator_retry_result[
                            "success"
                        ]
                        else "error"
                    ),
                    "arguments":
                        json.dumps(
                            {
                                "question":
                                    standalone_question,
                                "retry": True,
                                "strategy":
                                    "recalculate"
                            },
                            ensure_ascii=False
                        ),
                    "result":
                        calculator_retry_result.get(
                            "result",
                            ""
                        )
                }
            )

            # ====================================================
            # Validate Retry
            # ====================================================

            retry_validation = (
                validate_tool_result(
                    question=
                        standalone_question,
                    tool_name=
                        "calculator_retry",
                    tool_result=
                        calculator_retry_result
                )
            )

            tool_trace.append(
                {
                    "step":
                        len(tool_trace) + 1,
                    "tool":
                        "result_validator_retry",
                    "status": (
                        "success"
                        if retry_validation.get(
                            "valid",
                            False
                        )
                        else "rejected"
                    ),
                    "arguments":
                        json.dumps(
                            {
                                "validated_tool":
                                    "calculator_retry"
                            },
                            ensure_ascii=False
                        ),
                    "result":
                        retry_validation
                }
            )

            if retry_validation.get(
                "valid",
                False
            ):

                calculator_result = (
                    calculator_retry_result
                )

                validation_result = (
                    retry_validation
                )

        tool_results.append(
            {
                "tool":
                    "calculator",
                "data":
                    calculator_result,
                "validation":
                    validation_result
            }
        )

    # ============================================================
    # Current Information / Web Search
    # ============================================================

    if (
        "current_information" in intents
        and "web_search" not in planned_tools
    ):

        web_result = execute_web_search(
            standalone_question
        )

        tool_trace.append(
            {
                "step":
                    len(tool_trace) + 1,
                "tool":
                    "web_search",
                "status": (
                    "success"
                    if web_result["success"]
                    else "error"
                ),
                "arguments":
                    json.dumps(
                        {
                            "question":
                                standalone_question
                        },
                        ensure_ascii=False
                    ),
                "result":
                    web_result.get(
                        "result",
                        ""
                    )
            }
        )

        web_validation = validate_tool_result(
            question=
                standalone_question,
            tool_name=
                "web_search",
            tool_result=
                web_result
        )

        tool_trace.append(
            {
                "step":
                    len(tool_trace) + 1,
                "tool":
                    "result_validator",
                "status": (
                    "success"
                    if web_validation.get(
                        "valid",
                        False
                    )
                    else "rejected"
                ),
                "arguments":
                    json.dumps(
                        {
                            "validated_tool":
                                "web_search"
                        },
                        ensure_ascii=False
                    ),
                "result":
                    web_validation
            }
        )

        # --------------------------------------------------------
        # Web Retry
        # --------------------------------------------------------

        if (
            not web_validation.get(
                "valid",
                False
            )
            and web_validation.get(
                "needs_retry",
                False
            )
            and web_validation.get(
                "retry_strategy"
            ) == "new_search"
        ):

            web_retry_result = (
                execute_web_search_retry(
                    standalone_question
                )
            )

            tool_trace.append(
                {
                    "step":
                        len(tool_trace) + 1,
                    "tool":
                        "web_search_retry",
                    "status": (
                        "success"
                        if web_retry_result[
                            "success"
                        ]
                        else "error"
                    ),
                    "arguments":
                        json.dumps(
                            {
                                "question":
                                    standalone_question,
                                "retry": True,
                                "strategy":
                                    "new_search"
                            },
                            ensure_ascii=False
                        ),
                    "result":
                        web_retry_result.get(
                            "result",
                            ""
                        )
                }
            )

            retry_web_validation = (
                validate_tool_result(
                    question=
                        standalone_question,
                    tool_name=
                        "web_search_retry",
                    tool_result=
                        web_retry_result
                )
            )

            tool_trace.append(
                {
                    "step":
                        len(tool_trace) + 1,
                    "tool":
                        "result_validator_retry",
                    "status": (
                        "success"
                        if retry_web_validation.get(
                            "valid",
                            False
                        )
                        else "rejected"
                    ),
                    "arguments":
                        json.dumps(
                            {
                                "validated_tool":
                                    "web_search_retry"
                            },
                            ensure_ascii=False
                        ),
                    "result":
                        retry_web_validation
                }
            )

            if retry_web_validation.get(
                "valid",
                False
            ):

                web_result = (
                    web_retry_result
                )

                web_validation = (
                    retry_web_validation
                )

        tool_results.append(
            {
                "tool":
                    "web_search",
                "data":
                    web_result,
                "validation":
                    web_validation
            }
        )

        try:

            parsed_web = json.loads(
                web_result.get(
                    "result",
                    "[]"
                )
            )

            if isinstance(
                parsed_web,
                list
            ):

                web_sources.extend(
                    parsed_web
                )

        except Exception:

            pass

    # ============================================================
    # Quiz Tool
    # ============================================================

    if (
        router_tool == "quiz_generator"
        or "quiz" in intents
    ):

        quiz_result = execute_quiz(
            standalone_question
        )

        tool_trace.append(
            {
                "step":
                    len(tool_trace) + 1,
                "tool":
                    "quiz_generator",
                "status": (
                    "success"
                    if quiz_result["success"]
                    else "error"
                ),
                "arguments":
                    json.dumps(
                        {
                            "question":
                                standalone_question
                        },
                        ensure_ascii=False
                    ),
                "result":
                    quiz_result.get(
                        "result",
                        ""
                    )
            }
        )

        tool_results.append(
            {
                "tool":
                    "quiz_generator",
                "data":
                    quiz_result
            }
        )

    # ============================================================
    # Study Plan
    # ============================================================

    if (
        "study_plan" in intents
        and "study_plan_generator"
        not in planned_tools
    ):

        study_result = execute_study_plan(
            standalone_question,
            ""
        )

        tool_trace.append(
            {
                "step":
                    len(tool_trace) + 1,
                "tool":
                    "study_plan_generator",
                "status": (
                    "success"
                    if study_result["success"]
                    else "error"
                ),
                "arguments":
                    json.dumps(
                        {
                            "question":
                                standalone_question
                        },
                        ensure_ascii=False
                    ),
                "result":
                    study_result.get(
                        "result",
                        ""
                    )
            }
        )

        tool_results.append(
            {
                "tool":
                    "study_plan_generator",
                "data":
                    study_result
            }
        )

    # ============================================================
    # Weak Topic Tool
    # ============================================================

    if (
        "weak_topic" in intents
        and "weak_topic_detector"
        not in planned_tools
    ):

        weak_result = execute_weak_topic(
            standalone_question
        )

        tool_trace.append(
            {
                "step":
                    len(tool_trace) + 1,
                "tool":
                    "weak_topic_detector",
                "status": (
                    "success"
                    if weak_result["success"]
                    else "error"
                ),
                "arguments":
                    json.dumps(
                        {
                            "question":
                                standalone_question
                        },
                        ensure_ascii=False
                    ),
                "result":
                    weak_result.get(
                        "result",
                        ""
                    )
            }
        )

        tool_results.append(
            {
                "tool":
                    "weak_topic_detector",
                "data":
                    weak_result
            }
        )

    # ============================================================
    # Quiz Result Analyzer
    # ============================================================

    if (
        "quiz_result" in intents
        and "quiz_result_analyzer"
        not in planned_tools
    ):

        quiz_analysis = execute_quiz_result(
            standalone_question
        )

        tool_trace.append(
            {
                "step":
                    len(tool_trace) + 1,
                "tool":
                    "quiz_result_analyzer",
                "status": (
                    "success"
                    if quiz_analysis["success"]
                    else "error"
                ),
                "arguments":
                    json.dumps(
                        {
                            "question":
                                standalone_question
                        },
                        ensure_ascii=False
                    ),
                "result":
                    quiz_analysis.get(
                        "result",
                        ""
                    )
            }
        )

        tool_results.append(
            {
                "tool":
                    "quiz_result_analyzer",
                "data":
                    quiz_analysis
            }
        )

    # ============================================================
    # Targeted Revision
    # ============================================================

    if "targeted_revision" in intents:

        revision_result = generate_targeted_revision(
            standalone_question
        )

        tool_trace.append(
            {
                "step":
                    len(tool_trace) + 1,
                "tool":
                    "targeted_revision",
                "status":
                    "success",
                "arguments":
                    json.dumps(
                        {
                            "question":
                                standalone_question
                        },
                        ensure_ascii=False
                    ),
                "result":
                    revision_result
            }
        )

        tool_results.append(
            {
                "tool":
                    "targeted_revision",
                "data":
                    revision_result
            }
        )

    # ============================================================
    # Final Answer
    # ============================================================

    final_answer = generate_final_answer(
        question=
            standalone_question,
        tool_results=
            tool_results,
        conversation=
            conversation_memory.get(
                session_id,
                []
            )
    )

    tool_trace.append(
        {
            "step":
                len(tool_trace) + 1,
            "tool":
                "final_response",
            "status":
                "success",
            "arguments":
                json.dumps(
                    {
                        "question":
                            standalone_question
                    },
                    ensure_ascii=False
                ),
            "result":
                final_answer
        }
    )

    save_conversation(
        session_id,
        standalone_question,
        final_answer
    )

    return {
        "answer":
            final_answer,
        "tool_trace":
            tool_trace,
        "sources":
            web_sources
    }







        
                
          

                    
                

    
                    


    
                
                    
                    
            
                
                        
                        
                
    
                
                
            
                    
                        


                
        
                    
        

            
                    

        
                

        
                    
                
        




      
                    
                
        
        
                

            
                    
                    
                        
     
        
                
            
            

        
        

        
                
        
        


# ============================================================
# Helper: Generate Final Answer
# ============================================================

def generate_final_answer(
    question: str,
    progress: list,
    tool_results: list,
    conversation_history: list
):

    try:

        response = client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "system",
                    "content": """
You are the final answer generator of EduAgent AI.

Use ONLY the information supplied in the current
user question, learning progress, and tool results.

Never invent facts.

Never invent:
- topics
- subtopics
- scores
- dates
- chapters
- formulas
- study material
- academic details
- claims not supported by the supplied data

For learning progress:

- Copy topic names exactly from the supplied data.
- Copy scores exactly from the supplied data.
- Do not create new subtopics.

For study plans:

- Use only topics explicitly present in the tool results.
- You may create generic activities such as:
  revision
  practice questions
  self-testing
  error review
  mixed practice
  assessment
- Do NOT invent detailed academic subtopics.

Example:

If the tool says:

Force = 52%
Work Energy = 64%

you may say:

Force — 52%
Work Energy — 64%

You must NOT invent:
- Newton's laws
- friction
- tension
- normal force
- kinetic energy
- potential energy
- conservation of energy
- springs
- vectors

unless those details were explicitly provided.

For a multi-day plan:

- Preserve the exact number of days supplied by the planner.
- Preserve the exact hours per day supplied by the planner.
- Use only the topics supplied by the planner/tool.
- Do not silently change the schedule.

If the tool result is insufficient for a specific claim,
say that the detailed information was not provided.

The final answer should be clear, concise, practical,
and user-friendly.

Never mention internal tools, routing, validation,
planner implementation, retries, or agent architecture.
"""
                },

                *conversation_history,

                {
                    "role": "user",
                    "content": f"""
User question:

{question}

Learning progress:

{json.dumps(
    progress,
    ensure_ascii=False
)}

Tool results:

{json.dumps(
    tool_results,
    ensure_ascii=False
)}

Write the final answer using ONLY the information above.
"""
                }
            ],

            temperature=0.2,

            tool_choice="none"
        )

        return (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

    except Exception as e:

      return (
            f"Final response error: {str(e)}"
        )
    

    
                




