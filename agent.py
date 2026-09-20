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

    standalone_question = question

    conversation_history = get_conversation_history(
        session_id
    )

    # ========================================================
    # Direct Quiz Detection
    # ========================================================

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

        save_conversation(
            session_id,
            question,
            answer
        )

        return {
            "answer": answer,
            "tool_trace": tool_trace,
            "sources": []
        }

    # ========================================================
    # Learning Progress Extraction
    # ========================================================

    progress_update = extract_learning_progress(
        question
    )

    save_learning_progress(
        session_id,
        progress_update
    )

    progress = learning_progress.get(
        session_id,
        []
    )

    # ========================================================
    # Direct Weak Topic Detection
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

    if (
        weak_topic_match
        and not re.search(
            r"\b(study plan|revision plan|study schedule|"
            r"timetable|schedule|plan for \d+\s*[-]?\s*day)\b",
            standalone_question,
            re.IGNORECASE
        )
    ):

        weak_topics_result = weak_topic_detector(
            progress
        )

        revision_result = generate_targeted_revision(
            progress
        )

        tool_trace = [
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
            },
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
        ]

        final_answer = generate_final_answer(
            question=standalone_question,
            progress=progress,
            tool_results=[
                {
                    "tool": "weak_topic_detector",
                    "result": weak_topics_result
                },
                {
                    "tool": "targeted_revision",
                    "result": revision_result.get(
                        "result",
                        ""
                    )
                }
            ],
            conversation_history=conversation_history
        )

        tool_trace.append(
            {
                "step": len(tool_trace) + 1,
                "tool": "final_response",
                "status": "success",
                "result": final_answer
            }
        )

        save_conversation(
            session_id,
            question,
            final_answer
        )

        return {
            "answer": final_answer,
            "tool_trace": tool_trace,
            "sources": []
        }

    # ========================================================
    # Direct Quiz Result Detection
    # ========================================================

    quiz_result_match = re.search(
        r"\b(?:scored|got|achieved|received)\s+"
        r"(\d+(?:\.\d+)?)\s*%",
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

        save_conversation(
            session_id,
            question,
            answer
        )

        return {
            "answer": answer,
            "tool_trace": tool_trace,
            "sources": []
        }

    # ========================================================
    # Multi-Step Agent Planner
    # ========================================================

    plan = create_agent_plan(
        standalone_question
    )

    planner_trace = None

    if plan.get(
        "needs_planning",
        False
    ):

        planner_trace = {
            "step": 1,
            "tool": "agent_planner",
            "status": "success",
            "arguments": json.dumps(
                {
                    "question": standalone_question
                },
                ensure_ascii=False
            ),
            "result": plan
        }

    # ========================================================
    # Router
    # ========================================================

    router_tools = [
        tool
        for tool in TOOLS
        if tool["function"]["name"]
        != "quiz_result_analyzer"
    ]

    router_messages = [

        {
            "role": "system",
            "content": """
You are the intent router for an educational AI agent.

Classify the user's request into one or more intents:

1. explanation
2. numerical
3. quiz
4. study_plan
5. current_information
6. weak_topic
7. quiz_result

Rules:

- Numerical questions require calculator.
- Explanation questions require explanation.
- Quiz requests require quiz_generator.
- Study-plan requests require study_plan_generator.
- Current/latest/recent questions require web_search.
- Weak-topic requests require weak_topic_detector.
- Quiz-result requests are handled by the application.

Priority:

quiz_result
weak_topic
study_plan
quiz
numerical
current_information
explanation

Use tools only when appropriate.
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

    tool_calls = (
        router_response
        .choices[0]
        .message
        .tool_calls
    )

    intents = []

    if tool_calls:

        for tool_call in tool_calls:

            tool_name = (
                tool_call.function.name
            )

            if tool_name == "calculator":

                intents.append(
                    "numerical"
                )

            elif tool_name == "web_search":

                intents.append(
                    "current_information"
                )

            elif tool_name == "quiz_generator":

                intents.append(
                    "quiz"
                )

            elif tool_name == "study_plan_generator":

                intents.append(
                    "study_plan"
                )

            elif tool_name == "weak_topic_detector":

                intents.append(
                    "weak_topic"
                )

            elif tool_name == "quiz_result_analyzer":

                intents.append(
                    "quiz_result"
                )

    # Remove duplicates while preserving order

    intents = list(
        dict.fromkeys(intents)
    )

    # ========================================================
    # Tool Execution
    # ========================================================

    tool_trace = []
    tool_results = []
    web_sources = []

    if planner_trace:

        tool_trace.append(
            planner_trace
        )

    # ========================================================
    # Execute Planned Steps
    # ========================================================

    planned_tools = set()

    if plan.get(
        "needs_planning",
        False
    ):

        weak_result = None

        for planned_step in plan.get(
            "steps",
            []
        ):

            planned_tool = planned_step.get(
                "tool"
            )

            planned_tools.add(
                planned_tool
            )

            # ------------------------------------------------
            # Weak Topic Detector
            # ------------------------------------------------

            if planned_tool == "weak_topic_detector":

                weak_result = weak_topic_detector(
                    progress
                )

                tool_results.append(
                    {
                        "tool": "weak_topic_detector",
                        "result": weak_result
                    }
                )

                tool_trace.append(
                    {
                        "step": len(tool_trace) + 1,
                        "tool": "weak_topic_detector",
                        "status": "success",
                        "arguments": json.dumps(
                            {
                                "progress_items": len(progress)
                            },
                            ensure_ascii=False
                        ),
                        "result": weak_result
                    }
                )

            # ------------------------------------------------
            # Study Plan Generator
            # ------------------------------------------------

            elif planned_tool == "study_plan_generator":

                subject = plan.get(
                    "subject",
                    "General Studies"
                )

                days = plan.get(
                    "days",
                    7
                )

                hours_per_day = plan.get(
                    "hours_per_day",
                    1
                )

                plan_result = study_plan_generator(
                    subject=subject,
                    days=days,
                    hours_per_day=hours_per_day,
                    topics=weak_result
                )

                tool_results.append(
                    {
                        "tool": "study_plan_generator",
                        "result": plan_result
                    }
                )

                tool_trace.append(
                    {
                        "step": len(tool_trace) + 1,
                        "tool": "study_plan_generator",
                        "status": "success",
                        "arguments": json.dumps(
                            {
                                "subject": subject,
                                "days": days,
                                "hours_per_day": hours_per_day
                            },
                            ensure_ascii=False
                        ),
                        "result": plan_result
                    }
                )

    # ========================================================
    # Numerical
    # ========================================================

    if (
        "numerical" in intents
        and "calculator" not in planned_tools
    ):

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
                    if validation_result.get(
                        "valid",
                        False
                    )
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

            retry_result = execute_calculator(
                standalone_question
            )

            retry_validation = validate_tool_result(
                question=standalone_question,
                tool_name="calculator_retry",
                tool_result=retry_result
            )

            tool_trace.append(
                {
                    "step": len(tool_trace) + 1,
                    "tool": "calculator_retry",
                    "status": (
                        "success"
                        if retry_result["success"]
                        else "error"
                    ),
                    "arguments": json.dumps(
                        {
                            "question": standalone_question,
                            "retry": True
                        },
                        ensure_ascii=False
                    ),
                    "result": retry_result.get(
                        "result",
                        ""
                    )
                }
            )

            tool_trace.append(
                {
                    "step": len(tool_trace) + 1,
                    "tool": "result_validator_retry",
                    "status": (
                        "success"
                        if retry_validation.get(
                            "valid",
                            False
                        )
                        else "rejected"
                    ),
                    "arguments": json.dumps(
                        {
                            "validated_tool":
                                "calculator_retry"
                        },
                        ensure_ascii=False
                    ),
                    "result": retry_validation
                }
            )

            calculator_result = retry_result

            validation_result = retry_validation

        tool_results.append(
            {
                "tool": "calculator",
                "data": calculator_result,
                "validation": validation_result
            }
        )

    # ========================================================
    # Current Information
    # ========================================================

    if (
        "current_information" in intents
        and "web_search" not in planned_tools
    ):

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

        web_validation = validate_tool_result(
            question=standalone_question,
            tool_name="web_search",
            tool_result=web_result
        )

        version_check = detect_version_conflict(
            question=standalone_question,
            tool_result=web_result
        )

        if version_check.get(
            "conflict",
            False
        ):

            web_validation = {
                "valid": False,
                "reason": version_check.get(
                    "reason",
                    "Version conflict detected."
                ),
                "needs_retry": True,
                "retry_strategy": "new_search"
            }

        tool_trace.append(
            {
                "step": len(tool_trace) + 1,
                "tool": "result_validator",
                "status": (
                    "success"
                    if web_validation.get(
                        "valid",
                        False
                    )
                    else "rejected"
                ),
                "arguments": json.dumps(
                    {
                        "validated_tool": "web_search"
                    },
                    ensure_ascii=False
                ),
                "result": web_validation
            }
        )

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

            retry_trace = {
                "step": len(tool_trace) + 1,
                "tool": "web_search_retry",
                "status": "running",
                "arguments": json.dumps(
                    {
                        "question": standalone_question
                    },
                    ensure_ascii=False
                )
            }

            tool_trace.append(
                retry_trace
            )

            web_result = execute_web_search_retry(
                standalone_question
            )

            retry_trace["status"] = (
                "success"
                if web_result["success"]
                else "error"
            )

            retry_trace["result"] = web_result.get(
                "result",
                ""
            )

            web_validation = validate_tool_result(
                question=standalone_question,
                tool_name="web_search",
                tool_result=web_result
            )

            tool_trace.append(
                {
                    "step": len(tool_trace) + 1,
                    "tool": "result_validator_retry",
                    "status": (
                        "success"
                        if web_validation.get(
                            "valid",
                            False
                        )
                        else "rejected"
                    ),
                    "arguments": json.dumps(
                        {
                            "validated_tool": "web_search"
                        },
                        ensure_ascii=False
                    ),
                    "result": web_validation
                }
            )

        tool_results.append(
            {
                "tool": "web_search",
                "data": web_result,
                "validation": web_validation
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

    # ========================================================
    # Quiz
    # ========================================================

    if (
        "quiz" in intents
        and "quiz_generator" not in planned_tools
    ):

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

    # ========================================================
    # Study Plan
    # ========================================================

    if (
        "study_plan" in intents
        and "study_plan_generator" not in planned_tools
    ):

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

    # ========================================================
    # Weak Topic Detector
    # ========================================================

    if (
        "weak_topic" in intents
        and "weak_topic_detector" not in planned_tools
    ):

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

    # ========================================================
    # Targeted Revision
    # ========================================================

    if (
        "weak_topic" in intents
        and "weak_topic_detector" not in planned_tools
    ):

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

    # ========================================================
    # Quiz Result Analyzer
    # ========================================================

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

    # ========================================================
    # Final Answer
    # ========================================================

    final_answer = generate_final_answer(
        question=standalone_question,
        progress=progress,
        tool_results=tool_results,
        conversation_history=conversation_history
    )

    # ========================================================
    # Final Response Trace
    # ========================================================

    tool_trace.append(
        {
            "step": len(tool_trace) + 1,
            "tool": "final_response",
            "status": "success",
            "result": final_answer
        }
    )

    # ========================================================
    # Save Conversation Memory
    # ========================================================

    save_conversation(
        session_id,
        question,
        final_answer
    )

    # ========================================================
    # Final Return
    # ========================================================

    return {
        "answer": final_answer,
        "tool_trace": tool_trace,
        "sources": web_sources
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
    

    
                




