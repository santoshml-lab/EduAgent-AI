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
    """
    Deterministic sanity check for software version questions.

    Prevents an older maintenance/security release from
    being incorrectly treated as the latest feature release.
    """

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

    # Extract semantic versions such as:
    # 3.14.7, 3.11.16, 4.2.1
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

    # If multiple materially different versions are present,
    # flag the result so the LLM validator cannot blindly accept
    # an older maintenance release.
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
                "The result requires careful validation to ensure "
                "an older maintenance/security release is not "
                "mistaken for the latest feature release."
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

Check whether the tool result is useful and relevant
for answering the user's question.

Return ONLY valid JSON:

{
  "valid": true,
  "reason": "",
  "needs_retry": false,
  "retry_strategy": "none"
}

Rules:

You are the validation layer of EduAgent AI.

Your job is NOT just to check whether the tool result
contains an answer.

You must verify whether the result actually supports
the correct answer to the user's question.

Return ONLY valid JSON:

{
  "valid": true,
  "reason": "",
  "needs_retry": false,
  "retry_strategy": "none"
}

Validation rules:

1. Relevance
- Check whether the result directly answers the user's question.

2. Accuracy
- Check whether the factual claim in the result is actually
  supported by the retrieved sources.

3. Conflicting sources
- If multiple sources disagree, do NOT automatically accept
  the result.
- Compare the sources and identify the most authoritative
  and relevant source.
- Prefer official primary sources over third-party sources,
  forums, aggregators, or Wikipedia when available.

4. Latest/current questions
- For questions containing words such as:
  latest, current, recent, today, newest, official,
  first verify that the retrieved information represents
  the current state.
- Do not treat an older version, legacy release, or historical
  release as the latest version.
- Distinguish stable releases from beta, alpha, release
  candidates, previews, development versions, and nightly builds.

5. Version questions
- When the user asks for the latest version of software,
  compare the version numbers and release status.
- A maintenance/security release of an older major/minor series
  must NOT be treated as the latest feature release if a newer
  stable feature release exists.

6. Retry decision
- valid = false when the retrieved sources contain conflicting
  information that prevents a reliable answer.
- valid = false when the search result is outdated for a
  current/latest question.
- needs_retry = true when a better web search could reasonably
  resolve the uncertainty.
- retry_strategy = "new_search" for such cases.

7. Calculator
- Use "recalculate" only when the calculator result itself
  appears incorrect or incomplete.

8. Web search
- Use "new_search" when the web result is outdated,
  contradictory, ambiguous, or insufficient.

9. No retry
- Use "none" only when the result is sufficiently reliable
  to answer the question.

Never invent facts.
Never assume that the first search result is correct.
- needs_retry = true only when another tool attempt
  could reasonably fix the problem.
- Do not invent facts.
- retry_strategy must be one of:
  "recalculate",
  "new_search",
  "none"

- Use "recalculate" when the calculator result
  should be recalculated.

- Use "new_search" when a web search should be
  attempted again with a better query.

- Use "none" when retry is unnecessary.
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
            "needs_retry": False
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

- Use only the allowed tools.
- Do not invent tool names.
- Simple questions should have
  needs_planning = false and an empty steps list.
- Use multiple steps when one tool's result is needed
  by a later step.
- Keep the plan concise.
- Steps must be ordered logically.
- Do not answer the user's question.
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

        plan = json.loads(content)

        return plan

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

    # --------------------------------------------------------
    # Number of Questions
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Difficulty
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Normalize Subject
    # --------------------------------------------------------

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

        topic_match = re.search(
            r"(?:about|on)\s+"
            r"(.+?)(?:\s+at|\s+with|\s*$)",
            question,
            re.IGNORECASE
        )

        if topic_match:

            extracted_topic = (
                topic_match.group(1).strip()
            )

            subject_pattern = re.escape(
                subject
            )

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

    topic = topic.strip(
        " .,?!"
    )

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

1. Extract every topic and score explicitly provided by the user.

2. If the user gives topic-wise scores, create one record
   for each topic.

Example:

Biology:
Cell Biology 55%
Genetics 68%
Ecology 82%

Return:

{
  "records": [
    {
      "subject": "Biology",
      "topic": "Cell Biology",
      "score": 55,
      "score_type": "percentage",
      "note": ""
    },
    {
      "subject": "Biology",
      "topic": "Genetics",
      "score": 68,
      "score_type": "percentage",
      "note": ""
    },
    {
      "subject": "Biology",
      "topic": "Ecology",
      "score": 82,
      "score_type": "percentage",
      "note": ""
    }
  ]
}

3. If only an overall subject score is provided,
   keep the topic as an empty string.

Example:

"My Biology score is 62%"

Return:

{
  "records": [
    {
      "subject": "Biology",
      "topic": "",
      "score": 62,
      "score_type": "percentage",
      "note": ""
    }
  ]
}

4. Never invent topics.

5. Never invent scores.

6. If no score is provided, return:

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
    # Multi-Step Agent Planner
    # ========================================================

    plan = create_agent_plan(
        standalone_question
    )

    planner_trace = None

    if plan.get("needs_planning", False):

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

IMPORTANT INTENT RULES:

For numerical:
- Use numerical when the user asks to calculate, solve, find the value of,
  multiply, divide, add, subtract, or evaluate a mathematical expression.
- Mathematical expressions using symbols such as:
  +, -, *, /, ×, ÷, =, %, ^
  should normally be classified as numerical when the user wants a result.
- Examples:
  "What is 125 * 48?"
  "Calculate 25 + 75"
  "Solve 12 × 8"
  "What is 500 / 25?"
  "Find 20% of 500"
- These requests MUST be classified as numerical.

For explanation:
- Use explanation when the user wants a concept, definition,
  theory, or educational explanation.

For quiz:
- Use quiz when the user asks to create, generate, give, or take a quiz.

For study_plan:
- Use study_plan when the user asks for a study schedule,
  timetable, or study plan.

For current_information:
- Use current_information when the user asks for latest,
  recent, current, today's information, news, or information
  that requires web search.

For weak_topic:
- Use weak_topic when the user asks about weak topics,
  what they should revise, what they should study,
  or where they should focus based on learning progress.

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

Always choose the most relevant intent.

When a mathematical calculation is explicitly requested,
prefer numerical over explanation.

Use the available tools only when appropriate.
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

    # --------------------------------------------------------
    # Tool Execution
    # --------------------------------------------------------

    tool_trace = []
    tool_results = []
    web_sources = []

    if planner_trace:

        tool_trace.append(
            planner_trace
    )

    # --------------------------------------------------------
    # Execute Planned Steps
    # --------------------------------------------------------

    if plan.get("needs_planning", False):

        for planned_step in plan.get("steps", []):

            planned_tool = planned_step.get("tool")

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

            elif planned_tool == "study_plan_generator":

                plan_result = study_plan_generator(
                subject="Biology",
                days=7,
                hours_per_day=1,
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
                                "subject": "Biology",
                                "days": 7,
                                "hours_per_day": 1
                            },
                            ensure_ascii=False
                        ),
                        "result": plan_result
                    }
                )

   
    

    

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

        # ----------------------------------------------------
        # Smart Retry Calculator Once
        # ----------------------------------------------------

        retry_strategy = validation_result.get(
            "retry_strategy",
            "none"
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
            and retry_strategy == "recalculate"
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
                            "retry": True,
                            "strategy": "recalculate"
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
                                "calculator_retry",
                            "strategy":
                                "recalculate"
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

        # ----------------------------------------------------
        # Validate Web Search Result
        # ----------------------------------------------------

        web_validation = validate_tool_result(
        question=standalone_question,
        tool_name="web_search",
        tool_result=web_result
        )

        # ----------------------------------------------------
        # Deterministic Version Conflict Check
        # ----------------------------------------------------

        version_check = detect_version_conflict(
            question=standalone_question,
            tool_result=web_result
        )

        if version_check.get("conflict", False):

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
                    if web_validation.get("valid", False)
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
  

        # ----------------------------------------------------
        # Smart Web Search Retry
        # ----------------------------------------------------

        if (
            not web_validation.get("valid", False)
            and web_validation.get("needs_retry", False)
            and web_validation.get("retry_strategy") == "new_search"
        ):

            retry_trace = {
                "step": len(tool_trace) + 1,
                "tool": "web_search_retry",
                "status": "running",
                "arguments": json.dumps(
                    {
                        "question": standalone_question,
                        "strategy": "new_search"
                    },
                    ensure_ascii=False
                )
            }

            tool_trace.append(retry_trace)

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

            # Validate retry result
            web_validation = validate_tool_result(
                question=standalone_question,
                tool_name="web_search",
                tool_result=web_result
            )

            tool_trace.append(
    {
        "step": len(tool_trace) + 1,
        "tool": "result_validator",
        "status": (
            "success"
            if web_validation.get("valid", False)
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

            messages=[
                {
                    "role": "system",
                    "content": """
You are the final answer generator of EduAgent AI.

IMPORTANT:
- Do NOT call any tools.
- Do NOT generate tool calls.
- Do NOT request web search.
- Do NOT request calculator.
- Use ONLY the information already provided.
- Return ONLY the final natural-language answer for the user.
- Never output JSON.
- Never mention internal tools, routing, validation, retries,
  agent architecture, or implementation details.
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

Available tool results:

{json.dumps(tool_results, ensure_ascii=False)}

Using ONLY the information above, write the final answer.
"""
                }
            ],

            temperature=0.2,

            tool_choice="none"
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
                




