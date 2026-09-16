import json
from tavily import TavilyClient


# ========================================
# Calculator Tool
# ========================================
def calculator(expression: str):
    try:
        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return f"Calculation result: {result}"

    except Exception as e:
        return f"Calculator error: {str(e)}"


# ========================================
# Web Search Tool
# ========================================
def web_search(query: str):
    try:
        tavily = TavilyClient()

        response = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=False
        )

        results = []

        for index, item in enumerate(
            response.get("results", []),
            start=1
        ):
            results.append(
                {
                    "source_id": index,
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", "")
                }
            )

        return json.dumps(
            results,
            ensure_ascii=False
        )

    except Exception as e:
        return json.dumps(
            {
                "error": f"Web search failed: {str(e)}"
            },
            ensure_ascii=False
        )


# ========================================
# Education Router
# ========================================
def education_router(intent: str):
    """
    Identify the type of education task
    and return the recommended workflow.
    """

    workflows = {
        "explanation": (
            "Workflow: explanation. "
            "Explain the educational concept clearly "
            "and at an appropriate level."
        ),

        "numerical": (
            "Workflow: numerical. "
            "Use the calculator tool when mathematical "
            "calculation is required, then explain the result."
        ),

        "quiz": (
            "Workflow: quiz. "
            "Use the quiz_generator tool before generating "
            "the final educational quiz."
        ),

        "study_plan": (
            "Workflow: study_plan. "
            "Use the study_plan_generator tool before "
            "generating the final study plan."
        ),

        "current_information": (
            "Workflow: current_information. "
            "Use the web_search tool to retrieve current "
            "or up-to-date educational information."
        )
    }

    if intent not in workflows:
        return (
            "Unknown education intent. "
            "Use one of: explanation, numerical, quiz, "
            "study_plan, current_information."
        )

    return workflows[intent]


# ========================================
# Quiz Generator
# ========================================
def quiz_generator(
    subject: str,
    topic: str,
    number_of_questions: int = 10,
    difficulty: str = "medium"
):
    """
    Prepare structured instructions for
    generating an educational quiz.
    """

    if number_of_questions < 1:
        return (
            "Quiz error: number_of_questions "
            "must be at least 1."
        )

    if number_of_questions > 20:
        return (
            "Quiz error: maximum 20 questions "
            "are allowed."
        )

    allowed_difficulties = {
        "easy",
        "medium",
        "hard"
    }

    if difficulty.lower() not in allowed_difficulties:
        return (
            "Quiz error: difficulty must be "
            "easy, medium, or hard."
        )

    return (
        f"Create a {number_of_questions}-question quiz.\n"
        f"Subject: {subject}\n"
        f"Topic: {topic}\n"
        f"Difficulty: {difficulty.lower()}\n\n"
        "Include clear questions and options "
        "where appropriate. "
        "Provide an answer key at the end."
    )


# ========================================
# Study Plan Generator
# ========================================
def study_plan_generator(
    subject: str,
    days: int,
    hours_per_day: float,
    topics: str = ""
):
    """
    Prepare structured instructions for
    generating a personalized educational
    study plan.
    """

    if days < 1:
        return (
            "Study plan error: days must be "
            "at least 1."
        )

    if days > 30:
        return (
            "Study plan error: maximum 30 days "
            "are allowed."
        )

    if hours_per_day <= 0:
        return (
            "Study plan error: hours_per_day "
            "must be greater than 0."
        )

    if hours_per_day > 12:
        return (
            "Study plan error: hours_per_day "
            "cannot exceed 12."
        )

    return (
        f"Create a {days}-day study plan.\n"
        f"Subject: {subject}\n"
        f"Study time per day: {hours_per_day} hours\n"
        f"Topics: {topics if topics else 'Not specified'}\n\n"
        "Create a practical day-by-day study schedule. "
        "Include learning, revision, practice, and "
        "self-assessment where appropriate."
    )


# ========================================
# Tool Definitions
# ========================================
TOOLS = [

    # ------------------------------------
    # Calculator
    # ------------------------------------
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Perform mathematical calculations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "Mathematical expression "
                            "to calculate."
                        )
                    }
                },
                "required": [
                    "expression"
                ]
            }
        }
    },

    # ------------------------------------
    # Web Search
    # ------------------------------------
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current, "
                "recent, or up-to-date information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The web search query."
                        )
                    }
                },
                "required": [
                    "query"
                ]
            }
        }
    },

    # ------------------------------------
    # Education Router
    # ------------------------------------
    {
        "type": "function",
        "function": {
            "name": "education_router",
            "description": (
                "Identify the type of education "
                "task requested by the user and "
                "return the appropriate workflow."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": [
                            "explanation",
                            "numerical",
                            "quiz",
                            "study_plan",
                            "current_information"
                        ],
                        "description": (
                            "The education task category."
                        )
                    }
                },
                "required": [
                    "intent"
                ]
            }
        }
    },

    # ------------------------------------
    # Quiz Generator
    # ------------------------------------
    {
        "type": "function",
        "function": {
            "name": "quiz_generator",
            "description": (
                "Generate a structured educational "
                "quiz for a given subject and topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": (
                            "The academic subject."
                        )
                    },
                    "topic": {
                        "type": "string",
                        "description": (
                            "The topic for the quiz."
                        )
                    },
                    "number_of_questions": {
                        "type": "integer",
                        "description": (
                            "Number of quiz questions."
                        )
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": [
                            "easy",
                            "medium",
                            "hard"
                        ],
                        "description": (
                            "Quiz difficulty level."
                        )
                    }
                },
                "required": [
                    "subject",
                    "topic"
                ]
            }
        }
    },

    # ------------------------------------
    # Study Plan Generator
    # ------------------------------------
    {
        "type": "function",
        "function": {
            "name": "study_plan_generator",
            "description": (
                "Generate a structured educational "
                "study plan for a subject."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": (
                            "The academic subject."
                        )
                    },
                    "days": {
                        "type": "integer",
                        "description": (
                            "Number of days available "
                            "for study."
                        )
                    },
                    "hours_per_day": {
                        "type": "number",
                        "description": (
                            "Available study hours "
                            "per day."
                        )
                    },
                    "topics": {
                        "type": "string",
                        "description": (
                            "Topics that should be "
                            "included in the study plan."
                        )
                    }
                },
                "required": [
                    "subject",
                    "days",
                    "hours_per_day"
                ]
            }
        }
    }

]

                    


        
