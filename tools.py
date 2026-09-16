import json
from tavily import TavilyClient


def calculator(expression: str):
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Calculation result: {result}"

    except Exception as e:
        return f"Calculator error: {str(e)}"


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


def education_router(intent: str):
    """
    Identify the type of education task requested by the user.
    """

    allowed_intents = {
        "explanation",
        "numerical",
        "quiz",
        "study_plan",
        "current_information"
    }

    if intent not in allowed_intents:
        return (
            "Unknown education intent. "
            "Use one of: explanation, numerical, quiz, "
            "study_plan, current_information."
        )

    return f"Education task identified: {intent}"

def quiz_generator(
    subject: str,
    topic: str,
    number_of_questions: int = 10,
    difficulty: str = "medium"
):
    """
    Prepare structured instructions for generating an educational quiz.
    """

    if number_of_questions < 1:
        return "Quiz error: number_of_questions must be at least 1."

    if number_of_questions > 20:
        return "Quiz error: maximum 20 questions are allowed."

    allowed_difficulties = {
        "easy",
        "medium",
        "hard"
    }

    if difficulty.lower() not in allowed_difficulties:
        return (
            "Quiz error: difficulty must be easy, medium, or hard."
        )

    return (
        f"Create a {number_of_questions}-question quiz.\n"
        f"Subject: {subject}\n"
        f"Topic: {topic}\n"
        f"Difficulty: {difficulty.lower()}\n\n"
        "Include clear questions and options where appropriate. "
        "Provide an answer key at the end."
    )


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform mathematical calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to calculate."
                    }
                },
                "required": ["expression"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current, recent, or "
                "up-to-date information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The web search query."
                    }
                },
                "required": ["query"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "education_router",
            "description": (
                "Identify the type of education task requested "
                "by the user."
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
                "required": ["intent"]
            }
        }
    }

    {
    "type": "function",
    "function": {
        "name": "quiz_generator",
        "description": (
            "Generate a structured educational quiz "
            "for a given subject and topic."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "The academic subject."
                },
                "topic": {
                    "type": "string",
                    "description": "The topic for the quiz."
                },
                "number_of_questions": {
                    "type": "integer",
                    "description": "Number of quiz questions."
                },
                "difficulty": {
                    "type": "string",
                    "enum": [
                        "easy",
                        "medium",
                        "hard"
                    ],
                    "description": "Quiz difficulty level."
                }
            },
            "required": [
                "subject",
                "topic"
            ]
        }
    }
},
]


        
