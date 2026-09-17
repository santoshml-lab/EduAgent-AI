import os
import json
import serpapi
from dotenv import load_dotenv

load_dotenv()

serpapi_client = serpapi.Client(
api_key=os.getenv("SERPAPI_KEY")
)

#========================================

Calculator Tool

#========================================

def calculator(expression: str):
try:
result = eval(
expression,
{"builtins": {}},
{}
)

    return f"Calculation result: {result}"

except Exception as e:
    return f"Calculator error: {str(e)}"

#========================================

Web Search Tool

#========================================

def web_search(query: str):
"""
Search the web using SerpApi Google Search.
Returns structured search results for the agent.
"""

try:

    results = serpapi_client.search({
        "engine": "google",
        "q": query
    })

    organic_results = results.get(
        "organic_results",
        []
    )

    if not organic_results:

        return json.dumps({
            "error": "No search results found."
        })

    sources = []

    for index, result in enumerate(
        organic_results[:5],
        start=1
    ):

        sources.append({
            "source_id": f"source_{index}",
            "title": result.get(
                "title",
                ""
            ),
            "url": result.get(
                "link",
                ""
            ),
            "content": result.get(
                "snippet",
                ""
            )
        })

    return json.dumps(
        sources,
        ensure_ascii=False
    )

except Exception as e:

    return json.dumps({
        "error": (
            f"Web search failed: {str(e)}"
        )
    })

#========================================

Education Router

#========================================

def education_router(intents):
"""
Identify one or more education task types
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
        "or up-to-date information."
    ),

    "weak_topic": (
        "Workflow: weak_topic. "
        "Use the weak_topic_detector tool to identify "
        "learning topics that need additional revision "
        "based on explicitly recorded performance."
    )
}

# ----------------------------------------
# Convert single string to list
# ----------------------------------------
if isinstance(intents, str):

    intents = [intents]

# ----------------------------------------
# Validate intents
# ----------------------------------------
valid_intents = []

for intent in intents:

    if intent in workflows:

        valid_intents.append(intent)

# ----------------------------------------
# No valid intent
# ----------------------------------------
if not valid_intents:

    return (
        "Unknown education intent. "
        "Use one or more of: explanation, "
        "numerical, quiz, study_plan, "
        "current_information, weak_topic."
    )

# ----------------------------------------
# Build workflow result
# ----------------------------------------
selected_workflows = []

for intent in valid_intents:

    selected_workflows.append(
        workflows[intent]
    )

return "\n".join(
    selected_workflows
)

#========================================

Quiz Generator

#========================================

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

# ----------------------------------------
# Validate question count
# ----------------------------------------
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

# ----------------------------------------
# Validate difficulty
# ----------------------------------------
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

# ----------------------------------------
# Generate instructions
# ----------------------------------------
return (
    f"Create a {number_of_questions}-question quiz.\n"
    f"Subject: {subject}\n"
    f"Topic: {topic}\n"
    f"Difficulty: {difficulty.lower()}\n\n"
    "Include clear questions and options "
    "where appropriate. "
    "Provide an answer key at the end."
)

#========================================

Study Plan Generator

#========================================

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

# ----------------------------------------
# Validate days
# ----------------------------------------
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

# ----------------------------------------
# Validate hours
# ----------------------------------------
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

# ----------------------------------------
# Generate instructions
# ----------------------------------------
return (
    f"Create a {days}-day study plan.\n"
    f"Subject: {subject}\n"
    f"Study time per day: "
    f"{hours_per_day} hours\n"
    f"Topics: "
    f"{topics if topics else 'Not specified'}\n\n"
    "Create a practical day-by-day study schedule. "
    "Include learning, revision, practice, and "
    "self-assessment where appropriate."
)

#========================================

Weak Topic Detector

#========================================

def weak_topic_detector(progress):
"""
Identify learning topics that need more revision
based on explicitly recorded performance scores.
"""

weak_topics = []

for item in progress:

    score = item.get("score")

    if score is None:
        continue

    score_type = item.get(
        "score_type"
    )

    if (
        score_type == "percentage"
        and score < 70
    ):

        weak_topics.append({
            "subject": item.get(
                "subject",
                ""
            ),
            "topic": item.get(
                "topic",
                ""
            ),
            "score": score,
            "reason": "Score is below 70%"
        })

return json.dumps(
    weak_topics,
    ensure_ascii=False
)

#========================================

Tool Definitions

#========================================

TOOLS = [

# ====================================
# Calculator
# ====================================
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


# ====================================
# Web Search
# ====================================
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


# ====================================
# Education Router
# ====================================
{
    "type": "function",

    "function": {

        "name": "education_router",

        "description": (
            "Identify ALL education task types "
            "requested by the user. "
            "For a multi-task question, return "
            "ALL applicable task categories."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "intents": {

                    "type": "array",

                    "items": {

                        "type": "string",

                        "enum": [
                            "explanation",
                            "numerical",
                            "quiz",
                            "study_plan",
                            "current_information",
                            "weak_topic"
                        ]
                    },

                    "minItems": 1,

                    "description": (
                        "One or more education task "
                        "categories. For multiple "
                        "tasks, include ALL applicable "
                        "categories."
                    )
                }
            },

            "required": [
                "intents"
            ]
        }
    }
},


# ====================================
# Quiz Generator
# ====================================
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


# ====================================
# Study Plan Generator
# ====================================
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
},


# ====================================
# Weak Topic Detector
# ====================================
{
    "type": "function",

    "function": {

        "name": "weak_topic_detector",

        "description": (
            "Identify learning topics that need "
            "additional revision based on explicitly "
            "recorded performance scores."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "progress": {

                    "type": "array",

                    "description": (
                        "Previously recorded learning "
                        "progress and quiz scores."
                    ),

                    "items": {

                        "type": "object",

                        "properties": {

                            "subject": {

                                "type": "string",

                                "description": (
                                    "Academic subject."
                                )
                            },

                            "topic": {

                                "type": "string",

                                "description": (
                                    "Learning topic."
                                )
                            },

                            "score": {

                                "type": "number",

                                "description": (
                                    "Recorded performance "
                                    "score."
                                )
                            },

                            "score_type": {

                                "type": "string",

                                "description": (
                                    "Type of score, such "
                                    "as percentage or marks."
                                )
                            },

                            "note": {

                                "type": "string",

                                "description": (
                                    "Explicit performance "
                                    "note."
                                )
                            }
                        }
                    }
                }
            },

            "required": [
                "progress"
            ]
        }
    }
}

]

                    


        
