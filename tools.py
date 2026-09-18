import os
import json
import re
import serpapi
from dotenv import load_dotenv

load_dotenv()
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


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

        return str(result)

    except Exception as e:
        return f"Calculator error: {str(e)}"

# ========================================
# Web Search Tool
# ========================================

def web_search(query: str):

    try:

        client = serpapi.Client(
            api_key=os.getenv("SERPAPI_KEY")
        )

        results = client.search({
            "engine": "google",
            "q": query
        })

        organic_results = results.get(
            "organic_results",
            []
        )

        formatted_results = []

        for index, item in enumerate(
            organic_results[:5],
            start=1
        ):

            formatted_results.append(
                {
                    "source_id": index,
                    "title": item.get(
                        "title",
                        ""
                    ),
                    "url": item.get(
                        "link",
                        ""
                    ),
                    "content": item.get(
                        "snippet",
                        ""
                    )
                }
            )

        return json.dumps(
            formatted_results,
            ensure_ascii=False
        )

    except Exception as e:

        return json.dumps(
            {
                "error": f"Web search error: {str(e)}"
            },
            ensure_ascii=False
        )



                    


# ========================================
# Education Router
# ========================================

def education_router(intents):

    valid_intents = [
       "explanation",
       "numerical",
       "quiz",
       "study_plan",
       "current_information",
       "weak_topic",
       "quiz_result"
    ]

    
        

    if isinstance(
        intents,
        str
    ):

        intents = [intents]

    valid = [
        intent
        for intent in intents
        if intent in valid_intents
    ]

    return json.dumps(
        {
            "intents": valid
        },
        ensure_ascii=False
    )


# ========================================
# Quiz Generator
# ========================================

def quiz_generator(
    subject: str,
    topic: str,
    number_of_questions: int = 10,
    difficulty: str = "medium"
):

    # Keep quiz size within allowed range
    number_of_questions = max(
        1,
        min(number_of_questions, 20)
    )

    # Validate difficulty
    if difficulty not in [
        "easy",
        "medium",
        "hard"
    ]:
        difficulty = "medium"

    prompt = f"""
Generate exactly {number_of_questions} multiple-choice questions.

Subject: {subject}
Topic: {topic}
Difficulty: {difficulty}

Requirements:

1. Generate exactly {number_of_questions} questions.
2. Each question must have exactly 4 options.
3. Options must be A, B, C and D.
4. Only one option must be correct.
5. Include the correct answer for every question.
6. Include a short explanation for every correct answer.
7. Do not add questions outside the requested topic.
8. Return ONLY valid JSON.
9. Do not use Markdown.
10. Do not include ```json or ```.

Use this exact JSON structure:

{{
  "subject": "{subject}",
  "topic": "{topic}",
  "difficulty": "{difficulty}",
  "questions": [
    {{
      "question": "Question text",
      "options": {{
        "A": "Option A",
        "B": "Option B",
        "C": "Option C",
        "D": "Option D"
      }},
      "correct_answer": "A",
      "explanation": "Short explanation"
    }}
  ]
}}
"""

    try:

        response = client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert educational "
                        "quiz generator. Return only valid JSON."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.3
        )

        content = response.choices[0].message.content.strip()

        # Remove accidental Markdown fences
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

        quiz_data = json.loads(content)

        # Basic validation
        if not isinstance(
            quiz_data.get("questions"),
            list
        ):

            raise ValueError(
                "Quiz questions are missing."
            )

        if len(
            quiz_data["questions"]
        ) != number_of_questions:

            raise ValueError(
                f"Expected {number_of_questions} questions, "
                f"but received {len(quiz_data['questions'])}."
            )

        return json.dumps(
            quiz_data,
            ensure_ascii=False
        )

    except Exception as e:

        return json.dumps(
            {
                "error": f"Quiz generation error: {str(e)}"
            },
            ensure_ascii=False
        )


            


# ========================================
# Study Plan Generator
# ========================================

def study_plan_generator(
    subject: str,
    days: int = 7,
    hours_per_day: float = 2,
    topics: str = ""
):

    if days < 1:
        days = 1

    if days > 30:
        days = 30

    if hours_per_day <= 0:
        hours_per_day = 1

    if hours_per_day > 12:
        hours_per_day = 12

    return json.dumps(
        {
            "subject": subject,
            "days": days,
            "hours_per_day": hours_per_day,
            "topics": topics,
            "instruction": (
                f"Create a {days}-day study plan "
                f"for {subject} with "
                f"{hours_per_day} hours per day."
            )
        },
        ensure_ascii=False
    )


# ========================================
# Weak Topic Detector
# ========================================

def weak_topic_detector(progress):

    weak_topics = []

    for item in progress:

        score = item.get(
            "score"
        )

        if score is None:
            continue

        score_type = item.get(
            "score_type"
        )

        if (
            score_type == "percentage"
            and score < 70
        ):

            weak_topics.append(
                {
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
                }
            )

    return json.dumps(
        weak_topics,
        ensure_ascii=False
    )

# ========================================
# Quiz Result Analyzer
# ========================================

def quiz_result_analyzer(
    subject: str,
    topic: str,
    score: float,
    score_type: str = "percentage",
    total_questions: int = 0,
    correct_answers: int = 0
):

    result = {
        "subject": subject,
        "topic": topic,
        "score": score,
        "score_type": score_type,
        "total_questions": total_questions,
        "correct_answers": correct_answers
    }

    if score_type == "percentage":

        if score < 70:
            result["performance"] = "weak"
            result["recommendation"] = (
                "Revise this topic before moving to new topics."
            )

        elif score < 85:
            result["performance"] = "developing"
            result["recommendation"] = (
                "Practice more questions to strengthen understanding."
            )

        else:
            result["performance"] = "strong"
            result["recommendation"] = (
                "Topic performance is strong. Continue with periodic revision."
            )

    return json.dumps(
        result,
        ensure_ascii=False
    )


# ========================================
# Tool Definitions
# ========================================

TOOLS = [

    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Calculate a mathematical expression."
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

    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query."
                        )
                    }
                },
                "required": [
                    "query"
                ]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "education_router",
            "description": (
                "Route an education request "
                "to one or more applicable workflows."
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
                                "weak_topic",
                                "quiz_result"
                            
                                
                            ]
                        },
                        "description": (
                            "Applicable education intents."
                        )
                    }
                },
                "required": [
                    "intents"
                ]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "quiz_generator",
            "description": (
                "Generate quiz parameters "
                "for an education topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string"
                    },
                    "topic": {
                        "type": "string"
                    },
                    "number_of_questions": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": [
                            "easy",
                            "medium",
                            "hard"
                        ]
                    }
                },
                "required": [
                    "subject",
                    "topic"
                ]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "study_plan_generator",
            "description": (
                "Generate study plan parameters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string"
                    },
                    "days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 30
                    },
                    "hours_per_day": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 12
                    },
                    "topics": {
                        "type": "string"
                    }
                },
                "required": [
                    "subject"
                ]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "weak_topic_detector",
            "description": (
                "Identify learning topics "
                "that need more revision "
                "based on recorded performance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "progress": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subject": {
                                    "type": "string"
                                },
                                "topic": {
                                    "type": "string"
                                },
                                "score": {
                                    "type": [
                                        "number",
                                        "null"
                                    ]
                                },
                                "score_type": {
                                    "type": [
                                        "string",
                                        "null"
                                    ]
                                },
                                "note": {
                                    "type": "string"
                                }
                            },
                            "required": [
                                "subject",
                                "topic",
                                "score",
                                "score_type",
                                "note"
                            ]
                        }
                    }
                },
                "required": [
                    "progress"
                ]
            }
        }
    },


        {
        "type": "function",
        "function": {
            "name": "quiz_result_analyzer",
            "description": (
                "Analyze a quiz result and classify "
                "learning performance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string"
                    },
                    "topic": {
                        "type": "string"
                    },
                    "score": {
                        "type": "number"
                    },
                    "score_type": {
                        "type": "string",
                        "enum": [
                            "percentage"
                        ]
                    },
                    "total_questions": {
                        "type": "integer"
                    },
                    "correct_answers": {
                        "type": "integer",
                        "default": 0
                    }
                    
                        
                    
                },
                "required": [
                    "subject",
                    "topic",
                    "score"
                ]
            }
        }
        }




    


    

]





                    


        
