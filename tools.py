from tavily import TavilyClient
import json


def calculator(expression: str):
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Calculation result: {result}"
    except Exception:
        return "Unable to calculate this expression."

def web_search(query: str):
    try:
        tavily = TavilyClient()

        response = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=5
        )

        results = []

        for item in response.get("results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", "")
                }
            )

        return json.dumps(results, ensure_ascii=False)

    except Exception as e:
        return json.dumps(
            {"error": f"Web search failed: {str(e)}"},
            ensure_ascii=False
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
            "description": "Search the web for current or up-to-date information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    }
                },
                "required": ["query"]
            }
        }
    }
]
