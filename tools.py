from tavily import TavilyClient


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
            search_depth="basic",
            max_results=5
        )

        results = []

        for item in response.get("results", []):
            results.append(
                f"Title: {item.get('title')}\n"
                f"Content: {item.get('content')}\n"
                f"URL: {item.get('url')}"
            )

        return "\n\n".join(results)

    except Exception as e:
        return f"Web search failed: {str(e)}"


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
