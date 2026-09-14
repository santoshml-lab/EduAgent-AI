def calculator(expression: str):
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Calculation result: {result}"
    except Exception:
        return "Unable to calculate this expression."
