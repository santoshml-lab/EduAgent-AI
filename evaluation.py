import requests


BASE_URL = "https://eduagent-ai-osvz.onrender.com"


def run_test(test_name, question, session_id):
    payload = {
        "question": question,
        "session_id": session_id
    }

    try:
        response = requests.post(
            f"{BASE_URL}/ask",
            json=payload,
            timeout=120
        )

        if response.status_code != 200:
            return {
                "test": test_name,
                "status": "FAIL",
                "http_status": response.status_code,
                "error": response.text
            }

        data = response.json()

        return {
            "test": test_name,
            "status": "PASS",
            "http_status": response.status_code,
            "answer": data.get("answer", ""),
            "tool_trace": data.get("tool_trace", [])
        }

    except Exception as e:
        return {
            "test": test_name,
            "status": "FAIL",
            "error": str(e)
        }


def run_all_tests():
    tests = [
        {
            "name": "E1 - Basic Agent",
            "question": "What is 25 × 16?",
            "session_id": "evaluation-auto-E1"
        },
        {
            "name": "E2 - Web Search",
            "question": "What is the current latest Python version?",
            "session_id": "evaluation-auto-E2"
        },
        {
            "name": "E3 - Calculator Recovery",
            "question": "Calculate 125 × 48.",
            "session_id": "evaluation-auto-E3"
        },
        {
            "name": "E4 - Quiz",
            "question": "Give me a 5-question Physics quiz on Force.",
            "session_id": "evaluation-auto-E4"
        },
        {
            "name": "E5 - Multi-Step Agent",
            "question": (
                "My Physics scores are: Force 45%, "
                "Work Energy 55%, Light 90%. "
                "Find my weak topics and make a 7-day "
                "revision plan with 2 hours per day."
            ),
            "session_id": "evaluation-auto-E5"
        }
    ]

    results = []

    for test in tests:
        result = run_test(
            test_name=test["name"],
            question=test["question"],
            session_id=test["session_id"]
        )

        results.append(result)

    passed = sum(
        1
        for result in results
        if result["status"] == "PASS"
    )

    total = len(results)

    return {
        "summary": {
            "passed": passed,
            "failed": total - passed,
            "total": total
        },
        "results": results
    }


if __name__ == "__main__":

    evaluation = run_all_tests()

    print("\n" + "=" * 60)
    print("EduAgent AI Evaluation")
    print("=" * 60)

    print(
        f"Passed: "
        f"{evaluation['summary']['passed']}/"
        f"{evaluation['summary']['total']}"
    )

    for result in evaluation["results"]:
        print(
            f"{result['test']} → "
            f"{result['status']}"
        )
