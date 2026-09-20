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
                "answer": "",
                "tool_trace": [],
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
            "http_status": 0,
            "answer": "",
            "tool_trace": [],
            "error": str(e)
        }


def calculate_metrics(results):

    total = len(results)

    passed = sum(
        1
        for result in results
        if result["status"] == "PASS"
    )

    failed = total - passed

    validation_rejects = 0
    recovery_successes = 0
    multi_step_tests = 0
    multi_step_successes = 0
    final_response_successes = 0

    for result in results:

        trace = result.get("tool_trace", [])

        # -----------------------------
        # Validation rejection count
        # -----------------------------

        rejects = [
            item
            for item in trace
            if item.get("status") == "rejected"
        ]

        validation_rejects += len(rejects)

        # -----------------------------
        # Recovery detection
        # -----------------------------

        has_retry = any(
            "retry" in item.get("tool", "").lower()
            for item in trace
        )

        has_retry_success = any(
            "retry" in item.get("tool", "").lower()
            and item.get("status") == "success"
            for item in trace
        )

        if has_retry and has_retry_success:
            recovery_successes += 1

        # -----------------------------
        # Multi-step detection
        # -----------------------------

        planner_steps = 0

        for item in trace:

            if item.get("tool") == "agent_planner":

                planner_result = item.get(
                    "result",
                    {}
                )

                if isinstance(planner_result, dict):

                    planner_steps = len(
                        planner_result.get(
                            "steps",
                            []
                        )
                    )

                break

        if planner_steps > 1:

            multi_step_tests += 1

            executed_tools = [
                item.get("tool")
                for item in trace
            ]

            if (
                "weak_topic_detector"
                in executed_tools
                and
                "study_plan_generator"
                in executed_tools
            ):
                multi_step_successes += 1

        # -----------------------------
        # Final response
        # -----------------------------

        final_response_found = any(
            item.get("tool") == "final_response"
            and item.get("status") == "success"
            for item in trace
        )

        if final_response_found:
            final_response_successes += 1

    pass_rate = (
        (passed / total) * 100
        if total > 0
        else 0
    )

    recovery_rate = (
        (recovery_successes / validation_rejects) * 100
        if validation_rejects > 0
        else 100
    )

    multi_step_rate = (
        (multi_step_successes / multi_step_tests) * 100
        if multi_step_tests > 0
        else 100
    )

    return {
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(pass_rate, 2),
        "validation_rejects": validation_rejects,
        "recovery_successes": recovery_successes,
        "recovery_rate": round(recovery_rate, 2),
        "multi_step_tests": multi_step_tests,
        "multi_step_successes": multi_step_successes,
        "multi_step_success_rate": round(
            multi_step_rate,
            2
        ),
        "final_response_successes":
            final_response_successes
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
            "question":
                "What is the current latest Python version?",
            "session_id": "evaluation-auto-E2"
        },
        {
            "name": "E3 - Calculator Recovery",
            "question":
                "Calculate 125 × 48.",
            "session_id": "evaluation-auto-E3"
        },
        {
            "name": "E4 - Quiz",
            "question":
                "Give me a 5-question Physics quiz on Force.",
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

    metrics = calculate_metrics(results)

    return {
        "metrics": metrics,
        "results": results
    }


if __name__ == "__main__":

    evaluation = run_all_tests()

    print("\n" + "=" * 60)
    print("EduAgent AI Evaluation Metrics")
    print("=" * 60)

    metrics = evaluation["metrics"]

    print(
        f"Total Tests          : "
        f"{metrics['total_tests']}"
    )

    print(
        f"Passed               : "
        f"{metrics['passed']}"
    )

    print(
        f"Failed               : "
        f"{metrics['failed']}"
    )

    print(
        f"Pass Rate            : "
        f"{metrics['pass_rate']}%"
    )

    print(
        f"Validation Rejects   : "
        f"{metrics['validation_rejects']}"
    )

    print(
        f"Recovery Successes   : "
        f"{metrics['recovery_successes']}"
    )

    print(
        f"Recovery Rate        : "
        f"{metrics['recovery_rate']}%"
    )

    print(
        f"Multi-Step Tests     : "
        f"{metrics['multi_step_tests']}"
    )

    print(
        f"Multi-Step Success   : "
        f"{metrics['multi_step_successes']}"
    )

    print(
        f"Multi-Step Rate      : "
        f"{metrics['multi_step_success_rate']}%"
    )

    print(
        f"Final Responses     : "
        f"{metrics['final_response_successes']}/"
        f"{metrics['total_tests']}"
    )
