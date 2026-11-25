from datetime import date

from agent import MAX_STEPS, run_agent
from eval_answers import judge
from rag import load_chunks
from rag_embed import embed_chunks

TODAY = date.today().strftime("%B %d, %Y")

TEST_CASES = [
    {
        "question": "Hi, what can you help with?",
        "reference": "A friendly reply saying it can help with Acme Labs handbook or HR policy questions.",
        "must_use": [],
        "must_not_use": ["search_handbook", "calculate"],
    },
    {
        "question": "How many vacation days do I get?",
        "reference": "20 days of paid vacation per year, plus 10 company holidays.",
        "must_use": ["search_handbook"],
        "must_not_use": [],
    },
    {
        "question": "Can I work remotely full time?",
        "reference": "Only with VP approval. Otherwise employees can work from home up to 3 days per week with manager approval.",
        "must_use": ["search_handbook"],
        "must_not_use": [],
    },
    {
        "question": "What's the most I could get reimbursed for meals on a 4-day business trip?",
        "reference": "$300 ($75 per day for 4 days).",
        "must_use": ["search_handbook", "calculate"],
        "must_not_use": [],
    },
    {
        "question": "I started on March 1 this year. Roughly how many vacation days have I accrued so far?",
        "reference": (
            f"Today is {TODAY}. Vacation is 20 days per year, accruing monthly (about 1.67 days per month). "
            "The answer should multiply that by the number of months since March 1 of this year. "
            "Any reasonable rounding is fine."
        ),
        "must_use": ["search_handbook", "get_current_datetime", "calculate"],
        "must_not_use": [],
    },
    {
        "question": "What's 15% of the yearly learning budget?",
        "reference": "$300 (15% of $2,000).",
        "must_use": ["search_handbook", "calculate"],
        "must_not_use": [],
    },
    {
        "question": "What's the dress code?",
        "reference": "UNANSWERABLE",
        "must_use": ["search_handbook"],
        "must_not_use": [],
    },
]


def check_tools(tool_calls, case):
    failures = []
    for tool in case["must_use"]:
        if tool not in tool_calls:
            failures.append(f"didn't use {tool}")
    for tool in case["must_not_use"]:
        if tool in tool_calls:
            failures.append(f"shouldn't have used {tool}")
    return failures


def main():
    chunks = load_chunks("handbook.txt")
    embed_chunks(chunks)

    passed = 0
    total_tool_calls = 0

    for case in TEST_CASES:
        print(f"\n=== {case['question']} ===")
        messages = [{"role": "user", "content": case["question"]}]
        answer, tool_calls = run_agent(messages, chunks)
        total_tool_calls += len(tool_calls)

        failures = check_tools(tool_calls, case)
        if answer.startswith(f"Stopped after {MAX_STEPS} steps"):
            failures.append("hit the step limit")

        verdict = judge(case["question"], case["reference"], answer)
        if verdict["verdict"] != "pass":
            failures.append(f"judge: {verdict['reason']}")

        print(f"Answer: {answer}")
        print(f"Tools used: {tool_calls}")
        if failures:
            print("FAIL")
            for failure in failures:
                print(f"  - {failure}")
        else:
            print("PASS")
            passed += 1

    print(f"\nScore: {passed}/{len(TEST_CASES)} passed")
    print(f"Average tool calls per question: {total_tool_calls / len(TEST_CASES):.1f}")


if __name__ == "__main__":
    main()
