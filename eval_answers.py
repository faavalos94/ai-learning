import json

from anthropic import Anthropic
from dotenv import load_dotenv

from rag import load_chunks
from rag_embed import EMBED_MODEL, embed_chunks, generate_answer, rank_chunks, vo

load_dotenv()
client = Anthropic()

TEST_CASES = [
    {
        "question": "How many vacation days do I get?",
        "reference": "20 days of paid vacation per year, plus 10 company holidays.",
    },
    {
        "question": "Can I telecommute?",
        "reference": "Yes, up to 3 days per week with manager approval. Fully remote requires VP approval.",
    },
    {
        "question": "Will I get a computer?",
        "reference": "Yes, a MacBook Pro, refreshed every 3 years.",
    },
    {
        "question": "Do I need a doctor's note if I'm sick for two days?",
        "reference": "No. A doctor's note isn't needed for absences of 3 days or less.",
    },
    {
        "question": "How much can I spend on meals per day on a business trip?",
        "reference": "Up to $75 per day.",
    },
    {
        "question": "How big is the learning budget, and does it roll over?",
        "reference": "$2,000 per year. Unused budget does not carry over.",
    },
    {
        "question": "How many unused vacation days roll over?",
        "reference": "Up to 5 days.",
    },
    {
        "question": "How many sick days can I roll over to next year?",
        "reference": "None. Sick days do not roll over.",
    },
    {
        "question": "What's the dress code?",
        "reference": "UNANSWERABLE",
    },
    {
        "question": "Does the company offer a 401(k) match?",
        "reference": "UNANSWERABLE",
    },
]

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
    },
    "required": ["reason", "verdict"],
    "additionalProperties": False,
}

JUDGE_PROMPT = """You are grading an answer from a company handbook assistant.

Question: {question}
Reference answer: {reference}
Assistant's answer: {answer}

Grade "pass" if the assistant's answer contains the key facts in the reference answer and \
doesn't add facts that contradict it or aren't supported by it. Wording can differ.
If the reference answer is "UNANSWERABLE", grade "pass" only if the assistant says it \
doesn't know or that the handbook doesn't cover it.
Explain your reasoning briefly in "reason" before giving the verdict."""


def judge(question, reference, answer):
    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(question=question, reference=reference, answer=answer),
            }
        ],
        output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
    )
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Judge reply was cut off by max_tokens.")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def main():
    chunks = load_chunks("handbook.txt")
    embed_chunks(chunks)

    questions = [case["question"] for case in TEST_CASES]
    query_vectors = vo.embed(questions, model=EMBED_MODEL, input_type="query").embeddings

    passed = 0
    for case, query_vector in zip(TEST_CASES, query_vectors):
        retrieved = rank_chunks(query_vector, chunks)
        answer = generate_answer(case["question"], retrieved)
        result = judge(case["question"], case["reference"], answer)

        if result["verdict"] == "pass":
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"

        print(f"\n[{status}] {case['question']}")
        print(f"  Retrieved: {[c['title'] for c in retrieved]}")
        print(f"  Answer:    {answer}")
        print(f"  Judge:     {result['reason']}")

    print(f"\nScore: {passed}/{len(TEST_CASES)} passed")


if __name__ == "__main__":
    main()
