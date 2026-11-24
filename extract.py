import json

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "company": {"type": "string"},
        "location": {"type": "string"},
        "remote": {"type": "boolean"},
        "salary_range": {"type": "string"},
        "required_skills": {"type": "array", "items": {"type": "string"}},
        "nice_to_have_skills": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "title",
        "company",
        "location",
        "remote",
        "salary_range",
        "required_skills",
        "nice_to_have_skills",
        ],
        "additionalProperties": False,
}


def extract_job(posting_text):
    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=4000,
        system="Extract details from the job posting. If a text field isn't mentioned, use 'Not listed'.",
        messages=[{"role": "user", "content": posting_text}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
    )
    print(f"[tokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out]\n")
    print(f"[stop reason: {response.stop_reason}]\n")

    if response.stop_reason == "max_tokens":
        raise RuntimeError("Reply was cut off by max_tokens. Increase max_tokens and try again.")
    
    text = next(block.text for block in response.content if block.type == "text")
    return json.loads(text)


def main():
    with open("posting.txt") as f:
        posting = f.read()

    job = extract_job(posting)

    print("Raw data:")
    print(json.dumps(job, indent=2))

    print(f"\n{job['title']} at {job['company']} ({job['location']})")
    print(f"Remote: {'yes' if job['remote'] else 'no'}")
    print(f"Salary: {job['salary_range']}")
    print(f"\n{len(job['required_skills'])} required skills:")
    for skill in job["required_skills"]:
        print(f"  - {skill}")

    with open("job.json", "w") as f:
        json.dump(job, f, indent=2)
    print("\nSaved to job.json")


if __name__ == "__main__":
    main()
