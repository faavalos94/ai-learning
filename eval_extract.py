from extract import extract_job

TEST_CASES = [
    {
        "name": "remote with salary",
        "posting": "AI Engineer at Acme Labs. Fully remote (US). Salary: $150k-$180k. "
        "Required: Python, LLM APIs, Docker. Nice to have: Kubernetes.",
        "expected": {
            "company": "Acme Labs",
            "remote": True,
            "salary_listed": True,
            "required_skills": ["Python", "Docker"],
        },
    },
    {
        "name": "on-site, no salary",
        "posting": "Machine Learning Engineer, Brightpath Health. On-site in Austin, TX, "
        "5 days a week. Must know Python and SQL. Experience with RAG is a plus.",
        "expected": {
            "company": "Brightpath Health",
            "remote": False,
            "salary_listed": False,
            "required_skills": ["Python", "SQL"],
        },
    },
    {
        "name": "hybrid",
        "posting": "Northwind is hiring an LLM Application Developer. Hybrid: 2 days a week "
        "in our Seattle office. Pay range $130,000 to $160,000. "
        "You need TypeScript and prompt engineering experience.",
        "expected": {
            "company": "Northwind",
            "remote": False,
            "salary_listed": True,
            "required_skills": ["TypeScript"],
        },
    },
    {
        "name": "nice-to-have kept separate",
        "posting": "Data Scientist at Globex. Remote-friendly. Requirements: Python, statistics. "
        "Bonus points for PyTorch or Go.",
        "expected": {
            "company": "Globex",
            "remote": True,
            "salary_listed": False,
            "required_skills": ["Python"],
            "not_required": ["PyTorch"],
        },
    },
]


def check(job, expected):
    failures = []

    if job["company"] != expected["company"]:
        failures.append(f"company: expected '{expected['company']}', got '{job['company']}'")

    if job["remote"] != expected["remote"]:
        failures.append(f"remote: expected {expected['remote']}, got {job['remote']}")

    salary_listed = job["salary_range"] != "Not listed"
    if salary_listed != expected["salary_listed"]:
        failures.append(f"salary: expected listed={expected['salary_listed']}, got '{job['salary_range']}'")

    required = [s.lower() for s in job["required_skills"]]
    for skill in expected["required_skills"]:
        if not any(skill.lower() in s for s in required):
            failures.append(f"missing required skill: {skill}")

    for skill in expected.get("not_required", []):
        if any(skill.lower() in s for s in required):
            failures.append(f"'{skill}' should not be a required skill")

    return failures


def main():
    passed = 0

    for case in TEST_CASES:
        print(f"--- {case['name']} ---")
        job = extract_job(case["posting"])
        failures = check(job, case["expected"])

        if failures:
            print("FAIL")
            for failure in failures:
                print(f"  - {failure}")
        else:
            print("PASS")
            passed += 1
        print()

    print(f"Score: {passed}/{len(TEST_CASES)} passed")


if __name__ == "__main__":
    main()
