import re

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()

STOPWORDS = {
    "a", "an", "the", "is", "are", "do", "does", "i", "my", "we", "to", "of",
    "in", "on", "for", "and", "or", "how", "what", "when", "can", "much",
    "many", "it", "be", "at", "with", "you", "our",
}


def load_chunks(path):
    with open(path) as f:
        text = f.read()

    chunks = []
    for section in text.split("## ")[1:]:
        title, _, body = section.partition("\n")
        chunks.append({"title": title.strip(), "text": body.strip()})
    return chunks


def words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def retrieve(question, chunks, k=2):
    question_words = words(question)

    def score(chunk):
        return len(question_words & words(chunk["title"] + " " + chunk["text"]))

    ranked = sorted(chunks, key=score, reverse=True)
    return [chunk for chunk in ranked[:k] if score(chunk) > 0]


def answer(question, chunks):
    retrieved = retrieve(question, chunks)
    print(f"  [retrieved: {[c['title'] for c in retrieved]}]")

    context = "\n\n".join(f"### {c['title']}\n{c['text']}" for c in retrieved)

    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=2000,
        system=(
            "Answer the question using only the handbook sections provided. "
            "Say which section the answer came from. "
            "If the answer isn't in the sections, say you don't know."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Handbook sections:\n\n{context}\n\nQuestion: {question}",
            }
        ],
    )
    return next((b.text for b in response.content if b.type == "text"), "")


def main():
    chunks = load_chunks("handbook.txt")
    print(f"Loaded {len(chunks)} sections. Type 'quit' to exit.")

    while True:
        question = input("\nQuestion: ")
        if question.strip().lower() == "quit":
            break
        print("Answer:", answer(question, chunks))


if __name__ == "__main__":
    main()
