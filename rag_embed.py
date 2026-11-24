import math

import voyageai
from anthropic import Anthropic
from dotenv import load_dotenv

from rag import load_chunks

load_dotenv()
client = Anthropic()
vo = voyageai.Client()  # automatically reads VOYAGE_API_KEY

EMBED_MODEL = "voyage-4-lite"


def embed_chunks(chunks):
    texts = [f"{c['title']}\n{c['text']}" for c in chunks]
    result = vo.embed(texts, model=EMBED_MODEL, input_type="document")
    for chunk, vector in zip(chunks, result.embeddings):
        chunk["embedding"] = vector
    print(f"[embedded {len(chunks)} chunks, {len(result.embeddings[0])} numbers each]")


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    length_a = math.sqrt(sum(x * x for x in a))
    length_b = math.sqrt(sum(y * y for y in b))
    return dot / (length_a * length_b)


def retrieve(question, chunks, k=2):
    query_vector = vo.embed([question], model=EMBED_MODEL, input_type="query").embeddings[0]

    for chunk in chunks:
        chunk["score"] = cosine_similarity(query_vector, chunk["embedding"])

    ranked = sorted(chunks, key=lambda c: c["score"], reverse=True)
    for chunk in ranked:
        print(f"  {chunk['score']:.3f}  {chunk['title']}")
    return ranked[:k]


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
    embed_chunks(chunks)
    print("Type 'quit' to exit.")

    while True:
        question = input("\nQuestion: ")
        if question.strip().lower() == "quit":
            break
        print("Answer:", answer(question, chunks))


if __name__ == "__main__":
    main()
