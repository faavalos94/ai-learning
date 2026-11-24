from rag import load_chunks
from rag import retrieve as keyword_retrieve
from rag_embed import EMBED_MODEL, cosine_similarity, embed_chunks, vo

TEST_CASES = [
    ("How many vacation days do I get?", "Paid Time Off"),
    ("Can unused vacation carry over to next year?", "Paid Time Off"),
    ("How far ahead should I request time off?", "Paid Time Off"),
    ("Do I need a doctor's note if I'm ill for two days?", "Sick Leave"),
    ("What if I'm feeling under the weather?", "Sick Leave"),
    ("Can I telecommute?", "Remote Work"),
    ("What are the core hours?", "Remote Work"),
    ("Will I get a computer?", "Equipment"),
    ("Is there money for a desk and chair at home?", "Equipment"),
    ("I lost my work laptop, what should I do?", "Equipment"),
    ("How much can I spend on dinner during a business trip?", "Expenses"),
    ("Do I need approval to buy something expensive?", "Expenses"),
    ("Is there a budget for training courses?", "Professional Development"),
    ("Will the company pay for a conference ticket?", "Professional Development"),
]


def embedding_ranking(query_vector, chunks):
    return sorted(
        chunks,
        key=lambda c: cosine_similarity(query_vector, c["embedding"]),
        reverse=True,
    )


def rank_of(expected_title, ranked_chunks):
    titles = [c["title"] for c in ranked_chunks]
    if expected_title in titles:
        return titles.index(expected_title) + 1
    return None


def hit_rate(ranks, k):
    hits = sum(1 for r in ranks if r is not None and r <= k)
    return f"{hits}/{len(ranks)}"


def main():
    chunks = load_chunks("handbook.txt")
    embed_chunks(chunks)

    questions = [question for question, _ in TEST_CASES]
    query_vectors = vo.embed(questions, model=EMBED_MODEL, input_type="query").embeddings

    keyword_ranks = []
    embedding_ranks = []

    print(f"\n{'Question':<52} {'Expected':<26} {'Keyword':>8} {'Embed':>6}")
    print("-" * 95)

    for (question, expected), query_vector in zip(TEST_CASES, query_vectors):
        kw_rank = rank_of(expected, keyword_retrieve(question, chunks, k=len(chunks)))
        em_rank = rank_of(expected, embedding_ranking(query_vector, chunks))
        keyword_ranks.append(kw_rank)
        embedding_ranks.append(em_rank)

        kw_text = str(kw_rank) if kw_rank else "miss"
        em_text = str(em_rank) if em_rank else "miss"
        print(f"{question[:50]:<52} {expected:<26} {kw_text:>8} {em_text:>6}")

    print("\nCorrect section ranked #1 (hit@1):")
    print(f"  Keyword:   {hit_rate(keyword_ranks, 1)}")
    print(f"  Embedding: {hit_rate(embedding_ranks, 1)}")
    print("Correct section in top 2 (hit@2, what rag.py actually sends):")
    print(f"  Keyword:   {hit_rate(keyword_ranks, 2)}")
    print(f"  Embedding: {hit_rate(embedding_ranks, 2)}")


if __name__ == "__main__":
    main()
