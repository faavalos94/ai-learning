def fake_reply(messages):
    last = messages[-1]["content"]
    return f"You said '{last}'. I've seen {len(messages)} messages so far."

def main():
    messages = []
    print("Type 'quit' to exit.")

    while True:
        user_text = input("You: ")
        if user_text.strip().lower() == "quit":
            break

        if user_text.strip().lower() == "history":
            for m in messages:
                print(f"{m['role']}: {m['content']}")
            continue

        messages.append({"role": "user", "content": user_text})
        reply = fake_reply(messages)
        messages.append({"role": "assistant", "content": reply})
        print("Bot:", reply)


main()