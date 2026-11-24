from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()          # loads ANTHROPIC_API_KEY from .env into the environment
client = Anthropic()   # automatically reads ANTHROPIC_API_KEY


def llm_reply(messages):
    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=500,
        system="You are a terse pirate. Answer in one sentence.",
        messages=messages,
    )
    print(f"  [tokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out]")
    print(f"  [stop reason: {response.stop_reason}]")

    if response.stop_reason == "max_tokens":
        raise RuntimeError("Reply was cut off by max_tokens. Increase max_tokens and try again.")
    
    text = next(block.text for block in response.content if block.type == "text")
    return text

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
        reply = llm_reply(messages)
        messages.append({"role": "assistant", "content": reply})
        print("Bot:", reply)


main()