import json

from anthropic import Anthropic
from dotenv import load_dotenv

from rag import load_chunks
from rag_embed import embed_chunks, retrieve
from tools_chat import TOOLS as BASIC_TOOLS
from tools_chat import run_tool as run_basic_tool

load_dotenv()
client = Anthropic()

MAX_STEPS = 8

SEARCH_TOOL = {
    "name": "search_handbook",
    "description": (
        "Search the Acme Labs employee handbook. It covers paid time off, sick leave, "
        "remote work, equipment, expenses, and the professional development budget. "
        "Returns the most relevant sections. Use a short, specific query. "
        "If the results don't answer the question, try again with different wording."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}

TOOLS = BASIC_TOOLS + [SEARCH_TOOL]

SYSTEM_PROMPT = (
    "You are an HR assistant for Acme Labs employees. "
    "Answer policy questions only from the handbook, using the search_handbook tool. Never guess a policy. "
    "Use calculate for any arithmetic and get_current_datetime for anything involving today's date. "
    "Say which handbook section your answer came from. "
    "If the handbook doesn't cover something, say so."
)


def search_handbook(query, chunks):
    results = retrieve(query, chunks)
    return "\n\n".join(f"### {c['title']}\n{c['text']}" for c in results)


def run_tool(name, tool_input, chunks):
    if name == "search_handbook":
        return search_handbook(tool_input["query"], chunks)
    return run_basic_tool(name, tool_input)


def run_agent(messages, chunks):
    tool_calls = []

    for step in range(MAX_STEPS):
        response = client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            answer = next((b.text for b in response.content if b.type == "text"), "")
            return answer, tool_calls

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [step {step + 1}] {block.name}({json.dumps(block.input)})")
                tool_calls.append(block.name)
                result = run_tool(block.name, block.input, chunks)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    }
                )
        messages.append({"role": "user", "content": tool_results})

    return f"Stopped after {MAX_STEPS} steps without a final answer.", tool_calls


def main():
    chunks = load_chunks("handbook.txt")
    embed_chunks(chunks)
    messages = []
    print("Type 'quit' to exit.")

    while True:
        user_text = input("\nYou: ")
        if user_text.strip().lower() == "quit":
            break

        messages.append({"role": "user", "content": user_text})
        answer, tool_calls = run_agent(messages, chunks)
        print("Agent:", answer)


if __name__ == "__main__":
    main()
