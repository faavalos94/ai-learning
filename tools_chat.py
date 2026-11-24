import json
from datetime import datetime

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()

TOOLS = [
    {
        "name": "get_current_datetime",
        "description": "Get the current local date and time.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "calculate",
        "description": "Do exact arithmetic on two numbers. Always use this for math instead of calculating yourself.",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                },
            },
            "required": ["a", "b", "operation"],
        },
    },
]


def get_current_datetime():
    return datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")


def calculate(a, b, operation):
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        if b == 0:
            return "Error: division by zero"
        return a / b
    return f"Error: unknown operation {operation}"


def run_tool(name, tool_input):
    if name == "get_current_datetime":
        return get_current_datetime()
    if name == "calculate":
        return calculate(tool_input["a"], tool_input["b"], tool_input["operation"])
    return f"Error: unknown tool {name}"


def get_reply(messages):
    while True:
        response = client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=4000,
            system="You are a helpful assistant. Use tools whenever they help.",
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return next((block.text for block in response.content if block.type == "text"), "")

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [tool call: {block.name}({json.dumps(block.input)})]")
                result = run_tool(block.name, block.input)
                print(f"  [tool result: {result}]")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    }
                )
        messages.append({"role": "user", "content": tool_results})


def main():
    messages = []
    print("Type 'quit' to exit.")

    while True:
        user_text = input("You: ")
        if user_text.strip().lower() == "quit":
            break

        messages.append({"role": "user", "content": user_text})
        reply = get_reply(messages)
        print("Bot:", reply)


if __name__ == "__main__":
    main()
