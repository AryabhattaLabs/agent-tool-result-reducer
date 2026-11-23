"""agent-tool-result-reducer in an OpenAI Chat Completions tool loop.

    pip install agent-tool-result-reducer openai
    export OPENAI_API_KEY=...
    python examples/openai_tool_loop.py

The reducer sits in one place: between running the tool and appending its
result to `messages`. Everything else is a standard tool loop.
"""

import json
import os

from openai import OpenAI

from demo_tool import TOOL_DESCRIPTION, TOOL_NAME, TOOL_PARAMETERS, search_companies
from tool_result_reducer import reduce_tool_result

BUDGET = 8_000  # characters per tool result sent to the model
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")

client = OpenAI()
tools = [
    {
        "type": "function",
        "function": {"name": TOOL_NAME, "description": TOOL_DESCRIPTION, "parameters": TOOL_PARAMETERS},
    }
]
messages = [
    {
        "role": "user",
        "content": "Find logistics companies and give me a table of every company's name, city and employees.",
    }
]

while True:
    response = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    message = response.choices[0].message
    messages.append(message)
    if not message.tool_calls:
        break

    for call in message.tool_calls:
        args = json.loads(call.function.arguments)
        raw = search_companies(**args)

        # The one change: reduce before the result enters the conversation.
        result = reduce_tool_result(raw, call.function.name, select=args.get("select"), budget=BUDGET)
        print(result.summary(call.function.name), f"({len(result.text):,} of {len(json.dumps(raw)):,} characters)")

        messages.append({"role": "tool", "tool_call_id": call.id, "content": result.text})

print()
print(message.content)
