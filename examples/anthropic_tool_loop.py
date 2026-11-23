"""agent-tool-result-reducer in an Anthropic Messages API tool loop.

    pip install agent-tool-result-reducer anthropic
    export ANTHROPIC_API_KEY=...
    python examples/anthropic_tool_loop.py

The reducer sits in one place: between running the tool and appending its
`tool_result` block. Everything else is a standard manual tool loop.
"""

import json
import os

import anthropic

from demo_tool import TOOL_DESCRIPTION, TOOL_NAME, TOOL_PARAMETERS, search_companies
from tool_result_reducer import reduce_tool_result

BUDGET = 8_000  # characters per tool result sent to the model
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5")

client = anthropic.Anthropic()
tools = [{"name": TOOL_NAME, "description": TOOL_DESCRIPTION, "input_schema": TOOL_PARAMETERS}]
messages = [
    {
        "role": "user",
        "content": "Find logistics companies and give me a table of every company's name, city and employees.",
    }
]

while True:
    response = client.messages.create(model=MODEL, max_tokens=16_000, tools=tools, messages=messages)
    if response.stop_reason == "refusal":
        raise SystemExit(f"The model declined: {response.stop_details}")
    messages.append({"role": "assistant", "content": response.content})
    if response.stop_reason != "tool_use":
        break

    tool_results = []
    for block in response.content:
        if block.type != "tool_use":
            continue
        raw = search_companies(**block.input)

        # The one change: reduce before the result enters the conversation.
        result = reduce_tool_result(raw, block.name, select=block.input.get("select"), budget=BUDGET)
        print(result.summary(block.name), f"({len(result.text):,} of {len(json.dumps(raw)):,} characters)")

        tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result.text})
    messages.append({"role": "user", "content": tool_results})

print()
print("".join(block.text for block in response.content if block.type == "text"))
