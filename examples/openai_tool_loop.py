"""agent-tool-result-reducer in an OpenAI Responses API tool loop.

    pip install agent-tool-result-reducer openai
    export OPENAI_API_KEY=...
    python examples/openai_tool_loop.py

The reducer sits in one place: between running the tool and appending its
`function_call_output`. Everything else is a standard tool loop. (Reasoning
models such as gpt-6-luna take function tools on the Responses API, not on
Chat Completions.)
"""

import json
import os

from openai import OpenAI

from demo_tool import TOOL_DESCRIPTION, TOOL_NAME, TOOL_PARAMETERS, search_companies
from tool_result_reducer import reduce_tool_result

BUDGET = 8_000  # characters per tool result sent to the model
MODEL = os.environ.get("OPENAI_MODEL", "gpt-6-luna")

client = OpenAI()
tools = [{"type": "function", "name": TOOL_NAME, "description": TOOL_DESCRIPTION, "parameters": TOOL_PARAMETERS}]
conversation = [
    {
        "role": "user",
        "content": "Find logistics companies and give me a table of every company's name, city and employees.",
    }
]

while True:
    response = client.responses.create(model=MODEL, input=conversation, tools=tools)
    conversation += response.output  # keeps reasoning items alongside the calls
    calls = [item for item in response.output if item.type == "function_call"]
    if not calls:
        break

    for call in calls:
        args = json.loads(call.arguments)
        raw = search_companies(**args)

        # The one change: reduce before the result enters the conversation.
        result = reduce_tool_result(raw, call.name, select=args.get("select"), budget=BUDGET)
        print(result.summary(call.name), f"({len(result.text):,} of {len(json.dumps(raw)):,} characters)")

        conversation.append({"type": "function_call_output", "call_id": call.call_id, "output": result.text})

print()
print(response.output_text)
