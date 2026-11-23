"""agent-tool-result-reducer in a LangChain tool.

    pip install agent-tool-result-reducer langchain langchain-openai
    export OPENAI_API_KEY=...
    python examples/langchain_tool.py

In LangChain the tool's return value is what the model sees, so the reducer
goes inside the tool. Any chat model and any agent built on LangChain tools
works the same way; `create_agent` with OpenAI is only the demo.
"""

import json
import os
from typing import Optional

from langchain.agents import create_agent
from langchain_core.tools import tool

from demo_tool import TOOL_DESCRIPTION, TOOL_NAME, search_companies
from tool_result_reducer import reduce_tool_result

BUDGET = 8_000  # characters per tool result sent to the model
MODEL = os.environ.get("LANGCHAIN_MODEL", "openai:gpt-6-luna")


@tool(TOOL_NAME, description=TOOL_DESCRIPTION)
def search_companies_tool(query: str, select: Optional[str] = None) -> str:
    """Search the company directory.

    Args:
        query: What to search for.
        select: Optional comma-separated fields to keep from each company, e.g. id,name,industry.
    """
    raw = search_companies(query)

    # The one change: return the reduced text, not the raw payload.
    result = reduce_tool_result(raw, TOOL_NAME, select=select, budget=BUDGET)
    print(result.summary(TOOL_NAME), f"({len(result.text):,} of {len(json.dumps(raw)):,} characters)")
    return result.text


agent = create_agent(MODEL, tools=[search_companies_tool])
state = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Find logistics companies and give me a table of every company's name, city and employees.",
            }
        ]
    }
)

print()
print(state["messages"][-1].content)
