"""A stand-in search tool shared by the agent-loop examples.

`search_companies` returns what many real APIs return: a page of 30 records,
each with a long description and a list of news items, about 110,000
characters in all. That is far more than an agent loop should re-send to the
model on every turn, which is what the examples show the reducer handling.

Replace it with your own tool; nothing else in the examples depends on it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

INDUSTRIES = ["software", "logistics", "fintech", "healthcare", "retail", "energy"]
CITIES = ["Toronto", "Bengaluru", "Austin", "Berlin", "Singapore", "Lagos"]

TOOL_NAME = "search_companies"
TOOL_DESCRIPTION = (
    "Search the company directory. Returns one page of matching companies. "
    "Each company has: id, name, industry, city, employees, founded, website, "
    "description, news. Large results are reduced to fit; the `_meta` block "
    "in the result says what was left out and how to get it."
)
# JSON Schema for the tool's input, shared by every example.
TOOL_PARAMETERS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "What to search for."},
        "select": {
            "type": "string",
            "description": (
                "Optional. Comma-separated fields to keep from each company, "
                "e.g. id,name,industry. Omit to get all fields."
            ),
        },
    },
    "required": ["query"],
}


def search_companies(query: str, select: Optional[str] = None) -> Dict[str, Any]:
    """Return one page of 30 companies. `select` is handled by the reducer."""
    companies = [
        {
            "id": i,
            "name": f"{query.title()} Company {i}",
            "industry": INDUSTRIES[i % len(INDUSTRIES)],
            "city": CITIES[i % len(CITIES)],
            "employees": 40 + 17 * i,
            "founded": 1995 + i % 28,
            "website": f"https://company{i}.example.com",
            "description": f"Company {i} builds products for its customers. " * 45,
            "news": [
                {"date": f"2026-0{1 + j}-15", "headline": f"Company {i} update {j}", "body": "Details. " * 30}
                for j in range(4)
            ],
        }
        for i in range(30)
    ]
    return {"results": companies, "page": 1, "per_page": 30, "total": 412}
