"""Run: python examples/basic.py"""

import json

from tool_result_reducer import reduce_tool_result

# A typical paginated API response: 25 records, each with a long description.
response = {
    "content": [
        {
            "id": i,
            "name": f"Company {i}",
            "website": f"https://example{i}.com",
            "summary": {"industry": "software", "employees": 50 + i},
            "description": "Lorem ipsum " * 300,
        }
        for i in range(25)
    ],
    "page": {"totalElements": 443, "number": 0, "size": 25},
}

# 1) No select: the widest field is dropped so every item survives.
r = reduce_tool_result(response, "companies.search", budget=8_000)
print(r.summary("companies.search"))
print(json.dumps(json.loads(r.text)["_meta"], indent=2)[:600], "\n")

# 2) With select: only the fields the agent needs, all 25 items.
r = reduce_tool_result(response, "companies.search", select="id,name,summary.industry")
print(r.summary("companies.search"))
print(r.text[:300])
