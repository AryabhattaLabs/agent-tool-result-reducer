import json

from tool_result_reducer import Messages, reduce_tool_result


def companies(n, desc_len=2000):
    return [
        {
            "id": i,
            "name": f"Company {i}",
            "summary": {"name": f"Co {i}", "industry": "software"},
            "description": "x" * desc_len,
        }
        for i in range(n)
    ]


def test_small_result_is_untouched():
    payload = {"items": [{"id": 1}]}
    r = reduce_tool_result(payload)
    assert r.complete
    assert json.loads(r.text) == payload


def test_never_cuts_mid_object_and_reports_counts():
    # Few fields per item, so auto-fit cannot help and items must be dropped.
    payload = {"results": [{"id": i, "blob": "y" * 3000} for i in range(11)], "total": 443}
    r = reduce_tool_result(json.dumps(payload), "search", budget=12_000)
    out = json.loads(r.text)  # always valid JSON
    meta = out["_meta"]
    assert not r.complete and meta["complete"] is False
    assert meta["items_included"] == len(out["items"]) < 11
    assert meta["items_available"] == 11
    assert meta["total_reported_by_source"] == 443
    assert "warning" in meta and "field_costs" in meta
    assert len(r.text) <= 12_000


def test_select_lets_every_item_through():
    r = reduce_tool_result({"data": companies(11)}, select="id,summary.name")
    out = json.loads(r.text)
    assert r.complete
    assert out["items"][0] == {"id": 0, "name": "Co 0"}
    assert out["_meta"]["source_path"] == "data"


def test_auto_fit_drops_widest_field_before_dropping_items():
    r = reduce_tool_result({"items": companies(11)}, budget=12_000)
    assert r.complete
    assert "description" in r.fields_dropped
    meta = json.loads(r.text)["_meta"]
    assert meta["items_included"] == 11
    assert "fields_dropped_note" in meta


def test_select_that_matches_nothing_is_not_served_as_success():
    r = reduce_tool_result({"items": companies(3, 10)}, select="nope")
    assert r.select_error and "id" in r.select_error
    assert json.loads(r.text)["items"][0]["id"] == 0


def test_null_valued_field_counts_as_a_match():
    items = [{"id": i, "stage": None} for i in range(3)]
    r = reduce_tool_result({"items": items}, select="stage")
    assert r.select_error is None
    assert json.loads(r.text)["items"] == [{"stage": None}] * 3


def test_envelope_prefix_in_select_is_repaired():
    r = reduce_tool_result({"items": companies(3, 10)}, select="items.id,items.name")
    meta = json.loads(r.text)["_meta"]
    assert "select_prefix_stripped" in meta
    assert json.loads(r.text)["items"][0] == {"id": 0, "name": "Company 0"}


def test_jsonapi_prefers_data_over_longer_included():
    payload = {
        "data": [{"id": "1", "type": "co"}],
        "included": [{"id": str(i), "type": "rel"} for i in range(50)],
    }
    r = reduce_tool_result(payload, select="id")
    assert json.loads(r.text)["_meta"]["source_path"] == "data"


def test_large_single_object_becomes_parseable_outline():
    payload = {"title": "t", "body": "z" * 50_000, "nested": {"a": list(range(1000))}}
    r = reduce_tool_result(payload, budget=4_000)
    out = json.loads(r.text)
    assert r.outlined and not r.complete
    assert set(out["outline"]) == {"title", "body", "nested"}


def test_non_json_falls_back_to_char_cut():
    r = reduce_tool_result("plain text " * 2000, "fetch", budget=1_000)
    assert r.fallback_char_cut and not r.complete
    assert "truncated" in r.text


def test_messages_are_configurable():
    msgs = Messages(incomplete="CUSTOM WARNING")
    payload = {"items": [{"id": i, "blob": "y" * 3000} for i in range(10)]}
    r = reduce_tool_result(payload, budget=5_000, messages=msgs)
    assert json.loads(r.text)["_meta"]["warning"] == "CUSTOM WARNING"


def test_summary_line():
    payload = {"items": [{"id": i, "blob": "y" * 3000} for i in range(10)]}
    r = reduce_tool_result(payload, "crm.search", budget=5_000)
    assert r.summary("crm.search").startswith("crm.search: INCOMPLETE")


def test_summary_counts_items_when_select_fits():
    payload = {"results": companies(30), "total": 412}
    r = reduce_tool_result(payload, "crm.search", select="id,name", budget=8_000)
    assert r.complete and not r.fields_dropped
    assert r.summary("crm.search") == (
        "crm.search: complete, agent received 30 of 30 items "
        "(source reports 412 in total), selected fields: id, name"
    )


def test_meta_explains_items_beyond_this_response():
    # A page of 30 from a source reporting 412: complete, but not the full set.
    payload = {"results": companies(30), "total": 412}
    meta = json.loads(reduce_tool_result(payload, select="id,name", budget=8_000).text)["_meta"]
    assert meta["complete"] is True
    assert "412 in total" in meta["total_note"] and "full set" in meta["total_note"]

    # No note when the source's total matches what was returned.
    meta = json.loads(reduce_tool_result({"results": companies(30), "total": 30}, select="id").text)["_meta"]
    assert "total_note" not in meta


def test_summary_for_pass_through():
    r = reduce_tool_result({"ok": True}, "ping")
    assert r.summary("ping") == f"ping: complete ({len(r.text):,} chars)"


# ── invariants, checked over many random payloads ──────────────────────


def _random_payload(rng):
    def text(n):
        return "".join(rng.choice("abcdefghij klmnop") for _ in range(n))

    def value(depth=0):
        roll = rng.random()
        if roll < 0.4:
            return text(rng.choice([3, 20, 200, 2000, 9000]))
        if roll < 0.55:
            return rng.randint(-1000, 1000)
        if roll < 0.6:
            return None
        if roll < 0.8 and depth < 2:
            return {text(4).strip() or "k": value(depth + 1) for _ in range(rng.randint(0, 4))}
        if depth < 2:
            return [value(depth + 1) for _ in range(rng.randint(0, 3))]
        return text(8)

    keys = ["id", "name", "description", "summary", "tags", "notes", "score"]
    items = [
        {k: value() for k in rng.sample(keys, rng.randint(1, len(keys)))}
        for _ in range(rng.randint(0, 40))
    ]
    shape = rng.choice(["list", "items", "nested", "paged"])
    if shape == "list":
        return items
    if shape == "items":
        return {"items": items}
    if shape == "nested":
        return {"data": {"results": items}, "errors": []}
    return {"content": items, "totalElements": len(items) + rng.randint(0, 500)}


def test_invariants_hold_for_random_payloads():
    import random

    rng = random.Random(20261122)
    for _ in range(120):
        payload = _random_payload(rng)
        budget = rng.choice([1_500, 4_000, 12_000, 40_000])
        select = rng.choice([None, None, "id", "id,name", "summary.name,id", "nope", "items.id"])
        raw = json.dumps(payload) if rng.random() < 0.5 else payload
        r = reduce_tool_result(raw, "t", select=select, budget=budget)

        out = json.loads(r.text)  # always valid JSON
        if r.items_included is None:
            continue
        if isinstance(out, dict) and "_meta" in out:
            assert out["_meta"]["items_included"] == len(out["items"]) == r.items_included
            assert out["_meta"]["complete"] is r.complete
        # The budget holds whenever more than one item is kept. (The one
        # exception, a first item that alone exceeds it, has its own test.)
        assert len(r.text) <= budget or r.items_included <= 1, (budget, len(r.text), r.items_included)


def test_budget_is_met_when_the_refit_passes_fall_short():
    # Seeds where three refit passes used to leave the text a few characters
    # over the budget with several items kept (12,031 / 12,014 / 12,010).
    import random

    for seed in (1, 2, 5):
        rng = random.Random(seed)
        for _ in range(60):
            payload = _random_payload(rng)
            select = rng.choice([None, "items.id,items.name"])
            r = reduce_tool_result(payload, budget=12_000, select=select)
            if r.items_included and r.items_included > 1:
                assert len(r.text) <= 12_000, (seed, len(r.text), r.items_included)


def test_max_overshoot_keeps_the_item_strict_mode_would_drop():
    # Seeds with payloads that land just over the budget. Strict mode drops
    # one item to get back under it; an allowance of 200 characters keeps it.
    import random

    differed = 0
    for seed in (7, 9):
        rng = random.Random(seed)
        for _ in range(60):
            payload = _random_payload(rng)
            select = rng.choice([None, "items.id,items.name"])
            strict = reduce_tool_result(payload, budget=12_000, select=select)
            lenient = reduce_tool_result(payload, budget=12_000, select=select, max_overshoot=200)
            if strict.items_included == lenient.items_included:
                assert strict.text == lenient.text
                continue
            differed += 1
            assert lenient.items_included == strict.items_included + 1
            assert len(strict.text) <= 12_000 < len(lenient.text) <= 12_200
            json.loads(lenient.text)
    # How many random payloads land within a few characters of the budget
    # depends on the exact size of `_meta`, so require the behaviour, not a count.
    assert differed >= 1


def test_items_are_not_dropped_when_the_first_one_alone_is_over_budget():
    # With a long runtime-specific warning, the first item plus the _meta block
    # is already over the budget. Dropping the second item cannot bring the
    # result back under it, so it is kept.
    import random

    long_warning = Messages(incomplete="Read the stored response instead. " * 20)
    rng = random.Random(28)
    for _ in range(11):
        payload = _random_payload(rng)
        select = rng.choice([None, "items.id,items.name"])
    r = reduce_tool_result(payload, budget=12_000, select=select, messages=long_warning)
    assert r.items_included == 2 and r.items_available == 28
    assert 12_000 < len(r.text) < 12_100
    assert json.loads(r.text)["_meta"]["items_included"] == 2


# ── text that is not ASCII ─────────────────────────────────────────────


def test_non_ascii_text_is_measured_and_returned_as_written():
    items = [{"id": i, "name": "आर्यभट्ट लैब्स", "note": "नमस्ते दुनिया " * 20} for i in range(20)]
    as_object = reduce_tool_result({"items": items}, budget=12_000)
    as_string = reduce_tool_result(json.dumps({"items": items}, ensure_ascii=False), budget=12_000)
    for r in (as_object, as_string):
        assert r.complete and not r.fields_dropped  # ~6,500 real characters fit easily
        assert "नमस्ते" in r.text and "\\u09" not in r.text
    assert abs(as_object.chars_raw - as_string.chars_raw) < 200  # same payload, same measure


def test_non_ascii_text_survives_reduction():
    items = [{"id": i, "title": "東京 " * 400, "blob": "y" * 3000} for i in range(12)]
    r = reduce_tool_result({"items": items}, budget=6_000, select="id,title")
    out = json.loads(r.text)
    assert "東京" in r.text and out["_meta"]["items_included"] == len(out["items"])


def test_lone_surrogate_falls_back_to_escapes_instead_of_breaking_utf8():
    items = [{"id": i, "note": "broken \ud83d emoji", "blob": "y" * 3000} for i in range(10)]
    r = reduce_tool_result({"items": items}, budget=5_000)
    r.text.encode("utf-8")  # must not raise
    r.raw.encode("utf-8")
    assert json.loads(r.text)["items"][0]["id"] == 0


# ── totals ─────────────────────────────────────────────────────────────


def test_page_size_is_not_reported_as_a_total():
    # A Spring "Slice": page size 20, 11 items returned, no total anywhere.
    payload = {"content": [{"id": i, "blob": "y" * 3000} for i in range(11)], "size": 20, "number": 0}
    r = reduce_tool_result(payload, budget=12_000)
    assert r.total_reported is None
    assert "total_reported_by_source" not in json.loads(r.text)["_meta"]


def test_nested_page_total_is_still_found():
    payload = {
        "_embedded": {"companies": [{"id": i, "blob": "y" * 3000} for i in range(11)]},
        "page": {"size": 20, "totalElements": 443, "number": 0},
    }
    r = reduce_tool_result(payload, budget=12_000)
    assert r.total_reported == 443


# ── pagination fields (issue #9) ───────────────────────────────────────


def test_cursor_reaches_the_model_when_the_whole_page_is_included():
    payload = {"results": companies(11), "total": 443, "next_cursor": "abc123", "has_more": True}
    r = reduce_tool_result(payload, budget=12_000)
    meta = json.loads(r.text)["_meta"]
    assert r.complete and meta["items_included"] == 11
    assert meta["source_paging"] == {"next_cursor": "abc123", "has_more": True}
    assert "next page" in meta["paging_note"]
    assert r.paging == {"next_cursor": "abc123", "has_more": True}
    assert len(r.text) <= 12_000


def test_cursor_is_withheld_from_the_model_when_part_of_the_page_was_dropped():
    # The fixture from the feedback thread: following the cursor here would
    # skip B and C for good, so only the caller gets it.
    payload = {
        "results": [{"id": "A", "text": "a"}, {"id": "B", "text": "x" * 5000}, {"id": "C", "text": "c"}],
        "total": 12,
        "next_cursor": "abc123",
    }
    r = reduce_tool_result(payload, budget=1800)
    meta = json.loads(r.text)["_meta"]
    assert not r.complete and meta["items_included"] == 1
    assert "source_paging" not in meta and "abc123" not in r.text
    assert r.paging == {"next_cursor": "abc123"}


def test_nested_paging_fields_keep_their_path_and_null_means_last_page():
    payload = {
        "data": {"items": companies(11), "pagination": {"next_page_token": None, "limit": 11}},
        "links": {"next": "https://api.example.com/items?page=2"},
    }
    r = reduce_tool_result(payload, budget=12_000)
    assert json.loads(r.text)["_meta"]["source_paging"] == {
        "data.pagination.next_page_token": None,
        "data.pagination.limit": 11,
        "links.next": "https://api.example.com/items?page=2",
    }


def _page(n=20):
    return [{"id": i, "name": f"n{i}", "bio": "x" * 900} for i in range(n)]


# How real APIs return pagination state, and what must reach the model.
REAL_API_PAGING = {
    "slack": ({"ok": True, "members": _page(), "response_metadata": {"next_cursor": "dXNlcjpV"}},
              {"response_metadata.next_cursor": "dXNlcjpV"}),
    "notion": ({"object": "list", "results": _page(), "next_cursor": "fe2c", "has_more": True},
               {"next_cursor": "fe2c", "has_more": True}),
    "stripe": ({"object": "list", "data": _page(), "has_more": True, "url": "/v1/customers"}, {"has_more": True}),
    "hubspot": ({"results": _page(), "paging": {"next": {"after": "NTI1", "link": "https://h/x?after=NTI1"}}},
                {"paging.next.after": "NTI1"}),
    "google": ({"files": _page(), "nextPageToken": "CkQK"}, {"nextPageToken": "CkQK"}),
    "airtable": ({"records": _page(), "offset": "itr/rec"}, {"offset": "itr/rec"}),
    "drf": ({"count": 443, "next": "https://x/?page=2", "previous": None, "results": _page()},
            {"next": "https://x/?page=2"}),
    "salesforce": ({"totalSize": 443, "done": False, "nextRecordsUrl": "/query/01gD-2000", "records": _page()},
                   {"done": False, "nextRecordsUrl": "/query/01gD-2000"}),
    "graphql": ({"data": {"projects": {"nodes": _page(), "pageInfo": {"hasNextPage": True, "endCursor": "eyJp"}}}},
                {"data.projects.pageInfo.hasNextPage": True, "data.projects.pageInfo.endCursor": "eyJp"}),
    "elasticsearch": ({"_scroll_id": "FGlu", "hits": {"total": {"value": 443}, "hits": _page()}},
                      {"_scroll_id": "FGlu"}),
}


def test_paging_fields_of_real_api_shapes_reach_the_model():
    for name, (payload, expected) in REAL_API_PAGING.items():
        r = reduce_tool_result(payload, budget=20_000, select="id,name")
        assert r.complete, name
        assert json.loads(r.text)["_meta"]["source_paging"] == expected, name


def test_envelope_fields_that_are_not_paging_stay_out():
    payload = {"access_token": "SECRET", "request_id": "r1", "results": _page()}
    r = reduce_tool_result(payload, budget=8_000)
    assert "SECRET" not in r.text and r.paging == {}
    # Fields inside the items are never read as the envelope's paging.
    items = [{"id": i, "next_step": "call", "page": 3, "bio": "x" * 900} for i in range(20)]
    assert reduce_tool_result({"results": items}, budget=8_000).paging == {}


def test_a_very_long_cursor_is_left_to_the_caller():
    long_cursor = "c" * 2000
    payload = {"results": companies(11), "next_cursor": long_cursor, "has_more": True}
    r = reduce_tool_result(payload, budget=12_000)
    assert json.loads(r.text)["_meta"]["source_paging"] == {"has_more": True}
    assert r.paging["next_cursor"] == long_cursor


def test_no_paging_block_without_paging_fields():
    r = reduce_tool_result({"results": companies(11), "total": 443}, budget=12_000)
    assert "source_paging" not in json.loads(r.text)["_meta"] and r.paging == {}


# ── building blocks used outside reduce_tool_result ────────────────────


def test_locate_collection_returns_items_path_and_total():
    from tool_result_reducer import locate_collection

    rows = [{"id": 1}, {"id": 2}]
    assert locate_collection({"data": {"results": rows}, "total": 443}) == (rows, "data.results", 443)
    assert locate_collection(rows) == (rows, None, None)
    assert locate_collection({"title": "one object"}) == (None, None, None)
    assert locate_collection("not json") == (None, None, None)


def test_resolve_tells_absent_from_null():
    from tool_result_reducer import MISSING, resolve

    item = {"id": 7, "summary": {"name": "Acme", "stage": None}, "tags": ["a", "b"]}
    assert resolve(item, "summary.name") == "Acme"
    assert resolve(item, "summary.stage") is None
    assert resolve(item, "summary.nope") is MISSING
    assert resolve(item, "tags.1") == "b"
    assert resolve(item, "tags.9") is MISSING


def test_available_paths_for_one_item():
    from tool_result_reducer import available_paths

    item = {"id": 1, "summary": {"name": "x", "deep": {"a": 1}}, "tags": ["a"]}
    assert available_paths(item) == ["id", "summary.name", "summary.deep", "tags"]
    assert available_paths(item, depth=1) == ["id", "summary", "tags"]
    assert available_paths("scalar") == []
