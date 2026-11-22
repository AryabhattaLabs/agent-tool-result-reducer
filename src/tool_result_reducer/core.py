"""Bound a tool result by what the agent consumes, not by raw byte count.

A tool result is re-sent to the model on every later turn of an agent loop,
so an unbounded one is billed many times over. The common fix is a flat
character cap: keep the first N characters, drop the rest.

That bounds the wrong thing. In production, an API returned 11 companies in
77,539 characters; a 12,000-character cap cut mid-object
(``..."name": "content delivery networ``). The model parsed the two whole
records it could see, acted on those, and reported success. Nothing said
that 65,539 characters had been dropped.

This module fixes that class of failure:

1. **Project before bounding.** If the caller names the fields it needs
   (``select``), keep only those. Eleven records x 14 fields fit the same
   budget several times over.
2. **Drop fields before items.** When everything still does not fit, drop the
   widest fields across all items (measured, not guessed) so every item
   survives, and say which fields went.
3. **Cut on item boundaries and say so.** What survives is always valid JSON,
   wrapped in a ``_meta`` block stating how many items were included out of
   how many exist. A partial result can no longer pass for a complete one.
4. **Outline, don't truncate, large single objects.** Every top-level key
   stays visible and the text stays parseable.

Keep ``ReducedResult.raw`` wherever you store full payloads; only
``ReducedResult.text`` should enter the model's context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "DEFAULT_RESULT_BUDGET",
    "MISSING",
    "Messages",
    "ReducedResult",
    "available_paths",
    "available_paths_for",
    "locate_collection",
    "parse_select",
    "project",
    "projection_matched_nothing",
    "reduce_tool_result",
    "repair_envelope_prefix",
    "resolve",
]

# Default per-result budget in characters.
DEFAULT_RESULT_BUDGET = 12_000

# Room reserved inside the budget for the `_meta` envelope. It is an
# estimate; the main loop re-fits when the finished text overshoots.
_ENVELOPE_RESERVE = 700

# Keys that conventionally hold the collection in an API envelope. The
# longest list wins regardless; this set lets the search descend one level
# deeper through these names.
_COLLECTION_KEYS = (
    "content",
    "data",
    "items",
    "results",
    "records",
    "rows",
    "elements",
    "entries",
    "values",
    "list",
)

# Keys that conventionally report how many items exist server-side. This is
# what lets the note say "25 of 443" rather than just "25". "size" is left
# out on purpose: in paginated responses it is the page size, and reading it
# as a total tells the model "20 exist" when 11 were returned and more may.
_TOTAL_KEYS = (
    "total",
    "totalElements",
    "totalCount",
    "total_count",
    "totalResults",
    "total_results",
    "count",
)

# How deep to look for the collection. Three levels covers
# {"data": {"result": {"items": [...]}}} without wandering into item bodies.
_MAX_SEARCH_DEPTH = 3

# Auto-fit never trims below this many fields per item. Past that point the
# problem is item count, not field width, and the item-boundary cut with its
# explicit "incomplete" accounting is the better failure.
_AUTO_FIT_MIN_FIELDS = 2

# How much of a long string survives in a structure outline.
_OUTLINE_VALUE_CHARS = 200


@dataclass(frozen=True)
class Messages:
    """Text the reducer writes into results for the model to read.

    The defaults assume only that the model can call the same tool again with
    a ``select`` argument. If your runtime can answer a re-select from a
    cached response, or offers a tool to query the stored full payload, say
    so here: models follow these instructions closely, and a vague hint like
    "narrow the request" can send them back to a metered API repeatedly.

    ``fields_dropped`` and ``outline`` are format strings; see their
    placeholders below.
    """

    incomplete: str = (
        "You did NOT receive all items. Do not describe this result as "
        "complete, and do not report the items you can see as if they were "
        "the whole set. To get every item, call this tool again with the SAME "
        "arguments plus `select=` naming only the fields you need (dot-paths "
        "relative to one item, e.g. id,summary.name)."
    )
    # Placeholders: {count} items kept, {fields} comma-separated names.
    fields_dropped: str = (
        "All {count} items are included, but these fields were dropped to "
        "fit: {fields}. If a dropped field is needed, call this tool again "
        "with the SAME arguments but set `select=` to ONLY an id field plus "
        "the dropped field(s)."
    )
    # Placeholder: {chars} original size, formatted with thousands separators.
    outline: str = (
        "This response is a single large object ({chars} chars), shown as a "
        "structural outline: long strings are cut, nested objects/lists are "
        "summarised as {{_type, _keys/_len, _first}}. Every key that exists "
        "is listed."
    )
    # Placeholders: {dropped} characters cut, {tool} tool name.
    char_cut: str = (
        "\n\n... [truncated {dropped:,} characters of this {tool} result. "
        "Narrow the request (filters, a smaller page size, fewer fields) if "
        "you need the part that was cut.]"
    )


DEFAULT_MESSAGES = Messages()


@dataclass
class ReducedResult:
    """What the model gets, plus everything needed to explain it."""

    text: str
    raw: str
    complete: bool = True
    chars_raw: int = 0
    chars_kept: int = 0
    items_included: Optional[int] = None
    items_available: Optional[int] = None
    total_reported: Optional[int] = None
    fields_kept: List[str] = field(default_factory=list)
    source_path: Optional[str] = None
    # Set when a supplied `select` matched no field on any item, so the
    # projection was discarded rather than served as an empty result.
    select_error: Optional[str] = None
    # Set when the payload was not JSON and we fell back to cutting
    # characters: the one path that can hand the model broken syntax.
    fallback_char_cut: bool = False
    # Set when a large single object was replaced by a structure outline.
    outlined: bool = False
    # field -> approx chars/item removed by auto-fit. Non-empty means every
    # item IS included (complete=True) but these fields were dropped.
    fields_dropped: Dict[str, int] = field(default_factory=dict)

    @property
    def items_dropped(self) -> int:
        if self.items_available is None or self.items_included is None:
            return 0
        return max(0, self.items_available - self.items_included)

    @property
    def chars_dropped(self) -> int:
        return max(0, self.chars_raw - self.chars_kept)

    def log_fields(self) -> Dict[str, Any]:
        """Structured fields for a log/metrics event."""
        return {
            "chars_raw": self.chars_raw,
            "chars_kept": self.chars_kept,
            "items_included": self.items_included,
            "items_available": self.items_available,
            "total_reported": self.total_reported,
            "complete": self.complete,
            "fallback_char_cut": self.fallback_char_cut or None,
            "outlined": self.outlined or None,
            "fields_dropped": len(self.fields_dropped) or None,
        }

    def summary(self, tool_name: str = "tool") -> str:
        """One line for humans (logs, run UIs) describing what was lost."""
        if self.complete and not self.fields_dropped:
            return f"{tool_name}: complete ({self.chars_kept:,} chars)"
        if self.items_available is not None and self.items_included is not None:
            scope = f"{self.items_included} of {self.items_available} items"
            if self.total_reported and self.total_reported > self.items_available:
                scope += f" (source reports {self.total_reported:,} in total)"
            if self.fields_dropped:
                scope += f", dropped fields: {', '.join(sorted(self.fields_dropped))}"
        else:
            scope = f"{self.chars_kept:,} of {self.chars_raw:,} characters"
        state = "complete" if self.complete else "INCOMPLETE"
        return f"{tool_name}: {state}, agent received {scope}"


# ── select / projection ────────────────────────────────────────────────


def parse_select(select: Any) -> List[str]:
    """Normalise a `select` argument into a list of dot-paths.

    Accepts a comma-separated string (what a model passes) or a list (what a
    config holds). Empty/absent means "no projection".
    """
    if not select:
        return []
    if isinstance(select, str):
        parts: Iterable[str] = select.split(",")
    elif isinstance(select, (list, tuple)):
        parts = [str(p) for p in select]
    else:
        return []
    return [p.strip() for p in parts if str(p).strip()]


# "No such path", as distinct from "the path exists and holds null". The
# projection renders both as None, but deciding whether a select matched
# anything has to tell them apart.
_MISSING = object()


def _resolve(item: Any, path: str) -> Any:
    """Read a dot-path, returning ``_MISSING`` when the path does not exist."""
    cur = item
    for segment in path.split("."):
        if isinstance(cur, dict):
            if segment not in cur:
                return _MISSING
            cur = cur[segment]
        elif isinstance(cur, list) and segment.isdigit():
            idx = int(segment)
            if idx >= len(cur):
                return _MISSING
            cur = cur[idx]
        else:
            return _MISSING
    return cur


def _dig(item: Any, path: str) -> Any:
    value = _resolve(item, path)
    return None if value is _MISSING else value


#: Returned by :func:`resolve` when a path does not exist on an item.
MISSING = _MISSING


def resolve(item: Any, path: str) -> Any:
    """Read a dot-path from one item, telling "absent" from "present and null".

    Returns the value at ``path`` (``summary.name``, ``tags.0``), or
    :data:`MISSING` when the path does not exist. A path that exists and holds
    ``None`` returns ``None``::

        resolve({"a": {"b": None}}, "a.b") is None
        resolve({"a": {}}, "a.b") is MISSING
    """
    return _resolve(item, path)


def project(item: Any, select: Sequence[str], *, full_path_keys: bool = False) -> Any:
    """Keep only the selected dot-paths of one item.

    ``full_path_keys=False`` (default) flattens each path to its last
    segment: ``summary.name`` becomes ``name``, the shape an LLM reads most
    easily. On a collision the later path keeps its full name so no selected
    field is lost.

    ``full_path_keys=True`` keys by the path verbatim, for callers that must
    reference fields later by exactly the path they selected.
    """
    if not isinstance(item, dict):
        return item
    out: Dict[str, Any] = {}
    for path in select:
        value = _dig(item, path)
        if full_path_keys:
            out[path] = value
            continue
        key = path.rsplit(".", 1)[-1]
        if key in out:
            key = path
        out[key] = value
    return out


def _available_paths(item: Any, prefix: str = "", depth: int = 2) -> List[str]:
    if not isinstance(item, dict) or depth <= 0:
        return []
    out: List[str] = []
    for key, value in item.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and depth > 1:
            nested = _available_paths(value, path, depth - 1)
            out.extend(nested or [path])
        else:
            out.append(path)
    return out


def available_paths(item: Any, depth: int = 2) -> List[str]:
    """Dot-paths that can be selected on one item, ``depth`` levels deep.

    ``{"id": 1, "summary": {"name": "x"}}`` gives ``["id", "summary.name"]``.
    Anything that is not a dict gives ``[]``.
    """
    return _available_paths(item, "", depth)


def available_paths_for(items: Sequence[Any]) -> List[str]:
    """Dot-paths a caller could select, read from the first dict item."""
    for item in items:
        if isinstance(item, dict):
            return _available_paths(item)
    return []


def projection_matched_nothing(items: Sequence[Any], fields: Sequence[str]) -> bool:
    """True when no selected path exists on any item.

    Asks about *existence*, not value: a provider that answers
    ``"funding_stage": null`` for every item has answered the question, and
    the selection must be kept. Only a path found nowhere is a mistake.
    """
    if not fields or not items:
        return False
    saw_dict = False
    for item in items:
        if not isinstance(item, dict):
            continue
        saw_dict = True
        if any(_resolve(item, path) is not _MISSING for path in fields):
            return False
    return saw_dict


def repair_envelope_prefix(
    items: Sequence[Any], fields: Sequence[str]
) -> Optional[List[str]]:
    """Selected paths minus a shared leading segment, when that rescues them.

    Models write selects against the shape they see, and the reducer's own
    envelope names its collection ``items``, so ``items.id,items.name`` is a
    reasonable guess. Returns the stripped paths when every path is dotted,
    all share one first segment, and stripping makes the select match;
    otherwise None.
    """
    if not fields or not all("." in f for f in fields):
        return None
    prefixes = {f.split(".", 1)[0] for f in fields}
    if len(prefixes) != 1:
        return None
    stripped = [f.split(".", 1)[1] for f in fields]
    if projection_matched_nothing(items, stripped):
        return None
    return stripped


# ── sizing ─────────────────────────────────────────────────────────────


def _dumps(value: Any, *, ensure_ascii: bool = False) -> str:
    # Non-ASCII text is kept as written. Escaping it ("\u0906...") makes each
    # character cost six, so Hindi or Japanese text would be measured several
    # times larger than it is and trimmed for no reason.
    return json.dumps(value, default=str, separators=(",", ":"), ensure_ascii=ensure_ascii)


def _encodable(text: str) -> bool:
    """False when ``text`` holds a lone surrogate and cannot be sent as UTF-8."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _field_costs(items: Sequence[Any]) -> Dict[str, int]:
    """Total serialized chars each top-level field costs across all items.

    Empty when the items are not uniformly dicts.
    """
    costs: Dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            return {}
        for key, value in item.items():
            # + key, quotes, colon, separator
            costs[key] = costs.get(key, 0) + len(_dumps(value)) + len(key) + 4
    return costs


def _auto_fit(
    items: Sequence[Any], room: int
) -> Tuple[Optional[List[Any]], Dict[str, int]]:
    """Make ALL items fit by dropping the widest fields, not the last items.

    The model cannot see the budget, so letting it discover one by trial and
    error costs whole LLM turns. The reducer holds every field's measured
    size; dropping the most expensive fields until everything fits is one
    deterministic pass (widest first, ties by name).

    Returns ``(fitted_items, {field: ~chars_per_item_dropped})``, or
    ``(None, {})`` when trimming fields cannot help.
    """
    costs = _field_costs(items)
    if len(costs) <= _AUTO_FIT_MIN_FIELDS:
        return None, {}

    keep = set(costs)
    dropped: Dict[str, int] = {}
    order = sorted(costs, key=lambda k: (-costs[k], k))

    def measure(keys: set) -> int:
        return (
            sum(
                len(_dumps({k: v for k, v in item.items() if k in keys})) + 1
                for item in items
            )
            + 2
        )

    size = measure(keep)
    for key in order:
        if size <= room or len(keep) <= _AUTO_FIT_MIN_FIELDS:
            break
        keep.discard(key)
        dropped[key] = max(1, costs[key] // max(1, len(items)))
        size = measure(keep)

    if size > room or not dropped:
        return None, {}
    # Comprehension over item.items() preserves the provider's key order.
    fitted = [{k: v for k, v in item.items() if k in keep} for item in items]
    return fitted, dropped


# ── locating the collection ────────────────────────────────────────────


def _first_total(container: Any) -> Optional[int]:
    """Pull a server-reported total off an envelope dict."""
    if not isinstance(container, dict):
        return None
    for key in _TOTAL_KEYS:
        value = container.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        if isinstance(value, dict):  # e.g. {"page": {"totalElements": N}}
            nested = _first_total(value)
            if nested is not None:
                return nested
    for key in ("page", "pagination", "meta", "paging"):
        nested_container = container.get(key)
        if isinstance(nested_container, dict):
            nested = _first_total(nested_container)
            if nested is not None:
                return nested
    return None


def _locate_collection(
    parsed: Any,
) -> Tuple[Optional[List[Any]], Optional[str], Optional[int]]:
    """Find the payload's main collection: ``(items, path, total_reported)``.

    The longest list of objects wins, so a short sibling ("errors",
    "warnings") never displaces the real payload.
    """
    if isinstance(parsed, list):
        return parsed, None, None
    if not isinstance(parsed, dict):
        return None, None, None

    # JSON:API: `data` is the payload and `included` is a sidecar of related
    # resources that is often the LONGER list. Longest-wins would project
    # the sidecar and hand the model relationship stubs.
    data = parsed.get("data")
    if (
        isinstance(data, list)
        and data
        and isinstance(data[0], dict)
        and isinstance(parsed.get("included"), list)
    ):
        return data, "data", _first_total(parsed)

    best: Optional[List[Any]] = None
    best_path: Optional[str] = None
    best_owner: Any = None

    def walk(node: Any, path: str, depth: int) -> None:
        nonlocal best, best_path, best_owner
        if depth > _MAX_SEARCH_DEPTH or not isinstance(node, dict):
            return
        for key, value in node.items():
            child_path = f"{path}.{key}" if path else key
            if isinstance(value, list):
                # Only a list of objects is worth projecting; a list of
                # scalars is cheap and better left alone.
                if value and isinstance(value[0], dict):
                    if best is None or len(value) > len(best):
                        best, best_path, best_owner = value, child_path, node
                continue
            if isinstance(value, dict) and (
                depth < _MAX_SEARCH_DEPTH or key in _COLLECTION_KEYS
            ):
                walk(value, child_path, depth + 1)

    walk(parsed, "", 1)

    total = _first_total(best_owner) or _first_total(parsed)
    return best, best_path, total


def locate_collection(parsed: Any) -> Tuple[Optional[List[Any]], Optional[str], Optional[int]]:
    """Find the main list of items in a parsed API response.

    Returns ``(items, path, total_reported)``:

    * ``items``: the list, or ``None`` when the response holds no list of
      objects (a single object, a scalar).
    * ``path``: where it was found (``"data.results"``), or ``None`` when the
      response itself is the list.
    * ``total_reported``: the total the server reports (``total``,
      ``totalElements``, ``page.totalElements``, ...), or ``None``.

    This is the same search :func:`reduce_tool_result` uses: up to three
    levels deep, the longest list of objects wins, and JSON:API ``data`` beats
    a longer ``included``.
    """
    return _locate_collection(parsed)


# ── non-collection fallbacks ───────────────────────────────────────────


def _outline_value(value: Any, depth: int) -> Any:
    """One node of a structure outline: keep scalars, summarise containers."""
    if isinstance(value, str):
        if len(value) > _OUTLINE_VALUE_CHARS:
            return value[:_OUTLINE_VALUE_CHARS] + f"… [{len(value):,} chars]"
        return value
    if isinstance(value, dict):
        if depth <= 0:
            return {"_type": "object", "_keys": list(value)[:24]}
        return {k: _outline_value(v, depth - 1) for k, v in value.items()}
    if isinstance(value, list):
        summary: Dict[str, Any] = {"_type": "list", "_len": len(value)}
        if value and depth > 0:
            summary["_first"] = _outline_value(value[0], depth - 1)
        return summary
    return value


def _structure_outline(parsed: Dict[str, Any], budget: int) -> Optional[str]:
    """A valid-JSON structural view of a large non-collection object.

    Tries progressively shallower depths; None when even the shallowest view
    cannot fit, which only a pathologically wide object hits.
    """
    for depth in (2, 1, 0):
        outlined = {k: _outline_value(v, depth) for k, v in parsed.items()}
        text = _dumps(outlined)
        if len(text) <= budget:
            return text
    return None


# ── entry point ────────────────────────────────────────────────────────


def reduce_tool_result(
    payload: Any,
    tool_name: str = "tool",
    *,
    select: Any = None,
    budget: int = DEFAULT_RESULT_BUDGET,
    messages: Messages = DEFAULT_MESSAGES,
) -> ReducedResult:
    """Bound a tool result to what the agent can use, and say what was left out.

    Args:
        payload: The tool's output: a JSON string, or any JSON-serialisable
            value (dict, list, ...). Non-JSON strings are accepted.
        tool_name: Used in fallback text and summaries.
        select: Fields to keep per item, as a comma-separated string or a
            list of dot-paths relative to one item (``"id,summary.name"``).
            Applied whenever supplied, not only when the payload is oversized:
            a caller that knows which fields it consumes should not pay for
            the rest on every later turn.
        budget: Maximum characters for ``ReducedResult.text``. It is never
            exceeded while more than one item is kept. One item is always
            kept, so a single item larger than the budget still exceeds it,
            and a budget under about 1,000 characters has no room for the
            ``_meta`` block.
        messages: Instructions written into the result for the model.
    """
    raw_text = payload if isinstance(payload, str) else _dumps(payload)
    # A lone surrogate (a broken "\ud83d" from upstream) cannot be encoded as
    # UTF-8, so in that rare case the output falls back to ASCII escapes.
    ascii_only = not _encodable(raw_text)
    if ascii_only and not isinstance(payload, str):
        raw_text = _dumps(payload, ensure_ascii=True)
    chars_raw = len(raw_text)
    fields = parse_select(select)
    budget = max(1, int(budget or DEFAULT_RESULT_BUDGET))

    if chars_raw <= budget and not fields:
        return ReducedResult(
            text=raw_text, raw=raw_text, chars_raw=chars_raw, chars_kept=chars_raw
        )

    try:
        parsed = json.loads(raw_text) if isinstance(payload, str) else payload
    except (json.JSONDecodeError, TypeError, ValueError):
        parsed = None

    items, source_path, total_reported = _locate_collection(parsed)

    if items is None:
        # Not a collection: a single large object, or not JSON at all.
        if chars_raw <= budget:
            return ReducedResult(
                text=raw_text, raw=raw_text, chars_raw=chars_raw, chars_kept=chars_raw
            )
        if isinstance(parsed, dict):
            outline = _structure_outline(parsed, max(1, budget - _ENVELOPE_RESERVE))
            if outline is not None:
                envelope = {
                    "outline": json.loads(outline),
                    "_meta": {
                        "complete": False,
                        "chars_raw": chars_raw,
                        "note": messages.outline.format(chars=f"{chars_raw:,}"),
                    },
                }
                text = _dumps(envelope, ensure_ascii=ascii_only)
                return ReducedResult(
                    text=text,
                    raw=raw_text,
                    complete=False,
                    chars_raw=chars_raw,
                    chars_kept=len(text),
                    outlined=True,
                )
        dropped = chars_raw - budget
        text = raw_text[:budget] + messages.char_cut.format(dropped=dropped, tool=tool_name)
        return ReducedResult(
            text=text,
            raw=raw_text,
            complete=False,
            chars_raw=chars_raw,
            chars_kept=len(text),
            fallback_char_cut=True,
        )

    # A select written against the envelope (`items.id`) instead of one item:
    # strip the shared prefix, but only when the select as given matches
    # nothing, so a genuine `items` field on an item is never touched.
    prefix_stripped: Optional[str] = None
    if fields and projection_matched_nothing(items, fields):
        repaired = repair_envelope_prefix(items, fields)
        if repaired is not None:
            prefix_stripped = fields[0].split(".", 1)[0]
            fields = repaired

    projected = [project(item, fields) for item in items] if fields else list(items)

    # A select that matched nothing must never be served as a complete
    # result: that is a silent empty answer wearing a success badge. Drop the
    # projection and say which paths actually exist.
    select_error: Optional[str] = None
    if projection_matched_nothing(items, fields):
        available = available_paths_for(items)
        select_error = (
            f"The select {list(fields)} matched no field on any of the "
            f"{len(items)} items, so it was ignored and the full records are "
            f"shown instead (trimmed to fit). Select from these paths: "
            f"{', '.join(available[:40])}"
        )
        projected = list(items)
        fields = []

    def render(kept: List[Any], fields_dropped: Dict[str, int]) -> Tuple[str, bool]:
        """The finished text for these items, and whether it is complete."""
        complete = len(kept) == len(items)
        meta: Dict[str, Any] = {
            "items_included": len(kept),
            "items_available": len(items),
            "complete": complete,
        }
        if total_reported is not None and total_reported != len(items):
            meta["total_reported_by_source"] = total_reported
        if source_path:
            meta["source_path"] = source_path
        if fields:
            # Only what actually survived auto-fit.
            meta["fields_selected"] = [
                f
                for f in fields
                if f.rsplit(".", 1)[-1] not in fields_dropped and f not in fields_dropped
            ]
        if prefix_stripped:
            meta["select_prefix_stripped"] = (
                f"Your select paths were prefixed with '{prefix_stripped}.'; "
                f"that prefix names the collection, not a field, so it was "
                f"stripped. Select paths are relative to a single item "
                f"(e.g. summary.name)."
            )
        if select_error:
            meta["select_error"] = select_error
        if fields_dropped:
            names = sorted(fields_dropped)
            listed = ", ".join(names[:8]) + (" …" if len(names) > 8 else "")
            meta["fields_dropped"] = fields_dropped
            meta["fields_dropped_note"] = messages.fields_dropped.format(
                count=len(kept), fields=listed
            )
        if not complete:
            meta["items_omitted"] = len(items) - len(kept)
            meta["warning"] = messages.incomplete
            # Which fields are eating the budget, so a follow-up select is
            # informed rather than guessed. Measured on the original items.
            costs = _field_costs(items)
            if costs:
                top = sorted(costs.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
                meta["field_costs"] = dict(top)
        text = json.dumps({"items": kept, "_meta": meta}, default=str, ensure_ascii=ascii_only)
        return text, complete

    # Build the whole result; if the finished text still overflows (the
    # envelope reserve is an estimate), shrink the room by the overshoot and
    # rebuild. The fill always keeps at least one item.
    projected_full = projected
    room = max(1, budget - _ENVELOPE_RESERVE)
    for _ in range(3):
        projected = projected_full

        fields_dropped: Dict[str, int] = {}
        projected_size = sum(len(_dumps(item)) + 1 for item in projected) + 2
        if projected_size > room:
            fitted, fields_dropped = _auto_fit(projected, room)
            if fitted is not None:
                projected = fitted

        # Fill on item boundaries; serialise each item once (O(n)).
        kept: List[Any] = []
        used = 2  # the enclosing "[]"
        for item in projected:
            size = len(_dumps(item)) + 1
            if kept and used + size > room:
                break
            kept.append(item)
            used += size

        text, complete = render(kept, fields_dropped)
        if len(text) <= budget:
            break
        room = max(1, room - (len(text) - budget))

    # The passes above usually land inside the budget. When they do not, keep
    # the largest number of whole items that does: the text grows with every
    # item kept, so a binary search finds it in a few renders.
    if len(text) > budget and len(kept) > 1:
        low, high, fits = 1, len(kept) - 1, 1
        while low <= high:
            middle = (low + high) // 2
            if len(render(projected[:middle], fields_dropped)[0]) <= budget:
                fits, low = middle, middle + 1
            else:
                high = middle - 1
        kept = projected[:fits]
        text, complete = render(kept, fields_dropped)

    return ReducedResult(
        text=text,
        raw=raw_text,
        complete=complete,
        chars_raw=chars_raw,
        chars_kept=len(text),
        items_included=len(kept),
        items_available=len(items),
        total_reported=total_reported,
        fields_kept=list(fields),
        source_path=source_path,
        select_error=select_error,
        fields_dropped=fields_dropped,
    )
