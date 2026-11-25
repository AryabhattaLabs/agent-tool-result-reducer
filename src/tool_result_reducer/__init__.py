"""Shrink large JSON tool results for LLM agents without silently losing data."""

from .core import (
    DEFAULT_RESULT_BUDGET,
    MISSING,
    Messages,
    ReducedResult,
    available_paths,
    available_paths_for,
    locate_collection,
    parse_select,
    paths_not_found,
    project,
    projection_matched_nothing,
    reduce_tool_result,
    repair_envelope_prefix,
    resolve,
)

__all__ = [
    "DEFAULT_RESULT_BUDGET",
    "MISSING",
    "Messages",
    "ReducedResult",
    "available_paths",
    "available_paths_for",
    "locate_collection",
    "parse_select",
    "paths_not_found",
    "project",
    "projection_matched_nothing",
    "reduce_tool_result",
    "repair_envelope_prefix",
    "resolve",
]
__version__ = "0.2.5"
