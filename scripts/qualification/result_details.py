"""Bound qualification detail text without corrupting structured JSON."""

from __future__ import annotations

import hashlib
import json
from typing import Any


TRUNCATION_KEY = "_truncation"


def _compact_text(value: str, max_characters: int) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    marker = f"...[truncated sha256:{digest} chars={len(value)}]..."
    if max_characters <= len(marker):
        return marker[:max_characters]
    available = max_characters - len(marker)
    head = available // 2
    tail = available - head
    return value[:head] + marker + value[-tail:]


def _json_text(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def compact_details(value: str | None, max_characters: int) -> str | None:
    """Return bounded text, preserving a JSON object as a valid JSON object."""
    if value is None or len(value) <= max_characters:
        return value
    if max_characters < 128:
        raise ValueError("qualification detail limit must be at least 128 characters")

    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return _compact_text(value, max_characters)
    if not isinstance(parsed, dict):
        return _compact_text(value, max_characters)

    original_digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    bounded = dict(parsed)
    bounded[TRUNCATION_KEY] = {
        "characters": len(value),
        "omitted_fields": 0,
        "sha256": f"sha256:{original_digest}",
    }
    initial_string_limit = min(512, max(160, max_characters // 3))
    for key, item in list(bounded.items()):
        if key != TRUNCATION_KEY and isinstance(item, str) and len(item) > initial_string_limit:
            bounded[key] = _compact_text(item, initial_string_limit)

    while len(_json_text(bounded)) > max_characters:
        strings = [
            (len(item), key, item)
            for key, item in bounded.items()
            if key != TRUNCATION_KEY and isinstance(item, str) and len(item) > 96
        ]
        if strings:
            _, key, item = max(strings)
            excess = len(_json_text(bounded)) - max_characters
            target = max(96, len(item) - excess - 32)
            if target >= len(item):
                target = max(96, len(item) // 2)
            bounded[key] = _compact_text(item, target)
            continue

        removable = [
            (len(json.dumps(item, separators=(",", ":"), allow_nan=False)), key)
            for key, item in bounded.items()
            if key not in {TRUNCATION_KEY, "error", "milestones"}
        ]
        if removable:
            _, key = max(removable)
            del bounded[key]
            bounded[TRUNCATION_KEY]["omitted_fields"] += 1
            continue

        fallback: dict[str, Any] = {TRUNCATION_KEY: bounded[TRUNCATION_KEY]}
        if "milestones" in bounded:
            fallback["milestones"] = bounded["milestones"]
        if isinstance(bounded.get("error"), str):
            fallback["error"] = _compact_text(
                bounded["error"], max(96, max_characters // 2)
            )
        bounded = fallback
        if len(_json_text(bounded)) > max_characters and "milestones" in bounded:
            del bounded["milestones"]
            bounded[TRUNCATION_KEY]["omitted_fields"] += 1
        if len(_json_text(bounded)) > max_characters and "error" in bounded:
            overhead = len(_json_text({TRUNCATION_KEY: bounded[TRUNCATION_KEY]})) + 16
            bounded["error"] = _compact_text(
                bounded["error"], max(96, max_characters - overhead)
            )
        break

    result = _json_text(bounded)
    if len(result) > max_characters:
        raise ValueError("qualification detail metadata exceeds its character limit")
    return result


def join_details(*values: str | None, max_characters: int) -> str | None:
    """Join runner diagnostics while retaining a command's JSON detail envelope."""
    parts: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            parts.append(value)
            seen.add(value)
    if not parts:
        return None

    structured: list[dict[str, Any]] = []
    plain: list[str] = []
    for part in parts:
        try:
            parsed = json.loads(part)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            structured.append(parsed)
        else:
            plain.append(part)
    first = structured[0] if structured else None
    if not isinstance(first, dict):
        return compact_details("; ".join(parts), max_characters)

    merged = dict(first)
    if len(structured) > 1:
        merged["additional_case_details"] = structured[1:]
    runner_failures = merged.get("runner_failures")
    if runner_failures is None:
        failures: list[str] = []
    elif isinstance(runner_failures, list) and all(
        isinstance(item, str) for item in runner_failures
    ):
        failures = list(runner_failures)
    else:
        failures = [f"malformed prior runner_failures={runner_failures!r}"]
    for part in plain:
        if part not in failures:
            failures.append(part)
    if failures:
        merged["runner_failures"] = failures
    return compact_details(_json_text(merged), max_characters)
