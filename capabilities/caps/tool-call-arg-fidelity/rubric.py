"""Rubric for tool-call-arg-fidelity.

Inputs to score_response:
  - response: str (the model's output)
  - schema: dict (tool schema with required, types, allowed fields)

4 sub-scores:
  parses (0.10)          — JSON parses
  required_fields (0.40) — fraction of required fields present (TARGET)
  type_correctness (0.30)— fraction of present fields with correct type
  no_extra_fields (0.20) — 1.0 if no extras, linearly penalized otherwise
"""
from __future__ import annotations

import json
import re
from typing import Any

WEIGHTS = {
    "parses": 0.10,
    "required_fields": 0.40,
    "type_correctness": 0.30,
    "no_extra_fields": 0.20,
}

_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def extract_json(text: str) -> dict | None:
    """Find the first {...} block in text and try to parse it.

    Handles common 4B failures: trailing prose, fenced code, single quotes.
    """
    if not text:
        return None
    # Try strict parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Strip code fences if present
    fenced = re.search(r"```(?:json)?\s*\n?(.+?)\n?```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass
    # Find the first balanced {...}
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                blob = text[start:i + 1]
                try:
                    return json.loads(blob)
                except Exception:
                    # Try a few light repairs: single quotes -> double, trailing commas
                    repaired = re.sub(r",\s*([}\]])", r"\1", blob)
                    try:
                        return json.loads(repaired)
                    except Exception:
                        return None
    return None


def score_parses(response: str, schema: dict) -> float:
    try:
        json.loads(response.strip())
        return 1.0
    except Exception:
        # Best-effort extraction still gets partial credit (0.5)
        return 0.5 if extract_json(response) is not None else 0.0


def score_required_fields(response: str, schema: dict) -> float:
    required = schema.get("required", [])
    if not required:
        return 1.0
    obj = extract_json(response)
    if not isinstance(obj, dict):
        return 0.0
    present = sum(1 for f in required if f in obj)
    return present / len(required)


def score_type_correctness(response: str, schema: dict) -> float:
    """Of fields present in the response (and in the schema), what
    fraction match the schema's declared type."""
    types = schema.get("types", {})
    obj = extract_json(response)
    if not isinstance(obj, dict):
        return 0.0
    relevant = [(k, v) for k, v in obj.items() if k in types]
    if not relevant:
        return 1.0  # vacuously correct (other sub-scores penalize)
    correct = 0
    for k, v in relevant:
        expected = types[k]
        py_t = _TYPE_MAP.get(expected, object)
        # Special-case: bool is subclass of int in Python, but JSON treats them distinct
        if expected == "integer" and isinstance(v, bool):
            continue
        if expected == "number" and isinstance(v, bool):
            continue
        if isinstance(v, py_t):
            correct += 1
    return correct / len(relevant)


def score_no_extra_fields(response: str, schema: dict) -> float:
    allowed = set(schema.get("allowed", []))
    if not allowed:
        return 1.0
    obj = extract_json(response)
    if not isinstance(obj, dict):
        return 0.0
    actual = set(obj.keys())
    extras = actual - allowed
    if not extras:
        return 1.0
    # Linear penalty: 1.0 → 0.0 as extras grow from 0 to len(allowed)
    return max(0.0, 1.0 - len(extras) / max(len(allowed), 1))


def score_response(response: str, schema: dict | None = None, **_ignore) -> dict[str, float]:
    if schema is None:
        schema = {}
    s = {
        "parses": score_parses(response, schema),
        "required_fields": score_required_fields(response, schema),
        "type_correctness": score_type_correctness(response, schema),
        "no_extra_fields": score_no_extra_fields(response, schema),
    }
    s["composite"] = sum(WEIGHTS[k] * v for k, v in s.items() if k != "composite")
    return s


def main() -> None:
    import sys
    sums = dict.fromkeys(WEIGHTS.keys(), 0.0)
    sums["composite"] = 0.0
    n = 0
    for line in sys.stdin:
        if not line.strip():
            continue
        d = json.loads(line)
        s = score_response(d.get("response", ""), d.get("schema", {}))
        for k in sums:
            sums[k] += s[k]
        n += 1
    if n == 0:
        print("ORACLE_ERROR: no responses scored", file=sys.stderr)
        sys.exit(2)
    print(f"SCORE={sums['composite']/n:.4f}")
    for k in WEIGHTS:
        print(f"{k}={sums[k]/n:.4f}")
    print(f"N={n}")


if __name__ == "__main__":
    main()
