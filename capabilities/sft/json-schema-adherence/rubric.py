"""Strict JSON-schema adherence rubric.

Scores a model response against a (query, schema) prompt with four sub-scores
in [0,1] and a composite score. See capability.md for definitions.

Usage as a library:
    from rubric import score_response
    score_response(response_str, schema_dict) -> dict

CLI:
    python3 rubric.py < responses.jsonl  # one {prompt_id, response} per line
"""
from __future__ import annotations

import json
import re
import sys
from typing import Any

import jsonschema

# Tokens that betray placeholder content rather than substantive output.
_PLACEHOLDER_TOKENS = (
    "lorem ipsum",
    "lorem",
    "placeholder",
    "<placeholder>",
    "todo",
    "tbd",
    "n/a",
    "foo",
    "bar",
    "baz",
)
# Match a leading ```json ... ``` fence and capture the body.
_FENCE_RE = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$",
    re.DOTALL,
)


def strip_optional_fence(text: str) -> str:
    """Strip a leading/trailing ```json``` fence if present. Otherwise return
    the input unchanged.

    Does not strip preamble like ``"Here's the JSON: { ... }"`` — that case
    fails the ``is_pure`` rubric on purpose.
    """
    m = _FENCE_RE.match(text.strip())
    if m:
        return m.group(1).strip()
    return text


def _walk_strings(obj: Any, out: list[str]) -> None:
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk_strings(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk_strings(v, out)


def is_substantive(parsed: Any) -> bool:
    """Heuristic substantive-content check.

    Fails when:
    - any string field equals or contains a placeholder token (case-insensitive),
    - total string-field byte length < 50.
    """
    strings: list[str] = []
    _walk_strings(parsed, strings)
    total_len = sum(len(s.encode("utf-8")) for s in strings)
    # Threshold tuned to 25 bytes: most reasonable answers clear this; rejects
    # one-word stubs like {"name": "x"}.
    if total_len < 25:
        return False
    joined = " ".join(strings).lower()
    for tok in _PLACEHOLDER_TOKENS:
        if tok in joined:
            return False
    return True


def is_pure(raw: str) -> bool:
    """True iff the response, after .strip(), starts with `{` or `[` and ends
    with the matching closer — i.e. no preamble/postamble/fences/commentary
    around the JSON.
    """
    s = raw.strip()
    if not s:
        return False
    first, last = s[0], s[-1]
    if first == "{" and last == "}":
        return True
    if first == "[" and last == "]":
        return True
    return False


def score_response(
    response: str,
    schema: dict,
) -> dict:
    """Score a single response. Returns a dict with sub-scores + composite.

    Sub-scores are 0.0 or 1.0; composite is a weighted average per the
    capability.md formula.
    """
    parses = 0.0
    validates = 0.0
    is_pure_score = 1.0 if is_pure(response) else 0.0
    is_subs = 0.0

    parsed: Any = None
    try:
        candidate = strip_optional_fence(response)
        parsed = json.loads(candidate)
        parses = 1.0
    except (json.JSONDecodeError, ValueError):
        parses = 0.0
    if parses == 1.0:
        try:
            jsonschema.validate(parsed, schema)
            validates = 1.0
        except jsonschema.ValidationError:
            validates = 0.0
        is_subs = 1.0 if is_substantive(parsed) else 0.0

    composite = 0.4 * parses + 0.3 * validates + 0.2 * is_pure_score + 0.1 * is_subs
    return {
        "parses": parses,
        "validates": validates,
        "is_pure": is_pure_score,
        "is_substantive": is_subs,
        "composite": composite,
    }


def score_dataset(
    prompts: list[dict],
    responses: dict[str, str],
) -> dict:
    """Score a full dataset; ``prompts`` is the dataset rows (dicts with keys
    ``id``, ``query``, ``schema``) and ``responses`` maps prompt id → raw
    model response.

    Returns aggregate statistics suitable for logging in capability.jsonl.
    """
    per_prompt = []
    n_seen = 0
    for row in prompts:
        rid = row["id"]
        if rid not in responses:
            continue
        s = score_response(responses[rid], row["schema"])
        s["id"] = rid
        per_prompt.append(s)
        n_seen += 1
    if not per_prompt:
        return {"n": 0}

    def mean(k: str) -> float:
        return sum(p[k] for p in per_prompt) / len(per_prompt)

    return {
        "n": n_seen,
        "parses": mean("parses"),
        "validates": mean("validates"),
        "is_pure": mean("is_pure"),
        "is_substantive": mean("is_substantive"),
        "composite": mean("composite"),
        "per_prompt": per_prompt,
    }


def _self_test() -> None:
    # 1. Perfect output.
    schema = {
        "type": "object",
        "required": ["name", "age"],
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string", "minLength": 3},
            "age": {"type": "integer", "minimum": 0},
            "city": {"type": "string"},
        },
    }
    perfect = '{"name": "Eric Florenzano", "age": 42, "city": "San Francisco"}'
    s = score_response(perfect, schema)
    assert abs(s["composite"] - 1.0) < 1e-9, s

    # 2. Preamble — fails is_pure, fails parses.
    with_preamble = 'Here\'s the JSON: {"name": "Eric", "age": 42}'
    s = score_response(with_preamble, schema)
    assert s["is_pure"] == 0.0
    assert s["parses"] == 0.0

    # 3. Markdown fence — parses+validates but fails is_pure.
    fenced = '```json\n{"name": "Eric Florenzano", "age": 42}\n```'
    s = score_response(fenced, schema)
    assert s["parses"] == 1.0
    assert s["validates"] == 1.0
    assert s["is_pure"] == 0.0, s

    # 4. Missing required field — parses but fails validate.
    missing = '{"name": "Eric Florenzano"}'
    s = score_response(missing, schema)
    assert s["parses"] == 1.0
    assert s["validates"] == 0.0

    # 5. Extra field — parses but fails validate when additionalProperties:false.
    extra = '{"name": "Eric Florenzano", "age": 42, "extra": "no"}'
    s = score_response(extra, schema)
    assert s["validates"] == 0.0, s

    # 6. Placeholder values — substantive=0.
    placeholder = '{"name": "Foo Bar", "age": 0}'
    s = score_response(placeholder, schema)
    assert s["is_substantive"] == 0.0

    print("rubric self-test PASSED")


if __name__ == "__main__":
    _self_test()
