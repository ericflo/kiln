"""Parse pi session JSONL into the canonical kiln Trajectory schema.

The canonical schema (kiln-train::trajectory::ScoredRollout) carries an
optional list of TurnSegments where each segment has:

    {
        "role": "system" | "user" | "assistant" | "tool",
        "content": <raw text the model emitted or saw>,
        "kind": "context" | "action" | "observation",
        "tool_call_id": <optional correlation id>,
        "warning_prefix_len": <bytes of harness warning to strip from env mask>
    }

Action segments are policy-gradient targets (assistant turns).
Observation segments are ECHO's env-CE targets (tool results / environment).
Context segments are system/user prompts — no gradient.

This module reads pi's `~/.pi/sessions/<uuid>.jsonl` format. The format
(verified against capabilities/caps/pi-doctest/rollout.py on
2026-05-18):

    {"type": "message", "message": {
        "role": "system" | "user" | "assistant" | "tool",
        "content": [{
            "type": "text" | "thinking" | "toolCall" | "toolResult",
            ...role-specific fields...
        }]
    }}

Tool-call args have appeared under both b["input"] and b["arguments"]
across pi versions. Tool result blocks use b["content"] (string) and
b["toolCallId"] for correlation.

See `docs/plans/echo-integration-plan.md` §3.3 and §B.8 for the design.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional


# Renderers from pi's content blocks to the Qwen-XML form the chat template
# expects. Mirror what rollout.py:280-318 was doing inline; centralised here
# so every cap (not just pi-doctest) gets the same conversion.
def _render_assistant_block(b: dict) -> Optional[str]:
    bt = b.get("type")
    if bt == "text" and isinstance(b.get("text"), str):
        return b["text"]
    if bt == "thinking" and isinstance(b.get("thinking"), str):
        # Qwen-style <think> tags so the chat template round-trip is closer
        # to what the model emitted at sample time.
        return f"<think>{b['thinking']}</think>"
    if bt == "toolCall":
        name = b.get("name", "")
        # Pi version drift: 0.75.1 used "input"; current builds emit
        # "arguments". Preserve both so action tokens retain tool payloads.
        args_obj = b.get("input") or b.get("arguments") or {}
        args_json = json.dumps(args_obj)
        return (
            f"<tool_call>"
            f'{{"name": "{name}", "arguments": {args_json}}}'
            f"</tool_call>"
        )
    return None


def _render_assistant_content(blocks: list) -> str:
    parts: list[str] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        rendered = _render_assistant_block(b)
        if rendered is not None:
            parts.append(rendered)
    return "".join(parts)


def _extract_tool_call_ids(blocks: list) -> list[str]:
    """Tool-call IDs emitted by the assistant in this turn; matched against
    subsequent tool-result events for correlation."""
    ids: list[str] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "toolCall":
            tid = b.get("id") or b.get("toolCallId")
            if isinstance(tid, str):
                ids.append(tid)
    return ids


def _render_tool_result_content(blocks: list) -> tuple[str, Optional[str]]:
    """Render a tool-role message's content blocks. Returns
    (rendered_text, tool_call_id-or-None).

    Handles two pi formats:
      - pi 0.75.1: role="tool", content=[{type:"toolResult", content:..., toolCallId:...}]
      - pi 0.75.3: role="toolResult", content=[{type:"text", text:...}]
    """
    parts: list[str] = []
    tool_call_id: Optional[str] = None
    for b in blocks:
        if not isinstance(b, dict):
            continue
        bt = b.get("type")
        if bt == "toolResult":
            content = b.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for inner in content:
                    if isinstance(inner, dict) and isinstance(inner.get("text"), str):
                        parts.append(inner["text"])
            tid = b.get("toolCallId") or b.get("tool_call_id") or b.get("id")
            if isinstance(tid, str) and tool_call_id is None:
                tool_call_id = tid
        elif bt == "text" and isinstance(b.get("text"), str):
            parts.append(b["text"])
    return "".join(parts), tool_call_id


def _detect_warning_prefix(text: str) -> Optional[int]:
    """Detect the harness warning prefix kiln emits when a tool call fails
    parsing or violates format constraints. Returns the byte length of the
    prefix (so the masker can advance past it), or None if no warning.

    Paper §3.2: warning tokens memorize within ~60 steps and stop providing
    useful gradient; ECHO's env_mask excludes them when
    MaskConfig::warning_filter is true (the default).
    """
    if not text.startswith("WARNINGS:\n"):
        return None
    # Find the start of real terminal output. kiln's harness wraps it in
    # <command_output>...</command_output> — strip up to that.
    idx = text.find("<command_output>")
    if idx > 0:
        return idx
    # Fallback: any double-newline after the warning block.
    idx = text.find("\n\n")
    if idx > 0:
        return idx + 2
    return len("WARNINGS:\n")


def _iter_events(path: Path) -> Iterator[dict]:
    """Yield JSON-decoded events from a pi session JSONL, silently skipping
    malformed lines. Mirrors the lenient parse in rollout.py."""
    if not path or not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue


def parse_pi_session(
    session_path: Path,
    include_context: bool = False,
) -> list[dict]:
    """Parse a pi session JSONL file into a list of TurnSegment dicts.

    Returns a list compatible with the canonical kiln Trajectory schema:

        [
            {"role": "system",    "content": "...", "kind": "context"},
            {"role": "user",      "content": "...", "kind": "context"},
            {"role": "assistant", "content": "...", "kind": "action"},
            {"role": "tool",      "content": "...", "kind": "observation",
             "tool_call_id": "...", "warning_prefix_len": null|int},
            ...
        ]

    When `include_context=False` (default) system/user turns are filtered out
    because they belong in the surrounding `AgenticGroup.messages` field
    rather than per-rollout trajectory. When `include_context=True` every
    turn is emitted, useful for end-to-end fidelity.
    """
    segments: list[dict] = []

    for event in _iter_events(session_path):
        if event.get("type") != "message":
            continue
        msg = event.get("message") or {}
        role = msg.get("role")
        content = msg.get("content")

        if role in ("system", "user"):
            if not include_context:
                continue
            text = _stringify_content(content)
            if text:
                segments.append({"role": role, "content": text, "kind": "context"})

        elif role == "assistant":
            if isinstance(content, list):
                rendered = _render_assistant_content(content)
                if rendered:
                    segments.append(
                        {"role": role, "content": rendered, "kind": "action"}
                    )

        elif role in ("tool", "toolResult"):
            if isinstance(content, list):
                rendered, tool_call_id = _render_tool_result_content(content)
                if rendered:
                    # Normalize the role label to "tool" — the Qwen chat
                    # template only accepts that name; pi 0.75.1 used
                    # "tool" while pi 0.75.3 emits "toolResult" but both
                    # carry the same observation tokens. (chat template
                    # raised "Unexpected message role" otherwise.)
                    seg: dict[str, Any] = {
                        "role": "tool",
                        "content": rendered,
                        "kind": "observation",
                    }
                    if tool_call_id is not None:
                        seg["tool_call_id"] = tool_call_id
                    warning_prefix_len = _detect_warning_prefix(rendered)
                    if warning_prefix_len is not None:
                        seg["warning_prefix_len"] = warning_prefix_len
                    segments.append(seg)

    return segments


def _stringify_content(content: Any) -> str:
    """Flatten a pi content blob into a single string (used for system/user
    blocks which are typically already strings, but handle list form too)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for b in content:
            if isinstance(b, dict) and isinstance(b.get("text"), str):
                parts.append(b["text"])
            elif isinstance(b, str):
                parts.append(b)
        return "".join(parts)
    return str(content) if content is not None else ""


def flatten_action_text(trajectory: list[dict]) -> str:
    """Mirror of `ScoredRollout::from_trajectory`'s text-flattening logic in
    Rust: join Action segment contents with `<TURN_BREAK>` so the legacy
    single-string `text` field stays populated even when `trajectory` is
    the canonical representation."""
    return "<TURN_BREAK>".join(
        seg["content"] for seg in trajectory if seg.get("kind") == "action"
    )


def build_scored_rollout(
    session_path: Path,
    reward: float,
    include_context: bool = False,
) -> dict:
    """One-shot helper: parse a pi session and emit a ScoredRollout-shaped
    dict. The result deserializes into kiln-train's ScoredRollout type
    via serde with no further transformation.

        {
            "text": "<flattened action segments joined by <TURN_BREAK>>",
            "reward": <reward>,
            "trajectory": [<TurnSegment>, ...],
        }
    """
    trajectory = parse_pi_session(session_path, include_context=include_context)
    text = flatten_action_text(trajectory) or "(empty)"
    return {"text": text, "reward": reward, "trajectory": trajectory}
