"""Build pi-compaction rollout prompts.

Mirrors pi's `serializeConversation()` + `SUMMARIZATION_PROMPT` from
`packages/coding-agent/src/core/compaction/{utils,compaction}.ts`, so the
rollout input matches what production pi actually sends the model when
compaction fires.

A "task" is one long conversation that needs compaction. The training and
eval JSONLs hold one task per line:

    {
      "task_id": "task_0000",
      "source_messages": [...Anthropic-format messages...],
      "system_prompt": "...",
      "ground_truth": {...},   # populated by extract_ground_truth.py
      "source_text": "...",    # cached serialized form for the rubric
    }
"""

from __future__ import annotations

import json
import re
from typing import Any


# Exact strings from pi's source (utils.ts + compaction.ts).
PI_SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a "
    "conversation between a user and an AI coding assistant, then produce "
    "a structured summary following the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in "
    "the conversation. ONLY output the structured summary."
)


PI_SUMMARIZATION_USER_TEMPLATE = """## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""


PI_SUMMARIZATION_BASE_PROMPT = (
    "The messages above are a conversation to summarize. Create a "
    "structured context checkpoint summary that another LLM will use to "
    "continue the work.\n\nUse this EXACT format:\n\n"
    + PI_SUMMARIZATION_USER_TEMPLATE
)


TOOL_RESULT_MAX_CHARS = 2000


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[... {truncated} more characters truncated]"


def _flatten_content_blocks(content: Any) -> tuple[list[str], list[str], list[dict[str, Any]], list[str]]:
    """Split an Anthropic-style content array into (text, thinking, tool_uses, tool_results)."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_uses: list[dict[str, Any]] = []
    tool_results: list[str] = []

    if isinstance(content, str):
        text_parts.append(content)
        return text_parts, thinking_parts, tool_uses, tool_results

    if not isinstance(content, list):
        text_parts.append(str(content))
        return text_parts, thinking_parts, tool_uses, tool_results

    for block in content:
        if not isinstance(block, dict):
            text_parts.append(str(block))
            continue
        btype = block.get("type", "text")
        if btype == "text":
            t = block.get("text", "")
            if t:
                text_parts.append(t)
        elif btype == "thinking":
            t = block.get("thinking", "")
            if t:
                thinking_parts.append(t)
        elif btype == "tool_use":
            tool_uses.append({
                "name": block.get("name", ""),
                "input": block.get("input", {}) or {},
            })
        elif btype == "tool_result":
            tc = block.get("content", "")
            if isinstance(tc, list):
                tc = "\n".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in tc
                )
            tool_results.append(str(tc))
        # silently skip image / other blocks; pi serializer also drops them
    return text_parts, thinking_parts, tool_uses, tool_results


def _format_tool_call_args(args: dict[str, Any]) -> str:
    """Mirror pi's `${k}=${JSON.stringify(v)}, ...` format."""
    if not args:
        return ""
    parts: list[str] = []
    for k, v in args.items():
        parts.append(f"{k}={json.dumps(v, ensure_ascii=False)}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Cross-agent tool-name normalisation
#
# Pi serializes tool calls with lowercase names: read, write, edit, bash.
# Claude Code uses Capitalised names: Read, Write, Edit, Bash, Glob, Grep,
# Bash, TodoWrite, etc. To make the rollout input *look like real pi* even
# when sourced from a Claude Code session, we normalize.
# ---------------------------------------------------------------------------

PI_TOOL_NAME_MAP = {
    "Read": "read",
    "Write": "write",
    "Edit": "edit",
    "MultiEdit": "edit",
    "str_replace_based_edit_tool": "edit",
    "Bash": "bash",
    "Shell": "bash",
    "Glob": "glob",
    "Grep": "grep",
    "WebFetch": "fetch",
    "WebSearch": "search",
    "NotebookEdit": "edit",
    "TodoWrite": "todo",
    "Task": "task",
}

PI_ARG_KEY_MAP = {
    "file_path": "path",
    "filePath": "path",
    "filename": "path",
    "abs_path": "path",
    "command": "cmd",
    "pattern": "pattern",
}


def _normalize_tool_call(name: str, input_args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Map Claude Code tool names and arg keys onto pi's canonical names."""
    pi_name = PI_TOOL_NAME_MAP.get(name, name.lower())
    pi_args: dict[str, Any] = {}
    for k, v in (input_args or {}).items():
        pi_k = PI_ARG_KEY_MAP.get(k, k)
        pi_args[pi_k] = v
    return pi_name, pi_args


SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL | re.IGNORECASE)


def serialize_conversation(anthropic_messages: list[dict[str, Any]]) -> str:
    """Mirror pi's `serializeConversation(messages)` from utils.ts.

    Output lines like:
        [User]: text
        [Assistant thinking]: text
        [Assistant]: text
        [Assistant tool calls]: name(args); name2(args2)
        [Tool result]: text (truncated to 2000 chars)
    Each separated by blank lines.
    """
    parts: list[str] = []
    for msg in anthropic_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        texts, thinkings, tool_uses, tool_results = _flatten_content_blocks(content)

        if role == "user":
            # In Anthropic format, tool_result blocks ride inside user messages.
            if texts:
                joined = "".join(texts).strip()
                if joined:
                    parts.append(f"[User]: {joined}")
            for tr in tool_results:
                if tr.strip():
                    parts.append(f"[Tool result]: {_truncate(tr, TOOL_RESULT_MAX_CHARS)}")
        elif role == "assistant":
            if thinkings:
                parts.append(f"[Assistant thinking]: {chr(10).join(thinkings)}")
            if texts:
                parts.append(f"[Assistant]: {chr(10).join(texts)}")
            if tool_uses:
                call_strs = []
                for tu in tool_uses:
                    pi_name, pi_args = _normalize_tool_call(tu["name"], tu["input"])
                    call_strs.append(f"{pi_name}({_format_tool_call_args(pi_args)})")
                parts.append(f"[Assistant tool calls]: {'; '.join(call_strs)}")
        # other roles: tool, system — usually not in `messages` array here
    return "\n\n".join(parts)


def build_compaction_user_message(serialized_conversation: str) -> str:
    """Wrap the serialized conversation in pi's <conversation> tags + prompt."""
    return f"<conversation>\n{serialized_conversation}\n</conversation>\n\n{PI_SUMMARIZATION_BASE_PROMPT}"


def build_rollout_messages(anthropic_messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return OpenAI-style messages list ready for kiln /v1/chat/completions."""
    serialized = serialize_conversation(anthropic_messages)
    return [
        {"role": "system", "content": PI_SUMMARIZATION_SYSTEM_PROMPT},
        {"role": "user", "content": build_compaction_user_message(serialized)},
    ]


# ============================================================================
# Ground-truth extraction (single source of truth for the rubric)
# ============================================================================

def _strip_system_reminders(text: str) -> str:
    """Remove `<system-reminder>...</system-reminder>` blocks from user text."""
    return SYSTEM_REMINDER_RE.sub("", text).strip()


def _first_user_text(messages: list[dict[str, Any]]) -> str:
    """Return the first *real* user message — system-reminder noise stripped.

    Claude Code injects a `<system-reminder>` block on the first user message
    carrying boilerplate context (date, env, instructions). The user's actual
    question follows. We strip the reminder and return what's left.
    """
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content", "")
        if isinstance(content, str):
            cleaned = _strip_system_reminders(content)
            if cleaned:
                return cleaned
        elif isinstance(content, list):
            text_parts: list[str] = []
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    t = b.get("text", "")
                    if t:
                        text_parts.append(t)
            joined = _strip_system_reminders("\n".join(text_parts))
            if joined:
                return joined
    return ""


def _walk_tool_calls(messages: list[dict[str, Any]]) -> tuple[set[str], set[str], set[str]]:
    """Walk tool_use blocks, classify file ops by tool name into (read_only, modified, all_paths).

    Handles both pi-canonical names (`read`/`write`/`edit`) and Claude Code
    capitalised names (`Read`/`Write`/`Edit`/`MultiEdit`) after normalisation.
    """
    read_paths: set[str] = set()
    write_paths: set[str] = set()
    edit_paths: set[str] = set()
    all_paths: set[str] = set()
    for m in messages:
        if m.get("role") != "assistant":
            continue
        content = m.get("content", "")
        if not isinstance(content, list):
            continue
        for b in content:
            if not isinstance(b, dict) or b.get("type") != "tool_use":
                continue
            raw_name = b.get("name", "")
            raw_input = b.get("input", {}) or {}
            pi_name, pi_args = _normalize_tool_call(raw_name, raw_input)
            path = pi_args.get("path") or pi_args.get("file") or ""
            if isinstance(path, str) and path:
                all_paths.add(path)
                if pi_name == "read":
                    read_paths.add(path)
                elif pi_name == "write":
                    write_paths.add(path)
                elif pi_name == "edit":
                    edit_paths.add(path)
    modified = write_paths | edit_paths
    read_only = read_paths - modified
    return read_only, modified, all_paths


def _collect_tool_results(messages: list[dict[str, Any]]) -> str:
    """All tool_result text, joined, for identifier/error extraction."""
    out: list[str] = []
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content", "")
        if not isinstance(content, list):
            continue
        for b in content:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                tc = b.get("content", "")
                if isinstance(tc, list):
                    for sub in tc:
                        if isinstance(sub, dict) and sub.get("type") == "text":
                            out.append(sub.get("text", ""))
                else:
                    out.append(str(tc))
    return "\n".join(out)


PATH_RE = re.compile(
    r"(?:(?<![A-Za-z0-9_/])"
    r"(?:[~./]?[A-Za-z0-9_-]+(?:/[A-Za-z0-9_.-]+)+)"
    r"(?:\.[A-Za-z0-9]{1,8})?)"
)

IDENT_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]{2,}|[a-z][a-z0-9_]{4,}(?:\.[a-z_]+)?)\b")

ERROR_PAT = re.compile(
    r"\b(?:Traceback|Error|Exception|FAILED|fatal:|SyntaxError|"
    r"TypeError|ValueError|NameError|AttributeError|KeyError|"
    r"IndexError|ImportError|ModuleNotFoundError|FileNotFoundError|"
    r"OSError|RuntimeError|AssertionError|RecursionError|StopIteration|"
    r"command not found|No such file|Permission denied|exit status [1-9])\b",
    re.IGNORECASE,
)

# Same common-words filter as rubric.py
COMMON_WORDS = {
    "the", "this", "that", "these", "those", "what", "which", "where",
    "when", "while", "with", "without", "should", "would", "could",
    "result", "output", "error", "function", "method", "class", "module",
    "system", "string", "number", "value", "field", "object", "array",
    "request", "response", "session", "context", "conversation", "summary",
    "assistant", "user", "tool", "tools", "model", "models", "prompt",
    "prompts", "argument", "arguments", "parameter", "parameters",
    "return", "returns", "import", "imports", "package", "packages",
    "library", "libraries", "version", "versions",
    "completed", "started", "finished", "process", "running",
    "current", "previous", "following", "above", "below",
    "section", "sections", "summary", "summaries",
}


def extract_ground_truth(anthropic_messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Pre-compute the ground-truth facts the rubric consumes.

    Heuristics:
    - first_user_goal = the first user message's plain text
    - source_paths = union of read/write/edit tool-call paths + paths
      mentioned in tool results (regex)
    - modified_paths = write+edit tool-call paths
    - read_only_paths = read-only tool-call paths
    - source_identifiers = camelCase / snake_case names from tool inputs +
      tool results (top-N by frequency)
    - source_errors = error-keyword lines from tool results (top-N)
    """
    first_user_goal = _first_user_text(anthropic_messages)
    read_only, modified, all_tool_paths = _walk_tool_calls(anthropic_messages)
    tool_results_text = _collect_tool_results(anthropic_messages)
    user_text_blob = "\n".join(
        _first_user_text([m]) for m in anthropic_messages if m.get("role") == "user"
    )

    # Also pull paths from tool results / user text (mentions like "see /tmp/foo.py")
    extra_paths = set()
    for blob in (tool_results_text, user_text_blob):
        for m in PATH_RE.finditer(blob):
            p = m.group(0).rstrip(".,;:!?)]}")
            if "://" in p:
                continue
            if not ("/" in p or "." in p):
                continue
            if 4 <= len(p) <= 200:
                extra_paths.add(p)

    source_paths = set(all_tool_paths) | extra_paths

    # Identifiers — from tool results (heaviest source) and from tool-call args
    blob_for_idents = tool_results_text + "\n" + user_text_blob
    idents_freq: dict[str, int] = {}
    for m in IDENT_RE.finditer(blob_for_idents):
        ident = m.group(1)
        if ident.lower() in COMMON_WORDS:
            continue
        if len(ident) < 5:
            continue
        idents_freq[ident] = idents_freq.get(ident, 0) + 1
    source_identifiers = [
        i for i, _ in sorted(idents_freq.items(), key=lambda kv: -kv[1])[:30]
    ]

    # Errors
    errors: list[str] = []
    for line in tool_results_text.splitlines():
        if ERROR_PAT.search(line):
            stripped = line.strip()
            if stripped and len(stripped) < 400 and stripped not in errors:
                errors.append(stripped)
            if len(errors) >= 20:
                break

    return {
        "first_user_goal": first_user_goal[:4000],  # cap to avoid mega payloads
        "source_paths": sorted(source_paths),
        "modified_paths": sorted(modified),
        "read_only_paths": sorted(read_only),
        "source_identifiers": source_identifiers,
        "source_errors": errors,
    }


if __name__ == "__main__":  # pragma: no cover
    sample = [
        {"role": "user", "content": "Please fix the doctest failure in circular_shift"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Let me read the file first."},
            {"type": "tool_use", "name": "read", "input": {"path": "/tmp/solution.py"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "1",
             "content": "def circular_shift(x, shift): pass"},
        ]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "name": "edit", "input": {"path": "/tmp/solution.py", "old": "pass", "new": "return x[shift:] + x[:shift]"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "2", "content": "AssertionError: doctest example 3 failed"},
        ]},
    ]
    serialized = serialize_conversation(sample)
    print("--- SERIALIZED ---")
    print(serialized)
    print()
    gt = extract_ground_truth(sample)
    print("--- GROUND TRUTH ---")
    print(json.dumps(gt, indent=2))
