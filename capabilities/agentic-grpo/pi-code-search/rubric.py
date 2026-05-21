"""Reward function for pi-code-search — v1, multi-component.

CAPABILITY: Given a code-locate question over a repository (where is X
defined; what file:line references Y), the model should answer correctly
with the MINIMUM bytes of tool output consumed — preferring grep / glob /
find over Read of large files, and choosing the cheapest pattern that
distinguishes the right hit.

DESIGN DISCIPLINE (capability.md §0 adversarial review):

The composite is `outcome × (weighted agentic sub-scores + base)`. The
multiplication by outcome makes "guess without searching" and "submit
empty" both score 0 regardless of process. This is the v1 "no-empty
solution cheating" guard required for any agentic-grpo rubric.

| Sub-score        | Weight | Measures                                | Cannot be cheated by |
|------------------|--------|-----------------------------------------|----------------------|
| outcome          | mult   | F1 over (file, line) tuples (±N tol)    | empty/garbage → 0    |
| efficiency       | 0.30   | bytes consumed vs gold-optimal target   | many small reads still accumulate bytes |
| tool_choice      | 0.20   | penalty per large-file Read (>2KB)      | a few Reads OK; many are not |
| grounding        | 0.15   | each predicted answer appears in some tool-result body | guess-without-searching gets 0 |
| format_compliance| 0.05   | final answer matches file:line regex    | prose answer → 0     |
| base             | 0.30   | flat floor when outcome > 0             | (no signal alone)    |

  Composite = outcome × (0.30·efficiency + 0.20·tool_choice
                       + 0.15·grounding + 0.05·format
                       + 0.30·base)

Weights sum to 1.00 inside the parentheses. When outcome=1.0 and all
sub-scores=1.0, composite=1.0. When outcome=0.0, composite=0.0.

ADVERSARIAL REVIEW

Q: cheapest path to 1.0 without searching?
A: Guess. Mitigation: outcome F1 over (file, line) makes uniform
   guessing ~0 (gold spans the whole repo).

Q: Read whole codebase in one tool call.
A: Mitigation: efficiency penalizes bytes consumed. One whole-file Read
   of >50KB tanks efficiency to ~0. tool_choice also penalizes >2KB Read.

Q: Emit grep call, ignore output, guess the answer.
A: Mitigation: grounding sub-score = 0 unless predicted file:line
   appears in some tool-result body. **This is critical** — without it
   the model would learn to guess instead of search.

Q: Submit empty answer / no file:line.
A: format_compliance and outcome both 0; composite=0.

Q: Always answer "crates/kiln-train/src/trainer.rs:1" for every symbol.
A: Mitigation: outcome F1 with gold answers spread across the repo →
   uniform guessing yields F1 near 0.

PI 0.75.1 SESSION SCHEMA (verified via lib/pi_trajectory.py 2026-05-18):

  Each non-empty line is an event {type, id, parentId, timestamp, message}
  message = {role, content: [block, ...]}
    role one of: "user", "assistant", "tool"  (pi 0.75 uses "tool" not "toolResult")
    block.type one of: "text", "thinking", "toolCall", "toolResult"
      toolCall: {type, name, id, input: {...}}
      toolResult: {type, toolCallId, content: str | list[block]}

Tool names in pi 0.75: bash (shell), read, write, edit, glob, grep, etc.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

# Import shared helpers from pi-doctest's rubric_v0_outcome_only WITHOUT
# polluting sys.path (otherwise pi-doctest's task_scaffold.py shadows
# ours when rollout.py later does `import task_scaffold`).
_PARENT = Path(__file__).resolve().parent
_SIBLING_DOCTEST = _PARENT.parent / "pi-doctest"

def _load_sibling_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_doctest_rubric_v0 = _load_sibling_module(
    "_pi_doctest_rubric_v0",
    _SIBLING_DOCTEST / "archive" / "rubric_v0_outcome_only.py",
)
_iter_messages = _doctest_rubric_v0._iter_messages
_tool_calls_in = _doctest_rubric_v0._tool_calls_in
_tool_input = _doctest_rubric_v0._tool_input
_format_compliance = _doctest_rubric_v0._format_compliance


# -----------------------------------------------------------------------------
# Answer parsing
# -----------------------------------------------------------------------------

# Matches a file:line pair like `crates/kiln-train/src/trainer.rs:7270`
# or `src/main.rs:42`. The file part is path-like (slash-separated, with
# common code extensions). The line is a positive integer.
ANSWER_PAIR = re.compile(
    r"([A-Za-z0-9_./\-]+\.(?:rs|py|toml|md|json|sh|c|h|cpp|cc|ts|tsx|js|go|java|yaml|yml|txt|cu|cuh|html|css|toml|cfg|lock)):"
    r"(\d+)"
)


def _last_assistant_text(transcript: list) -> str:
    """Concatenate all text/thinking blocks from the LAST assistant turn."""
    last = ""
    for _, msg in _iter_messages(transcript):
        if msg.get("role") != "assistant":
            continue
        parts = []
        for b in msg.get("content") or []:
            if isinstance(b, dict):
                if b.get("type") == "text" and isinstance(b.get("text"), str):
                    parts.append(b["text"])
                # We do NOT scrape <think> for answers; only public text.
        if parts:
            last = "".join(parts)
    return last


def _all_assistant_text(transcript: list) -> str:
    """Concatenate all assistant turn TEXT (no thinking, no tool calls)."""
    parts: list[str] = []
    for _, msg in _iter_messages(transcript):
        if msg.get("role") != "assistant":
            continue
        for b in msg.get("content") or []:
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str):
                parts.append(b["text"])
    return "\n".join(parts)


def _normalize_path(p: str) -> str:
    """Strip common workdir prefixes (`repo/`, `./repo/`, `./`) so the
    model's path is comparable to gold which is repo-relative."""
    while True:
        if p.startswith("./"):
            p = p[2:]
        elif p.startswith("repo/"):
            p = p[5:]
        elif p.startswith("/repo/"):
            p = p[6:]
        else:
            break
    return p


def parse_predicted_pairs(answer_text: str) -> list[tuple[str, int]]:
    """Extract all (file, line) pairs from the answer text.

    Dedupes while preserving order. Lines must be positive integers.
    The `repo/` prefix is stripped if present (the model may include
    it even though the prompt asks for a path relative to repo/)."""
    pairs: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for m in ANSWER_PAIR.finditer(answer_text or ""):
        try:
            ln = int(m.group(2))
        except ValueError:
            continue
        if ln <= 0:
            continue
        f = _normalize_path(m.group(1))
        pair = (f, ln)
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


# -----------------------------------------------------------------------------
# Sub-score: outcome (F1 with tolerance)
# -----------------------------------------------------------------------------

def _f1_with_tolerance(
    predicted: list[tuple[str, int]],
    gold: list[tuple[str, int]],
    line_tol: int = 2,
) -> float:
    """F1 between predicted and gold (file, line) sets.

    A predicted pair (f, p_line) matches a gold pair (g_file, g_line) iff
    f == g_file AND |p_line - g_line| <= line_tol. Each gold entry can
    match at most one predicted entry (greedy nearest)."""
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    # Greedy match: for each predicted, find closest unmatched gold.
    gold_used = [False] * len(gold)
    matched = 0
    for p_file, p_line in predicted:
        best_i = -1
        best_diff = line_tol + 1
        for i, (g_file, g_line) in enumerate(gold):
            if gold_used[i]:
                continue
            if p_file != g_file:
                continue
            diff = abs(p_line - g_line)
            if diff <= line_tol and diff < best_diff:
                best_diff = diff
                best_i = i
        if best_i >= 0:
            gold_used[best_i] = True
            matched += 1
    if matched == 0:
        return 0.0
    precision = matched / len(predicted)
    recall = matched / len(gold)
    return 2 * precision * recall / (precision + recall)


def _outcome(transcript: list, task: dict, line_tol: int = 2) -> tuple[float, dict]:
    """Outcome = F1 between predicted (file, line) and gold set."""
    last = _last_assistant_text(transcript)
    predicted = parse_predicted_pairs(last)
    gold_raw = task.get("gold") or []
    gold: list[tuple[str, int]] = []
    for g in gold_raw:
        if isinstance(g, list) and len(g) == 2:
            try:
                gold.append((str(g[0]), int(g[1])))
            except (TypeError, ValueError):
                continue
        elif isinstance(g, dict) and "file" in g and "line" in g:
            try:
                gold.append((str(g["file"]), int(g["line"])))
            except (TypeError, ValueError):
                continue
    f1 = _f1_with_tolerance(predicted, gold, line_tol=line_tol)
    diag = {
        "n_predicted": len(predicted),
        "n_gold": len(gold),
        "predicted_head": predicted[:5],
        "gold_head": gold[:5],
    }
    return f1, diag


# -----------------------------------------------------------------------------
# Sub-score: efficiency (bytes consumed)
# -----------------------------------------------------------------------------

def _iter_tool_results(transcript: list) -> list[str]:
    """Return all tool-result body strings the model received (in order)."""
    out: list[str] = []
    for _, msg in _iter_messages(transcript):
        if msg.get("role") not in ("tool", "toolResult"):
            continue
        for b in msg.get("content") or []:
            if not isinstance(b, dict):
                continue
            content = b.get("content")
            if isinstance(content, str):
                out.append(content)
            elif isinstance(content, list):
                for inner in content:
                    if isinstance(inner, dict) and isinstance(inner.get("text"), str):
                        out.append(inner["text"])
            # Some pi versions store tool-result text as a plain text block.
            elif b.get("type") == "text" and isinstance(b.get("text"), str):
                out.append(b["text"])
    return out


def _bytes_consumed(transcript: list) -> int:
    return sum(len(s.encode("utf-8", errors="replace")) for s in _iter_tool_results(transcript))


def _efficiency(
    transcript: list,
    target_bytes: int,
    span: int = 5,
    floor: int = 100,
) -> tuple[float, dict]:
    """efficiency = clip(1 - max(0, bytes - target) / (span * max(target, floor)), 0, 1).

    target_bytes is the size of the gold optimal grep output.
    span=5 means: at 5x over target → 0.0, at 1x target → 1.0.
    floor prevents tiny targets from creating an impossible bar."""
    bytes_consumed = _bytes_consumed(transcript)
    eff_target = max(target_bytes, floor)
    over = max(0, bytes_consumed - target_bytes)
    eff = 1.0 - over / (span * eff_target)
    eff = max(0.0, min(1.0, eff))
    return eff, {
        "bytes_consumed": bytes_consumed,
        "target_bytes": target_bytes,
        "effective_target": eff_target,
    }


# -----------------------------------------------------------------------------
# Sub-score: tool_choice (penalty for >2KB Reads)
# -----------------------------------------------------------------------------

# pi tool names. We treat any tool with name starting with "read" or
# containing "read" or "view" as a Read-style call. grep/find/glob/rg/
# ast-grep are the "search" tools we reward implicitly via efficiency.
_READ_TOOLS = {"read", "fileRead", "Read", "view", "cat"}


def _tool_choice(
    transcript: list,
    large_file_bytes: int = 2048,
) -> tuple[float, dict]:
    """1.0 if no >threshold Read; -0.20 per large Read, floored at 0.

    'Large Read' means a tool call whose paired tool result is >2KB AND
    whose name is a Read-style tool. We use the SIZE OF THE TOOL RESULT
    (not the file on disk) so a Read that pi truncated automatically only
    counts up to its actual visible size."""
    # Build index of toolCall id → tool name from assistant turns.
    name_by_id: dict[str, str] = {}
    for _, msg in _iter_messages(transcript):
        if msg.get("role") != "assistant":
            continue
        for tc in _tool_calls_in(msg):
            tid = tc.get("id") or tc.get("toolCallId")
            name = tc.get("name") or ""
            if isinstance(tid, str) and isinstance(name, str):
                name_by_id[tid] = name

    # Sum penalty over tool-role messages whose paired call is a Read.
    n_large_reads = 0
    n_reads = 0
    for _, msg in _iter_messages(transcript):
        if msg.get("role") not in ("tool", "toolResult"):
            continue
        for b in msg.get("content") or []:
            if not isinstance(b, dict):
                continue
            tcid = b.get("toolCallId") or b.get("tool_call_id") or b.get("id")
            content = b.get("content")
            if isinstance(content, str):
                body = content
            elif isinstance(content, list):
                body = "".join(
                    inner.get("text", "") for inner in content
                    if isinstance(inner, dict) and isinstance(inner.get("text"), str)
                )
            else:
                continue
            name = name_by_id.get(tcid or "", "")
            if name in _READ_TOOLS:
                n_reads += 1
                if len(body.encode("utf-8", errors="replace")) >= large_file_bytes:
                    n_large_reads += 1
    score = max(0.0, 1.0 - 0.20 * n_large_reads)
    return score, {"n_reads": n_reads, "n_large_reads": n_large_reads}


# -----------------------------------------------------------------------------
# Sub-score: grounding (predicted answers appear in tool output)
# -----------------------------------------------------------------------------

def _grounding(
    transcript: list,
    predicted: list[tuple[str, int]],
) -> tuple[float, dict]:
    """For each predicted (file, line) check whether evidence of that
    file:line appears in some tool result. Accept any of:

    1. `<file>:<line>` (or `<file>:<line±2>`) substring anywhere
    2. `<basename>:<line>` (or `<basename>:<line±2>`) substring anywhere
    3. file basename appears in some result + line appears as `<line>:`
       at start of any line in any tool result (the rg `-n` format with
       a separate file-context grep).

    Returns fraction matched."""
    if not predicted:
        return 0.0, {"reason": "no predictions"}
    bodies = _iter_tool_results(transcript)
    haystack = "\n".join(bodies)
    matched = 0
    misses: list[str] = []
    for f, ln in predicted:
        basename = f.split("/")[-1]
        ok = False
        # Direct file:line (or basename:line) check, ±2 tolerance.
        for tline in range(ln - 2, ln + 3):
            for hint in (f"{f}:{tline}", f"{basename}:{tline}",
                         f"repo/{f}:{tline}"):
                if hint in haystack:
                    ok = True
                    break
            if ok:
                break
        # Fallback: basename appears AND `<line>:` (rg `-n` line prefix)
        # appears somewhere in tool results — covers the "narrow grep
        # returning just line: content" pattern.
        if not ok:
            if basename in haystack or f in haystack:
                for tline in range(ln - 2, ln + 3):
                    if f"\n{tline}:" in haystack or haystack.startswith(f"{tline}:"):
                        ok = True
                        break
        if ok:
            matched += 1
        else:
            misses.append(f"{f}:{ln}")
    score = matched / len(predicted)
    return score, {"matched": matched, "total": len(predicted), "misses": misses[:3]}


# -----------------------------------------------------------------------------
# Sub-score: format_compliance (file:line regex)
# -----------------------------------------------------------------------------

def _format_compliance_answer(transcript: list) -> tuple[float, dict]:
    """Fraction of file:line tokens in the final answer that match the
    canonical regex. We expect AT LEAST ONE file:line pair in the final
    answer; if zero, format=0. Otherwise 1.0."""
    last = _last_assistant_text(transcript)
    pairs = parse_predicted_pairs(last)
    if not pairs:
        return 0.0, {"reason": "no file:line in final answer"}
    return 1.0, {"n_pairs": len(pairs)}


# -----------------------------------------------------------------------------
# Composite
# -----------------------------------------------------------------------------

def score_rollout(transcript: list, workdir: str, task: dict) -> dict:
    line_tol = int(task.get("line_tol", 2))
    target_bytes = int(task.get("target_bytes", 200))

    outcome_val, outcome_diag = _outcome(transcript, task, line_tol=line_tol)
    eff_val, eff_diag = _efficiency(transcript, target_bytes)
    tc_val, tc_diag = _tool_choice(transcript)
    fmt_val, fmt_diag = _format_compliance_answer(transcript)

    last = _last_assistant_text(transcript)
    predicted = parse_predicted_pairs(last)
    grd_val, grd_diag = _grounding(transcript, predicted)

    # Composite combines an additive "search quality" term with a
    # multiplicative grounding factor (so guessing without searching
    # can never score above 0.40 even with a lucky correct answer).
    #
    #   agentic        = 0.50·efficiency + 0.30·tool_choice + 0.20·format
    #   grounding_fact = 0.40 + 0.60·grounding   # 0.40 floor; 1.0 ceiling
    #   composite      = outcome × grounding_fact × agentic
    agentic = 0.50 * eff_val + 0.30 * tc_val + 0.20 * fmt_val
    grounding_factor = 0.40 + 0.60 * grd_val
    composite = outcome_val * grounding_factor * agentic

    # Diagnostics (prefixed with `_` so eval summary ignores them).
    n_tool_calls = sum(
        len(_tool_calls_in(m)) for _, m in _iter_messages(transcript)
        if m.get("role") == "assistant"
    )

    return {
        "outcome": outcome_val,
        "efficiency": eff_val,
        "tool_choice": tc_val,
        "grounding": grd_val,
        "format_compliance": fmt_val,
        "composite": composite,
        "_n_tool_calls": n_tool_calls,
        "_bytes_consumed": eff_diag["bytes_consumed"],
        "_target_bytes": eff_diag["target_bytes"],
        "_n_large_reads": tc_diag.get("n_large_reads", 0),
        "_n_reads": tc_diag.get("n_reads", 0),
        "_outcome_diag": outcome_diag,
        "_grounding_diag": grd_diag,
        "_format_diag": fmt_diag,
    }


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: rubric.py <transcript.jsonl> <workdir> <task.json>",
              file=sys.stderr)
        sys.exit(2)
    transcript = []
    with open(sys.argv[1]) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                transcript.append(json.loads(line))
            except Exception:
                pass
    task = json.loads(Path(sys.argv[3]).read_text())
    out = score_rollout(transcript, sys.argv[2], task)
    print(json.dumps(out, indent=2, default=str))
