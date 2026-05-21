"""Composite reward function for pi-search-then-read (v0).

Multiplicative-gate composite:

    composite = outcome × format_compliance × (
        0.35 · search_efficiency
      + 0.25 · search_before_read
      + 0.20 · no_redundant_reads
      + 0.10 · precision_of_first_read
      + 0.10                             # base
    )

`outcome` = 1.0 iff the final assistant message contains the gold
answer for the query.
`format_compliance` = 1.0 iff the answer cites a file:line.
"""
from __future__ import annotations
import json
import re
from typing import Any

RUBRIC_VERSION = "v0"


def _iter_messages(transcript):
    for ev in transcript or []:
        if isinstance(ev, dict) and ev.get("type") == "message":
            msg = ev.get("message")
            if isinstance(msg, dict):
                yield msg


def _tool_calls(msg):
    content = msg.get("content")
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == "toolCall"]


def _final_text(transcript):
    final = ""
    for msg in _iter_messages(transcript):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or []
        text = "".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
        if text.strip():
            final = text.strip()
    return final


def _calls_with_idx(transcript):
    out = []
    for i, msg in enumerate(_iter_messages(transcript)):
        if msg.get("role") != "assistant":
            continue
        for tc in _tool_calls(msg):
            args = tc.get("input") or tc.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            out.append((i, tc.get("name", ""), args))
    return out


SEARCH_PAT = re.compile(r"\b(grep|rg|ag|find|locate|fd|ack)\b")
READ_PAT = re.compile(r"\b(cat|less|head|tail|sed -n)\b")


def _is_search_call(name, args):
    if name in ("bash", "shell", "exec"):
        cmd = (args.get("command") or args.get("cmd") or "")
        return bool(SEARCH_PAT.search(cmd))
    return False


def _is_read_call(name, args):
    """A read call is one that fetches file contents. Grep / find / rg are
    NOT reads (they're searches) — even if piped into `head` to truncate
    the result.
    """
    if name in ("read", "cat", "open"):
        return True
    if name in ("bash", "shell", "exec"):
        cmd = (args.get("command") or args.get("cmd") or "")
        if SEARCH_PAT.search(cmd):
            return False  # search-then-head pipelines are still searches
        return bool(READ_PAT.search(cmd))
    return False


CHARS_PER_LINE = 30


def _bytes_read(rollout) -> int:
    """Approximate chars read by inspecting each read call's args.

    Priority:
      1. If args has offset+limit (numeric) → limit * CHARS_PER_LINE.
      2. If sed -n 'A,Bp' in bash command → (B-A) * CHARS_PER_LINE.
      3. Else: assume whole-file read; use task.file_size_lines * CHARS_PER_LINE.
      4. Else: fall back to tool-result content length.

    This approach is robust against placeholder strings in calibration
    fixtures (which would otherwise fool a content-length count).
    """
    task = rollout.get("task") or {}
    full_chars = int(task.get("file_size_lines", 0)) * CHARS_PER_LINE
    transcript = rollout.get("transcript") or []
    total = 0
    msgs = list(_iter_messages(transcript))
    for i, msg in enumerate(msgs):
        if msg.get("role") != "assistant":
            continue
        for tc in _tool_calls(msg):
            args = tc.get("input") or tc.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            if not _is_read_call(tc.get("name", ""), args):
                continue
            chars = None
            off, lim = args.get("offset"), args.get("limit")
            if isinstance(off, int) and isinstance(lim, int):
                chars = lim * CHARS_PER_LINE
            else:
                cmd = (args.get("command") or args.get("cmd") or "")
                m = re.search(r"sed -n '(\d+),(\d+)p'", cmd)
                if m:
                    chars = (int(m.group(2)) - int(m.group(1)) + 1) * CHARS_PER_LINE
                elif _is_read_call(tc.get("name", ""), args):
                    chars = full_chars  # whole-file read fallback
            if chars is None:
                # Fall back to next tool result length.
                for j in range(i + 1, len(msgs)):
                    nm = msgs[j]
                    if nm.get("role") != "tool":
                        continue
                    for tr in nm.get("content") or []:
                        if isinstance(tr, dict) and tr.get("type") == "toolResult":
                            c = tr.get("content")
                            if isinstance(c, str):
                                chars = len(c)
                    break
            total += chars or 0
    return total


def _outcome(rollout) -> float:
    """Final reply contains the gold answer (case-insensitive substring)."""
    task = rollout.get("task") or {}
    gold = (task.get("gold_answer") or "").lower().strip()
    if not gold:
        return 0.0
    final = (rollout.get("format_text") or _final_text(rollout.get("transcript") or [])).lower()
    return 1.0 if gold in final else 0.0


def _format_compliance(rollout) -> float:
    final = (rollout.get("format_text") or _final_text(rollout.get("transcript") or [])).strip()
    if not final:
        return 0.0
    # Looking for file:line citation in the format "path:42" or "lib/x.py line 42"
    if re.search(r"\b[\w./_-]+\.py:\d+\b", final):
        return 1.0
    if re.search(r"\b[\w./_-]+\.py\b.*\bline\s+\d+", final, re.IGNORECASE):
        return 1.0
    if re.search(r"\bline\s+\d+\b.*\.py", final, re.IGNORECASE):
        return 0.8
    return 0.3 if ":" in final else 0.0


SMALL_FILE_THRESHOLD = 250


def _search_efficiency(rollout) -> float:
    """1 - (bytes_read / file_size_chars).

    For files <= SMALL_FILE_THRESHOLD lines, award full credit regardless
    (small files are fine to read whole).
    """
    task = rollout.get("task") or {}
    n_lines = int(task.get("file_size_lines", 0))
    if n_lines <= SMALL_FILE_THRESHOLD:
        return 1.0
    file_size_chars = n_lines * CHARS_PER_LINE
    br = _bytes_read(rollout)
    if file_size_chars <= 0:
        return 1.0
    eff = 1.0 - min(1.0, br / file_size_chars)
    return float(max(0.0, eff))


def _search_before_read(rollout) -> float:
    """1.0 iff a search call referencing the target symbol appears before
    any read of the target file.

    For small files (<= SMALL_FILE_THRESHOLD lines), award full credit
    regardless — a tiny file doesn't need pre-search.
    """
    task = rollout.get("task") or {}
    n_lines = int(task.get("file_size_lines", 0))
    if n_lines and n_lines <= SMALL_FILE_THRESHOLD:
        return 1.0
    target_symbol = (task.get("target_symbol") or "").lower()
    target_file = (task.get("target_file") or "").lower()
    transcript = rollout.get("transcript") or []
    calls = _calls_with_idx(transcript)
    first_read_idx = None
    for i, n, a in calls:
        if _is_read_call(n, a):
            p = a.get("path") or ""
            cmd = (a.get("command") or a.get("cmd") or "")
            if target_file in (p + " " + cmd).lower():
                first_read_idx = i
                break
    if first_read_idx is None:
        # No target-read at all → no chance to have searched first
        return 0.5
    for i, n, a in calls:
        if i >= first_read_idx:
            break
        if _is_search_call(n, a):
            cmd = (a.get("command") or a.get("cmd") or "").lower()
            if target_symbol and target_symbol in cmd:
                return 1.0
    return 0.0


def _no_redundant_reads(rollout) -> float:
    transcript = rollout.get("transcript") or []
    sigs = []
    for _, n, a in _calls_with_idx(transcript):
        if not _is_read_call(n, a):
            continue
        # Read signature: path + (offset, limit) or full command
        s = json.dumps([n, a.get("path", ""), a.get("offset", ""), a.get("limit", ""), a.get("command", "")], sort_keys=True, default=str)
        sigs.append(s)
    if not sigs:
        return 1.0
    dups = len(sigs) - len(set(sigs))
    return float(max(0.0, 1.0 - dups / max(1, len(sigs))))


def _precision_of_first_read(rollout) -> float:
    """If the first target-file read lands in the gold window, score 1;
    else score by overlap fraction (line range).
    """
    task = rollout.get("task") or {}
    target_file = (task.get("target_file") or "").lower()
    gold_start = int(task.get("gold_window_line_start", 0))
    gold_end = int(task.get("gold_window_line_end", 0))
    if gold_end <= gold_start:
        return 0.5
    transcript = rollout.get("transcript") or []
    for _, n, a in _calls_with_idx(transcript):
        if not _is_read_call(n, a):
            continue
        p = (a.get("path") or "").lower()
        cmd = (a.get("command") or a.get("cmd") or "").lower()
        if target_file not in (p + " " + cmd):
            continue
        # Try to extract offset/limit from `read` args or sed pattern
        off = a.get("offset")
        lim = a.get("limit")
        if isinstance(off, int) and isinstance(lim, int):
            read_start, read_end = off, off + lim
        else:
            m = re.search(r"sed -n '(\d+),(\d+)p'", cmd)
            if m:
                read_start, read_end = int(m.group(1)), int(m.group(2))
            else:
                # Reading whole file → low precision for large files
                return 0.0 if int(task.get("file_size_lines", 0)) >= 800 else 0.5
        # Compute overlap
        overlap = max(0, min(read_end, gold_end) - max(read_start, gold_start))
        window = gold_end - gold_start
        return float(min(1.0, overlap / max(1, window)))
    return 0.0


# Composite
W_EFF = 0.35
W_BEFORE = 0.25
W_NO_REDUND = 0.20
W_PRECISION = 0.10
BASE = 0.10


def score_one(rollout: dict) -> dict[str, Any]:
    outcome = _outcome(rollout)
    fmt = _format_compliance(rollout)
    eff = _search_efficiency(rollout)
    before = _search_before_read(rollout)
    no_redund = _no_redundant_reads(rollout)
    prec = _precision_of_first_read(rollout)
    process = (
        W_EFF * eff
        + W_BEFORE * before
        + W_NO_REDUND * no_redund
        + W_PRECISION * prec
        + BASE
    )
    composite = outcome * fmt * process
    return {
        "composite": float(max(0.0, min(1.0, composite))),
        "outcome": float(outcome),
        "format_compliance": float(fmt),
        "search_efficiency": float(eff),
        "search_before_read": float(before),
        "no_redundant_reads": float(no_redund),
        "precision_of_first_read": float(prec),
    }


CHEAT_PROBES = [
    "always-grep-even-on-small-files (mitigation: efficiency caps small files at full credit)",
    "grep-any-keyword (mitigation: search must reference query symbol)",
    "read-many-small-files (mitigation: per-file averaged, not aggregate)",
    "first-30-lines-everywhere (mitigation: outcome verifies actual answer)",
]
