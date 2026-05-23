"""Composite reward function for pi-context-aware-edits (v0).

Multiplicative-gate composite:

    composite = outcome × format_compliance × (
        0.40 · convention_consistency
      + 0.20 · read_before_edit
      + 0.15 · no_redundant_imports
      + 0.10 · no_style_drift
      + 0.15                             # base floor
    )

`convention_consistency` is the average of 4-6 binary checks (one per
convention category named in task.expected_conventions_in_edit). Each
check inspects the agent's edit against the file's existing style.
"""
from __future__ import annotations
import json
import re
from typing import Any

RUBRIC_VERSION = "v0"


# ---------------------------------------------------------------------------
# Transcript helpers
# ---------------------------------------------------------------------------

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


def _edited_files_from_transcript(transcript) -> dict[str, str]:
    """Map of file_path → final content the agent wrote. Last write wins."""
    out = {}
    for _, name, args in _calls_with_idx(transcript):
        if name in ("write", "edit", "replace"):
            p = args.get("path")
            c = args.get("content")
            if isinstance(p, str) and isinstance(c, str):
                out[p] = c
    return out


def _edited_files(rollout_or_transcript) -> dict[str, str]:
    """Map file_path -> final content, preferring captured sandbox state.

    Pi's edit tool usually stores patch blocks in an `edits` argument rather
    than whole-file content. The rollout harness captures final file contents
    so convention checks can score those edits directly.
    """
    if isinstance(rollout_or_transcript, dict):
        task = rollout_or_transcript.get("task") or {}
        initial = task.get("init_files") or {}
        final = rollout_or_transcript.get("final_files") or {}
        changed = {
            path: content
            for path, content in final.items()
            if initial.get(path) != content
        }
        if changed:
            return changed
        return _edited_files_from_transcript(rollout_or_transcript.get("transcript") or [])
    return _edited_files_from_transcript(rollout_or_transcript)


def _edited_paths(transcript) -> list[str]:
    out = []
    for _, name, args in _calls_with_idx(transcript):
        if name not in ("write", "edit", "replace"):
            continue
        p = args.get("path")
        if isinstance(p, str):
            out.append(p)
    return out


def _same_or_suffix(a: str, b: str) -> bool:
    a = a.strip()
    b = b.strip()
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


def _is_read(name, args):
    if name in ("read", "cat", "open"):
        return True
    if name in ("bash", "shell", "exec"):
        cmd = (args.get("command") or args.get("cmd") or "")
        return bool(re.search(r"^\s*(cat|less|head|tail|rg|grep|sed -n)\b", cmd))
    return False


def _read_paths(transcript) -> list[str]:
    out = []
    for _, name, args in _calls_with_idx(transcript):
        if not _is_read(name, args):
            continue
        p = args.get("path")
        if isinstance(p, str):
            out.append(p)
            continue
        cmd = (args.get("command") or args.get("cmd") or "")
        m = re.search(r"\b(?:cat|less|head|tail|grep|sed -n)\s+\S*\s*([\w./-]+)", cmd)
        if m:
            out.append(m.group(1))
    return out


# ---------------------------------------------------------------------------
# Per-convention checkers
# ---------------------------------------------------------------------------

def _check_naming_case(edit: str, target_case: str) -> float:
    # Find function/method defs in the edit
    if "def " in edit:
        names = re.findall(r"def\s+(\w+)\s*\(", edit)
    elif "fn " in edit:
        names = re.findall(r"\bfn\s+(\w+)\s*[\(<]", edit)
    elif "func " in edit:
        names = re.findall(r"\bfunc\s+(?:\([^)]*\)\s+)?(\w+)\s*\(", edit)
    else:
        return 0.5
    if not names:
        return 0.5
    def _is_snake(s):
        return s.islower() and "_" in s or (s.islower() and len(s) <= 8)
    def _is_camel(s):
        return s and s[0].islower() and any(c.isupper() for c in s[1:])
    def _is_pascal(s):
        return s and s[0].isupper()
    if target_case == "snake_case":
        ok = sum(1 for n in names if _is_snake(n))
    elif target_case == "camelCase":
        ok = sum(1 for n in names if _is_camel(n))
    elif target_case in ("PascalCase", "PascalCase_pub_camel_priv"):
        ok = sum(1 for n in names if _is_pascal(n) or _is_camel(n))
    else:
        return 0.5
    return ok / len(names)


def _check_type_annotations(edit: str, target: str) -> float:
    if target == "absent":
        return 1.0 if "->" not in edit and ": " not in re.sub(r":\s*\n", "\n", edit) else 0.5
    if target == "strict":
        defs = re.findall(r"def\s+\w+\([^)]*\)(\s*->\s*\S+)?", edit)
        if not defs:
            defs2 = re.findall(r"fn\s+\w+\([^)]*\)\s*->\s*\S+", edit)
            return 1.0 if defs2 else 0.5
        annotated = sum(1 for d in defs if d.strip())
        return annotated / max(1, len(defs))
    return 0.5


def _check_logging_style(edit: str, target: str) -> float:
    if target == "logging":
        return 1.0 if re.search(r"\blogger\.|\blogging\.", edit) else (0.0 if "print(" in edit else 0.5)
    if target == "print":
        return 1.0 if "print(" in edit else (0.0 if re.search(r"\blogger\.|\blogging\.", edit) else 0.5)
    if target == "log_crate":
        return 1.0 if re.search(r"log::|tracing::", edit) else 0.5
    if target == "log_package":
        return 1.0 if re.search(r"log\.(Print|Fatal|Errorf)", edit) else 0.5
    return 0.5


def _check_error_handling(edit: str, target: str) -> float:
    if target == "try_except":
        return 1.0 if re.search(r"\btry:\s", edit) and "except" in edit else 0.0
    if target == "raise_only":
        return 1.0 if "raise" in edit and "try:" not in edit else 0.5
    if target == "result":
        return 1.0 if "Result<" in edit or "?;" in edit else 0.5
    if target == "explicit_err":
        return 1.0 if re.search(r"if\s+err\s*!=\s*nil", edit) or "return.*err" in edit else 0.5
    return 0.5


def _check_comment_style(edit: str, target: str) -> float:
    if target == "docstrings":
        return 1.0 if re.search(r'"""|/\*\*|///', edit) else 0.0
    if target == "inline":
        return 1.0 if re.search(r"(?m)^\s*#\s", edit) and '"""' not in edit else 0.5
    if target == "godoc":
        return 1.0 if re.search(r"(?m)^//\s+[A-Z]", edit) else 0.5
    if target == "minimal":
        return 1.0 if not re.search(r'"""', edit) else 0.5
    return 0.5


def _check_import_style(edit: str, target: str) -> float:
    # We can't really judge import style from just the edited file without
    # context; award 1.0 if no new imports were added (the existing style is
    # preserved by default), 0.5 otherwise.
    if re.search(r"(?m)^\s*(import |from )", edit):
        return 0.6
    return 1.0


CONVENTION_CHECKERS = {
    "naming_case": _check_naming_case,
    "type_annotations": _check_type_annotations,
    "logging_style": _check_logging_style,
    "error_handling": _check_error_handling,
    "comment_style": _check_comment_style,
    "import_style": _check_import_style,
}


# ---------------------------------------------------------------------------
# Sub-scores
# ---------------------------------------------------------------------------

def _outcome(rollout) -> float:
    """1.0 iff outcome_passed (set by rollout.py after running verify_cmd)."""
    v = rollout.get("outcome_passed")
    return 1.0 if v is True else 0.0


def _format_compliance(rollout) -> float:
    final = (rollout.get("format_text") or _final_text(rollout.get("transcript") or [])).strip().lower()
    if not final:
        return 0.0
    score = 0.0
    # Mentions the file modified
    edited = _edited_files(rollout)
    if edited:
        any_match = False
        for fp in edited:
            if fp.lower() in final or fp.lower().split("/")[-1] in final:
                any_match = True
                break
        if any_match:
            score += 0.5
    # Mentions a convention preserved
    if re.search(r"\b(style|convention|consistent|match|preserved|naming|imports?|type|logging)\b", final):
        score += 0.5
    return min(1.0, score)


def _convention_consistency(rollout) -> float:
    task = rollout.get("task") or {}
    expected = task.get("expected_conventions_in_edit") or {}
    if not expected:
        return 0.5
    edits = _edited_files(rollout)
    if not edits:
        return 0.0
    # Concatenate all edits — convention applies across file
    edit_text = "\n".join(edits.values())
    scores = []
    for k, v in expected.items():
        checker = CONVENTION_CHECKERS.get(k)
        if checker is None:
            continue
        scores.append(checker(edit_text, v))
    if not scores:
        return 0.5
    return sum(scores) / len(scores)


def _read_before_edit(rollout) -> float:
    transcript = rollout.get("transcript") or []
    calls = _calls_with_idx(transcript)
    first_edit_idx = None
    for i, n, a in calls:
        if n in ("write", "edit", "replace"):
            first_edit_idx = i
            break
    if first_edit_idx is None:
        return 0.0
    # Was there a read of any source file before that edit?
    read_paths = []
    for i, n, a in calls:
        if i >= first_edit_idx:
            break
        if _is_read(n, a):
            p = a.get("path") or ""
            read_paths.append(p)
            cmd = (a.get("command") or a.get("cmd") or "")
            read_paths.extend(re.findall(r"[\w./-]+\.(?:py|rs|go|js|ts)", cmd))
    # Did the read cover at least one of the edited files?
    edits = _edited_paths(transcript)
    for ef in edits:
        for rp in read_paths:
            if _same_or_suffix(rp, ef):
                return 1.0
    return 0.0 if not read_paths else 0.4


def _no_redundant_imports(rollout) -> float:
    edits = _edited_files(rollout)
    if not edits:
        return 1.0
    # Trivial heuristic: did the agent add an import for a function that
    # already exists in the same file? We approximate by checking if any
    # new import line names something present in the original module text.
    for path, content in edits.items():
        for m in re.findall(r"(?m)^\s*from\s+(\S+)\s+import\s+(\w+)", content):
            module, sym = m
            if sym in content[: content.find(f"import {sym}")]:
                return 0.5
    return 1.0


def _no_style_drift(rollout) -> float:
    """Adjacent functions in the same file are stylistically uniform.

    Crude: if the file's existing style (e.g. uses docstrings) is broken
    by the new addition (e.g. comments instead of docstrings).
    """
    task = rollout.get("task") or {}
    init = (task.get("init_files") or {})
    edits = _edited_files(rollout)
    drift = 0
    total = 0
    for path, new_content in edits.items():
        orig = init.get(path)
        if not orig:
            continue
        total += 1
        had_docstring = '"""' in orig or "///" in orig or "/**" in orig
        has_docstring = '"""' in new_content or "///" in new_content or "/**" in new_content
        if had_docstring and not has_docstring:
            drift += 1
    if total == 0:
        return 1.0
    return float(max(0.0, 1.0 - drift / total))


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

W_CONV = 0.40
W_READ = 0.20
W_NO_REDUND = 0.15
W_NO_DRIFT = 0.10
BASE = 0.15


def score_one(rollout: dict) -> dict[str, Any]:
    outcome = _outcome(rollout)
    fmt = _format_compliance(rollout)
    conv = _convention_consistency(rollout)
    read = _read_before_edit(rollout)
    no_redund = _no_redundant_imports(rollout)
    no_drift = _no_style_drift(rollout)
    process = (
        W_CONV * conv
        + W_READ * read
        + W_NO_REDUND * no_redund
        + W_NO_DRIFT * no_drift
        + BASE
    )
    composite = outcome * fmt * process
    return {
        "composite": float(max(0.0, min(1.0, composite))),
        "outcome": float(outcome),
        "format_compliance": float(fmt),
        "convention_consistency": float(conv),
        "read_before_edit": float(read),
        "no_redundant_imports": float(no_redund),
        "no_style_drift": float(no_drift),
    }


CHEAT_PROBES = [
    "always-read-then-edit-ignore-contents (mitigation: conv-consistency punishes wrong style)",
    "copy-paste-existing-function (mitigation: outcome checks the actual function works)",
    "constant-import-line (mitigation: tasks span multiple profiles → averages low)",
    "skip-edit-claim-already-exists (mitigation: outcome verifies workspace state)",
]
