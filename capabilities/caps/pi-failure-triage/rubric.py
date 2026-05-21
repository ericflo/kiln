"""Reward function for pi-failure-triage — v1 (multi-component).

The capability: given a failing test or stack trace in a small project
with a planted bug, the agent reproduces the failure, isolates the
root cause, and proposes a fix that makes both the visible test AND a
held-out related test pass — without papering over symptoms.

Composite design philosophy:
  - `outcome` is a HARD MULTIPLICATIVE FLOOR. If the visible test
    doesn't pass after the agent's edits, composite = 0. No reward for
    "almost solved" stack-trace prose.
  - `held_out_passes` is the LARGEST agentic weight (0.30) — it is the
    whole point of the cap. Symptom fixes pass `outcome` but fail
    here.
  - `fix_localised_correctly` rewards touching the gold region; this
    discourages the "fix-everything-everywhere" reward shortcut.
  - `no_test_mutation` makes editing the test illegal. Oracle re-runs
    from a clean test directory anyway, but this provides per-rollout
    gradient.
  - `no_blanket_except` AST-checks the diff to penalize the cheapest
    symptom-fix pattern.
  - `format_compliance` requires a structured fix summary so the
    output is machine-readable.
  - `reproduced_before_fixing` rewards running the failing test BEFORE
    editing — the literal "debugging" loop we want to instill.
  - `diff_minimality` is a tiebreaker: smaller, focused diffs over
    sprawling ones.
  - `no_dependency_changes` penalizes the "I'll just upgrade the lib"
    shortcut.

Composite (v2 — multiplicative format gate):
    process = (
        0.30 * held_out_passes
      + 0.15 * fix_localised_correctly
      + 0.10 * no_test_mutation
      + 0.10 * no_blanket_except
      + 0.10 * reproduced_before_fixing
      + 0.05 * diff_minimality
      + 0.05 * no_dependency_changes
      + 0.15 * base
    )
    composite = outcome × format_compliance × process

Range: [0, 1]. Both outcome and format are MULTIPLICATIVE GATES (round
2 — see capability.md). The v1 additive rubric trapped +12.5pp of format
gain that round 1 actually produced; v2 makes that gain visible as
composite movement. Base term (0.15) is the floor inside process for any
correct visible-test fix.

Adversarial design (see capability.md §Adversarial design (§0)):
  - Delete the failing test → outcome=0 (oracle re-runs from clean dir).
  - try/except around the failing line → outcome=1, held_out=0,
    no_blanket_except penalised.
  - Edit the test → no_test_mutation=0.
  - Fix everything everywhere → fix_localised_correctly low.
  - Read the held-out test → out of agent workspace; scaffold doesn't
    mount it.
  - Upgrade a dependency → no_dependency_changes penalised.
  - Add `# type: ignore` to silence the error → no_blanket_except
    penalised (catches `# noqa` too).
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Transcript helpers
# ---------------------------------------------------------------------------


def _iter_messages(transcript: list[dict]):
    """Yield (event_idx, message) tuples from a pi session JSONL.

    Pi session events look like:
      {"type": "message", "message": {"role": "...", "content": [...]}}
    """
    for i, event in enumerate(transcript):
        if not isinstance(event, dict):
            continue
        if event.get("type") != "message":
            continue
        msg = event.get("message")
        if isinstance(msg, dict):
            yield i, msg


def _tool_calls_in(msg: dict) -> list[dict]:
    """Return the tool calls (if any) in an assistant message."""
    content = msg.get("content")
    if not isinstance(content, list):
        return []
    out = []
    for b in content:
        if isinstance(b, dict) and b.get("type") == "toolCall":
            out.append(b)
    return out


def _tool_results_in(msg: dict) -> list[dict]:
    """Return tool result text blocks. Handles both pi formats:
    - pi 0.75.1: role="tool", content=[{type:"toolResult", content:...}]
    - pi 0.75.3: role="toolResult", content=[{type:"text", text:...}]
    """
    content = msg.get("content")
    if not isinstance(content, list):
        return []
    out = []
    for b in content:
        if isinstance(b, dict) and b.get("type") in ("toolResult", "text"):
            out.append(b)
    return out


def _final_assistant_text(transcript: list[dict]) -> str:
    """Return the text content of the last assistant message (no tool calls)."""
    last_text = ""
    for _, msg in _iter_messages(transcript):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        text_parts: list[str] = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                text_parts.append(b.get("text", ""))
        if text_parts:
            last_text = "".join(text_parts)
    return last_text


def _bash_commands_run(transcript: list[dict]) -> list[str]:
    """All shell commands run via the bash tool, in order."""
    out: list[str] = []
    for _, msg in _iter_messages(transcript):
        if msg.get("role") != "assistant":
            continue
        for tc in _tool_calls_in(msg):
            if tc.get("name") in ("bash", "shell", "run", "execute"):
                args = tc.get("input") or tc.get("arguments") or {}
                if isinstance(args, dict):
                    cmd = args.get("command") or args.get("cmd") or args.get("script")
                    if isinstance(cmd, str):
                        out.append(cmd)
    return out


# ---------------------------------------------------------------------------
# Outcome: run the visible test inside the workdir
# ---------------------------------------------------------------------------


def _run_test(workdir: str, test_cmd: list[str], timeout: int = 30) -> tuple[bool, str]:
    """Run a test command in workdir. Returns (passed, stdout+stderr tail)."""
    try:
        proc = subprocess.run(
            test_cmd,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except FileNotFoundError as e:
        return False, f"COMMAND_NOT_FOUND: {e}"
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out[-600:]


def _outcome(workdir: str, task: dict) -> tuple[float, str]:
    """Run the visible test inside the workdir. Returns (score, tail)."""
    cmd = task.get("visible_test_cmd")
    if not cmd:
        return 0.0, "NO_VISIBLE_TEST_CMD"
    passed, tail = _run_test(workdir, cmd, timeout=task.get("test_timeout_s", 30))
    return (1.0 if passed else 0.0), tail


def _held_out_passes(workdir: str, task: dict) -> tuple[float, str]:
    """Mount the held-out test into the workdir and run it.

    Critical: the held-out test must NOT have been visible to the agent
    during the rollout. We write it now (after rollout completion) and
    immediately execute. Held-out is described by:
      task["held_out_test_path"]    — relative path under workdir
      task["held_out_test_content"] — full file content (string)
      task["held_out_test_cmd"]     — argv to invoke the test
    """
    path = task.get("held_out_test_path")
    content = task.get("held_out_test_content")
    cmd = task.get("held_out_test_cmd")
    if not (path and content and cmd):
        return 0.0, "NO_HELD_OUT_DATA"
    held_out_file = Path(workdir) / path
    held_out_file.parent.mkdir(parents=True, exist_ok=True)
    held_out_file.write_text(content)
    passed, tail = _run_test(workdir, cmd, timeout=task.get("test_timeout_s", 30))
    return (1.0 if passed else 0.0), tail


# ---------------------------------------------------------------------------
# Diff-based sub-scores
# ---------------------------------------------------------------------------


_TEST_PATH_RE = re.compile(r"(^|/)tests?/")


def _collect_files(workdir: str) -> dict[str, str]:
    """Return {relpath -> content} for all python/rust source under workdir,
    excluding venv / .pytest_cache / __pycache__ / .git."""
    root = Path(workdir)
    out: dict[str, str] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if any(part.startswith(".") for part in rel.split("/")):
            continue
        if "__pycache__" in rel or ".pytest_cache" in rel:
            continue
        if rel.endswith(".pyc"):
            continue
        if p.stat().st_size > 200_000:
            # Skip giant files (large binaries / generated artifacts).
            continue
        try:
            out[rel] = p.read_text(errors="ignore")
        except OSError:
            continue
    return out


def _baseline_files(task: dict) -> dict[str, str]:
    """Reconstruct the scaffolded file map for the task. Mirror of what
    `task_scaffold.init_workdir` writes."""
    files = task.get("workspace_files") or {}
    if isinstance(files, dict):
        return {k: v for k, v in files.items() if isinstance(v, str)}
    return {}


def _diff_paths(baseline: dict[str, str], final: dict[str, str]) -> dict[str, str]:
    """Return {relpath -> tag} where tag is one of:
        "added", "removed", "modified", "unchanged"
    Only modified / added / removed are interesting downstream.
    """
    tags: dict[str, str] = {}
    for k in baseline.keys() | final.keys():
        if k not in baseline:
            tags[k] = "added"
        elif k not in final:
            tags[k] = "removed"
        elif baseline[k] != final[k]:
            tags[k] = "modified"
        else:
            tags[k] = "unchanged"
    return tags


def _no_test_mutation(diff: dict[str, str]) -> float:
    """1.0 iff no file under tests/ was added/removed/modified."""
    for path, tag in diff.items():
        if tag == "unchanged":
            continue
        if _TEST_PATH_RE.search(path):
            return 0.0
    return 1.0


def _fix_localised_correctly(diff: dict[str, str], task: dict) -> float:
    """1.0 iff the only modified non-test file is the gold fix region (or a
    file in the gold module). 0.5 if neighbour file. 0 if only test or
    unrelated files."""
    gold = task.get("gold_fix_region") or {}
    gold_file = gold.get("file")
    gold_module = gold.get("module", "")
    touched_non_test: list[str] = []
    for path, tag in diff.items():
        if tag == "unchanged":
            continue
        if _TEST_PATH_RE.search(path):
            continue
        touched_non_test.append(path)
    if not touched_non_test:
        return 0.0
    if gold_file and gold_file in touched_non_test:
        # If they ALSO touched many other files, partial credit.
        return 1.0 if len(touched_non_test) <= 2 else 0.6
    if gold_module:
        # 0.5 if any touched file is in the gold module's parent dir.
        for path in touched_non_test:
            if path.startswith(gold_module + "/") or path.startswith(gold_module):
                return 0.5
    return 0.0


# Patterns we want to penalize: blanket exceptions + lint silencers.
_BLANKET_EXCEPT_RE = re.compile(
    r"except\s*(?:Exception\s*(?:as\s+\w+)?)?\s*:",
)
_BARE_EXCEPT_RE = re.compile(r"except\s*:")
_NOQA_RE = re.compile(r"#\s*noqa\b", re.IGNORECASE)
_TYPE_IGNORE_RE = re.compile(r"#\s*type\s*:\s*ignore", re.IGNORECASE)
_PRAGMA_NO_COVER_RE = re.compile(r"#\s*pragma\s*:\s*no\s*cover", re.IGNORECASE)
_PYLINT_DISABLE_RE = re.compile(r"#\s*pylint\s*:\s*disable", re.IGNORECASE)


def _no_blanket_except(
    baseline: dict[str, str],
    final: dict[str, str],
    diff: dict[str, str],
) -> tuple[float, dict]:
    """Penalize new blanket-except / bare-except / noqa / type:ignore /
    pragma:no cover / pylint:disable introduced in the diff.

    Score = max(0, 1 - sum_of_penalties).
      - +0.5 penalty per new `except Exception` / `except:`
      - +0.3 penalty per new `# noqa`, `# type: ignore`, `# pragma: no cover`,
        `# pylint: disable`
    """
    new_blanket = 0
    new_bare = 0
    new_noqa = 0
    new_type_ignore = 0
    new_pragma = 0
    new_pylint = 0
    for path, tag in diff.items():
        if tag != "modified" and tag != "added":
            continue
        if not path.endswith(".py"):
            continue
        before = baseline.get(path, "")
        after = final.get(path, "")
        # Compare match counts; positive deltas count as "new".
        before_bare = len(_BARE_EXCEPT_RE.findall(before))
        after_bare = len(_BARE_EXCEPT_RE.findall(after))
        new_bare += max(0, after_bare - before_bare)

        before_blanket = len(_BLANKET_EXCEPT_RE.findall(before)) - before_bare
        after_blanket = len(_BLANKET_EXCEPT_RE.findall(after)) - after_bare
        new_blanket += max(0, after_blanket - before_blanket)

        new_noqa += max(0, len(_NOQA_RE.findall(after)) - len(_NOQA_RE.findall(before)))
        new_type_ignore += max(
            0,
            len(_TYPE_IGNORE_RE.findall(after))
            - len(_TYPE_IGNORE_RE.findall(before)),
        )
        new_pragma += max(
            0,
            len(_PRAGMA_NO_COVER_RE.findall(after))
            - len(_PRAGMA_NO_COVER_RE.findall(before)),
        )
        new_pylint += max(
            0,
            len(_PYLINT_DISABLE_RE.findall(after))
            - len(_PYLINT_DISABLE_RE.findall(before)),
        )
    penalty = (
        0.5 * (new_blanket + new_bare)
        + 0.3 * (new_noqa + new_type_ignore + new_pragma + new_pylint)
    )
    diagnostics = {
        "new_blanket_except": new_blanket,
        "new_bare_except": new_bare,
        "new_noqa": new_noqa,
        "new_type_ignore": new_type_ignore,
        "new_pragma_no_cover": new_pragma,
        "new_pylint_disable": new_pylint,
    }
    return max(0.0, 1.0 - penalty), diagnostics


def _no_dependency_changes(diff: dict[str, str]) -> float:
    """1.0 iff no manifest / lockfile was modified."""
    sentinels = {
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "Pipfile",
        "Pipfile.lock",
        "setup.py",
        "setup.cfg",
        "Cargo.toml",
        "Cargo.lock",
        "package.json",
        "package-lock.json",
        "poetry.lock",
    }
    for path, tag in diff.items():
        if tag == "unchanged":
            continue
        basename = path.rsplit("/", 1)[-1]
        if basename in sentinels:
            return 0.0
    return 1.0


def _diff_line_count(baseline: dict[str, str], final: dict[str, str], diff: dict[str, str]) -> int:
    """Approximate line count of the diff (added + removed)."""
    total = 0
    for path, tag in diff.items():
        if tag == "unchanged":
            continue
        before = baseline.get(path, "").splitlines()
        after = final.get(path, "").splitlines()
        # Coarse: line count delta + length-mismatch overlap.
        if tag == "added":
            total += len(after)
        elif tag == "removed":
            total += len(before)
        else:
            # Modified: count differing lines linearly.
            n = max(len(before), len(after))
            same = sum(
                1 for i in range(min(len(before), len(after))) if before[i] == after[i]
            )
            total += (n - same) * 2  # both sides of the diff
    return total


def _diff_minimality(
    baseline: dict[str, str], final: dict[str, str], diff: dict[str, str], task: dict
) -> tuple[float, int]:
    """Reward small, focused diffs.

    Scoring:
      score = 1.0 if <= expected lines
      score = linear ramp to 0 between expected and 5x expected
      score = 0 above 5x expected
    """
    expected = max(1, int(task.get("expected_diff_lines", 6)))
    actual = _diff_line_count(baseline, final, diff)
    if actual <= expected:
        return 1.0, actual
    if actual >= 5 * expected:
        return 0.0, actual
    # Linear ramp
    ramp = 1.0 - (actual - expected) / (4 * expected)
    return max(0.0, ramp), actual


# ---------------------------------------------------------------------------
# Transcript-based sub-scores
# ---------------------------------------------------------------------------


def _reproduced_before_fixing(transcript: list[dict], task: dict) -> tuple[float, dict]:
    """1.0 iff the agent ran the failing test (or any pytest/cargo test
    invocation) BEFORE any edit/write tool call.

    The 'edit' here means a `write` or `edit` tool call on a non-test file.
    If the agent only edited tests (or never edited), this fires 0.0 since
    no debugging behavior was demonstrated.
    """
    first_test_run_idx = None
    first_source_edit_idx = None
    test_indicators = ("pytest", "doctest", "unittest", "cargo test", "python -m pytest")
    for i, msg in _iter_messages(transcript):
        if msg.get("role") != "assistant":
            continue
        for tc in _tool_calls_in(msg):
            name = tc.get("name") or ""
            args = tc.get("input") or tc.get("arguments") or {}
            if not isinstance(args, dict):
                continue
            if name in ("bash", "shell", "run"):
                cmd = (
                    args.get("command")
                    or args.get("cmd")
                    or args.get("script")
                    or ""
                )
                if isinstance(cmd, str) and any(t in cmd for t in test_indicators):
                    if first_test_run_idx is None:
                        first_test_run_idx = i
            elif name in ("write", "edit", "str_replace"):
                path = args.get("path") or args.get("file") or args.get("file_path") or ""
                if isinstance(path, str) and not _TEST_PATH_RE.search(path):
                    if first_source_edit_idx is None:
                        first_source_edit_idx = i
    diag = {
        "first_test_run_idx": first_test_run_idx,
        "first_source_edit_idx": first_source_edit_idx,
    }
    if first_test_run_idx is None:
        return 0.0, diag
    if first_source_edit_idx is None:
        # Ran tests but never edited — they get partial credit (debugging
        # without fix, but at least exhibited reproduction behavior).
        return 0.5, diag
    return (1.0 if first_test_run_idx < first_source_edit_idx else 0.0), diag


_FIX_SUMMARY_RE = re.compile(
    r"(?:^|\n)\s*(?:Fix(?:ed)?|Root\s+cause)\s*:\s*[`'\"]?([^`'\"\n]+)[`'\"]?",
    re.IGNORECASE,
)
_FILE_FUNC_RE = re.compile(r"[\w/.-]+::?\s*[\w.]+|[\w/.-]+\s*\(?[\w.]*\)?")


def _format_compliance(transcript: list[dict], task: dict) -> tuple[float, dict]:
    """Final assistant text must include a one-line fix summary roughly
    matching `Fix: <file>::<function>: <one-line root cause>`.

    Score:
      1.0 — line matches Fix:/Root cause: prefix AND references the gold
            file or function name.
      0.5 — line matches Fix: prefix but doesn't reference gold file/function.
      0.0 — no such line.
    """
    text = _final_assistant_text(transcript)
    if not text:
        return 0.0, {"final_text_present": False}
    gold_file = (task.get("gold_fix_region") or {}).get("file", "")
    gold_func = (task.get("gold_fix_region") or {}).get("function", "")
    m = _FIX_SUMMARY_RE.search(text)
    if not m:
        return 0.0, {"final_text_has_fix_summary": False}
    summary_body = m.group(1).strip()
    refs_gold = (
        (gold_file and gold_file.rsplit("/", 1)[-1].split(".")[0].lower() in summary_body.lower())
        or (gold_func and gold_func.lower() in summary_body.lower())
    )
    return (1.0 if refs_gold else 0.5), {
        "final_text_has_fix_summary": True,
        "refs_gold": bool(refs_gold),
        "summary_body": summary_body[:200],
    }


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


def score_rollout(transcript: list[dict], workdir: str, task: dict) -> dict:
    """Score one (transcript, workdir, task) rollout.

    Returns a dict with all sub-scores plus the composite.
    """
    # 1. Visible test outcome — the hard floor.
    outcome_val, outcome_tail = _outcome(workdir, task)

    # 2. Snapshot files NOW, before we materialize the held-out test
    #    (otherwise the held-out file would show up as an "added test"
    #    and falsely trip no_test_mutation).
    baseline = _baseline_files(task)
    final = _collect_files(workdir)
    diff = _diff_paths(baseline, final)
    no_test_mut = _no_test_mutation(diff)
    fix_local = _fix_localised_correctly(diff, task)
    no_blanket, blanket_diag = _no_blanket_except(baseline, final, diff)
    no_dep = _no_dependency_changes(diff)
    diff_min_score, diff_size = _diff_minimality(baseline, final, diff, task)

    # 4. Held-out test passes — the whole point. Mutates workdir (writes
    #    the held-out test) so order matters; ran AFTER the diff snapshot.
    held_out_val, held_out_tail = _held_out_passes(workdir, task)

    # 4. Transcript-based.
    repro_score, repro_diag = _reproduced_before_fixing(transcript, task)
    fmt_score, fmt_diag = _format_compliance(transcript, task)

    # v2 — multiplicative format gate.
    #
    # Round 1 (additive): outcome × (0.30·held_out + 0.15·fix_local +
    # 0.10·no_test_mut + 0.10·no_blanket + 0.10·repro + 0.05·fmt +
    # 0.05·diff_min + 0.05·no_dep + 0.10·base)
    #
    # Round 1 result: format moved +12.5pp on this cap, composite barely
    # moved (+0.6pp). That signal was trapped by additive weighting on a
    # saturated outcome. v2 changes format from an additive sub-score
    # (weight 0.05) to a multiplicative gate on the whole composite, so a
    # +12.5pp format gain now produces a +12.5pp composite movement.
    #
    # See capability.md "## Round 2 improvement plan" for the rationale.
    process = (
        0.30 * held_out_val
        + 0.15 * fix_local
        + 0.10 * no_test_mut
        + 0.10 * no_blanket
        + 0.10 * repro_score
        + 0.05 * diff_min_score
        + 0.05 * no_dep
        + 0.15  # base (was 0.10 — bumped to keep process+base = 1.0 before gates)
    )
    # outcome and format are both multiplicative gates.
    composite = outcome_val * fmt_score * process

    n_tool_calls = sum(
        len(_tool_calls_in(m)) for _, m in _iter_messages(transcript) if m.get("role") == "assistant"
    )

    return {
        "outcome": outcome_val,
        "held_out_passes": held_out_val,
        "fix_localised_correctly": fix_local,
        "no_test_mutation": no_test_mut,
        "no_blanket_except": no_blanket,
        "reproduced_before_fixing": repro_score,
        "format_compliance": fmt_score,
        "diff_minimality": diff_min_score,
        "no_dependency_changes": no_dep,
        "composite": composite,
        "_outcome_tail": outcome_tail,
        "_held_out_tail": held_out_tail,
        "_diff_size_lines": diff_size,
        "_diff_paths": {k: v for k, v in diff.items() if v != "unchanged"},
        "_blanket_diag": blanket_diag,
        "_repro_diag": repro_diag,
        "_fmt_diag": fmt_diag,
        "_n_tool_calls": n_tool_calls,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    if len(sys.argv) != 4:
        print("usage: rubric.py <transcript.jsonl> <workdir> <task.json>", file=sys.stderr)
        sys.exit(2)
    transcript: list[dict] = []
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
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
