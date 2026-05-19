"""Pi-diff-patch-apply multi-component rubric (v1).

The capability under test: given a workspace and a unified diff (sometimes
clean, sometimes with offset drift, sometimes broken in subtle ways), the
agent must apply the patch, run the tests, and — if tests don't pass —
repair the workspace minimally without modifying unrelated files.

Design goals for this rubric:

1. **Outcome IS the load-bearing signal.** Tests passing at the end of
   the session is the single fact that distinguishes a real fix from
   confident-sounding hallucination. We treat `outcome` as a hard gate
   for full credit, but we DO award a *consolation* score when the
   model made directional progress (applied some hunks, ran the tests)
   so the GRPO gradient is not all-zero in early training. The
   all-zeros failure mode is the #1 killer of agentic-GRPO runs (see
   §10 of agentic-grpo-capability-creator/SKILL.md).

2. **Anti-shortcut by construction.** Each adversarial cheat from the
   capability.md §0 audit is mapped to a sub-score that *can't* be
   maxed by the cheat:

   - "Reset workspace and write gold files directly" → `no_unrelated_edits`
     and `minimality` together cap this: writing 50 lines fresh when the
     gold patch was 5 lines collapses minimality.
   - "Disable failing tests" → the oracle re-runs the test suite from
     the original test fixtures (not the agent's possibly-mutated test
     files); furthermore `no_unrelated_edits` punishes touching test
     files at all when the gold patch did not.
   - "Apply but never test" → `tested_before_done` is required for full
     credit even when tests happen to pass (which they usually won't
     if untested).
   - "Loop on git apply forever" → `repair_efficiency` decays past 2
     attempts and wall-clock budget caps the rollout.
   - "Spam ls / cat" → not directly penalized (cheap signal) but
     `tool_call_efficiency` (diagnostic only for v1) tracks it.

3. **Composite stays in [0, 1]** with a sharp 0.50 → 1.00 split between
   "test-passing rollouts" and "everything else". Inside the
   test-passing band we still differentiate by minimality / scope /
   repair / format / verification. This gives GRPO enough resolution
   to learn quality of the SOLVED case while still learning to solve
   in the FAILED case via consolation gradient.

4. **Deterministic + hermetic + fast.** All sub-scores read either
   the transcript (already on disk) or the workdir (run pytest once
   per rollout; ~2-5s on the typical seed). No LLM judge in the loop.
   Re-running the rubric on a stored workdir + transcript reproduces
   the score exactly.

API
---

    score_rollout(transcript: list[dict],
                  workdir: str,
                  task: dict) -> dict

Returns a dict with these keys:

    composite                       float in [0, 1]
    outcome                         0.0 or 1.0   (tests pass?)
    minimality                      float in [0, 1]
    no_unrelated_edits              float in [0, 1]
    repair_efficiency               float in [0, 1]
    format_compliance               float in [0, 1]
    tested_before_done              float in [0, 1]
    applied_fraction                float in [0, 1]   (consolation signal)
    no_loop                         float in [0, 1]   (1 - dup-call rate)
    tool_call_efficiency            float in [0, 1]   (diagnostic only)
    _n_tool_calls                   int
    _n_apply_attempts               int
    _n_test_runs                    int
    _failed_test_names              list[str]
    _touched_paths                  list[str]
    _patch_paths                    list[str]
    _final_diff_lines               int
    _gold_diff_lines                int

Each sub-score is in [0, 1]. The composite is:

    if outcome == 1:
        composite = 0.50                              # base credit for passing
                    + 0.20 * minimality
                    + 0.15 * no_unrelated_edits
                    + 0.10 * repair_efficiency
                    + 0.05 * format_compliance
                    (tested_before_done is a multiplicative discount: see below)
    else:
        # Consolation: gradient on directional progress.
        composite = 0.20 * applied_fraction
                  + 0.10 * tested_before_done
                  + 0.05 * format_compliance
                  + 0.05 * no_loop
        # capped at 0.40 so a failed-tests rollout cannot match a
        # passed-tests one even if all consolation components are 1.0.

Note the explicit ordering: 0.40 (consolation max) < 0.50 (passing
floor), so the model can never reward-hack by skipping the actual
patch-apply task.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


# ============================================================================
# Constants
# ============================================================================

# Composite weights — kept in one place for easy tuning later.
W_OUTCOME_BASE = 0.50  # constant reward for tests passing
W_MINIMALITY = 0.20
W_NO_UNRELATED = 0.15
W_REPAIR_EFF = 0.10
W_FORMAT = 0.05
# Sub-score weights for the FAIL-consolation path (must sum to <= 0.40):
CONSOLATION_W_APPLIED = 0.20
CONSOLATION_W_TESTED = 0.10
CONSOLATION_W_FORMAT = 0.05
CONSOLATION_W_NO_LOOP = 0.05
# Hard cap on consolation so it's strictly lower than the pass-floor.
CONSOLATION_CAP = 0.40
# When outcome=1 we apply a multiplicative `tested_before_done` discount.
# If the model lucked into passing without ever calling pytest, we still
# award the base 0.50 but withhold half the agentic sub-score budget.
TESTED_BEFORE_DONE_DISCOUNT_WHEN_MISSING = 0.5

# Minimality span: how many gold-diff multiples of "extra lines" zero
# out the minimality score. span=2 means a diff that is 3× the gold
# size scores 0.
MINIMALITY_SPAN = 2

# Repair-efficiency: after this many failed-apply attempts, decay sets in.
REPAIR_ATTEMPTS_FREE = 2
REPAIR_ATTEMPTS_DECAY_PER = 0.25  # lose 0.25 per extra attempt

# Tool-call efficiency (diagnostic): "expected" total tool calls.
TOOL_CALL_EXPECTED = 6  # 1 read patch + 1 apply + 1 test + buffer
TOOL_CALL_HARD_CAP = 20  # zero score at this many


# ============================================================================
# Transcript helpers (pi session JSONL shape)
# ============================================================================

def _iter_messages(transcript: Iterable[dict]) -> Iterable[tuple[int, dict]]:
    """Yield (idx, message_dict) for each message-event in a pi session."""
    for i, event in enumerate(transcript):
        if not isinstance(event, dict):
            continue
        if event.get("type") != "message":
            continue
        msg = event.get("message")
        if isinstance(msg, dict):
            yield i, msg


def _tool_calls_in(msg: dict) -> list[dict]:
    """Extract tool-call blocks from one message."""
    content = msg.get("content")
    if not isinstance(content, list):
        return []
    out: list[dict] = []
    for b in content:
        if isinstance(b, dict) and b.get("type") == "toolCall":
            out.append(b)
    return out


def _tool_results_in(msg: dict) -> list[dict]:
    """Extract tool-result blocks from one tool-role message."""
    content = msg.get("content")
    if not isinstance(content, list):
        return []
    out: list[dict] = []
    for b in content:
        if isinstance(b, dict) and b.get("type") == "toolResult":
            out.append(b)
    return out


def _assistant_text(msg: dict) -> str:
    """Concatenate text blocks of one assistant message."""
    content = msg.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for b in content:
        if isinstance(b, dict) and b.get("type") == "text":
            t = b.get("text")
            if isinstance(t, str):
                parts.append(t)
    return "".join(parts)


def _final_assistant_text(transcript: Iterable[dict]) -> str:
    """The final assistant message's text content. Empty string if absent."""
    final = ""
    for _, msg in _iter_messages(transcript):
        if msg.get("role") == "assistant":
            final = _assistant_text(msg)
    return final


def _bash_invocations(transcript: Iterable[dict]) -> list[str]:
    """Every bash-tool command string the model issued."""
    cmds: list[str] = []
    for _, msg in _iter_messages(transcript):
        if msg.get("role") != "assistant":
            continue
        for tc in _tool_calls_in(msg):
            if tc.get("name") in ("bash", "shell", "exec"):
                args = tc.get("input") or tc.get("arguments") or {}
                if isinstance(args, dict):
                    cmd = args.get("command") or args.get("cmd") or ""
                else:
                    cmd = ""
                if isinstance(cmd, str):
                    cmds.append(cmd)
    return cmds


def _all_tool_calls(transcript: Iterable[dict]) -> list[tuple[str, dict]]:
    """List of (tool_name, input_dict) pairs in chronological order."""
    out: list[tuple[str, dict]] = []
    for _, msg in _iter_messages(transcript):
        if msg.get("role") != "assistant":
            continue
        for tc in _tool_calls_in(msg):
            name = tc.get("name", "")
            args = tc.get("input") or tc.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            out.append((name, args))
    return out


def _tool_call_signatures(transcript: Iterable[dict]) -> list[str]:
    """Canonical (tool_name, input-as-json) signatures for dup detection."""
    out: list[str] = []
    for name, args in _all_tool_calls(transcript):
        out.append(json.dumps([name, args], sort_keys=True, default=str))
    return out


def _apply_attempts(transcript: Iterable[dict]) -> tuple[int, int]:
    """
    Returns (n_attempts, n_failed_attempts) where an attempt is any bash
    call that invoked `git apply`, `patch`, or `git am`.

    A "failed" attempt is one whose corresponding tool result contained
    obvious failure markers (non-zero exit, 'patch does not apply',
    'rejected hunk', etc.) OR where the next assistant turn explicitly
    re-tried with a modified command.
    """
    attempts = 0
    failed = 0
    # Walk messages in order; for each bash apply call, peek at the
    # following tool-result message.
    msgs = list(_iter_messages(transcript))
    for idx, (i, msg) in enumerate(msgs):
        if msg.get("role") != "assistant":
            continue
        for tc in _tool_calls_in(msg):
            if tc.get("name") not in ("bash", "shell", "exec"):
                continue
            args = tc.get("input") or tc.get("arguments") or {}
            if not isinstance(args, dict):
                continue
            cmd = (args.get("command") or args.get("cmd") or "")
            if not isinstance(cmd, str):
                continue
            if not re.search(
                r"\b(git\s+apply|patch\s+(?:-p\d|<)|git\s+am)\b", cmd
            ):
                continue
            attempts += 1
            # Find the next tool result by walking forward.
            next_text = ""
            for j in range(idx + 1, len(msgs)):
                _, next_msg = msgs[j]
                if next_msg.get("role") == "tool":
                    for tr in _tool_results_in(next_msg):
                        c = tr.get("content", "")
                        if isinstance(c, str):
                            next_text += c
                    break
            low = next_text.lower()
            if any(
                marker in low
                for marker in (
                    "does not apply",
                    "patch failed",
                    "rejected hunk",
                    "with conflict",
                    "error:",
                    "fatal:",
                    "no such file",
                    "while searching for",
                    ".rej",
                    "trailing whitespace",
                    "corrupt patch",
                )
            ):
                failed += 1
                continue
            # Heuristic: nonzero exit indicators (e.g. "exit=1") -> failed
            if re.search(r"\bexit(?:_code)?\s*[=:]\s*[1-9]", low):
                failed += 1
    return attempts, failed


def _test_runs(transcript: Iterable[dict]) -> tuple[int, int]:
    """
    Returns (n_runs, n_successful_runs) where a run is any bash call
    matching `pytest`, `unittest`, `python -m doctest`, `cargo test`, etc.
    Successful = the corresponding tool result contained 'passed' / 'ok'
    without 'failed' / 'error'.
    """
    runs = 0
    successful = 0
    msgs = list(_iter_messages(transcript))
    for idx, (i, msg) in enumerate(msgs):
        if msg.get("role") != "assistant":
            continue
        for tc in _tool_calls_in(msg):
            if tc.get("name") not in ("bash", "shell", "exec"):
                continue
            args = tc.get("input") or tc.get("arguments") or {}
            if not isinstance(args, dict):
                continue
            cmd = (args.get("command") or args.get("cmd") or "")
            if not isinstance(cmd, str):
                continue
            if not re.search(
                r"\b(pytest|python\s+-m\s+(?:doctest|unittest)|"
                r"python3?\s+-m\s+(?:doctest|unittest)|"
                r"cargo\s+test|nose|nosetests|tox)\b",
                cmd,
            ):
                continue
            runs += 1
            next_text = ""
            for j in range(idx + 1, len(msgs)):
                _, next_msg = msgs[j]
                if next_msg.get("role") == "tool":
                    for tr in _tool_results_in(next_msg):
                        c = tr.get("content", "")
                        if isinstance(c, str):
                            next_text += c
                    break
            low = next_text.lower()
            # pytest success indicators
            if (
                ("passed" in low or "ok" in low)
                and "failed" not in low
                and " error" not in low
                and "failure" not in low
            ):
                successful += 1
    return runs, successful


def _final_assistant_idx(msgs: list[tuple[int, dict]]) -> int:
    """Index into `msgs` of the final assistant message, or -1."""
    last = -1
    for k, (_, m) in enumerate(msgs):
        if m.get("role") == "assistant":
            last = k
    return last


# ============================================================================
# Workdir helpers
# ============================================================================

def _read_workdir_state(workdir: Path, paths: Iterable[str]) -> dict[str, str | None]:
    """Read the current text content of each path under workdir. Missing → None."""
    out: dict[str, str | None] = {}
    for rel in paths:
        p = workdir / rel
        if not p.exists():
            out[rel] = None
            continue
        try:
            out[rel] = p.read_text()
        except (UnicodeDecodeError, OSError):
            out[rel] = None
    return out


def _diff_line_count(a: str | None, b: str | None) -> int:
    """A naive diff-line counter: sum of added + removed lines between two
    file contents, computed with difflib's SequenceMatcher.

    None means "file absent" — treated as empty.
    """
    import difflib

    a_lines = (a or "").splitlines()
    b_lines = (b or "").splitlines()
    sm = difflib.SequenceMatcher(None, a_lines, b_lines, autojunk=False)
    added = 0
    removed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            removed += i2 - i1
            added += j2 - j1
        elif tag == "delete":
            removed += i2 - i1
        elif tag == "insert":
            added += j2 - j1
    return added + removed


def _workspace_changed_paths(init_files: dict[str, str], workdir: Path,
                              extra_known_paths: Iterable[str] = ()) -> list[str]:
    """Return all paths that differ between the initial state and the current
    workdir state.

    Considers:
      - paths that were in init_files
      - paths under extra_known_paths (e.g. gold-touched, patch-touched)
      - any new files we find anywhere under workdir
    """
    candidates: set[str] = set(init_files.keys())
    candidates.update(extra_known_paths)
    # Also enumerate anything under workdir that wasn't expected.
    if workdir.exists():
        for p in workdir.rglob("*"):
            if not p.is_file():
                continue
            # Skip rejects and .orig files (those are tool byproducts, not
            # model intent).
            rel = str(p.relative_to(workdir))
            if rel.endswith(".rej") or rel.endswith(".orig"):
                continue
            if rel.startswith(".git/"):
                continue
            if "/__pycache__/" in ("/" + rel) or rel.endswith(".pyc"):
                continue
            candidates.add(rel)

    changed: list[str] = []
    for rel in sorted(candidates):
        before = init_files.get(rel)
        p = workdir / rel
        after: str | None
        if p.exists():
            try:
                after = p.read_text()
            except (UnicodeDecodeError, OSError):
                after = None
        else:
            after = None
        if before != after:
            changed.append(rel)
    return changed


# ============================================================================
# Sub-score: minimality
# ============================================================================

def score_minimality(workdir: Path, task: dict) -> tuple[float, int, int]:
    """Return (score, final_diff_lines, gold_diff_lines).

    final_diff_lines: total added + removed lines across all initially-known
        files (we score against the *initial* state of the workspace).

    Score is 1.0 if final_diff_lines <= gold_diff_lines, decays linearly to
    0 at final_diff_lines == (1 + MINIMALITY_SPAN) * gold_diff_lines.
    """
    init_files: dict[str, str] = task.get("init_files") or {}
    gold_diff_lines = int(task.get("gold_diff_lines") or 0)
    final_state = _read_workdir_state(workdir, init_files.keys())
    final_diff_lines = 0
    for rel, before in init_files.items():
        after = final_state.get(rel)
        final_diff_lines += _diff_line_count(before, after)

    if gold_diff_lines <= 0:
        # No-op patch (rare) — any change is excess.
        score = 1.0 if final_diff_lines == 0 else 0.0
    elif final_diff_lines <= gold_diff_lines:
        score = 1.0
    else:
        ceiling = (1 + MINIMALITY_SPAN) * gold_diff_lines
        if final_diff_lines >= ceiling:
            score = 0.0
        else:
            score = 1.0 - (final_diff_lines - gold_diff_lines) / max(1, ceiling - gold_diff_lines)
    return float(max(0.0, min(1.0, score))), final_diff_lines, gold_diff_lines


# ============================================================================
# Sub-score: no_unrelated_edits
# ============================================================================

def score_no_unrelated(workdir: Path, task: dict) -> tuple[float, list[str], list[str]]:
    """Return (score, touched_paths, patch_paths).

    touched_paths: paths that differ between init and final.
    patch_paths: paths the gold patch is intended to modify.

    Score is 1.0 iff touched_paths is a subset of patch_paths.
    If extra files were touched, score = max(0, 1 - n_extra / max(1, n_patch + n_extra))
    """
    init_files: dict[str, str] = task.get("init_files") or {}
    patch_paths: list[str] = list(task.get("touched_paths") or [])
    touched = _workspace_changed_paths(init_files, workdir, extra_known_paths=patch_paths)
    patch_set = set(patch_paths)
    extra = [p for p in touched if p not in patch_set]
    intended = [p for p in touched if p in patch_set]
    n_extra = len(extra)
    if n_extra == 0:
        score = 1.0
    else:
        denom = max(1, len(intended) + n_extra)
        score = max(0.0, 1.0 - n_extra / denom)
    return float(score), touched, patch_paths


# ============================================================================
# Sub-score: repair_efficiency
# ============================================================================

def score_repair_efficiency(transcript: list, task: dict) -> tuple[float, int, int]:
    """Return (score, n_attempts, n_failed_attempts).

    For tasks where `expected_repair` is false (clean patch), this trivially
    scores 1.0 if the apply succeeded on the first try (the common case).

    For tasks where `expected_repair` is true (drift / incorrect hunk), the
    score is 1.0 for ≤ REPAIR_ATTEMPTS_FREE failed attempts, then decays
    REPAIR_ATTEMPTS_DECAY_PER per extra attempt.
    """
    attempts, failed = _apply_attempts(transcript)
    expected_repair = bool(task.get("expected_repair", False))

    if not expected_repair:
        # Clean patch: penalize ANY failed attempt (model botched a clean
        # apply), but allow recovery — single failure ~0.75, two ~0.5, etc.
        if failed == 0:
            score = 1.0
        else:
            score = max(0.0, 1.0 - 0.25 * failed)
    else:
        if failed <= REPAIR_ATTEMPTS_FREE:
            score = 1.0
        else:
            score = max(0.0, 1.0 - REPAIR_ATTEMPTS_DECAY_PER * (failed - REPAIR_ATTEMPTS_FREE))
    return float(score), attempts, failed


# ============================================================================
# Sub-score: format_compliance
# ============================================================================

# Format: final assistant message should contain
#   (a) a completion marker word ("done", "applied", "passed", "complete",
#       "fixed", "resolved" — case-insensitive)
#   (b) at least one reference to one of the touched files (or "tests" /
#       "pytest" / a summary heading like "## Summary")
COMPLETION_MARKERS = re.compile(
    r"\b(done|applied|passed|complete[d]?|fix(?:ed)?|"
    r"resolv(?:ed|ing)?|success(?:ful)?|test(?:s)?\s+pass)\b",
    re.IGNORECASE,
)
SUMMARY_HEADING = re.compile(r"^(?:#{1,3}\s+)?summary\b", re.IGNORECASE | re.MULTILINE)


def score_format(transcript: list, task: dict) -> float:
    """Final-turn format compliance. 1.0 when the closing message mentions
    completion AND references the touched-files set or includes a
    'Summary' heading. 0.0 when the final message is empty."""
    final = _final_assistant_text(transcript).strip()
    if not final:
        return 0.0
    points = 0.0
    if COMPLETION_MARKERS.search(final):
        points += 0.5
    patch_paths = task.get("touched_paths") or []
    # Reference: any patch path OR its basename mentioned in the final text.
    final_lower = final.lower()
    referenced = False
    for p in patch_paths:
        if p.lower() in final_lower:
            referenced = True
            break
        base = os.path.basename(p).lower()
        if base and len(base) >= 4 and base in final_lower:
            referenced = True
            break
    if referenced:
        points += 0.3
    if SUMMARY_HEADING.search(final) or "summary" in final_lower:
        points += 0.2
    return float(min(1.0, points))


# ============================================================================
# Sub-score: tested_before_done
# ============================================================================

def score_tested_before_done(transcript: list, task: dict) -> tuple[float, int, int]:
    """1.0 if the model ran the test command (pytest / doctest / cargo test)
    before its final assistant turn. Independent of whether tests passed —
    we want to reward the *habit* of verifying.

    Returns (score, n_test_runs_before_done, n_successful_runs)
    """
    msgs = list(_iter_messages(transcript))
    final_idx = _final_assistant_idx(msgs)
    if final_idx < 0:
        return 0.0, 0, 0
    n_runs_before = 0
    n_success_before = 0
    for k, (_, msg) in enumerate(msgs):
        if k >= final_idx:
            break
        if msg.get("role") != "assistant":
            continue
        for tc in _tool_calls_in(msg):
            if tc.get("name") not in ("bash", "shell", "exec"):
                continue
            args = tc.get("input") or tc.get("arguments") or {}
            if not isinstance(args, dict):
                continue
            cmd = (args.get("command") or args.get("cmd") or "")
            if not isinstance(cmd, str):
                continue
            if not re.search(
                r"\b(pytest|python\s+-m\s+(?:doctest|unittest)|"
                r"python3?\s+-m\s+(?:doctest|unittest)|"
                r"cargo\s+test|nose|nosetests|tox)\b",
                cmd,
            ):
                continue
            n_runs_before += 1
            # Peek next tool result for success.
            for j in range(k + 1, len(msgs)):
                _, next_msg = msgs[j]
                if next_msg.get("role") == "tool":
                    text = ""
                    for tr in _tool_results_in(next_msg):
                        c = tr.get("content", "")
                        if isinstance(c, str):
                            text += c
                    low = text.lower()
                    if ("passed" in low and "failed" not in low and "error" not in low):
                        n_success_before += 1
                    break
    score = 1.0 if n_runs_before > 0 else 0.0
    return score, n_runs_before, n_success_before


# ============================================================================
# Sub-score: applied_fraction (consolation)
# ============================================================================

def score_applied_fraction(workdir: Path, task: dict) -> float:
    """How much of the gold-final workspace state the model reached, on the
    files the gold patch touches.

    For each gold-touched path:
        - if model's final content == gold content → 1.0
        - else: line-level F1 between model-final and gold-final
    Mean over all touched paths is the score.

    This gives a non-zero gradient even when tests don't pass: the model
    that wrote out 80% of the gold lines correctly scores higher than the
    model that didn't touch the workspace at all.
    """
    gold_files: dict[str, str] = task.get("gold_files") or {}
    if not gold_files:
        return 0.0
    final_state = _read_workdir_state(workdir, gold_files.keys())
    init_files: dict[str, str] = task.get("init_files") or {}
    scores: list[float] = []
    for rel, gold_text in gold_files.items():
        cur = final_state.get(rel)
        init_text = init_files.get(rel)
        if cur is None:
            scores.append(0.0)
            continue
        if cur == gold_text:
            scores.append(1.0)
            continue
        # Line-level Dice coefficient (more forgiving than Jaccard).
        gold_lines = set(gold_text.splitlines()) - {""}
        cur_lines = set(cur.splitlines()) - {""}
        if not gold_lines and not cur_lines:
            scores.append(1.0)
            continue
        inter = len(gold_lines & cur_lines)
        denom = len(gold_lines) + len(cur_lines)
        dice = (2.0 * inter / denom) if denom > 0 else 0.0
        # Anchor: if the model didn't move from init at all, score 0.
        # We want to reward progress _toward_ gold, not just similarity.
        if init_text is not None and cur == init_text:
            scores.append(0.0)
            continue
        # If the model has at least achieved some of the lines unique to gold
        # (i.e. lines in gold but not in init), credit that explicitly.
        if init_text is not None:
            init_lines = set(init_text.splitlines()) - {""}
            gold_unique = gold_lines - init_lines
            cur_unique_match = (cur_lines & gold_unique)
            if gold_unique:
                progress = len(cur_unique_match) / len(gold_unique)
                # Blend: 70% gold-unique progress, 30% overall dice.
                scores.append(min(1.0, 0.7 * progress + 0.3 * dice))
            else:
                scores.append(dice)
        else:
            scores.append(dice)
    return float(sum(scores) / max(1, len(scores)))


# ============================================================================
# Sub-score: no_loop
# ============================================================================

def score_no_loop(transcript: list) -> tuple[float, int, int]:
    """1.0 - (fraction of tool calls that are exact duplicates of earlier
    calls in the same session). Returns (score, n_duplicates, n_calls)."""
    sigs = _tool_call_signatures(transcript)
    seen: set[str] = set()
    dups = 0
    for s in sigs:
        if s in seen:
            dups += 1
        else:
            seen.add(s)
    n = len(sigs)
    if n == 0:
        return 1.0, 0, 0
    score = 1.0 - dups / n
    return float(max(0.0, score)), dups, n


# ============================================================================
# Sub-score: tool_call_efficiency (diagnostic only)
# ============================================================================

def score_tool_call_efficiency(transcript: list) -> tuple[float, int]:
    """Diagnostic only — for tracking how chatty the model is. Not folded
    into composite for v1 to keep composite reading interpretable.

    1.0 when n_tool_calls ≤ TOOL_CALL_EXPECTED, linearly down to 0 at
    TOOL_CALL_HARD_CAP.
    """
    n = sum(len(_tool_calls_in(m)) for _, m in _iter_messages(transcript)
            if m.get("role") == "assistant")
    if n <= TOOL_CALL_EXPECTED:
        return 1.0, n
    if n >= TOOL_CALL_HARD_CAP:
        return 0.0, n
    score = 1.0 - (n - TOOL_CALL_EXPECTED) / (TOOL_CALL_HARD_CAP - TOOL_CALL_EXPECTED)
    return float(max(0.0, min(1.0, score))), n


# ============================================================================
# Outcome (run the test command in a sandboxed fresh-checkout style)
# ============================================================================

def _run_outcome_check(workdir: Path, task: dict) -> tuple[float, list[str]]:
    """Run the task's verify_cmd in the workdir; return (1.0/0.0,
    failing_test_names).

    The verifier is HERMETIC: it does not allow the agent to disable tests
    by mutating the test files at runtime — the oracle re-runs against a
    fresh checkout of the test fixtures so any model-side test mutations
    are reverted before scoring.
    """
    verify_cmd: str = task.get("verify_cmd") or "python3 -m pytest -q tests/"
    # The "fresh checkout" trick: copy the model's source files into a
    # tmpdir, but restore the *original* test files on top. This means any
    # model-side test mutations are erased before the verifier runs.
    fresh = workdir.parent / (workdir.name + "__verify")
    if fresh.exists():
        shutil.rmtree(fresh, ignore_errors=True)
    shutil.copytree(workdir, fresh)
    init_files: dict[str, str] = task.get("init_files") or {}
    for rel, original in init_files.items():
        # Restore only test-shaped files; do NOT restore source files (those
        # are what the patch is supposed to modify).
        if not _looks_like_test_path(rel):
            continue
        p = fresh / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(original)
    # Also restore any test files in `protected_paths` that the task spec
    # marks as "tests should not be modified".
    for rel in task.get("protected_paths") or []:
        if rel in init_files:
            p = fresh / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(init_files[rel])

    try:
        proc = subprocess.run(
            ["bash", "-c", verify_cmd],
            cwd=str(fresh),
            capture_output=True,
            text=True,
            timeout=int(task.get("verify_timeout_s") or 60),
        )
    except subprocess.TimeoutExpired:
        return 0.0, ["__verify_timeout__"]
    finally:
        # Best-effort cleanup. Keep the dir around when the env var is set
        # for postmortem (rarely useful).
        if not os.environ.get("DIFFPATCH_KEEP_VERIFY"):
            shutil.rmtree(fresh, ignore_errors=True)

    if proc.returncode == 0:
        return 1.0, []

    # Extract failing test names from stdout (pytest style).
    failures = []
    for line in (proc.stdout or "").splitlines():
        m = re.match(r"FAILED\s+(\S+)", line.strip())
        if m:
            failures.append(m.group(1))
    if not failures:
        # Fallback: any non-zero exit with no specific failure names.
        failures = ["__nonzero_exit__"]
    return 0.0, failures


def _looks_like_test_path(rel: str) -> bool:
    rel_lower = rel.lower()
    if "/tests/" in ("/" + rel_lower):
        return True
    if rel_lower.startswith("tests/"):
        return True
    basename = os.path.basename(rel_lower)
    if basename.startswith("test_") and basename.endswith(".py"):
        return True
    if basename.endswith("_test.py"):
        return True
    if basename.endswith("_test.rs"):
        return True
    return False


# ============================================================================
# Composite
# ============================================================================

def score_rollout(transcript: list, workdir: str, task: dict) -> dict[str, Any]:
    """Score a pi-diff-patch-apply rollout.

    transcript: list of pi-session events (parsed JSONL).
    workdir: path to the rollout's working directory (final state).
    task: the task spec (must include init_files, gold_files, touched_paths,
          gold_diff_lines, expected_repair, verify_cmd, verify_timeout_s).
    """
    wd = Path(workdir)
    if not wd.exists():
        # Pi never ran (or the workdir was clobbered). Score 0.
        return {
            "composite": 0.0,
            "outcome": 0.0,
            "minimality": 0.0,
            "no_unrelated_edits": 0.0,
            "repair_efficiency": 0.0,
            "format_compliance": 0.0,
            "tested_before_done": 0.0,
            "applied_fraction": 0.0,
            "no_loop": 1.0,
            "tool_call_efficiency": 1.0,
            "_n_tool_calls": 0,
            "_n_apply_attempts": 0,
            "_n_apply_failed": 0,
            "_n_test_runs": 0,
            "_n_test_success": 0,
            "_failed_tests": ["__no_workdir__"],
            "_touched_paths": [],
            "_patch_paths": task.get("touched_paths") or [],
            "_final_diff_lines": 0,
            "_gold_diff_lines": int(task.get("gold_diff_lines") or 0),
            "_reason": "workdir missing",
        }

    outcome, failing = _run_outcome_check(wd, task)
    minimality, final_diff_lines, gold_diff_lines = score_minimality(wd, task)
    nourel, touched, patch_paths = score_no_unrelated(wd, task)
    repair_eff, n_attempts, n_failed_attempts = score_repair_efficiency(transcript, task)
    fmt = score_format(transcript, task)
    tested, n_tests_before, n_tests_before_ok = score_tested_before_done(transcript, task)
    applied = score_applied_fraction(wd, task)
    no_loop, n_dups, n_calls = score_no_loop(transcript)
    tce, n_total_calls = score_tool_call_efficiency(transcript)
    n_test_total, n_test_ok_total = _test_runs(transcript)

    if outcome >= 1.0:
        # PASS path — full sub-score budget. Apply tested_before_done as a
        # discount: if the model never tested, we award the base 0.50 but
        # halve the agentic sub-score budget. A "passed-but-untested" rollout
        # is suspicious — the model may have lucked into a passing state
        # or rewritten test files.
        agentic = (
            W_MINIMALITY * minimality
            + W_NO_UNRELATED * nourel
            + W_REPAIR_EFF * repair_eff
            + W_FORMAT * fmt
        )
        if tested < 1.0:
            agentic *= TESTED_BEFORE_DONE_DISCOUNT_WHEN_MISSING
        composite = W_OUTCOME_BASE + agentic
    else:
        # FAIL path — consolation gradient on directional progress. Hard
        # capped at CONSOLATION_CAP so it can never match a passing rollout.
        composite = min(
            CONSOLATION_CAP,
            CONSOLATION_W_APPLIED * applied
            + CONSOLATION_W_TESTED * tested
            + CONSOLATION_W_FORMAT * fmt
            + CONSOLATION_W_NO_LOOP * no_loop,
        )

    return {
        "composite": float(max(0.0, min(1.0, composite))),
        "outcome": float(outcome),
        "minimality": float(minimality),
        "no_unrelated_edits": float(nourel),
        "repair_efficiency": float(repair_eff),
        "format_compliance": float(fmt),
        "tested_before_done": float(tested),
        "applied_fraction": float(applied),
        "no_loop": float(no_loop),
        "tool_call_efficiency": float(tce),
        "_n_tool_calls": int(n_total_calls),
        "_n_apply_attempts": int(n_attempts),
        "_n_apply_failed": int(n_failed_attempts),
        "_n_test_runs": int(n_test_total),
        "_n_test_success": int(n_test_ok_total),
        "_n_test_runs_before_done": int(n_tests_before),
        "_n_dup_calls": int(n_dups),
        "_failed_tests": list(failing),
        "_touched_paths": list(touched),
        "_patch_paths": list(patch_paths),
        "_final_diff_lines": int(final_diff_lines),
        "_gold_diff_lines": int(gold_diff_lines),
        "_expected_repair": bool(task.get("expected_repair", False)),
    }


# ============================================================================
# Tiny self-test (run `python3 rubric.py` to exercise the scorer)
# ============================================================================

if __name__ == "__main__":  # pragma: no cover
    import tempfile

    init = {
        "src/foo.py": "def add(a, b):\n    return a - b\n",
        "tests/test_foo.py": "from src.foo import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
    }
    gold = {
        "src/foo.py": "def add(a, b):\n    return a + b\n",
        "tests/test_foo.py": init["tests/test_foo.py"],
    }
    task = {
        "task_id": "test_0001",
        "init_files": init,
        "gold_files": gold,
        "touched_paths": ["src/foo.py"],
        "protected_paths": ["tests/test_foo.py"],
        "gold_diff_lines": 2,
        "expected_repair": False,
        "verify_cmd": "python3 -m pytest -q tests/",
        "verify_timeout_s": 30,
    }
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        wd = td_p / "wd"
        wd.mkdir()
        for rel, content in gold.items():
            p = wd / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        transcript = [
            {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "apply"}]}},
            {"type": "message", "message": {"role": "assistant", "content": [
                {"type": "toolCall", "name": "bash", "input": {"command": "git apply /tmp/incoming.patch"}, "id": "1"},
            ]}},
            {"type": "message", "message": {"role": "tool", "content": [
                {"type": "toolResult", "content": "patch applied cleanly", "toolCallId": "1"},
            ]}},
            {"type": "message", "message": {"role": "assistant", "content": [
                {"type": "toolCall", "name": "bash", "input": {"command": "pytest -q tests/"}, "id": "2"},
            ]}},
            {"type": "message", "message": {"role": "tool", "content": [
                {"type": "toolResult", "content": "1 passed", "toolCallId": "2"},
            ]}},
            {"type": "message", "message": {"role": "assistant", "content": [
                {"type": "text", "text": "Done. Applied patch to src/foo.py and all tests passed."},
            ]}},
        ]
        result = score_rollout(transcript, str(wd), task)
        for k, v in sorted(result.items()):
            print(f"  {k}: {v}")
        print(f"\nCOMPOSITE: {result['composite']:.4f}")
