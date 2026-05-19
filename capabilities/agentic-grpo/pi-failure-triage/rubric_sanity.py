"""Sanity-check the rubric: it MUST separate root-cause fixes from
symptom fixes.

Targets:
  - root-cause composite mean >= 0.70
  - symptom composite mean    <= 0.50
  - rc_min >= sy_max - 0.05  (clean separation)

We generate the calibration programmatically: for each of the first 6
training tasks, we synthesize a "root-cause fix" (the textbook
correct fix) and a "symptom fix" (try/except + hardcoded values that
satisfy the visible test only).

If those don't separate, the rubric is broken — fix it before training.
"""

from __future__ import annotations

import json
import sys
import tempfile
import textwrap
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import rubric
import task_scaffold


def _load_tasks(n: int = 6) -> list[dict]:
    out = []
    for path in [ROOT / "datasets/train.tasks.jsonl"]:
        with path.open() as f:
            for line in f:
                out.append(json.loads(line))
    return out[:n]


def _root_cause_fix(task: dict) -> str:
    """The textbook correct fix for each scaffolded task. Keyed by task_id."""
    return _ROOT_CAUSE_FIXES.get(task["task_id"], "")


def _symptom_fix(task: dict) -> str:
    """A try/except + hardcoded values that satisfies the visible test ONLY."""
    return _SYMPTOM_FIXES.get(task["task_id"], "")


# Curated fix bodies for the first 6 training tasks.
_ROOT_CAUSE_FIXES = {
    "task_0013_count_vowels": (
        "def count_vowels(s):\n"
        "    n = 0\n"
        "    for c in s:\n"
        "        if c.lower() in 'aeiou':\n"
        "            n += 1\n"
        "    return n\n"
    ),
    "task_0014_filter_even": (
        "def only_even(xs):\n"
        "    return [x for x in xs if x % 2 == 0]\n"
    ),
    "task_0015_last_n": (
        "def last_n_chars(s, n):\n"
        "    return s[-n:] if n > 0 else ''\n"
    ),
    "task_0016_first_match_key": (
        "def first_match(d, target):\n"
        "    for k, v in d.items():\n"
        "        if v == target:\n"
        "            return k\n"
        "    return None\n"
    ),
    "task_0017_factorial_zero": (
        "def factorial(n):\n"
        "    if n <= 1:\n"
        "        return 1\n"
        "    return n * factorial(n - 1)\n"
    ),
    "task_0018_default_zero": (
        "def safe_total(xs, zero=0):\n"
        "    total = 0\n"
        "    for x in xs:\n"
        "        if x is None:\n"
        "            total += zero\n"
        "        else:\n"
        "            total += x\n"
        "    return total\n"
    ),
}

# Symptom fixes: pass visible test by hardcoding or by wrapping with try/except.
_SYMPTOM_FIXES = {
    "task_0013_count_vowels": (
        "def count_vowels(s):\n"
        "    if s == 'hello':\n"
        "        return 2\n"
        "    try:\n"
        "        n = 0\n"
        "        for c in s:\n"
        "            if c in 'aeiou':\n"
        "                n += 1\n"
        "        return n\n"
        "    except Exception:\n"
        "        return 0\n"
    ),
    "task_0014_filter_even": (
        "def only_even(xs):\n"
        "    if xs == [1, 2, 3, 4]:\n"
        "        return [2, 4]\n"
        "    try:\n"
        "        return [x for x in xs if x % 2 == 1]\n"
        "    except Exception:\n"
        "        return []\n"
    ),
    "task_0015_last_n": (
        "def last_n_chars(s, n):\n"
        "    if s == 'hello' and n == 3:\n"
        "        return 'llo'\n"
        "    return s[-n+1:]\n"
    ),
    "task_0016_first_match_key": (
        "def first_match(d, target):\n"
        "    if d == {'a': 1, 'b': 2} and target == 2:\n"
        "        return 'b'\n"
        "    for k, v in d.items():\n"
        "        if v != target:\n"
        "            return k\n"
        "    return None\n"
    ),
    "task_0017_factorial_zero": (
        "def factorial(n):\n"
        "    if n == 5:\n"
        "        return 120\n"
        "    if n == 1:\n"
        "        return 1\n"
        "    return n * factorial(n - 1)\n"
    ),
    "task_0018_default_zero": (
        "def safe_total(xs, zero=0):\n"
        "    if xs == [1, None, 3]:\n"
        "        return 4\n"
        "    try:\n"
        "        return sum(x for x in xs if x is not None)\n"
        "    except Exception:\n"
        "        return 0\n"
    ),
}


def _synth_transcript(fix_text: str, source_path: str) -> list[dict]:
    """Synthesize a transcript with: bash test, write, bash test, final summary.

    Includes 'reproduced before fixing' behavior so the transcript-side
    sub-scores fire correctly when the rubric inspects them.
    """
    return [
        {"type": "message", "message": {"role": "assistant", "content": [
            {"type": "toolCall", "name": "bash", "input": {
                "command": "python3 -m pytest -x tests/test_visible.py"
            }}
        ]}},
        {"type": "message", "message": {"role": "tool", "content": [
            {"type": "toolResult", "content": "FAILED", "toolCallId": "t1"}
        ]}},
        {"type": "message", "message": {"role": "assistant", "content": [
            {"type": "toolCall", "name": "write", "input": {
                "path": source_path, "content": fix_text
            }}
        ]}},
        {"type": "message", "message": {"role": "tool", "content": [
            {"type": "toolResult", "content": "ok", "toolCallId": "t2"}
        ]}},
        {"type": "message", "message": {"role": "assistant", "content": [
            {"type": "toolCall", "name": "bash", "input": {
                "command": "python3 -m pytest -x tests/test_visible.py"
            }}
        ]}},
        {"type": "message", "message": {"role": "tool", "content": [
            {"type": "toolResult", "content": "PASSED", "toolCallId": "t3"}
        ]}},
        {"type": "message", "message": {"role": "assistant", "content": [
            {"type": "text", "text": f"Fix: {source_path}::function: corrected behavior."}
        ]}},
    ]


def main() -> int:
    tasks = _load_tasks(6)
    rc_rows = []
    sy_rows = []
    print("Sanity check (root-cause vs symptom):")
    for t in tasks:
        tid = t["task_id"]
        rc_fix = _ROOT_CAUSE_FIXES.get(tid)
        sy_fix = _SYMPTOM_FIXES.get(tid)
        if not rc_fix or not sy_fix:
            continue
        for label, fix in (("root", rc_fix), ("symptom", sy_fix)):
            with tempfile.TemporaryDirectory() as td:
                task_scaffold.init_workdir(t, td)
                src_path = Path(td) / t["gold_fix_region"]["file"]
                src_path.write_text(fix)
                tr = _synth_transcript(fix, t["gold_fix_region"]["file"])
                res = rubric.score_rollout(tr, td, t)
                row = {"task_id": tid, "label": label, **{k: v for k, v in res.items() if not k.startswith("_")}}
                if label == "root":
                    rc_rows.append(row)
                else:
                    sy_rows.append(row)
                print(f"  {tid:35s} {label:7s} composite={row['composite']:.3f} "
                      f"outcome={row['outcome']:.1f} ho={row['held_out_passes']:.1f} "
                      f"blanket={row['no_blanket_except']:.1f} fmt={row['format_compliance']:.1f}")

    if not rc_rows or not sy_rows:
        print("ERROR: no calibration rows produced", file=sys.stderr)
        return 2

    rc_c = [r["composite"] for r in rc_rows]
    sy_c = [r["composite"] for r in sy_rows]
    print()
    print(f"ROOT     n={len(rc_c)} mean={mean(rc_c):.3f} stdev={stdev(rc_c) if len(rc_c)>1 else 0:.3f} min={min(rc_c):.3f} max={max(rc_c):.3f}")
    print(f"SYMPTOM  n={len(sy_c)} mean={mean(sy_c):.3f} stdev={stdev(sy_c) if len(sy_c)>1 else 0:.3f} min={min(sy_c):.3f} max={max(sy_c):.3f}")
    # Primary: strict separation. Secondary: reasonable means.
    sep_ok = (
        min(rc_c) > max(sy_c) - 0.01
        and mean(rc_c) - mean(sy_c) >= 0.25
        and mean(rc_c) >= 0.80
    )
    print()
    print("PASS" if sep_ok else "FAIL", "— rubric separation")
    return 0 if sep_ok else 1


if __name__ == "__main__":
    sys.exit(main())
