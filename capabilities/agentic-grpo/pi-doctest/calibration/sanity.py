"""Calibration sanity gate for pi-doctest.

Builds 3 "good" rollouts (correct solution.py, near-empty transcript)
and 3 "bad" rollouts (stub solution.py, near-empty transcript), runs
the rubric, and asserts:
  - good rollouts score ≥ 0.7
  - bad rollouts score ≤ 0.3

Exit 0 only when both bounds hold for every case.

The rubric for v0 is solely doctest pass-rate; transcripts don't
affect composite. Therefore the calibration set just demonstrates
the rubric correctly maps workdir-state to composite.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import rubric
import task_scaffold


# 3 hand-picked tasks from the eval set + canonical correct solutions.
CASES = [
    {
        "task": {
            "task_id": "calib_rolling_max",
            "imports": "from typing import List\n",
            "function_signature": (
                "def rolling_max(numbers: List[int]) -> List[int]:\n"
                "    \"\"\" From a given list of integers, generate a list of rolling maximum.\n"
                "    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])\n"
                "    [1, 2, 3, 3, 3, 4, 4]\n"
                "    \"\"\"\n"
            ),
        },
        "good_body": (
            "    out=[]; m=None\n"
            "    for x in numbers:\n"
            "        m = x if m is None else max(m, x)\n"
            "        out.append(m)\n"
            "    return out\n"
        ),
    },
    {
        "task": {
            "task_id": "calib_max_element",
            "imports": "from typing import List\n",
            "function_signature": (
                "def max_element(l: List[int]) -> int:\n"
                "    \"\"\"Return maximum element in the list.\n"
                "    >>> max_element([1, 2, 3])\n"
                "    3\n"
                "    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 124, 1, -10])\n"
                "    124\n"
                "    \"\"\"\n"
            ),
        },
        "good_body": "    return max(l)\n",
    },
    {
        "task": {
            "task_id": "calib_strlen",
            "imports": "",
            "function_signature": (
                "def strlen(s: str) -> int:\n"
                "    \"\"\"Return length of a string.\n"
                "    >>> strlen('')\n"
                "    0\n"
                "    >>> strlen('abc')\n"
                "    3\n"
                "    \"\"\"\n"
            ),
        },
        "good_body": "    return len(s)\n",
    },
]


def run_case(task: dict, body_replacement: str | None) -> dict:
    """Returns rubric output for this case. body_replacement=None means
    stub (NotImplementedError)."""
    d = tempfile.mkdtemp(prefix="pi-doctest-sanity-")
    try:
        task_scaffold.init_workdir(task, d)
        if body_replacement is not None:
            src = Path(d, "solution.py").read_text()
            src = src.replace("    raise NotImplementedError\n",
                              body_replacement)
            Path(d, "solution.py").write_text(src)
        return rubric.score_rollout([], d, task)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    failures = []
    for case in CASES:
        good = run_case(case["task"], case["good_body"])
        bad = run_case(case["task"], None)
        print(f"{case['task']['task_id']}: good={good['composite']:.2f} "
              f"bad={bad['composite']:.2f}")
        if good["composite"] < 0.7:
            failures.append(
                f"  GOOD case {case['task']['task_id']} scored {good['composite']:.2f} (<0.7)"
                f": {good}"
            )
        if bad["composite"] > 0.3:
            failures.append(
                f"  BAD case {case['task']['task_id']} scored {bad['composite']:.2f} (>0.3)"
                f": {bad}"
            )
    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f)
        sys.exit(2)
    print("\nrubric_sanity OK — 3 good ≥ 0.7, 3 bad ≤ 0.3")


if __name__ == "__main__":
    main()
