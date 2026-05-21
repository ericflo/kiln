"""Calibration sanity gate for pi-doctest under the v1 multi-component rubric.

Builds 3 "good" rollouts (correct solution.py + synthetic clean transcript
with bash-doctest call before final assistant turn) and 3 "bad" rollouts
(stub solution.py, empty transcript), runs the rubric, and asserts:
  - good rollouts score ≥ 0.7
  - bad rollouts score ≤ 0.3

Exit 0 only when both bounds hold for every case.
"""

import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import task_scaffold
spec = importlib.util.spec_from_file_location(
    "rubric", str(Path(__file__).parent.parent / "rubric.py")
)
rubric = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rubric)


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


def synth_good_transcript() -> list:
    """A clean 3-tool-call session: read → edit → bash-doctest → DONE."""
    return [
        {"type": "message", "message": {"role": "user",
            "content": [{"type": "text", "text": "task prompt"}]}},
        {"type": "message", "message": {"role": "assistant",
            "content": [{"type": "toolCall", "name": "read",
                         "arguments": {"path": "solution.py"}}]}},
        {"type": "message", "message": {"role": "toolResult",
            "content": [{"type": "text", "text": "...source..."}]}},
        {"type": "message", "message": {"role": "assistant",
            "content": [{"type": "toolCall", "name": "edit",
                         "arguments": {"path": "solution.py"}}]}},
        {"type": "message", "message": {"role": "toolResult",
            "content": [{"type": "text", "text": "edited"}]}},
        {"type": "message", "message": {"role": "assistant",
            "content": [{"type": "toolCall", "name": "bash",
                         "arguments": {"command": "python3 -m doctest -v solution.py"}}]}},
        {"type": "message", "message": {"role": "toolResult",
            "content": [{"type": "text", "text": "tests passed"}]}},
        {"type": "message", "message": {"role": "assistant",
            "content": [{"type": "text", "text": "DONE"}]}},
    ]


def run_case(task: dict, body_replacement: str | None) -> dict:
    d = tempfile.mkdtemp(prefix="pi-doctest-sanity-")
    try:
        task_scaffold.init_workdir(task, d)
        if body_replacement is not None:
            src = Path(d, "solution.py").read_text()
            src = src.replace("    raise NotImplementedError\n",
                              body_replacement)
            Path(d, "solution.py").write_text(src)
        tr = synth_good_transcript() if body_replacement is not None else []
        return rubric.score_rollout(tr, d, task)
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
                f"  GOOD case {case['task']['task_id']} scored {good['composite']:.2f}"
                f": {good}"
            )
        if bad["composite"] > 0.3:
            failures.append(
                f"  BAD case {case['task']['task_id']} scored {bad['composite']:.2f}"
                f": {bad}"
            )
    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f)
        sys.exit(2)
    print("\nrubric_sanity OK — 3 good ≥ 0.7, 3 bad ≤ 0.3 under v1 rubric")


if __name__ == "__main__":
    main()
