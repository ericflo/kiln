"""Composite reward function for sft/python-algo (v0).

For SFT eval, the rubric checks whether the generated solution:
  1. compiles (parses as Python)
  2. defines the expected function
  3. passes the hidden tests

    composite = compiles × defines_fn × test_pass_rate
"""
from __future__ import annotations
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

RUBRIC_VERSION = "v0"


def _extract_code(response: str) -> str:
    """Strip optional markdown fences."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    return m.group(1) if m else response


def _compiles(code: str) -> bool:
    try:
        compile(code, "<rubric>", "exec")
        return True
    except SyntaxError:
        return False


def _run_tests(code: str, tests, fn_name: str | None) -> tuple[int, int]:
    """Return (n_passed, n_total). Tests are tuples like (input, expected)
    or (input_a, input_b, expected).
    """
    if not tests:
        return 0, 0
    if fn_name is None:
        m = re.search(r"def\s+(\w+)\s*\(", code)
        if not m:
            return 0, len(tests)
        fn_name = m.group(1)
    passed = 0
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        fp = f.name
    try:
        for test in tests:
            if len(test) == 2:
                args = (test[0],)
                expected = test[1]
            elif len(test) == 3:
                args = (test[0], test[1])
                expected = test[2]
            else:
                continue
            args_repr = ", ".join(repr(a) for a in args)
            try:
                r = subprocess.run(
                    ["python3", "-c", f"exec(open({fp!r}).read()); print({fn_name}({args_repr}))"],
                    capture_output=True, text=True, timeout=5,
                )
                out = r.stdout.strip()
                if out == repr(expected) or out == str(expected):
                    passed += 1
            except subprocess.TimeoutExpired:
                pass
    finally:
        Path(fp).unlink(missing_ok=True)
    return passed, len(tests)


def score_one(rollout: dict) -> dict[str, Any]:
    response = rollout.get("response") or rollout.get("completion") or ""
    task = rollout.get("task") or rollout
    tests = task.get("tests") or []
    code = _extract_code(response)
    compiles = _compiles(code)
    has_fn = bool(re.search(r"^def\s+\w+", code, re.MULTILINE))
    passed, total = _run_tests(code, tests, None) if compiles else (0, len(tests))
    test_rate = passed / total if total > 0 else (1.0 if compiles and has_fn else 0.0)
    composite = (1.0 if compiles else 0.0) * (1.0 if has_fn else 0.0) * test_rate
    return {
        "composite": float(max(0.0, min(1.0, composite))),
        "compiles": float(1.0 if compiles else 0.0),
        "defines_fn": float(1.0 if has_fn else 0.0),
        "test_pass_rate": float(test_rate),
        "n_passed": passed,
        "n_total": total,
    }


CHEAT_PROBES = [
    "emit pseudocode (mitigation: compiles=0)",
    "define wrong function name (mitigation: tests fail by exec)",
    "return correct value for visible tests, fail hidden (mitigation: tests are hidden)",
]
