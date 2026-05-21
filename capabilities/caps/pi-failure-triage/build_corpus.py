"""Build pi-failure-triage train + eval task JSONLs.

Strategy: each task is a planted bug in a small Python project. We
hand-author the bug + its visible test + a held-out related test such
that:
  - a SYMPTOM fix (try/except, hardcoded constant, suppressing error)
    will pass the visible test but FAIL the held-out test.
  - a ROOT-CAUSE fix will pass both.

We provide ~50 unique bugs across these families:
  off-by-one, missing-edge-case, swapped-operator, wrong-default,
  wrong-comparison-operator, type-confusion, swallowed-error,
  scope-bug, mutation-aliasing, missing-validation, classic-logic-flip.

Eval is the first 12 tasks; train is the rest.

Each task line:
  {
    "task_id": str,
    "description": str,
    "workspace_files": {rel: content},
    "visible_test_cmd": [argv...],
    "held_out_test_path": str,
    "held_out_test_content": str,
    "held_out_test_cmd": [argv...],
    "gold_fix_region": {file, function, module},
    "expected_diff_lines": int,
    "test_timeout_s": int
  }

Held-out tests are NEVER written to disk during rollout — they only
exist on the task spec. The rubric materializes them at score time.
"""

from __future__ import annotations

import json
from pathlib import Path

DST = Path(__file__).parent / "datasets"


# ----------------------------------------------------------------------
# Test runner helpers
# ----------------------------------------------------------------------

_PYTEST_VISIBLE = ["python3", "-m", "pytest", "-x", "-q", "tests/test_visible.py"]
_PYTEST_HELD_OUT = ["python3", "-m", "pytest", "-x", "-q", "tests/held_out/test_held_out.py"]


def _task(
    task_id: str,
    description: str,
    source_path: str,
    source_content_buggy: str,
    visible_test_content: str,
    held_out_test_content: str,
    gold_function: str,
    gold_module: str = "src",
    expected_diff_lines: int = 4,
) -> dict:
    """Helper to assemble a task spec."""
    workspace_files = {
        source_path: source_content_buggy,
        "tests/__init__.py": "",
        "tests/test_visible.py": visible_test_content,
        "conftest.py": "import sys, os\nsys.path.insert(0, os.path.dirname(__file__))\n",
        "README.md": (
            "## Project\n\n"
            f"There is a failing test in `tests/test_visible.py`. Find the\n"
            f"root cause in `{source_path}` and fix it.\n"
        ),
    }
    return {
        "task_id": task_id,
        "description": description,
        "workspace_files": workspace_files,
        "visible_test_cmd": _PYTEST_VISIBLE,
        "held_out_test_path": "tests/held_out/test_held_out.py",
        "held_out_test_content": held_out_test_content,
        "held_out_test_cmd": _PYTEST_HELD_OUT,
        "gold_fix_region": {
            "file": source_path,
            "function": gold_function,
            "module": gold_module,
        },
        "expected_diff_lines": expected_diff_lines,
        "test_timeout_s": 30,
    }


# ----------------------------------------------------------------------
# Bug-planted task templates
# ----------------------------------------------------------------------


def _tasks() -> list[dict]:
    out: list[dict] = []

    # ----- 1. Off-by-one: range upper bound
    out.append(_task(
        task_id="task_0001_offbyone_range_sum",
        description="`sum_first_n` should sum the first n positive integers (1..n inclusive).",
        source_path="src/sumrange.py",
        source_content_buggy=(
            "def sum_first_n(n):\n"
            "    # Sum of 1..n inclusive\n"
            "    total = 0\n"
            "    for i in range(1, n):  # bug: should be range(1, n+1)\n"
            "        total += i\n"
            "    return total\n"
        ),
        visible_test_content=(
            "from src.sumrange import sum_first_n\n"
            "def test_basic():\n"
            "    assert sum_first_n(5) == 15\n"
        ),
        held_out_test_content=(
            "from src.sumrange import sum_first_n\n"
            "def test_held_out_one():\n"
            "    assert sum_first_n(1) == 1\n"
            "def test_held_out_ten():\n"
            "    assert sum_first_n(10) == 55\n"
            "def test_held_out_hundred():\n"
            "    assert sum_first_n(100) == 5050\n"
        ),
        gold_function="sum_first_n",
    ))

    # ----- 2. Missing zero-check: divide by zero
    out.append(_task(
        task_id="task_0002_safe_divide",
        description="`safe_divide(a, b)` should return None when b == 0, else a/b.",
        source_path="src/divide.py",
        source_content_buggy=(
            "def safe_divide(a, b):\n"
            "    # When b is 0, return None; else return a/b.\n"
            "    return a / b\n"
        ),
        visible_test_content=(
            "from src.divide import safe_divide\n"
            "def test_normal():\n"
            "    assert safe_divide(10, 2) == 5.0\n"
            "def test_zero():\n"
            "    assert safe_divide(10, 0) is None\n"
        ),
        held_out_test_content=(
            "from src.divide import safe_divide\n"
            "def test_neg_zero():\n"
            "    assert safe_divide(-7, 0) is None\n"
            "def test_float_zero():\n"
            "    assert safe_divide(3.14, 0.0) is None\n"
            "def test_neg_neg():\n"
            "    assert safe_divide(-6, -2) == 3.0\n"
        ),
        gold_function="safe_divide",
    ))

    # ----- 3. Swapped condition: AND/OR
    out.append(_task(
        task_id="task_0003_is_weekend",
        description="`is_weekend(day)` should return True iff day is 'sat' or 'sun'.",
        source_path="src/weekend.py",
        source_content_buggy=(
            "def is_weekend(day):\n"
            "    # day is a lowercase 3-letter day name like 'mon'\n"
            "    return day == 'sat' and day == 'sun'\n"
        ),
        visible_test_content=(
            "from src.weekend import is_weekend\n"
            "def test_sat():\n"
            "    assert is_weekend('sat') is True\n"
        ),
        held_out_test_content=(
            "from src.weekend import is_weekend\n"
            "def test_sun():\n"
            "    assert is_weekend('sun') is True\n"
            "def test_mon():\n"
            "    assert is_weekend('mon') is False\n"
            "def test_fri():\n"
            "    assert is_weekend('fri') is False\n"
        ),
        gold_function="is_weekend",
    ))

    # ----- 4. Wrong default mutable argument
    out.append(_task(
        task_id="task_0004_mutable_default",
        description="`append_log(msg, log=None)` should not share state across calls.",
        source_path="src/logger.py",
        source_content_buggy=(
            "def append_log(msg, log=[]):\n"
            "    # Append msg to log and return it. log defaults to an empty\n"
            "    # list per call.\n"
            "    log.append(msg)\n"
            "    return log\n"
        ),
        visible_test_content=(
            "from src.logger import append_log\n"
            "def test_two_calls_independent():\n"
            "    a = append_log('a')\n"
            "    b = append_log('b')\n"
            "    assert b == ['b']\n"
        ),
        held_out_test_content=(
            "from src.logger import append_log\n"
            "def test_three_calls():\n"
            "    a = append_log('a')\n"
            "    b = append_log('b')\n"
            "    c = append_log('c')\n"
            "    assert a == ['a']\n"
            "    assert b == ['b']\n"
            "    assert c == ['c']\n"
            "def test_explicit_log_isolated():\n"
            "    out = append_log('x', log=['y'])\n"
            "    assert out == ['y', 'x']\n"
            "    again = append_log('z')\n"
            "    assert again == ['z']\n"
        ),
        gold_function="append_log",
        expected_diff_lines=6,
    ))

    # ----- 5. Wrong comparison: >= instead of >
    out.append(_task(
        task_id="task_0005_max_gte_bug",
        description="`first_strictly_greater(xs, threshold)` returns the first xs[i] strictly > threshold.",
        source_path="src/find.py",
        source_content_buggy=(
            "def first_strictly_greater(xs, threshold):\n"
            "    # Return first element strictly greater than threshold,\n"
            "    # or None.\n"
            "    for x in xs:\n"
            "        if x >= threshold:  # bug: should be > not >=\n"
            "            return x\n"
            "    return None\n"
        ),
        visible_test_content=(
            "from src.find import first_strictly_greater\n"
            "def test_basic():\n"
            "    assert first_strictly_greater([1, 2, 3], 1) == 2\n"
        ),
        held_out_test_content=(
            "from src.find import first_strictly_greater\n"
            "def test_equal_not_returned():\n"
            "    assert first_strictly_greater([5, 5, 6], 5) == 6\n"
            "def test_no_match():\n"
            "    assert first_strictly_greater([1, 2, 3], 10) is None\n"
            "def test_negatives():\n"
            "    assert first_strictly_greater([-3, -2, -1], -2) == -1\n"
        ),
        gold_function="first_strictly_greater",
    ))

    # ----- 6. Index error: missing length check
    out.append(_task(
        task_id="task_0006_safe_first",
        description="`safe_first(xs)` returns xs[0] if xs is non-empty, else None.",
        source_path="src/first.py",
        source_content_buggy=(
            "def safe_first(xs):\n"
            "    # Return xs[0], or None if xs is empty.\n"
            "    return xs[0]\n"
        ),
        visible_test_content=(
            "from src.first import safe_first\n"
            "def test_nonempty():\n"
            "    assert safe_first([1, 2, 3]) == 1\n"
            "def test_empty():\n"
            "    assert safe_first([]) is None\n"
        ),
        held_out_test_content=(
            "from src.first import safe_first\n"
            "def test_empty_tuple():\n"
            "    assert safe_first(()) is None\n"
            "def test_empty_string():\n"
            "    assert safe_first('') is None\n"
            "def test_single():\n"
            "    assert safe_first([42]) == 42\n"
        ),
        gold_function="safe_first",
    ))

    # ----- 7. Type confusion: int returned as string
    out.append(_task(
        task_id="task_0007_parse_int_returns_str",
        description="`parse_int(s)` returns an int, or None on parse failure.",
        source_path="src/parse.py",
        source_content_buggy=(
            "def parse_int(s):\n"
            "    # Return int(s) on success, else None.\n"
            "    try:\n"
            "        return s  # bug: should be int(s)\n"
            "    except ValueError:\n"
            "        return None\n"
        ),
        visible_test_content=(
            "from src.parse import parse_int\n"
            "def test_str_int():\n"
            "    assert parse_int('5') == 5\n"
        ),
        held_out_test_content=(
            "from src.parse import parse_int\n"
            "def test_int_type():\n"
            "    assert type(parse_int('10')) is int\n"
            "def test_bad():\n"
            "    assert parse_int('foo') is None\n"
            "def test_negative():\n"
            "    assert parse_int('-3') == -3\n"
        ),
        gold_function="parse_int",
    ))

    # ----- 8. Swallowed exception that hides bug
    out.append(_task(
        task_id="task_0008_dict_get_default",
        description="`get_or_default(d, k, default=0)` returns d[k] if present, else default.",
        source_path="src/getor.py",
        source_content_buggy=(
            "def get_or_default(d, k, default=0):\n"
            "    # If k is in d, return d[k]; else return default.\n"
            "    try:\n"
            "        return d[k]\n"
            "    except Exception:\n"
            "        return None  # bug: should return `default`\n"
        ),
        visible_test_content=(
            "from src.getor import get_or_default\n"
            "def test_default_int():\n"
            "    assert get_or_default({}, 'x', default=7) == 7\n"
        ),
        held_out_test_content=(
            "from src.getor import get_or_default\n"
            "def test_default_zero():\n"
            "    assert get_or_default({}, 'x') == 0\n"
            "def test_string_default():\n"
            "    assert get_or_default({}, 'x', default='hi') == 'hi'\n"
            "def test_present():\n"
            "    assert get_or_default({'a': 1}, 'a', default=99) == 1\n"
        ),
        gold_function="get_or_default",
    ))

    # ----- 9. String trim: only one side stripped
    out.append(_task(
        task_id="task_0009_trim_both",
        description="`trim(s)` should remove leading AND trailing whitespace.",
        source_path="src/trim.py",
        source_content_buggy=(
            "def trim(s):\n"
            "    # Strip whitespace from both ends.\n"
            "    return s.lstrip()  # bug: only strips left\n"
        ),
        visible_test_content=(
            "from src.trim import trim\n"
            "def test_left():\n"
            "    assert trim('   hello   ') == 'hello'\n"
        ),
        held_out_test_content=(
            "from src.trim import trim\n"
            "def test_only_right():\n"
            "    assert trim('hello   ') == 'hello'\n"
            "def test_tabs():\n"
            "    assert trim('\\thello\\t') == 'hello'\n"
            "def test_no_ws():\n"
            "    assert trim('hello') == 'hello'\n"
        ),
        gold_function="trim",
    ))

    # ----- 10. Inclusive vs exclusive range
    out.append(_task(
        task_id="task_0010_inclusive_range",
        description="`range_inclusive(a, b)` should return [a, a+1, ..., b].",
        source_path="src/inc.py",
        source_content_buggy=(
            "def range_inclusive(a, b):\n"
            "    # Returns [a..b] inclusive of both ends.\n"
            "    return list(range(a, b))  # bug: exclusive on b\n"
        ),
        visible_test_content=(
            "from src.inc import range_inclusive\n"
            "def test_basic():\n"
            "    assert range_inclusive(1, 5) == [1, 2, 3, 4, 5]\n"
        ),
        held_out_test_content=(
            "from src.inc import range_inclusive\n"
            "def test_single():\n"
            "    assert range_inclusive(3, 3) == [3]\n"
            "def test_negative():\n"
            "    assert range_inclusive(-2, 2) == [-2, -1, 0, 1, 2]\n"
            "def test_zero_to_n():\n"
            "    assert range_inclusive(0, 4) == [0, 1, 2, 3, 4]\n"
        ),
        gold_function="range_inclusive",
    ))

    # ----- 11. Reversed list mutation aliasing
    out.append(_task(
        task_id="task_0011_reversed_returns_iter",
        description="`reverse_list(xs)` should return a new list reversed, not mutate xs.",
        source_path="src/rev.py",
        source_content_buggy=(
            "def reverse_list(xs):\n"
            "    # Return a new list with the same elements in reverse order.\n"
            "    xs.reverse()  # bug: mutates input\n"
            "    return xs\n"
        ),
        visible_test_content=(
            "from src.rev import reverse_list\n"
            "def test_returns_reversed():\n"
            "    assert reverse_list([1, 2, 3]) == [3, 2, 1]\n"
        ),
        held_out_test_content=(
            "from src.rev import reverse_list\n"
            "def test_does_not_mutate_input():\n"
            "    xs = [1, 2, 3]\n"
            "    _ = reverse_list(xs)\n"
            "    assert xs == [1, 2, 3]\n"
            "def test_two_calls():\n"
            "    xs = [1, 2, 3]\n"
            "    a = reverse_list(xs)\n"
            "    b = reverse_list(xs)\n"
            "    assert a == [3, 2, 1]\n"
            "    assert b == [3, 2, 1]\n"
        ),
        gold_function="reverse_list",
    ))

    # ----- 12. Modulo bug (negative numbers)
    out.append(_task(
        task_id="task_0012_clamp_index",
        description="`clamp_index(i, n)` returns i clamped into [0, n-1].",
        source_path="src/clamp.py",
        source_content_buggy=(
            "def clamp_index(i, n):\n"
            "    # Clamp i into [0, n-1]\n"
            "    if i < 0:\n"
            "        return 0\n"
            "    if i > n:  # bug: should be >= n\n"
            "        return n - 1\n"
            "    return i\n"
        ),
        visible_test_content=(
            "from src.clamp import clamp_index\n"
            "def test_neg():\n"
            "    assert clamp_index(-5, 10) == 0\n"
        ),
        held_out_test_content=(
            "from src.clamp import clamp_index\n"
            "def test_exact_n():\n"
            "    assert clamp_index(10, 10) == 9\n"
            "def test_huge():\n"
            "    assert clamp_index(100, 10) == 9\n"
            "def test_zero_n_minus_1():\n"
            "    assert clamp_index(9, 10) == 9\n"
        ),
        gold_function="clamp_index",
    ))

    # ----- 13. Greedy count without explicit return
    out.append(_task(
        task_id="task_0013_count_vowels",
        description="`count_vowels(s)` returns count of vowels (a,e,i,o,u, case-insensitive).",
        source_path="src/vowels.py",
        source_content_buggy=(
            "def count_vowels(s):\n"
            "    # Count vowels case-insensitively.\n"
            "    n = 0\n"
            "    for c in s:\n"
            "        if c in 'aeiou':  # bug: misses uppercase\n"
            "            n += 1\n"
            "    return n\n"
        ),
        visible_test_content=(
            "from src.vowels import count_vowels\n"
            "def test_lower():\n"
            "    assert count_vowels('hello') == 2\n"
        ),
        held_out_test_content=(
            "from src.vowels import count_vowels\n"
            "def test_upper():\n"
            "    assert count_vowels('HELLO') == 2\n"
            "def test_mixed():\n"
            "    assert count_vowels('Hello') == 2\n"
            "def test_empty():\n"
            "    assert count_vowels('') == 0\n"
        ),
        gold_function="count_vowels",
    ))

    # ----- 14. List comprehension filter inverted
    out.append(_task(
        task_id="task_0014_filter_even",
        description="`only_even(xs)` returns a list of just the even numbers.",
        source_path="src/onlyev.py",
        source_content_buggy=(
            "def only_even(xs):\n"
            "    # Return a new list with only the even numbers.\n"
            "    return [x for x in xs if x % 2 == 1]  # bug: keeps odd\n"
        ),
        visible_test_content=(
            "from src.onlyev import only_even\n"
            "def test_basic():\n"
            "    assert only_even([1, 2, 3, 4]) == [2, 4]\n"
        ),
        held_out_test_content=(
            "from src.onlyev import only_even\n"
            "def test_all_even():\n"
            "    assert only_even([2, 4, 6]) == [2, 4, 6]\n"
            "def test_all_odd():\n"
            "    assert only_even([1, 3, 5]) == []\n"
            "def test_negative():\n"
            "    assert only_even([-2, -1, 0, 1, 2]) == [-2, 0, 2]\n"
        ),
        gold_function="only_even",
    ))

    # ----- 15. String slicing off-by-one
    out.append(_task(
        task_id="task_0015_last_n",
        description="`last_n_chars(s, n)` returns the last n characters of s.",
        source_path="src/lastn.py",
        source_content_buggy=(
            "def last_n_chars(s, n):\n"
            "    # Return last n chars; full string if n >= len.\n"
            "    return s[-n+1:]  # bug: off-by-one\n"
        ),
        visible_test_content=(
            "from src.lastn import last_n_chars\n"
            "def test_basic():\n"
            "    assert last_n_chars('hello', 3) == 'llo'\n"
        ),
        held_out_test_content=(
            "from src.lastn import last_n_chars\n"
            "def test_one():\n"
            "    assert last_n_chars('abc', 1) == 'c'\n"
            "def test_full():\n"
            "    assert last_n_chars('abc', 5) == 'abc'\n"
            "def test_two():\n"
            "    assert last_n_chars('hello', 2) == 'lo'\n"
        ),
        gold_function="last_n_chars",
    ))

    # ----- 16. Dict lookup early return
    out.append(_task(
        task_id="task_0016_first_match_key",
        description="`first_match(d, target)` returns the first key whose value equals target, or None.",
        source_path="src/firstm.py",
        source_content_buggy=(
            "def first_match(d, target):\n"
            "    # First key with value == target, else None.\n"
            "    for k, v in d.items():\n"
            "        if v != target:  # bug: inverted comparison\n"
            "            return k\n"
            "    return None\n"
        ),
        visible_test_content=(
            "from src.firstm import first_match\n"
            "def test_basic():\n"
            "    d = {'a': 1, 'b': 2}\n"
            "    assert first_match(d, 2) == 'b'\n"
        ),
        held_out_test_content=(
            "from src.firstm import first_match\n"
            "def test_no_match():\n"
            "    assert first_match({'a': 1, 'b': 2}, 5) is None\n"
            "def test_first():\n"
            "    assert first_match({'a': 9, 'b': 9}, 9) == 'a'\n"
            "def test_empty():\n"
            "    assert first_match({}, 1) is None\n"
        ),
        gold_function="first_match",
    ))

    # ----- 17. Recursive base case wrong
    out.append(_task(
        task_id="task_0017_factorial_zero",
        description="`factorial(n)` should return 1 for n=0 and n=1.",
        source_path="src/fact.py",
        source_content_buggy=(
            "def factorial(n):\n"
            "    # n! recursively. factorial(0) == 1.\n"
            "    if n == 1:  # bug: missing n == 0 case\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n"
        ),
        visible_test_content=(
            "from src.fact import factorial\n"
            "def test_basic():\n"
            "    assert factorial(5) == 120\n"
        ),
        held_out_test_content=(
            "from src.fact import factorial\n"
            "def test_zero():\n"
            "    assert factorial(0) == 1\n"
            "def test_one():\n"
            "    assert factorial(1) == 1\n"
            "def test_six():\n"
            "    assert factorial(6) == 720\n"
        ),
        gold_function="factorial",
    ))

    # ----- 18. Wrong default value
    out.append(_task(
        task_id="task_0018_default_zero",
        description="`safe_total(xs, zero=0)` returns sum of xs, treating None as the `zero` arg.",
        source_path="src/total.py",
        source_content_buggy=(
            "def safe_total(xs, zero=0):\n"
            "    # Sum xs, treating None as the `zero` argument.\n"
            "    total = 0\n"
            "    for x in xs:\n"
            "        total += x  # bug: None + 0 raises TypeError\n"
            "    return total\n"
        ),
        visible_test_content=(
            "from src.total import safe_total\n"
            "def test_with_none():\n"
            "    assert safe_total([1, None, 3]) == 4\n"
        ),
        held_out_test_content=(
            "from src.total import safe_total\n"
            "def test_all_none():\n"
            "    assert safe_total([None, None]) == 0\n"
            "def test_custom_zero():\n"
            "    # Non-zero `zero`: each None contributes 7.\n"
            "    assert safe_total([1, None, 3, None], zero=7) == 18\n"
            "def test_empty():\n"
            "    assert safe_total([]) == 0\n"
        ),
        gold_function="safe_total",
    ))

    # ----- 19. Path join missing separator
    out.append(_task(
        task_id="task_0019_path_join",
        description="`join_path(a, b)` joins two segments with exactly one '/'.",
        source_path="src/paths.py",
        source_content_buggy=(
            "def join_path(a, b):\n"
            "    # Join two path segments with exactly one '/' between them.\n"
            "    return a + '/' + b  # bug: doesn't handle trailing slash on a\n"
        ),
        visible_test_content=(
            "from src.paths import join_path\n"
            "def test_basic():\n"
            "    assert join_path('foo', 'bar') == 'foo/bar'\n"
        ),
        held_out_test_content=(
            "from src.paths import join_path\n"
            "def test_a_trailing_slash():\n"
            "    assert join_path('foo/', 'bar') == 'foo/bar'\n"
            "def test_b_leading_slash():\n"
            "    assert join_path('foo', '/bar') == 'foo/bar'\n"
            "def test_both():\n"
            "    assert join_path('foo/', '/bar') == 'foo/bar'\n"
        ),
        gold_function="join_path",
    ))

    # ----- 20. None propagation
    out.append(_task(
        task_id="task_0020_get_chain",
        description="`get_chain(d, *keys)` walks nested dicts; returns None if any key missing.",
        source_path="src/chain.py",
        source_content_buggy=(
            "def get_chain(d, *keys):\n"
            "    # Walk nested dicts. Return None if any key missing.\n"
            "    for k in keys:\n"
            "        d = d[k]  # bug: KeyError on missing\n"
            "    return d\n"
        ),
        visible_test_content=(
            "from src.chain import get_chain\n"
            "def test_present():\n"
            "    assert get_chain({'a': {'b': 5}}, 'a', 'b') == 5\n"
            "def test_missing():\n"
            "    assert get_chain({'a': {}}, 'a', 'b') is None\n"
        ),
        held_out_test_content=(
            "from src.chain import get_chain\n"
            "def test_deep():\n"
            "    d = {'a': {'b': {'c': 9}}}\n"
            "    assert get_chain(d, 'a', 'b', 'c') == 9\n"
            "def test_missing_top():\n"
            "    assert get_chain({}, 'a', 'b') is None\n"
            "def test_non_dict():\n"
            "    assert get_chain({'a': 1}, 'a', 'b') is None\n"
        ),
        gold_function="get_chain",
    ))

    # ----- 21. Bad regex (escaped wrong)
    out.append(_task(
        task_id="task_0021_extract_year",
        description="`extract_year(s)` returns the 4-digit year in s, or None.",
        source_path="src/year.py",
        source_content_buggy=(
            "import re\n"
            "def extract_year(s):\n"
            "    # Return the first 4-digit year found.\n"
            "    m = re.search(r'\\d{3}', s)  # bug: \\d{3} not \\d{4}\n"
            "    return m.group(0) if m else None\n"
        ),
        visible_test_content=(
            "from src.year import extract_year\n"
            "def test_basic():\n"
            "    assert extract_year('Year: 2024') == '2024'\n"
        ),
        held_out_test_content=(
            "from src.year import extract_year\n"
            "def test_three_digit():\n"
            "    assert extract_year('Year: 999 Project') is None\n"
            "def test_with_text():\n"
            "    assert extract_year('Released in 1999') == '1999'\n"
            "def test_no_year():\n"
            "    assert extract_year('No year here') is None\n"
        ),
        gold_function="extract_year",
    ))

    # ----- 22. Boolean short-circuit
    out.append(_task(
        task_id="task_0022_safe_len",
        description="`safe_len(x)` returns len(x), or 0 if x is None.",
        source_path="src/slen.py",
        source_content_buggy=(
            "def safe_len(x):\n"
            "    # Return len(x), or 0 if x is None.\n"
            "    return len(x)  # bug: AttributeError on None\n"
        ),
        visible_test_content=(
            "from src.slen import safe_len\n"
            "def test_basic():\n"
            "    assert safe_len([1, 2, 3]) == 3\n"
            "def test_none():\n"
            "    assert safe_len(None) == 0\n"
        ),
        held_out_test_content=(
            "from src.slen import safe_len\n"
            "def test_str():\n"
            "    assert safe_len('hello') == 5\n"
            "def test_empty():\n"
            "    assert safe_len([]) == 0\n"
            "def test_none_again():\n"
            "    assert safe_len(None) == 0\n"
        ),
        gold_function="safe_len",
    ))

    # ----- 23. Missing copy.deepcopy
    out.append(_task(
        task_id="task_0023_deep_update",
        description="`with_key(d, k, v)` returns a new dict with k->v, without mutating d.",
        source_path="src/withk.py",
        source_content_buggy=(
            "def with_key(d, k, v):\n"
            "    # Return a NEW dict with k->v; do not mutate d.\n"
            "    d[k] = v  # bug: mutates\n"
            "    return d\n"
        ),
        visible_test_content=(
            "from src.withk import with_key\n"
            "def test_basic():\n"
            "    d = {'a': 1}\n"
            "    out = with_key(d, 'b', 2)\n"
            "    assert out == {'a': 1, 'b': 2}\n"
            "    assert d == {'a': 1}\n"
        ),
        held_out_test_content=(
            "from src.withk import with_key\n"
            "def test_overwrite():\n"
            "    d = {'a': 1}\n"
            "    out = with_key(d, 'a', 99)\n"
            "    assert out == {'a': 99}\n"
            "    assert d == {'a': 1}\n"
            "def test_empty():\n"
            "    d = {}\n"
            "    out = with_key(d, 'x', 1)\n"
            "    assert out == {'x': 1}\n"
            "    assert d == {}\n"
        ),
        gold_function="with_key",
    ))

    # ----- 24. Class init wrong attribute
    out.append(_task(
        task_id="task_0024_counter_class",
        description="`Counter` class: increment() bumps count by 1.",
        source_path="src/counter.py",
        source_content_buggy=(
            "class Counter:\n"
            "    def __init__(self, start=0):\n"
            "        self.count = start\n"
            "    def increment(self):\n"
            "        # Bump count by 1.\n"
            "        self.count -= 1  # bug: decrement instead\n"
            "    def value(self):\n"
            "        return self.count\n"
        ),
        visible_test_content=(
            "from src.counter import Counter\n"
            "def test_basic():\n"
            "    c = Counter()\n"
            "    c.increment()\n"
            "    assert c.value() == 1\n"
        ),
        held_out_test_content=(
            "from src.counter import Counter\n"
            "def test_multiple():\n"
            "    c = Counter()\n"
            "    for _ in range(5):\n"
            "        c.increment()\n"
            "    assert c.value() == 5\n"
            "def test_start_nonzero():\n"
            "    c = Counter(start=10)\n"
            "    c.increment()\n"
            "    assert c.value() == 11\n"
        ),
        gold_function="increment",
    ))

    # ----- 25. Set membership uses list
    out.append(_task(
        task_id="task_0025_unique_count",
        description="`unique_count(xs)` returns the count of unique elements.",
        source_path="src/uniq.py",
        source_content_buggy=(
            "def unique_count(xs):\n"
            "    # Count distinct elements.\n"
            "    return len(xs)  # bug: doesn't dedupe\n"
        ),
        visible_test_content=(
            "from src.uniq import unique_count\n"
            "def test_dup():\n"
            "    assert unique_count([1, 1, 2, 2, 3]) == 3\n"
        ),
        held_out_test_content=(
            "from src.uniq import unique_count\n"
            "def test_all_unique():\n"
            "    assert unique_count([1, 2, 3]) == 3\n"
            "def test_all_same():\n"
            "    assert unique_count([7, 7, 7, 7]) == 1\n"
            "def test_empty():\n"
            "    assert unique_count([]) == 0\n"
        ),
        gold_function="unique_count",
    ))

    # ----- 26. Negative input not handled
    out.append(_task(
        task_id="task_0026_abs_diff",
        description="`abs_diff(a, b)` returns |a-b|.",
        source_path="src/absd.py",
        source_content_buggy=(
            "def abs_diff(a, b):\n"
            "    # Absolute difference.\n"
            "    return a - b  # bug: not abs\n"
        ),
        visible_test_content=(
            "from src.absd import abs_diff\n"
            "def test_basic():\n"
            "    assert abs_diff(7, 3) == 4\n"
        ),
        held_out_test_content=(
            "from src.absd import abs_diff\n"
            "def test_reversed():\n"
            "    assert abs_diff(3, 7) == 4\n"
            "def test_neg():\n"
            "    assert abs_diff(-3, 4) == 7\n"
            "def test_zero():\n"
            "    assert abs_diff(5, 5) == 0\n"
        ),
        gold_function="abs_diff",
    ))

    # ----- 27. String case-sensitive compare
    out.append(_task(
        task_id="task_0027_eq_ignore_case",
        description="`eq_ignore_case(a, b)` compares strings ignoring case.",
        source_path="src/eqic.py",
        source_content_buggy=(
            "def eq_ignore_case(a, b):\n"
            "    # Case-insensitive equality.\n"
            "    return a == b  # bug: case-sensitive\n"
        ),
        visible_test_content=(
            "from src.eqic import eq_ignore_case\n"
            "def test_same():\n"
            "    assert eq_ignore_case('hello', 'hello') is True\n"
            "def test_diff_case():\n"
            "    assert eq_ignore_case('Hello', 'hello') is True\n"
        ),
        held_out_test_content=(
            "from src.eqic import eq_ignore_case\n"
            "def test_upper_lower():\n"
            "    assert eq_ignore_case('FOO', 'foo') is True\n"
            "def test_diff():\n"
            "    assert eq_ignore_case('foo', 'bar') is False\n"
            "def test_mixed():\n"
            "    assert eq_ignore_case('AbC', 'aBc') is True\n"
        ),
        gold_function="eq_ignore_case",
    ))

    # ----- 28. Integer division when float expected
    out.append(_task(
        task_id="task_0028_average",
        description="`average(xs)` returns the mean (float).",
        source_path="src/avg.py",
        source_content_buggy=(
            "def average(xs):\n"
            "    # Arithmetic mean of xs. Return float.\n"
            "    return sum(xs) // len(xs)  # bug: integer division\n"
        ),
        visible_test_content=(
            "from src.avg import average\n"
            "def test_basic():\n"
            "    assert average([1, 2, 3, 4]) == 2.5\n"
        ),
        held_out_test_content=(
            "from src.avg import average\n"
            "def test_two():\n"
            "    assert average([1, 4]) == 2.5\n"
            "def test_one():\n"
            "    assert average([7]) == 7.0\n"
            "def test_negs():\n"
            "    assert average([-2, -4, -6]) == -4.0\n"
        ),
        gold_function="average",
    ))

    # ----- 29. String split missing maxsplit
    out.append(_task(
        task_id="task_0029_split_once",
        description="`split_once(s, sep)` splits at the first sep occurrence.",
        source_path="src/splito.py",
        source_content_buggy=(
            "def split_once(s, sep):\n"
            "    # Return (before, after) split at FIRST occurrence of sep.\n"
            "    # If sep not in s, return (s, '').\n"
            "    parts = s.split(sep)  # bug: splits at every occurrence\n"
            "    if len(parts) == 1:\n"
            "        return (s, '')\n"
            "    return (parts[0], parts[1])\n"
        ),
        visible_test_content=(
            "from src.splito import split_once\n"
            "def test_basic():\n"
            "    assert split_once('a=b', '=') == ('a', 'b')\n"
        ),
        held_out_test_content=(
            "from src.splito import split_once\n"
            "def test_multiple_sep():\n"
            "    assert split_once('a=b=c', '=') == ('a', 'b=c')\n"
            "def test_no_sep():\n"
            "    assert split_once('abc', '=') == ('abc', '')\n"
            "def test_at_end():\n"
            "    assert split_once('a=', '=') == ('a', '')\n"
        ),
        gold_function="split_once",
    ))

    # ----- 30. Stale variable: closure capture
    out.append(_task(
        task_id="task_0030_make_adders",
        description="`make_adders(ns)` returns a list of functions, each adds its corresponding n.",
        source_path="src/adders.py",
        source_content_buggy=(
            "def make_adders(ns):\n"
            "    # Return [lambda x: x+n0, lambda x: x+n1, ...]\n"
            "    out = []\n"
            "    for n in ns:\n"
            "        out.append(lambda x: x + n)  # bug: late binding\n"
            "    return out\n"
        ),
        visible_test_content=(
            "from src.adders import make_adders\n"
            "def test_basic():\n"
            "    adders = make_adders([1, 2, 3])\n"
            "    assert adders[0](10) == 11\n"
        ),
        held_out_test_content=(
            "from src.adders import make_adders\n"
            "def test_each_distinct():\n"
            "    adders = make_adders([1, 2, 3])\n"
            "    assert adders[0](10) == 11\n"
            "    assert adders[1](10) == 12\n"
            "    assert adders[2](10) == 13\n"
            "def test_empty():\n"
            "    assert make_adders([]) == []\n"
        ),
        gold_function="make_adders",
        expected_diff_lines=4,
    ))

    # ----- 31. Comparison with float epsilon
    out.append(_task(
        task_id="task_0031_almost_eq",
        description="`almost_eq(a, b, eps=1e-6)` returns True iff |a-b| < eps.",
        source_path="src/aeq.py",
        source_content_buggy=(
            "def almost_eq(a, b, eps=1e-6):\n"
            "    # True iff |a-b| < eps.\n"
            "    return a == b  # bug: doesn't account for float epsilon\n"
        ),
        visible_test_content=(
            "from src.aeq import almost_eq\n"
            "def test_exact():\n"
            "    assert almost_eq(1.0, 1.0) is True\n"
            "def test_near():\n"
            "    assert almost_eq(0.1 + 0.2, 0.3) is True\n"
        ),
        held_out_test_content=(
            "from src.aeq import almost_eq\n"
            "def test_far():\n"
            "    assert almost_eq(1.0, 2.0) is False\n"
            "def test_custom_eps():\n"
            "    assert almost_eq(0.0, 0.5, eps=1.0) is True\n"
            "    assert almost_eq(0.0, 1.5, eps=1.0) is False\n"
        ),
        gold_function="almost_eq",
    ))

    # ----- 32. Inverted boolean
    out.append(_task(
        task_id="task_0032_is_empty",
        description="`is_empty(xs)` returns True iff xs is empty.",
        source_path="src/iempty.py",
        source_content_buggy=(
            "def is_empty(xs):\n"
            "    # True iff xs is empty.\n"
            "    return len(xs) > 0  # bug: inverted\n"
        ),
        visible_test_content=(
            "from src.iempty import is_empty\n"
            "def test_empty_list():\n"
            "    assert is_empty([]) is True\n"
        ),
        held_out_test_content=(
            "from src.iempty import is_empty\n"
            "def test_nonempty():\n"
            "    assert is_empty([1]) is False\n"
            "def test_empty_str():\n"
            "    assert is_empty('') is True\n"
            "def test_empty_dict():\n"
            "    assert is_empty({}) is True\n"
        ),
        gold_function="is_empty",
    ))

    # ----- 33. Repeated computation
    out.append(_task(
        task_id="task_0033_sum_evens",
        description="`sum_evens(xs)` returns sum of even numbers in xs.",
        source_path="src/sume.py",
        source_content_buggy=(
            "def sum_evens(xs):\n"
            "    # Sum of even numbers in xs.\n"
            "    total = 0\n"
            "    for x in xs:\n"
            "        if x % 2 != 0:  # bug: should be == 0\n"
            "            total += x\n"
            "    return total\n"
        ),
        visible_test_content=(
            "from src.sume import sum_evens\n"
            "def test_basic():\n"
            "    assert sum_evens([1, 2, 3, 4]) == 6\n"
        ),
        held_out_test_content=(
            "from src.sume import sum_evens\n"
            "def test_all_odd():\n"
            "    assert sum_evens([1, 3, 5]) == 0\n"
            "def test_zero():\n"
            "    assert sum_evens([0, 1, 2]) == 2\n"
            "def test_neg_even():\n"
            "    assert sum_evens([-2, -1, 0, 1, 2]) == 0\n"
        ),
        gold_function="sum_evens",
    ))

    # ----- 34. Hash mod negative
    out.append(_task(
        task_id="task_0034_bucket",
        description="`bucket(h, n)` returns h mapped into [0, n).",
        source_path="src/buck.py",
        source_content_buggy=(
            "def bucket(h, n):\n"
            "    # Map h into [0, n).\n"
            "    return h % n if h >= 0 else h  # bug: returns h unchanged for neg\n"
        ),
        visible_test_content=(
            "from src.buck import bucket\n"
            "def test_basic():\n"
            "    assert 0 <= bucket(7, 4) < 4\n"
        ),
        held_out_test_content=(
            "from src.buck import bucket\n"
            "def test_neg():\n"
            "    assert 0 <= bucket(-3, 5) < 5\n"
            "def test_zero():\n"
            "    assert bucket(0, 7) == 0\n"
            "def test_large_neg():\n"
            "    b = bucket(-100, 7)\n"
            "    assert 0 <= b < 7\n"
        ),
        gold_function="bucket",
    ))

    # ----- 35. Dict update vs assign
    out.append(_task(
        task_id="task_0035_merge_dicts",
        description="`merge_dicts(a, b)` returns a new dict combining a and b (b wins on conflict).",
        source_path="src/merge.py",
        source_content_buggy=(
            "def merge_dicts(a, b):\n"
            "    # New dict; b wins on conflict.\n"
            "    out = {}\n"
            "    out.update(b)\n"
            "    out.update(a)  # bug: a should not win\n"
            "    return out\n"
        ),
        visible_test_content=(
            "from src.merge import merge_dicts\n"
            "def test_basic():\n"
            "    assert merge_dicts({'a': 1}, {'a': 2, 'b': 3}) == {'a': 2, 'b': 3}\n"
        ),
        held_out_test_content=(
            "from src.merge import merge_dicts\n"
            "def test_no_overlap():\n"
            "    assert merge_dicts({'a': 1}, {'b': 2}) == {'a': 1, 'b': 2}\n"
            "def test_all_b_wins():\n"
            "    assert merge_dicts({'a': 1, 'b': 2}, {'a': 9, 'b': 9}) == {'a': 9, 'b': 9}\n"
            "def test_empty_b():\n"
            "    assert merge_dicts({'a': 1}, {}) == {'a': 1}\n"
        ),
        gold_function="merge_dicts",
    ))

    # ----- 36. Wrong index in nested loop
    out.append(_task(
        task_id="task_0036_pairs",
        description="`pairs(xs)` returns list of (xs[i], xs[i+1]) for adjacent pairs.",
        source_path="src/pairs.py",
        source_content_buggy=(
            "def pairs(xs):\n"
            "    # Adjacent pairs (xs[i], xs[i+1]).\n"
            "    return [(xs[i], xs[i]) for i in range(len(xs) - 1)]  # bug: same idx twice\n"
        ),
        visible_test_content=(
            "from src.pairs import pairs\n"
            "def test_basic():\n"
            "    assert pairs([1, 2, 3]) == [(1, 2), (2, 3)]\n"
        ),
        held_out_test_content=(
            "from src.pairs import pairs\n"
            "def test_two():\n"
            "    assert pairs([5, 9]) == [(5, 9)]\n"
            "def test_one():\n"
            "    assert pairs([1]) == []\n"
            "def test_empty():\n"
            "    assert pairs([]) == []\n"
        ),
        gold_function="pairs",
    ))

    # ----- 37. None check after dict access
    out.append(_task(
        task_id="task_0037_lookup_or_default",
        description="`lookup_score(name, scores, default=0)` returns scores[name] or default.",
        source_path="src/lkup.py",
        source_content_buggy=(
            "def lookup_score(name, scores, default=0):\n"
            "    # Return scores.get(name) or default. None should be replaced by default.\n"
            "    return scores.get(name)  # bug: returns None instead of default\n"
        ),
        visible_test_content=(
            "from src.lkup import lookup_score\n"
            "def test_present():\n"
            "    assert lookup_score('a', {'a': 5}) == 5\n"
            "def test_missing():\n"
            "    assert lookup_score('x', {'a': 5}) == 0\n"
        ),
        held_out_test_content=(
            "from src.lkup import lookup_score\n"
            "def test_custom_default():\n"
            "    assert lookup_score('x', {}, default=99) == 99\n"
            "def test_string_default():\n"
            "    assert lookup_score('x', {}, default='unk') == 'unk'\n"
            "def test_empty_dict():\n"
            "    assert lookup_score('a', {}) == 0\n"
        ),
        gold_function="lookup_score",
    ))

    # ----- 38. Sort stability assumption
    out.append(_task(
        task_id="task_0038_sort_by_first",
        description="`sort_by_first(xs)` sorts a list of tuples by first element.",
        source_path="src/sortf.py",
        source_content_buggy=(
            "def sort_by_first(xs):\n"
            "    # Sort tuples by first element ascending.\n"
            "    return sorted(xs, key=lambda t: t[1])  # bug: sorts by second\n"
        ),
        visible_test_content=(
            "from src.sortf import sort_by_first\n"
            "def test_basic():\n"
            "    assert sort_by_first([(3, 'c'), (1, 'a'), (2, 'b')]) == [(1, 'a'), (2, 'b'), (3, 'c')]\n"
        ),
        held_out_test_content=(
            "from src.sortf import sort_by_first\n"
            "def test_dup_first():\n"
            "    out = sort_by_first([(1, 'x'), (1, 'y'), (0, 'z')])\n"
            "    assert out[0] == (0, 'z')\n"
            "def test_neg():\n"
            "    out = sort_by_first([(2, 'a'), (-1, 'b'), (0, 'c')])\n"
            "    assert [x[0] for x in out] == [-1, 0, 2]\n"
        ),
        gold_function="sort_by_first",
    ))

    # ----- 39. Range start
    out.append(_task(
        task_id="task_0039_squares",
        description="`squares_up_to(n)` returns [1, 4, 9, ..., n²].",
        source_path="src/sq.py",
        source_content_buggy=(
            "def squares_up_to(n):\n"
            "    # Squares 1..n inclusive.\n"
            "    return [i * i for i in range(n)]  # bug: 0..n-1\n"
        ),
        visible_test_content=(
            "from src.sq import squares_up_to\n"
            "def test_basic():\n"
            "    assert squares_up_to(3) == [1, 4, 9]\n"
        ),
        held_out_test_content=(
            "from src.sq import squares_up_to\n"
            "def test_one():\n"
            "    assert squares_up_to(1) == [1]\n"
            "def test_four():\n"
            "    assert squares_up_to(4) == [1, 4, 9, 16]\n"
            "def test_five():\n"
            "    assert squares_up_to(5) == [1, 4, 9, 16, 25]\n"
        ),
        gold_function="squares_up_to",
    ))

    # ----- 40. Multiplication when should be addition
    out.append(_task(
        task_id="task_0040_concat_list",
        description="`concat(a, b)` returns a new list = a followed by b.",
        source_path="src/concat.py",
        source_content_buggy=(
            "def concat(a, b):\n"
            "    # New list a followed by b.\n"
            "    return a * b  # bug: tries to multiply lists\n"
        ),
        visible_test_content=(
            "from src.concat import concat\n"
            "def test_basic():\n"
            "    assert concat([1, 2], [3, 4]) == [1, 2, 3, 4]\n"
        ),
        held_out_test_content=(
            "from src.concat import concat\n"
            "def test_first_empty():\n"
            "    assert concat([], [3, 4]) == [3, 4]\n"
            "def test_second_empty():\n"
            "    assert concat([1, 2], []) == [1, 2]\n"
            "def test_both_empty():\n"
            "    assert concat([], []) == []\n"
        ),
        gold_function="concat",
    ))

    # ----- 41. Tuple unpack arity
    out.append(_task(
        task_id="task_0041_minmax",
        description="`minmax(xs)` returns (min, max) of xs.",
        source_path="src/mm.py",
        source_content_buggy=(
            "def minmax(xs):\n"
            "    # Return (min, max).\n"
            "    return (max(xs), min(xs))  # bug: order reversed\n"
        ),
        visible_test_content=(
            "from src.mm import minmax\n"
            "def test_basic():\n"
            "    assert minmax([3, 1, 4, 1, 5]) == (1, 5)\n"
        ),
        held_out_test_content=(
            "from src.mm import minmax\n"
            "def test_neg():\n"
            "    assert minmax([-3, -1, -5]) == (-5, -1)\n"
            "def test_single():\n"
            "    assert minmax([7]) == (7, 7)\n"
            "def test_dup():\n"
            "    assert minmax([2, 2, 2]) == (2, 2)\n"
        ),
        gold_function="minmax",
    ))

    # ----- 42. Substring check vs equality
    out.append(_task(
        task_id="task_0042_starts_with_word",
        description="`starts_with(s, word)` returns True iff s starts with word as a whole word.",
        source_path="src/sw.py",
        source_content_buggy=(
            "def starts_with(s, word):\n"
            "    # True iff s begins with `word`.\n"
            "    return word in s  # bug: substring anywhere\n"
        ),
        visible_test_content=(
            "from src.sw import starts_with\n"
            "def test_basic():\n"
            "    assert starts_with('hello world', 'hello') is True\n"
        ),
        held_out_test_content=(
            "from src.sw import starts_with\n"
            "def test_not_at_start():\n"
            "    assert starts_with('world hello', 'hello') is False\n"
            "def test_full_match():\n"
            "    assert starts_with('abc', 'abc') is True\n"
            "def test_longer_word_than_s():\n"
            "    assert starts_with('hi', 'hello') is False\n"
        ),
        gold_function="starts_with",
    ))

    # ----- 43. Wrong cumulative computation
    out.append(_task(
        task_id="task_0043_running_sum",
        description="`running_sum(xs)` returns cumulative sum list.",
        source_path="src/rsum.py",
        source_content_buggy=(
            "def running_sum(xs):\n"
            "    # [xs[0], xs[0]+xs[1], xs[0]+xs[1]+xs[2], ...]\n"
            "    out = []\n"
            "    for x in xs:\n"
            "        out.append(x)  # bug: doesn't accumulate\n"
            "    return out\n"
        ),
        visible_test_content=(
            "from src.rsum import running_sum\n"
            "def test_basic():\n"
            "    assert running_sum([1, 2, 3, 4]) == [1, 3, 6, 10]\n"
        ),
        held_out_test_content=(
            "from src.rsum import running_sum\n"
            "def test_neg():\n"
            "    assert running_sum([1, -1, 2]) == [1, 0, 2]\n"
            "def test_zeros():\n"
            "    assert running_sum([0, 0, 1]) == [0, 0, 1]\n"
            "def test_one():\n"
            "    assert running_sum([5]) == [5]\n"
        ),
        gold_function="running_sum",
        expected_diff_lines=4,
    ))

    # ----- 44. Length parameter off
    out.append(_task(
        task_id="task_0044_truncate",
        description="`truncate(s, n)` returns the first n chars of s.",
        source_path="src/trunc.py",
        source_content_buggy=(
            "def truncate(s, n):\n"
            "    # First n chars of s. If s shorter than n, return s.\n"
            "    return s[:n-1]  # bug: off-by-one\n"
        ),
        visible_test_content=(
            "from src.trunc import truncate\n"
            "def test_basic():\n"
            "    assert truncate('hello', 3) == 'hel'\n"
        ),
        held_out_test_content=(
            "from src.trunc import truncate\n"
            "def test_full():\n"
            "    assert truncate('abc', 10) == 'abc'\n"
            "def test_one():\n"
            "    assert truncate('hello', 1) == 'h'\n"
            "def test_zero():\n"
            "    assert truncate('hello', 0) == ''\n"
        ),
        gold_function="truncate",
    ))

    # ----- 45. Bool coercion of int 0
    out.append(_task(
        task_id="task_0045_has_value",
        description="`has_value(d, k)` returns True iff k in d, including value=0/empty.",
        source_path="src/hv.py",
        source_content_buggy=(
            "def has_value(d, k):\n"
            "    # True iff key k is in d (regardless of value).\n"
            "    return bool(d.get(k))  # bug: 0/None/'' all return False\n"
        ),
        visible_test_content=(
            "from src.hv import has_value\n"
            "def test_missing():\n"
            "    assert has_value({}, 'x') is False\n"
            "def test_zero():\n"
            "    assert has_value({'x': 0}, 'x') is True\n"
        ),
        held_out_test_content=(
            "from src.hv import has_value\n"
            "def test_empty_string():\n"
            "    assert has_value({'x': ''}, 'x') is True\n"
            "def test_none():\n"
            "    assert has_value({'x': None}, 'x') is True\n"
            "def test_real_missing():\n"
            "    assert has_value({'a': 1}, 'x') is False\n"
        ),
        gold_function="has_value",
    ))

    # ----- 46. Loop early-exit condition
    out.append(_task(
        task_id="task_0046_find_index",
        description="`find_index(xs, target)` returns first index where xs[i] == target, else -1.",
        source_path="src/findi.py",
        source_content_buggy=(
            "def find_index(xs, target):\n"
            "    # First index of target, else -1.\n"
            "    for i, x in enumerate(xs):\n"
            "        if x == target:\n"
            "            continue  # bug: should be return i\n"
            "    return -1\n"
        ),
        visible_test_content=(
            "from src.findi import find_index\n"
            "def test_basic():\n"
            "    assert find_index([10, 20, 30], 20) == 1\n"
        ),
        held_out_test_content=(
            "from src.findi import find_index\n"
            "def test_first():\n"
            "    assert find_index([5, 5, 5], 5) == 0\n"
            "def test_missing():\n"
            "    assert find_index([1, 2, 3], 99) == -1\n"
            "def test_empty():\n"
            "    assert find_index([], 1) == -1\n"
        ),
        gold_function="find_index",
    ))

    # ----- 47. String formatting placeholder
    out.append(_task(
        task_id="task_0047_format_name",
        description="`format_name(first, last)` returns 'Last, First'.",
        source_path="src/fmtn.py",
        source_content_buggy=(
            "def format_name(first, last):\n"
            "    # 'Last, First'\n"
            "    return f'{first}, {last}'  # bug: order swapped\n"
        ),
        visible_test_content=(
            "from src.fmtn import format_name\n"
            "def test_basic():\n"
            "    assert format_name('Alice', 'Smith') == 'Smith, Alice'\n"
        ),
        held_out_test_content=(
            "from src.fmtn import format_name\n"
            "def test_other():\n"
            "    assert format_name('Bob', 'Jones') == 'Jones, Bob'\n"
            "def test_unicode():\n"
            "    assert format_name('Éric', 'Florenzano') == 'Florenzano, Éric'\n"
            "def test_short():\n"
            "    assert format_name('A', 'B') == 'B, A'\n"
        ),
        gold_function="format_name",
    ))

    # ----- 48. Numeric overflow handling
    out.append(_task(
        task_id="task_0048_pow_mod",
        description="`pow_mod(b, e, m)` computes (b**e) % m efficiently for any non-negative e.",
        source_path="src/pmod.py",
        source_content_buggy=(
            "def pow_mod(b, e, m):\n"
            "    # (b ** e) % m. Should handle e=0.\n"
            "    result = b\n"
            "    for _ in range(e - 1):  # bug: doesn't handle e=0\n"
            "        result = (result * b) % m\n"
            "    return result % m\n"
        ),
        visible_test_content=(
            "from src.pmod import pow_mod\n"
            "def test_basic():\n"
            "    assert pow_mod(3, 4, 100) == 81\n"
        ),
        held_out_test_content=(
            "from src.pmod import pow_mod\n"
            "def test_zero_exp():\n"
            "    assert pow_mod(7, 0, 11) == 1\n"
            "def test_mod_small():\n"
            "    assert pow_mod(2, 10, 1000) == 24\n"
            "def test_one_exp():\n"
            "    assert pow_mod(5, 1, 7) == 5\n"
        ),
        gold_function="pow_mod",
        expected_diff_lines=6,
    ))

    # ----- 49. Dict comprehension keys/values swapped
    out.append(_task(
        task_id="task_0049_invert",
        description="`invert(d)` returns {v: k for k, v in d.items()} (assumes 1-1).",
        source_path="src/inv.py",
        source_content_buggy=(
            "def invert(d):\n"
            "    # Swap keys and values.\n"
            "    return {k: v for k, v in d.items()}  # bug: doesn't swap\n"
        ),
        visible_test_content=(
            "from src.inv import invert\n"
            "def test_basic():\n"
            "    assert invert({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}\n"
        ),
        held_out_test_content=(
            "from src.inv import invert\n"
            "def test_empty():\n"
            "    assert invert({}) == {}\n"
            "def test_single():\n"
            "    assert invert({'x': 9}) == {9: 'x'}\n"
            "def test_three():\n"
            "    out = invert({'a': 1, 'b': 2, 'c': 3})\n"
            "    assert out == {1: 'a', 2: 'b', 3: 'c'}\n"
        ),
        gold_function="invert",
    ))

    # ----- 50. Wrong identity check
    out.append(_task(
        task_id="task_0050_is_none_or_zero",
        description="`is_none_or_zero(x)` returns True iff x is None or x == 0.",
        source_path="src/inoz.py",
        source_content_buggy=(
            "def is_none_or_zero(x):\n"
            "    # True iff x is None or x is 0.\n"
            "    return x is None  # bug: missing zero check\n"
        ),
        visible_test_content=(
            "from src.inoz import is_none_or_zero\n"
            "def test_none():\n"
            "    assert is_none_or_zero(None) is True\n"
            "def test_zero():\n"
            "    assert is_none_or_zero(0) is True\n"
        ),
        held_out_test_content=(
            "from src.inoz import is_none_or_zero\n"
            "def test_nonzero():\n"
            "    assert is_none_or_zero(5) is False\n"
            "def test_neg():\n"
            "    assert is_none_or_zero(-1) is False\n"
            "def test_float_zero():\n"
            "    assert is_none_or_zero(0.0) is True\n"
        ),
        gold_function="is_none_or_zero",
    ))

    return out


def main():
    DST.mkdir(parents=True, exist_ok=True)
    tasks = _tasks()
    # Quick uniqueness check
    seen = set()
    for t in tasks:
        assert t["task_id"] not in seen, f"duplicate task_id {t['task_id']}"
        seen.add(t["task_id"])

    eval_set = tasks[:12]
    train_set = tasks[12:]

    with (DST / "eval.tasks.jsonl").open("w") as f:
        for t in eval_set:
            f.write(json.dumps(t) + "\n")
    with (DST / "train.tasks.jsonl").open("w") as f:
        for t in train_set:
            f.write(json.dumps(t) + "\n")

    print(f"wrote {len(eval_set)} eval tasks, {len(train_set)} train tasks")


if __name__ == "__main__":
    main()
