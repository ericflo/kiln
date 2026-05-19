"""Build pi-diff-patch-apply train + eval task corpora.

Each task is a self-contained Python project with:
  - an INITIAL buggy implementation,
  - a small pytest test suite (covers happy path + edge cases),
  - a PATCH (unified diff) that the agent must apply to the workspace,
  - a GOLD final state (what the workspace should look like after a clean
    minimal application of the canonically-correct diff),
  - metadata flagging the patch class (clean / offset-drift / incorrect).

We synthesize tasks from a curated pool of small algorithm + data-manipulation
primitives. Synthesis is fully deterministic given the seed so this script is
re-runnable. Output is two JSONL files under `datasets/`.

The corpus mixes three patch classes (per capability.config.json):

  clean_patch_pct       0.50   patch applies cleanly with `git apply`.
  offset_drift_pct      0.30   patch's hunk @@ line numbers are off by ±3
                                (or extra context lines added/removed).
                                `git apply` may fail; agent must repair.
  incorrect_hunk_pct    0.20   patch applies cleanly but encodes a subtly
                                wrong change (e.g. wrong return value).
                                Tests fail after apply; agent must repair.

Usage:
    python3 build_corpus.py --train 60 --eval 24 --seed 3141592653

Outputs (gitignored):
    datasets/train.tasks.jsonl
    datasets/eval.tasks.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"


# ---------------------------------------------------------------------------
# Helpers for generating unified diffs
# ---------------------------------------------------------------------------

def _make_unified_diff(rel_path: str, before: str, after: str) -> str:
    """Generate a unified diff using the system `diff` command — matches the
    format `git apply` consumes. Returns the diff text (or empty if equal)."""
    if before == after:
        return ""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        a_dir = td_p / "a"
        b_dir = td_p / "b"
        a_dir.mkdir()
        b_dir.mkdir()
        ap = a_dir / rel_path
        bp = b_dir / rel_path
        ap.parent.mkdir(parents=True, exist_ok=True)
        bp.parent.mkdir(parents=True, exist_ok=True)
        ap.write_text(before)
        bp.write_text(after)
        proc = subprocess.run(
            ["diff", "-u", f"a/{rel_path}", f"b/{rel_path}"],
            cwd=td,
            capture_output=True,
            text=True,
        )
        # diff returns 1 when files differ, 0 when same, 2 on error.
        return proc.stdout


def _count_diff_changed_lines(diff_text: str) -> int:
    """Count added + removed lines in a unified diff (excluding the @@ and
    +++/--- file headers)."""
    n = 0
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("@@"):
            continue
        if line.startswith("+") or line.startswith("-"):
            n += 1
    return n


def _shift_hunk_headers(diff_text: str, line_shift: int) -> str:
    """Mutate every `@@ -A,B +C,D @@` hunk header by shifting A and C by
    `line_shift`. Used to manufacture offset-drift patches.
    """
    if line_shift == 0:
        return diff_text

    def repl(m: re.Match) -> str:
        old_start = max(1, int(m.group(1)) + line_shift)
        old_count = m.group(2)
        new_start = max(1, int(m.group(3)) + line_shift)
        new_count = m.group(4)
        suffix = m.group(5) or ""
        old_part = f"{old_start},{old_count}" if old_count else f"{old_start}"
        new_part = f"{new_start},{new_count}" if new_count else f"{new_start}"
        return f"@@ -{old_part} +{new_part} @@{suffix}"

    return re.sub(
        r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)",
        repl,
        diff_text,
    )


def _corrupt_hunk_line(diff_text: str, *, rng: random.Random) -> str:
    """Mutate one `+` line of the diff to encode an off-by-one or wrong-op
    bug. The resulting diff will apply cleanly but tests will fail."""
    lines = diff_text.splitlines()
    plus_indices = [
        i for i, line in enumerate(lines)
        if line.startswith("+") and not line.startswith("+++") and len(line) > 1
    ]
    if not plus_indices:
        return diff_text
    idx = rng.choice(plus_indices)
    original = lines[idx]
    body = original[1:]  # drop the leading "+"

    # Try a series of mutation strategies; first one that changes the line wins.
    mutations: list[Callable[[str], str]] = [
        lambda s: s.replace("==", "!="),
        lambda s: s.replace("!=", "=="),
        lambda s: s.replace(" + ", " - "),
        lambda s: s.replace(" - ", " + "),
        lambda s: s.replace(" * ", " // "),
        lambda s: s.replace(" // ", " * "),
        lambda s: s.replace(" < ", " > "),
        lambda s: s.replace(" > ", " < "),
        lambda s: s.replace(" <= ", " >= "),
        lambda s: s.replace(" >= ", " <= "),
        lambda s: s.replace("True", "False"),
        lambda s: s.replace("False", "True"),
        lambda s: re.sub(r"\brange\((\w+)\)", r"range(\1 - 1)", s),
        lambda s: re.sub(r"\brange\(0,\s*(\w+)\)", r"range(1, \1)", s),
    ]
    rng.shuffle(mutations)
    new_body = body
    for fn in mutations:
        cand = fn(body)
        if cand != body:
            new_body = cand
            break
    if new_body == body:
        # Fallback: just change a single integer literal by one.
        new_body = re.sub(r"\b(\d+)\b",
                          lambda m: str(int(m.group(1)) + 1),
                          body, count=1)
    lines[idx] = "+" + new_body
    return "\n".join(lines) + ("\n" if diff_text.endswith("\n") else "")


# ---------------------------------------------------------------------------
# Task primitives — (bug, fix, tests) triples
# ---------------------------------------------------------------------------

# Each primitive returns a dict with:
#   "module_name"   — the name of the source module (no .py)
#   "buggy_src"     — initial buggy content of src/<module_name>.py
#   "gold_src"      — correct content of src/<module_name>.py
#   "tests_src"     — content of tests/test_<module_name>.py


def prim_add() -> dict:
    return {
        "module_name": "addition",
        "buggy_src": (
            "def add(a, b):\n"
            "    return a - b\n"
        ),
        "gold_src": (
            "def add(a, b):\n"
            "    return a + b\n"
        ),
        "tests_src": (
            "from src.addition import add\n\n"
            "def test_add_positive():\n"
            "    assert add(2, 3) == 5\n\n"
            "def test_add_zero():\n"
            "    assert add(0, 7) == 7\n\n"
            "def test_add_negative():\n"
            "    assert add(-2, -3) == -5\n"
        ),
    }


def prim_factorial() -> dict:
    return {
        "module_name": "fact",
        "buggy_src": (
            "def factorial(n):\n"
            "    if n <= 0:\n"
            "        return 0\n"
            "    out = 1\n"
            "    for k in range(1, n):\n"
            "        out *= k\n"
            "    return out\n"
        ),
        "gold_src": (
            "def factorial(n):\n"
            "    if n < 0:\n"
            "        raise ValueError('n must be non-negative')\n"
            "    out = 1\n"
            "    for k in range(2, n + 1):\n"
            "        out *= k\n"
            "    return out\n"
        ),
        "tests_src": (
            "import pytest\n"
            "from src.fact import factorial\n\n"
            "def test_zero():\n"
            "    assert factorial(0) == 1\n\n"
            "def test_one():\n"
            "    assert factorial(1) == 1\n\n"
            "def test_five():\n"
            "    assert factorial(5) == 120\n\n"
            "def test_negative_raises():\n"
            "    with pytest.raises(ValueError):\n"
            "        factorial(-1)\n"
        ),
    }


def prim_fib() -> dict:
    return {
        "module_name": "fib",
        "buggy_src": (
            "def fib(n):\n"
            "    if n < 2:\n"
            "        return n\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n"
        ),
        "gold_src": (
            "def fib(n):\n"
            "    if n < 2:\n"
            "        return n\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n - 1):\n"
            "        a, b = b, a + b\n"
            "    return b\n"
        ),
        "tests_src": (
            "from src.fib import fib\n\n"
            "def test_0():\n"
            "    assert fib(0) == 0\n\n"
            "def test_1():\n"
            "    assert fib(1) == 1\n\n"
            "def test_10():\n"
            "    assert fib(10) == 55\n\n"
            "def test_2():\n"
            "    assert fib(2) == 1\n"
        ),
    }


def prim_reverse() -> dict:
    return {
        "module_name": "rev",
        "buggy_src": (
            "def reverse_string(s):\n"
            "    out = ''\n"
            "    for i in range(len(s)):\n"
            "        out += s[i]\n"
            "    return out\n"
        ),
        "gold_src": (
            "def reverse_string(s):\n"
            "    return s[::-1]\n"
        ),
        "tests_src": (
            "from src.rev import reverse_string\n\n"
            "def test_hello():\n"
            "    assert reverse_string('hello') == 'olleh'\n\n"
            "def test_empty():\n"
            "    assert reverse_string('') == ''\n\n"
            "def test_palindrome():\n"
            "    assert reverse_string('aba') == 'aba'\n"
        ),
    }


def prim_is_prime() -> dict:
    return {
        "module_name": "primes",
        "buggy_src": (
            "def is_prime(n):\n"
            "    if n <= 1:\n"
            "        return True\n"
            "    for i in range(2, n):\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "    return True\n"
        ),
        "gold_src": (
            "def is_prime(n):\n"
            "    if n < 2:\n"
            "        return False\n"
            "    if n < 4:\n"
            "        return True\n"
            "    if n % 2 == 0:\n"
            "        return False\n"
            "    i = 3\n"
            "    while i * i <= n:\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "        i += 2\n"
            "    return True\n"
        ),
        "tests_src": (
            "from src.primes import is_prime\n\n"
            "def test_one_not_prime():\n"
            "    assert not is_prime(1)\n\n"
            "def test_two_is_prime():\n"
            "    assert is_prime(2)\n\n"
            "def test_seven_is_prime():\n"
            "    assert is_prime(7)\n\n"
            "def test_nine_not_prime():\n"
            "    assert not is_prime(9)\n\n"
            "def test_zero_not_prime():\n"
            "    assert not is_prime(0)\n"
        ),
    }


def prim_max() -> dict:
    return {
        "module_name": "maxlist",
        "buggy_src": (
            "def max_in_list(xs):\n"
            "    m = 0\n"
            "    for x in xs:\n"
            "        if x > m:\n"
            "            m = x\n"
            "    return m\n"
        ),
        "gold_src": (
            "def max_in_list(xs):\n"
            "    if not xs:\n"
            "        raise ValueError('empty list')\n"
            "    m = xs[0]\n"
            "    for x in xs[1:]:\n"
            "        if x > m:\n"
            "            m = x\n"
            "    return m\n"
        ),
        "tests_src": (
            "import pytest\n"
            "from src.maxlist import max_in_list\n\n"
            "def test_basic():\n"
            "    assert max_in_list([1, 2, 3]) == 3\n\n"
            "def test_negatives():\n"
            "    assert max_in_list([-5, -2, -10]) == -2\n\n"
            "def test_single():\n"
            "    assert max_in_list([42]) == 42\n\n"
            "def test_empty():\n"
            "    with pytest.raises(ValueError):\n"
            "        max_in_list([])\n"
        ),
    }


def prim_count_vowels() -> dict:
    return {
        "module_name": "vowels",
        "buggy_src": (
            "VOWELS = 'aeiou'\n\n"
            "def count_vowels(s):\n"
            "    n = 0\n"
            "    for c in s:\n"
            "        if c in VOWELS:\n"
            "            n += 1\n"
            "    return n\n"
        ),
        "gold_src": (
            "VOWELS = 'aeiouAEIOU'\n\n"
            "def count_vowels(s):\n"
            "    return sum(1 for c in s if c in VOWELS)\n"
        ),
        "tests_src": (
            "from src.vowels import count_vowels\n\n"
            "def test_lower():\n"
            "    assert count_vowels('hello') == 2\n\n"
            "def test_mixed_case():\n"
            "    assert count_vowels('Apple') == 2\n\n"
            "def test_no_vowels():\n"
            "    assert count_vowels('xyz') == 0\n\n"
            "def test_all_vowels():\n"
            "    assert count_vowels('aeiouAEIOU') == 10\n"
        ),
    }


def prim_flatten() -> dict:
    return {
        "module_name": "flat",
        "buggy_src": (
            "def flatten(xss):\n"
            "    return [xss[i][0] for i in range(len(xss))]\n"
        ),
        "gold_src": (
            "def flatten(xss):\n"
            "    out = []\n"
            "    for xs in xss:\n"
            "        for x in xs:\n"
            "            out.append(x)\n"
            "    return out\n"
        ),
        "tests_src": (
            "from src.flat import flatten\n\n"
            "def test_simple():\n"
            "    assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]\n\n"
            "def test_empty_inner():\n"
            "    assert flatten([[], [1], []]) == [1]\n\n"
            "def test_strings():\n"
            "    assert flatten([['a', 'b'], ['c']]) == ['a', 'b', 'c']\n"
        ),
    }


def prim_dedup() -> dict:
    return {
        "module_name": "dedup",
        "buggy_src": (
            "def dedup(xs):\n"
            "    return list(set(xs))\n"
        ),
        "gold_src": (
            "def dedup(xs):\n"
            "    seen = set()\n"
            "    out = []\n"
            "    for x in xs:\n"
            "        if x not in seen:\n"
            "            seen.add(x)\n"
            "            out.append(x)\n"
            "    return out\n"
        ),
        "tests_src": (
            "from src.dedup import dedup\n\n"
            "def test_basic():\n"
            "    assert dedup([1, 2, 1, 3, 2]) == [1, 2, 3]\n\n"
            "def test_already_unique():\n"
            "    assert dedup([1, 2, 3]) == [1, 2, 3]\n\n"
            "def test_empty():\n"
            "    assert dedup([]) == []\n\n"
            "def test_preserves_order():\n"
            "    assert dedup(['b', 'a', 'b', 'c', 'a']) == ['b', 'a', 'c']\n"
        ),
    }


def prim_gcd() -> dict:
    return {
        "module_name": "gcd",
        "buggy_src": (
            "def gcd(a, b):\n"
            "    while a != b:\n"
            "        if a > b:\n"
            "            a -= b\n"
            "        else:\n"
            "            b -= a\n"
            "    return a\n"
        ),
        "gold_src": (
            "def gcd(a, b):\n"
            "    a, b = abs(a), abs(b)\n"
            "    while b:\n"
            "        a, b = b, a % b\n"
            "    return a\n"
        ),
        "tests_src": (
            "from src.gcd import gcd\n\n"
            "def test_basic():\n"
            "    assert gcd(12, 8) == 4\n\n"
            "def test_coprime():\n"
            "    assert gcd(17, 5) == 1\n\n"
            "def test_zero():\n"
            "    assert gcd(0, 5) == 5\n\n"
            "def test_negative():\n"
            "    assert gcd(-12, 8) == 4\n"
        ),
    }


def prim_palindrome() -> dict:
    return {
        "module_name": "pal",
        "buggy_src": (
            "def is_palindrome(s):\n"
            "    return s == s[1:]\n"
        ),
        "gold_src": (
            "def is_palindrome(s):\n"
            "    return s == s[::-1]\n"
        ),
        "tests_src": (
            "from src.pal import is_palindrome\n\n"
            "def test_yes():\n"
            "    assert is_palindrome('aba')\n\n"
            "def test_no():\n"
            "    assert not is_palindrome('abc')\n\n"
            "def test_empty():\n"
            "    assert is_palindrome('')\n\n"
            "def test_one():\n"
            "    assert is_palindrome('x')\n"
        ),
    }


def prim_sumlist() -> dict:
    return {
        "module_name": "sumlist",
        "buggy_src": (
            "def sumlist(xs):\n"
            "    s = 0\n"
            "    for x in xs:\n"
            "        s = s * x\n"
            "    return s\n"
        ),
        "gold_src": (
            "def sumlist(xs):\n"
            "    s = 0\n"
            "    for x in xs:\n"
            "        s = s + x\n"
            "    return s\n"
        ),
        "tests_src": (
            "from src.sumlist import sumlist\n\n"
            "def test_basic():\n"
            "    assert sumlist([1, 2, 3]) == 6\n\n"
            "def test_empty():\n"
            "    assert sumlist([]) == 0\n\n"
            "def test_negative():\n"
            "    assert sumlist([-1, 2, -3]) == -2\n"
        ),
    }


def prim_filter_even() -> dict:
    return {
        "module_name": "evens",
        "buggy_src": (
            "def filter_even(xs):\n"
            "    return [x for x in xs if x % 2 == 1]\n"
        ),
        "gold_src": (
            "def filter_even(xs):\n"
            "    return [x for x in xs if x % 2 == 0]\n"
        ),
        "tests_src": (
            "from src.evens import filter_even\n\n"
            "def test_basic():\n"
            "    assert filter_even([1, 2, 3, 4]) == [2, 4]\n\n"
            "def test_all_odd():\n"
            "    assert filter_even([1, 3, 5]) == []\n\n"
            "def test_empty():\n"
            "    assert filter_even([]) == []\n\n"
            "def test_zero():\n"
            "    assert filter_even([0, 1, 2]) == [0, 2]\n"
        ),
    }


def prim_caesar() -> dict:
    return {
        "module_name": "caesar",
        "buggy_src": (
            "def caesar(s, shift):\n"
            "    out = []\n"
            "    for c in s:\n"
            "        if c.isalpha():\n"
            "            out.append(chr(ord(c) + shift))\n"
            "        else:\n"
            "            out.append(c)\n"
            "    return ''.join(out)\n"
        ),
        "gold_src": (
            "def caesar(s, shift):\n"
            "    out = []\n"
            "    for c in s:\n"
            "        if 'a' <= c <= 'z':\n"
            "            out.append(chr((ord(c) - ord('a') + shift) % 26 + ord('a')))\n"
            "        elif 'A' <= c <= 'Z':\n"
            "            out.append(chr((ord(c) - ord('A') + shift) % 26 + ord('A')))\n"
            "        else:\n"
            "            out.append(c)\n"
            "    return ''.join(out)\n"
        ),
        "tests_src": (
            "from src.caesar import caesar\n\n"
            "def test_basic():\n"
            "    assert caesar('abc', 1) == 'bcd'\n\n"
            "def test_wrap():\n"
            "    assert caesar('xyz', 3) == 'abc'\n\n"
            "def test_uppercase():\n"
            "    assert caesar('ABC', 1) == 'BCD'\n\n"
            "def test_nonalpha():\n"
            "    assert caesar('a b!', 1) == 'b c!'\n"
        ),
    }


def prim_unique_chars() -> dict:
    return {
        "module_name": "unique",
        "buggy_src": (
            "def has_unique_chars(s):\n"
            "    return len(s) == len(set(s.upper()))\n"
        ),
        "gold_src": (
            "def has_unique_chars(s):\n"
            "    return len(s) == len(set(s))\n"
        ),
        "tests_src": (
            "from src.unique import has_unique_chars\n\n"
            "def test_yes():\n"
            "    assert has_unique_chars('abc')\n\n"
            "def test_no():\n"
            "    assert not has_unique_chars('aab')\n\n"
            "def test_case_sensitive():\n"
            "    assert has_unique_chars('aA')\n\n"
            "def test_empty():\n"
            "    assert has_unique_chars('')\n"
        ),
    }


def prim_runlength() -> dict:
    return {
        "module_name": "runs",
        "buggy_src": (
            "def runlength(s):\n"
            "    if not s:\n"
            "        return []\n"
            "    out = []\n"
            "    prev = s[0]\n"
            "    count = 0\n"
            "    for c in s:\n"
            "        if c == prev:\n"
            "            count += 1\n"
            "        else:\n"
            "            out.append((prev, count))\n"
            "            prev = c\n"
            "            count = 0\n"
            "    return out\n"
        ),
        "gold_src": (
            "def runlength(s):\n"
            "    if not s:\n"
            "        return []\n"
            "    out = []\n"
            "    prev = s[0]\n"
            "    count = 0\n"
            "    for c in s:\n"
            "        if c == prev:\n"
            "            count += 1\n"
            "        else:\n"
            "            out.append((prev, count))\n"
            "            prev = c\n"
            "            count = 1\n"
            "    out.append((prev, count))\n"
            "    return out\n"
        ),
        "tests_src": (
            "from src.runs import runlength\n\n"
            "def test_basic():\n"
            "    assert runlength('aaabb') == [('a', 3), ('b', 2)]\n\n"
            "def test_empty():\n"
            "    assert runlength('') == []\n\n"
            "def test_single():\n"
            "    assert runlength('x') == [('x', 1)]\n\n"
            "def test_alternate():\n"
            "    assert runlength('aba') == [('a', 1), ('b', 1), ('a', 1)]\n"
        ),
    }


def prim_min_max() -> dict:
    return {
        "module_name": "minmax",
        "buggy_src": (
            "def min_max(xs):\n"
            "    return min(xs), min(xs)\n"
        ),
        "gold_src": (
            "def min_max(xs):\n"
            "    return min(xs), max(xs)\n"
        ),
        "tests_src": (
            "from src.minmax import min_max\n\n"
            "def test_basic():\n"
            "    assert min_max([1, 2, 3]) == (1, 3)\n\n"
            "def test_single():\n"
            "    assert min_max([7]) == (7, 7)\n\n"
            "def test_negatives():\n"
            "    assert min_max([-1, -5, -3]) == (-5, -1)\n"
        ),
    }


def prim_titlecase() -> dict:
    return {
        "module_name": "title",
        "buggy_src": (
            "def titlecase(s):\n"
            "    return ' '.join(w.upper() for w in s.split())\n"
        ),
        "gold_src": (
            "def titlecase(s):\n"
            "    return ' '.join(w.capitalize() for w in s.split())\n"
        ),
        "tests_src": (
            "from src.title import titlecase\n\n"
            "def test_basic():\n"
            "    assert titlecase('hello world') == 'Hello World'\n\n"
            "def test_empty():\n"
            "    assert titlecase('') == ''\n\n"
            "def test_already_titled():\n"
            "    assert titlecase('Hello World') == 'Hello World'\n\n"
            "def test_mixed():\n"
            "    assert titlecase('hELLO wORLD') == 'Hello World'\n"
        ),
    }


def prim_anagram() -> dict:
    return {
        "module_name": "anagram",
        "buggy_src": (
            "def is_anagram(a, b):\n"
            "    return a == b\n"
        ),
        "gold_src": (
            "def is_anagram(a, b):\n"
            "    return sorted(a) == sorted(b)\n"
        ),
        "tests_src": (
            "from src.anagram import is_anagram\n\n"
            "def test_yes():\n"
            "    assert is_anagram('listen', 'silent')\n\n"
            "def test_no():\n"
            "    assert not is_anagram('listen', 'silenz')\n\n"
            "def test_same():\n"
            "    assert is_anagram('abc', 'abc')\n\n"
            "def test_lengths():\n"
            "    assert not is_anagram('abc', 'abcd')\n"
        ),
    }


def prim_chunked() -> dict:
    return {
        "module_name": "chunked",
        "buggy_src": (
            "def chunked(xs, n):\n"
            "    return [xs[i:i+n] for i in range(0, len(xs))]\n"
        ),
        "gold_src": (
            "def chunked(xs, n):\n"
            "    return [xs[i:i+n] for i in range(0, len(xs), n)]\n"
        ),
        "tests_src": (
            "from src.chunked import chunked\n\n"
            "def test_basic():\n"
            "    assert chunked([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]\n\n"
            "def test_uneven():\n"
            "    assert chunked([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\n\n"
            "def test_n_larger():\n"
            "    assert chunked([1, 2], 5) == [[1, 2]]\n\n"
            "def test_empty():\n"
            "    assert chunked([], 3) == []\n"
        ),
    }


def prim_word_count() -> dict:
    return {
        "module_name": "words",
        "buggy_src": (
            "def word_count(s):\n"
            "    if not s:\n"
            "        return 1\n"
            "    return len(s.split())\n"
        ),
        "gold_src": (
            "def word_count(s):\n"
            "    return len(s.split())\n"
        ),
        "tests_src": (
            "from src.words import word_count\n\n"
            "def test_basic():\n"
            "    assert word_count('hello world') == 2\n\n"
            "def test_empty():\n"
            "    assert word_count('') == 0\n\n"
            "def test_one():\n"
            "    assert word_count('hi') == 1\n\n"
            "def test_extra_space():\n"
            "    assert word_count('   a    b   ') == 2\n"
        ),
    }


def prim_clamp() -> dict:
    return {
        "module_name": "clamp",
        "buggy_src": (
            "def clamp(x, lo, hi):\n"
            "    return max(x, hi) if x < lo else min(x, lo)\n"
        ),
        "gold_src": (
            "def clamp(x, lo, hi):\n"
            "    return max(lo, min(x, hi))\n"
        ),
        "tests_src": (
            "from src.clamp import clamp\n\n"
            "def test_in_range():\n"
            "    assert clamp(5, 0, 10) == 5\n\n"
            "def test_below():\n"
            "    assert clamp(-3, 0, 10) == 0\n\n"
            "def test_above():\n"
            "    assert clamp(15, 0, 10) == 10\n\n"
            "def test_edge():\n"
            "    assert clamp(0, 0, 10) == 0\n"
        ),
    }


def prim_invert_dict() -> dict:
    return {
        "module_name": "invertd",
        "buggy_src": (
            "def invert_dict(d):\n"
            "    return d\n"
        ),
        "gold_src": (
            "def invert_dict(d):\n"
            "    return {v: k for k, v in d.items()}\n"
        ),
        "tests_src": (
            "from src.invertd import invert_dict\n\n"
            "def test_basic():\n"
            "    assert invert_dict({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}\n\n"
            "def test_empty():\n"
            "    assert invert_dict({}) == {}\n\n"
            "def test_single():\n"
            "    assert invert_dict({'x': 7}) == {7: 'x'}\n"
        ),
    }


def prim_strip_quotes() -> dict:
    return {
        "module_name": "stripq",
        "buggy_src": (
            "def strip_quotes(s):\n"
            "    if s.startswith('\"') and s.endswith('\"'):\n"
            "        return s[1:]\n"
            "    return s\n"
        ),
        "gold_src": (
            "def strip_quotes(s):\n"
            "    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('\"', \"'\"):\n"
            "        return s[1:-1]\n"
            "    return s\n"
        ),
        "tests_src": (
            "from src.stripq import strip_quotes\n\n"
            "def test_double():\n"
            "    assert strip_quotes('\"hi\"') == 'hi'\n\n"
            "def test_single():\n"
            "    assert strip_quotes(\"'hi'\") == 'hi'\n\n"
            "def test_none():\n"
            "    assert strip_quotes('hi') == 'hi'\n\n"
            "def test_mismatched():\n"
            "    assert strip_quotes(\"'hi\\\"\") == \"'hi\\\"\"\n"
        ),
    }


def prim_zip_longest() -> dict:
    return {
        "module_name": "ziplong",
        "buggy_src": (
            "def zip_longest_pairs(a, b, fill=None):\n"
            "    return list(zip(a, b))\n"
        ),
        "gold_src": (
            "from itertools import zip_longest as _z\n\n"
            "def zip_longest_pairs(a, b, fill=None):\n"
            "    return list(_z(a, b, fillvalue=fill))\n"
        ),
        "tests_src": (
            "from src.ziplong import zip_longest_pairs\n\n"
            "def test_equal_length():\n"
            "    assert zip_longest_pairs([1, 2], ['a', 'b']) == [(1, 'a'), (2, 'b')]\n\n"
            "def test_a_longer():\n"
            "    assert zip_longest_pairs([1, 2, 3], ['a']) == [(1, 'a'), (2, None), (3, None)]\n\n"
            "def test_fill():\n"
            "    assert zip_longest_pairs([1], ['a', 'b'], fill=0) == [(1, 'a'), (0, 'b')]\n"
        ),
    }


PRIMITIVES = [
    prim_add, prim_factorial, prim_fib, prim_reverse, prim_is_prime,
    prim_max, prim_count_vowels, prim_flatten, prim_dedup, prim_gcd,
    prim_palindrome, prim_sumlist, prim_filter_even, prim_caesar,
    prim_unique_chars, prim_runlength, prim_min_max, prim_titlecase,
    prim_anagram, prim_chunked, prim_word_count, prim_clamp,
    prim_invert_dict, prim_strip_quotes, prim_zip_longest,
]


# ---------------------------------------------------------------------------
# Task assembly
# ---------------------------------------------------------------------------

def assemble_task(
    prim_fn: Callable[[], dict],
    *,
    task_id: str,
    patch_class: str,  # 'clean' | 'drift' | 'incorrect'
    rng: random.Random,
) -> dict | None:
    """Build a task dict from a primitive."""
    prim = prim_fn()
    module = prim["module_name"]
    buggy = prim["buggy_src"]
    gold = prim["gold_src"]
    tests = prim["tests_src"]

    src_path = f"src/{module}.py"
    test_path = f"tests/test_{module}.py"
    init_pkg = "src/__init__.py"

    init_files = {
        src_path: buggy,
        test_path: tests,
        init_pkg: "",
        "conftest.py": "import sys, os\nsys.path.insert(0, os.path.dirname(__file__))\n",
    }
    gold_files = dict(init_files)
    gold_files[src_path] = gold

    # Canonical gold diff (buggy → gold).
    gold_diff = _make_unified_diff(src_path, buggy, gold)
    gold_diff_changed_lines = _count_diff_changed_lines(gold_diff)
    if gold_diff_changed_lines == 0:
        return None  # primitive's buggy == gold; skip.

    # Manufacture the patch the agent will see.
    if patch_class == "clean":
        patch_text = gold_diff
        expected_repair = False
    elif patch_class == "drift":
        shift = rng.choice([-3, -2, 2, 3])
        patch_text = _shift_hunk_headers(gold_diff, line_shift=shift)
        expected_repair = True
    elif patch_class == "incorrect":
        patch_text = _corrupt_hunk_line(gold_diff, rng=rng)
        expected_repair = True
    else:
        raise ValueError(f"unknown patch_class: {patch_class}")

    return {
        "task_id": task_id,
        "patch_class": patch_class,
        "module_name": module,
        "init_files": init_files,
        "gold_files": gold_files,
        "patch_text": patch_text,
        "gold_diff_text": gold_diff,
        "gold_diff_lines": gold_diff_changed_lines,
        "touched_paths": [src_path],
        "protected_paths": [test_path],
        "expected_repair": expected_repair,
        "verify_cmd": "python3 -m pytest -q tests/",
        "verify_timeout_s": 60,
    }


def stratified_pool(seed: int, n: int) -> list[tuple[Callable[[], dict], str]]:
    """Produce a (prim_fn, patch_class) tuples list of length n with the
    target mix from capability.config.json."""
    rng = random.Random(seed)
    n_clean = int(round(n * 0.50))
    n_drift = int(round(n * 0.30))
    n_incorrect = n - n_clean - n_drift  # absorb rounding
    classes = (
        ["clean"] * n_clean
        + ["drift"] * n_drift
        + ["incorrect"] * n_incorrect
    )
    rng.shuffle(classes)
    # Round-robin primitives, with shuffling each cycle.
    prims = list(PRIMITIVES)
    rng.shuffle(prims)
    pool: list[tuple[Callable[[], dict], str]] = []
    idx = 0
    cycle = list(prims)
    rng.shuffle(cycle)
    for cls in classes:
        prim_fn = cycle[idx]
        idx += 1
        if idx >= len(cycle):
            cycle = list(prims)
            rng.shuffle(cycle)
            idx = 0
        pool.append((prim_fn, cls))
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=60)
    ap.add_argument("--eval", type=int, default=24)
    ap.add_argument("--seed", type=int, default=3141592653)
    args = ap.parse_args()

    DATASETS.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    pool = stratified_pool(args.seed, args.train + args.eval)

    tasks: list[dict] = []
    for i, (prim_fn, cls) in enumerate(pool):
        task = assemble_task(
            prim_fn,
            task_id=f"task_{i:04d}",
            patch_class=cls,
            rng=rng,
        )
        if task is None:
            continue
        tasks.append(task)

    eval_set = tasks[: args.eval]
    train_set = tasks[args.eval:]

    eval_path = DATASETS / "eval.tasks.jsonl"
    train_path = DATASETS / "train.tasks.jsonl"

    with eval_path.open("w") as f:
        for t in eval_set:
            f.write(json.dumps(t) + "\n")
    with train_path.open("w") as f:
        for t in train_set:
            f.write(json.dumps(t) + "\n")

    def count_classes(ts):
        out: dict[str, int] = {}
        for t in ts:
            out[t["patch_class"]] = out.get(t["patch_class"], 0) + 1
        return out

    print(f"wrote {len(eval_set)} eval tasks ({count_classes(eval_set)}) -> {eval_path}")
    print(f"wrote {len(train_set)} train tasks ({count_classes(train_set)}) -> {train_path}")
    print(f"total: {len(tasks)} tasks across {len(PRIMITIVES)} primitives.")


if __name__ == "__main__":
    main()
