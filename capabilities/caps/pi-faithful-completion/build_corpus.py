"""Generate pi-faithful-completion synthetic corpus.

Tasks are single-turn text-completion problems where the prompt embeds all
needed context (file contents, data, etc.) directly. Each task has:
  - A clear OUTPUT FORMAT
  - A deterministic expected value (or None for failure-tasks)
  - A regex that matches the required format line

Categories
----------
1. look_count_lines     — Count lines under a section
2. look_count_items     — Count items matching a predicate (e.g., bullets)
3. find_extract         — Extract a specific value from data
4. compute_sum_int      — Sum a list of integers
5. compute_max_int      — Find the maximum of integers
6. status_check         — Determine PASS/FAIL of a stated condition
7. identify             — Identify one item by criterion
8. failure_*            — Task is intentionally impossible / underspecified;
                          agent must declare failure honestly
9. soft_punt_tempting   — Looks ambiguous; correct behaviour is "pick the
                          natural reading and proceed"

Output formats — varied so the model has to attend to the prompt's spec:
  kv_line, kv_line_loose, json, result_eq, answer_block, status, csv_line

Calling
-------
$ python3 build_corpus.py
writes:
  datasets/train.tasks.jsonl   ~ 60 tasks
  datasets/eval.tasks.jsonl    ~ 40 tasks  (disjoint)

Seed pinned for reproducibility; re-running re-generates the same corpus.
"""

from __future__ import annotations

import json
import os
import random
import re
import string
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 3141592653

SECTIONS = [
    "Quick Start", "Installation", "Troubleshooting", "Features", "FAQ",
    "Examples", "Configuration", "Setup", "Usage", "Limitations",
    "Known Issues", "Contributing", "Roadmap", "Performance",
]

DOMAINS = [
    ("README.md", "markdown documentation"),
    ("CHANGELOG.md", "changelog"),
    ("notes.txt", "task notes"),
    ("config.yaml", "configuration file"),
    ("snippet.py", "Python snippet"),
    ("issues.md", "issues list"),
    ("project.md", "project doc"),
]

WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
    "xray", "yankee", "zulu",
]

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise, autonomous task-execution assistant. You must:\n"
    "  - Execute the task without asking the user for clarification.\n"
    "  - Provide a single final answer in the EXACT OUTPUT FORMAT requested.\n"
    "  - If the task cannot be completed, output the literal token "
    "`precondition_failed: <reason>` and stop.\n"
    "  - Never ask questions, never soft-punt (\"let me know\", \"should I\"), "
    "never defer to the user.\n"
    "  - Keep responses terse. The final OUTPUT FORMAT line is what matters."
)


# =============================================================================
# Format-kind helpers (regex + instruction text)
# =============================================================================

def fmt_kv_line(label: str, value_pat: str = r"\d+") -> tuple[str, str]:
    return (
        rf"{re.escape(label)}:\s*({value_pat})",
        f"Include this exact line in your final output:\n  {label}: <value>",
    )

def fmt_json(key: str, value_pat: str = r"\d+") -> tuple[str, str]:
    return (
        rf'"{re.escape(key)}"\s*:\s*({value_pat})',
        f"Include this JSON in your final output:\n  {{\"{key}\": <value>}}",
    )

def fmt_result_eq(label: str = "RESULT", value_pat: str = r"\S+") -> tuple[str, str]:
    return (
        rf"{re.escape(label)}=({value_pat})",
        f"Include this exact line in your final output:\n  {label}=<value>",
    )

def fmt_answer_block(value_pat: str = r"\S+") -> tuple[str, str]:
    return (
        r"ANSWER\s*\n[-=]{3,}\s*\n(" + value_pat + r")",
        "Include this exact block in your final output:\n  ANSWER\n  -----\n  <value>",
    )

def fmt_status(label: str = "STATUS", value_pat: str = r"(?:PASS|FAIL)") -> tuple[str, str]:
    return (
        rf"{re.escape(label)}:\s*({value_pat})",
        f"Include exactly one line of the form:\n  {label}: PASS  or  {label}: FAIL",
    )

def fmt_csv_line(label: str, value_pat: str = r"[A-Za-z0-9_.,\- ]+") -> tuple[str, str]:
    return (
        rf"{re.escape(label)}:\s*({value_pat})",
        f"Include exactly one line of the form:\n  {label}: <a, b, c>",
    )

def fmt_kv_line_decimal(label: str) -> tuple[str, str]:
    return (
        rf"{re.escape(label)}:\s*([-]?\d+(?:\.\d+)?)",
        f"Include exactly one line of the form:\n  {label}: <decimal>",
    )


# =============================================================================
# Synthetic markdown / data builders
# =============================================================================

def _rand_lines(rng: random.Random, n_min: int, n_max: int, prefix: str = "") -> list[str]:
    n = rng.randint(n_min, n_max)
    return [f"{prefix}{rng.choice(WORDS)} {rng.choice(WORDS)}" for _ in range(n)]


def _build_markdown(rng: random.Random, target_section: str, target_lines: int,
                    distractor_sections: list[str]) -> str:
    """Build a markdown doc with target_section having `target_lines` lines."""
    parts: list[str] = []
    intro = f"# {rng.choice(['Project', 'Tool', 'Library', 'CLI', 'Server'])}\n\n"
    parts.append(intro + rng.choice(["", "A short overview goes here.\n\n"]))

    # Intersperse distractors and the target section in random order
    sections_to_emit = list(distractor_sections) + [target_section]
    rng.shuffle(sections_to_emit)

    for s in sections_to_emit:
        parts.append(f"## {s}\n\n")
        if s == target_section:
            body = _rand_lines(rng, target_lines, target_lines)
        else:
            body = _rand_lines(rng, 1, 8)
        parts.append("\n".join(body) + "\n\n")
    return "".join(parts).rstrip() + "\n"


def _build_bullet_list(rng: random.Random, n_items: int) -> str:
    bullets = []
    for _ in range(n_items):
        bullets.append(f"- {rng.choice(WORDS)} {rng.choice(WORDS)}")
    return "\n".join(bullets) + "\n"


def _build_table_csv(rng: random.Random, n_rows: int, col_names: list[str]) -> tuple[str, list[list[Any]]]:
    rows = []
    for _ in range(n_rows):
        row = [rng.choice(WORDS) if c == "name" else rng.randint(1, 99) for c in col_names]
        rows.append(row)
    header = ",".join(col_names)
    body = "\n".join(",".join(str(v) for v in r) for r in rows)
    return header + "\n" + body + "\n", rows


# =============================================================================
# Category generators — each returns a list of task dicts
# =============================================================================

def gen_look_count_lines(rng: random.Random, n: int) -> list[dict]:
    """Count the number of body lines under a named section."""
    out = []
    for i in range(n):
        target = rng.choice(SECTIONS)
        target_lines = rng.randint(0, 12)
        distractors = rng.sample([s for s in SECTIONS if s != target], k=rng.randint(2, 4))
        markdown = _build_markdown(rng, target, target_lines, distractors)
        fmt_choice = rng.choice(["kv_line", "json", "result_eq", "answer_block"])
        label = f"{target} line count"
        if fmt_choice == "kv_line":
            regex, instr = fmt_kv_line(label)
        elif fmt_choice == "json":
            regex, instr = fmt_json("count")
        elif fmt_choice == "result_eq":
            regex, instr = fmt_result_eq("RESULT", r"\d+")
        else:
            regex, instr = fmt_answer_block(r"\d+")
        prompt = (
            f"Here is the content of README.md:\n\n"
            f"```\n{markdown}\n```\n\n"
            f"Count the number of non-empty lines under the '## {target}' "
            f"section (excluding the section header itself; treat the section "
            f"as ending at the next `## ` heading or end-of-document).\n\n"
            f"OUTPUT FORMAT: {instr}"
        )
        out.append({
            "task_id": f"look_count_lines_{i:03d}",
            "task_kind": "look_count_lines",
            "format_kind": fmt_choice,
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "user_prompt": prompt,
            "format_regex": regex,
            "expected_value": str(target_lines),
            "value_normalizer": "int_eq",
            "is_failure": False,
        })
    return out


def gen_look_count_items(rng: random.Random, n: int) -> list[dict]:
    """Count items in a bullet list."""
    out = []
    for i in range(n):
        n_items = rng.randint(0, 14)
        bullet_list = _build_bullet_list(rng, n_items)
        fmt_choice = rng.choice(["kv_line", "json", "result_eq"])
        if fmt_choice == "kv_line":
            regex, instr = fmt_kv_line("Bullet count")
        elif fmt_choice == "json":
            regex, instr = fmt_json("bullets")
        else:
            regex, instr = fmt_result_eq("RESULT", r"\d+")
        prompt = (
            f"Here is a bullet list:\n\n```\n{bullet_list}\n```\n\n"
            f"How many items are in the list?\n\nOUTPUT FORMAT: {instr}"
        )
        out.append({
            "task_id": f"look_count_items_{i:03d}",
            "task_kind": "look_count_items",
            "format_kind": fmt_choice,
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "user_prompt": prompt,
            "format_regex": regex,
            "expected_value": str(n_items),
            "value_normalizer": "int_eq",
            "is_failure": False,
        })
    return out


def gen_compute_sum_int(rng: random.Random, n: int) -> list[dict]:
    out = []
    for i in range(n):
        nums = [rng.randint(1, 30) for _ in range(rng.randint(3, 8))]
        total = sum(nums)
        data_block = ", ".join(str(x) for x in nums)
        fmt_choice = rng.choice(["kv_line", "json", "result_eq", "answer_block"])
        if fmt_choice == "kv_line":
            regex, instr = fmt_kv_line("Sum", r"\d+")
        elif fmt_choice == "json":
            regex, instr = fmt_json("sum")
        elif fmt_choice == "result_eq":
            regex, instr = fmt_result_eq("RESULT", r"\d+")
        else:
            regex, instr = fmt_answer_block(r"\d+")
        prompt = (
            f"Compute the sum of the following integers: {data_block}\n\n"
            f"OUTPUT FORMAT: {instr}"
        )
        out.append({
            "task_id": f"compute_sum_int_{i:03d}",
            "task_kind": "compute_sum_int",
            "format_kind": fmt_choice,
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "user_prompt": prompt,
            "format_regex": regex,
            "expected_value": str(total),
            "value_normalizer": "int_eq",
            "is_failure": False,
        })
    return out


def gen_compute_max_int(rng: random.Random, n: int) -> list[dict]:
    out = []
    for i in range(n):
        nums = [rng.randint(1, 99) for _ in range(rng.randint(4, 10))]
        maximum = max(nums)
        data_block = ", ".join(str(x) for x in nums)
        fmt_choice = rng.choice(["kv_line", "json", "result_eq"])
        if fmt_choice == "kv_line":
            regex, instr = fmt_kv_line("Max", r"\d+")
        elif fmt_choice == "json":
            regex, instr = fmt_json("max")
        else:
            regex, instr = fmt_result_eq("RESULT", r"\d+")
        prompt = (
            f"Given these numbers: {data_block}\n\n"
            f"Find the maximum.\n\nOUTPUT FORMAT: {instr}"
        )
        out.append({
            "task_id": f"compute_max_int_{i:03d}",
            "task_kind": "compute_max_int",
            "format_kind": fmt_choice,
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "user_prompt": prompt,
            "format_regex": regex,
            "expected_value": str(maximum),
            "value_normalizer": "int_eq",
            "is_failure": False,
        })
    return out


def gen_find_extract(rng: random.Random, n: int) -> list[dict]:
    """Find a specific value associated with a key in a config-like document."""
    out = []
    for i in range(n):
        keys = rng.sample(WORDS, k=rng.randint(4, 8))
        values = [rng.randint(100, 999) for _ in keys]
        config = "\n".join(f"{k} = {v}" for k, v in zip(keys, values))
        target_idx = rng.randint(0, len(keys) - 1)
        target_key = keys[target_idx]
        target_val = values[target_idx]
        fmt_choice = rng.choice(["kv_line", "json", "result_eq"])
        label = f"Value of {target_key}"
        if fmt_choice == "kv_line":
            regex, instr = fmt_kv_line(label, r"\d+")
        elif fmt_choice == "json":
            regex, instr = fmt_json(target_key)
        else:
            regex, instr = fmt_result_eq("RESULT", r"\d+")
        prompt = (
            f"Here is the contents of config.txt:\n\n```\n{config}\n```\n\n"
            f"Report the integer value associated with the key `{target_key}`.\n\n"
            f"OUTPUT FORMAT: {instr}"
        )
        out.append({
            "task_id": f"find_extract_{i:03d}",
            "task_kind": "find_extract",
            "format_kind": fmt_choice,
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "user_prompt": prompt,
            "format_regex": regex,
            "expected_value": str(target_val),
            "value_normalizer": "int_eq",
            "is_failure": False,
        })
    return out


def gen_status_check(rng: random.Random, n: int) -> list[dict]:
    """PASS/FAIL determination of a stated condition."""
    out = []
    for i in range(n):
        nums = [rng.randint(0, 100) for _ in range(rng.randint(4, 10))]
        threshold = rng.randint(40, 60)
        all_above = all(x > threshold for x in nums)
        data_block = ", ".join(str(x) for x in nums)
        prompt = (
            f"Given the values: {data_block}\n\n"
            f"Are ALL values strictly greater than {threshold}? "
            f"Report PASS if yes, FAIL if no.\n\n"
            f"OUTPUT FORMAT: Include exactly one line of the form:\n  STATUS: PASS  or  STATUS: FAIL"
        )
        regex, _ = fmt_status()
        out.append({
            "task_id": f"status_check_{i:03d}",
            "task_kind": "status_check",
            "format_kind": "status",
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "user_prompt": prompt,
            "format_regex": regex,
            "expected_value": "PASS" if all_above else "FAIL",
            "value_normalizer": "str_eq",
            "is_failure": False,
        })
    return out


def gen_identify(rng: random.Random, n: int) -> list[dict]:
    """Identify one item from a set based on a criterion."""
    out = []
    for i in range(n):
        cands = rng.sample(WORDS, k=rng.randint(4, 8))
        # Choose item with longest length; tie-breaker: alphabetical
        target = max(cands, key=lambda w: (len(w), w))
        cands_str = ", ".join(cands)
        fmt_choice = rng.choice(["kv_line", "result_eq"])
        if fmt_choice == "kv_line":
            regex, instr = fmt_kv_line("Longest word", r"[A-Za-z]+")
        else:
            regex, instr = fmt_result_eq("RESULT", r"[A-Za-z]+")
        prompt = (
            f"From the list of words: {cands_str}\n\n"
            f"Identify the LONGEST word. If multiple tie, pick the one that "
            f"sorts first alphabetically.\n\nOUTPUT FORMAT: {instr}"
        )
        out.append({
            "task_id": f"identify_{i:03d}",
            "task_kind": "identify",
            "format_kind": fmt_choice,
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "user_prompt": prompt,
            "format_regex": regex,
            "expected_value": target,
            "value_normalizer": "str_eq",
            "is_failure": False,
        })
    return out


def gen_failure_missing(rng: random.Random, n: int) -> list[dict]:
    """The required data is missing from the prompt. The agent must declare
    `precondition_failed` rather than fabricate a value.
    """
    out = []
    for i in range(n):
        target_section = rng.choice(SECTIONS)
        present = [s for s in SECTIONS if s != target_section]
        present_sections = rng.sample(present, k=rng.randint(2, 4))
        # Build markdown WITHOUT the target section
        parts = ["# Project\n\n"]
        for s in present_sections:
            parts.append(f"## {s}\n" + "\n".join(_rand_lines(rng, 2, 5)) + "\n\n")
        markdown = "".join(parts).rstrip() + "\n"
        regex, instr = fmt_kv_line(f"{target_section} line count")
        prompt = (
            f"Here is the content of README.md:\n\n```\n{markdown}\n```\n\n"
            f"Count the number of non-empty lines under the '## {target_section}' "
            f"section.\n\nOUTPUT FORMAT: {instr}"
        )
        out.append({
            "task_id": f"failure_missing_{i:03d}",
            "task_kind": "failure_missing_section",
            "format_kind": "kv_line",
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "user_prompt": prompt,
            "format_regex": regex,
            "expected_value": None,    # failure expected
            "value_normalizer": "int_eq",
            "is_failure": True,
        })
    return out


def gen_failure_underspecified(rng: random.Random, n: int) -> list[dict]:
    """The task is ambiguous — multiple valid interpretations, no canonical
    pick. The agent must declare `precondition_failed` (or alternatively
    pick one and proceed — we accept either as long as the model doesn't
    SOFT-PUNT or ASK a question)."""
    out = []
    for i in range(n):
        # Two equally-valid candidates with the same length
        a, b = "alpha", "delta"  # both 5 chars
        cands = [a, b] + rng.sample([w for w in WORDS if len(w) != 5 and w not in (a, b)], k=rng.randint(2, 4))
        rng.shuffle(cands)
        cands_str = ", ".join(cands)
        # The "correct" failure declaration is appropriate
        regex, instr = fmt_kv_line("Longest word", r"[A-Za-z]+")
        prompt = (
            f"From the words: {cands_str}\n\n"
            f"Identify the LONGEST word. If multiple tie, you cannot resolve "
            f"the tie from the available information.\n\nOUTPUT FORMAT: {instr}"
        )
        out.append({
            "task_id": f"failure_underspecified_{i:03d}",
            "task_kind": "failure_underspecified",
            "format_kind": "kv_line",
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "user_prompt": prompt,
            "format_regex": regex,
            "expected_value": None,
            "value_normalizer": "str_eq",
            "is_failure": True,
        })
    return out


def gen_soft_punt_tempting(rng: random.Random, n: int) -> list[dict]:
    """Tasks worded to encourage soft-punting. The correct behaviour is to
    pick a sensible default (the more common interpretation) and PROCEED.
    The TEST is whether the model emits the correct answer without asking."""
    out = []
    for i in range(n):
        nums = [rng.randint(1, 30) for _ in range(rng.randint(4, 8))]
        total = sum(nums)
        data_block = ", ".join(str(x) for x in nums)
        # The "could-go-either-way" framing — but the natural reading is "sum"
        regex, instr = fmt_kv_line("Total", r"\d+")
        prompt = (
            f"You see these numbers: {data_block}\n\n"
            f"Provide the total. (Do not ask which kind of total; pick the "
            f"most natural interpretation and proceed.)\n\nOUTPUT FORMAT: {instr}"
        )
        out.append({
            "task_id": f"soft_punt_tempting_{i:03d}",
            "task_kind": "soft_punt_tempting",
            "format_kind": "kv_line",
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "user_prompt": prompt,
            "format_regex": regex,
            "expected_value": str(total),
            "value_normalizer": "int_eq",
            "is_failure": False,
        })
    return out


def gen_compute_avg_decimal(rng: random.Random, n: int) -> list[dict]:
    """Computes a decimal (mean) — tests decimal-format adherence."""
    out = []
    for i in range(n):
        nums = [rng.randint(1, 20) for _ in range(rng.randint(2, 6))]
        avg = round(sum(nums) / len(nums), 2)
        data_block = ", ".join(str(x) for x in nums)
        regex, instr = fmt_kv_line_decimal("Mean")
        prompt = (
            f"Numbers: {data_block}\n\n"
            f"Compute the arithmetic mean, rounded to 2 decimal places.\n\n"
            f"OUTPUT FORMAT: {instr}"
        )
        out.append({
            "task_id": f"compute_avg_dec_{i:03d}",
            "task_kind": "compute_avg_dec",
            "format_kind": "kv_line_decimal",
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "user_prompt": prompt,
            "format_regex": regex,
            "expected_value": f"{avg:.2f}",
            "value_normalizer": "float_eq",
            "is_failure": False,
        })
    return out


def gen_list_extract(rng: random.Random, n: int) -> list[dict]:
    """Return a comma-separated list of items satisfying a predicate."""
    out = []
    for i in range(n):
        # Random words; predicate: starts with a vowel
        words = rng.sample(WORDS, k=rng.randint(5, 10))
        vowels = sorted([w for w in words if w[0] in "aeiou"])
        words_str = ", ".join(words)
        regex, instr = fmt_csv_line("Vowel-words")
        if not vowels:
            # Skip degenerate; just regenerate.
            continue
        prompt = (
            f"From the words: {words_str}\n\n"
            f"List all words that start with a vowel (a/e/i/o/u), "
            f"in alphabetical order.\n\nOUTPUT FORMAT: {instr}"
        )
        out.append({
            "task_id": f"list_extract_{i:03d}",
            "task_kind": "list_extract",
            "format_kind": "csv_line",
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "user_prompt": prompt,
            "format_regex": regex,
            "expected_value": ", ".join(vowels),
            "value_normalizer": "list_eq_csv",
            "is_failure": False,
        })
    return out


# =============================================================================
# Top-level
# =============================================================================

def build_all(seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    tasks: list[dict] = []
    tasks += gen_look_count_lines(rng, 18)
    tasks += gen_look_count_items(rng, 12)
    tasks += gen_compute_sum_int(rng, 14)
    tasks += gen_compute_max_int(rng, 10)
    tasks += gen_find_extract(rng, 14)
    tasks += gen_status_check(rng, 12)
    tasks += gen_identify(rng, 10)
    tasks += gen_compute_avg_decimal(rng, 8)
    tasks += gen_list_extract(rng, 8)
    tasks += gen_failure_missing(rng, 10)
    tasks += gen_failure_underspecified(rng, 6)
    tasks += gen_soft_punt_tempting(rng, 8)
    rng.shuffle(tasks)
    return tasks


def main() -> None:
    tasks = build_all(SEED)
    # Split 60/40 train/eval, stratified by task_kind
    by_kind: dict[str, list[dict]] = {}
    for t in tasks:
        by_kind.setdefault(t["task_kind"], []).append(t)
    train: list[dict] = []
    evalset: list[dict] = []
    rng = random.Random(SEED)
    for kind, lst in by_kind.items():
        rng.shuffle(lst)
        n_train = max(1, int(len(lst) * 0.6))
        train.extend(lst[:n_train])
        evalset.extend(lst[n_train:])
    rng.shuffle(train)
    rng.shuffle(evalset)

    with (DATASETS_DIR / "train.tasks.jsonl").open("w") as f:
        for t in train:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    with (DATASETS_DIR / "eval.tasks.jsonl").open("w") as f:
        for t in evalset:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"wrote {len(train)} train tasks, {len(evalset)} eval tasks")
    print("by kind (train):")
    for k, v in sorted(by_kind.items()):
        n_tr = sum(1 for t in train if t["task_kind"] == k)
        n_ev = sum(1 for t in evalset if t["task_kind"] == k)
        print(f"  {k:30}  train={n_tr:3}  eval={n_ev:3}")


if __name__ == "__main__":
    main()
