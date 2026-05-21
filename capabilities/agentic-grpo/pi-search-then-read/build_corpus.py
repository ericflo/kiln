"""Task-corpus builder for pi-search-then-read.

Each task is a workspace with at least one large file (≥200 lines)
plus a natural-language query that requires reading a small window of
that file. The rubric measures whether the agent searched first and
how much it read.

Output:
  datasets/train.tasks.jsonl  (24 tasks across 3 size tiers)
  datasets/eval.tasks.jsonl   (12 held-out)
"""
from __future__ import annotations
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"
SEED = int(os.environ.get("SEED", 31418))


def _make_large_file(rng: random.Random, n_lines: int, target_line: int, target_name: str) -> str:
    lines = []
    target_emitted = False
    for i in range(n_lines):
        if not target_emitted and i + 1 >= target_line:
            lines.append(f"def {target_name}(x: int, y: int) -> int:")
            lines.append("    \"\"\"Add two integers and return the result.\"\"\"")
            lines.append("    return x + y")
            target_emitted = True
            continue
        fn_idx = i // 4
        mod = i % 4
        if mod == 0:
            lines.append(f"def util_fn_{fn_idx}(arg: str) -> str:")
        elif mod == 1:
            lines.append(f"    return arg.upper() + '_{fn_idx}'")
        elif mod == 2:
            lines.append("")
        else:
            lines.append(f"# Helper {fn_idx}")
    return "\n".join(lines) + "\n"


SIZE_TIERS = [
    ("small", 200),
    ("medium", 800),
    ("large", 2000),
]

QUERIES = [
    ("return_type", "What does `{target}` return?", "int"),
    ("first_arg_name", "What's the first argument name of `{target}`?", "x"),
    ("docstring_first_word", "What's the first word of the docstring of `{target}`?", "Add"),
]


def _make_task(rng: random.Random, idx: int, tier_name: str, n_lines: int) -> dict:
    target_name = f"compute_{rng.randint(100, 999)}"
    target_line = rng.randint(int(n_lines * 0.3), int(n_lines * 0.7))
    query_kind, query_template, gold_answer = rng.choice(QUERIES)
    content = _make_large_file(rng, n_lines, target_line, target_name)
    return {
        "task_id": f"search_read_{tier_name}_{idx:03d}",
        "size_tier": tier_name,
        "init_files": {"lib/big_module.py": content},
        "prompt": (
            f"Answer this question about `lib/big_module.py`:\n"
            f"{query_template.format(target=target_name)}\n"
            f"Cite the file:line where you found the answer in your final reply."
        ),
        "query_kind": query_kind,
        "target_file": "lib/big_module.py",
        "target_symbol": target_name,
        "gold_window_line_start": max(1, target_line - 1),
        "gold_window_line_end": target_line + 3,
        "gold_answer": gold_answer,
        "file_size_lines": n_lines,
    }


def main():
    DATASETS.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    train, eval_ = [], []
    for tier_name, n_lines in SIZE_TIERS:
        for i in range(8):
            train.append(_make_task(rng, i, tier_name, n_lines))
        for i in range(4):
            eval_.append(_make_task(rng, 100 + i, tier_name, n_lines))
    rng.shuffle(train)
    rng.shuffle(eval_)
    with open(DATASETS / "train.tasks.jsonl", "w") as f:
        for t in train:
            f.write(json.dumps(t) + "\n")
    with open(DATASETS / "eval.tasks.jsonl", "w") as f:
        for t in eval_:
            f.write(json.dumps(t) + "\n")
    print(f"wrote {len(train)} train tasks (8 small + 8 medium + 8 large)")
    print(f"wrote {len(eval_)} eval tasks (4 small + 4 medium + 4 large)")


if __name__ == "__main__":
    main()
