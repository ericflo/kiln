"""Build train/eval task split for pi-terminal-bench-lite.

Reads tasks from a source JSONL (the OpenThoughts-TBLite v1 export) and
emits 70-task train + 30-task eval splits compatible with rollout.py.

Until the upstream TBLite export is staged on the pod, this script can
also synthesize a minimal placeholder corpus from a Python algorithm
seed for end-to-end shape validation.

Usage:
    # From the real TBLite jsonl:
    python build_corpus.py --source /workspace/tblite/tasks.jsonl

    # Synthesized placeholder (no upstream data required):
    python build_corpus.py --synthesize
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def synthesize_placeholder_tasks(n: int = 12) -> list[dict]:
    """A tiny placeholder corpus so the cap structure can be smoke-tested
    end-to-end before the real TBLite export is wired in. Each task is a
    small file-manipulation problem with a deterministic verifier."""
    tasks: list[dict] = []
    for i in range(n):
        target_letter = chr(ord("a") + (i % 26))
        tasks.append({
            "task_id": f"placeholder-{i:04d}",
            "category": "data_processing",
            "scaffold_files": {
                "input.txt": "alpha\nbeta\ngamma\n",
                "README.md": (
                    f"Write the first line of input.txt to output.txt, "
                    f"and prepend the letter '{target_letter}' to it."
                ),
            },
            "user_prompt": (
                f"Read input.txt. Write a new file output.txt containing "
                f"'{target_letter}' followed by the first line of input.txt."
            ),
            "verifier": f"test -f output.txt && grep -q '^{target_letter}alpha' output.txt",
            "verifier_timeout_s": 10,
            "expected_files": {},
        })
    return tasks


def split_train_eval(tasks: list[dict], n_train: int, n_eval: int, seed: int = 42) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    rng.shuffle(tasks)
    return tasks[:n_train], tasks[n_train : n_train + n_eval]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", help="upstream TBLite tasks JSONL")
    p.add_argument("--synthesize", action="store_true", help="emit placeholder corpus")
    p.add_argument("--n-train", type=int, default=70)
    p.add_argument("--n-eval", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.source and not args.synthesize:
        p.error("either --source or --synthesize is required")

    out_train = HERE / "datasets" / "train.tasks.jsonl"
    out_eval = HERE / "datasets" / "eval.tasks.jsonl"
    out_train.parent.mkdir(parents=True, exist_ok=True)

    if args.synthesize:
        # Synthesize a corpus big enough for both splits.
        total = max(args.n_train + args.n_eval, args.n_train, args.n_eval)
        tasks = synthesize_placeholder_tasks(total)
    else:
        tasks = []
        with Path(args.source).open() as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))

    train, eval_ = split_train_eval(tasks, args.n_train, args.n_eval, args.seed)
    out_train.write_text("\n".join(json.dumps(t) for t in train) + "\n")
    out_eval.write_text("\n".join(json.dumps(t) for t in eval_) + "\n")
    print(f"wrote {len(train)} train tasks → {out_train}")
    print(f"wrote {len(eval_)} eval tasks → {out_eval}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
