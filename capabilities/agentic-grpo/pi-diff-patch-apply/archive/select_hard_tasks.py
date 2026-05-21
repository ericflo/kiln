"""Select a subset of train.tasks.jsonl biased toward drift + incorrect
classes (where the base model has the most failure variance).

Usage:
    python3 select_hard_tasks.py --tasks datasets/train.tasks.jsonl \\
        --out datasets/train.hard.tasks.jsonl \\
        --total 12 \\
        --clean-pct 0.25 \\
        --drift-pct 0.42 \\
        --incorrect-pct 0.33
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--total", type=int, default=12)
    ap.add_argument("--clean-pct", type=float, default=0.25)
    ap.add_argument("--drift-pct", type=float, default=0.42)
    ap.add_argument("--incorrect-pct", type=float, default=0.33)
    ap.add_argument("--seed", type=int, default=3141592653)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tasks = {"clean": [], "drift": [], "incorrect": []}
    with open(args.tasks) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            cls = t.get("patch_class", "clean")
            if cls in tasks:
                tasks[cls].append(t)

    n_clean = int(round(args.total * args.clean_pct))
    n_drift = int(round(args.total * args.drift_pct))
    n_incorrect = args.total - n_clean - n_drift

    selected = []
    for cls, n in [("clean", n_clean), ("drift", n_drift), ("incorrect", n_incorrect)]:
        rng.shuffle(tasks[cls])
        selected.extend(tasks[cls][:n])
    rng.shuffle(selected)

    Path(args.out).write_text("\n".join(json.dumps(t) for t in selected) + "\n")
    print(f"wrote {len(selected)} tasks ({n_clean} clean / {n_drift} drift / {n_incorrect} incorrect) -> {args.out}")


if __name__ == "__main__":
    main()
