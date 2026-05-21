"""Task-corpus builder for sft/math-broad.

Word problems and prose-style math (not symbolic). Output:
  datasets/train.jsonl   (SFT training data)
  datasets/eval.tasks.jsonl  (gitignored eval)
"""
from __future__ import annotations
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"
SEED = int(os.environ.get("SEED", 31426))


PROBLEMS = [
    {
        "id": "prose_addition",
        "prompt": "If Alice has 3 apples and Bob gives her 7 more, how many apples does Alice have?",
        "answer": "10",
    },
    {
        "id": "prose_mult",
        "prompt": "A pizza is cut into 8 slices. If 3 people each eat 2 slices, how many slices are left?",
        "answer": "2",
    },
    {
        "id": "prose_division",
        "prompt": "There are 24 cookies to share equally among 6 children. How many cookies does each child get?",
        "answer": "4",
    },
    {
        "id": "prose_time",
        "prompt": "A train leaves at 9:15 AM and arrives at 1:45 PM. How long is the trip in hours and minutes?",
        "answer": "4 hours 30 minutes",
    },
    {
        "id": "prose_percent",
        "prompt": "A shirt costs $40 and is on sale for 25% off. What's the sale price?",
        "answer": "$30",
    },
    {
        "id": "prose_avg",
        "prompt": "Test scores: 80, 90, 70, 85, 75. What is the average?",
        "answer": "80",
    },
]


def main():
    DATASETS.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    train, eval_ = [], []
    for p in PROBLEMS:
        for i in range(8):
            train.append({
                "task_id": f"{p['id']}_{i:03d}",
                "prompt": p["prompt"],
                "completion": p["answer"],
            })
        for i in range(3):
            eval_.append({
                "task_id": f"{p['id']}_eval_{i:03d}",
                "prompt": p["prompt"],
                "gold_answer": p["answer"],
            })
    rng.shuffle(train)
    rng.shuffle(eval_)
    with open(DATASETS / "train.jsonl", "w") as f:
        for t in train:
            f.write(json.dumps(t) + "\n")
    with open(DATASETS / "eval.tasks.jsonl", "w") as f:
        for t in eval_:
            f.write(json.dumps(t) + "\n")
    print(f"wrote {len(train)} train + {len(eval_)} eval")


if __name__ == "__main__":
    main()
