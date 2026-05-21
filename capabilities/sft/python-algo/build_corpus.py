"""Task-corpus builder for sft/python-algo.

Algorithmic programming problems with hidden tests. Output:
  datasets/train.jsonl   (SFT training data)
  datasets/eval.tasks.jsonl  (gitignored eval)

Tasks have prompt + reference solution. SFT trains on the reference;
eval scores accuracy.
"""
from __future__ import annotations
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"
SEED = int(os.environ.get("SEED", 31425))

PROBLEMS = [
    {
        "id": "fib",
        "prompt": "Write a Python function `fib(n)` that returns the nth Fibonacci number (0-indexed). fib(0)=0, fib(1)=1.",
        "solution": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
        "tests": [(0, 0), (1, 1), (10, 55)],
    },
    {
        "id": "reverse_list",
        "prompt": "Write `reverse_list(lst)` that returns the reverse of a list without using slicing.",
        "solution": "def reverse_list(lst):\n    out = []\n    for x in lst:\n        out.insert(0, x)\n    return out\n",
        "tests": [([1,2,3], [3,2,1]), ([], [])],
    },
    {
        "id": "binary_search",
        "prompt": "Write `binary_search(arr, target)` returning the index of target in sorted arr, or -1.",
        "solution": "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target: return mid\n        if arr[mid] < target: lo = mid + 1\n        else: hi = mid - 1\n    return -1\n",
        "tests": [([1,2,3,4,5], 3, 2), ([], 1, -1)],
    },
    {
        "id": "merge_sorted",
        "prompt": "Write `merge(a, b)` that merges two sorted lists into one sorted list.",
        "solution": "def merge(a, b):\n    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]: out.append(a[i]); i += 1\n        else: out.append(b[j]); j += 1\n    out.extend(a[i:]); out.extend(b[j:])\n    return out\n",
        "tests": [([1,3], [2,4], [1,2,3,4])],
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
                "completion": p["solution"],
            })
        for i in range(3):
            eval_.append({
                "task_id": f"{p['id']}_eval_{i:03d}",
                "prompt": p["prompt"],
                "reference_solution": p["solution"],
                "tests": p["tests"],
            })
    rng.shuffle(train)
    rng.shuffle(eval_)
    with open(DATASETS / "train.jsonl", "w") as f:
        for t in train:
            f.write(json.dumps(t) + "\n")
    with open(DATASETS / "eval.tasks.jsonl", "w") as f:
        for t in eval_:
            f.write(json.dumps(t) + "\n")
    print(f"wrote {len(train)} train (SFT) + {len(eval_)} eval tasks")


if __name__ == "__main__":
    main()
