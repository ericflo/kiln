"""Task-corpus builder for pi-tool-call-efficiency (eval-only).

Round 2 reshape: this cap is repurposed as a TRANSFER EVAL that
measures tool-call efficiency across ADAPTERS trained on OTHER caps,
not a standalone training cap.
"""
from __future__ import annotations
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"
SEED = int(os.environ.get("SEED", 31422))

SISTER_CAPS = [
    "pi-doctest",
    "pi-code-search",
    "pi-code-comprehension",
    "pi-error-recovery",
    "pi-search-then-read",
]


def _fallback_tasks() -> list[dict]:
    out = []
    for i in range(8):
        out.append({
            "task_id": f"tce_simple_{i:03d}",
            "scenario": "minimal_steps",
            "init_files": {"lib/util.py": f"def f{i}(x): return x + {i}\n"},
            "prompt": f"What does `f{i}` return when called with 3? Use the minimum number of tool calls.",
            "expected_n_tool_calls": 2,
            "expected_answer": str(i + 3),
        })
    return out


def main():
    DATASETS.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    train, eval_ = [], []
    sampled = 0
    for cap in SISTER_CAPS:
        sister_eval = HERE.parent / cap / "datasets" / "eval.tasks.jsonl"
        if not sister_eval.exists():
            continue
        with open(sister_eval) as f:
            tasks = [json.loads(line) for line in f if line.strip()]
        for t in tasks[:3]:
            t = dict(t)
            t["_source_cap"] = cap
            eval_.append(t)
            sampled += 1
    train.extend(_fallback_tasks())
    if not eval_:
        eval_.extend(_fallback_tasks()[:6])
    with open(DATASETS / "train.tasks.jsonl", "w") as f:
        for t in train:
            f.write(json.dumps(t) + "\n")
    with open(DATASETS / "eval.tasks.jsonl", "w") as f:
        for t in eval_:
            f.write(json.dumps(t) + "\n")
    print(f"wrote {len(train)} train + {len(eval_)} eval ({sampled} from sisters)")
    print("NOTE: this cap is EVAL-ONLY. run_iter.sh measures, doesn't train.")


if __name__ == "__main__":
    main()
