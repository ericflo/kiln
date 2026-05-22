"""iter17 hard-tail prep: identify and target hard tasks.

Strategy:
1. Read iter5 rollouts (292 strict-prompt rollouts, 73 tasks × 4 gens).
2. Group by task_id, find tasks where the MAX composite across the 4
   generations is LOW (< 0.7). These are the "hard tail" — strict
   prompt can't reliably get a good answer on them.
3. Identify the corresponding tasks in train.tasks.jsonl.
4. Output:
   - hard_tasks.jsonl: just the hard tasks (subset for re-rollout)
   - List of task_ids for verification

Hypothesis: iter8 has captured the easy tail. Targeted training on
the hard tail (with more rollouts and lower filter) may push past
0.77 by addressing tasks the easy-tail training never saw kept examples for.
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROLLOUTS = Path("/tmp/iter5-rollouts/rollouts.jsonl")
TRAIN_TASKS = ROOT / "datasets/train.tasks.jsonl"
HARD_OUT = ROOT / "datasets/hard.tasks.jsonl"
THRESHOLD = 0.7

# Load rollouts, group by task_id, find max composite per task
by_task: dict[str, float] = {}
with ROLLOUTS.open() as f:
    for line in f:
        r = json.loads(line)
        tid = r["task_id"]
        comp = r.get("reward", 0.0)
        by_task[tid] = max(by_task.get(tid, 0.0), comp)

# Load full train tasks
tasks: list[dict] = []
with TRAIN_TASKS.open() as f:
    for line in f:
        tasks.append(json.loads(line))

# Find hard tasks (max composite < THRESHOLD or task missing entirely)
hard_task_ids = set()
for t in tasks:
    tid = t["task_id"]
    max_c = by_task.get(tid, 0.0)
    if max_c < THRESHOLD:
        hard_task_ids.add(tid)

# Write hard task set
with HARD_OUT.open("w") as f:
    for t in tasks:
        if t["task_id"] in hard_task_ids:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

# Summary
print(f"total_tasks={len(tasks)}")
print(f"easy_tasks_with_max>=0.7={len(tasks)-len(hard_task_ids)}")
print(f"hard_tasks_with_max<0.7={len(hard_task_ids)}")
print(f"hard task_ids:")
for tid in sorted(hard_task_ids):
    print(f"  {tid}  max_composite={by_task.get(tid, 0.0):.3f}")
print(f"\nwrote -> {HARD_OUT}")
