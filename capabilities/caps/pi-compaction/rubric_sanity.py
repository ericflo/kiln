"""Run the rubric on calibration/good.jsonl and calibration/bad.jsonl.

The rubric MUST separate good from bad before any training begins. Targets:
  - good composite mean >= 0.55   (programmatic good — real model can do better)
  - bad composite mean  <= 0.30
  - good_min >= bad_max - 0.05    (no overlap that swallows the signal)

If those fail, fix the rubric — not the calibration. Per `agentic-grpo-capability-creator`
§0: rubric design is the highest-leverage activity in the session.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import rubric


def load_tasks() -> dict[str, dict]:
    """Load only the train tasks; calibration is built from those.
    (Eval tasks share task_id space — loading both would overwrite.)"""
    tasks: dict[str, dict] = {}
    path = ROOT / "datasets/train.tasks.jsonl"
    if path.exists():
        with path.open() as f:
            for line in f:
                t = json.loads(line)
                tasks[t["task_id"]] = t
    return tasks


def score_calibration(path: Path, tasks: dict[str, dict]) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            entry = json.loads(line)
            task = tasks.get(entry["task_id"])
            if task is None:
                continue
            score = rubric.score_rollout(
                entry["response"],
                task["source_text"],
                task["ground_truth"],
            )
            out.append({
                **entry,
                "composite": score["composite"],
                "format.score": score["format.score"],
                "content.score": score["content.score"],
                "faithfulness.score": score["faithfulness.score"],
                "compression.score": score["compression.score"],
                "continuability.score": score["continuability.score"],
                "outcome": score["outcome"],
            })
    return out


def summarize(name: str, rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    composites = [r["composite"] for r in rows]
    sub_keys = [
        "format.score",
        "content.score",
        "faithfulness.score",
        "compression.score",
        "continuability.score",
        "outcome",
    ]
    summary = {
        "n": len(rows),
        "composite_mean": mean(composites),
        "composite_min": min(composites),
        "composite_max": max(composites),
        "composite_stdev": stdev(composites) if len(composites) >= 2 else 0.0,
    }
    for k in sub_keys:
        vals = [r[k] for r in rows]
        summary[f"{k}_mean"] = mean(vals)
    return summary


def main() -> int:
    tasks = load_tasks()
    good = score_calibration(ROOT / "calibration/good.jsonl", tasks)
    bad = score_calibration(ROOT / "calibration/bad.jsonl", tasks)

    print(f"loaded {len(tasks)} tasks")
    print(f"good cases: {len(good)}")
    print(f"bad cases:  {len(bad)}")
    print()

    g = summarize("good", good)
    b = summarize("bad", bad)

    print(f"GOOD: mean={g.get('composite_mean', 0):.4f} min={g.get('composite_min', 0):.4f} max={g.get('composite_max', 0):.4f} stdev={g.get('composite_stdev', 0):.4f}")
    print(f"BAD:  mean={b.get('composite_mean', 0):.4f} min={b.get('composite_min', 0):.4f} max={b.get('composite_max', 0):.4f} stdev={b.get('composite_stdev', 0):.4f}")
    print()

    # Per-bad-shortcut breakdown
    by_shortcut: dict[str, list[float]] = {}
    for r in bad:
        by_shortcut.setdefault(r.get("shortcut", "?"), []).append(r["composite"])
    print("By shortcut:")
    for k in sorted(by_shortcut):
        vals = by_shortcut[k]
        print(f"  {k:<16s} n={len(vals)}  mean={mean(vals):.4f}  max={max(vals):.4f}")
    print()

    # Per-good detail
    print("Per-good detail:")
    for r in good:
        print(
            f"  {r['task_id']}  composite={r['composite']:.4f}  "
            f"format={r['format.score']:.2f}  content={r['content.score']:.2f}  "
            f"faith={r['faithfulness.score']:.2f}  compress={r['compression.score']:.2f}  "
            f"cont={r['continuability.score']:.2f}  outcome={r['outcome']:.0f}"
        )

    # Gates
    gate_good = g.get("composite_mean", 0) >= 0.55
    gate_bad = b.get("composite_mean", 1) <= 0.30
    no_overlap = g.get("composite_min", 0) >= b.get("composite_max", 1) - 0.05
    print()
    print(f"GATE good_mean >= 0.55: {gate_good}  ({g.get('composite_mean', 0):.4f})")
    print(f"GATE bad_mean  <= 0.30: {gate_bad}  ({b.get('composite_mean', 0):.4f})")
    print(f"GATE good_min  >= bad_max - 0.05: {no_overlap}  ({g.get('composite_min', 0):.4f} vs {b.get('composite_max', 0):.4f})")

    all_pass = gate_good and gate_bad and no_overlap
    print()
    print("VERDICT:", "PASS" if all_pass else "FAIL — fix the rubric, not the calibration")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
