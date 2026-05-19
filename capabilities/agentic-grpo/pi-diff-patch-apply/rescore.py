"""Re-score a completed iter's rollouts against the current rubric.

Walks <out_dir>/task_NNNN/genXX/workdir/ + sessions/*.jsonl, runs
rubric.score_rollout on each, and overwrites <out_dir>/rollouts.jsonl
and <out_dir>/summary.json with fresh scores.

Use this when the rubric changes mid-iter (e.g. you tightened a sub-score
or added an exclusion). Cheaper than re-running pi.

Usage:
    python3 rescore.py --out-dir /tmp/iter0-eval --tasks datasets/eval.tasks.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import rubric  # noqa: E402


def latest_session_jsonl(session_dir: Path) -> Path | None:
    candidates = list(session_dir.rglob("*.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_transcript(path: Path | None) -> list[dict]:
    if not path or not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tasks", required=True)
    args = ap.parse_args()
    out_root = Path(args.out_dir)

    tasks = {}
    with open(args.tasks) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            tasks[t["task_id"]] = t

    records = []
    for task_dir in sorted(out_root.glob("task_*")):
        tid = task_dir.name
        task = tasks.get(tid)
        if not task:
            continue
        for gen_dir in sorted(task_dir.glob("gen*")):
            gen_idx = int(gen_dir.name[3:])
            workdir = gen_dir / "workdir"
            session_dir = gen_dir / "sessions"
            transcript_path = latest_session_jsonl(session_dir) if session_dir.exists() else None
            transcript = parse_transcript(transcript_path)
            score = rubric.score_rollout(transcript, str(workdir), task)
            records.append({
                "task_id": tid,
                "patch_class": task.get("patch_class"),
                "gen": gen_idx,
                "reward": score["composite"],
                "sub_scores": {k: v for k, v in score.items() if not k.startswith("_")},
                "diagnostics": {k: v for k, v in score.items() if k.startswith("_")},
                "wall_clock_s": 0.0,  # not tracked here
                "exit_code": 0,
                "transcript_path": str(transcript_path) if transcript_path else None,
                "workdir": str(workdir),
                "stdout_tail": "",
                "stderr_tail": "",
            })

    records.sort(key=lambda r: (r["task_id"], r["gen"]))
    (out_root / "rollouts.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )

    composites = [r["reward"] for r in records]
    by_class = {}
    for r in records:
        by_class.setdefault(r.get("patch_class") or "?", []).append(r["reward"])
    class_means = {k: sum(v) / len(v) for k, v in by_class.items()}

    sub_keys = set()
    for r in records:
        for k in r.get("sub_scores") or {}:
            sub_keys.add(k)
    sub_means = {}
    for k in sub_keys:
        vals = [r["sub_scores"].get(k, 0.0) for r in records]
        sub_means[k] = sum(vals) / len(vals)

    summary = {
        "mode": "eval-rescore",
        "n_rollouts": len(records),
        "mean_composite": sum(composites) / max(1, len(composites)),
        "p50_composite": sorted(composites)[len(composites) // 2] if composites else 0.0,
        "rollouts_passed": sum(1 for r in records if r["sub_scores"].get("outcome", 0) >= 1.0),
        "class_means": class_means,
        "sub_score_means": sub_means,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
