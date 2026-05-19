"""Pull the latest iter result down from the pod and append it to capability.jsonl.

After run_iter.sh completes, this script:
  1. Downloads the iter's summary.json (eval) from the pod.
  2. Constructs a result row with iter number, mean composite, sub-scores,
     wall-clock, and notes.
  3. Appends it to capability.jsonl.

Usage:
    python3 record_iter.py --iter N --pod <pod_id> [--kind train]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RP = os.environ.get("RP") or "/data/.clouderic-internal/repos/apps/trajectory-trainer/scripts/runpod_api.py"
CAP_JSONL = ROOT / "capability.jsonl"


def download(pod: str, remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["python3", RP, "download", pod, remote, str(local)],
        capture_output=True, text=True, timeout=180,
    )
    return local.exists() and local.stat().st_size > 0


def safe_load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--pod", required=True)
    ap.add_argument("--kind", default="train")
    args = ap.parse_args()

    tmp = Path(f"/tmp/iter{args.iter}-record")
    tmp.mkdir(parents=True, exist_ok=True)

    eval_summary = tmp / "eval-summary.json"
    train_summary = tmp / "train-summary.json"
    eval_ok = download(args.pod, f"/tmp/iter{args.iter}-eval/summary.json", eval_summary)
    train_ok = download(args.pod, f"/tmp/iter{args.iter}-rollouts/summary.json", train_summary)

    e = safe_load(eval_summary) if eval_ok else None
    t = safe_load(train_summary) if train_ok else None

    row = {
        "iter": args.iter,
        "ts": dt.datetime.utcnow().isoformat(),
        "kind": args.kind,
        "pod_id": args.pod,
    }
    if e:
        row["eval"] = {
            "mean_composite":            e.get("mean_composite"),
            "mean_outcome":              e.get("mean_outcome"),
            "mean_grounding":            e.get("mean_grounding"),
            "mean_cross_file":           e.get("mean_cross_file_caller_recall"),
            "mean_invariant_coverage":   e.get("mean_invariant_coverage"),
            "mean_format_compliance":    e.get("mean_format_compliance"),
            "mean_wall_clock_s":         e.get("mean_wall_clock_s"),
            "n_rollouts":                e.get("n_rollouts"),
            "rollouts_nonzero":          e.get("rollouts_nonzero"),
            "rollouts_zero":             e.get("rollouts_zero"),
        }
    if t:
        row["train"] = {
            "mean_composite":            t.get("mean_composite"),
            "mean_outcome":              t.get("mean_outcome"),
            "mean_wall_clock_s":         t.get("mean_wall_clock_s"),
            "mean_within_group_variance": t.get("mean_within_group_variance"),
            "n_rollouts":                t.get("n_rollouts"),
        }

    with CAP_JSONL.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
