"""Append one row to capability.jsonl summarizing this iter.

Pulls /tmp/pft-iter<N>-eval/summary.json and /tmp/pft-iter<N>-rollouts/summary.json
from the pod, and stages a row.
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


def _fetch_json(pod_id: str, remote_path: str) -> dict | None:
    rp = os.environ.get("RP") or "/data/.clouderic-internal/repos/apps/trajectory-trainer/scripts/runpod_api.py"
    local = Path("/tmp") / f"appendlog-{os.getpid()}-{Path(remote_path).name}"
    res = subprocess.run(
        ["python3", rp, "download", pod_id, remote_path, str(local)],
        capture_output=True, text=True, timeout=120,
    )
    if not local.exists() or local.stat().st_size == 0:
        return None
    try:
        return json.loads(local.read_text())
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--pod", required=True)
    ap.add_argument("--recipe", required=True)
    args = ap.parse_args()

    eval_sum = _fetch_json(args.pod, f"/tmp/pft-iter{args.iter}-eval/summary.json") or {}
    train_sum = _fetch_json(args.pod, f"/tmp/pft-iter{args.iter}-rollouts/summary.json") or {}

    row = {
        "iter": args.iter,
        "ts": dt.datetime.utcnow().isoformat(),
        "recipe": args.recipe,
        "eval_mean_composite": eval_sum.get("mean_composite"),
        "eval_mean_outcome": eval_sum.get("mean_outcome"),
        "eval_mean_held_out": eval_sum.get("mean_held_out_passes"),
        "eval_mean_no_blanket": eval_sum.get("mean_no_blanket_except"),
        "eval_mean_repro": eval_sum.get("mean_reproduced_before_fixing"),
        "eval_mean_fix_local": eval_sum.get("mean_fix_localised_correctly"),
        "eval_mean_no_test_mut": eval_sum.get("mean_no_test_mutation"),
        "eval_mean_format": eval_sum.get("mean_format_compliance"),
        "eval_mean_diff_min": eval_sum.get("mean_diff_minimality"),
        "eval_mean_no_dep": eval_sum.get("mean_no_dependency_changes"),
        "eval_p05_composite": eval_sum.get("p05_composite"),
        "eval_p95_composite": eval_sum.get("p95_composite"),
        "eval_mean_wall_clock_s": eval_sum.get("mean_wall_clock_s"),
        "eval_n_rollouts": eval_sum.get("n_rollouts"),
        "train_mean_composite": train_sum.get("mean_composite"),
        "train_mean_group_var": train_sum.get("mean_within_group_variance"),
        "train_n_rollouts": train_sum.get("n_rollouts"),
    }
    out_path = ROOT / "capability.jsonl"
    with out_path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
