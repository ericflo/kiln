"""Append one row to capability.jsonl from an iter's eval + train summaries.

Usage:
  python3 log_iter.py --iter N --slug <slug> --family <h-family> \
                      [--train-summary <path>] --eval-summary <path> \
                      [--adapter <name>] --verdict "<short>" \
                      [--hyperparams '<json>'] [--baseline N]

Writes a single JSON line. Re-runnable: if a row for the same (iter, slug)
already exists, it is replaced.
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


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=8", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--family", default="")
    ap.add_argument("--train-summary", default=None)
    ap.add_argument("--eval-summary", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--verdict", default="")
    ap.add_argument("--hyperparams", default="{}")
    ap.add_argument("--baseline", type=float, default=None)
    ap.add_argument("--status", default="recorded")
    ap.add_argument("--what-worked", default="")
    ap.add_argument("--what-failed", default="")
    ap.add_argument("--next-focus", default="")
    args = ap.parse_args()

    eval_sum = json.loads(Path(args.eval_summary).read_text()) if args.eval_summary else {}
    train_sum = json.loads(Path(args.train_summary).read_text()) if args.train_summary else None

    composite = eval_sum.get("mean_composite", 0.0)
    delta = None
    if args.baseline is not None:
        delta = composite - args.baseline

    row = {
        "iter": args.iter,
        "slug": args.slug,
        "ts": dt.datetime.utcnow().isoformat() + "Z",
        "status": args.status,
        "family": args.family,
        "rubric_version": "v1.0",
        "verdict": args.verdict,
        "composite": composite,
        "delta_vs_baseline": delta,
        "adapter": args.adapter,
        "hyperparams": json.loads(args.hyperparams) if args.hyperparams else {},
        "eval_summary": eval_sum,
        "train_summary": train_sum,
        "asi": {
            "what_worked": args.what_worked,
            "what_failed": args.what_failed,
            "next_focus": args.next_focus,
        },
        "git_sha": git_sha(),
    }

    log_path = ROOT / "capability.jsonl"
    # Read existing rows; drop any with same (iter, slug) so we can re-run.
    rows: list[dict] = []
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            try:
                r = json.loads(line)
                if r.get("iter") == args.iter and r.get("slug") == args.slug:
                    continue
                rows.append(r)
            except json.JSONDecodeError:
                continue
    rows.append(row)
    rows.sort(key=lambda r: (r.get("iter", 0), r.get("ts", "")))
    with log_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"logged iter {args.iter} slug={args.slug} composite={composite:.4f} "
          f"delta={delta if delta is None else f'{delta:+.4f}'} adapter={args.adapter}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
