"""Analyze capability.jsonl across iters and emit a closeout summary.

Reports:
  - Best iter (by eval composite)
  - Top-5 iters
  - Family-level mean composite
  - Sub-score deltas (best vs baseline)
  - Null/positive/negative counts

Usage:
  python3 analyze.py [--baseline 0.7237] [--out closeout-data.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "capability.jsonl"


def load_rows() -> list[dict]:
    if not LOG.exists():
        return []
    out = []
    for line in LOG.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return sorted(out, key=lambda r: r.get("iter", -1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_rows()
    if not rows:
        print("No rows in capability.jsonl")
        return 1

    baseline_row = next((r for r in rows if r.get("iter") == 0), None)
    if baseline_row and args.baseline is None:
        args.baseline = baseline_row.get("composite", 0.0)

    iter_rows = [r for r in rows if r.get("iter", 0) > 0]
    iter_rows.sort(key=lambda r: r.get("composite", 0.0), reverse=True)

    print(f"baseline composite: {args.baseline:.4f}")
    print(f"# iters logged: {len(iter_rows)}")
    print()

    print("=== TOP-5 iters ===")
    for i, r in enumerate(iter_rows[:5], 1):
        comp = r.get("composite", 0.0)
        delta = comp - args.baseline if args.baseline else 0.0
        print(f"  {i}. iter={r['iter']:2d}  slug={r['slug']:35s}  composite={comp:.4f}  Δ={delta:+.4f}  family={r.get('family','')}")
    print()

    print("=== Family-level mean composite ===")
    families = defaultdict(list)
    for r in iter_rows:
        families[r.get("family", "?")].append(r.get("composite", 0.0))
    for fam, comps in sorted(families.items(), key=lambda kv: -sum(kv[1])/max(1,len(kv[1]))):
        mean = sum(comps) / len(comps)
        delta = mean - args.baseline if args.baseline else 0.0
        print(f"  {fam:30s}  n={len(comps):2d}  mean={mean:.4f}  Δ={delta:+.4f}")
    print()

    print("=== Status counts ===")
    statuses = defaultdict(int)
    for r in iter_rows:
        statuses[r.get("status", "?")] += 1
    for s, n in sorted(statuses.items(), key=lambda kv: -kv[1]):
        print(f"  {s:20s}  {n}")
    print()

    if iter_rows:
        best = iter_rows[0]
        best_eval = best.get("eval_summary", {})
        best_sub = best_eval.get("subscore_means", {})
        base_sub = (baseline_row or {}).get("eval_summary", {}).get("subscore_means", {})
        if best_sub and base_sub:
            print("=== Best iter sub-score deltas ===")
            keys = ["outcome.score", "outcome.value_correct", "honesty.score",
                    "no_question.score", "no_soft_punt.score", "format_strict.score",
                    "terseness.score", "composite"]
            for k in keys:
                b = base_sub.get(k, 0.0)
                v = best_sub.get(k, 0.0)
                delta = v - b
                print(f"  {k:30s}  base={b:.4f}  best={v:.4f}  Δ={delta:+.4f}")

    if args.out:
        out = {
            "baseline": args.baseline,
            "top5": [{"iter": r["iter"], "slug": r["slug"], "composite": r["composite"]} for r in iter_rows[:5]],
            "families": {f: {"n": len(cs), "mean": sum(cs)/len(cs)} for f, cs in families.items()},
            "status_counts": dict(statuses),
            "n_iters_logged": len(iter_rows),
        }
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
