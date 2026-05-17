#!/usr/bin/env python3
"""headroom.py — rubric headroom analysis for opd-capability-creator.

Reads `capability.jsonl` from stdin or the path argument, finds the most
recent baseline entry (`slug=="baseline"`), and prints a table of:

  sub_score | weight | baseline | headroom = w × (1 − b)

plus the theoretical composite ceiling (sum of headroom).

This is the rubric design step. After every baseline, run:

  python3 $SKILL/templates/headroom.py < capability.jsonl

The agent picks the target sub-score from the row with the most headroom
and records it in capability.md.

Expects each baseline entry to include:
  - status: "kept"
  - slug:   "baseline"
  - composite: <float>
  - sub_scores: {<name>: <float>, ...}

And `capability.config.json` (in current dir) to specify rubric weights:
  rubric.sub_scores = [{"name": ..., "weight": ...}, ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def find_latest_baseline(rows: list[dict]) -> dict | None:
    """Return the most recent baseline entry, or None."""
    for row in reversed(rows):
        if row.get("slug") == "baseline":
            return row
    return None


def load_rubric_weights() -> dict[str, float]:
    """Read rubric weights from capability.config.json."""
    cfg_path = Path("capability.config.json")
    if not cfg_path.exists():
        print("ERROR: capability.config.json not found in cwd", file=sys.stderr)
        sys.exit(2)
    cfg = json.loads(cfg_path.read_text())
    sub = cfg.get("rubric", {}).get("sub_scores", [])
    if not sub:
        print("ERROR: capability.config.json rubric.sub_scores is empty.", file=sys.stderr)
        print("       Add entries like [{\"name\":\"parses\",\"weight\":0.4}, ...]", file=sys.stderr)
        sys.exit(2)
    return {s["name"]: float(s["weight"]) for s in sub}


def main() -> None:
    # Read jsonl from stdin or file arg
    if len(sys.argv) > 1 and sys.argv[1] not in ("-", "/dev/stdin"):
        text = Path(sys.argv[1]).read_text()
    else:
        text = sys.stdin.read()
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows:
        print("ERROR: no log entries found in capability.jsonl", file=sys.stderr)
        sys.exit(2)

    baseline = find_latest_baseline(rows)
    if baseline is None:
        print("ERROR: no entry with slug=\"baseline\" in capability.jsonl", file=sys.stderr)
        sys.exit(2)

    weights = load_rubric_weights()
    sub_scores = baseline.get("sub_scores", {})
    missing = set(weights) - set(sub_scores)
    if missing:
        print(f"WARNING: baseline missing sub-scores: {sorted(missing)}", file=sys.stderr)

    # Compute per-sub-score headroom
    rows_out: list[tuple[str, float, float, float]] = []
    for name, weight in weights.items():
        baseline_score = float(sub_scores.get(name, 0.0))
        headroom = weight * (1.0 - baseline_score)
        rows_out.append((name, weight, baseline_score, headroom))
    rows_out.sort(key=lambda r: -r[3])  # sort by headroom desc

    total_headroom = sum(r[3] for r in rows_out)
    baseline_composite = sum(w * sub_scores.get(n, 0.0) for n, w in weights.items())

    # Print a markdown table the agent can paste into capability.md
    print(f"\n## Headroom analysis (from baseline @ {baseline.get('ts', '?')})\n")
    print(f"Baseline composite: **{baseline_composite:.4f}**")
    print(f"Theoretical ceiling (composite): **{baseline_composite + total_headroom:.4f}**")
    print(f"Total movable headroom: **{total_headroom:.4f}**\n")
    print("| Sub-score | Weight | Baseline | Headroom (w×(1−b)) | Share |")
    print("|-----------|--------|----------|---------------------|-------|")
    for name, weight, baseline_score, headroom in rows_out:
        share = headroom / total_headroom if total_headroom > 0 else 0.0
        marker = "← target" if name == rows_out[0][0] and rows_out[0][3] > 1e-6 else ""
        print(f"| {name} | {weight:.2f} | {baseline_score:.4f} | {headroom:.4f} | {share:.0%} {marker} |")
    print()

    # Stop-and-report if rubric is saturated
    if total_headroom < 0.05:
        print(
            "ALERT: total headroom < 0.05. The rubric is essentially saturated;\n"
            "OPD will not produce a meaningful composite lift here. Flag to user.",
            file=sys.stderr,
        )
        sys.exit(3)


if __name__ == "__main__":
    main()
