"""Summarise capability.jsonl into a sorted table for quick eyeballing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

rows = []
LOG = Path(__file__).parent / "capability.jsonl"
for line in LOG.read_text().splitlines():
    if line.strip():
        rows.append(json.loads(line))

if not rows:
    print("no rows")
    sys.exit(0)

baseline = next((r for r in rows if r["name"] == "baseline"), None)
b_comp = baseline["composite"] if baseline else None


def fmt(v, n=4):
    if v is None:
        return " " * n
    return f"{v:.{n}f}"


print(f"{'name':<32} {'kind':<12} {'composite':>9} {'Δ':>7} {'parses':>7} {'val':>5} {'pure':>5} {'subs':>5} {'train':>6} {'eval':>5}")
print("-" * 110)
sorted_rows = sorted(
    rows,
    key=lambda r: r["composite"],
    reverse=True,
)
for r in sorted_rows:
    delta = (r["composite"] - b_comp) if b_comp is not None else None
    delta_s = f"{'+' if delta is not None and delta >= 0 else ''}{delta:.4f}" if delta is not None else ""
    print(
        f"{r['name']:<32} "
        f"{r.get('kind','opd'):<12} "
        f"{r['composite']:>9.4f} "
        f"{delta_s:>7} "
        f"{r['parses']:>7.3f} "
        f"{r['validates']:>5.3f} "
        f"{r['is_pure']:>5.3f} "
        f"{r['is_substantive']:>5.3f} "
        f"{r['train_secs']:>5}s "
        f"{r['eval_secs']:>4}s"
    )
