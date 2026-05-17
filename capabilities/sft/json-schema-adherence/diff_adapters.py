"""Compare two adapter results — show per-prompt diff and biggest movers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_", required=True, help="baseline judgment JSON")
    ap.add_argument("--to", required=True, help="post-train judgment JSON")
    args = ap.parse_args()

    a = json.load(open(args.from_))
    b = json.load(open(args.to))
    a_rows = {r["id"]: r for r in a["per_prompt"]}
    b_rows = {r["id"]: r for r in b["per_prompt"]}

    print(f"{args.from_} → {args.to}")
    print(f"composite {a['composite']:.4f} → {b['composite']:.4f} (Δ {b['composite']-a['composite']:+.4f})")
    print(f"validates {a['validates']:.3f} → {b['validates']:.3f} (Δ {b['validates']-a['validates']:+.3f})")
    print(f"is_pure   {a['is_pure']:.3f} → {b['is_pure']:.3f} (Δ {b['is_pure']-a['is_pure']:+.3f})")
    print(f"substant. {a['is_substantive']:.3f} → {b['is_substantive']:.3f} (Δ {b['is_substantive']-a['is_substantive']:+.3f})")
    print()

    # Per-prompt deltas
    deltas = []
    for rid, ar in a_rows.items():
        br = b_rows.get(rid)
        if br is None:
            continue
        d = br["composite"] - ar["composite"]
        deltas.append((d, rid, ar, br))
    deltas.sort()

    print("Biggest regressions:")
    for d, rid, ar, br in deltas[:5]:
        print(f"  {rid}: {ar['composite']:.2f} → {br['composite']:.2f} (Δ {d:+.2f})")
    print()
    print("Biggest gains:")
    for d, rid, ar, br in reversed(deltas[-5:]):
        print(f"  {rid}: {ar['composite']:.2f} → {br['composite']:.2f} (Δ {d:+.2f})")


if __name__ == "__main__":
    main()
