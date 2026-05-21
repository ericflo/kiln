"""Per-prompt composite matrix across multiple adapter judgments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", nargs="+", required=True)
    args = ap.parse_args()

    judgments = {}
    all_ids = set()
    for path in args.judgments:
        d = json.load(open(path))
        name = (d.get("adapter") or Path(path).stem).replace("-r16-lr1e4-2ep","").replace("-r32-lr1e4-2ep","-r32")
        rows = {r["id"]: r for r in d["per_prompt"]}
        judgments[name] = rows
        all_ids.update(rows.keys())

    names = list(judgments)
    print(f"{'prompt_id':<40}" + " ".join(f"{n[:14]:>14}" for n in names))
    for rid in sorted(all_ids):
        parts = []
        for n in names:
            r = judgments[n].get(rid)
            if r is None:
                parts.append(f"{'-':>14}")
            else:
                parts.append(f"{r['composite']:>14.3f}")
        print(f"{rid:<40}" + " ".join(parts))
    print()
    # Aggregates
    print(f"{'AGGREGATE':<40}" + " ".join(
        f"{sum(judgments[n][r]['composite'] for r in all_ids if r in judgments[n])/len(judgments[n]):>14.4f}"
        for n in names
    ))


if __name__ == "__main__":
    main()
