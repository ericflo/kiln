"""Per-domain breakdown of judgment results — composite/validates/substantive per domain."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", nargs="+", required=True, help="paths to judgment JSONs")
    args = ap.parse_args()

    all_data = {}  # adapter -> dom -> agg
    domains = set()
    for path in args.judgments:
        d = json.load(open(path))
        name = d.get("adapter") or Path(path).stem
        per_dom = {}
        for r in d["per_prompt"]:
            dom = r["id"].rsplit("_", 1)[0]
            domains.add(dom)
            s = per_dom.setdefault(dom, {"n": 0, "composite": 0, "validates": 0, "substant": 0})
            s["n"] += 1
            s["composite"] += r["composite"]
            s["validates"] += r["validates"]
            s["substant"] += r["is_substantive"]
        for dom, s in per_dom.items():
            n = s["n"]
            s["composite"] /= n
            s["validates"] /= n
            s["substant"] /= n
        all_data[name] = per_dom

    domains = sorted(domains)
    print(f"{'domain':<32} " + " ".join(f"{a[:24]:>24}" for a in all_data))
    for dom in domains:
        line = f"{dom:<32} "
        for a in all_data:
            d = all_data[a].get(dom, {})
            if d:
                line += f" {d['composite']:5.3f}/{d['validates']:4.2f}/{d['substant']:4.2f}    "
            else:
                line += f" {'-':>24}"
        print(line)


if __name__ == "__main__":
    main()
