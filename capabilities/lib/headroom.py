"""
headroom.py — per-sub-score headroom analysis.

A composite is a weighted sum of sub-scores. Each sub-score s_i with weight
w_i and current value v_i contributes at most `w_i × (1 - v_i)` to a future
composite uplift. Headroom = Σ w_i × (1 - v_i). Most of it usually lives in
one or two sub-scores; the rest are saturated.

Use this after the baseline eval and before every stage to know where
training should target.

Usage:
  python3 lib/headroom.py --eval-summary /tmp/<cap>-eval.json \\
    [--weights '{"outcome": 0.5, "format": 0.2, "process": 0.3}'] \\
    [--print]

If --weights is omitted, treats all sub-scores as equally weighted.
The rubric's actual weights live in rubric.py; pass them explicitly here
so the analysis matches the trained composite.
"""

import argparse
import json
import sys
from pathlib import Path


def analyze(sub_scores: dict, weights: dict | None = None) -> dict:
    """Compute per-sub-score headroom and overall headroom.

    Returns:
      {
        "per_sub_score": {<name>: {"value": v, "weight": w, "headroom": h, "share": h/total}},
        "total_headroom": ...,
        "ranked": [(name, headroom), ...],  # sorted descending
        "dominant": <name>,        # the largest-headroom sub-score
        "dominant_share": ...,
        "saturated": [<names>],    # sub-scores with headroom < 0.01
      }
    """
    if weights is None:
        n = len(sub_scores) or 1
        weights = {k: 1.0 / n for k in sub_scores}

    per = {}
    for name, value in sub_scores.items():
        w = weights.get(name, 0.0)
        h = w * (1.0 - value)
        per[name] = {"value": value, "weight": w, "headroom": h}

    total = sum(d["headroom"] for d in per.values())
    for d in per.values():
        d["share"] = (d["headroom"] / total) if total > 0 else 0.0

    ranked = sorted(per.items(), key=lambda kv: kv[1]["headroom"], reverse=True)
    dominant = ranked[0][0] if ranked else None
    dominant_share = ranked[0][1]["share"] if ranked else 0.0
    saturated = [k for k, d in per.items() if d["headroom"] < 0.01]

    return {
        "per_sub_score": per,
        "total_headroom": total,
        "ranked": [(name, d["headroom"]) for name, d in ranked],
        "dominant": dominant,
        "dominant_share": dominant_share,
        "saturated": saturated,
    }


def _format_human(result: dict) -> str:
    lines = []
    lines.append(f"Total headroom: {result['total_headroom']:.4f}")
    if result["dominant"]:
        lines.append(
            f"Dominant sub-score: {result['dominant']} "
            f"({result['dominant_share']:.0%} of headroom)"
        )
    lines.append("")
    lines.append("Per-sub-score (sorted by headroom desc):")
    lines.append(
        f"  {'sub_score':<24} {'value':>7} {'weight':>7} "
        f"{'headroom':>9} {'share':>7}"
    )
    for name, _ in result["ranked"]:
        d = result["per_sub_score"][name]
        lines.append(
            f"  {name:<24} {d['value']:>7.4f} {d['weight']:>7.3f} "
            f"{d['headroom']:>9.4f} {d['share']:>7.1%}"
        )
    if result["saturated"]:
        lines.append("")
        lines.append(f"Saturated (headroom < 0.01): {', '.join(result['saturated'])}")
    if result["total_headroom"] < 0.05:
        lines.append("")
        lines.append(
            "WARNING: total headroom < 0.05. Cap is near-saturated. Consider "
            "tightening the rubric, building a hard_eval pool, or shipping "
            "as-is. See METHODS.md Rule B + Rule G."
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--eval-summary", required=True, type=Path)
    ap.add_argument(
        "--weights",
        type=str,
        default=None,
        help='JSON dict of sub-score weights e.g. \'{"outcome": 0.5, "format": 0.5}\'',
    )
    ap.add_argument("--print", action="store_true", help="Human-readable output")
    args = ap.parse_args()

    eval_summary = json.loads(args.eval_summary.read_text())
    sub_scores = (
        eval_summary.get("sub_scores_mean") or eval_summary.get("sub_scores") or {}
    )
    weights = json.loads(args.weights) if args.weights else None

    result = analyze(sub_scores, weights)
    if args.print:
        print(_format_human(result))
    else:
        # ranked is a list of tuples — convert to list[dict] for JSON
        out = dict(result)
        out["ranked"] = [{"name": n, "headroom": h} for n, h in result["ranked"]]
        print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
