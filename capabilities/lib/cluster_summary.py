"""
cluster_summary.py — aggregate shipped pipelines into a distillation cluster manifest.

This is the Phase G prerequisite (see DISTILLATION.md). Given:

  - capabilities/caps/<cap>/pipeline.md headers across the whole tree
  - a sibling matrix from integration/cross-cap-coherence/ output

…produces capabilities/rounds/round-<N>/cluster_manifest.json describing
the greedy-compatible cluster chosen for distillation, the excluded caps
and reasons, and a placeholder for validation results.

Usage:
  python3 lib/cluster_summary.py \\
    --caps-root capabilities/caps \\
    --sibling-matrix rounds/round-3/sibling_matrix.json \\
    --round 3 \\
    [--threshold-soft -0.02] [--threshold-hard -0.05] \\
    [--output rounds/round-3/cluster_manifest.json]

The clustering algorithm is METHODS.md §2.3 greedy-by-composite-delta:
  1. Sort shipped pipelines by composite_delta descending.
  2. Iterate; add each pipeline to the cluster if it is compatible with
     every already-added member (sibling_delta > threshold-soft in both
     directions).
  3. Stop when no pipeline can be added.

Soft-incompatibility (sibling_delta ∈ [threshold-hard, threshold-soft])
is tolerated only if explicitly listed in --tolerate <cap-A,cap-B>.

Hard-incompatibility (sibling_delta < threshold-hard) is never tolerated.

This module does NOT do the actual distillation — that's a separate Phase G
trainer call.
"""

import argparse
import json
import sys
from pathlib import Path

# Reuse stage_manifest's pipeline.md header parser.
from stage_manifest import parse_pipeline_header  # type: ignore


def gather_shipped_pipelines(caps_root: Path) -> list[dict]:
    """Walk caps_root and collect pipeline.md headers for shipped caps.

    Returns a list of dicts:
      [{"cap": <name>, "header": <dict>, "path": <Path>}, ...]
    sorted by composite_delta descending (header.final_composite - header.baseline_composite).
    """
    out = []
    for child in sorted(caps_root.iterdir()):
        if not child.is_dir():
            continue
        pipeline_md = child / "pipeline.md"
        if not pipeline_md.exists():
            continue
        try:
            header = parse_pipeline_header(pipeline_md)
        except Exception as e:
            print(
                f"warning: failed to parse {pipeline_md}: {e}",
                file=sys.stderr,
            )
            continue
        if header.get("status") != "shipped":
            continue
        out.append({"cap": child.name, "header": header, "path": pipeline_md})

    def composite_delta(item):
        h = item["header"]
        base = h.get("baseline_composite") or 0.0
        final = h.get("final_composite") or 0.0
        return final - base

    out.sort(key=composite_delta, reverse=True)
    return out


def select_cluster(
    shipped: list[dict],
    sibling_matrix: dict,
    threshold_soft: float = -0.02,
    threshold_hard: float = -0.05,
    tolerate: set[tuple[str, str]] | None = None,
) -> dict:
    """Greedy cluster selection.

    sibling_matrix shape:
      {<cap_A>: {<cap_B>: float}}  meaning "applying cap_A's final adapter
      against cap_B's eval moved cap_B's composite by this delta."

    Returns:
      {
        "cluster_members": [<cap>, ...],
        "cluster_excluded": [{"cap": ..., "reason": ...}, ...],
        "soft_incompatible_pairs": [...],
        "hard_incompatible_pairs": [...],
        "metadata": {...},
      }
    """
    tolerate = tolerate or set()
    members = []
    excluded = []
    soft_pairs = []
    hard_pairs = []

    for item in shipped:
        cap = item["cap"]
        ok = True
        reason = None
        for m in members:
            ab = sibling_matrix.get(cap, {}).get(m)
            ba = sibling_matrix.get(m, {}).get(cap)
            for direction_key, delta in (("→", ab), ("←", ba)):
                if delta is None:
                    excluded.append({"cap": cap, "reason": f"no sibling delta with {m} ({direction_key})"})
                    ok = False
                    break
                if delta < threshold_hard:
                    hard_pairs.append({"a": cap, "b": m, "delta": delta, "direction": direction_key})
                    excluded.append(
                        {
                            "cap": cap,
                            "reason": f"hard-incompatible with {m} {direction_key} (delta={delta:.4f} < {threshold_hard})",
                        }
                    )
                    ok = False
                    break
                if delta < threshold_soft:
                    pair_key = tuple(sorted([cap, m]))
                    if pair_key in tolerate:
                        soft_pairs.append({"a": cap, "b": m, "delta": delta, "tolerated": True, "direction": direction_key})
                    else:
                        soft_pairs.append({"a": cap, "b": m, "delta": delta, "tolerated": False, "direction": direction_key})
                        excluded.append(
                            {
                                "cap": cap,
                                "reason": f"soft-incompatible with {m} {direction_key} (delta={delta:.4f}); not tolerated",
                            }
                        )
                        ok = False
                        break
            if not ok:
                break
        if ok:
            members.append(cap)

    return {
        "cluster_members": members,
        "cluster_excluded": excluded,
        "soft_incompatible_pairs": soft_pairs,
        "hard_incompatible_pairs": hard_pairs,
        "metadata": {
            "threshold_soft": threshold_soft,
            "threshold_hard": threshold_hard,
            "n_shipped": len(shipped),
            "n_cluster": len(members),
        },
    }


def build_manifest(
    *, round_n: int, shipped: list[dict], cluster: dict
) -> dict:
    member_set = set(cluster["cluster_members"])
    member_records = []
    for item in shipped:
        cap = item["cap"]
        if cap not in member_set:
            continue
        h = item["header"]
        member_records.append(
            {
                "cap": cap,
                "baseline_composite": h.get("baseline_composite"),
                "final_composite": h.get("final_composite"),
                "delta": (h.get("final_composite") or 0.0)
                - (h.get("baseline_composite") or 0.0),
                "final_adapter": h.get("final_adapter"),
                "stages": h.get("stages"),
            }
        )

    return {
        "round": round_n,
        "cluster_members": member_records,
        "cluster_excluded": cluster["cluster_excluded"],
        "soft_incompatible_pairs": cluster["soft_incompatible_pairs"],
        "hard_incompatible_pairs": cluster["hard_incompatible_pairs"],
        "selection_metadata": cluster["metadata"],
        "distillation_method": None,
        "distill_recipe": None,
        "validation": None,
        "promoted_to_base": False,
        "new_base_sha256": None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--caps-root", type=Path, default=Path("capabilities/caps"))
    ap.add_argument("--sibling-matrix", type=Path, required=True)
    ap.add_argument("--round", dest="round_n", type=int, required=True)
    ap.add_argument("--threshold-soft", type=float, default=-0.02)
    ap.add_argument("--threshold-hard", type=float, default=-0.05)
    ap.add_argument(
        "--tolerate",
        type=str,
        default="",
        help="Comma-separated pairs of soft-incompatible caps to tolerate "
        "(e.g. 'cap-A:cap-B,cap-C:cap-D')",
    )
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    sibling_matrix = json.loads(args.sibling_matrix.read_text())
    tolerate = set()
    if args.tolerate:
        for pair in args.tolerate.split(","):
            a, b = pair.split(":")
            tolerate.add(tuple(sorted([a.strip(), b.strip()])))

    shipped = gather_shipped_pipelines(args.caps_root)
    cluster = select_cluster(
        shipped, sibling_matrix, args.threshold_soft, args.threshold_hard, tolerate
    )
    manifest = build_manifest(round_n=args.round_n, shipped=shipped, cluster=cluster)

    payload = json.dumps(manifest, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"wrote {args.output}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
