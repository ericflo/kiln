#!/usr/bin/env python3
"""H15c — derive kiln MTP α per seed from PR #529's c1_attr CSVs.

Reads docs/archive/phase-c/phase-c29-v2/artifacts/c1_attr_seed{0,1,2}.csv and computes

    alpha_seed = sum(accepted) / n_rows

The `accepted` column is 0/1 per MTP step (k=1 speculative draft attempt).
We compute α per seed independently, then report the median — matching the
"compute per-seed first then median" rule from the task brief.

Writes `kiln_alpha_per_seed.json` for the compare step:
    {
      "per_seed": [{"seed": 0, "alpha": ..., "n_steps": ..., "n_accept": ...}, ...],
      "median_alpha": ...,
      "source_csvs": [...]
    }
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def compute_alpha(csv_path: Path) -> tuple[float, int, int]:
    n_steps = 0
    n_accept = 0
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_steps += 1
            n_accept += int(row["accepted"])
    if n_steps == 0:
        return 0.0, 0, 0
    return n_accept / n_steps, n_steps, n_accept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv-dir",
        default="docs/archive/phase-c/phase-c29-v2/artifacts",
        help="Directory containing c1_attr_seed{0,1,2}.csv",
    )
    ap.add_argument(
        "--out",
        default="docs/archive/phase-c/phase-c29-v3-vllm/kiln_alpha_per_seed.json",
        help="Output JSON path",
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir)
    per_seed = []
    alphas: list[float] = []
    sources = []
    for seed in args.seeds:
        csv_path = csv_dir / f"c1_attr_seed{seed}.csv"
        if not csv_path.exists():
            raise SystemExit(f"missing {csv_path}")
        alpha, n_steps, n_accept = compute_alpha(csv_path)
        per_seed.append(
            {
                "seed": seed,
                "alpha": alpha,
                "n_steps": n_steps,
                "n_accept": n_accept,
            }
        )
        alphas.append(alpha)
        sources.append(str(csv_path))

    result = {
        "per_seed": per_seed,
        "median_alpha": statistics.median(alphas),
        "mean_alpha": statistics.mean(alphas),
        "min_alpha": min(alphas),
        "max_alpha": max(alphas),
        "n_seeds": len(alphas),
        "source_csvs": sources,
        "methodology": (
            "alpha_seed = sum(accepted column) / n_rows in "
            "c1_attr_seed{N}.csv; median computed across seeds. "
            "Source run: PR #529 (scripts/c29_kiln_logits_dump.sh), "
            "KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 KILN_W4A16=1, "
            "greedy (temperature=0), --paged, seeds=[0,1,2] indexing "
            "PROMPT_POOL[seed%30] (= GSM8K prose prompts 0,1,2), "
            "--prompt-tokens 512, --max-output-tokens 16, no chat template, "
            "splice dumps captured at POSITIONS=0..3 × MAX_STEPS=2. "
            "NOTE: the task brief's 7/22 ≈ 0.318 hint refers to the "
            "subset of {accept-labeled, reject-labeled} splice rows "
            "used by the H15b C29 v2 top-K probe. This script uses all "
            "11/12/11 rows per seed from the full c1_attr CSV (every "
            "speculative_mtp_decode_step call, not only dumped rows)."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
