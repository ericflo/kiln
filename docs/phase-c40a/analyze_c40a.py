#!/usr/bin/env python3
"""Phase C40a analysis — HumanEval + chat-template OFF.

Reads /workspace/c40a-results/seed-{0..19}.json, computes median α,
95% bootstrap CI (rng=12345, 10_000 resamples), writes per-seed table
and summary to stdout. Same methodology as C39 (analyze_c39.py).
"""

import json
import random
import statistics
import sys
from pathlib import Path

RESULTS_DIR = Path("/workspace/c40a-results")
N = 20
BOOTSTRAP_RESAMPLES = 10_000
RNG_SEED = 12345


def main() -> int:
    rows = []
    for seed in range(N):
        f = RESULTS_DIR / f"seed-{seed}.json"
        if not f.exists():
            print(f"MISSING: {f}", file=sys.stderr)
            continue
        data = json.loads(f.read_text())
        lat = data.get("latency") or {}
        alpha = lat.get("acceptance_rate")
        tps = lat.get("decode_tokens_per_sec")
        itl = lat.get("mean_inter_token_ms")
        if alpha is None:
            print(f"NO ALPHA: seed={seed}", file=sys.stderr)
            continue
        rows.append(
            {
                "seed": seed,
                "alpha": alpha,
                "decode_tps": tps,
                "mean_itl_ms": itl,
            }
        )

    alphas = [r["alpha"] for r in rows]
    n = len(alphas)

    # Bootstrap CI
    rng = random.Random(RNG_SEED)
    resample_medians = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = [rng.choice(alphas) for _ in range(n)]
        resample_medians.append(statistics.median(sample))
    resample_medians.sort()
    lo = resample_medians[int(0.025 * BOOTSTRAP_RESAMPLES)]
    hi = resample_medians[int(0.975 * BOOTSTRAP_RESAMPLES)]

    median = statistics.median(alphas)
    mean = statistics.fmean(alphas)
    stdev = statistics.stdev(alphas) if n > 1 else 0.0
    min_alpha = min(alphas)
    max_alpha = max(alphas)
    min_seed = rows[alphas.index(min_alpha)]["seed"]
    max_seed = rows[alphas.index(max_alpha)]["seed"]

    itls = [r["mean_itl_ms"] for r in rows if r["mean_itl_ms"] is not None]
    tps = [r["decode_tps"] for r in rows if r["decode_tps"] is not None]
    median_itl = statistics.median(itls) if itls else None
    median_tps = statistics.median(tps) if tps else None

    # Per-seed table
    print("| seed | α      | decode_tps | mean_itl_ms |")
    print("| ---- | ------ | ---------- | ----------- |")
    for r in rows:
        a = f"{r['alpha']:.4f}"
        t = f"{r['decode_tps']:.2f}" if r["decode_tps"] is not None else "n/a"
        i = f"{r['mean_itl_ms']:.3f}" if r["mean_itl_ms"] is not None else "n/a"
        print(f"| {r['seed']:>4} | {a} | {t:>10} | {i:>11} |")

    print()
    print("## Summary statistics")
    print(f"N = {n}")
    print(f"median α = {median:.4f}")
    print(f"mean α = {mean:.4f}")
    print(f"stdev α = {stdev:.4f}")
    print(f"min α = {min_alpha:.4f} (seed {min_seed})")
    print(f"max α = {max_alpha:.4f} (seed {max_seed})")
    print(f"95% CI (bootstrap {BOOTSTRAP_RESAMPLES} resamples, rng={RNG_SEED}): [{lo:.4f}, {hi:.4f}]")
    print(f"median ITL ≈ {median_itl:.3f} ms" if median_itl else "median ITL: n/a")
    print(f"median decode tps ≈ {median_tps:.2f} tok/s" if median_tps else "median tps: n/a")
    print()
    print("## Floor check vs 0.72")
    print(f"CI lo = {lo:.4f}  {'PASS' if lo >= 0.72 else 'fail'}  vs 0.72")
    print(f"median = {median:.4f}  {'PASS' if median >= 0.72 else 'fail'}  vs 0.72")
    print(f"CI hi = {hi:.4f}  {'PASS' if hi >= 0.72 else 'fail'}  vs 0.72")
    print()
    if lo >= 0.72:
        print("VERDICT: CI fully ≥ 0.72 — CHAT-TEMPLATE IS THE BUG. Flag removal unlocks paper floor.")
    elif hi >= 0.72 and lo < 0.72:
        print("VERDICT: CI straddles 0.72 — partial gap-closer. Queue C40b/c.")
    else:
        print(f"VERDICT: CI fully < 0.72 — chat-template is NOT the bottleneck. Gap confirmed structural.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
