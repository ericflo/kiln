#!/usr/bin/env python3
"""Phase C40b — analyze N=20 seed JSONs and emit bootstrap CI on α median.

Mirrors C39/C40a stats: 10,000 bootstrap resamples, rng=12345, percentile
method (p2.5/p97.5 of resampled medians).
"""
from __future__ import annotations
import json, sys, glob, random, statistics
from pathlib import Path

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/c40b-results")
N_BOOT = 10_000
RNG_SEED = 12345

def extract_alpha(path: Path) -> float | None:
    """Return MTP α from a kiln-bench JSON, or None on failure."""
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  WARN: failed to parse {path.name}: {e}", file=sys.stderr)
        return None
    lat = data.get("latency", {})
    # Bench JSON uses "acceptance_rate" inside "latency" block.
    alpha = lat.get("acceptance_rate")
    if alpha is None:
        # Fallback: scan stderr log if present.
        return None
    return float(alpha)

def bootstrap_ci(values: list[float], n_boot: int = N_BOOT, seed: int = RNG_SEED, q_lo=0.025, q_hi=0.975):
    rng = random.Random(seed)
    n = len(values)
    medians = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        medians.append(statistics.median(sample))
    medians.sort()
    lo = medians[int(q_lo * n_boot)]
    hi = medians[int(q_hi * n_boot)]
    return lo, hi

def main():
    print(f"=== C40b analysis ({RESULTS}) ===")
    sanity_path = RESULTS / "sanity_seed0_temp0.json"
    if sanity_path.exists():
        sanity_alpha = extract_alpha(sanity_path)
        print(f"\nSanity (seed=0, temperature=0.0): α = {sanity_alpha}")
    else:
        sanity_alpha = None
        print(f"\nSanity file missing: {sanity_path}")

    # C40b N=20 seeds.
    rows: list[tuple[int, float]] = []
    missing: list[int] = []
    for seed in range(20):
        p = RESULTS / f"c40b_seed{seed}_temp0.1.json"
        if not p.exists():
            missing.append(seed)
            continue
        a = extract_alpha(p)
        if a is None:
            missing.append(seed)
            continue
        rows.append((seed, a))

    if missing:
        print(f"\nMissing or unparseable seeds: {missing}")

    rows.sort()
    print(f"\nC40b per-seed α (N={len(rows)} of 20, temperature=0.1, humaneval):")
    print("  seed | alpha")
    print("  -----+--------")
    for seed, a in rows:
        print(f"  {seed:4d} | {a:.4f}")

    if not rows:
        print("\nNo valid C40b seeds — cannot compute CI.")
        return 1

    alphas = [a for _, a in rows]
    median = statistics.median(alphas)
    mean = statistics.fmean(alphas)
    stdev = statistics.stdev(alphas) if len(alphas) > 1 else 0.0
    lo, hi = bootstrap_ci(alphas)
    print(f"\nC40b summary (N={len(alphas)}):")
    print(f"  median = {median:.4f}")
    print(f"  mean   = {mean:.4f}")
    print(f"  stdev  = {stdev:.4f}")
    print(f"  95% CI on median (bootstrap, 10000 resamples, rng={RNG_SEED}): [{lo:.4f}, {hi:.4f}]")

    # C39 baseline reference (HumanEval, temperature=0.0, chat-template ON, N=20):
    C39_MEDIAN = 0.6933
    C39_LO = 0.6602
    C39_HI = 0.7162
    PAPER_FLOOR = 0.72
    print(f"\nC39 baseline: median {C39_MEDIAN:.4f}, CI [{C39_LO:.4f}, {C39_HI:.4f}]")
    print(f"Qwen3.5 paper floor: α ≥ {PAPER_FLOOR}")

    delta_median = median - C39_MEDIAN
    print(f"\nC40b - C39 median delta: {delta_median:+.4f}")

    # Verdict
    print("\n=== Verdict ===")
    if lo > PAPER_FLOOR:
        print(f"  CI ENTIRELY ABOVE paper floor ({PAPER_FLOOR}). PASS — temperature=0.1 recovers α.")
    elif hi < PAPER_FLOOR:
        print(f"  CI ENTIRELY BELOW paper floor ({PAPER_FLOOR}). FAIL.")
    else:
        print(f"  CI STRADDLES paper floor ({PAPER_FLOOR}).")

    if delta_median >= 0.03:
        print(f"  Median improved by ≥ 0.03 absolute (Δ={delta_median:+.4f}). Hypothesis SUPPORTED.")
    elif delta_median <= -0.03:
        print(f"  Median worsened by ≥ 0.03 absolute (Δ={delta_median:+.4f}). Hypothesis INVERTED.")
    else:
        print(f"  Median within ±0.03 of C39 (Δ={delta_median:+.4f}). Hypothesis FALSIFIED — greedy is not uniquely harmful.")

    if sanity_alpha is not None:
        print(f"\n  Sanity check: temperature=0.0 seed=0 α = {sanity_alpha:.4f}.")
        print( "    Default path is byte-identical to pre-change if this matches a C39 seed=0 α.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
