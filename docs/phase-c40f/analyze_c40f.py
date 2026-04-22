#!/usr/bin/env python3
"""Phase C40f — analyze N=20 BF16 HumanEval JSONs and emit summary stats.

Mirrors the C39/C40a/C40b bootstrap method:
- 10,000 bootstrap resamples
- rng seed 12345
- percentile CI over resampled medians
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/phase-c40f")
N_BOOT = 10_000
RNG_SEED = 12345
PAPER_FLOOR = 0.72

C39_MEDIAN = 0.6933
C39_MEAN = 0.6868
C39_STDEV = 0.0499
C39_MIN = 0.5679
C39_MAX = 0.7778
C39_CI_LO = 0.6602
C39_CI_HI = 0.7162
C39_CLEARS = 4
C39_MEDIAN_DECODE_TPS = 41.6
C39_MEDIAN_ITL_MS = 24.0

C40E_W4A16_SEED0 = 0.6282051282051282
C40E_BF16_SEED0 = 0.7887323943661971


def load_rows(results: Path) -> list[tuple[int, float, float, float]]:
    rows = []
    for seed in range(20):
        path = results / f"seed-{seed}.json"
        with path.open() as f:
            data = json.load(f)
        lat = data["latency"]
        rows.append(
            (
                seed,
                float(lat["acceptance_rate"]),
                float(lat["decode_tokens_per_sec"]),
                float(lat["mean_inter_token_ms"]),
            )
        )
    return rows


def bootstrap_ci(values: list[float]) -> tuple[float, float]:
    rng = random.Random(RNG_SEED)
    n = len(values)
    medians = []
    for _ in range(N_BOOT):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        medians.append(statistics.median(sample))
    medians.sort()
    lo = medians[int(0.025 * N_BOOT)]
    hi = medians[int(0.975 * N_BOOT)]
    return lo, hi


def main() -> int:
    rows = load_rows(RESULTS)
    alphas = [row[1] for row in rows]
    decode_tps = [row[2] for row in rows]
    itl_ms = [row[3] for row in rows]

    median = statistics.median(alphas)
    mean = statistics.fmean(alphas)
    stdev = statistics.stdev(alphas)
    min_alpha = min(alphas)
    max_alpha = max(alphas)
    ci_lo, ci_hi = bootstrap_ci(alphas)
    clears = sum(alpha >= PAPER_FLOOR for alpha in alphas)
    median_decode = statistics.median(decode_tps)
    median_itl = statistics.median(itl_ms)

    print(f"=== C40f analysis ({RESULTS}) ===")
    print("\nPer-seed metrics:")
    print("  seed | alpha      | decode_tps | mean_itl_ms")
    print("  -----+------------+------------+------------")
    for seed, alpha, dtps, itl in rows:
        print(f"  {seed:4d} | {alpha:10.10f} | {dtps:10.4f} | {itl:10.4f}")

    print("\nSummary:")
    print(f"  N                         = {len(alphas)}")
    print(f"  median alpha              = {median:.10f}")
    print(f"  mean alpha                = {mean:.10f}")
    print(f"  stdev alpha               = {stdev:.10f}")
    print(f"  min / max alpha           = {min_alpha:.10f} / {max_alpha:.10f}")
    print(
        f"  95% CI on median alpha    = [{ci_lo:.10f}, {ci_hi:.10f}] "
        f"(bootstrap, {N_BOOT} resamples, rng={RNG_SEED})"
    )
    print(f"  seeds clearing 0.72       = {clears}/20")
    print(f"  median decode tok/s       = {median_decode:.10f}")
    print(f"  median mean ITL ms        = {median_itl:.10f}")

    print("\nComparison vs C39 W4A16 N=20:")
    print(f"  median delta              = {median - C39_MEDIAN:+.10f}")
    print(f"  mean delta                = {mean - C39_MEAN:+.10f}")
    print(f"  stdev delta               = {stdev - C39_STDEV:+.10f}")
    print(f"  lower-bound delta         = {ci_lo - C39_CI_LO:+.10f}")
    print(f"  upper-bound delta         = {ci_hi - C39_CI_HI:+.10f}")
    print(f"  clears delta              = {clears - C39_CLEARS:+d}")
    print(f"  median decode delta       = {median_decode - C39_MEDIAN_DECODE_TPS:+.10f}")
    print(f"  median ITL delta          = {median_itl - C39_MEDIAN_ITL_MS:+.10f}")

    print("\nComparison vs C40e paired seed 0:")
    print(f"  seed-0 alpha              = {rows[0][1]:.10f}")
    print(f"  seed-0 BF16 delta         = {rows[0][1] - C40E_BF16_SEED0:+.10f}")
    print(f"  seed-0 vs W4A16 delta     = {rows[0][1] - C40E_W4A16_SEED0:+.10f}")

    print("\nVerdict:")
    if ci_lo >= PAPER_FLOOR:
        print("  BF16 clears the 0.72 HumanEval floor distributionally.")
    elif median > C39_MEDIAN and ci_hi >= PAPER_FLOOR:
        print("  BF16 improves over C39 but remains distributionally ambiguous.")
    else:
        print("  BF16 does not beat the C39 W4A16 distribution on the median;")
        print("  the seed-0 win from C40e does not generalize cleanly across N=20.")
        if ci_hi >= PAPER_FLOOR:
            print("  The CI still straddles 0.72, so the distributional floor question remains ambiguous.")
        else:
            print("  The CI stays below 0.72, so BF16 fails the floor distributionally.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
