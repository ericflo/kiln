#!/usr/bin/env python3
"""H15a: Marlin pack determinism correlation on docs/phase-c40f/summary.json.

Tests whether load-time variance in Marlin packed weights (model_load_secs)
correlates with downstream MTP acceptance variance (acceptance_rate).

Stdlib only — no numpy / scipy. Pearson + Spearman computed by hand.
"""

import json
import math
import os
import sys
from statistics import mean, stdev


HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(HERE, "summary.json")


def pearson_r(xs, ys):
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def average_ranks(values):
    """Return average ranks (1-indexed) handling ties via mean rank."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(xs, ys):
    return pearson_r(average_ranks(xs), average_ranks(ys))


def fisher_z_ci(r, n, conf=0.95):
    """95% CI for Pearson r via Fisher z-transform.

    Returns (lo, hi). Returns (nan, nan) if undefined (n<=3, |r|>=1).
    """
    if n <= 3 or abs(r) >= 1.0:
        return float("nan"), float("nan")
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - 3)
    # 1.96 = z(0.975) for 95% two-sided
    z_crit = 1.959964
    z_lo, z_hi = z - z_crit * se, z + z_crit * se
    r_lo = (math.exp(2 * z_lo) - 1) / (math.exp(2 * z_lo) + 1)
    r_hi = (math.exp(2 * z_hi) - 1) / (math.exp(2 * z_hi) + 1)
    return r_lo, r_hi


def pearson_t_stat(r, n):
    if abs(r) >= 1.0 or n <= 2:
        return float("nan")
    return r * math.sqrt((n - 2) / (1 - r * r))


# Critical |r| thresholds for two-tailed Pearson at df = n - 2.
# Hand-tabulated for n = 20 (df = 18) from the standard t-distribution.
# t_crit / sqrt(df + t_crit^2) gives r_crit; we hard-code the result for n=20.
PEARSON_CRIT_N20 = {
    0.05: 0.4438,
    0.01: 0.5614,
    0.001: 0.6787,
}

# Critical |rho| thresholds for two-tailed Spearman at n = 20 (Zar 1999, Tab. B.20).
SPEARMAN_CRIT_N20 = {
    0.05: 0.4500,
    0.01: 0.5908,
    0.001: 0.7155,
}


def significance_label(absval, table):
    for alpha in (0.001, 0.01, 0.05):
        if absval >= table[alpha]:
            return alpha
    return None


def main():
    with open(SUMMARY) as f:
        data = json.load(f)

    rows = data["rows"]
    xs = [r["model_load_secs"] for r in rows]
    ys = [r["acceptance_rate"] for r in rows]
    n = len(xs)
    if n != 20:
        print(f"WARNING: expected n=20, got n={n}", file=sys.stderr)

    pr = pearson_r(xs, ys)
    sr = spearman_rho(xs, ys)
    pr_lo, pr_hi = fisher_z_ci(pr, n)
    pr_t = pearson_t_stat(pr, n)

    pearson_alpha = significance_label(abs(pr), PEARSON_CRIT_N20) if n == 20 else None
    spearman_alpha = significance_label(abs(sr), SPEARMAN_CRIT_N20) if n == 20 else None

    print("=" * 72)
    print("H15a: Marlin pack determinism correlation")
    print("Source:  docs/phase-c40f/summary.json")
    print("Metrics: rows[].model_load_secs  vs  rows[].acceptance_rate")
    print("=" * 72)
    print()
    print(f"n                         = {n}")
    print(f"model_load_secs  mean     = {mean(xs):.6f}  sd = {stdev(xs):.6f}")
    print(f"                 min/max  = {min(xs):.6f} / {max(xs):.6f}")
    print(f"acceptance_rate  mean     = {mean(ys):.6f}  sd = {stdev(ys):.6f}")
    print(f"                 min/max  = {min(ys):.6f} / {max(ys):.6f}")
    print()
    print("--- Pearson product-moment correlation ---")
    print(f"  r                         = {pr:+.6f}")
    print(f"  95% CI (Fisher z)         = [{pr_lo:+.6f}, {pr_hi:+.6f}]")
    print(f"  t = r*sqrt((n-2)/(1-r^2)) = {pr_t:+.6f}  (df = {n - 2})")
    if pearson_alpha is None:
        print(f"  significance              = NOT SIGNIFICANT at p<0.05")
        print(f"  critical |r| at p<0.05    = {PEARSON_CRIT_N20[0.05]:.4f}")
    else:
        print(f"  significance              = significant at p<{pearson_alpha}")
    print()
    print("--- Spearman rank correlation ---")
    print(f"  rho                       = {sr:+.6f}")
    if spearman_alpha is None:
        print(f"  significance              = NOT SIGNIFICANT at p<0.05")
        print(f"  critical |rho| at p<0.05  = {SPEARMAN_CRIT_N20[0.05]:.4f}")
    else:
        print(f"  significance              = significant at p<{spearman_alpha}")
    print()
    print("--- Per-seed pairs ---")
    print(f"  {'seed':>4}  {'model_load_secs':>16}  {'acceptance_rate':>16}")
    for r in rows:
        print(f"  {r['seed']:>4}  {r['model_load_secs']:>16.9f}  {r['acceptance_rate']:>16.9f}")
    print()
    print("--- Verdict ---")
    # Marlin pack determinism is the hypothesis. RULED OUT iff neither
    # correlation is significant at p<0.05 AND |rho| < 0.5 (the bench-plan
    # threshold quoted in PR #527 §"Free pre-step (do BEFORE GPU spend)").
    # SUPPORTED iff either correlation is significant at p<0.05 AND |rho|
    # >= 0.5. INCONCLUSIVE otherwise (e.g. one is significant but |rho|<0.5,
    # which means a real but small effect that doesn't justify swapping H15b
    # for the deterministic-pack repro).
    rho_strong = abs(sr) >= 0.5
    pearson_sig = pearson_alpha is not None
    spearman_sig = spearman_alpha is not None
    any_sig = pearson_sig or spearman_sig

    if rho_strong and any_sig:
        verdict = "SUPPORTED"
    elif (not any_sig) and abs(sr) < 0.5:
        verdict = "RULED OUT"
    else:
        verdict = "INCONCLUSIVE"

    print(
        f"H15a verdict: {verdict} Marlin pack determinism as alpha-gap mechanism"
    )
    print()
    print("Decision rule applied:")
    print("  SUPPORTED   iff |rho| >= 0.5 AND (Pearson OR Spearman) sig at p<0.05")
    print("  RULED OUT   iff |rho| < 0.5 AND neither correlation sig at p<0.05")
    print("  INCONCLUSIVE otherwise")
    print()
    print("Recommendation:")
    if verdict == "RULED OUT":
        print("  - Marlin pack determinism is NOT the alpha-gap mechanism on this anchor.")
        print("  - Queue H15b: stratified C29 v2 reject-row top-K Jaccard / cos_sim")
        print("    probe (~30 min A6000, ~$0.25). Decision belongs to next planning")
        print("    cycle, not this PR.")
    elif verdict == "SUPPORTED":
        print("  - Marlin pack nondeterminism plausibly drives alpha variance.")
        print("  - Queue a deterministic-pack repro (e.g. KILN_W4A16=0 or pinned-seed")
        print("    Marlin pack) BEFORE H15b. Decision belongs to next planning cycle.")
    else:
        print("  - Effect direction is real but small, OR significant by one test only.")
        print("  - Queue H15b anyway (the cheaper, more direct probe) and revisit")
        print("    Marlin determinism if H15b is also null.")


if __name__ == "__main__":
    main()
