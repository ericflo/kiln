# Phase 7 Audit: H15a — Marlin Pack Determinism Correlation

Date: 2026-04-24
kiln main at audit time: `2f38eb6` (post-PR #527)
Parent PR: [#527 — phase7: MTP acceptance-rate state-of-play audit (doc-only)](https://github.com/ericflo/kiln/pull/527) (merged 2026-04-24T23:38)
Scope: $0 doc-only correlation analysis on existing `docs/archive/phase-c/phase-c40f/summary.json`. No pod, no SSH, no GPU spend, no Rust changes.

## Summary

PR #527's bench plan §"Free pre-step (do BEFORE GPU spend)" called for a $0 Spearman correlation between `model_load_secs` and `acceptance_rate` on the canonical N=20 C40f anchor (PR #379) to test whether load-time variance in Marlin packed weights drives MTP α variance. This audit executes that pre-step.

**Verdict: RULED OUT.** Marlin pack determinism is NOT the alpha-gap mechanism on this anchor.

- **Pearson r = +0.349** (95% CI via Fisher z: [-0.111, +0.686]; t = +1.580, df = 18) — NOT significant at p<0.05 (critical |r| at n=20 is 0.4438).
- **Spearman ρ = +0.111** — NOT significant at p<0.05 (critical |ρ| at n=20 is 0.4500).
- The weak Pearson signal is driven entirely by **seed 0**, the only seed with `model_load_secs` in the >3 s tail (3.548 s vs ~1.8 s for the other 19). Seed 0 also happens to have the joint-highest α (0.789). Spearman, which is rank-robust to that outlier, falls to +0.111 — i.e. essentially no monotone relationship.
- The bench plan threshold quoted in PR #527 (|ρ| > 0.5 → swap H15b for a Marlin determinism repro) is missed by 4.5×.

**Bonus structural finding:** the C40f anchor's `acceptance_rate` field has only **10 unique values across 20 seeds** — seeds {0, 10}, {1, 11}, {2, 12}, … {9, 19} produce bit-identical α pairs (0.788732394, 0.684210526, 0.628205128, …) while their `model_load_secs` differ. This is itself near-conclusive evidence that α at this anchor is determined by something deterministic in the workload (most plausibly the seeded shuffle of the humaneval prompt subset cycling with period 10), NOT by physical Marlin pack output. If Marlin nondeterminism drove α, paired seeds with different load times would NOT produce bit-identical α. See §3 for the table.

**Recommendation: queue H15b** (stratified C29 v2 reject-row top-K Jaccard / cos_sim probe, ~30 min A6000, ~$0.25) per PR #527 §"Recommended next H." Decision belongs to the next planning cycle, not this PR.

## Verdict

**Marlin pack determinism is RULED OUT as the alpha-gap mechanism at the C40f anchor.**

Reopen precondition: a fresh anchor (e.g. post-Marlin-refactor, larger N, or different prompt distribution) where (a) `model_load_secs` distribution is materially wider than the 1.57–3.55 s observed here AND (b) `acceptance_rate` is no longer collision-degenerate across seed-pairs. Until then, Marlin pack output is not measurably driving α variance on this checkpoint.

## Methodology

### Inputs

- `docs/archive/phase-c/phase-c40f/summary.json` — N=20 paper-floor sweep, anchor PR [#379](https://github.com/ericflo/kiln/pull/379) (2026-04-22). Each row contains `seed`, `model_load_secs`, `acceptance_rate`, plus throughput / VRAM fields not used here.
- 20 seed indices: 0–19. All `model_vram_mb` values are 16791 (one row at 16790 — within rounding noise).

### Metrics extracted

For each row, the pair `(model_load_secs, acceptance_rate)` is the unit of analysis. `model_load_secs` is the wall-clock cost of `kiln-bench` initialization, dominated by Marlin pack across 104 W4A16 projections (per `kiln-marlin-parallel-pack-pattern` agent note and PR #210 description). `acceptance_rate` is the fraction of MTP draft tokens accepted by the verifier, computed by `bench_latency_paged_mtp` per the C40f harness configuration in PR #379.

### Correlation method

Stdlib only — no numpy / scipy. Implemented in `scripts/phase-c40f/h15a_correlation.py`:

- **Pearson r** computed by hand from sums of products and standard deviations.
- **Spearman ρ** computed as Pearson on average-ranks (1-indexed, mean-rank tie handling).
- **95% CI for Pearson r** via Fisher z-transform: `z = 0.5·ln((1+r)/(1-r))`, `SE = 1/√(n-3)`, `z ± 1.959964·SE` → back-transform via `tanh(z) = (e²ᶻ−1)/(e²ᶻ+1)`.
- **t-statistic for Pearson:** `t = r·√((n-2)/(1-r²))` with `df = n-2 = 18`.
- **Significance** evaluated against hand-tabulated critical values for n=20 from the standard t-distribution (Pearson) and Zar (1999) Table B.20 (Spearman). Two-tailed thresholds:

| α | Pearson |r|crit (n=20, df=18) | Spearman |ρ|crit (n=20) |
| --- | --- | --- |
| 0.05 | 0.4438 | 0.4500 |
| 0.01 | 0.5614 | 0.5908 |
| 0.001 | 0.6787 | 0.7155 |

### Decision rule (set in advance, matches PR #527 bench plan)

- **SUPPORTED** iff |ρ| ≥ 0.5 AND (Pearson OR Spearman) significant at p<0.05.
- **RULED OUT** iff |ρ| < 0.5 AND neither correlation significant at p<0.05.
- **INCONCLUSIVE** otherwise (e.g. one test significant but |ρ| < 0.5, or |ρ| ≥ 0.5 but neither test significant — both shapes mean "real but not strong enough to swap H15b for a determinism repro").

The 0.5 |ρ| floor is the threshold quoted directly in PR #527 §"Free pre-step (do BEFORE GPU spend)": *"If the correlation is significant (|ρ| > 0.5), the next H becomes 'MTP cold-load determinism' instead of H15b."*

### Sample size justification

n=20 from the canonical C40f sweep is the largest single-anchor MTP α distribution in the project (per `mtp-bench-workload-sensitivity` and `kiln-mtp-alpha-regression-qwen35-4b` notes; PR #527 §3 confirms it is still the canonical N≥20 anchor as of `2f38eb6`). Power at n=20 to detect |ρ| ≥ 0.5 at α=0.05 two-tailed is approximately 0.65 — adequate for ruling out a strong effect, marginal for detecting a moderate one. A negative result at this n still supports RULED OUT in the |ρ| < 0.5 regime, which is what the bench plan asks. We do not have to detect a small effect to make the H15b decision.

## Findings

### Raw script output

See `docs/archive/phase-c/phase-c40f/h15a_correlation_output.txt`. Salient numbers:

| Metric | Value |
| --- | --- |
| n | 20 |
| `model_load_secs` mean ± sd | 1.897 ± 0.462 s |
| `model_load_secs` min / max | 1.571 / 3.548 s |
| `acceptance_rate` mean ± sd | 0.694 ± 0.052 |
| `acceptance_rate` min / max | 0.628 / 0.789 |
| Pearson r | **+0.349** |
| Pearson 95% CI (Fisher z) | [−0.111, +0.686] |
| Pearson t-stat (df=18) | +1.580 |
| Pearson p<0.05 critical \|r\| | 0.4438 → **NOT significant** |
| Spearman ρ | **+0.111** |
| Spearman p<0.05 critical \|ρ\| | 0.4500 → **NOT significant** |

### Outlier diagnostic

- 19 of 20 seeds have `model_load_secs` in [1.57, 2.72] s. Seed 0 alone is at 3.548 s, ~1.7× the median (1.694 s).
- Seed 0's α (0.789) is the joint-maximum across the sweep (tied with seed 10 at 0.789, whose load time is 1.571 s — the *minimum*).
- Spearman's near-zero rank correlation (+0.111) is the right summary because it is robust to seed 0's load-time outlier. Pearson's +0.349 inflates the apparent association by giving seed 0's 1.65 s leverage on the regression line.

### Period-10 collision in `acceptance_rate`

The C40f anchor's `acceptance_rate` field exhibits a strong structural artifact:

| seed pair | α (both seeds) | load_secs (a, b) |
| --- | --- | --- |
| 0, 10 | 0.788732394 | 3.548 / 1.571 |
| 1, 11 | 0.684210526 | 1.624 / 1.835 |
| 2, 12 | 0.628205128 | 1.643 / 1.653 |
| 3, 13 | 0.706666667 | 1.676 / 2.721 |
| 4, 14 | 0.641025641 | 1.676 / 1.974 |
| 5, 15 | 0.739726027 | 1.797 / 1.736 |
| 6, 16 | 0.662337662 | 1.654 / 1.848 |
| 7, 17 | 0.641025641 | 2.034 / 1.752 |
| 8, 18 | 0.693333333 | 1.939 / 1.729 |
| 9, 19 | 0.753424658 | 1.644 / 1.894 |

Across all 10 seed-pairs, α is bit-identical despite `model_load_secs` differing by 8–116%. **If Marlin pack output were physically nondeterministic and α-determining, paired seeds with materially different load times would not converge on the same α to nine decimal places.** This is independent corroboration of the RULED OUT verdict.

The most parsimonious mechanistic explanation is that the C40f harness seeds a workload sub-selection (humaneval prompt subset shuffle) with period 10, while the model state per-prompt is seed-deterministic given the same packed weights — i.e. the harness re-uses the same 10 effective workloads across the 20 seeds. This is consistent with the C40f command template (`docs/archive/phase-c/phase-c40f/command-template.txt`) using `--prompt-subset humaneval` with `--seeds 0..19` against a fixed prompt pool. Confirming this is out of scope for H15a (it does not affect the determinism verdict) but is filed as an open observation in §4.

### Cross-check vs known structure of the C40f anchor

- C40f's bootstrap-median 95% CI is [0.652, 0.723] (PR #379, `summary.median_alpha = 0.689`). The H15a Pearson 95% CI for the load-time / α correlation [−0.111, +0.686] crosses zero — i.e. we cannot reject the null of no correlation.
- C40f's 6/20 seeds-≥-0.72 count is consistent with PR #527's identity-bias diagnosis: a small fraction of seeds clear the paper floor, and the per-seed regime split is much larger than any plausible Marlin pack effect on the underlying α distribution.

## Recommended next step

**Queue H15b: stratified C29 v2 reject-row top-K Jaccard / cos_sim probe** per PR #527 §"Recommended next H." This is the cheaper, more direct test of the leading remaining hypothesis (verifier-side numerical drift on Class B reject rows) and does not depend on resolving the Marlin determinism question. Cost: ~30 min on A6000, ~$0.25 GPU spend. Owner: next planning cycle, not this PR.

**Do NOT queue:**

- ❌ A deterministic-pack repro (e.g. KILN_W4A16=0 or pinned-seed Marlin pack). The H15a verdict eliminates the precondition for that work. If H15b is also null and we re-anchor at a wider load-time distribution where the period-10 α collision is broken, this can be revisited.
- ❌ A larger-N MTP α sweep purely to chase the Pearson +0.349 signal. The Spearman ρ = +0.111 plus the outlier diagnostic plus the period-10 collision say the residual signal is single-seed leverage, not a population effect. Spending a pod on bigger-N would burn $$ for no decision yield.

## Anti-duplication evidence

Verified at audit time:

| Search | Result |
| --- | --- |
| `gh pr list -R ericflo/kiln --state all --search "H15a marlin determinism" --limit 5` | Only #527 (parent PR) — no duplicate H15a PR. |
| `gh pr list -R ericflo/kiln --state all --search "marlin pack determinism correlation" --limit 5` | Only #527. |
| `gh pr list -R ericflo/kiln --state all --search "model_load_secs acceptance_rate correlation" --limit 5` | Only #527. |

PR #527 (the parent) explicitly documents H15a as a "free pre-step (do BEFORE GPU spend)" that no PR has yet executed. This PR is that execution.

## Sources

### Files read

- `docs/archive/phase-c/phase-c40f/summary.json` (lines 1–222) — input data; 20 rows in `rows[]`, summary block in `summary{}`.
- `docs/audits/phase7-mtp-acceptance-state-of-play.md` (PR #527) — bench plan §"Free pre-step (do BEFORE GPU spend)" defining the |ρ| > 0.5 threshold and the H15b vs deterministic-pack-repro fork.
- `docs/archive/phase-c/phase-c40f/command-template.txt` — C40f harness invocation (for the period-10 hypothesis in §3).
- `PROFILING.md` §"Phase 7 MTP acceptance-rate state-of-play audit (2026-04-24)" — pointer pattern reused below.

### Files written

- `scripts/phase-c40f/h15a_correlation.py` — stdlib-only Pearson + Spearman correlation analysis script.
- `docs/archive/phase-c/phase-c40f/h15a_correlation_output.txt` — verbatim script output captured 2026-04-24.
- `docs/audits/phase7-h15a-marlin-determinism.md` — this audit.
- `PROFILING.md` — top-of-file pointer added.

### Agent notes consulted

- `kiln-marlin-parallel-pack-pattern` — confirms `model_load_secs` is dominated by Marlin pack of 128 projections (32 layers × 4) under KILN_W4A16=1.
- `marlin-w4a16-weight-vs-output-drift` — distinguishes weight-level vs output-level drift; relevant if H15b ever gets re-scoped to per-projection numerical drift.
- `marlin-coverage-threshold-for-decode-speedup` — confirms which projections are Marlin-packed at the C40f anchor (gate/up/down + q across all layers), which is the population whose pack-time variance would in principle propagate to α.
- `mtp-bench-workload-sensitivity` — supports the period-10 collision interpretation (workload distribution dominates α at this anchor).
- `kiln-bench-anchor-config-must-match-env-flags` — confirms the C40f anchor is a single env-flag configuration; we are not mixing W4A16=0 / =1 in this dataset.

### vLLM cross-references

None directly relevant — this is a kiln-internal Marlin pipeline question. vLLM uses Marlin too but does not publish per-seed α + load-time pairs at this granularity.

## Reopen / re-revisit triggers

H15a should be re-run if any of the following changes:

1. **Anchor refresh with wider load-time distribution.** If a future N≥20 sweep produces `model_load_secs` with sd > 0.8 s (vs 0.46 s here) — e.g. via cold-cache vs warm-cache mixing or a Marlin pack refactor — re-test the correlation. The current dataset's narrow load-time band is itself a power-limiting factor.
2. **Deterministic-pack landed.** If `kiln-marlin-parallel-pack-pattern` is replaced with a pinned-seed pack and the per-seed load-time variance collapses to <0.1 s, the H15a question becomes moot but the period-10 α collision should be re-investigated separately.
3. **C40f anchor superseded by a larger-N or differently-shuffled anchor.** If the workload changes such that α takes more than 10 unique values across 20 seeds, re-run H15a — the period-10 collision is the dominant suppressor of Spearman ρ in this dataset.
4. **H15b returns null.** If the stratified C29 v2 probe on reject rows is also clean (cos_sim ≥ 0.999 / Jaccard@5 ≥ 0.95), the project goal pivots to "kiln-native ceiling, escalate to vLLM α microbench" per PR #527 §4 fallback. H15a does not need to be re-run unless one of triggers (1)–(3) also applies.

## References

- [PR #527 — Phase 7 MTP acceptance-rate state-of-play audit](https://github.com/ericflo/kiln/pull/527) (parent; merged 2026-04-24)
- [PR #379 — Phase C40f N=20 paper-floor sweep](https://github.com/ericflo/kiln/pull/379) (anchor; merged 2026-04-22)
- [PR #525 — Phase 7 vLLM GDN audit (doc-only)](https://github.com/ericflo/kiln/pull/525) — doc-only audit precedent
- [PR #526 — Phase 7 SGLang radix audit (doc-only)](https://github.com/ericflo/kiln/pull/526) — doc-only audit precedent
- [PR #210 — Marlin parallel pack](https://github.com/ericflo/kiln/pull/210) — pack pipeline this audit is testing
- `docs/archive/phase-c/phase-c36/c36-identity-bias.md` — the leading remaining hypothesis (identity-bias regime split) that H15b targets
- `PROFILING.md` §"Phase 7 MTP acceptance-rate state-of-play audit (2026-04-24)" — the parent state-of-play
