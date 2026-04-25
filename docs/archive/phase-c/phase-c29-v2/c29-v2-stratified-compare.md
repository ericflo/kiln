# Phase C29 v2 — Stratified MTP-Logits Reject-Row Probe (H15b)

**Primary tap**: `c14__logits` — `[1, 1, 248320]` (full base-vocab head over the MTP-block output).
**Total dump pairs**: 22 (0 errors).
**Labeled coverage**: accept n=7, reject n=15, unlabeled n=0.

## Verdict

**kiln_native_ceiling**

- reject_cos_sim_median = 0.9999776060227605
- reject_cos_sim_p10    = 0.9999706320862227
- accept_cos_sim_median = 0.999979479133117
- accept_cos_sim_p10    = 0.9999682113961627

**Next action**: Queue a vLLM α microbench to establish the external-reference upper bound on Qwen3.5-4B A6000 bs=1 at this workload. Kiln's MTP head logits are at the BF16-noise cosine ceiling even on rejected drafts.

Decision rule (from PR #527 §"Recommended next H"):

- `reject_cos_sim_median >= 0.999` → **kiln_native_ceiling** → queue vLLM α microbench
- `reject_cos_sim_median <  0.99 ` → **verifier_numerical_drift** → queue per-layer bisect on reject rows
- `0.99 <= reject_cos_sim_median < 0.999` → **ambiguous** → expand seeds and re-run

## Accept vs Reject stratified aggregate

### Accept-only — n=7

| metric | median | mean | p10 | p90 | min | max |
|--------|-------:|-----:|----:|----:|----:|----:|
| `cos_sim` | 0.999979 | 0.999976 | 0.999968 | 0.999982 | 0.999956 | 0.999982 |
| `max_abs_delta` | 0.112608 | 0.111731 | 0.107673 | 0.114927 | 0.106192 | 0.116446 |
| `top1_match_rate` | 1 | 1 | 1 | 1 | 1 | 1 |
| `jaccard_k1` | 1 | 1 | 1 | 1 | 1 | 1 |
| `jaccard_k5` | 1 | 1 | 1 | 1 | 1 | 1 |
| `jaccard_k10` | 1 | 1 | 1 | 1 | 1 | 1 |
| `jaccard_k20` | 1 | 0.959184 | 0.904762 | 1 | 0.904762 | 1 |
| `kl_kiln_to_ref` | 0.000201713 | 0.000271072 | 5.25815e-06 | 0.00064876 | 1.16609e-06 | 0.000985153 |
| `kl_ref_to_kiln` | 0.000201657 | 0.000270778 | 5.26956e-06 | 0.000646506 | 1.1647e-06 | 0.000977172 |
| `ref_prob_at_kiln_top1` | 0.641565 | 0.695396 | 0.397155 | 0.994204 | 0.255464 | 0.995776 |

### Reject-only — n=15

| metric | median | mean | p10 | p90 | min | max |
|--------|-------:|-----:|----:|----:|----:|----:|
| `cos_sim` | 0.999978 | 0.999978 | 0.999971 | 0.999984 | 0.999957 | 0.999991 |
| `max_abs_delta` | 0.115157 | 0.114381 | 0.101696 | 0.124478 | 0.0990372 | 0.14489 |
| `top1_match_rate` | 1 | 1 | 1 | 1 | 1 | 1 |
| `jaccard_k1` | 1 | 1 | 1 | 1 | 1 | 1 |
| `jaccard_k5` | 1 | 0.955556 | 0.8 | 1 | 0.666667 | 1 |
| `jaccard_k10` | 1 | 0.963636 | 0.818182 | 1 | 0.818182 | 1 |
| `jaccard_k20` | 1 | 0.980952 | 0.904762 | 1 | 0.904762 | 1 |
| `kl_kiln_to_ref` | 0.000178244 | 0.000414249 | 4.93353e-06 | 0.0010894 | 2.79728e-06 | 0.00240375 |
| `kl_ref_to_kiln` | 0.000170655 | 0.000407592 | 4.81977e-06 | 0.00108554 | 2.71323e-06 | 0.00235354 |
| `ref_prob_at_kiln_top1` | 0.94523 | 0.832804 | 0.621821 | 0.995848 | 0.128099 | 0.99915 |

### All labeled + unlabeled (reference) — n=22

| metric | median | mean | p10 | p90 | min | max |
|--------|-------:|-----:|----:|----:|----:|----:|
| `cos_sim` | 0.999978 | 0.999977 | 0.99997 | 0.999984 | 0.999956 | 0.999991 |
| `max_abs_delta` | 0.113045 | 0.113538 | 0.102356 | 0.122483 | 0.0990372 | 0.14489 |
| `top1_match_rate` | 1 | 1 | 1 | 1 | 1 | 1 |
| `jaccard_k1` | 1 | 1 | 1 | 1 | 1 | 1 |
| `jaccard_k5` | 1 | 0.969697 | 1 | 1 | 0.666667 | 1 |
| `jaccard_k10` | 1 | 0.975207 | 0.836364 | 1 | 0.818182 | 1 |
| `jaccard_k20` | 1 | 0.974026 | 0.904762 | 1 | 0.904762 | 1 |
| `kl_kiln_to_ref` | 0.000189979 | 0.000368692 | 4.35204e-06 | 0.000936919 | 1.16609e-06 | 0.00240375 |
| `kl_ref_to_kiln` | 0.000186156 | 0.000364061 | 4.25927e-06 | 0.00092996 | 1.1647e-06 | 0.00235354 |
| `ref_prob_at_kiln_top1` | 0.901257 | 0.789083 | 0.499074 | 0.995728 | 0.128099 | 0.99915 |

## Per-position stratified cos_sim medians

| mtp_pos | accept n | accept median cos | accept p10 | reject n | reject median cos | reject p10 |
|--------:|---------:|------------------:|-----------:|---------:|------------------:|-----------:|
| 0 | 2 | 0.999979 | 0.999977 | 3 | 0.999977 | 0.999977 |
| 1 | 1 | 0.999980 | 0.999980 | 5 | 0.999978 | 0.999973 |
| 2 | 2 | 0.999981 | 0.999980 | 4 | 0.999977 | 0.999961 |
| 3 | 2 | 0.999966 | 0.999958 | 3 | 0.999983 | 0.999978 |

## Secondary taps (sanity, not stratified)

| tap | n | median cos | min cos | median max\|Δ\| |
|-----|--:|-----------:|--------:|----------------:|
| `h_main` | 22 | 1.000000 | 1.000000 | 0.000e+00 |
| `c14__post_block` | 22 | 0.999972 | 0.999960 | 4.258e-02 |
| `c14__post_norm` | 22 | 0.999971 | 0.999958 | 1.284e-01 |

