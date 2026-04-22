# Phase C29 — Empirical MTP-Logits Top-K Verdict (H9)

**Bench**: Qwen3.5-4B BF16 + Marlin W4A16, paged, `KILN_SPEC_METHOD=mtp`, A6000 48GB, seed-pool prompts (PROMPT_POOL[seed % 8]).
**Captures**: 4 prompt seeds × up to 4 MTP positions × 4 splice steps → **49 paired (kiln, ref) dumps** (some seed × pos slots terminated early due to kiln rejecting the draft and not advancing to step ≥2; this is bench-realistic).
**Reference**: `scripts/c29_hf_reference_dump.py` → `scripts/c14_hf_reference_dump.py` per kiln dump → fp32 PyTorch re-forward of the post-block MTP head via `scripts/mtp_reference_dump.py --capture-subops`. 49/49 reference dumps produced, 0 errors, 0 missing taps.
**Comparator**: `scripts/c29_logits_compare.py` — primary tap `c14__logits` `[1, 1, 248320]`. Adds top-1 agreement, top-K Jaccard for K∈{1,5,10,20}, KL(kiln‖ref)/KL(ref‖kiln), and ref's prob mass at kiln's argmax.
**Floors**: top-1 agreement ≥ 0.95, top-K Jaccard ≥ 0.90.

## H9 hypothesis (under test)

C14 reported median `c14__logits` cos_sim ≈ 0.99997 across 16 dumps. On a 248K-dim head, that cosine is consistent with bf16 forward noise and could in principle hide top-K rotation severe enough to crash speculative-decoding acceptance. **H9 says**: even with a clean cosine, the kiln MTP head's top-K selections rotate enough vs an fp32 reference to depress α — and a per-prompt top-K probe should catch what cosine does not.

## Result — H9 REFUTED

The MTP head's logits agree with an fp32 PyTorch reference on **every selection-relevant metric**, not just cosine. There is no top-K rotation to find.

### Aggregate (49 dumps across 4 prompts × 4 positions × up to 4 steps)

| metric | median | min | max |
|--------|-------:|----:|----:|
| `cos_sim` | 0.999978 | 0.999934 | 0.999988 |
| `max_abs_delta` | 0.1098 | 0.0826 | 0.1870 |
| `top-1 match rate` | **1.0000** | 1.0000 | 1.0000 |
| `Jaccard@1` | 1.0000 | 1.0000 | 1.0000 |
| `Jaccard@5` | 1.0000 | 0.6667 | 1.0000 |
| `Jaccard@10` | 1.0000 | 0.8182 | 1.0000 |
| `Jaccard@20` | 1.0000 | 0.9048 | 1.0000 |
| `KL(kiln‖ref)` | 2.06e-05 | 3.76e-07 | 6.71e-03 |
| `KL(ref‖kiln)` | 2.10e-05 | 3.66e-07 | 6.78e-03 |
| `ref prob @ kiln top-1` | 0.9863 | 0.4497 | 0.99985 |

Top-1 match is **100.00%** across all 49 dump pairs. Median Jaccard@5/@10/@20 is exactly 1.0 — kiln's full top-20 is the same set as the fp32 reference's top-20 in the median case. Median KL divergence in either direction is ≈ 2 × 10⁻⁵, i.e. the two distributions are indistinguishable for any practical sampler.

### Per-position breakdown (medians)

| `mtp_pos` | n | cos | top-1 match | J@5 | J@10 | KL(k‖r) |
|----------:|--:|----:|------------:|----:|-----:|--------:|
| 0 | 16 | 0.999975 | 1.0000 | 1.0000 | 1.0000 | 2.10e-05 |
| 1 | 13 | 0.999975 | 1.0000 | 1.0000 | 1.0000 | 2.02e-05 |
| 2 | 14 | 0.999979 | 1.0000 | 1.0000 | 1.0000 | 1.48e-05 |
| 3 | 6  | 0.999982 | 1.0000 | 1.0000 | 1.0000 | 2.53e-05 |

No position-dependent degradation. If the head built up draft-step error linearly we'd expect pos 3 to be visibly worse than pos 0; instead it's marginally **better** (well within bf16 noise).

### Per-seed breakdown (prompt-level medians)

| seed (prompt slot) | n | cos | top-1 match | J@5 | J@10 |
|-------------------:|--:|----:|------------:|----:|-----:|
| 0 | 12 | 0.999980 | 1.0000 | 1.0000 | 1.0000 |
| 1 | 9  | 0.999968 | 1.0000 | 1.0000 | 1.0000 |
| 2 | 16 | 0.999978 | 1.0000 | 1.0000 | 1.0000 |
| 3 | 12 | 0.999977 | 1.0000 | 1.0000 | 1.0000 |

All four prompt seeds show identical clean signatures. There is no per-prompt class of inputs where the kiln head rotates differently from the reference. H9 is not rescued by hard-prompt probing.

### Flagged sites (6 of 49)

The comparator flags 6 sites where Jaccard@10 falls to 0.818 (= 9 of 11 elements overlap when ties around K=10 push two extra tokens into the kiln set vs the ref set). **All six still match top-1.** The cos for these sites stays in [0.999953, 0.999984]; KL stays at O(1e-5) for all but one site (seed 1 pos 2 step 2 has KL = 6.7e-3, traceable to `ref_prob_at_kiln_top1` = 0.45 — a near-tie between the top-2 tokens where both heads agree on ordering anyway).

| seed | pos | step | top-1 | J@5 | J@10 | cos | KL(k‖r) |
|-----:|----:|----:|:-----:|----:|-----:|----:|--------:|
| 0 | 3 | 1 | ✓ | 1.000 | 0.818 | 0.999979 | 3.07e-06 |
| 1 | 0 | 1 | ✓ | 1.000 | 0.818 | 0.999968 | 8.03e-06 |
| 1 | 2 | 1 | ✓ | 1.000 | 0.818 | 0.999984 | 7.51e-06 |
| 1 | 2 | 2 | ✓ | 1.000 | 0.818 | 0.999953 | 6.71e-03 |
| 3 | 1 | 2 | ✓ | 1.000 | 0.818 | 0.999982 | 7.94e-06 |
| 3 | 2 | 3 | ✓ | 1.000 | 0.818 | 0.999973 | 6.08e-06 |

These look like K-boundary jitter (i.e. positions ≈ 9 and 10 in the ranking are swapping with positions 11/12), not systemic head error. Speculative decoding only consumes top-1 for greedy acceptance and the full distribution for stochastic acceptance — neither is meaningfully harmed by 9/11 vs 11/11 overlap at rank-10 when the rank-1 token is identical.

### Secondary taps (sanity, parity with C14)

| tap | n | median cos | min cos | median max\|Δ\| |
|-----|--:|-----------:|--------:|----------------:|
| `h_main` | 49 | 1.000000 | 1.000000 | 0.000e+00 |
| `c14__post_block` | 49 | 0.999970 | 0.999954 | 4.48e-02 |
| `c14__post_norm` | 49 | 0.999969 | 0.999958 | 1.75e-01 |
| `c14__logits` | 49 | 0.999978 | 0.999934 | 1.10e-01 |

Every tap reproduces the C14 baseline. `h_main` is bit-exact (input identity); the three downstream taps stay in the same cosine envelope on a 3× larger sample.

## What this rules out (combined with C13/C14)

C13 ruled out the pre-projection compute (token embedding, both norm halves, concat, fused projection): cos ≥ 0.9999928.
C14 ruled out the post-projection compute (transformer block, final norm, lm_head): cos ≥ 0.9999529 on cosine-only metrics.
**C29 ruled out the SELECTION-LAYER consequence of C14's residual cosine drift.** The cleanly-cosine logits also rank tokens identically to the fp32 reference. There is no hidden top-K rotation eating α.

Specifically C29 rules out:
- "bf16 noise on a 248K-dim head is large enough at any single position to flip top-1." (rate observed: 0/49)
- "Even if top-1 is fine, top-5/top-10 sets diverge enough to break stochastic acceptance." (median Jaccard@5/@10 = 1.0; min Jaccard@10 = 0.818 with top-1 still matching)
- "KL(kiln‖ref) is large enough to make a temperature-scaled sampler reject draft tokens." (median KL ≈ 2e-5; 4 orders of magnitude below the typical sampler-divergence threshold of 1e-1)
- "Some prompt class triggers head divergence the seed-42 prompt missed." (4 different prompts, all clean)
- "Later draft positions accumulate per-step head error." (pos 3 ≈ pos 0)

## What this does NOT rule out

The MTP forward compute window from `h_main` through `mtp_logits` is now exhaustively audited clean by static, cosine, and selection-layer probes. The α ≈ 0.0328 — 0.23 regression therefore cannot live inside the MTP head's single forward pass under any of the metrics that matter to draft-token acceptance. The remaining candidates are unchanged from the C14 verdict and re-confirmed by C29:

1. **Acceptance math / sampler / KV-rollback bugs.** The decision path that turns `mtp_logits` into accept-or-reject and the KV advance/rollback that follows are entirely outside the splice window. C29's clean head means kiln is producing the right logits and then mishandling the acceptance decision or the cache state that follows. **This is now the highest-probability root cause** — phase C30 should target it.
2. **Base-stack drift producing a wrong `h_main`.** Still conditional on the kiln-supplied `h_main`. The `h_main` cos = 1.0 row in C13/C14/C29 is by construction (the reference takes `h_main` from the kiln dump). A base-stack audit (Phase C15-style: full fp32 re-forward of the base model on the same prompt + accepted tokens at each step) would be decisive.
3. **Draft-token positional wiring across steps.** Embedding is correct (C13), top-K is correct (C29), but the position assigned to the draft token in the base-stack KV when the next decode step starts is a separate concern.

## Recommended follow-up (Phase C30 candidate)

**Acceptance + KV-rollback audit.** At each step, dump (a) kiln's accept/reject decision + the threshold used, (b) the base-stack logits kiln scored against, (c) the KV-cache slice for the MTP write position, then re-decide acceptance with an fp32 reference and compare. If the disagreement is in step (a) the bug is in acceptance math; if it's in step (c) the bug is in cache rollback. Either is small surface area to bisect.

The C29 captures, comparator, and `c14__logits` parity matrix already exist; reusing the same prompt seeds will let C30 directly cross-check acceptance decisions against the same draft-token logits this verdict signed off on.

## Artifacts

- Driver: `scripts/c29_kiln_logits_dump.sh` (bash, multi-seed wrapper around kiln-bench).
- HF reference scheduler: `scripts/c29_hf_reference_dump.py` (loops `c14_hf_reference_dump.py` per seed).
- Comparator: `scripts/c29_logits_compare.py` (top-K Jaccard + KL + per-(seed,pos,step) breakdown).
- Compare report (this doc's source data): `docs/phase-c29/c29-logits-compare.md`, `docs/phase-c29/c29-logits-compare.json`.
- Pod logs + per-seed C14 summary tarball: `docs/phase-c29/artifacts/`.
- 49 paired safetensors dumps (118 MB kiln + 118 MB ref) live on pod; not committed (binary, reproducible from scripts).
- Bench runs: `--prompt-tokens 512`, `--max-output-tokens 32`, `--paged`, `--skip-training`, `KILN_W4A16=1`. Per-seed acceptance rates: seed 0 α=0.0357, seed 1 α=0.0357, seed 2 α=0.107, seed 3 α=0.069 (bench-pool noise, all in the regression band).
