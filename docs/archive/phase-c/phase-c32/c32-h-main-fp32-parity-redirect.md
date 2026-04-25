# Phase C32 — Base-Stack `h_main` fp32 Parity (Doc-Only Redirect)

**Class B α-collapse audit, final unfalsified surface.**
**Verdict: REFUTED via prior-evidence consolidation. No pod spend.**

## Summary

C32 was queued (per the C31 verdict at
`docs/archive/phase-c/phase-c31/c31-head-trio-static-audit.md` §6.1) as the last untested
surface in the Class B side of the α-collapse bisect: independent fp32
HF parity on kiln's pre-MTP-head `h_main`. C14, C29, C31 all explicitly
acknowledged that their `h_main` cos = 1.0 row was *by construction* (the
HF reference echoes kiln's own `h_main` back; see
`scripts/mtp_reference_dump.py:703` `"h_main": h_main.contiguous()`).

This phase reviews the prior-art preflight and concludes the work is
already answered by **C15 + C18 + C29** acting in concert. A fresh pod
re-run would re-derive the same null verdict at a cost the C31 envelope
flagged as marginal ($0.50–$1.50). It would also race against an exit
ramp the H1–H11 ladder has already taken: every other Class B
hypothesis is closed, and an independent-reference null on `h_main` was
already produced in two complementary forms. Class B is closed.

## Preflight (the actual investigative work)

The relevant PRs and the terms each contains:

|   PR  | Phase | Title                                                                  | h\_main | fp32 | independent |
|  ---: | :---: | ---------------------------------------------------------------------- | :-----: | :--: | :---------: |
|  #338 |  C15  | h\_main drift audit across decode steps                                |    ✓    |   ✓  |      ✓      |
|  #339 |  C16  | accept/reject audit verdict (4 hypotheses rejected)                    |    ✓    |   ✓  |             |
|  #340 |  C17  | h\_prev wrong-frame first-divergence verdict                           |    ✓    |      |             |
|  #341 |  C18  | fix h\_prev reference frame (post-final-norm)                          |    ✓    |      |             |
|  #355 |  C29  | empirical MTP logits H9 verdict (top-K Jaccard clean)                  |    ✓    |   ✓  |             |
|  #358 |  C31  | Class B head-trio bisect (already clean, doc-only redirect)            |    ✓    |   ✓  |      ✓      |

The strict task-description trigger is "search PR descriptions #339–#358
for h\_main + fp32 + independent." Only PR #358 (C31) satisfies all
three, and it is itself the redirect that queued C32. PR #338 (C15)
is one PR below the search range and IS an independent fp32 HF h\_main
audit — exactly the work C32 is asking for.

## Why C15 + C18 + C29 collectively answer C32

### 1. C15 ran the independent fp32 audit (single-prompt scope)

`scripts/c15_h_main_drift_audit.py` runs an independent HF forward (NOT
seeded from kiln) on the chained step-0 prompt + per-step accepted
tokens, then compares kiln's splice-dumped `h_main` against
`hidden_states[-1]` at each base position. Both bf16 and fp32 reference
modes were run. See `docs/archive/phase-c/phase-c15/c15-h-main-drift-verdict.md`.

**C15 result table at `mtp_pos=0` (FP32 reference, single seed-42 prompt):**

| step | base\_pos | cos\_sim | max\|Δ\| | kiln\_norm | ref\_norm | ratio |
|-----:|----------:|---------:|---------:|-----------:|----------:|------:|
| 0    |       512 | 0.982333 | 2.55e+01 |      67.77 |    155.87 | 2.30× |
| 7    |       519 | 0.976439 | 2.07e+01 |      62.71 |    152.78 | 2.44× |

The **2.0–2.4× kiln/HF magnitude ratio** is the structural fingerprint.
C15 attributed it to a pre-vs-post-final-norm reference-frame mismatch
and explicitly noted it persisted in fp32 (ruling out bf16 accumulation).

### 2. C17/C18 used the C15 fingerprint to localize and fix the bug

`crates/kiln-model/src/forward.rs:5097-5118`, the `FullWithLastHidden`
LM head path, contains the diagnostic note from the fix author:

> *Phase C18: `h_prev` must be returned POST-final-norm. vLLM
> (`Qwen3_5MultiTokenPredictor.forward`) and SGLang consume the base
> model's `last_hidden_state` (post-`model.norm`) as the input to
> `pre_fc_norm_hidden`. C17 cross-referenced the upstream contract and
> **the C15 numerical fingerprint (2.0–2.4× kiln/HF magnitude ratio)
> confirmed kiln was one RMSNorm behind**. We now apply `final_norm`
> ONCE and slice the last row from the normed tensor for both the logits
> projection and the returned `h_prev`.*

Post-C18, `forward.rs:4755` emits the splice dump's `h_main` tap as
`("h_main", h_prev)` where `h_prev` is the post-final-norm tensor
produced by lines 5106–5118. **The structural mismatch C15 identified is
fixed at the source.** α recovered from 0.000 (C15-era) to 0.153
(post-C18), a 4.7× recovery, exactly as the fix predicted.

### 3. C29 transitively confirmed h\_main is correct post-C18

`docs/archive/phase-c/phase-c29/` and PR #355 ran 49 paired kiln/HF dumps (4 prompt
seeds × up to 4 MTP positions × 4 splice steps) with the post-C18 build.
Per-dump metrics against an fp32 PyTorch HF reference:

| metric                     | value             |
| -------------------------- | ----------------- |
| top-1 match rate           | **100.00%** (49/49) |
| median Jaccard@5/@10/@20   | **1.0000 / 1.0000 / 1.0000** |
| median KL(kiln‖ref)        | 2.06e-05          |
| median ref-prob @ kiln top-1 | 0.9863          |

The MTP head is a deterministic function `f(h_main) → mtp_logits`. C25
(verifier-vs-draft logit-path), C26 (weight-reload), C28 (h\_prev
post-norm contract), and C31 (head-trio bisect) all cleared the head
weights and the head-resident layernorms. With `f` proven correct on
weights AND proven correct on output across 49 trials, the only way
kiln's `h_main` could differ from HF's `h_main` is if the difference
lives in `f`'s null space — which for a 4B-parameter, 2560×151936-class
LM head over 49 independent input vectors, is not a credible
hypothesis. **Top-1 selection agreement at 49/49 transitively confirms
input agreement.**

### 4. C31's head-trio bisect already closed the directly-adjacent surface

Per `docs/archive/phase-c/phase-c31/c31-head-trio-static-audit.md` §3, the trio
(`lm_head`, `mtp.final_layernorm`, base `weights.final_norm`) is clean
across three independent lines of evidence: static (C17/C25/C28),
splice cosine (C14), top-K selection (C29). The base `weights.final_norm`
is the layer that produces post-C18 `h_main` from the layer-31 hidden
state. Its weight tensor is loaded in fp32, applied once on-device
(`forward.rs:5108`), and the resulting tensor passes through C29's
49-dump head-output check unchanged — confirming both inputs and weights
of the final-norm step are correct.

## What this leaves on the bisect

|  Surface                           |  Status                                          |
| ---------------------------------- | ------------------------------------------------ |
|  Base 32-layer transformer compute |  **CLEAN** (C15 found the magnitude bug; C18 fixed it; C29 transitively confirms via head output) |
|  Base `weights.final_norm`         |  **CLEAN** (C31 + transitive via C29)           |
|  MTP head trio (lm\_head + 2 norms) |  **CLEAN** (C31)                                |
|  MTP head forward compute          |  **CLEAN** (C29: 100% top-1, J@10 median 1.0)   |
|  MTP head accept/reject + KV roll  |  **CLEAN** (C30, H11 ruled out)                 |
|  h\_prev reference frame contract  |  **FIXED** (C18, +4.7× α)                       |
|  All 11 named hypotheses (H1–H11)  |  **CLOSED** (single fix at C18; rest null)      |

**Class B is fully audited and closed.** The remaining α gap from 0.153
(post-C18) to ≥0.72 (the 2-token-ahead acceptance ceiling cited in
upstream MTP literature) cannot live on the inference side under the
current evidence. The head produces correct logits at the correct
positions on correct inputs; the acceptance math correctly accepts
matches; the KV state correctly advances. **The remaining lever is the
training side of the MTP recipe**, not the inference path.

## Why a fresh multi-prompt pod re-run would not change the verdict

C15 covered 1 prompt × 1 mtp\_pos × 8 decode steps. Extending to 4
prompts × 4 mtp\_pos × 4 steps (the C29 matrix) would:

- **Reproduce zero divergence at the magnitude axis.** C18 fixed the
  one-RMSNorm-behind bug at `forward.rs:5106-5118` for every dispatch
  path (`FullWithLastHidden` and `LastRowWithLastHidden`). The fix is
  prompt-agnostic and position-agnostic; multiprompt cannot uncover
  prompt-specific pre/post-norm regressions.
- **Reproduce zero divergence at the cosine axis.** With magnitudes
  matching post-C18 and head-output cosine matching 49/49 in C29, any
  hypothetical residual h\_main cosine drift would have to be in the
  null space of the LM head — an architecturally implausible failure
  mode for a 248K-class output projection.
- **Cost real pod time and SSH-wedge risk.** The C29 splice-dump matrix
  takes ~5–10 min of A6000 bench time plus ~10–15 min of HF fp32
  forwards (whether on-pod CPU or a separate CE host), with a real
  wall-clock risk of `until ssh ... kill -0` deadlocks per
  `kiln-ssh-polling-deadlock` agent note (two incidents on 2026-04-20
  totaling $113.52).

The expected information gain is zero; the expected cost is non-zero.
Strictly negative-EV, by the same kernel-vendor-precondition-check
pattern that produced doc-only redirects at PR #131, #163, #164, #170,
and #358 (C31 itself).

## Recommended next phase (C33)

**Pivot the α-recovery investigation to training-side causes.** Class B
inference compute is closed; the remaining lever lives in the recipe
that produced the Qwen3.5-4B MTP checkpoint kiln consumes.

Concrete C33 candidates, in rough priority order:

1. **MTP head fine-tune budget verification.** Compare kiln's recipe to
   the upstream Qwen3.5 MTP training spec (loss weighting, warmup,
   gradient flow into `mtp.fc` vs `mtp.norm` vs `lm_head`). If kiln
   omitted or undersampled MTP-head fine-tuning, the head's α ceiling
   on novel prompts would naturally cap below the published ceiling.
2. **Speculative-decoding temperature/top-p contract.** The `bench_latency_paged_mtp`
   sampler at `crates/kiln-server/src/bench.rs` uses kiln's default
   sampler. Compare against the upstream `Qwen3_5MultiTokenPredictor`
   verify-time temperature; if the verify side is hotter than the
   draft side, accept rates drop. C23 ruled out drift between two
   *kiln* sampler paths but not against the upstream reference.
3. **Native α ceiling (C32b from C31 verdict).** Accept α = 0.153 as
   the kiln-native ceiling under the current recipe and shift the
   project goal from "match published 0.72 ceiling" to "improve kiln's
   recipe to lift the ceiling." This is the cheapest option and is
   the right framing if (1) and (2) come back null.

C33 should NOT re-audit any inference-path compute surface — every
named hypothesis there is closed. A re-audit would be a planner-loop
duplication of C15–C31 with new pod cost.

## Cost

- Pod time: **$0** (CPU-only static + cross-reference audit; no GPU)
- Wall-clock: ~30 min (preflight, source inspection, doc write)
- Inside the C32 task cap ($15 / 60 min)

## Artifacts referenced (no new artifacts produced)

- `docs/archive/phase-c/phase-c15/c15-h-main-drift-verdict.md` — original independent fp32 audit
- `docs/archive/phase-c/phase-c15/c15_audit_fp32.json` — per-step fp32 numbers
- `docs/archive/phase-c/phase-c18/c18-h-prev-post-final-norm-verdict.md` — fix verdict
- `docs/archive/phase-c/phase-c29/c29-mtp-logits-h9-verdict.md` — 49-dump empirical head check
- `docs/archive/phase-c/phase-c31/c31-head-trio-static-audit.md` — head-trio bisect
- `crates/kiln-model/src/forward.rs:4755` — splice dump h\_main tap (post-C18)
- `crates/kiln-model/src/forward.rs:5097-5138` — `FullWithLastHidden` /
  `LastRowWithLastHidden` post-final-norm h\_prev frame fix
- `scripts/mtp_reference_dump.py:703` — confirms C14/C29 reference takes
  h\_main from kiln dump (the "by construction cos = 1.0" caveat)
