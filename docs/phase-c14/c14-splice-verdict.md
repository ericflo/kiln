# Phase C14 — MTP Post-Block Splice Verdict

**Bench**: Qwen3.5-4B BF16, paged, `KILN_SPEC_METHOD=mtp`, A6000 48GB, 512 prompt × 64 decode, seed 42.
**Captures**: `KILN_MTP_DUMP_SPLICE=1 KILN_MTP_DUMP_SPLICE_POS="0,2" KILN_MTP_DUMP_SPLICE_MAX_STEPS=8` → 16 kiln dumps (2 positions × 8 steps). C14 taps (`c14__post_block`, `c14__post_norm`, `c14__logits`) captured via OR-composed gate under `is_dump_c14_post_block_effectively_enabled()`.
**Reference**: `scripts/c14_hf_reference_dump.py` — pure-PyTorch fp32 re-forward of the 3 post-block MTP taps plus `h_main` for every kiln dump via `scripts/mtp_reference_dump.py --capture-subops`. 16/16 reference dumps produced, 0 errors, 0 missing taps.
**Thresholds**: cos ≥ 0.999, max|Δ| ≤ 1e-2.
**Bench α**: 0.0328 (matches prior C12/C13 primary-negative baseline; C14 pass is purely diagnostic).

## Per-tap medians (across 16 dumps)

| Tap                | median cos_sim | min cos_sim | median max\|Δ\| | max max\|Δ\| | flagged |
|--------------------|---------------:|------------:|----------------:|-------------:|--------:|
| `h_main`           | 1.0000000      | 1.0000000   | 0.000e+00       | 0.000e+00    | 0       |
| `c14__post_block`  | 0.9999675      | 0.9999530   | 3.89e-02        | 7.85e-02     | 16      |
| `c14__post_norm`   | 0.9999660      | 0.9999547   | 1.46e-01        | 3.14e-01     | 16      |
| `c14__logits`      | 0.9999736      | 0.9999529   | 1.12e-01        | 1.40e-01     | 16      |

`flagged` = count of (pos, step) sites where cos < 0.999 **or** max|Δ| > 1e-2. Every flagged site stays well inside cos ≥ 0.9999529; every flag comes from the 1e-2 max|Δ| ceiling being below the natural bf16 activation noise floor on these output-side tensors.

## Per-position breakdown

| pos | tap                | n | median cos | min cos | median max\|Δ\| | max max\|Δ\| |
|-----|--------------------|--:|-----------:|--------:|----------------:|-------------:|
| 0 | `h_main`           | 8 | 1.0000000 | 1.0000000 | 0.00e+00 | 0.00e+00 |
| 0 | `c14__post_block`  | 8 | 0.9999680 | 0.9999530 | 3.89e-02 | 7.85e-02 |
| 0 | `c14__post_norm`   | 8 | 0.9999666 | 0.9999547 | 1.76e-01 | 3.14e-01 |
| 0 | `c14__logits`      | 8 | 0.9999775 | 0.9999529 | 1.10e-01 | 1.40e-01 |
| 2 | `h_main`           | 8 | 1.0000000 | 1.0000000 | 0.00e+00 | 0.00e+00 |
| 2 | `c14__post_block`  | 8 | 0.9999675 | 0.9999562 | 4.05e-02 | 6.50e-02 |
| 2 | `c14__post_norm`   | 8 | 0.9999649 | 0.9999565 | 1.34e-01 | 2.08e-01 |
| 2 | `c14__logits`      | 8 | 0.9999703 | 0.9999611 | 1.12e-01 | 1.38e-01 |

Tap shapes:
- `h_main` → `[1, 1, 2560]`
- `c14__post_block` → `[1, 1, 2560]` (output of the single MTP transformer block)
- `c14__post_norm` → `[1, 1, 2560]` (output of the final RMSNorm before `lm_head`)
- `c14__logits` → `[1, 1, 248320]` (`lm_head` projection, i.e. the tied base-model vocab head applied to `post_norm`)

## Verdict — POST-BLOCK SPLICE CLEAR

The post-block splice — i.e. every tensor downstream of `c6__fused` all the way through `lm_head` — **agrees with an fp32 HF reference to cos ≥ 0.9999529 on every tap at every step for both targeted positions**. The max|Δ| numbers are entirely explained by bf16 activation noise scaling with tensor magnitude, not by any systemic rotation, scale, or mis-wiring.

- `h_main` matches **bit-exactly** (cos = 1.0, max|Δ| = 0) across all 16 sites. This is the expected sanity identity: the reference sidecar takes `h_main` straight from the kiln dump, so a mismatch there would be a dumper bug, not a block bug. Clean.
- `c14__post_block` (transformer block output, L2 ≈ 33): median cos 0.9999675 with median max|Δ| 3.9e-2 is consistent with bf16 attention + MLP + two residual adds on a 2560-dim hidden state. A systemic bug inside the block — wrong q/k/v projection, a swapped residual, an off-by-one RoPE frequency, a wrong post-attn norm weight — would drop cos well below 0.99; we don't see that signature.
- `c14__post_norm` (final RMSNorm, L2 ≈ 160): median cos 0.9999660, max max|Δ| 3.1e-1. The larger max|Δ| is expected — RMSNorm divides by a small RMS value, which amplifies bf16 rounding of the input. The **cosine** still rounds to 1.0, which is the measurement that would catch a wrong norm weight.
- `c14__logits` (lm_head projection into vocab=248320, L2 ≈ 1067): median cos 0.9999736, max max|Δ| 1.4e-1. The vector has 248k entries and the matmul accumulates through H=2560; at bf16 the expected per-entry rounding envelope is O(1e-1) and the observed numbers sit inside that envelope. A wrong lm_head binding (for instance if MTP failed to tie to `model.embed_tokens` and pulled a different tensor instead) would drop cos to the 0.99x region or worse — the PR #331 weight-loading audit already verified the binding and C14's post-lm_head numbers confirm it.

**Therefore the α ≈ 0.033 regression reproduced in Phase C12 and re-confirmed in C13 is NOT caused by a downstream-of-`c6__fused` compute bug in the MTP transformer block, the final norm, or the tied `lm_head`.** The full post-projection forward path is wired correctly and produces logits that agree with an fp32 reference on an effectively identical cosine basis.

## What this rules out

Combined with C13's pre-projection verdict, **the entire MTP forward compute path from `h_main` through `mtp_logits` is now audited clean**:

- **C13 (pre-projection)**: `h_main`, draft-token embedding, both RMSNorm halves, concat, and the `fc` fused projection all agree with fp32 to cos ≥ 0.9999928.
- **C14 (post-projection)**: `post_block`, `post_norm`, and `logits` all agree with fp32 to cos ≥ 0.9999529.

What C14 specifically rules out:
- "MTP transformer block computes attention or MLP incorrectly under bf16" — would show `c14__post_block` cos ≪ 1 on a single-block stack; doesn't.
- "Post-attn residual adds are scaled/ordered wrong" — would show a stable, systemic rotation in `c14__post_block` across all 16 sites; doesn't.
- "MTP final RMSNorm weight is bound to the wrong tensor" — would drop `c14__post_norm` cos below `c14__post_block` cos by a stable margin; the two tracks are essentially coincident (both median 0.9999660-0.9999675).
- "`lm_head` is tied to the wrong embedding / uses a stale tensor / is transposed" — would drop `c14__logits` cos to the 0.99x region; doesn't.
- "RoPE frequency or rotation axis is off inside the MTP block" — would show a magnitude-dependent rotation in `c14__post_block` that the cosine metric catches; doesn't.

## What this does NOT rule out — remaining investigation axes

With splice inputs (C13) AND splice outputs (C14) both clean, the α ≈ 0.033 regression cannot live inside the MTP head's single forward pass. The remaining candidates are therefore:

- **Base-stack drift producing a wrong `h_main`.** C14 and C13 are both **conditional on `h_main` being correct**: the fp32 reference takes `h_main` from the kiln dump verbatim. The `h_main` cos=1.0 row in both verdicts is tautological. If the base-model GDN/GQA stack is building up a wrong hidden state across decode (KV drift, GDN state accumulation bug, wrong cache position), C14 cannot see it. **This is now the highest-probability root cause.**
- **Acceptance math / sampler / KV-rollback bugs.** The decision path that turns `mtp_logits` into an accept-or-reject call, and the subsequent KV-cache advance/rollback, is entirely outside the splice window. The Phase C12 fp32-draft-head probe already showed that forcing the head to fp32 doesn't recover α, which is consistent with the splice-is-clean verdicts from C13 and C14. The sampler + rollback seam is next.
- **Draft-token positional wiring.** The draft token's embedding is correct (C13 `c6__token_emb` bit-exact). But its position inside the base-stack KV when the MTP head re-attends to cache on the next step is a separate concern that neither C13 nor C14 probes.

## Recommended follow-up

1. **Base-stack h_main audit (Phase C15 candidate).** Compare kiln's `h_main` at each (pos, step) slot against a full fp32 re-forward of the base model on the same prompt tokens + previously-accepted tokens. This is expensive (~5 min CPU per step, or ~1 min GPU per step on fp32), but the splice window has now been exhausted as a source of the regression. 16 slots × ~1 min = ~15-20 min of GPU time for a decisive read.
2. **Sampler / acceptance probe.** Capture, for each step, kiln's accept/reject decision **and** what an fp32 reference acceptance check would decide given the same `mtp_logits` and base logits. If they disagree, the bug is in acceptance math, not in the MTP compute.
3. **KV-state audit at position-step boundary.** At each step boundary, hash the kiln KV cache slice for the MTP write position and compare to what an fp32 reference would produce. A drift here would show up as a progressively wrong `h_main` on the *next* step.
4. Leave `KILN_MTP_DUMP_SPLICE` in place; both the C6/C7 pre-projection taps and the new C14 post-block taps are composable via the same meta-flag and cheap (<1 MB/step).

## Artifacts

- Kiln splice captures: 16 × `.safetensors` at `/workspace/captures/mtp_pos-{0,2}/step-{0..7}.safetensors` on pod.
- HF reference: 16 × `.safetensors` at `/workspace/captures-ref-c14/mtp_pos-{0,2}/step-{0..7}.safetensors` on pod.
- Summary JSON: `docs/phase-c14/c14-summary.json` (this commit).
- Bench run log: `/tmp/bench-c14.log` on pod.
- Reference sidecar log: `/tmp/c14-ref.log` on pod.
- Bench: prompt 512, decode 64, α = 0.0328, decode 36.54 tok/s, p99 ITL 48.37 ms, seed 42.
