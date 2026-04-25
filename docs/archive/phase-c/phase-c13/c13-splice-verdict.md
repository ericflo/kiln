# Phase C13 — MTP Splice Verdict

**Bench**: Qwen3.5-4B BF16, paged, `KILN_SPEC_METHOD=mtp`, H100 NVL, 512 prompt × 64 decode, seed 42.
**Captures**: `KILN_MTP_DUMP_SPLICE=1 KILN_MTP_DUMP_SPLICE_POS="0,2" KILN_MTP_DUMP_SPLICE_MAX_STEPS=8` → 16 kiln dumps (2 positions × 8 steps).
**Reference**: `scripts/c13_hf_reference_dump.py` — pure-PyTorch fp32 re-forward of the 5 pre-RoPE MTP taps plus `h_main` for every kiln dump. 16/16 reference dumps produced, 0 errors, 0 missing taps.
**Thresholds**: cos ≥ 0.999, max|Δ| ≤ 1e-2.
**Bench α**: 0.033 (matches prior C12 primary-negative baseline; C13 pass is purely diagnostic).

## Per-tap medians (across 16 dumps)

| Tap               | median cos_sim | min cos_sim | median max\|Δ\| | max max\|Δ\| | flagged |
|-------------------|---------------:|------------:|----------------:|-------------:|--------:|
| `h_main`          | 1.0000000      | 1.0000000   | 0.000e+00       | 0.000e+00    | 0       |
| `c6__token_emb`   | 1.0000000      | 1.0000000   | 0.000e+00       | 0.000e+00    | 0       |
| `c6__norm_emb`    | 0.9999986      | 0.9999985   | 7.42e-03        | 1.52e-02     | 2       |
| `c6__norm_h`      | 0.9999987      | 0.9999984   | 2.33e-02        | 3.02e-02     | 16      |
| `c6__concat`      | 0.9999986      | 0.9999984   | 2.33e-02        | 3.02e-02     | 16      |
| `c6__fused`       | 0.9999948      | 0.9999928   | 8.69e-03        | 2.32e-02     | 7       |

`flagged` = count of (pos, step) sites where cos < 0.999 **or** max|Δ| > 1e-2. Every flagged site stays well inside cos ≥ 0.9999928; the only way the site gets flagged is via the 1e-2 max|Δ| ceiling.

## Per-position breakdown

| pos | tap | n | median cos | min cos | median max\|Δ\| | max max\|Δ\| |
|-----|-----|--:|-----------:|--------:|----------------:|-------------:|
| 0 | `h_main`        | 8 | 1.0000000 | 1.0000000 | 0.00e+00 | 0.00e+00 |
| 0 | `c6__token_emb` | 8 | 1.0000000 | 1.0000000 | 0.00e+00 | 0.00e+00 |
| 0 | `c6__norm_emb`  | 8 | 0.9999986 | 0.9999986 | 7.48e-03 | 7.78e-03 |
| 0 | `c6__norm_h`    | 8 | 0.9999987 | 0.9999985 | 2.30e-02 | 3.02e-02 |
| 0 | `c6__concat`    | 8 | 0.9999986 | 0.9999985 | 2.30e-02 | 3.02e-02 |
| 0 | `c6__fused`     | 8 | 0.9999949 | 0.9999929 | 7.50e-03 | 1.51e-02 |
| 2 | `h_main`        | 8 | 1.0000000 | 1.0000000 | 0.00e+00 | 0.00e+00 |
| 2 | `c6__token_emb` | 8 | 1.0000000 | 1.0000000 | 0.00e+00 | 0.00e+00 |
| 2 | `c6__norm_emb`  | 8 | 0.9999986 | 0.9999985 | 7.31e-03 | 1.52e-02 |
| 2 | `c6__norm_h`    | 8 | 0.9999987 | 0.9999984 | 2.64e-02 | 3.00e-02 |
| 2 | `c6__concat`    | 8 | 0.9999987 | 0.9999984 | 2.64e-02 | 3.00e-02 |
| 2 | `c6__fused`     | 8 | 0.9999948 | 0.9999928 | 1.23e-02 | 2.32e-02 |

## Verdict — SPLICE INPUTS CLEAR

The pre-projection splice — i.e. the tensors feeding the MTP `fc` head — **agrees with an fp32 HF reference to cos ≥ 0.9999928 on every tap at every step for both targeted positions**. The only flags come from the 1e-2 max|Δ| threshold being below the natural bf16 activation noise floor on tensors with magnitude O(1–10): every single flagged entry still has cosine similarity that would round to 1.0 at fp16 precision.

- `h_main` and `c6__token_emb` match **bit-exactly** (cos = 1.0, max|Δ| = 0) across all 16 sites. This is the expected sanity identity: both the source-side h_main and the draft-token embedding are reproduced verbatim from the kiln dump when the HF reference re-forwards, so any divergence there would point to a dumper bug rather than a splice bug. Clean.
- `c6__norm_emb`, `c6__norm_h`, `c6__concat`, `c6__fused` track the RMSNorm and `fc` projection outputs — the places where the hypothesis "wrong weight bound to wrong half" would show up as a systemic rotation or sign flip. We observe neither. The deltas are isotropic bf16 rounding, symmetric across positions 0 and 2, and identical in character between `norm_emb` and `norm_h` (which is what you expect when the two RMSNorm weights are correctly paired: the delta distribution is a property of the bf16 activation precision, not of a weight-binding flip).
- `c6__concat` max|Δ| equals `c6__norm_h` max|Δ| at every step, confirming concat is a trivial interleave of the two halves — no stale-buffer or stride bug.
- `c6__fused` is slightly noisier (matmul accumulation over H=2560), but still cos ≥ 0.9999928. A systemic mis-weighting of either `pre_fc_norm_embed` or `pre_fc_norm_hidden` against the wrong half would drop `c6__fused` cos well below 0.99 — the Phase B2 `KILN_MTP_SWAP_FC_NORMS=1` probe is the canonical counter-example. We don't see that signature.

**Therefore the α≈0.033 regression reproduced in Phase C12 is NOT caused by a pre-projection splice/weight-binding bug.** The splice is wired correctly; both halves of the `fc` input are produced in the expected order and with the expected numerical fidelity; `h_main` (the source-stack hidden state) enters the MTP head cleanly.

## What this rules out

- "Weights are loaded, but paired to the wrong half of the `fc` input" (Phase B2 secondary-hypothesis A/B via `KILN_MTP_SWAP_FC_NORMS`).
- "h_main is produced from a stale KV / wrong position" (would show `h_main` cos ≪ 1 — doesn't).
- "Draft-token embedding uses a different tied-lm_head/embed_tokens tensor than HF" (would show `c6__token_emb` cos ≪ 1 — doesn't).
- "`concat` is accidentally using a different stride / interleave order than HF" (`c6__concat` matches `c6__norm_h` max|Δ| bit-for-bit).
- Any systemic bias in the `fc` projection bigger than bf16 matmul rounding (cos would drop to the 0.99x region — doesn't).

## What this does NOT rule out — next investigation axes

- **Post-`fc` MTP transformer block drift**: taps beyond `fc_output` (`pre_layer`, `post_layer`, `post_final_ln`, `mtp_logits`) are captured by the B6/B7 legacy latch but are outside the splice window. An α≈0.033 with clean splice inputs points the investigation at the MTP transformer block itself or at the lm_head tie.
- **Base-stack hidden-state drift in long sequences**: we match the kiln h_main against an HF fp32 re-forward that starts *from the same kiln h_main* (the reference sidecar takes h_main and draft_token_id from the kiln dump rather than re-running the whole base model). So if the bug is in the base-stack producing a wrong h_main from a drifting KV state, C13 cannot see it — the splice test is conditional on h_main being correct. The `h_main` row showing `cos = 1.0` is tautological here and should not be read as "the base model is fine."
- **Acceptance math / sampler / KV-rollback**: α is determined by the MTP distribution *and* by which draft tokens the sampler emits, and the Phase C12 fp32-draft-head probe already showed that forcing the head itself to fp32 doesn't recover α. C13 confirms splice inputs don't. So the remaining candidates are: (a) the MTP transformer block itself under bf16, (b) the tied lm_head output normalization, (c) KV-cache advance / rollback under speculative decode.

## Recommended follow-up

1. Extend the splice dump through the MTP transformer block output (`post_layer`) and final norm (`post_final_ln`) with an HF fp32 reference. The B6/B7 outer taps already collect these; a sibling `scripts/c13_hf_reference_dump.py` mode that also re-forwards the MTP transformer block (not just the `fc` projection) would be the next cheap cut.
2. Add a base-model h_main comparison at the same (pos, step) slots. The kiln dump preserves the prompt tokens; a companion fp32 re-forward of the *base* model on those tokens is slow but tractable (~5 min CPU per step). A mismatch there moves the investigation onto the base-stack / GDN / GQA.
3. Leave `KILN_MTP_DUMP_SPLICE` in place as-is; it's cheap, composable, and correctly wired.

## Artifacts

- Kiln splice captures: 16 × `.safetensors` at `/workspace/captures/mtp_pos-{0,2}/step-{0..7}.safetensors` on pod.
- HF reference: 16 × `.safetensors` at `/workspace/captures-ref/mtp_pos-{0,2}/step-{0..7}.safetensors` on pod.
- Summary JSON: `docs/archive/phase-c/phase-c13/c13-summary.json` (this commit).
- Bench run log: `/tmp/bench-splice2.log` on pod.
- Reference sidecar log: `/tmp/c13-ref.log` on pod.
