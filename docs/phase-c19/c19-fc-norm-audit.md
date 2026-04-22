# Phase C19 — `mtp.fc_norm` audit (hypothesis 1, doc-only redirect)

## Verdict

**Hypothesis 1 is DISPROVEN. No code change. No pod spend.** Qwen3.5-4B's
published checkpoint does not ship a single `mtp.fc_norm.weight` tensor; it
ships two distinct norms — `mtp.pre_fc_norm_embedding.weight` and
`mtp.pre_fc_norm_hidden.weight` — and **kiln already loads both and already
applies them to the correct halves of the `fc` input** before the head's
`[2H → H]` projection. The C17 / C18 secondary hypothesis was a misreading.
Hand the residual α gap (median 0.1532, floor 0.5) to hypothesis 2: dual-norm
inversion **inside the MTP transformer block**.

## C18 hypothesis as stated

From `docs/phase-c18/c18-h-prev-post-norm-fix.md` (Phase C19 handoff, item 1):

> **`mtp.fc_norm` vs. `model.norm` reuse.** HF Transformers instantiates a
> *separate* RMSNorm (`fc_norm`) inside the MTP head and applies it to
> `h_main` before the `fc` projection. Kiln currently reuses the main
> final-norm weights for both. Swap to the dedicated `fc_norm` tensors from
> the checkpoint and re-bench α.

Two factual claims:

1. **HF claim:** Qwen3.5-4B's MTP head contains a single `fc_norm` RMSNorm
   that normalizes `h_main` before `fc`.
2. **Kiln claim:** Kiln reuses `model.norm` weights (the main-stack
   final-norm) where it should be using `fc_norm`.

Both claims are false against the current code and checkpoint, as evidenced
below.

## Evidence

### 1. Direct enumeration of the `Qwen/Qwen3.5-4B` safetensors index

CPU-only `huggingface_hub` + `safetensors` enumeration of every key in the
published checkpoint. The full set of `mtp.*` tensors is:

```
mtp.fc.weight
mtp.layers.0.input_layernorm.weight
mtp.layers.0.mlp.down_proj.weight
mtp.layers.0.mlp.gate_proj.weight
mtp.layers.0.mlp.up_proj.weight
mtp.layers.0.post_attention_layernorm.weight
mtp.layers.0.self_attn.k_norm.weight
mtp.layers.0.self_attn.k_proj.weight
mtp.layers.0.self_attn.o_proj.weight
mtp.layers.0.self_attn.q_norm.weight
mtp.layers.0.self_attn.q_proj.weight
mtp.layers.0.self_attn.v_proj.weight
mtp.norm.weight
mtp.pre_fc_norm_embedding.weight
mtp.pre_fc_norm_hidden.weight
```

Exactly 15 tensors. **There is no `mtp.fc_norm.weight`.** The only
`fc_norm`-named tensors are `mtp.pre_fc_norm_embedding.weight` and
`mtp.pre_fc_norm_hidden.weight`, both `[hidden]` = `[2560]`.

This matches the C13 weight-loading audit
([`docs/phase-c13/mtp-weight-loading-audit.md`](../phase-c13/mtp-weight-loading-audit.md))
table, which already enumerated all 15 MTP tensors and validated their
shapes.

### 2. Kiln loader covers both norms

`load_mtp_if_present` ([`crates/kiln-model/src/loader.rs:612-755`](../../crates/kiln-model/src/loader.rs))
extracts both pre-fc norms and stores them as distinct `MtpWeights` fields:

```rust
// loader.rs:636-654
let pre_fc_norm_embedding = extract_tensor(
    tensor_map,
    &format!("{mtp_prefix}pre_fc_norm_embedding.weight"),
)?;
validate_shape(&pre_fc_norm_embedding, &[config.hidden_size], &ctx("pre_fc_norm_embedding"))?;

let pre_fc_norm_hidden = extract_tensor(
    tensor_map,
    &format!("{mtp_prefix}pre_fc_norm_hidden.weight"),
)?;
validate_shape(&pre_fc_norm_hidden, &[config.hidden_size], &ctx("pre_fc_norm_hidden"))?;
```

The loader docstring at lines 598-604 explicitly enumerates the two
`pre_fc_norm_*` tensors as part of the loaded set. There is no fallback
to `weights.final_norm` and no silent substitution: if either tensor is
missing in the checkpoint, `extract_tensor` errors out and load fails loudly.

### 3. Kiln forward applies both norms to the correct halves

`mtp_forward_step` ([`crates/kiln-model/src/forward.rs:4474-4489`](../../crates/kiln-model/src/forward.rs))
applies the two norms to the embedding half and the hidden half before
concatenation:

```rust
// forward.rs:4474-4489
let swap_fc_norms = crate::mtp_debug::is_swap_fc_norms_enabled();
let (norm_emb_weight, norm_h_weight) = if swap_fc_norms {
    (&mtp.pre_fc_norm_hidden, &mtp.pre_fc_norm_embedding)
} else {
    (&mtp.pre_fc_norm_embedding, &mtp.pre_fc_norm_hidden)
};
let norm_emb = rms_norm(&token_emb, norm_emb_weight, config.rms_norm_eps)?;
let norm_h = rms_norm(h_prev, norm_h_weight, config.rms_norm_eps)?;
// concat + fc at lines 4498-4525
```

Production path uses the unswapped pairing: `pre_fc_norm_embedding → token_emb`,
`pre_fc_norm_hidden → h_prev`. `weights.final_norm` is not referenced
anywhere in `mtp_forward_step`. It is only used (correctly, on the main stack)
inside `model_forward_paged` to produce the post-final-norm `h_prev` that
`mtp_forward_step` then receives.

### 4. HF Python reference confirms the dual-norm pattern

The reference dump at [`scripts/mtp_reference_dump.py`](../../scripts/mtp_reference_dump.py)
documents HF Transformers' Qwen3-Next MTP forward as a 7-step routine with
**two pre-fc norms applied to two halves**, identical to kiln's structure.
There is no single `fc_norm` step in the reference either; the reference and
kiln match 1:1.

### 5. Phase B3 multi-prompt swap A/B already tested the swap

`KILN_MTP_SWAP_FC_NORMS=1` exists precisely to A/B-test the secondary
hypothesis "the loader paired the two `[H]` norm vectors to the wrong halves
of the `fc` input" (forward.rs:4467-4472 comment block). Phase B3 ran this
swap and α was statistically unchanged, disproving the within-pair swap as
well. Combined with the no-substitution evidence above, the entire
fc-pre-norm region is exonerated.

## Why this hypothesis was tempting

The "single `fc_norm` before `fc`" pattern *does* exist in some other
multi-token-prediction architectures (e.g. DeepSeek-style MoE MTP heads
publish a single `fc_norm`). Qwen3.5-4B's MTP head splits this into the
two-half formulation and Phase C18's handoff text appears to have crossed
wires with the alternate convention. The checkpoint enumeration above is the
authoritative refutation.

## Hypothesis 2 — the next target

Carrying the C18 handoff item 2 forward as the new head of the queue:

> **Dual-norm inversion inside the MTP block.** The MTP decoder block has
> its own pre-attn / pre-MLP RMSNorms. If kiln's `MtpBlock` currently applies
> these in the wrong order (or substitutes the main-model norms), token
> logits will be subtly miscalibrated but still on-manifold — exactly the
> α ≈ 0.15 shape.

Concrete tensors in scope:

- `mtp.layers.0.input_layernorm.weight` — pre-attention RMSNorm
- `mtp.layers.0.post_attention_layernorm.weight` — pre-MLP RMSNorm
- `mtp.layers.0.self_attn.q_norm.weight` — per-head Q-norm
- `mtp.layers.0.self_attn.k_norm.weight` — per-head K-norm

C13 already verified these load (`docs/phase-c13/mtp-weight-loading-audit.md`,
table rows 5, 6, 11, 12). The remaining audit is **runtime ordering and
binding** inside `transformer_block_paged` (called from `mtp_forward_step`
around forward.rs:4541+) — i.e. confirm the block consumes the four MTP
norms (not the equivalent main-stack norms loaded under `model.layers.N.*`)
in the same call positions HF's `Qwen3NextMtpDecoderLayer.forward` does.

### Suggested next-step (CPU-only, no pod)

1. **Targeted forward read.** Trace the `transformer_block_paged` call in
   `mtp_forward_step` and confirm:
    - `mtp.layer.norms.input` (= checkpoint
      `mtp.layers.0.input_layernorm.weight`) is applied immediately before
      attention.
    - `mtp.layer.norms.post_attention` (= checkpoint
      `mtp.layers.0.post_attention_layernorm.weight`) is applied immediately
      before MLP.
    - `q_norm` / `k_norm` are sourced from `mtp.layer.attention.Full.{q,k}_norm`
      and not from any main-layer slot.
2. **Retarget the existing parity harness.** `scripts/mtp_compare.py` +
   `scripts/mtp_h_main_reference_dump.py` already bracket the inner-MTP
   splice (Phase B6). Repoint the comparison taps at the four norms above
   and dump per-tensor deltas vs. HF's `Qwen3NextMtpDecoderLayer.forward`.
   Inversion or main/MTP norm crossover will show as a non-trivial delta at
   exactly one tap.
3. **If parity diverges:** ship the wiring fix and re-run the C18
   α-recovery bench (3 seeds, 512 prompt × 128 decode, A6000,
   `KILN_SPEC_METHOD=mtp`, `KILN_W4A16=1`).
4. **If parity matches:** disprove H2 with a Phase C20 doc-only PR and
   advance to H3 (rotary `mtp_pos` offset).

The CPU steps cost no pod time. Only step 3 — once an actual fix is in hand
— justifies acquiring an A6000.

## Cost / budget

- Wall-clock: ~10 min (no pod, CPU enumeration only).
- Pod cost: $0. Hypothesis was disprovable from the checkpoint index alone
  plus the existing C13 audit and `mtp_forward_step` source.
- Well under the 120 min / $50 task cap.
