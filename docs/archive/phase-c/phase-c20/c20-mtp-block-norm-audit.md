# Phase C20 — MTP-block dual-norm wiring audit (hypothesis 2, doc-only redirect)

## Verdict

**Hypothesis 2 is DISPROVEN. No code change. No pod spend.** Kiln's MTP
transformer block already binds the four MTP-specific RMSNorm tensors —
`mtp.layers.0.input_layernorm.weight`,
`mtp.layers.0.post_attention_layernorm.weight`,
`mtp.layers.0.self_attn.q_norm.weight`, and
`mtp.layers.0.self_attn.k_norm.weight` — to the call positions HF's
`Qwen3NextMtpDecoderLayer.forward` uses. The loader path makes
main-stack substitution structurally impossible, and the runtime path
applies the four norms in the canonical pre-attn → qk → pre-MLP
ordering that matches the HF reference. Hand off the residual α gap
(C18 median 0.1532, floor 0.5) to hypothesis 3: rotary `mtp_pos`
offset.

## C18 hypothesis as stated

From [`docs/archive/phase-c/phase-c18/c18-h-prev-post-norm-fix.md`](../phase-c18/c18-h-prev-post-norm-fix.md)
(Phase C19 handoff, item 2), forwarded by C19 as the new head of queue:

> **Dual-norm inversion inside the MTP block.** The MTP decoder block
> has its own pre-attn / pre-MLP RMSNorms. If kiln's `MtpBlock`
> currently applies these in the wrong order (or substitutes the
> main-model norms), token logits will be subtly miscalibrated but
> still on-manifold — exactly the α ≈ 0.15 shape.

Two factual claims to test:

1. **Substitution claim:** kiln's MTP block silently substitutes
   main-stack norms (`model.layers.N.*`) for the MTP-specific norms
   (`mtp.layers.0.*`).
2. **Ordering claim:** the four norms are applied in the wrong sequence
   relative to attention / MLP / RoPE versus the HF reference.

Both claims are false against the current code, as evidenced below.

## Tensors in scope

C13's weight-loading audit (`docs/archive/phase-c/phase-c13/mtp-weight-loading-audit.md`,
table rows 5, 6, 11, 12) already confirmed all four tensors load and
shape-validate. Their roles in the HF `Qwen3NextMtpDecoderLayer.forward`
sequence:

| Checkpoint key | Shape | HF call position |
| --- | --- | --- |
| `mtp.layers.0.input_layernorm.weight` | `[hidden]` = `[2560]` | Pre-attention norm |
| `mtp.layers.0.post_attention_layernorm.weight` | `[hidden]` = `[2560]` | Pre-MLP norm |
| `mtp.layers.0.self_attn.q_norm.weight` | `[head_dim]` = `[256]` | Per-head Q-norm, post-projection / pre-RoPE |
| `mtp.layers.0.self_attn.k_norm.weight` | `[head_dim]` = `[256]` | Per-head K-norm, post-projection / pre-RoPE |

## Evidence

### 1. Loader path — substitution is structurally impossible

`load_mtp_if_present` ([`crates/kiln-model/src/loader.rs:684-685`](../../crates/kiln-model/src/loader.rs))
constructs the MTP layer prefix from the MTP root and dispatches into
the same `load_layer` helper used by every main-stack layer:

```rust
// loader.rs:684-685
let mtp_layer_prefix = format!("{mtp_prefix}layers.0.");
let layer = load_layer(tensor_map, &mtp_layer_prefix, 3, config)
    .context("mtp layer 0")?;
```

`mtp_prefix` is `"mtp."` (resolved earlier in the same function), so
`mtp_layer_prefix` evaluates to `"mtp.layers.0."`. `load_layer`
([`loader.rs:357-396`](../../crates/kiln-model/src/loader.rs))
then string-formats the four norm keys against the caller's prefix:

```rust
// loader.rs:363
let input_layernorm = extract_tensor(tensor_map,
    &format!("{prefix}input_layernorm.weight"))?;

// loader.rs:372
let post_attention_layernorm = extract_tensor(tensor_map,
    &format!("{prefix}post_attention_layernorm.weight"))?;

// loader.rs:383
AttentionWeights::Full(load_full_attention(tensor_map, prefix, ...)?)
```

`load_full_attention` ([`loader.rs:399-438`](../../crates/kiln-model/src/loader.rs))
extends the same prefix with `self_attn.`:

```rust
// loader.rs:424-428
let q_norm = extract_tensor(tensor_map, &format!("{attn}q_norm.weight"))?;
validate_shape(&q_norm, &[config.head_dim], &ctx("q_norm"))?;

let k_norm = extract_tensor(tensor_map, &format!("{attn}k_norm.weight"))?;
validate_shape(&k_norm, &[config.head_dim], &ctx("k_norm"))?;
```

The four resolved keys are therefore exactly:

- `mtp.layers.0.input_layernorm.weight`
- `mtp.layers.0.post_attention_layernorm.weight`
- `mtp.layers.0.self_attn.q_norm.weight`
- `mtp.layers.0.self_attn.k_norm.weight`

`extract_tensor` errors out with no fallback if any key is missing —
there is no silent main-stack substitution path. The same loader code
that builds main-layer `LayerWeights` from `model.layers.N.` builds
the MTP `LayerWeights` from `mtp.layers.0.`; the *only* difference is
the prefix string. Substitution would require either re-using the
main-stack `prefix` value (it doesn't), or mutating
`MtpWeights.layer.{input_layernorm,post_attention_layernorm,
attention.Full.{q,k}_norm}` after load (no such code path exists).

### 2. Runtime path — the MTP block consumes the MTP layer struct

`mtp_forward_step` ([`crates/kiln-model/src/forward.rs:4605-4622`](../../crates/kiln-model/src/forward.rs))
calls the same `transformer_block_paged` used by main-stack full-attn
layers, but passes `&mtp.layer` (the MTP-specific `GpuLayerWeights`)
as the layer argument:

```rust
// forward.rs:4605-4622
let mtp_hidden_result = transformer_block_paged(
    backend,
    &fused,
    &mtp.layer,                          // <- MTP layer, not main
    config,
    &positions,
    mtp_pos,
    config.num_attention_heads,
    config.num_kv_heads,
    config.head_dim,
    config.rotary_dim(),
    &weights.rotary_inv_freq,
    config.rms_norm_eps,
    mtp_cache,
    mtp_block_table,
    /* full_attn_layer_idx = */ 0,
    /* lora = */ None,
);
```

Inside `transformer_block_paged`
([`forward.rs:3770-3868`](../../crates/kiln-model/src/forward.rs))
the four norm taps reference fields of the layer struct passed in —
no main-stack lookup, no global state:

```rust
// forward.rs:3788-3795
let attn_weights = match &layer.attention {
    GpuAttentionWeights::Full(w) => w,
    GpuAttentionWeights::Linear(_) => {
        anyhow::bail!(
            "transformer_block_paged only supports full attention layers (not linear/GDN)"
        )
    }
};

// forward.rs:3798-3801   (pre-attention norm)
let normed = {
    kiln_nvtx::range!(c"kiln/norm/pre_attn");
    rms_norm(x, &layer.input_layernorm, rms_norm_eps)?
};

// forward.rs:3811-3828   (gqa_attention_paged receives MTP attn_weights)
let attn_out = gqa_attention_paged(
    backend, &normed, attn_weights, ...
)?;

// forward.rs:3838-3842   (post-attention / pre-MLP norm)
let normed = {
    kiln_nvtx::range!(c"kiln/norm/pre_mlp");
    rms_norm(&x, &layer.post_attention_layernorm, rms_norm_eps)?
};
```

`gqa_attention_paged` ([`forward.rs:3196-3201`](../../crates/kiln-model/src/forward.rs))
applies the per-head Q/K norms from the same `attn_weights` it was
handed:

```rust
// forward.rs:3196-3201   (post-projection, post-reshape, pre-RoPE)
let (q, k) = {
    kiln_nvtx::range!(c"kiln/attn/qk_norm");
    let q = rms_norm(&q, &attn_weights.q_norm, rms_norm_eps)?;
    let k = rms_norm(&k, &attn_weights.k_norm, rms_norm_eps)?;
    (q, k)
};
```

When `mtp_forward_step` is the caller, `attn_weights` is the
`GpuFullAttentionWeights` extracted at `forward.rs:3788-3795` from
`mtp.layer.attention.Full(w)` — i.e. the MTP-specific `q_norm` /
`k_norm` loaded at `loader.rs:424-428` against the `mtp.layers.0.`
prefix. Main-stack `model.layers.N.self_attn.{q,k}_norm` tensors are
never reachable from this call chain.

### 3. Ordering matches HF `Qwen3NextMtpDecoderLayer.forward`

The HF reference applies the four norms in this sequence (paraphrased
from `Qwen3NextMtpDecoderLayer.forward` and verified via
[`scripts/mtp_reference_dump.py`](../../scripts/mtp_reference_dump.py)):

```
hidden = input_layernorm(hidden)
q, k, v = q_proj(hidden), k_proj(hidden), v_proj(hidden)
q = q_norm(reshape_per_head(q))                     # pre-RoPE
k = k_norm(reshape_per_head(k))                     # pre-RoPE
q, k = rope(q, k, position_ids)
attn_out = attention(q, k, v, ...)
hidden = residual + o_proj(attn_out)
hidden_for_mlp = post_attention_layernorm(hidden)
hidden = hidden + down_proj(silu(gate_proj(...)) * up_proj(...))
```

Kiln's `transformer_block_paged` + `gqa_attention_paged` produces the
same sequence, in the same order, against the MTP-specific norm
tensors:

| HF step | kiln tap | kiln file:line |
| --- | --- | --- |
| `input_layernorm(hidden)` | `rms_norm(x, &layer.input_layernorm, ...)` | `forward.rs:3800` |
| Q/K projections | `q_proj` / `k_proj` linear (inside `gqa_attention_paged`) | `forward.rs:3096-3186` |
| `q_norm` (per-head, pre-RoPE) | `rms_norm(&q, &attn_weights.q_norm, ...)` | `forward.rs:3198` |
| `k_norm` (per-head, pre-RoPE) | `rms_norm(&k, &attn_weights.k_norm, ...)` | `forward.rs:3199` |
| `rope(q, k, position_ids)` | `kiln/attn/rope` block | `forward.rs:3215-3219` |
| Attention + `o_proj` + residual | `gqa_attention_paged` return + `(x + attn_out)?` | `forward.rs:3811-3835` |
| `post_attention_layernorm(hidden)` | `rms_norm(&x, &layer.post_attention_layernorm, ...)` | `forward.rs:3841` |
| MLP + residual | `swiglu_ffn(&normed, &layer.mlp, ...)` + `(x + ffn_out)?` | `forward.rs:3853-3868` |

Order is byte-for-byte identical: pre-attn norm → qk projection →
qk-norm → RoPE → attention → o_proj+residual → pre-MLP norm → MLP +
residual. There is no inversion, no missed step, and no extra step.

### 4. C13 weight-loading audit corroborates

The C13 weight-loading audit
([`docs/archive/phase-c/phase-c13/mtp-weight-loading-audit.md`](../phase-c13/mtp-weight-loading-audit.md))
already enumerated all 15 MTP-prefixed tensors, validated their
shapes, and confirmed the loader stores them as distinct fields under
`MtpWeights.layer.*`. C20 closes the runtime side: the same struct
fields are the ones bound at the four call positions above.

## Why this hypothesis was tempting

The α ≈ 0.15 plateau after the C18 reference-frame fix has the shape
of "block topology is right but one weight is off-by-one inside the
block." In other architectures with looser MTP-block conventions
(some MoE MTP heads reuse main-stack norms by design, for example),
the "MTP block silently shares the main-stack norms" pattern is a
real bug. For Qwen3.5-4B, the loader prefix split + struct
isolation in `MtpWeights.layer` (a fresh `LayerWeights` instance, not
a borrow of any main-stack layer) closes the door on that class of
bug at compile time.

## Hypothesis 3 — the next target

Carrying the C18 handoff item 3 forward as the new head of the queue:

> **Rotary `mtp_pos` offset.** MTP inner-block rotary should receive
> `pos + 1` (the next position) relative to the main block's rotary at
> `pos`. If kiln feeds the same position index to both, the rotary
> frequency is off-by-one and draft tokens will consistently miss.

Concrete artifacts in scope:

- `mtp_forward_step` ([`forward.rs:4605-4622`](../../crates/kiln-model/src/forward.rs))
  passes `mtp_pos` and `&positions` into `transformer_block_paged` as
  the `start_pos` and `positions` arguments.
- `gqa_attention_paged` ([`forward.rs:3215-3296`](../../crates/kiln-model/src/forward.rs))
  applies RoPE using the `positions` tensor and `inv_freq`.
- HF reference: `Qwen3NextMtpDecoderLayer.forward` builds
  `position_ids` for the MTP step from `position_ids + 1` relative to
  the main-block step at the same decode tick.

### Suggested next-step (CPU-only, no pod)

1. **Targeted forward read.** Trace where `mtp_pos` and the
   `positions` tensor handed to `mtp_forward_step` are computed in the
   speculative-decode caller (`crates/kiln-model/src/speculative.rs`
   and `crates/kiln-server/src/bench.rs`). Confirm the value passed in
   is the *next* position (`main_pos + 1`), not the same position the
   main block just consumed.
2. **Reference-dump comparison.** Reuse `scripts/mtp_reference_dump.py`
   + `scripts/mtp_compare.py` to dump HF's `position_ids` for the MTP
   step and compare against kiln's `positions` tensor at the same
   prefix. A consistent +1 offset (or its absence) is the single
   diagnostic.
3. **If parity diverges:** ship the position-offset fix and re-run the
   C18 α-recovery bench (3 seeds, 512 prompt × 128 decode, A6000,
   `KILN_SPEC_METHOD=mtp`, `KILN_W4A16=1`).
4. **If parity matches:** disprove H3 with a Phase C21 doc-only PR
   and advance to H4 (MTP-head gate / residual / `fc` parameterization)
   or H5 (draft-side sampler mismatch).

The CPU-only steps cost no pod time. Only step 3 — once an actual
fix is in hand — justifies acquiring an A6000.

## Cost / budget

- Wall-clock: ~25 min (no pod, source audit and HF cross-reference
  only).
- Pod cost: $0. Hypothesis was disprovable from the existing
  `loader.rs` + `forward.rs` source plus the C13 weight-loading
  audit and the HF reference dumped in
  `scripts/mtp_reference_dump.py`.
- Well under the 120 min / $50 task cap.
