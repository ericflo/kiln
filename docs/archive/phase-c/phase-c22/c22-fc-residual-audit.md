# Phase C22 — MTP `fc` and residual parameterization audit (hypothesis 4, doc-only redirect)

## Verdict

**Hypothesis 4 is DISPROVEN on static audit. No code change. No pod
spend.** Kiln's MTP `fc` projection (`pre_fc_norm_embedding`,
`pre_fc_norm_hidden`, `concat`, bias-free `[2H → H]` matmul) and its
inner-block residual handoff (`fused → transformer_block_paged → +x →
+ffn_out → final_layernorm`) match vLLM's
`Qwen3NextMultiTokenPredictor.forward` and the inner
`Qwen3NextDecoderLayer.forward` line-for-line on every parameter the
C21 handoff named: concat order, dtype, bias presence,
pre-`fc` dual-norm assignment, the `residual = None` precondition on
entry to the inner block, the post-attn / post-mlp residual sequence
inside the inner block, and the final-norm fusion of the
second residual on the outer caller. Hand off the residual α gap
(C18 median 0.1532, floor 0.5) to **H5 (draft-side sampler /
temperature / penalty drift)** as the next cheap-first hypothesis,
keeping H3 parked for a hardware A/B if H5 also disproves cleanly.

## C21 hypothesis as stated

From [`docs/archive/phase-c/phase-c21/c21-mtp-rotary-pos-audit.md`](../phase-c21/c21-mtp-rotary-pos-audit.md)
(handoff to C22, "Hypothesis 4 — the next target"):

> **MTP-head gate / residual / `fc` parameterization.** The
> `mtp.fc` projection collapses `[h_main || embed]` → `H` *before*
> the inner block. If kiln's gating, residual addition, or
> normalization around `fc` differs from HF's by even a sign or a
> missed scale, the MTP block sees a uniformly mis-scaled input and
> α pegs near a constant low.

The C21 doc enumerated three concrete parity claims to test on static
audit:

1. **`fc` projection parity:** `concat([pre_fc_norm_embedding(embed),
   pre_fc_norm_hidden(h_prev)])` → `[2H → H]`. Cross-check that
   residual / bias / dtype exactly mirrors
   `Qwen3NextMultiTokenPredictor.forward`'s `fc` application
   ([`forward.rs:4527-4543`](#references-kiln)).
2. **Inner-block residual init:** the inner `transformer_block_paged`
   call expects `residual = None` on entry per vLLM
   ([`qwen3_next_mtp.py:115`](#references-vllm) sets
   `residual = None`). Verify kiln's call honours this and does
   not accidentally feed in a stale residual from a prior MTP step
   ([`forward.rs:4605-4622`](#references-kiln)).
3. **Inner-block forward parity:** verify the exact pre-residual /
   post-residual sequence inside the inner block matches
   `Qwen3NextDecoderLayer.forward`.

The audit below disproves all three claims of mismatch (i.e. confirms
parity).

## Tensors and counters in scope

For one `mtp_forward_step` invocation at draft index `mtp_pos`:

| Symbol | Shape | Meaning |
| --- | --- | --- |
| `h_prev` | `[1, 1, H]` | Pre-final-norm hidden from base model (or prior MTP step), output of the C18 fix |
| `token_emb` | `[1, 1, H]` | Embedded draft token id, lookup into shared `embed_tokens` |
| `norm_emb` | `[1, 1, H]` | `pre_fc_norm_embedding(token_emb)` |
| `norm_h` | `[1, 1, H]` | `pre_fc_norm_hidden(h_prev)` |
| `concat` | `[1, 1, 2H]` | `cat([norm_emb, norm_h], dim=-1)` |
| `fused` | `[1, 1, H]` | `concat @ fc.T`, bias-free |
| `attn_out` | `[1, 1, H]` | Output of `gqa_attention_paged` (post-`o_proj`, post `attn_output_gate`) |
| `x_post_attn` | `[1, 1, H]` | `fused + attn_out` (first residual) |
| `ffn_out` | `[1, 1, H]` | Output of `swiglu_ffn` on `post_attention_layernorm(x_post_attn)` |
| `out` | `[1, 1, H]` | `x_post_attn + ffn_out = fused + attn_out + ffn_out` (second residual; equals `transformer_block_paged` return) |
| `normed` | `[1, 1, H]` | `final_layernorm(out)`, fed to weight-tied `lm_head` |

Two reference checkpoint key conventions exist for the final norm
weight; both work:

| Loader key | Source | Notes |
| --- | --- | --- |
| `mtp.norm.weight` | Stock Qwen3.5-4B safetensors | Used by both kiln (`loader.rs:660`) and vLLM (`self.norm` attribute name) |
| `mtp.final_layernorm.weight` | Older HF / vLLM doc convention | Accepted by kiln loader as a fallback ([`loader.rs:659-666`](#references-kiln)) |

## Evidence

### 1. `fc` projection — bias-free, dtype-correct, dual-norm before concat

**Kiln** ([`forward.rs:4502-4554`](#references-kiln),
[`loader.rs:628-654`](#references-kiln)):

```rust
// (B2 swap A/B is the identity branch in production)
let (norm_emb_weight, norm_h_weight) =
    (&mtp.pre_fc_norm_embedding, &mtp.pre_fc_norm_hidden);

let norm_emb = rms_norm(&token_emb, norm_emb_weight, config.rms_norm_eps)?;
let norm_h   = rms_norm(h_prev,     norm_h_weight,   config.rms_norm_eps)?;

let concat = Tensor::cat(&[&norm_emb, &norm_h], 2)?.contiguous()?;
let fused  = concat.broadcast_matmul(&mtp.fc_t)?; // bias-free
```

The loader proves bias absence: `load_mtp_if_present` extracts only
`{prefix}fc.weight`, validates the `[hidden, 2*hidden]` shape, and
never touches `fc.bias`. There is no field on `MtpWeights` to hold
one ([`forward.rs:196-215`](#references-kiln); see field list:
`fc`, `fc_t`, `pre_fc_norm_embedding`, `pre_fc_norm_hidden`,
`layer`, `final_layernorm`).

**vLLM** ([`qwen3_next_mtp.py:64-72,107-114`](#references-vllm)):

```python
self.fc = ColumnParallelLinear(
    self.config.hidden_size * 2,
    self.config.hidden_size,
    gather_output=True,
    bias=False,
    return_bias=False,
    quant_config=quant_config,
    prefix=f"{prefix}.fc",
)
# ...
inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
hidden_states = self.pre_fc_norm_hidden(hidden_states)
hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
hidden_states = self.fc(hidden_states)
residual = None
```

**Parity check (per claim):**

| Aspect | Kiln | vLLM | Match |
| --- | --- | --- | --- |
| `fc` bias | absent (no field, loader rejects unloaded keys) | `bias=False` | ✓ |
| `fc` shape | `[hidden, 2*hidden]` (BF16) | `ColumnParallelLinear(hidden*2, hidden)` | ✓ |
| Pre-`fc` norm on embed | `pre_fc_norm_embedding(token_emb)` BEFORE concat | `self.pre_fc_norm_embedding(inputs_embeds)` BEFORE concat | ✓ |
| Pre-`fc` norm on hidden | `pre_fc_norm_hidden(h_prev)` BEFORE concat | `self.pre_fc_norm_hidden(hidden_states)` BEFORE concat | ✓ |
| Concat order | `[norm_emb, norm_h]` (dim=-1) | `torch.cat([inputs_embeds, hidden_states], dim=-1)` (post-norm names) | ✓ |
| Activation between concat and fc | none | none | ✓ |
| Post-`fc` activation | none (next op is the inner-block input layernorm) | none (next op is the inner-block forward) | ✓ |
| Dtype default | BF16 in/out (with optional `KILN_MTP_FC_FP32_ACCUM` opt-in for noise study) | BF16 in/out | ✓ |

The C9 fp32-accum kill switch (`KILN_MTP_FC_FP32_ACCUM=1` /
`KILN_MTP_FP32_HEAD=1`) is opt-in only; the production path is BF16
and matches vLLM's default precision.

### 2. Inner-block call site — `residual = None` enforced by construction

**Kiln** ([`forward.rs:4616-4633`](#references-kiln)):

```rust
let mtp_hidden_result = transformer_block_paged(
    backend,
    &fused,                // ← `x` argument; this is the ONLY hidden input
    &mtp.layer,
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

**vLLM** ([`qwen3_next_mtp.py:114-126`](#references-vllm)):

```python
hidden_states = self.fc(hidden_states)
residual = None
# ...
current_step_idx = spec_step_idx % self.num_mtp_layers
hidden_states, residual = self.layers[current_step_idx](
    positions=positions,
    hidden_states=hidden_states,
    residual=residual,
)
```

**Parity check:** Kiln's `transformer_block_paged` has **no `residual`
parameter** ([`forward.rs:3781-3798`](#references-kiln)). The
"residual = None on entry" semantic is therefore enforced by
construction: there is no API surface to feed in a stale residual
from a prior MTP step. The function always uses its `x` argument
(here, `fused`) as the first-residual source. This is structurally
safer than vLLM's explicit-`residual` design — kiln cannot leak a
prior-step residual even by accident.

vLLM's `Qwen3NextDecoderLayer.forward` enters the
`if residual is None` branch, which is exactly equivalent to kiln's
implicit handling: the layer takes `hidden_states` as both the input
and the implicit residual seed (see step 3 below).

### 3. Inner block — pre-residual / post-residual sequence

**Kiln** ([`forward.rs:3808-3878`](#references-kiln)):

```rust
// Pre-attention norm
let normed = rms_norm(x, &layer.input_layernorm, rms_norm_eps)?;

// Self-attention with paged cache (handles rope + attn_output_gate)
let attn_out = gqa_attention_paged(
    backend, &normed, attn_weights, positions, start_pos,
    num_heads, num_kv_heads, head_dim, rotary_dim,
    inv_freq, rms_norm_eps, paged_cache, block_table,
    full_attn_layer_idx,
    config.attn_output_gate,    // ← true for Qwen3.5-4B
    lora,
)?;

// Residual connection
let x = (x + attn_out)?;

// Post-attention norm
let normed = rms_norm(&x, &layer.post_attention_layernorm, rms_norm_eps)?;

// Feed-forward network
let ffn_out = swiglu_ffn(&normed, &layer.mlp, lora)?;

// Residual connection
let out = (x + ffn_out)?;
Ok(out)
```

**vLLM** ([`qwen3_next.py:388-447`](#references-vllm)):

```python
def forward(self, hidden_states, residual, positions=None, **kwargs):
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    # attention populates self_attention_output
    self.self_attn(positions=positions, output=self_attention_output,
                   hidden_states=hidden_states)
    hidden_states = self_attention_output
    # layer_scale block — Qwen3.5-4B has layer_scale=False (default),
    # so this branch is bypassed entirely (no additional scaling).
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    # second layer_scale branch also bypassed.
    return hidden_states, residual
```

vLLM's `Qwen3NextRMSNorm.forward(hidden, residual)` is the
fused-residual norm:
`return norm(hidden + residual), hidden + residual`. Substituting in
the chain:

| Step | vLLM expression (residual is None on entry, so `residual = hidden_states = fused`) | Equivalent kiln expression |
| --- | --- | --- |
| Pre-attn norm | `hidden = input_layernorm(fused)` | `normed = rms_norm(x=fused, input_layernorm)` |
| Attn | `attn_out = self_attn(hidden=normed)` | `attn_out = gqa_attention_paged(normed, …)` |
| Post-attn fused norm | `(hidden, residual) = post_attn_norm(attn_out, residual=fused)` ⇒ `hidden = norm(attn_out + fused)`, `residual = attn_out + fused` | `x = fused + attn_out`; `normed = rms_norm(x, post_attention_layernorm)` |
| MLP | `hidden = mlp(hidden)` | `ffn_out = swiglu_ffn(normed, mlp)` |
| Layer return | returns `(mlp_out, attn_out + fused)` | returns `out = x + ffn_out = (fused + attn_out) + ffn_out` |

vLLM's caller then closes the second residual via the outer
`self.norm(hidden, residual)` fused-norm
([`qwen3_next_mtp.py:133`](#references-vllm)):

```python
hidden_states, _ = self.norm(hidden_states, residual)
```

which evaluates to
`norm(mlp_out + (attn_out + fused)) = norm(fused + attn_out + mlp_out)`.

Kiln's caller closes the same second residual implicitly (because
`transformer_block_paged` already returned `out = fused + attn_out +
ffn_out`) and then applies the same final norm
([`forward.rs:4668-4671`](#references-kiln)):

```rust
let normed = rms_norm(&mtp_hidden, &mtp.final_layernorm, config.rms_norm_eps)?;
```

= `final_layernorm(fused + attn_out + ffn_out)` = vLLM's
`self.norm(mlp_out + (attn_out + fused))`. **Identical scalar value
for identical inputs.**

### 4. `attn_output_gate` parity

The MTP layer is loaded as a full-attention layer
([`loader.rs:678-697`](#references-kiln)) sharing `attn_output_gate`
with the main-model full-attention layers (`config.attn_output_gate
= true` for Qwen3.5-4B,
[`config.rs:33,94`](#references-kiln)).

**Kiln** ([`forward.rs:3837`](#references-kiln)) passes
`config.attn_output_gate` into `gqa_attention_paged`, which fans the
gate dim into the fused `q_proj` output and applies
`sigmoid(gate) * attn_output` before `o_proj`
([`forward.rs:2691-2728`](#references-kiln) for the unfused path,
[`forward.rs:3122-3298`](#references-kiln) for the paged-decode
fused-Marlin path).

**vLLM** ([`qwen3_next.py:227,232,281-289,304-308`](#references-vllm)):

```python
self.attn_output_gate = getattr(config, "attn_output_gate", True)
self.qkv_proj = QKVParallelLinear(
    config.hidden_size, self.head_dim,
    self.total_num_heads * (1 + self.attn_output_gate),
    self.total_num_kv_heads, ...
)
# forward:
if self.attn_output_gate:
    q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
    q, gate = torch.chunk(q_gate.view(*orig_shape, self.num_heads, -1), 2, dim=-1)
# ...
attn_output = self.attn(q, k, v)
if self.attn_output_gate:
    gate = torch.sigmoid(gate)
    attn_output = attn_output * gate
output[:], _ = self.o_proj(attn_output)
```

Identical: `q_proj` is fused-doubled to carry the gate, Q and gate
are split halves, gate is `sigmoid`'d and elementwise-multiplied with
the attention output before `o_proj`. Kiln's `[8192, 2560]`
`q_proj` shape (= `2 · 16 · 256 · 2560`) is exactly the gated
projection vLLM's `total_num_heads * (1 + attn_output_gate)` produces
and is documented in the C13 weight-loading audit
([`docs/archive/phase-c/phase-c13/mtp-weight-loading-audit.md`](../phase-c13/mtp-weight-loading-audit.md)).

### 5. `layer_scale` — both paths are no-ops on Qwen3.5-4B

vLLM's `Qwen3NextDecoderLayer.__init__` has
`self.layer_scale = getattr(config, "layer_scale", False)`
([`qwen3_next.py:373`](#references-vllm)). The Qwen3.5-4B
`Qwen3NextConfig` does not set `layer_scale`, so the attribute
defaults to `False` and both `if self.layer_scale:` branches
([`qwen3_next.py:419-431,433-447`](#references-vllm)) are skipped
entirely.

Kiln does not implement `layer_scale` at all (no `attn_layer_scale`
or `ffn_layer_scale` field on any weights struct;
`grep -r layer_scale crates/` returns zero hits). For Qwen3.5-4B
this is structurally equivalent: both code paths are no-ops, and the
math is unchanged.

If any future Qwen3-Next variant ships with `layer_scale=True`, kiln
will need to add the per-head `(scale + 1) * hidden` multiplications
on both the attn-out and mlp-out paths, but that is a forward-looking
parity concern, not a correctness issue for the current target
checkpoint.

### 6. Final-norm tensor key parity

Both kiln and vLLM accept either `mtp.norm.weight` or
`mtp.final_layernorm.weight` as the final RMSNorm weight key:

| Aspect | Kiln | vLLM | Match |
| --- | --- | --- | --- |
| Loader key fallback | accepts both ([`loader.rs:659-666`](#references-kiln)) | uses attribute `self.norm` keyed off `mtp.norm.weight` ([`qwen3_next_mtp.py:87`](#references-vllm)) | ✓ |
| Stock Qwen3.5-4B key in checkpoint | `mtp.norm.weight` (per [C13 audit](../phase-c13/mtp-weight-loading-audit.md)) | same | ✓ |
| Application | `rms_norm(out, mtp.final_layernorm, rms_norm_eps)` | `self.norm(hidden, residual)` (fused-residual norm: `norm(hidden + residual)`) | ✓ (same scalar value, since kiln's `out` already equals vLLM's `hidden + residual`) |

### 7. End-to-end scalar equivalence (summary)

For one MTP step with the same draft token id, the same `h_prev`,
the same checkpoint, identical RoPE positions, BF16 throughout, and
no debug flags:

| Tap | vLLM expression | Kiln expression | Equal? |
| --- | --- | --- | --- |
| `norm_emb` | `pre_fc_norm_embedding(embed)` | `rms_norm(token_emb, pre_fc_norm_embedding)` | ✓ |
| `norm_h` | `pre_fc_norm_hidden(h_prev)` | `rms_norm(h_prev, pre_fc_norm_hidden)` | ✓ |
| `fused` | `fc(cat([norm_emb, norm_h], -1))`, bias-free | `cat([norm_emb, norm_h], 2) @ fc.T`, bias-free | ✓ |
| `attn_out` | `o_proj(sigmoid(gate) * attn(q_norm(q),k_norm(k),v))` after RoPE | same path, same `attn_output_gate=true` | ✓ |
| `x_post_attn` | implicit via fused `post_attn_norm(attn_out, fused)` ⇒ `residual_after = attn_out + fused` | `x = fused + attn_out` | ✓ |
| `ffn_out` | `mlp(post_attn_norm(attn_out + fused))` | `swiglu_ffn(rms_norm(x_post_attn, post_attention_layernorm))` | ✓ |
| Inner-layer return | `(ffn_out, attn_out + fused)` | `out = (fused + attn_out) + ffn_out` (kiln folds the second residual inside) | ✓ |
| Final norm | `self.norm(ffn_out, attn_out + fused) = norm(ffn_out + attn_out + fused)` | `final_layernorm(out) = norm(fused + attn_out + ffn_out)` | ✓ |
| `logits` | `lm_head(normed)` (weight-tied to `embed_tokens`) | `lm_head_forward(normed, embed_tokens_t)` (weight-tied) | ✓ |

No expression in the right column differs from the left. **There is
no `fc`/residual/gating bug to fix here.**

## Why this hypothesis was tempting

C18 framed H4 with two converging cues:

- **Surface complexity of the `fc` step.** "Concat two normed
  hidden vectors and project them down" looks like a place a sign
  flip, missed bias, or wrong norm assignment could hide. The
  pre-fc-norms are name-only distinguishable (both `[H]`-vectors),
  which is exactly the case where a swap goes unnoticed; the B2
  `KILN_MTP_SWAP_FC_NORMS=1` A/B was added defensively for this
  reason ([`forward.rs:4502-4507`](#references-kiln)).
- **Residual hand-off across two different code shapes.** vLLM's
  fused-residual norm style (`norm(hidden, residual)` returns
  `(norm(h+r), h+r)`) and kiln's explicit-residual style
  (`x = x + delta`) look superficially different even though they
  compute identical scalars. "Different code shape" is a natural
  trigger for "different result," especially for a cross-runtime
  audit.

The audit shows both cues were over-interpreted:

- The pre-fc-norm assignment is correct as loaded; the C2 A/B exists
  in the codebase (it has been since Phase B2) and would have
  surfaced any swap by changing α materially. It does not.
- The two residual styles are mathematically equivalent: vLLM's
  outer `self.norm(hidden, residual)` closes the second residual the
  same way kiln's `transformer_block_paged` body does internally,
  yielding the same final `normed` tensor.

## Hypothesis 5 — the next target

Carrying the C18 handoff item 5 forward as the new head of the
queue:

> **Draft-side sampler / temperature / penalty drift.** If the MTP
> draft path samples with a different temperature, repetition
> penalty, top-k/top-p, or logit-processor wiring than the verifier
> (or than the trained head was calibrated for at inference), the
> drafted tokens will be systematically off-distribution and α will
> peg low even though the head's logits are bit-correct.

Concrete artifacts in scope:

- The MTP draft sampler call site in
  `crates/kiln-model/src/speculative.rs` — verify temperature,
  top-k, top-p, repetition penalty, presence/frequency penalty, and
  any logit-bias / suppression / forced-token policy is identical to
  the verifier's sampler path.
- `crates/kiln-server/src/bench.rs:1086-1138` — the speculative loop
  body. Check that the verifier and drafter share the same `LogitsProcessor`
  state (or, if separate, that the configurations match exactly).
- vLLM cross-reference: `vllm/v1/spec_decode/eagle.py` and
  `vllm/v1/spec_decode/utils.py` for how the draft logits are
  processed before sampling, plus
  `vllm/v1/sample/sampler.py` for the verifier-side equivalent.
- C8 / C12 already pinned `attn_output_gate`, `attn_layer_idx`, and
  the fp32-head dtype path; H5 should focus on what happens
  **after** `lm_head_forward` returns logits.

If H5 also disproves cleanly, return to **H3 (rotary `mtp_pos` offset
hardware A/B)** and run the cheap A6000 discriminator
(`abs_pos = base_pos + mtp_pos` vs `abs_pos = base_pos - 1 +
mtp_pos`, ~25 min, ~$5) before opening a deeper investigation into
the trained MTP head's expected calibration.

### Suggested next-step (CPU-only, no pod)

1. **Targeted speculative read.** Trace the draft sampler in
   `speculative_mtp_decode_step` and the verifier sampler in the
   speculative loop body. Compare temperature, top-k, top-p,
   repetition / presence / frequency penalties, and any logit-bias
   wiring side by side.
2. **Verifier ↔ drafter parity diff.** If the configurations differ
   on any single dimension, that is the H5 candidate fix. If they
   are identical, confirm vLLM's MTP/EAGLE drafter applies the same
   sampler config; a missing logit-processor stage (e.g. a missing
   penalty pass on draft logits) would explain a uniform draft
   shift.
3. **If parity diverges:** ship the sampler fix and re-run the C18
   α-recovery bench (3 seeds, 512 prompt × 128 decode, A6000,
   `KILN_SPEC_METHOD=mtp`, `KILN_W4A16=1`).
4. **If parity matches:** disprove H5 with a Phase C23 doc-only PR
   and fall back to **H3 (cheap hardware A/B)** as the discriminator
   of last resort before declaring "the trained MTP head is
   calibrated against an unknown convention" and escalating to a
   training-time data audit.

## Cost / budget

- Wall-clock: ~50 min (no pod, source audit + vLLM cross-reference
  of `qwen3_next_mtp.py` + `qwen3_next.py` decoder layer + kiln
  loader + kiln transformer block + doc).
- Pod cost: $0. Hypothesis was reduced to "kiln matches vLLM at
  every audited point" from existing kiln source +
  `qwen3_next_mtp.py` + `qwen3_next.py`.
- Well under the 120 min / $50 task cap.

## References (kiln)

- `crates/kiln-model/src/forward.rs:196-215` — `MtpGpuWeights`
  struct: `fc`, `fc_t`, `pre_fc_norm_embedding`, `pre_fc_norm_hidden`,
  `layer`, `final_layernorm`. **No bias field.**
- `crates/kiln-model/src/forward.rs:4408-4683` — `mtp_forward_step`:
  embed → dual norms → concat → bias-free `fc` → inner block →
  `final_layernorm` → weight-tied `lm_head`.
- `crates/kiln-model/src/forward.rs:4502-4507` — Phase B2
  `KILN_MTP_SWAP_FC_NORMS` A/B (identity in production).
- `crates/kiln-model/src/forward.rs:4527-4554` — `concat` and `fc`
  (with optional `KILN_MTP_FC_FP32_ACCUM` opt-in for noise study).
- `crates/kiln-model/src/forward.rs:4616-4633` — call into
  `transformer_block_paged` with `&fused` as the `x` argument.
- `crates/kiln-model/src/forward.rs:4668-4671` — final-norm and
  `lm_head_forward`.
- `crates/kiln-model/src/forward.rs:3781-3878` —
  `transformer_block_paged`: pre-attn norm → GQA → first residual
  (`x = x + attn_out`) → post-attn norm → SwiGLU MLP → second
  residual (`out = x + ffn_out`). **No `residual` parameter.**
- `crates/kiln-model/src/forward.rs:2691-2728,3122-3298` —
  `attn_output_gate` handling in unfused and paged-decode fused
  paths.
- `crates/kiln-model/src/loader.rs:612-706` — `load_mtp_if_present`:
  loads `fc`, `pre_fc_norm_embedding`, `pre_fc_norm_hidden`, full
  GQA layer (forced full-attention via `idx=3`), final norm with
  fallback key. **Never loads `fc.bias`.**
- `crates/kiln-core/src/config.rs:33,94,156-160` —
  `attn_output_gate=true` for Qwen3.5-4B; `full_attn_q_proj_dim`
  doubles when the gate is on.
- `docs/archive/phase-c/phase-c18/c18-h-prev-post-norm-fix.md` — origin of the
  H1/H2/H3/H4/H5 hypothesis queue.
- `docs/archive/phase-c/phase-c19/c19-fc-norm-audit.md` — H1 disproof (no
  `mtp.fc_norm` tensor in checkpoint).
- `docs/archive/phase-c/phase-c20/c20-mtp-block-norm-audit.md` — H2 disproof (block
  dual-norm wiring is correct).
- `docs/archive/phase-c/phase-c21/c21-mtp-rotary-pos-audit.md` — H3 inconclusive
  (parked for hardware A/B); direct C22 handoff at "Hypothesis 4 —
  the next target".

## References (vLLM)

Sparse-cloned at `/tmp/vllm-c22/` (read-only, CPU-only):

- `vllm/model_executor/models/qwen3_next_mtp.py:43-134` —
  `Qwen3NextMultiTokenPredictor.__init__` and `.forward`. Bias-free
  `ColumnParallelLinear(hidden*2, hidden)` for `fc`; dual
  pre-fc-norms applied BEFORE concat; concat order `[embed,
  hidden]`; `residual = None` set explicitly before the inner-layer
  call; outer `self.norm(hidden, residual)` closes the second
  residual.
- `vllm/model_executor/models/qwen3_next.py:217-308` —
  `Qwen3NextAttention.__init__` and `.forward`:
  `attn_output_gate=getattr(config, "attn_output_gate", True)`,
  fused-doubled `qkv_proj`, `sigmoid(gate) * attn_output` before
  `o_proj`.
- `vllm/model_executor/models/qwen3_next.py:373-447` —
  `Qwen3NextDecoderLayer.__init__` and `.forward`: `layer_scale`
  defaults to `False` (Qwen3.5-4B does not set it), so the
  per-layer-scale branches are bypassed. Fused-residual
  `input_layernorm(hidden, residual)` and
  `post_attention_layernorm(hidden, residual)` close residuals
  along with the norm.

## Next: kick off Phase C23 (H5)

Open a Phase C23 task to audit the draft-side sampler against the
verifier-side sampler and the vLLM EAGLE/MTP drafter reference. Same
$0-pod CPU-only static audit pattern as C19/C20/C21/C22. If H5 also
disproves cleanly, run the H3 hardware A/B (~$5, ~25 min A6000) as
the last cheap discriminator before escalating.
