# Phase C17 — MTP `h_prev` Reference-Frame Audit

**Scope (from C16 handoff)**: With the accept/reject machinery (`speculative.rs`) and the MTP block-internal compute (C13 splice inputs + C14 splice outputs) both certified clean, the remaining structural candidate for α-collapse is the **input contract** between the base model and the MTP head — specifically, which hidden-state frame the MTP head's `pre_fc_norm_hidden` is supposed to consume.

**Method**: pure source-reading. Cross-reference the vLLM Qwen3-Next + Qwen3.5 MTP modules, the SGLang Qwen3.5 MTP module, and the HF transformers Qwen3-Next base model against kiln's `LmHeadMode::FullWithLastHidden` extraction path.

**Bench α during C17**: not re-measured. C17 is diagnosis-only on the same broken build C15/C16 audited (α ≈ 0.000–0.033 vs the 0.72 ship floor).

## Verdict

**kiln passes the WRONG hidden-state frame to the MTP head.**

- **vLLM/SGLang contract**: `Qwen3NextMultiTokenPredictor.forward(hidden_states=…)` and `Qwen3_5MultiTokenPredictor.forward(hidden_states=…)` expect the BASE model's POST-final-norm `last_hidden_state`. The MTP module then applies its own `pre_fc_norm_hidden` to that already-normalized input.
- **kiln behaviour**: `LmHeadMode::FullWithLastHidden` (`crates/kiln-model/src/forward.rs:5035–5057`) and `LmHeadMode::LastRowWithLastHidden` (lines 5059–5070) extract `last_hidden = hidden.narrow(1, seq_len - 1, 1)?.contiguous()?` **BEFORE** applying `weights.final_norm`, then return that pre-norm tensor as `h_prev`. The capture tap is named `h_pre_final_norm` in the dumps — kiln itself acknowledges in code that the dispatched `h_main` is pre-norm.
- **Code comment that documents the intent is wrong**: `forward.rs:5036–5043` claims "MTP needs `h_prev` as the base model's output between the final transformer block and the final RMSNorm (matches vLLM's Qwen3-Next reference where `previous_hidden_states` is passed into the MTP head before `final_layernorm`)." The cross-reference below shows that vLLM does the opposite.

The 2× `h_main` norm gap surfaced in C15 (kiln 67.77 vs HF 155.82 at step 0, FP32 reference) is the direct numerical fingerprint of this frame mismatch — exactly the magnitude shift produced by Qwen3-Next's `(1 + w) · x · rsqrt(mean(x²)+ε)` final RMSNorm.

## Cross-reference: vLLM `Qwen3NextMultiTokenPredictor.forward`

`vllm/model_executor/models/qwen3_next_mtp.py` (snapshot 2026-04-21):

```python
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,        # <-- argument from caller
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    spec_step_idx: int = 0,
) -> torch.Tensor:
    if get_pp_group().is_first_rank:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)   # <-- norm applied
        hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
        hidden_states = self.fc(hidden_states)
        residual = None
    ...
```

The `hidden_states` argument is the BASE model's output. `Qwen3NextModel.forward` in `vllm/model_executor/models/qwen3_next.py` ends with:

```python
hidden_states, _ = self.norm(hidden_states, residual)   # final RMSNorm + residual fuse
if aux_hidden_states:
    return hidden_states, aux_hidden_states
return hidden_states
```

i.e. the `hidden_states` returned from base — and therefore passed in to MTP — is **POST-final-norm**. The optional `aux_hidden_states` tuple is for EAGLE-3-style intermediate-layer auxiliary loss targets; it is not what the MTP forward consumes as `hidden_states`.

## Cross-reference: vLLM `Qwen3_5MultiTokenPredictor.forward`

`vllm/model_executor/models/qwen3_5_mtp.py`, identical pattern, lines 121–157:

```python
def forward(self, input_ids, positions, hidden_states, ..., spec_step_idx=0):
    if get_pp_group().is_first_rank:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
        hidden_states = self.fc(hidden_states)
    ...
    hidden_states, _ = self.norm(hidden_states, residual)   # MTP's own final RMSNorm
    return hidden_states
```

The Qwen3.5 path that kiln targets matches the Qwen3-Next pattern bit for bit: incoming `hidden_states` is whatever the base returns from its own `model.norm(...)`, then `pre_fc_norm_hidden` is the MTP head's own normalization step on top.

## Cross-reference: SGLang `Qwen3_5ForCausalLMMTP.forward`

`sglang/.../qwen3_5_mtp.py`, lines 113–155:

```python
hidden_states = forward_batch.spec_info.hidden_states

if not forward_batch.forward_mode.is_idle():
    input_embeds = self.pre_fc_norm_embedding(input_embeds)
    hidden_states = self.pre_fc_norm_hidden(hidden_states)
hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)
hidden_states = self.fc(hidden_states)
```

`spec_info.hidden_states` is populated by SGLang's spec-decode worker from the base model's last-step output, which is the `last_hidden_state` (post-final-norm) — same contract.

## Cross-reference: HF transformers `Qwen3NextModel.forward`

`transformers/models/qwen3_next/modeling_qwen3_next.py` lines 952–973:

```python
hidden_states = inputs_embeds
position_embeddings = self.rotary_emb(hidden_states, position_ids)

for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
    layer_mask = linear_attn_mask if self.config.layer_types[i] == "linear_attention" else causal_mask
    hidden_states = decoder_layer(
        hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=layer_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        **kwargs,
    )

hidden_states = self.norm(hidden_states)            # final norm applied unconditionally

return MoeModelOutputWithPast(
    last_hidden_state=hidden_states,                 # POST-norm
    past_key_values=past_key_values,
)
```

`MoeModelOutputWithPast.last_hidden_state` is therefore the post-final-norm hidden. HF transformers itself does not load MTP weights (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` at line 879), so it has no in-tree MTP head — the only working HF-aligned MTP references are vLLM and SGLang, and both consume `last_hidden_state` (post-norm).

## What kiln does

`crates/kiln-model/src/forward.rs:5035–5057` — the active `LmHeadMode::FullWithLastHidden` arm used for MTP:

```rust
LmHeadMode::FullWithLastHidden => {
    // Extract the last-row pre-final-norm hidden BEFORE normalising.
    // MTP needs `h_prev` as the base model's output between the final
    // transformer block and the final RMSNorm (matches vLLM's
    // Qwen3-Next reference where `previous_hidden_states` is passed
    // into the MTP head before `final_layernorm`). Logits are
    // produced over every position so speculative verification can
    // compare draft predictions at position 0 and sample a bonus at
    // position 1 in a single pass.
    let last_hidden = hidden.narrow(1, seq_len - 1, 1)?.contiguous()?;
    if crate::mtp_debug::is_h_main_capture_armed() {
        let _ = crate::mtp_debug::capture_h_main_tap("h_pre_final_norm", &last_hidden);
    }
    let logits = {
        kiln_nvtx::range!(c"kiln/lm_head");
        let normed = rms_norm(&hidden, &weights.final_norm, config.rms_norm_eps)?;
        normed.broadcast_matmul(&weights.embed_tokens_t)?
    };
    Ok((Some(logits), Some(last_hidden)))            // h_prev = pre-norm
}
```

`last_hidden` is captured **before** any `rms_norm(..., &weights.final_norm, ...)` is applied. The base model's logits are then computed by normalizing a separate copy of `hidden` and projecting through the tied LM head, which is correct for the BASE forward — but the `Some(last_hidden)` returned to the caller as `h_prev` is the pre-norm tensor.

`mtp_forward_step` (`crates/kiln-model/src/forward.rs:4520–4647`) takes that pre-norm `h_prev` and immediately runs:

```rust
let norm_h = rms_norm(h_prev, &mtp.pre_fc_norm_hidden, config.rms_norm_eps)?;
```

i.e. it applies `pre_fc_norm_hidden` to a tensor that was never run through `model.norm`. The MTP module's `pre_fc_norm_hidden` was trained against post-`model.norm` inputs (the only way the upstream reference frames it), so the per-feature scale and direction this norm sees in kiln are off-distribution.

## Why C13 / C14 / `mtp_reference_dump.py` did not catch this

`scripts/mtp_reference_dump.py:644–710` consumes kiln's captured `h_main` verbatim:

```python
h_main = kiln["h_main"].to(torch.float32)            # pre-final-norm, from kiln dump
...
norm_h = rms_norm(h_main, norm_h_w, args.rms_eps)    # apply MTP pre_fc_norm_hidden
...
fc_input = torch.cat([norm_emb, norm_h], dim=-1)
fc_output = torch.matmul(fc_input, w.fc_weight.t())
post_layer = mtp_inner_block(pre_layer, ...)
post_final_ln = rms_norm(post_layer, w.final_layernorm, args.rms_eps)
mtp_logits = torch.matmul(post_final_ln, w.embed_tokens.t())
```

The reference dump runs the SAME (potentially wrong) input through the SAME MTP block math, so the comparator at every C13 splice tap (`c6__token_emb`, `c6__fused`, `c6__fc_input`, …) and every C14 post-block tap (`c14__post_block`, `c14__post_norm`, `c14__logits`) sees cos ≈ 0.99995 — both sides agree because both sides start from the wrong frame. The cleanliness verdicts in C13 and C14 still hold conditionally: given `h_main` as input, the MTP block's internal compute matches HF. They do not certify that `h_main` was the right input.

C15 was the first script to compare kiln's `h_main` against an independent HF forward (`scripts/c15_h_main_drift_audit.py:161–203`). Its reference uses `out.hidden_states[-1]` from `output_hidden_states=True`, with the in-script comment "hs[-1] is output after final decoder layer = pre-final-norm hidden" (line 201). Empirically, however, C15 measured `kiln_norm ≈ 67.77` vs `ref_norm ≈ 155.82` at FP32 — a 2.30× ratio that matches the per-element scaling produced by Qwen3-Next's `(1 + w) · x · rsqrt(mean(x²)+ε)` final RMSNorm. The most consistent reading is that the HF reference frame at `hs[-1]` for this model is post-final-norm (or that the wrapper that materializes `output_hidden_states` for Qwen3-Next includes the final-norm output as the last tuple element). Either way, the magnitude evidence supports kiln being a frame behind the HF reference — not the other way around.

## Numerical fingerprint

From `docs/archive/phase-c/phase-c15/c15-h-main-drift-verdict.md` (FP32 column, reproduced for traceability):

| step | base_pos | cos_sim   | max\|Δ\|  | rel_err_mean | kiln_norm | ref_norm | ratio |
|-----:|---------:|----------:|----------:|-------------:|----------:|---------:|------:|
| 0    | 512      | 0.982333  | 2.550e+01 | 7.059e-01    | 67.77     | 155.87   | 2.30× |
| 1    | 513      | 0.986784  | 3.139e+01 | 8.925e-01    | 69.44     | 159.40   | 2.30× |
| 2    | 514      | 0.963464  | 1.770e+01 | 6.296e-01    | 63.46     | 150.51   | 2.37× |
| 3    | 515      | 0.961570  | 2.349e+01 | 5.264e-01    | 80.25     | 153.22   | 1.91× |
| 4    | 516      | 0.986800  | 1.264e+01 | 5.770e-01    | 70.92     | 158.02   | 2.23× |
| 5    | 517      | 0.952725  | 2.112e+01 | 5.885e-01    | 69.11     | 146.79   | 2.12× |
| 6    | 518      | 0.972997  | 2.289e+01 | 8.605e-01    | 70.44     | 153.16   | 2.17× |
| 7    | 519      | 0.976439  | 2.074e+01 | 6.673e-01    | 62.71     | 152.78   | 2.44× |

The ratios cluster tightly around the predicted `mean(|1 + w|) / 1 ≈ 2.0–2.4` for Qwen3-Next's final-norm weight distribution. This is exactly the per-element rescaling a missing `(1 + w) · rsqrt(mean(x²)+ε)` step produces when `mean(x²)` is already O(1) (which it is, after 32 layers of residual + RMSNorm in the base stack).

## Why this would tank α specifically

- The MTP block is a single attention + MLP stack on top of `fc(concat(norm_h, norm_emb))`. It is far smaller than the base stack and has very limited capacity to "re-learn" a 2× scale shift on its primary input across one block.
- `pre_fc_norm_hidden` divides by the local RMS, which partially absorbs the magnitude difference — but the per-element WEIGHTS `(1 + w_i)` of the missing base `model.norm` are gone, so the directional content fed into `fc` is in a different basis from what `fc` was trained against.
- Even cos = 0.982 between the post-norm-vs-pre-norm `h_main` is large enough at the input of a one-block transformer to scramble the top-1 token ranking. A draft logit that should rank-match the verifier ~72% of the time degenerating to 0–3% is consistent with a compounded direction-and-scale error at the MTP input.

## Open questions left for C18

1. **Is `LmHeadMode::FullWithLastHidden` the right place to fix this, or should `model_forward_paged_with_last_hidden` always return `model.norm`-applied `h_prev`?** Either change the `narrow(... seq_len-1, 1)` slice to operate on `normed` instead of `hidden`, or normalize `last_hidden` in place before the `Ok((Some(logits), Some(last_hidden)))` return. The math-cheaper fix is to drop the second `rms_norm(&hidden, ...)` call and re-use the same normed slice for both the logits projection and `h_prev`.
2. **Does `mtp_reference_dump.py` need a frame correction pass before C18 lands?** Yes — once kiln starts shipping post-norm `h_main`, the reference dump must apply the same `model.norm` to its kiln-derived input (or accept post-norm `h_main` directly). A one-line guard at line 645 (`h_main = rms_norm(kiln["h_main"].to(torch.float32), final_norm_w, eps)`) is sufficient until kiln's capture is fixed.
3. **Does kiln have any production code path that relies on the pre-norm `h_main` being returned to a non-MTP caller?** A one-day C17.5 spike could grep for callers of `LmHeadMode::*WithLastHidden` and `model_forward_paged_with_last_hidden` to confirm there are no silent dependencies on the current (wrong) frame outside of MTP.

## Artifacts

- This document.
- Companion verdict: `docs/archive/phase-c/phase-c17/c17-head-forward-first-divergence.md`.
- Source pointers all live in-tree (`crates/kiln-model/src/forward.rs`, `scripts/mtp_reference_dump.py`, `scripts/c15_h_main_drift_audit.py`).
- Cross-reference pins for the upstream contracts (snapshot 2026-04-21):
  - vLLM Qwen3-Next MTP: `qwen3_next_mtp.py:98–134`
  - vLLM Qwen3-Next base: `qwen3_next.py:451–531`
  - vLLM Qwen3.5 MTP: `qwen3_5_mtp.py:121–157`
  - SGLang Qwen3.5 MTP: `qwen3_5_mtp.py:39–155`
  - HF transformers Qwen3-Next base: `modeling_qwen3_next.py:879, 918–973`
