# MTP Training Plan — keep the draft head aligned with the tuned model

**Status:** PR-A (serving + adapter format) shipped: `LoraWeights.mtp`,
LoRA-threaded MTP verify/replay, draft-block LoRA application, dispatch
gate lifted. This document specifies PR-B: actually *training* the MTP
draft block whenever an adapter trains.

## Why

Qwen3.5-4B ships one native MTP layer (k=1). Speculative decoding with it
drafts ~2 tokens per base forward when the draft agrees with the model.
The draft head was distilled against the BASE model — every LoRA step
moves the served distribution away from it, so acceptance rate (and the
entire speedup) decays exactly in proportion to how much the user has
personalized their model. "Your model gets better every time you use it"
must not mean "…and slower."

PR-A guarantees output fidelity under adapters (the verify pass applies
the LoRA), so the ONLY thing at stake in PR-B is restoring acceptance
rate — a pure speed win with no correctness risk.

## Shape of the feature

A post-SFT **MTP alignment phase** inside `sft_train` (GRPO/OPD later),
gated on `config.train_mtp` (default ON when the checkpoint has `mtp.*`
tensors) and skipped silently when the model has no MTP weights.

1. **Trainable params:** LoRA A/B for the MTP block's
   `{q,k,v,o,gate,up,down}_proj` at the run's rank/alpha. The `fc`
   (2H→H), the three norms, and the tied lm_head stay frozen — the block
   carries the capacity.
2. **Data:** the same tokenized SFT examples (tokens + label_mask) the
   main phase just trained on.
3. **Hiddens:** one inference forward per example with the FRESHLY
   TRAINED adapter applied —
   `model_forward_paged_normed_hidden(tokens, …, lora)` returns the
   post-final-norm hiddens `[1, T, H]` (exactly what `mtp_forward_step`
   consumes as `h_prev` at serve time). Detach; allocate a temporary
   PagedKvCache + BlockTable sized for T (freed after the phase).
4. **MTP forward (tape scope armed):** for positions `t` in `0..T-2`:
   `fused_t = fc( concat( pre_fc_norm_embedding(embed(tok_{t+1})),
   pre_fc_norm_hidden(h_t) ) )` → MTP transformer block (causal
   full-attention over the fused sequence, RoPE at absolute positions,
   LoRA on the seven projections via the tape's lora-linear) →
   `final_layernorm` → logits via the tied `embed_tokens_t`.
   All ops exist as tape primitives (`try_tape_matmul_kt`,
   `try_tape_rms_norm_kt`, `try_tape_concat_kt`, `try_tape_rope_kt`,
   `try_tape_flash_attn_kt`, `try_tape_lora_linear_kt`,
   `try_tape_cross_entropy_from_logits_kt`).
5. **Loss:** CE against the +2-shifted labels, masked to positions where
   `label_mask[t+2]` (the MTP target must itself be a supervised token).
   Skip examples with < 3 supervised positions.
6. **Optimizer:** the run's optimizer (Muon default) over just the MTP
   LoRA params; a handful of epochs over the same examples (cheap: one
   tiny block).
7. **Save:** write the trained pairs under the PR-A key scheme
   `base_model.model.model.mtp.layers.0.{self_attn,mlp}.{module}.lora_{A,B}.weight`
   into the same `adapter_model.safetensors`.
8. **Receipt:** `mtp_alignment: { trained: bool, examples, final_ce,
   skipped_reason }` so the adapter records whether its draft head is
   aligned.

## Implementation notes

- Mirror how `standard_forward_backward_tape_authoritative_kt` arms the
  tape scope and registers LoRA `Parameter`s as leaves; the MTP block is
  structurally identical to a main full-attention layer, so its training
  forward is the same composition with the fc/concat prologue and
  tied-head epilogue added.
- The hiddens forward (step 3) runs OUTSIDE the tape scope (no leaves →
  no recording) — gradients never flow into the main model. The
  alignment phase is fully separable from the main SFT step.
- Validation (bounded, attended): finite-difference check on one MTP
  LoRA pair (mirror `grpo_logit_grad_matches_finite_difference_f32`);
  CE-decreases-over-epochs assertion on a tiny fixture; then one real
  adapter on the rig with the C1 attribution CSV
  (`KILN_C1_ATTR_PATH`) comparing acceptance rate: base draft vs
  aligned draft on the same prompts.

## Follow-ups after PR-B

- GRPO/OPD runs get the same phase (hiddens from their own forwards).
- `kiln self-improve` includes MTP alignment automatically (it's SFT
  under the hood once the agent-traces bridge resolves prompts).
- Dashboard: show draft acceptance rate per adapter (the data already
  exists in `MtpGenerationOutput::draft_accepted_count`).

## Implementation anchors (recon 2026-06-11)

- **Save**: `TrainableLoraParams::save_peft` (trainer.rs:~1426) writes
  `base_model.model.model.layers.{i}.{self_attn|mlp}.{module}.lora_{A,B}.weight`
  via per-module `save_proj` closures. MTP: add
  `TrainableLoraParams.mtp: Option<TrainableLoraLayerParams>` and a second
  loop emitting `base_model.model.model.mtp.layers.0.{sub}.{module}...`
  (PR-A loader already parses these keys — `parse_peft_key.is_mtp`).
- **Init**: mirror `TrainableLoraParams::initialize_seeded` (trainer.rs:687+)
  for the 7 MTP modules; shapes come from `MtpWeights.layer`
  (q/k/v/o from `AttentionWeights::Full`, gate/up/down from `FfnWeights`).
- **Hiddens**: `model_forward_no_head` + `model_forward_final_norm`
  (both already used by the GRPO step fn, trainer.rs:~8875) — NO paged
  cache needed, returns post-final-norm [1,T,H] with the trained adapter
  applied. Run OUTSIDE the tape scope, detach.
- **Tape scope**: mirror `grpo_step_forward_backward_tape_authoritative_kt`
  (trainer.rs:8849): `kiln_kt_bridge::tape_bridge::with_tape_authoritative_scope_kt`
  + GradStore extraction keyed by `Parameter::tensor_id()` (the
  `decode_kt_param_deposit` loop).
- **MTP block forward under tape**: the serve path uses
  `transformer_block_paged` (needs a paged cache). For training, compose
  from tape primitives like the main training forward does, OR construct
  a small throwaway PagedKvCache (the MTP debug/bench path shows how) and
  reuse `transformer_block_paged` with `lora: Some((&mtp_lora_view, scale))`
  — the lora matmuls route through the tape when the scope is armed
  (same mechanism the main layers use).
- **Loss**: `try_tape_cross_entropy_from_logits_kt` (the SFT CE root) over
  the tied-head logits (`weights.embed_tokens_t` matmul), labels = +2
  shift, masked to positions where `label_mask[t+2]`.
- **Optimizer**: reuse the SFT step's optimizer construction on the MTP
  param list only; a handful of epochs over the same examples.
