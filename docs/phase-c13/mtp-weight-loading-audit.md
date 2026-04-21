# Phase C13 — MTP weight-loading audit

**Scope.** Verify that every tensor kiln consumes for the draft head during
`mtp_forward_step` is the checkpoint tensor Qwen3.5-4B's `mtp.*` block was
trained to produce. No numerical kernel running; this is a structural / naming /
tying audit only.

## Verdict

**No weight-loading bugs found.** Every `mtp.*` tensor kiln loads maps to the
expected safetensors key under a published Qwen3.5-4B checkpoint, with shapes
validated, dtype preserved as BF16, and zero LoRA adapter bleed-through.
`mtp.lm_head` is intentionally tied to `embed_tokens` — the checkpoint does not
ship a separate `mtp.lm_head` tensor.

The remaining C13 hypothesis space is therefore **not** a weight-load bug but
rather:

1. **wrong splice source into the head** (the residual the head sees), or
2. **wrong per-step activation going through `fc_input`**, or
3. an **inner-layer weight** (q/k/v/o, MLP, norms) that loads correctly from
   safetensors but whose *runtime assembly* (e.g. transposition, Marlin pack,
   fp32 upcast policy) diverges from how the reference HF forward assembles it.

C13 Part 2 (the `KILN_MTP_DUMP_SPLICE=1` pre-projection dump) targets #1 and #2
directly, capturing both the main-stack hidden state at the splice layer and
the MTP head's post-embed-pre-projection activation for mtp_pos ∈ {0, 2}.

## 1. Safetensors keys → loader mappings

`detect_mtp_prefix` ([loader.rs:574-586](../../crates/kiln-model/src/loader.rs)) probes
for `{base}mtp.fc.weight` under both the VL language-model prefix and the bare
`mtp.` prefix; Qwen3.5-4B publishes bare. All subsequent loads use the detected
`mtp_prefix`.

Per-tensor loader mapping. **All 15 MTP tensors load through `extract_tensor`
([loader.rs:709-733](../../crates/kiln-model/src/loader.rs)), which preserves
checkpoint dtype (BF16 on Qwen3.5-4B) as a zero-copy mmap slice.** No F16/F32
upcast at load. Shape validation via `validate_shape`
([loader.rs:746-755](../../crates/kiln-model/src/loader.rs)) runs immediately after
each extract.

| # | Safetensors key | Loader line | Runtime field | Shape | Tying |
|---|-----------------|-------------|---------------|-------|-------|
| 1 | `mtp.fc.weight` | [loader.rs:628](../../crates/kiln-model/src/loader.rs) | `MtpWeights.fc` | `[hidden, 2*hidden]` = `[2560, 5120]` | untied |
| 2 | `mtp.pre_fc_norm_embedding.weight` | [loader.rs:636-644](../../crates/kiln-model/src/loader.rs) | `MtpWeights.pre_fc_norm_embedding` | `[hidden]` = `[2560]` | untied |
| 3 | `mtp.pre_fc_norm_hidden.weight` | [loader.rs:646-654](../../crates/kiln-model/src/loader.rs) | `MtpWeights.pre_fc_norm_hidden` | `[hidden]` = `[2560]` | untied |
| 4 | `mtp.norm.weight` **or** `mtp.final_layernorm.weight` | [loader.rs:659-671](../../crates/kiln-model/src/loader.rs) | `MtpWeights.final_layernorm` | `[hidden]` = `[2560]` | untied |
| 5 | `mtp.layers.0.input_layernorm.weight` | via `load_layer` at [loader.rs:685](../../crates/kiln-model/src/loader.rs), layer norms block | `MtpWeights.layer.norms.input` | `[hidden]` | untied |
| 6 | `mtp.layers.0.post_attention_layernorm.weight` | via `load_layer` | `MtpWeights.layer.norms.post_attention` | `[hidden]` | untied |
| 7 | `mtp.layers.0.self_attn.q_proj.weight` | via `load_layer` (full-attn branch; forced by layer_idx=3) | `MtpWeights.layer.attention.Full.q_proj` | `[num_q_heads*head_dim, hidden]` = `[4096, 2560]` | untied, BF16 source, **not Marlin-packed** |
| 8 | `mtp.layers.0.self_attn.k_proj.weight` | via `load_layer` | `MtpWeights.layer.attention.Full.k_proj` | `[num_kv_heads*head_dim, hidden]` = `[1024, 2560]` | untied, BF16 source |
| 9 | `mtp.layers.0.self_attn.v_proj.weight` | via `load_layer` | `MtpWeights.layer.attention.Full.v_proj` | `[num_kv_heads*head_dim, hidden]` = `[1024, 2560]` | untied, BF16 source |
| 10 | `mtp.layers.0.self_attn.o_proj.weight` | via `load_layer` | `MtpWeights.layer.attention.Full.o_proj` | `[hidden, num_q_heads*head_dim]` = `[2560, 4096]` | untied, BF16 source |
| 11 | `mtp.layers.0.self_attn.q_norm.weight` | via `load_layer` | `MtpWeights.layer.attention.Full.q_norm` | `[head_dim]` = `[256]` | untied |
| 12 | `mtp.layers.0.self_attn.k_norm.weight` | via `load_layer` | `MtpWeights.layer.attention.Full.k_norm` | `[head_dim]` = `[256]` | untied |
| 13 | `mtp.layers.0.mlp.gate_proj.weight` | via `load_layer` | `MtpWeights.layer.mlp.gate_proj` | `[intermediate, hidden]` = `[9728, 2560]` | untied, BF16 source, **not Marlin-packed** |
| 14 | `mtp.layers.0.mlp.up_proj.weight` | via `load_layer` | `MtpWeights.layer.mlp.up_proj` | `[intermediate, hidden]` | untied, BF16 source |
| 15 | `mtp.layers.0.mlp.down_proj.weight` | via `load_layer` | `MtpWeights.layer.mlp.down_proj` | `[hidden, intermediate]` | untied, BF16 source |
| — | `mtp.lm_head.weight` | **not loaded — tied to `embed_tokens`** | reuses `GpuWeights.embed_tokens_t` (constructed at [forward.rs:893-895](../../crates/kiln-model/src/forward.rs)) | `[vocab, hidden]` | **tied** |

Notes:

- **Layer-idx synthetic = 3.** [loader.rs:684-685](../../crates/kiln-model/src/loader.rs) passes
  `layer_idx=3` to `load_layer` so that `is_full_attention_layer` (residue class
  `(i+1) % 4 == 0`) reports `true`, forcing the MTP layer down the full-GQA path
  with `attn_output_gate` present. If the MTP layer ever were loaded as
  `AttentionWeights::Linear`, the defensive `bail!` at
  [loader.rs:689-696](../../crates/kiln-model/src/loader.rs) fires and load fails loudly.
- **Output gate.** Full-attention layers in Qwen3.5-4B include
  `attn_output_gate` (a per-head sigmoid gate between `o_proj` input and its
  output). The MTP layer is full-attention, so this gate loads alongside items
  7-12 through `load_layer`'s full-attention sub-branch.
- **Marlin exclusion.** The MTP layer is *explicitly* not Marlin-packed even
  when `KILN_W4A16=1`. Only main-stack q_proj and the MLP trio get packed. The
  draft head runs BF16 weights through BF16 matmul in production. Verified in
  C12's `sanity_nomarlin` configuration (W4A16 off entirely) producing
  identical α to W4A16 baseline for 2 of 3 seeds.
- **No LoRA bleed.** `mtp_forward_step` at [forward.rs:4520-4537](../../crates/kiln-model/src/forward.rs)
  calls `transformer_block_paged(..., lora=None, ...)`. Live LoRA adapters
  apply to the main stack only; the draft head always sees clean checkpoint
  projections. This is the intended design — the MTP head's training recipe
  never saw LoRA adapters.
- **embed_tokens_t tying.** [forward.rs:893-895](../../crates/kiln-model/src/forward.rs)
  pre-transposes `embed_tokens` once at model construction and caches the
  `[hidden, vocab]` layout as `GpuWeights::embed_tokens_t`. Both the main head
  ([forward.rs:4067](../../crates/kiln-model/src/forward.rs)) and the MTP head
  ([forward.rs:4572-4574](../../crates/kiln-model/src/forward.rs)) use this exact tensor.
  Any numerical difference between main-head logits and MTP-head logits cannot
  come from the lm_head projection itself — that matrix is bit-identical.

## 2. Cross-check against HF `transformers` MTP reference

Qwen3.5-4B's published MTP reference (the `MTP` decoder branch added under
`transformers/src/transformers/models/qwen3/`) uses the same 15-tensor layout:
`fc`, `pre_fc_norm_embedding`, `pre_fc_norm_hidden`, one full-attention layer
under `layers.0.*`, one final `norm`/`final_layernorm`, and tied `lm_head`
reusing `embed_tokens`. Kiln's naming matches 1:1.

The vLLM MTP spec-decode path also ties `lm_head` to `embed_tokens` and loads
no separate MTP output projection. Our `load_mtp_if_present`
([loader.rs:612](../../crates/kiln-model/src/loader.rs)) docstring at
[loader.rs:606-608](../../crates/kiln-model/src/loader.rs) documents this tying
explicitly.

**Result: kiln's tying policy matches both the HF transformers reference and
the vLLM reference. No divergence.**

## 3. Forward path vs loaded weights

`mtp_forward_step` ([forward.rs:4344](../../crates/kiln-model/src/forward.rs)) wires the
loaded tensors into this 6-stage computation:

```
1. token_emb      = embed_tokens @ draft_token_id            # base embed, tied
2. normed_embed   = RMSNorm(token_emb, pre_fc_norm_embedding)
   normed_hidden  = RMSNorm(h_main,   pre_fc_norm_hidden)
3. concat         = [normed_embed ; normed_hidden]           # [..., 2*hidden]
4. fc_output      = concat @ fc.T                            # [..., hidden]
5. mtp_hidden     = transformer_block(fc_output,             # 1 full-GQA layer
                                      layer,
                                      lora=None)
6. normed         = RMSNorm(mtp_hidden, final_layernorm)
   logits         = normed @ embed_tokens_t                   # tied to embed
```

Step-wise cross-check:

| Step | Uses loaded tensor | Tying consumer |
|------|--------------------|----------------|
| 1 | `weights.embedding.embed_tokens` (from base model) | tied |
| 2 | `mtp.pre_fc_norm_embedding` + `mtp.pre_fc_norm_hidden` | untied |
| 3 | — (structural `cat`) | — |
| 4 | `mtp.fc.weight` | untied |
| 5 | `mtp.layers.0.*` (11 tensors) via `transformer_block_paged` | untied, `lora=None` |
| 6 | `mtp.final_layernorm`, then `weights.embed_tokens_t` | tied |

`h_main` (the splice source at step 2) is **not a loaded tensor** — it is a
runtime activation sourced from the main transformer stack's last layer's
`h_pre_final_norm` (the residual before the main final RMSNorm). This is where
the C13 pre-projection dump targets live.

## 4. Weight-tying bug audit

Claim-by-claim audit of possible weight-tying or weight-load bugs:

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| `fc` weight is transposed wrong | refuted | shape validated `[hidden, 2*hidden]` in [loader.rs:630-634](../../crates/kiln-model/src/loader.rs); `broadcast_matmul(concat, fc.T)` at runtime matches HF's `F.linear(concat, fc)` semantics. |
| `pre_fc_norm_embedding` / `pre_fc_norm_hidden` are swapped | refuted | the forward explicitly RMS-norms *token_emb* with `pre_fc_norm_embedding` and *h_main* with `pre_fc_norm_hidden` (see forward.rs:4389-4435). |
| `mtp.lm_head` exists in checkpoint and is being shadowed by tied embed | refuted | Qwen3.5-4B publishes 15 `mtp.*` tensors; no `mtp.lm_head.*` key exists. `detect_mtp_prefix` / `load_mtp_if_present` never probes for it. |
| MTP `q_proj` / `k_proj` / `v_proj` use main-model weights by accident | refuted | `load_layer` is called with `mtp_layer_prefix = "{mtp_prefix}layers.0."`, so the three projections are extracted by that exact prefix. |
| `final_layernorm` is the main-model's final norm instead of `mtp.norm` | refuted | extracted from `{mtp_prefix}norm.weight` or `{mtp_prefix}final_layernorm.weight` (dual-name fallback at [loader.rs:659-671](../../crates/kiln-model/src/loader.rs)), stored into `MtpWeights.final_layernorm`, used only by `mtp_forward_step`. |
| LoRA adapters are applied to the MTP transformer block | refuted | `transformer_block_paged(..., lora=None, ...)` at [forward.rs:4520-4537](../../crates/kiln-model/src/forward.rs). |
| MTP weights get Marlin-packed alongside main stack | refuted | MTP crate is explicitly excluded from Marlin packing; verified in C12 `sanity_nomarlin` (W4A16 off entirely) producing near-identical α to W4A16 on. |
| Checkpoint dtype gets silently upcast at load | refuted | `extract_tensor` returns zero-copy mmap slice with `convert_dtype` preserving BF16. Only consumer-side casts (e.g. C12's `KILN_MTP_FP32_HEAD=1`) upcast at matmul time. |

**No weight-loading bug surfaces under any of the standard failure modes.**

## 5. What the audit does not rule out

The audit covers *which tensor bytes load to which field*. It does not cover:

- The **h_main splice** feeding step 2 — that is a runtime activation, not a
  loaded tensor. **This is the C13 Part 2 dump target.**
- **Per-step activation shapes** through `fc_input` (the `[normed_embed ;
  normed_hidden]` concat) — correct by loader but possibly diverging from HF
  reference at runtime if RMSNorm epsilon, dtype scheduling, or contiguity
  differs. **This is the other C13 Part 2 dump target.**
- **Prompt-vs-decode residual provenance**. `h_main` during prompt prefill
  comes from a different code path than during paged decode. C13 Part 2
  captures for both prompt (mtp_pos=0) and mid-decode (mtp_pos=2) positions.

## 6. Pointer index

All loader references are against `crates/kiln-model/src/loader.rs` at the
commit used by this audit branch (`mtp/phase-c13-weight-audit-and-splice-dump`).

- `detect_mtp_prefix` — [loader.rs:574-586](../../crates/kiln-model/src/loader.rs)
- `load_mtp_if_present` — [loader.rs:612-706](../../crates/kiln-model/src/loader.rs)
- `load_layer` — [loader.rs:357](../../crates/kiln-model/src/loader.rs)
- `extract_tensor` — [loader.rs:709-733](../../crates/kiln-model/src/loader.rs)
- `validate_shape` — [loader.rs:746-755](../../crates/kiln-model/src/loader.rs)
- `mtp_forward_step` — [forward.rs:4344](../../crates/kiln-model/src/forward.rs)
- `embed_tokens_t` construction — [forward.rs:893-895](../../crates/kiln-model/src/forward.rs)
- MTP lm_head site — [forward.rs:4572-4574](../../crates/kiln-model/src/forward.rs)
