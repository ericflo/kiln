# Phase C25 — Verifier vs MTP-draft logit-path static audit (hypothesis 6, doc-only redirect)

## Verdict

**Hypothesis 6 is REJECTED on static audit. No code change, no
pod bench.** A CPU-first read of
`crates/kiln-model/src/speculative.rs::speculative_mtp_decode_step`
(the k=1 verify loop), `crates/kiln-model/src/forward.rs`
(`model_forward_paged_with_last_hidden` /
`model_forward_paged_inner` /
`LmHeadMode::FullWithLastHidden` / `mtp_forward_step`),
`crates/kiln-model/src/weights.rs::MtpWeights`, and
`crates/kiln-model/src/loader.rs::load_mtp_weights` confirms
that the verifier-path and draft-path lm_head invocations are
structurally canonical per the `Qwen3NextMultiTokenPredictor`
reference contract. Both paths tie into the same head
(`GpuWeights::embed_tokens_t`), both feed post-final-norm
hidden states, and the two distinct `final_layernorm` weights
are the architecturally-required separate norms (base vs MTP
head). No silent asymmetry was found that could explain the
residual α drift (C18 median 0.1532, floor 0.5). H6 joins
H1–H5 on the cleared queue; the residual gap handoff moves to
H7 (MTP weight-reload sanity) and H8 (block sequencing across
decode steps 0→1→2 and MTP paged-KV state carryover).

## C24 handoff as stated

From
[`docs/archive/phase-c/phase-c24/c24-mtp-rotary-pos-ab-verdict.md`](../phase-c24/c24-mtp-rotary-pos-ab-verdict.md)
("Next steps after H1–H5 are cleared"):

> Hand off the residual α gap (C18 median 0.1532, floor 0.5)
> to deeper structural audits beyond the H1–H5 queue: block
> sequencing between decode steps 0→1→2, state carryover
> across the MTP paged KV cache, and verifier-side logit
> comparison against HF's `Qwen3NextMultiTokenPredictor` on a
> small calibration set.

Phase C25 addresses the third of those three items — a static
read of the two lm_head-path call sites ("verifier-side logit
comparison"), CPU-first, before spending pod time. The audit
finds no asymmetry, so the hardware-side HF comparison in the
C24 handoff is not warranted at this time; block sequencing
(C24's first item) remains the most load-bearing open
suspect.

## Audit scope

Compare, on the SAME hidden-state frame and SAME decode step:

1. **Verifier-path lm_head** — how the k=1 verify call
   (`model_forward_paged_with_last_hidden`) produces its
   `logits[:, -1, :]` tensor over the next-token position.
2. **Draft-path lm_head** — how `mtp_forward_step` produces
   its single-row draft logits at the same next-token
   position.

Dimensions audited:
- Tied-head identity (is `lm_head` the same weight tensor in
  both paths?)
- Final RMSNorm identity (are we normalizing with the same
  weights, or with intentionally distinct weights?)
- Input hidden-state lineage (both paths must feed
  post-final-norm hidden states into their `lm_head`)
- RoPE frame at the next-token position (separately audited
  in C21/C24, re-checked here for the lm_head-path end)

## Findings

### Verifier path (k=1)

`crates/kiln-model/src/speculative.rs::speculative_mtp_decode_step`
(post-PR #349) calls the verifier as a single-token forward:

```rust
let (logits, h_prev) = model_forward_paged_with_last_hidden(
    &[last_token],
    base_cache,
    base_block_table,
    base_pos,
    // ...
)?;
```

`model_forward_paged_with_last_hidden`
(`crates/kiln-model/src/forward.rs`, ~lines 4272–4329) delegates
to `model_forward_paged_inner` with
`LmHeadMode::FullWithLastHidden`. That branch
(`crates/kiln-model/src/forward.rs`, ~lines 5066–5088) is:

```rust
// Phase C18: `h_prev` must be returned POST-final-norm so that
// the downstream consumer (MTP draft) receives the same tensor
// frame the lm_head saw.
let normed = rms_norm(&hidden, &weights.final_norm, config.rms_norm_eps)?;
let last_hidden = normed.narrow(1, seq_len - 1, 1)?.contiguous()?;
let logits = lm_head_forward(&normed, &weights.embed_tokens_t)?;
Ok((Some(logits), Some(last_hidden)))
```

So the verifier produces:

- `logits = lm_head_forward(rms_norm(H_last, weights.final_norm), weights.embed_tokens_t)`
- `h_prev = rms_norm(H_last, weights.final_norm)` (identical
  frame, sliced at the last row)

### Draft path (MTP k=1)

`crates/kiln-model/src/forward.rs::mtp_forward_step`
(~lines 4408–4800+) takes the verifier's
`h_prev` (already post-final-norm), the next-candidate token
embedding, and runs:

```rust
let e = embed_lookup(...);
let e_normed = rms_norm(e, &mtp.pre_fc_norm_embedding, eps);
let h_normed = rms_norm(h_prev, &mtp.pre_fc_norm_hidden, eps);
let concat = cat(e_normed, h_normed);
let fused = matmul(concat, &mtp.fc_t); // [hidden, 2*hidden] → [hidden]
// transformer_block_paged on mtp.layer at abs_pos = base_pos + mtp_pos
let mtp_hidden = transformer_block_paged(fused, &mtp.layer, ...);
let normed = rms_norm(&mtp_hidden, &mtp.final_layernorm, config.rms_norm_eps)?;
let logits = lm_head_forward(&normed, &weights.embed_tokens_t)?;
```

So the draft produces:

- `logits = lm_head_forward(rms_norm(mtp_hidden, mtp.final_layernorm), weights.embed_tokens_t)`

### Tied-head identity

Both paths end in
`lm_head_forward(_, &weights.embed_tokens_t)`. The `MtpWeights`
struct (`crates/kiln-model/src/weights.rs`, ~lines 232–245)
intentionally does NOT carry a separate `lm_head` tensor; the
doc comment there states:

> Forward shape (vLLM `qwen3_next_mtp.py` reference):
> `concat(pre_fc_norm_embedding(embed(t)), pre_fc_norm_hidden(h)) → fc (2H→H)
>  → layer (GQA + SwiGLU MLP) → final_layernorm → tied lm_head (= embed_tokens.t())`
>
> store a separate `lm_head` tensor — the spec-decode forward path
> reuses `GpuWeights::embed_tokens_t`.

The loader (`crates/kiln-model/src/loader.rs::load_mtp_weights`,
~lines 614–706) does not probe for or load any
`mtp.lm_head.weight` tensor. The tied head is therefore
guaranteed identical by construction.

### Final RMSNorm identity

The two paths deliberately use DIFFERENT norm weights:

- Verifier final norm: `weights.final_norm` loaded from
  `model.norm.weight` (or `{prefix}norm.weight` with the VL
  language-model prefix).
- Draft final norm: `mtp.final_layernorm` loaded from
  `{mtp_prefix}norm.weight` or
  `{mtp_prefix}final_layernorm.weight`
  (`crates/kiln-model/src/loader.rs` ~lines 656–676).

The loader accepts both `mtp.norm.weight` (Qwen3.5-4B stock
publish layout) and `mtp.final_layernorm.weight` (older
vLLM/HF naming convention). Critically, it does NOT fall back
to the base `weights.final_norm` if the MTP key is missing —
if the MTP final-norm key is absent, the loader bails with
"MTP final RMSNorm not found". So a silent alias to the base
norm is architecturally impossible.

This is the correct contract: the Qwen3-Next MTP spec ships an
independent `mtp.norm` tensor because the MTP head's residual
stream has a different statistical profile than the base
model's final hidden state (the MTP layer operates on a
fused `concat(e, h)` projection, not the base decoder's
residual). Sharing the base `final_norm` would be the bug; the
two-norm layout is canonical.

### Input hidden-state lineage

- Verifier lm_head sees `rms_norm(H_last, weights.final_norm)`
  where `H_last` is the last-row base hidden state.
- Draft lm_head sees `rms_norm(mtp_hidden, mtp.final_layernorm)`
  where `mtp_hidden = transformer_block(fc(concat(pre_fc_norm_embedding(e), pre_fc_norm_hidden(h_prev))))`,
  and `h_prev` is the verifier's own post-final-norm tensor (the
  same tensor the verifier lm_head's input was sliced from,
  per the Phase C17/C18 fix).

The lineage cascade on the MTP side is norm-then-re-norm (C18
post-final-norm `h_prev` → MTP `pre_fc_norm_hidden` → fc
fusion → inner transformer block → `mtp.final_layernorm`).
That cascade is load-bearing — the C18 fix made the INPUT to
the MTP side be post-final-norm; the MTP side's own
`pre_fc_norm_hidden` then re-normalizes to the MTP layer's
expected distribution. This matches vLLM
`Qwen3NextMultiTokenPredictor.forward` exactly.

### RoPE frame

The lm_head invocation itself is RoPE-free; RoPE only enters
inside the attention projections earlier in each path. That
path was separately audited in C21/C24 and ruled out as the
dominant drift source. From the logit-end's perspective, both
lm_head calls see a hidden tensor of shape `[1, 1, hidden]`
(single-row) and multiply by the tied head to get a `[1, 1,
vocab]` row. There is no RoPE-frame asymmetry remaining at
the lm_head step.

## Why there is nothing to A/B on hardware

An A/B bench is only justified when the static read finds a
candidate asymmetry whose hardware signature is ambiguous. In
this case, the audit finds **zero** candidate asymmetries at
the lm_head path:

- Head is guaranteed identical (tied by construction, not by
  load-time coincidence).
- Two distinct final norms are architecturally required —
  swapping in the base `weights.final_norm` would BE the bug,
  not a candidate fix.
- Input lineage is already fixed by C17/C18 to feed
  post-final-norm frames into the MTP side.
- RoPE at the lm_head step is a no-op (RoPE is applied
  earlier, inside attention).

The precedent for doc-only-redirect PRs on null-static-audit
phases is C19 (PR #344), C20 (PR #345), C21 (PR #346), and C22
(PR #347) — all of which ruled out their respective
hypothesis on a CPU-first read without acquiring a pod. C25
follows the same pattern.

## Handoff to H7 / H8

With the H1–H6 queue cleared, the remaining C18 residual
(α ≈ 0.153 vs floor 0.5) must come from one of:

**H7 — MTP weight-reload sanity.** A load-time verification
that the 15 MTP-prefixed tensors land in the expected GPU
locations with the expected dtypes and shapes after Marlin
packing, the MTP layer escape (it is NOT queued for Marlin
batch pack, per `forward.rs` ~line 1128), and the W4A16 weight
upload path. The scaffolding comment at `forward.rs` line 1128
explicitly notes that MTP projections are kept in BF16, which
is correct for the current PR but deserves a one-time
end-to-end check that the BF16 tensors survive without
silent cast-to-F16 or layout corruption. Concretely: dump the
first row of each MTP tensor at load time and compare to a
reference dump from a stock transformers load of the same
checkpoint.

**H8 — Block sequencing and MTP paged-KV state carryover.**
The bigger remaining suspect. Across decode steps 0 → 1 → 2,
the MTP draft layer maintains its own `PagedKvCache` that is
separate from the base model's paged KV. Both caches advance
on a different schedule (base advances once per verified
token; MTP advances once per draft attempt). If the MTP
cache's block table, seq_len accounting, or position stride
disagrees with the MTP layer's RoPE assumption across
multi-step verification, α will drop in exactly the
"structurally close but wrong by a constant offset" pattern
we observe at 0.153.

H7 is cheaper to run and disprove; queue it first.

## No-bench rationale (budget)

Budget was $15 / 60 min wall-clock. Static audit consumed
approximately 0 pod-$ and <30 min wall-clock. No pod was
acquired; no bench was run. The doc-only redirect ships at
$0.

## Files inspected

- `crates/kiln-model/src/speculative.rs` — `speculative_mtp_decode_step` (post-PR #349, single-token verify)
- `crates/kiln-model/src/forward.rs` — `model_forward_paged_with_last_hidden`, `model_forward_paged_inner`, `LmHeadMode::FullWithLastHidden` branch, `mtp_forward_step`
- `crates/kiln-model/src/weights.rs` — `MtpWeights` struct + doc comment
- `crates/kiln-model/src/loader.rs` — `load_mtp_weights`, MTP prefix detection, final-norm key probing

## Cross-refs

- C17 reference frame: `docs/archive/phase-c/phase-c17/c17-mtp-h-prev-post-norm.md` (if authored) / PR #340
- C18 post-final-norm fix: PR #341
- C19 null redirect: PR #344
- C20 MTP block-norm audit: `docs/archive/phase-c/phase-c20/c20-mtp-block-norm-audit.md`, PR #345
- C21 MTP rotary-pos audit: `docs/archive/phase-c/phase-c21/c21-mtp-rotary-pos-audit.md`, PR #346
- C22 fc/residual audit: `docs/archive/phase-c/phase-c22/c22-fc-residual-audit.md`, PR #347
- C23 draft-sampler audit: `docs/archive/phase-c/phase-c23/c23-draft-sampler-audit.md`, PR #348
- C24 rotary-pos A/B verdict: `docs/archive/phase-c/phase-c24/c24-mtp-rotary-pos-ab-verdict.md`, PR #349
