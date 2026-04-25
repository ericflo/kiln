# Phase C17 — MTP Head-Forward First-Divergence Verdict

**Bench**: not re-measured. C17 is diagnosis-only on the same broken build C15/C16 audited (α ≈ 0.000–0.033 vs the 0.72 ship floor). No pod was acquired for this phase — every audit below is satisfied by static source reading against in-tree kiln code, in-tree dump comparators, and the upstream Qwen3-Next / Qwen3.5 MTP references.

**Scope (from C16 handoff)**: With the accept/reject machinery (`crates/kiln-server/src/speculative.rs`) and the MTP block-internal compute (Phase C13 splice inputs + Phase C14 splice outputs) both certified clean, the remaining structural candidate for α-collapse is the **input contract** between the base model and the MTP head. C17 audits the four narrow surfaces called out in the C16 handoff:

1. `lm_head` tying / transpose
2. `final_layernorm` (post-norm) wiring
3. The hidden-state frame the HF/vLLM/SGLang MTP reference consumes
4. `mtp_compare.py` rerun against an HF reference that consumes the same frame kiln consumes — i.e. whether the splice comparator can detect the divergence at all

## Verdict

**FIRST-DIVERGENCE = base→MTP `h_prev` reference-frame mismatch.**

kiln passes the BASE model's PRE-final-norm hidden state into MTP as `h_prev`; the upstream contract (vLLM Qwen3-Next, vLLM Qwen3.5, SGLang Qwen3.5, HF transformers `Qwen3NextModel`) consumes the BASE model's POST-final-norm `last_hidden_state`. The MTP head's `pre_fc_norm_hidden` is therefore being applied to a tensor in a different basis from what it was trained against, and the resulting `fc(concat(norm_emb, norm_h))` lives in a downstream-of-`fc` distribution the MTP transformer block was never trained on. This is the root cause of α-collapse. Full reference-frame analysis (every cross-check, every snippet) is in the companion document `docs/archive/phase-c/phase-c17/c17-reference-frame.md`.

The other three audits are CLEAN and stay clean under this verdict:

| Audit | Result | Notes |
|------|--------|-------|
| `lm_head` tying / transpose | CLEAN | MTP reuses the base `embed_tokens_t` (verified at `crates/kiln-model/src/forward.rs:4647`); transpose is the same one the base lm_head uses; this matched the C14 `c14__logits` pass-through cosine of 0.9999736. |
| `mtp.final_layernorm` (post-norm) wiring | CLEAN | MTP applies its own loaded `mtp.final_layernorm` weight to `mtp_hidden` at `crates/kiln-model/src/forward.rs:4639` (NVTX range `kiln/mtp/final_layernorm`), distinct from the base model's `weights.final_norm` at `forward.rs:912`. C14 `c14__post_norm` cos = 0.9999660 confirms the MTP-side norm matches the FP32 reference numerically. |
| HF MTP reference frame | WRONG-FRAME identified in kiln | vLLM `pre_fc_norm_hidden` is applied to `Qwen3NextModel.last_hidden_state` (post-norm); kiln applies it to a pre-norm tap. Documented in companion. |
| `mtp_compare.py` first-divergence re-run | NOT RUN — STATIC FINDING SUFFICIENT | The reference dump (`scripts/mtp_reference_dump.py:645`) consumes kiln's captured `h_main` verbatim, so it cannot detect a wrong-frame `h_main`. Re-running `mtp_compare` would re-print the same C13/C14 cleanliness verdicts (cos ≈ 0.99995 at every tap) at non-trivial pod cost. The fix is in the reference, not another comparator run. See the "Why no pod was needed" section. |

## Method

Each audit was satisfied by reading specific source locations against a specific upstream reference. No GPU work, no captures, no benchmarks. Total wall-clock for all four audits: well inside the 90 min cap. Total cost: $0 pod spend.

### Audit 1 — `lm_head` tying / transpose

Source location: `crates/kiln-model/src/forward.rs:4645–4648`:

```rust
let logits = {
    kiln_nvtx::range!(c"kiln/mtp/lm_head");
    normed.broadcast_matmul(&weights.embed_tokens_t)?
};
```

`weights.embed_tokens_t` is set up at `crates/kiln-model/src/forward.rs:897–911`:

```rust
let (embed_tokens, embed_tokens_t) = if matches!(device, Device::Metal(_)) {
    let embed_tokens_t =
        weight_to_transposed_tensor_2d(&weights.embedding.embed_tokens, device)?;
    let embed_tokens = dropped_weight_stub(&weights.embedding.embed_tokens, device)?;
    (embed_tokens, embed_tokens_t)
} else {
    let embed_tokens =
        weight_to_tensor(&weights.embedding.embed_tokens, device)?;
    let embed_tokens_t =
        cached_transpose_for_weight(&weights.embedding.embed_tokens, &embed_tokens, device)?;
    (embed_tokens, embed_tokens_t)
};
```

Both branches construct `embed_tokens_t` from the BASE model's single `weights.embedding.embed_tokens` tensor. The MTP head reuses that same `embed_tokens_t` (no separate `mtp.lm_head` is ever loaded — see `MtpWeights` at `crates/kiln-model/src/weights.rs:231–245`, which has no `lm_head` field and an explicit doc-comment "lm_head is tied to the base model's embed_tokens, so we do NOT store a separate lm_head tensor"). This matches the vLLM `Qwen3NextMTP` contract and is consistent with PR #331's weight-loading audit.

Numerical corroboration: Phase C14 (`docs/archive/phase-c/phase-c14/c14-splice-verdict.md`) measured `c14__logits` cos = 0.9999736 against an FP32 HF reference that uses the SAME tied tensor. A wrong tying or wrong transpose would have shown cos ≪ 1 at this tap; it didn't. CLEAN.

### Audit 2 — `mtp.final_layernorm` post-norm wiring

Source location: `crates/kiln-model/src/forward.rs:4637–4640`:

```rust
let normed = {
    kiln_nvtx::range!(c"kiln/mtp/final_layernorm");
    rms_norm(&mtp_hidden, &mtp.final_layernorm, config.rms_norm_eps)?
};
```

Three properties to verify:

1. The weight passed to `rms_norm` is `mtp.final_layernorm`, the MTP head's OWN final norm, not the base model's `weights.final_norm`. Confirmed by grep: `weights.final_norm` is referenced at `forward.rs:912, 3970, 4099, 4115, 4151, 5021, 5030`; `mtp.final_layernorm` only at `forward.rs:1143` (load) and `forward.rs:4639` (use). No accidental cross-wiring.
2. `mtp.final_layernorm` is loaded from the safetensors `mtp.*` namespace separately from the base `final_norm` tensor (`crates/kiln-model/src/forward.rs:1143`: `weight_to_tensor(&mtp_w.final_layernorm, device).context("mtp.final_layernorm")?`). Loaded tensor is owned by `MtpWeights` (`weights.rs:244`).
3. The norm is applied to `mtp_hidden` (the MTP transformer block's output), which is the correct frame for the inner MTP final norm in the vLLM/SGLang reference (`Qwen3_5MultiTokenPredictor.forward` ends with `hidden_states, _ = self.norm(hidden_states, residual)` immediately before the tied lm_head). C14 `c14__post_norm` measured cos = 0.9999660 against the FP32 reference applying the same MTP final norm.

CLEAN. The post-norm wiring is correct.

### Audit 3 — HF MTP reference frame

Full chain documented in `docs/archive/phase-c/phase-c17/c17-reference-frame.md`. Summary:

- vLLM `Qwen3NextMultiTokenPredictor.forward(hidden_states=…)` and `Qwen3_5MultiTokenPredictor.forward(hidden_states=…)` apply `self.pre_fc_norm_hidden(hidden_states)` to the incoming arg.
- That arg is the BASE model's `last_hidden_state` from `Qwen3NextModel.forward`, which ends with `hidden_states, _ = self.norm(hidden_states, residual); return hidden_states` — POST-final-norm.
- HF transformers `Qwen3NextModel.forward` (line 968 of `modeling_qwen3_next.py`) applies `hidden_states = self.norm(hidden_states)` then returns `MoeModelOutputWithPast(last_hidden_state=hidden_states, …)` — also POST-norm.
- HF transformers IGNORES MTP weights entirely (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` at line 879), so transformers itself has no in-tree MTP head; the only working HF-aligned MTP references are vLLM and SGLang, and both consume `last_hidden_state` (post-norm).
- SGLang `Qwen3_5ForCausalLMMTP.forward` reads `forward_batch.spec_info.hidden_states`, populated by the spec-decode worker from the base's `last_hidden_state` — same contract.

kiln's `LmHeadMode::FullWithLastHidden` (`crates/kiln-model/src/forward.rs:5035–5057`) returns `last_hidden = hidden.narrow(1, seq_len - 1, 1)?.contiguous()?` BEFORE applying `weights.final_norm`. That pre-norm tensor is what `mtp_forward_step` then passes to `pre_fc_norm_hidden`. The capture tap's name in dumps (`h_pre_final_norm`) is itself an admission of the frame.

WRONG-FRAME identified.

### Audit 4 — `mtp_compare.py` against same-frame HF reference

The C16 handoff suggested "Run `scripts/mtp_compare.py` against an HF reference that consumes the SAME frame kiln consumes; identify FIRST tap divergence." On reading `scripts/mtp_reference_dump.py:644–710` it became clear that this is exactly what the existing comparator already does, and that running it again costs pod time without producing new information:

```python
h_main = kiln["h_main"].to(torch.float32)            # pre-final-norm, from kiln dump
norm_emb_w = w.pre_fc_norm_embedding
norm_h_w = w.pre_fc_norm_hidden
norm_h = rms_norm(h_main, norm_h_w, args.rms_eps)
norm_emb = rms_norm(token_emb, norm_emb_w, args.rms_eps)
fc_input = torch.cat([norm_emb, norm_h], dim=-1)
fc_output = torch.matmul(fc_input, w.fc_weight.t())
post_layer = mtp_inner_block(pre_layer, …)
post_final_ln = rms_norm(post_layer, w.final_layernorm, args.rms_eps)
mtp_logits = torch.matmul(post_final_ln, w.embed_tokens.t())
```

The reference dump consumes kiln's captured `h_main` verbatim and runs the SAME MTP block math the kiln side runs, with the SAME wrong frame in. So `mtp_compare.py` has already been certifying both sides agree — that's exactly what the C13 / C14 verdict tables show (cos ≈ 0.99995 at every tap, both inputs and outputs of the splice). Re-running `mtp_compare.py` against a same-frame reference would simply re-print those numbers.

To make the comparator detect the first divergence, the FIX has to land first — either:

1. kiln must capture `h_main` AFTER `weights.final_norm` is applied (the C18 fix), OR
2. `mtp_reference_dump.py` must apply `weights.final_norm` to its kiln-derived `h_main` before passing it to `pre_fc_norm_hidden`.

Either change converts `mtp_compare.py` into a frame-divergence detector. Until then, more `mtp_compare` runs add no signal. Static finding is sufficient.

NOT RUN — by design. $0 pod spend.

## Why no pod was needed

The four audits are all answerable by reading source. Three of them are tying / wiring questions that have unambiguous answers in the kiln source tree, and one (the `mtp_compare` rerun) was already known from C13/C14 to be incapable of catching a wrong-frame `h_main`. Acquiring an A6000 lease to re-run a comparator that already reports cos = 0.99995 at every tap would have burned $5–$8 of pod time for zero new evidence; the wall-clock budget is preserved for the C18 fix-and-verify cycle that actually needs the GPU.

## Numerical fingerprint corroborating the verdict

From `docs/archive/phase-c/phase-c15/c15-h-main-drift-verdict.md` (FP32 reference column):

| step | kiln_norm | ref_norm | ratio |
|-----:|----------:|---------:|------:|
| 0    | 67.77     | 155.87   | 2.30× |
| 1    | 69.44     | 159.40   | 2.30× |
| 2    | 63.46     | 150.51   | 2.37× |
| 3    | 80.25     | 153.22   | 1.91× |
| 4    | 70.92     | 158.02   | 2.23× |
| 5    | 69.11     | 146.79   | 2.12× |
| 6    | 70.44     | 153.16   | 2.17× |
| 7    | 62.71     | 152.78   | 2.44× |

Ratios cluster around `mean(|1 + w_final|) / 1 ≈ 2.0–2.4` for Qwen3-Next's `(1 + w) · x · rsqrt(mean(x²)+ε)` final RMSNorm. This is the per-element rescaling produced by the MISSING base-model final norm — the exact magnitude shift predicted by the wrong-frame hypothesis. Cosine remaining at ≈ 0.96–0.99 (rather than collapsing) is also expected: `pre_fc_norm_hidden` divides by local RMS, which absorbs the magnitude part but cannot recover the per-element `(1 + w_i)` directional content.

## What this leaves for C18

C17 is diagnosis-only. C18 owns the fix and the bench-validated re-measurement of α. The companion `c17-reference-frame.md` includes the proposed minimal fix (drop the second `rms_norm(&hidden, …)` call in `LmHeadMode::FullWithLastHidden` and re-use the same normed slice for both the logits projection and the `h_prev` return), the open question about whether `LmHeadMode::LastRowWithLastHidden` needs the same treatment, and the matching one-line guard for `scripts/mtp_reference_dump.py` so the comparator stays valid through the transition.

## Artifacts

- This document.
- Companion: `docs/archive/phase-c/phase-c17/c17-reference-frame.md` (full reference-frame cross-check, numerical fingerprint, upstream snippet pins).
- Source pointers: `crates/kiln-model/src/forward.rs:897–911, 1143, 4520–4647, 5025–5070`; `crates/kiln-model/src/weights.rs:220–245`; `scripts/mtp_reference_dump.py:644–710`; `scripts/c15_h_main_drift_audit.py:161–203`.
- Upstream pins (snapshot 2026-04-21): vLLM `qwen3_next_mtp.py:98–134`, vLLM `qwen3_next.py:451–531`, vLLM `qwen3_5_mtp.py:121–157`, SGLang `qwen3_5_mtp.py:39–155`, HF transformers `modeling_qwen3_next.py:879, 918–973`.
- Predecessor verdicts: `docs/archive/phase-c/phase-c14/c14-splice-verdict.md`, `docs/archive/phase-c/phase-c15/c15-h-main-drift-verdict.md`, `docs/archive/phase-c/phase-c16/c16-accept-reject-verdict.md`.
