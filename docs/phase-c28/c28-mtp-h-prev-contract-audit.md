# Phase C28 — `h_prev` post-norm contract static audit (H10, doc-only)

## Verdict

**H10 RULED OUT. No code change. No pod spend.** All four steps of the
vLLM `Qwen3NextMultiTokenPredictor.forward` contract match kiln's current
source (HEAD `72d75c0`) bit-for-bit structurally: kiln returns `h_prev`
in the post-final-norm frame (C18 fix), applies the two dedicated
`pre_fc_norm_*` RMSNorms to the correct halves with `config.rms_norm_eps`,
concatenates with embedding first / hidden second, and projects through
the `[H, 2H]` `fc` weight as a cached transpose. The residual α gap
(median 0.1532, ship floor 0.5) must lie **outside** the `h_prev`
post-norm splice pipeline. Hand the residual α gap to Phase C29: an
empirical dump of kiln vs. vLLM MTP top-K logits on a fixed set of
low-α prompts, to catch whichever numerical drift is still
corrupting draft-token sampling.

## Hypothesis under test

From Phase C27 (`docs/phase-c27/c27-mtp-block-sequencing-audit.md`,
handoff section):

> **H10 — `h_prev` post-final-norm contract drift.** C18 fixed the frame
> but only verified the fix on two call sites and the α bench. We never
> re-audited the full 4-step vLLM chain — (1) base → `model.norm` →
> h_prev, (2) `pre_fc_norm_hidden(h_prev)`, (3) concat with
> `pre_fc_norm_embedding(embed(token))`, (4) `fc` projection — step by
> step against kiln's source. A per-step drift anywhere in that chain
> (wrong weight tensor, wrong eps, wrong order, wrong residual contract,
> wrong dtype) would land exactly where we are: α ≈ 0.15, on-manifold,
> no crash.

Static audit only — no pod spend. Each of the four steps is checked for
six sub-properties:

1. **Present** — the step exists at all
2. **Right weight tensor** — the RMSNorm / Linear weight is sourced from
   the correct checkpoint key
3. **Right eps** — RMSNorm eps matches `config.rms_norm_eps`
4. **Right order** — relative ordering within the pipeline
5. **Right residual contract** — whether residual is fused or not, and
   whether that matches vLLM
6. **Right dtype** — input/output dtypes match vLLM's bf16 contract

Trivially-clean sub-properties are collapsed; anything suspicious is
called out with the precise line range.

## vLLM reference (canonical source of truth)

`vllm/model_executor/models/qwen3_next_mtp.py`
(`Qwen3NextMultiTokenPredictor.forward`, lines 98–134):

```python
def forward(self, input_ids, positions, hidden_states, ...):
    if get_pp_group().is_first_rank:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
        inputs_embeds  = self.pre_fc_norm_embedding(inputs_embeds)   # Step 3a
        hidden_states  = self.pre_fc_norm_hidden(hidden_states)      # Step 2
        hidden_states  = torch.cat([inputs_embeds, hidden_states], dim=-1)  # Step 3b
        hidden_states  = self.fc(hidden_states)                      # Step 4
        residual = None
    ...
    hidden_states, residual = self.layers[current_step_idx](
        positions=positions,
        hidden_states=hidden_states,
        residual=residual,
    )
    ...
    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states
```

Module-init (lines 59–93):

```python
self.embed_tokens          = VocabParallelEmbedding(V, H)
self.fc                    = ColumnParallelLinear(H * 2, H, bias=False)
self.layers                = ModuleList([Qwen3NextDecoderLayer(...,
                               layer_type="full_attention")])
self.norm                  = Qwen3NextRMSNorm(H, eps=config.rms_norm_eps)
self.pre_fc_norm_hidden    = Qwen3NextRMSNorm(H, eps=config.rms_norm_eps)
self.pre_fc_norm_embedding = Qwen3NextRMSNorm(H, eps=config.rms_norm_eps)
```

`hidden_states` arriving in `forward` is the base model's
`last_hidden_state` — which, per the vLLM `Qwen3NextModel` contract, is
already **post-final-norm** (the base stack runs `self.norm(...)` before
returning).

Step-by-step: this gives us the 4-step chain:

1. **Step 1.** Base stack → `model.norm` → `h_prev` (post-final-norm frame)
2. **Step 2.** `h_prev ← pre_fc_norm_hidden(h_prev)`
3. **Step 3.** `concat ← [pre_fc_norm_embedding(embed(tok)), h_prev]` along
   the hidden dim
4. **Step 4.** `fused ← fc(concat)` — `ColumnParallelLinear(2H, H)`,
   equivalent to `concat @ fc.weight.T`

## Preflight

- HEAD: `72d75c0` (Phase C27 merged). Tree clean.
- No overlapping open PRs. (`gh pr list --repo ericflo/kiln --state open`
  returned empty at audit start.)
- Prior C28 search: `gh pr list --state all --search "C28 OR h_prev in:title"`
  returned only #245 (native MTP wiring WIP, merged), #340 (C17
  first-divergence audit, merged), #341 (C18 h_prev post-norm fix,
  merged). No prior C28 doc or open h_prev work.
- Files inspected:
  - `crates/kiln-model/src/forward.rs` (HEAD) — `mtp_forward_step`
    (lines 4380–4700), `model_forward_paged_with_last_hidden` LM-head
    arms (lines 5040–5110), `MtpGpuWeights` struct (lines 195–215)
  - `crates/kiln-model/src/loader.rs` (HEAD) — `load_mtp_if_present`
    (lines 612–755), the `pre_fc_norm_*` extraction block and fc/norm
    key handling
  - vLLM reference: `/tmp/vllm_qwen3_next_mtp.py`
  - Prior-phase docs: `docs/phase-c18/c18-h-prev-post-norm-fix.md`,
    `docs/phase-c19/c19-fc-norm-audit.md`,
    `docs/phase-c27/c27-mtp-block-sequencing-audit.md`

## Step 1 — Base → `model.norm` → `h_prev`

vLLM: `hidden_states` arriving in MTP `forward` is already
post-`model.norm` (base stack applies `self.norm(hidden_states, residual)`
before returning `last_hidden_state`).

Kiln: `model_forward_paged_with_last_hidden` populates `h_prev` in the
two MTP-consuming variants. Both arms now compute `final_norm` before
slicing (`FullWithLastHidden` at `forward.rs:5066-5088`,
`LastRowWithLastHidden` at `forward.rs:5089-5107`):

```rust
// FullWithLastHidden — lines 5075-5081
let normed = {
    kiln_nvtx::range!(c"kiln/final_norm");
    rms_norm(&hidden, &weights.final_norm, config.rms_norm_eps)?
};
let last_hidden = normed.narrow(1, seq_len - 1, 1)?.contiguous()?;
if crate::mtp_debug::is_h_main_capture_armed() {
    let _ = crate::mtp_debug::capture_h_main_tap("h_post_final_norm", &last_hidden);
}
```

```rust
// LastRowWithLastHidden — lines 5094-5101
let last_pre_norm = hidden.narrow(1, seq_len - 1, 1)?.contiguous()?;
let last_hidden = {
    kiln_nvtx::range!(c"kiln/final_norm");
    rms_norm(&last_pre_norm, &weights.final_norm, config.rms_norm_eps)?
};
```

Sub-property matrix:

| # | Property | vLLM | Kiln | Status |
| --- | --- | --- | --- | --- |
| 1 | Present | `self.norm(hidden_states, residual)` in base stack | `rms_norm(..., weights.final_norm, ...)` in both LM-head arms | ✅ CLEAN |
| 2 | Weight tensor | `model.norm.weight` | `weights.final_norm` (loaded from `model.norm.weight`, see loader.rs) | ✅ CLEAN |
| 3 | Eps | `config.rms_norm_eps` | `config.rms_norm_eps` (passed through) | ✅ CLEAN |
| 4 | Order | After base stack, before returning to MTP | After base `hidden`, before `narrow`/`lm_head`; Phase C18 fix applied once and fed both paths | ✅ CLEAN |
| 5 | Residual | vLLM fuses residual into `self.norm(hidden, residual)` | Kiln's `transformer_block_paged` already sums residual into `hidden` before returning; passing `hidden` (not `(hidden, residual)`) to a non-fused `rms_norm` is equivalent | ✅ CLEAN (see note) |
| 6 | Dtype | bf16 | bf16 (kiln inherits from base `hidden`, no explicit cast) | ✅ CLEAN |

Residual note: vLLM's `Qwen3NextRMSNorm(hidden, residual)` is a fused
`hidden + residual → normed, residual` kernel; the two outputs feed the
next block's residual stream. At the **end** of the stack, the returned
`hidden` has already absorbed the residual, so the post-`self.norm`
tensor that vLLM calls `last_hidden_state` is `rms_norm(hidden +
residual)`. Kiln's `transformer_block_paged` returns a single `hidden`
that has already accumulated the residual (same pattern as the base
model block, which has been running for months without divergence), so
`rms_norm(hidden, final_norm, eps)` produces the same tensor
numerically. Phase B6 bit-for-bit compare already verified the base
stack's `last_hidden_state` matches HF reference to within dtype tol.

**Step 1 verdict: CLEAN.** C18 landed the fix and it is still in place.

## Step 2 — `pre_fc_norm_hidden(h_prev)`

vLLM (line 112): `hidden_states = self.pre_fc_norm_hidden(hidden_states)`.

Kiln (`forward.rs:4515-4518`, production path with `swap_fc_norms=false`):

```rust
let norm_h = {
    kiln_nvtx::range!(c"kiln/mtp/pre_fc_norm_hidden");
    rms_norm(h_prev, norm_h_weight, config.rms_norm_eps)?
};
```

`norm_h_weight` resolves to `&mtp.pre_fc_norm_hidden` when the A/B swap
flag is off (`forward.rs:4502-4507`; default behaviour).

Sub-property matrix:

| # | Property | vLLM | Kiln | Status |
| --- | --- | --- | --- | --- |
| 1 | Present | `self.pre_fc_norm_hidden(hidden_states)` | `rms_norm(h_prev, norm_h_weight, ...)` | ✅ CLEAN |
| 2 | Weight tensor | `mtp.pre_fc_norm_hidden.weight` | `mtp.pre_fc_norm_hidden` (loader.rs:641-647 — `[hidden]` shape validated, no fallback) | ✅ CLEAN |
| 3 | Eps | `config.rms_norm_eps` | `config.rms_norm_eps` | ✅ CLEAN |
| 4 | Order | Before concat, after receiving `h_prev` post-model.norm | Same — between Step 1 (post-final-norm h_prev) and Step 3 (concat) | ✅ CLEAN |
| 5 | Residual | Non-fused (this norm takes no residual arg in vLLM — the `pre_fc_norm_*` variants deliberately do not take residual; they are the "pre-concat" normalisation layer) | Non-fused (kiln also calls the plain `rms_norm`) | ✅ CLEAN |
| 6 | Dtype | bf16 | bf16 (kiln passes `h_prev` through unchanged) | ✅ CLEAN |

**Step 2 verdict: CLEAN.** Already covered by C19 hypothesis-1
refutation (`docs/phase-c19/c19-fc-norm-audit.md`). Re-verified here
against HEAD — no regression.

## Step 3 — `concat([pre_fc_norm_embedding(embed(tok)), h_prev], dim=-1)`

vLLM (lines 109–113):

```python
inputs_embeds = self.embed_input_ids(input_ids)
...
inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
hidden_states = self.pre_fc_norm_hidden(hidden_states)
hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
```

Concat order: **embedding first**, **hidden second**, along the last
(hidden) dim.

Kiln (`forward.rs:4487-4527`):

```rust
// 1. Token embedding for the draft token.
let token_ids = [draft_token_id];
let token_emb = embedding_lookup_from_weights(&token_ids, weights)?; // [1, H]
let token_emb = token_emb.unsqueeze(0)?;                             // [1, 1, H]

// 2-3. Dual RMSNorms.
let (norm_emb_weight, norm_h_weight) = if swap_fc_norms {
    (&mtp.pre_fc_norm_hidden, &mtp.pre_fc_norm_embedding)
} else {
    (&mtp.pre_fc_norm_embedding, &mtp.pre_fc_norm_hidden)
};
let norm_emb = rms_norm(&token_emb, norm_emb_weight, config.rms_norm_eps)?;
let norm_h   = rms_norm(h_prev,     norm_h_weight,   config.rms_norm_eps)?;

// 4. Concat along the hidden dim and fuse.
let concat = Tensor::cat(&[&norm_emb, &norm_h], 2)?.contiguous()?;
```

Sub-property matrix:

| # | Property | vLLM | Kiln | Status |
| --- | --- | --- | --- | --- |
| 1 | Present (3a embed norm) | `self.pre_fc_norm_embedding(inputs_embeds)` | `rms_norm(&token_emb, norm_emb_weight, eps)` | ✅ CLEAN |
| 2 | Weight tensor (3a) | `mtp.pre_fc_norm_embedding.weight` | `mtp.pre_fc_norm_embedding` (loader.rs:636-640) | ✅ CLEAN |
| 3 | Eps (3a) | `config.rms_norm_eps` | `config.rms_norm_eps` | ✅ CLEAN |
| 4 | Embedding source | `VocabParallelEmbedding` (tied to base embed_tokens via shared weights) | `embedding_lookup_from_weights(&[draft_token_id], weights)` — uses the base `embed_tokens` (tied), not a separate MTP embedding | ✅ CLEAN |
| 5 | Concat order | `[inputs_embeds, hidden_states]` (embedding FIRST, hidden SECOND) | `[&norm_emb, &norm_h]` (embedding FIRST, hidden SECOND) | ✅ CLEAN |
| 6 | Concat dim | `dim=-1` (hidden dim) on `[1, 1, H]` → `[1, 1, 2H]` | `dim=2` on `[1, 1, H]` → `[1, 1, 2H]` (equivalent: dim 2 IS dim -1 for a 3-D tensor) | ✅ CLEAN |
| 7 | Dtype | bf16 on both halves | bf16 on both halves | ✅ CLEAN |

The concat-order property was the highest-risk sub-check. vLLM uses
`torch.cat([inputs_embeds, hidden_states], dim=-1)` — embedding first.
Kiln uses `Tensor::cat(&[&norm_emb, &norm_h], 2)` — `norm_emb` first.
Both produce a `[1, 1, 2H]` tensor where the **first** H columns are
the embedding half and the **second** H columns are the hidden half.
The `fc` weight was trained against this specific half-ordering; any
swap would show as a catastrophic α collapse (on-manifold but wrong
half-to-half projection), not α ≈ 0.15.

**Step 3 verdict: CLEAN.**

## Step 4 — `fc` projection

vLLM (line 114 + module-init line 64–72):

```python
self.fc = ColumnParallelLinear(
    self.config.hidden_size * 2,    # 2H input
    self.config.hidden_size,        # H output
    bias=False,
)
...
hidden_states = self.fc(hidden_states)   # [seq, 2H] -> [seq, H]
```

`ColumnParallelLinear` is semantically `y = x @ W^T`, with `W` stored
as `[out, in] = [H, 2H]` on disk.

Kiln:

- Weight layout (forward.rs:195-215):
  - `mtp.fc`: `[H, 2H]`, bf16 on device — matches vLLM's on-disk layout.
  - `mtp.fc_t`: `[2H, H]`, pre-transposed at load time (PR #117/#124/#128
    cached-transpose pattern — same trick the base model uses for
    `*_proj_t`).
- Loader (loader.rs:628): `let fc = extract_tensor(tensor_map,
  &format!("{mtp_prefix}fc.weight"))?;` — pulls directly from the
  checkpoint's `mtp.fc.weight` key.
- Shape assertion: `crates/kiln-model/src/loader.rs` unit test (line 1612):
  `assert_eq!(mtp.fc.shape, vec![64, 128])` — confirms `[H, 2H]` for
  the unit-test `H=64`. For production `Qwen3.5-4B` with `H=2560`, the
  expected shape is `[2560, 5120]`.
- Matmul (forward.rs:4552, production path):

  ```rust
  concat.broadcast_matmul(&mtp.fc_t)?   // [1, 1, 2H] @ [2H, H] = [1, 1, H]
  ```

Sub-property matrix:

| # | Property | vLLM | Kiln | Status |
| --- | --- | --- | --- | --- |
| 1 | Present | `self.fc(hidden_states)` | `concat.broadcast_matmul(&mtp.fc_t)` | ✅ CLEAN |
| 2 | Weight tensor | `mtp.fc.weight` | `mtp.fc` loaded from `mtp.fc.weight` (loader.rs:628), cached as transpose `mtp.fc_t` | ✅ CLEAN |
| 3 | In-dim | `2H` | `2H` (concat produces `[1, 1, 2H]`) | ✅ CLEAN |
| 4 | Out-dim | `H` | `H` (fc_t is `[2H, H]`) | ✅ CLEAN |
| 5 | Math equivalence | `y = x @ W^T` with `W: [H, 2H]` | `y = concat @ fc_t` with `fc_t = fc.T: [2H, H]`. Structurally identical — kiln pre-applies the transpose once at load, so the hot-path matmul is the same tensor contraction. | ✅ CLEAN |
| 6 | Bias | `bias=False` | No bias term in the matmul | ✅ CLEAN |
| 7 | Dtype | bf16 (ColumnParallel holds bf16) | bf16 by default. `KILN_MTP_FC_FP32_ACCUM=1` / `KILN_MTP_FP32_HEAD=1` optionally promote to f32 for debugging (Phase C9 — kill switch, not prod default) | ✅ CLEAN |

**Step 4 verdict: CLEAN.** The fc projection matches vLLM's
`ColumnParallelLinear(2H, H, bias=False)` exactly. The only divergence
is the cached-transpose optimisation, which is a layout detail — it
changes *where* the transpose happens (load time vs. matmul time) but
not *what* tensor the matmul computes.

## Overall verdict

**H10 RULED OUT.** All four steps of the vLLM
`Qwen3NextMultiTokenPredictor.forward` contract match kiln's source at
HEAD `72d75c0` across all 6 sub-properties (weight tensor, eps, order,
residual contract, concat-order / direction, dtype).

Cumulative Phase-C audit result (hypotheses eliminated so far):

| PR | Phase | Hypothesis | Result |
| --- | --- | --- | --- |
| #340 | C17 | h_prev first-divergence (frame) | CANDIDATE → confirmed |
| #341 | C18 | h_prev post-final-norm frame fix | PARTIAL (α: 0.0325 → 0.1532, 4.7× but below floor) |
| #343 | C19 | H1: `mtp.fc_norm` vs. `model.norm` reuse | RULED OUT (no `fc_norm` in checkpoint) |
| #344 | C20 | H2: MTP block dual-norm inversion | RULED OUT |
| #346 | C21 | — (empirical bench) | — |
| #347 | C22 | H3: rotary mtp_pos offset (static) | RULED OUT |
| #348 | C23 | H4: gate / residual param in splice | RULED OUT |
| #350 | C24 | H3 hardware A/B | RULED OUT |
| #351 | C25 | H6: verifier-vs-draft logit path | RULED OUT |
| #352 | C26 | H7: MTP weight-reload sanity | RULED OUT |
| #353 | C27 | H8: MTP block sequencing + paged-KV | RULED OUT |
| **#TBD** | **C28** | **H10: h_prev post-norm contract** | **RULED OUT (this doc)** |

Every structural / static hypothesis touching the MTP splice pipeline
has now been refuted. The residual α gap (0.1532 → floor 0.5) must
therefore come from a place we cannot see with source-reading alone —
either a numerical drift that is too small to catch by eye, or a
behavioural divergence in the draft-token sampler.

## Handoff — Phase C29 (H9 empirical MTP-logits dump)

The only remaining high-signal test is empirical: dump kiln's and
vLLM's MTP top-K logits on the same input on a fixed set of prompts,
and see where the first-divergence occurs.

### Sketch of the dump script

Two halves, both under 200 lines each.

**Half 1 — `scripts/c29_mtp_logits_dump_kiln.py`** (already have
`KILN_MTP_DUMP_PATH` + `KILN_MTP_DUMP_POS` from Phase B6/B7):

1. Pick 10 prompts where production kiln α ≤ 0.10 (filter from the
   `speculative.rs` telemetry or regenerate from the C18 α harness).
2. For each prompt, run kiln with `KILN_MTP_DUMP_PATH=/tmp/c29_kiln/p{N}/
   KILN_MTP_DUMP_POS="0,1,2,3"` to capture the full 8-tap safetensors
   dump at MTP positions 0–3.
3. Also capture the final `logits` tensor (already tap 8/8 in the
   existing dump schema).

**Half 2 — `scripts/c29_mtp_logits_dump_vllm.py`** (new, HF
`transformers` + published Qwen3.5-4B checkpoint):

1. Load Qwen3.5-4B with `trust_remote_code=True` and the reference
   `Qwen3NextForCausalLM` class (includes MTP head).
2. For each of the same 10 prompts, run the base forward to get
   `hidden_states[-1]` (post-final-norm), then manually run the
   4-step MTP chain against `mtp.*` weights loaded from the same
   checkpoint.
3. Capture the same 8 taps: `token_emb`, `norm_emb`, `norm_h`,
   `concat`, `fused` (`fc` output), `post_block`, `post_norm`,
   `logits`.
4. Dump as safetensors files with identical names so pairwise diff is
   trivial.

**Half 3 — `scripts/c29_mtp_logits_compare.py`** (new, ~50 lines):

1. For each prompt × position × tap, compute `max|Δ|`, `rms(Δ)`, and
   the top-K token overlap between kiln's and vLLM's distributions.
2. Print the first tap where `max|Δ|` exceeds a configurable threshold
   (suggested: 1e-2 for bf16 taps, 1e-3 for f32 taps).
3. Report top-K Jaccard overlap on the final `logits` tap — if this
   drops below ~0.8 while earlier taps are clean, the drift is
   concentrated in the MTP block itself (attention / MLP), not the
   splice.

### Cost envelope

- **Kiln dump half:** `KILN_MTP_DUMP_PATH` path already ships; C18 α bench
  takes ~6 min of A6000 time per prompt pair (prefill + 128 decode).
  10 prompts ≈ 60 min of GPU time, ~$0.50 on A6000 spot.
- **vLLM dump half:** CPU is fine (Qwen3.5-4B in bf16 fits in ~8 GB RAM
  for a single-prompt forward). 10 prompts × ~15s each ≈ 3 min — no
  pod needed.
- **Compare half:** pure numpy, no GPU. Instant.

Total C29 budget: 1× A6000 pod acquire, ~90 min wall-clock, ~$15. This
stays well under the 120 min / $50 kiln task cap.

### Success criterion

C29 succeeds if the compare script identifies a specific tap where
kiln first diverges from vLLM beyond the bf16 noise floor. That tap
names the hypothesis for Phase C30. If *no* tap exceeds threshold and
the top-K Jaccard is already > 0.8, the problem lies in the sampler /
acceptance logic (H5 revisit), not in the draft token's logit
distribution.

## Bundled cleanup (none)

The C27 handoff also flagged a stale comment at `speculative.rs:708-710`
contradicting `:678-682` as an optional bundleable cleanup. On
re-inspection this audit is already at its natural scope cap (the
four-step chain + verdict + C29 handoff); bundling the comment fix
would bloat this PR and muddy its "doc-only static audit" framing.
Split to a separate trivial PR if/when anyone touches `speculative.rs`
for unrelated reasons.

## Cost / budget

- Wall-clock: ~45 min (static source read-through of `forward.rs`,
  `loader.rs`, `mtp_debug.rs`, and the vLLM reference; doc drafting).
- Pod cost: **$0.** No GPU acquired, no bench run.
- Well under the 60 min / $15 Phase C28 task cap.
