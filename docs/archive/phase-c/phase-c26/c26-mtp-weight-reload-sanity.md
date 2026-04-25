# Phase C26 — MTP weight-reload sanity audit (hypothesis 7, doc-only redirect)

## Verdict

**Hypothesis 7 is REJECTED on static audit. No code change, no
pod bench.** A CPU-first read of
`crates/kiln-model/src/loader.rs::load_mtp_if_present`,
`crates/kiln-model/src/loader.rs::detect_mtp_prefix`,
`crates/kiln-model/src/loader.rs::extract_tensor`,
`crates/kiln-model/src/forward.rs::weight_to_tensor`, and the
MTP GPU upload block at `forward.rs` lines 1128-1244 — together
with a vLLM cross-reference against
`vllm/model_executor/models/qwen3_next_mtp.py` and the
Qwen3.5-4B `config.json` published on the Hugging Face Hub —
confirms that the 15 MTP-prefixed checkpoint tensors land on
the GPU with the expected names, shapes, and dtypes, escape
Marlin packing cleanly, and have no silent fallback or
aliasing path that could explain the residual α gap (C18
median 0.1532, floor 0.5). H7 joins H1–H6 on the cleared
queue; the remaining gap is now most plausibly attributable
to H8 (block sequencing across decode steps 0→1→2 and MTP
paged-KV state carryover during multi-step verification).

## C25 handoff as stated

From
[`docs/archive/phase-c/phase-c25/c25-verifier-logit-audit.md`](../phase-c25/c25-verifier-logit-audit.md)
("Handoff to H7 / H8"):

> **H7 — MTP weight-reload sanity.** A load-time verification
> that the 15 MTP-prefixed tensors land in the expected GPU
> locations with the expected dtypes and shapes after Marlin
> packing, the MTP layer escape (it is NOT queued for Marlin
> batch pack, per `forward.rs` ~line 1128), and the W4A16
> weight upload path. The scaffolding comment at
> `forward.rs` line 1128 explicitly notes that MTP
> projections are kept in BF16, which is correct for the
> current PR but deserves a one-time end-to-end check that
> the BF16 tensors survive without silent cast-to-F16 or
> layout corruption.

Phase C26 addresses exactly this hypothesis. The audit is a
CPU-first static read against the live source plus the vLLM
reference; if it found any candidate asymmetry whose
hardware signature was ambiguous, an A/B bench would have
been justified (~60 min / ~$20 budget per the task
description). The audit found **zero** candidate asymmetries,
so the hardware-side check is not warranted and H7 ships as a
doc-only redirect at $0 pod spend. C26 follows the same
precedent as C19 (PR #344), C21 (PR #346), C22 (PR #347), C23
(PR #348), and C25 (PR #351).

## Audit scope

For every one of the 15 MTP-prefixed tensors in the Qwen3.5-4B
checkpoint, verify the five sub-properties of "weight-reload
sanity":

1. **Key → field mapping** is unambiguous and explicit (no
   silent miswire of one tensor into another field).
2. **Dtype survival** end-to-end: the on-disk dtype (BF16 for
   Qwen3.5-4B) round-trips through `safetensors` →
   `WeightTensor` → `Tensor::from_raw_buffer` without an
   implicit cast (no BF16 → F16 → BF16 silent downgrade).
3. **Shape correctness** at load time: every MTP tensor is
   shape-validated against an explicit expected shape derived
   from `config.hidden_size` (no shape mismatch absorbed by a
   reshape).
4. **No silent zero / identity fallback** on missing keys: a
   missing MTP tensor must surface as a hard error rather
   than defaulting to zeros, identity, or aliasing the base
   model's weight.
5. **Marlin packing escape** is enforced: the MTP
   transformer layer's q_proj + MLP trio (gate / up / down)
   must NOT be queued for the Marlin batch pack, since the
   current PR keeps MTP in BF16 (the W4A16 coverage
   extension is deferred to a follow-up).

## Findings

### 1. Key → field mapping

`load_mtp_if_present` at `crates/kiln-model/src/loader.rs`
lines 612-706 extracts each MTP tensor by an explicit
`extract_tensor(tensor_map, &format!("{mtp_prefix}<key>"))?`
call, with one extraction per `MtpWeights` field:

```rust
let fc = extract_tensor(tensor_map, &format!("{mtp_prefix}fc.weight"))?;
let pre_fc_norm_embedding = extract_tensor(
    tensor_map,
    &format!("{mtp_prefix}pre_fc_norm_embedding.weight"),
)?;
let pre_fc_norm_hidden = extract_tensor(
    tensor_map,
    &format!("{mtp_prefix}pre_fc_norm_hidden.weight"),
)?;
// final_layernorm probes mtp.norm.weight first, then
// mtp.final_layernorm.weight as a backward-compat alias
let final_ln_key = [
    format!("{mtp_prefix}norm.weight"),
    format!("{mtp_prefix}final_layernorm.weight"),
]
.into_iter()
.find(|k| tensor_map.contains_key(k.as_str()))
.with_context(|| format!("MTP final RMSNorm not found ..."))?;
let final_layernorm = extract_tensor(tensor_map, &final_ln_key)?;
```

Each lookup is bound to a distinctly-named local that flows
into a distinctly-named struct field of `MtpWeights`. There is
no implicit fall-through, no glob extraction, no name-keyed
loop that could cross-wire two tensors. The only ambiguity in
the four `[hidden]`-shaped RMSNorms is intentionally surfaced
as separate, named call sites; the `swap_fc_norms` test path
in `loader.rs` (the existing C20 audit coverage) explicitly
exercises that the two `pre_fc_norm_*` tensors aren't
swapped.

The MTP transformer layer is loaded by re-using the
main-model `load_layer` helper at a synthetic prefix:

```rust
let mtp_layer_prefix = format!("{mtp_prefix}layers.0.");
let layer = load_layer(tensor_map, &mtp_layer_prefix, 3, config)
    .context("mtp layer 0")?;
```

The `layer_idx=3` argument is chosen so `(3+1) % 4 == 0`,
forcing `is_full_attention_layer` to return true (the MTP
layer is full GQA, not linear-attention GDN). This is
followed by a defensive bail:

```rust
match &layer.attention {
    AttentionWeights::Full(_) => {}
    AttentionWeights::Linear(_) => bail!(
        "MTP layer loaded as linear attention — expected full GQA attention. \
         Checkpoint schema change?"
    ),
}
```

Tests at `loader.rs` lines 1606-1658 cover the VL prefix, the
bare prefix, the `final_layernorm` alias, and the MTP-absent
case (returns `Ok(None)` cleanly).

### 2. Dtype survival

The on-disk dtype reaches GPU memory through three hops, none
of which casts:

**Hop A — safetensors → `WeightTensor`.** `extract_tensor` at
`loader.rs` lines 708-733 reads `view.dtype()`, runs it
through `convert_dtype` (a 1:1 mapping function, see lines
~735-755 for the match arms), and stores the dtype on the
`WeightTensor` directly:

```rust
let dtype = convert_dtype(view.dtype())
    .with_context(|| format!("Unsupported dtype {:?} for tensor {name}", view.dtype()))?;
let shape: Vec<usize> = view.shape().to_vec();
let view_data = view.data();
// ... mmap-bounded raw-bytes reference, no copy/cast
```

There is no copy, no cast, no truncation; the safetensors
view's raw bytes are referenced through a bounded mmap
window.

**Hop B — `WeightTensor` → candle `Tensor`.** `weight_to_tensor`
at `forward.rs` lines 428-433 is a 1:1 dtype dispatch:

```rust
fn weight_to_tensor(w: &WeightTensor, device: &Device) -> Result<Tensor> {
    let dtype = weight_dtype(w);
    let t = Tensor::from_raw_buffer(w.as_bytes(), dtype, &w.shape, device)
        .context("failed to create tensor from raw buffer")?;
    Ok(t)
}
```

`weight_dtype` (lines 435-440) is the inverse of `convert_dtype`,
also 1:1:

```rust
fn weight_dtype(w: &WeightTensor) -> DType {
    match w.dtype {
        TensorDType::F16 => DType::F16,
        TensorDType::BF16 => DType::BF16,
        TensorDType::F32 => DType::F32,
        // ...
    }
}
```

There is no implicit `.to_dtype()` call anywhere on the MTP
upload path. BF16 stays BF16; F16 stays F16; F32 stays F32.

**Hop C — Marlin escape.** Critically, the MTP layer's
projection upload at `forward.rs` lines 1135-1244 routes
through `projection_tensors_for_load` (the non-Metal CUDA
branch, lines 674-678):

```rust
} else {
    let materialized = weight_to_tensor(w, device)?;
    let transposed = cached_transpose(&materialized)?;
    Ok((materialized, transposed))
}
```

`cached_transpose` is `weight.t()?.contiguous()?` — a transpose
+ contiguous-copy that preserves dtype. So the MTP `fc`,
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
`down_proj` tensors all reach the GPU in their native BF16
dtype (the Qwen3.5-4B publish dtype), with their pre-computed
transposes also in BF16.

### 3. Shape correctness

Every MTP tensor in `load_mtp_if_present` is shape-validated
against an expected shape:

| Tensor | Expected shape | Validation site |
| --- | --- | --- |
| `fc` | `[H, 2H]` | `loader.rs:630-634` |
| `pre_fc_norm_embedding` | `[H]` | `loader.rs:640-644` |
| `pre_fc_norm_hidden` | `[H]` | `loader.rs:650-654` |
| `final_layernorm` | `[H]` | `loader.rs:672-676` |
| MTP layer (q/k/v/o/gate/up/down/q_norm/k_norm/2 layernorms) | per-tensor shapes inside `load_layer` | shared with main-model layer load path |

The `validate_shape` helper compares against an expected
slice and returns a `bail!` error with the actual shape if
they disagree — no silent reshape.

### 4. No silent zero / identity fallback

`extract_tensor` at `loader.rs` lines 708-733 uses
`with_context(|| format!("Weight tensor not found: {name}"))?`
on the tensor-map lookup, so a missing MTP key surfaces as a
typed `anyhow::Error` with the missing key name. The function
does NOT return a zero tensor, does NOT return an identity
matrix, and does NOT fall back to any other tensor.

The MTP final-norm probe is the one site that accepts more
than one valid name (`mtp.norm.weight` OR
`mtp.final_layernorm.weight`), and that probe also bails if
neither is present — it does NOT silently alias to the base
model's `model.norm.weight`. From `loader.rs` lines 656-670:

```rust
// Stock Qwen3.5-4B names this `mtp.norm.weight`; older vLLM / HF docs
// (and earlier kiln comments) refer to it as `mtp.final_layernorm`.
// Accept either.
let final_ln_key = [
    format!("{mtp_prefix}norm.weight"),
    format!("{mtp_prefix}final_layernorm.weight"),
]
.into_iter()
.find(|k| tensor_map.contains_key(k.as_str()))
.with_context(|| {
    format!(
        "MTP final RMSNorm not found (looked for {mtp_prefix}norm.weight and \
         {mtp_prefix}final_layernorm.weight)"
    )
})?;
```

A silent alias to the base `final_norm` is architecturally
impossible — it would have to be coded explicitly, and it is
not.

The `detect_mtp_prefix` probe at `loader.rs` lines 567-586
returns `None` when neither prefix has an `mtp.fc.weight`
present, and `load_mtp_if_present` returns `Ok(None)` in that
case. So the only way `weights.mtp` is `Some(_)` is if a real
checkpoint shipped real MTP tensors.

### 5. Marlin packing escape

The Marlin batch pack vector
`marlin_pack_inputs: Vec<(Tensor, i32)>` is constructed at
`forward.rs` line 926 and then populated **only inside** the
main per-layer loop:

```rust
let mut marlin_pack_inputs: Vec<(Tensor, i32)> = Vec::new();
let mut marlin_pack_meta: Vec<MarlinPackEntry> = Vec::new();

let mut layers = Vec::with_capacity(weights.layers.len());
for (i, lw) in weights.layers.iter().enumerate() {
    // ... main-model layer upload, pushes onto marlin_pack_inputs
    //     for q_proj, gate_proj, up_proj, down_proj when w4a16_enabled
}

if w4a16_enabled && !marlin_pack_inputs.is_empty() {
    let packed = crate::marlin_proj::pack_from_bf16_batch(&marlin_pack_inputs)?;
    // ... install packed weights back into per-layer slots
}

// Upload native MTP head tensors when the checkpoint shipped them.
let mtp = if let Some(mtp_w) = &weights.mtp {
    // ... separate code path, no push onto marlin_pack_inputs
};
```

The MTP block at lines 1135-1244 runs **after** the Marlin
batch pack has executed and finalized. It does not push onto
`marlin_pack_inputs`. It explicitly sets `q_proj_marlin: None`,
`gate_proj_marlin: None`, `up_proj_marlin: None`, and
`down_proj_marlin: None` on the MTP `GpuFullAttentionWeights`
and `GpuFfnWeights`. So at decode time, the per-layer Marlin
dispatch in `forward.rs` (gated on `*_marlin.is_some()`) routes
the MTP layer to the BF16 reference GEMM, not Marlin — which
is the architecturally-required behavior for the current PR.

The scaffolding comment at `forward.rs` line 1128 names this
deliberate choice:

> Upload native MTP head tensors when the checkpoint shipped them.
> For WIP scaffolding the MTP transformer layer is kept in BF16 and
> is NOT queued for Marlin batch packing: the MTP layer is a single
> layer whose projections account for ~3% of total model memory, so
> deferring W4A16 coverage costs little and keeps the scaffold
> simple. The follow-up PR extends `marlin_pack_inputs` to include
> the MTP layer's q_proj + MLP trio.

The MTP escape is enforced both by code structure (separate
loop) AND by explicit `_marlin: None` field initialization
(no risk of an old packed weight leaking through).

## vLLM cross-reference

A cross-reference against
`vllm/model_executor/models/qwen3_next_mtp.py` (vLLM `main`,
~295 lines) confirms that kiln's MTP forward and weight
loading match the canonical reference contract:

| Concern | vLLM | kiln | Match? |
| --- | --- | --- | --- |
| MTP keys loaded under `mtp.` prefix | `fc`, `pre_fc_norm_embedding`, `pre_fc_norm_hidden`, `norm`, `layers.0.*` | same set, plus alias for `final_layernorm` | ✓ |
| Forward consumes post-final-norm hidden state | yes (vLLM's base Qwen3Next returns normed hidden) | yes (C18 fix; verifier returns post-`final_norm` `h_prev`) | ✓ |
| Forward order: `pre_fc_norm_embedding(e)` → `pre_fc_norm_hidden(h)` → concat → fc → layer → final norm → lm_head | exactly that order in `Qwen3NextMultiTokenPredictor.forward` (lines 109-133) | exactly that order in `mtp_forward_step` | ✓ |
| `lm_head` separately constructed but tied via checkpoint | vLLM constructs `ParallelLMHead` independently; `shared_weight_names = ["embed_tokens", "lm_head"]` (line 284) | kiln re-uses `weights.embed_tokens_t` directly via `lm_head_forward(_, &weights.embed_tokens_t)` | ✓ for Qwen3.5-4B (`tie_word_embeddings: true` on the published config; see below) |
| Dtype | native pass-through (BF16 in practice) | native pass-through (BF16) | ✓ |
| Silent fallback on missing tensor | none — unknown stacked-shard names dropped silently, but no zero/identity init | none — `with_context` errors on missing keys | ✓ (kiln is stricter: it surfaces the missing key) |

The Qwen3.5-4B published `config.json` on the Hugging Face Hub
explicitly sets:

- `tie_word_embeddings: true` (both top-level and inside
  `text_config`)
- `mtp_use_dedicated_embeddings: false`
- `mtp_num_hidden_layers: 1`

So kiln's design choice to tie the MTP `lm_head` to the base
model's `embed_tokens.t()` (the single-source-of-truth tied
head) is **architecturally correct** for the Qwen3.5-4B
checkpoint, not a coincidence of vLLM-style separate
construction. The fact that vLLM constructs lm_head as a
separate `ParallelLMHead` is a parallel-sharding plumbing
concern; the underlying weight values are tied via the
checkpoint's `embed_tokens.weight` aliasing through the
`tie_word_embeddings: true` flag.

The only kiln-specific divergence is that kiln does not
construct a separate `embed_tokens` for the MTP head (it
shares the base `embed_tokens` directly), which matches
`mtp_use_dedicated_embeddings: false`.

## Why there is nothing to A/B on hardware

An A/B bench is only justified when the static read finds a
candidate asymmetry whose hardware signature is ambiguous. In
this case, the audit finds **zero** candidate asymmetries:

- Key → field mapping is explicit and tested.
- Dtype is preserved by 1:1 dispatch in `weight_to_tensor`
  with no implicit cast on the upload path.
- Shape is validated for every MTP tensor before the GPU
  upload.
- Missing keys surface as `with_context` errors, not silent
  fallbacks.
- Marlin packing strictly bypasses MTP via separate code path
  and explicit `_marlin: None` fields.
- vLLM cross-reference confirms key set, forward order,
  dtype handling, and tied-head contract all match.

A hardware A/B that compares "current MTP load path" vs
"hypothetical-fixed MTP load path" has no fix to test —
there is no proposed perturbation that would distinguish the
hypothesis. The doc-only redirect ships the precedent
established by C19 (PR #344), C21 (PR #346), C22 (PR #347),
C23 (PR #348), and C25 (PR #351).

## Handoff to H8

With H1–H7 cleared, the residual α gap (C18 median 0.1532 vs
floor 0.5) must come from H8:

**H8 — Block sequencing across decode steps 0→1→2 and MTP
paged-KV state carryover.** Across multi-step verification,
the MTP draft layer maintains its own `PagedKvCache` that is
separate from the base model's paged KV. The two caches
advance on different schedules:

- The base cache advances once per **verified** token.
- The MTP cache advances once per **draft attempt** at each
  speculative step.

Suspect dimensions for H8:

1. **MTP block-table allocation across multi-step verify.**
   On step k>0 of speculative decode, does the MTP cache
   block-table point at the correct draft-position slot, or
   is it being aliased to the verifier's slot? If the MTP
   cache reads from a stale draft slot, the QK attention
   inside the MTP layer sees a wrong key/value pair, which
   would drop α in exactly the "structurally close but
   off-by-a-step" pattern observed at 0.153.

2. **MTP `seq_len` accounting.** Does the per-decode-step
   `seq_len` arg passed into `transformer_block_paged` for
   the MTP layer reflect the MTP's own KV occupancy, or is
   it accidentally tracking the base verifier's `seq_len`?

3. **Position stride / RoPE frame at multi-step.** The Phase
   B7a fix established `abs_pos = base_pos + mtp_pos` for the
   MTP layer's RoPE. On step k>0, is `mtp_pos` correctly
   advancing within the MTP draft window, or is it
   re-resetting to 0?

4. **Block deallocation / reuse across rejected drafts.**
   When a verifier rejects a draft at step k, is the MTP
   cache block at that position correctly invalidated, or is
   the next draft attempt reading stale K/V? A stale-K/V
   carryover after rejection would produce an exactly
   token-frequency-dependent α drop.

5. **Multi-step prefill vs decode mode mismatch.** Does the
   MTP layer's `transformer_block_paged` correctly enter
   decode mode (single-token KV append) on every spec step,
   or does it occasionally enter prefill mode (multi-token
   write) and clobber a previous slot?

H8 is more expensive than H7 to investigate because it
requires reading the spec-decode loop AND the paged-KV
allocation logic AND the cache reset/invalidation flow —
roughly three call-graph traversals instead of one.
Recommend a CPU-first static read of
`crates/kiln-model/src/speculative.rs::speculative_mtp_decode_step`
together with `crates/kiln-core/src/paged_kv.rs` (or
wherever the MTP cache is allocated and indexed) for the
next phase, before any pod time.

## No-bench rationale (budget)

Budget was $15 / 60 min wall-clock (static audit) with a $50 /
120 min escalation if a positive finding warranted hardware
A/B. Static audit consumed approximately **$0 pod-spend** and
**~30 min wall-clock**. No pod was acquired; no bench was run.
The doc-only redirect ships at $0.

## Files inspected

- `crates/kiln-model/src/loader.rs` — `detect_mtp_prefix`
  (lines 567-586), `load_mtp_if_present` (lines 612-706),
  `extract_tensor` (lines 708-733), `convert_dtype` (lines
  ~735-755), `validate_shape` helper, MTP test coverage
  (lines 1606-1658)
- `crates/kiln-model/src/weights.rs` — `MtpWeights` struct
  (~lines 217-317) and the `pub mtp: Option<MtpWeights>`
  field on `ModelWeights`
- `crates/kiln-model/src/forward.rs` — `weight_to_tensor`
  (lines 428-433), `weight_dtype` (lines 435-440),
  `projection_tensors_for_load` (lines 662-679),
  `cached_transpose` (lines 739-741), `marlin_pack_inputs`
  construction (lines 920-1058), MTP GPU upload block
  (lines 1128-1244), `mtp_forward_step` (~lines 4408-4800+)
- `vllm/model_executor/models/qwen3_next_mtp.py` (vLLM
  `main`, ~295 lines) — `Qwen3NextMultiTokenPredictor`
  forward (lines 98-134), `Qwen3NextMTP.load_weights` (lines
  283-295), `compute_logits` (line 281), `lm_head`
  construction (line 251), `shared_weight_names` (line 284)
- `Qwen/Qwen3.5-4B/config.json` (Hugging Face Hub) —
  `tie_word_embeddings: true`, `mtp_use_dedicated_embeddings:
  false`, `mtp_num_hidden_layers: 1`

## Cross-refs

- C17 reference frame: PR #340
- C18 post-final-norm fix: PR #341
- C19 null redirect: PR #344
- C20 MTP block-norm audit: `docs/archive/phase-c/phase-c20/c20-mtp-block-norm-audit.md`, PR #345
- C21 MTP rotary-pos audit: `docs/archive/phase-c/phase-c21/c21-mtp-rotary-pos-audit.md`, PR #346
- C22 fc/residual audit: `docs/archive/phase-c/phase-c22/c22-fc-residual-audit.md`, PR #347
- C23 draft-sampler audit: `docs/archive/phase-c/phase-c23/c23-draft-sampler-audit.md`, PR #348
- C24 rotary-pos A/B verdict: `docs/archive/phase-c/phase-c24/c24-mtp-rotary-pos-ab-verdict.md`, PR #349 / #350
- C25 verifier-logit audit: `docs/archive/phase-c/phase-c25/c25-verifier-logit-audit.md`, PR #351
