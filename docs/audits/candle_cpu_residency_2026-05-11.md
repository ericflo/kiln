# Candle CPU residency audit (Strix Halo, post-AdamW + lazy-sync)

**Date:** 2026-05-11
**Commits referenced:** `28f7f2ca` (on-device AdamW), `d2eb4da` (lazy
candle-storage sync), and the earlier projection-drop + embed-tokens-stub
infrastructure documented inline below.

## Goal

Catalogue what candle CPU storage actually still holds at full size during a
`kiln serve` LoRA training run on AMD Strix Halo (Vulkan compute, unified
memory, 30 GB DRAM), and document why the remaining items can't be stubbed
without architectural changes beyond Option C.

This is the answer to the question "no second CPU copy?" — with concrete
file/line references so a future reader (or a regression bisect) can verify
the claims.

## TL;DR

On the Vulkan path, all multi-GB base-model duplicates **are already
stubbed by default**. The candle CPU side holds three remaining
categories:

| Category | Approx size (Qwen3.5-4B / rank=8 / T=918) | Stubbable inside Option C? |
| --- | --- | --- |
| (A) Multi-GB base-model weights | ~5 GB → ~0 (stubbed) | **Already done** |
| (B) LoRA Vars + AdamW moments | ~45 MB | **No** — candle `Var` binds storage size to shape |
| (C) Runtime intermediates | ~400 MB peak per gradient-checkpoint segment | **No** — `CustomOp::cpu_fwd` contractually returns real bytes |

To eliminate (B) or (C) requires Option A (replace candle's tensor type
with our own) — out of scope for Option C.

## (A) What's already stubbed (default-on, Vulkan path)

### Embedding table

`crates/kiln-model/src/forward.rs:1999` — `stub_embed_tokens_after_upload()`
returns `true` when `Device::Metal` or `kiln::backend::vulkan_active()`.
The full `embed_tokens` table (~700 MB at vocab=152064 × hidden=2560 × 2
bytes BF16) is replaced with `Tensor::zeros((1,), DType::BF16, device)`
immediately after the transposed cache is built. Downstream lookups use
`embed_tokens_t` and never read the stub.

Existing test: `forward.rs:12060` `test_stub_embed_tokens_decision_negative_only`
asserts the predicate is false on plain CPU.

### Per-layer projection originals

`crates/kiln-model/src/forward.rs:2069` —
`projection_original_drop_enabled_for_device()` returns `true` on Metal,
CUDA, or Vulkan-active processes. `dropped_bf16_stub()` (line 2243)
returns a 1-element BF16 tensor that replaces every layer's
`q_proj_t`/`k_proj_t`/`v_proj_t`/`o_proj_t`/`gate_proj_t`/`up_proj_t`/`down_proj_t`
**original** copies (the contiguous-bf16 view, ~30-100 MB each ×
~210 sites = ~3-4 GB total).

The transposed views remain in candle storage because they're the source
for the Vulkan kernel weight uploads — but those are reference-counted
via `Arc` and freed once the registry buffer absorbs them.

Kill-switch: `KILN_KEEP_PROJECTION_ORIGINALS=1` disables the drop.
Default behaviour (drop on Vulkan) is the contract this audit pins.

### Marlin-absorbed BF16 weights

`crates/kiln-model/src/forward.rs:2252` —
`marlin_bf16_drop_disabled()` defaults to `false` (drop enabled). After
Marlin-quantized weights are absorbed into their packed format, the
intermediate full-resolution BF16 copies are dropped via the same
`dropped_bf16_stub` pattern. Kill-switch:
`KILN_DISABLE_MARLIN_BF16_DROP=1`.

### Boundary states between gradient-checkpoint segments

`crates/kiln-train/src/trainer.rs` (around `segment_input_via_registry_or_clone`)
— after a tiled checkpointed forward returns, the
`boundary_states[seg_idx]` candle CPU mirror is replaced with a 1-element
BF16 stub. The real bytes live in the resident activation registry; the
next segment's forward reads them via `resolve_resident_activation`.

## (B) LoRA Vars + AdamW moment Vars (~45 MB, not stubbable in Option C)

**What:** Per-LoRA-module `Var` for A and B (rank=8: ~9 MB total across
all modules) plus AdamW first-moment `m` and second-moment `v` Vars
allocated by `TrainableLoraParams::allocate_adamw_state` (~9 MB each =
~18 MB combined). Param + m + v = ~27 MB candle CPU storage.

**State as of `d2eb4da`:** Registry buffers are the canonical source of
truth between training steps. `apply_sgd_update` and
`apply_adamw_update` on the on-device path **do not** call `var.set` —
candle CPU storage is intentionally stale until
`TrainableLoraParams::sync_to_candle` runs (called from `sft_train` /
`grpo_train` immediately before `save_peft`, final + per-checkpoint).
`VulkanLoraOp::bwd` reads A and B directly from registry buffers via
`kernels::buffer_to_tensor` (commit `d2eb4da`).

**Why we can't fully stub:** Candle's `Var::set` requires the new
tensor's shape to match the Var's current shape — the underlying
storage allocation is sized by the shape. A `Var::zeros((1,),
DType::BF16, device)` is a *different* Var with shape `(1,)`. To replace
a `[rank, in_features]` Var's storage with a 1-element stub *in place*
would require either:

1. A candle API to "mark this Var as stub-storage with logical shape
   `[rank, in_features]`" — does not exist in upstream candle.
2. Owning the tensor type ourselves (Option A).

Net effect: the ~45 MB stays allocated for the lifetime of training, but
no per-step CPU work touches it. It's pure dead memory, not a compute
leak. On 30 GB UMA, that's 0.15% of available memory — measurable but
not load-bearing.

## (C) Runtime intermediates (~400 MB peak per segment, not stubbable)

**What:** Q/K/V projection outputs, attention scores, softmax outputs,
RMSNorm outputs, MLP gate/up/down intermediates, per-layer hidden
states, residual sums.

For each layer and each forward pass these are produced as full-size
candle CPU tensors by either:

- `CustomOp1/2/3::cpu_fwd` — contractually returns `CpuStorage` with
  real bytes (see `VulkanLinearOp::cpu_fwd` in
  `crates/kiln-model/src/backend/vulkan_linear_op.rs:182`).
- Raw candle ops between CustomOps: `(x + attn_out)?`, `.silu()`,
  `.softmax(D::Minus1)`, `.to_dtype(BF16)`, etc. — these run on candle
  CPU and materialize their output.

With gradient checkpointing enabled (default; 4 segments for Qwen3.5),
peak candle CPU footprint is bounded by one segment's worth of
intermediates: roughly 7-8 layers × ~10 tensors × ~5 MB at T=918 ≈
**~400 MB**.

**Why we can't stub:** Two compounding constraints:

1. `CustomOp::cpu_fwd` returns `CpuStorage` by contract; the returned
   bytes become candle's storage for that intermediate. There's no
   side-channel to say "the real data is over there, this is just a
   placeholder."

2. Even with a side-channel, downstream raw candle ops (`+`, `.silu()`,
   etc.) read the storage directly. They'd see stub bytes and produce
   garbage.

Working around either requires inserting a kiln-owned wrapper around
*every* op in the forward pass — at which point we're rewriting the
tensor type, which is Option A.

## What Option A would require

A summary of the structural work to take (B) and (C) to true zero
candle CPU residency:

1. **`VkTensor` / `VkVar` type.** Replaces `candle_core::Tensor` /
   `candle_core::Var` throughout production crates. Holds
   `(Arc<VulkanBuffer>, Shape, DType, autograd_node_id)`. No CPU
   storage.

2. **Custom autograd tape.** Thread-local stack of backward closures,
   tape-replay on `loss.backward()`. ~300 lines of Rust core; existing
   `CustomOp::bwd` impls port mostly as-is (their math is dtype-agnostic).

3. **Vulkan kernels for every op currently provided by candle.**
   The audit at line 23 of `phase2_cpu_matmul_leaks_2026-05-11.md`
   covered the heavy matmul cases. Option A also needs kernels (or pure
   shape/metadata implementations) for: `+`, `-`, `*`, `.affine()`,
   `.softmax()`, `.silu()`, `.sqrt()`, `.exp()`, `.sum()`, `.mean()`,
   `.to_dtype()` (BF16↔F32), `.transpose()`, `.reshape()`, `.contiguous()`,
   `.narrow()`, `.gather()`. Maybe 30-50 distinct ops. Most are 1-day
   kernels each; the surface area is the issue.

4. **Testing oracle replacement.** Move `candle-core` to
   `[dev-dependencies]` and write a custom parity layer that compares
   our `VkTensor` outputs against candle's CPU output for every op. The
   oracle is the highest-risk piece — without it, kernel correctness
   regressions get harder to catch.

**Realistic estimate:** 8-12 weeks of focused work. The math is
mechanical; the bug surface is wide. A successful Option A produces a
production binary with no `candle-core` runtime dependency — only
`[dev-dependencies]` for the parity oracle.

## Recommendation

Option A is real value but not a tonight project. Until there's a
concrete reason to do it (binary size, large-model support, dependency
cleanup, or a specific candle bottleneck we hit), the current state
(post-`d2eb4da`) is a stable end-point:

- All weights GPU-resident
- All optimizer state GPU-resident
- LoRA backward reads from registry (no candle storage dep)
- Lazy sync to candle only at `save_peft` time
- ~445 MB of candle CPU residency total (45 MB Vars + ~400 MB peak
  intermediates), all of which is structural and doesn't impact perf
  hot paths.

The compute side of `kiln serve` on Strix Halo is fully on the GPU; the
remaining candle CPU footprint is autograd-tape bookkeeping and the
intermediate-tensor backing memory candle requires by design.

## See also

- `docs/audits/phase2_cpu_matmul_leaks_2026-05-11.md` — companion audit
  of CPU-fallback matmul leaks (different axis: GPU-vs-CPU compute, not
  residency).
- `docs/audits/phase2_env_vars_reference_2026-05-11.md` — environment
  variable reference including all kill-switches mentioned here.
- `CHANGELOG.md` Unreleased section — incremental log of the Vulkan
  residency work since v0.2.15.
