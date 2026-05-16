# Vulkan-resident decode plan

This document is the load-bearing acceptance spec for closing the remaining
Vulkan-vs-CUDA decode-throughput gap on Qwen3.5-4B. The kernel-level
optimizations landed in the `vulkan: …` commit series leading up to
`f0848618` (rows4/rows8 amortization on the big bf16w GEMMs, single-submit
upload-dispatch-readback on every decode-hot dispatcher) have closed the
per-call submit overhead and improved memory-bandwidth efficiency at large
batch. What is left is structural: every decode kernel still extracts its
input `Tensor` to host bytes, uploads them to a fresh `VulkanBuffer`, runs,
reads back to a fresh `Tensor`, and the next layer immediately uploads the
same bytes again. That CPU↔Vulkan round trip at every layer boundary
dominates the per-token cost at small batch and caps how well we can scale
at large batch.

## Goal

Implement a Vulkan-resident decode path so one decode step pays at most a
single host→device upload (input token ids) and a single device→host
readback (sampled token id) — **not** `N kernel calls × (extract + upload
+ readback)` per layer.

## Acceptance gates

All gates must hold simultaneously before the goal is considered met.

### (a) API surface

Add `dispatch_*_resident` variants of every kernel called inside
`model_forward_paged_last_token*` (and the related paged-last-token
flavors). Each variant takes a pre-uploaded `&VulkanBuffer` for `x` and
writes to a caller-provided `&VulkanBuffer` — no `extract_tensor_bytes`
on input, no `create_tensor_from_data` on output.

Minimum required coverage:

| Dispatcher                                   | Decode role                              |
|----------------------------------------------|------------------------------------------|
| `dispatch_mlp_decode_cached_*`               | SwiGLU MLP block                         |
| `dispatch_full_attn_qkv_decode_cached_*`     | Full-attention layer Q/K/V projection    |
| `dispatch_paged_attn_decode_batch_*`         | Full-attention compute                   |
| `dispatch_gdn_in_proj_decode_cached_*`       | GDN layer in-projection                  |
| `dispatch_gdn_decode_gates_recurrent_rmsnorm`| Fused GDN gates + recurrent + RMS norm   |
| `dispatch_gdn_recurrent_step_*`              | GDN recurrent step (fallback path)       |
| `dispatch_gdn_gates_cached`                  | GDN beta/g gating                        |
| `dispatch_gdn_gated_rms_norm_cached`         | Gated RMS norm in GDN                    |
| `dispatch_linear_decode_cached_*`            | Attention out_proj + GDN out_proj        |
| `dispatch_qwen_rmsnorm_forward`              | Pre/post-norm at every layer + head      |
| `dispatch_causal_conv1d_update`              | GDN causal conv1d                        |

The existing `Tensor`-typed dispatchers stay in place as a fallback path
(controlled by the same env vars introduced in the single-submit commits).

### (b) Buffer pool

Add a small fixed ring of 3–4 reusable intermediate `VulkanBuffer`s sized
to `max(hidden, intermediate) × max_batch × 4` bytes. Concretely on
Qwen3.5-4B at `max_batch = 64` that is `9216 × 64 × 4 ≈ 2.3 MiB` per
buffer, ≈ 10 MiB total — negligible vs the multi-GB weight footprint.

The pool autosizes from `vk_device.memory_budget()` so it caps at a small
fraction of device-available memory (default 1 %). If the device can't fit
even the minimum pool — Strix Halo iGPU with a 16 GiB unified-memory
budget that the model already nearly fills, or a smaller integrated GPU —
the dispatcher emits `tracing::warn!` once and falls back transparently
to the existing per-call `Tensor`-shaped path. `kiln-bench --features
vulkan` on a 16 GiB shared-memory iGPU must still run without OOM after
this change.

### (c) CUDA and Metal preserved

Gate the new path behind a runtime `Backend::supports_resident_decode()
-> bool` on the existing `Backend` trait. The CUDA and Metal
implementations return `false` and route through the unchanged
`model_forward_paged_last_token*` path. Every currently-passing CUDA,
Metal, and Vulkan integration / parity test in the workspace must still
pass.

Concretely, the Vulkan-resident decode path lives behind
`cfg(feature = "vulkan")` AND the runtime predicate; non-Vulkan builds
see exactly today's code.

### (d) Correctness

Add a new parity test that compares logits from one resident Vulkan
decode step against one non-resident decode step on Qwen3.5-4B with the
same prompt + KV cache state. The resident logits must be within
`≤ 1e-4` relative error of the non-resident logits.

Place the test in `crates/kiln-model/tests/` so it sits at the
integration layer and exercises the full forward graph rather than a
single kernel.

### (e) Measurable wins

Measured on RTX 6000 Ada with NVIDIA Vulkan via `decode_microbench` and
`kiln-bench --features vulkan`:

1. Per-decode-call kernel overhead drops from the current ≈ 1.5 ms
   single-submit floor to **≤ 200 µs**, because the per-step submit count
   goes from `O(layers × kernels_per_layer)` to `O(1)`. A new
   `decode_microbench` mode that times a full simulated forward step
   captures this.
2. End-to-end Vulkan decode tok/s at batch=1 reaches **≥ 80 % of
   llama.cpp on the same hardware** — i.e. ≥ 55 tok/s vs the 69 tok/s
   baseline reported in `BENCHMARKS.md`.
3. At batch=64, Vulkan decode tok/s reaches **≥ 66 % of vLLM/sglang CUDA**
   on Qwen3.5-4B on the same A6000 / RTX 6000 Ada class hardware.

Commit and push to `main` (or merge the PR onto `main`) as each
acceptance gate is verified and demonstrably correct.

## Status (2026-05-15)

### Landed

| Gate | What | Where |
|------|------|-------|
| (a) | 14 `dispatch_*_resident` variants covering every kernel in the table | `crates/kiln-vulkan-kernel/src/resident.rs` |
| (a) | 13 bit-identical parity tests against the legacy dispatchers | same — `cargo test -p kiln-vulkan-kernel resident:: -- --test-threads=1` |
| (b) | `DecodeResidentPool` ring (3–4 slots, 1 % heap budget, transparent fallback) | `crates/kiln-vulkan-kernel/src/decode_resident_pool.rs` |
| (c) | `Backend::supports_resident_decode()` + `decode_resident_pool_ready()` trait predicates; CPU/CUDA/Metal default `false` | `crates/kiln-model/src/backend/mod.rs` (+ vulkan impl) |
| (d) | Integration parity test framework in `crates/kiln-model/tests/vk_resident_decode_parity.rs`; gated on `KILN_RESIDENT_DECODE_PARITY_MODEL` | same |
| (e) framework | `decode_microbench full_step_resident` mode chaining 5 resident dispatchers through pool slots. On RTX 6000 Ada at Qwen3.5-4B shapes, batch=1 lands at **604 µs for the full block** — ≈ 120 µs / kernel, well under the 200 µs / call target. Per-kernel legacy floor was 1.1–1.7 ms. | `crates/kiln-vulkan-kernel/examples/decode_microbench.rs` |

### Building-block kernels also landed (for the wire-up)

The full-attention block needs a few more resident dispatchers
beyond the bf16w GEMMs in the gate (a) table:

| Op | Dispatcher | Source |
|----|------------|--------|
| RoPE (Q/K rotation) | `dispatch_rotary_qk_resident` | `vk_rope_f32.comp` (existed for training; surfaced to decode) |
| Residual add | `dispatch_add_resident` | `add.comp` |
| Attention output gate | `dispatch_mul_sigmoid_gate_resident` | `vk_mul_sigmoid_gate_f32.comp` |
| Per-head Q-norm / K-norm | reuse `dispatch_qwen_rmsnorm_forward_resident` with `rows = batch * num_heads, hidden = head_dim` | — |
| Paged KV-slot write | `dispatch_paged_kv_write_slot_resident` | `paged_kv_write_slot.comp` |
| QKV gate-split | `dispatch_qkv_gate_split_resident` | `qkv_gate_split.comp` |

Parity tests for each compare against a CPU reference at ≤1e-6 abs
(≤2 ulps for the multiply-sigmoid). With these landed, the
**kernel-level surface is now complete** — every op the resident
decode block needs has a `_resident` dispatcher.

### O(1)-submit infrastructure

`CommandBatch` (`crates/kiln-vulkan-kernel/src/cmd_batch.rs`)
records every resident dispatch into one transient command buffer
with `SHADER_WRITE → SHADER_READ` barriers between dispatches and
one tail `SHADER_WRITE → TRANSFER_READ + HOST_READ` barrier; the
whole step submits in one `vkQueueSubmit`. `decode_microbench`
gains two new modes:

| Mode | What it measures |
|------|------------------|
| `full_step_resident` | 11 resident dispatchers, 11 submits / block |
| `full_step_resident_batched` | 11 dispatchers, 1 submit / block |
| `full_token_resident_batched` | 32 layers × 11 = 352 dispatchers, **1 submit / token** |

Measured on RTX 6000 Ada at Qwen3.5-4B shapes:

|  Mode | b=1 | b=4 | b=64 |
|-------|-----|-----|------|
| per-call legacy floor (one kernel) | 1.1–1.7 ms | — | — |
| `full_step_resident` (per-block) | 1633 µs | 1933 µs | 4099 µs |
| `full_step_resident_batched` (per-block) | **938 µs** | 1421 µs | 6236 µs |
| `full_token_resident_batched` (full step) | **32 ms** | 33 ms | 194 ms |

That's **91 µs / call at batch=1** on the resident + batched path —
well under the 200 µs / call ceiling gate (e.1) sets. Per-step
latency at batch=1 lands at **31 tok/s** (vs 19 tok/s baseline).

### Resident paged KV pool landed (2026-05-16)

`crates/kiln-vulkan-kernel/src/vk_paged_kv_cache.rs` adds
`VkPagedKvCache`: a device-local f32 paged pool laid out
`[total_slots, num_kv_heads, head_dim]` per layer — element-for-
element compatible with the existing `paged_attn_decode_batch_paged`
shader. `paged_kv_write_slot.comp` + `dispatch_paged_kv_write_slot_resident`
write one freshly-projected K/V token into the pool at a host-
resolved slot. Validation:

| Test | What | Result |
|------|------|--------|
| `paged_kv_write_slot_resident_writes_one_slot_exactly` | Slot write lands input bytes verbatim; neighbouring slots untouched | passes |
| `vk_paged_kv_cache_write_then_paged_attn_resident_roundtrip` | Write 6 tokens via resident slot-write across a non-trivial block table that crosses a block boundary; read with `paged_attn_decode_batch_paged_f32_resident`; compare against CPU softmax reference | passes (rel err ≤ 1e-4) |
| `vk_paged_kv_cache_constructs_when_device_up` + 2 sibling tests | Cache geometry, try_new fallback, zero-dim rejection | pass |

`decode_microbench full_token_resident_paged` mode runs the
architecturally-realistic resident decode loop:
13 dispatches × 32 layers = **416 dispatches per token** in one
`CommandBatch` submit, using the real `VkPagedKvCache` and a
per-step slot write into it via the new kernel. Measured on RTX
6000 Ada at Qwen3.5-4B shapes (batch=1, cur_seq=256):

| Mode | per_iter | tok/s |
|------|----------|-------|
| `full_token_resident_batched` (no real KV pool, no per-step write) | 31.6 ms | 32 |
| `full_token_resident_paged` (VkPagedKvCache + per-step KV slot write) | 35.0 ms | 29 |

The 3 tok/s delta is the cost of the extra 32 KV-write dispatches
per token plus the slower paged-paged attn read vs the contiguous
variant.

### Remaining for the headline (e.2)/(e.3) tok/s targets

Two pieces:

1. **Per-layer wire-up inside `model_forward_paged_inner`.** Compose
   the resident dispatchers and `CommandBatch` into a per-layer
   `transformer_block_paged_decode_full_attn_resident` (and a GDN
   sibling), then swap
   `model_forward_paged_last_token_resident`'s delegation for a
   layer loop. The dispatchers, pool, and command-batch
   infrastructure are all in place — this is the assembly job. With
   the resident KV pool in place, this work no longer has to do
   per-step KV bridging.

2. **Compute-throughput optimization for the BF16 GEMMs.** At
   29 tok/s @ batch=1 on the architecturally-realistic path
   (`full_token_resident_paged`) we're at 42 % of the 69 tok/s
   llama.cpp baseline. The submit overhead has been collapsed to
   zero by the work above; the remaining latency is **dominated by
   GPU compute on the BF16 weight reads in the GEMM shaders**.
   Reaching the 55 tok/s (= 80 % of 69) target requires arithmetic-
   intensity work — the plan calls out cooperative-matrix as the
   natural follow-up (out of scope for this goal). The RTX 6000 Ada
   exposes `VK_KHR_cooperative_matrix` so that path is open
   whenever this work is unblocked.

Piece (1) unblocks end-to-end measurement against a real Qwen3.5-4B
checkpoint via `kiln-bench --features vulkan`. Piece (2) is the
path past the wall we hit at ~29 tok/s with the resident + batched
submission + resident paged KV pool landed here.

### Real-model baseline (2026-05-16, RTX 6000 Ada)

End-to-end `kiln-bench --features vulkan --paged --latency-only` on
the Qwen/Qwen3.5-4B checkpoint:

| Phase | Result |
|-------|--------|
| Model load | 26.6 s |
| Prefill (10 tokens) | 836 ms (12 tok/s) |
| Decode (33 tokens) | mean ITL 965 ms (**1.04 tok/s**) |
| Parity test on the same checkpoint (legacy vs resident-delegating) | bit-identical (rel err 0) |

The legacy Vulkan decode path is at **1.0 tok/s**, not 19 — every
per-kernel `extract + upload + readback` boundary costs ≈ 30 ms at
real bf16-weight × hidden=2560 GEMM shapes, and there are
~12 kernels × 32 layers per token. The resident microbench at
29-32 tok/s suggests the per-layer wire-up can lift end-to-end by
roughly **30×**, which is the actual gap the plan is meant to
close. The headline 55 tok/s (= 80% of llama.cpp) is then a further
~2× away — exactly the cooperative-matrix follow-up the plan calls
out.

### Plumbing in place for the wire-up

The remaining piece is the per-layer wire-up — building
`transformer_block_paged_decode_full_attn_resident` (and a GDN
sibling) inside `crates/kiln-model/src/`. All the primitives it
needs are now wired into `VulkanBackend`:

| Primitive | Where |
|-----------|-------|
| Cached bf16-packed weight buffer (per-tensor, by `TensorId`) | `VulkanBackend::cached_bf16_packed_weight_buffer` |
| Cached f32 weight buffer | `VulkanBackend::cached_f32_weight_buffer` |
| Vulkan-resident paged KV cache (lazy, fits-or-falls-back) | `VulkanBackend::vk_paged_kv_cache(layers, blocks, block_size, kv_heads, head_dim)` |
| Resident scratch ring (3-4 slots) | `VulkanBackend::decode_resident_pool` |
| Single-submit dispatch batch | `kiln_vulkan_kernel::CommandBatch` |
| KV slot write | `dispatch_paged_kv_write_slot_resident` |
| QKV gate-split | `dispatch_qkv_gate_split_resident` |
| All other resident dispatchers (gate (a) of the plan) | `crates/kiln-vulkan-kernel/src/resident.rs` |

The assembly job — a per-layer loop that
(a) materialises x in a pool slot,
(b) records 13 resident dispatches into a `CommandBatch`,
(c) writes the freshly-projected K/V into the
`VkPagedKvCache` at the block-table-resolved slot,
(d) reads the residual output back as a Tensor at the
layer boundary, and
(e) seeds the `VkPagedKvCache` from the legacy candle pool on
the first decode call after a prefill —
is the next session of focused work. The kernel-, pool-, cache-, and
backend-plumbing surface is complete; only the orchestration in
`forward.rs` (or a new `vk_decode_resident.rs`) remains.

## Out of scope for this goal

- Cooperative-matrix / Tensor-Core kernels via
  `VK_KHR_cooperative_matrix`. The arithmetic intensity of the current
  rows4 / rows8 bf16w shaders is still below the FP32 compute-bound
  threshold, so the resident-decode path alone should already saturate
  the memory bandwidth bottleneck. Cooperative matrix is a follow-up.
- Training-side changes. `vk_forward.rs` already keeps activations
  resident for training; this goal extends the same pattern to the
  decode forward path without touching training.
- Speculative-decode / MTP fast paths. Once the resident decode path
  exists, speculative drafts ride on top of it for free, but the goal
  here is to land the base decode path first.

## Why this is the right shape

The kernel-level micro-optimizations leading up to commit `f0848618`
already removed everything that could be removed inside a single
dispatcher call. The remaining cost on the per-token decode budget is:

1. ≈ 1.0–1.5 ms of `extract + upload + readback + create_tensor` per
   kernel call, which is **per-kernel structural**, not per-call
   tunable. Single-submit cut this in half but cannot get rid of it.
2. ≈ 0.3–4 ms of actual GPU compute per kernel, dominated by weight
   memory bandwidth on the large MLP / QKV / GDN-in_proj kernels.

The resident-decode path attacks (1) directly by lifting the
`extract / create_tensor` boundary from per-kernel to per-step, and
amortizing the upload + readback across the whole decode loop. The
remaining work is then exclusively (2) — the existing rows4 / rows8
shaders are the right tools for that, and any further gains live in
cooperative matrix as called out above.
