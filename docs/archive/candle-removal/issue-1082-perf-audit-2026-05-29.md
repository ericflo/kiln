# Issue #1082 — Perf audit of migration work so far (2026-05-29)

> Multi-agent audit (6 dimensions) for baked-in CPU roundtrips / abstraction copies / CPU-only-where-should-be-device. Verdict + ranked remediation backlog. See also docs/issue-1082-inc0-grad-substrate-plan-2026-05-29.md.

All findings confirmed against source. The observers are unconditional (no env gate at lines 2277/2303). The GDN decay is 3 separate bridged ops (lines 14106-14117). The greedy argmax does a kt->candle copy (line 13062) then `to_scalar` (line 13066) — the copy is pure waste since the scalar must go to host anyway. I have enough verification to synthesize confidently.

---

# #1082 Perf Audit — Remediation Backlog: "What else got baked in?"

## Executive answer to "what else got baked in?"

The good news: **the production CUDA decode and train-step paths are NOT silently CPU-bound** in the way you feared — the actual data flow stays on-device (the kt↔candle crossings bottom out in `memcpy_dtod_async`, device-to-device, not host RAM). The bad news: the migration baked in **two systemic device-side taxes** on every hot-loop op — (1) a wasted full-buffer `cudaMemsetAsync` because the only kt CUDA allocator is `alloc_zeros` (no uninit primitive exists) and the buffer is then 100% overwritten, and (2) a heavyweight `candle_core::Device::new_cuda()` (fresh cuBLAS + cuRAND handle) constructed *per copy-back* — dozens of times per decoded token. There are also **three genuine CPU-roundtrip findings on live paths**: the unconditional per-step LoRA grad-norm observers (~640 `to_scalar` host syncs/step), the gradient-checkpointing path's `to_device(&cpu_device())` offload+reload, and the **entire Vulkan kt op suite** (softmax/rmsnorm/argmax/index_select all D2H→CPU→H2D per call). The kiln-optim AdamW/SGD CPU loop and `clip_grad_norm`/`einsum` are CPU-only **but not yet wired into any production path** — they are latent traps to fix before adoption, not active regressions.

Honest verdict: nothing catastrophic is live on CUDA decode today, but the rewrite has institutionalized "alloc-zeros + new device + dtod copy-back" as the standard kt-API gate shape. If that pattern is left in when the forward.rs flip completes, it becomes the permanent steady-state cost. **This is the moment to add the uninit-alloc + device-cache + zero-copy-borrow-back trio.**

---

## P0 — fix NOW during the rewrite

Ranked by (hot_path × severity × confidence). These three structural fixes (uninit alloc, cached/threaded candle device, zero-copy borrow-back) collectively neutralize ~20 of the 40 findings, so do them first as shared infrastructure.

### P0-1. Per-call `candle_core::Device::new_cuda()` in the kt→candle copy-back (fresh cuBLAS+cuRAND handle every crossing)
**`crates/kiln-kt-bridge/src/lib.rs:676`** (`kt_tensor_to_candle_cuda_copy`)
Confirmed: line 676 calls `candle_core::Device::new_cuda(device_index)` on **every** invocation; `CudaDevice::new` builds a fresh `CudaBlas::new` (cublasCreate) + `CudaRng::new` + context retain/bind. This is the copy-back primitive behind **every** kt-API gate that returns a result. In `gdn_single_token_recurrence` alone (24 GDN layers/token) the decay cluster triggers ~3 crossings/layer = ~72 cuBLAS-handle creations per decoded token, on top of rms_norm (~64/token), rotary (2×8), qk_norm (2×24), marlin projections (4×32), embedding, argmax.
**Note the asymmetry that proves it's avoidable:** the inverse `kt_tensor_from_candle_cuda_borrow` (lib.rs:840) and `..._copy` (lib.rs:514) cheaply reuse the existing device via `cuda_st.device().clone()` (Arc clone). Only the to-candle direction throws the live context away.
**Fix:** Cache the candle `CudaDevice` per ordinal in a `OnceLock`/thread-local map (it's `Arc`-cloneable), or thread the live `Arc<CudaContext>` the kt `CudaStorage` already holds (`cuda_storage.rs:157`) through to the destination alloc. **Effort: S. Mergeable independently** (pure bridge-internal change, no forward.rs flip needed). **Highest leverage / lowest risk in the whole backlog.**

### P0-2. No uninit CUDA alloc — every kt output is `alloc_zeros` (memset) then fully overwritten
**`crates/kiln-tensor/src/cuda_storage.rs:134-144`** (`zeros_ctx`, the *sole* CUDA constructor) + **`crates/kiln-kt-bridge/src/lib.rs:701`** (`Tensor::zeros` dst) + **`crates/kiln-tensor/src/cuda_matmul.rs:236`**
Confirmed: `zeros_ctx` always does `ctx.default_stream().alloc_zeros::<u8>(byte_len)` (alloc + cudaMemsetAsync). There is no uninit path. Every matmul output, elementwise/norm output, allocator pool miss, and both bridge copy-backs memset a buffer that the very next kernel/memcpy overwrites 100%. Comments throughout even *admit* the overwrite ("zeros; we overwrite via dtod memcpy", lib.rs:698-700). On the decode path this is one wasted full-DRAM-write memset kernel per allocating op.
**Fix:** Add `CudaStorage::alloc_uninit_ctx` backed by cudarc's `unsafe alloc::<u8>` (core.rs:1464, right next to `alloc_zeros` at 1493). Use it for outputs fully defined by their producer: matmul outputs (`cuda_matmul.rs:236`), full-tensor elementwise/unary/cast outputs (`cuda_storage.rs:1646`), the allocator pool path (`cuda_allocator.rs:298`), and both bridge dsts (`lib.rs:701`, `lib.rs:533`). Keep `zeros_ctx` only where the consumer reads before writing (accumulation/scatter). **Effort: M. Mergeable independently** — it's an additive primitive; wire it site-by-site. **This is the single highest-frequency baked-in waste.**

### P0-3. No zero-copy kt→candle borrow-back — every gate pays alloc + dtod copy-back
**`crates/kiln-kt-bridge/src/lib.rs:616`** (`kt_tensor_to_candle_cuda_copy`)
Confirmed: the bridge exports `kt_tensor_from_candle_cuda_borrow` (zero-copy, lib.rs:810) but there is **no inverse** `kt_tensor_to_candle_cuda_borrow` — only the copy variant. Every gated op (neg/exp/add_scalar/recip/matmul/broadcast_mul/to_dtype/embedding/rms_norm/rotary/qk_norm/gqa_sdpa) ends with an alloc + dtod memcpy, even though the kt op already produced freshly-allocated owned device memory. The dtod copy exists *only* because the surrounding forward code is still candle-typed.
**Fix:** Add `kt_tensor_to_candle_cuda_borrow` that wraps the kt-owned `CudaSlice` as candle `Borrowed` storage (mirror the existing kt-side `from_borrowed_ctx`, keep the kt tensor alive via a `keep_alive` Arc). The kt op output is already owned device memory — hand candle its pointer. Removes one alloc + one dtod from every gate. **Effort: M. Partially coupled** to the forward.rs flip (some candle call sites still reach `.slice()` which panics on Borrowed — per the migration note at lib.rs:777-786), but the adapter itself + the safe call sites can land now.

### P0-4. ~640 CUDA→host `to_scalar` syncs per training step for LoRA grad-norm logging (unconditional, no gate)
**`crates/kiln-train/src/train_receipt.rs:1562`** (`tensor_l2_norm` → `.to_scalar::<f32>()`)
Confirmed: called from `accumulate_lora_grad_sum_sq` (trainer.rs:5243), invoked once per Var by `observe_lora_grad_norms_from_grad_store` (loops `all_vars_with_modules`, trainer.rs:5229) / `_from_map` (trainer.rs:5214). The observers fire **unconditionally every step** — SFT at trainer.rs:2277 & 2303, GRPO at 5078/5121, OPD at 4818 — with **no env gate**. ~10 modules × 32 layers × {A,B} ≈ 640 individual device→host syncs/step, each forcing a stream sync that serializes the optimizer against the host.
**Fix:** Either (a) gate the entire observation behind `KILN_OBSERVE_LORA_GRAD_NORMS` (off in production) — trivial, or (b) batch the reduction: stack per-module sum-of-squares on device, do ONE `to_scalar` per module (the on-device `tensor_l2_norm_kt` / `l2_norm_scalar` already exists at train_receipt.rs:1574). **Effort: S for the gate, M for batched reduction. Mergeable independently.** This is the **most clear-cut live CPU-roundtrip** in the audit.

### P0-5. Gradient-checkpointing path round-trips every grad CUDA→host→CUDA per step
**`crates/kiln-train/src/trainer.rs:7214`** (`accumulate_grads`: `.to_device(&cpu_device())`) ↔ **`trainer.rs:7325-7328`** (`optimizer_step_from_map`: `.to_device(var...)` back)
Confirmed: `accumulate_grads` moves every grad to host RAM and accumulates in a CPU HashMap; `optimizer_step_from_map` moves each grad back to device. Gradient checkpointing is VRAM-auto-enabled (the `if let Some(ref segs) = segments` branch), so this is a live CUDA path for long-context SFT and token-level GRPO. All ~640 LoRA grads bounce CUDA→host→CUDA each step purely to use a CPU-side accumulator.
**Fix:** Keep a persistent device-resident accumulator keyed by TensorId (in-place device add via the existing resident-activation registry) instead of the CPU offload. CPU offload only makes sense for true activation spilling, not small LoRA grad tensors. **Effort: M. Mergeable independently** of forward.rs.

### P0-6. cuBLAS batched/4-D GEMMs serialized in a single-stream Rust for-loop (`concurrent_streams: 1`)
**`crates/kiln-tensor/src/cuda_matmul.rs:271`** (`concurrent_streams: 1`) + **:274** (serial `for batch_i in 0..batch`)
Confirmed: the batched case is unrolled into a serial loop on one stream with `concurrent_streams: 1` hardcoded. The GQA-SDPA decode path calls this on 4-D `q@kᵀ` [B,16,T,T] (forward.rs:18610) and `p@v` (forward.rs:18638) — B×16 independent head-GEMMs back-to-back with zero overlap. This is exactly the epic's "single-stream serialization of independent matmuls" pain point. kiln-blas even documents that `concurrent_streams=3` is SM-light vs `=1` (backend_matmul.rs:120) yet the field is pinned to 1.
**Fix:** Add a strided/batched cublasLt GEMM (`cublasLtMatmul` with `batchCount` + strideA/B/C) to `CublasLtMatmulHandle`, issue the batch as one strided-batched call; or fan out over a small stream pool driven by `concurrent_streams`. **Effort: M-L. Mergeable independently.** Highest *algorithmic* upside on the decode path.

### P0-7. W4A16 marlin decode projection: kt result copied back to candle only to re-cast f16→bf16 one line later
**`crates/kiln-model/src/marlin_proj.rs:313`** (`kt_tensor_to_candle_cuda_copy(&y_kt)`) + **:315** (`to_dtype(BF16)`)
Confirmed pattern: this is the `KILN_W4A16=1` production decode projection (q_proj forward.rs:18142 + MLP forward.rs:12236, 4 projections × 32 layers/token). The copy-back materializes a candle f16 tensor (alloc+memset+dtod+device-build) consumed and discarded by the f16→bf16 cast on the next line. Plus redundant `.contiguous()` at marlin_proj.rs:294 & 342 on already-contiguous tensors.
**Fix:** Run the f16→bf16 cast as a kt op (`cuda_cast`) directly on `y_kt`, then a single borrow-back (P0-3). Drop the redundant `.contiguous()`. **Effort: S. Coupled** to P0-3 for the full win; the cast-fold + contiguous-drop can land now.

---

## P1 — fix during the relevant phase (real, not innermost hot loop, or coupled to in-flight work)

- **GDN single-token decay = 3 separate bridged ops/token × 24 layers** — `forward.rs:14106-14117` (confirmed: `to_dtype`→`exp`→`to_dtype`, each a full copy-back). Fuse into one cast-exp-cast kt kernel, or keep `g` F32 end-to-end and feed the result kt-side into the subsequent `broadcast_mul`. Mostly subsumed once P0-1/2/3 land; the kernel fusion is the residual win. **Effort: M.**
- **Greedy decode argmax does a pointless kt→candle copy before the host readback** — `forward.rs:13062-13066` (confirmed: `kt_tensor_to_candle_cuda_copy` then `to_scalar`). Replace with `out_kt.to_vec::<i64>()?[0] as u32` — one 8-byte DtoH, dropping the alloc+memset+dtod+device-build entirely. Same for `try_kt_sampling_argmax_rows` (forward.rs:13134). **Effort: S. Mergeable independently** — clean independent win.
- **Eager GQA core bounces kt→candle→kt around softmax/mask** — `forward.rs:18605-18640` (left candle "for bit-exactness with the parity oracle"). Do scale+mask+softmax on kt (`cuda_softmax_last_axis` already exists, used in sampling.rs). Coupled to the parity-oracle migration. **Effort: M.**
- **MatmulBackward forces `.contiguous()` on both transposed operands every backward** — `kiln-autograd/src/backwards/matmul.rs:62-63` (confirmed; forced because `ops/matmul.rs:200-205` rejects non-contiguous). For a tape-wired lm_head matmul (b=[2560,152064]) `b_t.contiguous()` copies ~780MB BF16/step. Fix: teach `cuda_matmul` to accept transpose flags (`CUBLAS_OP_T`) so backward passes transposed *views* at zero copy cost. **Effort: M. Coupled** to CP-4 tape-authoritative training; high per-step VRAM/bandwidth win when that path is live.
- **CrossEntropy backward builds a dense [num_active,152064] one-hot on HOST then H2D-uploads it/step** — `forward.rs:15434-15442` (confirmed: `vec![0.0f32; num_active*vocab]` + host loop + `Tensor::from_vec`). ~156MB host vec + H2D for 256 active tokens, to encode one nonzero/row. Replace with on-device `scatter_add` of [num_active] indices/values. **Effort: M.**
- **GRPO loss backward re-runs the full candle GRPO forward+backward over [1,T,152064] logits/step** — `grpo_candle_shim.rs:213-227` (confirmed). On-device (not a CPU roundtrip) but a redundant second forward+autograd over the vocab dim. Save softmax `p` from the forward tape and reuse as `dlogits=(p-target)*seed*inv_n`. **Effort: M. Coupled** to CP-4.
- **Per-step grad-merge does `loss.backward()` on a detached loss just to get a private GradStore, then dtod-copies each grad in** — `trainer.rs:10756, 10781, 11294`. Migration-seam tax (GradStore::new() is private). Expose a public GradStore constructor / kt-native grad map. **Effort: S-M. Coupled** to CP-4.
- **GdnRecurrentBackward contiguify+copies all 5 output grads/step × 24 layers** — `tape_forward.rs:2196-2208`. Gate `.contiguous()` on `!is_contiguous()` (FlashAttnBackward already does, tape_forward.rs:1788); mostly subsumed by P0-2 uninit alloc. **Effort: S.**

---

## P2 — known / acceptable (don't re-flag)

- **`memcpy_dtod_async` in the bridge (lib.rs:738) is device-to-device, NOT a CPU roundtrip** — confirmed src+dst both CUDA pointers. Acceptable per the audit rubric; only avoidable via the zero-copy borrow (P0-3), already captured.
- **kiln-optim AdamW / SGD / GradAccumulator are CPU-only (`read_to_f32` + scalar host loop)** — `adamw.rs:142-180`, `sgd.rs:128-171`, `grad_accumulator.rs`. **Confirmed NOT on any production path** — the live optimizer is `trainer.rs::apply_adamw_update` → on-device `dispatch_adamw_step` (cuda.rs:341/386). The module doc says "CPU reference path; per-backend impls in subsequent PRs." **Latent trap, not active cost.** Must add CUDA/Metal/Vulkan `OptimStep` impls *before* kiln-train adopts kiln-optim — wiring the current code onto CUDA would be a ~50× #1063-class regression. Do not re-flag as a live finding.
- **`clip_grad_norm` (ops/grad_clip.rs:63-145) and `einsum` (ops/einsum.rs:81) are CPU-only** — exported but **no production caller** in kiln-train/kiln-model (kt's own tests only). Latent gaps; fix before any GPU caller is added. Not active regressions.
- **AdamW on-device fallback chain (~15 candle kernels/Var when dtype mismatch)** — `trainer.rs:7161-7185`. Stays on-device (no host readback — "CPU fallback" is a misnomer). Real but it's a launch-count issue, not a roundtrip; ensure tape grad dtype == moment dtype (BF16) so `dispatch_adamw_step` fires, and add a debug counter so silent fallback is observable. Low priority.
- **Per-token RoPE cos/sin table rebuild in the eager non-graph path** — `forward.rs:9027-9044`. Only hit when caller passes no precomputed tables; the table variant exists to avoid it. Thread tables into the eager path. Low severity.

---

## Cross-cutting patterns (the real signal behind the worry)

**Pattern A — "alloc-zeros + new_cuda device + dtod copy-back" is the institutionalized shape of every kt-API gate.**
Appears at: `lib.rs:676/701` (to-candle copy), `lib.rs:533` (from-candle copy), and transitively at *every* gate call site in forward.rs (84 `kt_tensor_to_candle_cuda_copy` calls) — rms_norm (8098), rotary (9064/9129), qk_norm (17215), marlin (313), matmul (6930), GQA-SDPA (18614/18642), embedding (7887), argmax (13062), the sigmoid composite (106-148), GDN decay (14106). **Structural fix = the P0-1/2/3 trio:** cache the candle device, add `alloc_uninit_ctx`, add `kt_tensor_to_candle_cuda_borrow`. Land those three and the *entire* pattern's per-crossing cost drops from {alloc + memset + cuBLAS-handle-build + dtod} to {Arc clone + pointer wrap} — across dozens of crossings per decoded token and hundreds per training step. **This is the one thing to get right while the rewrite is open.**

**Pattern B — the Vulkan backend round-trips through host RAM on every kt op.**
Appears at: `vulkan_storage.rs` — softmax (222/264/325/359), rmsnorm (425), l2norm (698), activation (884), index_select (1080), argmax (1842, CPU-scan). Each does D2H `read_back` → upload to VkTensor → kernel → D2H → H2D back into kt storage — two full host round-trips per call even though both buffers are GPU-resident. On the Vulkan decode backend, rmsnorm (~64/token), index_select (per token), and argmax (per token, full [1,152064] logits CPU-scan) are **genuine per-token CPU roundtrips**. **Structural fix:** implement the zero-copy kt-VulkanStorage ↔ VkTensor bridge (share the `Arc<VulkanBuffer>`, wrap as a VkTensor leaf without read_back/upload) and a SPIR-V argmax-last-axis reduction kernel. This is a whole-backend gap — Vulkan is currently a correctness shim, not a perf path. Treat as its own phase; it does not block the CUDA forward.rs flip but should be flagged loudly so nobody benchmarks Vulkan decode and panics.

**Pattern C — CPU-only reference impls staged ahead of their device kernels (latent landmines).**
kiln-optim AdamW/SGD/GradAccumulator, `clip_grad_norm`, `einsum`. All correct, all `read_to_f32`/CpuStorage-gated, none wired to a GPU production path *yet*. The risk is a future PR adopting one onto CUDA without noticing the host loop. **Structural fix:** add a `device().is_cpu()` guard that hard-errors (or returns `Ok(None)` to fall through) on GPU storage in each, so adoption can't silently regress, until the device kernel lands.

**Files to touch first (shared infra, unblocks the most findings):** `crates/kiln-tensor/src/cuda_storage.rs` (add `alloc_uninit_ctx`), `crates/kiln-kt-bridge/src/lib.rs` (cache device + add borrow-back + use uninit), `crates/kiln-train/src/trainer.rs` (gate observers, kill CPU grad offload), `crates/kiln-tensor/src/cuda_matmul.rs` (strided-batched GEMM).
