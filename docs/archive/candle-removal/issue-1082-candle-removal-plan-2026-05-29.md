# kiln-model candle-core Removal — Authoritative Migration Plan (#1082 Tier-3)

> **Historical snapshot, not current operating guidance.** This document records
> migration state from May 2026. The `KILN_USE_TAPE_*` and
> `KILN_USE_TAPE_AUTHORITATIVE` switches mentioned below were removed without
> aliases or replacement fields. Current GPU training uses an internal tape
> scope as its sole routing authority. See [Configuration](../../CONFIGURATION.md)
> and [Native SFT Profile](../../NATIVE_SFT_PROFILE.md) for current behavior.

## 1. Executive Reframe: The Substrate Is Done — This Is Mechanical Type-Swapping

**The kiln-tensor substrate is essentially complete.** This migration is a `candle_core::Tensor` → `kiln_tensor::Tensor` type-swap across ~6,076 candle reference tokens in `kiln-model/src/`, plus the cargo-feature flip that drops the dependency. It is **not** a substrate-building exercise. A prior plan wrongly claimed five substrate gaps that all already existed; this plan does not repeat that error. Every "gap" below was checked against the kiln-tensor op surface and the existing kt ports (`paged_kv_cache_kt.rs`, `metal_types`, `device_op.rs`, `GumbelSampler`).

### The genuinely-real substrate gaps (survived verification)

There is exactly **one** op-level functional gap and **one** performance-only gap. Everything else the subsystem specs flagged as gap-shaped is either a name change, a method→free-fn shift, a two-op composition, or a missing host-readback ergonomic — none of which block correctness.

| Real gap | Kind | Blocked subsystem(s) | Resolution |
| --- | --- | --- | --- |
| **`slice_set` — in-place scatter write into a tensor sub-region** | Functional (CUDA + CPU) | `cuda_graph.rs` (graph-stable logits buffer), `paged_kv_cache.rs` / `kv_cache.rs` CPU write path, `forward.rs` conv-state roll + logits buffer (~6 sites) | **CUDA path already solved** in `paged_kv_cache_kt.rs` via `cudarc::memcpy_dtod_async` into the pool's raw device pointer. CPU path needs `&mut`/`Mutex` + alloc-and-assign, or a small kt `copy_into_at(dst, src, offset)` primitive. This is the one primitive worth landing early. |
| **CUDA `sort` / `top_k` (on-device)** | Performance-only | `sampling.rs::try_topk_on_device` | `kiln_tensor::ops::sort` and `ops::top_k` are CPU-only (unconditional `CpuStorage` downcast). The candle path ran an on-device sort. Migration falls back to one D2H copy + kt host top_k. **Correctness is preserved** (host-fallback already exists); only the on-device top-k fast path regresses. Not a blocker — flag for a follow-up CUDA sort kernel. |

### Non-gaps explicitly debunked (do NOT plan substrate work for these)

- **`sqr()`** → `ops::mul(&x,&x)` or `ops::pow(&x,2.0)`. Trivial.
- **`recip()`** → `ops::reciprocal` (name change only).
- **`log()`** → `ops::ln` (candle `.log()` is natural log).
- **keepdim reductions (`sum_keepdim`/`mean_keepdim`/`max_keepdim`)** → `*_axis` + `unsqueeze(axis)`. Two-op compose.
- **`broadcast_mul`/`broadcast_add`/`broadcast_sub`/`broadcast_div`** → `ops::broadcast_to` + elementwise `mul`/`add`/`sub`/`div`. kt does **not** implicitly broadcast — this is a *semantic gotcha*, not a missing op.
- **`broadcast_matmul`** → `matmul` after explicit rank alignment (`unsqueeze`). Inference-side LoRA only; CP-4 tape already owns the training LoRA path.
- **`Tensor::empty` (uninitialized)** → `zeros_on`. Loses a memset micro-opt on a path where the buffer is fully overwritten anyway. Acceptable.
- **`dims4()`/`dims3()`/`dims2()`/`dims1()`** → `t.shape()` slice indexing with a rank check. Mechanical noise, ~600 sites; consider a local helper macro or thin kt `dims_n` helpers to reduce churn (optional, not blocking).
- **`to_vec1::<T>()`/`to_scalar::<T>()`** → `t.to_device(Cpu)?.contiguous()?` then `bytemuck::cast_slice` on `CpuStorage::as_bytes()`. Missing *ergonomic*, not capability. **Strongly recommend landing a `Tensor::to_vec<T>()` convenience early** — it appears in tests, FP8 host fallback, sampling readback, and mtp_debug, and writing the byte-cast inline ~200 times is the single biggest source of avoidable churn.
- **`from_raw_buffer`** → `bytemuck::cast_slice(bytes)` + `from_slice` + `to_device`. One site (`lora_loader::safetensor_to_kt`).
- **`GumbelSampler` on-device** → exists in kt but reads back to host. Performance note for large-batch prefill sampling only; single-token decode is fine.

**Bottom line:** land two small things early — a `copy_into_at`/`slice_set` kt primitive (CPU + a thin CUDA wrapper around the existing `memcpy_dtod_async` pattern) and a `Tensor::to_vec<T>()` host-readback helper — and the entire rest of the migration is mechanical or semantic-shift type-swapping with zero new kernels required (except the optional CUDA sort follow-up).

---

## 2. The All-Or-Nothing Boundary and the Highest-Leverage Sequencing Question

**Question: Can the default/cuda build drop `candle-core` *before* the Metal backend is migrated?**

**Answer: YES.** This is the single most important sequencing decision and it unlocks an early partial win.

The reasoning, verified against the cargo-features map (`crates/kiln-model/Cargo.toml`):

- `cuda` and `metal` are **independent additive cargo feature sets**. `cuda = ["candle-core/cuda", "candle-nn/cuda", ...]`; `metal = ["candle-core/metal", "dep:objc2", "dep:objc2-metal", "kiln-tensor/metal"]`. Dropping `candle-core/cuda` from the `cuda` line does not touch the `metal` line.
- The blocker is **not** the feature graph — it is that `candle-core` is currently an **unconditional** base dep (`Cargo.toml:18`), and the `metal` backend (`backend/metal.rs`, ~1,298 candle refs) plus the always-on `candle-nn` (`gumbel_softmax` ×2) still need it.

**The strategy that produces an early win:**

1. Make `candle-core` **`optional = true`** and the `metal` feature carry it explicitly: keep `metal = ["candle-core/metal", ...]` (the `dep:` is implied by the existing path-feature reference) and add a temporary transition feature so Metal still compiles against candle.
2. Migrate everything in the **CUDA + CPU + Vulkan + sampling + forward** path off candle.
3. Replace the always-on `candle-nn::gumbel_softmax` calls with `kiln_tensor::GumbelSampler` so `candle-nn` can also become optional/Metal-gated.
4. **Flip the `cuda` feature to drop `candle-core/cuda` + `candle-nn/cuda`.** At this point a `cargo build --features cuda` (the production RunPod build) **compiles with zero candle-core in the dependency tree** — while `--features metal` still pulls candle behind its own gate. This is the early partial win: production decode/training runs candle-free months before the Metal lift finishes.
5. Migrate `backend/metal.rs` last (macOS-CI-gated), add the `metal_types::sdpa` kt-native shim, then drop `candle-core/metal` and `kiln-tensor/metal`'s candle dep.
6. Delete `candle-core` and `candle-nn` from the manifest entirely; delete `legacy-candle-parity`.

**Prerequisite that gates even the cuda-first drop** (sibling-crate ordering): `kiln-kt-bridge` currently has an **unconditional `candle-core` dep** (`kiln-kt-bridge/Cargo.toml:44`). kiln-model cannot drop candle-core from a feature while it transitively pulls it through an always-on bridge dep. So **`kiln-kt-bridge` must become candle-optional (or its candle surface gated) before the cuda-feature flip** — the bridge is "deleted last" for its *API*, but its *candle dep* must go optional at the cuda-flip step. This is the one cross-crate constraint that is easy to miss.

---

## 3. Dependency-Ordered Phases

The spine, validated against the five inter-subsystem dependency lists: the `BackendRuntime` trait and the `GpuWeights`/cache struct fields are the two "everything-downstream-of-me" types. They must flip early, and the safe pattern (already established in this codebase via `for_device_kt` / `device_kt()` / `new_kt`) is **kt-typed twin methods/constructors alongside the candle ones**, then retire the candle originals.

### Phase 0 — Land the two early primitives (CPU-validatable, ~S)

**Files:** `crates/kiln-tensor/src/` (new/extended ops), `crates/kiln-tensor/src/tensor.rs`.
**Migrates:** Nothing in kiln-model yet — removes the two real blockers.
**Work:**
- `Tensor::to_vec<T: Element>(&self) -> Result<Vec<T>>` — `to_device(Cpu)? + contiguous()? + bytemuck::cast_slice` on `CpuStorage::as_bytes()`. Kills ~200 inline byte-cast sites.
- `slice_set` / `copy_into_at(dst, src, axis, offset)` — CPU impl + a thin CUDA wrapper over the `memcpy_dtod_async` pattern already proven in `paged_kv_cache_kt.rs`.
- (Optional, churn-reducing) `dims2/dims3/dims4` tuple helpers on `Tensor`.
**Swap patterns:** N/A (substrate additions).
**Gotchas:** `to_vec` must force contiguity; `slice_set` CPU path needs interior mutability (`&mut`/`Mutex`) since kt tensors are immutable-by-clone.
**Validation:** `cargo nextest run -p kiln-tensor` (CPU/CI). No GPU needed for the CPU paths; CUDA `copy_into_at` validated on pod in Phase 3.
**Size:** S.

### Phase 1 — `GpuWeights` + caches kt-typing (the data spine) (~XL)

**Files:** `forward.rs` (struct defs `GpuWeights`/`GpuLayerWeights`/`GpuFullAttentionWeights`/`GpuLinearAttentionWeights`/`GpuFfnWeights`/`LinearAttentionState`/`MtpGpuWeights`, ~lines 4648–5050), `kv_cache.rs`, `paged_kv_cache.rs`, `fp8.rs`, `lora_loader.rs`, `marlin_proj.rs`, `packed_weight_registry.rs`, `decode_buffers.rs`, `loader.rs`.
**Migrates:** Every struct field from `candle_core::Tensor` → `kiln_tensor::Tensor`. Constructors gain `*_kt` twins where not already present (`KvCache::new_kt`, `PagedKvCache::new_kt` already exist; add for `DecodeBuffer::allocate`, `MarlinPackedProj::pack_from_bf16`, `PackedWeightStorage::Bf16`).
**Migration order within the phase:** `MarlinPackedProj` → `LoraProjectionWeights` → `GpuWeights` family → `PackedWeightStorage` → `DecodeBuffer` → `PagedKvCache` interior → `KvCache` interior → `fp8.rs` surface.
**candle→kt swaps:**
- `Tensor::zeros(shape,dtype,dev)` → `Tensor::zeros_on(dev,shape,dtype)` / `zeros_cpu`.
- `Tensor::from_slice(data,shape,dev)` → `from_slice(data,shape)` + `.to_device(dev)` (two-step; CPU-first).
- `Tensor::from_vec(vec,shape,dev)` → `from_vec(vec,shape)` + `.to_device(dev)`.
- `.to_dtype(dt)` → `ops::cast(&t,dt)`.
- `.index_select(&idx,dim)` → `ops::index_select(&t,axis,&idx)` (axis-first).
- `.narrow/.squeeze/.unsqueeze/.transpose/.t/.contiguous/.reshape` → identical kt methods.
- `Tensor::cat(&v,dim)` → `ops::concat(&v,axis)`.
- `Storage::Cuda(s).as_cuda_slice(...).device_ptr(stream)` → `kiln_tensor::CudaStorage::device_ptr_raw()` → `(CUdeviceptr, usize)`.
- `slice_set` write loops → `copy_into_at` (Phase 0) on CPU; `memcpy_dtod_async` on CUDA (already in `paged_kv_cache_kt.rs`).
- `from_raw_buffer` → `bytemuck::cast_slice` + `from_slice` + `to_device` (`lora_loader::safetensor_to_kt`).
**Semantic gotchas:** two-step device placement adds H2D copies on paths that previously allocated on-device; replicate the `cpu_compatible_compute_dtype` BF16→F32-on-CPU downgrade guard explicitly (kt won't do it); `t.id()` identity must be stable across views for the Vulkan/weight caches keyed on `TensorId` (confirm in Phase 0/1); `U8` is the FP8 E4M3 wire dtype (no semantic gap, just a naming reminder).
**Validation:** `cargo nextest run -p kiln-model` (CPU/CI) for all `#[test]` blocks; CUDA pod for `test_write_token_major_native_cuda_kt_*` parity vs candle `slice_set` (`KILN_CUDA_ARCHS=86 cargo nextest run -p kiln-model --features cuda`); `kiln-bench --paged --max-output-tokens 128` median-of-3 parity.
**Size:** XL (this is the largest single lift; `paged_kv_cache_kt.rs` is the reference impl to mirror).

### Phase 2 — `BackendRuntime` trait kt-typing (~L)

**Files:** `backend/mod.rs` (trait def, 257 refs, 51 candle-typed methods), `backend/cpu.rs` (6 refs).
**Migrates:** All 51 trait method signatures `&candle_core::Tensor`/`&mut`/`Result<Option<...>>` → `kiln_tensor::Tensor`; the single `candle_core::DType` (`resolve_resident_activation`) → `kt::DType`; `for_device`/`for_device_kt` consolidation; `CpuBackend` drops its `device: candle_core::Device` field, keeps `device_kt`.
**candle→kt swaps:** mechanical type substitution in 48 default-body methods (bodies return `Ok(None)`/`Ok(false)` — untouched). `name()` match → `kiln_tensor::Device` arms. `for_device(&candle::Device)` → deleted; `for_device_kt` becomes sole factory with `matches!(dev, kt::Device::Vulkan(_))` + `vulkan::vulkan_is_available()` runtime detection preserved.
**Semantic gotchas:** three methods need attention from their concrete-impl owners (defaults are safe): `gdn_full_chunk_forward_head_last_into` (in-place `_out` buffer), `scatter_gdn_recurrent_resident_batch_rows` (`&mut [&mut Tensor]` true scatter — needs `storage_mut`/raw-ptr in CUDA impl), `resolve_resident_activation` (materializes via `zeros_on` free-fn form). `prewarm_decode_weights`/`drop_uploaded_bf16_weights` take `&GpuWeights` — already kt after Phase 1. The trait flip **breaks all backend impls at compile time simultaneously** — that's expected and surfaces the full blast radius at `cargo check`.
**Validation:** `cargo check` (CPU, no features) is the gate — it lights up every concrete-backend mismatch (cuda/metal/vulkan). Full validation deferred to per-backend phases.
**Size:** L (signature surface is large but bodies are untouched).

### Phase 3 — CUDA backend + CUDA train/graph (~XL)

**Files:** `backend/cuda.rs` (289 refs), `cuda_train.rs` (426 refs), `cuda_graph.rs` (138 refs).
**Migrates:** The CUDA concrete impl of the now-kt trait; `CudaTrainTensor` inner field flips to `kt::Tensor` (cascades through all `.as_tensor().op()` chains); `CudaGraphRunner::new(&candle::Device)` → `kt::Device`.
**candle→kt swaps:**
- Trait-boundary `&candle::Tensor` → `&kt::Tensor`; the `kt_tensor_from_candle_cuda_borrow`/`..._copy` bridge calls **evaporate** (inputs are already kt).
- `broadcast_add/sub/mul/div` → `broadcast_to` + elementwise (add shape asserts).
- `mean_keepdim/sum_keepdim/max_keepdim` → `*_axis` + `unsqueeze`.
- `Tensor::new(slice,dev)` → `from_slice(slice,shape)` (+`to_device` for CUDA).
- `log_sum_exp(D::Minus1)` → in-file numerically-stable manual expansion (the pattern already in `cuda_shifted_linear_cross_entropy_loss`).
- `candle_core::cuda_backend::cudarc::*` → direct `cudarc::driver::*`.
- `unsafe Tensor::empty` → `zeros_on`.
- `TensorId::from_raw(id.as_raw() as u64)` → `kt_tensor.id()` directly.
- `cuda_graph.rs` logits-buffer `slice_set` → `copy_into_at`/`memcpy_dtod` (Phase 0 primitive).
**Semantic gotchas:** `track_op()` (×6) has no kt equivalent — post-CP-4 it becomes `bridge_scope_active()`/tape-scope checks; do NOT assume a mechanical substitute. `kiln-rmsnorm-kernel::matmul_f32_bf16w` must expose a kt-native entry before `cuda_train.rs:1277,1351` migrate. The raw-CUDA-pointer accessor for `memcpy_htod_async` in `update_cuda_scalar` needs `CudaStorage::device_ptr_raw` (kt-native path).
**Validation:** CUDA pod mandatory. `KILN_DISABLE_GDN_KERNEL=1 KILN_DISABLE_FUSED_CONV1D=1 KILN_DISABLE_FUSED_GDN_GATES=1 KILN_DISABLE_FUSED_GDN_GATED_RMS_NORM=1 ./kiln-bench --paged` to confirm dispatches still engage; optimizer round-trip tests (`cuda_sgd_step_resident_round_trip_f32`, `cuda_adamw_step_resident_round_trip_bf16`) as regression canaries; `kiln-bench --training-steps 5` gated on `secs_per_step ±10%` + `peak_vram_mb ±15%` (Tier-2 baseline); `KILN_CUDA_GRAPHS=true` capture+replay for `cuda_graph.rs`.
**Size:** XL.

### Phase 4 — Vulkan backend (~L)

**Files:** `backend/vulkan.rs` (~480 refs), `vulkan_linear_op.rs` (~138), `vulkan_lora_op.rs` (~100). (`vk_decode_resident.rs`/`vk_forward.rs` do not exist — content is inline in `vulkan.rs`.)
**Migrates:** The Vulkan trait impl; `candle_core::TensorId` HashMap keys (incl. the two `OnceLock<Mutex<HashMap<TensorId, Arc<VulkanBuffer>>>>` registries) → `kiln_tensor::TensorId`; the `CustomOp1`/`CustomOp3` impls → `kiln_tensor::DeviceOp1`/`DeviceOp3`.
**candle→kt swaps:** `TensorId` (import path only), `Device::Cpu` matches → `is_cpu()`, `DType` checks, `from_vec`/`from_slice`/`zeros`, `reshape/narrow/unsqueeze/contiguous/t`, `flatten_all`→`ops::flatten`, `to_dtype`→`ops::cast`, `index_select`→free-fn, `cat`→`concat`, `broadcast_as`→`broadcast_to`, `broadcast_matmul`→`matmul`, `* f64`→`mul_scalar(f32)`. The `tensor_to_f32_bytes_with_shape`/`tensor_from_f32_bytes` helpers (duplicated across all three files) rewrite to use `to_vec<f32>()` (Phase 0).
**Semantic gotchas:** **`DeviceOp` rewrite is architectural** — `CpuStorage`/`Layout`/`from_storage`/`BackpropOp::none()`/`Storage::Cpu` plumbing is torn out (input to `cpu_fwd` is already a `kt::Tensor`); `bwd` returns a closure-based `BackwardOp`. **Order constraint: migrate `VulkanLinearOp`/`VulkanLoraOp` to `DeviceOp` *before* the surrounding `lora_delta_resident` BackendRuntime methods** (which call `apply_op3` → `device_op::dispatch3`). `broadcast_to` must produce a stride-0 view, not a materialized copy, or `drop_uploaded_bf16_weights` trades 6 GB of weights for 6 GB of zeros. `t.id()` stability across views (cache correctness). `to_vec`/byte extraction requires `contiguous()` after `narrow`/`reshape`.
**Validation:** **CI-buildable, no GPU required** — Vulkan tests self-skip (`if !backend.has_vulkan()`). `cargo check` (no `--features cuda`) is the primary gate; `cargo test -p kiln-model` runs non-Vulkan tests on CI. Full suite on a Vulkan box (Strix Halo / any AMD/Intel iGPU): `vulkan_linear_backward_parity_small`, `lora_delta_resident_reflects_post_update_weights`, `resolve_resident_activation_round_trip`; `KILN_VULKAN_LINEAR=1 ./kiln-bench --paged --skip-training`.
**Size:** L (half the refs are test code).

### Phase 5 — Sampling / generate / speculative / mtp_debug (~M)

**Files:** `sampling.rs` (~29 prod refs), `generate.rs` (~27), `speculative.rs` (~14), `mtp_debug.rs` (~21).
**Migrates:** The inference glue; function signatures flip to `kt::Tensor`; `PagedPrefixNextToken::Logits`/`PrefillSampleSource::Logits` enum payloads → `kt::Tensor`; `DecodeBatcherConfig` candle-typed variants deleted (kt twins `from_env_for_backend_kt`/`from_env_for_device_kt`/`enabled_for_device_kt` already exist).
**candle→kt swaps:** `flatten_all`→`flatten`, `to_dtype`→`cast`/`to_f32`, `narrow/squeeze`→same, `affine(1/T,0)`→`mul_scalar(1.0/T as f32)`, `dims1()`→rank-check+`shape()[0]`, `dims()`→`shape()`, `to_vec1`→`to_vec` (Phase 0), `KvCache::new`→`KvCache::new_kt` (exists).
**Semantic gotchas (the architectural sites):**
- `argmax(0).to_scalar::<u32>()` → `ops::argmax_last_dim` returns **I64** tensor → extract `i64`, cast to `u32`.
- `index_add(&idx,&delta,0)` → `flat + scatter_add(deltas,0,idx,vocab)` (**zero-based** scatter vs additive-into-existing — must add the two-step, not a direct swap; CPU penalty path).
- `gumbel_softmax(&scaled,1.0,0)` → `GumbelSampler::sample()` — requires **rank-2 `[B,V]`** (`unsqueeze(0)`/`squeeze(0)`), owns a seeded Mutex RNG (construct/share an instance), returns **I64 `[B]`**, **reads back to CPU** (on-device guarantee not preserved — fine for single-token decode).
- `sort_last_dim(false)` in `try_topk_on_device` → **CUDA sort gap**: D2H + kt host `top_k` (one DtoH/step on the CUDA top-k path; correctness preserved).
- `contiguous()` before any kt borrow on `narrow`/`squeeze` views.
- **`mtp_debug.rs` must NOT be deleted** — 263 call sites in `forward.rs`; its `is_mtp_*_armed` flags affect production control flow.
**Validation:** `cargo nextest run -p kiln-model --features cuda` on A6000 (penalty correctness, min-p, top-k heap-vs-sort, greedy determinism, gumbel, CUDA softmax); `test_top_k_matches_host_topk` (CPU) for the sort fallback; `KILN_USE_TAPE_AUTHORITATIVE=1` parity gate (kiln-train) for end-to-end generation; CPU CI for fallback paths.
**Size:** M (most refs flip mechanically once forward.rs lands; ~14 architectural sites).

### Phase 6 — `forward.rs` core (the bulk) (~XXL)

**Files:** `forward.rs` (~3,862 candle refs, ~33,646 lines), plus co-migration of `kv_cache.rs`/`lora_loader.rs` consumers if not fully done in Phase 1.
**Migrates:** Every function signature and the entire forward-pass body to `kt::Tensor`. This is the largest file and depends on Phases 1–2 (structs + trait) being done. **Of the ~3,862 refs: ~1,200 are inside CustomOp impl bodies + try_kt bridge helpers (deleted/collapsed, not migrated), ~745 are metadata reads (mechanical), ~1,900 are real op calls (per-site).**
**candle→kt swaps (high-frequency):** `LAST_DIM`/`D::Minus1` → `rank()-1` (everywhere — #1 silent off-by-one risk); `to_dtype`→`ops::cast` (~368); `contiguous` (~366, **must stay** — kt borrow requires it); `broadcast_mul/add/sub` (~158) → `broadcast_to`+elementwise (RMSNorm `[B,T,1]`×`[B,T,H]` — silently wrong if omitted); `Tensor::cat`→`concat` (~63); `matmul`/`broadcast_matmul`→`ops::matmul` (the 2D-flatten wrapper `matmul_no_broadcast_copy` stays in Rust); `sum/sum_keepdim/sum_all`→`sum_axis`+`unsqueeze`/`sum_all`; `sqr()`→`mul(x,x)`; `recip()`→`reciprocal`; `.log()`→`ln`; unary activation/trig → free fns; `where_cond`→`where_select`; `Tensor::new`/`from_slice`→CPU-first+`to_device`; `index_select`→axis-first free fn; `softmax(LAST_DIM)`→`softmax_last_dim`; `slice_set`→`copy_into_at` (conv-state roll @ ~12910, logits buffer @ ~23939); `dims3()/dims2()/dim(n)`→`shape()` destructure.
**Architectural deletions (post-CP-4, do NOT migrate — delete):** the 7 `CustomOp1/2/3` impl bodies (`CudaLoraAddF32`/`CudaLoraAddBf16`/`CudaLoraLinearBf16`/`CudaSigmoidMulTrainingBf16`/`CudaFlashAttentionTrainingBf16`/`CudaRotaryOneBf16`/`VulkanRmsNorm*`) and their `Storage`/`Layout`/`Shape`/`from_storage`/`CudaStorage`/`alloc_uninit` plumbing (~1,200 refs); `track_op()` guards (~35) → `tape_forward_enabled() && tape_scope_active()` (leaving a stale `!track_op()` guard silently bypasses backward nodes in training — critical); `device().synchronize()` (×3, inside CustomOp backward — vanishes with the impls); `Var::from_tensor` (test-only finite-diff — keep candle as a `#[cfg(test)]` dep or port to kt `Tape`).
**Validation:** `cargo check --features cuda` (Linux, no GPU) catches all type errors first; `cargo nextest run -p kiln-model` for RMSNorm/RoPE/GDN-sequential/FlashAttn CPU parity; A6000 full parity gate `kiln-bench --paged --prompt-tokens 512 --max-output-tokens 128` within ±2% tok/s, all kernels off vs production; Tier-1b `perf_regression_sft_train_cpu_smoke_completes_under_30s`. **Validate via finite-diff + convergence, NOT candle-autograd parity** (per the `kiln-candle-autograd-drops-attn-conv-grads` learning — candle's `loss.backward()` silently severs full-attn + GDN-conv grads; CP-4 tape is the more-correct reference).
**Size:** XXL — break into per-region sub-PRs (embedding/lm-head → MLP/SwiGLU → GQA attention → GDN recurrence/chunkwise, the heaviest at ~1,800 refs).

### Phase 7 — Flip `cuda` feature off candle (THE EARLY WIN) (~S, gated on 1–6 + bridge)

**Files:** `crates/kiln-model/Cargo.toml`, `crates/kiln-kt-bridge/Cargo.toml`.
**Migrates:** Dependency manifest. Make `candle-core` `optional`; make `kiln-kt-bridge`'s candle dep optional/gated; remove `candle-core/cuda` + `candle-nn/cuda` from the `cuda` feature; replace the last `candle_core::cuda_backend::CudaDevice` (`paged_kv_cache_kt.rs:89,116`) with direct `cudarc::CudaDevice`.
**Result:** `cargo build --features cuda` (production RunPod build) is **candle-free**. `--features metal` still pulls candle behind its gate.
**Validation:** `KILN_CUDA_ARCHS=86 cargo build --release --features cuda --bin kiln-bench` on A6000 with **zero `candle` in `cargo tree`**; full decode + SFT bench parity.
**Size:** S (but only flips once 1–6 land and `kiln-kt-bridge` is candle-optional).

### Phase 8 — Metal backend (~XL, macOS-CI-gated)

**Files:** `backend/metal.rs` (~1,298 refs), `crates/kiln-tensor/src/metal_types.rs` (sdpa shim).
**Migrates:** The Metal trait impl; `MetalBackend.device: candle::Device` field dropped (use `device_kt`); buffer extraction `storage_and_layout()`+`Storage::Metal(s)` → `t.storage().as_any().downcast_ref::<MetalStorage>()` + `t.layout()`; `kt_layout_from_candle`/`kt_dtype_from_candle` bridges deleted; pipeline cache keys `DeviceId` → `u64` (`companion.device_id()`); `precompile_custom_kernels(&candle::Device)` → `MetalCompanion`.
**candle→kt swaps:** dtype/device guards (~386+170, mechanical), shape inspection → `shape()`/`rank()`, `index_select`→axis-first, `flatten_all`→`reshape(product)`, `to_vec1`→`to_vec`/CPU-readback, `unsafe Tensor::empty`→`zeros_on` (free zero-fill on UMA Shared buffers).
**The Metal-specific real gap — `sdpa`:** `metal_types::sdpa` re-exports `candle_nn::ops::sdpa` (candle-typed). Once the trait passes kt tensors, add a **bridge shim at the `metal_types::sdpa` chokepoint (kt→candle→kt)** until a native kt Metal SDPA lands. This is the documented Phase-7 follow-up; it does not block the cuda-first drop.
**Semantic gotchas:** `index_select` axis order (~6 sites in paged-decode — silent dim/idx swap); `candle DType::I32` → kt `DType::U32` (4-byte, kiln convention); `companion.device_id()` returns `u64` not `DeviceId(usize)` (cache-key type annotation); preserve `is_contiguous()` guards before `buffer_o_kt`; sdpa transpose round-trip allocates two MTL buffers (memory budget awareness).
**Validation:** macOS-only. `cargo build --release --features metal` on M-series; `kiln-bench` with/without `KILN_DISABLE_METAL_SDPA=1`; existing `metal.rs` parity tests (~line 20300+) candle-ref vs Metal kernel; macOS GHA `ci.yml` lane covers the build check (GPU parity needs an M-series runner).
**Size:** XL.

### Phase 9 — Drop candle-nn, flip metal feature, delete candle-core (~S)

**Files:** `crates/kiln-model/Cargo.toml`, `crates/kiln-tensor/Cargo.toml`, `crates/kiln-kt-bridge/` (final API deletion).
**Migrates:** Remove `candle-core/metal` from the `metal` feature; remove `kiln-tensor/metal`'s `candle-core` dep; delete `candle-core` + `candle-nn` from the manifest; delete `legacy-candle-parity` and its gated parity tests; delete `kiln-kt-bridge` (the migration shim, last to go).
**Validation:** `cargo build --features cuda`, `cargo build --features metal`, `cargo check` (CPU) — all with `! cargo tree | grep candle`. Full A6000 + macOS bench parity sweep.
**Size:** S.

---

## 4. First Executable Increment

**The smallest concrete migration that compiles, validates, and reduces candle refs — and is NOT substrate-building (the substrate exists).**

There are two valid starting points; do **Phase 0's `to_vec<T>()` + `copy_into_at` first** (they are pure kt-tensor additions that unblock everything and are CPU-test-validated with no kiln-model churn), then the first *kiln-model* increment:

**First kiln-model increment: migrate `MarlinPackedProj` to kt fields** (`crates/kiln-model/src/marlin_proj.rs`, ~18 refs).

- **Why this one:** It is the upstream-most leaf in the Phase-1 dependency order (`MarlinPackedProj → LoraProjectionWeights → GpuWeights → PackedWeightStorage → DecodeBuffer`). It has only 18 refs, a clean kt path (`marlin_w4a16_gemm_kt` already takes kt tensors), and both its consumers (`marlin_proj.rs` itself and `packed_weight_registry.rs::PackedWeightStorage::MarlinW4A16`) are local.
- **The swap:** `MarlinPackedProj.b_packed` / `.scales` from `candle_core::Tensor` → `kiln_tensor::Tensor`; `pack_from_bf16` constructor `from_vec(vec, shape, dev)` → `from_vec(vec, shape) + to_device(dev)`; `to_dtype(F16)`/`to_dtype(BF16)` → `ops::cast`; `.transpose`/`.reshape`/`.contiguous`/`.to_device` stay as methods. The `kt_tensor_from_candle_cuda_borrow` calls in `matmul_bf16_2d_kt` evaporate (input is already kt). Raw-ptr path in `packed_weight_registry.rs::with_bf16_device_ptr` switches from `Storage::Cuda(...).device_ptr(stream)` to `CudaStorage::device_ptr_raw()`.
- **Validation:** `cargo check` (CPU) confirms types; `cargo nextest run -p kiln-model --features cuda` on a pool A6000 pod runs the Marlin pack/GEMM round-trip; net candle-ref reduction is immediate and measurable (`rg -c candle_core crates/kiln-model/src/marlin_proj.rs`).
- **Size:** S — single small file, one local consumer, no trait or forward.rs dependency.

This proves the toolchain (kt twin-constructor pattern, two-step device placement, raw-ptr accessor swap) on a contained surface before the XL `GpuWeights`/`forward.rs` lifts.

---

## 5. Blast Radius Into kiln-train / kiln-server (Coupled Crates)

### kiln-server — **near-zero source churn**

- **22 candle references in `src/`, ALL comments — zero production code.** kiln-server already migrated to `for_device_kt`, `GpuWeights::device_kt()`, `kiln_kt_bridge::candle_device_from_kt`.
- It holds `&GpuWeights` and passes it to kiln-train. Once `GpuWeights` fields go kt (Phase 1), kiln-server **needs no source changes** — the type flips under it.
- **One mechanical exception:** `training_queue.rs:962` — `weights.embed_tokens.device().clone()` currently returns `&candle_core::Device`; after Phase 1 it returns `kiln_tensor::Device` (use `weights.device_kt()`). Single-line change.

### kiln-train — **non-trivial but concentrated (~48 real production candle refs across 8 files)**

- **`cd_types.rs`** — the type-alias module (`pub type Tensor = candle_core::Tensor`, etc.). These aliases are the seam: flip them to `kiln_tensor::Tensor`/`Device`/`DType` **in lockstep with Phase 1's `GpuWeights` migration**. Everything downstream in kiln-train that uses `cd_types::Tensor` then flips for free.
- **`trainer.rs`** (28 refs) — mostly `candle_core::Device::new_cuda(0)` in tests + `Device::Cuda(_)` matches in the tape-authoritative guard. Mechanical: `kiln_tensor::Device::Cuda`.
- **`opd.rs`** (4 non-comment) — `candle_nn::optim::AdamW` in `#[cfg(test)]` only. Test-dev dep; can stay candle-gated under `#[cfg(test)]` until the very end or port to a kt optimizer.
- **`tape_step.rs`, `echo.rs`, `train_receipt.rs`** — minor.
- **`GradStore` / `Var` / `TensorId`** — kiln-train's autograd-adjacent types need parallel migration once `GpuWeights` flips; since CP-4 (tape-authoritative) is **done and default-on**, the candle-autograd training path is dead-ish, so these are mostly test-surface and the tape path is the authority.
- **Coupling rule:** the `cd_types` alias flip and the `GpuWeights` field flip (Phase 1) **must land together or in adjacent PRs** — a mismatch (kt `GpuWeights` fed into a candle `cd_types::Tensor` signature) won't compile. Plan Phase 1 to include the `cd_types.rs` alias swap and the `training_queue.rs:962` one-liner in the same change set.

**Net:** kiln-server is effectively free; kiln-train's real cost is the `cd_types` alias flip (which cascades cleanly) plus ~30 mechanical `Device`/`DType` test-site renames. Neither crate gates the cuda-first candle drop beyond the Phase-1 `GpuWeights` lift they already depend on.
---

## 6. Reality-check addendum (post-Phase-0, verified against source `2ea7f1cd`)

Two corrections from validating the plan against the actual source — material to execution order:

### 6.1 `forward.rs` uses a bare `Tensor` import alias, not `candle_core::Tensor` qualified
`forward.rs:25` is `use candle_core::{backend::BackendDevice, D, DType, Device, Tensor, Var};`. So the ~1,240 bare `Tensor` occurrences and ~847 `candle`-qualified occurrences (the latter includes comments) ARE the candle surface — **not ~3,862**. The map's 3,862 conflated the bare-`Tensor` token count with candle refs. Consequence for migration: forward.rs is a **contained per-file flip** — change the import to `kiln_tensor::{Tensor, ...}` (or `use kiln_tensor::Tensor;`), then fix the now-broken op-call *syntax* (candle methods → kt free-fns: `.to_dtype`→`ops::cast`, `.index_select(idx,d)`→`ops::index_select(t,d,idx)`, `softmax(D::Minus1)`→`softmax_last_dim`, etc.). The op SEMANTICS map 1:1 (substrate complete); the churn is syntactic + the broadcast/keepdim gotchas already cataloged in §3 Phase 6.

### 6.2 Phase 1 (GpuWeights) and Phase 6 (forward.rs) are COUPLED via bare `Tensor`
`GpuWeights`/`GpuLayerWeights`/… fields are bare `Tensor` (= candle), and forward.rs does candle ops directly on them. Flipping the struct fields to `kiln_tensor::Tensor` therefore **breaks every forward.rs op on them at compile time** — Phase 1 cannot land standalone-compilable without either (a) doing forward.rs in the same change (the big core flip — GpuWeights + forward.rs together, the XXL atomic-ish lift), or (b) flipping the fields to kt + wrapping each forward.rs field-read in a candle-bridge (`kt_tensor_to_candle_cuda_borrow`) so forward.rs stays candle-typed until its own flip — hundreds of temporary bridge sites, messy but incrementally compilable.

**Revised recommendation:** treat **GpuWeights + forward.rs as one migration unit** (the core), decomposed into compilable sub-steps by *kt-twin accessor* rather than per-field bridges: keep candle fields, add `*_kt()` accessor methods (precedent: `device_kt`/`from_model_weights_kt`/`model_forward_kt` already exist) that return kt views, migrate forward.rs region-by-region (embedding/lm-head → MLP → GQA attn → GDN) to consume the `_kt()` accessors, and only flip the underlying fields to kt once the last candle consumer is gone. The backends (metal 1298 / vulkan 480 / cuda 289 — these DO use `candle_core::` qualified heavily, unlike forward.rs) flip after the trait (Phase 2), independently.

**Next concrete increment:** begin the forward.rs region migration at the **embedding + lm_head** region (smallest, most upstream, already has `model_forward_kt`/`from_model_weights_kt` kt scaffolding to build on), behind `_kt()` accessors, validated on a CUDA pod against the candle path. Then MLP/SwiGLU, then GQA attention, then GDN (heaviest).
