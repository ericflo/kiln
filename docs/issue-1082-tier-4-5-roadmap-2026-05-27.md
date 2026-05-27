# Issue #1082 — Tier 4 → Tier 5 consolidation + roadmap (2026-05-27)

This document is the **single-shot bootstrap** for a fresh agent picking up
issue #1082 (full candle removal). It supersedes the snapshot in
[`CANDLE_REMOVAL_PLAN.md`](./CANDLE_REMOVAL_PLAN.md) (which remains the canonical
multi-tier reference, but is now lagging recent same-day work) by:

1. Summarizing what Tier 1–3 work actually landed (with commit SHAs).
2. Re-auditing the candle footprint at HEAD (`main@269d8f88`).
3. Naming the Tier-4/5 critical-path substrate items and which downstream
   work each unblocks.
4. Recommending 3 substrate tasks to tackle next.
5. Linking every existing STOP doc.

Doc-only — no code changes in this pass. The point of the doc is that the
next agent should be able to pick the highest-leverage substrate task
**without re-doing this audit**.

---

## 1. Tier 1 → Tier 3 — what landed

Across roughly 100 commits over the past 24 hours, the Tier-1/2/3 work
consolidated around four breakthroughs:

### 1a. Kernel-crate Tier-1 closes (candle dropped from `Cargo.toml`)

| Crate | Closing commit | Status |
| --- | --- | --- |
| `kiln-conv1d-kernel` | `577f8b0c` (2026-05-26) | ✅ `candle-core` dropped, `kt_v2_smoke` rewritten via `Tensor::cuda_from_slice` |
| `kiln-marlin-gemm`   | `4a862711` (2026-05-26) | ✅ same template — parity.rs deleted, kt-only public surface |
| `kiln-flash-attn`    | `981dc190` (earlier 2026-05-27) | ✅ candle-typed surface fully deleted (`flash_attn_fwd`, `flash_attn_bwd`, `flash_attn_paged_decode*`, `paged_kv_write_*`); 5/5 kt_api smoke tests green |

The remaining Tier-1 kernel crates (`kiln-rmsnorm-kernel`, `kiln-gdn-kernel`,
`kiln-opd-loss-kernel`, `kiln-flce-kernel`) **still carry `candle-core` in
their Cargo.toml**, but their production-caller posture is now:

- `kiln-gdn-kernel`: the candle-typed decode entries
  (`gdn_decode_gates_recurrent[_qk_norm[_rmsnorm]]`) and the
  `with_decode_gates_recurrent_outputs` thread-local wrapper were deleted
  in `60b7ab07` (2026-05-27). `cuda_graph.rs` no longer imports any
  candle-typed gdn entries. Remaining surface is the chunk-family
  (`gdn_chunk_prep`, `gdn_chunk_scan`, `gdn_full_chunk_forward[_multiblock]`)
  — still production-active in forward.rs prefill loops.
- `kiln-rmsnorm-kernel`: 5 default-on flag helpers deleted in `58607c30`.
  ~32 candle-typed callers remain across forward.rs, cuda.rs, cuda_train.rs
  (`supports_*` predicates, `lora_add_*_storage`, `rotary_one_bf16_storage`,
  matmul training paths, depthwise conv1d forward+backward).
- `kiln-opd-loss-kernel`: substrate complete. CUDA kt-typed backward
  (`opd_top_k_reverse_kl_phase_b_bwd_kt`) landed in `dc092849`; Phase B
  `CustomOp::bwd` migrated to kt-bridge in `0c1be227` / `34524a7b`.
  Remaining: 3 candle-typed forward callers in `kiln-train::opd.rs` are
  now routed through the `KtForwardOp1` shim (see 1c).
- `kiln-flce-kernel`: substrate complete. Phase B CUDA backward
  (`fused_linear_cross_entropy_phase_b_backward_kt`) landed earlier;
  `FlceCustomOp::bwd` migrated to kt-bridge in `ab2da23f` / `34524a7b`;
  `KtForwardOp1` shim entry landed in `72339698`.

### 1b. `CustomOp::bwd` kt-bridge migrations (substrate template)

Four production CustomOp backward bodies now route their FFI calls through
`kiln-kt-bridge` instead of candle FFI directly. All four use the same
template: borrow candle tensors as kt via
`kt_tensor_from_candle_cuda_borrow`, call the kt-typed kernel wrapper, copy
results back via `kt_tensor_to_candle_cuda_copy`. Bit-exact-by-construction
because the same FFI symbol is being invoked. Each migration ships with a
`KILN_DISABLE_*_BWD_KT_BRIDGE=1` kill switch + on-error candle fallback.

| Migration | Crate / file | Commit |
| --- | --- | --- |
| `RmsNormCustomOp::bwd` | `kiln-rmsnorm-kernel/src/lib.rs` | `341da876` |
| `CudaRotaryOneBf16::bwd` | `kiln-model/src/forward.rs` | `d99a15a3` |
| `OpdLossCustomOp::bwd` | `kiln-opd-loss-kernel/src/phase_b.rs` | `0c1be227`, `34524a7b` |
| `FlceCustomOp::bwd` | `kiln-flce-kernel/src/phase_b.rs` | `ab2da23f` (PR #1389) |

Two `CustomOp::bwd` families were **deliberately not migrated**:
- `CudaSigmoidMulTrainingBf16::bwd` — STOPped in
  `186e2c2b` because it has no single fused backward FFI; the
  per-step Phase-7 `try_kt_*` gates are already the closest
  status-quo-preserving migration.
- 3× `CudaLora*::bwd` — STOPped in `cca536cb` and re-confirmed by
  [`docs/lora-bwd-kt-migration-stop-2026-05-27.md`](./lora-bwd-kt-migration-stop-2026-05-27.md).
  Same shape as sigmoid_mul: multi-step composites without a fused
  backward FFI. Revisit when `kiln_lora_*_bwd` kernels land in
  `kiln-rmsnorm-kernel/csrc/`.

### 1c. `KtForwardOp{1,2,3}` shim (kt-autograd interop)

The breakthrough for **Tier-1 close on opd-loss + flce + rmsnorm** is the
generic `KtForwardOp{1,2,3}` candle-autograd shim. Without it, the
candle-typed phase-A forward entries (the fat candle-autograd composites
built from generic ops + `mean_kl.backward()`) had no migration path —
`kiln-tensor` has no `Var`/`backward()` machinery in production.

| Substrate | Commit |
| --- | --- |
| `kiln-kt-bridge::forward_op::KtForwardOp{1,2,3}` candle `CustomOp{1,2,3}` shim parameterized by forward + backward **closures** | `095f1c74` |
| `kiln-rmsnorm-kernel/src/kt_forward_op.rs` `fused_rmsnorm_via_kt_forward_op` entry | `aba53219` |
| `kiln-flce-kernel/src/kt_forward_op.rs` `fused_linear_cross_entropy_phase_b_via_kt_forward_op` entry | `72339698` |
| `kiln-opd-loss-kernel/src/kt_forward_op.rs` `opd_top_k_reverse_kl_per_position_via_kt_forward_op` entry | (under `34524a7b` / earlier) |
| First production caller flipped: `kiln-model::forward::rms_norm` | `0442782c` |
| First production caller flipped: `kiln-train::opd.rs` per-position | `f214f168` |
| OpdLoss KtForwardOp forward closure now full-kt path | `7dfb6639` |

The shim takes a **forward closure** (kt-typed kernel call) + **backward
closure** (kt-typed bwd), wraps them in candle's `CustomOp{1,2,3}`, plumbs
the `cuda_fwd` hook through (constructing a leaf candle `Tensor` from
the supplied `(CudaStorage, Layout)` pair so the closure sees a normal
`&Tensor`), and forwards the candle backward callback to the bwd closure
unchanged. Caller code keeps using `loss.backward()` on candle's tape;
the inside of every fused kernel becomes pure kt.

This is the unblocker that converts the rest of the rmsnorm /
flce / opd Tier-1 production migrations from "requires new substrate" to
"~50–80 LOC per caller, fully testable in isolation."

### 1d. Tier-3 boundary API — `*_kt` parallel entries in kiln-model

`kiln-model` is the deeply candle-typed crate, but a parallel kt-typed
surface now exists for every Tier-3 boundary type. None of these change
behaviour; they are mechanical alternates that kiln-server / kiln-train
can call without a candle Device/DType import:

| Entry | Commit |
| --- | --- |
| `runtime_backend::for_device_kt(&kiln_tensor::Device)` | `89bd66c1` |
| `GpuWeights::from_model_weights_kt(...)` | `53e8149a` |
| `GpuWeights.<weight>.device_kt()` accessor | `f9fbcc2a` |
| `PagedKvCache::new_kt(...)` | `efdb16d2` |
| `model_forward_kt(...) -> kiln_tensor::Tensor` | `5bd48ef1` |
| `kiln-server::device::select_device_kt()` | `dfd44932` |
| `kiln-kt-bridge::{kt_device_from_candle, candle_device_from_kt, candle_dtype_to_kt, kt_dtype_to_candle}` | `12a33cf5` (helpers), `42031508` (split bridge helpers from cuda-only adapters), `fa8730d6` (drop cuda gate from kt entries) |
| `kiln-kt-bridge` metal feature + `Device::Metal` mapping | `ae315509` (+ panic-catch test in `344c6d4a`) |

All of these are kept thin on purpose: each `_kt` entry bridges Device/DType
at the boundary and calls the existing candle-typed implementation. The
trait surface (`BackendRuntime`) is still candle-typed today. The point of
the `_kt` entries is so kiln-server / kiln-train can *swap their own imports
off candle* without first migrating the kiln-model internals — that
internal migration is Tier 3's biggest remaining work item.

### 1e. cuda-gate cleanup + Cargo hygiene

- `kiln-blas`: candle cudarc re-exports swapped for direct cudarc deps
  (`0d201199`). Still keeps `candle-core` as an optional `cublaslt`/`probe`
  dep.
- `kiln-tensor::fp8.rs`: candle cudarc re-exports swapped for direct
  cudarc (`c84ac6f4`).
- `kiln-tensor-id`: hoisted into its own dependency-free leaf crate
  (`fea65fbe`) so vulkan-kernel can depend on it without forming a path-dep
  cycle.
- `kiln-vulkan-kernel`: `publish=false` (`3c92a7e2`); `candle::TensorId`
  swap for `kiln_tensor_id::TensorId` (`07e8d342`).
- `kiln-opd-loss-kernel`: `kiln-kt-bridge` dep gated behind cuda feature
  (`fe4cfd1e`) so non-CUDA CI doesn't drag `cudarc` through nvcc probes.
- `kiln-model`: cuda gates dropped from `Device/DType _kt` parallel
  entries (`fa8730d6`) so multi-backend builds (Metal/Vulkan/CPU) can
  call them without `--features cuda`.

### 1f. kiln-server / kiln-train surface cleanup

Several mechanical migrations off candle Device/DType references where the
underlying API didn't actually require candle:

- `kiln-server::completions.rs` Metal device match → kt-typed
  (`f93e21fa`)
- `kiln-server::bench.rs` candle dtype matches consolidated
  (`8997dede`)
- `kiln-server::main.rs` precompile helpers retyped to `kt::Device`
  (`c49c1d74`)
- `kiln-server::state.rs` test CPU device usage consolidated
  (`1351087d`), Metal/Cuda dispatch routed through kt-bridge
  (`269d8f88` — HEAD)
- `kiln-train::cuda_train.rs` redundant `candle_core` test imports
  dropped (`0cd36f96`); cuda-gated candle imports consolidated in
  `trainer.rs` (`176bd897`)
- `kiln-train::echo.rs` + `opd.rs` test-mod candle imports dropped
  (`c6d9b009`, `8f9d38fe`)
- `kiln-train` adapter L2-norm receipt path migrated to kt (`806f839d`)
- `kiln-train` vk_*parity/smoke tests dropped redundant candle imports
  (`bb6ab0f2`)

### 1g. STOP docs (audited, no-code-change)

When a tier looked superficially migratable but the substrate revealed
deeper blockers, the audit landed as a STOP doc rather than churn:

- [`docs/kiln-server-candle-removal-stop-2026-05-27.md`](./kiln-server-candle-removal-stop-2026-05-27.md)
  (`02d5e88b`) — kiln-server cannot drop candle without Tier 3.
- [`docs/lora-bwd-kt-migration-stop-2026-05-27.md`](./lora-bwd-kt-migration-stop-2026-05-27.md)
  (`038cd756`) — the three LoRA `CustomOp::bwd` bodies cannot bridge-migrate
  without fused `kiln_lora_*_bwd` FFI kernels.
- In-source STOPs on test/example files (each `(#1082)` commit):
  - `kiln-rmsnorm-kernel::examples/phase10_microbench.rs` (`acd00bb4`)
  - `kiln-flce-kernel::src/tests.rs` parity tests (`9a95adc2`)
  - `kiln-opd-loss-kernel::src/tests.rs` parity tests (`6d3fc88d`)
  - `kiln-server::tests/real_model_integration.rs` (`684f968a`)
  - `kiln-vulkan-kernel::examples/{decode_microbench,vk_mlp_probe}.rs` (`46a838ff`)
  - `kiln-tensor::src/cuda_allocator.rs` module docstring (`06636f64`,
    in-source — this STOP lives **inside the source file**, not in `docs/`)

---

## 2. Candle footprint at HEAD (`main@269d8f88`)

Audit method:
```bash
grep -rln "use candle_core\|use candle_nn" crates/
```

Files importing candle, by crate (descending):

| Crate | Files | Category |
| --- | --- | --- |
| `kiln-kt-bridge` | 43 | **By design** — this crate IS the candle ↔ kt boundary. Almost all files are CUDA parity tests under `tests/` and the public bridge surface in `src/{lib,forward_op}.rs`. |
| `kiln-model` | 27 | **Tier 3 — production migration work item**. Spread across forward.rs, cuda_train.rs, paged_kv_cache.rs, all 4 backends, vk_forward.rs, fp8.rs, marlin_proj.rs, sampling.rs, lora_loader.rs, etc. |
| `kiln-vulkan-kernel` | 8 | **Tier 2 — Family 1/2/3 blockers**. `vk_tensor.rs` (5 imports), `kernels.rs` (1 import — the legacy dispatch surface), `resident.rs`, 3 tests + 2 examples. Bulk of footprint already migrated (was 20 a few weeks ago). |
| `kiln-train` | 7 | **Tier 3-downstream**. cuda_train.rs, trainer.rs, opd.rs, echo.rs, train_receipt.rs, 2 vk tests. Each call site is gated on `GpuWeights` / `LinearAttentionState` / `ModelRunner` candle-typed signatures. |
| `kiln-opd-loss-kernel` | 5 | 1 production file (`lib.rs`), 1 phase_b.rs (kt-bridge migrated, still uses candle types in signatures), kt_forward_op.rs (by-design candle shim), 1 test, 1 example. |
| `kiln-flce-kernel` | 5 | Same shape: `lib.rs`, `phase_b.rs` (kt-bridge migrated), `kt_forward_op.rs`, `kt_api.rs`, `tests.rs`. |
| `kiln-tensor` | 4 | **The substrate blocker**. `cuda_allocator.rs` + `cuda_storage.rs` hold `Arc<candle_core::cuda_backend::CudaDevice>` for stream affinity; `metal_allocator.rs` + `metal_storage.rs` hold `Arc<candle_core::metal_backend::MetalDevice>` for the same reason. |
| `kiln-rmsnorm-kernel` | 4 | `lib.rs` (the candle-typed surface + `RmsNormCustomOp` shells), `kt_api.rs` (bridge surface), `kt_forward_op.rs` (by-design shim), `examples/phase10_microbench.rs` (STOPped). |
| `kiln-server` | 3 | `device.rs`, `state.rs`, `tests/real_model_integration.rs` — all already STOP-doc'd as downstream of Tier 3. |
| `kiln-gdn-kernel` | 2 | `lib.rs` (chunk-family candle-typed surface + in-lib parity tests), `examples/gates_bench.rs`. |
| `kiln-blas` | 2 | `cublaslt_handle.rs` (optional `cublaslt`/`probe` features), `tests/cublaslt_handle_smoke.rs`. |

Total: **110 files** across **11 crates** that import candle, vs. the earlier
plan's snapshot of 15 crates and ~150 imports — meaningful reduction has
already shipped at Tier 1, but Tier 3 dominates the residual.

### Breakdown by "removal class"

**A. By-design / shim — stay until Tier 4 deletes the bridge:**
- `kiln-kt-bridge/*` (43 files): the bridge.
- `kiln-{rmsnorm,flce,opd-loss}-kernel/src/kt_forward_op.rs` (3 files):
  the candle `CustomOp{1,2,3}` integration shim.
- `kiln-{rmsnorm,flce,opd-loss}-kernel/src/kt_api.rs` (3 files):
  the kt-typed FFI wrappers (use candle's `cudarc` re-exports for
  CUDA stream plumbing).

**B. Substrate (kiln-tensor) — block all downstream:**
- `kiln-tensor/src/{cuda_storage,cuda_allocator,metal_storage,metal_allocator}.rs`
  (4 files). Every other CUDA / Metal kt-API call ultimately threads through
  these.

**C. Production code — migrates as Tier 3 closes:**
- `kiln-model/{forward,cuda_train,paged_kv_cache,backend/*,fp8,marlin_proj,
  sampling,lora_loader,packed_weight_registry,kv_cache,decode_buffers,
  speculative,adapter_merge,mtp_debug,generate,vk_forward,vk_decode_resident,
  paged_kv_cache_kt,cuda_graph}.rs` (≈22 production files)
- `kiln-train/{cuda_train,trainer,opd,echo,train_receipt}.rs` (5 files)
- `kiln-server/{device,state}.rs` (2 files)
- `kiln-vulkan-kernel/src/{kernels,vk_tensor,resident}.rs` (3 files)
- `kiln-{rmsnorm,gdn,flce,opd-loss}-kernel/src/lib.rs` and `phase_b.rs`
  (≈6 files, candle-typed surface + parity scaffolds)
- `kiln-gdn-kernel/src/lib.rs`, `kiln-blas/src/cublaslt_handle.rs`,
  `kiln-tensor/src/fp8.rs` candle cudarc re-exports already swapped — these
  are just the source-file usages of candle types in signatures.

**D. Tests / examples — delete when candle removed:**
- `kiln-{vulkan-kernel,model,kt-bridge}/tests/*.rs` (most of kt-bridge's 41
  parity tests, several model parity tests).
- `kiln-{rmsnorm,opd-loss,flce}-kernel/src/tests.rs` (in-lib parity tests
  comparing candle reference to kt kernel — already STOP-doc'd as
  "stay on candle until candle is removed entirely").
- `kiln-{rmsnorm,gdn,vulkan,opd-loss}-kernel/examples/*.rs` (5 files —
  STOP-doc'd in source).
- `kiln-server/tests/real_model_integration.rs` (STOP-doc'd).
- `kiln-train/tests/{vk_cuda_opd_parity,vk_train_smoke}.rs` (Phase 7
  gates).

---

## 3. Tier-4 / Tier-5 critical path

This is the section a fresh agent should optimize against. Each item below
is rated by **(a) downstream blast radius** (how many other migrations it
unblocks) and **(b) substrate vs production cost** (whether the work is in
kiln-tensor / kiln-kt-bridge or in a hot Rust production file).

### CP-1. `kiln-tensor::CudaStorage` candle removal (HIGHEST LEVERAGE)

**Location:** `crates/kiln-tensor/src/cuda_storage.rs` (the file holds an
`Arc<candle_core::cuda_backend::CudaDevice>` in every `CudaStorage`).

**Why it's the bottleneck:** every CUDA kt_api call in every kernel crate
ultimately reaches `.candle_device().cuda_stream()` to plumb a CUDA stream
to the FFI symbol. As of `06636f64` (the cuda_allocator STOP doc-only commit),
the order-of-operations is now formally captured:

1. Add a parallel `CudaStorage::zeros_kt(ctx: Arc<CudaContext>, ...)` that
   allocates via `ctx.default_stream().alloc_zeros::<u8>` and stores
   `Arc<CudaContext>` instead of `Arc<CudaDevice>` on `CudaStorage`
   (likely via an internal enum + dual accessors so existing kernel-crate
   FFI sites keep compiling unchanged).
2. Migrate the dozen-plus kernel-crate FFI call sites that reach
   `.candle_device().cuda_stream()` to a `cuda_stream_raw()` / `CUstream`
   accessor that **already exists** on `CudaStorage` (`d561dbf8`).
3. Flip `CudaStorage::zeros` itself to take `Arc<CudaContext>`, drop the
   `candle_device` field.
4. `CudaAllocator` becomes a one-line type substitution (it has zero
   external callers; see `cuda_allocator.rs` STOP note line 39).

**Unblocks:**
- `kiln-conv1d-kernel` / `kiln-marlin-gemm` / `kiln-flash-attn` Tier-1
  closures' last residual blocker (the candle-typed `cudarc` re-exports in
  `kt_api.rs` files).
- `kiln-rmsnorm-kernel` / `kiln-gdn-kernel` / `kiln-opd-loss-kernel` /
  `kiln-flce-kernel` candle-core dep drops (after their respective
  production-caller migrations complete).
- `kiln-tensor::cuda_allocator.rs` candle drop (mechanical follow-on).

**Cost:** ~1–2 substrate PRs in `kiln-tensor`, then a sweep of ≤15 FFI
sites across the kernel crates to swap `candle_device().cuda_stream()` →
`cuda_stream_raw()`. Step 2 is sweep-only; step 1 + step 3 are the actual
substrate change.

### CP-2. `kiln-tensor::MetalStorage` candle removal (parallel to CP-1)

**Location:** `crates/kiln-tensor/src/{metal_storage,metal_allocator}.rs`.

**Shape:** identical to CP-1. `MetalStorage` holds
`Arc<candle_core::metal_backend::MetalDevice>` for command-queue ownership.
The same enum-internal-with-dual-accessors trick lets a kt-native Metal
handle (`MTLDevice` + `MTLCommandQueue` directly from `metal-rs` /
`candle_metal_kernels::metal`) coexist while production sites migrate.

**Unblocks:**
- `kiln-mps` Tier-2 close (Apple Silicon native Metal backend).
- Multi-backend `kiln-server` builds (today's Metal path bridges through
  the candle Metal Device; without candle, kiln-server's Metal smoke would
  break).
- The `kiln_tensor::metal_storage` candle parity test in `softmax_last_dim`
  (lines 196–298) — the doc-comment there explicitly calls out the candle
  dispatch as the planned substrate replacement.

**Cost:** ~1–2 PRs. Parallel-track with CP-1 — neither depends on the
other.

### CP-3. `model_forward_kt` for non-CUDA backends (multi-backend unblock)

**Location:** `crates/kiln-model/src/forward.rs:21081` —
`pub fn model_forward_kt(...)`.

**Current state:** the entry is `#[cfg(feature = "cuda")]` only. It
delegates to `model_forward(...)` and bridges the returned candle Tensor
to `kiln_tensor::Tensor` via `kt_tensor_from_candle_cuda_borrow` — which
hard-requires CUDA storage on the returned tensor.

**Why it matters:** `kiln-server`'s STOP doc identifies this as one of
five Tier-3 prerequisites for kiln-server to drop candle. Without
non-CUDA `model_forward_kt`, kiln-server is forced to keep
`model_forward(...)` (candle-typed return) for the Metal/Vulkan/CPU code
paths — even after CP-1 lands.

**Unblocks:**
- `kiln-server::completions.rs` prompt-logprobs path (Metal + CPU).
- `kiln-server::bench.rs` non-CUDA backends.
- `kiln-train` non-CUDA training paths (for echo + opd).

**Cost:** ~1 PR. Bridge variant per backend:
`kt_tensor_from_candle_metal_borrow`, `kt_tensor_from_candle_cpu_borrow`,
or a single `kt_tensor_from_candle_borrow` that dispatches at the
boundary. Already partially shaped by `kt_dtype_from_kc` helpers
referenced in the kiln-server STOP doc.

### CP-4. kt-native autograd `Var` / `Tape` adoption in production

**Status:** `kiln-autograd` crate exists with `BackwardOp` trait + `Tape` +
30+ `BackwardOp` impls + `KILN_DETECT_ANOMALY` end-to-end. **Zero
production callers** — see `CANDLE_REMOVAL_PLAN.md` §"kt-autograd
readiness investigation" (`14f02e8c`).

**Why it matters:** the `KtForwardOp{1,2,3}` shim (CP-`1c` above) is a
*stop-gap*. It lets the inside of each fused op be pure kt while the
outside is still candle autograd. To drop the kt-bridge entirely (Tier 4)
and the candle vendor tree (Tier 5), the training loop has to swap from
`loss.backward()` → `tape.backward(loss_id, ...)` in a single coordinated
PR.

**Unblocks:**
- Final deletion of `kiln-kt-bridge::forward_op::KtForwardOp{1,2,3}` shim
  (replaced by `kiln-autograd::BackwardOp` impls per fused kernel).
- All `kiln-train` Var/AdamW/Variables → kt-optim equivalents (the
  `kiln-optim` crate already has the kt-side equivalents wired to
  `kiln-autograd` per its dev-dep).
- `kiln-model::forward.rs`'s 7 `CudaLora*` / `CudaSigmoidMul*` /
  `CudaFlashAttention*` / `CudaRotaryOne*` `CustomOp{2,3}` impls (they
  become `BackwardOp` impls).

**Cost:** **MULTI-PR substrate work**. Need:
- `kiln-tensor::Tensor` to hold an `Option<TensorId>` tape-handle.
- `kiln-autograd::Var` (or equivalent) to mirror candle's `Var`.
- `kiln-autograd::Tape::backward` to coexist with `loss.backward()` during
  the migration period.
- Cross-crate parity tests proving SFT/GRPO/OPD numeric parity vs candle's
  autograd for at least 1k training steps.

**This is the heaviest single item on the Tier-4 critical path.** Other
substrate items above (CP-1/CP-2/CP-3) are 1–2 PRs each. This one is
likely 5–10 PRs spanning multiple weeks.

### CP-5. `LinearAttentionState::new_kt` parallel entry

**Location:** `crates/kiln-model/src/forward.rs:4985` —
`LinearAttentionState::new(config, device: &candle_core::Device)`.

**Why it matters:** kiln-server's STOP doc names this as one of five
Tier-3 prerequisites. `kiln-server::state.rs:3320-3327` and
`completions.rs:3607` construct linear-attn state with a candle Device,
forcing the candle import on every code path that boots a model.

**Unblocks:**
- Same set as CP-3 (kiln-server multi-backend).
- `kiln-server::state.rs`'s 25 `candle_core::*` references collapse by
  ~5–8.

**Cost:** ~1 PR. The constructor body uses `device` only for
`Tensor::zeros(... device)` — a kt-typed sibling allocates via
`kiln_tensor::Tensor::zeros_on` (now device-parametric, `c41e3870`) and
returns the kt-typed state. The struct's internal `conv_states:
Vec<Tensor>` field stays candle-typed for now (full migration is part of
CP-7 below).

### CP-6. `kiln-vulkan-kernel` legacy dispatch surface `_bytes` siblings

**Location:** `crates/kiln-vulkan-kernel/src/kernels.rs` — the 49
`dispatch_*` functions returning candle `Tensor`.

**Why it matters:** the `CANDLE_REMOVAL_PLAN.md` §"kiln-vulkan-kernel
blocker breakdown" identifies this as the largest single piece of Family 1
work. The pattern is established (`dispatch_kernel_bytes:436` precedent in
`60c48916`, `upload_*_buffer_from_slice` helpers in `6f1cabdc`): factor
dispatch bodies to take `&[u8]` + shape instead of `&Tensor`, keep the
candle-typed wrapper as a thin shim.

**Unblocks:**
- `kiln-model::backend/{vulkan,vulkan_linear_op,vulkan_lora_op}.rs` +
  `vk_decode_resident.rs` candle drops (≈30 call sites across the
  workspace).
- `kiln-train::vk_train.rs` candle drop.
- `kiln-vulkan-kernel/examples/{vk_mlp_probe,decode_microbench}.rs` candle
  drops (already STOP-doc'd; this is the unblocker).

**Cost:** ≈49 mechanical `_bytes` sibling additions, can be done in 5–10
small PRs. The plan calls for starting with
`dispatch_mlp_gate_up_decode_cached_bytes` (smallest self-contained
example).

### CP-7. `kiln-model` internal candle migration (Tier 3 closeout)

**Location:** 22+ production files in `crates/kiln-model/src/`.

**Status:** 40+ `try_kt_*` opt-in gates landed; the
`KILN_USE_KT_API_ALL=1` master switch is the integration point.
`PagedKvCacheKt` migration is partial (~10 decode call sites remain on
candle).

**Why it's hard:** the candle surface is the most diffuse and the most
performance-sensitive in the workspace. Every `try_kt_*` migration has to
prove a parity cycle before its env gate flips default-on. Several Phase-7
gates already flipped (`fec54f8b` for LORA_ADD, `57fbf5e0` for CAT_*,
`329df3be` for EMBEDDING, `b1e971d1` for ARGMAX, plus
`c156b5a7`/`45131f04`/`786ad3c1`/`26b4d0bf`/`1155cc47`/`6c8d5ade` for
softmax/l2/flash-attn).

**Unblocks:**
- Tier-4 (`kiln-kt-bridge` deletion) becomes meaningful only after this is
  done.
- `kiln-server` STOP doc converts to a mechanical migration PR (per its
  "What unblocks this work" section).
- `kiln-train` candle drop.

**Cost:** **MULTI-MONTH residual work**. Probably the largest single
work category remaining after CP-1/CP-2/CP-3/CP-4 land. Best handled as
ongoing Phase-7 gate-flipping work in parallel with the higher-leverage
substrate items above.

### CP-8. `kiln-kt-bridge` deletion (Tier 4 endpoint)

**Location:** `crates/kiln-kt-bridge/`.

**Status:** by design last. The bridge IS the candle ↔ kt boundary. Its
`candle-core` dep is removed only when every production path is on
kt-typed tensors; at that point the bridge either collapses to a thin
device-id translation layer (probably absorbed into `kiln-tensor`
directly) or is deleted entirely.

**What changes:** the 6 helper functions
(`{candle_dtype_to_kt,kt_dtype_to_candle,kt_device_from_candle,
candle_device_from_kt,kt_tensor_from_candle_cuda_borrow,
kt_tensor_to_candle_cuda_copy}`) and the `KtForwardOp{1,2,3}` shim all
become dead code at this point. Tests under `tests/cuda_*_parity.rs` (41
of them) get deleted because they're comparing candle ops to kt ops —
once candle is gone, the kt side is the only side.

### CP-9. Vendor delete (Tier 5 endpoint)

```bash
cargo tree --workspace -i candle-core    # must be empty
rm -rf vendor/candle-core/
sed -i "/candle-core.*path.*vendor/d" Cargo.toml
```

Doc-only / config-only at this point. The only thing this is gated on is
every prior step succeeding.

---

## 4. Recommended sequence (next 3 substrate items)

If a fresh agent has bandwidth for **one** task, do **CP-1**.
If they have bandwidth for **three**, do **CP-1 + CP-2 + CP-3**.

**1. CP-1 (`kiln-tensor::CudaStorage` → `Arc<CudaContext>`).** Highest
leverage in the workspace today. Unblocks Tier-1 Cargo drops on four
kernel crates **and** removes the substrate-blocker that's been called
out in every kernel-crate audit since the plan was written. The
order-of-operations is fully captured in
`crates/kiln-tensor/src/cuda_allocator.rs:18-78` — a fresh agent can read
the STOP note and execute. ~1–2 PR.

**2. CP-2 (`kiln-tensor::MetalStorage` → kt-native handle).**
Parallel-track to CP-1; same shape; ~1–2 PR. Independent unblock for
Apple Silicon Metal builds. Pair the two because the in-source comments
on `metal_storage.rs:196-298` already specify what kt-native Metal
dispatch should look like (the `softmax_last_dim` body is the smallest
candle-bridge in the file and a good first migration target).

**3. CP-3 (`model_forward_kt` non-CUDA backends).** ~1 PR. After CP-1
+ CP-2 give kiln-tensor a candle-free CUDA + Metal substrate, this
extends `model_forward_kt` to dispatch through the correct kt borrow
helper per backend. After this lands, `kiln-server`'s STOP doc becomes
re-auditable — and the per-file migrations identified in the STOP doc
become mechanical 5-line edits.

**Why these three first:** they together account for the substrate that
gates everything else. Once they land, CP-5 (`LinearAttentionState::new_kt`)
becomes a trivial follow-on; CP-7 (`kiln-model` internal migration)
acquires a much smaller surface to migrate (because the per-call kt
boundary will already work for all 4 backends); and CP-4 (kt-native
autograd in production) becomes the only remaining substrate item
between "Tier-4 starts" and "kt-bridge can be deleted."

**What NOT to do next:** CP-6 (vulkan kernel `_bytes` sweep) is
mechanical and high-volume but has no downstream amplifier; it can be
done at any time without affecting any other CP. CP-4 (kt-native
autograd) is the largest single item on the critical path but is
multi-week and partly research-coded — it shouldn't be the first
substrate task a new agent picks up. CP-7 (`kiln-model` Phase-7 gate
flipping) is ongoing maintenance work and benefits from CP-1/CP-3
landing first (so the gates flip with fewer code-path forks).

---

## 5. References — every existing STOP / audit doc

External (`docs/`):
- [`CANDLE_REMOVAL_PLAN.md`](./CANDLE_REMOVAL_PLAN.md) — Tier 0–5
  canonical roadmap. Last refreshed 2026-05-26 in-doc; supersedeable
  per-tier marks below this consolidation.
- [`kiln-server-candle-removal-stop-2026-05-27.md`](./kiln-server-candle-removal-stop-2026-05-27.md)
  — kiln-server is downstream of Tier 3 (commit `02d5e88b`).
- [`lora-bwd-kt-migration-stop-2026-05-27.md`](./lora-bwd-kt-migration-stop-2026-05-27.md)
  — three `CudaLora*::bwd` bodies blocked on fused `kiln_lora_*_bwd`
  kernels (commit `038cd756`).

In-source (module docstrings — NOT in `docs/`):
- `crates/kiln-tensor/src/cuda_allocator.rs:18-78` — Phase 7 swap of
  `Arc<CudaDevice>` → `Arc<CudaContext>` blocked on `CudaStorage::zeros`
  migrating first (commit `06636f64`).
- `crates/kiln-tensor/src/cuda_storage.rs:181-275` — Phase 7 zero-copy
  candle→kt adapter docstring; the parallel STOP for the
  `candle_device: Arc<CudaDevice>` field migration.
- `crates/kiln-rmsnorm-kernel/examples/phase10_microbench.rs` — header
  documents why this example stays on candle (commit `acd00bb4`).
- `crates/kiln-flce-kernel/src/tests.rs` — parity tests intentionally
  stay on candle until candle is removed (commit `9a95adc2`).
- `crates/kiln-opd-loss-kernel/src/tests.rs` — same (commit `6d3fc88d`).
- `crates/kiln-server/tests/real_model_integration.rs` — why this stays
  on candle (commit `684f968a`).
- `crates/kiln-vulkan-kernel/examples/{decode_microbench,vk_mlp_probe}.rs`
  — candle-removal blockers (commit `46a838ff`).

Re-audited via `git log --grep '#1082'` against `main@269d8f88` —
approximately **100 commits** are in flight today on issue #1082, all of
them prefixed `(#1082)` per kiln PR conventions (no `CE:` prefix; no
`closes #1082` text).

---

For #1082.
