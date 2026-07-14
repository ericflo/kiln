# ROCm Backend — First-Class Parity Plan

> Plan to add a first-class ROCm/HIP backend to Kiln at full parity with the
> existing CUDA, Vulkan, and Metal backends. Produced by a multi-agent
> mapping + design pass and verified against the source.

## Verified facts (spot-checked against the tree)

- **ROCm 7.2.4 is already installed** at `/opt/rocm` — `hipcc`,
  `libhipblaslt.so.1`, `librocblas.so.5` all present. The *compute path* is the
  entire value-add; detection (`kiln-core/src/vram.rs`, DRM sysfs vendor
  `0x1002`) already exists.
- **The kernel FFI is backend-neutral `extern "C"`.** There is **zero**
  PTX/nvrtc/`launch_builder` on the Rust side (grep clean). The 28
  `kiln-tensor/csrc/*.cu` (57 `.cu` repo-wide) are compiled by `nvcc` in
  `build.rs` into a static lib and called by stable C symbol — so hipified
  `.cu` drop into the same `extern "C"` lib with no Rust FFI changes.
- **cudarc usage is narrow** — ~8 symbols: `DevicePtr` ×38, `CudaContext` ×13,
  `CudaSlice` ×7, `CudaGraph` ×4, `CudaStream` ×3, plus a few `sys`/`result`
  calls. A bounded `kiln-hip` mirror crate is tractable.
- **GEMM is cuBLASLt** (`kiln-blas/csrc/cublaslt_{matmul,probe}.cu`) → maps 1:1
  to **hipBLASLt** (installed).
- **Wave-size hazard is real and pervasive** — 14 `.cu` files hardcode 32-lane
  warps (`__shfl_xor_sync(0xFFFFFFFF, …)`, `tid & 31`, `(blk + 31) / 32`).
  These compile fine but **silently corrupt on AMD's 64-wide wavefronts**. This
  is the single highest correctness risk.
- `Device` is a `#[non_exhaustive]` enum (`Cpu/Cuda/Metal/Vulkan`) with sibling
  `Backend` + `DeviceLocation` and a `Backend::ALL: [Backend; 4]`. `Storage` is
  `Arc<dyn StorageBackend>` (trait, not enum) — so `RocmStorage` plugs in
  cleanly.

---

## Executive summary

Kiln dispatches over a `#[non_exhaustive] Device { Cpu, Cuda, Metal, Vulkan }`
enum, and every backend is a *parallel cascade*: one top-level `--features cuda`
on `kiln-server` fans out across ~13 crates via a documented feature graph,
terminating in (a) a narrow cudarc runtime surface (~8 symbols) and (b) 28
`extern "C"` AOT-compiled `.cu` kernels invoked through a backend-neutral C-ABI
that passes raw `void*` streams and `u64` device pointers. There is **no
PTX/nvrtc/launch_builder on the Rust side** — kernels are compiled by `nvcc` in
`build.rs` into a static lib (`kiln_tensor_cuda_ops`) and called by stable C
symbol. First-class ROCm means replicating that cascade edge-for-edge with a
`rocm` feature, adding `Device::Rocm(usize)` + all internal match arms + growing
`Backend::ALL` to `[Backend; 5]`, vendoring a bounded `kiln-hip` binding that
mirrors cudarc's used surface 1:1, compiling the same 28 `.cu` with `hipcc`
behind a `kt_gpu_compat.cuh` shim, wiring `hipBLASLt` as the cublasLt drop-in,
and porting all 9 kernel crates (forward AND backward) — gated so every
default-members build stays green on hosts with no ROCm toolchain.

**Headline approach.** A three-tier hybrid: **shared-source hipify** for the ~28
elementwise/reduce/norm tensor kernels + the loss/rmsnorm/gdn/conv1d families —
one `.cu` source compiled by two compilers behind a 40-line
`#ifdef __HIP_PLATFORM_AMD__` shim, with a mandatory **wave-size fix**; a bounded
**`kiln-hip`** binding so `RocmStorage`/`RocmAllocator`/`active_rocm_stream` are
mechanical retypes of the candle-free `cuda_*.rs` files; **hipBLASLt** as the
cublasLt 1:1 drop-in in a `kiln-rocblas` crate; and for the two genuinely
un-hipifiable hot kernels (`kiln-flash-attn`'s vendored CUTLASS/CuTe,
`kiln-marlin-gemm`'s inline `mma.sync`/`ldmatrix`/`cp.async` PTX): ship a correct
kt-composite/dequant→hipBLASLt fallback first so `--features rocm` is end-to-end
functional and parity-green from day one, then climb the MFMA/CK perf ladder as
profiler-justified follow-up PRs.

---

## Chosen strategy: a three-tier hybrid

Approach **1 (Source-Port Maximalist) as the foundation**, sequenced with
**Approach 3 (Incremental-via-fallback Pragmatist)**'s always-green ordering and
correctness floor, with **Approach 2 (Native-HIP Performance)**'s MFMA/CK work
as the deferred perf ladder for exactly the two kernels that cannot be hipified.

### Why Approach 1 is the core
1. **The kernel FFI is backend-neutral extern-C** (`cuda_storage.rs:522`
   declares `unsafe extern "C"` launchers taking `*const c_void` stream) — no
   PTX on the Rust side, so hipifying the `.cu` into the same `extern "C"`
   static lib drops in with **zero** Rust FFI changes.
2. **cudarc usage is narrow (~8 symbols)** — a `kiln-hip` mirror is *bounded*
   work, and `cuda_stream_priority.rs` already demonstrates the raw-FFI-RAII
   pattern. Maintenance win: the 28 tensor kernels + rmsnorm/gdn/conv1d/opd
   families stay ONE source compiled by two compilers behind
   `kt_gpu_compat.cuh`, so future CUDA kernel fixes land on ROCm with no second
   edit — the strongest defense against backend drift.

### Why Approach 3's sequencing wraps it
`kiln-flash-attn` (vendored CUTLASS/CuTe, ~520 headers, SM80+
`cp.async`/`ldmatrix`/GMMA) and `kiln-marlin-gemm` (inline
`mma.sync.aligned.m16n8k16`/`ldmatrix`/`cp.async`/`lop3` PTX) **physically
cannot be hipified** (hipify-perl translates neither inline asm nor CUTLASS).
Ship those two through a **kt-composite / dequant→hipBLASLt fallback first**, so
`--features rocm` is correct, complete, and parity-green from the first kernel
PR. (Reject Approach 3's *literal* route-through-Vulkan idea — it forces a dual
GPU stack inside one logical `Device::Rocm`. The kt-composite fallback gets the
same correctness floor on the pure HIP stack.)

### Why Approach 2 is the ladder, not the start
Flash-attn and marlin are where AMD throughput is won, and hand-MFMA/CK is
unavoidable for them — but starting there front-loads the rarest skill set and
leaves the backend non-building for months. So MFMA marlin and CK/ck-tile
flash-attn are **Phase R.10 perf follow-ups behind the same `rocm` feature**,
each gated by a numerical-parity test on real gfx942. Approach 2's other
contribution — the **wave-size-64 reduction audit** — is *not* deferred; it is
mandatory Tier-1 work (Phase R.5).

### Concrete technical decisions
- **Binding layer:** new `crates/kiln-hip` (sys via bindgen over
  `hip/hip_runtime_api.h` + hand-written safe RAII mirroring the ~8 cudarc
  symbols), pinned once in root `[workspace.dependencies]`, dynamic-linking
  against `libamdhip64.so` so toolchain-less `cargo check` passes.
- **Kernel port:** shared single source + `kt_gpu_compat.cuh`
  (`#ifdef __HIP_PLATFORM_AMD__` → `hip/hip_runtime.h`, `hip_bf16.h`,
  `hip_fp16.h`; `KILN_WARP` 32/64; portable `warp_reduce_*`), injected via
  `-include`. Compile with `hipcc` (not `cc::Build::cuda(true)`, hardwired to
  nvcc) in `build_rocm()` arms.
- **BLAS:** `kiln-rocblas` wrapping **hipBLASLt** (heuristic + algo-blob +
  epilogue + bias + F32-compute-on-bf16), reusing kiln-blas's
  `algo_cache`/`workspace_pool`/`backend_matmul` verbatim, **rocBLAS gemmEx
  fallback** for gfx1100. Reached via `kiln-tensor/rocm features=["hipblaslt"]`.
- **Matrix cores:** MFMA (`__builtin_amdgcn_mfma_*`) on CDNA gfx90a/gfx942,
  WMMA/rocWMMA on RDNA3 gfx1100 — only in the Phase R.10 ladder.
- **Graph capture:** `kiln-graph-rocm` mirroring `kiln-graph-cuda`, real
  `hipGraph_t` capture mirroring `cuda_graph.rs` (all 6 capture calls have
  direct hip* analogs).
- **gfx targets:** primary **gfx942** (MI300X), then **gfx90a**
  (MI200/210/250), then **gfx1100** (RX 7900 RDNA3 — thinner hipBLASLt, WMMA not
  MFMA). `KILN_ROCM_ARCHS=gfx90a;gfx942;gfx1100`.

---

## Definition of Done — parity surface

Organized by subsystem. Every box must be checked for "as first-class as CUDA."

### 1. Device / enum substrate (`crates/kiln-tensor/src/device.rs`)
- [ ] `Device::Rocm(usize)` variant added to the `#[non_exhaustive]` enum.
- [ ] `DeviceLocation::Rocm { gpu_id }` added (mirror `Cuda { gpu_id }` exactly).
- [ ] `Backend::Rocm` added.
- [ ] `Device::short_name` → `"rocm:{i}"`; `index` → folds into `Some(i)`;
      `backend` → `Backend::Rocm`; `location` → `DeviceLocation::Rocm`.
- [ ] `Backend::short_name` → `"rocm"`; `Backend::ALL` grown `[;4]`→`[;5]`,
      appending `Rocm` **last** (minimize CSV/parity-table churn).
- [ ] `backend_all_order` test (device.rs:230) asserts
      `[cpu, cuda, metal, vulkan, rocm]`.
- [ ] All ~40 internal exhaustive `match Device::` sites
      (kiln-tensor/kiln-model/kiln-server) get a `Device::Rocm(_)` arm.
- [ ] All 6 `match Backend::` sites (bench.rs, state.rs,
      graph-{metal,vulkan,cuda}/lib.rs) get a `Backend::Rocm` arm.
- [ ] Grep `[_; 4]` / `[Backend; 4]` in `bench-results/` + fixtures; widen each.

### 2. Rust↔HIP binding (`crates/kiln-hip`, new — excluded from default-members)
- [ ] Pinned in root `[workspace.dependencies]` next to the cudarc block, with
      `default-features=false` + dynamic-linking so `cargo check --features
      rocm` passes with NO ROCm.
- [ ] `kiln-hip-sys` raw FFI: bindgen over `<hip/hip_runtime_api.h>`
      (hipMalloc/MallocAsync/Free/FreeAsync, hipMemcpy{Htod,Dtoh,Dtod}Async,
      hipMemsetD8Async, hipStreamCreate/WithPriority/Destroy/Synchronize,
      hipStreamBeginCapture/EndCapture/IsCapturing, hipGraph*). Links `amdhip64`.
- [ ] Safe RAII mirroring the *used* cudarc surface ONLY:
      `RocmContext::{new, default_stream, new_stream}`, `RocmStream` (raw
      `hipStream_t`, `Drop=hipStreamDestroy`, **public** `hip_stream()`,
      `synchronize`, `alloc_zeros`, `alloc`, `memcpy_dtoh`), `RocmSlice<u8>`
      (`hipDeviceptr_t`+len, `Drop=hipFreeAsync`, `device_ptr(&stream)`).
- [ ] Free fns `memcpy_htod_async`/`memcpy_dtod_async`/`memset_d8_async`.
- [ ] `RocmGraph`/`RocmGraphExec` wrapping `hipGraph_t`/`hipGraphExec_t`
      (begin_capture Relaxed, end_capture AutoFreeOnLaunch, capture_status,
      launch).
- [ ] Explicit `unsafe impl Send/Sync` for `RocmContext`/`RocmStream`.
- [ ] `hipDeviceptr_t` (`void*`) normalized to a `usize`/pointer wrapper so the
      `SliceOwner::Borrowed { ptr, .. }` retype is clean (CUdeviceptr is `u64`).
- [ ] Stream-priority sign convention verified (lower=higher, like CUDA).
- [ ] `hipRuntimeGetVersion` runtime assert against a documented minimum.

### 3. Storage / allocator / stream (`crates/kiln-tensor/src/`)
- [ ] `rocm_storage.rs` — retype of `cuda_storage.rs`
      (`SliceOwner{Owned|Borrowed}`, `zeros_ctx`/`alloc_uninit_ctx`/
      `from_borrowed`, FFI accessors `device_ptr_raw`/`rocm_stream_raw`).
- [ ] `RocmStorage` implements the full `StorageBackend` trait.
- [ ] `rocm_allocator.rs` — retype of `cuda_allocator.rs` (Owned/Pool/Frozen +
      cache + reserved/peak accounting + `warm()`).
- [ ] `active_rocm_stream` + unsafe graph-only `with_rocm_graph_capture_stream` + `ActiveStreamGuard`
      thread-local.
- [ ] `capture_alloc` ROCm Borrowed-view path (HIP-graph pointer pinning).
- [ ] `primary_rocm_context(idx)` accessor + `rocm_is_available()` free fn.
- [ ] `rocm_stream_priority.rs` via `hipStreamCreateWithPriority` +
      `hipDeviceGetStreamPriorityRange`.

### 4. RocmStorage op surface (the 28 csrc ops — fwd AND bwd where applicable)
- [ ] contiguous/copy_async, index_select, elementwise, activation, cast (incl.
      fp8 e4m3/e5m2), softmax, reduce_last/arbitrary_axis, argmax, topk,
      masked_fill, scatter_add (+ bf16 atomics — `-munsafe-fp-atomics` /
      f32-accumulate fallback), cross_entropy (fwd+bwd), concat, rope, dropout
      (rocRAND RNG parity), rmsnorm (fwd+bwd), layernorm (fwd+bwd), scalar_op,
      clamp_pow, compare, where_select, diag, binary_minmax, scan_axis, lerp,
      fp8, is_finite_reduce.
- [ ] **Wave-size fix applied** to every two-level reduction — `KILN_WARP`
      (32/64) constexpr replacing `0xFFFFFFFF`, `tid & 31`, `tid / 32`, `>> 5`,
      `(blk+31)/32`.

### 5. Matmul / BLAS (`crates/kiln-rocblas`, new + `kiln-tensor/src/rocm_matmul.rs`)
- [ ] `kiln-rocblas` mirrors `kiln-blas`: reuse `algo_cache.rs` +
      `workspace_pool.rs` + `backend_matmul.rs` verbatim.
- [ ] `csrc/hipblaslt_matmul.cu` — 1:1 port of `cublaslt_matmul.cu`
      (`cublasLt*→hipblasLt*`, `CUBLAS_COMPUTE_32F→HIPBLASLT_COMPUTE_F32`,
      `CUDA_R_16BF/16F/32F→HIP_R_*`, keep `C^T=B^T@A^T` col-major swap, per-N
      bias pointer, verify `HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE`). Preserve
      `KILN_BLAS_ERR_*` codes.
- [ ] `csrc/hipblaslt_probe.cu` — heuristic + algo-blob (re-check
      `sizeof(hipblasLtMatmulAlgo_t)` vs `KILN_BLAS_ALGO_BLOB_MAX=256`).
- [ ] Epilogue parity: BIAS/RELU/GELU/BIAS_GELU → `HIPBLASLT_EPILOGUE_*`; SiLU
      keeps kt-activation fallback.
- [ ] dtype: `a.dtype()==b.dtype()` BF16/F16/F32 with F32 compute (CUDA parity).
- [ ] `rocm_matmul.rs` — clone of `cuda_matmul.rs` with per-device
      `HipblasLtMatmulHandle` registry keyed by device_index.
- [ ] Autotune cache: distinct `hipblaslt-` prefix + gfx-arch fingerprint.
- [ ] rocBLAS `gemmEx` fallback for shapes/arches hipBLASLt can't tune (esp.
      gfx1100) → no `KILN_BLAS_ERR_HEURISTIC` crash.
- [ ] Reached via `kiln-tensor/rocm features=["hipblaslt"]` (NOT standalone like
      the dead `kiln-vulkan-blas/vulkan`).

### 6. DeviceOp dispatch
- [ ] Every op-dispatch site that matches on backend routes `Device::Rocm` to
      the ROCm storage/kernel path.
- [ ] `kiln-kt-bridge` device-ptr seam: backend-neutral
      `device_input_ptr`/`device_output_ptr` (or `rocm_*` analogs) under
      `cfg(any(cuda,rocm))`. `kiln-kt-bridge` feature `rocm = ["kiln-tensor/rocm"]`;
      `tape_bridge` cfg includes `rocm`.

### 7. Kernel crates — all 9, forward AND backward
- [ ] **kiln-tensor csrc** (28 ops) — Tier-1 shared-source hipify (§4).
- [ ] **kiln-flce-kernel** — pure-kt composite; rides `kiln-tensor/rocm`.
- [ ] **kiln-opd-loss-kernel** — composite bwd ships immediately; `opd_topk_kl.cu`
      (hipify-clean) fast path. Weak-dep feature shape preserved.
- [ ] **kiln-rmsnorm-kernel** — Tier-1 hipify of fused_rmsnorm (fwd+bwd),
      l2_qk_norm, attn_qkv_prep, rotary_qk, mlp_silu_mul, sigmoid_mul, lora_add,
      causal_conv1d, **optimizer_step (sgd/adamw — training parity)**. Wave-fix.
- [ ] **kiln-conv1d-kernel** — Tier-1 hipify of causal_conv1d_update.cu.
- [ ] **kiln-gdn-kernel** — Tier-1 hipify of recurrent_gdn_fwd + 6 chunk-scan
      files; wave-fix on recurrent reductions.
- [ ] **kiln-marlin-gemm** (W4A16, HARD) — Phase A: int4-dequant→hipBLASLt
      composite; Phase B (ladder): MFMA (gfx9) / WMMA (gfx11) hand-port.
- [ ] **kiln-flash-attn** (HARDEST) — Phase A: kt-composite SDPA (hipBLASLt QK^T
      → kt softmax → hipBLASLt PV + recompute bwd), bf16, hdim 128/256,
      causal+non-causal; Phase B (ladder): CK/ck-tile fmha (or AOTriton).
- [ ] Each kernel crate: in-place `[features] rocm` gating `kiln-tensor`/
      `kiln-kt-bridge` `features=["rocm"]` (not hardwired `["cuda"]`). **CUDA
      build byte-unchanged.** No crate compiled-but-dead under `rocm`.

### 8. Graph capture (`crates/kiln-graph-rocm`, new + `kiln-model/src/rocm_graph.rs`)
- [ ] `kiln-graph-rocm` scaffold (copy graph-cuda's 89-line lib.rs).
- [ ] Real `hipGraph` capture mirroring `cuda_graph.rs`.
- [ ] HIP-graph device-pointer pinning across replay validated (the
      `hipMallocAsync`/`hipFreeAsync` pool-under-capture analog — riskiest area).

### 9. Feature flags (the full parallel cascade)
- [ ] root: `kiln-hip` + `kiln-rocblas` path deps + new crates in `members`,
      excluded from `default-members`.
- [ ] `kiln-server`: `rocm = [kiln-model/rocm, kiln-tensor/rocm, kiln-train/rocm,
      kiln-kt-bridge/rocm]`.
- [ ] `kiln-model`, `kiln-tensor`, `kiln-train`, `kiln-kt-bridge`, `kiln-core`
      (**CUDA shape `[dep:kiln-tensor, kiln-tensor/rocm]`, NOT the vulkan shape**),
      `kiln-flce-kernel`, `kiln-opd-loss-kernel` (weak-dep), `kiln-rocblas`
      rocm/probe/hipblaslt edges wired.
- [ ] `roctx` optional feature (rocTX/roctracer) mirroring `nvtx`, off-by-default.
- [ ] `cargo check --features rocm` green on a host WITHOUT ROCm; `cargo
      build`/`check`/`test` (default-members) **byte-unchanged**; cuda
      candle-free guard still passes.

### 10. Build scripts
- [ ] Shared `find_rocm_root()` (ROCM_PATH/HIP_PATH/`/opt/rocm` + `which hipcc`)
      + `hipcc`-invoke helper (cc::Build::cuda(true) CANNOT drive hipcc).
- [ ] `kiln-tensor/build.rs` `build_rocm()` on `CARGO_FEATURE_ROCM`: compile 28
      csrc + `kt_gpu_compat.cuh` via `hipcc --offload-arch=$KILN_ROCM_ARCHS` →
      `kiln_tensor_rocm_ops`; link `amdhip64`. No-op + `cargo:warning` when
      ROCm absent.
- [ ] `kiln-rocblas`, `kiln-rmsnorm-kernel`, `kiln-conv1d-kernel`,
      `kiln-gdn-kernel`, `kiln-opd-loss-kernel`, `kiln-marlin-gemm`,
      `kiln-flash-attn` build.rs each gain a `CARGO_FEATURE_ROCM` hipcc arm.
- [ ] `KILN_ROCM_ARCHS` (default `gfx90a;gfx942;gfx1100`) → `--offload-arch`;
      `cargo:rerun-if-env-changed` for ROCM_PATH/HIP_PATH/HIPCC/KILN_ROCM_ARCHS.

### 11. Runtime selection (`kiln-server/src/device.rs`)
- [ ] `#[cfg(feature="rocm")] if kiln_tensor::rocm_is_available() { return
      Ok(Device::Rocm(0)) }` arm — **before** Vulkan (AMD prefers ROCm).
- [ ] `feature="rocm"` added to BOTH aggregate cfg lists (device.rs:86, :89).
- [ ] `mark_rocm_active()` mirroring `mark_vulkan_active()`; main.rs banner/probe
      rocm arms (~9 sites).
- [ ] `kiln-core/src/vram.rs`: DRM sysfs already covers 0x1002; optional
      `VramSource::RocmSmi` for amd-smi precision parity.

### 12. Parity tests
- [ ] `#[cfg(feature="rocm")]` arm in `kiln-tensor/tests/new_ops_parity.rs`
      (CPU-vs-ROCm, bf16 rtol ~2e-2 / atol ~1e-2 fwd; f32 rtol 1e-4).
- [ ] `gdn-kernel/tests/{gates_parity,gated_rms_norm_parity}.rs` rocm arms.
- [ ] Every kernel crate's `tests/kt_v2_smoke.rs` runs under rocm on real gfx.
- [ ] Backward parity (finite-diff): rmsnorm_bwd, cross_entropy_bwd,
      layernorm_bwd, opd composite bwd, flash bwd, optimizer_step.
- [ ] Tests gated on `KILN_ROCM_DEVICE`. `parity-tolerance.csv` widened.

### 13. CI / release
This checklist records the original implementation plan. Hosted compile jobs
are now manual compatibility checks, not ROCm qualification, and no self-hosted
runner is required. Runtime evidence is captured locally on Strix Halo under
`qualification/receipts/rocm/`; see `docs/ci-policy.md`.

- [ ] `.github/workflows/ci.yml` `linux-rocm` job (modeled on `linux-vulkan`):
      `cargo check --locked -p kiln-server --features rocm` toolchain-less.
- [ ] (Optional, self-hosted) AMD gfx942/gfx1100 runner running the parity suite.
- [ ] `server-release.yml` `linux-rocm` job (modeled on `linux-cuda`): install
      ROCm, `cargo build --release --features rocm`, package
      `kiln-${VERSION}-x86_64-unknown-linux-gnu-rocm6X.tar.gz`.
- [ ] `desktop/src/installer.rs`: `LINUX_ROCM_TARGET` const, reroute vendor
      0x1002 → ROCm with `KILN_DESKTOP_GPU_BACKEND=vulkan` escape hatch.

### 14. Docs
- [ ] README/build docs: `--features rocm`, `KILN_ROCM_ARCHS`, ROCM_PATH/
      HIP_PATH, supported gfx targets.
- [ ] `kiln-hip` crate doc enumerating the cudarc surface mirrored.
- [ ] Parity status table (op × backend) updated with the rocm column.

---

## Phased roadmap

**Invariant for every phase:** `cargo build`/`check`/`test` on default-members
stays byte-unchanged on non-ROCm hosts, and `cargo check -p kiln-server
--features rocm` passes toolchain-less. Each phase is independently
parity-testable.

| Phase | Goal | Key files | Exit criterion |
|---|---|---|---|
| **R.0** | Enum substrate + match fan-out | `device.rs` (5 arms + `ALL`→`[;5]` + test), ~40 `Device::` sites, 6 `Backend::` sites | `cargo build` green; `backend_all_order` asserts `[cpu,cuda,metal,vulkan,rocm]`; zero ROCm code |
| **R.1** | `kiln-hip` binding crate + workspace pin | new `crates/kiln-hip`, root `[workspace.dependencies]` | `cargo check -p kiln-hip` green toolchain-less; RAII unit test on this gfx box |
| **R.2** | build.rs `build_rocm()` no-op + `kt_gpu_compat.cuh` | `kiln-tensor/build.rs`, `csrc/kt_gpu_compat.cuh` (shim + `KILN_WARP` + portable `warp_reduce_*`) | default build untouched; arm compiles empty TU on this box |
| **R.3** | Leaf feature edges + `RocmStorage`/allocator/stream | `kiln-tensor/Cargo.toml`, `rocm_storage.rs`, `rocm_allocator.rs`, `active_rocm_stream`, `primary_rocm_context`, `kiln-rocblas` skeleton | `cargo check -p kiln-tensor --features rocm` green toolchain-less; alloc/memcpy round-trip on gfx |
| **R.4** | Mid feature edges | kt-bridge/core/flce/opd/model/train/server `rocm` features + neutral device-ptr seam | `cargo check -p kiln-server --features rocm` green toolchain-less |
| **R.5** | Tier-1 tensor kernels (28 csrc) + **WAVE-SIZE FIX** *(parallel)* | wire 28 `.cu` into `build_rocm()`; `KILN_WARP` on all reduction/softmax/norm; `new_ops_parity.rs` rocm arm | every tensor op passes CPU-vs-ROCm parity on gfx942 — **hard correctness gate** |
| **R.6** | hipBLASLt dense GEMM | `kiln-rocblas/csrc/hipblaslt_{matmul,probe}.cu`, `rocm_matmul.rs`, autotune cache | dense bf16/f16/f32 + epilogues parity on gfx942; gfx1100 rocBLAS fallback; small-model fwd matches CUDA |
| **R.7** | Tier-1 kernel crates (rmsnorm/conv1d/gdn) + optimizer_step *(parallel)* | 3 kernel crates' `[features] rocm` + hipcc arms; model/train `rocm` | GDN/conv fwd parity; rmsnorm bwd + adamw/sgd step parity (training converges); CUDA build byte-unchanged |
| **R.8** | flash-attn + marlin **composite fallback** | flash-attn kt-composite SDPA (fwd+bwd); marlin int4-dequant→hipBLASLt | **full model inference + short training run match CUDA — the "first-class parity" milestone**; ship-able |
| **R.9** | Graph capture + runtime selection + CI/release/desktop | `kiln-graph-rocm` + `rocm_graph.rs` hipGraph; server device probe; CI/release/desktop | captured-graph decode == eager; release artifact builds; CI `linux-rocm` green |
| **R.10+** | **PERF LADDER** (profiler-gated, post-parity) | marlin MFMA/WMMA; flash-attn CK/ck-tile (or AOTriton) | each native kernel ≥ composite throughput AND passes the same parity test; independent PRs |

---

## Execution workflow (multi-agent build orchestration)

A phase-gated orchestrator with bounded fan-out inside the embarrassingly-parallel
kernel phases, worktree isolation per kernel, and an **adversarial parity gate**
that loops-until-green before any phase advances. The phase DAG (R.0→R.10) is
strictly serial at the *gate* level; parallelism lives *inside* R.1, R.5, R.7,
R.8, R.10.

```
Orchestrator (owns the phase DAG + the parity oracle)
│
├─ R.0/R.1/R.2/R.3/R.4 → SERIAL single-builder agents (each ends with a cargo check gate)
├─ R.5 → FAN-OUT: 6 kernel-group agents + 1 shim-owner (kt_gpu_compat.cuh)
├─ R.6 → SERIAL single agent (BLAS is one coherent unit)
├─ R.7 → FAN-OUT: 3 agents (rmsnorm | conv1d | gdn)
├─ R.8 → FAN-OUT: 2 agents (flash-attn composite | marlin composite)
├─ R.9 → SERIAL single agent (graph + runtime + CI coupled)
└─ R.10 → FAN-OUT: 2 long-running specialists (MFMA marlin | CK flash-attn)
```

The shim (`kt_gpu_compat.cuh` with `KILN_WARP`) is a **shared dependency** of
every R.5 kernel agent → built FIRST by a dedicated shim-owner and frozen before
fan-out, otherwise N agents race-edit one header.

**Worktree isolation.** Each fan-out agent runs in its own `git worktree`
branched off the phase-base commit. Kernel `.cu` edits are disjoint by file; the
only shared files (`build.rs` file list, per-crate `Cargo.toml` `rocm` feature)
are edited by appending, and the orchestrator serializes the **merge** of those
hunks. Merge order: shim-owner → kernel agents (rebased on shim) → orchestrator
merges build.rs/Cargo.toml hunks → run the phase gate.

**Adversarial parity verification (the gate).** A parity-oracle agent (distinct
from the builder) independently, in a clean worktree:
1. Re-runs `cargo test --features rocm` on the self-hosted gfx942 runner (+
   gfx1100 for fallback) with `KILN_ROCM_DEVICE`.
2. Runs each parity target against the **CPU reference** (canonical oracle — NOT
   CUDA output, since `hipcc -ffast-math` reassociates differently than nvcc
   `--use_fast_math`; bf16 tolerances absorb this).
3. Runs the **finite-difference** oracle for backward ops — never trusts a
   hand-written bwd.
4. **Adversarial probes:** reduction kernels at block widths spanning a 64-lane
   wavefront boundary {33, 63, 64, 65, 96, 128, 1024} to catch wave64
   truncation a power-of-two-only test misses; hipBLASLt at decode-skinny
   (M=1..8) + large shapes; toggles `KILN_ROCM_ARCHS` to confirm gfx1100
   rocBLAS fallback fires.
5. Diffs `cargo tree -p kiln-server --features cuda` before/after to assert the
   **CUDA build is byte-unchanged** and still candle-free.

**Loop-until-green gating.** Builder commits → oracle verifies → on any `fail`
the orchestrator returns failing rows + adversarial-probe diffs to the SAME
builder (preserving its worktree) with `retry:N`; merges only when all-green. A
phase gate opens only when (a) all fan-out tasks merged green, (b) the phase
exit-criterion suite is green on gfx942, and (c) the non-ROCm-host invariant
holds. Peak concurrency ≈ 7 (R.5). Oracle + orchestrator are persistent across
all phases.

---

## Validation

- **CPU reference is the canonical oracle**, not CUDA output (fast-math
  reassociation differs across compilers). CUDA path is a secondary sanity check.
- **Forward:** `new_ops_parity.rs` rocm arm, bf16 rtol 2e-2 / atol 1e-2 (the
  metal/vulkan band), f32 rtol 1e-4.
- **Backward:** finite-difference oracle for every hand-written bwd.
- **Wavefront-boundary sweep** at {33,63,64,65,96,128,1024} on real gfx942 —
  the single highest-value test (wave64 bugs compile cleanly, manifest only
  numerically).
- **bf16 atomics** with/without `-munsafe-fp-atomics`; f32-accumulate fallback.
- **GEMM shape coverage**: decode-skinny (M=1..8), square, large-K; gfx1100
  asserts rocBLAS fallback fires.
- **Graph-capture replay** bit-identical to eager decode.
- **CI:** `linux-rocm` check-only (toolchain-less) always-on; self-hosted
  gfx942/gfx1100/gfx90a parity suite; `server-release.yml` real build; cuda
  candle-free guard unchanged.

---

## Top risks

1. **Wave64 silently corrupts every two-level `__shfl` reduction (HIGHEST).**
   Confirmed: 14 `.cu` files. Mitigation: `KILN_WARP` (32/64) + portable
   `warp_reduce_*` in `kt_gpu_compat.cuh` (R.5, mandatory) + the boundary sweep.
   No kernel merges without it.
2. **flash-attn (CUTLASS/CuTe) + marlin (inline PTX) cannot be hipified.**
   Mitigation: R.8 composite fallback first; native MFMA/CK is R.10. Backend is
   first-class without R.10.
3. **No drop-in cudarc analog; `kiln-hip` is bespoke surface.** Mitigation: ~8
   symbol histogram bounds it; `cuda_stream_priority.rs` is the template; pin
   once for type unification. Riskiest sub-area: `DevicePtr::device_ptr(&stream)`
   guard semantics (38 sites) + `hipMallocAsync` pool-under-capture pinning →
   dedicated graph-replay bit-identity tests.
4. **`hipDeviceptr_t` is `void*`, CUdeviceptr is `u64`.** Mitigation: normalize
   to one wrapper; grep all transmute/bytemuck sites.
5. **`Backend::ALL` `[;4]`→`[;5]` fan-out** ripples into ~40 matches + CSV
   fixtures. Mitigation: R.0 atomic commit; oracle greps `[_;4]`; append Rocm
   LAST.
6. **Editing the 5 hardwired CUDA-only kernel crates can break the CUDA build,
   and the candle-only guard won't catch it.** Mitigation: preserve cuda dep
   edges byte-identical; oracle diffs `cargo tree --features cuda`; `linux-cuda`
   release build is the net.
7. **`cc::Build::cuda(true)` is hardwired to nvcc.** Mitigation: `build_rocm()`
   invokes `hipcc` directly (`--offload-arch`, not `-gencode`).
8. **hipBLASLt arch coverage thin on gfx1100.** Mitigation: rocBLAS `gemmEx`
   fallback; gfx1100 parity asserts fallback fires; gfx-fingerprinted cache.
9. **`kiln-core/vulkan` has a DIFFERENT feature shape** (no kiln-tensor pull) —
   copying it for rocm is a trap. Mitigation: rocm uses the CUDA shape.
10. **`kiln-vulkan-blas/vulkan` is a dead standalone feature.** Mitigation:
    `kiln-rocblas` pulled via `kiln-tensor/rocm features=["hipblaslt"]`, verified
    by an end-to-end GEMM-on-Rocm test.
11. **Toolchain-less `cargo check --features rocm` must pass.** Mitigation:
    `kiln-hip` dynamic-linking + every `build_rocm()` no-ops with `cargo:warning`
    when ROCm absent. Single most important toolchain choice; a phase-gate
    criterion.
12. **Parallel kernel agents race-editing shared files.** Mitigation: worktree
    isolation; shim frozen before R.5 fan-out; orchestrator serializes shared-hunk
    merges; disjoint `.cu` ownership.

---

## Open questions (need a decision)

1. **gfx target set:** ship gfx942 + gfx90a + gfx1100 in default
   `KILN_ROCM_ARCHS`, or CDNA-only (gfx942;gfx90a) for v1 with RDNA3 as
   follow-up? Confirm which physical GPU the self-hosted parity runner has.
2. **Minimum ROCm version:** pin `kiln-hip` bindgen against ROCm 6.x (broadest
   install base) or 7.x (this box)? Affects hipBLASLt/CK APIs assumed.
3. **`kiln-hip` authorship:** hand-roll bindgen+RAII in-repo (panel consensus —
   no mature crate exists) or build on crates.io `hip-sys`? Recommend hand-roll.
4. **R.10 staffing:** is there a GPU-kernel engineer to own MFMA marlin + CK
   flash-attn long-term? If not, ship first-class on the composite fallback
   (R.8); R.10 deferred. Confirm whether deployment is AMD-first at scale.
5. **Desktop reroute:** flip vendor 0x1002 Vulkan→ROCm now, or keep Vulkan
   default until the ROCm artifact is proven across consumer AMD GPUs?
6. **Self-hosted CI:** is an AMD GPU runner available for on-hardware parity, or
   does validation rely on the release-build compile + manual runs on this box?
   The loop-until-green gate assumes a runner.
7. **Native f32-act × bf16-weight GEMM:** match CUDA parity by casting to a
   common dtype only (recommended v1), or also implement `rocm_matmul_bf16w` to
   recover the Vulkan-only VRAM win? hipBLASLt has no native mixed-input path.
8. **roctx/roctracer profiling:** wire the optional `roctx` feature now
   (mirroring `nvtx`, off-by-default) or defer?
