# Candle Removal Plan (#1082)

This document inventories every `candle_core` and `candle_nn` reference in
the kiln workspace and the migration path to a candle-free build. It is
the canonical artifact for tracking Phase 7 closeout against issue #1082.

Last refreshed: **2026-05-26**, against `main` through `aa969ceb` —
post Phase 5 revert (`909e2e61`,
`KILN_CUDA_GRAPHS_BATCHED=0` + `_KV_FUSED=0` remain the CUDA
default) and KILN_DETECT_ANOMALY substrate + regression-test
landing (`3c90d064` + `fdeace4b` + `28514162` + `aa969ceb`).

**Today's batch headline:**
- **Phase 5 production posture corrected** — bs>1 captured-batched
  graphs + fused KV writer were reverted to opt-in in `909e2e61`
  after production bench exposed HTTP 500s. The eager-batched path is
  healthy and remains the default (`498 tok/s @ bs=64` on A6000,
  archived in `a215efd2`). Captured bs>1 shape-capture is fixed
  (`68aa19c8` + `c78c4f90`), but replay still hits
  `CUDA_ERROR_ILLEGAL_ADDRESS`; see `bench-results/cuda-graph-status.md`.
- 🎉 **Tier 1 closed** for `kiln-conv1d-kernel` (`577f8b0c`),
  `kiln-marlin-gemm` (`4a862711`), and now `kiln-flash-attn` —
  `candle-core` dropped from all three `[dependencies]` blocks.
- 🎉 **KILN_DETECT_ANOMALY** Phase 9 trap wired end-to-end:
  scaffold (`72c2c16f`) + `Tensor::all_finite()` substrate
  primitive (`3c90d064`) + `Tape::backward` integration
  (`fdeace4b`) + CUDA D2H bridge (`28514162`) + direct tape
  regression test (`aa969ceb`). Set `KILN_DETECT_ANOMALY=1`
  and the autograd tape panics at the producing op's tape
  position on the first NaN/Inf gradient.
- Substrate accessors added: `CudaStorage::cuda_stream_raw()`
  (`d561dbf8`), `Tensor::cuda_from_slice` + `cuda_zeros_on` +
  `primary_cuda_device` (`a5da6152`),
  `flash_attn_paged_decode_dyn_seqlen_kt_with_graph_outputs`
  (`aab07fa7`).
- Cuda.rs candle fallback branches removed (kt-only): conv1d
  (`2ebcfb08`), GDN 10 sites (`86c7f134`), flash-attn 4 sites
  including `_with_graph_outputs` (`9ac211e9` + `aab07fa7`),
  GDN single-block (`29321870`).
- Forward.rs cleanups: 5 default-on rmsnorm-family flag helpers
  deleted (`58607c30`), flash-attn captured-graph site migrated
  to kt (`fe5418a4`).
- GDN auto-cast for non-BF16 a_log/dt_bias + heavy q/k/v/a/b/z
  tensors (`010b5e40` + `66e5cb29`), GDN supports_kt 4D vs
  kernel 3D wire fix (`a2cb9edb`).

**Today's net diff:** ~37 commits, large net deletions across
kiln-model (~-3000 LOC of candle fallback branches and
intra-process parity tests).

## Workspace candle footprint

```
.rs files importing candle, by crate (descending):

41  kiln-kt-bridge
27  kiln-model
20  kiln-vulkan-kernel  (NB: actually 8 as of 2026-05-26 HEAD eea56b23; table stale)
 9  kiln-train
 6  kiln-server
 6  kiln-gdn-kernel
 5  kiln-tensor
 4  kiln-rmsnorm-kernel
 4  kiln-opd-loss-kernel
 4  kiln-marlin-gemm
 3  kiln-flce-kernel
 3  kiln-flash-attn
 3  kiln-conv1d-kernel
 2  kiln-blas

15  total crates with `candle*` in `[dependencies]` or `[dev-dependencies]`
```

## Removal sequencing

Removal proceeds bottom-up: leaf crates first (they have the smallest blast
radius), then the high-traffic kernel crates, then `kiln-model`, then the
bridges, finally the vendor delete.

### Tier 0 — already candle-free (kt-substrate primary)

- `kiln-core`        — never depended on candle directly.
- `kiln-scheduler`   — never depended on candle directly.
- `kiln-nvtx`        — never depended on candle directly.

### Tier 1 — leaf crates with minimal candle surface

| Crate | candle deps | Migration path |
|---|---|---|
| `kiln-blas`             | `dep:candle-core` (optional, `cublaslt`/`probe` features) | Drop once the cublasLt probe binary is moved to a standalone debug crate or rewritten against `cudarc` directly. |
| `kiln-conv1d-kernel`    | `candle-core` (cuda feature) | Drop the candle-typed surface once all callers use `kt_api::*_kt`. |
| `kiln-marlin-gemm`      | `candle-core` (cuda feature) | Same as conv1d. |
| `kiln-flash-attn`       | `candle-core` (cuda feature) | Same; 5/5 kt_api smoke tests already green. |
| `kiln-rmsnorm-kernel`   | `candle-core` (cuda feature) | Same; 25/25 kt_api smoke tests already green. |
| `kiln-gdn-kernel`       | `candle-core` (cuda feature) + `candle-nn` | Same; 20/20 kt_api smoke tests already green. |

For each Tier-1 crate, removing candle means:

1. Migrate every production caller in `kiln-model::forward` and
   `kiln-train` from the candle-typed entry to the `kt_api::*_kt` entry
   (zero-copy borrow path).
2. Delete the candle-typed `lib.rs` surface.
3. Delete the `candle-core` line from the crate `Cargo.toml`.
4. Verify `cargo check --release -p kiln-<crate> --features cuda`.

### Tier 2 — opd-loss + flce + mps + vulkan

| Crate | Notes |
|---|---|
| `kiln-opd-loss-kernel`  | **Blocked by 3 live candle-typed callers in `kiln-train`** (investigated 2026-05-26 against HEAD `8e21c8fe`): `src/opd.rs:68` imports `opd_top_k_reverse_kl_phase_a_per_position` + `DEFAULT_CHUNK_SIZE` for the production forward path; `src/opd.rs:3034` imports `opd_top_k_reverse_kl_phase_b` (cfg-test); `tests/vk_cuda_opd_parity.rs:20` imports `opd_top_k_reverse_kl_phase_b_per_position`. kt-typed entry points already exist (`opd_top_k_reverse_kl_kt`, `opd_top_k_reverse_kl_per_position_kt`, `compute_per_position_metrics_kt`) — production path migration is the unblocker; same shape as the rmsnorm/gdn blockers. Crate `Cargo.toml` still depends on `candle-core` + `candle-nn` (dev). |
| `kiln-flce-kernel`      | **Largest single rewrite.** Phase A is pure-candle, Phase B is the raw-CUDA replacement. Driving Phase B to closeout is its own multi-PR effort (see crate-level `phase_b.rs`). |
| `kiln-mps`              | Apple Silicon only; Metal storage substrate lives here. |
| `kiln-vulkan-kernel`    | **Investigated 2026-05-26 against HEAD `eea56b23`** — 8 .rs files importing candle, 13 import lines total (61 .rs files in the crate). Footprint is now ~60% smaller than the earlier "20 files" estimate after 14+ #1082 cleanup commits over the past 3 weeks (latest: `e047c579` de-candle `vk_flce_parity` test factories). Crate `vk_ops/` (31 files) is **fully candle-free**. Top 3 remaining blockers are all rooted in the **legacy candle-typed `kernels::dispatch_*` surface** — see "kiln-vulkan-kernel blocker breakdown" below. |

#### kiln-vulkan-kernel blocker breakdown (2026-05-26, HEAD `eea56b23`)

The 13 remaining candle import lines split into 3 distinct blocker families.
**Crucially, none of them require kt-autograd** (the bottleneck for rmsnorm /
gdn / opd-loss closures). The blocker here is just the legacy
candle-typed dispatch API surface.

**Family 1 — `kernels.rs` legacy dispatch surface (1 import line, ~10K LOC of
fns):** `src/kernels.rs:5` imports `{DType, Device, Tensor}`. ~30 of the 49
`pub fn dispatch_*` functions return `Tensor` or `(Tensor, …)` and take `&Tensor`
inputs (e.g. `dispatch_mlp_gate_up_decode_cached:4553`,
`dispatch_qwen_rmsnorm_forward:2003`,
`dispatch_gdn_in_proj_decode_cached_bf16_weights:1176`). Six public helper fns
also have candle in their signature (`extract_tensor_bytes:908`,
`extract_tensor_packed_bf16_bytes_pub:926`, `create_tensor_from_data:951`,
`buffer_to_tensor:977`, `upload_tensor_f32_buffer:1099`,
`upload_tensor_bf16_packed_buffer:1126`). These are called from
**off-limits crates** (`kiln-model/src/{forward.rs, backend/vulkan.rs,
backend/vulkan_linear_op.rs, backend/vulkan_lora_op.rs, vk_decode_resident.rs}`
and `kiln-train/src/vk_train.rs`) — ~30 call sites total across the workspace.
The legacy dispatch surface cannot move until those production callers
migrate, which is a kiln-model concern outside this crate.

**Family 2 — `vk_tensor.rs` candle bridge (5 import lines):** `src/vk_tensor.rs:16`
imports `{CpuStorage, DType, Device, Storage, Tensor}`. Used by
`VkTensor::from_candle` / `to_candle` / `fresh_param_id` (`TensorId` minting
via candle 1-element `Tensor::zeros`). The doc-comment on
`fresh_param_id:340` explicitly flags this as the contained candle dep — only
this one fn body needs to swap once a kt-native `TensorId` lands. External
callers (`kiln-model/src/{vk_forward.rs, forward.rs, backend/vulkan.rs}`,
`kiln-train/src/{trainer.rs, vk_train.rs, echo.rs}`, and 2 train tests)
all use `from_candle` / `to_candle` to bridge their candle Tensor inputs
into the vk-native autograd tape and read results back out. Blocked on
**kt-native `TensorId` substitute**, *not* kt-autograd (vk_autograd already
runs in pure VkTensor space).

**Family 3 — parity tests + examples calling the legacy dispatch surface (7
import lines across 5 files):** `tests/gdn_parity.rs:2` (106 legacy
`dispatch_*` call sites in this test alone), `tests/vk_matmul_parity.rs:292`
(one scoped `Var`-based analytical-gradient oracle inside `mod tests`),
`tests/vk_flce_parity.rs:203` (same pattern — scoped candle autograd oracle),
`examples/decode_microbench.rs:15` (microbench feeds `dispatch_*_cached*`),
`examples/vk_mlp_probe.rs:45` (only uses candle to build `make_x() -> Tensor`
because `dispatch_mlp_gate_up_decode_cached` takes `&Tensor`), and
`src/resident.rs:1832` (the `#[cfg(test)] mod tests` block — 22 parity tests
that compare `_resident` raw-buffer dispatchers against the legacy candle
baselines). All of Family 3 either consumes the legacy dispatch surface
(blocked on Family 1) or uses `Var`-based candle autograd as a deliberate
oracle (acceptable scoped use).

**Recommended next concrete steps (in priority order):**

1. **Add `_bytes` siblings for the most-used legacy dispatchers.** The
   `dispatch_kernel_bytes:436` precedent (landed in `60c48916`) and the
   `upload_*_buffer_from_slice` helpers (`6f1cabdc`) already show the
   pattern: factor the dispatch body to take `&[u8]` + shape instead of
   `&Tensor`, keep the candle-typed wrapper for backward compat. Start
   with `dispatch_mlp_gate_up_decode_cached_bytes` — the single change
   would let `examples/vk_mlp_probe.rs` drop candle entirely (file is
   already 95% candle-free; only `make_x() -> Tensor` and one
   `Tensor::from_vec` call need to swap). This is the smallest
   self-contained #1082 commit available in the crate right now and
   establishes the pattern for the rest of Family 1.

2. **Mint a kt-native `TensorId` substitute.** The `vk_tensor.rs`
   blocker is gated on one new type — a 64-bit monotonic id minted by
   `kiln-tensor` (or `kiln-core`). Once that exists, change `pub use
   candle_core::TensorId` to point at the new type, swap
   `fresh_param_id`'s body from `Tensor::zeros(…).id()` to the new
   minter, and `vk_tensor.rs`'s candle surface goes from 5 import
   lines to 0. All external `VkTensor::from_candle` / `to_candle`
   callers stay on candle (they live in off-limits crates and bridge
   candle-typed inputs/outputs — they're naturally part of the
   kiln-model / kiln-train migrations later). **This unblocker is
   independent of kt-autograd.**

3. **Defer Family 3 entirely.** Parity tests + microbenches that
   consume the legacy `dispatch_*` API are correctly blocked on Family
   1; the scoped `Var` oracles in `vk_matmul_parity.rs` and
   `vk_flce_parity.rs` are acceptable as long as candle remains a
   workspace dep, and become trivial to remove once kt has an
   analytical-grad oracle. No standalone action needed here.

**What's NOT a blocker:** kt-autograd (CustomOp1/2/3 → BackwardOp). The
vk-native autograd tape (`vk_autograd.rs` + `VkBackwardOp` trait) already
runs in pure VkTensor space — it imports `TensorId` only to key the
gradient store, and that import flows through `vk_tensor.rs`'s
`pub use candle_core::TensorId` (Family 2). Once the kt-native `TensorId`
substitute lands (step 2), `vk_autograd.rs` is candle-free with no other
changes.

### Tier 3 — `kiln-model` (the forward pass)

27 .rs files. The candle surface here is the most diffuse and the most
performance-sensitive. Migration proceeds op-family by op-family, with
each migration env-gated by a `KILN_USE_KT_API_*` flag so the gate can be
flipped to default-on after a clean parity-test cycle.

State today (post-#f569b7be):

- 40+ env-gated kt-API parallel paths landed; each defaults off.
- The `KILN_USE_KT_API_ALL=1` master switch flips all gates on at once.
- `PagedKvCache` ↔ `PagedKvCacheKt` migration is partial — constructor +
  writer wired, first end-to-end production site flipped via bench
  (`d5bba062`), ~10 decode call sites remain on the candle path.

Remaining work to drop the candle dep from `kiln-model`:

1. Finish the PagedKvCacheKt call-site migration.
2. Promote every `try_kt_*` helper from opt-in to default-on by removing
   its env gate (after a parity cycle).
3. Delete the candle fallback branches in the `if let Some(out) =
   try_kt_*` patterns.

### Tier 4 — `kiln-kt-bridge` (the bridge itself)

By design, `kiln-kt-bridge` IS the candle↔kt boundary. Its `candle-core`
dep is removed last: once every production path is on kt-typed tensors,
the bridge collapses to a thin device-id translation layer (or is
deleted entirely).

### Tier 5 — vendor delete

After every `crates/*/Cargo.toml` has no candle entry:

```bash
# Verify no transitive candle dep anywhere
cargo tree --workspace -i candle-core   # should be empty

# Delete the vendor tree
rm -rf vendor/candle-core/

# Remove the workspace patch
sed -i "/candle-core.*path.*vendor/d" Cargo.toml
```

## Phase 5 (CUDA graph capture) — bs>1 captured path opt-in (2026-05-26)

`KILN_CUDA_GRAPHS_BATCHED=1` and `KILN_CUDA_GRAPHS_BATCHED_KV_FUSED=1`
briefly flipped from default-off to default-on in commit `6d564b9a`,
then reverted in `909e2e61` after production validation exposed
bs>=2 HTTP 500s. The current production default is the eager-batched
decode path, which is healthy (`498 tok/s @ bs=64` on A6000, archived
in `a215efd2`).

Validation gate that gated the flip (passed on A6000 at HEAD
`a2cb9edb`):
- Live `kiln-bench --paged` Qwen3.5-4B W4A16 + paged + batched +
  KV-fused: exit 0, decode 74.4 tok/s, mean ITL 13.45 ms, peak VRAM
  11.2 GB at batch=1.
- `compute-sanitizer memcheck` under the same configuration:
  `========= ERROR SUMMARY: 0 errors` across batches 1/4/8/16.

Bug chain from the attempted default-on flip:
1. **sccache CUDA dlink stale cache** dropped `gdn_gates` device SASS
   from the binary → `cudaLaunchKernel → cudaErrorSymbolNotFound`.
   Workaround for every fresh RunPod build: `cargo clean -p
   kiln-gdn-kernel && SCCACHE_RECACHE=1`. A permanent build.rs fix
   (hash the .o into the dlink cache key) is tracked separately.
2. **GDN BF16-envelope mismatch** — kt path required all-BF16 but
   `kiln-conv1d-kernel` emits F32 q/k/v by design + safetensors
   loader keeps `a_log` / `dt_bias` "as-is" (often F32). Fix:
   auto-cast at dispatch (`010b5e40` + `66e5cb29`).
3. **supports_kt shape contract** — kt predicate expects 4D
   `[B, 1, q_heads, dk]` but kt kernel expects 3D
   `[B, q_heads, dk]`. cuda.rs was passing 3D to both → predicate
   always declined. Fix: predicate on 4D, squeeze for kernel
   (`a2cb9edb`).
4. **Batched graph key/validator formula mismatch** — capture key
   used the 128-token K/V chunk bucket, while the validator compared
   against the actual current decode length. Fixed in `68aa19c8` and
   refined in `c78c4f90` so stable buffer shapes stay bucketed while
   kernel read bounds use actual `max_seqlen_k`.
5. **Next-layer replay failure** — captured bs>=2 replay now reaches
   post-launch sampling but hits `CUDA_ERROR_ILLEGAL_ADDRESS`, likely
   from lm-head matmul output allocation inside the capture window.
   This remains unfixed and documented in
   `bench-results/cuda-graph-status.md`.

See `bench-results/cuda-graph-status.md` "Phase 5 sanitizer
sweep — 2026-05-26" for the full validation trail.

Opt-in flags: `KILN_CUDA_GRAPHS_BATCHED=1` and
`KILN_CUDA_GRAPHS_BATCHED_KV_FUSED=1`. Keep them off in production
until the lm-head captured-output lifetime fix lands and passes a
fresh RunPod validation cycle.

## KILN_DETECT_ANOMALY (Phase 9 trap) — wired end-to-end (2026-05-26)

The `KILN_DETECT_ANOMALY=1` NaN/Inf trap is now live across the
autograd tape:
- Scaffold (`72c2c16f`): `anomaly_detection_enabled()` +
  `anomaly_panic()` in `kiln-autograd::anomaly`.
- Substrate (`3c90d064`): `Tensor::all_finite()` walks the strided
  view (CPU storage). Handles F32, BF16, F16, FP8E4M3 (no-Inf
  format), FP8E5M2.
- Tape wire (`fdeace4b`): `Tape::backward` reads
  `anomaly_detection_enabled()` once at top, then after each
  `op.apply()` scans returned grads. On first non-finite,
  `anomaly_panic` with the producing op's tape position.
- CUDA bridge (`28514162`): `Tensor::all_finite()` initially used a
  `cuda_to_host_copy` D2H bridge for CUDA-resident tensors so
  the trap worked end-to-end for GPU training paths,
  paying an O(numel) D2H copy per scanned tensor. Covered by
  6 new CPU unit tests in `kiln-tensor::tensor::tests` (NaN,
  +Inf, -Inf, integer vacuous-true, post-transpose stride walk).
- CUDA kernel (`is_finite_reduce.cu` + `cuda_is_finite`): replaces
  the D2H bridge with a per-backend atomicOr reduction on the
  device. Kernel walks `n_elements` and sets a single 4-byte u32
  flag; `Tensor::all_finite()` reads back only those 4 bytes per
  call instead of the full tensor. Supported dtypes: F32, BF16,
  F16, F8E4M3, F8E5M2. Non-contiguous inputs are contiguified via
  `cuda_contiguous` first (matching the kt-CUDA reduction
  convention). Net D2H per scanned tensor: O(numel) → 4 bytes.
- Tape regression (`aa969ceb`): `Tape::backward` now has a direct
  unit test with a fake backward op returning NaN, asserting the
  panic includes the anomaly prefix, tape position, op name, and
  gradient detail. A companion test proves non-finite grads still
  propagate when the env flag is disabled.

Cost: O(numel) per backward step on CPU when enabled, ~5% per
the issue body. CUDA paths now pay only 4 bytes of D2H per scanned
tensor (down from O(numel) under the bridge). Off-by-default in
production; CI training-parity tests opt in via `KILN_DETECT_ANOMALY=1`.

Remaining: per-backend `is_finite_storage` reduction kernels for
Metal and Vulkan (CUDA done). Then enable the trap in the SFT
parity CI.

## Build matrix coverage required before each tier closes

| Tier | Required CI green |
|---|---|
| Tier 1 crate close | `linux-default` + `linux-cuda` (the kernel crates own gate) |
| Tier 2 crate close | Same + `macos-metal` for `kiln-mps`, `linux-vulkan` for `kiln-vulkan-kernel` |
| Tier 3 close       | Full matrix + opd/sft regression nightlies green |
| Tier 4 close       | Full matrix |
| Tier 5 delete      | Full matrix + a fresh `cargo tree -i candle-core` returning empty |

## Status snapshot (2026-05-26, refreshed)

- Tier 0: ✅ already candle-free (kiln-core, kiln-scheduler, kiln-nvtx).
- Tier 1: 🟢 **conv1d-kernel + marlin-gemm closed (`577f8b0c` + `4a862711`)**;
  3 kernel crates still in flight. The substrate accessor
  `Tensor::cuda_from_slice` (`a5da6152`) is the template that
  unblocks the rest — each crate just needs a candle-typed surface
  delete + a candle-free test rewrite.
  **Production wiring breakthrough** (#695587df): first kt_api production wire
  for a non-rmsnorm kernel crate. kiln-conv1d-kernel's `causal_conv1d_update_kt`
  + `causal_conv1d_prefill_kt` are now both wired in `CudaBackend` at
  `crates/kiln-model/src/backend/cuda.rs:1004` and `:1038`.
  Status by crate:
    - **kiln-conv1d-kernel: ✅ CLOSED** (`577f8b0c`, 2026-05-26) —
      candle-typed lib.rs surface deleted, in-lib parity scaffolds
      removed, `tests/kt_v2_smoke.rs` rewritten candle-free via
      `Tensor::cuda_from_slice`, `candle-core` dropped from
      Cargo.toml. `cargo tree -i candle-core` is empty at the
      crate's direct-dep level.
    - **kiln-marlin-gemm: ✅ CLOSED** (`4a862711`, 2026-05-26) —
      same template: candle-typed `marlin_w4a16_gemm` deleted,
      `tests/parity.rs` removed (replaced by candle-free kt smoke),
      `candle-core` dropped from Cargo.toml.
    - **kiln-gdn-kernel: 🟡 cuda.rs fully kt-only** — all 10
      dispatch sites + the single-block `gdn_full_chunk_forward`
      path migrated (`86c7f134`, `29321870`). Auto-cast added for
      non-BF16 a_log/dt_bias/q/k/v/a/b/z (`010b5e40`, `66e5cb29`)
      because the upstream conv1d kernel emits F32 q/k/v + the
      safetensors loader keeps a_log/dt_bias "as-is" (often F32).
      supports_kt 4D vs kernel 3D shape contract split fix
      (`a2cb9edb`). Remaining: 6 forward.rs prefill-chunk-loop
      sites (tests), 2 cuda_graph.rs sites (`with_decode_gates_
      recurrent_outputs` — see "Scope clarification" below: the
      helper is effectively no-op-in-production today, so a naive
      kt sibling closes the type wire but ships no behavior change;
      the real fix is deeper). `kt_api.rs` already
      candle-import-free.
    - **kiln-flash-attn: 🟡 cuda.rs fully kt-only** — all 4 sites
      migrated, including `_with_graph_outputs` via the new
      `flash_attn_paged_decode_dyn_seqlen_kt_with_graph_outputs`
      substrate (`aab07fa7`). Forward.rs captured-graph site also
      kt (`fe5418a4`). Remaining: 3 candle-typed CustomOp2 shim
      sites in forward.rs (`impl CustomOp2 for
      CudaFlashAttentionTrainingBf16` — autograd integration);
      `paged_kv_cache.rs` 3 sites (`paged_kv_write_*` —
      PagedKvCacheKt migration); 2 env-gated cuda_train.rs sites
      (default-off pending parity testing). `kt_api.rs` already
      candle-import-free.
    - **kiln-rmsnorm-kernel: 🟡 5 default-on flag helpers removed
      from forward.rs** (`58607c30`) — rmsnorm / rotary_qk×2 /
      sigmoid_mul / mlp_silu_mul×2 / l2_qk_norm now kt-only.
      Remaining: ~32 candle-typed callers across forward.rs
      (CustomOp1 internals + lora_add_*_storage + supports_*
      predicates + rotary_one_bf16_storage); cuda.rs (sgd_step,
      adamw_step — env-gated opt-in); cuda_train.rs (25 sites in
      training-only paths). `kt_api.rs` already
      candle-import-free.
    - kiln-opd-loss-kernel: separate phase (full rewrite for Phase B)
- Tier 2: 🟡 in flight — flce-kernel Phase B is the standout rewrite effort.
  Metal + Vulkan backends now have real kernel wires landed for 9 ops
  (rmsnorm, layernorm, l2norm, silu, sigmoid, softmax, elementwise binary
  add/sub/mul/div, cast f32↔bf16, index_select_dim0).
- Tier 3: 🟡 in flight — kiln-model has 40+ env-gated kt-API parallel paths
  in forward.rs + cuda_train.rs. PagedKvCacheKt accessor migration now
  has 5 parity helpers (`try_kt_paged_kv_{block_size, is_fp8,
  pool_tensors_present, num_layers, num_blocks}`) + 9 production accessor
  read sites migrated. All defaults remain off.
- Tier 4: ⏳ blocked on Tier 1 + Tier 3.
- Tier 5: ⏳ blocked on everything above.

This document is updated as tiers close — search-replace the 🟡/⏳ marks
as each tier reaches the relevant `Required CI green` gate.

## Tier 1 — per-crate candle-typed caller audit (2026-05-25)

Audit goal: for each Tier 1 kernel crate, determine whether the candle-typed
public surface can be deleted yet (and therefore whether `candle-core` can be
dropped from the crate's `Cargo.toml`). Methodology: count remaining
`kiln_<crate>::<symbol>` references in `crates/kiln-model/src/**` and other
workspace crates, excluding `_kt`/`kt_api` references (those go through the
kt-typed surface and do not require the candle dep on the kernel crate).

Run against `main` post-`fc5b6b7f`. Counts produced via:
`grep -rn 'kiln_<crate>::' crates/ --include='*.rs' | grep -v '_kt\|kt_api'`.

### kiln-conv1d-kernel — **0 production callers** (substrate-blocked on dep drop)

After `453ed5d3` (kt-typed `supports{,_update,_prefill}_kt` predicates
landed) and `2ebcfb08` (`CudaBackend::causal_conv1d_{update,prefill}`
migrated to use the kt-typed surface as the only path; the dedicated
`cuda_use_kt_api_conv1d` flag + `KILN_DISABLE_KT_API_CONV1D` env gate
removed; intra-process A/B parity tests in forward.rs deleted),
production no longer touches the candle-typed
`kiln_conv1d_kernel::supports*` / `causal_conv1d_*` surface anywhere.

Remaining candle-typed callers are all `#[cfg(test)]` inside the crate
itself (`crates/kiln-conv1d-kernel/src/lib.rs:587/646/712/725`).
These are parity scaffolds that build a candle reference and compare
to the fused kernel.

The crate can drop `candle-core` from `[dependencies]` once the
following structural blockers clear (none of these are specific to
`kiln-conv1d-kernel` — they affect every CUDA-side kt_api):

1. **`kt_api.rs` itself still uses `candle_core::cuda_backend::cudarc`
   for `DevicePtr` + `candle_device().cuda_stream()` to plumb the CUDA
   stream into the FFI.** This is the `kiln_tensor::CudaStorage` →
   candle round-trip that every kernel crate's kt_api shares. Needs
   a candle-free `kiln_tensor::CudaStorage::cuda_stream() -> *mut
   c_void` accessor before `candle-core` can leave the runtime deps.
2. **In-lib parity tests use `candle_core::Tensor::from_vec` + the
   candle-typed reference implementation.** Migrating to kt requires
   either a candle-free `kiln_tensor::Tensor::host_to_cuda_copy` (the
   existing one takes `Arc<CudaDevice>` from candle), or moving the
   reference computation to a candle-free CPU implementation.

When #1 lands across the substrate, this crate's `Cargo.toml` drop is
a one-line PR. Until then the `pub fn supports*` / `pub fn causal_conv1d_*`
candle-typed surface stays as test-internal scaffolding only — no
production caller remains.

### kiln-marlin-gemm — **0 production candle-typed callers** (substrate-blocked on dep drop)

After `0841c266` (`marlin_proj::matmul_bf16` migrated to use
`matmul_bf16_2d_kt` as the only path; `cuda_use_kt_api_marlin()`
gate + `KILN_DISABLE_KT_API_MARLIN` env removed; the
`test_marlin_w4a16_gemm_kt_api_parity` scaffold in forward.rs
deleted), production no longer calls candle-typed
`kiln_marlin_gemm::marlin_w4a16_gemm`.

Audit clarification: `kiln_marlin_gemm::pack::quantize_and_pack`
(`marlin_proj.rs:131`) is **not** candle-typed — its signature is
`(weight: &[f32], k: usize, n: usize, groupsize: i64) -> (Vec<i32>,
Vec<f16>, Vec<f32>)`. Pure host types, no candle dep. The audit
grep matched on crate path; the call site does not impose a
candle dep on `kiln-marlin-gemm`.

Remaining candle-typed callers are all `#[cfg(test)]`:
- `crates/kiln-marlin-gemm/tests/parity.rs:20` — crate-local parity
- `crates/kiln-model/tests/marlin_qproj_parity.rs:82` — pack test

Same substrate-level blockers as `kiln-conv1d-kernel` gate the
`Cargo.toml` `candle-core` drop:
1. `kt_api.rs` uses `kiln_kt_bridge::cuda_storage_and_byte_offset`
   which still goes through the candle-typed CudaStorage. (Resolved
   by `338b1b88` for stream extraction; same accessor pattern needed
   for the `Arc<CudaDevice>` itself before pure-kt allocator routes
   land.)
2. In-lib `#[cfg(test)]` parity scaffolds use candle for input
   construction.

Once the substrate-side blockers clear, the `Cargo.toml` drop is a
one-line PR.

### kiln-rmsnorm-kernel — **BLOCKED** (53 candle-typed callers in kiln-model)

By far the largest residual surface. Candle-typed entry points still
reachable in production:

**`crates/kiln-model/src/backend/cuda.rs`** (6 sites):
- `supports_optimizer_step` × 2, `sgd_step_inplace`, `adamw_step_inplace`,
  `supports_lora_decode_add`, `lora_decode_add`

**`crates/kiln-model/src/cuda_train.rs`** (15 sites):
- `sgd_step_inplace`, `adamw_step_inplace`, `matmul_f32_bf16w` × 3,
  `lora_add_inplace_f32`, `silu_inplace_save_sigmoid_f32`,
  `causal_depthwise_conv1d_f32` (forward), `causal_depthwise_conv1d_f32_inplace`,
  `matmul_f32_bf16w_bwd_lhs` × 2, `causal_depthwise_conv1d_f32_bwd_input` × 2,
  `causal_depthwise_conv1d_f32_bwd_weight`, `causal_depthwise_conv1d_f32_bwd_state`

**`crates/kiln-model/src/forward.rs`** (28+ sites):
- `lora_add_inplace_f32_storage`, `lora_add_bf16_storage` × 2,
  `supports_sigmoid_mul` × 2, `fused_sigmoid_mul`, `fused_sigmoid_mul_storage`,
  `supports` (rmsnorm), `fused_rmsnorm`, `fused_rmsnorm_with_autograd`,
  `supports_rotary_qk` × 2, `fused_rotary_qk` × 2, plus
  `supports_attn_decode_qkv_prep`, `fused_attn_decode_qkv_prep`,
  `supports_mlp_silu_mul[_packed]`, `fused_mlp_silu_mul[_packed]`,
  `supports_l2_qk_norm[_gqa]`, `fused_l2_qk_norm[_gqa]`,
  `rotary_one_bf16_storage`, `rotary_one_bwd_bf16*`.

Plus `crates/kiln-rmsnorm-kernel/examples/phase10_microbench.rs` exercising
the candle path directly.

This crate is the rmsnorm + miscellaneous-elementwise + lora + matmul +
depthwise-conv1d catch-all. Multiple op families (optimizer steps,
training matmuls, lora delta inserts, attn decode prep, depthwise conv1d
forward+backward) still have **no kt-API counterparts** in `kt_api.rs` —
the kt-API surface today is partial. Dropping candle from this crate
requires landing kt_api scaffolding for every op family first, then
migrating each call site, which is multi-PR work.

Follow-up steps (not done in this PR):
1. Expand `kt_api` to cover: optimizer steps, training matmuls
   (`matmul_f32_bf16w` + bwd), lora add families, depthwise conv1d
   forward+backward, attn decode prep, mlp silu mul.
2. Migrate every cuda_train.rs caller (the training hot path) — these
   are deep in autograd machinery and may need bridge extensions.
3. Migrate the forward.rs candle sites (many of which take
   `&Tensor` storage views — needs careful kt-storage parity).
4. Delete candle surface from `lib.rs`.

**Progress 2026-05-26** — kt-typed predicate substrate complete
(commits `8eb37f5d`, `e092671c`, `54ad5ab6`). Every candle-typed
`supports_*` predicate in `kiln-rmsnorm-kernel/src/lib.rs` that has
a live caller in kiln-model now has a kt-typed twin in
`kiln-rmsnorm-kernel/src/kt_api.rs`:

  - `supports_mlp_silu_mul_kt`
  - `supports_mlp_silu_mul_packed_kt`
  - `supports_sigmoid_mul_kt`
  - `supports_rotary_qk_kt`
  - `supports_attn_decode_qkv_prep_kt`
  - `supports_l2_qk_norm_kt`
  - `supports_l2_qk_norm_gqa_kt`
  - `supports_lora_decode_add_kt`
  - `supports_optimizer_step_kt`

Each twin mirrors the candle predicate's shape/dtype/contig/device
invariants over `KtTensor`. Substrate-only — no caller swaps in
those commits. Unblocks step 3 above: forward.rs + cuda.rs sites
can swap the candle-typed `supports_*(...)` gate to its kt twin
once the operands are bridged once, removing the candle dependency
at the predicate layer.

### kiln-gdn-kernel — **1 production caller** (single-block fall-through remains)

After `86c7f134` removed all 10 candle fallback branches + the
`cuda_use_kt_api_gdn` flag + 9 redundant kt-vs-candle parity tests
in forward.rs (483 insertions / 2,155 deletions = -1672 net LOC),
the GDN dispatch surface in `backend/cuda.rs` is kt-only with one
documented exception:

- `gdn_full_chunk_forward` single-block fall-through at
  `backend/cuda.rs:~1043` — preserved because no kt-typed single-
  block kernel exists yet; the multi-block path is fully kt-only.

Remaining candle-typed callers across kiln-model:
- `crates/kiln-model/src/forward.rs` (6 sites): direct calls to
  `gdn_chunk_prep`, `gdn_chunk_scan`, `gdn_full_chunk_forward`,
  `gdn_full_chunk_forward_multiblock` in prefill chunk loops
  outside the backend dispatcher path.
- `crates/kiln-model/src/cuda_graph.rs` (2 sites):
  `with_decode_gates_recurrent_outputs` — graph-output wrapper that
  takes caller-owned outputs (`Vec<Tensor>`) for CUDA-graph
  capture; no kt-typed sibling exists yet.

#### Scope clarification: `with_decode_gates_recurrent_outputs`

Audited 2026-05-26 (worktree `agent-a3fe6452f68908706`). The two
cuda_graph.rs callers (lines `~1385` and `~1574`) wrap the captured
forward in `kiln_gdn_kernel::with_decode_gates_recurrent_outputs`,
which sets up the `DECODE_GATES_RECURRENT_OUTPUTS` thread-local so
the candle-typed `gdn_decode_gates_recurrent[_qk_norm[_rmsnorm]]`
kernels can pick pre-allocated outputs via
`next_decode_gates_recurrent_output` instead of `Tensor::zeros`-
ing inside the capture window.

That thread-local has **no production reader anymore**. The
production decode dispatcher in `backend/cuda.rs:1011`,
`:1197`, `:1378` already routes through the kt-typed kernels
(`gdn_decode_gates_recurrent_bf16_kt` and friends in
`kiln-gdn-kernel/src/kt_api.rs`). Each kt kernel allocates its
own output via `kiln_kt_bridge::alloc_cuda_tensor` and the result
is copied back to candle via `kt_tensor_to_candle_cuda_copy`. The
candle-typed `gdn_decode_*` functions in `lib.rs` (the only
readers of `DECODE_GATES_RECURRENT_OUTPUTS`) are now exercised
only from in-lib parity tests at `lib.rs:3517`, `:3672`, `:3788`.

Consequence: the cuda_graph.rs `with_decode_gates_recurrent_
outputs(...)` calls are effectively **no-ops in production**
today. The "AUTO_FREE_ON_LAUNCH would free intra-capture
allocations" comment at `cuda_graph.rs:1565-1573` describes a
defense that no longer applies — the kt path's `alloc_cuda_
tensor` + dtod-copy both happen inside the capture window
anyway. Either the captured graphs survive those freed
allocations by some other mechanism (e.g., the AUTO_FREE
semantics only apply to specific allocator pools that kt's
`cuda_zeros` and candle's `Tensor::zeros` don't use), or there
is a latent graph-replay-stability issue at `bs>=16` waiting to
be tripped. Either way, that question is orthogonal to the
candle-removal task.

A cosmetic kt sibling (`with_decode_gates_recurrent_outputs_kt`
taking `Vec<KtTensor>`) is implementable but has no caller path
that would benefit:
- The struct fields in `CapturedDecodeGraph` and
  `CapturedBatchedDecodeGraph` (`_gdn_decode_outputs: Vec<Tensor>`)
  store candle tensors for graph-stable lifetime ownership.
- The kt-typed decode kernels in `kt_api.rs` do not consult any
  thread-local — they allocate fresh outputs per call.
- Wiring a kt-typed `DECODE_GATES_RECURRENT_OUTPUTS_KT` would
  require rewriting all 8 kt-typed decode kernels to consult it,
  plus changing the `CapturedDecodeGraph` field types and the
  `Self::new_gdn_decode_outputs` allocator. Out of scope for a
  single follow-up PR.

Updated follow-up steps:
1. Land a kt-typed single-block `gdn_full_chunk_forward_kt` to
   eliminate the cuda.rs fall-through.
2. **`with_decode_gates_recurrent_outputs` cleanup (one of):**
   a. **Delete the helper and both call sites** once it is
      confirmed that the kt path's graph-replay-stability does
      not regress at `bs>=16` (the helper sets up state nobody
      reads in production — removing it is a pure scaffolding
      cleanup). Requires the graph-stability audit above.
   b. **Wire the kt kernels to a kt-typed thread-local** if the
      audit finds we actually need graph-stable pre-allocated
      outputs. Adds `with_decode_gates_recurrent_outputs_kt` +
      `next_decode_gates_recurrent_output_kt`, updates all 8
      kt-typed decode kernels in `kt_api.rs`, changes
      `CapturedDecodeGraph._gdn_decode_outputs` to `Vec<KtTensor>`,
      and changes `Self::new_gdn_decode_outputs` to allocate kt
      tensors via `kiln_kt_bridge::alloc_cuda_tensor`.
3. Migrate the forward.rs prefill-chunk-loop sites once #1 lands.
4. Delete candle surface from `lib.rs`.
5. Drop `candle-core` from Cargo.toml (substrate-blocked on the
   same kt-bridge cleanup as conv1d/marlin).

### kiln-flash-attn — ✅ **CANDLE-FREE** (Tier-1 complete)

All public entry points operate on `kiln_tensor::Tensor`. The
candle-typed parallel surface (`flash_attn_fwd`, `flash_attn_bwd`,
`flash_attn_paged_decode*`, `paged_kv_write_*`) was deleted from
`crates/kiln-flash-attn/src/lib.rs` after every `kiln-model`
production caller migrated to the `*_kt` wrappers:
- `crates/kiln-model/src/backend/cuda.rs` — FlashAttention forward,
  paged decode, dyn-seqlen paged decode, and graph-output dyn-seqlen
  paged decode all route through kt wrappers.
- `crates/kiln-model/src/cuda_train.rs` — SFT/GRPO prefill
  FlashAttention forward and backward route through kt wrappers.
- `crates/kiln-model/src/forward.rs` — `CudaFlashAttentionTrainingBf16`
  forward, backward recompute, and backward route through kt wrappers;
  backward collapses expanded GQA dk/dv back to heads_kv before
  returning gradients.
- `crates/kiln-model/src/paged_kv_cache.rs` — the 3 CUDA BF16 paged-KV
  writer fast paths route through `paged_kv_write_token_major_bf16_slot_kt`,
  `paged_kv_write_token_major_bf16_batch_slot_kt`, and
  `paged_kv_write_token_major_bf16_kt` unconditionally.

The in-lib parity tests (`kt_flash_attn_regression`) and the candle
parity comparison in `tests/kt_v2_smoke.rs` were dropped alongside the
candle shell; the remaining kt smoke tests construct CUDA inputs
candle-free via `kiln_tensor::Tensor::cuda_from_slice`. `Cargo.toml`
no longer depends on `candle-core` (matches the `kiln-conv1d-kernel` /
`kiln-marlin-gemm` Tier-1 precedent).

Remaining `kiln-flash-attn` follow-up (out of scope for this PR):
1. Complete the PagedKvCacheKt migration so `paged_kv_cache.rs` can be
   removed entirely.

## Audit summary

| Crate | Candle-typed callers (kiln-model) | Can drop candle now? | Blocker |
|---|---|---|---|
| `kiln-conv1d-kernel`  | 0   | **No** | substrate (kt_api uses candle CudaStorage internals) |
| `kiln-marlin-gemm`    | 0   | **No** | substrate (same as conv1d) |
| `kiln-rmsnorm-kernel` | 53  | **No** | kt_api surface partial; needs op-family expansion |
| `kiln-gdn-kernel`     | 9   | **No** | 1 single-block fall-through + 6 forward.rs prefill + 2 cuda_graph |
| `kiln-flash-attn`     | 0   | ✅ **Yes (done)** | Tier-1 complete; candle dep dropped from `Cargo.toml`. |

**Production-caller status:** As of `0841c266`,
`kiln-conv1d-kernel` and `kiln-marlin-gemm` have **zero**
candle-typed production callers in `kiln-model`. Their
`Cargo.toml` `candle-core` deps remain only because the substrate
itself (`kiln-tensor::CudaStorage` + `kiln-kt-bridge`) still goes
through candle for CUDA context ownership.

**Substrate path forward**: extend the
`CudaStorage::cuda_stream_raw()` pattern landed in `d561dbf8` to
the full Arc<CudaDevice> internals, then the kernel-crate
`kt_api.rs` files can call `CudaStorage` accessors without any
candle import — at which point `kiln-conv1d-kernel` and
`kiln-marlin-gemm` drop candle in one mechanical PR each. The
in-lib `#[cfg(test)]` parity scaffolds + a candle-free CPU
construction path on `kiln_tensor::Tensor` are the parallel
unblock.

The audit numbers in this section supersede the per-crate counts in the
status snapshot above when they disagree (the status snapshot reports
"X/Y kt_api functions wired", which counts kt-API _coverage_; this
section reports the candle-typed _caller count_, which is what gates
dropping the dep).

## kiln-autograd readiness for CustomOp migration (investigated 2026-05-26)

Investigation into whether `crates/kiln-autograd/` is ready to replace
candle's `CustomOp1/2/3` traits in production training paths (for
`#1082`). Findings:

**Substrate state.** `kiln-autograd` exposes a complete tape over
`kiln_tensor::Tensor` + `TensorId`: `BackwardOp` trait
(`name`/`input_count`/`apply`/`requires_input`), `Tape`
(record/backward/clear/reachable_from) with anti-pattern 16
version-counter enforcement, `GradStore`, `KILN_DETECT_ANOMALY` wired
end-to-end, and 30+ concrete `BackwardOp` implementations under
`src/backwards/` (matmul, reduce, embedding, rmsnorm, layernorm, rope,
cross-entropy, all elementwise + activation + trig families, etc.).
`MatmulBackward`/`AddBackward`/`MulBackward`/`EmbeddingBackward` all
run on CUDA-resident tensors today, verified by
`crates/kiln-kt-bridge/tests/cuda_backward_parity.rs`. Gap:
`SoftmaxLastDimBackward` and other `backwards/activation.rs` ops use a
`load_f32` helper that hard-requires `CpuStorage` and error out on
CUDA (intentional; tracked by `cuda_softmax_backward_currently_errors_
on_cuda_storage`). Same applies to several other backwards that hand-
roll their math; the CUDA-ready set today is the subset that goes
through `kiln_tensor::ops::*` which already dispatch to CUDA.

**Production caller count: zero.** `grep kiln_autograd crates/`
shows only test-only consumers (`kiln-kt-bridge` dev-dep,
`kiln-tensor` dev-dep, `kiln-optim` dev-dep) plus `kiln-autograd`'s
own internal tests. No `kiln-model`/`kiln-train`/`kiln-server` file
imports `kiln_autograd::*`. The tape is built but unused by
production. Production training uses candle's `Var` + `AdamW` +
`loss.backward()` exclusively, and every fused-with-backward kernel
(`RmsNormCustomOp`, `FlceCustomOp`, `OpdLossCustomOp`,
`InjectTensorGradient`, the 7 `CudaLora*`/`CudaSigmoidMul*`/
`CudaFlashAttention*`/`CudaRotaryOne*` ops in `forward.rs`) is a
candle `CustomOp{1,2,3}` whose `bwd()` returns candle Tensors.

**Bridge story.** `crates/kiln-kt-bridge/src/lib.rs` already has every
primitive needed for a candle `CustomOp::bwd` body to compute via kt
ops and hand the result back to candle's autograd. The relevant
helpers (all `pub`, all CUDA-tested):
`kt_tensor_from_candle_cuda_borrow` (zero-copy candle→kt),
`kt_tensor_from_candle_cuda_copy` (copying variant),
`kt_tensor_to_candle_cuda_copy` (kt→candle, used in 14+ kt-API
production sites in `forward.rs::rms_norm`). Candle's autograd
(`vendor/candle-core/src/backprop.rs:644-670`) consumes whatever
`Tensor` value `c.bwd(...)` returns and `grads.insert_or_add`'s it
keyed on the input `Tensor`'s id — it does not require the gradient
to have been produced by candle ops, only that it's a candle `Tensor`
of matching shape and dtype on the same device. **Therefore: candle
backward can be satisfied by a kt-computed gradient, round-tripped
through `kt_tensor_to_candle_cuda_copy` at the `bwd()` boundary.**
There is no need for kt-autograd's `Tape` to drive candle's backprop
walker — the seam is at one op's `bwd()` body, not at the graph
level. (One device-to-device memcpy per migrated op per training step
is the bridge cost; ungated by anything except the v2 borrow direction
for kt→candle, which is a follow-up.)

**Recommended first migration: `RmsNormCustomOp::bwd`** (in
`crates/kiln-rmsnorm-kernel/src/lib.rs:830`). Rationale:
(a) the forward and backward kt entry points already exist
(`fused_rmsnorm_kt`, `fused_rmsnorm_backward_kt` in
`crates/kiln-rmsnorm-kernel/src/kt_api.rs:65,130`) and call the same
FFI symbols as the candle `bwd()` body, so bit-exact parity is by
construction; (b) inputs are just `(x, weight)` BF16 + scalar `eps`,
no per-position masks or chunk-loop state; (c) the failure mode is
isolated — a regression shows up at one call site
(`kiln-model/src/forward.rs::rms_norm`'s training branch) instead of
spreading across 32 layers' worth of fused-attn or 14 trainer call
sites; (d) the crate is already on the Tier-1 closure list as
**BLOCKED**, with this exact gap (autograd-aware migration) called
out in the audit. The other candidates are strictly more entangled:
`InjectTensorGradient` is a clever hack that only stores an upstream
Tensor — there's nothing to migrate, the bwd already just returns
the stored tensor; `OpdLossCustomOp::bwd` and `FlceCustomOp::bwd`
both close over `Vec<u32>`/`Vec<bool>`/`chunk_size` and call into
chunk-loop helpers (`backward_dhidden` and friends) that are still
pure-candle multi-hundred-line bodies; the `CudaLora*`/
`CudaFlashAttention*` ops in `forward.rs` are all `CustomOp3` with
LoRA-state entanglement and live inside the 5000-line forward path.
`VulkanRmsNormOp` is similarly small but Vulkan-only — CUDA migrates
more callers per unit work.

**Recommended approach: (a) the bridge approach for the first
migration, sequenced into (b) over many later PRs.** The bridge
approach replaces `RmsNormCustomOp::bwd`'s body — currently a
candle-side `kiln_fused_rmsnorm_bwd` FFI call wrapped in
`Tensor::from_storage` and `BackpropOp::none()` — with the kt-typed
chain `kt_tensor_from_candle_cuda_borrow(x) +
kt_tensor_from_candle_cuda_borrow(weight) +
kt_tensor_from_candle_cuda_borrow(grad_y) →
fused_rmsnorm_backward_kt(...) → kt_tensor_to_candle_cuda_copy(...)`.
Candle's `CustomOp2 for RmsNormCustomOp` itself stays as the
candle-autograd integration surface; only the body changes. This:
(1) proves the bridge end-to-end on one of the smallest possible
surfaces; (2) is reversible — flip back to the candle-FFI body if
parity fails; (3) deletes zero LOC of candle structurally, so it
doesn't yet advance Tier 5, but it does prove that the
`kt_tensor_to_candle_cuda_copy` boundary works for autograd-tracked
tensors at training time (today it's only exercised on inference
paths via the `track_op()` check). Once the bridge pattern is
proven for rmsnorm, the same template applies to the other 10
`CustomOp{1,2,3}` impls, and only after every candle-typed `bwd`
body is using the bridge does the full migration to (b) make sense
— at which point the whole training loop swaps from `loss.backward()`
to `tape.backward(loss_id, ...)` in a single coordinated PR, the kt
`Tape`/`BackwardOp` replace the candle `Op::CustomOp{1,2,3}`
recording sites, and the candle `Var`/`AdamW`/`Variables` machinery
gets replaced (the `kiln-optim` crate already has the kt-side
equivalents wired to `kiln-autograd` per its dev-dep). Smallest
plausible "first proof" PR: ~50 LOC in `kiln-rmsnorm-kernel`
swapping the bwd body, plus a parity-vs-current test, with no
changes outside the crate. Not yet executed; this paragraph is the
plan, not a status report.

**Status update (2026-05-26): first migration shipped on
`rmsnorm-bwd-kt-bridge-1082`** (commits `4a76f5d6` + `6a0f9dfe`).
The `RmsNormCustomOp::bwd` body now routes through
`fused_rmsnorm_backward_via_kt_bridge` in
`crates/kiln-rmsnorm-kernel/src/lib.rs:768`, which does exactly the
chain described above: 3 candle→kt `kt_tensor_from_candle_cuda_borrow`
calls + `fused_rmsnorm_backward_kt` + `kiln_f32_to_bf16` direct FFI
(to cast only the first `hidden` slots of the kt path's over-
allocated `[rows, hidden]` F32 partial buffer) + 2
`kt_tensor_to_candle_cuda_copy` calls. Kill switch
`KILN_DISABLE_RMSNORM_BWD_KT_BRIDGE=1` keeps the candle
`fused_rmsnorm_backward` reachable as the parity-test fallback;
the bwd body also falls through to the candle path on any bridge
error (borrow/alloc/FFI/copy-back failure). Verified on L40S
(`KILN_CUDA_ARCHS=89`): all 26 `kiln-rmsnorm-kernel` tests pass
including the new `test_cuda_rmsnorm_bwd_kt_bridge_default_matches_
candle_path` (decode `[1, 2560]`, prefill `[512, 2560]`, tiny
`[4, 128]`). `cargo build --release --features cuda --bin kiln`
also passes (14m 23s on L40S). `grad_x` is bit-exact between paths
(no cross-row reduction); `grad_w` may differ by up to one BF16 ULP
across separate launches because the kernel's `atomicAdd` row
reduction is order-non-deterministic (test tolerance 5e-2, matching
the `parity_backward_multi_row_cuda` precedent of `tol=2e-2`
against the candle reference). This is the **first** production
CustomOp::bwd body using the bridge; the template now applies to
`OpdLossCustomOp::bwd` and `FlceCustomOp::bwd` (modulo the
chunk-loop-helper porting work called out above).
