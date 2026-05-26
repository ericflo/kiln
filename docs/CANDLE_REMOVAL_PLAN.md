# Candle Removal Plan (#1082)

This document inventories every `candle_core` and `candle_nn` reference in
the kiln workspace and the migration path to a candle-free build. It is
the canonical artifact for tracking Phase 7 closeout against issue #1082.

Last refreshed: **2026-05-26**, against `main` at `fdeace4b` —
post Phase 5 default-on flip (`6d564b9a`,
`KILN_CUDA_GRAPHS_BATCHED=1` + `_KV_FUSED=1` are now the default
on CUDA) and KILN_DETECT_ANOMALY substrate landing
(`3c90d064` + `fdeace4b`).

**Today's batch headline:**
- 🎉 **Phase 5 default-on** (`6d564b9a`) — bs>1 captured-batched
  graphs + fused KV writer are default. End-to-end
  `compute-sanitizer memcheck` clean on A6000 at the gating HEAD
  (`a2cb9edb`).
- 🎉 **Tier 1 closed** for `kiln-conv1d-kernel` (`577f8b0c`) and
  `kiln-marlin-gemm` (`4a862711`) — `candle-core` dropped from
  both `[dependencies]` blocks.
- 🎉 **KILN_DETECT_ANOMALY** Phase 9 trap wired end-to-end:
  scaffold (`72c2c16f`) + `Tensor::all_finite()` substrate
  primitive (`3c90d064`) + `Tape::backward` integration
  (`fdeace4b`). Set `KILN_DETECT_ANOMALY=1` and the autograd
  tape panics at the producing op's tape position on the first
  NaN/Inf gradient.
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
20  kiln-vulkan-kernel
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
| `kiln-opd-loss-kernel`  | Smaller surface; mirror the rmsnorm path. |
| `kiln-flce-kernel`      | **Largest single rewrite.** Phase A is pure-candle, Phase B is the raw-CUDA replacement. Driving Phase B to closeout is its own multi-PR effort (see crate-level `phase_b.rs`). |
| `kiln-mps`              | Apple Silicon only; Metal storage substrate lives here. |
| `kiln-vulkan-kernel`    | 20 .rs files importing candle. Most are integration shims that can be removed once `kiln-tensor`s Vulkan backend is fully wired (current scaffolds: ~30 `vulkan_fwd: Ok(None)` sites). |

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

## Phase 5 (CUDA graph capture) — DEFAULT ON (2026-05-26)

`KILN_CUDA_GRAPHS_BATCHED=1` and `KILN_CUDA_GRAPHS_BATCHED_KV_FUSED=1`
flipped from default-off to **default-on** in commit `6d564b9a`.
This was the headline #1082 Phase 5 perf-gate item — the unblock for
closing the kiln vs vLLM bs=64 decode-throughput gap.

Validation gate that gated the flip (passed on A6000 at HEAD
`a2cb9edb`):
- Live `kiln-bench --paged` Qwen3.5-4B W4A16 + paged + batched +
  KV-fused: exit 0, decode 74.4 tok/s, mean ITL 13.45 ms, peak VRAM
  11.2 GB at batch=1.
- `compute-sanitizer memcheck` under the same configuration:
  `========= ERROR SUMMARY: 0 errors` across batches 1/4/8/16.

Bug chain it took to get here (all fixed in main):
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

See `bench-results/cuda-graph-status.md` "Phase 5 sanitizer
sweep — 2026-05-26" for the full validation trail.

Escape hatches: `KILN_CUDA_GRAPHS_BATCHED=0` and
`KILN_CUDA_GRAPHS_BATCHED_KV_FUSED=0`.

## KILN_DETECT_ANOMALY (Phase 9 trap) — wired end-to-end (2026-05-26)

The `KILN_DETECT_ANOMALY=1` NaN/Inf trap is now live across the
autograd tape:
- Scaffold (`72c2c16f`): `anomaly_detection_enabled()` +
  `anomaly_panic()` in `kiln-autograd::anomaly`.
- Substrate (`3c90d064`): `Tensor::all_finite()` walks the strided
  view (CPU storage; non-CPU returns `Err` pending per-backend
  is_finite kernel). Handles F32, BF16, F16, FP8E4M3 (no-Inf
  format), FP8E5M2.
- Tape wire (`fdeace4b`): `Tape::backward` reads
  `anomaly_detection_enabled()` once at top, then after each
  `op.apply()` scans returned grads. On first non-finite,
  `anomaly_panic` with the producing op's tape position.

Cost: O(numel) per backward step on CPU when enabled, ~5% per
the issue body. Off-by-default in production; CI training-parity
tests opt in via `KILN_DETECT_ANOMALY=1`.

Remaining: per-backend `is_finite_storage` kernels
(CUDA/Metal/Vulkan) for the GPU training paths, then enable the
trap in the SFT parity CI.

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
      recurrent_outputs` needs a kt sibling that accepts caller-
      owned outputs). `kt_api.rs` already candle-import-free.
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
  takes caller-owned outputs (`(out, lse)` pair) for CUDA-graph
  capture; no kt-typed sibling exists yet.

Follow-up steps:
1. Land a kt-typed single-block `gdn_full_chunk_forward_kt` to
  eliminate the cuda.rs fall-through.
2. Add kt-typed sibling for `with_decode_gates_recurrent_outputs`
  with caller-owned outputs; migrate cuda_graph.rs sites.
3. Migrate the forward.rs prefill-chunk-loop sites once #1 lands.
4. Delete candle surface from `lib.rs`.
5. Drop `candle-core` from Cargo.toml (substrate-blocked on the
  same kt-bridge cleanup as conv1d/marlin).

### kiln-flash-attn — **BLOCKED** (15 candle-typed callers in kiln-model)

Candle-typed entry points still reachable in production:

**`crates/kiln-model/src/paged_kv_cache.rs`** (3 sites):
- `paged_kv_write_token_major_bf16_slot`,
  `paged_kv_write_token_major_bf16_batch_slot`,
  `paged_kv_write_token_major_bf16`.

**`crates/kiln-model/src/backend/cuda.rs`** (4 sites): candle fallbacks
for `flash_attn`, `flash_attn_paged_decode`,
`flash_attn_paged_decode_dyn_seqlen` × 2.

**`crates/kiln-model/src/forward.rs`** (4 sites):
- `flash_attn_fwd` × 2 (decode + train forward)
- `flash_attn_bwd` (training backward)
- `flash_attn_paged_decode_dyn_seqlen` (direct call outside backend
  dispatcher)

**`crates/kiln-model/src/cuda_train.rs`** (2 sites): `flash_attn_fwd` +
`flash_attn_bwd` for the SFT/GRPO training path.

The default-on flip for `KILN_DISABLE_KT_API_FLASH_ATTN` covers the 4
wired entry points (forward, paged decode, paged decode dyn seqlen).
What remains:
- `flash_attn_bwd` has no kt-API entry yet (status: "1 entry point
  remaining for training path").
- `paged_kv_write_*` (kv-cache writer family) has kt-API counterparts
  used by `paged_kv_cache_kt.rs`, but `paged_kv_cache.rs` is the
  candle-typed path still in use; the `Kt` cache type isn't fully wired
  through yet (status: "PagedKvCacheKt accessor migration is partial").

Follow-up steps (not done in this PR):
1. Land `flash_attn_bwd_kt` (training backward kt-API entry).
2. Migrate `cuda_train.rs:3447` + `:3588` to the kt path.
3. Migrate `forward.rs` direct-call sites (`:4751`, `:4776`, `:4852`,
   `:17472`) once the kt-API train backward is available.
4. Complete the PagedKvCacheKt migration so `paged_kv_cache.rs` can be
   removed and the 3 `paged_kv_write_*` candle entries with it.
5. Delete candle surface from `lib.rs`.

## Audit summary

| Crate | Candle-typed callers (kiln-model) | Can drop candle now? | Blocker |
|---|---|---|---|
| `kiln-conv1d-kernel`  | 0   | **No** | substrate (kt_api uses candle CudaStorage internals) |
| `kiln-marlin-gemm`    | 0   | **No** | substrate (same as conv1d) |
| `kiln-rmsnorm-kernel` | 53  | **No** | kt_api surface partial; needs op-family expansion |
| `kiln-gdn-kernel`     | 9   | **No** | 1 single-block fall-through + 6 forward.rs prefill + 2 cuda_graph |
| `kiln-flash-attn`     | ~10 | **No** | `paged_kv_write_*` + PagedKvCacheKt migration |

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
