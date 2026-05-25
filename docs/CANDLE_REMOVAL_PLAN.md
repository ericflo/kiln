# Candle Removal Plan (#1082)

This document inventories every `candle_core` and `candle_nn` reference in
the kiln workspace and the migration path to a candle-free build. It is
the canonical artifact for tracking Phase 7 closeout against issue #1082.

Last refreshed: 2026-05-25, against `main` post-merge of `ce/1082-pagedkv-accessors-migrate` (`40666082`) + 9 Metal/Vulkan kernel wires (`ddee82e6`, `1471dafb`, `b5ec4874`, `a88d0842`, `93ad0360`, `bb2e6283`, `5d9dcef3`, `742cf7e4`, `c846f188`) + 10 cuda_train.rs kt-API wires (`ddd8379c` through `e99a1a42`).

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

## Build matrix coverage required before each tier closes

| Tier | Required CI green |
|---|---|
| Tier 1 crate close | `linux-default` + `linux-cuda` (the kernel crates own gate) |
| Tier 2 crate close | Same + `macos-metal` for `kiln-mps`, `linux-vulkan` for `kiln-vulkan-kernel` |
| Tier 3 close       | Full matrix + opd/sft regression nightlies green |
| Tier 4 close       | Full matrix |
| Tier 5 delete      | Full matrix + a fresh `cargo tree -i candle-core` returning empty |

## Status snapshot (2026-05-25, refreshed)

- Tier 0: ✅ already candle-free (kiln-core, kiln-scheduler, kiln-nvtx).
- Tier 1: 🟡 in flight — all 5 kernel crates have kt_api surfaces.
  **Production wiring breakthrough** (#695587df): first kt_api production wire
  for a non-rmsnorm kernel crate. kiln-conv1d-kernel's `causal_conv1d_update_kt`
  + `causal_conv1d_prefill_kt` are now both wired in `CudaBackend` at
  `crates/kiln-model/src/backend/cuda.rs:1004` and `:1038`.
  Status by crate:
    - kiln-rmsnorm-kernel: 7 production `_kt` callers in forward.rs ✓
    - kiln-conv1d-kernel: 2/2 kt_api functions wired in CudaBackend ✓ + byte-exact parity tests at B=1/C=8192 (`69a5f68c` update + `1cb0c107` prefill, 0 mismatches across 8192/65536 output elements and 24576 conv_state elements). **Default ON** — env gate flipped from `KILN_USE_KT_API_CONV1D` opt-in to `KILN_DISABLE_KT_API_CONV1D` opt-out as the first Tier 1 default flip.
    - kiln-marlin-gemm: 1/1 kt_api function wired in marlin_proj::matmul_bf16 ✓ (`8b415107` + `668b0847` bridge extension for I32→U32)
    - kiln-gdn-kernel: 5/10 kt_api functions wired in CudaBackend ✓ (`forward_substitution`/`14c17570`, `recurrent_forward`/`7a357a3d`, `chunk_prep`/`68a3667e`, `chunk_scan`/`1edc82dc`, `full_chunk_forward_multiblock`/`57b37c00`) — 5 remaining (decode_gates_recurrent variants + gated_rms_norm)
    - kiln-flash-attn: 4/5 production wires ✓ (`flash_attn_fwd_kt` at `f3b7e797`, `flash_attn_paged_decode_kt` at `c49c1995`, `flash_attn_paged_decode_dyn_seqlen_kt` at `7fe3011f`, no-graph-outputs `dyn_seqlen` variant at `276482d6`) + 2/5 functions used internally by `paged_kv_cache_kt.rs` (`paged_kv_write_token_major_bf16_slot_kt` + `_bf16_kt`); 1 entry point remaining (`flash_attn_bwd_kt` for the training path) + the `with_graph_outputs` kt-API extension. **Default ON** — env gate flipped from `KILN_USE_KT_API_FLASH_ATTN` opt-in to `KILN_DISABLE_KT_API_FLASH_ATTN` opt-out. Bit-exact by construction: all 4 wired sites bottom out in the same FFI symbols as the candle shim. The `with_graph_outputs` site keeps its `graph_outputs.is_none()` guard so the CUDA-graph caller-owned-output path still uses the candle wrapper.
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

### kiln-conv1d-kernel — **BLOCKED** (4 candle-typed callers in kiln-model)

Candle-typed entry points still reachable in production:
- `kiln_conv1d_kernel::supports` — `crates/kiln-model/src/backend/cuda.rs:1627`
- `kiln_conv1d_kernel::causal_conv1d_update` — `crates/kiln-model/src/backend/cuda.rs:1649`
- `kiln_conv1d_kernel::supports_prefill` — `crates/kiln-model/src/backend/cuda.rs:1664`
- `kiln_conv1d_kernel::causal_conv1d_prefill` — `crates/kiln-model/src/backend/cuda.rs:1683`

The kt-API default-on flip means the candle path is unreachable when
`KILN_DISABLE_KT_API_CONV1D` is unset (production default). However, the
candle path remains as the **escape-hatch fallback** when the env gate is
set, and `supports`/`supports_prefix` are still called pre-dispatch as
shape predicates. Both fallback and predicate callers must be migrated
(or `supports*` exposed as kt-typed predicates) before candle can be
dropped from the crate.

Follow-up steps (not done in this PR):
1. Add `kt_api::supports_kt` / `supports_prefill_kt` (or a kt-typed
   replacement predicate that takes `&kiln_tensor::Tensor`).
2. Migrate the four call sites to the kt-typed surface.
3. Delete the candle public functions from `lib.rs` (keep FFI + kt_api).
4. Drop `candle-core = { workspace = true, features = ["cuda"] }`.

### kiln-marlin-gemm — **BLOCKED** (5 candle-typed callers in kiln-model + 1 test)

Candle-typed entry points still reachable in production:
- `kiln_marlin_gemm::pack::quantize_and_pack` — `crates/kiln-model/src/marlin_proj.rs:131`
- `kiln_marlin_gemm::marlin_w4a16_gemm` — `crates/kiln-model/src/marlin_proj.rs:381`
- `kiln_marlin_gemm::marlin_w4a16_gemm` — `crates/kiln-model/src/marlin_proj.rs:402`
- `kiln_marlin_gemm::marlin_w4a16_gemm` — `crates/kiln-model/src/forward.rs:30696` (parity-test scaffolding `out_candle` reference)

Other workspace consumers of the candle-typed surface:
- `crates/kiln-marlin-gemm/tests/parity.rs:20` — own crate's parity test
- `crates/kiln-model/tests/marlin_qproj_parity.rs:82` — `kiln_marlin_gemm::pack::quantize_and_pack`

The kt-API wire (`marlin_w4a16_gemm_kt`) is default-on for the matmul
path, but `pack::quantize_and_pack` (the offline weight-packing routine)
still hands back candle tensors. Until pack is migrated to the kt-typed
surface (or moved behind a feature flag separate from the matmul path),
the crate cannot drop candle-core.

Follow-up steps (not done in this PR):
1. Add `kt_api::quantize_and_pack_kt` returning `kiln_tensor::Tensor`s.
2. Migrate `marlin_proj.rs:131` to it.
3. Migrate the parity-test scaffolding in `forward.rs:30696` to compare
   kt outputs (or wrap the comparison in `#[cfg(test)]`).
4. Delete the candle public functions from `lib.rs`.

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

### kiln-gdn-kernel — **BLOCKED** (30 candle-typed callers in kiln-model)

Candle-typed entry points still reachable in production:

**`crates/kiln-model/src/backend/cuda.rs`** (22 sites — all `_supports`
predicates plus candle fallbacks for the 10 kt-wired entry points):
- `gdn_forward_substitution`, `gdn_recurrent_forward`,
  `gdn_chunk_prep[_supports]`, `gdn_chunk_scan[_supports]`,
  `gdn_full_chunk_forward[_supports]`,
  `gdn_full_chunk_forward_multiblock[_supports]`,
  `gdn_decode_gates_recurrent[_supports]`,
  `gdn_decode_qk_norm_gates_recurrent[_supports]`,
  `gdn_decode_qk_norm_gates_recurrent_rmsnorm[_supports]`,
  `gdn_gates_decline_reason`, `gdn_gates`, `gdn_gated_rms_norm[_supports]`,
  plus the `GDN_FULL_CHUNK_FORWARD_MULTIBLOCK_DV_TILE` constant.

**`crates/kiln-model/src/forward.rs`** (6 sites): direct calls to
`gdn_chunk_prep`, `gdn_chunk_scan`, `gdn_full_chunk_forward`,
`gdn_full_chunk_forward_multiblock` (called outside the backend
dispatcher path, e.g. during prefill chunk loops).

**`crates/kiln-model/src/cuda_graph.rs`** (2 sites):
`with_decode_gates_recurrent_outputs` — graph-output wrapper.

Plus crate-internal examples + tests using the candle-typed surface.

The kt-API wire status (5/10 functions wired) plus the **default-on**
flip means many entry points already route through kt at runtime, but
the candle fallback path is still compiled and reachable. The
`_supports` predicate calls and the `forward.rs` direct-chunked-prefill
calls are non-trivial extra work because their inputs are still
candle-typed at the call site.

Follow-up steps (not done in this PR):
1. Land the remaining 5/10 kt-API wires (decode_gates_recurrent
   variants + gated_rms_norm — already listed in the status block).
2. Add kt-typed `_supports` predicates.
3. Migrate the forward.rs prefill-chunk-loop sites (these likely
   require a kt-typed sibling for `with_decode_gates_recurrent_outputs`).
4. Delete candle surface from `lib.rs`.

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
| `kiln-conv1d-kernel`  | 4   | **No** | `supports*` predicates + fallback path callers |
| `kiln-marlin-gemm`    | 5+1 | **No** | `pack::quantize_and_pack` + matmul fallback |
| `kiln-rmsnorm-kernel` | 53  | **No** | kt_api surface partial; needs op-family expansion |
| `kiln-gdn-kernel`     | 30  | **No** | 5/10 wires remaining + `_supports` + forward.rs prefill sites |
| `kiln-flash-attn`     | 15  | **No** | `flash_attn_bwd_kt` + PagedKvCacheKt migration |

**Conclusion: none of the 5 Tier 1 kernel crates can drop `candle-core`
from their `Cargo.toml` in this pass.** Each has known follow-up work
listed above. The smallest residual surface is on `kiln-conv1d-kernel`
(4 sites), which is therefore the closest to candle removal once kt-API
`supports*` predicates land and the fallback path is removed (or guarded
to a `#[cfg(test)]` parity scaffold).

The audit numbers in this section supersede the per-crate counts in the
status snapshot above when they disagree (the status snapshot reports
"X/Y kt_api functions wired", which counts kt-API _coverage_; this
section reports the candle-typed _caller count_, which is what gates
dropping the dep).
