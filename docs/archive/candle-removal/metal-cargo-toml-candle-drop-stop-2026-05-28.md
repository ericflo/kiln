# STOP: dropping `candle-core` from `kiln-tensor`'s metal feature in Cargo.toml (#1082, 2026-05-28)

## Bottom line

The task asked: **drop `candle-core` from `crates/kiln-tensor/Cargo.toml`'s
`[dependencies]`**.

After Wave-14 (commits `3f00a979`, `a07e7781`, `ae08652c`, `d8d43c6d`),
`kiln-tensor`'s `metal_storage.rs` and `metal_allocator.rs` are **fully
candle-core-free** — the 7 in-file substrate ops, the storage field
state, and the allocator field state all reach Metal substrate
primitives through `kiln_tensor::metal_types::MetalCompanion` (a
kt-native struct holding `candle_metal_kernels::{Device, Kernels,
Commands}` directly — none of those types depend on `candle-core`).

**However, `candle-core` cannot be dropped from `kiln-tensor`'s
`metal` feature in `Cargo.toml` in scope for this commit chain.** Four
items in `crates/kiln-tensor/src/metal_types.rs` continue to require
it, and all four are consumed at ~48 callsites in
`crates/kiln-model/src/backend/metal.rs` that the task's "touch ONLY"
list explicitly forbids editing.

## What still pulls `candle-core` under `metal`

`crates/kiln-tensor/src/metal_types.rs`:

1. `pub use candle_core::metal_backend::MetalDevice;` (line 108)
2. `pub use candle_core::metal_backend::DeviceId;` (line 114)
3. `pub use candle_core::Storage;` (line 123)
4. `pub use candle_nn::ops::sdpa;` (line 140)
5. `pub fn buffer_o<'a>(buffer, l: &candle_core::Layout, dtype: candle_core::DType)`
   (lines 167-176)

Each is consumed by `crates/kiln-model/src/backend/metal.rs` at a
non-trivial number of call sites (counts from `git grep` on `main` at
`d8d43c6d`):

| Symbol                                      | Callers in `kiln-model::backend::metal`                          |
|---------------------------------------------|------------------------------------------------------------------|
| `metal_types::MetalDevice`                  | ~48 (every `metal_*_pipeline` / `metal_*` helper signature)      |
| `metal_types::DeviceId`                     | Cache keys in the `HashMap<DeviceId, Arc<ComputePipeline>>` per-fn pipeline caches |
| `metal_types::Storage`                      | `Storage::Metal(s) => s` downcast in every kernel-helper FFI body |
| `metal_types::sdpa`                         | 9 callsites (3 production SDPA dispatch, 6 test/parity helpers)  |
| `metal_types::buffer_o`                     | 232+ callsites (the kernel-FFI `BufferOffset` constructor)        |

## What it would take to drop `candle-core` from kiln-tensor's `metal` feature

Each of the five chokepoints needs a kt-native replacement, AND every
caller in `kiln-model::backend::metal` needs to migrate onto it:

### 1. `MetalDevice` → kt-native handle

`MetalDevice` is consumed in helper signatures (`fn
metal_X_pipeline(device: &MetalDevice) -> ComputePipeline`). It's used
internally to reach `device.metal_device()` / `device.kernels()` /
`device.command_encoder()`. The kt-native replacement is the
`kiln_tensor::metal_types::MetalCompanion` introduced in `3f00a979` —
swap the parameter type to `&MetalCompanion`, swap the three accessor
calls to `.device()` / `.kernels()` / `.command_encoder()`. Mechanical
but touches ~48 sites.

### 2. `DeviceId` → kt-native device identity

`DeviceId` is candle's `usize` newtype for caching per-device pipelines.
The per-function `OnceLock<Mutex<HashMap<DeviceId, ...>>>` pattern
needs a kt-native equivalent — either a kt-side `DeviceId` newtype
keyed off `MetalCompanion::device_id()` (which exposes the underlying
`MTLDevice::registryID()` already used by candle), or the cache key
becomes `usize` directly. The kt-native version is cleaner — `usize`
keys remove a candle-typed cache parameter from every helper.

### 3. `Storage` → kt-native storage match-arm

The `Storage::Metal(s) => s` downcast in kernel-helper bodies receives
a candle `Tensor::storage()` lock guard. After the in-flight kt-tensor
migration in kiln-model (already underway — see Wave 13's `kt_metal_X`
pattern), these helpers will accept `&kiln_tensor::Tensor` instead and
downcast to `kiln_tensor::MetalStorage` via the existing
`storage().as_any().downcast_ref::<MetalStorage>()` path. ~48 call
sites mirror the `MetalDevice` count.

### 4. `sdpa` → kt-native fused SDPA

`candle_nn::ops::sdpa` is the MLX-style fused scaled-dot-product
attention kernel. Replacing it requires either:
- Adding a kt-native `kiln_tensor::metal::sdpa(q, k, v, scale)` op (a
  new `metal_sdpa_*_axis` substrate op in `metal_storage.rs` calling
  the same MSL kernel candle_nn dispatches into), OR
- Routing the 9 callsites through `kiln-mps` (the kt-native MSL kernel
  crate) once it lands.

The 9 callsites are all in `kiln-model::backend::metal` (3 production,
6 test).

### 5. `buffer_o` signature → kt-native Layout/DType

The `buffer_o` helper formula is `l.start_offset() *
dtype.size_in_bytes()`. Re-typing the signature as `(buffer: &Buffer,
l: &kiln_tensor::Layout, dtype: kiln_tensor::DType) -> BufferOffset`
is a one-line change in `metal_types.rs`. The hard part is the 232+
callers in `kiln-model::backend::metal` that currently pass
`&candle_core::Layout` (from `tensor.layout()` on a candle Tensor) and
`candle_core::DType` (from `tensor.dtype()`).

Migration shape: every caller swaps `tensor.layout()` →
`kt_tensor.layout()` and `tensor.dtype()` → `kt_tensor.dtype()`. The
kt-side `Layout` and `DType` types already exist (see
`kiln-tensor/src/layout.rs` and `kiln-tensor/src/dtype.rs`); they
mirror candle's shape exactly for the `start_offset()` /
`size_in_bytes()` accessors `buffer_o` uses.

This migration is also gated on the kt-tensor adoption in
`kiln-model::backend::metal`'s helpers — same 232+ sites that need
`Storage::Metal(s) => s` → kt-side downcast.

## Why this is out of scope for the current commit chain

The task description explicitly limited the file scope:

> Coordinate
> ...
> 4 other agents running. You touch ONLY:
> - `crates/kiln-tensor/src/metal_storage.rs`
> - `crates/kiln-tensor/src/metal_types.rs`
> - `crates/kiln-tensor/src/metal_allocator.rs`
> - `crates/kiln-tensor/Cargo.toml`

Every chokepoint above requires editing
`crates/kiln-model/src/backend/metal.rs` — explicitly excluded. The
file is also large (~14,000 lines) and concurrent with other agents'
Wave 12-13 work touching the same surface area (commits `1b7f6b80`,
`7b0e9dbd`).

## What this commit chain DID land

After Wave-14 (`3f00a979` → `d8d43c6d`):

- **`kiln-tensor::metal_storage.rs` is candle-core-free** at both the
  field level (was already, post-CP-1 final lift in earlier waves)
  AND the op-dispatch level (new in Wave 14).
- **`kiln-tensor::metal_allocator.rs` is candle-core-free** at the
  field level (was already, post-CP-1) and now has no broken intra-doc
  links to deleted symbols.
- **`MetalCompanion`** — a kt-native substrate type — lives in
  `metal_types.rs` and is the canonical accessor for the in-file ops.
- **`primary_metal_device`** and **`MetalStorage::candle_device`** —
  the two candle-derivation shims — are gone.
- **Five candle-core lib re-exports** (`MetalDevice`, `DeviceId`,
  `Storage`, `sdpa`) remain in `metal_types.rs` for downstream
  consumption by `kiln-model::backend::metal` only.

## Substrate readiness for the eventual Cargo.toml drop

The kt-native substrate to replace the candle re-exports is in place
or trivially within reach:

| Candle re-export   | kt-native substrate                             | Status         |
|--------------------|-------------------------------------------------|----------------|
| `MetalDevice`      | `kiln_tensor::metal_types::MetalCompanion`      | **In place**   |
| `DeviceId`         | `usize` from `MetalCompanion::device_id()`      | One method add |
| `Storage`          | `kiln_tensor::Storage` + `.as_any().downcast_ref::<MetalStorage>()` | **In place** |
| `sdpa`             | new `metal_sdpa_*_axis` substrate op            | Out of scope   |
| `buffer_o` types   | `kiln_tensor::Layout` + `kiln_tensor::DType`    | **In place**   |

The `sdpa` op is the only substrate gap. The remaining work is
mechanical caller migration in `kiln-model::backend::metal`.

## Recommended next commit chain (a separate scoped task)

1. **kiln-tensor**: Add `MetalCompanion::device_id()` returning
   `usize` (one-line method off the underlying `MTLDevice::registryID`
   the wrapper already exposes).
2. **kiln-tensor**: Add `buffer_o_kt(buffer, l: &kiln_tensor::Layout,
   dtype: kiln_tensor::DType)` alongside the existing `buffer_o` —
   same formula, kt-typed args.
3. **kiln-tensor**: Add `metal_sdpa_last_axis` substrate op (mirrors
   `metal_softmax_last_axis` in shape).
4. **kiln-model**: Mass-migrate `kiln-model::backend::metal`'s 48
   helper signatures from `&MetalDevice` → `&MetalCompanion`. ~48
   per-fn search/replace passes; same for `.metal_device()` /
   `.kernels()` / `.command_encoder()` site-by-site.
5. **kiln-model**: Mass-migrate `buffer_o(...)` callers to
   `buffer_o_kt(...)` (232+ sites).
6. **kiln-model**: Migrate the 9 `sdpa(...)` callsites to
   `kiln_tensor::metal_sdpa_last_axis`.
7. **kiln-model**: Migrate `Storage::Metal(s) => s` pattern-matches
   to kt-side downcast (already partially done — Wave 13 introduced
   `kt_metal_X` callers that use the kt-tensor path).
8. **kiln-tensor**: Delete the 5 candle re-exports from
   `metal_types.rs` once `kiln-model::backend::metal` no longer
   imports them.
9. **kiln-tensor**: Drop `candle-core` and `candle-nn` from the
   `metal` feature in `Cargo.toml`. (May still need them under the
   `cuda` feature — that's the parallel CP-1 work.)

Steps 4-7 are individually mechanical but volume-intensive. The
right way to schedule this is one PR per helper-family group
(SDPA, conv1d, gdn, mlp, embedding, ...) — same pattern Wave 12-13
used.

## See also

- `docs/archive/candle-removal/issue-1082-tier-4-5-roadmap-2026-05-27.md` — the broader #1082
  roadmap. Section CP-2 ("`kiln-tensor::MetalStorage` candle removal
  (parallel to CP-1)") is now complete; the remaining work this STOP
  doc describes is the chokepoint-flip side of the same item.
- `crates/kiln-tensor/src/metal_storage.rs:53-68` — the module-level
  comment now reflects the candle-core-free state.
- Commits `3f00a979` (MetalCompanion type), `a07e7781`
  (accessor + cache), `ae08652c` (op migration), `d8d43c6d` (shim
  deletion).
