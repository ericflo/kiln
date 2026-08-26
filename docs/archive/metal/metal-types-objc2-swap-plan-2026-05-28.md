# metal_types chokepoint → objc2-metal substrate swap plan (2026-05-28)

This doc lays out the **step-by-step PR sequence** for retiring the
`candle_metal_kernels::*` and `candle_core::*` re-exports in
`crates/kiln-tensor/src/metal_types.rs` and moving every caller in
`crates/kiln-model/src/backend/metal.rs` onto kt-native objc2-metal
substrate.

It supersedes the **scope-blocked** STOP doc
[`metal-cargo-toml-candle-drop-stop-2026-05-28.md`](../../metal-cargo-toml-candle-drop-stop-2026-05-28.md):
that doc was correctly written when the touch scope was constrained to
`kiln-tensor`-only; this doc covers the multi-PR sequence that crosses
into `kiln-model` (the dominant consumer) and lands the substrate flip
all the way through to the Cargo.toml drop.

This is one of two architectural pieces (alongside CP-4) that gate
Tier-5 candle removal. It is **independent of CP-4** — kt-native
autograd is not required to flip the Metal substrate. They can ship
in parallel.

Companion docs:
- [`candle-removal-status-2026-05-28-pm.md`](../../candle-removal-status-2026-05-28-pm.md) — overall dashboard
- [`issue-1082-tier-4-5-roadmap-2026-05-27.md`](../../issue-1082-tier-4-5-roadmap-2026-05-27.md) — CP-2 description (parallel to CP-1)
- [`metal-cargo-toml-candle-drop-stop-2026-05-28.md`](../../metal-cargo-toml-candle-drop-stop-2026-05-28.md) — the prior scope-blocked attempt

---

## 1. Inventory — what's in `metal_types.rs` today

`crates/kiln-tensor/src/metal_types.rs` (465 LOC) exposes **5 candle
re-exports** + **1 helper function** + **5 native `Raw*` objc2-metal
aliases** (already in place, parallel) + **1 kt-native `MetalCompanion`
substrate type** (already in place, internal only).

| Line | Name | Today's substrate | Target substrate | Native `Raw*` exists? |
|---|---|---|---|---|
| 70  | `ComputePipeline`        | `candle_metal_kernels::metal::ComputePipeline` | `RawComputePipelineState` (objc2) | ✅ line 220 |
| 76  | `Library`                | `candle_metal_kernels::metal::Library` | `RawLibrary` (objc2) | ✅ line 226 |
| 87  | `BufferOffset<'a>`       | `candle_metal_kernels::utils::BufferOffset` | kt-native struct (`{ buffer: &RawBuffer, offset_in_bytes: usize }`) | partial — `RawBuffer` exists line 236; new struct needs writing |
| 108 | `MetalDevice`            | `candle_core::metal_backend::MetalDevice` | `MetalCompanion` (kt-native) | ✅ struct at line 336 |
| 114 | `DeviceId`               | `candle_core::metal_backend::DeviceId` | `u64` (via `MetalCompanion::device_id()`) | ✅ method at line 424 |
| 123 | `Storage`                | `candle_core::Storage` | `kiln_tensor::Storage` | already in kiln-tensor |
| 140 | `sdpa`                   | `candle_nn::ops::sdpa` | new `metal_sdpa_last_axis` substrate op in `metal_storage.rs` (or `kiln-mps`) | ❌ — substrate gap |
| 167 | `buffer_o(buffer, l: &candle_core::Layout, dtype: candle_core::DType)` | n/a (helper, not re-export) | new `buffer_o_kt(buffer, l: &kiln_tensor::Layout, dtype: kiln_tensor::DType)` | ❌ — needs writing |

**Already candle-free, sitting parallel for the flip** (lines 210-254):
- `RawComputePipelineState` — `Retained<ProtocolObject<dyn MTLComputePipelineState>>`
- `RawLibrary` — `Retained<ProtocolObject<dyn MTLLibrary>>`
- `RawBuffer` — `Retained<ProtocolObject<dyn MTLBuffer>>`
- `RawDevice` — `Retained<ProtocolObject<dyn MTLDevice>>`
- `RawCommandQueue` — `Retained<ProtocolObject<dyn MTLCommandQueue>>`

**Already candle-free, internal-only kt-native substrate** (lines 256-465):
- `MetalCompanion` — `{ device, kernels, commands }` triple, drop-in for `MetalDevice`
  - `MetalCompanion::device_id() -> u64` already implemented (line 424)
  - `MetalCompanion::command_encoder()`, `.kernels()`, `.device()` accessors ready

> **Net surface area:** 5 candle-typed names + 1 helper signature, 8 unique
> consumer-side sites if you count downcast patterns (`Storage::Metal(s)`),
> spread across **a single file**: `crates/kiln-model/src/backend/metal.rs`.

---

## 2. Consumer call sites (as of `139e33c5`, 2026-05-28 pm)

All consumers live in `crates/kiln-model/src/backend/metal.rs` (19523 LOC).
A single hoisted import block at line 19 funnels every reference through
one chokepoint:

```rust
use kiln_tensor::metal_types::{
    buffer_o, sdpa, ComputePipeline, DeviceId, Library, MetalDevice, Storage,
};
```

`grep -c` counts on that file:

| Symbol | Call sites in `kiln-model::backend::metal` | Other crates |
|---|---|---|
| `ComputePipeline` | 91 (pipeline-builder helpers + cache types) | 0 |
| `Library`         | 3 (the shared MSL library cache singleton) | 0 |
| `MetalDevice`     | 48 (every `metal_*_pipeline()` parameter, every helper that needs `.device()` / `.kernels()` / `.command_buffer()`) | 0 |
| `DeviceId`        | 47 (every `OnceLock<Mutex<HashMap<DeviceId, ComputePipeline>>>` cache key) | 0 |
| `Storage::Metal`  | 232 (every kernel-FFI body's `match &*storage` downcast) | 0 |
| `sdpa`            | 15 total — 3 production (lines 495, 526, 602) + 12 test/parity (lines 14849, 14919, 14938, 14947, 17928, 19487, …) | 0 |
| `buffer_o(`       | 232 (every kernel-FFI body's `BufferOffset` constructor — matches Storage::Metal site count exactly) | 0 |
| `BufferOffset`    | 0 in `metal.rs`; **9 in `metal_storage.rs`** (named explicitly via `use candle_metal_kernels::BufferOffset` per-fn) | 9 in `kiln-tensor::metal_storage` itself |

**Internal kt-tensor consumers** (already candle-free at the call site,
but lean on the candle-shipped substrate):
- `crates/kiln-tensor/src/metal_storage.rs` — 7 in-file substrate ops
  (softmax, rmsnorm, layernorm, index_select_dim0, cast, elementwise_binary,
  activation_unary). They already use `MetalCompanion` via
  `MetalStorage::companion()` (lines 301-303); they construct
  `BufferOffset` directly with `candle_metal_kernels::BufferOffset { ... }`.
- `crates/kiln-tensor/src/ops/{rmsnorm,layernorm}.rs` — minor in-crate
  consumers.

**Files referencing `candle_metal_kernels` anywhere**: 5 — all in `kiln-tensor`:
1. `metal_allocator.rs`
2. `metal_types.rs`
3. `ops/rmsnorm.rs`
4. `ops/layernorm.rs`
5. `metal_storage.rs`

**`candle_metal_kernels` references in `kiln-model::backend::metal.rs`:
0.** The chokepoint is real and effective — that crate name does not appear
on the import line.

**`candle_core` references in `kiln-model::backend::metal.rs`: 1289** —
but those are `Tensor`, `Device`, `DType`, `Layout` from the larger Tier-3
(`kiln-model` internal) migration (CP-7), not from the metal_types
chokepoint. They are out of scope for *this* doc.

---

## 3. Migration sequence

The strategy is **substrate-first, callers-second, drop-last**.

> **Key invariant for every step:** the candle re-exports in
> `metal_types.rs` stay live as type aliases until the last caller
> migrates off them. Each step compiles green and the existing
> `cargo nextest run --features metal` smoke + production
> `kiln-bench --backend metal` parity check passes.

### Step 1 — `buffer_o_kt` helper (substrate add)

- **Touches:** `crates/kiln-tensor/src/metal_types.rs` (single function add, ~20 LOC).
- **Adds:** `pub fn buffer_o_kt<'a>(buffer: &'a candle_metal_kernels::metal::Buffer, l: &kiln_tensor::Layout, dtype: kiln_tensor::DType) -> BufferOffset<'a>` — same formula (`l.start_offset() * dtype.size_in_bytes()`), kt-typed args.
- **Risk:** **Low** — additive, no caller migration.
- **Validation:** `cargo check -p kiln-tensor --features metal` on the agent's session pod; no Apple Silicon hardware needed for the type-check.
- **Why first:** every Step 5 caller migration needs this entry to land first.

### Step 2 — `metal_sdpa_last_axis` substrate op (substrate add)

- **Touches:** `crates/kiln-tensor/src/metal_storage.rs` (add new `pub fn metal_sdpa_last_axis(q, k, v, scale, causal) -> Result<Tensor>` mirror of `metal_softmax_last_axis`).
- **Risk:** **Med** — new MSL kernel dispatch (or `candle_metal_kernels::call_sdpa_*` wrapper) — must be parity-tested against `candle_nn::ops::sdpa` on an Apple Silicon pod.
- **Validation:**
  - `cargo check -p kiln-tensor --features metal` — type-checks anywhere
  - Apple Silicon (M-series local or RunPod Mac): new unit test in `metal_storage.rs` comparing `metal_sdpa_last_axis` vs `candle_nn::ops::sdpa` on a 128×64×128 BF16 tensor triple. Max-abs ≤ 1e-3.
- **Why second:** this is the only substrate gap — the other 7 chokepoint flips are pure caller migrations.
- **Dependency:** wraps the same MSL kernel `candle_nn::ops::sdpa` dispatches into (which itself is a `candle_metal_kernels::call_sdpa_*` call); the wire-level FFI is identical and bit-exact. The op only adds the kt-typed signature.

### Step 3 — Hoist `MetalCompanion::companion()` accessor from `MetalStorage` onto `MetalDevice` re-export migration target (substrate prep)

- **Touches:** none. **No-op** — `MetalStorage::companion()` already returns `Arc<MetalCompanion>` (line 301). Step exists only to make the dependency on this accessor explicit in the sequence.
- **Verify:** `MetalCompanion::device_id() -> u64` is already implemented (line 424); `MetalCompanion::command_encoder()`, `.kernels()`, `.device()` are all ready.

### Step 4 — Storage downcast migration (`Storage::Metal(s) => s` → kt-side)

- **Touches:** `crates/kiln-model/src/backend/metal.rs` (232 sites).
- **Shape per site:**
  ```rust
  // before
  let (x_storage, x_layout) = x.storage_and_layout();
  let x_metal = match &*x_storage {
      Storage::Metal(s) => s,
      _ => anyhow::bail!("..."),
  };
  let x_buf = buffer_o(x_metal.buffer(), &x_layout, x.dtype());

  // after
  let x_kt: &kiln_tensor::Tensor = /* already candle-typed today;
                                      requires per-call-site kt borrow */;
  let x_storage = x_kt.storage();
  let x_metal = x_storage.as_any()
      .downcast_ref::<kiln_tensor::MetalStorage>()
      .ok_or_else(|| anyhow::anyhow!("..."))?;
  let x_buf = buffer_o_kt(x_metal.buffer(), x_kt.layout(), x_kt.dtype());
  ```
- **Risk:** **High** — touches the largest call-site count (232 × 2 patterns = 464 individual edits). Coordinating with the in-flight `kt_metal_X` migration (Wave 13, see commit `7b0e9dbd`) is mandatory; cherry-pick by helper-family to avoid merge collisions.
- **Recommended decomposition:** one PR per helper family, smallest first:
  1. embedding (~5 sites)
  2. lm_head (~3 sites)
  3. rmsnorm (~10 sites)
  4. rotary_qk (~8 sites)
  5. mlp (~15 sites)
  6. gdn family (~50 sites)
  7. conv1d (~20 sites)
  8. paged_kv / paged_attn (~40 sites)
  9. gemv / matmul (~80 sites — last and largest)
- **Validation per PR:** `cargo nextest run -p kiln-model --features metal` plus `kiln-bench --backend metal --paged --prompt-tokens 512 --max-output-tokens 128` (median of 3, accept-rate ≥ baseline). Apple Silicon hardware required.
- **Dependency:** **Step 1** (`buffer_o_kt`) must land first.

### Step 5 — `&MetalDevice` → `&MetalCompanion` helper-signature swap

- **Touches:** `crates/kiln-model/src/backend/metal.rs` (~48 helper signatures + 47 cache key sites).
- **Shape:**
  ```rust
  // before
  fn metal_rms_norm_pipeline(device: &MetalDevice) -> Result<ComputePipeline> {
      static PIPELINES: OnceLock<Mutex<HashMap<DeviceId, ComputePipeline>>> = OnceLock::new();
      if let Some(p) = cache.get(&device.id()) { return Ok(p.clone()); }
      let library = metal_shared_library(device)?;
      let function = library.get_function("kiln_rmsnorm_bf16", None)?;
      let pipeline = device.device().new_compute_pipeline_state_with_function(&function)?;
      cache.insert(device.id(), pipeline.clone());
      Ok(pipeline)
  }

  // after
  fn metal_rms_norm_pipeline(companion: &MetalCompanion) -> Result<ComputePipeline> {
      static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
      if let Some(p) = cache.get(&companion.device_id()) { return Ok(p.clone()); }
      let library = metal_shared_library(companion)?;
      let function = library.get_function("kiln_rmsnorm_bf16", None)?;
      let pipeline = companion.device().new_compute_pipeline_state_with_function(&function)?;
      cache.insert(companion.device_id(), pipeline.clone());
      Ok(pipeline)
  }
  ```
- **Risk:** **Med** — mechanical but volume-intensive; mid-sequence merge collisions with Step 4 are likely. Sequence Steps 4 and 5 carefully or interleave per helper family.
- **Recommended decomposition:** Same family-based splitting as Step 4; can land each family's Step 4 + Step 5 in a single PR to halve coordination cost.
- **Validation:** same as Step 4.
- **Dependency:** Step 3 (already done) provides the substrate; Step 4 should land first per-family so the body migrations chain naturally.

### Step 6 — `sdpa` call-site migration (15 sites total)

- **Touches:** `crates/kiln-model/src/backend/metal.rs` — 3 production sites (lines 495, 526, 602) + 12 test sites.
- **Shape per site:** `sdpa(q, k, v, None, causal, scale, 1.0)` → `kiln_tensor::metal_sdpa_last_axis(q_kt, k_kt, v_kt, scale, causal)`. Production sites already prepare contiguous `q_t`/`k_t`/`v_t` candle tensors; the migration also kt-borrows them.
- **Risk:** **Med** — small surface but kernel-level parity must hold across the 3 production sites (prefill SDPA, head-major SDPA, paged SDPA). The substrate change is bit-exact-by-construction (Step 2 wraps the same MSL kernel), so the risk is in the call-site borrow plumbing, not the kernel.
- **Validation:**
  - `cargo nextest run -p kiln-model --features metal sdpa` (the existing parity tests).
  - `kiln-bench --backend metal` — SDPA tok/s within ±2% of pre-flip baseline.
- **Dependency:** Step 2 (`metal_sdpa_last_axis` substrate).

### Step 7 — `Library` migration (3 sites — the shared library cache)

- **Touches:** `crates/kiln-model/src/backend/metal.rs` lines 5741, 5743, 5784.
- **Shape:** `Library` return type stays as-is (it's already the right shape; the question is whether to keep the `candle_metal_kernels::metal::Library` re-export or flip to `objc2_metal::MTLLibrary` directly). Since `Library` is `Retained<ProtocolObject<dyn MTLLibrary>>` under the hood (the same as `RawLibrary`), the chokepoint flips by changing line 76 of `metal_types.rs`:
  ```rust
  // before
  pub use candle_metal_kernels::metal::Library;

  // after
  pub type Library = RawLibrary;
  ```
  Call sites compile unchanged (the wrapping `Retained<ProtocolObject<...>>` is identical), but `device.device().new_library_with_source(...)` becomes `companion.device().new_library_with_source(...)` (already covered by Step 5 device-handle migration).
- **Risk:** **Low** — single-line `pub use` → `pub type` swap; callers compile unchanged.
- **Validation:** `cargo check -p kiln-model --features metal`.
- **Dependency:** Step 5 (so `device.device()` → `companion.device()` is already done at the 3 sites).

### Step 8 — `ComputePipeline` migration (91 sites — pure rename)

- **Touches:** `crates/kiln-tensor/src/metal_types.rs` (1-line swap).
- **Shape:**
  ```rust
  // before
  pub use candle_metal_kernels::metal::ComputePipeline;

  // after
  pub type ComputePipeline = RawComputePipelineState;
  ```
- **Risk:** **Low** — `RawComputePipelineState` is `Retained<ProtocolObject<dyn MTLComputePipelineState>>`, the same type `candle_metal_kernels::metal::ComputePipeline` is internally. Callers compile unchanged because `.clone()` and the `.set_compute_pipeline_state(...)` encoder method are protocol-object inherent methods on the Retained handle.
- **Validation:** `cargo check -p kiln-model --features metal` + a smoke test of one pipeline-builder per family.
- **Dependency:** Step 5 (helper signatures all migrated to `&MetalCompanion` — so the candle wrapper layer is fully off the helper bodies).

### Step 9 — `BufferOffset` flip + kt-native struct

- **Touches:** `crates/kiln-tensor/src/metal_types.rs`.
- **Shape:** Either:
  - **Option A — `pub type` flip**: keep the field layout, swap the `&Buffer` field to `&RawBuffer`. This is bit-equivalent because `candle_metal_kernels::metal::Buffer` is a `Retained<ProtocolObject<dyn MTLBuffer>>` newtype wrapper.
  - **Option B — kt-native struct**: define `pub struct BufferOffset<'a> { pub buffer: &'a RawBuffer, pub offset_in_bytes: usize }`. Slightly more invasive but visually flat — no candle name anywhere.
- **Recommended:** Option B (it's 4 lines of code and removes the last `candle_metal_kernels::utils` import).
- **Risk:** **Med** — every `BufferOffset { buffer: ..., offset_in_bytes: ... }` literal in `metal_storage.rs` (9 sites) needs the `buffer:` field type to align with the new struct. Mechanical.
- **Validation:** `cargo check -p kiln-tensor --features metal` then `cargo check -p kiln-model --features metal`.
- **Dependency:** none — can land any time after Step 1.

### Step 10 — `MetalDevice` + `DeviceId` re-export deletion

- **Touches:** `crates/kiln-tensor/src/metal_types.rs` (delete lines 108, 114).
- **Risk:** **Low** — by this point every caller has migrated to `&MetalCompanion` / `u64` (via Steps 4-5).
- **Validation:** `cargo check -p kiln-model --features metal` succeeds; the `use kiln_tensor::metal_types::{...}` import block in `metal.rs` no longer names `MetalDevice` or `DeviceId`.
- **Dependency:** Steps 4 + 5 fully landed.

### Step 11 — `Storage` re-export deletion

- **Touches:** `crates/kiln-tensor/src/metal_types.rs` (delete line 123).
- **Risk:** **Low** — by this point every caller has migrated to kt-side `Storage` downcast (Step 4).
- **Validation:** `cargo check -p kiln-model --features metal`.
- **Dependency:** Step 4 fully landed.

### Step 12 — `sdpa` re-export deletion

- **Touches:** `crates/kiln-tensor/src/metal_types.rs` (delete line 140).
- **Risk:** **Low** — Step 6 migrated all 15 callers.
- **Dependency:** Step 6 fully landed.

### Step 13 — `buffer_o` (candle-typed signature) deletion

- **Touches:** `crates/kiln-tensor/src/metal_types.rs` (delete lines 167-176).
- **Risk:** **Low** — Step 4 migrated all 232 callers to `buffer_o_kt`.
- **Dependency:** Step 4 fully landed.

### Step 14 — `Cargo.toml` `candle-core` / `candle-nn` drop from `metal` feature

- **Touches:** `crates/kiln-tensor/Cargo.toml`.
- **Shape:**
  ```toml
  # before
  metal = ["dep:candle-core", "candle-core?/metal", "dep:candle-metal-kernels",
           "dep:candle-nn", "candle-nn?/metal", "dep:objc2", "dep:objc2-metal"]

  # after
  metal = ["dep:candle-metal-kernels", "dep:objc2", "dep:objc2-metal"]
  ```
  Plus the corresponding `[dependencies]` line removals.
- **Risk:** **Low** if all prior steps have landed; `cargo tree -p kiln-tensor --features metal -i candle-core` should show an empty edge set before this flip.
- **Validation:**
  - `cargo build -p kiln-tensor --features metal --no-default-features` succeeds.
  - `cargo build -p kiln-model --features metal --no-default-features` succeeds.
  - `cargo tree --workspace -i candle-core` no longer lists `kiln-tensor` or `kiln-model` (Metal-only) as candle-core consumers under the metal feature.
- **Dependency:** Steps 4-13 all landed.

### Step 15 — `candle-metal-kernels` itself — out of scope for THIS doc

`candle-metal-kernels` is a *sibling* of candle-core, not part of the
candle-vendored tree. It's a thin wrapper around MSL JIT compilation +
MSL kernel dispatch; dropping it requires a kt-native MSL kernel cache
(`kiln-mps` will eventually hold this). That's covered by Phase 2.x of
`kiln-mps` (see `crates/kiln-mps/src/lib.rs` head-of-file comment) and
is out of scope for the candle-core/candle-nn drop this doc tracks.

---

## 4. Dependency gates

### What other workstreams must land before each step?

| Step | Cross-workstream gate | Status |
|---|---|---|
| 1 (buffer_o_kt) | none | ready |
| 2 (metal_sdpa_last_axis) | none — substrate is local to `kiln-tensor::metal_storage` | ready |
| 3 (companion accessor) | done — landed in commit `a07e7781` | ✅ |
| 4 (Storage downcast) | **Wave 13 `kt_metal_X` migration** must be coordinated to avoid merge collisions on the same helper bodies. See commit `1b7f6b80`, `7b0e9dbd`. Per-family PRs minimize collision risk. | partially in flight |
| 5 (`&MetalCompanion` signature) | same as Step 4 — Wave 13 coordination | partially in flight |
| 6 (sdpa callers) | Step 2 substrate landed | gated on Step 2 |
| 7-13 (chokepoint flips + deletions) | Steps 4-6 caller migrations complete | gated on Steps 4-6 |
| 14 (Cargo.toml drop) | Steps 7-13 complete | gated |

### What CP-* item is this work?

This is **CP-2** in the `issue-1082-tier-4-5-roadmap-2026-05-27.md`
critical-path table. Note:
- The "CP-2 (`kiln-tensor::MetalStorage` candle removal)" sub-item is
  **complete** (Wave 14, commits `3f00a979`, `a07e7781`, `ae08652c`,
  `d8d43c6d`). MetalStorage and MetalAllocator no longer touch candle-core
  field types.
- This doc covers the **second half of CP-2** — the chokepoint-flip side
  that the Wave 14 STOP doc identified as out-of-scope.

### Independence from CP-4 (kt-native autograd)

Per the dashboard, **CP-4 (kt-tape adoption in production training) is
independent of this work**. The metal_types swap is inference-path-only
substrate; CP-4 is training-loop autograd substrate. They can ship in
parallel.

### Independence from CP-1 (CUDA `Arc<CudaContext>` migration)

CP-1 is the CUDA twin of this work. They are independent — neither
substrate touches the other's storage type. They can ship in parallel
(and the roadmap explicitly recommends doing so).

### Independence from `kiln-mps` Tier-2

`kiln-mps` is the eventual home for a kt-native MSL kernel cache. The
substrate gap for `sdpa` (Step 2) could route through `kiln-mps` instead
of adding `metal_sdpa_last_axis` to `metal_storage.rs`. **Recommended
path: ship Step 2 in `metal_storage.rs` first** (mirrors `metal_softmax_last_axis`,
~50 LOC). When `kiln-mps` Phase 2.x lands, the implementation can be
delegated; the public surface (`kiln_tensor::metal_sdpa_last_axis`) stays
constant.

---

## 5. The Tier-5 endgame

### Once `metal_types.rs` is candle-free, does anything else need to land for Tier-5?

**For the `metal` feature path: no.** Step 14 above completes the
`candle-core` + `candle-nn` drop from `kiln-tensor`'s `metal` feature.
After it lands:

- `cargo tree -p kiln-tensor --features metal -i candle-core` → empty
- `cargo tree -p kiln-model --features metal -i candle-core` → still
  shows references from the larger Tier-3 candle migration (`Tensor`,
  `Device`, `DType` — CP-7 territory), NOT from `metal_types`.

### What still pulls `candle-core` workspace-wide after this lands?

Per the dashboard:
- `kiln-flce-kernel` (blocked on CP-4)
- `kiln-kt-bridge` (by-design — Tier-4 deletion target)
- `kiln-model` (this work covers the `metal` chokepoint; the broader
  `forward.rs` candle migration is CP-7)
- `kiln-opd-loss-kernel` (blocked on CP-4)
- `kiln-rmsnorm-kernel` (blocked on CP-4)
- `kiln-train` (one `CustomOp1` impl — CP-4)

So this work **does not by itself close Tier-5**. It closes one of two
gating items (alongside CP-4) named in the dashboard's "Two parallel
architectural pieces" section. Tier-5 requires both this AND CP-4 to
land.

### Should this update the existing STOP doc?

When **Step 14 lands**, the existing STOP doc
[`metal-cargo-toml-candle-drop-stop-2026-05-28.md`](../../metal-cargo-toml-candle-drop-stop-2026-05-28.md)
should get a closing entry like:

```markdown
## 2026-MM-DD — STOP resolved
The chokepoint flip described in `metal-types-objc2-swap-plan-2026-05-28.md`
landed in PRs #NNNN-#NNNN. The Cargo.toml drop blocked by this STOP doc
landed in PR #NNNN. This STOP doc is closed.
```

The dashboard row in
[`candle-removal-status-2026-05-28-pm.md`](../../candle-removal-status-2026-05-28-pm.md)
also flips from `kiln-tensor: Yes (metal_types re-exports)` to `No` (still
`Yes` if CP-1 hasn't landed for the CUDA side).

---

## 6. Recommended first PR

**Step 1 — add `buffer_o_kt` helper.**

Rationale:
1. **No pod cost.** Type-checks on any host with `cargo check -p kiln-tensor --features metal`.
2. **Smallest unit.** ~20 LOC additive change in a single file. Reviewable in 2 minutes.
3. **Unblocks the largest step (Step 4).** With `buffer_o_kt` available,
   the 232 caller migrations can begin in per-family PRs immediately.
4. **No CP-4, no kt-tape gate, no merge-collision risk.** Pure substrate add.

Suggested PR shell:

```bash
git checkout -b add-buffer-o-kt-for-metal-types-chokepoint
# Edit crates/kiln-tensor/src/metal_types.rs: add buffer_o_kt below
# buffer_o (lines 142-176).
cargo check -p kiln-tensor --features metal
git add crates/kiln-tensor/src/metal_types.rs
git commit -m "kiln-tensor: add buffer_o_kt helper for metal_types chokepoint (#1082)"
git push origin add-buffer-o-kt-for-metal-types-chokepoint
gh pr create --repo ericflo/kiln --title "kiln-tensor: add buffer_o_kt helper for metal_types chokepoint" \
  --body "Refs #1082. Step 1 of the metal_types -> objc2-metal swap sequence documented in docs/metal-types-objc2-swap-plan-2026-05-28.md."
gh pr merge $NUM --squash --auto
```

**Recommended second PR:** Step 2 (`metal_sdpa_last_axis` substrate op).
This DOES require Apple Silicon pod time for the parity test (~30 min on
M2/M3). The implementation body is a copy-paste of `metal_softmax_last_axis`
with the kernel name swapped. After Step 1 + Step 2 land, the 9 caller-
migration PRs (Steps 4-6 decomposed by helper family) can ship in parallel
since they touch disjoint helper families in `metal.rs`.

---

## 7. Why this doc supersedes the prior STOP

The earlier STOP doc
[`metal-cargo-toml-candle-drop-stop-2026-05-28.md`](../../metal-cargo-toml-candle-drop-stop-2026-05-28.md)
was scope-blocked: the task that produced it constrained edits to
`kiln-tensor` only. That STOP correctly identified the 5 chokepoint
re-exports as the blocker; this doc completes the analysis by:

1. Confirming the substrate-readiness counts (all `Raw*` and
   `MetalCompanion` are in place).
2. Sequencing the per-family caller migrations to avoid merge collisions
   with Wave 13's in-flight `kt_metal_X` work.
3. Naming the single substrate gap (`sdpa`) and a concrete recipe
   (mirror `metal_softmax_last_axis`).
4. Splitting the 232-site `Storage::Metal` downcast migration by helper
   family so each PR is independently reviewable and bench-able.
5. Identifying Step 1 (`buffer_o_kt`) as the smallest, lowest-risk,
   no-pod-cost first PR.

The STOP doc remains valuable as a historical "what-was-out-of-scope"
record; this doc is the forward plan.

---

## See also

- [`candle-removal-status-2026-05-28-pm.md`](../../candle-removal-status-2026-05-28-pm.md) — dashboard
- [`issue-1082-tier-4-5-roadmap-2026-05-27.md`](../../issue-1082-tier-4-5-roadmap-2026-05-27.md) — CP-2 description
- [`metal-cargo-toml-candle-drop-stop-2026-05-28.md`](../../metal-cargo-toml-candle-drop-stop-2026-05-28.md) — predecessor STOP doc
- `crates/kiln-tensor/src/metal_types.rs:47-62` — in-source migration notes (the `Raw*` substrate's intent)
- `crates/kiln-tensor/src/metal_storage.rs:455-560` — `metal_softmax_last_axis` (the implementation template for Step 2)
- Commits `3f00a979`, `a07e7781`, `ae08652c`, `d8d43c6d`, `56bdaffd` — the Wave 14 substrate-prep landings this doc builds on.
