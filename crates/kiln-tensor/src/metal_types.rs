//! Candle-free type aliases for the metal-rs–style types used by every
//! kernel-crate FFI site in `kiln-model::backend::metal`.
//!
//! # Why this module exists (#1082 Phase 7)
//!
//! Roughly all `candle_metal_kernels::*` references in
//! `kiln-model::backend::metal` are simple uses of two type names —
//! `ComputePipeline` (the pipeline-state object) and `Library` (the
//! compiled MSL library handle). Each call site looks like:
//!
//! ```text
//! fn metal_X_pipeline(
//!     device: &candle_core::metal_backend::MetalDevice,
//! ) -> Result<candle_metal_kernels::metal::ComputePipeline> {
//!     use candle_metal_kernels::metal::ComputePipeline;
//!     ...
//! }
//! ```
//!
//! Funneling those through one re-export consolidates the
//! `candle_metal_kernels::*` surface area to a single chokepoint that
//! lives in `kiln-tensor`. Once every caller imports
//! `kiln_tensor::metal_types::{ComputePipeline, Library}`, the
//! underlying substrate can swap (e.g. to a direct `objc2_metal::*`
//! protocol-object pair, or to a non-candle `metal-rs` wrapper) without
//! touching the ~92 kernel-helper functions in `metal.rs`.
//!
//! The wire types are identical to candle's re-export — they ARE the
//! same `candle-metal-kernels::metal::{ComputePipeline, Library}`
//! types, just reached through a `kiln_tensor::` path so that the
//! kernel helpers no longer name `candle_metal_kernels` in their
//! signatures or bodies. This is the same bookkeeping pattern that
//! `kiln-tensor::metal_storage::metal_device_handle()` uses on the
//! storage side: keep the candle-shipped substrate in place for now
//! while the call-site footprint drops to zero.
//!
//! # Phase 7 follow-up
//!
//! When the kernel crates migrate off candle-metal-kernels'
//! pipeline/library abstractions (e.g. to objc2_metal directly), this
//! module flips its `pub use candle_metal_kernels::metal::*` line to
//! point at the new substrate. The metal.rs migration that converts
//! the existing 92 sites to `kiln_tensor::metal_types::*` is the
//! enabling step — until those imports are all centralized here, the
//! substrate swap must touch every helper individually.
//!
//! # Native objc2-metal substrate (#1082 Phase 7 lift, partial)
//!
//! The `Raw*` aliases below expose the direct `objc2_metal::MTL*`
//! protocol-object handles that sit underneath every
//! `candle_metal_kernels::metal::*` wrapper. They are the substrate
//! the chokepoint will eventually flip its existing aliases to point at
//! — adding them now (parallel to the candle re-exports) lets future
//! call-site migrations land on a kt-native objc2-metal path without
//! re-introducing a candle string anywhere on the import line. The
//! `Retained<ProtocolObject<dyn MTL*>>` shape is the same one
//! `candle_metal_kernels::metal::{ComputePipeline, Library, Buffer,
//! Device}` hold internally (via `.raw` fields backed by the same
//! `objc2-metal 0.3` crate), so passing a `Raw*` to `candle_metal_kernels::call_*`
//! works as soon as the candle wrapper exposes `From<Raw*>` conversions
//! (already in place for `Device` and `Buffer`; pipeline/library wrap
//! the same Retained internally).

#![cfg(feature = "metal")]

/// MSL compute-pipeline-state handle. Re-exported from candle's
/// in-house objc2 wrapper today; flips to a direct `objc2_metal`
/// `Retained<ProtocolObject<dyn MTLComputePipelineState>>` when the
/// kernel crates retire the candle bridge.
pub use candle_metal_kernels::metal::ComputePipeline;

/// MSL library handle. Re-exported from candle's in-house objc2
/// wrapper today; flips to a direct `objc2_metal`
/// `Retained<ProtocolObject<dyn MTLLibrary>>` when the kernel crates
/// retire the candle bridge.
pub use candle_metal_kernels::metal::Library;

/// `BufferOffset` — `{ buffer: &Buffer, offset_in_bytes: usize }`
/// pair consumed by every `candle_metal_kernels::call_*` MSL kernel
/// entry point as a positional argument.
///
/// Re-exported from `candle_metal_kernels::utils::BufferOffset` today;
/// when the kernel crates retire the candle bridge, this flips to a
/// kt-native struct holding `(Retained<ProtocolObject<dyn MTLBuffer>>,
/// usize)` without touching the ~232 call sites in
/// `kiln-model::backend::metal` that construct one.
pub use candle_metal_kernels::utils::BufferOffset;

/// Candle `MetalDevice` re-export — used by every pipeline-builder
/// helper in `kiln-model::backend::metal` as a function parameter
/// (`device: &kiln_tensor::metal_types::MetalDevice`).
///
/// The wire type is candle's `MetalDevice` wrapper; the kt-side path
/// lets the model crate stop naming `candle_core::metal_backend::*`
/// in its helper signatures so the eventual substrate lift to a raw
/// `MTLDevice` + `MTLCommandQueue` handle pair is a single edit
/// here, not 46 separate signature edits in `metal.rs`.
///
/// # Phase 7 follow-up
///
/// Substrate lift retires the candle wrapper itself — this re-export
/// then flips to a `kiln_tensor::MetalRawDevice` companion that holds
/// `(Retained<ProtocolObject<dyn MTLDevice>>,
///   Retained<ProtocolObject<dyn MTLCommandQueue>>)` directly. The
/// helper-function bodies (which only call `device.id()`,
/// `device.device()`, `device.command_buffer()`, etc.) get migrated
/// alongside that flip.
pub use candle_core::metal_backend::MetalDevice;

/// The raw `candle_metal_kernels` Metal device — the candle-CORE-free
/// substrate handle every MSL pipeline / library build and every
/// `call_*` kernel entry actually consumes. `MetalCompanion::device()`
/// returns `&MetalRawDevice`; candle's `MetalDevice::device()` does too.
///
/// (#1082) The `metal_*_pipeline` helpers in `kiln-model::backend::metal`
/// migrate their `device.id()` / `device.device()` candle calls onto a
/// `&dyn MetalPipelineHost` parameter whose `pipeline_raw_device()`
/// returns this type — so the same getter serves both a candle
/// `MetalDevice` caller (during migration) and a kt `MetalCompanion`
/// caller (the candle-free end state) with no per-call-site change.
pub use candle_metal_kernels::metal::Device as MetalRawDevice;

/// Candle `DeviceId` re-export — the `usize` newtype used as the cache
/// key in every per-device pipeline / library `HashMap` in the
/// `metal_*_pipeline` helpers. Same Phase-7 rationale as `MetalDevice`:
/// keep the wire type identical, move the path through `kiln_tensor`.
pub use candle_core::metal_backend::DeviceId;

/// Candle `Storage` enum re-export — pattern-matched in every kernel
/// helper to downcast `Storage::Metal(s) => s` and extract the
/// candle `MetalStorage` for the `.buffer()` / `.dtype()` accessors.
/// Phase 7 substrate-lift replaces the match arm with a direct
/// `kiln_tensor::MetalStorage` downcast off `kiln_tensor::Storage`
/// (already in flight on the kt-side); the chokepoint here is the
/// path-naming bookkeeping step.
pub use candle_core::Storage;

// `D` re-export retired (#1082): the production `kiln-model::backend::metal`
// call sites now compute the last-axis index via kt-native `tensor.rank() - 1`
// arithmetic, and the test helpers in that file import `candle_core::D`
// directly since they operate on `candle_core::Tensor` already.

/// Candle `sdpa` re-export — the MLX-style fused scaled-dot-product
/// attention kernel shipped in `candle_nn::ops::sdpa`. Used in 9
/// callsites across `kiln-model::backend::metal` (3 in the metal
/// SDPA dispatch + 6 in test/parity helpers).
///
/// Same Phase-7 chokepoint rationale as the other re-exports: keep
/// the wire signature identical, move the path through `kiln_tensor`
/// so callers stop naming `candle_nn::ops::sdpa` directly. The
/// substrate-lift step replaces this re-export with a kt-native
/// fused SDPA op that takes `kiln_tensor::Tensor` arguments.
pub use candle_nn::ops::sdpa;

/// Build a `BufferOffset` from a candle `Buffer` + `kiln_tensor::Layout`
/// + `kiln_tensor::DType`.
///
/// This is the kt-typed chokepoint entry point for byte-offset
/// computation: callers in `kiln-model::backend::metal` reach it
/// through the `kiln_tensor` path so their bodies no longer name
/// `candle_core::metal_backend::*`. The formula is the candle-bundled
/// one (`l.start_offset() * dtype.size_in_bytes()`); inlining it here
/// gives the same byte offset that `candle_metal_kernels::call_*`
/// kernels expect, while keeping the chokepoint free to evolve (a
/// `pub use` of `candle_core::metal_backend::buffer_o` would lock the
/// chokepoint to candle-core's API surface forever).
///
/// As of the Step-4 caller migration (swap plan
/// `docs/metal-types-objc2-swap-plan-2026-05-28.md`), all 232 metal.rs
/// call sites name this kt-typed helper; the legacy candle-typed
/// `buffer_o` has been retired.
///
/// The `buffer` argument still names
/// `candle_metal_kernels::metal::Buffer` because that's the type the
/// downstream `candle_metal_kernels::call_*` MSL kernels expect today.
/// On Apple Silicon UMA, that Buffer wrapper holds the same
/// `Retained<ProtocolObject<dyn MTLBuffer>>` that [`RawBuffer`]
/// aliases — they're bit-identical and the swap is a renaming, not
/// a re-allocation. The buffer-arg flip is Step 6 of the plan.
///
/// # (#1082)
#[inline]
pub fn buffer_o_kt<'a>(
    buffer: &'a candle_metal_kernels::metal::Buffer,
    l: &crate::Layout,
    dtype: crate::DType,
) -> BufferOffset<'a> {
    BufferOffset {
        buffer,
        offset_in_bytes: l.start_offset() * dtype.size_in_bytes(),
    }
}

// ----------------------------------------------------------------------
// Native objc2-metal substrate (#1082 Phase 7 — partial lift)
// ----------------------------------------------------------------------
//
// Every `candle_metal_kernels::metal::*` wrapper above ultimately holds
// a `Retained<ProtocolObject<dyn MTL*>>` from the `objc2-metal 0.3`
// crate. The `Raw*` aliases below expose those handles directly —
// they are the substrate the existing chokepoint aliases (`ComputePipeline`,
// `Library`, `MetalDevice`, etc.) will eventually flip to point at, once
// the kernel-crate FFI sites in `kiln-model::backend::metal` migrate off
// the candle wrapper signatures (`device.command_buffer()`,
// `device.kernels()`, `pipeline.set_compute_pipeline_state(...)`).
//
// Adding them here in parallel with the candle re-exports above does not
// break any existing call site — the candle aliases stay live — and gives
// future migration commits a kt-native objc2-metal path to land on
// without re-introducing a `candle_metal_kernels::metal::*` string
// anywhere on the import line. The eventual chokepoint flip replaces
// `pub use candle_metal_kernels::metal::ComputePipeline` with
// `pub type ComputePipeline = RawComputePipelineState`, etc.
//
// # Why the objc2-metal version is pinned to candle's
//
// `candle-metal-kernels 0.10.2` depends on `objc2-metal = "0.3.2"`.
// kiln-tensor's `metal` feature pulls the same `objc2-metal = "0.3"`,
// so cargo unifies on one version and the `Retained<ProtocolObject<dyn
// MTL*>>` types these aliases name are exactly the ones candle's
// wrappers hold internally. That bit-identity is what lets a future
// caller pass a `RawComputePipelineState` straight to
// `candle_metal_kernels::call_*` without a `From<>` shim — the wire
// type that crosses the FFI is the same Retained handle either way.

/// MSL compute-pipeline-state handle as a raw `objc2-metal` protocol
/// object. The eventual substrate for `ComputePipeline`.
///
/// This is the same Retained handle that
/// `candle_metal_kernels::metal::ComputePipeline` holds internally,
/// just reached without the candle wrapper layer. Future kernel-helper
/// migrations can pivot away from `pipeline.set_compute_pipeline_state(...)`
/// (a candle wrapper method) onto direct
/// `MTLComputeCommandEncoder::setComputePipelineState_` calls against
/// the `Retained` handle.
pub type RawComputePipelineState = objc2::rc::Retained<
    objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
>;

/// MSL library handle as a raw `objc2-metal` protocol object. The
/// eventual substrate for `Library`.
pub type RawLibrary =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLLibrary>>;

/// MTLBuffer handle as a raw `objc2-metal` protocol object. The
/// eventual substrate the existing `candle_metal_kernels::metal::Buffer`
/// wrapper flips to point at.
///
/// On Apple Silicon UMA this is the same Retained handle that backs
/// every `MetalStorage::buffer()` accessor today — flipping the
/// chokepoint here is a renaming, not a re-allocation.
pub type RawBuffer =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>;

/// MTLDevice handle as a raw `objc2-metal` protocol object. The
/// eventual substrate for `MetalDevice` once the kernel-helper
/// signatures retire the candle wrapper's `.command_buffer()` /
/// `.kernels()` accessors in favor of direct
/// `MTLCommandQueue::commandBuffer` + a kt-side `Kernels` cache.
pub type RawDevice =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>;

/// MTLCommandQueue handle as a raw `objc2-metal` protocol object.
/// The eventual companion that lands on `MetalStorage` alongside
/// `metal_handle: MetalRawDevice` so that the in-storage substrate
/// ops in `metal_storage.rs` can build `MTLCommandBuffer`s without
/// derivating a candle `MetalDevice` per call (the
/// `primary_metal_device` shim retires once this lands).
pub type RawCommandQueue =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandQueue>>;

// ----------------------------------------------------------------------
// MetalCompanion — kt-native substrate for the 7 in-file substrate ops
// (#1082 Phase 7 — Wave 14 lift)
// ----------------------------------------------------------------------
//
// `MetalCompanion` collects the substrate primitives the 7 internal
// substrate ops in `metal_storage.rs` currently reach for through a
// derived candle `MetalDevice` wrapper:
//   - `.metal_device()`   -> the `&Device` for `candle_metal_kernels::call_*`
//   - `.kernels()`        -> the `&Kernels` MSL pipeline-state cache
//   - `.command_encoder()` -> a `ComputeCommandEncoder` from a pool
//
// All three primitives live in `candle_metal_kernels` itself (not
// `candle-core`) — `Kernels::new()`, `Commands::new(command_queue)`,
// and the Device's `new_command_queue()` are candle-core-free
// constructors. Holding them on a kt-native struct lets the 7 ops
// dispatch through `candle_metal_kernels::call_*` without ever
// materializing a candle `MetalDevice` per call (the
// `primary_metal_device` shim and its `candle_core::Device::new_metal`
// call site retire once the ops migrate).
//
// # Wave-14 status
//
// This commit adds the type only — no callers yet. The 7 substrate ops
// still derive a candle `MetalDevice` per call via the existing
// `MetalStorage::candle_device()` back-compat shim. A follow-up commit
// adds the `MetalStorage::companion()` accessor + a per-device-index
// `OnceLock<HashMap>` cache (parallel to `primary_metal_device`); a
// subsequent commit migrates the ops + drops the candle-core hook from
// `metal_storage.rs`.
//
// # Why this is the right substrate
//
// `candle_metal_kernels::call_*` consumes:
//   - `device: &Device` — the candle-metal-kernels Device wrapper
//   - `ep: impl EncoderProvider` — satisfied by `&ComputeCommandEncoder`
//   - `kernels: &Kernels` — pipeline cache
//
// Every primitive on the right is in `candle_metal_kernels`, not
// `candle-core`. `MetalCompanion` simply holds owned/`Arc`-ed copies
// of each so the ops can read `&Device` / `&Kernels` / construct a
// fresh `ComputeCommandEncoder` per call without going through candle.
//
// The eventual Cargo.toml `candle-core` drop is blocked by the
// `MetalDevice` / `DeviceId` / `Storage` / `sdpa` re-exports above
// (consumed at ~48 callsites in `kiln-model::backend::metal`), not by
// the in-file ops — those become candle-free as soon as `MetalCompanion`
// + accessor land.

/// Kt-native substrate for the 7 in-file Metal substrate ops in
/// `metal_storage.rs`. Holds the candle-core-free primitives every
/// `candle_metal_kernels::call_*` invocation needs:
///   - A `candle_metal_kernels::metal::Device` for the `&Device` parameter
///   - An `Arc<candle_metal_kernels::Kernels>` MSL pipeline cache
///   - An `Arc<RwLock<candle_metal_kernels::metal::Commands>>` command-
///     buffer pool for `ComputeCommandEncoder` materialization
///
/// Constructed via [`MetalCompanion::from_raw`] from a
/// `candle_metal_kernels::metal::Device` (which is itself just a
/// thin wrapper around `Retained<ProtocolObject<dyn MTLDevice>>` —
/// candle-core is not involved at any step). Mirror in spirit of
/// candle's `MetalDevice` struct (`vendor/candle-core/src/metal_backend/device.rs`)
/// but stripped of every non-substrate field (buffer pools, random
/// seed, DeviceId — none of which the 7 ops touch).
///
/// # Why fields are `Arc`-shared
///
/// `Kernels` and `Commands` are designed for cross-thread shared use
/// (candle wraps them in `Arc` for the same reason). `Device` is
/// internally `Retained<ProtocolObject<...>>` — cheap to clone via
/// NSObject `retain` — so it's held by value.
///
/// # Phase 7 follow-up
///
/// When `kiln-mps` lands its kt-native MSL kernel cache, this companion
/// flips its `kernels: Arc<Kernels>` field from
/// `candle_metal_kernels::Kernels` to a kt-native equivalent without
/// touching any caller. Same for `commands` if a kt-native command-
/// buffer pool replaces candle's.
#[derive(Clone)]
pub struct MetalCompanion {
    device: candle_metal_kernels::metal::Device,
    kernels: std::sync::Arc<candle_metal_kernels::Kernels>,
    commands: std::sync::Arc<std::sync::RwLock<candle_metal_kernels::metal::Commands>>,
}

impl std::fmt::Debug for MetalCompanion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalCompanion")
            .field("device", &"<MTLDevice>")
            .field("kernels", &"<Kernels cache>")
            .field("commands", &"<Commands pool>")
            .finish()
    }
}

impl MetalCompanion {
    /// Build a fresh companion from a raw `candle_metal_kernels` Device
    /// handle. Allocates a new `Kernels` cache, a new `MTLCommandQueue`,
    /// and a new `Commands` pool — all candle-core-free.
    ///
    /// Same shape as candle's `MetalDevice::new` (the `Device::all()` /
    /// `Device::system_default()` + `Kernels::new()` +
    /// `Commands::new(command_queue)` triple) but without the surrounding
    /// buffer-map / seed-buffer / DeviceId machinery the 7 ops never use.
    ///
    /// Returns `Err` if the command-queue or `Commands` pool fails to
    /// build — both forward the underlying `MetalKernelError` as a
    /// `kiln_tensor::Error::Msg`.
    pub fn from_raw(device: candle_metal_kernels::metal::Device) -> crate::Result<Self> {
        use candle_metal_kernels::metal::Commands;
        use candle_metal_kernels::Kernels;
        let command_queue = device.new_command_queue().map_err(|e| {
            crate::Error::Msg(format!(
                "MetalCompanion::from_raw: new_command_queue failed: {e:?}"
            ))
        })?;
        let commands = Commands::new(command_queue).map_err(|e| {
            crate::Error::Msg(format!(
                "MetalCompanion::from_raw: Commands::new failed: {e:?}"
            ))
        })?;
        Ok(MetalCompanion {
            device,
            kernels: std::sync::Arc::new(Kernels::new()),
            commands: std::sync::Arc::new(std::sync::RwLock::new(commands)),
        })
    }

    /// Borrow the underlying `candle_metal_kernels` Device — the
    /// `&Device` parameter every `call_*` MSL kernel entry consumes.
    pub fn device(&self) -> &candle_metal_kernels::metal::Device {
        &self.device
    }

    /// Stable per-device identifier suitable for `HashMap` cache keys in
    /// per-function MSL pipeline / library caches. Returns the
    /// `MTLDevice::registryID()` value of the underlying device — a
    /// 64-bit unsigned that is unique across Metal devices visible to
    /// the system and stable across the device's lifetime.
    ///
    /// # Why this exists
    ///
    /// Step 1 in the substrate-readiness checklist of
    /// `docs/metal-cargo-toml-candle-drop-stop-2026-05-28.md`. The
    /// existing `kiln-model::backend::metal` per-function
    /// `OnceLock<Mutex<HashMap<DeviceId, ComputePipeline>>>` caches key
    /// off candle's `DeviceId(usize)` newtype. Once those helper
    /// signatures migrate from `&MetalDevice` to `&MetalCompanion`, the
    /// cache key shape becomes `HashMap<u64, ComputePipeline>` (or
    /// `HashMap<usize, _>` after a trivial cast), with `device_id()`
    /// supplying the key value at every `cache.get(&...)` /
    /// `cache.insert(..., _)` site.
    ///
    /// The `registryID` value is what candle's own `DeviceId` was
    /// originally a wrapper around (see `candle-core` 0.10's
    /// `metal_backend::device::DeviceId(usize)` — internally a
    /// monotonically incrementing counter, semantically identical to a
    /// stable device handle).
    ///
    /// # Phase 7 follow-up
    ///
    /// When the kernel-helper signatures migrate off `&MetalDevice` and
    /// onto `&MetalCompanion`, every `device.id()` call site swaps to
    /// `companion.device_id()`. The HashMap key type changes from the
    /// candle `DeviceId` to `u64`, dropping the
    /// `kiln_tensor::metal_types::DeviceId` chokepoint re-export from
    /// the import line in `kiln-model::backend::metal`.
    pub fn device_id(&self) -> u64 {
        self.device.registry_id()
    }

    /// Borrow the MSL pipeline cache — the `&Kernels` parameter every
    /// `call_*` MSL kernel entry consumes.
    pub fn kernels(&self) -> &candle_metal_kernels::Kernels {
        &self.kernels
    }

    /// Materialize a fresh `ComputeCommandEncoder` from the underlying
    /// `Commands` pool. Mirror of candle's
    /// `MetalDevice::command_encoder()` — same `Commands::command_encoder`
    /// call underneath, just reached through the kt-native companion
    /// rather than the candle wrapper.
    ///
    /// The returned encoder ends encoding on drop (the
    /// `candle_metal_kernels::metal::ComputeCommandEncoder` `Drop` impl
    /// calls `end_encoding()`), so callers don't need to manage it
    /// manually.
    pub fn command_encoder(
        &self,
    ) -> crate::Result<candle_metal_kernels::metal::ComputeCommandEncoder> {
        let commands = self.commands.write().map_err(|e| {
            crate::Error::Msg(format!(
                "MetalCompanion::command_encoder: commands.write() poisoned: {e}"
            ))
        })?;
        let (_flush, encoder) = commands.command_encoder().map_err(|e| {
            crate::Error::Msg(format!(
                "MetalCompanion::command_encoder: Commands::command_encoder failed: {e:?}"
            ))
        })?;
        // The `flush` bool from candle's pool signals when the pool
        // should sweep buffer maps. We don't keep buffer maps on the
        // companion — sweeping is candle's concern, not kt-tensor's —
        // so we discard it. (The 7 ops never touched candle's buffer-map
        // sweep either; they only used `command_encoder()` for its
        // encoder return value.)
        Ok(encoder)
    }

    /// Commit any pending command buffer and block until the GPU has
    /// finished every encoded op on this companion's queue.
    ///
    /// This is the **host-read synchronization point** (#1082): the
    /// `command_encoder()` path defers commit to the
    /// `candle_metal_kernels::Commands` pool, so a freshly-written
    /// `StorageModeShared` buffer's `contents()` pointer is not
    /// guaranteed to reflect the GPU write until the encoding command
    /// buffer has been committed and completed. Every Metal→host
    /// readback (`metal_to_host_copy`, `Tensor::to_vec` on a Metal
    /// tensor) calls this first.
    ///
    /// Mirror of candle's `MetalDevice::wait_until_completed()` — same
    /// `Commands::wait_until_completed` underneath, reached through the
    /// kt-native companion. Idempotent and cheap when nothing is
    /// pending.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Msg`] if the `Commands` lock is poisoned
    /// or the underlying commit/wait fails.
    pub fn wait_until_completed(&self) -> crate::Result<()> {
        let commands = self.commands.write().map_err(|e| {
            crate::Error::Msg(format!(
                "MetalCompanion::wait_until_completed: commands.write() poisoned: {e}"
            ))
        })?;
        commands.wait_until_completed().map_err(|e| {
            crate::Error::Msg(format!(
                "MetalCompanion::wait_until_completed: Commands::wait_until_completed failed: {e:?}"
            ))
        })?;
        Ok(())
    }
}
