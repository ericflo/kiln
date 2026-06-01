//! Candle-free type aliases for the objc2-metal substrate types used by
//! every kernel-crate FFI site in `kiln-model::backend::metal` and by the
//! in-crate Metal ops (`metal_storage.rs`, `metal_matmul.rs`,
//! `metal_kernels.rs`, `metal_allocator.rs`).
//!
//! # Why this module exists (#1082)
//!
//! It is the single chokepoint through which the whole workspace reaches
//! the Metal protocol-object substrate. Funneling every `Device` /
//! `Buffer` / `ComputePipeline` / `Library` / `ComputeCommandEncoder` /
//! `BufferOffset` reference through one re-export consolidated the
//! substrate surface area so the substrate provider could be swapped
//! without touching the ~232 kernel-helper call sites in
//! `kiln-model::backend::metal`.
//!
//! # Candle drop (#1082 final step)
//!
//! The substrate provider is now [`crate::metal_rt`] — a kiln-owned objc2
//! wrapper vendored from candle-metal-kernels 0.10.2's `src/metal/`
//! boilerplate. The aliases below repoint from
//! `candle_metal_kernels::metal::*` / `candle_metal_kernels::utils::*` to
//! `crate::metal_rt::*`. The wire types are bit-identical (both wrap the
//! same `Retained<ProtocolObject<dyn MTL*>>` from `objc2-metal 0.3.2`), so
//! the alias NAMES + method signatures are preserved verbatim and every
//! caller compiles unchanged. `kiln-tensor` no longer depends on
//! `candle_metal_kernels`.

#![cfg(feature = "metal")]

/// MSL compute-pipeline-state handle. Kiln-owned objc2 wrapper
/// ([`crate::metal_rt::ComputePipeline`]) over
/// `Retained<ProtocolObject<dyn MTLComputePipelineState>>`.
pub use crate::metal_rt::ComputePipeline;

/// MSL library handle. Kiln-owned objc2 wrapper
/// ([`crate::metal_rt::Library`]) over
/// `Retained<ProtocolObject<dyn MTLLibrary>>`.
pub use crate::metal_rt::Library;

/// `BufferOffset` — `{ buffer: &Buffer, offset_in_bytes: usize }`
/// pair consumed by every MSL kernel entry point as a positional
/// argument. Kiln-owned ([`crate::metal_rt::BufferOffset`]).
pub use crate::metal_rt::BufferOffset;

/// The raw Metal device handle every MSL pipeline / library build and
/// every kernel dispatch consumes. Kiln-owned objc2 wrapper
/// ([`crate::metal_rt::Device`]) over
/// `Retained<ProtocolObject<dyn MTLDevice>>`.
/// `MetalCompanion::device()` returns `&MetalRawDevice`.
pub use crate::metal_rt::Device as MetalRawDevice;

/// Build a [`BufferOffset`] from a [`MetalRawDevice`]-allocated `Buffer`
/// + `kiln_tensor::Layout` + `kiln_tensor::DType`.
///
/// The kt-typed chokepoint entry point for byte-offset computation. The
/// formula is `l.start_offset() * dtype.size_in_bytes()` — the same byte
/// offset every kiln MSL kernel expects.
///
/// # (#1082)
#[inline]
pub fn buffer_o_kt<'a>(
    buffer: &'a crate::metal_rt::Buffer,
    l: &crate::Layout,
    dtype: crate::DType,
) -> BufferOffset<'a> {
    BufferOffset {
        buffer,
        offset_in_bytes: l.start_offset() * dtype.size_in_bytes(),
    }
}

// ----------------------------------------------------------------------
// Native objc2-metal substrate handles (#1082)
// ----------------------------------------------------------------------
//
// Direct `Retained<ProtocolObject<dyn MTL*>>` aliases — the raw handles
// that sit underneath every `crate::metal_rt::*` wrapper. Held by
// callers that need a protocol object without the kiln wrapper layer.

/// MSL compute-pipeline-state handle as a raw `objc2-metal` protocol
/// object.
pub type RawComputePipelineState = objc2::rc::Retained<
    objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
>;

/// MSL library handle as a raw `objc2-metal` protocol object.
pub type RawLibrary =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLLibrary>>;

/// MTLBuffer handle as a raw `objc2-metal` protocol object.
pub type RawBuffer =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>;

/// MTLDevice handle as a raw `objc2-metal` protocol object.
pub type RawDevice =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>;

/// MTLCommandQueue handle as a raw `objc2-metal` protocol object.
pub type RawCommandQueue =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandQueue>>;

// ----------------------------------------------------------------------
// MetalCompanion — kt-native substrate for the in-crate Metal ops (#1082)
// ----------------------------------------------------------------------
//
// `MetalCompanion` collects the substrate primitives the in-crate Metal
// ops (the GEMM in `metal_matmul.rs` + the 9 op families in
// `metal_kernels.rs`) need:
//   - `.device()`           -> the `&MetalRawDevice` for pipeline/library builds
//   - `.command_encoder()`  -> a `ComputeCommandEncoder` from the pool
//   - `.wait_until_completed()` -> the host-read sync point
//
// All primitives live in `crate::metal_rt` (`Device::all()`,
// `Commands::new(command_queue)`, the Device's `new_command_queue()`) —
// candle-free. Holding them on a kt-native struct lets the ops dispatch
// their kiln-owned MSL kernels without ever materializing a candle type.

/// Kt-native substrate for the in-crate Metal ops. Holds the candle-free
/// primitives every kiln MSL kernel dispatch needs:
///   - A [`crate::metal_rt::Device`] for the device parameter
///   - An `Arc<RwLock<crate::metal_rt::Commands>>` command-buffer pool for
///     `ComputeCommandEncoder` materialization
///
/// Constructed via [`MetalCompanion::from_raw`] from a
/// [`crate::metal_rt::Device`] (a thin wrapper around
/// `Retained<ProtocolObject<dyn MTLDevice>>` — candle is not involved at
/// any step).
///
/// # Why fields are `Arc`-shared
///
/// `Commands` is designed for cross-thread shared use. `Device` is
/// internally `Retained<ProtocolObject<...>>` — cheap to clone via
/// NSObject `retain` — so it's held by value.
#[derive(Clone)]
pub struct MetalCompanion {
    device: crate::metal_rt::Device,
    commands: std::sync::Arc<std::sync::RwLock<crate::metal_rt::Commands>>,
}

impl std::fmt::Debug for MetalCompanion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalCompanion")
            .field("device", &"<MTLDevice>")
            .field("commands", &"<Commands pool>")
            .finish()
    }
}

impl MetalCompanion {
    /// Build a fresh companion from a raw [`crate::metal_rt::Device`]
    /// handle. Allocates a new `MTLCommandQueue` and a new `Commands`
    /// pool — all candle-free.
    ///
    /// Returns `Err` if the command-queue or `Commands` pool fails to
    /// build — both forward the underlying `MetalRtError` as a
    /// `kiln_tensor::Error::Msg`.
    pub fn from_raw(device: crate::metal_rt::Device) -> crate::Result<Self> {
        use crate::metal_rt::Commands;
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
            commands: std::sync::Arc::new(std::sync::RwLock::new(commands)),
        })
    }

    /// Borrow the underlying [`crate::metal_rt::Device`] — the device
    /// parameter every MSL pipeline / library build consumes.
    pub fn device(&self) -> &crate::metal_rt::Device {
        &self.device
    }

    /// Stable per-device identifier suitable for `HashMap` cache keys in
    /// per-function MSL pipeline / library caches. Returns the
    /// `MTLDevice::registryID()` value of the underlying device — a
    /// 64-bit unsigned that is unique across Metal devices visible to
    /// the system and stable across the device's lifetime.
    pub fn device_id(&self) -> u64 {
        self.device.registry_id()
    }

    /// Materialize a fresh `ComputeCommandEncoder` from the underlying
    /// `Commands` pool.
    ///
    /// The returned encoder ends encoding on drop (the
    /// [`crate::metal_rt::ComputeCommandEncoder`] `Drop` impl calls
    /// `end_encoding()`), so callers don't need to manage it manually.
    pub fn command_encoder(
        &self,
    ) -> crate::Result<crate::metal_rt::ComputeCommandEncoder> {
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
        // The `flush` bool from the pool signals when the pool recycled a
        // command buffer; the in-crate ops don't act on it (they wait via
        // `wait_until_completed` on the host-read path), so it's discarded.
        Ok(encoder)
    }

    /// Commit any pending command buffer and block until the GPU has
    /// finished every encoded op on this companion's queue.
    ///
    /// This is the **host-read synchronization point** (#1082): the
    /// `command_encoder()` path defers commit to the
    /// [`crate::metal_rt::Commands`] pool, so a freshly-written
    /// `StorageModeShared` buffer's `contents()` pointer is not
    /// guaranteed to reflect the GPU write until the encoding command
    /// buffer has been committed and completed. Every Metal→host
    /// readback (`metal_to_host_copy`, `Tensor::to_vec` on a Metal
    /// tensor) calls this first.
    ///
    /// Idempotent and cheap when nothing is pending.
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
