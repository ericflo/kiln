//! `kiln-hip` — bounded, safe Rust bindings to the AMD ROCm/HIP runtime.
//!
//! This is the **cudarc analog** for kiln's ROCm backend (Phase R.1). It mirrors
//! the exact ~8-symbol surface the CUDA substrate uses
//! (`CudaContext`/`CudaStream`/`CudaSlice`/`DevicePtr` + a few `result`/`sys`
//! free functions) so that `rocm_storage.rs` / `rocm_allocator.rs` (Phase R.3)
//! are mechanical retypes of the candle-free `cuda_*.rs` files.
//!
//! Design mirrors `kiln-tensor/src/cuda_stream_priority.rs`: own the raw HIP
//! handle, implement `Drop`, expose a checked FFI accessor
//! (`hip_stream_for_execution()`), and carry `unsafe impl Send + Sync` with the
//! same justification cudarc uses.
//!
//! The crate compiles on hosts with no ROCm toolchain (the FFI block has no
//! `#[link]`; `build.rs` links `amdhip64` only when ROCm is present). Calling a
//! function with no runtime present returns `Err(HipError)` rather than
//! aborting — except linking, which only a ROCm host performs.

pub mod sys;

use std::collections::HashMap;
use std::ffi::CStr;
use std::fmt;
use std::os::raw::{c_int, c_uint, c_void};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

/// Result alias for HIP calls.
pub type Result<T> = std::result::Result<T, HipError>;

static CLEANUP_QUARANTINE_DROP_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);

fn device_cleanup_quarantine(ordinal: c_int) -> Arc<AtomicBool> {
    static QUARANTINES: OnceLock<Mutex<HashMap<c_int, Arc<AtomicBool>>>> = OnceLock::new();
    let mut quarantines = QUARANTINES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    Arc::clone(
        quarantines
            .entry(ordinal)
            .or_insert_with(|| Arc::new(AtomicBool::new(false))),
    )
}

fn warn_cleanup_quarantine(resource: &str) {
    if !CLEANUP_QUARANTINE_DROP_WARNING_EMITTED.swap(true, Ordering::AcqRel) {
        eprintln!(
            "{resource}: ROCm cleanup quarantined after a fatal execution or cleanup failure; retaining possibly in-flight HIP resources until process exit"
        );
    }
}

fn bind_device_for_cleanup(
    ordinal: c_int,
    cleanup_quarantined: &AtomicBool,
    resource: &str,
) -> bool {
    if cleanup_quarantined.load(Ordering::Acquire) {
        warn_cleanup_quarantine(resource);
        return false;
    }
    let rc = unsafe { sys::hipSetDevice(ordinal) };
    if rc != sys::HIP_SUCCESS {
        cleanup_quarantined.store(true, Ordering::Release);
        eprintln!("{resource}: hipSetDevice failed during cleanup (hipError {rc})");
        warn_cleanup_quarantine(resource);
        return false;
    }
    true
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// A failed HIP runtime call: the numeric `hipError_t`, the API symbol that
/// returned it, and the driver's human-readable string.
#[derive(Clone)]
pub struct HipError {
    /// The raw `hipError_t` code (`hipSuccess == 0` never appears here).
    pub code: i32,
    /// The HIP API function that returned the error (e.g. `"hipMalloc"`).
    pub api: &'static str,
    /// `hipGetErrorString(code)`, if resolvable.
    pub message: String,
}

impl fmt::Debug for HipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HipError({} from {}: {})",
            self.code, self.api, self.message
        )
    }
}

impl fmt::Display for HipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} failed: {} (hipError {})",
            self.api, self.message, self.code
        )
    }
}

impl std::error::Error for HipError {}

/// Convert a raw `hipError_t` into a `Result`, attaching the API name and the
/// driver's error string.
#[inline]
fn check(code: sys::hipError_t, api: &'static str) -> Result<()> {
    if code == sys::HIP_SUCCESS {
        return Ok(());
    }
    // HIP runtime errors are sticky per host thread. If a graph/runtime API
    // returns an error and leaves it pending, the next kt kernel wrapper's
    // post-launch hipGetLastError/cudaGetLastError can incorrectly report that
    // stale graph error as the kernel's launch failure. Preserve this call's
    // direct return code for the caller, but clear the sticky slot here.
    let _ = unsafe { sys::hipGetLastError() };
    // SAFETY: hipGetErrorString returns a static NUL-terminated string for any
    // code (an "unknown error" string for unrecognized codes). Returns a valid
    // pointer; never null in practice.
    let message = unsafe {
        let ptr = sys::hipGetErrorString(code);
        if ptr.is_null() {
            String::from("<no error string>")
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    Err(HipError {
        code: code as i32,
        api,
        message,
    })
}

/// The HIP runtime version (`hipRuntimeGetVersion`), or an error if no runtime
/// is linked/available. Encoded as `10_000_000*major + 100_000*minor + patch`
/// in recent ROCm.
pub fn runtime_version() -> Result<i32> {
    let mut v: c_int = 0;
    check(
        unsafe { sys::hipRuntimeGetVersion(&mut v) },
        "hipRuntimeGetVersion",
    )?;
    Ok(v)
}

/// Number of visible HIP devices, or `0` if the runtime reports none.
///
/// Never errors on a missing runtime in the "no device" sense — it surfaces the
/// underlying `hipError_t` so callers can distinguish "no GPU" from "no driver".
pub fn device_count() -> Result<i32> {
    let mut n: c_int = 0;
    check(
        unsafe { sys::hipGetDeviceCount(&mut n) },
        "hipGetDeviceCount",
    )?;
    Ok(n)
}

/// Best-effort availability probe: `true` iff the HIP runtime links and reports
/// at least one device. Swallows errors into `false` so callers can branch
/// without handling a `Result` (mirrors `cuda_is_available()` usage).
pub fn is_available() -> bool {
    matches!(device_count(), Ok(n) if n > 0)
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/// Host synchronization discipline for work queued through a [`RocmContext`].
///
/// The legacy mode preserves kiln's historical host barriers exactly. The
/// stream-ordered mode only removes barriers that a caller explicitly marks as
/// same-stream dependencies; explicit drains, host readbacks, allocation
/// lifetime boundaries, graph transitions, and memory reclamation still wait.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RocmSynchronizationMode {
    /// Preserve the historical device/stream synchronization behavior.
    #[default]
    LegacyHostBarriers,
    /// Trust FIFO ordering for explicitly proven same-stream dependencies.
    StreamOrdered,
}

/// Immutable execution policy installed when a [`RocmContext`] is created.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RocmExecutionPolicy {
    /// Synchronization discipline for steady-state ROCm execution.
    pub synchronization_mode: RocmSynchronizationMode,
}

impl RocmExecutionPolicy {
    /// Construct a policy with the requested synchronization discipline.
    pub const fn new(synchronization_mode: RocmSynchronizationMode) -> Self {
        Self {
            synchronization_mode,
        }
    }
}

/// Fixed-cardinality reasons for a host-visible ROCm synchronization boundary.
///
/// These values are stable metric dimensions. Add a variant for a genuinely
/// new boundary class rather than recording free-form operation names.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum RocmSyncReason {
    /// A caller explicitly requested a device-wide drain.
    ExplicitDeviceDrain,
    /// A caller explicitly requested an active-stream drain.
    ExplicitStreamDrain,
    /// A tensor is crossing into a consumer with an external stream contract.
    TensorHandoff,
    /// Generated output is about to become visible outside the backend.
    ExternalYield,
    /// An eager activation result is consumed on the same stream.
    ActivationOutput,
    /// An eager elementwise result is consumed on the same stream.
    ElementwiseOutput,
    /// An eager cast result is consumed on the same stream.
    CastOutput,
    /// A concat input is handed to same-stream assembly.
    ConcatInput,
    /// A concat result is consumed on the same stream.
    ConcatOutput,
    /// A contiguous-copy result is consumed on the same stream.
    ContiguousOutput,
    /// A repeated-head result is consumed on the same stream.
    RepeatHeadsOutput,
    /// A GEMM result is consumed on the same stream.
    MatmulOutput,
    /// A GEMM FP32-output fallback crosses its cast boundary.
    MatmulCastBoundary,
    /// An in-place mutation must be complete before returning to its caller.
    InPlaceMutation,
    /// Device work must drain before releasing pooled memory.
    MemoryReclaim,
    /// Graph capture, instantiation, or replay crosses a host boundary.
    GraphBoundary,
    /// Full attention crosses into a consumer stream or implementation.
    FullAttentionHandoff,
    /// A model-level tensor crosses into a consumer stream or implementation.
    ModelHandoff,
    /// A device result is about to be read by the host.
    HostReadback,
    /// Work must complete before an allocation or borrowed owner can expire.
    AllocationLifetime,
    /// Synchronization is required to contain or diagnose an execution error.
    ErrorRecovery,
    /// Synchronization protects a process- or device-global state transition.
    GlobalStateMutation,
}

impl RocmSyncReason {
    /// Every synchronization reason in stable metric order.
    pub const ALL: [Self; ROCM_SYNC_REASON_COUNT] = [
        Self::ExplicitDeviceDrain,
        Self::ExplicitStreamDrain,
        Self::TensorHandoff,
        Self::ExternalYield,
        Self::ActivationOutput,
        Self::ElementwiseOutput,
        Self::CastOutput,
        Self::ConcatInput,
        Self::ConcatOutput,
        Self::ContiguousOutput,
        Self::RepeatHeadsOutput,
        Self::MatmulOutput,
        Self::MatmulCastBoundary,
        Self::InPlaceMutation,
        Self::MemoryReclaim,
        Self::GraphBoundary,
        Self::FullAttentionHandoff,
        Self::ModelHandoff,
        Self::HostReadback,
        Self::AllocationLifetime,
        Self::ErrorRecovery,
        Self::GlobalStateMutation,
    ];

    /// Stable lower-snake-case metric label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExplicitDeviceDrain => "explicit_device_drain",
            Self::ExplicitStreamDrain => "explicit_stream_drain",
            Self::TensorHandoff => "tensor_handoff",
            Self::ExternalYield => "external_yield",
            Self::ActivationOutput => "activation_output",
            Self::ElementwiseOutput => "elementwise_output",
            Self::CastOutput => "cast_output",
            Self::ConcatInput => "concat_input",
            Self::ConcatOutput => "concat_output",
            Self::ContiguousOutput => "contiguous_output",
            Self::RepeatHeadsOutput => "repeat_heads_output",
            Self::MatmulOutput => "matmul_output",
            Self::MatmulCastBoundary => "matmul_cast_boundary",
            Self::InPlaceMutation => "in_place_mutation",
            Self::MemoryReclaim => "memory_reclaim",
            Self::GraphBoundary => "graph_boundary",
            Self::FullAttentionHandoff => "full_attention_handoff",
            Self::ModelHandoff => "model_handoff",
            Self::HostReadback => "host_readback",
            Self::AllocationLifetime => "allocation_lifetime",
            Self::ErrorRecovery => "error_recovery",
            Self::GlobalStateMutation => "global_state_mutation",
        }
    }

    #[inline]
    const fn index(self) -> usize {
        self as usize
    }
}

/// Number of fixed ROCm synchronization-reason metric dimensions.
pub const ROCM_SYNC_REASON_COUNT: usize = 22;

/// Atomic synchronization counters for one fixed [`RocmSyncReason`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RocmSyncReasonStats {
    /// Stable reason dimension.
    pub reason: RocmSyncReason,
    /// Completed or failed device-wide wait attempts.
    pub device_wait_count: u64,
    /// Completed or failed stream wait attempts.
    pub stream_wait_count: u64,
    /// Host wall-clock nanoseconds spent in all wait attempts.
    pub waited_ns: u64,
    /// Barriers omitted because stream-ordered execution proved them redundant.
    pub skipped_count: u64,
}

/// Point-in-time synchronization telemetry for one [`RocmContext`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RocmSyncTelemetrySnapshot {
    /// Per-reason counters in [`RocmSyncReason::ALL`] order.
    pub reasons: [RocmSyncReasonStats; ROCM_SYNC_REASON_COUNT],
    /// A fatal execution or synchronization failure quarantined this device.
    /// The quarantine is process-lifetime sticky: new execution is rejected and
    /// Drop implementations retain possibly in-flight HIP resources rather than
    /// risking use-after-free. Restart the process to recover the device.
    pub cleanup_quarantined: bool,
}

impl RocmSyncTelemetrySnapshot {
    /// Total host wait attempts across every reason and wait scope.
    pub fn total_wait_count(&self) -> u64 {
        self.reasons.iter().fold(0u64, |total, stats| {
            total
                .saturating_add(stats.device_wait_count)
                .saturating_add(stats.stream_wait_count)
        })
    }

    /// Total host wall-clock nanoseconds spent waiting across every reason.
    pub fn total_waited_ns(&self) -> u64 {
        self.reasons
            .iter()
            .fold(0u64, |total, stats| total.saturating_add(stats.waited_ns))
    }

    /// Total same-stream barriers omitted by the execution policy.
    pub fn total_skipped_count(&self) -> u64 {
        self.reasons.iter().fold(0u64, |total, stats| {
            total.saturating_add(stats.skipped_count)
        })
    }
}

#[derive(Debug)]
struct RocmSyncTelemetry {
    device_wait_counts: [AtomicU64; ROCM_SYNC_REASON_COUNT],
    stream_wait_counts: [AtomicU64; ROCM_SYNC_REASON_COUNT],
    waited_ns: [AtomicU64; ROCM_SYNC_REASON_COUNT],
    skipped_counts: [AtomicU64; ROCM_SYNC_REASON_COUNT],
}

impl Default for RocmSyncTelemetry {
    fn default() -> Self {
        Self {
            device_wait_counts: std::array::from_fn(|_| AtomicU64::new(0)),
            stream_wait_counts: std::array::from_fn(|_| AtomicU64::new(0)),
            waited_ns: std::array::from_fn(|_| AtomicU64::new(0)),
            skipped_counts: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
}

#[derive(Clone, Copy)]
enum RocmSyncScope {
    Device,
    Stream,
}

impl RocmSyncTelemetry {
    fn saturating_add(counter: &AtomicU64, value: u64) {
        let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            Some(current.saturating_add(value))
        });
    }

    fn record_wait(&self, reason: RocmSyncReason, scope: RocmSyncScope, waited_ns: u64) {
        let index = reason.index();
        let counter = match scope {
            RocmSyncScope::Device => &self.device_wait_counts[index],
            RocmSyncScope::Stream => &self.stream_wait_counts[index],
        };
        Self::saturating_add(counter, 1);
        Self::saturating_add(&self.waited_ns[index], waited_ns);
    }

    fn record_skipped(&self, reason: RocmSyncReason) {
        Self::saturating_add(&self.skipped_counts[reason.index()], 1);
    }

    fn snapshot(&self) -> RocmSyncTelemetrySnapshot {
        RocmSyncTelemetrySnapshot {
            reasons: std::array::from_fn(|index| RocmSyncReasonStats {
                reason: RocmSyncReason::ALL[index],
                device_wait_count: self.device_wait_counts[index].load(Ordering::Relaxed),
                stream_wait_count: self.stream_wait_counts[index].load(Ordering::Relaxed),
                waited_ns: self.waited_ns[index].load(Ordering::Relaxed),
                skipped_count: self.skipped_counts[index].load(Ordering::Relaxed),
            }),
            cleanup_quarantined: false,
        }
    }
}

/// A HIP device handle + its default stream. The runtime-API analog of
/// `cudarc::driver::CudaContext`.
///
/// HIP's runtime API has no explicit context object — the "context" is the
/// device ordinal, bound per-thread via `hipSetDevice`. `RocmContext` carries
/// the ordinal and an owned default stream so the surface matches cudarc's
/// `CudaContext { default_stream(), new_stream() }`.
#[derive(Debug)]
pub struct RocmContext {
    ordinal: c_int,
    default_stream: Arc<RocmStream>,
    execution_policy: RocmExecutionPolicy,
    sync_telemetry: Arc<RocmSyncTelemetry>,
    cleanup_quarantined: Arc<AtomicBool>,
}

// SAFETY: the ordinal is a plain int; the default stream is itself Send+Sync
// (see RocmStream). Binding is per-thread via hipSetDevice on every call, so a
// RocmContext is safe to share across threads — the same reasoning cudarc uses
// for CudaContext.
unsafe impl Send for RocmContext {}
unsafe impl Sync for RocmContext {}

impl RocmContext {
    /// Create a context for device `ordinal`, validating that the runtime is
    /// present and the ordinal is in range, then creating a non-blocking
    /// default stream.
    pub fn new(ordinal: usize) -> Result<Arc<Self>> {
        Self::new_with_execution_policy(ordinal, RocmExecutionPolicy::default())
    }

    /// Create a context with an immutable steady-state execution policy.
    ///
    /// The policy is copied into the context before any stream is exposed, so
    /// every tensor and backend sharing the context observes one discipline.
    pub fn new_with_execution_policy(
        ordinal: usize,
        execution_policy: RocmExecutionPolicy,
    ) -> Result<Arc<Self>> {
        let ordinal = ordinal as c_int;
        let count = device_count()?;
        if ordinal < 0 || ordinal >= count {
            return Err(HipError {
                code: -1,
                api: "RocmContext::new",
                message: format!("device ordinal {ordinal} out of range (count={count})"),
            });
        }
        // HIP's runtime context is device-global within this process. Every
        // RocmContext for one ordinal must therefore share one sticky quarantine;
        // otherwise a second context could execute or free resources after a
        // device-wide settlement failure in the first.
        let cleanup_quarantined = device_cleanup_quarantine(ordinal);
        if cleanup_quarantined.load(Ordering::Acquire) {
            return Err(HipError {
                code: -1,
                api: "RocmContext::new",
                message: format!(
                    "ROCm device {ordinal} is quarantined after a fatal execution failure; restart the process"
                ),
            });
        }
        check(unsafe { sys::hipSetDevice(ordinal) }, "hipSetDevice")?;
        // Pin the stream-ordered allocator's pool to NEVER release freed memory
        // back to the OS (hipMemPoolAttrReleaseThreshold = u64::MAX). The default
        // threshold (0) makes hipMallocAsync hand freed pages back aggressively;
        // under the decode alloc/free churn that races in-flight kernels and
        // corrupts output (it's what the paged-decode order-sync was masking).
        // Keeping freed blocks pooled removes the hazard AND avoids OS-roundtrip
        // alloc latency. Best-effort: ignore if the runtime lacks mempools.
        const HIP_MEM_POOL_ATTR_RELEASE_THRESHOLD: c_uint = 4;
        let mut pool: *mut c_void = ptr::null_mut();
        if unsafe { sys::hipDeviceGetDefaultMemPool(&mut pool, ordinal) } == sys::HIP_SUCCESS
            && !pool.is_null()
        {
            let mut threshold: u64 = u64::MAX;
            let _ = unsafe {
                sys::hipMemPoolSetAttribute(
                    pool,
                    HIP_MEM_POOL_ATTR_RELEASE_THRESHOLD,
                    &mut threshold as *mut u64 as *mut c_void,
                )
            };
        }
        let sync_telemetry = Arc::new(RocmSyncTelemetry::default());
        let default_stream = RocmStream::create(
            ordinal,
            None,
            Arc::clone(&sync_telemetry),
            Arc::clone(&cleanup_quarantined),
        )?;
        Ok(Arc::new(RocmContext {
            ordinal,
            default_stream,
            execution_policy,
            sync_telemetry,
            cleanup_quarantined,
        }))
    }

    /// The device ordinal this context targets.
    pub fn ordinal(&self) -> usize {
        self.ordinal as usize
    }

    /// Bind this device to the calling thread (`hipSetDevice`). Cheap; called
    /// before driver work, mirroring cudarc's `bind_to_thread`.
    pub fn bind_to_thread(&self) -> Result<()> {
        self.ensure_execution_available("RocmContext::bind_to_thread")?;
        check(unsafe { sys::hipSetDevice(self.ordinal) }, "hipSetDevice")
    }

    /// Whether a fatal failure made further execution and resource cleanup
    /// unsafe for this device. This state is shared by every context for the
    /// same ordinal and remains set until process restart.
    pub fn cleanup_quarantined(&self) -> bool {
        self.cleanup_quarantined.load(Ordering::Acquire)
    }

    /// Reject new execution after a fatal failure. A narrowly scoped
    /// [`Self::synchronize_device_for`] call with `ErrorRecovery` may still try
    /// to settle physical work for diagnosis, but never clears the quarantine.
    pub fn ensure_execution_available(&self, api: &'static str) -> Result<()> {
        if self.cleanup_quarantined() {
            return Err(HipError {
                code: -1,
                api,
                message: "ROCm context cleanup is quarantined after synchronization failure"
                    .to_string(),
            });
        }
        Ok(())
    }

    /// Permanently quarantine execution and cleanup for this device until the
    /// process restarts. Higher layers use this when logical model state cannot
    /// be proven valid even if the HIP device itself synchronized successfully.
    pub fn quarantine_execution(&self) {
        self.cleanup_quarantined.store(true, Ordering::Release);
    }

    /// The context's default (non-blocking) stream.
    pub fn default_stream(&self) -> Arc<RocmStream> {
        self.default_stream.clone()
    }

    /// Immutable steady-state execution policy for this context.
    pub const fn execution_policy(&self) -> RocmExecutionPolicy {
        self.execution_policy
    }

    /// Point-in-time fixed-cardinality synchronization telemetry.
    pub fn sync_telemetry_snapshot(&self) -> RocmSyncTelemetrySnapshot {
        let mut snapshot = self.sync_telemetry.snapshot();
        snapshot.cleanup_quarantined = self.cleanup_quarantined.load(Ordering::Acquire);
        snapshot
    }

    fn timed_synchronize(
        &self,
        reason: RocmSyncReason,
        scope: RocmSyncScope,
        synchronize: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        let started = Instant::now();
        let result = synchronize();
        let waited_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.sync_telemetry.record_wait(reason, scope, waited_ns);
        if result.is_err() {
            self.cleanup_quarantined.store(true, Ordering::Release);
        }
        result
    }

    /// Always perform and account for a device-wide synchronization.
    ///
    /// Use this for true global boundaries such as reclamation and global-state
    /// mutation. Same-stream steady-state dependencies must use
    /// [`Self::synchronize_same_stream_dependency`] instead.
    pub fn synchronize_device_for(&self, reason: RocmSyncReason) -> Result<()> {
        if reason != RocmSyncReason::ErrorRecovery {
            self.ensure_execution_available("RocmContext::synchronize_device_for")?;
        }
        self.timed_synchronize(reason, RocmSyncScope::Device, || {
            // Recovery synchronization must remain available while cleanup is
            // quarantined, so bind directly instead of calling the guarded
            // public `bind_to_thread` entry point.
            check(unsafe { sys::hipSetDevice(self.ordinal) }, "hipSetDevice")?;
            check(
                unsafe { sys::hipDeviceSynchronize() },
                "hipDeviceSynchronize",
            )
        })
    }

    /// Always perform and account for a synchronization on `stream`.
    pub fn synchronize_stream_for(
        &self,
        stream: &RocmStream,
        reason: RocmSyncReason,
    ) -> Result<()> {
        if stream.ordinal != self.ordinal {
            return Err(HipError {
                code: -1,
                api: "RocmContext::synchronize_stream_for",
                message: format!(
                    "context belongs to device {} but stream belongs to device {}",
                    self.ordinal, stream.ordinal
                ),
            });
        }
        self.timed_synchronize(reason, RocmSyncScope::Stream, || stream.synchronize())
    }

    /// Order a producer and consumer already known to use the same stream.
    ///
    /// Legacy mode performs the historical host wait. Stream-ordered mode
    /// records a skipped barrier and relies on the stream's FIFO dependency.
    pub fn synchronize_same_stream_dependency(
        &self,
        stream: &RocmStream,
        reason: RocmSyncReason,
    ) -> Result<()> {
        self.ensure_execution_available("RocmContext::synchronize_same_stream_dependency")?;
        match self.execution_policy.synchronization_mode {
            RocmSynchronizationMode::LegacyHostBarriers => {
                self.synchronize_stream_for(stream, reason)
            }
            RocmSynchronizationMode::StreamOrdered => {
                if stream.ordinal != self.ordinal {
                    return Err(HipError {
                        code: -1,
                        api: "RocmContext::synchronize_same_stream_dependency",
                        message: format!(
                            "context belongs to device {} but stream belongs to device {}",
                            self.ordinal, stream.ordinal
                        ),
                    });
                }
                self.sync_telemetry.record_skipped(reason);
                Ok(())
            }
        }
    }

    /// Preserve a historical device-wide barrier in legacy mode while allowing
    /// stream-ordered mode to omit it for a proven same-stream dependency.
    ///
    /// This is intentionally narrower than [`Self::synchronize_device_for`]. It
    /// exists for compatibility barriers around libraries such as hipBLASLt
    /// that were historically device-wide even though their producer and
    /// consumer are submitted to the same explicit stream.
    pub fn synchronize_legacy_device_same_stream_dependency(
        &self,
        stream: &RocmStream,
        reason: RocmSyncReason,
    ) -> Result<()> {
        self.ensure_execution_available(
            "RocmContext::synchronize_legacy_device_same_stream_dependency",
        )?;
        if stream.ordinal != self.ordinal {
            return Err(HipError {
                code: -1,
                api: "RocmContext::synchronize_legacy_device_same_stream_dependency",
                message: format!(
                    "context belongs to device {} but stream belongs to device {}",
                    self.ordinal, stream.ordinal
                ),
            });
        }
        match self.execution_policy.synchronization_mode {
            RocmSynchronizationMode::LegacyHostBarriers => self.synchronize_device_for(reason),
            RocmSynchronizationMode::StreamOrdered => {
                self.sync_telemetry.record_skipped(reason);
                Ok(())
            }
        }
    }

    /// Synchronize work before exposing generated output outside the backend.
    ///
    /// Legacy mode preserves the historical device-wide drain. Stream-ordered
    /// mode drains the context's default stream; graph replay records an event
    /// from its capture stream into this stream before reaching the yield.
    pub fn synchronize_external_yield(&self) -> Result<()> {
        self.synchronize_external_yield_for(&self.default_stream)
    }

    /// Synchronize the actual producer stream before publishing progress.
    /// Callers that dispatch through a private stream use this form after the
    /// active-stream scope has ended.
    pub fn synchronize_external_yield_for(&self, producer_stream: &RocmStream) -> Result<()> {
        if producer_stream.ordinal != self.ordinal {
            return Err(HipError {
                code: -1,
                api: "RocmContext::synchronize_external_yield_for",
                message: format!(
                    "context belongs to device {} but producer stream belongs to device {}",
                    self.ordinal, producer_stream.ordinal
                ),
            });
        }
        match self.execution_policy.synchronization_mode {
            RocmSynchronizationMode::LegacyHostBarriers => {
                self.synchronize_device_for(RocmSyncReason::ExternalYield)
            }
            RocmSynchronizationMode::StreamOrdered => {
                self.synchronize_stream_for(producer_stream, RocmSyncReason::ExternalYield)
            }
        }
    }

    /// Return pooled-but-unused VRAM to the OS, keeping at least
    /// `min_keep_bytes` cached for fast reuse. This is how kiln gives memory
    /// back when a coexisting process needs it (the pool otherwise hoards freed
    /// blocks via the release-threshold pin, by design, to avoid the async-free
    /// decode race).
    ///
    /// When spare bytes exist, synchronizes the device before releasing them so
    /// no in-flight kernel is reading a released block. Returns the measured
    /// reduction in pool-reserved bytes. A pool with no spare bytes returns
    /// without synchronizing the device.
    pub fn trim_pool(&self, min_keep_bytes: usize) -> Result<u64> {
        self.bind_to_thread()?;
        let (reserved_before, used_before) = self.pool_stats()?;
        let spare_before = reserved_before.saturating_sub(used_before);
        let requested_release = reserved_before
            .saturating_sub(min_keep_bytes as u64)
            .min(spare_before);
        if requested_release == 0 {
            return Ok(0);
        }
        // Drain in-flight work so freed pages aren't yanked from under a kernel.
        self.synchronize_device_for(RocmSyncReason::MemoryReclaim)?;
        let mut pool: *mut c_void = ptr::null_mut();
        check(
            unsafe { sys::hipDeviceGetDefaultMemPool(&mut pool, self.ordinal) },
            "hipDeviceGetDefaultMemPool",
        )?;
        if pool.is_null() {
            return Ok(0);
        }
        let effective_min_keep = usize::try_from(reserved_before.saturating_sub(requested_release))
            .map_err(|_| HipError {
                code: -1,
                api: "RocmContext::trim_pool",
                message: "pool byte count does not fit usize".to_string(),
            })?;
        check(
            unsafe { sys::hipMemPoolTrimTo(pool, effective_min_keep) },
            "hipMemPoolTrimTo",
        )?;
        let (reserved_after, _) = self.pool_stats()?;
        Ok(reserved_before.saturating_sub(reserved_after))
    }

    /// `(reserved, used)` bytes of THIS device's default stream-ordered memory
    /// pool: `reserved` = total VRAM the pool currently holds from the OS (its
    /// high-water mark, since we pin the release threshold), `used` = bytes
    /// actively allocated out of it right now. `reserved - used` is pooled-free
    /// memory available for reuse WITHOUT a new OS reservation.
    ///
    /// Unlike the DRM/`hipMemGetInfo` counters, these are PROCESS-ISOLATED — they
    /// measure only kiln's pool, immune to a coexisting llama-server etc. This is
    /// the right signal for "did a freed KV pool get reused vs grow our
    /// footprint": a reuse leaves `reserved` flat; a leak grows it. Returns
    /// `(0,0)` if the runtime lacks mempools.
    pub fn pool_stats(&self) -> Result<(u64, u64)> {
        self.bind_to_thread()?;
        const HIP_MEM_POOL_ATTR_RESERVED_MEM_CURRENT: c_uint = 5;
        const HIP_MEM_POOL_ATTR_USED_MEM_CURRENT: c_uint = 7;
        let mut pool: *mut c_void = ptr::null_mut();
        if unsafe { sys::hipDeviceGetDefaultMemPool(&mut pool, self.ordinal) } != sys::HIP_SUCCESS
            || pool.is_null()
        {
            return Ok((0, 0));
        }
        let mut reserved: u64 = 0;
        let mut used: u64 = 0;
        check(
            unsafe {
                sys::hipMemPoolGetAttribute(
                    pool,
                    HIP_MEM_POOL_ATTR_RESERVED_MEM_CURRENT,
                    &mut reserved as *mut u64 as *mut c_void,
                )
            },
            "hipMemPoolGetAttribute(reserved)",
        )?;
        check(
            unsafe {
                sys::hipMemPoolGetAttribute(
                    pool,
                    HIP_MEM_POOL_ATTR_USED_MEM_CURRENT,
                    &mut used as *mut u64 as *mut c_void,
                )
            },
            "hipMemPoolGetAttribute(used)",
        )?;
        Ok((reserved, used))
    }

    /// Device-reported `(free, total)` bytes via `hipMemGetInfo`. On a discrete
    /// GPU this is the driver's own view; on a unified APU it reflects the GTT
    /// budget and is best cross-checked against the OS-level `kiln-memory` probe.
    pub fn mem_get_info(&self) -> Result<(usize, usize)> {
        self.bind_to_thread()?;
        let mut free: usize = 0;
        let mut total: usize = 0;
        check(
            unsafe { sys::hipMemGetInfo(&mut free, &mut total) },
            "hipMemGetInfo",
        )?;
        Ok((free, total))
    }

    /// Create a fresh non-blocking stream bound to this device.
    pub fn new_stream(&self) -> Result<Arc<RocmStream>> {
        self.bind_to_thread()?;
        RocmStream::create(
            self.ordinal,
            None,
            Arc::clone(&self.sync_telemetry),
            Arc::clone(&self.cleanup_quarantined),
        )
    }

    /// Create a non-blocking stream at an explicit scheduling priority. Lower
    /// integer = higher priority (HIP follows the CUDA convention). See
    /// [`stream_priority_range`].
    pub fn new_stream_with_priority(&self, priority: i32) -> Result<Arc<RocmStream>> {
        self.bind_to_thread()?;
        RocmStream::create(
            self.ordinal,
            Some(priority),
            Arc::clone(&self.sync_telemetry),
            Arc::clone(&self.cleanup_quarantined),
        )
    }

    /// Create a reusable ordering-only event on this device.
    pub fn new_event(&self) -> Result<Arc<RocmEvent>> {
        self.bind_to_thread()?;
        RocmEvent::create(self.ordinal, Arc::clone(&self.cleanup_quarantined))
    }

    /// Block until all work on the device completes (`hipDeviceSynchronize`).
    pub fn synchronize(&self) -> Result<()> {
        self.synchronize_device_for(RocmSyncReason::ExplicitDeviceDrain)
    }
}

/// Query the device's stream-priority integer range as
/// `(least_priority, greatest_priority)`. `greatest <= least` (lower int =
/// higher priority), both `0` on devices without priority support. Mirrors
/// `cuda_stream_priority_range`.
pub fn stream_priority_range() -> Result<(i32, i32)> {
    let mut least: c_int = 0;
    let mut greatest: c_int = 0;
    check(
        unsafe { sys::hipDeviceGetStreamPriorityRange(&mut least, &mut greatest) },
        "hipDeviceGetStreamPriorityRange",
    )?;
    Ok((least, greatest))
}

// ---------------------------------------------------------------------------
// Stream
// ---------------------------------------------------------------------------

/// RAII owner of a HIP stream. The runtime-API analog of
/// `cudarc::driver::CudaStream`; `hip_stream()` matches `cu_stream()` so kernel
/// launch FFI accepts it unchanged.
#[derive(Debug)]
pub struct RocmStream {
    handle: sys::hipStream_t,
    ordinal: c_int,
    sync_telemetry: Arc<RocmSyncTelemetry>,
    cleanup_quarantined: Arc<AtomicBool>,
}

/// RAII owner of a reusable HIP event configured for ordering only.
#[derive(Debug)]
pub struct RocmEvent {
    handle: sys::hipEvent_t,
    ordinal: c_int,
    cleanup_quarantined: Arc<AtomicBool>,
}

// SAFETY: a hipEvent_t is an opaque runtime handle bound to one device. HIP
// permits recording and waiting on it from different host threads/streams;
// every operation below rebinds the owning device first.
unsafe impl Send for RocmEvent {}
unsafe impl Sync for RocmEvent {}

impl RocmEvent {
    fn create(ordinal: c_int, cleanup_quarantined: Arc<AtomicBool>) -> Result<Arc<Self>> {
        check(unsafe { sys::hipSetDevice(ordinal) }, "hipSetDevice")?;
        let mut handle: sys::hipEvent_t = ptr::null_mut();
        check(
            unsafe { sys::hipEventCreateWithFlags(&mut handle, sys::HIP_EVENT_DISABLE_TIMING) },
            "hipEventCreateWithFlags",
        )?;
        Ok(Arc::new(Self {
            handle,
            ordinal,
            cleanup_quarantined,
        }))
    }

    fn ensure_same_device(&self, stream: &RocmStream) -> Result<()> {
        if self.ordinal != stream.ordinal {
            return Err(HipError {
                code: -1,
                api: "RocmEvent device validation",
                message: format!(
                    "event belongs to device {} but stream belongs to device {}",
                    self.ordinal, stream.ordinal
                ),
            });
        }
        Ok(())
    }
}

// SAFETY: a hipStream_t is a raw handle bound to one device; it is safe to move
// and share across threads (binding is re-applied per call). Same reasoning as
// cudarc's `unsafe impl Send/Sync for CudaStream`.
unsafe impl Send for RocmStream {}
unsafe impl Sync for RocmStream {}

impl RocmStream {
    /// Create a non-blocking stream bound to `ordinal`, optionally at a given
    /// priority. Internal — callers go through `RocmContext`.
    fn create(
        ordinal: c_int,
        priority: Option<i32>,
        sync_telemetry: Arc<RocmSyncTelemetry>,
        cleanup_quarantined: Arc<AtomicBool>,
    ) -> Result<Arc<Self>> {
        check(unsafe { sys::hipSetDevice(ordinal) }, "hipSetDevice")?;
        let mut handle: sys::hipStream_t = ptr::null_mut();
        match priority {
            None => check(
                unsafe { sys::hipStreamCreateWithFlags(&mut handle, sys::HIP_STREAM_NON_BLOCKING) },
                "hipStreamCreateWithFlags",
            )?,
            Some(p) => check(
                unsafe {
                    sys::hipStreamCreateWithPriority(
                        &mut handle,
                        sys::HIP_STREAM_NON_BLOCKING,
                        p as c_int,
                    )
                },
                "hipStreamCreateWithPriority",
            )?,
        }
        Ok(Arc::new(RocmStream {
            handle,
            ordinal,
            sync_telemetry,
            cleanup_quarantined,
        }))
    }

    /// The raw `hipStream_t` for crate-internal identity checks and tests.
    /// External launch code must use [`Self::hip_stream_for_execution`].
    pub(crate) fn hip_stream(&self) -> sys::hipStream_t {
        self.handle
    }

    /// Acquire the raw HIP stream handle for an external kernel launch.
    ///
    /// Unlike [`Self::hip_stream`], this is a fallible execution boundary: it
    /// rejects a cleanup-quarantined context and binds the stream's device to
    /// the calling thread immediately before the caller crosses FFI. Kernel
    /// launchers outside `kiln-hip` must use this accessor so a failed recovery
    /// cannot be bypassed by retaining an otherwise-valid raw handle.
    pub fn hip_stream_for_execution(&self) -> Result<sys::hipStream_t> {
        self.bind()?;
        Ok(self.handle)
    }

    /// The device ordinal this stream is bound to.
    pub fn ordinal(&self) -> usize {
        self.ordinal as usize
    }

    #[inline]
    fn bind(&self) -> Result<()> {
        if self.cleanup_quarantined.load(Ordering::Acquire) {
            return Err(HipError {
                code: -1,
                api: "RocmStream::bind",
                message: "ROCm stream cleanup is quarantined after synchronization failure"
                    .to_string(),
            });
        }
        check(unsafe { sys::hipSetDevice(self.ordinal) }, "hipSetDevice")
    }

    /// Block until all work queued on this stream completes.
    pub fn synchronize(&self) -> Result<()> {
        self.bind()?;
        check(
            unsafe { sys::hipStreamSynchronize(self.handle) },
            "hipStreamSynchronize",
        )
    }

    fn synchronize_for(&self, reason: RocmSyncReason) -> Result<()> {
        let started = Instant::now();
        let result = self.synchronize();
        let waited_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.sync_telemetry
            .record_wait(reason, RocmSyncScope::Stream, waited_ns);
        if result.is_err() {
            self.cleanup_quarantined.store(true, Ordering::Release);
        }
        result
    }

    /// Record `event` after all work currently queued on this stream.
    pub fn record_event(&self, event: &RocmEvent) -> Result<()> {
        event.ensure_same_device(self)?;
        self.bind()?;
        check(
            unsafe { sys::hipEventRecord(event.handle, self.handle) },
            "hipEventRecord",
        )
    }

    /// Queue a dependency on the most recent recording of `event` without
    /// blocking the host thread.
    pub fn wait_event(&self, event: &RocmEvent) -> Result<()> {
        event.ensure_same_device(self)?;
        self.bind()?;
        check(
            unsafe { sys::hipStreamWaitEvent(self.handle, event.handle, 0) },
            "hipStreamWaitEvent",
        )
    }

    /// Allocate `len` bytes on the device, zeroed. Stream-ordered
    /// (`hipMallocAsync`) when supported, falling back to synchronous
    /// `hipMalloc` on arches/runtimes without the stream-ordered allocator.
    pub fn alloc_zeros(self: &Arc<Self>, len: usize) -> Result<RocmSlice> {
        let slice = self.alloc(len)?;
        if len > 0 {
            self.bind()?;
            // SAFETY: slice.ptr is a valid device allocation of `len` bytes.
            check(
                unsafe { sys::hipMemsetD8Async(slice.ptr, 0, len, self.handle) },
                "hipMemsetD8Async",
            )?;
        }
        Ok(slice)
    }

    /// Allocate `len` (uninitialized) bytes on the device. See [`Self::alloc_zeros`].
    pub fn alloc(self: &Arc<Self>, len: usize) -> Result<RocmSlice> {
        self.bind()?;
        // A zero-length allocation is legal and yields a null/!owned slice.
        if len == 0 {
            return Ok(RocmSlice {
                ptr: ptr::null_mut(),
                len: 0,
                async_alloc: false,
                stream: self.clone(),
            });
        }
        let mut ptr: *mut c_void = ptr::null_mut();
        // Prefer the stream-ordered allocator (needed for HIP-graph capture in
        // R.9); fall back to plain hipMalloc if the runtime/arch rejects it.
        let async_rc = unsafe { sys::hipMallocAsync(&mut ptr, len, self.handle) };
        let async_alloc = if async_rc == sys::HIP_SUCCESS {
            true
        } else {
            ptr = ptr::null_mut();
            check(unsafe { sys::hipMalloc(&mut ptr, len) }, "hipMalloc")?;
            false
        };
        Ok(RocmSlice {
            ptr,
            len,
            async_alloc,
            stream: self.clone(),
        })
    }

    /// Copy host bytes into a device slice, then synchronize (the host buffer is
    /// only borrowed for the call, so the async copy must complete first).
    pub fn memcpy_htod(&self, dst: &mut RocmSlice, src: &[u8]) -> Result<()> {
        if src.len() != dst.len {
            return Err(HipError {
                code: -1,
                api: "RocmStream::memcpy_htod",
                message: format!("length mismatch: src {} != dst {}", src.len(), dst.len),
            });
        }
        if src.is_empty() {
            return Ok(());
        }
        self.bind()?;
        // SAFETY: dst.ptr is a valid device allocation of dst.len bytes; src is
        // a valid host buffer of the same length. We synchronize before return.
        check(
            unsafe {
                sys::hipMemcpyHtoDAsync(
                    dst.ptr,
                    src.as_ptr() as *mut c_void,
                    src.len(),
                    self.handle,
                )
            },
            "hipMemcpyHtoDAsync",
        )?;
        self.synchronize_for(RocmSyncReason::AllocationLifetime)
    }

    /// Async H2D copy into a caller-supplied raw device pointer, WITHOUT a
    /// trailing synchronize. The HIP-graph replay path (R.9) uses this to
    /// refresh a graph-stable buffer's contents *in place*: the destination
    /// pointer is the one baked into the captured graph, so it must not change
    /// (no realloc). The copy is queued on this stream and is ordered before
    /// any subsequent launch on the same stream.
    ///
    /// Unlike [`Self::memcpy_htod`], this does NOT synchronize — the caller is
    /// responsible for (a) keeping `src` alive until the copy completes and
    /// (b) synchronizing (this stream, or the launch's stream after an event)
    /// before the host or another stream reads the destination.
    ///
    /// # Safety
    /// `dst` must point to at least `src.len()` bytes of a live device
    /// allocation reachable from this stream's device.
    pub unsafe fn memcpy_htod_raw_async(&self, dst: *mut c_void, src: &[u8]) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        self.bind()?;
        // SAFETY: caller guarantees `dst` addresses >= src.len() live device
        // bytes; `src` is a valid host slice of the same length.
        check(
            unsafe {
                sys::hipMemcpyHtoDAsync(dst, src.as_ptr() as *mut c_void, src.len(), self.handle)
            },
            "hipMemcpyHtoDAsync",
        )
    }

    /// Zero `len` bytes at a caller-supplied raw device pointer on this stream,
    /// WITHOUT synchronizing. The HIP-graph capture arena (R.9) uses this during
    /// the replay (capture) pass: issued on the active capture stream it is
    /// RECORDED into the graph, so every replay re-zeros the read-before-write
    /// arena buffers. The ROCm analog of cudarc's `result::memset_d8_async`.
    ///
    /// # Safety
    /// `dst` must point to at least `len` bytes of a live device allocation
    /// reachable from this stream's device.
    pub unsafe fn memset_zero_async(&self, dst: *mut c_void, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        self.bind()?;
        // SAFETY: caller guarantees `dst` addresses >= len live device bytes.
        check(
            unsafe { sys::hipMemsetD8Async(dst, 0, len, self.handle) },
            "hipMemsetD8Async",
        )
    }

    /// Allocate a device buffer of `src.len()` bytes and copy `src` into it
    /// (H2D), synchronizing before return. The one-shot analog of cudarc's
    /// `CudaStream::clone_htod`.
    pub fn clone_htod(self: &Arc<Self>, src: &[u8]) -> Result<RocmSlice> {
        let mut slice = self.alloc(src.len())?;
        self.memcpy_htod(&mut slice, src)?;
        Ok(slice)
    }

    /// Copy a device slice back to a freshly allocated host `Vec`, synchronizing
    /// before returning.
    pub fn memcpy_dtoh(&self, src: &RocmSlice) -> Result<Vec<u8>> {
        let mut out = vec![0u8; src.len];
        if src.len == 0 {
            return Ok(out);
        }
        self.bind()?;
        // SAFETY: src.ptr is a valid device allocation of src.len bytes; out is
        // a host buffer of the same length. Synchronized before return.
        check(
            unsafe {
                sys::hipMemcpyDtoHAsync(
                    out.as_mut_ptr() as *mut c_void,
                    src.ptr,
                    src.len,
                    self.handle,
                )
            },
            "hipMemcpyDtoHAsync",
        )?;
        self.synchronize_for(RocmSyncReason::HostReadback)?;
        Ok(out)
    }

    /// Copy a caller-supplied device byte range back to a fresh host `Vec`,
    /// synchronizing before returning. Unlike [`Self::memcpy_dtoh`], this can
    /// read a validated subrange of a larger allocation without first creating
    /// a temporary device buffer.
    ///
    /// # Safety
    /// `src` must point to at least `len` bytes of a live device allocation
    /// reachable from this stream's device. The allocation must remain live
    /// until this method returns.
    pub unsafe fn memcpy_dtoh_raw(&self, src: *const c_void, len: usize) -> Result<Vec<u8>> {
        let mut out = vec![0u8; len];
        if len == 0 {
            return Ok(out);
        }
        self.bind()?;
        // SAFETY: the caller guarantees `src` addresses at least `len` live
        // device bytes; `out` owns exactly `len` writable host bytes. The
        // stream synchronization below completes the copy before either
        // pointer can become invalid.
        check(
            unsafe {
                sys::hipMemcpyDtoHAsync(
                    out.as_mut_ptr() as *mut c_void,
                    src as *mut c_void,
                    len,
                    self.handle,
                )
            },
            "hipMemcpyDtoHAsync",
        )?;
        self.synchronize_for(RocmSyncReason::HostReadback)?;
        Ok(out)
    }

    /// Device-to-device copy on this stream (async; caller orders via the
    /// stream). `dst` and `src` must have equal length.
    pub fn memcpy_dtod(&self, dst: &mut RocmSlice, src: &RocmSlice) -> Result<()> {
        if src.len != dst.len {
            return Err(HipError {
                code: -1,
                api: "RocmStream::memcpy_dtod",
                message: format!("length mismatch: src {} != dst {}", src.len, dst.len),
            });
        }
        if src.len == 0 {
            return Ok(());
        }
        self.bind()?;
        // SAFETY: both are valid device allocations of equal length.
        check(
            unsafe { sys::hipMemcpyDtoDAsync(dst.ptr, src.ptr, src.len, self.handle) },
            "hipMemcpyDtoDAsync",
        )
    }

    /// Async D2D copy between caller-supplied raw device pointers, WITHOUT a
    /// trailing synchronize. The copy is queued on this stream and ordered with
    /// subsequent work on the same stream.
    ///
    /// # Safety
    /// `dst` and `src` must each point to at least `len` bytes of live device
    /// allocations reachable from this stream's device. The regions must not
    /// overlap in a way that violates HIP memcpy requirements.
    pub unsafe fn memcpy_dtod_raw_async(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        self.bind()?;
        // SAFETY: caller guarantees both raw pointers address >= len live
        // device bytes on this stream's device.
        check(
            unsafe { sys::hipMemcpyDtoDAsync(dst, src as *mut c_void, len, self.handle) },
            "hipMemcpyDtoDAsync",
        )
    }
}

impl Drop for RocmStream {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if !bind_device_for_cleanup(self.ordinal, &self.cleanup_quarantined, "RocmStream::drop") {
            self.handle = ptr::null_mut();
            return;
        }
        // SAFETY: handle was created by hipStreamCreate* and not yet destroyed.
        let rc = unsafe { sys::hipStreamDestroy(self.handle) };
        if rc != sys::HIP_SUCCESS {
            self.cleanup_quarantined.store(true, Ordering::Release);
            eprintln!("RocmStream::drop: hipStreamDestroy failed (hipError {rc})");
            warn_cleanup_quarantine("RocmStream::drop");
        }
        self.handle = ptr::null_mut();
    }
}

impl Drop for RocmEvent {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if !bind_device_for_cleanup(self.ordinal, &self.cleanup_quarantined, "RocmEvent::drop") {
            self.handle = ptr::null_mut();
            return;
        }
        let rc = unsafe { sys::hipEventDestroy(self.handle) };
        if rc != sys::HIP_SUCCESS {
            self.cleanup_quarantined.store(true, Ordering::Release);
            eprintln!("RocmEvent::drop: hipEventDestroy failed (hipError {rc})");
            warn_cleanup_quarantine("RocmEvent::drop");
        }
        self.handle = ptr::null_mut();
    }
}

// ---------------------------------------------------------------------------
// Slice (device allocation)
// ---------------------------------------------------------------------------

/// An owned device byte buffer. The analog of `cudarc::driver::CudaSlice<u8>`.
///
/// Holds an `Arc<RocmStream>` so `Drop` can free on the same stream it was
/// allocated on (`hipFreeAsync` for stream-ordered allocations, `hipFree`
/// otherwise). Device pointers are plain `void*` normalized to a single
/// accessor so the Phase R.3 `SliceOwner::Borrowed { ptr, .. }` retype is clean.
#[derive(Debug)]
pub struct RocmSlice {
    ptr: *mut c_void,
    len: usize,
    async_alloc: bool,
    stream: Arc<RocmStream>,
}

// SAFETY: the device pointer is bound to one device and not aliased mutably
// across threads by this type; ownership is unique. Same reasoning as cudarc's
// CudaSlice (which is Send + Sync).
unsafe impl Send for RocmSlice {}
unsafe impl Sync for RocmSlice {}

impl RocmSlice {
    /// The raw device pointer (`hipDeviceptr_t == void*`). Valid while `self` is
    /// alive. The single normalized accessor referenced by the storage layer.
    pub fn device_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// The device pointer as a `usize` — convenient for the `SliceOwner` retype
    /// and for passing to `extern "C"` kernel launchers as a `u64`/pointer.
    pub fn device_ptr_usize(&self) -> usize {
        self.ptr as usize
    }

    /// Physical byte length of the allocation.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the allocation is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The stream this slice was allocated on (and is freed on).
    pub fn stream(&self) -> &Arc<RocmStream> {
        &self.stream
    }
}

impl Drop for RocmSlice {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        if !bind_device_for_cleanup(
            self.stream.ordinal,
            &self.stream.cleanup_quarantined,
            "RocmSlice::drop",
        ) {
            self.ptr = ptr::null_mut();
            return;
        }
        // SAFETY: ptr was produced by hipMallocAsync/hipMalloc on self.stream
        // and not yet freed. Free with the matching API.
        let rc = if self.async_alloc {
            unsafe { sys::hipFreeAsync(self.ptr, self.stream.handle) }
        } else {
            unsafe { sys::hipFree(self.ptr) }
        };
        if rc != sys::HIP_SUCCESS {
            self.stream
                .cleanup_quarantined
                .store(true, Ordering::Release);
            eprintln!("RocmSlice::drop: hipFree failed (hipError {rc})");
            warn_cleanup_quarantine("RocmSlice::drop");
        }
        self.ptr = ptr::null_mut();
    }
}

// ---------------------------------------------------------------------------
// Graph capture (wired into kiln-graph-rocm in Phase R.9)
// ---------------------------------------------------------------------------

/// RAII owner of a captured `hipGraph_t`. Analog of `cudarc::driver::CudaGraph`.
#[derive(Debug)]
pub struct RocmGraph {
    graph: sys::hipGraph_t,
    ordinal: c_int,
    cleanup_quarantined: Arc<AtomicBool>,
}

unsafe impl Send for RocmGraph {}
unsafe impl Sync for RocmGraph {}

/// RAII owner of an instantiated `hipGraphExec_t`. Analog of `CudaGraphExec`.
#[derive(Debug)]
pub struct RocmGraphExec {
    exec: sys::hipGraphExec_t,
    ordinal: c_int,
    cleanup_quarantined: Arc<AtomicBool>,
}

unsafe impl Send for RocmGraphExec {}
unsafe impl Sync for RocmGraphExec {}

impl RocmStream {
    /// Begin capturing work issued on this stream into a graph
    /// (`hipStreamBeginCapture`, relaxed mode — matches the CUDA path).
    pub fn begin_capture(&self) -> Result<()> {
        self.bind()?;
        check(
            unsafe {
                sys::hipStreamBeginCapture(self.handle, sys::HIP_STREAM_CAPTURE_MODE_RELAXED)
            },
            "hipStreamBeginCapture",
        )
    }

    /// End capture and return the resulting graph (`hipStreamEndCapture`).
    pub fn end_capture(&self) -> Result<RocmGraph> {
        self.bind()?;
        let mut graph: sys::hipGraph_t = ptr::null_mut();
        check(
            unsafe { sys::hipStreamEndCapture(self.handle, &mut graph) },
            "hipStreamEndCapture",
        )?;
        Ok(RocmGraph {
            graph,
            ordinal: self.ordinal,
            cleanup_quarantined: Arc::clone(&self.cleanup_quarantined),
        })
    }

    /// Whether a capture is currently active on this stream.
    pub fn is_capturing(&self) -> Result<bool> {
        self.bind()?;
        let mut status: c_uint = 0;
        check(
            unsafe { sys::hipStreamIsCapturing(self.handle, &mut status) },
            "hipStreamIsCapturing",
        )?;
        Ok(status == sys::HIP_STREAM_CAPTURE_STATUS_ACTIVE)
    }
}

impl RocmGraph {
    /// Instantiate into an executable graph.
    ///
    /// Uses `flags = 0` (plain instantiation). The R.9 decode graph
    /// pre-allocates every buffer it touches OUTSIDE the capture window (the
    /// freeze-pointers arena), so the captured graph contains NO stream-ordered
    /// alloc nodes — `AUTO_FREE_ON_LAUNCH` (the CUDA discipline) has nothing to
    /// free and was rejected with `hipErrorInvalidValue` on gfx1151 / ROCm 7.2.4.
    pub fn instantiate(&self) -> Result<RocmGraphExec> {
        if self.cleanup_quarantined.load(Ordering::Acquire) {
            return Err(HipError {
                code: -1,
                api: "RocmGraph::instantiate",
                message: "ROCm device is quarantined after a fatal execution failure; restart the process"
                    .to_string(),
            });
        }
        check(unsafe { sys::hipSetDevice(self.ordinal) }, "hipSetDevice")?;
        let mut exec: sys::hipGraphExec_t = ptr::null_mut();
        check(
            unsafe { sys::hipGraphInstantiateWithFlags(&mut exec, self.graph, 0) },
            "hipGraphInstantiateWithFlags",
        )?;
        Ok(RocmGraphExec {
            exec,
            ordinal: self.ordinal,
            cleanup_quarantined: Arc::clone(&self.cleanup_quarantined),
        })
    }
}

impl Drop for RocmGraph {
    fn drop(&mut self) {
        if !self.graph.is_null() {
            if !bind_device_for_cleanup(self.ordinal, &self.cleanup_quarantined, "RocmGraph::drop")
            {
                self.graph = ptr::null_mut();
                return;
            }
            let rc = unsafe { sys::hipGraphDestroy(self.graph) };
            if rc != sys::HIP_SUCCESS {
                self.cleanup_quarantined.store(true, Ordering::Release);
                eprintln!("RocmGraph::drop: hipGraphDestroy failed (hipError {rc})");
                warn_cleanup_quarantine("RocmGraph::drop");
            }
            self.graph = ptr::null_mut();
        }
    }
}

impl RocmGraphExec {
    /// Launch the executable graph on `stream`.
    pub fn launch(&self, stream: &RocmStream) -> Result<()> {
        if self.ordinal != stream.ordinal {
            return Err(HipError {
                code: -1,
                api: "RocmGraphExec::launch",
                message: format!(
                    "graph executable belongs to device {} but stream belongs to device {}",
                    self.ordinal, stream.ordinal
                ),
            });
        }
        stream.bind()?;
        check(
            unsafe { sys::hipGraphLaunch(self.exec, stream.handle) },
            "hipGraphLaunch",
        )
    }
}

impl Drop for RocmGraphExec {
    fn drop(&mut self) {
        if !self.exec.is_null() {
            if !bind_device_for_cleanup(
                self.ordinal,
                &self.cleanup_quarantined,
                "RocmGraphExec::drop",
            ) {
                self.exec = ptr::null_mut();
                return;
            }
            let rc = unsafe { sys::hipGraphExecDestroy(self.exec) };
            if rc != sys::HIP_SUCCESS {
                self.cleanup_quarantined.store(true, Ordering::Release);
                eprintln!("RocmGraphExec::drop: hipGraphExecDestroy failed (hipError {rc})");
                warn_cleanup_quarantine("RocmGraphExec::drop");
            }
            self.exec = ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — run only where a real HIP device is present; skip otherwise (mirrors
// the cuda_stream_priority.rs `try_ctx` pattern).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_policy_defaults_to_legacy_host_barriers() {
        assert_eq!(
            RocmExecutionPolicy::default().synchronization_mode,
            RocmSynchronizationMode::LegacyHostBarriers
        );
    }

    #[test]
    fn sync_reason_metrics_are_fixed_and_accounted() {
        let mut labels = std::collections::HashSet::new();
        for reason in RocmSyncReason::ALL {
            assert!(
                labels.insert(reason.as_str()),
                "duplicate sync reason label"
            );
        }
        assert_eq!(labels.len(), ROCM_SYNC_REASON_COUNT);

        let telemetry = RocmSyncTelemetry::default();
        telemetry.record_wait(RocmSyncReason::ExternalYield, RocmSyncScope::Stream, 17);
        telemetry.record_skipped(RocmSyncReason::ActivationOutput);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.total_wait_count(), 1);
        assert_eq!(snapshot.total_waited_ns(), 17);
        assert_eq!(snapshot.total_skipped_count(), 1);
    }

    fn try_ctx() -> Option<Arc<RocmContext>> {
        if !is_available() {
            eprintln!("ROCm device not available; skipping");
            return None;
        }
        RocmContext::new(0).ok()
    }

    #[test]
    fn runtime_and_count_are_sane() {
        let Some(_ctx) = try_ctx() else { return };
        let v = runtime_version().expect("hipRuntimeGetVersion");
        assert!(v > 0, "expected a positive HIP runtime version, got {v}");
        assert!(device_count().unwrap() >= 1);
    }

    #[test]
    fn priority_range_well_formed() {
        let Some(_ctx) = try_ctx() else { return };
        let (least, greatest) = stream_priority_range().expect("priority range");
        assert!(
            greatest <= least,
            "greatest {greatest} should be <= least {least}"
        );
    }

    #[test]
    fn alloc_memset_roundtrip() {
        let Some(ctx) = try_ctx() else { return };
        let stream = ctx.default_stream();
        // zeros
        let z = stream.alloc_zeros(256).expect("alloc_zeros");
        assert_eq!(z.len(), 256);
        assert!(!z.device_ptr().is_null());
        let host = stream.memcpy_dtoh(&z).expect("dtoh");
        assert!(
            host.iter().all(|&b| b == 0),
            "alloc_zeros must zero the buffer"
        );
    }

    #[test]
    fn htod_dtoh_roundtrip() {
        let Some(ctx) = try_ctx() else { return };
        let stream = ctx.default_stream();
        let before = ctx.sync_telemetry_snapshot();
        let src: Vec<u8> = (0..1024u32).map(|i| (i % 251) as u8).collect();
        let mut dev = stream.alloc(src.len()).expect("alloc");
        stream.memcpy_htod(&mut dev, &src).expect("htod");
        let back = stream.memcpy_dtoh(&dev).expect("dtoh");
        assert_eq!(src, back, "H2D->D2H must round-trip the bytes exactly");
        let after = ctx.sync_telemetry_snapshot();
        let allocation_lifetime = RocmSyncReason::AllocationLifetime.index();
        let host_readback = RocmSyncReason::HostReadback.index();
        assert_eq!(
            after.reasons[allocation_lifetime].stream_wait_count,
            before.reasons[allocation_lifetime]
                .stream_wait_count
                .saturating_add(1)
        );
        assert_eq!(
            after.reasons[host_readback].stream_wait_count,
            before.reasons[host_readback]
                .stream_wait_count
                .saturating_add(1)
        );
    }

    #[test]
    fn dtod_roundtrip() {
        let Some(ctx) = try_ctx() else { return };
        let stream = ctx.default_stream();
        let src_host: Vec<u8> = (0..512u32).map(|i| (i * 7 % 251) as u8).collect();
        let mut a = stream.alloc(src_host.len()).expect("alloc a");
        stream.memcpy_htod(&mut a, &src_host).expect("htod");
        let mut b = stream.alloc(src_host.len()).expect("alloc b");
        stream.memcpy_dtod(&mut b, &a).expect("dtod");
        stream.synchronize().expect("sync");
        let back = stream.memcpy_dtoh(&b).expect("dtoh");
        assert_eq!(src_host, back, "D2D copy must preserve bytes");
    }

    #[test]
    fn event_orders_work_across_streams() {
        let Some(ctx) = try_ctx() else { return };
        let producer = ctx.default_stream();
        let consumer = ctx.new_stream().expect("consumer stream");
        let input_ready = ctx.new_event().expect("input-ready event");
        let copy_ready = ctx.new_event().expect("copy-ready event");
        let expected: Vec<u8> = (0..4096u32).map(|i| (i * 13 % 251) as u8).collect();
        let mut src = producer.alloc(expected.len()).expect("source allocation");
        let dst = producer
            .alloc_zeros(expected.len())
            .expect("destination allocation");

        producer
            .memcpy_htod(&mut src, &expected)
            .expect("source upload");
        producer
            .record_event(&input_ready)
            .expect("record source readiness");
        consumer
            .wait_event(&input_ready)
            .expect("wait for source readiness");
        unsafe {
            consumer
                .memcpy_dtod_raw_async(dst.device_ptr(), src.device_ptr(), expected.len())
                .expect("cross-stream copy");
        }
        consumer
            .record_event(&copy_ready)
            .expect("record copy readiness");
        producer
            .wait_event(&copy_ready)
            .expect("wait for copy readiness");

        let actual = producer.memcpy_dtoh(&dst).expect("ordered download");
        assert_eq!(actual, expected);
    }

    #[test]
    fn new_stream_with_priority_creates() {
        let Some(ctx) = try_ctx() else { return };
        let (least, greatest) = stream_priority_range().expect("range");
        let hi = ctx
            .new_stream_with_priority(greatest)
            .expect("high-priority stream");
        let lo = ctx
            .new_stream_with_priority(least)
            .expect("low-priority stream");
        assert!(!hi.hip_stream().is_null());
        assert!(!lo.hip_stream().is_null());
    }
}
