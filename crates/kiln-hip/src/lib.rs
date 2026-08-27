//! `kiln-hip` — bounded, safe Rust bindings to the AMD ROCm/HIP runtime.
//!
//! This is the **cudarc analog** for kiln's ROCm backend (Phase R.1). It mirrors
//! the bounded context/stream/allocation/graph surface the CUDA substrate uses
//! so `rocm_storage.rs` and the allocator layers can share the same ownership
//! model without exposing unchecked runtime handles.
//!
//! Design mirrors `kiln-tensor/src/cuda_stream_priority.rs`: own the raw HIP
//! handle, implement `Drop`, expose raw streams only through an owning
//! submission token, and carry `unsafe impl Send + Sync` with the same
//! justification cudarc uses.
//!
//! The crate compiles on hosts with no ROCm toolchain (the FFI block has no
//! `#[link]`; `build.rs` links `amdhip64` only when ROCm is present). Calling a
//! function with no runtime present returns `Err(HipError)` rather than
//! aborting — except linking, which only a ROCm host performs.

pub mod sys;

use std::collections::HashMap;
use std::ffi::CStr;
use std::fmt;
use std::marker::PhantomData;
use std::os::raw::{c_int, c_uint, c_void};
use std::ptr;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Result alias for HIP calls.
pub type Result<T> = std::result::Result<T, HipError>;

static CLEANUP_QUARANTINE_DROP_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);

const EXECUTION_GATE_STOPPED: usize = 1usize << (usize::BITS - 1);
const EXECUTION_GATE_FINAL: usize = 1usize << (usize::BITS - 2);
const EXECUTION_GATE_ACTIVE_MASK: usize = EXECUTION_GATE_FINAL - 1;
const EXECUTION_SETTLEMENT_TIMEOUT: Duration = Duration::from_secs(5);

/// Per-device admission gate shared by every context and resource in the
/// process. The high bits stop admission and mark final quarantine; the low
/// bits count host threads currently crossing a HIP FFI boundary.
#[derive(Debug)]
struct RocmExecutionGate {
    state: AtomicUsize,
    settled_lock: Mutex<()>,
    settled: Condvar,
    recovery_lock: Mutex<()>,
}

impl Default for RocmExecutionGate {
    fn default() -> Self {
        Self {
            state: AtomicUsize::new(0),
            settled_lock: Mutex::new(()),
            settled: Condvar::new(),
            recovery_lock: Mutex::new(()),
        }
    }
}

impl RocmExecutionGate {
    fn try_acquire(self: &Arc<Self>, api: &'static str) -> Result<RocmExecutionPermit> {
        let mut observed = self.state.load(Ordering::Acquire);
        loop {
            if observed & EXECUTION_GATE_STOPPED != 0 {
                return Err(HipError {
                    code: -1,
                    api,
                    message: "ROCm device is quarantined after a fatal execution failure; restart the process"
                        .to_string(),
                });
            }
            let active = observed & EXECUTION_GATE_ACTIVE_MASK;
            if active == EXECUTION_GATE_ACTIVE_MASK {
                return Err(HipError {
                    code: -1,
                    api,
                    message: "ROCm execution admission counter overflow".to_string(),
                });
            }
            match self.state.compare_exchange_weak(
                observed,
                observed + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Ok(RocmExecutionPermit {
                        gate: Arc::clone(self),
                        active: true,
                        _not_send: PhantomData,
                    });
                }
                Err(actual) => observed = actual,
            }
        }
    }

    fn request_quarantine(&self) {
        let previous = self
            .state
            .fetch_or(EXECUTION_GATE_STOPPED, Ordering::AcqRel);
        if previous & EXECUTION_GATE_ACTIVE_MASK == 0 {
            self.mark_final();
        }
    }

    fn mark_final(&self) {
        // Serialize the predicate transition with waiter enrollment. Without
        // this lock a waiter could observe !FINAL, lose the notification just
        // before wait_timeout, and sleep until the full recovery deadline.
        let _settled = self
            .settled_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.state.fetch_or(EXECUTION_GATE_FINAL, Ordering::Release);
        self.settled.notify_all();
    }

    fn release(&self) {
        let previous = self.state.fetch_sub(1, Ordering::AcqRel);
        debug_assert_ne!(previous & EXECUTION_GATE_ACTIVE_MASK, 0);
        if previous & EXECUTION_GATE_STOPPED != 0 && previous & EXECUTION_GATE_ACTIVE_MASK == 1 {
            self.mark_final();
        }
    }

    fn execution_stopped(&self) -> bool {
        self.state.load(Ordering::Acquire) & EXECUTION_GATE_STOPPED != 0
    }

    fn require_stopped_for_error_recovery(&self) -> Result<()> {
        if self.execution_stopped() {
            return Ok(());
        }
        Err(HipError {
            code: -1,
            api: "RocmContext::synchronize_device_for(ErrorRecovery)",
            message: "ErrorRecovery requires execution quarantine to be published before the drain; use CaptureRollback for recoverable gate-open settlement"
                .to_string(),
        })
    }

    fn wait_until_final_for(&self, timeout: Duration) -> bool {
        if self.state.load(Ordering::Acquire) & EXECUTION_GATE_FINAL != 0 {
            return true;
        }
        let started = Instant::now();
        let mut settled = self
            .settled_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        while self.state.load(Ordering::Acquire) & EXECUTION_GATE_FINAL == 0 {
            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                return false;
            }
            let (next, wait_result) = self
                .settled
                .wait_timeout(settled, remaining)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            settled = next;
            if wait_result.timed_out()
                && self.state.load(Ordering::Acquire) & EXECUTION_GATE_FINAL == 0
            {
                return false;
            }
        }
        true
    }
}

/// Admission token for exactly one host-side HIP submission. It is deliberately
/// not `Send`: `hipSetDevice` binds the acquiring host thread.
#[derive(Debug)]
struct RocmExecutionPermit {
    gate: Arc<RocmExecutionGate>,
    active: bool,
    _not_send: PhantomData<Rc<()>>,
}

impl RocmExecutionPermit {
    fn quarantine(&self) {
        self.gate.request_quarantine();
    }
}

impl Drop for RocmExecutionPermit {
    fn drop(&mut self) {
        if self.active {
            // A Rust unwind can bypass the status-classification statement
            // immediately following an FFI call. Conservatively publish STOP
            // before releasing this admission; normal non-panicking internal
            // calls remain lightweight, while public submissions additionally
            // require explicit complete/quarantine on every path.
            if std::thread::panicking() {
                self.gate.request_quarantine();
            }
            self.active = false;
            self.gate.release();
        }
    }
}

/// Owns host memory for the complete lifetime of an admitted async copy and its
/// mandatory settlement wait. If control exits before settlement is classified,
/// the host allocation is retained for process lifetime and execution is
/// quarantined rather than risking a driver use-after-free.
struct RocmAdmittedHostTransfer<T> {
    permit: RocmExecutionPermit,
    host: Option<T>,
    classified: bool,
}

impl<T> RocmAdmittedHostTransfer<T> {
    fn new(permit: RocmExecutionPermit, host: T) -> Self {
        Self {
            permit,
            host: Some(host),
            classified: false,
        }
    }

    fn permit(&self) -> &RocmExecutionPermit {
        &self.permit
    }

    fn host(&self) -> &T {
        self.host.as_ref().expect("host transfer buffer is live")
    }

    fn host_mut(&mut self) -> &mut T {
        self.host.as_mut().expect("host transfer buffer is live")
    }

    /// Finish a classified settlement. Host memory is returned only when the
    /// stream wait proved that HIP no longer owns it; otherwise it is leaked.
    fn finish(mut self, host_memory_settled: bool) -> Option<T> {
        let host = self.host.take();
        self.classified = true;
        if host_memory_settled {
            host
        } else {
            self.permit.quarantine();
            if let Some(host) = host {
                std::mem::forget(host);
            }
            None
        }
    }
}

impl<T> Drop for RocmAdmittedHostTransfer<T> {
    fn drop(&mut self) {
        if self.classified {
            return;
        }
        self.permit.quarantine();
        if let Some(host) = self.host.take() {
            std::mem::forget(host);
        }
    }
}

fn device_execution_gate(ordinal: c_int) -> Arc<RocmExecutionGate> {
    static GATES: OnceLock<Mutex<HashMap<c_int, Arc<RocmExecutionGate>>>> = OnceLock::new();
    let mut gates = GATES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    Arc::clone(
        gates
            .entry(ordinal)
            .or_insert_with(|| Arc::new(RocmExecutionGate::default())),
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
    execution_gate: &Arc<RocmExecutionGate>,
    resource: &str,
) -> Option<RocmExecutionPermit> {
    let permit = match execution_gate.try_acquire("ROCm cleanup") {
        Ok(permit) => permit,
        Err(_) => {
            warn_cleanup_quarantine(resource);
            return None;
        }
    };
    let rc = unsafe { sys::hipSetDevice(ordinal) };
    if rc != sys::HIP_SUCCESS {
        permit.quarantine();
        eprintln!("{resource}: hipSetDevice failed during cleanup (hipError {rc})");
        warn_cleanup_quarantine(resource);
        return None;
    }
    Some(permit)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HipCallFailureClass {
    PureQuery,
    OptionalConfiguration,
    PostDrainPoolMaintenance,
    CaptureState,
    ExecutionMutation,
}

impl HipCallFailureClass {
    const fn quarantines(self) -> bool {
        matches!(self, Self::CaptureState | Self::ExecutionMutation)
    }
}

fn check_call_status(
    permit: &RocmExecutionPermit,
    code: sys::hipError_t,
    api: &'static str,
    class: HipCallFailureClass,
) -> Result<()> {
    // Publish STOP from the direct return code before querying an error string
    // or clearing HIP's sticky slot. The permit remains active through all
    // classification work, but no concurrent caller may enter after this point.
    if code != sys::HIP_SUCCESS && class.quarantines() {
        permit.quarantine();
    }
    check(code, api)
}

/// Check a device-affecting call whose failure can mean an earlier async fault
/// or uncertain partial mutation.
fn check_execution_mutation(
    permit: &RocmExecutionPermit,
    code: sys::hipError_t,
    api: &'static str,
) -> Result<()> {
    check_call_status(permit, code, api, HipCallFailureClass::ExecutionMutation)
}

fn host_transfer_result(
    enqueue: Result<()>,
    settlement: Result<()>,
    api: &'static str,
) -> Result<()> {
    match (enqueue, settlement) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(enqueue_error), Err(settlement_error)) => Err(HipError {
            code: settlement_error.code,
            api,
            message: format!(
                "async copy enqueue failed ({enqueue_error}); stream settlement also failed \
                 ({settlement_error})"
            ),
        }),
    }
}

fn clean_no_publication_error(code: sys::hipError_t) -> bool {
    matches!(
        code,
        sys::HIP_ERROR_INVALID_VALUE | sys::HIP_ERROR_OUT_OF_MEMORY | sys::HIP_ERROR_NOT_SUPPORTED
    )
}

fn resource_creation_status_is_fatal(code: sys::hipError_t, handle_is_null: bool) -> bool {
    if code == sys::HIP_SUCCESS {
        return handle_is_null;
    }
    !handle_is_null || !clean_no_publication_error(code)
}

fn async_allocation_fallback_allowed(code: sys::hipError_t, pointer_is_null: bool) -> bool {
    pointer_is_null
        && matches!(
            code,
            sys::HIP_ERROR_INVALID_VALUE | sys::HIP_ERROR_NOT_SUPPORTED
        )
}

/// Resource creation may fail recoverably only when the runtime published no
/// handle and returned a documented local/OOM/unsupported code. A non-null
/// handle paired with failure is an ambiguous partial publication and poisons
/// execution so cleanup cannot race it.
fn check_resource_creation(
    permit: &RocmExecutionPermit,
    code: sys::hipError_t,
    handle_is_null: bool,
    api: &'static str,
) -> Result<()> {
    if code == sys::HIP_SUCCESS {
        if resource_creation_status_is_fatal(code, handle_is_null) {
            permit.quarantine();
            return Err(HipError {
                code: -1,
                api,
                message: "HIP reported successful resource creation without publishing a handle"
                    .to_string(),
            });
        }
        return Ok(());
    }
    if resource_creation_status_is_fatal(code, handle_is_null) {
        permit.quarantine();
    }
    check(code, api)
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
    Err(HipError { code, api, message })
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

/// Route policy for hipBLASLt strided-batched matmul.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RocmStridedBatchedMatmulMode {
    /// Historical qualification-only shape and dtype guard.
    #[cfg(any(test, feature = "hardware-qualification"))]
    Auto,
    /// Always use hipBLASLt strided batching when the logical batch is larger
    /// than one. This is an experimental correctness-comparison route.
    Enabled,
    /// Always issue one hipBLASLt operation per logical batch row.
    #[default]
    Disabled,
}

/// Output route for BF16-input, BF16-output hipBLASLt matmul.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RocmBf16MatmulOutputMode {
    /// Historical qualification-only size guard.
    #[cfg(any(test, feature = "hardware-qualification"))]
    Auto,
    /// Always request native BF16 output from hipBLASLt.
    NativeBf16,
    /// Always request F32 output and cast it to BF16 on the device.
    #[default]
    F32ThenCast,
}

/// Immutable ROCm matmul route policy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RocmMatmulPolicy {
    pub strided_batched_mode: RocmStridedBatchedMatmulMode,
    pub bf16_output_mode: RocmBf16MatmulOutputMode,
}

impl RocmMatmulPolicy {
    pub const fn new(
        strided_batched_mode: RocmStridedBatchedMatmulMode,
        bf16_output_mode: RocmBf16MatmulOutputMode,
    ) -> Self {
        Self {
            strided_batched_mode,
            bf16_output_mode,
        }
    }
}

/// Selection discipline for a ROCm flash-attention route.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RocmFlashAttentionRouteMode {
    /// Apply the route's device-neutral shape and memory-admission guards.
    #[default]
    Auto,
    /// Prefer the route whenever its hard correctness and memory guards pass.
    Enabled,
    /// Do not prefer the route. Exact fallback paths remain available.
    Disabled,
}

/// Immutable ROCm flash-attention route and geometry policy.
///
/// The fields normalize the former collection of positive, negative, force,
/// and threshold environment variables into one closed policy. They are read
/// from the input tensor's owning [`RocmContext`] before dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RocmFlashAttentionPolicy {
    pub f32_matmul_inner_tile: usize,
    pub online_matmul_batch_group: usize,
    pub native_scalar_forward: bool,
    pub native_scalar_forward_max_sequence: usize,
    pub native_single_forward_max_sequence: usize,
    pub native_tiled_forward: bool,
    pub native_forward_query_tile: usize,
    pub native_streaming_forward: bool,
    pub native_streaming_forward_min_sequence: usize,
    pub native_streaming_forward_key_tile: usize,
    pub native_rectangular_causal_forward: bool,
    pub online_forward: bool,
    pub online_backward: bool,
    pub materialized_backward_mode: RocmFlashAttentionRouteMode,
    pub native_backward_preference: RocmFlashAttentionRouteMode,
    pub native_backward_d128_max_sequence: usize,
    pub native_backward_d256_max_sequence: usize,
    pub native_backward_long_min_sequence: usize,
    pub collapsed_gqa_backward: bool,
    pub native_direct_collapsed_gqa_backward: bool,
    pub native_gqa_qblock_forward: bool,
    pub native_gqa_qblock_forward_min_sequence: usize,
    pub wmma_gqa_qblock_forward: bool,
    pub wmma_gqa_r64k32_forward: bool,
    pub wmma_gqa_r64k32_forward_min_sequence: usize,
    pub wmma_gqa_r64k32_log2_forward: bool,
    pub wmma_gqa_r64k32_log2_forward_min_sequence: usize,
    pub backward_precompute_delta_max_sequence: usize,
    pub native_direct_collapsed_gqa_query_parallelism: usize,
}

impl RocmFlashAttentionPolicy {
    /// Historical Strix Halo flash-attention qualification fixture.
    #[cfg(any(test, feature = "hardware-qualification"))]
    pub const fn qualified() -> Self {
        Self {
            f32_matmul_inner_tile: 4096,
            online_matmul_batch_group: 4,
            native_scalar_forward: true,
            native_scalar_forward_max_sequence: 4096,
            native_single_forward_max_sequence: 32768,
            native_tiled_forward: true,
            native_forward_query_tile: 4096,
            native_streaming_forward: true,
            native_streaming_forward_min_sequence: 8192,
            native_streaming_forward_key_tile: 4096,
            native_rectangular_causal_forward: true,
            online_forward: true,
            online_backward: true,
            materialized_backward_mode: RocmFlashAttentionRouteMode::Auto,
            native_backward_preference: RocmFlashAttentionRouteMode::Auto,
            native_backward_d128_max_sequence: 1024,
            native_backward_d256_max_sequence: 512,
            native_backward_long_min_sequence: 4096,
            collapsed_gqa_backward: true,
            native_direct_collapsed_gqa_backward: true,
            native_gqa_qblock_forward: true,
            native_gqa_qblock_forward_min_sequence: 256,
            wmma_gqa_qblock_forward: true,
            wmma_gqa_r64k32_forward: true,
            wmma_gqa_r64k32_forward_min_sequence: 256,
            wmma_gqa_r64k32_log2_forward: true,
            wmma_gqa_r64k32_log2_forward_min_sequence: 256,
            backward_precompute_delta_max_sequence: 1024,
            native_direct_collapsed_gqa_query_parallelism: 1,
        }
    }

    /// Reference-oriented policy retaining exact, bounded composite routes.
    pub const fn portable_fallback() -> Self {
        Self {
            f32_matmul_inner_tile: 4096,
            online_matmul_batch_group: 4,
            native_scalar_forward: false,
            native_scalar_forward_max_sequence: 4096,
            native_single_forward_max_sequence: 32768,
            native_tiled_forward: false,
            native_forward_query_tile: 4096,
            native_streaming_forward: false,
            native_streaming_forward_min_sequence: 8192,
            native_streaming_forward_key_tile: 4096,
            native_rectangular_causal_forward: false,
            online_forward: true,
            online_backward: true,
            materialized_backward_mode: RocmFlashAttentionRouteMode::Auto,
            native_backward_preference: RocmFlashAttentionRouteMode::Disabled,
            native_backward_d128_max_sequence: 1024,
            native_backward_d256_max_sequence: 512,
            native_backward_long_min_sequence: 4096,
            collapsed_gqa_backward: false,
            native_direct_collapsed_gqa_backward: false,
            native_gqa_qblock_forward: false,
            native_gqa_qblock_forward_min_sequence: 256,
            wmma_gqa_qblock_forward: false,
            wmma_gqa_r64k32_forward: false,
            wmma_gqa_r64k32_forward_min_sequence: 256,
            wmma_gqa_r64k32_log2_forward: false,
            wmma_gqa_r64k32_log2_forward_min_sequence: 256,
            backward_precompute_delta_max_sequence: 1024,
            native_direct_collapsed_gqa_query_parallelism: 1,
        }
    }

    /// The multiblock experiment changes no flash-attention route.
    #[cfg(any(test, feature = "hardware-qualification"))]
    pub const fn experimental_multiblock() -> Self {
        Self::qualified()
    }

    /// Return the first invalid invariant without touching the device.
    pub const fn validation_error(self) -> Option<&'static str> {
        if self.f32_matmul_inner_tile == 0 {
            return Some("flash_attention.f32_matmul_inner_tile must be positive");
        }
        if self.online_matmul_batch_group == 0 {
            return Some("flash_attention.online_matmul_batch_group must be positive");
        }
        if self.native_scalar_forward_max_sequence == 0 {
            return Some("flash_attention.native_scalar_forward_max_sequence must be positive");
        }
        if self.native_single_forward_max_sequence == 0 {
            return Some("flash_attention.native_single_forward_max_sequence must be positive");
        }
        if self.native_forward_query_tile == 0 {
            return Some("flash_attention.native_forward_query_tile must be positive");
        }
        if self.native_streaming_forward_min_sequence == 0 {
            return Some("flash_attention.native_streaming_forward_min_sequence must be positive");
        }
        if self.native_streaming_forward_key_tile == 0 {
            return Some("flash_attention.native_streaming_forward_key_tile must be positive");
        }
        if self.native_backward_d128_max_sequence == 0
            || self.native_backward_d256_max_sequence == 0
        {
            return Some("flash_attention native backward maxima must be positive");
        }
        if self.native_backward_long_min_sequence == 0 {
            return Some("flash_attention.native_backward_long_min_sequence must be positive");
        }
        if self.native_gqa_qblock_forward_min_sequence == 0 {
            return Some("flash_attention.native_gqa_qblock_forward_min_sequence must be positive");
        }
        if self.native_gqa_qblock_forward_min_sequence > i32::MAX as usize {
            return Some("flash_attention.native_gqa_qblock_forward_min_sequence must fit i32");
        }
        if self.wmma_gqa_qblock_forward && !self.native_gqa_qblock_forward {
            return Some("flash_attention WMMA GQA qblock requires native GQA qblock");
        }
        if self.wmma_gqa_r64k32_forward && !self.wmma_gqa_qblock_forward {
            return Some("flash_attention WMMA GQA r64k32 requires WMMA GQA qblock");
        }
        if self.wmma_gqa_r64k32_forward_min_sequence == 0 {
            return Some("flash_attention.wmma_gqa_r64k32_forward_min_sequence must be positive");
        }
        if self.wmma_gqa_r64k32_forward_min_sequence > i32::MAX as usize {
            return Some("flash_attention.wmma_gqa_r64k32_forward_min_sequence must fit i32");
        }
        if self.wmma_gqa_r64k32_log2_forward && !self.wmma_gqa_r64k32_forward {
            return Some("flash_attention WMMA GQA log2 requires WMMA GQA r64k32");
        }
        if self.wmma_gqa_r64k32_log2_forward_min_sequence == 0 {
            return Some(
                "flash_attention.wmma_gqa_r64k32_log2_forward_min_sequence must be positive",
            );
        }
        if self.wmma_gqa_r64k32_log2_forward_min_sequence > i32::MAX as usize {
            return Some("flash_attention.wmma_gqa_r64k32_log2_forward_min_sequence must fit i32");
        }
        if self.backward_precompute_delta_max_sequence == 0 {
            return Some("flash_attention.backward_precompute_delta_max_sequence must be positive");
        }
        if self.backward_precompute_delta_max_sequence > i32::MAX as usize {
            return Some("flash_attention.backward_precompute_delta_max_sequence must fit i32");
        }
        if !matches!(
            self.native_direct_collapsed_gqa_query_parallelism,
            1 | 2 | 4
        ) {
            return Some(
                "flash_attention.native_direct_collapsed_gqa_query_parallelism must be 1, 2, or 4",
            );
        }
        None
    }
}

impl Default for RocmFlashAttentionPolicy {
    fn default() -> Self {
        Self::portable_fallback()
    }
}

/// Immutable low-level ROCm tensor-kernel policy.
///
/// These values are fixed before the primary context is created. Tensor and
/// kernel crates read the policy from the owning context instead of consulting
/// process environment in operation paths.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RocmTensorKernelPolicy {
    pub split_paged_attention: bool,
    pub split_paged_attention_min_sequence: usize,
    pub paged_attention_split_tokens: usize,
    pub paged_attention_max_splits: usize,
    pub gqa_paged_attention: bool,
    pub gqa_d128_parallel: bool,
    pub gqa_d256_parallel: bool,
    pub concat_safe_row_assembly: bool,
    pub concat_safe_row_assembly_min_elements: usize,
    pub is_finite_host_scan_min_elements: Option<usize>,
    pub rmsnorm_row_tile_rows: usize,
    pub flash_attention: RocmFlashAttentionPolicy,
}

impl RocmTensorKernelPolicy {
    /// Historical Strix Halo tensor-kernel qualification fixture.
    #[cfg(any(test, feature = "hardware-qualification"))]
    pub const fn qualified() -> Self {
        Self {
            split_paged_attention: true,
            split_paged_attention_min_sequence: 2048,
            paged_attention_split_tokens: 128,
            paged_attention_max_splits: 256,
            gqa_paged_attention: true,
            gqa_d128_parallel: true,
            gqa_d256_parallel: true,
            concat_safe_row_assembly: true,
            concat_safe_row_assembly_min_elements: 1_000_000,
            is_finite_host_scan_min_elements: Some(16 * 1024 * 1024),
            rmsnorm_row_tile_rows: 4096,
            flash_attention: RocmFlashAttentionPolicy::qualified(),
        }
    }

    /// Reference-oriented policy that declines accelerated tensor routes while
    /// retaining fixed correctness and bounded-work geometries.
    pub const fn portable_fallback() -> Self {
        Self {
            split_paged_attention: false,
            split_paged_attention_min_sequence: 2048,
            paged_attention_split_tokens: 128,
            paged_attention_max_splits: 256,
            gqa_paged_attention: false,
            gqa_d128_parallel: false,
            gqa_d256_parallel: false,
            concat_safe_row_assembly: true,
            concat_safe_row_assembly_min_elements: 1_000_000,
            is_finite_host_scan_min_elements: Some(16 * 1024 * 1024),
            rmsnorm_row_tile_rows: 4096,
            flash_attention: RocmFlashAttentionPolicy::portable_fallback(),
        }
    }

    /// The experimental model profile changes no low-level tensor route.
    #[cfg(any(test, feature = "hardware-qualification"))]
    pub const fn experimental_multiblock() -> Self {
        Self {
            flash_attention: RocmFlashAttentionPolicy::experimental_multiblock(),
            ..Self::qualified()
        }
    }

    /// Return the first invalid invariant without touching the device.
    pub const fn validation_error(self) -> Option<&'static str> {
        if self.split_paged_attention_min_sequence == 0 {
            return Some("split_paged_attention_min_sequence must be positive");
        }
        if self.paged_attention_split_tokens == 0 {
            return Some("paged_attention_split_tokens must be positive");
        }
        if self.paged_attention_max_splits < 2 {
            return Some("paged_attention_max_splits must be at least two");
        }
        if (self.gqa_d128_parallel || self.gqa_d256_parallel) && !self.gqa_paged_attention {
            return Some("parallel GQA routes require gqa_paged_attention");
        }
        if self.concat_safe_row_assembly_min_elements == 0 {
            return Some("concat_safe_row_assembly_min_elements must be positive");
        }
        if let Some(elements) = self.is_finite_host_scan_min_elements
            && elements == 0
        {
            return Some("is_finite_host_scan_min_elements must be positive when present");
        }
        if self.rmsnorm_row_tile_rows == 0 {
            return Some("rmsnorm_row_tile_rows must be positive");
        }
        if let Some(error) = self.flash_attention.validation_error() {
            return Some(error);
        }
        None
    }
}

impl Default for RocmTensorKernelPolicy {
    fn default() -> Self {
        Self::portable_fallback()
    }
}

/// Immutable execution policy installed when a [`RocmContext`] is created.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RocmExecutionPolicy {
    /// Synchronization discipline for steady-state ROCm execution.
    pub synchronization_mode: RocmSynchronizationMode,
    /// Matmul route policy fixed before the primary context is created.
    pub matmul: RocmMatmulPolicy,
    /// Low-level tensor-kernel policy fixed before context creation.
    pub tensor_kernels: RocmTensorKernelPolicy,
}

impl RocmExecutionPolicy {
    /// Construct a policy with the requested synchronization discipline.
    pub const fn new(synchronization_mode: RocmSynchronizationMode) -> Self {
        Self {
            synchronization_mode,
            matmul: RocmMatmulPolicy::new(
                RocmStridedBatchedMatmulMode::Disabled,
                RocmBf16MatmulOutputMode::F32ThenCast,
            ),
            tensor_kernels: RocmTensorKernelPolicy::portable_fallback(),
        }
    }

    /// Attach an immutable matmul route policy before context creation.
    pub const fn with_matmul_policy(mut self, matmul: RocmMatmulPolicy) -> Self {
        self.matmul = matmul;
        self
    }

    /// Attach the immutable tensor-kernel policy before context creation.
    pub const fn with_tensor_kernel_policy(
        mut self,
        tensor_kernels: RocmTensorKernelPolicy,
    ) -> Self {
        self.tensor_kernels = tensor_kernels;
        self
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
    /// A permanently quarantined device is drained for diagnosis/containment.
    ErrorRecovery,
    /// Synchronization protects a process- or device-global state transition.
    GlobalStateMutation,
    /// A failed graph capture is physically drained before a proven logical
    /// rollback while execution admission remains recoverable.
    CaptureRollback,
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
        Self::CaptureRollback,
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
            Self::CaptureRollback => "capture_rollback",
        }
    }

    #[inline]
    const fn index(self) -> usize {
        self as usize
    }
}

/// Number of fixed ROCm synchronization-reason metric dimensions.
pub const ROCM_SYNC_REASON_COUNT: usize = 23;

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

/// Admission token for one device-level HIP FFI call.
///
/// The token is host-thread affine because acquisition binds the device with
/// `hipSetDevice`. Keep it alive until the corresponding FFI call returns, then
/// consume it with [`Self::complete`] or [`Self::quarantine`]. Dropping an
/// unclassified token permanently quarantines the device, including during a
/// panic unwind.
#[derive(Debug)]
#[must_use = "consume the ROCm submission with complete() or quarantine() after the HIP FFI call"]
pub struct RocmDeviceSubmission {
    permit: RocmExecutionPermit,
    settled: bool,
}

impl RocmDeviceSubmission {
    /// Complete a successful or explicitly nonfatal device FFI call. Consuming
    /// the token makes the end of the admission interval visible in review.
    pub fn complete(mut self) {
        self.settled = true;
    }

    /// Stop all future execution admission for this device. This is nonblocking
    /// and consumes this token so a subsequent `ErrorRecovery` drain cannot
    /// deadlock waiting for the caller's own admission to settle.
    pub fn quarantine(mut self) {
        self.permit.quarantine();
        self.settled = true;
    }
}

impl Drop for RocmDeviceSubmission {
    fn drop(&mut self) {
        if !self.settled {
            self.permit.quarantine();
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
    execution_gate: Arc<RocmExecutionGate>,
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
        if let Some(message) = execution_policy.tensor_kernels.validation_error() {
            return Err(HipError {
                code: -1,
                api: "RocmContext::new",
                message: format!("invalid ROCm tensor-kernel policy: {message}"),
            });
        }
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
        let execution_gate = device_execution_gate(ordinal);
        if execution_gate.execution_stopped() {
            return Err(HipError {
                code: -1,
                api: "RocmContext::new",
                message: format!(
                    "ROCm device {ordinal} is quarantined after a fatal execution failure; restart the process"
                ),
            });
        }
        let initialization = execution_gate.try_acquire("RocmContext::new")?;
        check_call_status(
            &initialization,
            unsafe { sys::hipSetDevice(ordinal) },
            "hipSetDevice",
            HipCallFailureClass::ExecutionMutation,
        )?;
        // Pin the stream-ordered allocator's pool to NEVER release freed memory
        // back to the OS (hipMemPoolAttrReleaseThreshold = u64::MAX). The default
        // threshold (0) makes hipMallocAsync hand freed pages back aggressively;
        // under the decode alloc/free churn that races in-flight kernels and
        // corrupts output (it's what the paged-decode order-sync was masking).
        // Keeping freed blocks pooled removes the hazard AND avoids OS-roundtrip
        // alloc latency. Best-effort: ignore if the runtime lacks mempools.
        const HIP_MEM_POOL_ATTR_RELEASE_THRESHOLD: c_uint = 4;
        let mut pool: *mut c_void = ptr::null_mut();
        let pool_code = unsafe { sys::hipDeviceGetDefaultMemPool(&mut pool, ordinal) };
        if pool_code == sys::HIP_SUCCESS && !pool.is_null() {
            let mut threshold: u64 = u64::MAX;
            let set_code = unsafe {
                sys::hipMemPoolSetAttribute(
                    pool,
                    HIP_MEM_POOL_ATTR_RELEASE_THRESHOLD,
                    &mut threshold as *mut u64 as *mut c_void,
                )
            };
            if set_code != sys::HIP_SUCCESS {
                let _ = check_call_status(
                    &initialization,
                    set_code,
                    "hipMemPoolSetAttribute(release threshold)",
                    HipCallFailureClass::OptionalConfiguration,
                );
            }
        } else if pool_code != sys::HIP_SUCCESS {
            let _ = check_call_status(
                &initialization,
                pool_code,
                "hipDeviceGetDefaultMemPool",
                HipCallFailureClass::PureQuery,
            );
        }
        let sync_telemetry = Arc::new(RocmSyncTelemetry::default());
        drop(initialization);
        let default_stream = RocmStream::create(
            ordinal,
            None,
            Arc::clone(&sync_telemetry),
            Arc::clone(&execution_gate),
        )?;
        Ok(Arc::new(RocmContext {
            ordinal,
            default_stream,
            execution_policy,
            sync_telemetry,
            execution_gate,
        }))
    }

    /// The device ordinal this context targets.
    pub fn ordinal(&self) -> usize {
        self.ordinal as usize
    }

    /// Bind this device to the calling thread (`hipSetDevice`). Cheap; called
    /// before driver work, mirroring cudarc's `bind_to_thread`.
    pub fn bind_to_thread(&self) -> Result<()> {
        let _permit = self.execution_permit("RocmContext::bind_to_thread")?;
        Ok(())
    }

    /// Whether a fatal failure made further execution and resource cleanup
    /// unsafe for this device. This state is shared by every context for the
    /// same ordinal and remains set until process restart.
    pub fn cleanup_quarantined(&self) -> bool {
        self.execution_gate.execution_stopped()
    }

    /// Reject new execution after a fatal failure. A narrowly scoped
    /// [`Self::synchronize_device_for`] call with `ErrorRecovery` may still try
    /// to settle physical work for diagnosis, but never clears the quarantine.
    /// This is validation only and does not reserve admission; code about to
    /// call HIP must acquire [`Self::execution_submission`] instead.
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
        self.execution_gate.request_quarantine();
    }

    /// Acquire admission for one device-level HIP FFI call and bind this
    /// context's device to the current host thread. The token must remain alive
    /// until that FFI call returns.
    pub fn execution_submission(&self, api: &'static str) -> Result<RocmDeviceSubmission> {
        let permit = self.execution_permit(api)?;
        Ok(RocmDeviceSubmission {
            permit,
            settled: false,
        })
    }

    fn execution_permit(&self, api: &'static str) -> Result<RocmExecutionPermit> {
        let permit = self.execution_gate.try_acquire(api)?;
        check_call_status(
            &permit,
            unsafe { sys::hipSetDevice(self.ordinal) },
            "hipSetDevice",
            HipCallFailureClass::ExecutionMutation,
        )?;
        Ok(permit)
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
        snapshot.cleanup_quarantined = self.execution_gate.execution_stopped();
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
            self.execution_gate.request_quarantine();
        }
        result
    }

    /// Always perform and account for a device-wide synchronization.
    ///
    /// Use this for true global boundaries such as reclamation and global-state
    /// mutation. Same-stream steady-state dependencies must use
    /// [`Self::synchronize_same_stream_dependency`] instead. `ErrorRecovery`
    /// requires quarantine to have been published first, then waits until every
    /// pre-quarantine submission token has been dropped. Use `CaptureRollback`
    /// for a recoverable capture drain that deliberately leaves admission open.
    /// Public token quarantine methods consume their token to make the fatal
    /// ordering explicit.
    pub fn synchronize_device_for(&self, reason: RocmSyncReason) -> Result<()> {
        if reason == RocmSyncReason::ErrorRecovery {
            self.execution_gate.require_stopped_for_error_recovery()?;
            return self.timed_synchronize(reason, RocmSyncScope::Device, || {
                if !self
                    .execution_gate
                    .wait_until_final_for(EXECUTION_SETTLEMENT_TIMEOUT)
                {
                    return Err(HipError {
                        code: -1,
                        api: "RocmContext::synchronize_device_for(ErrorRecovery)",
                        message: format!(
                            "timed out after {} ms waiting for admitted HIP calls to return; execution remains quarantined",
                            EXECUTION_SETTLEMENT_TIMEOUT.as_millis()
                        ),
                    });
                }
                let _recovery = self
                    .execution_gate
                    .recovery_lock
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                check(unsafe { sys::hipSetDevice(self.ordinal) }, "hipSetDevice")?;
                check(
                    unsafe { sys::hipDeviceSynchronize() },
                    "hipDeviceSynchronize",
                )
            });
        }
        let submission = self.execution_permit("RocmContext::synchronize_device_for")?;
        self.timed_synchronize(reason, RocmSyncScope::Device, || {
            check_call_status(
                &submission,
                unsafe { sys::hipDeviceSynchronize() },
                "hipDeviceSynchronize",
                HipCallFailureClass::ExecutionMutation,
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
        let submission = self.execution_permit("RocmContext::trim_pool")?;
        let mut pool: *mut c_void = ptr::null_mut();
        check_call_status(
            &submission,
            unsafe { sys::hipDeviceGetDefaultMemPool(&mut pool, self.ordinal) },
            "hipDeviceGetDefaultMemPool",
            HipCallFailureClass::PureQuery,
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
        // The device is already drained and trim only releases currently
        // unused pool pages. A rejected or partial trim leaves live allocation
        // ownership unchanged, so surface the error without quarantining.
        check_call_status(
            &submission,
            unsafe { sys::hipMemPoolTrimTo(pool, effective_min_keep) },
            "hipMemPoolTrimTo",
            HipCallFailureClass::PostDrainPoolMaintenance,
        )?;
        drop(submission);
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
        let _submission = self.execution_permit("RocmContext::pool_stats")?;
        const HIP_MEM_POOL_ATTR_RESERVED_MEM_CURRENT: c_uint = 5;
        const HIP_MEM_POOL_ATTR_USED_MEM_CURRENT: c_uint = 7;
        let mut pool: *mut c_void = ptr::null_mut();
        let pool_code = unsafe { sys::hipDeviceGetDefaultMemPool(&mut pool, self.ordinal) };
        if pool_code != sys::HIP_SUCCESS {
            let _ = check_call_status(
                &_submission,
                pool_code,
                "hipDeviceGetDefaultMemPool",
                HipCallFailureClass::PureQuery,
            );
            return Ok((0, 0));
        }
        if pool.is_null() {
            return Ok((0, 0));
        }
        let mut reserved: u64 = 0;
        let mut used: u64 = 0;
        check_call_status(
            &_submission,
            unsafe {
                sys::hipMemPoolGetAttribute(
                    pool,
                    HIP_MEM_POOL_ATTR_RESERVED_MEM_CURRENT,
                    &mut reserved as *mut u64 as *mut c_void,
                )
            },
            "hipMemPoolGetAttribute(reserved)",
            HipCallFailureClass::PureQuery,
        )?;
        check_call_status(
            &_submission,
            unsafe {
                sys::hipMemPoolGetAttribute(
                    pool,
                    HIP_MEM_POOL_ATTR_USED_MEM_CURRENT,
                    &mut used as *mut u64 as *mut c_void,
                )
            },
            "hipMemPoolGetAttribute(used)",
            HipCallFailureClass::PureQuery,
        )?;
        Ok((reserved, used))
    }

    /// Device-reported `(free, total)` bytes via `hipMemGetInfo`. On a discrete
    /// GPU this is the driver's own view; on a unified APU it reflects the GTT
    /// budget and is best cross-checked against the OS-level `kiln-memory` probe.
    pub fn mem_get_info(&self) -> Result<(usize, usize)> {
        let _submission = self.execution_permit("RocmContext::mem_get_info")?;
        let mut free: usize = 0;
        let mut total: usize = 0;
        check_call_status(
            &_submission,
            unsafe { sys::hipMemGetInfo(&mut free, &mut total) },
            "hipMemGetInfo",
            HipCallFailureClass::PureQuery,
        )?;
        Ok((free, total))
    }

    /// Create a fresh non-blocking stream bound to this device.
    pub fn new_stream(&self) -> Result<Arc<RocmStream>> {
        RocmStream::create(
            self.ordinal,
            None,
            Arc::clone(&self.sync_telemetry),
            Arc::clone(&self.execution_gate),
        )
    }

    /// Create a non-blocking stream at an explicit scheduling priority. Lower
    /// integer = higher priority (HIP follows the CUDA convention). See
    /// [`stream_priority_range`].
    pub fn new_stream_with_priority(&self, priority: i32) -> Result<Arc<RocmStream>> {
        RocmStream::create(
            self.ordinal,
            Some(priority),
            Arc::clone(&self.sync_telemetry),
            Arc::clone(&self.execution_gate),
        )
    }

    /// Create a reusable ordering-only event on this device.
    pub fn new_event(&self) -> Result<Arc<RocmEvent>> {
        RocmEvent::create(self.ordinal, Arc::clone(&self.execution_gate))
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
/// `cudarc::driver::CudaStream`; raw access is submission-scoped so a copied
/// handle cannot race process-lifetime execution quarantine.
#[derive(Debug)]
pub struct RocmStream {
    handle: sys::hipStream_t,
    ordinal: c_int,
    sync_telemetry: Arc<RocmSyncTelemetry>,
    execution_gate: Arc<RocmExecutionGate>,
}

/// Opaque, non-executable identity for comparing live ROCm streams.
///
/// Unlike a raw `hipStream_t`, this value cannot be passed to an FFI call and
/// therefore does not require execution admission. It is only meaningful while
/// the compared stream owners remain alive.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RocmStreamId(usize);

/// RAII owner of a reusable HIP event configured for ordering only.
#[derive(Debug)]
pub struct RocmEvent {
    handle: sys::hipEvent_t,
    ordinal: c_int,
    execution_gate: Arc<RocmExecutionGate>,
}

/// Admission token carrying the raw stream for exactly one external FFI call.
/// The owned stream reference prevents destruction while the caller submits.
/// After the call returns, consume the token with [`Self::complete`] or
/// [`Self::quarantine`]. Dropping an unclassified token permanently quarantines
/// the device, including during a panic unwind.
#[derive(Debug)]
#[must_use = "consume the ROCm stream submission with complete() or quarantine() after the FFI call"]
pub struct RocmStreamSubmission {
    stream: Arc<RocmStream>,
    permit: RocmExecutionPermit,
    settled: bool,
}

impl RocmStreamSubmission {
    /// Raw `hipStream_t` for the immediate FFI call protected by this token.
    pub fn raw_stream(&self) -> sys::hipStream_t {
        self.stream.handle
    }

    /// Complete a successful or explicitly nonfatal external FFI call.
    pub fn complete(mut self) {
        self.settled = true;
    }

    /// Permanently stop new submissions after the protected FFI call reports a
    /// fatal error. Consuming the token settles this caller's admission before
    /// it can enter recovery; no later call can acquire a token.
    pub fn quarantine(mut self) {
        self.permit.quarantine();
        self.settled = true;
    }

    /// Device ordinal of the protected stream.
    pub fn ordinal(&self) -> usize {
        self.stream.ordinal()
    }
}

impl Drop for RocmStreamSubmission {
    fn drop(&mut self) {
        if !self.settled {
            self.permit.quarantine();
        }
    }
}

// SAFETY: a hipEvent_t is an opaque runtime handle bound to one device. HIP
// permits recording and waiting on it from different host threads/streams;
// every operation below rebinds the owning device first.
unsafe impl Send for RocmEvent {}
unsafe impl Sync for RocmEvent {}

impl RocmEvent {
    fn create(ordinal: c_int, execution_gate: Arc<RocmExecutionGate>) -> Result<Arc<Self>> {
        let permit = execution_gate.try_acquire("RocmEvent::create")?;
        check_call_status(
            &permit,
            unsafe { sys::hipSetDevice(ordinal) },
            "hipSetDevice",
            HipCallFailureClass::ExecutionMutation,
        )?;
        let mut handle: sys::hipEvent_t = ptr::null_mut();
        let code =
            unsafe { sys::hipEventCreateWithFlags(&mut handle, sys::HIP_EVENT_DISABLE_TIMING) };
        check_resource_creation(&permit, code, handle.is_null(), "hipEventCreateWithFlags")?;
        Ok(Arc::new(Self {
            handle,
            ordinal,
            execution_gate,
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
        execution_gate: Arc<RocmExecutionGate>,
    ) -> Result<Arc<Self>> {
        let permit = execution_gate.try_acquire("RocmStream::create")?;
        check_call_status(
            &permit,
            unsafe { sys::hipSetDevice(ordinal) },
            "hipSetDevice",
            HipCallFailureClass::ExecutionMutation,
        )?;
        let mut handle: sys::hipStream_t = ptr::null_mut();
        let (code, api) = match priority {
            None => (
                unsafe { sys::hipStreamCreateWithFlags(&mut handle, sys::HIP_STREAM_NON_BLOCKING) },
                "hipStreamCreateWithFlags",
            ),
            Some(p) => (
                unsafe {
                    sys::hipStreamCreateWithPriority(
                        &mut handle,
                        sys::HIP_STREAM_NON_BLOCKING,
                        p as c_int,
                    )
                },
                "hipStreamCreateWithPriority",
            ),
        };
        check_resource_creation(&permit, code, handle.is_null(), api)?;
        Ok(Arc::new(RocmStream {
            handle,
            ordinal,
            sync_telemetry,
            execution_gate,
        }))
    }

    /// Acquire a typed admission token for one external kernel launch.
    /// Returning a token instead of a copyable raw handle closes the
    /// check-to-FFI quarantine race.
    pub fn execution_submission(
        self: &Arc<Self>,
        api: &'static str,
    ) -> Result<RocmStreamSubmission> {
        let permit = self.execution_permit(api)?;
        Ok(RocmStreamSubmission {
            stream: Arc::clone(self),
            permit,
            settled: false,
        })
    }

    /// The device ordinal this stream is bound to.
    pub fn ordinal(&self) -> usize {
        self.ordinal as usize
    }

    /// Stable identity for this live stream wrapper. This does not bind a
    /// device or grant permission to submit work.
    pub fn id(&self) -> RocmStreamId {
        RocmStreamId(self as *const Self as usize)
    }

    #[inline]
    fn execution_permit(&self, api: &'static str) -> Result<RocmExecutionPermit> {
        let permit = self.execution_gate.try_acquire(api)?;
        check_call_status(
            &permit,
            unsafe { sys::hipSetDevice(self.ordinal) },
            "hipSetDevice",
            HipCallFailureClass::ExecutionMutation,
        )?;
        Ok(permit)
    }

    /// Block until all work queued on this stream completes.
    pub fn synchronize(&self) -> Result<()> {
        let submission = self.execution_permit("RocmStream::synchronize")?;
        check_call_status(
            &submission,
            unsafe { sys::hipStreamSynchronize(self.handle) },
            "hipStreamSynchronize",
            HipCallFailureClass::ExecutionMutation,
        )
    }

    /// Settle a stream while retaining an admission acquired before the async
    /// operation. A concurrent STOP cannot open a gap between enqueue and wait.
    fn synchronize_admitted_for(
        &self,
        submission: &RocmExecutionPermit,
        reason: RocmSyncReason,
    ) -> Result<()> {
        let started = Instant::now();
        let result = check_call_status(
            submission,
            unsafe { sys::hipStreamSynchronize(self.handle) },
            "hipStreamSynchronize",
            HipCallFailureClass::ExecutionMutation,
        );
        let waited_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.sync_telemetry
            .record_wait(reason, RocmSyncScope::Stream, waited_ns);
        if result.is_err() {
            self.execution_gate.request_quarantine();
        }
        result
    }

    /// Record `event` after all work currently queued on this stream.
    pub fn record_event(&self, event: &RocmEvent) -> Result<()> {
        event.ensure_same_device(self)?;
        let submission = self.execution_permit("RocmStream::record_event")?;
        check_call_status(
            &submission,
            unsafe { sys::hipEventRecord(event.handle, self.handle) },
            "hipEventRecord",
            HipCallFailureClass::ExecutionMutation,
        )
    }

    /// Queue a dependency on the most recent recording of `event` without
    /// blocking the host thread.
    pub fn wait_event(&self, event: &RocmEvent) -> Result<()> {
        event.ensure_same_device(self)?;
        let submission = self.execution_permit("RocmStream::wait_event")?;
        check_call_status(
            &submission,
            unsafe { sys::hipStreamWaitEvent(self.handle, event.handle, 0) },
            "hipStreamWaitEvent",
            HipCallFailureClass::ExecutionMutation,
        )
    }

    /// Allocate `len` bytes on the device, zeroed. Stream-ordered
    /// (`hipMallocAsync`) when supported, falling back to synchronous
    /// `hipMalloc` on arches/runtimes without the stream-ordered allocator.
    pub fn alloc_zeros(self: &Arc<Self>, len: usize) -> Result<RocmSlice> {
        let slice = self.alloc(len)?;
        if len > 0 {
            let submission = self.execution_permit("RocmStream::alloc_zeros")?;
            // SAFETY: slice.ptr is a valid device allocation of `len` bytes.
            check_execution_mutation(
                &submission,
                unsafe { sys::hipMemsetD8Async(slice.ptr, 0, len, self.handle) },
                "hipMemsetD8Async",
            )?;
        }
        Ok(slice)
    }

    /// Allocate `len` (uninitialized) bytes on the device. See [`Self::alloc_zeros`].
    pub fn alloc(self: &Arc<Self>, len: usize) -> Result<RocmSlice> {
        let submission = self.execution_permit("RocmStream::alloc")?;
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
        // R.9). Invalid-value/not-supported with no published pointer is the
        // documented capability fallback. OOM is returned unchanged for the
        // allocation governor; ambiguous failures quarantine execution.
        let async_rc = unsafe { sys::hipMallocAsync(&mut ptr, len, self.handle) };
        let async_alloc = match async_rc {
            sys::HIP_SUCCESS => {
                check_resource_creation(&submission, async_rc, ptr.is_null(), "hipMallocAsync")?;
                true
            }
            code if async_allocation_fallback_allowed(code, ptr.is_null()) => {
                // Clear HIP's thread-local sticky slot before trying the
                // documented synchronous capability fallback.
                let _ = check(async_rc, "hipMallocAsync(capability fallback)");
                let sync_rc = unsafe { sys::hipMalloc(&mut ptr, len) };
                check_resource_creation(&submission, sync_rc, ptr.is_null(), "hipMalloc")?;
                false
            }
            _ => {
                check_resource_creation(&submission, async_rc, ptr.is_null(), "hipMallocAsync")?;
                unreachable!("failed hipMallocAsync cannot pass creation classification")
            }
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
        // Own staging memory until the wait is classified. Borrowing `src`
        // directly would make a failed wait or unwind return control while HIP
        // could still dereference caller-owned memory.
        let staging = src.to_vec();
        let submission = self.execution_permit("RocmStream::memcpy_htod")?;
        let transfer = RocmAdmittedHostTransfer::new(submission, staging);
        // SAFETY: dst.ptr is a valid device allocation of dst.len bytes; src is
        // a valid host buffer of the same length. We synchronize before return.
        let enqueue = check_execution_mutation(
            transfer.permit(),
            unsafe {
                sys::hipMemcpyHtoDAsync(
                    dst.ptr,
                    transfer.host().as_ptr() as *mut c_void,
                    transfer.host().len(),
                    self.handle,
                )
            },
            "hipMemcpyHtoDAsync",
        );
        let settlement =
            self.synchronize_admitted_for(transfer.permit(), RocmSyncReason::AllocationLifetime);
        let host_memory_settled = settlement.is_ok();
        let _staging = transfer.finish(host_memory_settled);
        host_transfer_result(enqueue, settlement, "RocmStream::memcpy_htod")
    }

    /// Async H2D copy into a caller-supplied raw device pointer, WITHOUT a
    /// trailing synchronize. The HIP-graph replay path (R.9) uses this to
    /// refresh a graph-stable buffer's contents *in place*: the destination
    /// pointer is the one baked into the captured graph, so it must not change
    /// (no realloc). The copy is queued on this stream and is ordered before
    /// any subsequent launch on the same stream.
    ///
    /// Unlike [`Self::memcpy_htod`], this does NOT add a trailing stream wait.
    /// It deliberately uses generic `hipMemcpyAsync`: ROCm's runtime contract
    /// says an unpinned host source is consumed synchronously before that call
    /// returns. That makes fresh pageable staging buffers safe without adding a
    /// graph-replay barrier. A pinned or registered `src` can remain async and
    /// must stay alive until the stream completes.
    ///
    /// # Safety
    /// `dst` must point to at least `src.len()` bytes of a live device
    /// allocation reachable from this stream's device. If `src` is pinned or
    /// registered host memory, it must remain live until the stream completes.
    pub unsafe fn memcpy_htod_raw_async(&self, dst: *mut c_void, src: &[u8]) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        let submission = self.execution_permit("RocmStream::memcpy_htod_raw_async")?;
        // SAFETY: caller guarantees `dst` addresses >= src.len() live device
        // bytes; `src` is a valid host slice of the same length. The generic
        // API's installed-header contract explicitly makes unpinned host
        // copies synchronous with respect to this borrow.
        check_execution_mutation(
            &submission,
            unsafe {
                sys::hipMemcpyAsync(
                    dst,
                    src.as_ptr() as *const c_void,
                    src.len(),
                    sys::HIP_MEMCPY_HOST_TO_DEVICE,
                    self.handle,
                )
            },
            "hipMemcpyAsync(HostToDevice)",
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
        let submission = self.execution_permit("RocmStream::memset_zero_async")?;
        // SAFETY: caller guarantees `dst` addresses >= len live device bytes.
        check_execution_mutation(
            &submission,
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
        let out = vec![0u8; src.len];
        if src.len == 0 {
            return Ok(out);
        }
        let submission = self.execution_permit("RocmStream::memcpy_dtoh")?;
        let mut transfer = RocmAdmittedHostTransfer::new(submission, out);
        let out_ptr = transfer.host_mut().as_mut_ptr();
        // SAFETY: src.ptr is a valid device allocation of src.len bytes; out is
        // a host buffer of the same length. Synchronized before return.
        let enqueue = check_execution_mutation(
            transfer.permit(),
            unsafe {
                sys::hipMemcpyDtoHAsync(out_ptr as *mut c_void, src.ptr, src.len, self.handle)
            },
            "hipMemcpyDtoHAsync",
        );
        let settlement =
            self.synchronize_admitted_for(transfer.permit(), RocmSyncReason::HostReadback);
        let host_memory_settled = settlement.is_ok();
        let out = transfer.finish(host_memory_settled);
        host_transfer_result(enqueue, settlement, "RocmStream::memcpy_dtoh")?;
        Ok(out.expect("successful host transfer settlement returns its buffer"))
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
        let out = vec![0u8; len];
        if len == 0 {
            return Ok(out);
        }
        let submission = self.execution_permit("RocmStream::memcpy_dtoh_raw")?;
        let mut transfer = RocmAdmittedHostTransfer::new(submission, out);
        let out_ptr = transfer.host_mut().as_mut_ptr();
        // SAFETY: the caller guarantees `src` addresses at least `len` live
        // device bytes; `out` owns exactly `len` writable host bytes. The
        // stream synchronization below completes the copy before either
        // pointer can become invalid.
        let enqueue = check_execution_mutation(
            transfer.permit(),
            unsafe {
                sys::hipMemcpyDtoHAsync(
                    out_ptr as *mut c_void,
                    src as *mut c_void,
                    len,
                    self.handle,
                )
            },
            "hipMemcpyDtoHAsync",
        );
        let settlement =
            self.synchronize_admitted_for(transfer.permit(), RocmSyncReason::HostReadback);
        let host_memory_settled = settlement.is_ok();
        let out = transfer.finish(host_memory_settled);
        host_transfer_result(enqueue, settlement, "RocmStream::memcpy_dtoh_raw")?;
        Ok(out.expect("successful host transfer settlement returns its buffer"))
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
        let submission = self.execution_permit("RocmStream::memcpy_dtod")?;
        // SAFETY: both are valid device allocations of equal length.
        check_execution_mutation(
            &submission,
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
        let submission = self.execution_permit("RocmStream::memcpy_dtod_raw_async")?;
        // SAFETY: caller guarantees both raw pointers address >= len live
        // device bytes on this stream's device.
        check_execution_mutation(
            &submission,
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
        let Some(submission) =
            bind_device_for_cleanup(self.ordinal, &self.execution_gate, "RocmStream::drop")
        else {
            self.handle = ptr::null_mut();
            return;
        };
        // SAFETY: handle was created by hipStreamCreate* and not yet destroyed.
        let rc = unsafe { sys::hipStreamDestroy(self.handle) };
        if rc != sys::HIP_SUCCESS {
            submission.quarantine();
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
        let Some(submission) =
            bind_device_for_cleanup(self.ordinal, &self.execution_gate, "RocmEvent::drop")
        else {
            self.handle = ptr::null_mut();
            return;
        };
        let rc = unsafe { sys::hipEventDestroy(self.handle) };
        if rc != sys::HIP_SUCCESS {
            submission.quarantine();
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
        let Some(submission) = bind_device_for_cleanup(
            self.stream.ordinal,
            &self.stream.execution_gate,
            "RocmSlice::drop",
        ) else {
            self.ptr = ptr::null_mut();
            return;
        };
        // SAFETY: ptr was produced by hipMallocAsync/hipMalloc on self.stream
        // and not yet freed. Free with the matching API.
        let rc = if self.async_alloc {
            unsafe { sys::hipFreeAsync(self.ptr, self.stream.handle) }
        } else {
            unsafe { sys::hipFree(self.ptr) }
        };
        if rc != sys::HIP_SUCCESS {
            submission.quarantine();
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
    execution_gate: Arc<RocmExecutionGate>,
}

unsafe impl Send for RocmGraph {}
unsafe impl Sync for RocmGraph {}

/// RAII owner of an instantiated `hipGraphExec_t`. Analog of `CudaGraphExec`.
#[derive(Debug)]
pub struct RocmGraphExec {
    exec: sys::hipGraphExec_t,
    ordinal: c_int,
    execution_gate: Arc<RocmExecutionGate>,
}

unsafe impl Send for RocmGraphExec {}
unsafe impl Sync for RocmGraphExec {}

impl RocmStream {
    /// Begin capturing work issued on this stream into a graph
    /// (`hipStreamBeginCapture`, relaxed mode — matches the CUDA path).
    pub fn begin_capture(&self) -> Result<()> {
        let submission = self.execution_permit("RocmStream::begin_capture")?;
        check_call_status(
            &submission,
            unsafe {
                sys::hipStreamBeginCapture(self.handle, sys::HIP_STREAM_CAPTURE_MODE_RELAXED)
            },
            "hipStreamBeginCapture",
            HipCallFailureClass::ExecutionMutation,
        )
    }

    /// End capture and return the resulting graph (`hipStreamEndCapture`).
    pub fn end_capture(&self) -> Result<RocmGraph> {
        let submission = self.execution_permit("RocmStream::end_capture")?;
        let mut graph: sys::hipGraph_t = ptr::null_mut();
        let code = unsafe { sys::hipStreamEndCapture(self.handle, &mut graph) };
        if code == sys::HIP_SUCCESS && graph.is_null() {
            submission.quarantine();
            return Err(HipError {
                code: -1,
                api: "hipStreamEndCapture",
                message: "HIP reported successful graph capture without publishing a graph"
                    .to_string(),
            });
        }
        check_call_status(
            &submission,
            code,
            "hipStreamEndCapture",
            HipCallFailureClass::ExecutionMutation,
        )?;
        Ok(RocmGraph {
            graph,
            ordinal: self.ordinal,
            execution_gate: Arc::clone(&self.execution_gate),
        })
    }

    /// Whether a capture is currently active on this stream.
    pub fn is_capturing(&self) -> Result<bool> {
        let submission = self.execution_permit("RocmStream::is_capturing")?;
        let mut status: c_uint = 0;
        check_call_status(
            &submission,
            unsafe { sys::hipStreamIsCapturing(self.handle, &mut status) },
            "hipStreamIsCapturing",
            HipCallFailureClass::CaptureState,
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
        let submission = self.execution_gate.try_acquire("RocmGraph::instantiate")?;
        check_call_status(
            &submission,
            unsafe { sys::hipSetDevice(self.ordinal) },
            "hipSetDevice",
            HipCallFailureClass::ExecutionMutation,
        )?;
        let mut exec: sys::hipGraphExec_t = ptr::null_mut();
        let code = unsafe { sys::hipGraphInstantiateWithFlags(&mut exec, self.graph, 0) };
        check_resource_creation(
            &submission,
            code,
            exec.is_null(),
            "hipGraphInstantiateWithFlags",
        )?;
        Ok(RocmGraphExec {
            exec,
            ordinal: self.ordinal,
            execution_gate: Arc::clone(&self.execution_gate),
        })
    }
}

impl Drop for RocmGraph {
    fn drop(&mut self) {
        if !self.graph.is_null() {
            let Some(submission) =
                bind_device_for_cleanup(self.ordinal, &self.execution_gate, "RocmGraph::drop")
            else {
                self.graph = ptr::null_mut();
                return;
            };
            let rc = unsafe { sys::hipGraphDestroy(self.graph) };
            if rc != sys::HIP_SUCCESS {
                submission.quarantine();
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
        let submission = stream.execution_permit("RocmGraphExec::launch")?;
        check_call_status(
            &submission,
            unsafe { sys::hipGraphLaunch(self.exec, stream.handle) },
            "hipGraphLaunch",
            HipCallFailureClass::ExecutionMutation,
        )
    }
}

impl Drop for RocmGraphExec {
    fn drop(&mut self) {
        if !self.exec.is_null() {
            let Some(submission) =
                bind_device_for_cleanup(self.ordinal, &self.execution_gate, "RocmGraphExec::drop")
            else {
                self.exec = ptr::null_mut();
                return;
            };
            let rc = unsafe { sys::hipGraphExecDestroy(self.exec) };
            if rc != sys::HIP_SUCCESS {
                submission.quarantine();
                eprintln!("RocmGraphExec::drop: hipGraphExecDestroy failed (hipError {rc})");
                warn_cleanup_quarantine("RocmGraphExec::drop");
            }
            self.exec = ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// CPU-only policy/concurrency tests run everywhere. Device integration tests
// use the `try_ctx` skip pattern when no real HIP device is present.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::time::Duration;

    fn gate_state(gate: &RocmExecutionGate) -> (bool, bool, usize) {
        let state = gate.state.load(Ordering::Acquire);
        (
            state & EXECUTION_GATE_STOPPED != 0,
            state & EXECUTION_GATE_FINAL != 0,
            state & EXECUTION_GATE_ACTIVE_MASK,
        )
    }

    #[test]
    fn execution_gate_quarantine_waits_for_admitted_call() {
        let gate = Arc::new(RocmExecutionGate::default());
        let permit = gate
            .try_acquire("execution_gate_quarantine_waits_for_admitted_call")
            .expect("initial admission");

        gate.request_quarantine();
        assert_eq!(gate_state(&gate), (true, false, 1));
        assert!(
            gate.try_acquire("post-quarantine admission").is_err(),
            "the stop transition must reject every later admission"
        );

        drop(permit);
        assert_eq!(gate_state(&gate), (true, true, 0));
    }

    #[test]
    fn execution_gate_linearizes_concurrent_quarantine() {
        let gate = Arc::new(RocmExecutionGate::default());
        let entered_ffi = Arc::new(Barrier::new(2));
        let leave_ffi = Arc::new(Barrier::new(2));
        let worker = {
            let gate = Arc::clone(&gate);
            let entered_ffi = Arc::clone(&entered_ffi);
            let leave_ffi = Arc::clone(&leave_ffi);
            std::thread::spawn(move || {
                let permit = gate
                    .try_acquire("simulated admitted HIP call")
                    .expect("worker admission");
                entered_ffi.wait();
                leave_ffi.wait();
                drop(permit);
            })
        };

        entered_ffi.wait();
        gate.request_quarantine();
        assert_eq!(gate_state(&gate), (true, false, 1));
        assert!(gate.try_acquire("racing later HIP call").is_err());

        leave_ffi.wait();
        worker.join().expect("worker must return normally");
        assert_eq!(gate_state(&gate), (true, true, 0));
    }

    #[test]
    fn execution_gate_waiter_unblocks_only_after_final_settlement() {
        let gate = Arc::new(RocmExecutionGate::default());
        let permit = gate.try_acquire("held HIP call").expect("admission");
        gate.request_quarantine();

        let waiter_ready = Arc::new(Barrier::new(2));
        let (settled_tx, settled_rx) = std::sync::mpsc::channel();
        let waiter = {
            let gate = Arc::clone(&gate);
            let waiter_ready = Arc::clone(&waiter_ready);
            std::thread::spawn(move || {
                waiter_ready.wait();
                assert!(gate.wait_until_final_for(Duration::from_secs(1)));
                settled_tx.send(()).expect("settlement receiver alive");
            })
        };
        waiter_ready.wait();
        assert!(settled_rx.try_recv().is_err());

        drop(permit);
        settled_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("last permit must wake recovery waiters");
        waiter.join().expect("waiter must return normally");
    }

    #[test]
    fn execution_gate_settlement_wait_is_bounded_and_fail_closed() {
        let gate = Arc::new(RocmExecutionGate::default());
        let permit = gate.try_acquire("hung HIP call").expect("admission");
        gate.request_quarantine();

        assert!(!gate.wait_until_final_for(Duration::from_millis(10)));
        assert_eq!(gate_state(&gate), (true, false, 1));
        assert!(gate.try_acquire("post-timeout admission").is_err());

        drop(permit);
        assert!(gate.wait_until_final_for(Duration::ZERO));
        assert_eq!(gate_state(&gate), (true, true, 0));
    }

    #[test]
    fn unclassified_public_submission_drop_is_fail_closed() {
        let gate = Arc::new(RocmExecutionGate::default());
        let submission = RocmDeviceSubmission {
            permit: gate
                .try_acquire("unclassified public FFI")
                .expect("admission"),
            settled: false,
        };

        drop(submission);

        assert_eq!(gate_state(&gate), (true, true, 0));
        assert!(
            gate.try_acquire("submission after unclassified drop")
                .is_err()
        );
    }

    #[test]
    fn explicit_public_submission_completion_keeps_admission_open() {
        let gate = Arc::new(RocmExecutionGate::default());
        RocmDeviceSubmission {
            permit: gate
                .try_acquire("successful public FFI")
                .expect("admission"),
            settled: false,
        }
        .complete();

        assert_eq!(gate_state(&gate), (false, false, 0));
        assert!(gate.try_acquire("submission after completion").is_ok());
    }

    #[test]
    fn concurrent_stop_waits_for_admitted_host_transfer_lifetime() {
        struct DropProbe(Arc<AtomicBool>);
        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }

        let gate = Arc::new(RocmExecutionGate::default());
        let host_dropped = Arc::new(AtomicBool::new(false));
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (settle_tx, settle_rx) = std::sync::mpsc::channel();
        let worker = {
            let gate = Arc::clone(&gate);
            let host_dropped = Arc::clone(&host_dropped);
            std::thread::spawn(move || {
                let permit = gate.try_acquire("simulated async host transfer").unwrap();
                let transfer =
                    RocmAdmittedHostTransfer::new(permit, DropProbe(host_dropped.clone()));
                entered_tx.send(()).unwrap();
                settle_rx.recv().unwrap();
                assert!(!host_dropped.load(Ordering::Acquire));
                let host = transfer
                    .finish(true)
                    .expect("successful settlement releases host memory");
                drop(host);
            })
        };

        entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("host transfer did not acquire admission");
        gate.request_quarantine();
        assert_eq!(gate_state(&gate), (true, false, 1));
        assert!(!host_dropped.load(Ordering::Acquire));
        settle_tx.send(()).unwrap();
        worker.join().unwrap();
        assert!(host_dropped.load(Ordering::Acquire));
        assert_eq!(gate_state(&gate), (true, true, 0));
    }

    #[test]
    fn failed_or_unwound_host_transfer_retains_host_memory() {
        struct DropProbe(Arc<AtomicBool>);
        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }

        let failed_gate = Arc::new(RocmExecutionGate::default());
        let failed_drop = Arc::new(AtomicBool::new(false));
        let failed = RocmAdmittedHostTransfer::new(
            failed_gate.try_acquire("failed host transfer").unwrap(),
            DropProbe(failed_drop.clone()),
        );
        assert!(failed.finish(false).is_none());
        assert!(!failed_drop.load(Ordering::Acquire));
        assert_eq!(gate_state(&failed_gate), (true, true, 0));

        let unwind_gate = Arc::new(RocmExecutionGate::default());
        let unwind_drop = Arc::new(AtomicBool::new(false));
        let unwind = std::panic::catch_unwind({
            let unwind_gate = Arc::clone(&unwind_gate);
            let unwind_drop = Arc::clone(&unwind_drop);
            move || {
                let _transfer = RocmAdmittedHostTransfer::new(
                    unwind_gate.try_acquire("unwound host transfer").unwrap(),
                    DropProbe(unwind_drop),
                );
                panic!("simulated unwind before host-transfer settlement");
            }
        });
        assert!(unwind.is_err());
        assert!(!unwind_drop.load(Ordering::Acquire));
        assert_eq!(gate_state(&unwind_gate), (true, true, 0));
    }

    #[test]
    fn public_submission_unwind_publishes_sticky_stop() {
        let gate = Arc::new(RocmExecutionGate::default());
        let result = std::panic::catch_unwind({
            let gate = Arc::clone(&gate);
            move || {
                let _submission = RocmDeviceSubmission {
                    permit: gate.try_acquire("panicking public FFI").expect("admission"),
                    settled: false,
                };
                panic!("simulated panic after external FFI");
            }
        });

        assert!(result.is_err());
        assert_eq!(gate_state(&gate), (true, true, 0));
        assert!(gate.try_acquire("submission after unwind").is_err());
    }

    #[test]
    fn public_stream_submission_unwind_publishes_sticky_stop() {
        let gate = Arc::new(RocmExecutionGate::default());
        let stream = Arc::new(RocmStream {
            handle: ptr::null_mut(),
            ordinal: 1_000_000_003,
            sync_telemetry: Arc::new(RocmSyncTelemetry::default()),
            execution_gate: Arc::clone(&gate),
        });
        let result = std::panic::catch_unwind({
            let gate = Arc::clone(&gate);
            let stream = Arc::clone(&stream);
            move || {
                let _submission = RocmStreamSubmission {
                    stream,
                    permit: gate.try_acquire("panicking stream FFI").expect("admission"),
                    settled: false,
                };
                panic!("simulated panic after external stream FFI");
            }
        });

        assert!(result.is_err());
        assert_eq!(gate_state(&gate), (true, true, 0));
        assert!(gate.try_acquire("stream submission after unwind").is_err());
        // This synthetic stream has no HIP handle and exists only to exercise
        // the token's unwind behavior without touching the runtime.
        std::mem::forget(stream);
    }

    #[test]
    fn error_recovery_requires_a_preexisting_stop_transition() {
        let gate = RocmExecutionGate::default();
        let error = gate
            .require_stopped_for_error_recovery()
            .expect_err("gate-open ErrorRecovery must be rejected");
        assert!(error.message.contains("use CaptureRollback"));

        gate.request_quarantine();
        gate.require_stopped_for_error_recovery()
            .expect("STOPped gate admits the bounded recovery drain");
    }

    #[test]
    fn internal_execution_permit_unwind_is_fail_closed() {
        let gate = Arc::new(RocmExecutionGate::default());
        let result = std::panic::catch_unwind({
            let gate = Arc::clone(&gate);
            move || {
                let _permit = gate
                    .try_acquire("panicking HIP wrapper")
                    .expect("admission");
                panic!("simulated wrapper panic");
            }
        });
        assert!(result.is_err());
        assert_eq!(gate_state(&gate), (true, true, 0));
        assert!(gate.try_acquire("admission after unwind").is_err());
    }

    #[test]
    fn execution_gates_are_isolated_by_device() {
        const FIRST_TEST_ORDINAL: c_int = 1_000_000_001;
        const SECOND_TEST_ORDINAL: c_int = 1_000_000_002;
        let first = device_execution_gate(FIRST_TEST_ORDINAL);
        let same_device = device_execution_gate(FIRST_TEST_ORDINAL);
        let second = device_execution_gate(SECOND_TEST_ORDINAL);
        assert!(Arc::ptr_eq(&first, &same_device));
        assert!(!Arc::ptr_eq(&first, &second));
        first.request_quarantine();

        assert!(same_device.try_acquire("stopped device").is_err());
        assert!(second.try_acquire("independent device").is_ok());
        assert_eq!(gate_state(&first), (true, true, 0));
        assert_eq!(gate_state(&second), (false, false, 0));
    }

    #[test]
    fn resource_creation_policy_preserves_clean_no_publication_failures() {
        for code in [
            sys::HIP_ERROR_INVALID_VALUE,
            sys::HIP_ERROR_OUT_OF_MEMORY,
            sys::HIP_ERROR_NOT_SUPPORTED,
        ] {
            assert!(!resource_creation_status_is_fatal(code, true));
            assert!(
                resource_creation_status_is_fatal(code, false),
                "an error with a published handle is ambiguous"
            );
        }

        assert!(!resource_creation_status_is_fatal(sys::HIP_SUCCESS, false));
        assert!(resource_creation_status_is_fatal(sys::HIP_SUCCESS, true));
        assert!(resource_creation_status_is_fatal(
            sys::HIP_ERROR_PRIOR_LAUNCH_FAILURE,
            true
        ));
    }

    #[test]
    fn async_allocation_fallback_is_capability_only() {
        assert!(async_allocation_fallback_allowed(
            sys::HIP_ERROR_INVALID_VALUE,
            true
        ));
        assert!(async_allocation_fallback_allowed(
            sys::HIP_ERROR_NOT_SUPPORTED,
            true
        ));
        assert!(!async_allocation_fallback_allowed(
            sys::HIP_ERROR_OUT_OF_MEMORY,
            true
        ));
        assert!(!async_allocation_fallback_allowed(
            sys::HIP_ERROR_PRIOR_LAUNCH_FAILURE,
            true
        ));
        assert!(!async_allocation_fallback_allowed(
            sys::HIP_ERROR_NOT_SUPPORTED,
            false
        ));
    }

    #[test]
    fn call_failure_policy_distinguishes_queries_from_unknown_device_state() {
        for class in [
            HipCallFailureClass::PureQuery,
            HipCallFailureClass::OptionalConfiguration,
            HipCallFailureClass::PostDrainPoolMaintenance,
        ] {
            assert!(!class.quarantines(), "{class:?} must remain recoverable");
        }
        for class in [
            HipCallFailureClass::CaptureState,
            HipCallFailureClass::ExecutionMutation,
        ] {
            assert!(class.quarantines(), "{class:?} must quarantine");
        }
    }

    #[test]
    fn execution_policy_defaults_to_portable_kernels_and_legacy_host_barriers() {
        let policy = RocmExecutionPolicy::default();
        assert_eq!(
            policy.synchronization_mode,
            RocmSynchronizationMode::LegacyHostBarriers
        );
        assert_eq!(
            policy.tensor_kernels,
            RocmTensorKernelPolicy::portable_fallback()
        );
    }

    #[test]
    fn tensor_kernel_profiles_preserve_safety_and_close_accelerated_routes() {
        let qualified = RocmTensorKernelPolicy::qualified();
        let fallback = RocmTensorKernelPolicy::portable_fallback();
        let experimental = RocmTensorKernelPolicy::experimental_multiblock();

        assert_eq!(
            [
                qualified.split_paged_attention,
                qualified.gqa_paged_attention,
                qualified.gqa_d128_parallel,
                qualified.gqa_d256_parallel,
            ],
            [true; 4]
        );
        assert_eq!(
            [
                fallback.split_paged_attention,
                fallback.gqa_paged_attention,
                fallback.gqa_d128_parallel,
                fallback.gqa_d256_parallel,
            ],
            [false; 4]
        );
        assert_eq!(experimental, qualified);

        for policy in [qualified, fallback, experimental] {
            assert_eq!(policy.validation_error(), None);
            assert_eq!(policy.split_paged_attention_min_sequence, 2048);
            assert_eq!(policy.paged_attention_split_tokens, 128);
            assert_eq!(policy.paged_attention_max_splits, 256);
            assert!(policy.concat_safe_row_assembly);
            assert_eq!(policy.concat_safe_row_assembly_min_elements, 1_000_000);
            assert_eq!(
                policy.is_finite_host_scan_min_elements,
                Some(16 * 1024 * 1024)
            );
            assert_eq!(policy.rmsnorm_row_tile_rows, 4096);
        }

        assert_eq!(
            RocmTensorKernelPolicy {
                rmsnorm_row_tile_rows: 0,
                ..qualified
            }
            .validation_error(),
            Some("rmsnorm_row_tile_rows must be positive")
        );
    }

    #[test]
    fn flash_attention_profiles_preserve_bounded_composites_and_close_native_routes() {
        let qualified = RocmFlashAttentionPolicy::qualified();
        let fallback = RocmFlashAttentionPolicy::portable_fallback();
        let experimental = RocmFlashAttentionPolicy::experimental_multiblock();

        assert_eq!(qualified.validation_error(), None);
        assert_eq!(fallback.validation_error(), None);
        assert_eq!(experimental, qualified);
        assert_eq!(qualified.f32_matmul_inner_tile, 4096);
        assert_eq!(qualified.native_forward_query_tile, 4096);
        assert_eq!(qualified.native_streaming_forward_key_tile, 4096);
        assert_eq!(qualified.backward_precompute_delta_max_sequence, 1024);
        assert!(qualified.online_forward && qualified.online_backward);
        assert!(fallback.online_forward && fallback.online_backward);
        assert_eq!(
            [
                qualified.native_scalar_forward,
                qualified.native_tiled_forward,
                qualified.native_streaming_forward,
                qualified.native_rectangular_causal_forward,
                qualified.collapsed_gqa_backward,
                qualified.native_direct_collapsed_gqa_backward,
                qualified.native_gqa_qblock_forward,
                qualified.wmma_gqa_qblock_forward,
                qualified.wmma_gqa_r64k32_forward,
                qualified.wmma_gqa_r64k32_log2_forward,
            ],
            [true; 10]
        );
        assert_eq!(
            [
                fallback.native_scalar_forward,
                fallback.native_tiled_forward,
                fallback.native_streaming_forward,
                fallback.native_rectangular_causal_forward,
                fallback.collapsed_gqa_backward,
                fallback.native_direct_collapsed_gqa_backward,
                fallback.native_gqa_qblock_forward,
                fallback.wmma_gqa_qblock_forward,
                fallback.wmma_gqa_r64k32_forward,
                fallback.wmma_gqa_r64k32_log2_forward,
            ],
            [false; 10]
        );
        assert_eq!(
            fallback.native_backward_preference,
            RocmFlashAttentionRouteMode::Disabled
        );
        assert_eq!(
            RocmFlashAttentionPolicy {
                wmma_gqa_qblock_forward: true,
                native_gqa_qblock_forward: false,
                ..qualified
            }
            .validation_error(),
            Some("flash_attention WMMA GQA qblock requires native GQA qblock")
        );
    }

    #[test]
    fn invalid_flash_attention_policy_fails_before_device_probe() {
        let flash_attention = RocmFlashAttentionPolicy {
            native_forward_query_tile: 0,
            ..RocmFlashAttentionPolicy::qualified()
        };
        let invalid = RocmTensorKernelPolicy {
            flash_attention,
            ..RocmTensorKernelPolicy::qualified()
        };
        let error = RocmContext::new_with_execution_policy(
            usize::MAX,
            RocmExecutionPolicy::default().with_tensor_kernel_policy(invalid),
        )
        .expect_err("invalid flash policy must fail before the impossible ordinal is probed");

        assert_eq!(error.api, "RocmContext::new");
        assert_eq!(error.code, -1);
        assert_eq!(
            error.message,
            "invalid ROCm tensor-kernel policy: flash_attention.native_forward_query_tile must be positive"
        );
    }

    #[test]
    fn invalid_tensor_kernel_policy_fails_before_device_probe() {
        let invalid = RocmTensorKernelPolicy {
            rmsnorm_row_tile_rows: 0,
            ..RocmTensorKernelPolicy::qualified()
        };
        let error = RocmContext::new_with_execution_policy(
            usize::MAX,
            RocmExecutionPolicy::default().with_tensor_kernel_policy(invalid),
        )
        .expect_err("invalid policy must fail before the impossible ordinal is probed");

        assert_eq!(error.api, "RocmContext::new");
        assert_eq!(error.code, -1);
        assert_eq!(
            error.message,
            "invalid ROCm tensor-kernel policy: rmsnorm_row_tile_rows must be positive"
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
        assert_eq!(RocmSyncReason::CaptureRollback.as_str(), "capture_rollback");

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
        let hi_submission = hi
            .execution_submission("new_stream_with_priority_creates(high)")
            .expect("high-priority stream admission");
        let lo_submission = lo
            .execution_submission("new_stream_with_priority_creates(low)")
            .expect("low-priority stream admission");
        assert!(!hi_submission.raw_stream().is_null());
        assert!(!lo_submission.raw_stream().is_null());
        hi_submission.complete();
        lo_submission.complete();
    }
}
