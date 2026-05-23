//! `CaptureSession` — RAII guard for a capture lifetime.
//!
//! Wraps a `begin() → end()` block during which the allocator is in
//! `Frozen` mode and every tensor referenced by the captured graph
//! must outlive the session.
//!
//! # Anti-pattern: dangling pointers across capture/replay
//!
//! Per the Phase 5 bullet:
//!
//! > Any tensor whose `.device_ptr()` enters a captured graph must
//! > outlive every replay of that graph. Lifetimes are encoded in the
//! > type system where possible (`CapturedGraph<'a>` borrows from a
//! > `FrozenAllocator<'a>`) and enforced by a debug-assertion
//! > `kiln_tensor::audit_captured_pointers()` that walks the graph and
//! > verifies every recorded pointer still resolves to a live
//! > allocation.
//!
//! Today's session records the **set of pinned `TensorId`s** that the
//! per-backend graph captured. On replay (via
//! [`CaptureSession::audit_pinned`]) the session re-checks that
//! every recorded TensorId is still live in the registered Tensor
//! map. This is the runtime hook; the Phase 5.x `'a`-lifetime
//! encoding lands when `kiln-graph-cuda` ships and gets exercised
//! against a real capture.

use std::collections::HashSet;

use kiln_tensor::{Tensor, TensorId};

use crate::{AllocatorMode, CaptureError};

/// One pinned pointer that the captured graph dereferences during
/// replay. Today we just record the `TensorId`; Phase 5.x's per-
/// backend impls also record the raw device pointer for fast O(1)
/// dangling-pointer detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PinnedPointer {
    pub tensor_id: TensorId,
}

impl PinnedPointer {
    pub const fn new(tensor_id: TensorId) -> Self {
        PinnedPointer { tensor_id }
    }
}

/// RAII guard for a capture lifetime.
///
/// Usage shape (executed by per-backend impls):
///
/// ```ignore
/// let mut session = CaptureSession::begin();
/// // ... per-backend record-then-finalize calls ...
/// let captured = backend.finalize_capture(&mut session)?;
/// // session is dropped here; subsequent replay()s of `captured`
/// // call session.audit_pinned(&live_tensors) to detect dangling
/// // pointers.
/// ```
#[derive(Debug, Default)]
pub struct CaptureSession {
    mode: AllocatorMode,
    pinned: HashSet<PinnedPointer>,
    finalized: bool,
}

impl CaptureSession {
    /// Begin a capture. The session starts in `Frozen` allocator mode;
    /// any allocation during the session is an error.
    pub fn begin() -> Self {
        CaptureSession {
            mode: AllocatorMode::Frozen,
            pinned: HashSet::new(),
            finalized: false,
        }
    }

    /// Current allocator mode. Always `Frozen` while the session is
    /// open; the per-backend impl reads this to assert no allocation
    /// happens during capture.
    pub fn mode(&self) -> AllocatorMode {
        self.mode
    }

    /// Has the session been finalized (a `CapturedGraph` produced)?
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }

    /// Mark the session as finalized. Called by the per-backend impl
    /// after `cuGraphInstantiate` (or the equivalent) succeeds.
    pub fn finalize(&mut self) {
        self.finalized = true;
    }

    /// Record a pinned pointer for a tensor referenced by the captured
    /// graph. Phase 5.x's per-backend impls call this for every
    /// device pointer they bake into the graph.
    pub fn pin(&mut self, tensor: &Tensor) {
        self.pinned.insert(PinnedPointer::new(tensor.id()));
    }

    /// Borrow the set of pinned pointers.
    pub fn pinned(&self) -> &HashSet<PinnedPointer> {
        &self.pinned
    }

    /// Audit the pinned set against a list of live tensors.
    ///
    /// **Debug-build assertion** — released to runtime so the per-
    /// backend `CapturedGraph::replay()` impl can opt into the check
    /// via `cfg(debug_assertions)` (or always-on under a
    /// `KILN_AUDIT_GRAPHS=1` env). Returns
    /// [`CaptureError::DanglingPointer`] for the first pinned
    /// `TensorId` that no longer appears in `live`.
    pub fn audit_pinned(&self, live: &HashSet<TensorId>) -> Result<(), CaptureError> {
        for p in &self.pinned {
            if !live.contains(&p.tensor_id) {
                return Err(CaptureError::DanglingPointer {
                    tensor_id: p.tensor_id,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_tensor::{DType, Tensor};

    fn t() -> Tensor {
        Tensor::zeros_cpu(vec![2, 3], DType::F32)
    }

    #[test]
    fn begin_session_is_frozen() {
        let s = CaptureSession::begin();
        assert_eq!(s.mode(), AllocatorMode::Frozen);
        assert!(!s.is_finalized());
        assert!(s.pinned().is_empty());
    }

    #[test]
    fn pin_records_tensor_id() {
        let mut s = CaptureSession::begin();
        let a = t();
        s.pin(&a);
        assert_eq!(s.pinned().len(), 1);
        assert!(s.pinned().contains(&PinnedPointer::new(a.id())));
    }

    #[test]
    fn finalize_flips_flag() {
        let mut s = CaptureSession::begin();
        assert!(!s.is_finalized());
        s.finalize();
        assert!(s.is_finalized());
    }

    #[test]
    fn audit_pinned_ok_when_all_live() {
        let mut s = CaptureSession::begin();
        let a = t();
        let b = t();
        s.pin(&a);
        s.pin(&b);
        let live: HashSet<TensorId> = [a.id(), b.id()].into_iter().collect();
        s.audit_pinned(&live).unwrap();
    }

    #[test]
    fn audit_pinned_errors_on_dangling() {
        let mut s = CaptureSession::begin();
        let a = t();
        s.pin(&a);
        let live: HashSet<TensorId> = HashSet::new();
        let e = s.audit_pinned(&live).unwrap_err();
        match e {
            CaptureError::DanglingPointer { tensor_id } => assert_eq!(tensor_id, a.id()),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn pin_is_idempotent() {
        let mut s = CaptureSession::begin();
        let a = t();
        s.pin(&a);
        s.pin(&a);
        s.pin(&a);
        assert_eq!(s.pinned().len(), 1);
    }
}
