//! Capture-lifetime integration test (Phase 5 + Phase 1.27/1.28).
//!
//! Demonstrates the full capture-lifetime contract end-to-end on the
//! CPU smoke path:
//!
//! 1. Allocator starts in `Pool` mode and is pre-warmed for the
//!    sizes the captured graph will need (`warm(dtype, n, count)`).
//! 2. `CaptureSession::begin()` is called; the allocator transitions
//!    to `Frozen` mode.
//! 3. Each tensor the captured graph reads is pinned via
//!    `session.pin(&tensor)`. The graph-equivalent ops run on already-
//!    warm Frozen-mode allocations; cache-miss allocations error.
//! 4. `session.finalize()` marks the session ready for replay.
//! 5. On each replay, `session.audit_pinned(&live)` verifies no pinned
//!    TensorId has gone missing from the live set.
//!
//! The full per-backend impls (`kiln-graph-cuda`, `kiln-graph-metal`,
//! `kiln-graph-vulkan`) reuse this exact contract — the CPU smoke is
//! the canonical reference.

use std::collections::HashSet;

use kiln_graph::{AllocatorMode, CaptureError, CaptureSession};
use kiln_tensor as kt;

/// Happy path: pre-warm, begin, pin, alloc-from-warm, finalize,
/// audit, succeed.
#[test]
fn capture_lifetime_happy_path() {
    let mut allocator = kt::CpuAllocator::new();

    // 1. Pre-warm the pool with the sizes the captured graph needs.
    allocator.warm(kt::DType::F32, 16, 2);
    allocator.warm(kt::DType::BF16, 32, 1);

    // 2. Enter the capture window.
    let mut session = CaptureSession::begin();
    allocator.set_mode(AllocatorMode::Frozen).unwrap();
    assert_eq!(allocator.mode(), AllocatorMode::Frozen);

    // 3. Allocate from the warm pool. These tensors get pinned to
    //    the session because the captured graph will dereference
    //    their device pointers on replay.
    let f32_a = allocator.alloc(kt::DType::F32, 16).unwrap();
    let f32_b = allocator.alloc(kt::DType::F32, 16).unwrap();
    let bf16_a = allocator.alloc(kt::DType::BF16, 32).unwrap();
    let t_f32_a = kt::Tensor::from_parts(
        f32_a,
        kt::Layout::contiguous(vec![16]),
        kt::TensorId::next(),
    )
    .unwrap();
    let t_f32_b = kt::Tensor::from_parts(
        f32_b,
        kt::Layout::contiguous(vec![16]),
        kt::TensorId::next(),
    )
    .unwrap();
    let t_bf16_a = kt::Tensor::from_parts(
        bf16_a,
        kt::Layout::contiguous(vec![32]),
        kt::TensorId::next(),
    )
    .unwrap();
    session.pin(&t_f32_a);
    session.pin(&t_f32_b);
    session.pin(&t_bf16_a);
    assert_eq!(session.pinned().len(), 3);

    // 4. Finalize the session (a per-backend impl would now have a
    //    CapturedGraph in hand).
    session.finalize();
    assert!(session.is_finalized());

    // 5. Audit: every pinned id must appear in `live`.
    let live: HashSet<kt::TensorId> = [t_f32_a.id(), t_f32_b.id(), t_bf16_a.id()]
        .into_iter()
        .collect();
    session.audit_pinned(&live).unwrap();
}

/// Frozen-mode rejection: trying to allocate a size that wasn't
/// pre-warmed returns the standard allocator error.
#[test]
fn capture_lifetime_rejects_non_warm_alloc() {
    let mut allocator = kt::CpuAllocator::new();
    allocator.warm(kt::DType::F32, 16, 1);

    let mut session = CaptureSession::begin();
    allocator.set_mode(AllocatorMode::Frozen).unwrap();

    // 16-element F32 was warmed; 32-element F32 was not.
    let _ok = allocator.alloc(kt::DType::F32, 16).unwrap();
    let e = allocator.alloc(kt::DType::F32, 32).unwrap_err();
    assert!(e.to_string().contains("Frozen"));
    assert!(e.to_string().contains("Pre-warm"));

    // The session itself isn't aware of the failed alloc — that's the
    // per-backend impl's concern. We assert here that we *can*
    // continue using the session for the items that did alloc.
    session.finalize();
    assert!(session.is_finalized());
}

/// Dangling-pointer detection: if a pinned tensor is dropped before
/// replay, `audit_pinned` returns `CaptureError::DanglingPointer`.
#[test]
fn capture_lifetime_dangling_pointer_detection() {
    let mut allocator = kt::CpuAllocator::new();
    allocator.warm(kt::DType::F32, 16, 1);

    let mut session = CaptureSession::begin();
    allocator.set_mode(AllocatorMode::Frozen).unwrap();

    let storage = allocator.alloc(kt::DType::F32, 16).unwrap();
    let tensor = kt::Tensor::from_parts(
        storage,
        kt::Layout::contiguous(vec![16]),
        kt::TensorId::next(),
    )
    .unwrap();
    let pinned_id = tensor.id();
    session.pin(&tensor);

    // "Drop" the tensor — represented by an empty live set on replay.
    drop(tensor);

    let live: HashSet<kt::TensorId> = HashSet::new();
    let e = session.audit_pinned(&live).unwrap_err();
    match e {
        CaptureError::DanglingPointer { tensor_id } => assert_eq!(tensor_id, pinned_id),
        other => panic!("expected DanglingPointer, got {other:?}"),
    }
}

/// Mode transitions across the lifecycle: Pool -> Frozen -> Pool.
/// After the session ends, the allocator can return to Pool mode and
/// continue normal operation.
#[test]
fn capture_lifetime_post_session_returns_to_pool() {
    let mut allocator = kt::CpuAllocator::new();
    allocator.set_mode(AllocatorMode::Pool).unwrap();
    allocator.warm(kt::DType::F32, 8, 1);

    // Enter capture
    let _session = CaptureSession::begin();
    allocator.set_mode(AllocatorMode::Frozen).unwrap();
    let _s = allocator.alloc(kt::DType::F32, 8).unwrap();

    // Exit capture (real callers drop the session + restore mode)
    allocator.set_mode(AllocatorMode::Pool).unwrap();
    assert_eq!(allocator.mode(), AllocatorMode::Pool);

    // Pool mode: cache miss falls back to fresh alloc, no error.
    let s = allocator.alloc(kt::DType::F32, 64).unwrap();
    assert_eq!(s.byte_len(), 256);
}

/// Reserved-bytes accounting carries through the lifecycle without
/// double-counting cache reuse.
#[test]
fn capture_lifetime_reserved_bytes_accounting() {
    let mut allocator = kt::CpuAllocator::new();
    allocator.warm(kt::DType::F32, 16, 3);
    let warm_bytes = allocator.reserved_bytes();
    assert_eq!(warm_bytes, 3 * 64);

    let _session = CaptureSession::begin();
    allocator.set_mode(AllocatorMode::Frozen).unwrap();

    // Allocating from the cache does NOT increment reserved_bytes
    // (the bytes are already accounted for in `warm()`).
    let _s = allocator.alloc(kt::DType::F32, 16).unwrap();
    assert_eq!(allocator.reserved_bytes(), warm_bytes);

    // Peak unchanged.
    assert_eq!(allocator.peak_reserved_bytes(), warm_bytes);
}
