//! Phase R.9-2 — ROCm capture arena (freeze-pointers) invariants.
//!
//! The arena is what keeps the captured decode graph's Q/K/V/activation device
//! pointers STABLE across capture→replay: Record pass mints real owned buffers
//! and hands out Borrowed views; Replay pass hands out Borrowed views of the
//! SAME buffers (no `hipMallocAsync` inside the capture window) so every
//! recorded device pointer is frozen. `zero=true` buffers get a captured
//! `hipMemsetD8Async` on replay (read-before-write correctness).
//!
//! `Tensor::zeros_on(Rocm)` routes through `RocmStorage::zeros_ctx`, so the
//! arena intercepts Tensor allocations — we drive it through Tensors and read
//! back with the public copy/write helpers.
//!
//! Run: `cargo test -p kiln-tensor --features rocm --test rocm_capture_arena`
#![cfg(feature = "rocm")]

use std::cell::RefCell;
use std::rc::Rc;

use kiln_tensor::{
    primary_rocm_context, rocm_capture_arena_active, with_rocm_capture_arena, DType, Device,
    RocmCaptureArena, RocmStorage, Tensor,
};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping capture-arena test");
        true
    } else {
        false
    }
}

/// Raw device base pointer behind a (ROCm-backed) tensor.
fn tptr(t: &Tensor) -> u64 {
    t.storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .expect("rocm storage")
        .device_ptr_raw()
        .0
}

fn t_is_borrowed(t: &Tensor) -> bool {
    t.storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .expect("rocm storage")
        .is_borrowed()
}

fn new_arena() -> Rc<RefCell<RocmCaptureArena>> {
    let ctx = primary_rocm_context(0).expect("primary_rocm_context");
    Rc::new(RefCell::new(RocmCaptureArena::new_record(ctx, 0)))
}

#[test]
fn freeze_pointers_and_replay_memset() {
    if no_rocm() {
        return;
    }
    let shapes: [(usize, DType); 3] = [(8, DType::F32), (16, DType::F32), (4, DType::F32)];
    let arena = new_arena();

    // --- Record pass: real owned buffers, Borrowed views handed back.
    let (recorded, t0_record): (Vec<u64>, Tensor) = with_rocm_capture_arena(arena.clone(), || {
        let mut ptrs = Vec::new();
        let mut first = None;
        for (n, dt) in shapes {
            let t = Tensor::zeros_on(Device::Rocm(0), vec![n], dt).expect("zeros_on record");
            assert!(t_is_borrowed(&t), "arena must hand out Borrowed views");
            ptrs.push(tptr(&t));
            if first.is_none() {
                first = Some(t);
            }
        }
        (ptrs, first.unwrap())
    });
    assert_eq!(arena.borrow().buffer_count(), 3, "3 distinct buffers recorded");

    // Dirty the first frozen buffer with a non-zero sentinel (writes through the
    // stable pointer). On replay the arena's captured memset must re-zero it.
    kiln_tensor::rocm_write_host_in_place(&t0_record, &[7.0f32; 8]).expect("dirty write");
    kiln_tensor::rocm_synchronize_compute_stream(0).expect("sync");

    // --- Replay pass: SAME buffers, frozen pointers, zero=true → re-zeroed.
    arena.borrow_mut().begin_replay();
    let replayed: Vec<u64> = with_rocm_capture_arena(arena.clone(), || {
        shapes
            .iter()
            .map(|&(n, dt)| tptr(&Tensor::zeros_on(Device::Rocm(0), vec![n], dt).expect("zeros_on replay")))
            .collect()
    });
    assert_eq!(recorded, replayed, "FREEZE-POINTERS: replay must reuse the recorded device pointers");

    // The replay memset ran on the (default) active stream; sync then read back.
    kiln_tensor::rocm_synchronize_default_stream(0).expect("sync default");
    let back = kiln_tensor::rocm_to_host_copy(&t0_record)
        .expect("dtoh")
        .to_vec::<f32>()
        .expect("to_vec");
    assert_eq!(back, vec![0.0f32; 8], "replay memset must re-zero the dirtied zero=true buffer");
}

#[test]
fn replay_shape_mismatch_errors() {
    if no_rocm() {
        return;
    }
    let arena = new_arena();
    with_rocm_capture_arena(arena.clone(), || {
        let _ = Tensor::zeros_on(Device::Rocm(0), vec![8], DType::F32).expect("record");
    });
    arena.borrow_mut().begin_replay();
    let err = with_rocm_capture_arena(arena.clone(), || {
        // Different element count than recorded → the arena must reject it.
        Tensor::zeros_on(Device::Rocm(0), vec![16], DType::F32)
    });
    assert!(err.is_err(), "replay with a different shape must error (non-determinism guard)");
}

#[test]
fn replay_overrun_errors() {
    if no_rocm() {
        return;
    }
    let arena = new_arena();
    with_rocm_capture_arena(arena.clone(), || {
        let _ = Tensor::zeros_on(Device::Rocm(0), vec![8], DType::F32).expect("record one");
    });
    arena.borrow_mut().begin_replay();
    let (ok, overrun) = with_rocm_capture_arena(arena.clone(), || {
        let a = Tensor::zeros_on(Device::Rocm(0), vec![8], DType::F32);
        let b = Tensor::zeros_on(Device::Rocm(0), vec![8], DType::F32); // one past the recorded count
        (a, b)
    });
    assert!(ok.is_ok(), "first replay alloc reuses the recorded buffer");
    assert!(overrun.is_err(), "replaying more allocs than recorded must error");
}

#[test]
fn hook_is_noop_outside_scope() {
    if no_rocm() {
        return;
    }
    assert!(!rocm_capture_arena_active(), "no arena active outside a scope");
    // Outside any scope: a normal Owned allocation, zero behavior change.
    let t = Tensor::zeros_on(Device::Rocm(0), vec![8], DType::F32).expect("zeros_on");
    assert!(
        t.storage()
            .as_any()
            .downcast_ref::<RocmStorage>()
            .unwrap()
            .is_owned(),
        "outside the arena, allocations are Owned"
    );

    // Inside a Record scope: arena active, allocations are Borrowed views.
    let arena = new_arena();
    with_rocm_capture_arena(arena, || {
        assert!(rocm_capture_arena_active(), "arena active inside the scope");
        let t = Tensor::zeros_on(Device::Rocm(0), vec![8], DType::F32).expect("zeros_on in-scope");
        assert!(t_is_borrowed(&t), "inside the arena, allocations are Borrowed");
    });
    assert!(!rocm_capture_arena_active(), "scope restored on exit");
}

#[test]
fn record_single_alloc_pushes_one_buffer() {
    if no_rocm() {
        return;
    }
    // Re-entry proof: one zeros_on must push exactly ONE arena buffer (the inner
    // real constructor took the direct path, it did not recurse into the hook).
    let arena = new_arena();
    with_rocm_capture_arena(arena.clone(), || {
        let _ = Tensor::zeros_on(Device::Rocm(0), vec![8], DType::F32).expect("zeros_on");
    });
    assert_eq!(arena.borrow().buffer_count(), 1, "exactly one buffer per zeros_on (no recursion)");
}
