//! Phase R.9-1 — `rocm_write_host_in_place` round-trip + device-pointer
//! stability. This is the foundational sync-free (modulo a trailing stream
//! sync) in-place H2D writer the HIP-graph replay path uses to refresh the
//! per-step decode inputs (token id, position, rotary tables, paged metadata)
//! WITHOUT reallocating — the captured graph bakes the destination pointer, so
//! the core invariant is that the pointer never changes across a write.
//!
//! Run: `cargo test -p kiln-tensor --features rocm --test rocm_write_in_place`
#![cfg(feature = "rocm")]

use kiln_tensor::{Device, RocmStorage, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping rocm_write_host_in_place test");
        true
    } else {
        false
    }
}

/// Raw device base pointer behind a ROCm tensor (the value the captured graph
/// would bake in).
fn dev_ptr(t: &Tensor) -> u64 {
    t.storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .expect("rocm storage")
        .device_ptr_raw()
        .0
}

#[test]
fn write_in_place_roundtrip_and_pointer_stable() {
    if no_rocm() {
        return;
    }
    // Owned f32 buffer, contiguous, start_offset == 0.
    let t = Tensor::from_vec_on(Device::Rocm(0), vec![0f32; 8], vec![8]).expect("from_vec_on");
    let p0 = dev_ptr(&t);

    let host = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    kiln_tensor::rocm_write_host_in_place(&t, &host).expect("write_host_in_place");
    // The fn syncs internally, but sync the compute stream too to be explicit.
    kiln_tensor::rocm_synchronize_compute_stream(0).expect("sync");

    let back = kiln_tensor::rocm_to_host_copy(&t)
        .expect("dtoh")
        .to_vec::<f32>()
        .expect("to_vec f32");
    assert_eq!(back, host.to_vec(), "in-place write must land the new contents");

    let p1 = dev_ptr(&t);
    assert_eq!(p0, p1, "device pointer MUST be stable across an in-place write");
}

#[test]
fn write_in_place_last_write_wins() {
    if no_rocm() {
        return;
    }
    let t = Tensor::from_vec_on(Device::Rocm(0), vec![0f32; 4], vec![4]).expect("from_vec_on");
    kiln_tensor::rocm_write_host_in_place(&t, &[9.0f32, 9.0, 9.0, 9.0]).expect("write 1");
    kiln_tensor::rocm_write_host_in_place(&t, &[1.0f32, 2.0, 3.0, 4.0]).expect("write 2");
    kiln_tensor::rocm_synchronize_compute_stream(0).expect("sync");
    let back = kiln_tensor::rocm_to_host_copy(&t)
        .expect("dtoh")
        .to_vec::<f32>()
        .expect("to_vec");
    assert_eq!(back, vec![1.0, 2.0, 3.0, 4.0], "second write must win (same-stream ordering)");
}

#[test]
fn write_in_place_rejects_bad_inputs() {
    if no_rocm() {
        return;
    }
    let t = Tensor::from_vec_on(Device::Rocm(0), vec![0f32; 8], vec![8]).expect("from_vec_on");
    // length mismatch
    assert!(
        kiln_tensor::rocm_write_host_in_place(&t, &[1.0f32, 2.0]).is_err(),
        "host len != element count must error"
    );
    // wrong element byte width (u8, 1 byte, into a 4-byte f32 tensor)
    assert!(
        kiln_tensor::rocm_write_host_in_place(&t, &[0u8; 8]).is_err(),
        "mismatched element byte width must error"
    );
    // non-contiguous / start_offset != 0 (a narrow view)
    let big = Tensor::from_vec_on(Device::Rocm(0), (0..16).map(|i| i as f32).collect(), vec![16])
        .expect("from_vec_on big");
    let view = big.narrow(0, 4, 8).expect("narrow");
    assert!(
        kiln_tensor::rocm_write_host_in_place(&view, &[0f32; 8]).is_err(),
        "start_offset != 0 must error"
    );
}
