#![cfg(feature = "cuda")]
//! #1082 CP-4 Step-A kt-chaining primitives. The exercised logic is CPU-only
//! (candle/kt CPU tensors), but `kiln_kt_bridge::tape_bridge` is cuda-gated,
//! so the test is gated to the `cuda` feature to compile.
//!
//! `retain_output_for_chaining` + `kt_input_for_candle` let a downstream
//! adapter reuse an upstream adapter's kt output tensor (threading the same
//! kt `TensorId` output→input), so the recorded kt `Tape` is CONNECTED — the
//! prerequisite for a tape-authoritative backward walk over an adapter chain.

use candle_core::{Device as CDevice, Tensor as CTensor};
use kiln_kt_bridge::tape_bridge::{
    kt_input_for_candle, retain_output_for_chaining, with_io_mapping_scope,
};

#[test]
fn kt_input_for_candle_reuses_retained_output_in_scope() {
    let c = CTensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), &CDevice::Cpu).unwrap();
    let kt = kiln_tensor::Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
    let kt_id = kt.id();

    // Outside any scope: always None (no-op).
    assert!(kt_input_for_candle(c.id()).is_none());

    with_io_mapping_scope(|| {
        // Before retain: unknown candle id -> None.
        assert!(kt_input_for_candle(c.id()).is_none());

        retain_output_for_chaining(&kt, c.id());

        // After retain: returns the SAME kt tensor (same id) — this is the
        // connectivity that lets a downstream adapter thread the producer's
        // kt output into its own input, keeping the recorded tape connected.
        let got = kt_input_for_candle(c.id()).expect("retained kt output present");
        assert_eq!(got.id(), kt_id, "must reuse the producer's kt tensor id");

        // A candle tensor NOT produced by an adapter -> None (fresh-borrow).
        let other = CTensor::from_vec(vec![9.0f32], (1,), &CDevice::Cpu).unwrap();
        assert!(kt_input_for_candle(other.id()).is_none());
    });

    // Scope closed: retained entries are gone.
    assert!(kt_input_for_candle(c.id()).is_none());
}

#[test]
fn with_io_mapping_scope_clears_on_panic() {
    // The scope guard must clear even if the body panics, so a later scope
    // can open without hitting the nested-scope assertion.
    let r = std::panic::catch_unwind(|| {
        with_io_mapping_scope(|| {
            let c = CTensor::from_vec(vec![1.0f32], (1,), &CDevice::Cpu).unwrap();
            let kt = kiln_tensor::Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
            retain_output_for_chaining(&kt, c.id());
            panic!("boom");
        })
    });
    assert!(r.is_err(), "inner panic should propagate");
    // A fresh scope must open cleanly (no leaked scope from the panic).
    with_io_mapping_scope(|| {
        let c = CTensor::from_vec(vec![2.0f32], (1,), &CDevice::Cpu).unwrap();
        assert!(kt_input_for_candle(c.id()).is_none());
    });
}
