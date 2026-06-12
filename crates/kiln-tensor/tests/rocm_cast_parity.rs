//! Phase R.5 — CPU-vs-ROCm parity for the elementwise dtype `cast` kernel.
//!
//! `cast.cu` is one-thread-per-element (no cross-lane reductions), so it is not
//! a wave-size hazard — a couple of shapes exercise every cast tag. We sweep the
//! full F32 ↔ BF16 ↔ F16 matrix and compare against a CPU reference that uses
//! the `half` crate's round-to-nearest conversions (identical to the kernel's
//! `__float2bfloat16` / `__float2half`). Skips when no ROCm device is present.
//!
//! Run: `cargo test -p kiln-tensor --features rocm --test rocm_cast_parity`
#![cfg(feature = "rocm")]

use half::{bf16, f16};
use kiln_tensor::{DType, Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.5 cast parity test");
        true
    } else {
        false
    }
}

/// Deterministic value in ~[-8, 8) for index i — covers a spread of magnitudes
/// (and exact-zero) so BF16/F16 rounding is genuinely exercised.
fn val(i: usize) -> f32 {
    (((i * 37 + 11) % 1600) as f32) / 100.0 - 8.0
}

/// One-element f32 → BF16 → f32 reference round-trip (round-to-nearest).
fn ref_to_bf16(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}
/// One-element f32 → F16 → f32 reference round-trip (round-to-nearest).
fn ref_to_f16(x: f32) -> f32 {
    f16::from_f32(x).to_f32()
}

/// Build an f32 device tensor of `n` deterministic values on Rocm(0).
fn dev_f32(n: usize) -> (Tensor, Vec<f32>) {
    let data: Vec<f32> = (0..n).map(val).collect();
    let t = Tensor::from_vec_on(Device::Rocm(0), data.clone(), vec![n])
        .unwrap_or_else(|e| panic!("from_vec_on f32 (n={n}): {e}"));
    (t, data)
}

/// Read a device tensor back to host and return its values as f32, regardless of
/// the device dtype (BF16/F16 are widened via the `half` crate).
fn dev_to_f32(t: &Tensor) -> Vec<f32> {
    let host = kiln_tensor::rocm_to_host_copy(t).expect("rocm_to_host_copy");
    match t.dtype() {
        DType::F32 => host.to_vec::<f32>().expect("to_vec f32"),
        DType::BF16 => host
            .to_vec::<bf16>()
            .expect("to_vec bf16")
            .into_iter()
            .map(|v| v.to_f32())
            .collect(),
        DType::F16 => host
            .to_vec::<f16>()
            .expect("to_vec f16")
            .into_iter()
            .map(|v| v.to_f32())
            .collect(),
        other => panic!("unexpected dtype {other}"),
    }
}

fn assert_close(got: &[f32], reference: &[f32], label: &str) {
    assert_eq!(got.len(), reference.len(), "{label}: length mismatch");
    for (i, (g, r)) in got.iter().zip(reference.iter()).enumerate() {
        let diff = (g - r).abs();
        assert!(
            diff <= 1e-5 + 1e-4 * r.abs(),
            "{label} mismatch at idx={i}: got {g} ref {r} diff {diff}"
        );
    }
}

#[test]
fn cast_parity_full_matrix() {
    if no_rocm() {
        return;
    }
    // A couple of sizes (incl. non-multiple-of-block) suffice for an elementwise
    // kernel; the boundary sweep is for reductions, not this op.
    for &n in &[1usize, 5, 256, 257, 1000] {
        let (f32_dev, f32_host) = dev_f32(n);

        // F32 -> BF16
        let to_bf16 = kiln_tensor::rocm_cast(&f32_dev, DType::BF16)
            .unwrap_or_else(|e| panic!("rocm_cast f32->bf16 (n={n}): {e}"));
        assert_eq!(to_bf16.dtype(), DType::BF16);
        let ref_bf16: Vec<f32> = f32_host.iter().map(|&x| ref_to_bf16(x)).collect();
        assert_close(
            &dev_to_f32(&to_bf16),
            &ref_bf16,
            &format!("f32->bf16 n={n}"),
        );

        // F32 -> F16
        let to_f16 = kiln_tensor::rocm_cast(&f32_dev, DType::F16)
            .unwrap_or_else(|e| panic!("rocm_cast f32->f16 (n={n}): {e}"));
        assert_eq!(to_f16.dtype(), DType::F16);
        let ref_f16: Vec<f32> = f32_host.iter().map(|&x| ref_to_f16(x)).collect();
        assert_close(&dev_to_f32(&to_f16), &ref_f16, &format!("f32->f16 n={n}"));

        // BF16 -> F32 (round-trips the already-rounded bf16 values exactly).
        let bf16_to_f32 = kiln_tensor::rocm_cast(&to_bf16, DType::F32)
            .unwrap_or_else(|e| panic!("rocm_cast bf16->f32 (n={n}): {e}"));
        assert_eq!(bf16_to_f32.dtype(), DType::F32);
        assert_close(
            &dev_to_f32(&bf16_to_f32),
            &ref_bf16,
            &format!("bf16->f32 n={n}"),
        );

        // F16 -> F32
        let f16_to_f32 = kiln_tensor::rocm_cast(&to_f16, DType::F32)
            .unwrap_or_else(|e| panic!("rocm_cast f16->f32 (n={n}): {e}"));
        assert_eq!(f16_to_f32.dtype(), DType::F32);
        assert_close(
            &dev_to_f32(&f16_to_f32),
            &ref_f16,
            &format!("f16->f32 n={n}"),
        );

        // BF16 -> F16: kernel widens bf16 to f32 then narrows to f16. Reference
        // mirrors that exact two-step path.
        let bf16_to_f16 = kiln_tensor::rocm_cast(&to_bf16, DType::F16)
            .unwrap_or_else(|e| panic!("rocm_cast bf16->f16 (n={n}): {e}"));
        assert_eq!(bf16_to_f16.dtype(), DType::F16);
        let ref_bf16_f16: Vec<f32> = ref_bf16.iter().map(|&x| ref_to_f16(x)).collect();
        assert_close(
            &dev_to_f32(&bf16_to_f16),
            &ref_bf16_f16,
            &format!("bf16->f16 n={n}"),
        );

        // F16 -> BF16: widen f16 to f32 then narrow to bf16.
        let f16_to_bf16 = kiln_tensor::rocm_cast(&to_f16, DType::BF16)
            .unwrap_or_else(|e| panic!("rocm_cast f16->bf16 (n={n}): {e}"));
        assert_eq!(f16_to_bf16.dtype(), DType::BF16);
        let ref_f16_bf16: Vec<f32> = ref_f16.iter().map(|&x| ref_to_bf16(x)).collect();
        assert_close(
            &dev_to_f32(&f16_to_bf16),
            &ref_f16_bf16,
            &format!("f16->bf16 n={n}"),
        );
    }
    eprintln!("cast CPU-vs-ROCm parity passed across the F32<->BF16<->F16 matrix");
}
