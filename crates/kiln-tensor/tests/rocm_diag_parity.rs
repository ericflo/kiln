//! Phase R.5 — CPU-vs-ROCm parity for the `diag` kernels.
//!
//! `diag.cu` is one-thread-per-diagonal-index (no cross-lane reductions), so it
//! is not a wave-size hazard — a handful of `n` values exercise the scatter /
//! gather index math. We cover both directions:
//!   * `rocm_diagonal_extract`: `[n, n]` -> `[n]` main-diagonal gather.
//!   * `rocm_diag_build`: `[n]` -> `[n, n]` zero matrix with `v` on the diagonal.
//! across the F32 / BF16 / F16 dtype matrix, compared to a CPU reference. Skips
//! when no ROCm device is present.
//!
//! Run: `cargo test -p kiln-tensor --features rocm --test rocm_diag_parity`
#![cfg(feature = "rocm")]

use half::{bf16, f16};
use kiln_tensor::{DType, Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.5 diag parity test");
        true
    } else {
        false
    }
}

/// Deterministic value in ~[-8, 8) for index i.
fn val(i: usize) -> f32 {
    (((i * 37 + 11) % 1600) as f32) / 100.0 - 8.0
}

/// Round an f32 through the device dtype (so the CPU reference matches the value
/// the kernel actually stores/loads for BF16/F16). F32 is identity.
fn round_dtype(x: f32, dtype: DType) -> f32 {
    match dtype {
        DType::F32 => x,
        DType::BF16 => bf16::from_f32(x).to_f32(),
        DType::F16 => f16::from_f32(x).to_f32(),
        other => panic!("unexpected dtype {other}"),
    }
}

/// Build a device tensor of the given dtype from f32 source values, returning the
/// tensor plus the dtype-rounded host values (the kernel sees these).
fn dev_tensor(data_f32: &[f32], shape: Vec<usize>, dtype: DType) -> (Tensor, Vec<f32>) {
    let rounded: Vec<f32> = data_f32.iter().map(|&x| round_dtype(x, dtype)).collect();
    let t = match dtype {
        DType::F32 => Tensor::from_vec_on(Device::Rocm(0), rounded.clone(), shape),
        DType::BF16 => {
            let v: Vec<bf16> = data_f32.iter().map(|&x| bf16::from_f32(x)).collect();
            Tensor::from_vec_on(Device::Rocm(0), v, shape)
        }
        DType::F16 => {
            let v: Vec<f16> = data_f32.iter().map(|&x| f16::from_f32(x)).collect();
            Tensor::from_vec_on(Device::Rocm(0), v, shape)
        }
        other => panic!("unexpected dtype {other}"),
    }
    .unwrap_or_else(|e| panic!("from_vec_on ({dtype}): {e}"));
    (t, rounded)
}

/// Read a device tensor back to host as f32 (widening BF16/F16 via `half`).
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
fn diagonal_extract_parity() {
    if no_rocm() {
        return;
    }
    for &n in &[1usize, 2, 7, 32, 64, 100, 257] {
        // Full [n, n] matrix of deterministic values.
        let mat: Vec<f32> = (0..n * n).map(val).collect();
        for &dtype in &[DType::F32, DType::BF16, DType::F16] {
            let (t, rounded) = dev_tensor(&mat, vec![n, n], dtype);
            let out = kiln_tensor::rocm_diagonal_extract(&t)
                .unwrap_or_else(|e| panic!("rocm_diagonal_extract (n={n}, {dtype}): {e}"));
            assert_eq!(out.dtype(), dtype);
            assert_eq!(out.shape(), &[n]);
            // CPU reference: out[i] = mat[i*n + i].
            let reference: Vec<f32> = (0..n).map(|i| rounded[i * n + i]).collect();
            assert_close(
                &dev_to_f32(&out),
                &reference,
                &format!("diagonal_extract n={n} {dtype}"),
            );
        }
    }
    eprintln!("diagonal_extract CPU-vs-ROCm parity passed (F32/BF16/F16)");
}

#[test]
fn diag_build_parity() {
    if no_rocm() {
        return;
    }
    for &n in &[1usize, 2, 7, 32, 64, 100, 257] {
        let vec_data: Vec<f32> = (0..n).map(val).collect();
        for &dtype in &[DType::F32, DType::BF16, DType::F16] {
            let (v, rounded) = dev_tensor(&vec_data, vec![n], dtype);
            let out = kiln_tensor::rocm_diag_build(&v)
                .unwrap_or_else(|e| panic!("rocm_diag_build (n={n}, {dtype}): {e}"));
            assert_eq!(out.dtype(), dtype);
            assert_eq!(out.shape(), &[n, n]);
            // CPU reference: [n, n] zeros with rounded[i] on the diagonal.
            let mut reference = vec![0.0f32; n * n];
            for i in 0..n {
                reference[i * n + i] = rounded[i];
            }
            assert_close(
                &dev_to_f32(&out),
                &reference,
                &format!("diag_build n={n} {dtype}"),
            );
        }
    }
    eprintln!("diag_build CPU-vs-ROCm parity passed (F32/BF16/F16)");
}
