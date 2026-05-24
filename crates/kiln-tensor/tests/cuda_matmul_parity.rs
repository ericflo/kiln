//! Parity test: `kiln_tensor::cuda_matmul` vs candle's `Tensor::matmul`.
//!
//! Phase 2.x of #1082 — closes the loop on the cublasLt-backed kt
//! matmul. The candle CUDA matmul (cublasGemmEx + tensor cores) is
//! the production reference; our cublasLt path must match it
//! element-wise within the BF16/F32 tolerance bands documented in
//! `bench-results/parity-tolerance.csv`.
//!
//! Gated on `--features cuda`; silently skips when no CUDA device is
//! reachable so it runs unmodified on the `linux-default` CI profile.

#![cfg(feature = "cuda")]

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{cuda_matmul, ops};

fn try_cuda() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

/// Deterministic small-magnitude pattern. Small magnitudes keep the
/// BF16 multiplication out of denormals while still exercising the
/// full mantissa.
fn pattern(n: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for _ in 0..n {
        s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E3779B97F4A7C15);
        let f = ((s as u32 % 1024) as f32 - 512.0) / 5120.0;
        out.push(f);
    }
    out
}

fn run_and_compare(m: usize, n: usize, k: usize, dtype: CandleDType, tolerance: f32) {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };

    let a_f32 = pattern(m * k, 1);
    let b_f32 = pattern(k * n, 2);
    let a_cd = CandleTensor::from_vec(a_f32, (m, k), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let b_cd = CandleTensor::from_vec(b_f32, (k, n), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    // Borrow into kt (zero-copy adapter).
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();

    // Run our cublasLt path.
    let c_kt = cuda_matmul(&a_kt, &b_kt).expect("cuda_matmul");

    // Sync.
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    // Candle reference (its own cublasGemmEx default-tensor-op path).
    let c_ref = a_cd.matmul(&b_cd).unwrap();
    let c_ref_f32: Vec<f32> = c_ref
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((m * n,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Pull kt result into a candle tensor for readback.
    let c_kt_candle = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&c_kt).unwrap();
    let got_f32 = c_kt_candle
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((m * n,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert_eq!(c_ref_f32.len(), got_f32.len());
    let mut max_abs = 0.0f32;
    for (a, b) in c_ref_f32.iter().zip(got_f32.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    assert!(
        max_abs < tolerance,
        "shape=({m},{n},{k}) dtype={dtype:?} parity drift = {max_abs} (expected < {tolerance})"
    );
}

#[test]
fn cuda_matmul_bf16_64x96x128() {
    run_and_compare(64, 96, 128, CandleDType::BF16, 5e-2);
}

#[test]
fn cuda_matmul_bf16_mlp_gate_up_shape_small() {
    // Same aspect ratio as Qwen3.5-4B's MLP gate||up, scaled down so
    // the test runs fast: [B*T, 256] @ [256, 1536] mirrors the
    // [B*T, 2560] @ [2560, 18432] production shape (factor 10).
    run_and_compare(128, 1536, 256, CandleDType::BF16, 5e-2);
}

#[test]
fn cuda_matmul_f32_32x48x64() {
    run_and_compare(32, 48, 64, CandleDType::F32, 1e-3);
}

#[test]
fn cuda_matmul_dispatches_through_ops_matmul() {
    // Validates the generic dispatch entry point (`ops::matmul`)
    // routes through `MatmulOp::cuda_fwd` when both inputs are on
    // CUDA.
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };

    let m = 16usize;
    let n = 32usize;
    let k = 24usize;
    let a_cd = CandleTensor::from_vec(pattern(m * k, 5), (m, k), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let b_cd = CandleTensor::from_vec(pattern(k * n, 6), (k, n), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();

    let c_kt = ops::matmul(&a_kt, &b_kt).expect("ops::matmul dispatch");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let c_ref_f32: Vec<f32> = a_cd
        .matmul(&b_cd)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((m * n,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let got_f32 = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&c_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((m * n,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let mut max_abs = 0.0f32;
    for (a, b) in c_ref_f32.iter().zip(got_f32.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    assert!(max_abs < 5e-2, "dispatch parity drift = {max_abs}");
}

#[test]
fn cuda_matmul_algo_cache_populated_after_call() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };

    let m = 8usize;
    let n = 16usize;
    let k = 24usize;
    let a_cd = CandleTensor::from_vec(pattern(m * k, 7), (m, k), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let b_cd = CandleTensor::from_vec(pattern(k * n, 8), (k, n), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();

    // Drive the call. After it returns, the shared algo cache should
    // contain at least one entry (for this (m, n, k, bf16, RR) tuple).
    let _ = cuda_matmul(&a_kt, &b_kt).unwrap();

    let snap = kiln_tensor::snapshot_algo_cache();
    assert!(
        snap.len() >= 1,
        "shared algo cache should be populated after a cuda_matmul call (got len={})",
        snap.len()
    );
}
