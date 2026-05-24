//! Parity test: kt CUDA RMSNorm (`RmsNormOp::cuda_fwd` /
//! `cuda_rmsnorm_last_axis`) vs kt CPU reference (`ops::rms_norm`).
//!
//! Phase 4 substrate validation. Confirms the kernel in
//! `csrc/rmsnorm.cu` (per-row sum-of-squares + rsqrt + per-element
//! scale by `weight`) produces outputs matching the canonical CPU
//! reference.

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{cuda_rmsnorm_last_axis, ops, Tensor};

fn try_cuda() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

fn pattern(n: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for _ in 0..n {
        s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E3779B97F4A7C15);
        // Range roughly [-1, 1) — same shape as the softmax/l2norm
        // parity tests.
        let f = ((s as u32 % 2048) as f32 - 1024.0) / 1024.0;
        out.push(f);
    }
    out
}

fn pattern_weight(n: usize, seed: u64) -> Vec<f32> {
    // Weights are typically positive, ~0.5-1.5 range in real RMSNorm
    // layers. Generate that distribution rather than the symmetric
    // [-1, 1) used for the inputs.
    let mut out = Vec::with_capacity(n);
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for _ in 0..n {
        s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E3779B97F4A7C15);
        let f = 0.5 + ((s as u32 % 1024) as f32) / 1024.0;
        out.push(f);
    }
    out
}

fn cpu_reference(
    x_data: &[f32],
    w_data: &[f32],
    n_rows: usize,
    n_cols: usize,
    eps: f32,
) -> Vec<f32> {
    let x = Tensor::from_slice(x_data, vec![n_rows, n_cols]).unwrap();
    let w = Tensor::from_slice(w_data, vec![n_cols]).unwrap();
    let y = ops::rms_norm(&x, &w, eps).unwrap();
    let cpu_storage = y
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap();
    let bytes = cpu_storage.as_bytes();
    let mut out = Vec::with_capacity(n_rows * n_cols);
    for i in 0..(n_rows * n_cols) {
        out.push(f32::from_le_bytes(
            bytes[i * 4..i * 4 + 4].try_into().unwrap(),
        ));
    }
    out
}

fn run_rmsnorm_parity(
    n_rows: usize,
    n_cols: usize,
    dtype: CandleDType,
    eps: f32,
    tolerance: f32,
) {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n = n_rows * n_cols;
    let x_data = pattern(n, 13);
    let w_data = pattern_weight(n_cols, 47);

    let x_cd = CandleTensor::from_vec(x_data.clone(), (n_rows, n_cols), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let w_cd = CandleTensor::from_vec(w_data.clone(), (n_cols,), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();
    let w_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&w_cd).unwrap();

    let out_kt = cuda_rmsnorm_last_axis(&x_kt, &w_kt, eps).expect("rmsnorm");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    // Reference: kt CPU RMSNorm over the same F32 values.
    let ref_vec = cpu_reference(&x_data, &w_data, n_rows, n_cols, eps);

    let got_vec: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((n,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert_eq!(ref_vec.len(), got_vec.len());
    let mut max_abs = 0.0f32;
    for (a, b) in ref_vec.iter().zip(got_vec.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    assert!(
        max_abs < tolerance,
        "rows={n_rows} cols={n_cols} dtype={dtype:?} eps={eps} max_abs={max_abs} > {tolerance}"
    );
}

#[test]
fn cuda_rmsnorm_f32_4_rows_512_cols() {
    run_rmsnorm_parity(4, 512, CandleDType::F32, 1e-6, 1e-5);
}

#[test]
fn cuda_rmsnorm_f32_8_rows_2560_cols() {
    // Qwen3.5-4B hidden dim is 2560 — exercises the strided per-thread
    // accumulation when n_cols > MAX_THREADS.
    run_rmsnorm_parity(8, 2560, CandleDType::F32, 1e-6, 1e-5);
}

#[test]
fn cuda_rmsnorm_bf16_8_rows_64_cols() {
    // BF16 has ~3 decimal digits of precision; widen the tolerance.
    run_rmsnorm_parity(8, 64, CandleDType::BF16, 1e-6, 5e-2);
}

#[test]
fn cuda_rmsnorm_bf16_2_rows_2560_cols() {
    // BF16 at Qwen3.5-4B hidden dim — both the production hot-path
    // and the strided-per-thread reduction stress case.
    run_rmsnorm_parity(2, 2560, CandleDType::BF16, 1e-6, 5e-2);
}

#[test]
fn cuda_rmsnorm_dispatches_through_ops_rms_norm() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n_rows = 4;
    let n_cols = 256;
    let eps = 1e-6_f32;
    let x_data = pattern(n_rows * n_cols, 23);
    let w_data = pattern_weight(n_cols, 71);
    let x_cd = CandleTensor::from_vec(x_data.clone(), (n_rows, n_cols), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let w_cd = CandleTensor::from_vec(w_data.clone(), (n_cols,), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();
    let w_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&w_cd).unwrap();

    // ops::rms_norm should pick the CUDA path automatically via
    // RmsNormOp::cuda_fwd.
    let out_kt = ops::rms_norm(&x_kt, &w_kt, eps).expect("dispatch");

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got_vec: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((n_rows * n_cols,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Reference: kt CPU RMSNorm.
    let ref_vec = cpu_reference(&x_data, &w_data, n_rows, n_cols, eps);
    let mut max_abs = 0.0f32;
    for (a, b) in ref_vec.iter().zip(got_vec.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    assert!(max_abs < 5e-2, "dispatch parity max_abs={max_abs}");
}
