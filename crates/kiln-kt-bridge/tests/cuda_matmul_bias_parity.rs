//! Parity test: `cuda_matmul_with_bias` vs decomposed
//! `cuda_matmul + cuda_elementwise_binary(Add)`.
//!
//! Validates that the cublasLt `Bias` epilogue produces the same
//! result as a separate matmul + bias-add pair, within BF16/F32
//! tolerance bands.

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{cuda_matmul, cuda_matmul_with_bias};

fn try_cuda() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

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

fn run_parity(m: usize, n: usize, k: usize, dtype: CandleDType, tolerance: f32) {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };

    let a_data = pattern(m * k, 1);
    let b_data = pattern(k * n, 2);
    let bias_data = pattern(n, 3);

    let a_cd = CandleTensor::from_vec(a_data, (m, k), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let b_cd = CandleTensor::from_vec(b_data, (k, n), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let bias_cd = CandleTensor::from_vec(bias_data, (n,), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();
    let bias_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&bias_cd).unwrap();

    // Path A: fused matmul-with-bias via cublasLt's Bias epilogue.
    let c_fused = cuda_matmul_with_bias(&a_kt, &b_kt, &bias_kt).expect("fused");

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();
    let got_fused: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&c_fused)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((m * n,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Path B: candle baseline — matmul then broadcast-add bias.
    let c_baseline = a_cd
        .matmul(&b_cd)
        .unwrap()
        .broadcast_add(&bias_cd)
        .unwrap();
    let baseline_f32: Vec<f32> = c_baseline
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((m * n,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert_eq!(got_fused.len(), baseline_f32.len());
    let mut max_abs = 0.0f32;
    for (a, b) in baseline_f32.iter().zip(got_fused.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    assert!(
        max_abs < tolerance,
        "fused vs baseline drift = {max_abs} (expected < {tolerance})"
    );
}

#[test]
fn cuda_matmul_with_bias_bf16_64x96x128() {
    run_parity(64, 96, 128, CandleDType::BF16, 5e-2);
}

#[test]
fn cuda_matmul_with_bias_f32_32x48x64() {
    run_parity(32, 48, 64, CandleDType::F32, 1e-3);
}

#[test]
fn cuda_matmul_with_bias_validates_bias_shape() {
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
    // Bias of WRONG length.
    let bad_bias_cd = CandleTensor::from_vec(pattern(n + 7, 7), (n + 7,), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();

    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();
    let bad_bias_kt =
        kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&bad_bias_cd).unwrap();

    let result = cuda_matmul_with_bias(&a_kt, &b_kt, &bad_bias_kt);
    assert!(result.is_err(), "expected bias-length error");
}
