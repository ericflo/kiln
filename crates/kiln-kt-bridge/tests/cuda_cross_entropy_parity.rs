//! Parity test: kt CUDA cross-entropy
//! (`CrossEntropyOp::cuda_fwd` / `cuda_cross_entropy_loss`) vs kt CPU
//! reference (`ops::cross_entropy`).
//!
//! Phase 4 substrate validation. Confirms the kernel in
//! `csrc/cross_entropy.cu` produces a scalar loss matching the
//! canonical CPU reference for F32/BF16 logits and I64/U32 targets.

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{cuda_cross_entropy_loss, ops, Tensor};

fn try_cuda() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

fn pattern(n: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for _ in 0..n {
        s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E3779B97F4A7C15);
        let f = ((s as u32 % 2048) as f32 - 1024.0) / 1024.0;
        out.push(f);
    }
    out
}

fn cpu_reference_f32(logits: &[f32], targets: &[i64], batch: usize, vocab: usize) -> f32 {
    let x_kt = Tensor::from_slice(logits, vec![batch, vocab]).unwrap();
    let t_kt = Tensor::from_slice(targets, vec![batch]).unwrap();
    let loss = ops::cross_entropy(&x_kt, &t_kt).unwrap();
    let cpu_storage = loss
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap();
    f32::from_le_bytes(cpu_storage.as_bytes()[0..4].try_into().unwrap())
}

fn run_xent_parity(
    batch: usize,
    vocab: usize,
    dtype: CandleDType,
    target_dtype: CandleDType,
    tolerance: f32,
) {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n = batch * vocab;
    let data = pattern(n, 11);

    // Targets: a deterministic pseudo-random distribution into [0, vocab).
    let targets_i64: Vec<i64> = (0..batch)
        .map(|b| ((b as u64 * 0x9E3779B97F4A7C15) % vocab as u64) as i64)
        .collect();

    let x_cd = CandleTensor::from_vec(data.clone(), (batch, vocab), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let t_cd_i64 = CandleTensor::from_vec(targets_i64.clone(), (batch,), &dev).unwrap();
    let t_cd = t_cd_i64.to_dtype(target_dtype).unwrap();
    let t_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&t_cd).unwrap();

    let loss_kt = cuda_cross_entropy_loss(&x_kt, &t_kt).expect("cross_entropy");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    // Reference: kt CPU cross-entropy at the *cast* dtype, so any
    // precision skew from BF16 round-trip matches.
    let ref_data: Vec<f32> = match dtype {
        CandleDType::F32 => data.clone(),
        CandleDType::BF16 => data
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect(),
        CandleDType::F16 => data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect(),
        _ => panic!("unsupported dtype"),
    };
    // The CPU reference still operates on F32 logits + I64 targets;
    // we widen the comparison.
    let cpu_loss = cpu_reference_f32(&ref_data, &targets_i64, batch, vocab);

    // Pull GPU result back to host as F32.
    let got_cd = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&loss_kt).unwrap();
    let got_f32: f32 = got_cd
        .to_dtype(CandleDType::F32)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

    let d = (cpu_loss - got_f32).abs();
    assert!(
        d < tolerance,
        "batch={batch} vocab={vocab} dtype={dtype:?} target_dtype={target_dtype:?} cpu={cpu_loss} cuda={got_f32} diff={d} > {tolerance}"
    );
}

#[test]
fn cuda_xent_f32_i64_4_rows_128_cols() {
    run_xent_parity(4, 128, CandleDType::F32, CandleDType::I64, 1e-4);
}

#[test]
fn cuda_xent_f32_u32_8_rows_512_cols() {
    run_xent_parity(8, 512, CandleDType::F32, CandleDType::U32, 1e-4);
}

#[test]
fn cuda_xent_bf16_i64_8_rows_256_cols() {
    // BF16 ~3 decimal digits. The reduction tree sum-of-exps in F32
    // means the final-cast skew is ~1e-2 in the worst case.
    run_xent_parity(8, 256, CandleDType::BF16, CandleDType::I64, 5e-2);
}

#[test]
fn cuda_xent_bf16_i64_2_rows_2048_cols() {
    // Larger row size — exercises the strided per-thread reduction
    // when vocab > MAX_THREADS.
    run_xent_parity(2, 2048, CandleDType::BF16, CandleDType::I64, 5e-2);
}

#[test]
fn cuda_xent_uniform_logits_gives_log_vocab() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let batch = 3usize;
    let vocab = 4usize;
    let data = vec![0.0f32; batch * vocab];
    let targets_i64 = vec![0i64, 1, 2];

    let x_cd = CandleTensor::from_vec(data, (batch, vocab), &dev)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();
    let t_cd = CandleTensor::from_vec(targets_i64, (batch,), &dev).unwrap();
    let t_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&t_cd).unwrap();

    let loss_kt = cuda_cross_entropy_loss(&x_kt, &t_kt).expect("xent");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let loss_f32: f32 = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&loss_kt)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

    let expected = (vocab as f32).ln();
    assert!(
        (loss_f32 - expected).abs() < 1e-5,
        "uniform logits should give log(vocab)={expected}, got {loss_f32}"
    );
}

#[test]
fn cuda_xent_perfect_prediction_is_near_zero() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    // Logits with a single +100 peak at the target → P(target) ≈ 1
    // → loss ≈ 0.
    let logits = vec![100.0f32, 0.0, 0.0, 100.0];
    let targets_i64 = vec![0i64, 1];

    let x_cd = CandleTensor::from_vec(logits, (2, 2), &dev)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();
    let t_cd = CandleTensor::from_vec(targets_i64, (2,), &dev).unwrap();
    let t_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&t_cd).unwrap();

    let loss_kt = cuda_cross_entropy_loss(&x_kt, &t_kt).expect("xent");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let loss_f32: f32 = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&loss_kt)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

    assert!(
        loss_f32 < 1e-6,
        "perfect prediction should give near-zero loss, got {loss_f32}"
    );
}

#[test]
fn cuda_xent_rejects_out_of_range_target() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let logits = vec![0.0f32, 0.0];
    let targets_i64 = vec![5i64];  // vocab=2, target=5 is out of range
    let x_cd = CandleTensor::from_vec(logits, (1, 2), &dev)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();
    let t_cd = CandleTensor::from_vec(targets_i64, (1,), &dev).unwrap();
    let t_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&t_cd).unwrap();

    let res = cuda_cross_entropy_loss(&x_kt, &t_kt);
    assert!(res.is_err(), "expected out-of-range error, got {res:?}");
    let msg = res.unwrap_err().to_string();
    assert!(
        msg.contains("out of range"),
        "expected 'out of range' message, got {msg}"
    );
}

#[test]
fn cuda_xent_dispatches_through_ops_cross_entropy() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let batch = 4usize;
    let vocab = 64usize;
    let n = batch * vocab;
    let data = pattern(n, 31);
    let targets_i64: Vec<i64> =
        (0..batch).map(|b| ((b as u64 * 7) % vocab as u64) as i64).collect();

    let x_cd = CandleTensor::from_vec(data.clone(), (batch, vocab), &dev)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();
    let t_cd = CandleTensor::from_vec(targets_i64.clone(), (batch,), &dev).unwrap();
    let t_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&t_cd).unwrap();

    // ops::cross_entropy should pick the CUDA path automatically via
    // CrossEntropyOp::cuda_fwd.
    let loss_kt = ops::cross_entropy(&x_kt, &t_kt).expect("dispatch");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got_f32: f32 = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&loss_kt)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let cpu_loss = cpu_reference_f32(&data, &targets_i64, batch, vocab);

    assert!(
        (cpu_loss - got_f32).abs() < 1e-4,
        "dispatch path: cpu={cpu_loss} cuda={got_f32}"
    );
}
