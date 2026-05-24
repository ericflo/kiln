//! Parity test: kt CUDA masked-fill (`MaskedFillOp::cuda_fwd` /
//! `cuda_masked_fill`) vs kt CPU reference (`ops::masked_fill`).
//!
//! Phase 4 substrate validation. Confirms the kernel in
//! `csrc/masked_fill.cu` produces outputs matching the canonical CPU
//! reference (which writes `fill_value` where `mask != 0` and copies
//! `x` otherwise).



use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{cuda_masked_fill, ops, Tensor};

fn try_cuda() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

fn pattern_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for _ in 0..n {
        s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E3779B97F4A7C15);
        let f = ((s as u32 % 2048) as f32 - 1024.0) / 1024.0;
        out.push(f);
    }
    out
}

fn pattern_mask(n: usize, seed: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for _ in 0..n {
        s = s.wrapping_add(0x12345678).wrapping_mul(0x9E3779B97F4A7C15);
        // ~50% density.
        out.push(if (s & 1) == 0 { 0u8 } else { 1u8 });
    }
    out
}

fn cpu_reference_f32(x: &[f32], mask: &[u8], shape: Vec<usize>, fill: f32) -> Vec<f32> {
    let x_t = Tensor::from_slice(x, shape.clone()).unwrap();
    let m_t = Tensor::from_slice(mask, shape).unwrap();
    let y = ops::masked_fill(&x_t, &m_t, fill).unwrap();
    let cpu_storage = y
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap();
    let bytes = cpu_storage.as_bytes();
    let mut out = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        out.push(f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()));
    }
    out
}

fn run_masked_fill_parity(
    shape: Vec<usize>,
    dtype: CandleDType,
    fill: f32,
    seed: u64,
    tolerance: f32,
) {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n: usize = shape.iter().product();
    let data = pattern_f32(n, seed);
    let mask_bytes = pattern_mask(n, seed.wrapping_add(7));

    let x_cd = CandleTensor::from_vec(data.clone(), shape.clone(), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let m_cd = CandleTensor::from_vec(mask_bytes.clone(), shape.clone(), &dev).unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();
    let m_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&m_cd).unwrap();

    let out_kt = cuda_masked_fill(&x_kt, &m_kt, fill).expect("masked_fill");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    // Reference: kt CPU masked_fill over the same F32 values + same mask
    // bytes (CPU op promotes to F32, narrows back to dtype on store —
    // matches what the CUDA kernel does in BF16/F16).
    let ref_vec = cpu_reference_f32(&data, &mask_bytes, shape.clone(), fill);

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
        if a.is_infinite() && b.is_infinite() && a.signum() == b.signum() {
            // Both ±inf with matching sign — perfect match.
            continue;
        }
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    assert!(
        max_abs < tolerance,
        "shape={shape:?} dtype={dtype:?} fill={fill} max_abs={max_abs} > {tolerance}"
    );

    // Spot-check: at masked positions, output ≈ fill (within dtype
    // precision). At unmasked positions, output ≈ input.
    for i in 0..n {
        if mask_bytes[i] != 0 {
            let gv = got_vec[i];
            if fill.is_infinite() {
                assert!(
                    gv.is_infinite() && gv.signum() == fill.signum(),
                    "i={i} mask=1 got={gv} expected ±inf with sign {}",
                    fill.signum()
                );
            } else {
                assert!(
                    (gv - fill).abs() < tolerance,
                    "i={i} mask=1 got={gv} fill={fill}"
                );
            }
        } else {
            assert!(
                (got_vec[i] - data[i]).abs() < tolerance,
                "i={i} mask=0 got={} input={}",
                got_vec[i],
                data[i]
            );
        }
    }
}

#[test]
fn cuda_masked_fill_f32_basic() {
    run_masked_fill_parity(vec![4, 64], CandleDType::F32, -99.0, 31, 1e-6);
}

#[test]
fn cuda_masked_fill_bf16_neg_inf() {
    // The canonical "pre-softmax fill -inf" pattern. BF16 represents
    // -inf exactly, so the parity should be tight.
    run_masked_fill_parity(
        vec![8, 128],
        CandleDType::BF16,
        f32::NEG_INFINITY,
        41,
        // Tolerance is 0 when the comparison hits +inf vs +inf in the
        // diff; non-fill positions tolerate BF16 round-trip noise.
        1e-2,
    );
}

#[test]
fn cuda_masked_fill_f16_large() {
    // Larger contiguous tensor exercising more thread blocks.
    run_masked_fill_parity(vec![2, 4, 256], CandleDType::F16, -1024.0, 53, 1e-1);
}

#[test]
fn cuda_masked_fill_dispatches_through_ops_masked_fill() {
    // End-to-end dispatch check: ops::masked_fill on CUDA tensors
    // should pick the CUDA path via MaskedFillOp::cuda_fwd and produce
    // identical results to the CPU reference.
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n_rows = 4usize;
    let n_cols = 64usize;
    let n = n_rows * n_cols;
    let data = pattern_f32(n, 61);
    let mask = pattern_mask(n, 67);
    let fill = -7.5f32;

    let x_cd = CandleTensor::from_vec(data.clone(), (n_rows, n_cols), &dev)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap();
    let m_cd = CandleTensor::from_vec(mask.clone(), (n_rows, n_cols), &dev).unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();
    let m_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&m_cd).unwrap();

    let out_kt = ops::masked_fill(&x_kt, &m_kt, fill).expect("dispatch");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got_vec: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((n,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    for i in 0..n {
        if mask[i] != 0 {
            assert!(
                (got_vec[i] - fill).abs() < 1e-6,
                "i={i} mask=1 got={} fill={fill}",
                got_vec[i]
            );
        } else {
            assert!(
                (got_vec[i] - data[i]).abs() < 1e-6,
                "i={i} mask=0 got={} input={}",
                got_vec[i],
                data[i]
            );
        }
    }
}
