//! Parity test: kt CUDA `clamp(x, lo, hi)` and `pow(x, p)` vs kt CPU
//! reference.
//!
//! The CUDA kernel (`csrc/clamp_pow.cu`) does per-element math
//! promoted to F32, then narrowed back to storage dtype. The CPU
//! reference does the same scalar math in `ops::clamp_pow`. Parity
//! tolerance allows small BF16/F16 rounding error since
//! `__expf` / `powf` aren't bit-identical between host and device
//! libm. (#1082)

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::ops;

fn try_cuda() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

fn pattern_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for _ in 0..n {
        s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E3779B97F4A7C15);
        let f = ((s as u32 % 4096) as f32 - 2048.0) / 256.0;
        out.push(f);
    }
    out
}

fn read_f32(t: &kiln_tensor::Tensor) -> Vec<f32> {
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap();
    cpu.as_bytes()
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn approx(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "lengths differ");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let denom = x.abs().max(y.abs()).max(1.0);
        let rel = diff / denom;
        assert!(
            diff <= tol || rel <= tol,
            "idx {i}: got {y}, want {x} (abs_diff={diff}, rel={rel}, tol={tol})"
        );
    }
}

#[test]
fn cuda_clamp_f32_clips_to_range() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let data: Vec<f32> = vec![-5.0, -1.0, 0.0, 1.0, 5.0];
    let x_cd = CandleTensor::from_vec(data.clone(), (5,), &dev).unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let out_kt = ops::clamp(&x_kt, -1.0, 1.0).expect("clamp");
    assert_eq!(out_kt.shape(), &[5]);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(got, vec![-1.0, -1.0, 0.0, 1.0, 1.0]);
}

#[test]
fn cuda_clamp_f32_matches_cpu() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let data = pattern_f32(64, 23);
    let x_cd = CandleTensor::from_vec(data.clone(), (8, 8), &dev).unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let lo = -2.0f32;
    let hi = 3.0f32;
    let out_kt = ops::clamp(&x_kt, lo, hi).expect("clamp");

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .reshape((64,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // CPU reference.
    let x_cpu = kiln_tensor::Tensor::from_slice(&data, vec![8, 8]).unwrap();
    let ref_kt = ops::clamp(&x_cpu, lo, hi).unwrap();
    let want = read_f32(&ref_kt);

    approx(&want, &got, 0.0);
}

#[test]
fn cuda_clamp_bf16_matches_cpu() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let data = pattern_f32(32, 29);
    let x_cd = CandleTensor::from_vec(data.clone(), (4, 8), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let lo = -1.5f32;
    let hi = 1.5f32;
    let out_kt = ops::clamp(&x_kt, lo, hi).expect("clamp bf16");
    assert_eq!(out_kt.dtype(), kiln_tensor::DType::BF16);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((32,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // CPU reference: quantize through BF16 the same way.
    let bf16_data: Vec<f32> = data
        .iter()
        .map(|&v| half::bf16::from_f32(v).to_f32())
        .collect();
    let x_cpu_bf16: Vec<half::bf16> = data.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let x_cpu_kt = kiln_tensor::Tensor::from_slice(&x_cpu_bf16, vec![4, 8]).unwrap();
    let ref_kt = ops::clamp(&x_cpu_kt, lo, hi).unwrap();
    let ref_cpu = ref_kt
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap();
    let want: Vec<f32> = ref_cpu
        .as_bytes()
        .chunks(2)
        .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
        .collect();
    let _ = bf16_data;

    // BF16 has ~7 bits of mantissa, so allow small abs tolerance.
    approx(&want, &got, 1e-2);
}

#[test]
fn cuda_pow_f32_square() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, -4.0];
    let x_cd = CandleTensor::from_vec(data, (4,), &dev).unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let out_kt = ops::pow(&x_kt, 2.0).expect("pow square");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    approx(&[1.0, 4.0, 9.0, 16.0], &got, 1e-5);
}

#[test]
fn cuda_pow_f32_matches_cpu() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    // Use only positive values so non-integer powf is well-defined.
    let data: Vec<f32> = (0..32).map(|i| 0.5 + (i as f32) * 0.25).collect();
    let x_cd = CandleTensor::from_vec(data.clone(), (4, 8), &dev).unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let p = 1.5f32;
    let out_kt = ops::pow(&x_kt, p).expect("pow");

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .reshape((32,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // CPU reference.
    let x_cpu = kiln_tensor::Tensor::from_slice(&data, vec![4, 8]).unwrap();
    let ref_kt = ops::pow(&x_cpu, p).unwrap();
    let want = read_f32(&ref_kt);

    // powf isn't bit-identical between libm and CUDA's __powf, so
    // allow small relative tolerance.
    approx(&want, &got, 1e-5);
}

#[test]
fn cuda_pow_bf16_matches_cpu() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let data: Vec<f32> = (0..16).map(|i| 0.5 + (i as f32) * 0.25).collect();
    let x_cd = CandleTensor::from_vec(data.clone(), (4, 4), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let p = 2.0f32;
    let out_kt = ops::pow(&x_kt, p).expect("pow bf16");
    assert_eq!(out_kt.dtype(), kiln_tensor::DType::BF16);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((16,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // CPU reference: quantize through BF16.
    let x_cpu_bf16: Vec<half::bf16> = data.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let x_cpu_kt = kiln_tensor::Tensor::from_slice(&x_cpu_bf16, vec![4, 4]).unwrap();
    let ref_kt = ops::pow(&x_cpu_kt, p).unwrap();
    let ref_cpu = ref_kt
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap();
    let want: Vec<f32> = ref_cpu
        .as_bytes()
        .chunks(2)
        .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
        .collect();

    // BF16 has ~7 bits of mantissa; pow amplifies error. Loose tol.
    approx(&want, &got, 5e-2);
}

#[test]
fn cuda_clamp_lo_gt_hi_errors() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let x_cd = CandleTensor::from_vec(vec![1.0f32], (1,), &dev).unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();
    let e = ops::clamp(&x_kt, 1.0, -1.0).unwrap_err();
    assert!(e.to_string().contains("lo"));
}
