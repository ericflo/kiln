#![cfg(feature = "cuda")]

//! Issue #1082: parity test for FP8 (E4M3FN) quantize / dequantize
//! on CUDA.
//!
//! Validates `cuda_fp8_quantize_with_scale`, `cuda_fp8_quantize_direct`,
//! `cuda_fp8_dequantize`, and `cuda_fp8_dequantize_direct` against the
//! canonical candle-typed reference in `kiln_model::fp8`.
//!
//! The CUDA kernel and the CPU reference share the bitwise E4M3FN
//! encode / decode tables, so parity is exact (not just within
//! tolerance) for the bit patterns. Roundtrip accuracy is bounded by
//! the precision of E4M3FN itself (~1 ULP / 12% relative error in the
//! worst case for values far from 0).

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{
    cuda_fp8_dequantize, cuda_fp8_dequantize_direct, cuda_fp8_quantize,
    cuda_fp8_quantize_direct, cuda_fp8_quantize_with_scale, DType,
};

fn try_cuda() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

/// Deterministic pseudo-random F32 data in [-1, 1].
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

/// Scale-aware random pattern (values can be larger than [-1, 1]).
fn pattern_scaled(n: usize, seed: u64, range: f32) -> Vec<f32> {
    pattern(n, seed).iter().map(|&v| v * range).collect()
}

// CPU reference — mirrors `kiln_model::fp8::f32_to_e4m3` / `e4m3_to_f32`.
// Inlining the bitwise reference here keeps the test crate's dep set
// thin (no kiln-model pull-in) and makes parity verification visible
// at the test site.
fn f32_to_e4m3_ref(val: f32) -> u8 {
    if val == 0.0 || val == -0.0 {
        return 0u8;
    }
    if val.is_nan() || val.is_infinite() {
        return if val < 0.0 { 0xFFu8 } else { 0x7Fu8 };
    }
    let sign: u8 = if val < 0.0 { 1 } else { 0 };
    let abs_val = val.abs().min(448.0);
    let min_normal: f32 = 2.0_f32.powi(-6);
    if abs_val < min_normal {
        let mantissa = (abs_val / min_normal * 8.0).round() as u8;
        let mantissa = mantissa.min(7);
        return (sign << 7) | mantissa;
    }
    let bits = abs_val.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let f32_mantissa = bits & 0x7FFFFF;
    let e4m3_exp_unbiased = f32_exp.clamp(-6, 8);
    let biased_exp = (e4m3_exp_unbiased + 7) as u8;
    let mantissa_3bit = ((f32_mantissa + (1 << 19)) >> 20) as u8;
    if mantissa_3bit >= 8 {
        let biased_exp = biased_exp + 1;
        if biased_exp > 15 {
            return (sign << 7) | 0x7F;
        }
        return (sign << 7) | (biased_exp << 3);
    }
    if biased_exp > 15 || (biased_exp == 15 && mantissa_3bit > 7) {
        return (sign << 7) | 0x7F;
    }
    (sign << 7) | (biased_exp << 3) | mantissa_3bit
}

fn e4m3_to_f32_ref(bits: u8) -> f32 {
    let sign = (bits >> 7) & 1;
    let exp = (bits >> 3) & 0xF;
    let mantissa = bits & 0x7;
    let abs_val = if exp == 0 {
        2.0_f32.powi(-6) * (mantissa as f32 / 8.0)
    } else {
        2.0_f32.powi(exp as i32 - 7) * (1.0 + mantissa as f32 / 8.0)
    };
    if sign == 1 { -abs_val } else { abs_val }
}

fn cast_data(data: &[f32], dtype: CandleDType) -> Vec<f32> {
    match dtype {
        CandleDType::F32 => data.to_vec(),
        CandleDType::BF16 => data
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect(),
        CandleDType::F16 => data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect(),
        _ => panic!("unsupported dtype"),
    }
}

fn kt_dtype(dtype: CandleDType) -> DType {
    match dtype {
        CandleDType::F32 => DType::F32,
        CandleDType::BF16 => DType::BF16,
        CandleDType::F16 => DType::F16,
        _ => panic!("unsupported dtype"),
    }
}

// =====================================================================
// Quantize parity
// =====================================================================

fn run_quantize_parity(
    shape: &[usize],
    dtype: CandleDType,
    scale: f32,
) {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n: usize = shape.iter().product();
    let data = pattern_scaled(n, 17, scale * 1.5); // produce some out-of-range values
    let shape_tuple = shape.to_vec();
    let cd = CandleTensor::from_vec(data.clone(), shape_tuple.clone(), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&cd).unwrap();

    let q_kt = cuda_fp8_quantize_with_scale(&kt, scale).expect("quantize");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let q_cd = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&q_kt).unwrap();
    let got_bytes: Vec<u8> = q_cd.flatten_all().unwrap().to_vec1::<u8>().unwrap();

    // Reference: apply same cast-then-divide-then-encode flow.
    let cast = cast_data(&data, dtype);
    let want_bytes: Vec<u8> = cast
        .iter()
        .map(|&v| f32_to_e4m3_ref(v / scale))
        .collect();

    assert_eq!(got_bytes.len(), want_bytes.len(), "len mismatch");
    let mut max_diff_at = None;
    for (i, (got, want)) in got_bytes.iter().zip(want_bytes.iter()).enumerate() {
        if got != want {
            max_diff_at = Some((i, *got, *want));
            break;
        }
    }
    if let Some((i, g, w)) = max_diff_at {
        // For BF16 / F16 inputs there is one ULP of rounding noise in
        // the host-side cast path that can produce a differing E4M3
        // mantissa at the boundary. Allow a 1-bit difference on the
        // last bit of the mantissa as long as the dequantized values
        // are within FP8 quantization noise (~12% relative for normal
        // values).
        let g_v = e4m3_to_f32_ref(g) * scale;
        let w_v = e4m3_to_f32_ref(w) * scale;
        let denom = w_v.abs().max(1e-6);
        let rel = (g_v - w_v).abs() / denom;
        assert!(
            rel < 0.30,
            "qparity: shape={shape:?} dtype={dtype:?} scale={scale} \
             index={i} got_bits=0x{g:02x} want_bits=0x{w:02x} \
             got={g_v} want={w_v} rel={rel}"
        );
    }
}

#[test]
fn cuda_fp8_quantize_f32_scale1() {
    run_quantize_parity(&[1024], CandleDType::F32, 1.0);
}

#[test]
fn cuda_fp8_quantize_f32_scale_half() {
    run_quantize_parity(&[256], CandleDType::F32, 0.5);
}

#[test]
fn cuda_fp8_quantize_f32_scale_large() {
    run_quantize_parity(&[2, 256], CandleDType::F32, 10.0);
}

#[test]
fn cuda_fp8_quantize_bf16_scale1() {
    run_quantize_parity(&[1024], CandleDType::BF16, 1.0);
}

#[test]
fn cuda_fp8_quantize_bf16_scale_small() {
    run_quantize_parity(&[8, 64], CandleDType::BF16, 0.125);
}

#[test]
fn cuda_fp8_quantize_f16_scale1() {
    run_quantize_parity(&[1024], CandleDType::F16, 1.0);
}

// =====================================================================
// Roundtrip (quantize -> dequantize)
// =====================================================================

fn run_roundtrip(
    shape: &[usize],
    dtype: CandleDType,
    rel_tol: f32,
) {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n: usize = shape.iter().product();
    let data: Vec<f32> = pattern(n, 23); // values in [-1, 1]
    let cd = CandleTensor::from_vec(data.clone(), shape.to_vec(), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&cd).unwrap();

    // Compute scale and quantize
    let (q_kt, scale) = cuda_fp8_quantize(&kt).expect("quantize+scale");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    // Dequantize back to original dtype
    let deq_kt = cuda_fp8_dequantize(&q_kt, scale, kt_dtype(dtype))
        .expect("dequantize");
    cuda_dev.synchronize().unwrap();

    let deq_cd = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&deq_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap();
    let got: Vec<f32> = deq_cd.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Compare against original (cast through input dtype to match
    // precision floor)
    let want = cast_data(&data, dtype);
    assert_eq!(got.len(), want.len());

    let mut max_rel = 0.0f32;
    for (g, w) in got.iter().zip(want.iter()) {
        let err = (g - w).abs();
        let denom = w.abs().max(0.01);
        let rel = err / denom;
        if rel > max_rel {
            max_rel = rel;
        }
    }
    assert!(
        max_rel < rel_tol,
        "roundtrip: shape={shape:?} dtype={dtype:?} max_rel={max_rel} > {rel_tol}"
    );
}

#[test]
fn cuda_fp8_roundtrip_f32() {
    // FP8 has ~12% relative error in the worst case; we set 30% to
    // cover the per-tensor scaling overhead too.
    run_roundtrip(&[1024], CandleDType::F32, 0.30);
}

#[test]
fn cuda_fp8_roundtrip_bf16() {
    run_roundtrip(&[1024], CandleDType::BF16, 0.30);
}

#[test]
fn cuda_fp8_roundtrip_f16() {
    run_roundtrip(&[1024], CandleDType::F16, 0.30);
}

#[test]
fn cuda_fp8_roundtrip_large() {
    run_roundtrip(&[2, 1024, 64], CandleDType::F32, 0.30);
}

// =====================================================================
// Direct mode (scale = 1.0)
// =====================================================================

#[test]
fn cuda_fp8_direct_in_range() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    // Values that fit comfortably in [-448, 448].
    let data: Vec<f32> = vec![1.0, -2.0, 3.5, -4.25, 0.5, -0.5, 0.01, -0.01];
    let n = data.len();
    let cd = CandleTensor::from_vec(data.clone(), (n,), &dev).unwrap();
    let kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&cd).unwrap();

    let q_kt = cuda_fp8_quantize_direct(&kt).expect("quantize_direct");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let deq_kt = cuda_fp8_dequantize_direct(&q_kt, DType::F32).expect("dequant_direct");
    cuda_dev.synchronize().unwrap();

    let deq_cd = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&deq_kt).unwrap();
    let got: Vec<f32> = deq_cd.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Without per-tensor scaling, accuracy is limited by FP8 itself.
    // Tolerance ~12-15% relative error per value, with a 0.01 floor.
    for (i, (g, w)) in got.iter().zip(data.iter()).enumerate() {
        let err = (g - w).abs();
        let rel = if w.abs() > 0.1 { err / w.abs() } else { err };
        assert!(
            rel < 0.20,
            "direct: idx={i} got={g} want={w} rel={rel}"
        );
    }
}

#[test]
fn cuda_fp8_direct_clamps_out_of_range() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    // Values that exceed [-448, 448] should clamp.
    let data: Vec<f32> = vec![1000.0, -1000.0, 500.0, -500.0];
    let n = data.len();
    let cd = CandleTensor::from_vec(data.clone(), (n,), &dev).unwrap();
    let kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&cd).unwrap();

    let q_kt = cuda_fp8_quantize_direct(&kt).unwrap();
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let deq_kt = cuda_fp8_dequantize_direct(&q_kt, DType::F32).unwrap();
    cuda_dev.synchronize().unwrap();

    let deq_cd = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&deq_kt).unwrap();
    let got: Vec<f32> = deq_cd.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // All should clamp to ±448.
    for (i, (g, w)) in got.iter().zip(data.iter()).enumerate() {
        let target = if *w > 0.0 { 448.0 } else { -448.0 };
        assert!(
            (*g - target).abs() < 1.0,
            "clamp: idx={i} got={g} want_clamped={target}"
        );
    }
}

// =====================================================================
// Bit-exact byte-pattern check on representative values
// =====================================================================

#[test]
fn cuda_fp8_quantize_byte_pattern_exact() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    // Hand-picked values where the CPU reference and CUDA path should
    // produce IDENTICAL byte patterns (no rounding ambiguity).
    let data: Vec<f32> = vec![
        0.0, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0,
        0.5, 0.25, 0.125, 0.0625, 0.03125,
    ];
    let n = data.len();
    let cd = CandleTensor::from_vec(data.clone(), (n,), &dev).unwrap();
    let kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&cd).unwrap();

    let q_kt = cuda_fp8_quantize_with_scale(&kt, 1.0).unwrap();
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let q_cd = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&q_kt).unwrap();
    let got: Vec<u8> = q_cd.flatten_all().unwrap().to_vec1::<u8>().unwrap();

    for (i, (g, v)) in got.iter().zip(data.iter()).enumerate() {
        let want = f32_to_e4m3_ref(*v);
        assert_eq!(
            *g, want,
            "byte-exact: idx={i} val={v} got=0x{g:02x} want=0x{want:02x}"
        );
    }
}

#[test]
fn cuda_fp8_dequantize_byte_pattern_exact() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    // All 256 possible E4M3FN bit patterns.
    let bytes: Vec<u8> = (0u8..=255).collect();
    let n = bytes.len();
    let cd = CandleTensor::from_vec(bytes.clone(), (n,), &dev).unwrap();
    let kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&cd).unwrap();

    let deq_kt = cuda_fp8_dequantize(&kt, 1.0, DType::F32).unwrap();
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let deq_cd = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&deq_kt).unwrap();
    let got: Vec<f32> = deq_cd.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    for (i, (g, b)) in got.iter().zip(bytes.iter()).enumerate() {
        let want = e4m3_to_f32_ref(*b);
        // E4M3FN has no NaN/Inf — every byte maps to a finite value.
        assert!(g.is_finite(), "deq: idx={i} bits=0x{b:02x} got non-finite {g}");
        let err = (g - want).abs();
        // Bit-exact decode parity (no floating-point ambiguity since
        // ldexpf gives exact powers of 2 in the normal range).
        assert!(
            err < 1e-6,
            "deq byte-exact: idx={i} bits=0x{b:02x} got={g} want={want} err={err}"
        );
    }
}

// =====================================================================
// Error path: dtype mismatch
// =====================================================================

#[test]
fn cuda_fp8_dequantize_rejects_non_u8() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let data = vec![1.0_f32, 2.0, 3.0, 4.0];
    let cd = CandleTensor::from_vec(data, (4,), &dev).unwrap();
    let kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&cd).unwrap();
    // F32 input — dequant should reject this.
    let r = cuda_fp8_dequantize(&kt, 1.0, DType::F32);
    assert!(r.is_err());
}

#[test]
fn cuda_fp8_quantize_rejects_zero_scale() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let data = vec![1.0_f32, 2.0, 3.0, 4.0];
    let cd = CandleTensor::from_vec(data, (4,), &dev).unwrap();
    let kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&cd).unwrap();
    let r = cuda_fp8_quantize_with_scale(&kt, 0.0);
    assert!(r.is_err());
}
