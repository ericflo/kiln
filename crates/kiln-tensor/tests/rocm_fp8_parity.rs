//! R.5b — ROCm FP8 (E4M3FN) quantize/dequantize correctness.
//!
//! `rocm_fp8_quantize_direct` / `rocm_fp8_dequantize_direct` route through the
//! shared `csrc/fp8.cu` E4M3FN bit logic (now compiled for ROCm). This validates
//! the on-device round-trip against the analytic E4M3FN properties: values
//! exactly representable in E4M3 round-trip identically; out-of-range values
//! saturate to ±448; arbitrary values round-trip within E4M3's 3-mantissa-bit
//! relative precision. The kernel is pure elementwise bit math (no wave hazard).
//!
//! Run: `cargo test -p kiln-tensor --features rocm --test rocm_fp8_parity`
#![cfg(feature = "rocm")]

use kiln_tensor::{DType, Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.5b fp8 parity test");
        true
    } else {
        false
    }
}

fn quantize_dequantize_roundtrip(vals: &[f32]) -> Vec<f32> {
    let t = Tensor::from_vec_on(Device::Rocm(0), vals.to_vec(), vec![vals.len()])
        .expect("from_vec_on f32");
    let q = kiln_tensor::rocm_fp8_quantize_direct(&t).expect("rocm_fp8_quantize_direct");
    assert_eq!(q.dtype(), DType::U8, "quantized output must be U8");
    assert_eq!(q.dims(), &[vals.len()], "quantized shape preserved");
    let deq = kiln_tensor::rocm_fp8_dequantize_direct(&q, DType::F32)
        .expect("rocm_fp8_dequantize_direct");
    assert_eq!(deq.dtype(), DType::F32);
    deq.to_vec::<f32>().expect("to_vec f32")
}

#[test]
fn fp8_roundtrip_exact_for_representable_values() {
    if no_rocm() {
        return;
    }
    // Values exactly representable in E4M3FN (1+m/8 mantissa, power-of-two
    // exponents): zero, simple fractions, integers, and the ±448 max.
    let vals: Vec<f32> = vec![
        0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, 1.5, -1.5, 1.25, 1.125, 3.0, 4.0, 8.0, 16.0,
        -16.0, 256.0, 448.0, -448.0, 0.0625,
    ];
    let got = quantize_dequantize_roundtrip(&vals);
    for (i, (&want, &g)) in vals.iter().zip(got.iter()).enumerate() {
        assert!(
            (want - g).abs() < 1e-6,
            "representable value {want} at {i} did not round-trip exactly: got {g}"
        );
    }
}

#[test]
fn fp8_saturates_out_of_range() {
    if no_rocm() {
        return;
    }
    // E4M3FN saturates at ±448 (no inf encoding). Beyond-range values clamp.
    let vals = vec![1000.0f32, -1000.0, 500.0, -500.0, 1e9, -1e9];
    let got = quantize_dequantize_roundtrip(&vals);
    let expect = [448.0f32, -448.0, 448.0, -448.0, 448.0, -448.0];
    for (i, (&w, &g)) in expect.iter().zip(got.iter()).enumerate() {
        assert!(
            (w - g).abs() < 1e-6,
            "out-of-range value {} at {i} should saturate to {w}, got {g}",
            vals[i]
        );
    }
}

#[test]
fn fp8_roundtrip_within_e4m3_precision() {
    if no_rocm() {
        return;
    }
    // Arbitrary values in the normalized-attention range (~±10): round-trip
    // relative error must be within E4M3's 3-mantissa-bit precision (1/8 = 12.5%
    // worst case between representable steps; use 13% for headroom).
    let vals: Vec<f32> = (0..512)
        .map(|i| {
            let x = (i as f32 / 511.0) * 20.0 - 10.0; // [-10, 10]
            // avoid the exact-zero bucket dominating the relative-error check
            if x.abs() < 1e-3 {
                0.137
            } else {
                x
            }
        })
        .collect();
    let got = quantize_dequantize_roundtrip(&vals);
    for (i, (&w, &g)) in vals.iter().zip(got.iter()).enumerate() {
        let rel = (w - g).abs() / w.abs().max(1e-6);
        assert!(
            rel < 0.13,
            "value {w} at {i} round-tripped to {g} (rel err {rel:.3} exceeds E4M3 precision)"
        );
    }
}

#[test]
fn fp8_quantize_accepts_bf16_and_f16_sources() {
    if no_rocm() {
        return;
    }
    // The quantize kernel reads F32/BF16/F16 sources (dtype tags 0/1/2).
    let base: Vec<f32> = vec![0.0, 1.0, -2.0, 0.5, 4.0, -8.0, 16.0, 1.5];
    for &src_dtype in &[DType::BF16, DType::F16] {
        let f32_t = Tensor::from_vec_on(Device::Rocm(0), base.clone(), vec![base.len()])
            .expect("from_vec_on");
        let src = f32_t.to_dtype(src_dtype).expect("cast to src dtype");
        let q = kiln_tensor::rocm_fp8_quantize_direct(&src)
            .unwrap_or_else(|e| panic!("rocm_fp8_quantize_direct ({src_dtype:?}): {e}"));
        let deq = kiln_tensor::rocm_fp8_dequantize_direct(&q, DType::F32)
            .expect("dequantize");
        let got = deq.to_vec::<f32>().expect("to_vec");
        for (&w, &g) in base.iter().zip(got.iter()) {
            // These are all exactly E4M3-representable; bf16/f16 of them is exact too.
            assert!(
                (w - g).abs() < 1e-2,
                "{src_dtype:?} source: {w} round-tripped to {g}"
            );
        }
    }
}
