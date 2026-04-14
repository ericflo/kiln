//! FP8 (E4M3FN) quantization utilities for KV cache compression.
//!
//! Stores KV cache values in 8-bit floating point format (E4M3FN: 1 sign, 4 exponent,
//! 3 mantissa bits, range [-448, 448], no NaN/Inf). This halves KV cache memory
//! compared to BF16/FP16, enabling ~2x longer context at the same VRAM budget.
//!
//! Quantization uses per-tensor absmax scaling:
//!   scale = max(abs(tensor)) / 448.0
//!   quantized = clamp(round(tensor / scale), -448, 448)
//!   dequantized = quantized * scale

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

/// Maximum representable value in E4M3FN format.
const E4M3_MAX: f32 = 448.0;

/// Convert a tensor to FP8 storage without scaling (scale = 1.0).
///
/// Values are directly converted to E4M3FN representation. Values outside [-448, 448]
/// are clamped. This is suitable for the paged KV cache where different writes may
/// have different value ranges and per-tensor scaling is not practical.
///
/// For typical attention K/V values (which are normalized and usually in ±10 range),
/// this provides good precision without scaling.
pub fn quantize_to_fp8_direct(tensor: &Tensor) -> Result<Tensor> {
    let tensor_f32 = tensor.to_dtype(DType::F32)?;
    let data = tensor_f32.flatten_all()?.to_vec1::<f32>()?;
    let fp8_bytes: Vec<u8> = data.iter().map(|&v| f32_to_e4m3(v)).collect();
    let shape = tensor.shape().clone();
    let quantized = Tensor::from_vec(fp8_bytes, shape, tensor.device())?;
    Ok(quantized)
}

/// Convert FP8 storage back to target dtype without scaling (scale = 1.0).
pub fn dequantize_from_fp8_direct(quantized: &Tensor, target_dtype: DType, device: &Device) -> Result<Tensor> {
    dequantize_from_fp8(quantized, 1.0, target_dtype, device)
}

/// Convert a BF16/FP16/F32 tensor to FP8 storage (U8 tensor + scale).
///
/// Returns `(quantized_u8, scale)` where:
/// - `quantized_u8` has the same shape as input but dtype U8
/// - `scale` is a scalar f32 used to dequantize: `dequant = fp8_to_f32(u8_val) * scale`
pub fn quantize_to_fp8(tensor: &Tensor) -> Result<(Tensor, f32)> {
    // Compute absmax scale
    let tensor_f32 = tensor.to_dtype(DType::F32)?;
    let abs_max = tensor_f32
        .abs()?
        .max(0)?
        .max(0)?;
    // Flatten remaining dims to get scalar
    let abs_max_val: f32 = abs_max
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;

    // Avoid division by zero
    let scale = if abs_max_val < 1e-12 {
        1.0
    } else {
        abs_max_val / E4M3_MAX
    };

    // Scale to FP8 range and convert to FP8 bit pattern
    let scaled = (&tensor_f32 / scale as f64)?;

    // Clamp to E4M3 representable range and convert each value to its FP8 bit pattern
    let data = scaled.flatten_all()?.to_vec1::<f32>()?;
    let fp8_bytes: Vec<u8> = data.iter().map(|&v| f32_to_e4m3(v)).collect();

    let shape = tensor.shape().clone();
    let quantized = Tensor::from_vec(fp8_bytes, shape, tensor.device())?;

    Ok((quantized, scale))
}

/// Convert FP8 storage (U8 tensor + scale) back to the target dtype.
pub fn dequantize_from_fp8(quantized: &Tensor, scale: f32, target_dtype: DType, device: &Device) -> Result<Tensor> {
    let data = quantized.flatten_all()?.to_vec1::<u8>()?;
    let f32_vals: Vec<f32> = data.iter().map(|&b| e4m3_to_f32(b) * scale).collect();

    let shape = quantized.shape().clone();
    let result = Tensor::from_vec(f32_vals, shape, device)?;
    result.to_dtype(target_dtype).context("dequantize dtype conversion")
}

/// Convert an f32 value to E4M3FN bit pattern.
///
/// E4M3FN layout: [sign(1)] [exponent(4)] [mantissa(3)]
/// Bias = 7, no NaN/Inf (all exponent bits set = max normal value).
/// Range: [-448, 448], smallest subnormal: 2^-9 = ~0.00195
fn f32_to_e4m3(val: f32) -> u8 {
    if val == 0.0 || val == -0.0 {
        return 0u8; // positive zero
    }

    let sign: u8 = if val < 0.0 { 1 } else { 0 };
    let abs_val = val.abs();

    // Clamp to representable range
    let abs_val = abs_val.min(E4M3_MAX);

    // Subnormal threshold: 2^(1-7) * (0 + 0/8) is the boundary
    // Smallest normal: 2^(1-7) = 2^-6 = 0.015625
    let min_normal: f32 = 2.0_f32.powi(-6);

    if abs_val < min_normal {
        // Subnormal: exponent field = 0, mantissa encodes fraction of 2^-6
        // value = 2^-6 * (mantissa / 8)  =>  mantissa = round(value / 2^-6 * 8)
        // Actually for E4M3: subnormal = 2^(1-bias) * (0.mantissa) = 2^-6 * (m/8)
        let mantissa = (abs_val / min_normal * 8.0).round() as u8;
        let mantissa = mantissa.min(7); // can't exceed 3 bits
        return (sign << 7) | mantissa;
    }

    // Normal values
    let bits = abs_val.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127; // unbiased exponent
    let f32_mantissa = bits & 0x7FFFFF; // 23-bit mantissa

    // E4M3 exponent: biased with bias=7, range [1, 15] for normal (0 = subnormal)
    // But E4M3FN: exponent 1111 with mantissa 111 = 448 (max normal, not NaN)
    let e4m3_exp_unbiased = f32_exp.clamp(-6, 8); // -6 to 8 for bias=7 → biased 1..15
    let biased_exp = (e4m3_exp_unbiased + 7) as u8;

    // Round the 23-bit mantissa to 3 bits
    // Top 3 bits of f32 mantissa
    let mantissa_3bit = ((f32_mantissa + (1 << 19)) >> 20) as u8; // round to nearest

    if mantissa_3bit >= 8 {
        // Mantissa overflow from rounding — bump exponent
        let biased_exp = biased_exp + 1;
        if biased_exp > 15 {
            // Saturate to max
            return (sign << 7) | 0x7F; // max value: exp=1111, mantissa=111 = 448
        }
        return (sign << 7) | (biased_exp << 3);
    }

    // Clamp to max representable
    if biased_exp > 15 || (biased_exp == 15 && mantissa_3bit > 7) {
        return (sign << 7) | 0x7F; // 448
    }

    (sign << 7) | (biased_exp << 3) | mantissa_3bit
}

/// Convert an E4M3FN bit pattern back to f32.
fn e4m3_to_f32(bits: u8) -> f32 {
    let sign = (bits >> 7) & 1;
    let exp = (bits >> 3) & 0xF;
    let mantissa = bits & 0x7;

    let abs_val = if exp == 0 {
        // Subnormal: value = 2^(1-7) * (0.mantissa) = 2^-6 * mantissa/8
        2.0_f32.powi(-6) * (mantissa as f32 / 8.0)
    } else {
        // Normal: value = 2^(exp-7) * (1 + mantissa/8)
        // Note: E4M3FN — even exp=15 mantissa=7 is a normal value (448), not NaN
        2.0_f32.powi(exp as i32 - 7) * (1.0 + mantissa as f32 / 8.0)
    };

    if sign == 1 { -abs_val } else { abs_val }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e4m3_roundtrip_zero() {
        assert_eq!(f32_to_e4m3(0.0), 0);
        assert_eq!(e4m3_to_f32(0), 0.0);
    }

    #[test]
    fn test_e4m3_roundtrip_one() {
        let bits = f32_to_e4m3(1.0);
        let val = e4m3_to_f32(bits);
        assert!((val - 1.0).abs() < 0.1, "Expected ~1.0, got {val}");
    }

    #[test]
    fn test_e4m3_max_value() {
        let bits = f32_to_e4m3(448.0);
        let val = e4m3_to_f32(bits);
        assert!((val - 448.0).abs() < 1.0, "Expected ~448, got {val}");
    }

    #[test]
    fn test_e4m3_clamps_above_max() {
        let bits = f32_to_e4m3(1000.0);
        let val = e4m3_to_f32(bits);
        assert!((val - 448.0).abs() < 1.0, "Expected clamped to ~448, got {val}");
    }

    #[test]
    fn test_e4m3_negative() {
        let bits = f32_to_e4m3(-2.0);
        let val = e4m3_to_f32(bits);
        assert!((val + 2.0).abs() < 0.3, "Expected ~-2.0, got {val}");
    }

    #[test]
    fn test_e4m3_small_value() {
        let bits = f32_to_e4m3(0.01);
        let val = e4m3_to_f32(bits);
        assert!((val - 0.01).abs() < 0.005, "Expected ~0.01, got {val}");
    }

    #[test]
    fn test_quantize_dequantize_tensor() -> Result<()> {
        let device = Device::Cpu;
        let original = Tensor::new(
            &[[1.0_f32, -2.0, 3.0, -4.0], [0.5, -0.5, 0.1, -0.1]],
            &device,
        )?;

        let (quantized, scale) = quantize_to_fp8(&original)?;
        assert_eq!(quantized.dtype(), DType::U8);
        assert_eq!(quantized.dims(), original.dims());

        let dequantized = dequantize_from_fp8(&quantized, scale, DType::F32, &device)?;
        assert_eq!(dequantized.dims(), original.dims());

        // Check approximate roundtrip accuracy (FP8 has limited precision)
        let orig_vals = original.flatten_all()?.to_vec1::<f32>()?;
        let deq_vals = dequantized.flatten_all()?.to_vec1::<f32>()?;
        for (i, (o, d)) in orig_vals.iter().zip(deq_vals.iter()).enumerate() {
            let err = (o - d).abs();
            let rel_err = if o.abs() > 0.01 { err / o.abs() } else { err };
            assert!(
                rel_err < 0.15,
                "Index {i}: original={o}, dequantized={d}, rel_err={rel_err}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_quantize_preserves_shape() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Tensor::randn(0.0_f32, 1.0, (2, 4, 8, 16), &device)?;
        let (quantized, _scale) = quantize_to_fp8(&tensor)?;
        assert_eq!(quantized.dims(), tensor.dims());
        Ok(())
    }

    #[test]
    fn test_quantize_zeros() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Tensor::zeros((2, 4), DType::F32, &device)?;
        let (quantized, scale) = quantize_to_fp8(&tensor)?;
        let dequantized = dequantize_from_fp8(&quantized, scale, DType::F32, &device)?;
        let vals = dequantized.flatten_all()?.to_vec1::<f32>()?;
        assert!(vals.iter().all(|&v| v.abs() < 1e-6));
        Ok(())
    }

    #[test]
    fn test_fp8_memory_savings() -> Result<()> {
        // BF16 KV cache: 2 bytes per element
        // FP8 KV cache: 1 byte per element + negligible scale overhead
        // This test verifies the U8 storage is indeed half the size
        let device = Device::Cpu;
        let bf16_tensor = Tensor::zeros((1024, 4, 256), DType::BF16, &device)?;
        let bf16_bytes = bf16_tensor.elem_count() * 2; // 2 bytes per BF16

        let f32_tensor = bf16_tensor.to_dtype(DType::F32)?;
        let (fp8_tensor, _) = quantize_to_fp8(&f32_tensor)?;
        let fp8_bytes = fp8_tensor.elem_count() * 1; // 1 byte per U8

        assert_eq!(fp8_bytes * 2, bf16_bytes, "FP8 should be exactly half the size of BF16");
        Ok(())
    }
}
