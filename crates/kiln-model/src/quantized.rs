//! GPTQ INT4 weight quantization support.
//!
//! Loads pre-quantized GPTQ safetensors (INT4 packed weights + scales + zeros)
//! and dequantizes them to BF16 for use in the standard forward pass.
//!
//! GPTQ packing format (INT4, pack_factor=8):
//! - `qweight`: `[in_features/8, out_features]`, dtype I32
//!   Each i32 contains 8 consecutive INT4 values packed in 4-bit groups.
//! - `scales`: `[num_groups, out_features]`, dtype F16 or BF16
//!   Per-group per-output-feature scale factors.
//! - `qzeros`: `[num_groups, out_features/8]`, dtype I32
//!   Packed INT4 zero points, same packing as qweight.
//!
//! The dequantization formula: `weight_f = (weight_int4 - zero_int4) * scale`
//!
//! Current implementation: dequantize to BF16 on CPU during model loading.
//! The resulting weights are stored on GPU in the standard BF16 format.
//! A future follow-up will add GPU-native INT4 storage with a custom CUDA
//! dequantization kernel for ~50% GPU memory savings.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::path::Path;

use crate::weights::{TensorDType, WeightTensor};

/// GPTQ quantization configuration from `quantize_config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct GptqConfig {
    pub bits: usize,
    pub group_size: usize,
    #[serde(default)]
    pub sym: bool,
    #[serde(default)]
    pub desc_act: bool,
}

impl GptqConfig {
    /// Number of INT4 values packed in one 32-bit word.
    pub fn pack_factor(&self) -> usize {
        32 / self.bits
    }
}

/// Load GPTQ config from `quantize_config.json` in the model directory.
///
/// Returns `None` if the file doesn't exist (model is not GPTQ-quantized).
pub fn load_gptq_config(model_dir: &Path) -> Result<Option<GptqConfig>> {
    let config_path = model_dir.join("quantize_config.json");
    if !config_path.exists() {
        return Ok(None);
    }
    let contents = std::fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read {}", config_path.display()))?;
    let config: GptqConfig = serde_json::from_str(&contents)
        .with_context(|| format!("Failed to parse {}", config_path.display()))?;
    if config.bits != 4 {
        bail!(
            "Only 4-bit GPTQ quantization is supported, got {} bits",
            config.bits
        );
    }
    Ok(Some(config))
}

/// Dequantize a GPTQ-packed linear layer weight to a dense BF16 `WeightTensor`.
///
/// The GPTQ format stores weights column-major: `qweight[in/8, out]`.
/// The output is in kiln's standard layout: `[out_features, in_features]`.
pub fn dequantize_gptq_weight(
    qweight_data: &[u8],
    qweight_shape: &[usize],
    scales_data: &[u8],
    scales_shape: &[usize],
    scales_dtype: TensorDType,
    qzeros_data: &[u8],
    qzeros_shape: &[usize],
    group_size: usize,
) -> Result<WeightTensor> {
    let pack_factor = 8usize; // 32 / 4 bits

    if qweight_shape.len() != 2 {
        bail!(
            "qweight must be 2-D, got shape {:?}",
            qweight_shape
        );
    }

    let in_features_packed = qweight_shape[0];
    let out_features = qweight_shape[1];
    let in_features = in_features_packed * pack_factor;
    let num_groups = scales_shape[0];

    // Validate shapes
    if scales_shape.len() != 2 || scales_shape[1] != out_features {
        bail!(
            "scales shape mismatch: expected [*, {out_features}], got {:?}",
            scales_shape
        );
    }
    if qzeros_shape.len() != 2 || qzeros_shape[0] != num_groups {
        bail!(
            "qzeros shape mismatch: expected [{num_groups}, *], got {:?}",
            qzeros_shape
        );
    }

    // Interpret raw bytes as u32 slices (GPTQ stores as I32 but bit patterns are the same).
    let qweight = bytes_as_u32(qweight_data)
        .context("qweight byte alignment")?;
    let qzeros = bytes_as_u32(qzeros_data)
        .context("qzeros byte alignment")?;

    if qweight.len() < in_features_packed * out_features {
        bail!(
            "qweight data too short: need {} u32s, got {}",
            in_features_packed * out_features,
            qweight.len()
        );
    }

    // Convert scales to f32 for computation.
    let scales_f32 = scales_to_f32(scales_data, scales_dtype, num_groups * out_features)?;

    let out_features_packed = qzeros_shape[1];

    // Dequantize to BF16: output shape [out_features, in_features]
    let total_elems = out_features * in_features;
    let mut output_bf16 = vec![0u16; total_elems];

    for out_idx in 0..out_features {
        for in_idx in 0..in_features {
            let group = in_idx / group_size;

            // Extract INT4 weight value from packed qweight
            let pack_row = in_idx / pack_factor;
            let bit_offset = (in_idx % pack_factor) * 4;
            let packed_val = qweight[pack_row * out_features + out_idx];
            let weight_int = ((packed_val >> bit_offset) & 0xF) as i32;

            // Extract INT4 zero point from packed qzeros
            let zero_pack_col = out_idx / pack_factor;
            let zero_bit_offset = (out_idx % pack_factor) * 4;
            let zero_packed = qzeros[group * out_features_packed + zero_pack_col];
            let zero_int = ((zero_packed >> zero_bit_offset) & 0xF) as i32;

            // Get per-group per-output scale
            let scale = scales_f32[group * out_features + out_idx];

            // Dequantize: (weight - zero) * scale
            let dequantized = (weight_int - zero_int) as f32 * scale;

            // Store as BF16 in [out, in] layout
            output_bf16[out_idx * in_features + in_idx] = f32_to_bf16(dequantized);
        }
    }

    // Convert u16 slice to bytes
    let output_bytes: Vec<u8> = output_bf16
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    Ok(WeightTensor {
        data: output_bytes,
        shape: vec![out_features, in_features],
        dtype: TensorDType::BF16,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Reinterpret a byte slice as a u32 slice (little-endian assumed).
fn bytes_as_u32(data: &[u8]) -> Result<Vec<u32>> {
    if data.len() % 4 != 0 {
        bail!(
            "byte slice length {} is not a multiple of 4",
            data.len()
        );
    }
    Ok(data
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

/// Convert scale tensor raw bytes to a Vec<f32>.
fn scales_to_f32(data: &[u8], dtype: TensorDType, count: usize) -> Result<Vec<f32>> {
    match dtype {
        TensorDType::F16 => {
            if data.len() < count * 2 {
                bail!("scales data too short for {} F16 values", count);
            }
            Ok(data
                .chunks_exact(2)
                .take(count)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    f16_to_f32(bits)
                })
                .collect())
        }
        TensorDType::BF16 => {
            if data.len() < count * 2 {
                bail!("scales data too short for {} BF16 values", count);
            }
            Ok(data
                .chunks_exact(2)
                .take(count)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    bf16_to_f32(bits)
                })
                .collect())
        }
        TensorDType::F32 => {
            if data.len() < count * 4 {
                bail!("scales data too short for {} F32 values", count);
            }
            Ok(data
                .chunks_exact(4)
                .take(count)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect())
        }
    }
}

/// Convert IEEE 754 half-precision (F16) bits to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            // ±0
            f32::from_bits(sign << 31)
        } else {
            // Subnormal: normalize
            let mut f = frac;
            let mut e = 0u32;
            while (f & 0x400) == 0 {
                f <<= 1;
                e += 1;
            }
            f &= 0x3FF;
            let exp32 = 127 - 15 + 1 - e;
            f32::from_bits((sign << 31) | (exp32 << 23) | (f << 13))
        }
    } else if exp == 31 {
        // Inf / NaN
        f32::from_bits((sign << 31) | (0xFF << 23) | (frac << 13))
    } else {
        // Normal
        let exp32 = exp + 127 - 15;
        f32::from_bits((sign << 31) | (exp32 << 23) | (frac << 13))
    }
}

/// Convert BF16 bits to f32.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Convert f32 to BF16 bits (round-to-nearest-even).
fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    // Round to nearest even: add 0x7FFF + bit 16 (the LSB of the result)
    let round = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (round >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_to_f32_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0f32);
        assert_eq!(f16_to_f32(0x8000), -0.0f32);
    }

    #[test]
    fn test_f16_to_f32_one() {
        assert!((f16_to_f32(0x3C00) - 1.0f32).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_roundtrip() {
        for v in [0.0f32, 1.0, -1.0, 0.5, 100.0, -0.001] {
            let bf16 = f32_to_bf16(v);
            let back = bf16_to_f32(bf16);
            assert!(
                (back - v).abs() < v.abs() * 0.01 + 1e-6,
                "roundtrip failed for {v}: got {back}"
            );
        }
    }

    #[test]
    fn test_bytes_as_u32() {
        let data = [0x01, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00];
        let result = bytes_as_u32(&data).unwrap();
        assert_eq!(result, vec![1u32, 255u32]);
    }

    #[test]
    fn test_bytes_as_u32_bad_alignment() {
        let data = [0x01, 0x02, 0x03];
        assert!(bytes_as_u32(&data).is_err());
    }

    #[test]
    fn test_dequantize_gptq_tiny() {
        // Create a tiny GPTQ-quantized weight:
        // 2x2 linear layer (in=16, out=2) with group_size=8
        //
        // qweight: [in/8, out] = [2, 2] (each u32 packs 8 INT4 values)
        // scales: [num_groups, out] = [2, 2]
        // qzeros: [num_groups, out/8] = [2, 1]

        let in_features = 16;
        let out_features = 2;
        let group_size = 8;
        let pack_factor = 8;

        // Pack weight values: all 5s (INT4 value = 5)
        // u32 with 8 copies of 5: 5 | (5<<4) | (5<<8) | ... | (5<<28)
        let packed_5: u32 = (0..pack_factor)
            .map(|i| 5u32 << (i * 4))
            .sum();

        // qweight: [2, 2] flattened row-major
        let qweight: Vec<u32> = vec![packed_5; (in_features / pack_factor) * out_features];
        let qweight_bytes: Vec<u8> = qweight.iter().flat_map(|v| v.to_le_bytes()).collect();

        // qzeros: all 8s (midpoint for symmetric INT4)
        // [2, 1] = 2 u32 values, each packing 8 zero points
        let packed_8: u32 = (0..pack_factor)
            .map(|i| 8u32 << (i * 4))
            .sum();
        let qzeros: Vec<u32> = vec![packed_8; 2];
        let qzeros_bytes: Vec<u8> = qzeros.iter().flat_map(|v| v.to_le_bytes()).collect();

        // scales: all 0.5 in F32
        let scale_val = 0.5f32;
        let num_groups = in_features / group_size;
        let scales: Vec<f32> = vec![scale_val; num_groups * out_features];
        let scales_bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();

        let result = dequantize_gptq_weight(
            &qweight_bytes,
            &[in_features / pack_factor, out_features],
            &scales_bytes,
            &[num_groups, out_features],
            TensorDType::F32,
            &qzeros_bytes,
            &[num_groups, out_features / pack_factor],
            group_size,
        )
        .unwrap();

        assert_eq!(result.shape, vec![out_features, in_features]);
        assert_eq!(result.dtype, TensorDType::BF16);

        // Expected: (5 - 8) * 0.5 = -1.5 for all elements
        let expected_bf16 = f32_to_bf16(-1.5);
        let result_u16: Vec<u16> = result
            .data
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();

        for (i, &val) in result_u16.iter().enumerate() {
            let actual = bf16_to_f32(val);
            assert!(
                (actual - (-1.5)).abs() < 0.1,
                "element {i}: expected -1.5, got {actual} (bf16 bits: {val:#06X}, expected {expected_bf16:#06X})"
            );
        }
    }

    #[test]
    fn test_dequantize_gptq_varying_values() {
        // Test that different packed INT4 values dequantize correctly
        let in_features = 8;
        let out_features = 1;
        let group_size = 8;

        // Pack values 0,1,2,3,4,5,6,7 into one u32
        let packed: u32 = (0..8u32).map(|i| i << (i * 4)).sum();
        let qweight_bytes: Vec<u8> = packed.to_le_bytes().to_vec();

        // Zero point = 0, scale = 1.0
        let packed_zero: u32 = 0;
        let qzeros_bytes: Vec<u8> = packed_zero.to_le_bytes().to_vec();
        let scale = 1.0f32;
        let scales_bytes: Vec<u8> = scale.to_le_bytes().to_vec();

        let result = dequantize_gptq_weight(
            &qweight_bytes,
            &[1, 1],
            &scales_bytes,
            &[1, 1],
            TensorDType::F32,
            &qzeros_bytes,
            &[1, 1],
            group_size,
        )
        .unwrap();

        assert_eq!(result.shape, vec![1, 8]);

        let result_values: Vec<f32> = result
            .data
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();

        for (i, &val) in result_values.iter().enumerate() {
            let expected = i as f32; // (i - 0) * 1.0
            assert!(
                (val - expected).abs() < 0.1,
                "element {i}: expected {expected}, got {val}"
            );
        }
    }

    #[test]
    fn test_gptq_config_parse() {
        let json = r#"{
            "bits": 4,
            "group_size": 128,
            "sym": true,
            "desc_act": false
        }"#;
        let config: GptqConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.bits, 4);
        assert_eq!(config.group_size, 128);
        assert!(config.sym);
        assert!(!config.desc_act);
        assert_eq!(config.pack_factor(), 8);
    }

    #[test]
    fn test_gptq_config_defaults() {
        let json = r#"{"bits": 4, "group_size": 128}"#;
        let config: GptqConfig = serde_json::from_str(json).unwrap();
        assert!(!config.sym);
        assert!(!config.desc_act);
    }
}
