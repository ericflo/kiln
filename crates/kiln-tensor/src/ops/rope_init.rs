//! `precompute_rope_freqs` — build the sin/cos schedules for RoPE.
//!
//! Given a max sequence length and a rotary head dim, generates:
//!
//! ```text
//! freqs[s, i] = s / theta^(2i / rotary_dim) for s ∈ [0, max_seq),
//!                                            i ∈ [0, rotary_dim/2)
//! cos[s, i] = cos(freqs[s, i])
//! sin[s, i] = sin(freqs[s, i])
//! ```
//!
//! `theta = 10000.0` for the original transformer RoPE; Qwen3.5-4B
//! uses `theta = 10000000.0`. Returns `(cos, sin)` both
//! `[max_seq, rotary_dim/2]` F32.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Layout, Result, Storage, Tensor, TensorId};

pub fn precompute_rope_freqs(
    max_seq: usize,
    rotary_dim: usize,
    theta: f32,
) -> Result<(Tensor, Tensor)> {
    if max_seq == 0 {
        bail!("precompute_rope_freqs: max_seq must be > 0");
    }
    if rotary_dim < 2 || !rotary_dim.is_multiple_of(2) {
        bail!("precompute_rope_freqs: rotary_dim must be ≥ 2 and even");
    }
    if theta <= 1.0 {
        bail!("precompute_rope_freqs: theta must be > 1");
    }
    let half = rotary_dim / 2;
    // inv_freq[i] = 1 / theta^(2i / rotary_dim)
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0_f32 / theta.powf((2 * i) as f32 / rotary_dim as f32))
        .collect();
    let mut cos_bytes = vec![0u8; max_seq * half * 4];
    let mut sin_bytes = vec![0u8; max_seq * half * 4];
    for s in 0..max_seq {
        for i in 0..half {
            let a = (s as f32) * inv_freq[i];
            let c = a.cos();
            let ss = a.sin();
            let off = (s * half + i) * 4;
            cos_bytes[off..off + 4].copy_from_slice(&c.to_le_bytes());
            sin_bytes[off..off + 4].copy_from_slice(&ss.to_le_bytes());
        }
    }
    let cos_cpu = CpuStorage::from_bytes(DType::F32, cos_bytes)?;
    let sin_cpu = CpuStorage::from_bytes(DType::F32, sin_bytes)?;
    let cos_storage: Storage = Arc::new(cos_cpu);
    let sin_storage: Storage = Arc::new(sin_cpu);
    let cos = Tensor::from_parts(
        cos_storage,
        Layout::contiguous(vec![max_seq, half]),
        TensorId::next(),
    )?;
    let sin = Tensor::from_parts(
        sin_storage,
        Layout::contiguous(vec![max_seq, half]),
        TensorId::next(),
    )?;
    Ok((cos, sin))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn rope_freqs_at_position_zero_are_one_and_zero() {
        // For s=0, frequency * 0 = 0, so cos=1, sin=0 always.
        let (cos, sin) = precompute_rope_freqs(4, 8, 10000.0).unwrap();
        let cv = read_f32(&cos);
        let sv = read_f32(&sin);
        let half = 8 / 2;
        for i in 0..half {
            assert!((cv[i] - 1.0).abs() < 1e-6, "cos[0, {i}] = {}", cv[i]);
            assert!(sv[i].abs() < 1e-6, "sin[0, {i}] = {}", sv[i]);
        }
    }

    #[test]
    fn rope_freqs_shape() {
        let (cos, sin) = precompute_rope_freqs(16, 32, 10000.0).unwrap();
        assert_eq!(cos.shape(), &[16, 16]);
        assert_eq!(sin.shape(), &[16, 16]);
        assert_eq!(cos.dtype(), DType::F32);
    }

    #[test]
    fn rope_freqs_qwen_theta_compiles() {
        // Qwen3.5-4B uses theta = 10_000_000.
        let (cos, _sin) = precompute_rope_freqs(4, 4, 10_000_000.0).unwrap();
        assert_eq!(cos.shape(), &[4, 2]);
    }

    #[test]
    fn rope_freqs_with_kiln_tensor_rope_runs() {
        use crate::ops::rope;
        let (cos, sin) = precompute_rope_freqs(4, 4, 10000.0).unwrap();
        // Build [B=1, S=4, D=4] x and apply rope.
        let x = Tensor::from_slice(&[1.0f32; 16], vec![1, 4, 4]).unwrap();
        let y = rope(&x, &cos, &sin, 4).unwrap();
        assert_eq!(y.shape(), &[1, 4, 4]);
    }

    #[test]
    fn rope_freqs_max_seq_zero_errors() {
        let e = precompute_rope_freqs(0, 4, 10000.0).unwrap_err();
        assert!(e.to_string().contains("max_seq"));
    }

    #[test]
    fn rope_freqs_odd_rotary_dim_errors() {
        let e = precompute_rope_freqs(4, 3, 10000.0).unwrap_err();
        assert!(e.to_string().contains("rotary_dim"));
    }

    #[test]
    fn rope_freqs_invalid_theta_errors() {
        let e = precompute_rope_freqs(4, 4, 0.5).unwrap_err();
        assert!(e.to_string().contains("theta"));
    }
}
