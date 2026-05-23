//! `where_select` — ternary mask-based select.
//!
//! ```text
//! out[i] = mask[i] != 0 ? t[i] : f[i]
//! ```
//!
//! Different from [`crate::ops::masked_fill`] (which has a scalar
//! fill value) — `where_select` chooses between two same-shape
//! tensors elementwise. Used in:
//!
//! - **Attention pre-softmax masking**: `where(causal_mask,
//!   -inf_tensor, scores)`.
//! - **Gating circuits**: route values through one path or another
//!   based on a boolean predicate.
//!
//! # Semantics
//!
//! All three inputs (`mask`, `t`, `f`) must share shape. `mask`
//! must be U8; `t` and `f` must share dtype (F32/BF16/F16). Output
//! has the same shape and dtype as `t`/`f`.
//!
//! # Determinism
//!
//! `Constructive`. Pointwise; no reduction.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

/// Ternary mask-based select.
pub fn where_select(mask: &Tensor, t: &Tensor, f: &Tensor) -> Result<Tensor> {
    validate(mask, t, f)?;
    let dtype = t.dtype();
    let per = dtype.size_in_bytes();
    let n = t.element_count();
    let m_cpu = downcast_cpu(mask, "mask")?;
    let t_cpu = downcast_cpu(t, "t")?;
    let f_cpu = downcast_cpu(f, "f")?;
    let m_bytes = m_cpu.as_bytes();
    let t_bytes = t_cpu.as_bytes();
    let f_bytes = f_cpu.as_bytes();
    let mut out = vec![0u8; n * per];

    for i in 0..n {
        let src = if m_bytes[i] != 0 {
            &t_bytes[i * per..(i + 1) * per]
        } else {
            &f_bytes[i * per..(i + 1) * per]
        };
        out[i * per..(i + 1) * per].copy_from_slice(src);
    }
    let cpu = CpuStorage::from_bytes(dtype, out)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(t.shape().to_vec()), TensorId::next())
}

fn validate(mask: &Tensor, t: &Tensor, f: &Tensor) -> Result<()> {
    if mask.shape() != t.shape() || t.shape() != f.shape() {
        bail!(
            "where_select: shape mismatch: mask {:?}, t {:?}, f {:?}",
            mask.shape(),
            t.shape(),
            f.shape()
        );
    }
    if mask.dtype() != DType::U8 {
        bail!("where_select: mask dtype must be U8, got {}", mask.dtype());
    }
    if t.dtype() != f.dtype() {
        bail!(
            "where_select: t/f dtype mismatch: t={}, f={}",
            t.dtype(),
            f.dtype()
        );
    }
    if !matches!(t.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("where_select: dtype must be F32/BF16/F16, got {}", t.dtype());
    }
    if !mask.is_contiguous() || !t.is_contiguous() || !f.is_contiguous() {
        bail!("where_select: all inputs must be contiguous");
    }
    Ok(())
}

fn downcast_cpu<'a>(t: &'a Tensor, label: &str) -> Result<&'a CpuStorage> {
    t.storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::Msg(format!("where_select: {label} storage must be CpuStorage")))
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
    fn where_select_picks_per_position() {
        // mask = [1, 0, 1, 0]; t = [1, 2, 3, 4]; f = [10, 20, 30, 40]
        // out = [1, 20, 3, 40]
        let m = Tensor::from_slice(&[1u8, 0, 1, 0], vec![4]).unwrap();
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let f = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0], vec![4]).unwrap();
        let out = where_select(&m, &t, &f).unwrap();
        assert_eq!(read_f32(&out), vec![1.0, 20.0, 3.0, 40.0]);
    }

    #[test]
    fn where_select_all_mask_set_returns_t() {
        let m = Tensor::from_slice(&[1u8, 1, 1], vec![3]).unwrap();
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let f = Tensor::from_slice(&[7.0f32, 8.0, 9.0], vec![3]).unwrap();
        let out = where_select(&m, &t, &f).unwrap();
        assert_eq!(read_f32(&out), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn where_select_all_mask_clear_returns_f() {
        let m = Tensor::from_slice(&[0u8, 0, 0], vec![3]).unwrap();
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let f = Tensor::from_slice(&[7.0f32, 8.0, 9.0], vec![3]).unwrap();
        let out = where_select(&m, &t, &f).unwrap();
        assert_eq!(read_f32(&out), vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn where_select_rank2() {
        // 2x2 case
        let m = Tensor::from_slice(&[1u8, 0, 0, 1], vec![2, 2]).unwrap();
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let f = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0], vec![2, 2]).unwrap();
        let out = where_select(&m, &t, &f).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(read_f32(&out), vec![1.0, 20.0, 30.0, 4.0]);
    }

    #[test]
    fn where_select_bf16_path() {
        let m = Tensor::from_slice(&[1u8, 0], vec![2]).unwrap();
        let tv: Vec<half::bf16> = [1.0f32, 2.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let fv: Vec<half::bf16> = [10.0f32, 20.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let t = Tensor::from_slice(&tv, vec![2]).unwrap();
        let f = Tensor::from_slice(&fv, vec![2]).unwrap();
        let out = where_select(&m, &t, &f).unwrap();
        assert_eq!(out.dtype(), DType::BF16);
        // Re-read with bf16:
        let cpu = out.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let bytes = cpu.as_bytes();
        let v0 = half::bf16::from_le_bytes(bytes[..2].try_into().unwrap()).to_f32();
        let v1 = half::bf16::from_le_bytes(bytes[2..4].try_into().unwrap()).to_f32();
        assert!((v0 - 1.0).abs() < 1e-2);
        assert!((v1 - 20.0).abs() < 1e-2);
    }

    #[test]
    fn where_select_shape_mismatch_errors() {
        let m = Tensor::from_slice(&[1u8, 0], vec![2]).unwrap();
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let f = Tensor::from_slice(&[10.0f32, 20.0], vec![2]).unwrap();
        let e = where_select(&m, &t, &f).unwrap_err();
        assert!(e.to_string().contains("shape mismatch"));
    }

    #[test]
    fn where_select_mask_dtype_errors() {
        let m = Tensor::from_slice(&[1.0f32, 0.0], vec![2]).unwrap();
        let t = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let f = Tensor::from_slice(&[10.0f32, 20.0], vec![2]).unwrap();
        let e = where_select(&m, &t, &f).unwrap_err();
        assert!(e.to_string().contains("mask dtype"));
    }

    #[test]
    fn where_select_tf_dtype_mismatch_errors() {
        let m = Tensor::from_slice(&[1u8, 0], vec![2]).unwrap();
        let t = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let fv = vec![half::bf16::from_f32(10.0), half::bf16::from_f32(20.0)];
        let f = Tensor::from_slice(&fv, vec![2]).unwrap();
        let e = where_select(&m, &t, &f).unwrap_err();
        assert!(e.to_string().contains("t/f dtype"));
    }
}
