//! `masked_select` — flatten elements where a boolean mask is true.
//!
//! Given a data tensor and a same-shape boolean mask, returns a
//! 1-D tensor containing the data elements where `mask == 1` (in
//! row-major order). PyTorch parity with `torch.masked_select(x,
//! mask)`.
//!
//! Used for: token-level loss masking (where the label is real, not
//! ignore_index), preference filtering on a batched logit tensor,
//! and "pull the active rows" workflows in RL replay buffers.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Layout, Result, Storage, Tensor, TensorId};

pub fn masked_select(x: &Tensor, mask: &Tensor) -> Result<Tensor> {
    if mask.dtype() != DType::U8 {
        bail!(
            "masked_select: mask dtype must be U8 (0/1), got {}",
            mask.dtype()
        );
    }
    if x.shape() != mask.shape() {
        bail!(
            "masked_select: shape mismatch — x {:?} vs mask {:?}",
            x.shape(),
            mask.shape()
        );
    }
    if x.dtype().is_packed() {
        bail!(
            "masked_select: packed data dtype {} not supported",
            x.dtype()
        );
    }
    if !x.is_contiguous() || !mask.is_contiguous() {
        bail!("masked_select: inputs must be contiguous");
    }

    let per = x.dtype().size_in_bytes();
    let n = x.element_count();
    let x_cpu = x.storage();
    let x_cpu = x_cpu
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| crate::Error::from_str("masked_select: x must be CpuStorage"))?;
    let m_cpu = mask.storage();
    let m_cpu = m_cpu
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| crate::Error::from_str("masked_select: mask must be CpuStorage"))?;
    let xb = x_cpu.as_bytes();
    let mb = m_cpu.as_bytes();

    // First pass: count selected.
    let mut count = 0usize;
    for i in 0..n {
        if mb[i] != 0 {
            count += 1;
        }
    }
    let mut out = vec![0u8; count * per];
    let mut w = 0usize;
    for i in 0..n {
        if mb[i] != 0 {
            let src = i * per;
            let dst = w * per;
            out[dst..dst + per].copy_from_slice(&xb[src..src + per]);
            w += 1;
        }
    }

    let cpu_out = CpuStorage::from_bytes(x.dtype(), out)?;
    let storage: Storage = Arc::new(cpu_out);
    Tensor::from_parts(storage, Layout::contiguous(vec![count]), TensorId::next())
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
    fn masked_select_1d() {
        let x = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0, 50.0], vec![5]).unwrap();
        let mask = Tensor::from_slice(&[1u8, 0, 1, 0, 1], vec![5]).unwrap();
        let y = masked_select(&x, &mask).unwrap();
        assert_eq!(y.shape(), &[3]);
        assert_eq!(read_f32(&y), vec![10.0, 30.0, 50.0]);
    }

    #[test]
    fn masked_select_2d_row_major() {
        // x = [[1,2,3],[4,5,6]], mask picks (0,0), (1,1), (1,2) → [1, 5, 6]
        let x =
            Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let mask = Tensor::from_slice(&[1u8, 0, 0, 0, 1, 1], vec![2, 3]).unwrap();
        let y = masked_select(&x, &mask).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 5.0, 6.0]);
    }

    #[test]
    fn masked_select_all_false_returns_empty() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let mask = Tensor::from_slice(&[0u8, 0, 0], vec![3]).unwrap();
        let y = masked_select(&x, &mask).unwrap();
        assert_eq!(y.shape(), &[0]);
    }

    #[test]
    fn masked_select_all_true_returns_flat() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let mask = Tensor::from_slice(&[1u8, 1, 1, 1], vec![2, 2]).unwrap();
        let y = masked_select(&x, &mask).unwrap();
        assert_eq!(y.shape(), &[4]);
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn masked_select_shape_mismatch_errors() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let mask = Tensor::from_slice(&[1u8, 0], vec![2]).unwrap();
        let e = masked_select(&x, &mask).unwrap_err();
        assert!(e.to_string().contains("shape mismatch"));
    }

    #[test]
    fn masked_select_non_u8_mask_errors() {
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let mask = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = masked_select(&x, &mask).unwrap_err();
        assert!(e.to_string().contains("U8"));
    }

    #[test]
    fn masked_select_nonzero_treated_as_true() {
        // Any nonzero u8 selects.
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let mask = Tensor::from_slice(&[42u8, 0, 7], vec![3]).unwrap();
        let y = masked_select(&x, &mask).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 3.0]);
    }
}
