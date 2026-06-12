//! `MaximumBackward` — gradient of elementwise `maximum(a, b)`.
//!
//! Forward: `out[i] = max(a[i], b[i])`.
//!
//! Backward routes each gradient element to whichever side held the
//! max. Ties (`a[i] == b[i]`) are split 50/50 — the standard
//! subgradient convention; this is what PyTorch / Jax do as well.
//!
//! ```text
//! d_a[i] = grad_output[i] * (a[i] > b[i]) + grad_output[i] * 0.5 * (a[i] == b[i])
//! d_b[i] = grad_output[i] * (a[i] < b[i]) + grad_output[i] * 0.5 * (a[i] == b[i])
//! ```
//!
//! Both saved inputs `a` and `b` are required (only their byte
//! contents — broadcasting is *not* applied here; the forward
//! enforces equal shapes).

use std::sync::Arc;

use kiln_tensor::{CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId, bail};

use crate::BackwardOp;

#[derive(Debug)]
pub struct MaximumBackward {
    /// Saved forward `a` input.
    pub a: Tensor,
    /// Saved forward `b` input.
    pub b: Tensor,
}

fn read_one_f32(dtype: DType, bytes: &[u8], i: usize, per: usize) -> f32 {
    let s = i * per;
    match dtype {
        DType::F32 => f32::from_le_bytes(bytes[s..s + 4].try_into().unwrap()),
        DType::BF16 => half::bf16::from_le_bytes(bytes[s..s + 2].try_into().unwrap()).to_f32(),
        DType::F16 => half::f16::from_le_bytes(bytes[s..s + 2].try_into().unwrap()).to_f32(),
        _ => unreachable!(),
    }
}

fn write_one_f32(dtype: DType, bytes: &mut [u8], i: usize, per: usize, v: f32) {
    let s = i * per;
    match dtype {
        DType::F32 => bytes[s..s + 4].copy_from_slice(&v.to_le_bytes()),
        DType::BF16 => bytes[s..s + 2].copy_from_slice(&half::bf16::from_f32(v).to_le_bytes()),
        DType::F16 => bytes[s..s + 2].copy_from_slice(&half::f16::from_f32(v).to_le_bytes()),
        _ => unreachable!(),
    }
}

impl BackwardOp for MaximumBackward {
    fn name(&self) -> &'static str {
        "maximum_backward"
    }
    fn input_count(&self) -> usize {
        2
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let dtype = grad_output.dtype();
        if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
            bail!("MaximumBackward: dtype must be F32/BF16/F16, got {dtype}");
        }
        if self.a.dtype() != dtype || self.b.dtype() != dtype {
            bail!(
                "MaximumBackward: dtype mismatch — grad {dtype}, a {}, b {}",
                self.a.dtype(),
                self.b.dtype()
            );
        }
        if self.a.shape() != grad_output.shape() || self.b.shape() != grad_output.shape() {
            bail!(
                "MaximumBackward: shape mismatch — grad {:?}, a {:?}, b {:?}",
                grad_output.shape(),
                self.a.shape(),
                self.b.shape()
            );
        }
        if !grad_output.is_contiguous() || !self.a.is_contiguous() || !self.b.is_contiguous() {
            bail!("MaximumBackward: inputs must be contiguous");
        }
        let per = dtype.size_in_bytes();
        let n = grad_output.element_count();

        let a_cpu = self
            .a
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("MaximumBackward: a must be CpuStorage"))?;
        let b_cpu = self
            .b
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("MaximumBackward: b must be CpuStorage"))?;
        let g_cpu = grad_output
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("MaximumBackward: grad must be CpuStorage"))?;
        let a_bytes = a_cpu.as_bytes();
        let b_bytes = b_cpu.as_bytes();
        let g_bytes = g_cpu.as_bytes();

        let mut da = vec![0u8; n * per];
        let mut db = vec![0u8; n * per];
        for i in 0..n {
            let av = read_one_f32(dtype, a_bytes, i, per);
            let bv = read_one_f32(dtype, b_bytes, i, per);
            let gv = read_one_f32(dtype, g_bytes, i, per);
            let (da_v, db_v) = if av > bv {
                (gv, 0.0)
            } else if av < bv {
                (0.0, gv)
            } else {
                let half = gv * 0.5;
                (half, half)
            };
            write_one_f32(dtype, &mut da, i, per, da_v);
            write_one_f32(dtype, &mut db, i, per, db_v);
        }
        let shape = grad_output.shape().to_vec();
        let da_cpu = CpuStorage::from_bytes(dtype, da)?;
        let db_cpu = CpuStorage::from_bytes(dtype, db)?;
        let da_storage: Storage = Arc::new(da_cpu);
        let db_storage: Storage = Arc::new(db_cpu);
        let d_a = Tensor::from_parts(
            da_storage,
            Layout::contiguous(shape.clone()),
            TensorId::next(),
        )?;
        let d_b = Tensor::from_parts(db_storage, Layout::contiguous(shape), TensorId::next())?;
        Ok(vec![Some(d_a), Some(d_b)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        // Both saved on the struct.
        true
    }
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
    fn maximum_backward_routes_to_winner() {
        // a = [1, 5, 3], b = [4, 2, 3]. max = [4, 5, 3]. ties at index 2.
        // grad = [10, 20, 30]:
        //   d_a = [0, 20, 15]   (idx 0: b wins, idx 1: a wins, idx 2: tie)
        //   d_b = [10, 0, 15]
        let a = Tensor::from_slice(&[1.0f32, 5.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[4.0f32, 2.0, 3.0], vec![3]).unwrap();
        let grad = Tensor::from_slice(&[10.0f32, 20.0, 30.0], vec![3]).unwrap();
        let bo = MaximumBackward { a, b };
        let grads = bo.apply(&grad).unwrap();
        assert_eq!(read_f32(grads[0].as_ref().unwrap()), vec![0.0, 20.0, 15.0]);
        assert_eq!(read_f32(grads[1].as_ref().unwrap()), vec![10.0, 0.0, 15.0]);
    }

    #[test]
    fn maximum_backward_all_a_wins() {
        let a = Tensor::from_slice(&[5.0f32, 5.0, 5.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[1.0f32, 1.0, 1.0], vec![3]).unwrap();
        let grad = Tensor::from_slice(&[10.0f32, 20.0, 30.0], vec![3]).unwrap();
        let bo = MaximumBackward { a, b };
        let grads = bo.apply(&grad).unwrap();
        assert_eq!(read_f32(grads[0].as_ref().unwrap()), vec![10.0, 20.0, 30.0]);
        assert_eq!(read_f32(grads[1].as_ref().unwrap()), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn maximum_backward_all_tie() {
        // All tie → 50/50 split.
        let a = Tensor::from_slice(&[5.0f32, 5.0], vec![2]).unwrap();
        let b = Tensor::from_slice(&[5.0f32, 5.0], vec![2]).unwrap();
        let grad = Tensor::from_slice(&[2.0f32, 4.0], vec![2]).unwrap();
        let bo = MaximumBackward { a, b };
        let grads = bo.apply(&grad).unwrap();
        assert_eq!(read_f32(grads[0].as_ref().unwrap()), vec![1.0, 2.0]);
        assert_eq!(read_f32(grads[1].as_ref().unwrap()), vec![1.0, 2.0]);
    }

    #[test]
    fn maximum_backward_2d_shape_preserved() {
        let a = Tensor::from_slice(&[1.0f32, 4.0, 3.0, 2.0], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[2.0f32, 3.0, 5.0, 1.0], vec![2, 2]).unwrap();
        let grad = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0], vec![2, 2]).unwrap();
        let bo = MaximumBackward { a, b };
        let grads = bo.apply(&grad).unwrap();
        assert_eq!(grads[0].as_ref().unwrap().shape(), &[2, 2]);
        // a < b at [0,0], a > b at [0,1], a < b at [1,0], a > b at [1,1].
        assert_eq!(
            read_f32(grads[0].as_ref().unwrap()),
            vec![0.0, 20.0, 0.0, 40.0]
        );
        assert_eq!(
            read_f32(grads[1].as_ref().unwrap()),
            vec![10.0, 0.0, 30.0, 0.0]
        );
    }

    #[test]
    fn maximum_backward_bf16_round_trips() {
        let bf = |v: f32| half::bf16::from_f32(v);
        let av: Vec<half::bf16> = [1.0f32, 5.0].iter().map(|&v| bf(v)).collect();
        let bv: Vec<half::bf16> = [4.0f32, 2.0].iter().map(|&v| bf(v)).collect();
        let gv: Vec<half::bf16> = [10.0f32, 20.0].iter().map(|&v| bf(v)).collect();
        let a = Tensor::from_slice(&av, vec![2]).unwrap();
        let b = Tensor::from_slice(&bv, vec![2]).unwrap();
        let grad = Tensor::from_slice(&gv, vec![2]).unwrap();
        let bo = MaximumBackward { a, b };
        let grads = bo.apply(&grad).unwrap();
        let da_bytes = grads[0]
            .as_ref()
            .unwrap()
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .unwrap()
            .as_bytes();
        let vals: Vec<f32> = (0..2)
            .map(|i| {
                half::bf16::from_le_bytes(da_bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
            })
            .collect();
        // a wins at i=1 only.
        assert_eq!(vals, vec![0.0, 20.0]);
    }

    #[test]
    fn maximum_backward_shape_mismatch_errors() {
        let a = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let b = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let grad = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let bo = MaximumBackward { a, b };
        let e = bo.apply(&grad).unwrap_err();
        assert!(e.to_string().contains("shape mismatch"));
    }

    #[test]
    fn op_metadata() {
        let a = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let b = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let bo = MaximumBackward { a, b };
        assert_eq!(bo.name(), "maximum_backward");
        assert_eq!(bo.input_count(), 2);
        assert!(bo.requires_input(0));
        assert!(bo.requires_input(1));
    }
}
