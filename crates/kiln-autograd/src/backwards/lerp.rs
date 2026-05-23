//! Backward for `lerp(a, b, w)`:
//!
//! `lerp = a + w * (b - a)`
//!
//! Gradients:
//! - `dL/da = dy * (1 - w)`
//! - `dL/db = dy * w`
//!
//! Weight `w` is a scalar f32 captured at construction; not
//! differentiable here (would need a separate fwd op variant if `w`
//! is itself a tensor).

use std::sync::Arc;

use kiln_tensor::{
    bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId,
};

use crate::BackwardOp;

fn validate_same_shape(a: &Tensor, b: &Tensor, name: &str) -> Result<()> {
    if a.shape() != b.shape() {
        bail!("{name}: shape mismatch: {:?} vs {:?}", a.shape(), b.shape());
    }
    if a.dtype() != b.dtype() {
        bail!("{name}: dtype mismatch");
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        bail!("{name}: inputs must be contiguous");
    }
    if !matches!(a.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("{name}: dtype must be F32/BF16/F16, got {}", a.dtype());
    }
    Ok(())
}

fn scale_tensor(t: &Tensor, factor: f32) -> Result<Tensor> {
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("lerp_backward: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let n = t.element_count();
    let dtype = t.dtype();
    let mut out_bytes = vec![0u8; bytes.len()];
    match dtype {
        DType::F32 => {
            for i in 0..n {
                let v = f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
                let sv = v * factor;
                out_bytes[i * 4..i * 4 + 4].copy_from_slice(&sv.to_le_bytes());
            }
        }
        DType::BF16 => {
            for i in 0..n {
                let v =
                    half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32();
                let sv = half::bf16::from_f32(v * factor);
                out_bytes[i * 2..i * 2 + 2].copy_from_slice(&sv.to_le_bytes());
            }
        }
        DType::F16 => {
            for i in 0..n {
                let v =
                    half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32();
                let sv = half::f16::from_f32(v * factor);
                out_bytes[i * 2..i * 2 + 2].copy_from_slice(&sv.to_le_bytes());
            }
        }
        _ => unreachable!(),
    }
    let cpu_out = CpuStorage::from_bytes(dtype, out_bytes)?;
    let storage: Storage = Arc::new(cpu_out);
    Tensor::from_parts(storage, Layout::contiguous(t.shape().to_vec()), TensorId::next())
}

#[derive(Debug)]
pub struct LerpBackward {
    pub weight: f32,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl BackwardOp for LerpBackward {
    fn name(&self) -> &'static str {
        "lerp_backward"
    }
    fn input_count(&self) -> usize {
        2
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if grad_output.shape() != self.shape {
            bail!(
                "lerp_backward: grad shape {:?} != saved shape {:?}",
                grad_output.shape(),
                self.shape
            );
        }
        if grad_output.dtype() != self.dtype {
            bail!(
                "lerp_backward: grad dtype {} != saved dtype {}",
                grad_output.dtype(),
                self.dtype
            );
        }
        if !matches!(grad_output.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            bail!(
                "lerp_backward: dtype must be F32/BF16/F16, got {}",
                grad_output.dtype()
            );
        }
        let _ = validate_same_shape; // keep helper exported for symmetry across this module family
        let da = scale_tensor(grad_output, 1.0 - self.weight)?;
        let db = scale_tensor(grad_output, self.weight)?;
        Ok(vec![Some(da), Some(db)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_tensor::Tensor;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn lerp_bwd_weight_zero_routes_full_grad_to_a() {
        // lerp(a, b, 0) = a → da = dy, db = 0.
        let bwd = LerpBackward {
            weight: 0.0,
            shape: vec![3],
            dtype: DType::F32,
        };
        let dy = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let grads = bwd.apply(&dy).unwrap();
        let da = grads[0].as_ref().unwrap();
        let db = grads[1].as_ref().unwrap();
        assert_eq!(read_f32(da), vec![1.0, 2.0, 3.0]);
        assert_eq!(read_f32(db), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn lerp_bwd_weight_one_routes_full_grad_to_b() {
        let bwd = LerpBackward {
            weight: 1.0,
            shape: vec![3],
            dtype: DType::F32,
        };
        let dy = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let grads = bwd.apply(&dy).unwrap();
        let da = grads[0].as_ref().unwrap();
        let db = grads[1].as_ref().unwrap();
        assert_eq!(read_f32(da), vec![0.0, 0.0, 0.0]);
        assert_eq!(read_f32(db), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn lerp_bwd_weight_half_splits_evenly() {
        let bwd = LerpBackward {
            weight: 0.5,
            shape: vec![2],
            dtype: DType::F32,
        };
        let dy = Tensor::from_slice(&[2.0f32, 4.0], vec![2]).unwrap();
        let grads = bwd.apply(&dy).unwrap();
        assert_eq!(read_f32(grads[0].as_ref().unwrap()), vec![1.0, 2.0]);
        assert_eq!(read_f32(grads[1].as_ref().unwrap()), vec![1.0, 2.0]);
    }

    #[test]
    fn lerp_bwd_grads_sum_to_dy() {
        // da + db = dy for any weight.
        let bwd = LerpBackward {
            weight: 0.7,
            shape: vec![3],
            dtype: DType::F32,
        };
        let dy = Tensor::from_slice(&[10.0f32, 20.0, 30.0], vec![3]).unwrap();
        let grads = bwd.apply(&dy).unwrap();
        let da = read_f32(grads[0].as_ref().unwrap());
        let db = read_f32(grads[1].as_ref().unwrap());
        for ((a, b), &y) in da.iter().zip(db.iter()).zip([10.0f32, 20.0, 30.0].iter()) {
            assert!(((a + b) - y).abs() < 1e-5);
        }
    }

    #[test]
    fn lerp_bwd_shape_mismatch_errors() {
        let bwd = LerpBackward {
            weight: 0.5,
            shape: vec![3],
            dtype: DType::F32,
        };
        let dy = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let e = bwd.apply(&dy).unwrap_err();
        assert!(e.to_string().contains("shape"));
    }

    #[test]
    fn lerp_bwd_input_count_is_two() {
        let bwd = LerpBackward {
            weight: 0.5,
            shape: vec![1],
            dtype: DType::F32,
        };
        assert_eq!(bwd.input_count(), 2);
    }
}
