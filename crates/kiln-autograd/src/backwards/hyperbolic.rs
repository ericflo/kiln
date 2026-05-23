//! Backwards for `sinh` and `cosh`.
//!
//! | Op    | Backward          |
//! |-------|-------------------|
//! | sinh  | `dx = dy * cosh(x)` |
//! | cosh  | `dx = dy * sinh(x)` |

use std::sync::Arc;

use kiln_tensor::{
    bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId,
};

use crate::BackwardOp;

fn validate_same(a: &Tensor, b: &Tensor, name: &str) -> Result<()> {
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

fn load_f32(t: &Tensor) -> Result<Vec<f32>> {
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("hyperbolic_backward: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let n = t.element_count();
    let dtype = t.dtype();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(match dtype {
            DType::F32 => f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            DType::BF16 => {
                half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
            }
            DType::F16 => {
                half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
            }
            _ => unreachable!(),
        });
    }
    Ok(out)
}

fn store_f32(dtype: DType, shape: &[usize], values: &[f32]) -> Result<Tensor> {
    let per = dtype.size_in_bytes();
    let mut bytes = vec![0u8; values.len() * per];
    match dtype {
        DType::F32 => {
            for (i, &v) in values.iter().enumerate() {
                bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        DType::BF16 => {
            for (i, &v) in values.iter().enumerate() {
                bytes[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
            }
        }
        DType::F16 => {
            for (i, &v) in values.iter().enumerate() {
                bytes[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
        }
        _ => unreachable!(),
    }
    let cpu = CpuStorage::from_bytes(dtype, bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(shape.to_vec()), TensorId::next())
}

#[derive(Debug)]
pub struct SinhBackward {
    pub x: Tensor,
}
impl BackwardOp for SinhBackward {
    fn name(&self) -> &'static str {
        "sinh_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        validate_same(&self.x, grad_output, "sinh_backward")?;
        let x = load_f32(&self.x)?;
        let dy = load_f32(grad_output)?;
        let dx: Vec<f32> = x
            .iter()
            .zip(dy.iter())
            .map(|(&xi, &dyi)| dyi * xi.cosh())
            .collect();
        Ok(vec![Some(store_f32(self.x.dtype(), self.x.shape(), &dx)?)])
    }
}

#[derive(Debug)]
pub struct CoshBackward {
    pub x: Tensor,
}
impl BackwardOp for CoshBackward {
    fn name(&self) -> &'static str {
        "cosh_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        validate_same(&self.x, grad_output, "cosh_backward")?;
        let x = load_f32(&self.x)?;
        let dy = load_f32(grad_output)?;
        let dx: Vec<f32> = x
            .iter()
            .zip(dy.iter())
            .map(|(&xi, &dyi)| dyi * xi.sinh())
            .collect();
        Ok(vec![Some(store_f32(self.x.dtype(), self.x.shape(), &dx)?)])
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

    fn first(g: Vec<Option<Tensor>>) -> Tensor {
        g.into_iter().next().unwrap().unwrap()
    }

    #[test]
    fn sinh_bwd_at_zero_is_one() {
        // d/dx sinh(x) at x=0 = cosh(0) = 1
        let bwd = SinhBackward {
            x: Tensor::from_slice(&[0.0f32], vec![1]).unwrap(),
        };
        let dy = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let g = first(bwd.apply(&dy).unwrap());
        assert!((read_f32(&g)[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosh_bwd_at_zero_is_zero() {
        // d/dx cosh(x) at x=0 = sinh(0) = 0
        let bwd = CoshBackward {
            x: Tensor::from_slice(&[0.0f32], vec![1]).unwrap(),
        };
        let dy = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let g = first(bwd.apply(&dy).unwrap());
        assert!(read_f32(&g)[0].abs() < 1e-5);
    }

    #[test]
    fn sinh_bwd_finite_difference() {
        let x_val = 0.7f32;
        let eps = 1e-3f32;
        let analytic = x_val.cosh();
        let numeric = ((x_val + eps).sinh() - (x_val - eps).sinh()) / (2.0 * eps);
        assert!((analytic - numeric).abs() / analytic.abs() < 1e-3);

        let bwd = SinhBackward {
            x: Tensor::from_slice(&[x_val], vec![1]).unwrap(),
        };
        let dy = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let g = first(bwd.apply(&dy).unwrap());
        assert!((read_f32(&g)[0] - analytic).abs() < 1e-5);
    }

    #[test]
    fn cosh_bwd_finite_difference() {
        let x_val = 1.2f32;
        let eps = 1e-3f32;
        let analytic = x_val.sinh();
        let numeric = ((x_val + eps).cosh() - (x_val - eps).cosh()) / (2.0 * eps);
        assert!((analytic - numeric).abs() / analytic.abs() < 1e-3);

        let bwd = CoshBackward {
            x: Tensor::from_slice(&[x_val], vec![1]).unwrap(),
        };
        let dy = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let g = first(bwd.apply(&dy).unwrap());
        assert!((read_f32(&g)[0] - analytic).abs() < 1e-5);
    }

    #[test]
    fn input_count_is_one() {
        let bwd = SinhBackward {
            x: Tensor::from_slice(&[1.0f32], vec![1]).unwrap(),
        };
        assert_eq!(bwd.input_count(), 1);
    }
}
