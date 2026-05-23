//! Backwards for `sin`, `cos`, `tan`. Each saves `x` and recomputes
//! the chain-rule term in backward.
//!
//! | Op  | Backward |
//! |-----|----------|
//! | sin | `dx = dy * cos(x)`  |
//! | cos | `dx = -dy * sin(x)` |
//! | tan | `dx = dy * sec²(x) = dy * (1 + tan²(x))` |

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
        .ok_or_else(|| Error::from_str("trig_backward: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let n = t.element_count();
    let dtype = t.dtype();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(match dtype {
            DType::F32 => f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            DType::BF16 => half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                .to_f32(),
            DType::F16 => half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                .to_f32(),
            _ => unreachable!(),
        });
    }
    Ok(out)
}

fn store_f32(dtype: DType, shape: &[usize], data: &[f32]) -> Result<Tensor> {
    let per = dtype.size_in_bytes();
    let mut bytes = vec![0u8; data.len() * per];
    match dtype {
        DType::F32 => {
            for (i, &v) in data.iter().enumerate() {
                bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        DType::BF16 => {
            for (i, &v) in data.iter().enumerate() {
                bytes[i * 2..i * 2 + 2].copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
            }
        }
        DType::F16 => {
            for (i, &v) in data.iter().enumerate() {
                bytes[i * 2..i * 2 + 2].copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
        }
        _ => unreachable!(),
    }
    let cpu = CpuStorage::from_bytes(dtype, bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(shape.to_vec()), TensorId::next())
}

#[derive(Debug)]
pub struct SinBackward {
    pub x: Tensor,
}
impl BackwardOp for SinBackward {
    fn name(&self) -> &'static str {
        "sin_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        validate_same(&self.x, grad_output, "SinBackward")?;
        let x = load_f32(&self.x)?;
        let dy = load_f32(grad_output)?;
        let dx: Vec<f32> = x
            .iter()
            .zip(dy.iter())
            .map(|(&xi, &dyi)| dyi * xi.cos())
            .collect();
        Ok(vec![Some(store_f32(self.x.dtype(), self.x.shape(), &dx)?)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        true
    }
}

#[derive(Debug)]
pub struct CosBackward {
    pub x: Tensor,
}
impl BackwardOp for CosBackward {
    fn name(&self) -> &'static str {
        "cos_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        validate_same(&self.x, grad_output, "CosBackward")?;
        let x = load_f32(&self.x)?;
        let dy = load_f32(grad_output)?;
        let dx: Vec<f32> = x
            .iter()
            .zip(dy.iter())
            .map(|(&xi, &dyi)| -dyi * xi.sin())
            .collect();
        Ok(vec![Some(store_f32(self.x.dtype(), self.x.shape(), &dx)?)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        true
    }
}

#[derive(Debug)]
pub struct TanBackward {
    pub x: Tensor,
}
impl BackwardOp for TanBackward {
    fn name(&self) -> &'static str {
        "tan_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        validate_same(&self.x, grad_output, "TanBackward")?;
        let x = load_f32(&self.x)?;
        let dy = load_f32(grad_output)?;
        let dx: Vec<f32> = x
            .iter()
            .zip(dy.iter())
            .map(|(&xi, &dyi)| {
                let t = xi.tan();
                dyi * (1.0 + t * t) // sec²(x) = 1 + tan²(x)
            })
            .collect();
        Ok(vec![Some(store_f32(self.x.dtype(), self.x.shape(), &dx)?)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "idx {i}: got {x}, want {y} (tol {tol})"
            );
        }
    }

    #[test]
    fn sin_backward_at_zero_is_one() {
        let x = Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let bo = SinBackward { x };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();
        approx(&dx, &[1.0], 1e-6);
    }

    #[test]
    fn cos_backward_at_zero_is_zero() {
        let x = Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let bo = CosBackward { x };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();
        approx(&dx, &[0.0], 1e-6);
    }

    #[test]
    fn sin_backward_at_pi_half_is_zero() {
        let x = Tensor::from_slice(&[std::f32::consts::FRAC_PI_2], vec![1]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let bo = SinBackward { x };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();
        approx(&dx, &[0.0], 1e-6);
    }

    #[test]
    fn cos_backward_at_pi_half_is_neg_one() {
        let x = Tensor::from_slice(&[std::f32::consts::FRAC_PI_2], vec![1]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let bo = CosBackward { x };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();
        approx(&dx, &[-1.0], 1e-6);
    }

    #[test]
    fn tan_backward_at_zero_is_one() {
        // tan(0) = 0; sec²(0) = 1.
        let x = Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let bo = TanBackward { x };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();
        approx(&dx, &[1.0], 1e-6);
    }

    #[test]
    fn tan_backward_at_pi_quarter_is_two() {
        // tan(π/4) = 1; sec²(π/4) = 1 + 1 = 2.
        let x = Tensor::from_slice(&[std::f32::consts::FRAC_PI_4], vec![1]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let bo = TanBackward { x };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();
        approx(&dx, &[2.0], 1e-5);
    }

    #[test]
    fn sin_backward_finite_difference() {
        use kiln_tensor::ops::sin;
        let x_data = vec![0.3f32, -1.2, 2.5];
        let x = Tensor::from_slice(&x_data, vec![3]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32; 3], vec![3]).unwrap();
        let bo = SinBackward { x: x.clone() };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();
        let loss = |xv: &[f32]| -> f32 {
            let xt = Tensor::from_slice(xv, vec![3]).unwrap();
            load_f32(&sin(&xt).unwrap()).unwrap().iter().sum()
        };
        let step = 1e-3;
        let mut fd = Vec::with_capacity(3);
        for i in 0..3 {
            let mut up = x_data.clone();
            up[i] += step;
            let mut dn = x_data.clone();
            dn[i] -= step;
            fd.push((loss(&up) - loss(&dn)) / (2.0 * step));
        }
        approx(&dx, &fd, 1e-3);
    }

    #[test]
    fn op_metadata() {
        let one = Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
        assert_eq!(SinBackward { x: one.clone() }.name(), "sin_backward");
        assert_eq!(CosBackward { x: one.clone() }.name(), "cos_backward");
        assert_eq!(TanBackward { x: one }.name(), "tan_backward");
    }
}
