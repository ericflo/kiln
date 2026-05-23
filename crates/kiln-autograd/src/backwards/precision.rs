//! Backwards for the dtype-cast ops `to_f32`, `to_bf16`, `to_f16`.
//!
//! Cast is the identity gradient in the source dtype. The gradient
//! flowing in is in the *destination* dtype; the backward pass
//! casts it back to the source dtype before handing it upstream.
//!
//! | Forward       | Backward |
//! |---------------|----------|
//! | `to_bf16(x)`  | `dx = grad_out.cast(orig_dtype)` |
//! | `to_f16(x)`   | same |
//! | `to_f32(x)`   | same |

use kiln_tensor::{bail, DType, Result, Tensor};

use crate::BackwardOp;

/// Generic cast-backward. Saves the source dtype + shape and casts
/// `grad_output` back to the source dtype.
#[derive(Debug)]
pub struct CastBackward {
    pub orig_dtype: DType,
    pub shape: Vec<usize>,
}

impl BackwardOp for CastBackward {
    fn name(&self) -> &'static str {
        "cast_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if grad_output.shape() != self.shape {
            bail!(
                "cast_backward: shape mismatch — grad {:?} vs saved {:?}",
                grad_output.shape(),
                self.shape
            );
        }
        // If the saved dtype matches the incoming grad dtype, no cast.
        let out = if grad_output.dtype() == self.orig_dtype {
            grad_output.clone()
        } else {
            kiln_tensor::ops::cast(grad_output, self.orig_dtype)?
        };
        Ok(vec![Some(out)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_tensor::{CpuStorage, DType, Tensor};

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn read_bf16(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(2)
            .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
            .collect()
    }

    fn first(g: Vec<Option<Tensor>>) -> Tensor {
        g.into_iter().next().unwrap().unwrap()
    }

    #[test]
    fn cast_bwd_same_dtype_is_identity() {
        // Saved dtype = grad dtype → no cast, value preserved.
        let bwd = CastBackward {
            orig_dtype: DType::F32,
            shape: vec![3],
        };
        let dy = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let g = first(bwd.apply(&dy).unwrap());
        assert_eq!(g.dtype(), DType::F32);
        assert_eq!(read_f32(&g), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn cast_bwd_bf16_grad_back_to_f32() {
        // Forward was to_bf16(x); backward grad comes in as bf16
        // and we cast back to f32.
        let bwd = CastBackward {
            orig_dtype: DType::F32,
            shape: vec![3],
        };
        let dy_bf16 = Tensor::from_slice(
            &[
                half::bf16::from_f32(1.0),
                half::bf16::from_f32(2.0),
                half::bf16::from_f32(3.0),
            ],
            vec![3],
        )
        .unwrap();
        let g = first(bwd.apply(&dy_bf16).unwrap());
        assert_eq!(g.dtype(), DType::F32);
        let v = read_f32(&g);
        for (got, want) in v.iter().zip([1.0f32, 2.0, 3.0].iter()) {
            assert!((got - want).abs() < 1e-2);
        }
    }

    #[test]
    fn cast_bwd_f32_grad_back_to_bf16() {
        // Forward was to_f32(x_bf16); backward grad arrives as f32
        // and we cast back to bf16.
        let bwd = CastBackward {
            orig_dtype: DType::BF16,
            shape: vec![3],
        };
        let dy = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let g = first(bwd.apply(&dy).unwrap());
        assert_eq!(g.dtype(), DType::BF16);
        let v = read_bf16(&g);
        for (got, want) in v.iter().zip([1.0f32, 2.0, 3.0].iter()) {
            assert!((got - want).abs() < 1e-2);
        }
    }

    #[test]
    fn cast_bwd_shape_mismatch_errors() {
        let bwd = CastBackward {
            orig_dtype: DType::F32,
            shape: vec![3],
        };
        let dy = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let e = bwd.apply(&dy).unwrap_err();
        assert!(e.to_string().contains("shape"));
    }

    #[test]
    fn cast_bwd_input_count_is_one() {
        let bwd = CastBackward {
            orig_dtype: DType::F32,
            shape: vec![1],
        };
        assert_eq!(bwd.input_count(), 1);
    }

    #[test]
    fn cast_bwd_f16_round_trip() {
        // Forward to_f32 on f16 input; backward casts f32 grad → f16.
        let bwd = CastBackward {
            orig_dtype: DType::F16,
            shape: vec![2],
        };
        let dy = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let g = first(bwd.apply(&dy).unwrap());
        assert_eq!(g.dtype(), DType::F16);
    }
}
