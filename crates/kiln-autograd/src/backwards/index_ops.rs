//! Backwards for indexing and dtype-conversion ops:
//! `index_select`, `scatter_add`, and `cast`.
//!
//! `index_select` and `scatter_add` are mutually inverse on backward:
//!
//! - `index_select(source, axis, indices)` →
//!   `d_source = scatter_add(grad_output, axis, indices, target_dim=source.shape[axis])`
//! - `scatter_add(values, axis, indices, target_dim)` →
//!   `d_values = index_select(grad_output, axis, indices)`
//!
//! `cast(x, dtype)` is its own inverse: backward is just
//! `d_x = cast(grad_output, original_dtype)`.

use kiln_tensor::ops::{cast, index_select, scatter_add};
use kiln_tensor::{bail, DType, Result, Tensor};

use crate::BackwardOp;

// ----------------------------------------------------------------------
// IndexSelectBackward
// ----------------------------------------------------------------------

#[derive(Debug)]
pub struct IndexSelectBackward {
    /// `axis` from the forward call.
    pub axis: usize,
    /// `source.shape[axis]` from the forward call.
    pub source_axis_dim: usize,
    /// Saved indices from the forward call.
    pub indices: Tensor,
}

impl BackwardOp for IndexSelectBackward {
    fn name(&self) -> &'static str {
        "index_select_backward"
    }
    fn input_count(&self) -> usize {
        2
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if !matches!(self.indices.dtype(), DType::I64 | DType::U32) {
            bail!(
                "IndexSelectBackward: indices dtype must be I64/U32, got {}",
                self.indices.dtype()
            );
        }
        // scatter_add the gradient back into the source shape along axis.
        let d_source = scatter_add(grad_output, self.axis, &self.indices, self.source_axis_dim)?;
        Ok(vec![Some(d_source), None /* indices non-differentiable */])
    }
    fn requires_input(&self, idx: usize) -> bool {
        match idx {
            0 => false, // source not needed
            1 => true,  // indices saved
            _ => false,
        }
    }
}

// ----------------------------------------------------------------------
// ScatterAddBackward
// ----------------------------------------------------------------------

#[derive(Debug)]
pub struct ScatterAddBackward {
    /// `axis` from the forward call.
    pub axis: usize,
    /// Saved indices from the forward call.
    pub indices: Tensor,
}

impl BackwardOp for ScatterAddBackward {
    fn name(&self) -> &'static str {
        "scatter_add_backward"
    }
    fn input_count(&self) -> usize {
        2
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if !matches!(self.indices.dtype(), DType::I64 | DType::U32) {
            bail!(
                "ScatterAddBackward: indices dtype must be I64/U32, got {}",
                self.indices.dtype()
            );
        }
        // Gather the gradient at the saved indices.
        let d_values = index_select(grad_output, self.axis, &self.indices)?;
        Ok(vec![Some(d_values), None])
    }
    fn requires_input(&self, idx: usize) -> bool {
        match idx {
            0 => false, // values not needed
            1 => true,  // indices saved
            _ => false,
        }
    }
}

// ----------------------------------------------------------------------
// CastBackward
// ----------------------------------------------------------------------

#[derive(Debug)]
pub struct CastBackward {
    /// The dtype of the *forward input* (`x`). Backward casts the
    /// upstream gradient back to this dtype.
    pub source_dtype: DType,
}

impl BackwardOp for CastBackward {
    fn name(&self) -> &'static str {
        "cast_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if grad_output.dtype() == self.source_dtype {
            // No-op cast; clone and return.
            return Ok(vec![Some(grad_output.clone())]);
        }
        let d_x = cast(grad_output, self.source_dtype)?;
        Ok(vec![Some(d_x)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_tensor::CpuStorage;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    // ─── IndexSelectBackward ─────────────────────────────────────

    #[test]
    fn index_select_backward_scatters_into_source_shape() {
        // source shape [4, 2]. indices [1, 3]. grad_output [2, 2].
        // d_source should be zero everywhere except rows 1 and 3.
        let indices = Tensor::from_slice(&[1i64, 3], vec![2]).unwrap();
        let grad = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let bo = IndexSelectBackward {
            axis: 0,
            source_axis_dim: 4,
            indices,
        };
        let grads = bo.apply(&grad).unwrap();
        let d = grads[0].as_ref().unwrap();
        assert_eq!(d.shape(), &[4, 2]);
        assert_eq!(
            read_f32(d),
            vec![0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0]
        );
        assert!(grads[1].is_none());
    }

    #[test]
    fn index_select_backward_collisions_accumulate() {
        // indices [1, 1] → both go into row 1 of d_source.
        let indices = Tensor::from_slice(&[1i64, 1], vec![2]).unwrap();
        let grad = Tensor::from_slice(&[1.0f32, 1.0, 2.0, 2.0], vec![2, 2]).unwrap();
        let bo = IndexSelectBackward {
            axis: 0,
            source_axis_dim: 2,
            indices,
        };
        let d = bo.apply(&grad).unwrap()[0].as_ref().unwrap().clone();
        // Row 0: zero; row 1: [1+2, 1+2] = [3, 3].
        assert_eq!(read_f32(&d), vec![0.0, 0.0, 3.0, 3.0]);
    }

    // ─── ScatterAddBackward ──────────────────────────────────────

    #[test]
    fn scatter_add_backward_gathers_at_indices() {
        // Forward was scatter_add(values, axis=0, indices=[2, 0], target_dim=3).
        // Backward: d_values = index_select(grad_output, axis=0, indices=[2, 0]).
        // grad_output [3, 2] = [[10, 20], [30, 40], [50, 60]].
        // → d_values = grad_output[[2, 0]] = [[50, 60], [10, 20]].
        let indices = Tensor::from_slice(&[2i64, 0], vec![2]).unwrap();
        let grad = Tensor::from_slice(
            &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
            vec![3, 2],
        )
        .unwrap();
        let bo = ScatterAddBackward { axis: 0, indices };
        let grads = bo.apply(&grad).unwrap();
        let d = grads[0].as_ref().unwrap();
        assert_eq!(d.shape(), &[2, 2]);
        assert_eq!(read_f32(d), vec![50.0, 60.0, 10.0, 20.0]);
        assert!(grads[1].is_none());
    }

    // ─── CastBackward ────────────────────────────────────────────

    #[test]
    fn cast_backward_round_trips_dtype() {
        // Forward: cast F32 → BF16. Backward: cast BF16 grad → F32.
        let bf_grad: Vec<half::bf16> = [1.0f32, 2.0, 3.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let grad = Tensor::from_slice(&bf_grad, vec![3]).unwrap();
        let bo = CastBackward {
            source_dtype: DType::F32,
        };
        let g = bo.apply(&grad).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(g.dtype(), DType::F32);
        // BF16 → F32 is exact for small whole numbers.
        assert_eq!(read_f32(&g), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn cast_backward_same_dtype_is_clone() {
        let grad = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let bo = CastBackward {
            source_dtype: DType::F32,
        };
        let g = bo.apply(&grad).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(read_f32(&g), vec![1.0, 2.0]);
    }

    // ─── metadata ────────────────────────────────────────────────

    #[test]
    fn op_metadata() {
        let idx = Tensor::from_slice(&[0i64], vec![1]).unwrap();
        let bo = IndexSelectBackward {
            axis: 0,
            source_axis_dim: 1,
            indices: idx.clone(),
        };
        assert_eq!(bo.name(), "index_select_backward");
        assert_eq!(bo.input_count(), 2);
        assert!(!bo.requires_input(0));
        assert!(bo.requires_input(1));

        let bo = ScatterAddBackward {
            axis: 0,
            indices: idx,
        };
        assert_eq!(bo.name(), "scatter_add_backward");
        assert_eq!(bo.input_count(), 2);

        let bo = CastBackward {
            source_dtype: DType::F32,
        };
        assert_eq!(bo.name(), "cast_backward");
        assert_eq!(bo.input_count(), 1);
    }
}
