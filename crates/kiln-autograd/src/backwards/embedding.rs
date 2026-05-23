//! `EmbeddingBackward` — gradient of `embedding(weights, token_ids)`.
//!
//! Forward (from `kiln_tensor::ops::embedding`):
//! - `weights: [V, H]`, F32 / BF16 / F16
//! - `token_ids: [...]`, I64 or U32
//! - output: `[..., H]`, dtype matching weights
//!
//! Backward: `d_weights = scatter_add(grad_output, axis=0,
//! indices=token_ids, target_dim=V)`. `d_token_ids = None` (indices
//! are non-differentiable).
//!
//! Two positions in `token_ids` that collide on the same vocab row
//! both contribute additively to the corresponding row of
//! `d_weights` — the canonical NLP-training behaviour. On GPU
//! backends this uses `atomicAdd` and picks up the atomic-bwd
//! tolerance band; CPU iteration order is fixed and bit-stable.

use kiln_tensor::ops::scatter_add;
use kiln_tensor::{bail, DType, Result, Tensor};

use crate::BackwardOp;

#[derive(Debug)]
pub struct EmbeddingBackward {
    /// Original vocab size. `d_weights.shape[0]`.
    pub vocab_size: usize,
    /// Hidden dim. `d_weights.shape[1]`. Used only to validate
    /// grad_output's trailing axis.
    pub hidden: usize,
    /// Saved `token_ids` from the forward pass.
    pub token_ids: Tensor,
}

impl BackwardOp for EmbeddingBackward {
    fn name(&self) -> &'static str {
        "embedding_backward"
    }
    fn input_count(&self) -> usize {
        2
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let go_shape = grad_output.shape();
        if go_shape.is_empty() {
            bail!(
                "EmbeddingBackward: grad_output must have rank ≥ 1 (got rank 0)"
            );
        }
        let last = *go_shape.last().unwrap();
        if last != self.hidden {
            bail!(
                "EmbeddingBackward: grad_output trailing axis {last} != hidden {}",
                self.hidden
            );
        }
        if !matches!(grad_output.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            bail!(
                "EmbeddingBackward: grad_output dtype must be F32/BF16/F16, got {}",
                grad_output.dtype()
            );
        }
        if !matches!(self.token_ids.dtype(), DType::I64 | DType::U32) {
            bail!(
                "EmbeddingBackward: token_ids dtype must be I64/U32, got {}",
                self.token_ids.dtype()
            );
        }
        // scatter_add along axis 0: produces [vocab_size, hidden].
        let d_weights =
            scatter_add(grad_output, 0, &self.token_ids, self.vocab_size)?;
        Ok(vec![Some(d_weights), None /* token_ids non-differentiable */])
    }
    fn requires_input(&self, idx: usize) -> bool {
        // The backward needs token_ids (saved on the struct) but
        // does not read the forward weights — `requires_input(0)` is
        // false to enable Phase 6.5's selective-recompute.
        match idx {
            0 => false, // weights
            1 => true,  // token_ids (saved)
            _ => false,
        }
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

    #[test]
    fn embedding_backward_scatter_into_unique_rows() {
        // V=4, H=2. token_ids=[0, 2]. grad_output=[[1, 2], [3, 4]].
        // d_weights = scatter_add: row 0 += [1, 2], row 2 += [3, 4].
        // Expected: [[1, 2], [0, 0], [3, 4], [0, 0]].
        let token_ids = Tensor::from_slice(&[0i64, 2], vec![2]).unwrap();
        let grad_output =
            Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let bo = EmbeddingBackward {
            vocab_size: 4,
            hidden: 2,
            token_ids,
        };
        let grads = bo.apply(&grad_output).unwrap();
        let d_w = grads[0].as_ref().unwrap();
        assert_eq!(d_w.shape(), &[4, 2]);
        assert_eq!(read_f32(d_w), vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]);
        assert!(grads[1].is_none());
    }

    #[test]
    fn embedding_backward_collisions_accumulate() {
        // token_ids = [1, 1]. grad_output = [[1, 1], [2, 2]].
        // d_weights row 1 = [1+2, 1+2] = [3, 3]; others zero.
        let token_ids = Tensor::from_slice(&[1i64, 1], vec![2]).unwrap();
        let grad_output =
            Tensor::from_slice(&[1.0f32, 1.0, 2.0, 2.0], vec![2, 2]).unwrap();
        let bo = EmbeddingBackward {
            vocab_size: 2,
            hidden: 2,
            token_ids,
        };
        let d_w = bo.apply(&grad_output).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(read_f32(&d_w), vec![0.0, 0.0, 3.0, 3.0]);
    }

    #[test]
    fn embedding_backward_2d_tokens() {
        // token_ids shape [B=2, S=2], values [B, S, H=2]. Each (b,s)
        // scatters into d_weights[id].
        let token_ids = Tensor::from_slice(&[0i64, 1, 1, 0], vec![2, 2]).unwrap();
        let grad_output = Tensor::from_slice(
            &[1.0f32, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
            vec![2, 2, 2],
        )
        .unwrap();
        let bo = EmbeddingBackward {
            vocab_size: 2,
            hidden: 2,
            token_ids,
        };
        let d_w = bo.apply(&grad_output).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(d_w.shape(), &[2, 2]);
        // Row 0: tokens at flat positions 0 (val [1,1]) and 3 (val [4,4]) → [5, 5]
        // Row 1: tokens at flat positions 1 (val [2,2]) and 2 (val [3,3]) → [5, 5]
        assert_eq!(read_f32(&d_w), vec![5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn embedding_backward_rejects_bad_grad_shape() {
        let token_ids = Tensor::from_slice(&[0i64], vec![1]).unwrap();
        let bad = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let bo = EmbeddingBackward {
            vocab_size: 1,
            hidden: 2,
            token_ids,
        };
        let e = bo.apply(&bad).unwrap_err();
        assert!(e.to_string().contains("trailing axis"));
    }

    #[test]
    fn embedding_backward_rejects_rank0_grad() {
        let token_ids = Tensor::from_slice(&[0i64], vec![1]).unwrap();
        let bad = Tensor::from_slice(&[1.0f32], vec![]).unwrap();
        let bo = EmbeddingBackward {
            vocab_size: 1,
            hidden: 1,
            token_ids,
        };
        let e = bo.apply(&bad).unwrap_err();
        assert!(e.to_string().contains("rank ≥ 1"));
    }

    #[test]
    fn op_metadata() {
        let token_ids = Tensor::from_slice(&[0i64], vec![1]).unwrap();
        let bo = EmbeddingBackward {
            vocab_size: 1,
            hidden: 1,
            token_ids,
        };
        assert_eq!(bo.name(), "embedding_backward");
        assert_eq!(bo.input_count(), 2);
        assert!(!bo.requires_input(0)); // weights not needed
        assert!(bo.requires_input(1)); // token_ids saved
    }
}
