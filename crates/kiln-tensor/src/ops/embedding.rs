//! `EmbeddingOp` — token-id → row gather (a.k.a. `index_select` along axis 0).
//!
//! This is the first **real** `DeviceOp` impl in kiln-tensor: it
//! demonstrates the trait shape, the dispatch pattern, and the CPU
//! parity-test contract that every subsequent kernel port follows.
//!
//! # Semantics
//!
//! Given:
//!
//! - `weights: Tensor` — shape `[vocab_size, hidden]`, dtype `F32 | BF16 | F16`.
//! - `token_ids: Tensor` — shape `[N]` (1-D) or `[batch, seq_len]` (2-D),
//!   dtype `I64 | U32`.
//!
//! Produces:
//!
//! - `out: Tensor` — shape `[..token_ids.shape, hidden]`, same dtype
//!   as `weights`.
//!
//! Migration target: `candle_core::Tensor::index_select(...)` and
//! `candle_nn::Embedding`. Phase 0.1's audit grouped these under
//! "shape ops" — the gather here is the lossless replacement.
//!
//! # Why this is forward-only today
//!
//! Embedding's backward path is the `atomic-bwd` band from Phase 0.4
//! (`scatter_add` of the upstream gradient into a `[vocab_size, hidden]`
//! grad buffer keyed on `token_ids`). That lands when `kiln-autograd`
//! is up — the `bwd()` method here returns `None` today and is wired
//! in a follow-up.

use crate::{
    bail, dispatch2, BackwardOp, CpuStorage, DType, Determinism, DeviceOp2, Error, Layout, Result,
    Storage, Tensor, TensorId,
};
use std::sync::Arc;

/// Token-id gather along axis 0 of `weights`.
///
/// Stateless — the op is constructed once and reused. Phase 0.2's
/// shape proposal allows configuring per-op state (e.g., a Marlin
/// packing handle); embedding has none.
#[derive(Debug, Default, Clone, Copy)]
pub struct EmbeddingOp;

impl DeviceOp2 for EmbeddingOp {
    fn name(&self) -> &'static str {
        "embedding"
    }

    fn determinism(&self) -> Determinism {
        // Forward gather is order-stable; backward is atomic-bwd
        // tolerance-bounded. Forward only here -> Constructive.
        Determinism::Constructive
    }

    fn cpu_fwd(&self, weights: &Tensor, token_ids: &Tensor) -> Result<Option<Tensor>> {
        validate_inputs(weights, token_ids)?;

        let vocab_size = weights.shape()[0];
        let hidden = weights.shape()[1];
        let dtype = weights.dtype();
        let per = dtype.size_in_bytes();
        if per == 0 || dtype.is_packed() {
            bail!(
                "EmbeddingOp::cpu_fwd: packed dtype {dtype} for weights is not supported \
                 (embedding gather requires fixed per-element byte size)"
            );
        }

        // Output shape = token_ids.shape ++ [hidden].
        let mut out_shape: Vec<usize> = token_ids.shape().to_vec();
        out_shape.push(hidden);
        let n_out_elements: usize = token_ids.element_count();
        let row_bytes = hidden * per;

        // Pull token ids from CPU storage as u64 indices regardless of
        // I64/U32 dtype.
        let ids = read_token_ids_cpu(token_ids)?;

        // Pull weight bytes.
        let w_cpu = weights
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("EmbeddingOp::cpu_fwd: weights storage must be CpuStorage"))?;
        let w_bytes = w_cpu.as_bytes();
        if w_bytes.len() < vocab_size * row_bytes {
            bail!(
                "EmbeddingOp::cpu_fwd: weights byte len {} < vocab_size * hidden * per = {}",
                w_bytes.len(),
                vocab_size * row_bytes
            );
        }

        let mut out_bytes = vec![0u8; n_out_elements * row_bytes];
        for (i, &id) in ids.iter().enumerate() {
            if id as usize >= vocab_size {
                bail!(
                    "EmbeddingOp::cpu_fwd: token id {id} out of range (vocab_size={vocab_size}) at position {i}"
                );
            }
            let src = (id as usize) * row_bytes;
            let dst = i * row_bytes;
            out_bytes[dst..dst + row_bytes].copy_from_slice(&w_bytes[src..src + row_bytes]);
        }

        let cpu = CpuStorage::from_bytes(dtype, out_bytes)?;
        let storage: Storage = Arc::new(cpu);
        let tensor = Tensor::from_parts(storage, Layout::contiguous(out_shape), TensorId::next())?;
        Ok(Some(tensor))
    }

    // cuda_fwd / metal_fwd / vulkan_fwd inherit the default Ok(None) ->
    // dispatcher falls through to cpu_fwd. Phase 2+ overrides each.

    fn bwd(&self) -> Option<Box<dyn BackwardOp>> {
        // See module doc — wired up under kiln-autograd in a follow-up.
        None
    }
}

/// Convenience: dispatch `EmbeddingOp` on this Tensor pair without
/// constructing the op handle manually.
pub fn embedding(weights: &Tensor, token_ids: &Tensor) -> Result<Tensor> {
    dispatch2(&EmbeddingOp, weights, token_ids)
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

fn validate_inputs(weights: &Tensor, token_ids: &Tensor) -> Result<()> {
    if weights.rank() != 2 {
        bail!(
            "EmbeddingOp: weights must be rank-2 [vocab_size, hidden], got shape {:?}",
            weights.shape()
        );
    }
    if token_ids.rank() == 0 {
        bail!("EmbeddingOp: token_ids must have rank ≥ 1");
    }
    match token_ids.dtype() {
        DType::I64 | DType::U32 => {}
        other => bail!("EmbeddingOp: token_ids dtype must be I64 or U32, got {other}"),
    }
    if !weights.is_contiguous() {
        bail!(
            "EmbeddingOp: weights must be contiguous (got strides={:?} start_offset={})",
            weights.strides(),
            weights.layout().start_offset()
        );
    }
    if !token_ids.is_contiguous() {
        bail!("EmbeddingOp: token_ids must be contiguous");
    }
    Ok(())
}

fn read_token_ids_cpu(token_ids: &Tensor) -> Result<Vec<u64>> {
    let cpu = token_ids
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("EmbeddingOp: token_ids storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let n = token_ids.element_count();
    let mut ids = Vec::with_capacity(n);
    match token_ids.dtype() {
        DType::I64 => {
            if bytes.len() < n * 8 {
                bail!(
                    "EmbeddingOp: token_ids buffer is {} bytes, need {} for I64 x {}",
                    bytes.len(),
                    n * 8,
                    n
                );
            }
            for i in 0..n {
                let raw = &bytes[i * 8..(i + 1) * 8];
                let v = i64::from_le_bytes(raw.try_into().unwrap());
                if v < 0 {
                    bail!("EmbeddingOp: negative token id {v} at position {i}");
                }
                ids.push(v as u64);
            }
        }
        DType::U32 => {
            if bytes.len() < n * 4 {
                bail!(
                    "EmbeddingOp: token_ids buffer is {} bytes, need {} for U32 x {}",
                    bytes.len(),
                    n * 4,
                    n
                );
            }
            for i in 0..n {
                let raw = &bytes[i * 4..(i + 1) * 4];
                let v = u32::from_le_bytes(raw.try_into().unwrap());
                ids.push(v as u64);
            }
        }
        other => unreachable!("validate_inputs already rejected {other}"),
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_weights_f32(vocab: usize, hidden: usize) -> Tensor {
        let values: Vec<f32> = (0..vocab * hidden).map(|i| i as f32).collect();
        Tensor::from_slice(&values, vec![vocab, hidden]).unwrap()
    }

    #[test]
    fn embedding_f32_basic_lookup() {
        // vocab=5, hidden=3 -> weights row 0 = [0,1,2], row 1 = [3,4,5], ...
        let w = build_weights_f32(5, 3);
        let ids = Tensor::from_slice(&[2i64, 0, 4, 1], vec![4]).unwrap();
        let out = embedding(&w, &ids).unwrap();

        assert_eq!(out.shape(), &[4, 3]);
        assert_eq!(out.dtype(), DType::F32);

        let cpu = out.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let bytes = cpu.as_bytes();
        let back: Vec<f32> = bytemuck::cast_slice::<u8, f32>(bytes).to_vec();
        assert_eq!(
            back,
            vec![
                6.0, 7.0, 8.0, // row 2
                0.0, 1.0, 2.0, // row 0
                12.0, 13.0, 14.0, // row 4
                3.0, 4.0, 5.0, // row 1
            ]
        );
    }

    #[test]
    fn embedding_2d_token_ids_appends_hidden_axis() {
        let w = build_weights_f32(4, 2);
        // [[1, 0], [3, 2]] -> shape [2, 2, 2]
        let ids = Tensor::from_slice(&[1i64, 0, 3, 2], vec![2, 2]).unwrap();
        let out = embedding(&w, &ids).unwrap();
        assert_eq!(out.shape(), &[2, 2, 2]);
        let cpu = out.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let back: Vec<f32> = bytemuck::cast_slice::<u8, f32>(cpu.as_bytes()).to_vec();
        assert_eq!(back, vec![2.0, 3.0, 0.0, 1.0, 6.0, 7.0, 4.0, 5.0]);
    }

    #[test]
    fn embedding_u32_token_ids() {
        let w = build_weights_f32(3, 2);
        let ids = Tensor::from_slice(&[2u32, 0], vec![2]).unwrap();
        let out = embedding(&w, &ids).unwrap();
        let cpu = out.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let back: Vec<f32> = bytemuck::cast_slice::<u8, f32>(cpu.as_bytes()).to_vec();
        assert_eq!(back, vec![4.0, 5.0, 0.0, 1.0]);
    }

    #[test]
    fn embedding_rejects_out_of_range_id() {
        let w = build_weights_f32(3, 2);
        let ids = Tensor::from_slice(&[5i64], vec![1]).unwrap();
        let err = embedding(&w, &ids).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn embedding_rejects_negative_id() {
        let w = build_weights_f32(3, 2);
        let ids = Tensor::from_slice(&[-1i64], vec![1]).unwrap();
        let err = embedding(&w, &ids).unwrap_err();
        assert!(err.to_string().contains("negative token id"));
    }

    #[test]
    fn embedding_rejects_wrong_weight_rank() {
        let w = Tensor::zeros_cpu(vec![3, 2, 4], DType::F32);
        let ids = Tensor::from_slice(&[0i64], vec![1]).unwrap();
        let err = embedding(&w, &ids).unwrap_err();
        assert!(err.to_string().contains("rank-2"));
    }

    #[test]
    fn embedding_rejects_bad_token_dtype() {
        let w = build_weights_f32(3, 2);
        let ids = Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
        let err = embedding(&w, &ids).unwrap_err();
        assert!(err.to_string().contains("must be I64 or U32"));
    }

    #[test]
    fn embedding_op_metadata() {
        let op = EmbeddingOp;
        assert_eq!(op.name(), "embedding");
        assert!(op.determinism().is_constructive());
        assert!(op.bwd().is_none());
    }

    #[test]
    fn embedding_bf16_lookup() {
        // BF16 weight gather: byte-level copy should work regardless
        // of the float interpretation.
        let vocab = 3;
        let hidden = 4;
        let bf16_values: Vec<half::bf16> = (0..vocab * hidden)
            .map(|i| half::bf16::from_f32(i as f32))
            .collect();
        let w = Tensor::from_slice(&bf16_values, vec![vocab, hidden]).unwrap();
        let ids = Tensor::from_slice(&[2i64, 1], vec![2]).unwrap();
        let out = embedding(&w, &ids).unwrap();
        assert_eq!(out.dtype(), DType::BF16);
        assert_eq!(out.shape(), &[2, 4]);

        let cpu = out.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let bytes = cpu.as_bytes();
        // Verify the BF16 byte pattern matches: row 2 = bf16(8..12); row 1 = bf16(4..8).
        let row2_start = 0;
        for i in 0..hidden {
            let lo = bytes[row2_start + i * 2];
            let hi = bytes[row2_start + i * 2 + 1];
            let bf = half::bf16::from_le_bytes([lo, hi]);
            assert_eq!(bf, half::bf16::from_f32((8 + i) as f32));
        }
        let row1_start = hidden * 2;
        for i in 0..hidden {
            let lo = bytes[row1_start + i * 2];
            let hi = bytes[row1_start + i * 2 + 1];
            let bf = half::bf16::from_le_bytes([lo, hi]);
            assert_eq!(bf, half::bf16::from_f32((4 + i) as f32));
        }
    }
}
