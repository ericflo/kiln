//! `broadcast_to` — materialize broadcasting of size-1 axes.
//!
//! Given an input shape and a target shape of the same rank, each
//! axis must either match or be the expansion of a size-1 input
//! axis to the target axis size. The output is a fresh contiguous
//! Tensor with the target shape; the broadcast axes are replicated.
//!
//! # Why a forward op
//!
//! kiln-tensor's per-op surface (matmul, mul, add, etc.) requires
//! **contiguous** inputs. Layout-level broadcasting via zero-stride
//! cannot pass through those op signatures. A materializing
//! `broadcast_to` lets backward ops compose cleanly via the public
//! op surface instead of dropping to byte-level loops.
//!
//! # Determinism
//!
//! `Constructive`. Fixed iteration order; the only operation per
//! element is a byte copy from the source slot.
//!
//! # Dispatch
//!
//! Wrapped as a [`DeviceOp1`] so CUDA tensors route through
//! `cuda_index_select_dim0` on a flattened view of the input. The
//! gather indices are a flat-output-index → flat-input-index map
//! computed on the host once per call.

use std::sync::Arc;

use crate::{
    bail, dispatch1, BackwardOp, CpuStorage, Determinism, DeviceOp1, Error, Layout, Result,
    Storage, Tensor, TensorId,
};

/// Broadcast op — materializes broadcasting of size-1 axes.
#[derive(Debug, Clone)]
pub struct BroadcastOp {
    target_shape: Vec<usize>,
}

impl BroadcastOp {
    pub fn new(target_shape: &[usize]) -> Self {
        BroadcastOp {
            target_shape: target_shape.to_vec(),
        }
    }

    pub fn target_shape(&self) -> &[usize] {
        &self.target_shape
    }

    /// Compute the per-axis broadcast factor for `in_shape` →
    /// `target_shape`, validating that each axis is either equal
    /// or the input axis is size-1.
    fn broadcast_factors(&self, in_shape: &[usize]) -> Result<Vec<usize>> {
        if in_shape.len() != self.target_shape.len() {
            bail!(
                "broadcast_to: rank mismatch: input {:?} vs target {:?}",
                in_shape,
                self.target_shape
            );
        }
        let mut bf = vec![1usize; in_shape.len()];
        for (axis, (&in_d, &out_d)) in
            in_shape.iter().zip(self.target_shape.iter()).enumerate()
        {
            if in_d == out_d {
                bf[axis] = 1;
            } else if in_d == 1 {
                bf[axis] = out_d;
            } else {
                bail!(
                    "broadcast_to: cannot broadcast axis {axis} from {in_d} to {out_d} (input shape {:?}, target {:?})",
                    in_shape,
                    self.target_shape
                );
            }
        }
        Ok(bf)
    }

    /// Compute the flat-output-index → flat-input-index gather map.
    ///
    /// Used by both `cpu_fwd` (for byte copies) and `cuda_fwd` (as
    /// indices into a flattened CUDA buffer). Length = target_total.
    fn gather_indices(&self, in_shape: &[usize]) -> Vec<u32> {
        let rank = in_shape.len();
        let target_total: usize = self.target_shape.iter().product();
        let mut in_strides = vec![1usize; rank];
        for k in (0..rank.saturating_sub(1)).rev() {
            in_strides[k] = in_strides[k + 1] * in_shape[k + 1];
        }
        let mut out_strides = vec![1usize; rank];
        for k in (0..rank.saturating_sub(1)).rev() {
            out_strides[k] = out_strides[k + 1] * self.target_shape[k + 1];
        }
        let mut indices = Vec::with_capacity(target_total);
        for flat_out in 0..target_total {
            let mut rem = flat_out;
            let mut in_offset = 0usize;
            for k in 0..rank {
                let idx_out = rem / out_strides[k];
                rem %= out_strides[k];
                let idx_in = if in_shape[k] == 1 { 0 } else { idx_out };
                in_offset += idx_in * in_strides[k];
            }
            indices.push(in_offset as u32);
        }
        indices
    }
}

impl DeviceOp1 for BroadcastOp {
    fn name(&self) -> &'static str {
        "broadcast_to"
    }

    fn determinism(&self) -> Determinism {
        Determinism::Constructive
    }

    fn cpu_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        let in_shape = x.shape();
        let _bf = self.broadcast_factors(in_shape)?;

        if !x.is_contiguous() {
            bail!("broadcast_to: input must be contiguous");
        }
        let dtype = x.dtype();
        if dtype.is_packed() {
            bail!("broadcast_to: packed dtype {dtype} not supported");
        }
        let per = dtype.size_in_bytes();
        let in_cpu = x
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("broadcast_to: storage must be CpuStorage"))?;
        let in_bytes = in_cpu.as_bytes();

        let target_total: usize = self.target_shape.iter().product();
        let mut out_bytes = vec![0u8; target_total * per];

        // Walk every output index and copy the corresponding input element.
        // For each output index (i0, i1, …, i_{R-1}) compute the source
        // index (i0/bf0, i1/bf1, …) where bf_k = broadcast_factor[k].
        // The strides for the input are derived from `in_shape`.
        let rank = in_shape.len();
        let mut in_strides = vec![1usize; rank];
        for k in (0..rank.saturating_sub(1)).rev() {
            in_strides[k] = in_strides[k + 1] * in_shape[k + 1];
        }
        let mut out_strides = vec![1usize; rank];
        for k in (0..rank.saturating_sub(1)).rev() {
            out_strides[k] = out_strides[k + 1] * self.target_shape[k + 1];
        }
        for flat_out in 0..target_total {
            // Decompose flat_out into per-axis indices.
            let mut rem = flat_out;
            let mut in_offset = 0usize;
            for k in 0..rank {
                let idx_out = rem / out_strides[k];
                rem %= out_strides[k];
                let idx_in = if in_shape[k] == 1 { 0 } else { idx_out };
                in_offset += idx_in * in_strides[k];
            }
            let src = in_offset * per;
            let dst = flat_out * per;
            out_bytes[dst..dst + per].copy_from_slice(&in_bytes[src..src + per]);
        }
        let cpu = CpuStorage::from_bytes(dtype, out_bytes)?;
        let storage: Storage = Arc::new(cpu);
        Ok(Some(Tensor::from_parts(
            storage,
            Layout::contiguous(self.target_shape.clone()),
            TensorId::next(),
        )?))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        let in_shape = x.shape();
        let _bf = self.broadcast_factors(in_shape)?;
        let dtype = x.dtype();
        if dtype.is_packed() {
            return Ok(None);
        }
        if !x.is_contiguous() {
            return Ok(None);
        }

        // Fast path: shapes already match — no data movement.
        if in_shape == self.target_shape.as_slice() {
            return Ok(Some(x.reshape(x.shape().to_vec())?));
        }

        let target_total: usize = self.target_shape.iter().product();
        if target_total == 0 {
            // Empty output: build a zero-element tensor of the right shape.
            // Use reshape via cuda_contiguous? Simpler — fall through to CPU.
            return Ok(None);
        }

        let in_total: usize = in_shape.iter().product();

        // Resolve the candle CudaDevice from `x`'s storage so indices
        // land on the matching device.
        let x_storage = x
            .storage()
            .as_any()
            .downcast_ref::<crate::CudaStorage>()
            .ok_or_else(|| Error::from_str("broadcast_to::cuda_fwd: x must be CUDA storage"))?;
        let candle_device = x_storage.candle_device().clone();
        let device_index = match x.device() {
            crate::Device::Cuda(i) => i,
            _ => return Ok(None),
        };

        // Flatten the input to 1D so we can use cuda_index_select_dim0.
        let x_flat = x.reshape(vec![in_total])?;

        // Build the gather indices on CPU, then ship to CUDA.
        let indices_host = self.gather_indices(in_shape);
        let indices_cpu = Tensor::from_slice(&indices_host, vec![target_total])?;
        let indices_cuda =
            crate::host_to_cuda_copy(&indices_cpu, candle_device, device_index)?;

        let gathered = crate::cuda_index_select_dim0(&x_flat, &indices_cuda)?;

        // Reshape the 1D gathered output to the target shape.
        Ok(Some(gathered.reshape(self.target_shape.clone())?))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        // Gate on the same preconditions as cuda_fwd so future MSL
        // kernel work can drop in without changing the call site:
        //   - non-packed dtype
        //   - contiguous input
        //   - non-empty target shape
        //   - valid broadcast factors (delegated to broadcast_factors)
        let in_shape = x.shape();
        let _bf = self.broadcast_factors(in_shape)?;
        if x.dtype().is_packed() {
            return Ok(None);
        }
        if !x.is_contiguous() {
            return Ok(None);
        }
        if in_shape == self.target_shape.as_slice() {
            return Ok(Some(x.reshape(x.shape().to_vec())?));
        }
        let target_total: usize = self.target_shape.iter().product();
        if target_total == 0 {
            return Ok(None);
        }
        // TODO(#1082, phase 4 Metal): implement
        // `crate::metal_broadcast_to(x, target_shape)` analogous to
        // the CUDA gather-via-index-select strategy above. Until that
        // kernel lands, fall through to the CPU path
        // (numerics-correct, performance-wrong).
        // Candidate implementations:
        //   1. Custom MSL kernel: grid over `target_total`; each
        //      thread computes its source linear index via the same
        //      broadcast-factor math as the CPU path, then copies
        //      `block_bytes`. Avoids the index-buffer round-trip
        //      that the CUDA gather-emulation pays.
        //   2. Reuse `metal_index_select_dim0` once it lands —
        //      mirrors the CUDA approach by building a CPU index
        //      buffer, uploading, and gathering. Higher latency but
        //      simpler to wire if a gather kernel already exists.
        //   3. MPS Graph: `broadcast(_:shape:)` for one-shot
        //      dispatch. Cheap to wire but heavier per-call overhead.
        Ok(None)
    }

    #[cfg(feature = "vulkan")]
    fn vulkan_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        // Gate on the same preconditions as cuda_fwd / metal_fwd:
        //   - non-packed dtype
        //   - contiguous input
        //   - non-empty target shape
        //   - valid broadcast factors (delegated to broadcast_factors)
        let in_shape = x.shape();
        let _bf = self.broadcast_factors(in_shape)?;
        if x.dtype().is_packed() {
            return Ok(None);
        }
        if !x.is_contiguous() {
            return Ok(None);
        }
        if in_shape == self.target_shape.as_slice() {
            return Ok(Some(x.reshape(x.shape().to_vec())?));
        }
        let target_total: usize = self.target_shape.iter().product();
        if target_total == 0 {
            return Ok(None);
        }
        // TODO(#1082, phase 4 Vulkan): implement
        // `crate::vulkan_broadcast_to(x, target_shape)` analogous to
        // the CUDA gather-via-index-select strategy above. Until
        // that wrapper lands, fall through to the CPU path
        // (numerics-correct, performance-wrong).
        // Candidate implementations:
        //   1. SPIR-V compute shader: grid over `target_total`;
        //      each invocation computes its source linear index via
        //      broadcast-factor math and copies `block_bytes`. The
        //      shape metadata can be pushed via push-constants or a
        //      small uniform buffer.
        //   2. Reuse `vulkan_index_select_dim0` once it lands —
        //      build a CPU index buffer, upload, gather. Mirrors
        //      the current CUDA strategy.
        //   3. Dtype matrix gap: `VkDType` exposes F32 / BF16 today;
        //      other dtypes need either widening or a cast wrapper
        //      at the dispatch boundary.
        Ok(None)
    }

    fn bwd(&self) -> Option<Box<dyn BackwardOp>> {
        None
    }
}

/// Broadcast `x` to `target_shape`. Each axis of `target_shape`
/// must either equal the corresponding `x` axis or be a multiple
/// of a size-1 input axis.
pub fn broadcast_to(x: &Tensor, target_shape: &[usize]) -> Result<Tensor> {
    dispatch1(&BroadcastOp::new(target_shape), x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn broadcast_rank1_size1_to_3() {
        let x = Tensor::from_slice(&[5.0f32], vec![1]).unwrap();
        let y = broadcast_to(&x, &[3]).unwrap();
        assert_eq!(y.shape(), &[3]);
        assert_eq!(read_f32(&y), vec![5.0, 5.0, 5.0]);
    }

    #[test]
    fn broadcast_rank2_one_axis_only() {
        // [[1, 2, 3]] [1, 3] → [4, 3] = [[1,2,3], [1,2,3], [1,2,3], [1,2,3]]
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let y = broadcast_to(&x, &[4, 3]).unwrap();
        assert_eq!(y.shape(), &[4, 3]);
        assert_eq!(
            read_f32(&y),
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn broadcast_rank2_other_axis() {
        // [[1], [2]] [2, 1] → [2, 3] = [[1,1,1], [2,2,2]]
        let x = Tensor::from_slice(&[1.0f32, 2.0], vec![2, 1]).unwrap();
        let y = broadcast_to(&x, &[2, 3]).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn broadcast_both_axes() {
        // [[5]] [1, 1] → [3, 4]: every element 5.0
        let x = Tensor::from_slice(&[5.0f32], vec![1, 1]).unwrap();
        let y = broadcast_to(&x, &[3, 4]).unwrap();
        assert_eq!(read_f32(&y), vec![5.0; 12]);
    }

    #[test]
    fn broadcast_identity_when_shapes_match() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let y = broadcast_to(&x, &[3]).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn broadcast_rejects_rank_mismatch() {
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = broadcast_to(&x, &[1, 3]).unwrap_err();
        assert!(e.to_string().contains("rank mismatch"));
    }

    #[test]
    fn broadcast_rejects_incompatible_axis() {
        let x = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let e = broadcast_to(&x, &[3]).unwrap_err();
        assert!(e.to_string().contains("cannot broadcast"));
    }

    #[test]
    fn broadcast_bf16_path() {
        let bf: Vec<half::bf16> = [1.0f32, 2.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let x = Tensor::from_slice(&bf, vec![1, 2]).unwrap();
        let y = broadcast_to(&x, &[3, 2]).unwrap();
        assert_eq!(y.dtype(), DType::BF16);
        assert_eq!(y.shape(), &[3, 2]);
    }
}
