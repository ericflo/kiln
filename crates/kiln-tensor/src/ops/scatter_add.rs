//! `scatter_add` — the embedding-backward / index_select-backward
//! primitive.
//!
//! Inverse of [`crate::ops::index_select`]: given source values,
//! axis, indices, and a target shape (with `dim_along_axis`
//! specified), produces a zero-filled output and atomically adds
//! each source value at its indexed position.
//!
//! # Semantics
//!
//! Given:
//! - `values: Tensor` — shape matching `index_select(target, axis,
//!   indices)`'s output shape
//! - `indices: Tensor` — same shape as in the forward `index_select`
//! - `axis: usize` — the axis the original gather was over
//! - `target_dim: usize` — `target.shape[axis]` (the original axis size)
//!
//! Returns:
//! - `out: Tensor` — shape = `values.shape` with the indices-shaped
//!   prefix collapsed back to a single `target_dim` along `axis`.
//!   Zero-initialized; then for each index position the corresponding
//!   slice of `values` is added.
//!
//! # Determinism
//!
//! `ToleranceBounded` (atomic-bwd band). Two indices that collide on
//! the same target position both add to the output; the addition
//! order under multi-threaded execution is non-deterministic across
//! backends. On CPU, the iteration order is fixed (index 0 then 1
//! then 2 ...) so this single-threaded reference is bit-stable; GPU
//! backends use `atomicAdd` and pick up the tolerance band.

use crate::{
    bail, dispatch2, BackwardOp, CpuStorage, DType, Determinism, DeviceOp2, Error, Layout, Result,
    Storage, Tensor, TensorId,
};
use std::sync::Arc;

/// Scatter-add op handle. Carries the gather axis + the original
/// target dim along that axis.
#[derive(Debug, Clone, Copy)]
pub struct ScatterAddOp {
    axis: usize,
    target_dim: usize,
}

impl ScatterAddOp {
    pub const fn new(axis: usize, target_dim: usize) -> Self {
        ScatterAddOp { axis, target_dim }
    }
    pub const fn axis(self) -> usize {
        self.axis
    }
    pub const fn target_dim(self) -> usize {
        self.target_dim
    }
}

impl DeviceOp2 for ScatterAddOp {
    fn name(&self) -> &'static str {
        "scatter_add"
    }

    fn determinism(&self) -> Determinism {
        // CPU is constructive (fixed iteration order); GPU
        // implementations pick up the atomic-bwd tolerance band per
        // bench-results/parity-tolerance.csv.
        Determinism::ToleranceBounded {
            dtype_band_key: "atomic-bwd",
        }
    }

    fn cpu_fwd(&self, values: &Tensor, indices: &Tensor) -> Result<Option<Tensor>> {
        validate(values, indices, self.axis, self.target_dim)?;

        let dtype = values.dtype();
        let per = dtype.size_in_bytes();
        if per == 0 || dtype.is_packed() {
            bail!(
                "ScatterAddOp: packed dtype {dtype} for values is not supported"
            );
        }

        // Source shape is: prefix(values.shape[..axis]) ++ indices.shape ++ suffix(values.shape[axis+indices.rank..]).
        // Target shape is: prefix ++ [target_dim] ++ suffix.
        let v_shape = values.shape();
        let i_shape = indices.shape();
        let outer: usize = v_shape[..self.axis].iter().product();
        let inner_start = self.axis + i_shape.len();
        let inner: usize = v_shape[inner_start..].iter().product();
        let n_indices: usize = i_shape.iter().product::<usize>().max(1);

        // Output shape: outer-axes ++ [target_dim] ++ inner-axes.
        let mut out_shape: Vec<usize> = v_shape[..self.axis].to_vec();
        out_shape.push(self.target_dim);
        out_shape.extend_from_slice(&v_shape[inner_start..]);

        let ids = read_indices(indices)?;
        let v_cpu = downcast_cpu(values, "values")?;
        let v_bytes = v_cpu.as_bytes();

        // Promote everything to F32 for the accumulation.
        let mut acc = vec![0.0_f32; outer * self.target_dim * inner];

        let block_size = inner; // per-index "row" in the source
        for o in 0..outer {
            for (idx_pos, &id) in ids.iter().enumerate() {
                let target_idx = id as usize;
                if target_idx >= self.target_dim {
                    bail!(
                        "ScatterAddOp: index {target_idx} out of range (target_dim={}) at position {idx_pos}",
                        self.target_dim
                    );
                }
                let src_off = (o * n_indices + idx_pos) * block_size;
                let dst_off = (o * self.target_dim + target_idx) * block_size;
                for i in 0..inner {
                    let v = read_one_f32(dtype, v_bytes, src_off + i, per);
                    acc[dst_off + i] += v;
                }
            }
        }

        // Cast back to input dtype.
        let mut out_bytes = vec![0u8; acc.len() * per];
        match dtype {
            DType::F32 => {
                for (i, v) in acc.iter().enumerate() {
                    out_bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
            DType::BF16 => {
                for (i, v) in acc.iter().enumerate() {
                    out_bytes[i * 2..i * 2 + 2]
                        .copy_from_slice(&half::bf16::from_f32(*v).to_le_bytes());
                }
            }
            DType::F16 => {
                for (i, v) in acc.iter().enumerate() {
                    out_bytes[i * 2..i * 2 + 2]
                        .copy_from_slice(&half::f16::from_f32(*v).to_le_bytes());
                }
            }
            _ => unreachable!(),
        }
        let cpu = CpuStorage::from_bytes(dtype, out_bytes)?;
        let storage: Storage = Arc::new(cpu);
        let out = Tensor::from_parts(storage, Layout::contiguous(out_shape), TensorId::next())?;
        Ok(Some(out))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, values: &Tensor, indices: &Tensor) -> Result<Option<Tensor>> {
        // CUDA substrate currently only handles axis=0 + 1-D U32 indices.
        // Other axes / multi-D indices / I64 indices fall through to CPU.
        if self.axis != 0 {
            return Ok(None);
        }
        if indices.rank() != 1 {
            return Ok(None);
        }
        if indices.dtype() != DType::U32 {
            return Ok(None);
        }
        if values.dtype().is_packed() {
            return Ok(None);
        }
        // Kernel supports F32 + BF16 only. Fall through to CPU for F16
        // until a F16 atomicAdd is wired in.
        if !matches!(values.dtype(), DType::F32 | DType::BF16) {
            return Ok(None);
        }
        if !values.is_contiguous() || !indices.is_contiguous() {
            return Ok(None);
        }

        validate(values, indices, self.axis, self.target_dim)?;

        // Pull the candle_device + device index off the values tensor so we
        // can allocate a same-device zero-filled output.
        let cuda_storage = values
            .storage()
            .as_any()
            .downcast_ref::<crate::CudaStorage>()
            .ok_or_else(|| {
                crate::Error::Msg(
                    "ScatterAddOp::cuda_fwd: values must be CUDA storage".to_string(),
                )
            })?;
        let candle_device = cuda_storage.candle_device().clone();
        let device_index = match values.device() {
            crate::Device::Cuda(i) => i,
            other => {
                return Err(crate::Error::Msg(format!(
                    "ScatterAddOp::cuda_fwd: expected CUDA device, got {other}"
                )));
            }
        };

        // Output shape: [target_dim, ...values.shape[1..]] (axis==0 +
        // indices.rank()==1 means the prefix is empty).
        let mut out_shape: Vec<usize> = Vec::with_capacity(values.rank());
        out_shape.push(self.target_dim);
        out_shape.extend_from_slice(&values.shape()[1..]);
        let n_out_elements: usize = out_shape.iter().product();

        let storage = crate::cuda_zeros(
            candle_device,
            device_index,
            values.dtype(),
            n_out_elements,
        )?;
        let out = Tensor::from_parts(storage, Layout::contiguous(out_shape), TensorId::next())?;

        // In-place atomic scatter-add into the zero-filled output.
        crate::cuda_scatter_add_dim0(&out, indices, values)?;
        Ok(Some(out))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(&self, values: &Tensor, indices: &Tensor) -> Result<Option<Tensor>> {
        // Gate on the same preconditions as cuda_fwd so future MSL
        // kernel work can drop in without changing the call site:
        //   - axis == 0 only (dim0 fast path)
        //   - 1-D U32 indices (matches cuda_fwd; I64 / multi-D fall
        //     through to CPU until kernels are extended)
        //   - F32 / BF16 values (F16 needs an atomic-add fp16
        //     path on Metal before it can route through this)
        //   - contiguous inputs
        if self.axis != 0 {
            return Ok(None);
        }
        if indices.rank() != 1 {
            return Ok(None);
        }
        if indices.dtype() != DType::U32 {
            return Ok(None);
        }
        if values.dtype().is_packed() {
            return Ok(None);
        }
        if !matches!(values.dtype(), DType::F32 | DType::BF16) {
            return Ok(None);
        }
        if !values.is_contiguous() || !indices.is_contiguous() {
            return Ok(None);
        }
        // TODO(#1082, phase 4 Metal): implement
        // `crate::metal_scatter_add_dim0(out, indices, values)`
        // analogous to `crate::cuda_scatter_add_dim0` above. Until
        // that kernel lands, fall through to the CPU path
        // (numerics-correct, performance-wrong).
        // Candidate implementations:
        //   1. Custom MSL kernel: one thread per (i, inner_index)
        //      pair; load `values[i, inner]`, atomic-add into
        //      `out[ids[i], inner]`. Metal supports `atomic_float`
        //      add on iOS17+/macOS14+ — for BF16, two-step
        //      (cast-to-F32, atomic-add, cast back) is the portable
        //      path until native BF16 atomics ship.
        //   2. Output buffer must be zero-allocated before the
        //      atomic-add pass (analog of `cuda_zeros` here); the
        //      Metal device-allocator path lives in
        //      `kiln-model::backend::metal`.
        //   3. MPS Graph has no public scatter-add primitive that
        //      matches our determinism contract — avoid that route.
        Ok(None)
    }

    #[cfg(feature = "vulkan")]
    fn vulkan_fwd(&self, values: &Tensor, indices: &Tensor) -> Result<Option<Tensor>> {
        // Gate on the same preconditions as cuda_fwd / metal_fwd:
        //   - axis == 0 only
        //   - 1-D U32 indices
        //   - F32 / BF16 values (F16 needs atomic-add fp16 support
        //     before it can route through this)
        //   - contiguous inputs
        if self.axis != 0 {
            return Ok(None);
        }
        if indices.rank() != 1 {
            return Ok(None);
        }
        if indices.dtype() != DType::U32 {
            return Ok(None);
        }
        if values.dtype().is_packed() {
            return Ok(None);
        }
        if !matches!(values.dtype(), DType::F32 | DType::BF16) {
            return Ok(None);
        }
        if !values.is_contiguous() || !indices.is_contiguous() {
            return Ok(None);
        }
        // TODO(#1082, phase 4 Vulkan): implement
        // `crate::vulkan_scatter_add_dim0(out, indices, values)`
        // analogous to `crate::cuda_scatter_add_dim0` above. Until
        // that wrapper lands, fall through to the CPU path
        // (numerics-correct, performance-wrong).
        // Candidate implementations:
        //   1. SPIR-V compute shader: one invocation per (i,
        //      inner_index) pair; `atomicAdd` into
        //      `out[ids[i], inner]`. Requires the
        //      `VK_KHR_shader_atomic_float` extension for F32
        //      atomics; BF16 needs a cast-to-F32 / atomic-add /
        //      cast-back wrapper (analog of the Metal note above).
        //   2. Output buffer must be zero-allocated before the
        //      atomic-add pass; `VkTensor::zeros` (or equivalent) in
        //      `vk_tensor.rs` is the natural primitive.
        //   3. Dtype matrix gap: `VkDType` exposes F32 / BF16 today;
        //      F16 / I64-index paths need either widening `VkDType`
        //      or cast wrappers at the dispatch boundary.
        Ok(None)
    }

    fn bwd(&self) -> Option<Box<dyn BackwardOp>> {
        // scatter_add's bwd is index_select. Phase 6b wires this once
        // kiln-autograd's BackwardOp surface accepts the
        // dispatch-into-gather pattern.
        None
    }
}

/// Dispatch `ScatterAddOp` with the given axis + target_dim.
pub fn scatter_add(values: &Tensor, axis: usize, indices: &Tensor, target_dim: usize) -> Result<Tensor> {
    dispatch2(&ScatterAddOp::new(axis, target_dim), values, indices)
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

fn validate(values: &Tensor, indices: &Tensor, axis: usize, target_dim: usize) -> Result<()> {
    if values.rank() == 0 {
        bail!("ScatterAddOp: values must have rank ≥ 1");
    }
    if indices.rank() == 0 {
        bail!("ScatterAddOp: indices must have rank ≥ 1");
    }
    if !matches!(values.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("ScatterAddOp: values dtype must be F32/BF16/F16, got {}", values.dtype());
    }
    if !matches!(indices.dtype(), DType::I64 | DType::U32) {
        bail!("ScatterAddOp: indices dtype must be I64/U32, got {}", indices.dtype());
    }
    if axis >= values.rank() {
        bail!(
            "ScatterAddOp: axis {axis} out of bounds (values rank={})",
            values.rank()
        );
    }
    // Check that `values.shape[axis..axis + indices.rank()]` equals `indices.shape`.
    let i_shape = indices.shape();
    if axis + i_shape.len() > values.rank() {
        bail!(
            "ScatterAddOp: axis {axis} + indices.rank()={} exceeds values rank={}",
            i_shape.len(),
            values.rank()
        );
    }
    let v_shape = values.shape();
    for (k, &dim) in i_shape.iter().enumerate() {
        if v_shape[axis + k] != dim {
            bail!(
                "ScatterAddOp: values.shape[{}..{}]={:?} must match indices.shape={:?}",
                axis,
                axis + i_shape.len(),
                &v_shape[axis..axis + i_shape.len()],
                i_shape
            );
        }
    }
    if target_dim == 0 {
        bail!("ScatterAddOp: target_dim=0 — output would be empty");
    }
    if !values.is_contiguous() || !indices.is_contiguous() {
        bail!("ScatterAddOp: both inputs must be contiguous");
    }
    Ok(())
}

fn downcast_cpu<'a>(t: &'a Tensor, label: &str) -> Result<&'a CpuStorage> {
    t.storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::Msg(format!("ScatterAddOp: {label} storage must be CpuStorage")))
}

fn read_indices(t: &Tensor) -> Result<Vec<u64>> {
    let cpu = downcast_cpu(t, "indices")?;
    let bytes = cpu.as_bytes();
    let n = t.element_count();
    let mut out = Vec::with_capacity(n);
    match t.dtype() {
        DType::I64 => {
            for i in 0..n {
                let v = i64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
                if v < 0 {
                    bail!("ScatterAddOp: negative index {v} at position {i}");
                }
                out.push(v as u64);
            }
        }
        DType::U32 => {
            for i in 0..n {
                let v = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
                out.push(v as u64);
            }
        }
        _ => unreachable!(),
    }
    Ok(out)
}

fn read_one_f32(dtype: DType, bytes: &[u8], i: usize, per: usize) -> f32 {
    let off = i * per;
    match dtype {
        DType::F32 => f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()),
        DType::BF16 => {
            half::bf16::from_le_bytes(bytes[off..off + 2].try_into().unwrap()).to_f32()
        }
        DType::F16 => {
            half::f16::from_le_bytes(bytes[off..off + 2].try_into().unwrap()).to_f32()
        }
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        bytemuck::cast_slice::<u8, f32>(cpu.as_bytes()).to_vec()
    }

    #[test]
    fn scatter_basic_axis_0() {
        // values [10, 20, 30] at indices [1, 0, 2] target_dim=3
        // -> output [20, 10, 30]
        let values = Tensor::from_slice(&[10.0f32, 20.0, 30.0], vec![3]).unwrap();
        let indices = Tensor::from_slice(&[1i64, 0, 2], vec![3]).unwrap();
        let out = scatter_add(&values, 0, &indices, 3).unwrap();
        assert_eq!(out.shape(), &[3]);
        assert_eq!(read_f32(&out), vec![20.0, 10.0, 30.0]);
    }

    #[test]
    fn scatter_accumulates_on_collision() {
        // values [10, 20, 30] all at index 0 -> output [60, 0]
        let values = Tensor::from_slice(&[10.0f32, 20.0, 30.0], vec![3]).unwrap();
        let indices = Tensor::from_slice(&[0i64, 0, 0], vec![3]).unwrap();
        let out = scatter_add(&values, 0, &indices, 2).unwrap();
        assert_eq!(out.shape(), &[2]);
        assert_eq!(read_f32(&out), vec![60.0, 0.0]);
    }

    #[test]
    fn scatter_2d_inner_dim_preserved() {
        // values shape [3, 2] at indices [1, 0, 2] target_dim=3 axis=0
        // -> output shape [3, 2], rearranged by index.
        let values = Tensor::from_slice(
            &[10.0f32, 11.0, 20.0, 21.0, 30.0, 31.0],
            vec![3, 2],
        )
        .unwrap();
        let indices = Tensor::from_slice(&[1i64, 0, 2], vec![3]).unwrap();
        let out = scatter_add(&values, 0, &indices, 3).unwrap();
        assert_eq!(out.shape(), &[3, 2]);
        assert_eq!(
            read_f32(&out),
            vec![20.0, 21.0, 10.0, 11.0, 30.0, 31.0]
        );
    }

    #[test]
    fn scatter_with_outer_axis_independent() {
        // values shape [2, 3] axis=1; outer dim = 2 (two independent
        // sub-batches), indices [1, 0, 2] target_dim=3.
        // Batch 0: values [1, 2, 3] → output [2, 1, 3]
        // Batch 1: values [10, 20, 30] → output [20, 10, 30]
        let values = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0],
            vec![2, 3],
        )
        .unwrap();
        let indices = Tensor::from_slice(&[1i64, 0, 2], vec![3]).unwrap();
        let out = scatter_add(&values, 1, &indices, 3).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(
            read_f32(&out),
            vec![2.0, 1.0, 3.0, 20.0, 10.0, 30.0]
        );
    }

    #[test]
    fn round_trip_with_index_select_is_identity_on_no_collision() {
        // forward: pick indices [2, 0, 1] from a 3-row tensor;
        // backward: scatter_add at those indices should recover the
        // ORIGINAL row ordering up to permutation.
        let original =
            Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0], vec![3, 2]).unwrap();
        let indices = Tensor::from_slice(&[2i64, 0, 1], vec![3]).unwrap();
        let gathered = crate::ops::index_select(&original, 0, &indices).unwrap();
        let scattered = scatter_add(&gathered, 0, &indices, 3).unwrap();
        assert_eq!(scattered.shape(), original.shape());
        // Reconstructs original row-by-row.
        assert_eq!(read_f32(&scattered), read_f32(&original));
    }

    #[test]
    fn out_of_range_index_errors() {
        let values = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let indices = Tensor::from_slice(&[5i64], vec![1]).unwrap();
        let e = scatter_add(&values, 0, &indices, 2).unwrap_err();
        assert!(e.to_string().contains("out of range"));
    }

    #[test]
    fn target_dim_zero_errors() {
        let values = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let indices = Tensor::from_slice(&[0i64], vec![1]).unwrap();
        let e = scatter_add(&values, 0, &indices, 0).unwrap_err();
        assert!(e.to_string().contains("target_dim=0"));
    }

    #[test]
    fn shape_mismatch_errors() {
        // values [3, 2]; indices [2] but axis=0 — values.shape[0..1]=[3]
        // doesn't match indices.shape=[2].
        let values =
            Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let indices = Tensor::from_slice(&[0i64, 1], vec![2]).unwrap();
        let e = scatter_add(&values, 0, &indices, 4).unwrap_err();
        assert!(e.to_string().contains("must match"));
    }

    #[test]
    fn op_metadata() {
        let op = ScatterAddOp::new(0, 10);
        assert_eq!(op.name(), "scatter_add");
        match op.determinism() {
            Determinism::ToleranceBounded { dtype_band_key } => {
                assert_eq!(dtype_band_key, "atomic-bwd");
            }
            _ => panic!("scatter_add should be tolerance-bounded"),
        }
        assert_eq!(op.axis(), 0);
        assert_eq!(op.target_dim(), 10);
    }
}
