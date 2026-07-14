//! ROCm wrappers for the concat kernel(s) (Phase R.5).
use std::sync::Arc;

use crate::{Device, Error, Layout, Result, RocmStorage, Tensor, TensorId};

// The ROCm-side `concat` launcher, compiled by `build.rs::build_rocm()` into
// `libkiln_tensor_rocm_ops.a` with the same stable C ABI as the CUDA build.
// Signature is copied verbatim from the `kiln_concat_async` decl in
// `cuda_storage.rs` (same symbol, same args).
unsafe extern "C" {
    fn kiln_concat_async(
        dst: *mut core::ffi::c_void,
        src_ptrs: *const *const core::ffi::c_void,
        t_axis_lens: *const i64,
        n_inputs: i32,
        outer: i64,
        inner_bytes: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    fn kiln_concat_axis0_contiguous_async(
        dst: *mut core::ffi::c_void,
        src_ptrs: *const *const core::ffi::c_void,
        t_axis_lens: *const i64,
        n_inputs: i32,
        inner_bytes: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// ROCm-side `concat(inputs, axis)` — concatenate `inputs` along `axis` into a
/// freshly-allocated contiguous output. ROCm analog of `cuda_concat`.
///
/// Mirrors the CPU reference in `crates/kiln-tensor/src/ops/concat.rs`
/// byte-for-byte: per-outer-slab, each input's axis-slab is copied into the
/// running output offset. The kernel `kiln_concat_async` performs all the
/// copies in a single launch.
///
/// Requirements:
/// - At least one input (and no more than 32).
/// - All inputs must be ROCm-backed and contiguous.
/// - All inputs must share dtype and the same shape except along `axis`.
/// - `axis < rank`, all input ranks equal.
/// - Packed dtypes are not supported.
pub fn rocm_concat(inputs: &[&Tensor], axis: usize) -> Result<Tensor> {
    if inputs.is_empty() {
        return Err(Error::Msg(
            "rocm_concat: at least one input required".to_string(),
        ));
    }
    if inputs.len() > 32 {
        return Err(Error::Msg(format!(
            "rocm_concat: too many inputs ({}); MAX_INPUTS=32",
            inputs.len()
        )));
    }

    let rank = inputs[0].rank();
    if axis >= rank {
        return Err(Error::Msg(format!(
            "rocm_concat: axis {axis} out of range for rank-{rank} inputs"
        )));
    }
    let dtype = inputs[0].dtype();
    if dtype.is_packed() {
        return Err(Error::Msg(format!(
            "rocm_concat: packed dtype {dtype} is not supported"
        )));
    }
    let bpe = dtype.size_in_bytes();
    if bpe == 0 {
        return Err(Error::Msg(format!("rocm_concat: zero-size dtype {dtype}")));
    }

    for (i, t) in inputs.iter().enumerate() {
        if t.rank() != rank {
            return Err(Error::Msg(format!(
                "rocm_concat: input {i} rank {} != input 0 rank {rank}",
                t.rank()
            )));
        }
        if t.dtype() != dtype {
            return Err(Error::Msg(format!(
                "rocm_concat: input {i} dtype {} != input 0 dtype {dtype}",
                t.dtype()
            )));
        }
        if t.device() != inputs[0].device() {
            return Err(Error::Msg(format!(
                "rocm_concat: input {i} device {} != input 0 device {}",
                t.device(),
                inputs[0].device()
            )));
        }
        if !t.is_contiguous() {
            return Err(Error::Msg(format!(
                "rocm_concat: input {i} must be contiguous"
            )));
        }
        for (d, (&a, &b)) in t.shape().iter().zip(inputs[0].shape()).enumerate() {
            if d != axis && a != b {
                return Err(Error::Msg(format!(
                    "rocm_concat: input {i} shape {:?} differs from input 0 shape {:?} along axis {d}",
                    t.shape(),
                    inputs[0].shape()
                )));
            }
        }
    }

    // Output shape: input 0's shape with axis dim replaced by sum.
    let mut out_shape = inputs[0].shape().to_vec();
    let axis_total: usize = inputs.iter().map(|t| t.shape()[axis]).sum();
    out_shape[axis] = axis_total;

    let outer: usize = out_shape[..axis].iter().product::<usize>().max(1);
    let inner: usize = out_shape[axis + 1..].iter().product::<usize>().max(1);
    let inner_bytes = (inner * bpe) as i64;

    // Pull device + context from first input.
    let first_storage = inputs[0]
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_concat: input 0 must be ROCm storage".to_string()))?;
    let ctx = first_storage.context();
    let device_index = match inputs[0].device() {
        Device::Rocm(i) => i,
        _ => unreachable!("rocm_concat: input 0 device must be Rocm"),
    };

    // Allocate destination — same device, same dtype, total elements. The
    // kernel fully overwrites every output byte (one element per thread), so
    // skip the zero-fill.
    let n_out_elements: usize = out_shape.iter().product();
    let dst_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, dtype, n_out_elements)?;
    let (dst_base, _) = dst_storage.device_ptr_raw();
    let storage_arc: crate::Storage = Arc::new(dst_storage);
    let out = Tensor::from_parts(
        Arc::clone(&storage_arc),
        Layout::contiguous(out_shape.clone()),
        TensorId::next(),
    )?;

    // Collect per-input source pointers (base + start_offset bytes) and
    // per-input axis lengths.
    let mut src_ptrs: Vec<*const core::ffi::c_void> = Vec::with_capacity(inputs.len());
    let mut t_axis_lens: Vec<i64> = Vec::with_capacity(inputs.len());
    let mut input_storages: Vec<&RocmStorage> = Vec::with_capacity(inputs.len());
    for t in inputs {
        let st = t
            .storage()
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Msg("rocm_concat: input must be ROCm storage".to_string()))?;
        if !Arc::ptr_eq(&ctx, &st.context()) {
            return Err(Error::Msg(
                "rocm_concat: every input must share one ROCm context".to_string(),
            ));
        }
        let (base, _) = st.device_ptr_raw();
        let off = (t.layout().start_offset() * bpe) as u64;
        src_ptrs.push((base + off) as *const core::ffi::c_void);
        t_axis_lens.push(t.shape()[axis] as i64);
        input_storages.push(st);
    }

    let dst_ptr = dst_base as *mut core::ffi::c_void;

    // Large sequence-tile cats in model training have `outer == 1`, e.g.
    // [1, tile, hidden] pieces concatenated along sequence. On gfx115x, the
    // one-shot ROCm concat paths have exposed intermittent long-context
    // corruption under allocator pressure. Assemble those large row-major cats
    // through the same contiguous-copy primitive as `slice_set`; it is exact
    // and has been the stable path for long-context device-to-device row
    // copies.
    let safe_row_assembly_enabled = std::env::var("KILN_DISABLE_ROCM_CONCAT_SAFE_ROW_ASSEMBLY")
        .ok()
        .map(|v| {
            !matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(true);
    if safe_row_assembly_enabled && outer == 1 && n_out_elements >= 1_000_000 {
        let out_rows = out
            .reshape(vec![axis_total, inner])
            .map_err(|e| Error::Msg(format!("rocm_concat: reshape dst rows: {e}")))?;
        let mut row_offset = 0usize;
        for t in inputs {
            crate::rocm_synchronize_tensor_same_stream_dependency(
                t,
                crate::RocmSyncReason::ConcatInput,
            )
            .map_err(|e| {
                Error::Msg(format!(
                    "rocm_concat: synchronize input before safe row assembly: {e}"
                ))
            })?;
            let rows = t.shape()[axis];
            let src_rows = t
                .reshape(vec![rows, inner])
                .map_err(|e| Error::Msg(format!("rocm_concat: reshape src rows: {e}")))?;
            out_rows
                .slice_set(&src_rows, 0usize, row_offset)
                .map_err(|e| {
                    Error::Msg(format!(
                        "rocm_concat: safe row assembly slice_set offset={row_offset}: {e}"
                    ))
                })?;
            row_offset += rows;
        }
        crate::rocm_storage::rocm_synchronize_context_same_stream_dependency_with_inputs(
            &ctx,
            &input_storages,
            crate::RocmSyncReason::ConcatOutput,
        )
        .map_err(|e| {
            Error::Msg(format!(
                "rocm_concat: synchronize after safe row assembly: {e:?}"
            ))
        })?;
        return Ok(out);
    }

    // The specialized vectorized row-copy kernel is kept as an opt-in
    // diagnostic path for benchmarking and triage.
    let axis0_row_copy_enabled = std::env::var("KILN_ROCM_CONCAT_AXIS0_ROW_COPY")
        .ok()
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false);
    if axis0_row_copy_enabled && outer == 1 && n_out_elements >= 1_000_000 {
        let stream_submission = first_storage.rocm_stream_submission()?;
        let raw_stream = stream_submission.raw_stream();
        let status = unsafe {
            kiln_concat_axis0_contiguous_async(
                dst_ptr,
                src_ptrs.as_ptr(),
                t_axis_lens.as_ptr(),
                inputs.len() as i32,
                inner_bytes,
                raw_stream,
            )
        };
        if status != 0 {
            stream_submission.quarantine();
            return Err(Error::Msg(format!(
                "rocm_concat: axis0 contiguous FFI returned status {status}"
            )));
        }
        stream_submission.complete();
        crate::rocm_storage::rocm_synchronize_context_same_stream_dependency_with_inputs(
            &ctx,
            &input_storages,
            crate::RocmSyncReason::ConcatOutput,
        )
        .map_err(|e| {
            Error::Msg(format!(
                "rocm_concat: synchronize after axis0 contiguous concat: {e:?}"
            ))
        })?;
        return Ok(out);
    }

    let stream_submission = first_storage.rocm_stream_submission()?;
    let raw_stream = stream_submission.raw_stream();
    let status = unsafe {
        kiln_concat_async(
            dst_ptr,
            src_ptrs.as_ptr(),
            t_axis_lens.as_ptr(),
            inputs.len() as i32,
            outer as i64,
            inner_bytes,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(Error::Msg(format!(
            "rocm_concat: FFI returned status {status}"
        )));
    }
    stream_submission.complete();
    crate::rocm_storage::rocm_synchronize_context_same_stream_dependency_with_inputs(
        &ctx,
        &input_storages,
        crate::RocmSyncReason::ConcatOutput,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "rocm_concat: synchronize after async kernel launch: {e:?}"
        ))
    })?;

    Ok(out)
}
