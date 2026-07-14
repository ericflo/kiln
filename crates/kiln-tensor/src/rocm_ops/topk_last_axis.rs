//! ROCm wrapper for the `topk_last_axis` kernel (R.5b — sampling top-k).
use std::sync::Arc;

use crate::{DType, Device, Error, Layout, Result, RocmStorage, Tensor, TensorId};

// ROCm-side launcher for `topk_last_axis.cu`, compiled by
// `build.rs::build_rocm()` into `libkiln_tensor_rocm_ops.a` (same stable C ABI
// as the CUDA build). Signature is identical to the `cuda_storage.rs` extern
// `kiln_topk_last_axis_async`.
unsafe extern "C" {
    fn kiln_topk_last_axis_async(
        x: *const core::ffi::c_void,
        out_vals: *mut core::ffi::c_void,    // float[n_rows * k]
        out_indices: *mut core::ffi::c_void, // int64_t[n_rows * k]
        n_rows: i64,
        n_cols: i64,
        k: i32,
        dtype_tag: i32, // 0=F32, 1=BF16, 2=F16
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// On-device top-k over the trailing axis of a contiguous rank-1 ROCm tensor.
/// ROCm analog of `cuda_topk_last_axis`: keeps the full `[V]` logits row
/// resident on the device and transfers ONLY the `k` `(value, index)` pairs
/// back to host — instead of the `topk_via_host_sort` fallback's full-`[V]`
/// f32 D2H every sampled token. For discrete AMD GPUs this is the same PCIe win
/// the CUDA path gets; on UMA APUs the D2H is cheap either way.
///
/// Ranking matches `topk_via_host_sort` exactly: descending value, ties broken
/// by LOWEST index. F32 / BF16 / F16 inputs; comparison happens in F32. The
/// per-pass argmax reduction uses explicit 32-lane subgroups, so it is
/// wave32/64-correct.
pub fn rocm_topk_last_axis(x: &Tensor, k: usize) -> Result<(Vec<f32>, Vec<u32>)> {
    let in_dtype = x.dtype();
    let dtype_tag: i32 = match in_dtype {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
        other => {
            return Err(Error::Msg(format!(
                "rocm_topk_last_axis: unsupported dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(Error::Msg(
            "rocm_topk_last_axis: input must be contiguous".to_string(),
        ));
    }
    if x.rank() != 1 {
        return Err(Error::Msg(format!(
            "rocm_topk_last_axis: input must be rank 1, got rank {}",
            x.rank()
        )));
    }
    let vocab = x.dims()[0];
    if vocab == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let k = k.min(vocab);
    if k == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_topk_last_axis: input must be ROCm".to_string()))?;
    let ctx = x_storage.context();
    let device_index = match x.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("rocm_topk_last_axis: input must be ROCm"),
    };

    // Tiny on-device outputs: [k] F32 values + [k] I64 indices. The kernel
    // writes every slot; zeros_ctx is cheap defensiveness for the row-too-short
    // edge (k clamped to vocab above, so it never actually triggers).
    let vals_storage = RocmStorage::zeros_ctx(&ctx, device_index, DType::F32, k)?;
    let idx_storage = RocmStorage::zeros_ctx(&ctx, device_index, DType::I64, k)?;

    let raw_stream = x_storage.rocm_stream_raw()?;
    let (x_base, _) = x_storage.device_ptr_raw();
    let (vals_base, _) = vals_storage.device_ptr_raw();
    let (idx_base, _) = idx_storage.device_ptr_raw();
    let x_off = (x.layout().start_offset() * in_dtype.size_in_bytes()) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let vals_ptr = vals_base as *mut core::ffi::c_void;
    let idx_ptr = idx_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_topk_last_axis_async(
            x_ptr,
            vals_ptr,
            idx_ptr,
            1,
            vocab as i64,
            k as i32,
            dtype_tag,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_topk_last_axis: FFI returned status {status}"
        )));
    }

    // Small D2H: only k f32 values + k i64 indices cross the bus, via the
    // tensors' `to_vec1` (the same readback primitive the host-sort uses).
    let vals_t = Tensor::from_parts(
        Arc::new(vals_storage),
        Layout::contiguous(vec![k]),
        TensorId::next(),
    )?;
    let idx_t = Tensor::from_parts(
        Arc::new(idx_storage),
        Layout::contiguous(vec![k]),
        TensorId::next(),
    )?;
    let values = vals_t.to_vec1::<f32>()?;
    let indices: Vec<u32> = idx_t
        .to_vec1::<i64>()?
        .into_iter()
        .map(|v| v as u32)
        .collect();
    Ok((values, indices))
}
