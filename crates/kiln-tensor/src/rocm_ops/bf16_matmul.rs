//! ROCm BF16 matmul fallback for unstable hipBLASLt training shapes.

use std::sync::Arc;

use crate::{DType, Device, Error, Layout, Result, RocmStorage, Tensor, TensorId};

unsafe extern "C" {
    fn kiln_rocm_bf16_matmul_bf16_out_async(
        a: *const core::ffi::c_void,
        b: *const core::ffi::c_void,
        c: *mut core::ffi::c_void,
        m: i64,
        n: i64,
        k: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// Row-major contiguous BF16 matmul with FP32 accumulation and BF16 output.
///
/// This is not a general BLAS replacement; it exists for ROCm long-row
/// training projection slices where hipBLASLt has returned non-finite output.
pub fn rocm_bf16_matmul_bf16_out(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    const OP: &str = "rocm_bf16_matmul_bf16_out";
    if a.rank() != 2 || b.rank() != 2 {
        return Err(Error::Msg(format!(
            "{OP}: rank-2 inputs required, got a={} b={}",
            a.rank(),
            b.rank()
        )));
    }
    if a.dtype() != DType::BF16 || b.dtype() != DType::BF16 {
        return Err(Error::Msg(format!(
            "{OP}: BF16 inputs required, got a={} b={}",
            a.dtype(),
            b.dtype()
        )));
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(Error::Msg(format!("{OP}: contiguous inputs required")));
    }
    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[0];
    let k = a_shape[1];
    if b_shape[0] != k {
        return Err(Error::Msg(format!(
            "{OP}: contraction mismatch a.K={k} b.K={}",
            b_shape[0]
        )));
    }
    let n = b_shape[1];

    let a_storage = a
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg(format!("{OP}: a must be ROCm storage")))?;
    let b_storage = b
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg(format!("{OP}: b must be ROCm storage")))?;
    use crate::StorageBackend;
    if a_storage.device() != b_storage.device() {
        return Err(Error::Msg(format!(
            "{OP}: device mismatch a={} b={}",
            a_storage.device(),
            b_storage.device()
        )));
    }
    if !Arc::ptr_eq(&a_storage.context(), &b_storage.context()) {
        return Err(Error::Msg(format!(
            "{OP}: both inputs must share one ROCm context"
        )));
    }
    let device_index = match a.device() {
        Device::Rocm(i) => i,
        other => {
            return Err(Error::Msg(format!("{OP}: a must be on ROCm, got {other}")));
        }
    };
    if b.device() != Device::Rocm(device_index) {
        return Err(Error::Msg(format!(
            "{OP}: b device {} != a device rocm:{device_index}",
            b.device()
        )));
    }

    let ctx = a_storage.context();
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, DType::BF16, m * n)?;
    let stream = a_storage.rocm_stream_raw()?;
    let bpe = DType::BF16.size_in_bytes();
    let (a_base, _) = a_storage.device_ptr_raw();
    let (b_base, _) = b_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();
    let a_ptr = (a_base + (a.layout().start_offset() * bpe) as u64) as *const core::ffi::c_void;
    let b_ptr = (b_base + (b.layout().start_offset() * bpe) as u64) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_rocm_bf16_matmul_bf16_out_async(
            a_ptr, b_ptr, out_ptr, m as i64, n as i64, k as i64, stream,
        )
    };
    if status != 0 {
        return Err(Error::Msg(format!("{OP}: FFI returned status {status}")));
    }
    crate::rocm_storage::rocm_synchronize_context_same_stream_dependency_with_inputs(
        &ctx,
        &[a_storage, b_storage],
        crate::RocmSyncReason::MatmulOutput,
    )
    .map_err(|e| Error::Msg(format!("{OP}: synchronize after kernel launch: {e:?}")))?;

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(
        storage_arc,
        Layout::contiguous(vec![m, n]),
        TensorId::next(),
    )
    .map_err(|e| Error::Msg(format!("{OP}: wrap: {e}")))
}
