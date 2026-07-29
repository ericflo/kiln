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
        batch: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    fn kiln_rocm_bf16_matmul_f32_out_async(
        a: *const core::ffi::c_void,
        b: *const core::ffi::c_void,
        c: *mut core::ffi::c_void,
        m: i64,
        n: i64,
        k: i64,
        batch: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// Row-major contiguous BF16 matmul with FP32 accumulation and BF16 output.
///
/// This is not a general BLAS replacement; it exists for ROCm long-row
/// training projection slices where hipBLASLt has returned non-finite output.
pub fn rocm_bf16_matmul_bf16_out(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    rocm_bf16_matmul(a, b, DType::BF16, "rocm_bf16_matmul_bf16_out")
}

/// Row-major contiguous BF16 matmul with FP32 accumulation and output.
///
/// This is the portable fallback for small valid geometries that hipBLASLt
/// declines before it can establish a usable execution plan.
pub fn rocm_bf16_matmul_f32_out(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    rocm_bf16_matmul(a, b, DType::F32, "rocm_bf16_matmul_f32_out")
}

fn rocm_bf16_matmul(a: &Tensor, b: &Tensor, out_dtype: DType, op: &'static str) -> Result<Tensor> {
    let rank = a.rank();
    if rank < 2 || rank != b.rank() {
        return Err(Error::Msg(format!(
            "{op}: equal rank >= 2 inputs required, got a={} b={}",
            rank,
            b.rank()
        )));
    }
    if a.dtype() != DType::BF16 || b.dtype() != DType::BF16 {
        return Err(Error::Msg(format!(
            "{op}: BF16 inputs required, got a={} b={}",
            a.dtype(),
            b.dtype()
        )));
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(Error::Msg(format!("{op}: contiguous inputs required")));
    }
    let a_shape = a.shape();
    let b_shape = b.shape();
    for axis in 0..rank - 2 {
        if a_shape[axis] != b_shape[axis] {
            return Err(Error::Msg(format!(
                "{op}: batch axis {axis} mismatch: a={} b={}",
                a_shape[axis], b_shape[axis]
            )));
        }
    }
    let m = a_shape[rank - 2];
    let k = a_shape[rank - 1];
    if b_shape[rank - 2] != k {
        return Err(Error::Msg(format!(
            "{op}: contraction mismatch a.K={k} b.K={}",
            b_shape[rank - 2]
        )));
    }
    let n = b_shape[rank - 1];
    let batch = a_shape[..rank - 2].iter().product::<usize>().max(1);
    if batch > 65_535 {
        return Err(Error::Msg(format!(
            "{op}: logical batch {batch} exceeds the fallback kernel limit"
        )));
    }
    let mut out_shape = a_shape[..rank - 2].to_vec();
    out_shape.push(m);
    out_shape.push(n);

    let a_storage = a
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg(format!("{op}: a must be ROCm storage")))?;
    let b_storage = b
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg(format!("{op}: b must be ROCm storage")))?;
    use crate::StorageBackend;
    if a_storage.device() != b_storage.device() {
        return Err(Error::Msg(format!(
            "{op}: device mismatch a={} b={}",
            a_storage.device(),
            b_storage.device()
        )));
    }
    if !Arc::ptr_eq(&a_storage.context(), &b_storage.context()) {
        return Err(Error::Msg(format!(
            "{op}: both inputs must share one ROCm context"
        )));
    }
    let device_index = match a.device() {
        Device::Rocm(i) => i,
        other => {
            return Err(Error::Msg(format!("{op}: a must be on ROCm, got {other}")));
        }
    };
    if b.device() != Device::Rocm(device_index) {
        return Err(Error::Msg(format!(
            "{op}: b device {} != a device rocm:{device_index}",
            b.device()
        )));
    }

    let ctx = a_storage.context();
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, out_dtype, batch * m * n)?;
    let stream_submission = a_storage.rocm_stream_submission()?;
    let stream = stream_submission.raw_stream();
    let bpe = DType::BF16.size_in_bytes();
    let (a_base, _) = a_storage.device_ptr_raw();
    let (b_base, _) = b_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();
    let a_ptr = (a_base + (a.layout().start_offset() * bpe) as u64) as *const core::ffi::c_void;
    let b_ptr = (b_base + (b.layout().start_offset() * bpe) as u64) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        match out_dtype {
            DType::BF16 => kiln_rocm_bf16_matmul_bf16_out_async(
                a_ptr,
                b_ptr,
                out_ptr,
                m as i64,
                n as i64,
                k as i64,
                batch as i64,
                stream,
            ),
            DType::F32 => kiln_rocm_bf16_matmul_f32_out_async(
                a_ptr,
                b_ptr,
                out_ptr,
                m as i64,
                n as i64,
                k as i64,
                batch as i64,
                stream,
            ),
            _ => unreachable!("ROCm BF16 fallback output dtype is fixed"),
        }
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(Error::Msg(format!("{op}: FFI returned status {status}")));
    }
    stream_submission.complete();
    crate::rocm_storage::rocm_synchronize_context_same_stream_dependency_with_inputs(
        &ctx,
        &[a_storage, b_storage],
        crate::RocmSyncReason::MatmulOutput,
    )
    .map_err(|e| Error::Msg(format!("{op}: synchronize after kernel launch: {e:?}")))?;

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(storage_arc, Layout::contiguous(out_shape), TensorId::next())
        .map_err(|e| Error::Msg(format!("{op}: wrap: {e}")))
}
