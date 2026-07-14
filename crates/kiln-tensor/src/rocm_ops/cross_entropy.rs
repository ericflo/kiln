//! ROCm wrappers for the cross_entropy kernel(s) (Phase R.5).
use std::sync::Arc;

use crate::{DType, Device, Error, Layout, Result, RocmStorage, Tensor, TensorId};

// Same stable C ABI as the CUDA build (`csrc/cross_entropy.cu`). The kernel
// does the row-wise log-sum-exp + target-logit subtraction into a per-row F32
// scratch buffer, then a single-block finalize sums and divides by batch.
// Row-level validation errors are reported via the device-side `row_err_i32`
// flag, which we read back after launch.
unsafe extern "C" {
    fn kiln_cross_entropy_loss_async(
        logits: *const core::ffi::c_void,
        targets: *const core::ffi::c_void,
        row_loss_f32: *mut core::ffi::c_void,
        row_err_i32: *mut core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_rows: i64,
        n_cols: i64,
        dtype_tag: i32,
        targets_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// Cross-entropy loss over per-row softmax of `logits` against `targets`.
/// ROCm analog of `cuda_cross_entropy_loss`, routing through the wave-size-fixed
/// `cross_entropy.cu` kernel (Phase R.5).
///
/// ```text
/// for each row b of logits [B, V]:
///     m_b = max_v logits[b, v]
///     log_sum_exp_b = m_b + log(sum_v exp(logits[b, v] - m_b))
///     loss_b = log_sum_exp_b - logits[b, targets[b]]
/// loss = mean_b loss_b
/// ```
///
/// `logits` is `[batch, vocab]` (F32 / BF16 / F16), `targets` is `[batch]`
/// (I64 or U32). Output is a rank-0 scalar at the logits dtype. F32
/// accumulation throughout. An out-of-range target or an all-`-inf` row is
/// surfaced as `Result::Err` via a device-side error flag read back after the
/// launch.
pub fn rocm_cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let dtype = logits.dtype();
    let dtype_tag: i32 = match dtype {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
        other => {
            return Err(Error::Msg(format!(
                "rocm_cross_entropy_loss: logits dtype must be F32/BF16/F16, got {other}"
            )));
        }
    };
    let targets_tag: i32 = match targets.dtype() {
        DType::I64 => 0,
        DType::U32 => 1,
        other => {
            return Err(Error::Msg(format!(
                "rocm_cross_entropy_loss: targets dtype must be I64/U32, got {other}"
            )));
        }
    };
    if logits.rank() != 2 {
        return Err(Error::Msg(format!(
            "rocm_cross_entropy_loss: logits must be rank-2 [batch, vocab], got shape {:?}",
            logits.shape()
        )));
    }
    if targets.rank() != 1 {
        return Err(Error::Msg(format!(
            "rocm_cross_entropy_loss: targets must be rank-1 [batch], got shape {:?}",
            targets.shape()
        )));
    }
    if !logits.is_contiguous() || !targets.is_contiguous() {
        return Err(Error::Msg(
            "rocm_cross_entropy_loss: both inputs must be contiguous".to_string(),
        ));
    }
    let shape = logits.shape();
    let batch = shape[0];
    let vocab = shape[1];
    if batch == 0 {
        return Err(Error::Msg(
            "rocm_cross_entropy_loss: batch dim is 0 — mean is undefined".to_string(),
        ));
    }
    if targets.shape()[0] != batch {
        return Err(Error::Msg(format!(
            "rocm_cross_entropy_loss: batch mismatch — logits has batch={batch}, targets has batch={}",
            targets.shape()[0]
        )));
    }

    let logits_storage = logits
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| {
            Error::Msg("rocm_cross_entropy_loss: logits must be ROCm storage".to_string())
        })?;
    let targets_storage = targets
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| {
            Error::Msg("rocm_cross_entropy_loss: targets must be ROCm storage".to_string())
        })?;

    let ctx = logits_storage.context();
    let device_index = match logits.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("RocmStorage tensor is always Rocm"),
    };

    // Per-row F32 scratch and a 1-element U32 error flag, both zero-init.
    let row_loss_storage = RocmStorage::zeros_ctx(&ctx, device_index, DType::F32, batch)?;
    let row_err_storage = RocmStorage::zeros_ctx(&ctx, device_index, DType::U32, 1)?;
    // Scalar output (1 element at the input dtype); the kernel overwrites it.
    let out_storage = RocmStorage::zeros_ctx(&ctx, device_index, dtype, 1)?;

    let stream_submission = logits_storage.rocm_stream_submission()?;
    let raw_stream = stream_submission.raw_stream();

    let (logits_base, _) = logits_storage.device_ptr_raw();
    let (targets_base, _) = targets_storage.device_ptr_raw();
    let (row_loss_base, _) = row_loss_storage.device_ptr_raw();
    let (row_err_base, _) = row_err_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();

    let logits_off = (logits.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let targets_off = (targets.layout().start_offset() * targets.dtype().size_in_bytes()) as u64;
    let logits_ptr = (logits_base + logits_off) as *const core::ffi::c_void;
    let targets_ptr = (targets_base + targets_off) as *const core::ffi::c_void;
    let row_loss_ptr = row_loss_base as *mut core::ffi::c_void;
    let row_err_ptr = row_err_base as *mut core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_cross_entropy_loss_async(
            logits_ptr,
            targets_ptr,
            row_loss_ptr,
            row_err_ptr,
            out_ptr,
            batch as i64,
            vocab as i64,
            dtype_tag,
            targets_tag,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(Error::Msg(format!(
            "rocm_cross_entropy_loss: FFI returned status {status}"
        )));
    }
    stream_submission.complete();

    // Read back the row-error flag to surface validation failures. This forces
    // a sync via the D2H copy but is the only way to surface a Result-level
    // error from on-device checks.
    let stream = crate::active_rocm_stream(&ctx);
    let err_host = stream.memcpy_dtoh(row_err_storage.slice()).map_err(|e| {
        Error::Msg(format!(
            "rocm_cross_entropy_loss: row_err D2H failed: {e:?}"
        ))
    })?;
    if err_host.len() < 4 {
        return Err(Error::Msg(format!(
            "rocm_cross_entropy_loss: row_err D2H returned {} bytes, expected 4",
            err_host.len()
        )));
    }
    let err_code = u32::from_le_bytes([err_host[0], err_host[1], err_host[2], err_host[3]]);
    match err_code {
        0 => {}
        1 => {
            return Err(Error::Msg(format!(
                "rocm_cross_entropy_loss: target out of range (vocab={vocab})"
            )));
        }
        2 => {
            return Err(Error::Msg(
                "rocm_cross_entropy_loss: row has no finite logits (all -inf?); loss is undefined"
                    .to_string(),
            ));
        }
        other => {
            return Err(Error::Msg(format!(
                "rocm_cross_entropy_loss: unknown row_err code {other}"
            )));
        }
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(
        storage_arc,
        Layout::contiguous(Vec::<usize>::new()),
        TensorId::next(),
    )
}
