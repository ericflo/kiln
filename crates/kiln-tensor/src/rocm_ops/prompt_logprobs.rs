//! Compact ROCm prompt-logprob selection.

use std::sync::Arc;

use crate::{
    DType, Device, DevicePromptLogprobRow, Error, Layout, Result, RocmStorage, Tensor, TensorId,
};

unsafe extern "C" {
    fn kiln_prompt_logprobs_async(
        x: *const core::ffi::c_void,
        observed_ids: *const i64,
        out_row_max: *mut core::ffi::c_void,
        out_log_sum: *mut core::ffi::c_void,
        out_observed_logit: *mut core::ffi::c_void,
        out_observed_rank: *mut core::ffi::c_void,
        out_top_logits: *mut core::ffi::c_void,
        out_top_indices: *mut core::ffi::c_void,
        out_invalid_kind: *mut core::ffi::c_void,
        out_invalid_column: *mut core::ffi::c_void,
        out_invalid_value: *mut core::ffi::c_void,
        n_rows: i64,
        n_cols: i64,
        top_k: i32,
        dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// Select compact prompt-logprob rows without copying full vocabulary rows.
///
/// `logits` must be contiguous rank-2 ROCm F32/BF16/F16. Every input logit and
/// every derived F32 log-probability is checked for finiteness on device. The
/// only host transfer is O(rows * top_k): normalization scalars, observed
/// values/ranks, validation diagnostics, and selected `(logit, token_id)` pairs.
pub fn rocm_prompt_logprobs(
    logits: &Tensor,
    observed_token_ids: &[u32],
    top_k: usize,
) -> Result<Vec<DevicePromptLogprobRow>> {
    const NAME: &str = "rocm_prompt_logprobs";
    let dtype_tag = match logits.dtype() {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
        dtype => return Err(Error::Msg(format!("{NAME}: unsupported dtype {dtype}"))),
    };
    if !logits.is_contiguous() {
        return Err(Error::Msg(format!("{NAME}: logits must be contiguous")));
    }
    if logits.rank() != 2 {
        return Err(Error::Msg(format!(
            "{NAME}: logits must be rank 2, got rank {}",
            logits.rank()
        )));
    }
    let n_rows = logits.dims()[0];
    let n_cols = logits.dims()[1];
    if n_cols == 0 {
        return Err(Error::Msg(format!(
            "{NAME}: vocabulary width must be nonzero"
        )));
    }
    if observed_token_ids.len() != n_rows {
        return Err(Error::Msg(format!(
            "{NAME}: observed token count {} did not equal row count {n_rows}",
            observed_token_ids.len()
        )));
    }
    if top_k > n_cols {
        return Err(Error::Msg(format!(
            "{NAME}: top_k {top_k} exceeded vocabulary width {n_cols}"
        )));
    }
    let top_k_i32 = i32::try_from(top_k)
        .map_err(|_| Error::Msg(format!("{NAME}: top_k {top_k} did not fit i32")))?;
    for (row, &token_id) in observed_token_ids.iter().enumerate() {
        if token_id as usize >= n_cols {
            return Err(Error::Msg(format!(
                "{NAME}: row {row} observed token id {token_id} was outside vocabulary width {n_cols}"
            )));
        }
    }
    if n_rows == 0 {
        return Ok(Vec::new());
    }
    let n_rows_i64 = i64::try_from(n_rows)
        .map_err(|_| Error::Msg(format!("{NAME}: row count {n_rows} did not fit i64")))?;
    let n_cols_i64 = i64::try_from(n_cols)
        .map_err(|_| Error::Msg(format!("{NAME}: vocabulary width {n_cols} did not fit i64")))?;

    let storage = logits
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg(format!("{NAME}: logits must use ROCm storage")))?;
    let ctx = storage.context();
    let device_index = match logits.device() {
        Device::Rocm(index) => index,
        device => {
            return Err(Error::Msg(format!(
                "{NAME}: logits must be ROCm, got {device}"
            )));
        }
    };
    let observed_values = observed_token_ids
        .iter()
        .map(|&token_id| i64::from(token_id))
        .collect::<Vec<_>>();
    let observed_storage =
        RocmStorage::alloc_uninit_ctx(&ctx, device_index, DType::I64, observed_values.len())?;
    let observed = Tensor::from_parts(
        Arc::new(observed_storage),
        Layout::contiguous(vec![n_rows]),
        TensorId::next(),
    )?;
    crate::rocm_write_host_in_place(&observed, &observed_values)?;
    let observed_storage = observed
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg(format!("{NAME}: observed IDs must use ROCm storage")))?;

    let top_count = n_rows
        .checked_mul(top_k)
        .ok_or_else(|| Error::Msg(format!("{NAME}: top-k output length overflow")))?;
    let row_max_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, DType::F32, n_rows)?;
    let log_sum_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, DType::F32, n_rows)?;
    let observed_logit_storage =
        RocmStorage::alloc_uninit_ctx(&ctx, device_index, DType::F32, n_rows)?;
    let observed_rank_storage =
        RocmStorage::alloc_uninit_ctx(&ctx, device_index, DType::I64, n_rows)?;
    let top_logits_storage =
        RocmStorage::alloc_uninit_ctx(&ctx, device_index, DType::F32, top_count.max(1))?;
    let top_indices_storage =
        RocmStorage::alloc_uninit_ctx(&ctx, device_index, DType::I64, top_count.max(1))?;
    let invalid_kind_storage =
        RocmStorage::alloc_uninit_ctx(&ctx, device_index, DType::U32, n_rows)?;
    let invalid_column_storage =
        RocmStorage::alloc_uninit_ctx(&ctx, device_index, DType::I64, n_rows)?;
    let invalid_value_storage =
        RocmStorage::alloc_uninit_ctx(&ctx, device_index, DType::F32, n_rows)?;

    let submission = storage.rocm_stream_submission()?;
    let raw_stream = submission.raw_stream();
    let (x_base, _) = storage.device_ptr_raw();
    let (observed_base, _) = observed_storage.device_ptr_raw();
    let (row_max_base, _) = row_max_storage.device_ptr_raw();
    let (log_sum_base, _) = log_sum_storage.device_ptr_raw();
    let (observed_logit_base, _) = observed_logit_storage.device_ptr_raw();
    let (observed_rank_base, _) = observed_rank_storage.device_ptr_raw();
    let (top_logits_base, _) = top_logits_storage.device_ptr_raw();
    let (top_indices_base, _) = top_indices_storage.device_ptr_raw();
    let (invalid_kind_base, _) = invalid_kind_storage.device_ptr_raw();
    let (invalid_column_base, _) = invalid_column_storage.device_ptr_raw();
    let (invalid_value_base, _) = invalid_value_storage.device_ptr_raw();
    let x_offset = (logits.layout().start_offset() * logits.dtype().size_in_bytes()) as u64;

    let status = unsafe {
        kiln_prompt_logprobs_async(
            (x_base + x_offset) as *const core::ffi::c_void,
            observed_base as *const i64,
            row_max_base as *mut core::ffi::c_void,
            log_sum_base as *mut core::ffi::c_void,
            observed_logit_base as *mut core::ffi::c_void,
            observed_rank_base as *mut core::ffi::c_void,
            top_logits_base as *mut core::ffi::c_void,
            top_indices_base as *mut core::ffi::c_void,
            invalid_kind_base as *mut core::ffi::c_void,
            invalid_column_base as *mut core::ffi::c_void,
            invalid_value_base as *mut core::ffi::c_void,
            n_rows_i64,
            n_cols_i64,
            top_k_i32,
            dtype_tag,
            raw_stream,
        )
    };
    if status != 0 {
        submission.quarantine();
        return Err(Error::Msg(format!("{NAME}: FFI returned status {status}")));
    }
    submission.complete();

    fn flat_tensor(storage: RocmStorage, len: usize) -> Result<Tensor> {
        Tensor::from_parts(
            Arc::new(storage),
            Layout::contiguous(vec![len]),
            TensorId::next(),
        )
    }
    let row_maxes = flat_tensor(row_max_storage, n_rows)?.to_vec1::<f32>()?;
    let log_sums = flat_tensor(log_sum_storage, n_rows)?.to_vec1::<f32>()?;
    let observed_logits = flat_tensor(observed_logit_storage, n_rows)?.to_vec1::<f32>()?;
    let observed_ranks = flat_tensor(observed_rank_storage, n_rows)?.to_vec1::<i64>()?;
    let top_logits = if top_count == 0 {
        Vec::new()
    } else {
        flat_tensor(top_logits_storage, top_count)?.to_vec1::<f32>()?
    };
    let top_indices = if top_count == 0 {
        Vec::new()
    } else {
        flat_tensor(top_indices_storage, top_count)?.to_vec1::<i64>()?
    };
    let invalid_kinds = flat_tensor(invalid_kind_storage, n_rows)?.to_vec1::<u32>()?;
    let invalid_columns = flat_tensor(invalid_column_storage, n_rows)?.to_vec1::<i64>()?;
    let invalid_values = flat_tensor(invalid_value_storage, n_rows)?.to_vec1::<f32>()?;
    let _input_keepalive = (&observed, logits);

    crate::prompt_logprobs::finish_device_prompt_logprob_rows(
        NAME,
        n_rows,
        n_cols,
        top_k,
        row_maxes,
        log_sums,
        observed_logits,
        observed_ranks,
        top_logits,
        top_indices,
        invalid_kinds,
        invalid_columns,
        invalid_values,
    )
}
