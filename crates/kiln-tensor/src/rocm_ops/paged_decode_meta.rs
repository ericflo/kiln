//! ROCm wrappers for the paged-decode metadata kernels
//! (`csrc/paged_decode_meta.cu`, Phase R.9 prerequisite).
//!
//! These compute the paged gather index and the per-batch tail mask ON-DEVICE
//! from device-resident `block_table` / `seqused_k` U32 buffers, replacing the
//! host round-trips in `kiln-flash-attn`'s `rocm_sdpa.rs` paged-decode path
//! (which D2H'd the metadata, looped on the host, and H2D'd the result — a sync
//! per attention layer that also blocks HIP graph capture).
use std::sync::Arc;

use crate::{DType, Device, Error, Layout, Result, RocmStorage, Tensor, TensorId};

unsafe extern "C" {
    fn kiln_paged_gather_index_async(
        block_table_u32: *const core::ffi::c_void,
        out_idx_u32: *mut core::ffi::c_void,
        b: i64,
        seqlen_k: i64,
        max_blocks_per_seq: i64,
        page_block_size: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_paged_tail_mask_async(
        seqused_k_u32: *const core::ffi::c_void,
        out_mask_u8: *mut core::ffi::c_void,
        b: i64,
        h: i64,
        sk: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_paged_gather_rows_async(
        pool: *const core::ffi::c_void,
        block_table_u32: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        b: i64,
        seqlen_k: i64,
        max_blocks_per_seq: i64,
        page_block_size: i64,
        hk: i64,
        d: i64,
        pool_rows: i64,
        elem_bytes: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_gqa_repeat_heads_async(
        src: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        b: i64,
        sk: i64,
        hk: i64,
        h: i64,
        d: i64,
        elem_bytes: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

fn rocm_storage<'a>(t: &'a Tensor, who: &str) -> Result<&'a RocmStorage> {
    t.storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg(format!("{who}: tensor must be ROCm storage")))
}

/// Compute the flat physical-slot gather index `[b*seqlen_k]` (U32) on-device
/// from a device-resident `block_table` (U32 `[b, max_blocks_per_seq]`).
///
/// `idx[bi*seqlen_k + t] = block_table[bi, t/page_block_size]*page_block_size +
/// t%page_block_size` — bit-identical to the host loop it replaces. Feed the
/// result straight to [`crate::rocm_index_select_dim0`] (which already accepts a
/// device U32 index), so the paged gather never touches the host.
pub fn rocm_paged_gather_index(
    block_table: &Tensor,
    b: usize,
    seqlen_k: usize,
    max_blocks_per_seq: usize,
    page_block_size: usize,
) -> Result<Tensor> {
    if block_table.dtype() != DType::U32 {
        return Err(Error::Msg(format!(
            "rocm_paged_gather_index: block_table dtype must be U32, got {}",
            block_table.dtype()
        )));
    }
    let bt = if block_table.is_contiguous() && block_table.layout().start_offset() == 0 {
        block_table.clone()
    } else {
        block_table.contiguous()?
    };
    let bt_storage = rocm_storage(&bt, "rocm_paged_gather_index")?;
    let ctx = bt_storage.context();
    let device_index = match bt.device() {
        Device::Rocm(i) => i,
        _ => {
            return Err(Error::Msg(
                "rocm_paged_gather_index: not a ROCm device".to_string(),
            ));
        }
    };

    let n = b * seqlen_k;
    let out_storage = RocmStorage::zeros_ctx(&ctx, device_index, DType::U32, n)?;

    let raw_stream = bt_storage.rocm_stream_raw();
    let (bt_base, _) = bt_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();

    let status = unsafe {
        kiln_paged_gather_index_async(
            bt_base as *const core::ffi::c_void,
            out_base as *mut core::ffi::c_void,
            b as i64,
            seqlen_k as i64,
            max_blocks_per_seq as i64,
            page_block_size as i64,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_paged_gather_index: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(storage_arc, Layout::contiguous(vec![n]), TensorId::next())
        .map_err(|e| Error::Msg(format!("rocm_paged_gather_index: wrap: {e}")))
}

/// Gather token-major paged KV rows directly on-device.
///
/// `pool` must be `[total_slots, hk, d]` and `block_table` must be U32
/// `[b, max_blocks_per_seq]`. The output is `[b, seqlen_k, hk, d]`, with the
/// same dtype as `pool`. This replaces the old two-kernel
/// `rocm_paged_gather_index` + `rocm_index_select_dim0` path for ROCm paged
/// decode, avoiding the large-grid row-copy launch that fails on long contexts.
pub fn rocm_paged_gather_rows(
    pool: &Tensor,
    block_table: &Tensor,
    b: usize,
    seqlen_k: usize,
    max_blocks_per_seq: usize,
    page_block_size: usize,
) -> Result<Tensor> {
    if pool.dtype().is_packed() {
        return Err(Error::Msg(
            "rocm_paged_gather_rows: packed pool dtype not supported".to_string(),
        ));
    }
    if block_table.dtype() != DType::U32 {
        return Err(Error::Msg(format!(
            "rocm_paged_gather_rows: block_table dtype must be U32, got {}",
            block_table.dtype()
        )));
    }
    if pool.device() != block_table.device() {
        return Err(Error::Msg(format!(
            "rocm_paged_gather_rows: device mismatch pool={:?} block_table={:?}",
            pool.device(),
            block_table.device()
        )));
    }

    let pool_c = if pool.is_contiguous() {
        pool.clone()
    } else {
        pool.contiguous()?
    };
    let bt = if block_table.is_contiguous() {
        block_table.clone()
    } else {
        block_table.contiguous()?
    };

    let pool_shape = pool_c.shape();
    if pool_shape.len() != 3 {
        return Err(Error::Msg(format!(
            "rocm_paged_gather_rows: pool must have shape [total_slots, hk, d], got {pool_shape:?}"
        )));
    }
    let bt_shape = bt.shape();
    if bt_shape.len() != 2 {
        return Err(Error::Msg(format!(
            "rocm_paged_gather_rows: block_table must have shape [b, max_blocks], got {bt_shape:?}"
        )));
    }
    if b > bt_shape[0] || max_blocks_per_seq > bt_shape[1] {
        return Err(Error::Msg(format!(
            "rocm_paged_gather_rows: requested b={b}, max_blocks_per_seq={max_blocks_per_seq} \
             exceeds block_table shape {bt_shape:?}"
        )));
    }
    if page_block_size == 0 {
        return Err(Error::Msg(
            "rocm_paged_gather_rows: page_block_size must be > 0".to_string(),
        ));
    }
    if seqlen_k > max_blocks_per_seq.saturating_mul(page_block_size) {
        return Err(Error::Msg(format!(
            "rocm_paged_gather_rows: seqlen_k={seqlen_k} exceeds \
             max_blocks_per_seq*page_block_size={}",
            max_blocks_per_seq.saturating_mul(page_block_size)
        )));
    }

    let pool_rows = pool_shape[0];
    let hk = pool_shape[1];
    let d = pool_shape[2];
    let dtype = pool_c.dtype();
    let elem_bytes = dtype.size_in_bytes();
    if elem_bytes == 0 {
        return Err(Error::Msg(format!(
            "rocm_paged_gather_rows: unsupported zero-width dtype {dtype}"
        )));
    }

    let pool_storage = rocm_storage(&pool_c, "rocm_paged_gather_rows")?;
    let bt_storage = rocm_storage(&bt, "rocm_paged_gather_rows")?;
    let ctx = pool_storage.context();
    let device_index = match pool_c.device() {
        Device::Rocm(i) => i,
        _ => {
            return Err(Error::Msg(
                "rocm_paged_gather_rows: tensors must be on a ROCm device".to_string(),
            ));
        }
    };

    let n_out = b * seqlen_k * hk * d;
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, dtype, n_out)?;

    let raw_stream = pool_storage.rocm_stream_raw();
    let (pool_base, _) = pool_storage.device_ptr_raw();
    let (bt_base, _) = bt_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();

    let pool_off = (pool_c.layout().start_offset() * elem_bytes) as u64;
    let bt_off = (bt.layout().start_offset() * DType::U32.size_in_bytes()) as u64;

    let status = unsafe {
        kiln_paged_gather_rows_async(
            (pool_base + pool_off) as *const core::ffi::c_void,
            (bt_base + bt_off) as *const core::ffi::c_void,
            out_base as *mut core::ffi::c_void,
            b as i64,
            seqlen_k as i64,
            max_blocks_per_seq as i64,
            page_block_size as i64,
            hk as i64,
            d as i64,
            pool_rows as i64,
            elem_bytes as i64,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_paged_gather_rows: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(
        storage_arc,
        Layout::contiguous(vec![b, seqlen_k, hk, d]),
        TensorId::next(),
    )
    .map_err(|e| Error::Msg(format!("rocm_paged_gather_rows: wrap: {e}")))
}

/// Repeat GQA KV heads directly on ROCm.
///
/// Input is `[b, sk, hk, d]`; output is `[b, sk, h, d]`, with each input head
/// repeated `h / hk` times. This is equivalent to
/// `unsqueeze + expand + contiguous + reshape`, but avoids the generic ROCm
/// broadcast path's large flattened `index_select_dim0`.
pub fn rocm_gqa_repeat_heads(src: &Tensor, h: usize) -> Result<Tensor> {
    if src.dtype().is_packed() {
        return Err(Error::Msg(
            "rocm_gqa_repeat_heads: packed dtype not supported".to_string(),
        ));
    }
    let src_c = if src.is_contiguous() {
        src.clone()
    } else {
        src.contiguous()?
    };
    let shape = src_c.shape();
    if shape.len() != 4 {
        return Err(Error::Msg(format!(
            "rocm_gqa_repeat_heads: src must have shape [b, sk, hk, d], got {shape:?}"
        )));
    }
    let (b, sk, hk, d) = (shape[0], shape[1], shape[2], shape[3]);
    if hk == h {
        return Ok(src_c);
    }
    if hk == 0 || h == 0 || h % hk != 0 {
        return Err(Error::Msg(format!(
            "rocm_gqa_repeat_heads: output heads h={h} must be a non-zero multiple of hk={hk}"
        )));
    }

    let dtype = src_c.dtype();
    let elem_bytes = dtype.size_in_bytes();
    if elem_bytes == 0 {
        return Err(Error::Msg(format!(
            "rocm_gqa_repeat_heads: unsupported zero-width dtype {dtype}"
        )));
    }

    let src_storage = rocm_storage(&src_c, "rocm_gqa_repeat_heads")?;
    let ctx = src_storage.context();
    let device_index = match src_c.device() {
        Device::Rocm(i) => i,
        _ => {
            return Err(Error::Msg(
                "rocm_gqa_repeat_heads: src must be on a ROCm device".to_string(),
            ));
        }
    };

    let n_out = b * sk * h * d;
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, dtype, n_out)?;

    let raw_stream = src_storage.rocm_stream_raw();
    let (src_base, _) = src_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();
    let src_off = (src_c.layout().start_offset() * elem_bytes) as u64;

    let status = unsafe {
        kiln_gqa_repeat_heads_async(
            (src_base + src_off) as *const core::ffi::c_void,
            out_base as *mut core::ffi::c_void,
            b as i64,
            sk as i64,
            hk as i64,
            h as i64,
            d as i64,
            elem_bytes as i64,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_gqa_repeat_heads: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(
        storage_arc,
        Layout::contiguous(vec![b, sk, h, d]),
        TensorId::next(),
    )
    .map_err(|e| Error::Msg(format!("rocm_gqa_repeat_heads: wrap: {e}")))
}

/// Build the per-batch tail mask `[b*h, 1, sk]` (U8: `1` => masked) on-device
/// from a device-resident `seqused_k` (U32 `[b]`): `mask[bi,hi,j] = j >=
/// seqused_k[bi]`. Bit-identical to the host loop in `sdpa_forward_dyn_tail`.
/// Consumed by `rocm_masked_fill`.
pub fn rocm_build_tail_mask(seqused_k: &Tensor, b: usize, h: usize, sk: usize) -> Result<Tensor> {
    if seqused_k.dtype() != DType::U32 {
        return Err(Error::Msg(format!(
            "rocm_build_tail_mask: seqused_k dtype must be U32, got {}",
            seqused_k.dtype()
        )));
    }
    let su = if seqused_k.is_contiguous() && seqused_k.layout().start_offset() == 0 {
        seqused_k.clone()
    } else {
        seqused_k.contiguous()?
    };
    let su_storage = rocm_storage(&su, "rocm_build_tail_mask")?;
    let ctx = su_storage.context();
    let device_index = match su.device() {
        Device::Rocm(i) => i,
        _ => {
            return Err(Error::Msg(
                "rocm_build_tail_mask: not a ROCm device".to_string(),
            ));
        }
    };

    let n = b * h * sk;
    let out_storage = RocmStorage::zeros_ctx(&ctx, device_index, DType::U8, n)?;

    let raw_stream = su_storage.rocm_stream_raw();
    let (su_base, _) = su_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();

    let status = unsafe {
        kiln_paged_tail_mask_async(
            su_base as *const core::ffi::c_void,
            out_base as *mut core::ffi::c_void,
            b as i64,
            h as i64,
            sk as i64,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_build_tail_mask: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(
        storage_arc,
        Layout::contiguous(vec![b * h, 1, sk]),
        TensorId::next(),
    )
    .map_err(|e| Error::Msg(format!("rocm_build_tail_mask: wrap: {e}")))
}
