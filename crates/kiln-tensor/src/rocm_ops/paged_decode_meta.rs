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
        _ => return Err(Error::Msg("rocm_paged_gather_index: not a ROCm device".to_string())),
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

/// Build the per-batch tail mask `[b*h, 1, sk]` (U8: `1` => masked) on-device
/// from a device-resident `seqused_k` (U32 `[b]`): `mask[bi,hi,j] = j >=
/// seqused_k[bi]`. Bit-identical to the host loop in `sdpa_forward_dyn_tail`.
/// Consumed by `rocm_masked_fill`.
pub fn rocm_build_tail_mask(
    seqused_k: &Tensor,
    b: usize,
    h: usize,
    sk: usize,
) -> Result<Tensor> {
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
        _ => return Err(Error::Msg("rocm_build_tail_mask: not a ROCm device".to_string())),
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
    Tensor::from_parts(storage_arc, Layout::contiguous(vec![b * h, 1, sk]), TensorId::next())
        .map_err(|e| Error::Msg(format!("rocm_build_tail_mask: wrap: {e}")))
}
