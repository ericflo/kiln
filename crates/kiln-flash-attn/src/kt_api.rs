//! `kiln_tensor::Tensor`-typed surface alongside the candle-typed
//! API.
//!
//! Phase 7 prep — line 322 of #1082:
//!
//! > **`kiln-flash-attn` Rust API switch.** Current API takes
//! > `candle_core::Tensor`. Switch the Rust shell to
//! > `kiln_tensor::Tensor`; **do not touch the FFI/CUDA-side** —
//! > keep `kiln_flash_attn_fwd`, `_fwd_paged_decode`,
//! > `_fwd_paged_decode_dyn_seqlen`, `_bwd`, and
//! > `kiln_paged_kv_write_token_major_bf16{_slot,}`.
//!
//! Strategy: ship the kiln-tensor surface alongside the existing
//! candle-typed surface. Both call the same FFI symbols. Callers
//! migrate one site at a time. When the migration completes, the
//! candle-typed API is deleted.
//!
//! Today only `flash_attn_fwd_kt` lands here; the remaining four
//! public entry points port in subsequent PRs using the same
//! pattern.

use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use kiln_kt_bridge::BridgeError;
use kiln_tensor::{CudaStorage, DType as KtDType, Tensor as KtTensor};

use crate::{
    kiln_flash_attn_bwd, kiln_flash_attn_fwd, kiln_flash_attn_fwd_paged_decode,
    kiln_flash_attn_fwd_paged_decode_dyn_seqlen, kiln_paged_kv_write_token_major_bf16,
    kiln_paged_kv_write_token_major_bf16_slot, round_up,
};

/// Error type for the kiln-tensor-typed flash-attn surface. Stays
/// independent of candle's error so Phase 7 can delete candle
/// without rewriting this module. Carries the bridge error message
/// when storage validation fails.
#[derive(Debug)]
pub enum FlashAttnError {
    Msg(String),
}

impl std::fmt::Display for FlashAttnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FlashAttnError::Msg(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for FlashAttnError {}

impl From<BridgeError> for FlashAttnError {
    fn from(e: BridgeError) -> Self {
        FlashAttnError::Msg(e.message)
    }
}

/// Borrow the kiln-tensor's [`CudaStorage`], returning a typed
/// reference. Errors if the tensor isn't backed by CUDA, isn't
/// contiguous, or has the wrong dtype.
fn cuda_storage_of<'a>(
    t: &'a KtTensor,
    expected_dtype: KtDType,
    name: &'static str,
) -> Result<&'a CudaStorage, FlashAttnError> {
    let (st, _) = kiln_kt_bridge::cuda_storage_and_byte_offset(t, expected_dtype, name)?;
    Ok(st)
}

/// `flash_attn_fwd` over `kiln_tensor::Tensor` operands.
///
/// Mirrors [`crate::flash_attn_fwd`] one-for-one: same FFI, same
/// shape contract `[batch, seqlen, num_heads, head_dim]`, same
/// (output, softmax_lse) return tuple. Differences:
/// - Operand type is `kiln_tensor::Tensor` instead of `candle_core::Tensor`.
/// - Output + softmax_lse are allocated through `kiln_tensor`'s
///   `cuda_zeros` rather than `candle_core::Tensor::zeros`.
pub fn flash_attn_fwd_kt(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<(KtTensor, KtTensor), FlashAttnError> {
    let q_shape = q.shape();
    if q_shape.len() != 4 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: q must be rank-4 [batch, seqlen, num_heads, head_dim], got {q_shape:?}"
        )));
    }
    let k_shape = k.shape();
    if k_shape.len() != 4 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: k must be rank-4, got {k_shape:?}"
        )));
    }
    let v_shape = v.shape();
    if v_shape.len() != 4 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: v must be rank-4, got {v_shape:?}"
        )));
    }

    let (b, seqlen_q, num_heads, head_dim) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    let (_b, seqlen_k, num_heads_k, _hd) = (k_shape[0], k_shape[1], k_shape[2], k_shape[3]);

    if head_dim != 128 && head_dim != 256 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: only head_dim=128,256 supported, got {head_dim}"
        )));
    }

    // Owner-agnostic input pointers (Phase 7 v2).
    let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::BF16, "q")?;
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::BF16, "v")?;
    let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;

    // Output + softmax_lse allocated through the shared bridge.
    let out_t = alloc_cuda_tensor(q_st, KtDType::BF16, vec![b, seqlen_q, num_heads, head_dim])?;
    let lse_t = alloc_cuda_tensor(q_st, KtDType::F32, vec![b, num_heads, seqlen_q])?;
    let out_ptr = kiln_kt_bridge::cuda_output_device_ptr(&out_t);
    let lse_ptr = kiln_kt_bridge::cuda_output_device_ptr(&lse_t);

    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_flash_attn_fwd(
            q_ptr as *const _,
            k_ptr as *const _,
            v_ptr as *const _,
            out_ptr as *mut _,
            lse_ptr as *mut _,
            b as i32,
            seqlen_q as i32,
            seqlen_k as i32,
            num_heads as i32,
            num_heads_k as i32,
            head_dim as i32,
            softmax_scale,
            if causal { 1 } else { 0 },
            raw_stream,
        )
    };
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: kiln_flash_attn_fwd returned status {status}"
        )));
    }

    Ok((out_t, lse_t))
}

// ============================================================================
// Internal helpers — delegate to kiln-kt-bridge
// ============================================================================

fn cuda_storage_and_byte_offset<'a>(
    t: &'a KtTensor,
    expected_dtype: KtDType,
    name: &'static str,
) -> Result<(&'a CudaStorage, usize), FlashAttnError> {
    Ok(kiln_kt_bridge::cuda_storage_and_byte_offset(t, expected_dtype, name)?)
}

fn alloc_cuda_tensor(
    device_source: &CudaStorage,
    dtype: KtDType,
    shape: Vec<usize>,
) -> Result<KtTensor, FlashAttnError> {
    Ok(kiln_kt_bridge::alloc_cuda_tensor(device_source, dtype, shape)?)
}

// ============================================================================
// flash_attn_paged_decode_kt
// ============================================================================

/// `flash_attn_paged_decode` over `kiln_tensor::Tensor` operands.
/// Mirrors [`crate::flash_attn_paged_decode`] one-for-one.
pub fn flash_attn_paged_decode_kt(
    q: &KtTensor,
    k_pool: &KtTensor,
    v_pool: &KtTensor,
    block_table: &KtTensor,
    seqlen_k: usize,
    page_block_size: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<KtTensor, FlashAttnError> {
    let q_shape = q.shape();
    if q_shape.len() != 4 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: paged_decode q must be rank-4, got {q_shape:?}"
        )));
    }
    let (b, q_len, num_heads, head_dim) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    if q_len != 1 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: paged_decode requires query_len==1, got {q_len}"
        )));
    }
    let kp_shape = k_pool.shape();
    if kp_shape.len() != 3 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: paged_decode k_pool must be rank-3, got {kp_shape:?}"
        )));
    }
    let (_total_slots, num_heads_k, hd_k) = (kp_shape[0], kp_shape[1], kp_shape[2]);
    if hd_k != head_dim {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: k_pool head_dim ({hd_k}) != q head_dim ({head_dim})"
        )));
    }
    if v_pool.shape().len() != 3 || v_pool.shape()[2] != head_dim {
        return Err(FlashAttnError::Msg(
            "kt-flash-attn: v_pool head_dim mismatch".to_string(),
        ));
    }
    if num_heads % num_heads_k != 0 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: num_heads ({num_heads}) % num_heads_k ({num_heads_k}) != 0"
        )));
    }
    if head_dim != 128 && head_dim != 256 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: paged_decode only supports head_dim=128,256, got {head_dim}"
        )));
    }
    if page_block_size == 0 || 128 % page_block_size != 0 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: paged_decode page_block_size must divide 128, got {page_block_size}"
        )));
    }
    if block_table.dtype() != KtDType::U32 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: block_table dtype must be U32, got {:?}",
            block_table.dtype()
        )));
    }
    let bt_shape = block_table.shape();
    if bt_shape.len() != 2 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: block_table must be rank-2, got {bt_shape:?}"
        )));
    }
    let (bt_batch, max_blocks_per_seq) = (bt_shape[0], bt_shape[1]);
    if bt_batch != b {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: block_table batch ({bt_batch}) != q batch ({b})"
        )));
    }

    // Owner-agnostic input pointers (Phase 7 v2).
    let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::BF16, "q")?;
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k_pool, KtDType::BF16, "k_pool")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v_pool, KtDType::BF16, "v_pool")?;
    let bt_ptr = kiln_kt_bridge::cuda_input_device_ptr(block_table, KtDType::U32, "block_table")?;
    let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;

    let out_t = alloc_cuda_tensor(q_st, KtDType::BF16, vec![b, 1, num_heads, head_dim])?;
    let lse_t = alloc_cuda_tensor(q_st, KtDType::F32, vec![b, num_heads, 1])?;
    let out_ptr = kiln_kt_bridge::cuda_output_device_ptr(&out_t);
    let lse_ptr = kiln_kt_bridge::cuda_output_device_ptr(&lse_t);

    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_flash_attn_fwd_paged_decode(
            q_ptr as *const _,
            k_ptr as *const _,
            v_ptr as *const _,
            bt_ptr as *const i32,
            out_ptr as *mut _,
            lse_ptr as *mut _,
            b as i32,
            num_heads as i32,
            num_heads_k as i32,
            head_dim as i32,
            seqlen_k as i32,
            max_blocks_per_seq as i32,
            page_block_size as i32,
            softmax_scale,
            if causal { 1 } else { 0 },
            raw_stream,
        )
    };
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: paged_decode FFI returned {status}"
        )));
    }
    Ok(out_t)
}

// ============================================================================
// flash_attn_paged_decode_dyn_seqlen_kt
// ============================================================================

/// `flash_attn_paged_decode_dyn_seqlen` over `kiln_tensor::Tensor`
/// operands. Mirrors [`crate::flash_attn_paged_decode_dyn_seqlen`].
/// `seqused_k` is a per-batch u32 tensor of effective K/V lengths.
pub fn flash_attn_paged_decode_dyn_seqlen_kt(
    q: &KtTensor,
    k_pool: &KtTensor,
    v_pool: &KtTensor,
    block_table: &KtTensor,
    seqused_k: &KtTensor,
    max_seqlen_k: usize,
    page_block_size: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<KtTensor, FlashAttnError> {
    let q_shape = q.shape();
    if q_shape.len() != 4 || q_shape[1] != 1 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: dyn_seqlen q must be rank-4 [b, 1, h, d], got {q_shape:?}"
        )));
    }
    let (b, _q_len, num_heads, head_dim) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    let kp_shape = k_pool.shape();
    if kp_shape.len() != 3 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: dyn_seqlen k_pool must be rank-3, got {kp_shape:?}"
        )));
    }
    let num_heads_k = kp_shape[1];
    if head_dim != 128 && head_dim != 256 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: dyn_seqlen only supports head_dim=128,256, got {head_dim}"
        )));
    }
    if num_heads % num_heads_k != 0 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: dyn_seqlen num_heads ({num_heads}) % num_heads_k ({num_heads_k}) != 0"
        )));
    }
    if block_table.dtype() != KtDType::U32 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: dyn_seqlen block_table must be U32, got {:?}",
            block_table.dtype()
        )));
    }
    if seqused_k.dtype() != KtDType::U32 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: dyn_seqlen seqused_k must be U32, got {:?}",
            seqused_k.dtype()
        )));
    }
    let bt_shape = block_table.shape();
    if bt_shape.len() != 2 || bt_shape[0] != b {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: dyn_seqlen block_table must be [b, blocks], got {bt_shape:?}"
        )));
    }
    let max_blocks_per_seq = bt_shape[1];
    if seqused_k.shape() != [b] {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: dyn_seqlen seqused_k must be [b={b}], got {:?}",
            seqused_k.shape()
        )));
    }

    // Owner-agnostic input pointers (Phase 7 v2).
    let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::BF16, "q")?;
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k_pool, KtDType::BF16, "k_pool")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v_pool, KtDType::BF16, "v_pool")?;
    let bt_ptr = kiln_kt_bridge::cuda_input_device_ptr(block_table, KtDType::U32, "block_table")?;
    let sk_ptr = kiln_kt_bridge::cuda_input_device_ptr(seqused_k, KtDType::U32, "seqused_k")?;
    let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;

    let out_t = alloc_cuda_tensor(q_st, KtDType::BF16, vec![b, 1, num_heads, head_dim])?;
    let lse_t = alloc_cuda_tensor(q_st, KtDType::F32, vec![b, num_heads, 1])?;
    let out_ptr = kiln_kt_bridge::cuda_output_device_ptr(&out_t);
    let lse_ptr = kiln_kt_bridge::cuda_output_device_ptr(&lse_t);

    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_flash_attn_fwd_paged_decode_dyn_seqlen(
            q_ptr as *const _,
            k_ptr as *const _,
            v_ptr as *const _,
            bt_ptr as *const i32,
            sk_ptr as *const i32,
            out_ptr as *mut _,
            lse_ptr as *mut _,
            b as i32,
            num_heads as i32,
            num_heads_k as i32,
            head_dim as i32,
            max_seqlen_k as i32,
            max_blocks_per_seq as i32,
            page_block_size as i32,
            softmax_scale,
            if causal { 1 } else { 0 },
            raw_stream,
        )
    };
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: paged_decode_dyn_seqlen FFI returned {status}"
        )));
    }
    Ok(out_t)
}

// ============================================================================
// paged_kv_write_token_major_bf16_kt
// ============================================================================

/// `paged_kv_write_token_major_bf16` (host-slot variant) over kiln-tensor.
pub fn paged_kv_write_token_major_bf16_kt(
    k_pool: &KtTensor,
    v_pool: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    slot: usize,
) -> Result<(), FlashAttnError> {
    let kp_shape = k_pool.shape();
    if kp_shape.len() != 3 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: kv_write k_pool must be rank-3, got {kp_shape:?}"
        )));
    }
    let (total_slots, num_kv_heads, head_dim) = (kp_shape[0], kp_shape[1], kp_shape[2]);
    if v_pool.shape() != kp_shape {
        return Err(FlashAttnError::Msg(
            "kt-flash-attn: kv_write k/v pool shapes mismatch".to_string(),
        ));
    }
    if slot >= total_slots {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: kv_write slot {slot} >= total_slots {total_slots}"
        )));
    }
    let expected = num_kv_heads * head_dim;
    if k.element_count() != expected || v.element_count() != expected {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: kv_write requires one token row of {expected}, got {} and {}",
            k.element_count(),
            v.element_count()
        )));
    }
    let slot_u32 = u32::try_from(slot)
        .map_err(|_| FlashAttnError::Msg(format!("kt-flash-attn: slot {slot} exceeds u32")))?;
    let num_kv_heads_i32 = i32::try_from(num_kv_heads).map_err(|_| {
        FlashAttnError::Msg(format!("kt-flash-attn: num_kv_heads {num_kv_heads} exceeds i32"))
    })?;
    let head_dim_i32 = i32::try_from(head_dim)
        .map_err(|_| FlashAttnError::Msg(format!("kt-flash-attn: head_dim {head_dim} exceeds i32")))?;

    // Owner-agnostic input pointers (Phase 7 v2). k_pool/v_pool are
    // written in place; caller convention: pass Owned for the pools.
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::BF16, "v")?;
    let kp_ptr = kiln_kt_bridge::cuda_input_device_ptr(k_pool, KtDType::BF16, "k_pool")?;
    let vp_ptr = kiln_kt_bridge::cuda_input_device_ptr(v_pool, KtDType::BF16, "v_pool")?;
    let (k_st, _) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;

    let stream = k_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let status = unsafe {
        kiln_paged_kv_write_token_major_bf16(
            kp_ptr as *mut _,
            vp_ptr as *mut _,
            k_ptr as *const _,
            v_ptr as *const _,
            slot_u32,
            num_kv_heads_i32,
            head_dim_i32,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: kv_write FFI returned {status}"
        )));
    }
    Ok(())
}

// ============================================================================
// paged_kv_write_token_major_bf16_slot_kt
// ============================================================================

/// `paged_kv_write_token_major_bf16_slot` (device-slot variant) over kiln-tensor.
pub fn paged_kv_write_token_major_bf16_slot_kt(
    k_pool: &KtTensor,
    v_pool: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    slot: &KtTensor,
) -> Result<(), FlashAttnError> {
    let kp_shape = k_pool.shape();
    if kp_shape.len() != 3 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: kv_write_slot k_pool must be rank-3, got {kp_shape:?}"
        )));
    }
    let (_, num_kv_heads, head_dim) = (kp_shape[0], kp_shape[1], kp_shape[2]);
    if v_pool.shape() != kp_shape {
        return Err(FlashAttnError::Msg(
            "kt-flash-attn: kv_write_slot k/v pool mismatch".to_string(),
        ));
    }
    if slot.dtype() != KtDType::U32 || slot.shape() != [1] {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: kv_write_slot slot must be U32 [1], got {:?} {:?}",
            slot.dtype(),
            slot.shape()
        )));
    }
    let num_kv_heads_i32 = i32::try_from(num_kv_heads)
        .map_err(|_| FlashAttnError::Msg(format!("num_kv_heads {num_kv_heads} > i32")))?;
    let head_dim_i32 = i32::try_from(head_dim)
        .map_err(|_| FlashAttnError::Msg(format!("head_dim {head_dim} > i32")))?;

    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::BF16, "v")?;
    let kp_ptr = kiln_kt_bridge::cuda_input_device_ptr(k_pool, KtDType::BF16, "k_pool")?;
    let vp_ptr = kiln_kt_bridge::cuda_input_device_ptr(v_pool, KtDType::BF16, "v_pool")?;
    let sl_ptr = kiln_kt_bridge::cuda_input_device_ptr(slot, KtDType::U32, "slot")?;
    let (k_st, _) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;

    let stream = k_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let status = unsafe {
        kiln_paged_kv_write_token_major_bf16_slot(
            kp_ptr as *mut _,
            vp_ptr as *mut _,
            k_ptr as *const _,
            v_ptr as *const _,
            sl_ptr as *const u32,
            num_kv_heads_i32,
            head_dim_i32,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: kv_write_slot FFI returned {status}"
        )));
    }
    Ok(())
}

// ============================================================================
// flash_attn_bwd_kt
// ============================================================================

/// `flash_attn_bwd` over `kiln_tensor::Tensor` operands.
/// Returns `(dq, dk, dv)`. GQA expansion to num_heads happens
/// upstream (caller sums dk/dv across groups if needed); this
/// function only allocates expanded buffers matching the FFI's
/// shape contract.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bwd_kt(
    dout: &KtTensor,
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    out: &KtTensor,
    softmax_lse: &KtTensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<(KtTensor, KtTensor, KtTensor), FlashAttnError> {
    let q_shape = q.shape();
    if q_shape.len() != 4 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: bwd q must be rank-4, got {q_shape:?}"
        )));
    }
    let (b, seqlen_q, num_heads, head_dim) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    let k_shape = k.shape();
    if k_shape.len() != 4 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: bwd k must be rank-4, got {k_shape:?}"
        )));
    }
    let (_b, seqlen_k, num_heads_k, _hd) = (k_shape[0], k_shape[1], k_shape[2], k_shape[3]);
    if head_dim != 128 && head_dim != 256 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: bwd only supports head_dim=128,256, got {head_dim}"
        )));
    }

    let seqlen_q_rounded = round_up(seqlen_q, 128);
    let head_dim_rounded = round_up(head_dim, 32);

    // Owner-agnostic input pointers (Phase 7 v2).
    let dout_ptr = kiln_kt_bridge::cuda_input_device_ptr(dout, KtDType::BF16, "dout")?;
    let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::BF16, "q")?;
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::BF16, "v")?;
    let out_ptr = kiln_kt_bridge::cuda_input_device_ptr(out, KtDType::BF16, "out")?;
    let lse_ptr = kiln_kt_bridge::cuda_input_device_ptr(softmax_lse, KtDType::F32, "softmax_lse")?;
    let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;

    let dq = alloc_cuda_tensor(q_st, KtDType::BF16, vec![b, seqlen_q, num_heads, head_dim])?;
    let dk = alloc_cuda_tensor(q_st, KtDType::BF16, vec![b, seqlen_k, num_heads, head_dim])?;
    let dv = alloc_cuda_tensor(q_st, KtDType::BF16, vec![b, seqlen_k, num_heads, head_dim])?;
    let softmax_d =
        alloc_cuda_tensor(q_st, KtDType::F32, vec![b, num_heads, seqlen_q_rounded])?;
    let dq_accum = alloc_cuda_tensor(
        q_st,
        KtDType::F32,
        vec![b, seqlen_q_rounded, num_heads, head_dim_rounded],
    )?;
    let dq_ptr = kiln_kt_bridge::cuda_output_device_ptr(&dq);
    let dk_ptr = kiln_kt_bridge::cuda_output_device_ptr(&dk);
    let dv_ptr = kiln_kt_bridge::cuda_output_device_ptr(&dv);
    let sd_ptr = kiln_kt_bridge::cuda_output_device_ptr(&softmax_d);
    let da_ptr = kiln_kt_bridge::cuda_output_device_ptr(&dq_accum);

    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_flash_attn_bwd(
            dout_ptr as *const _,
            q_ptr as *const _,
            k_ptr as *const _,
            v_ptr as *const _,
            out_ptr as *const _,
            lse_ptr as *const _,
            dq_ptr as *mut _,
            dk_ptr as *mut _,
            dv_ptr as *mut _,
            sd_ptr as *mut _,
            da_ptr as *mut _,
            b as i32,
            seqlen_q as i32,
            seqlen_k as i32,
            num_heads as i32,
            num_heads_k as i32,
            head_dim as i32,
            softmax_scale,
            if causal { 1 } else { 0 },
            /* deterministic */ 1,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: bwd FFI returned {status}"
        )));
    }
    Ok((dq, dk, dv))
}
