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
use kiln_tensor::{CudaStorage, DType as KtDType, StorageBackend, Tensor as KtTensor};

use crate::{
    kiln_flash_attn_bwd, kiln_flash_attn_fwd, kiln_flash_attn_fwd_paged_decode,
    kiln_flash_attn_fwd_paged_decode_dyn_seqlen, kiln_paged_kv_write_token_major_bf16,
    kiln_paged_kv_write_token_major_bf16_slot, round_up,
};

/// Error type for the kiln-tensor-typed flash-attn surface. Stays
/// independent of candle's error so Phase 7 can delete candle
/// without rewriting this module.
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

/// Borrow the kiln-tensor's [`CudaStorage`], returning a typed
/// reference. Errors if the tensor isn't backed by CUDA, isn't
/// contiguous, or has the wrong dtype.
fn cuda_storage_of<'a>(
    t: &'a KtTensor,
    expected_dtype: KtDType,
    name: &'static str,
) -> Result<&'a CudaStorage, FlashAttnError> {
    if t.dtype() != expected_dtype {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: {name} must be {expected_dtype}, got {}",
            t.dtype()
        )));
    }
    if !t.is_contiguous() {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: {name} must be contiguous (call .contiguous() first)"
        )));
    }
    t.storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            FlashAttnError::Msg(format!("kt-flash-attn: {name} must be a CUDA tensor"))
        })
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

    let q_st = cuda_storage_of(q, KtDType::BF16, "q")?;
    let k_st = cuda_storage_of(k, KtDType::BF16, "k")?;
    let v_st = cuda_storage_of(v, KtDType::BF16, "v")?;

    // All three operands must share the same CUDA device (cudarc
    // pointers from different devices are not interchangeable).
    let candle_device = q_st.candle_device().clone();
    let device_index = q_st.device().index().unwrap_or(0);

    // Output + softmax_lse allocated through kiln-tensor's CUDA
    // helpers. Shapes match the FFI contract exactly.
    let n_out: usize = b * seqlen_q * num_heads * head_dim;
    let n_lse: usize = b * num_heads * seqlen_q;

    let out_storage =
        kiln_tensor::cuda_zeros(candle_device.clone(), device_index, KtDType::BF16, n_out)
            .map_err(|e| FlashAttnError::Msg(format!("kt-flash-attn: out alloc: {e}")))?;
    let lse_storage =
        kiln_tensor::cuda_zeros(candle_device.clone(), device_index, KtDType::F32, n_lse)
            .map_err(|e| FlashAttnError::Msg(format!("kt-flash-attn: lse alloc: {e}")))?;

    let out_cuda = out_storage
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("cuda_zeros produced non-CUDA storage");
    let lse_cuda = lse_storage
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("cuda_zeros produced non-CUDA storage");

    // Grab the CUDA stream from the device handle that allocated `q`.
    let stream = candle_device.cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    // Slice each operand by its layout's byte offset, then take the
    // device pointer through the cudarc 0.19 stream-bound API.
    let q_off_bytes = q.layout().start_offset() * KtDType::BF16.size_in_bytes();
    let k_off_bytes = k.layout().start_offset() * KtDType::BF16.size_in_bytes();
    let v_off_bytes = v.layout().start_offset() * KtDType::BF16.size_in_bytes();

    let q_slice = q_st.slice().slice(q_off_bytes..);
    let k_slice = k_st.slice().slice(k_off_bytes..);
    let v_slice = v_st.slice().slice(v_off_bytes..);
    let out_slice = out_cuda.slice().slice(0..);
    let lse_slice = lse_cuda.slice().slice(0..);

    let status = unsafe {
        let (q_ptr, _g1) = q_slice.device_ptr(&stream);
        let (k_ptr, _g2) = k_slice.device_ptr(&stream);
        let (v_ptr, _g3) = v_slice.device_ptr(&stream);
        let (out_ptr, _g4) = out_slice.device_ptr(&stream);
        let (lse_ptr, _g5) = lse_slice.device_ptr(&stream);

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

    // Wrap each storage into a kiln-tensor with the appropriate shape.
    let out_t = KtTensor::from_parts(
        out_storage,
        kiln_tensor::Layout::contiguous(vec![b, seqlen_q, num_heads, head_dim]),
        kiln_tensor::TensorId::next(),
    )
    .map_err(|e| FlashAttnError::Msg(format!("kt-flash-attn: out wrap: {e}")))?;
    let lse_t = KtTensor::from_parts(
        lse_storage,
        kiln_tensor::Layout::contiguous(vec![b, num_heads, seqlen_q]),
        kiln_tensor::TensorId::next(),
    )
    .map_err(|e| FlashAttnError::Msg(format!("kt-flash-attn: lse wrap: {e}")))?;
    Ok((out_t, lse_t))
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Convert a kiln_tensor (validated CUDA + dtype) to a (CudaStorage, byte_offset)
/// pair. The byte offset already includes `start_offset * dtype.size_in_bytes()`.
fn cuda_storage_and_byte_offset<'a>(
    t: &'a KtTensor,
    expected_dtype: KtDType,
    name: &'static str,
) -> Result<(&'a CudaStorage, usize), FlashAttnError> {
    let st = cuda_storage_of(t, expected_dtype, name)?;
    let byte_off = t.layout().start_offset() * expected_dtype.size_in_bytes();
    Ok((st, byte_off))
}

/// Allocate a fresh CUDA-backed `kiln_tensor::Tensor` of `dtype`,
/// `shape`, on the same CUDA device as `device_source`.
fn alloc_cuda_tensor(
    device_source: &CudaStorage,
    dtype: KtDType,
    shape: Vec<usize>,
) -> Result<KtTensor, FlashAttnError> {
    let candle_device = device_source.candle_device().clone();
    let device_index = device_source.device().index().unwrap_or(0);
    let n_elements: usize = shape.iter().product();
    let storage = kiln_tensor::cuda_zeros(candle_device, device_index, dtype, n_elements)
        .map_err(|e| FlashAttnError::Msg(format!("kt-flash-attn: alloc {dtype:?} {shape:?}: {e}")))?;
    KtTensor::from_parts(
        storage,
        kiln_tensor::Layout::contiguous(shape),
        kiln_tensor::TensorId::next(),
    )
    .map_err(|e| FlashAttnError::Msg(format!("kt-flash-attn: alloc wrap: {e}")))
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

    let (q_st, q_off) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;
    let (k_st, k_off) = cuda_storage_and_byte_offset(k_pool, KtDType::BF16, "k_pool")?;
    let (v_st, v_off) = cuda_storage_and_byte_offset(v_pool, KtDType::BF16, "v_pool")?;
    let (bt_st, bt_off) = cuda_storage_and_byte_offset(block_table, KtDType::U32, "block_table")?;

    let out_t = alloc_cuda_tensor(q_st, KtDType::BF16, vec![b, 1, num_heads, head_dim])?;
    let lse_t = alloc_cuda_tensor(q_st, KtDType::F32, vec![b, num_heads, 1])?;
    let out_st = out_t.storage();
    let out_cuda = out_st
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc returned CUDA");
    let lse_st = lse_t.storage();
    let lse_cuda = lse_st
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc returned CUDA");

    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let q_slice = q_st.slice().slice(q_off..);
    let k_slice = k_st.slice().slice(k_off..);
    let v_slice = v_st.slice().slice(v_off..);
    let bt_slice = bt_st.slice().slice(bt_off..);
    let out_slice = out_cuda.slice().slice(0..);
    let lse_slice = lse_cuda.slice().slice(0..);

    let status = unsafe {
        let (q_ptr, _g1) = q_slice.device_ptr(&stream);
        let (k_ptr, _g2) = k_slice.device_ptr(&stream);
        let (v_ptr, _g3) = v_slice.device_ptr(&stream);
        let (bt_ptr, _g4) = bt_slice.device_ptr(&stream);
        let (out_ptr, _g5) = out_slice.device_ptr(&stream);
        let (lse_ptr, _g6) = lse_slice.device_ptr(&stream);

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

    let (q_st, q_off) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;
    let (k_st, k_off) = cuda_storage_and_byte_offset(k_pool, KtDType::BF16, "k_pool")?;
    let (v_st, v_off) = cuda_storage_and_byte_offset(v_pool, KtDType::BF16, "v_pool")?;
    let (bt_st, bt_off) = cuda_storage_and_byte_offset(block_table, KtDType::U32, "block_table")?;
    let (sk_st, sk_off) = cuda_storage_and_byte_offset(seqused_k, KtDType::U32, "seqused_k")?;

    let out_t = alloc_cuda_tensor(q_st, KtDType::BF16, vec![b, 1, num_heads, head_dim])?;
    let lse_t = alloc_cuda_tensor(q_st, KtDType::F32, vec![b, num_heads, 1])?;
    let out_cuda = out_t
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");
    let lse_cuda = lse_t
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");

    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let q_slice = q_st.slice().slice(q_off..);
    let k_slice = k_st.slice().slice(k_off..);
    let v_slice = v_st.slice().slice(v_off..);
    let bt_slice = bt_st.slice().slice(bt_off..);
    let sk_slice = sk_st.slice().slice(sk_off..);
    let out_slice = out_cuda.slice().slice(0..);
    let lse_slice = lse_cuda.slice().slice(0..);

    let status = unsafe {
        let (q_ptr, _g1) = q_slice.device_ptr(&stream);
        let (k_ptr, _g2) = k_slice.device_ptr(&stream);
        let (v_ptr, _g3) = v_slice.device_ptr(&stream);
        let (bt_ptr, _g4) = bt_slice.device_ptr(&stream);
        let (sk_ptr, _g5) = sk_slice.device_ptr(&stream);
        let (out_ptr, _g6) = out_slice.device_ptr(&stream);
        let (lse_ptr, _g7) = lse_slice.device_ptr(&stream);

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

    let (k_st, k_off) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;
    let (v_st, v_off) = cuda_storage_and_byte_offset(v, KtDType::BF16, "v")?;
    let (kp_st, kp_off) = cuda_storage_and_byte_offset(k_pool, KtDType::BF16, "k_pool")?;
    let (vp_st, vp_off) = cuda_storage_and_byte_offset(v_pool, KtDType::BF16, "v_pool")?;

    let stream = k_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let k_slice = k_st.slice().slice(k_off..);
    let v_slice = v_st.slice().slice(v_off..);
    let kp_slice = kp_st.slice().slice(kp_off..);
    let vp_slice = vp_st.slice().slice(vp_off..);
    let status = unsafe {
        let (k_ptr, _g1) = k_slice.device_ptr(&stream);
        let (v_ptr, _g2) = v_slice.device_ptr(&stream);
        let (kp_ptr, _g3) = kp_slice.device_ptr(&stream);
        let (vp_ptr, _g4) = vp_slice.device_ptr(&stream);
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

    let (k_st, k_off) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;
    let (v_st, v_off) = cuda_storage_and_byte_offset(v, KtDType::BF16, "v")?;
    let (kp_st, kp_off) = cuda_storage_and_byte_offset(k_pool, KtDType::BF16, "k_pool")?;
    let (vp_st, vp_off) = cuda_storage_and_byte_offset(v_pool, KtDType::BF16, "v_pool")?;
    let (sl_st, sl_off) = cuda_storage_and_byte_offset(slot, KtDType::U32, "slot")?;

    let stream = k_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let k_slice = k_st.slice().slice(k_off..);
    let v_slice = v_st.slice().slice(v_off..);
    let kp_slice = kp_st.slice().slice(kp_off..);
    let vp_slice = vp_st.slice().slice(vp_off..);
    let sl_slice = sl_st.slice().slice(sl_off..);
    let status = unsafe {
        let (k_ptr, _g1) = k_slice.device_ptr(&stream);
        let (v_ptr, _g2) = v_slice.device_ptr(&stream);
        let (kp_ptr, _g3) = kp_slice.device_ptr(&stream);
        let (vp_ptr, _g4) = vp_slice.device_ptr(&stream);
        let (sl_ptr, _g5) = sl_slice.device_ptr(&stream);
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

    let (dout_st, dout_off) = cuda_storage_and_byte_offset(dout, KtDType::BF16, "dout")?;
    let (q_st, q_off) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;
    let (k_st, k_off) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;
    let (v_st, v_off) = cuda_storage_and_byte_offset(v, KtDType::BF16, "v")?;
    let (out_st, out_off) = cuda_storage_and_byte_offset(out, KtDType::BF16, "out")?;
    let (lse_st, lse_off) = cuda_storage_and_byte_offset(softmax_lse, KtDType::F32, "softmax_lse")?;

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

    let dq_cuda = dq.storage().as_any().downcast_ref::<CudaStorage>().unwrap();
    let dk_cuda = dk.storage().as_any().downcast_ref::<CudaStorage>().unwrap();
    let dv_cuda = dv.storage().as_any().downcast_ref::<CudaStorage>().unwrap();
    let sd_cuda = softmax_d.storage().as_any().downcast_ref::<CudaStorage>().unwrap();
    let da_cuda = dq_accum.storage().as_any().downcast_ref::<CudaStorage>().unwrap();

    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let dout_slice = dout_st.slice().slice(dout_off..);
    let q_slice = q_st.slice().slice(q_off..);
    let k_slice = k_st.slice().slice(k_off..);
    let v_slice = v_st.slice().slice(v_off..);
    let out_slice = out_st.slice().slice(out_off..);
    let lse_slice = lse_st.slice().slice(lse_off..);
    let dq_slice = dq_cuda.slice().slice(0..);
    let dk_slice = dk_cuda.slice().slice(0..);
    let dv_slice = dv_cuda.slice().slice(0..);
    let sd_slice = sd_cuda.slice().slice(0..);
    let da_slice = da_cuda.slice().slice(0..);

    let status = unsafe {
        let (dout_ptr, _g1) = dout_slice.device_ptr(&stream);
        let (q_ptr, _g2) = q_slice.device_ptr(&stream);
        let (k_ptr, _g3) = k_slice.device_ptr(&stream);
        let (v_ptr, _g4) = v_slice.device_ptr(&stream);
        let (out_ptr, _g5) = out_slice.device_ptr(&stream);
        let (lse_ptr, _g6) = lse_slice.device_ptr(&stream);
        let (dq_ptr, _g7) = dq_slice.device_ptr(&stream);
        let (dk_ptr, _g8) = dk_slice.device_ptr(&stream);
        let (dv_ptr, _g9) = dv_slice.device_ptr(&stream);
        let (sd_ptr, _g10) = sd_slice.device_ptr(&stream);
        let (da_ptr, _g11) = da_slice.device_ptr(&stream);

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
