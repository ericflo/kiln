//! `kiln_tensor::Tensor`-typed surface for the flash-attention kernels.
//!
//! Phase 7 (#1082) — this is now the **only** Rust surface for the
//! flash-attn FFI. The previous candle-typed parallel API
//! (`flash_attn_fwd`, `flash_attn_bwd`, `flash_attn_paged_decode*`,
//! `paged_kv_write_*`) was deleted after every `kiln-model` production
//! caller migrated to these `*_kt` wrappers.
//!
//! Surface coverage:
//! - [`flash_attn_fwd_kt`] — dense forward
//! - [`flash_attn_bwd_kt`] — dense backward (expanded GQA dk/dv)
//! - [`flash_attn_paged_decode_kt`] — single-step paged decode
//! - [`flash_attn_paged_decode_dyn_seqlen_kt`] — graph-stable dyn-seqlen paged decode
//! - [`flash_attn_paged_decode_dyn_seqlen_kt_with_graph_outputs`] — caller-owned
//!   outputs for CUDA-graph capture
//! - `paged_kv_write_token_major_bf16{,_slot,_batch_slot}_kt` — paged-KV writers
//!
//! All `*_kt` shells bottom out in the same FFI symbols
//! (`kiln_flash_attn_*` / `kiln_paged_kv_write_*`) declared in `lib.rs`.

use kiln_kt_bridge::BridgeError;
#[cfg(any(feature = "cuda", feature = "rocm"))]
use kiln_tensor::Device as KtDevice;
use kiln_tensor::{DType as KtDType, Tensor as KtTensor};

// CUDA-only imports: the FFI symbols + the `CudaStorage` downcast helper are
// only reached on the CUDA path. When building `--no-default-features --features
// rocm` (no `cuda`), every `*_kt` body dispatches into `crate::rocm_sdpa` before
// touching these, so they would be dead — gate them on `cuda` to keep the ROCm
// build warning-clean and CUDA-feature byte-identical.
#[cfg(feature = "cuda")]
use crate::{
    kiln_flash_attn_bwd, kiln_flash_attn_fwd, kiln_flash_attn_fwd_paged_decode,
    kiln_flash_attn_fwd_paged_decode_dyn_seqlen, kiln_paged_kv_write_token_major_bf16,
    kiln_paged_kv_write_token_major_bf16_batch_slot, kiln_paged_kv_write_token_major_bf16_slot,
    round_up,
};
#[cfg(feature = "cuda")]
use kiln_tensor::CudaStorage;

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

// `cuda_storage_of` helper was dead code post-338b1b88 — every
// caller now uses `cuda_storage_and_byte_offset` directly via the
// bridge or extracts pointers through `cuda_input_device_ptr` /
// `cuda_output_device_ptr`. Removed to silence dead-code warning.

/// `flash_attn_fwd` over `kiln_tensor::Tensor` operands.
///
/// Mirrors [`crate::flash_attn_fwd`] one-for-one: same FFI, same
/// shape contract `[batch, seqlen, num_heads, head_dim]`, same
/// (output, softmax_lse) return tuple. Differences:
/// - Operand type is `kiln_tensor::Tensor` instead of `candle_core::Tensor`.
/// - Output + softmax_lse are allocated through `kiln_tensor`'s
///   `cuda_zeros` rather than `candle_core::Tensor::zeros`.
// Some validation locals are only consumed by the CUDA FFI tail; on a rocm-only
// build the ROCm composite dispatch returns first, so they are unused there.
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
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

    // ROCm composite SDPA dispatch (Phase R.8) — no CUTLASS on ROCm, so the
    // attention path runs through the fully on-device kiln_tensor composite.
    #[cfg(feature = "rocm")]
    if matches!(q.device(), KtDevice::Rocm(_)) {
        return crate::rocm_sdpa::flash_attn_fwd_rocm(q, k, v, softmax_scale, causal);
    }

    #[cfg(feature = "cuda")]
    {
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

        let raw_stream = q_st.cuda_stream_raw();

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

        return Ok((out_t, lse_t));
    }

    // Neither a ROCm device (handled above) nor a CUDA build: no backend can
    // service this operand.
    #[cfg(not(feature = "cuda"))]
    Err(FlashAttnError::Msg(format!(
        "kt-flash-attn: flash_attn_fwd_kt has no backend for device {:?} \
         (cuda feature off; only Device::Rocm is supported in this build)",
        q.device()
    )))
}

#[cfg_attr(not(any(feature = "cuda", feature = "rocm")), allow(unused_variables))]
pub fn flash_attn_fwd_no_lse_kt(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Option<KtTensor>, FlashAttnError> {
    let q_shape = q.shape();
    if q_shape.len() != 4 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn-no-lse: q must be rank-4 [batch, seqlen, num_heads, head_dim], got {q_shape:?}"
        )));
    }
    let k_shape = k.shape();
    if k_shape.len() != 4 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn-no-lse: k must be rank-4, got {k_shape:?}"
        )));
    }
    if v.shape() != k_shape {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn-no-lse: k/v shape mismatch {k_shape:?}/{:?}",
            v.shape()
        )));
    }

    #[cfg(feature = "rocm")]
    if matches!(q.device(), KtDevice::Rocm(_)) {
        return crate::rocm_sdpa::flash_attn_fwd_rocm_no_lse(q, k, v, softmax_scale, causal);
    }

    #[cfg(feature = "cuda")]
    {
        if matches!(q.device(), KtDevice::Cuda(_)) {
            let (out, _lse) = flash_attn_fwd_kt(q, k, v, softmax_scale, causal)?;
            return Ok(Some(out));
        }
    }

    Ok(None)
}

#[cfg_attr(not(feature = "rocm"), allow(unused_variables))]
pub fn flash_attn_fwd_head_major_kt(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<(KtTensor, KtTensor), FlashAttnError> {
    let q_shape = q.shape();
    if q_shape.len() != 4 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn-head-major: q must be rank-4 [batch, heads, seqlen, head_dim], got {q_shape:?}"
        )));
    }
    let k_shape = k.shape();
    if k_shape.len() != 4 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn-head-major: k must be rank-4, got {k_shape:?}"
        )));
    }
    let v_shape = v.shape();
    if v_shape.len() != 4 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn-head-major: v must be rank-4, got {v_shape:?}"
        )));
    }

    let (b, num_heads, _seqlen_q, head_dim) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    let (kb, num_heads_k, _seqlen_k, khd) = (k_shape[0], k_shape[1], k_shape[2], k_shape[3]);
    if v_shape != k_shape {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn-head-major: k/v shape mismatch {k_shape:?}/{v_shape:?}"
        )));
    }
    if kb != b
        || khd != head_dim
        || num_heads == 0
        || num_heads_k == 0
        || num_heads % num_heads_k != 0
    {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn-head-major: invalid q/k shapes q={q_shape:?} k={k_shape:?}"
        )));
    }
    if head_dim != 128 && head_dim != 256 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn-head-major: only head_dim=128,256 supported, got {head_dim}"
        )));
    }

    #[cfg(feature = "rocm")]
    if matches!(q.device(), KtDevice::Rocm(_)) {
        return crate::rocm_sdpa::flash_attn_fwd_head_major_rocm(q, k, v, softmax_scale, causal);
    }

    Err(FlashAttnError::Msg(
        "kt-flash-attn-head-major: backend does not support head-major prefill".to_string(),
    ))
}

// ============================================================================
// Internal helpers — delegate to kiln-kt-bridge
// ============================================================================

#[cfg(feature = "cuda")]
fn cuda_storage_and_byte_offset<'a>(
    t: &'a KtTensor,
    expected_dtype: KtDType,
    name: &'static str,
) -> Result<(&'a CudaStorage, usize), FlashAttnError> {
    Ok(kiln_kt_bridge::cuda_storage_and_byte_offset(
        t,
        expected_dtype,
        name,
    )?)
}

#[cfg(feature = "cuda")]
fn alloc_cuda_tensor(
    device_source: &CudaStorage,
    dtype: KtDType,
    shape: Vec<usize>,
) -> Result<KtTensor, FlashAttnError> {
    Ok(kiln_kt_bridge::alloc_cuda_tensor(
        device_source,
        dtype,
        shape,
    )?)
}

#[cfg(feature = "cuda")]
fn flash_attn_bwd_deterministic() -> bool {
    let raw = std::env::var("KILN_FLASH_ATTN_BWD_DETERMINISTIC").ok();
    let lower = raw.as_deref().map(str::trim).map(str::to_ascii_lowercase);
    match lower.as_deref() {
        Some("1") | Some("true") | Some("yes") => true,
        Some("0") | Some("false") | Some("no") => false,
        _ => false,
    }
}

// ============================================================================
// flash_attn_paged_decode_kt
// ============================================================================

/// `flash_attn_paged_decode` over `kiln_tensor::Tensor` operands.
/// Mirrors [`crate::flash_attn_paged_decode`] one-for-one.
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
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

    // ROCm composite paged-decode dispatch (Phase R.8).
    #[cfg(feature = "rocm")]
    if matches!(q.device(), KtDevice::Rocm(_)) {
        return crate::rocm_sdpa::flash_attn_paged_decode_rocm(
            q,
            k_pool,
            v_pool,
            block_table,
            seqlen_k,
            page_block_size,
            softmax_scale,
            causal,
        );
    }

    #[cfg(feature = "cuda")]
    {
        // Owner-agnostic input pointers (Phase 7 v2).
        let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::BF16, "q")?;
        let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k_pool, KtDType::BF16, "k_pool")?;
        let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v_pool, KtDType::BF16, "v_pool")?;
        let bt_ptr =
            kiln_kt_bridge::cuda_input_device_ptr(block_table, KtDType::U32, "block_table")?;
        let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;

        let out_t = alloc_cuda_tensor(q_st, KtDType::BF16, vec![b, 1, num_heads, head_dim])?;
        let lse_t = alloc_cuda_tensor(q_st, KtDType::F32, vec![b, num_heads, 1])?;
        let out_ptr = kiln_kt_bridge::cuda_output_device_ptr(&out_t);
        let lse_ptr = kiln_kt_bridge::cuda_output_device_ptr(&lse_t);

        let raw_stream = q_st.cuda_stream_raw();

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
        return Ok(out_t);
    }

    #[cfg(not(feature = "cuda"))]
    Err(FlashAttnError::Msg(format!(
        "kt-flash-attn: flash_attn_paged_decode_kt has no backend for device {:?} \
         (cuda feature off; only Device::Rocm is supported in this build)",
        q.device()
    )))
}

// ============================================================================
// flash_attn_paged_decode_dyn_seqlen_kt
// ============================================================================

/// `flash_attn_paged_decode_dyn_seqlen` over `kiln_tensor::Tensor`
/// operands. Mirrors [`crate::flash_attn_paged_decode_dyn_seqlen`].
/// `seqused_k` is a per-batch u32 tensor of effective K/V lengths.
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
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

    // ROCm composite dyn-seqlen paged-decode dispatch (Phase R.8).
    #[cfg(feature = "rocm")]
    if matches!(q.device(), KtDevice::Rocm(_)) {
        return crate::rocm_sdpa::flash_attn_paged_decode_dyn_seqlen_rocm(
            q,
            k_pool,
            v_pool,
            block_table,
            seqused_k,
            max_seqlen_k,
            page_block_size,
            softmax_scale,
            causal,
        );
    }

    #[cfg(feature = "cuda")]
    {
        // Owner-agnostic input pointers (Phase 7 v2).
        let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::BF16, "q")?;
        let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k_pool, KtDType::BF16, "k_pool")?;
        let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v_pool, KtDType::BF16, "v_pool")?;
        let bt_ptr =
            kiln_kt_bridge::cuda_input_device_ptr(block_table, KtDType::U32, "block_table")?;
        let sk_ptr = kiln_kt_bridge::cuda_input_device_ptr(seqused_k, KtDType::U32, "seqused_k")?;
        let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;

        let out_t = alloc_cuda_tensor(q_st, KtDType::BF16, vec![b, 1, num_heads, head_dim])?;
        let lse_t = alloc_cuda_tensor(q_st, KtDType::F32, vec![b, num_heads, 1])?;
        let out_ptr = kiln_kt_bridge::cuda_output_device_ptr(&out_t);
        let lse_ptr = kiln_kt_bridge::cuda_output_device_ptr(&lse_t);

        let raw_stream = q_st.cuda_stream_raw();

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
        return Ok(out_t);
    }

    #[cfg(not(feature = "cuda"))]
    Err(FlashAttnError::Msg(format!(
        "kt-flash-attn: flash_attn_paged_decode_dyn_seqlen_kt has no backend for device {:?} \
         (cuda feature off; only Device::Rocm is supported in this build)",
        q.device()
    )))
}

/// `flash_attn_paged_decode_dyn_seqlen` over `kiln_tensor::Tensor`
/// operands — **caller-owned-output variant** for CUDA-graph capture.
///
/// Mirrors [`flash_attn_paged_decode_dyn_seqlen_kt`] but takes the
/// `(out, lse)` buffers as caller-owned kt-Tensors. The kernel writes
/// in place; pointer addresses are baked into a captured CUDA graph
/// and survive replays as long as the caller's tensors do.
///
/// Companion to the candle-typed
/// [`crate::flash_attn_paged_decode_dyn_seqlen`] when `graph_outputs
/// = Some((out, lse))`. Same shape contract:
/// - `out` must be BF16 `[b, 1, num_heads, head_dim]`
/// - `lse` must be F32 `[b, num_heads, 1]`
///
/// Returns `()` — the result is in the caller's `out` tensor. Errors
/// on dtype/shape mismatches before any FFI dispatch.
///
/// Substrate addition (#1082) that closes the last candle fallback
/// in `kiln-model::backend::cuda::flash_attn_paged_decode_contiguous_
/// batch_dyn_seqlen_with_graph_outputs`. Bit-exact by construction —
/// bottoms out in the same `kiln_flash_attn_fwd_paged_decode_dyn_
/// seqlen` FFI symbol as the candle path.
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub fn flash_attn_paged_decode_dyn_seqlen_kt_with_graph_outputs(
    q: &KtTensor,
    k_pool: &KtTensor,
    v_pool: &KtTensor,
    block_table: &KtTensor,
    seqused_k: &KtTensor,
    out: &KtTensor,
    lse: &KtTensor,
    max_seqlen_k: usize,
    page_block_size: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<(), FlashAttnError> {
    let q_shape = q.shape();
    if q_shape.len() != 4 || q_shape[1] != 1 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: with_graph_outputs q must be rank-4 [b, 1, h, d], got {q_shape:?}"
        )));
    }
    let (b, _q_len, num_heads, head_dim) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    let kp_shape = k_pool.shape();
    if kp_shape.len() != 3 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: with_graph_outputs k_pool must be rank-3, got {kp_shape:?}"
        )));
    }
    let num_heads_k = kp_shape[1];
    if head_dim != 128 && head_dim != 256 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: with_graph_outputs only supports head_dim=128,256, got {head_dim}"
        )));
    }
    if num_heads % num_heads_k != 0 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: with_graph_outputs num_heads ({num_heads}) % num_heads_k ({num_heads_k}) != 0"
        )));
    }
    if block_table.dtype() != KtDType::U32 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: with_graph_outputs block_table must be U32, got {:?}",
            block_table.dtype()
        )));
    }
    if seqused_k.dtype() != KtDType::U32 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: with_graph_outputs seqused_k must be U32, got {:?}",
            seqused_k.dtype()
        )));
    }
    let bt_shape = block_table.shape();
    if bt_shape.len() != 2 || bt_shape[0] != b {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: with_graph_outputs block_table must be [b, blocks], got {bt_shape:?}"
        )));
    }
    let max_blocks_per_seq = bt_shape[1];
    if seqused_k.shape() != [b] {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: with_graph_outputs seqused_k must be [b={b}], got {:?}",
            seqused_k.shape()
        )));
    }

    // Caller-owned output validation: shape + dtype must match the
    // kernel's write contract exactly (the kernel writes by index,
    // not by appending).
    let out_expected = [b, 1, num_heads, head_dim];
    if out.dtype() != KtDType::BF16 || out.shape() != out_expected {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: with_graph_outputs out must be BF16 {out_expected:?}, got {:?} {:?}",
            out.dtype(),
            out.shape()
        )));
    }
    let lse_expected = [b, num_heads, 1];
    if lse.dtype() != KtDType::F32 || lse.shape() != lse_expected {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: with_graph_outputs lse must be F32 {lse_expected:?}, got {:?} {:?}",
            lse.dtype(),
            lse.shape()
        )));
    }

    // ROCm composite dispatch (Phase R.8). There is no CUDA-graph capture on the
    // ROCm composite path; we compute the same result and copy it into the
    // caller-owned `out` buffer device-to-device so the contract (result lands
    // in `out`) holds. `lse` is left as-is (the composite decode path does not
    // surface lse, and no ROCm caller reads it back from this entry point).
    #[cfg(feature = "rocm")]
    if matches!(q.device(), KtDevice::Rocm(_)) {
        let computed = crate::rocm_sdpa::flash_attn_paged_decode_dyn_seqlen_rocm(
            q,
            k_pool,
            v_pool,
            block_table,
            seqused_k,
            max_seqlen_k,
            page_block_size,
            softmax_scale,
            causal,
        )?;
        crate::rocm_sdpa::rocm_copy_into(&computed, out)?;
        let _ = lse;
        return Ok(());
    }

    #[cfg(feature = "cuda")]
    {
        // Owner-agnostic input pointers (same pattern as the non-graph variant).
        let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::BF16, "q")?;
        let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k_pool, KtDType::BF16, "k_pool")?;
        let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v_pool, KtDType::BF16, "v_pool")?;
        let bt_ptr =
            kiln_kt_bridge::cuda_input_device_ptr(block_table, KtDType::U32, "block_table")?;
        let sk_ptr = kiln_kt_bridge::cuda_input_device_ptr(seqused_k, KtDType::U32, "seqused_k")?;
        // Caller-owned output pointers — the kernel writes through these
        // addresses. They must outlive every replay of any captured CUDA
        // graph that records this dispatch (kt-graph capture lifetime
        // contract from `kiln-graph::CaptureSession::pin`).
        let out_ptr = kiln_kt_bridge::cuda_input_device_ptr(out, KtDType::BF16, "out")?;
        let lse_ptr = kiln_kt_bridge::cuda_input_device_ptr(lse, KtDType::F32, "lse")?;
        let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;

        let raw_stream = q_st.cuda_stream_raw();

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
                "kt-flash-attn: with_graph_outputs FFI returned {status}"
            )));
        }
        return Ok(());
    }

    #[cfg(not(feature = "cuda"))]
    Err(FlashAttnError::Msg(format!(
        "kt-flash-attn: with_graph_outputs has no backend for device {:?} \
         (cuda feature off; only Device::Rocm is supported in this build)",
        q.device()
    )))
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
    // ROCm composite KV-write dispatch (Phase R.8): in-place device-to-device
    // copy of the token row into the pool at `slot`.
    #[cfg(feature = "rocm")]
    if matches!(k_pool.device(), KtDevice::Rocm(_)) {
        return crate::rocm_sdpa::paged_kv_write_token_major_bf16_rocm(
            k_pool,
            v_pool,
            k,
            v,
            slot,
            num_kv_heads,
            head_dim,
        );
    }

    #[cfg(feature = "cuda")]
    {
        let slot_u32 = u32::try_from(slot)
            .map_err(|_| FlashAttnError::Msg(format!("kt-flash-attn: slot {slot} exceeds u32")))?;
        let num_kv_heads_i32 = i32::try_from(num_kv_heads).map_err(|_| {
            FlashAttnError::Msg(format!(
                "kt-flash-attn: num_kv_heads {num_kv_heads} exceeds i32"
            ))
        })?;
        let head_dim_i32 = i32::try_from(head_dim).map_err(|_| {
            FlashAttnError::Msg(format!("kt-flash-attn: head_dim {head_dim} exceeds i32"))
        })?;

        // Owner-agnostic input pointers (Phase 7 v2). k_pool/v_pool are
        // written in place; caller convention: pass Owned for the pools.
        let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
        let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::BF16, "v")?;
        let kp_ptr = kiln_kt_bridge::cuda_input_device_ptr(k_pool, KtDType::BF16, "k_pool")?;
        let vp_ptr = kiln_kt_bridge::cuda_input_device_ptr(v_pool, KtDType::BF16, "v_pool")?;
        let (k_st, _) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;

        let raw_stream = k_st.cuda_stream_raw();
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
        return Ok(());
    }

    #[cfg(not(feature = "cuda"))]
    Err(FlashAttnError::Msg(format!(
        "kt-flash-attn: paged_kv_write_token_major_bf16_kt has no backend for device {:?} \
         (cuda feature off; only Device::Rocm is supported in this build)",
        k_pool.device()
    )))
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

    // ROCm composite KV-write (device-slot) dispatch (Phase R.8).
    #[cfg(feature = "rocm")]
    if matches!(k_pool.device(), KtDevice::Rocm(_)) {
        return crate::rocm_sdpa::paged_kv_write_token_major_bf16_slot_rocm(
            k_pool,
            v_pool,
            k,
            v,
            slot,
            num_kv_heads,
            head_dim,
        );
    }

    #[cfg(feature = "cuda")]
    {
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

        let raw_stream = k_st.cuda_stream_raw();
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
        return Ok(());
    }

    #[cfg(not(feature = "cuda"))]
    Err(FlashAttnError::Msg(format!(
        "kt-flash-attn: paged_kv_write_token_major_bf16_slot_kt has no backend for device {:?} \
         (cuda feature off; only Device::Rocm is supported in this build)",
        k_pool.device()
    )))
}

// ============================================================================
// paged_kv_write_token_major_bf16_batch_slot_kt
// ============================================================================

/// `paged_kv_write_token_major_bf16_batch_slot` (batched device-slot variant)
/// over kiln-tensor.
///
/// Mirrors [`crate::paged_kv_write_token_major_bf16_batch_slot`] one-for-one:
/// same FFI symbol (`kiln_paged_kv_write_token_major_bf16_batch_slot`), same
/// shape contract. Bottoms out in the same kernel; only the Rust shell types
/// differ.
///
/// Shapes:
/// - `k_pool`, `v_pool`: `[total_slots, num_kv_heads, head_dim]` BF16
/// - `k`, `v`: contiguous BF16 with `element_count == batch * num_kv_heads * head_dim`
/// - `slots`: U32 `[batch]` device tensor (one slot index per row)
///
/// The candle version calls `.contiguous()` on k/v/slots internally. The
/// kt path requires the caller to pass contiguous storage (validated by
/// `cuda_input_device_ptr`), matching the convention of the other
/// kt-paged_kv_write entry points.
pub fn paged_kv_write_token_major_bf16_batch_slot_kt(
    k_pool: &KtTensor,
    v_pool: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    slots: &KtTensor,
) -> Result<(), FlashAttnError> {
    let kp_shape = k_pool.shape();
    if kp_shape.len() != 3 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: kv_write_batch_slot k_pool must be rank-3, got {kp_shape:?}"
        )));
    }
    let (_, num_kv_heads, head_dim) = (kp_shape[0], kp_shape[1], kp_shape[2]);
    if v_pool.shape() != kp_shape {
        return Err(FlashAttnError::Msg(
            "kt-flash-attn: kv_write_batch_slot k/v pool mismatch".to_string(),
        ));
    }
    if slots.dtype() != KtDType::U32 || slots.shape().len() != 1 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: kv_write_batch_slot slots must be U32 rank-1, got {:?} {:?}",
            slots.dtype(),
            slots.shape()
        )));
    }
    let batch = slots.shape()[0];
    if batch == 0 {
        return Err(FlashAttnError::Msg(
            "kt-flash-attn: kv_write_batch_slot requires non-empty batch".to_string(),
        ));
    }
    let expected_per_row = num_kv_heads * head_dim;
    let expected_total = batch * expected_per_row;
    if k.element_count() != expected_total || v.element_count() != expected_total {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: kv_write_batch_slot expects batch*({num_kv_heads}*{head_dim})={expected_total} elements per K/V, got k={} v={}",
            k.element_count(),
            v.element_count()
        )));
    }
    // ROCm composite KV-write (batched device-slot) dispatch (Phase R.8).
    #[cfg(feature = "rocm")]
    if matches!(k_pool.device(), KtDevice::Rocm(_)) {
        return crate::rocm_sdpa::paged_kv_write_token_major_bf16_batch_slot_rocm(
            k_pool,
            v_pool,
            k,
            v,
            slots,
            batch,
            num_kv_heads,
            head_dim,
        );
    }

    #[cfg(feature = "cuda")]
    {
        let batch_i32 = i32::try_from(batch)
            .map_err(|_| FlashAttnError::Msg(format!("batch {batch} > i32")))?;
        let num_kv_heads_i32 = i32::try_from(num_kv_heads)
            .map_err(|_| FlashAttnError::Msg(format!("num_kv_heads {num_kv_heads} > i32")))?;
        let head_dim_i32 = i32::try_from(head_dim)
            .map_err(|_| FlashAttnError::Msg(format!("head_dim {head_dim} > i32")))?;

        // Owner-agnostic input pointers (Phase 7 v2). k_pool/v_pool are
        // written in place; caller convention: pass Owned for the pools.
        let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
        let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::BF16, "v")?;
        let kp_ptr = kiln_kt_bridge::cuda_input_device_ptr(k_pool, KtDType::BF16, "k_pool")?;
        let vp_ptr = kiln_kt_bridge::cuda_input_device_ptr(v_pool, KtDType::BF16, "v_pool")?;
        let sl_ptr = kiln_kt_bridge::cuda_input_device_ptr(slots, KtDType::U32, "slots")?;
        let (k_st, _) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;

        let raw_stream = k_st.cuda_stream_raw();
        let status = unsafe {
            kiln_paged_kv_write_token_major_bf16_batch_slot(
                kp_ptr as *mut _,
                vp_ptr as *mut _,
                k_ptr as *const _,
                v_ptr as *const _,
                sl_ptr as *const u32,
                batch_i32,
                num_kv_heads_i32,
                head_dim_i32,
                raw_stream,
            )
        };
        if status != 0 {
            return Err(FlashAttnError::Msg(format!(
                "kt-flash-attn: kv_write_batch_slot FFI returned {status}"
            )));
        }
        return Ok(());
    }

    #[cfg(not(feature = "cuda"))]
    Err(FlashAttnError::Msg(format!(
        "kt-flash-attn: paged_kv_write_token_major_bf16_batch_slot_kt has no backend for \
         device {:?} (cuda feature off; only Device::Rocm is supported in this build)",
        k_pool.device()
    )))
}

// ============================================================================
// flash_attn_bwd_kt
// ============================================================================

/// `flash_attn_bwd` over `kiln_tensor::Tensor` operands.
/// Returns `(dq, dk, dv)`. GQA expansion to num_heads happens
/// upstream (caller sums dk/dv across groups if needed); this
/// function only allocates expanded buffers matching the FFI's
/// shape contract.
///
/// CUDA defaults to the fast non-deterministic FA2 backward accumulation path.
/// Set `KILN_FLASH_ATTN_BWD_DETERMINISTIC=1` to opt into the deterministic
/// split-accumulation path for exact replay/debug runs.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
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

    // ROCm native/composite backward dispatch.
    #[cfg(feature = "rocm")]
    if matches!(q.device(), KtDevice::Rocm(_)) {
        return crate::rocm_sdpa::flash_attn_bwd_rocm(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale,
            causal,
        );
    }

    #[cfg(feature = "cuda")]
    {
        let seqlen_q_rounded = round_up(seqlen_q, 128);
        let head_dim_rounded = round_up(head_dim, 32);

        // Owner-agnostic input pointers (Phase 7 v2).
        let dout_ptr = kiln_kt_bridge::cuda_input_device_ptr(dout, KtDType::BF16, "dout")?;
        let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::BF16, "q")?;
        let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
        let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::BF16, "v")?;
        let out_ptr = kiln_kt_bridge::cuda_input_device_ptr(out, KtDType::BF16, "out")?;
        let lse_ptr =
            kiln_kt_bridge::cuda_input_device_ptr(softmax_lse, KtDType::F32, "softmax_lse")?;
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

        let raw_stream = q_st.cuda_stream_raw();
        let deterministic = flash_attn_bwd_deterministic();

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
                if deterministic { 1 } else { 0 },
                raw_stream,
            )
        };
        if status != 0 {
            return Err(FlashAttnError::Msg(format!(
                "kt-flash-attn: bwd FFI returned {status}"
            )));
        }
        return Ok((dq, dk, dv));
    }

    #[cfg(not(feature = "cuda"))]
    Err(FlashAttnError::Msg(format!(
        "kt-flash-attn: flash_attn_bwd_kt has no backend for device {:?} \
         (cuda feature off; only Device::Rocm is supported in this build)",
        q.device()
    )))
}

// Note: the previous `kt_flash_attn_regression` parity tests against the
// candle-typed `flash_attn_fwd` / `flash_attn_bwd` were removed when the
// candle-typed surface was deleted from `lib.rs` (Phase 7 / #1082). The
// kt-only smoke tests at `tests/kt_v2_smoke.rs` still exercise the FFI
// against real CUDA inputs via the candle-free
// `kiln_tensor::Tensor::cuda_from_slice` substrate.
