//! Vulkan linear/lm-head operation helpers.
//!
//! `backend/vulkan.rs` owns the `BackendRuntime` facade. This module keeps the
//! per-submit safety policy and cached-weight decode dispatches next to the
//! linear concern without hiding the explicit VkTensor/buffer dispatch boundary.

use anyhow::{Context, Result};
use std::sync::OnceLock;

use super::vulkan::VulkanBackend;
use super::vulkan_tensor_bridge::{kt_tensor_from_f32_bytes, kt_tensor_to_f32_bytes_with_shape};
use super::{BackendMatmulLayout, requested_matmul_layout};

/// (#1082) Per-dispatch FLOP ceiling for the Vulkan-routed matmul.
///
/// Migrated inline from the deleted `backend::vulkan_linear_op` module
/// (its `candle_core::CustomOp1` training wrapper was removed when the kt
/// autograd tape became the sole grad producer). The forward-only FLCE
/// offset path in `linear_prefill_apply_offset` still needs the ceiling to
/// sub-chunk oversized dispatches: the host hard-hung twice on Strix Halo
/// when a single oversized submit (~4.36M workgroups) was queued, so the
/// ceiling caps per-submit FLOP. Tunable via `KILN_VULKAN_LINEAR_MAX_GFLOP`
/// (parsed once; `0` disables the guard).
const DEFAULT_MAX_FLOP_PER_DISPATCH: u64 = 20_000_000_000;

/// FLOP estimate for `[batch, hidden] @ [hidden, out_dim]` (one mul + one
/// add per inner term).
fn matmul_flop(batch: usize, hidden: usize, out_dim: usize) -> u64 {
    (batch as u64)
        .saturating_mul(hidden as u64)
        .saturating_mul(out_dim as u64)
        .saturating_mul(2)
}

fn max_flop_per_dispatch() -> u64 {
    static CEILING: OnceLock<u64> = OnceLock::new();
    *CEILING.get_or_init(|| {
        std::env::var("KILN_VULKAN_LINEAR_MAX_GFLOP")
            .ok()
            .as_deref()
            .map(str::trim)
            .and_then(|s| s.parse::<f64>().ok())
            .map(|gflop| {
                if gflop <= 0.0 {
                    u64::MAX
                } else {
                    (gflop * 1.0e9_f64).round() as u64
                }
            })
            .unwrap_or(DEFAULT_MAX_FLOP_PER_DISPATCH)
    })
}

/// True when the requested matmul shape would exceed the per-dispatch FLOP
/// ceiling; the caller sub-chunks via [`max_chunk_dim_for_flop`].
pub(super) fn dispatch_exceeds_safety_ceiling(batch: usize, hidden: usize, out_dim: usize) -> bool {
    matmul_flop(batch, hidden, out_dim) > max_flop_per_dispatch()
}

/// Largest `chunk_dim` such that `2 x other_dim_product x chunk_dim <=
/// max_flop_per_dispatch()`. Always >= 1; returns `usize::MAX` when the
/// guard is disabled.
pub(super) fn max_chunk_dim_for_flop(other_dim_product: usize) -> usize {
    let max_flop = max_flop_per_dispatch();
    if max_flop == u64::MAX {
        return usize::MAX;
    }
    let denom = (other_dim_product as u64).saturating_mul(2).max(1);
    let chunk = (max_flop / denom) as usize;
    chunk.max(1)
}

pub(super) fn matmul(
    backend: &VulkanBackend,
    req: &super::capability::MatmulRequest,
    lhs: &kiln_tensor::Tensor,
    rhs: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    let Some(layout) = requested_matmul_layout(req, lhs, rhs) else {
        return Ok(None);
    };

    if matches!(lhs.device(), kiln_tensor::Device::Vulkan(_))
        && matches!(rhs.device(), kiln_tensor::Device::Vulkan(_))
    {
        return resident_matmul(req, lhs, rhs, layout);
    }

    if layout == BackendMatmulLayout::Plain
        && matches!(lhs.device(), kiln_tensor::Device::Cpu)
        && matches!(rhs.device(), kiln_tensor::Device::Cpu)
        && lhs.is_contiguous()
        && rhs.is_contiguous()
        && lhs.dtype() == kiln_tensor::DType::F32
        && matches!(
            rhs.dtype(),
            kiln_tensor::DType::F32 | kiln_tensor::DType::BF16
        )
        && req.out_dtype == kiln_tensor::DType::F32
    {
        return cached_linear_matmul(backend, lhs, rhs);
    }

    Ok(None)
}

fn resident_matmul(
    req: &super::capability::MatmulRequest,
    lhs: &kiln_tensor::Tensor,
    rhs: &kiln_tensor::Tensor,
    layout: BackendMatmulLayout,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !lhs.is_contiguous()
        || !rhs.is_contiguous()
        || req.out_dtype != lhs.dtype()
        || req.lhs_dtype != req.rhs_dtype
        || !matches!(
            lhs.dtype(),
            kiln_tensor::DType::F32 | kiln_tensor::DType::BF16
        )
    {
        return Ok(None);
    }

    let out = match layout {
        BackendMatmulLayout::Plain => {
            if lhs.rank() == 2 {
                if lhs.dtype() != kiln_tensor::DType::F32 {
                    return Ok(None);
                }
                kiln_tensor::vulkan_matmul(lhs, rhs)?
            } else {
                kiln_tensor::vulkan_matmul_batched(lhs, rhs)?
            }
        }
        BackendMatmulLayout::LhsTransposed => kiln_tensor::vulkan_matmul_lhs_transposed(lhs, rhs)?,
        BackendMatmulLayout::RhsTransposed => kiln_tensor::vulkan_matmul_rhs_transposed(lhs, rhs)?,
        BackendMatmulLayout::BothTransposed => {
            let rank = lhs.rank();
            let lhs_t = lhs.transpose(rank - 2, rank - 1)?.contiguous()?;
            let rhs_t = rhs.transpose(rank - 2, rank - 1)?.contiguous()?;
            if lhs_t.rank() == 2 {
                if lhs_t.dtype() != kiln_tensor::DType::F32 {
                    return Ok(None);
                }
                kiln_tensor::vulkan_matmul(&lhs_t, &rhs_t)?
            } else {
                kiln_tensor::vulkan_matmul_batched(&lhs_t, &rhs_t)?
            }
        }
    };
    Ok(Some(out))
}

fn cached_linear_matmul(
    backend: &VulkanBackend,
    lhs: &kiln_tensor::Tensor,
    rhs: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if lhs.rank() < 2 || rhs.dims().len() != 2 {
        return Ok(None);
    }
    let l_dims = lhs.dims().to_vec();
    let hidden = *l_dims.last().unwrap();
    let Ok((weight_hidden, out_dim)) = rhs.dims2() else {
        return Ok(None);
    };
    if weight_hidden != hidden {
        return Ok(None);
    }

    let lead = l_dims[..l_dims.len() - 1].iter().product::<usize>();
    let dispatch_x = if l_dims.len() == 3 {
        lhs.clone()
    } else {
        lhs.reshape((lead, 1usize, hidden))?
    };
    let Some(out) = linear_decode(backend, &dispatch_x, rhs)? else {
        return Ok(None);
    };

    let mut out_shape = l_dims[..l_dims.len() - 1].to_vec();
    out_shape.push(out_dim);
    Ok(Some(out.reshape(out_shape.as_slice())?))
}

pub(super) fn linear_decode(
    backend: &VulkanBackend,
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    // kt guards read directly off the kt args before the bridge.
    if !backend.has_vulkan()
        || !backend.linear_decode_enabled
        || x.dtype() != kiln_tensor::DType::F32
    {
        return Ok(None);
    }
    if !matches!(x.device(), kiln_tensor::Device::Cpu)
        || !matches!(weight_t.device(), kiln_tensor::Device::Cpu)
    {
        return Ok(None);
    }
    // (#1082) Fully kt-native: read shapes off the kt tensors, extract
    // f32 bytes straight from kt storage, and key the weight buffer cache
    // on the **stable** kt `TensorId`. The old path bridged BOTH x and the
    // (large) weight through `kt_logits_to_candle` every call -- minting a
    // fresh candle id per token so the weight cache missed every step and
    // re-uploaded ~1 GB/token. Now the weight uploads exactly once.
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return Ok(None);
    };
    let Ok((weight_hidden, out_dim)) = weight_t.dims2() else {
        return Ok(None);
    };
    if weight_hidden != hidden {
        return Ok(None);
    }

    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
    let row_count = batch * seq_len;
    // x is [batch, seq_len, hidden] contiguous F32; the kernel consumes a
    // flat [row_count, hidden] f32 buffer, so the [.,1,.] reshape the candle
    // path did is a no-op on the bytes -- extract them straight from kt.
    let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
    let packed = backend.use_bf16_packed_linear_weight_kt(weight_t);
    let weight_buf = if packed {
        backend.cached_bf16_packed_weight_buffer_kt(weight_t)?
    } else {
        backend.cached_f32_weight_buffer_kt(weight_t)?
    };
    let out_data = kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached_bytes(
        vk_device,
        &x_data,
        &weight_buf,
        row_count,
        hidden,
        out_dim,
        packed,
    )
    .context("linear_decode kernel failed")?;
    Ok(Some(kt_tensor_from_f32_bytes(
        &out_data,
        &[batch, seq_len, out_dim],
        kiln_tensor::DType::F32,
    )?))
}

pub(super) fn linear_prefill_apply(
    _backend: &VulkanBackend,
    _x: &kiln_tensor::Tensor,
    _weight_t: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    // (#1082) Decline. This hook previously routed the training-time
    // projection matmul through `VulkanLinearOp` (a
    // `candle_core::CustomOp1`) so candle's `loss.backward()` could
    // produce the input gradient. With the kt autograd tape
    // (`kiln_autograd`) as the sole grad producer that candle autograd
    // island is gone -- the projection matmul is recorded onto the tape
    // by the portable kt matmul path in forward.rs, and
    // `Tape::backward()` produces the gradient. Returning `Ok(None)`
    // routes the caller to that kt-recorded path.
    //
    // NOTE: the forward-only inference linear kernel still lives in
    // `linear_decode` (declines tracked tensors); only the
    // autograd-wrapping prefill path is removed here.
    Ok(None)
}

pub(super) fn linear_prefill_apply_offset(
    backend: &VulkanBackend,
    x: &kiln_tensor::Tensor,
    full_weight_t: &kiln_tensor::Tensor,
    chunk_start: usize,
    chunk_len: usize,
) -> Result<Option<kiln_tensor::Tensor>> {
    // kt guards read directly off the kt args before the bridge.
    if !backend.has_vulkan() || !backend.linear_decode_enabled {
        return Ok(None);
    }
    if !matches!(x.device(), kiln_tensor::Device::Cpu)
        || !matches!(full_weight_t.device(), kiln_tensor::Device::Cpu)
    {
        return Ok(None);
    }
    // Only the bf16-packed kernel has an offset variant today; require
    // bf16 weights so the cached buffer matches the dispatch shader.
    if full_weight_t.dtype() != kiln_tensor::DType::BF16 {
        return Ok(None);
    }
    // (#1082) kt-native: the cached-weight offset kernel + FLOP-ceiling
    // sub-chunking run directly on the kt args (the FLCE caller owns its
    // own analytic backward, so this is forward-only).
    let Ok((_batch, _seq_len, hidden_x)) = x.dims3() else {
        return Ok(None);
    };
    let Ok((hidden_w, full_out_dim)) = full_weight_t.dims2() else {
        return Ok(None);
    };
    if hidden_x != hidden_w {
        return Ok(None);
    }
    if chunk_start + chunk_len > full_out_dim {
        return Ok(None);
    }
    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?
        .clone();
    let weight_buffer = backend.cached_bf16_packed_weight_buffer_kt(full_weight_t)?;
    // Promote x to f32 for the kernel (kernel expects f32 input).
    let x_f32 = if x.dtype() == kiln_tensor::DType::F32 {
        x.clone()
    } else {
        x.to_dtype(kiln_tensor::DType::F32)?
    };
    let dims = x_f32.dims().to_vec();
    let row_count: usize = dims[..dims.len() - 1].iter().product();
    let dispatch_x = if dims.len() == 3 && dims[1] == 1 {
        x_f32
    } else {
        x_f32.reshape((row_count, 1usize, hidden_x))?
    };
    // Per-dispatch FLOP guard. FLCE chunks at chunk_size=4096 sit
    // right at the 20 GFLOP ceiling for T=918; longer T or larger
    // chunk_len passed by future callers would put a single submit
    // over the safety limit. Sub-chunk along the chunk_len dim so
    // each submit fits -- that's strictly better than bailing to
    // FLCE's CPU fallback because each sub-chunk still uses the
    // same offset kernel with no re-upload of the weight buffer.
    let sub_chunk_len = if dispatch_exceeds_safety_ceiling(row_count, hidden_x, chunk_len) {
        max_chunk_dim_for_flop(row_count.saturating_mul(hidden_x)).min(chunk_len)
    } else {
        chunk_len
    };
    let out = if sub_chunk_len == chunk_len {
        let x_data = kt_tensor_to_f32_bytes_with_shape(&dispatch_x)?.0;
        let out_bytes =
            kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached_bf16_weights_offset_bytes(
                vk_device.as_ref(),
                &x_data,
                weight_buffer.as_ref(),
                row_count,
                hidden_x,
                chunk_len,
                chunk_start,
                full_out_dim,
            )
            .context("VulkanBackend: linear_prefill_apply_offset dispatch failed")?;
        kt_tensor_from_f32_bytes(
            &out_bytes,
            &[row_count, 1, chunk_len],
            kiln_tensor::DType::F32,
        )?
    } else {
        // One-shot trace so the operator can see when FLCE chunks
        // are themselves being sub-chunked. Combined with the
        // VulkanLinearOp chunking traces, gives a complete picture
        // of which paths are exceeding the safety ceiling.
        static FIRST_OFFSET_SUBCHUNK_LOGGED: OnceLock<()> = OnceLock::new();
        FIRST_OFFSET_SUBCHUNK_LOGGED.get_or_init(|| {
            let total_gflop = (2u64
                .saturating_mul(row_count as u64)
                .saturating_mul(hidden_x as u64)
                .saturating_mul(chunk_len as u64)) as f64
                / 1.0e9;
            let sub_count = chunk_len.div_ceil(sub_chunk_len);
            tracing::info!(
                row_count,
                hidden_x,
                chunk_len,
                full_out_dim,
                total_gflop,
                sub_chunk_len,
                sub_count,
                "linear_prefill_apply_offset first sub-chunked dispatch"
            );
        });
        // Walk chunk_len in sub_chunk_len-sized strides; concat
        // outputs along the last axis. Same kernel/buffer per
        // sub-dispatch, just different `chunk_start` offsets and
        // smaller `chunk_len` per submit.
        let mut sub_outputs: Vec<kiln_tensor::Tensor> = Vec::new();
        let mut sub_offset = 0usize;
        let x_data = kt_tensor_to_f32_bytes_with_shape(&dispatch_x)?.0;
        while sub_offset < chunk_len {
            let cur_len = (chunk_len - sub_offset).min(sub_chunk_len);
            let sub_bytes =
                kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached_bf16_weights_offset_bytes(
                    vk_device.as_ref(),
                    &x_data,
                    weight_buffer.as_ref(),
                    row_count,
                    hidden_x,
                    cur_len,
                    chunk_start + sub_offset,
                    full_out_dim,
                )
                .with_context(|| {
                    format!(
                        "VulkanBackend: linear_prefill_apply_offset sub-chunk \
                         (sub_offset={sub_offset}, cur_len={cur_len}, \
                          chunk_start={chunk_start}, chunk_len={chunk_len}) failed"
                    )
                })?;
            let sub = kt_tensor_from_f32_bytes(
                &sub_bytes,
                &[row_count, 1, cur_len],
                kiln_tensor::DType::F32,
            )?;
            sub_outputs.push(sub);
            sub_offset += cur_len;
        }
        let sub_refs: Vec<&kiln_tensor::Tensor> = sub_outputs.iter().collect();
        kiln_tensor::ops::concat(&sub_refs, 2).context("offset sub-chunk concat")?
    };
    // Output from kernel is `[row_count, 1, chunk_len]`. Restore the
    // caller's leading dims with chunk_len in the last position.
    let mut out_dims = dims;
    *out_dims.last_mut().unwrap() = chunk_len;
    let reshaped = out.reshape(out_dims.as_slice())?;
    Ok(Some(reshaped))
}

pub(super) fn supports_linear_decode_argmax(backend: &VulkanBackend) -> bool {
    backend.has_vulkan() && backend.linear_decode_enabled
}

pub(super) fn linear_decode_argmax(
    backend: &VulkanBackend,
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> Result<Option<u32>> {
    // kt guards read directly off the kt args before the bridge.
    if !backend.has_vulkan()
        || !backend.linear_decode_enabled
        || x.dtype() != kiln_tensor::DType::F32
    {
        return Ok(None);
    }
    if !matches!(x.device(), kiln_tensor::Device::Cpu)
        || !matches!(weight_t.device(), kiln_tensor::Device::Cpu)
    {
        return Ok(None);
    }
    // (#1082) Fully kt-native: the lm_head weight (the 778 MB table) was
    // re-bridged + re-uploaded per token under the candle-id cache; key on
    // the stable kt id so it uploads once.
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return Ok(None);
    };
    if batch != 1 || seq_len != 1 {
        return Ok(None);
    }
    let Ok((weight_hidden, out_dim)) = weight_t.dims2() else {
        return Ok(None);
    };
    if weight_hidden != hidden {
        return Ok(None);
    }

    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
    let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
    let token = if backend.use_bf16_packed_linear_weight_kt(weight_t) {
        let weight_buf = backend.cached_bf16_packed_weight_buffer_kt(weight_t)?;
        kiln_vulkan_kernel::kernels::dispatch_linear_decode_argmax_cached_bf16_weights_bytes(
            vk_device,
            &x_data,
            &weight_buf,
            hidden,
            out_dim,
        )
    } else {
        let weight_buf = backend.cached_f32_weight_buffer_kt(weight_t)?;
        kiln_vulkan_kernel::kernels::dispatch_linear_decode_argmax_cached_bytes(
            vk_device,
            &x_data,
            &weight_buf,
            hidden,
            out_dim,
        )
    }
    .context("linear_decode_argmax kernel failed")?;
    Ok(Some(token))
}

pub(super) fn supports_linear_decode_argmax_batch(backend: &VulkanBackend) -> bool {
    backend.has_vulkan() && backend.linear_decode_enabled && backend.linear_argmax_batch_enabled
}

pub(super) fn supports_linear_decode_sample(backend: &VulkanBackend, top_k: u32) -> bool {
    // The fused sample kernel only handles top_k in `1..=TOPK_SAMPLE_KERNEL_K_MAX`.
    // Larger requests fall back to the host sampler.
    backend.has_vulkan()
        && backend.linear_decode_enabled
        && top_k > 0
        && top_k <= kiln_vulkan_kernel::kernels::TOPK_SAMPLE_KERNEL_K_MAX
}

pub(super) fn linear_decode_sample(
    backend: &VulkanBackend,
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
    history_indices: &[u32],
    history_counts: &[u32],
    repetition_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
    seed: u64,
) -> Result<Option<u32>> {
    // kt guards read directly off the kt args before the bridge.
    if !supports_linear_decode_sample(backend, top_k) || x.dtype() != kiln_tensor::DType::F32 {
        return Ok(None);
    }
    if !matches!(x.device(), kiln_tensor::Device::Cpu)
        || !matches!(weight_t.device(), kiln_tensor::Device::Cpu)
    {
        return Ok(None);
    }
    // (#1082) Fully kt-native: lm_head weight keyed on the stable kt id.
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return Ok(None);
    };
    if batch != 1 || seq_len != 1 {
        return Ok(None);
    }
    let Ok((weight_hidden, out_dim)) = weight_t.dims2() else {
        return Ok(None);
    };
    if weight_hidden != hidden {
        return Ok(None);
    }

    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
    let packed_bf16 = backend.use_bf16_packed_linear_weight_kt(weight_t);
    let weight_buf = if packed_bf16 {
        backend.cached_bf16_packed_weight_buffer_kt(weight_t)?
    } else {
        backend.cached_f32_weight_buffer_kt(weight_t)?
    };
    let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
    let token = kiln_vulkan_kernel::kernels::dispatch_linear_decode_sample_bytes(
        vk_device,
        &x_data,
        &weight_buf,
        packed_bf16,
        hidden,
        out_dim,
        history_indices,
        history_counts,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        temperature,
        top_k,
        top_p,
        min_p,
        seed,
    )
    .context("fused linear_decode_sample dispatch failed")?;
    Ok(Some(token))
}

pub(super) fn supports_linear_decode_sample_batch(
    backend: &VulkanBackend,
    top_k: &[u32],
    temperatures: &[f32],
) -> bool {
    backend.has_vulkan()
        && backend.linear_decode_enabled
        && top_k.len() == temperatures.len()
        && !top_k.is_empty()
        && top_k.iter().zip(temperatures.iter()).all(|(&k, &temp)| {
            let greedy = temp == 0.0 || (k == 1 && temp.is_finite() && temp > 0.0);
            greedy
                || (temp.is_finite()
                    && temp > 0.0
                    && k > 0
                    && k <= kiln_vulkan_kernel::kernels::TOPK_SAMPLE_KERNEL_K_MAX)
        })
}

pub(super) fn linear_decode_sample_batch(
    backend: &VulkanBackend,
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
    history_rows: &[u32],
    history_indices: &[u32],
    history_counts: &[u32],
    repetition_penalties: &[f32],
    presence_penalties: &[f32],
    frequency_penalties: &[f32],
    temperatures: &[f32],
    top_k: &[u32],
    top_p: &[f32],
    min_p: &[f32],
    seeds: &[u64],
) -> Result<Option<Vec<u32>>> {
    if !supports_linear_decode_sample_batch(backend, top_k, temperatures)
        || x.dtype() != kiln_tensor::DType::F32
    {
        return Ok(None);
    }
    if !matches!(x.device(), kiln_tensor::Device::Cpu)
        || !matches!(weight_t.device(), kiln_tensor::Device::Cpu)
    {
        return Ok(None);
    }
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return Ok(None);
    };
    if batch == 0 || seq_len != 1 {
        return Ok(None);
    }
    let Ok((weight_hidden, out_dim)) = weight_t.dims2() else {
        return Ok(None);
    };
    if weight_hidden != hidden {
        return Ok(None);
    }

    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
    let packed_bf16 = backend.use_bf16_packed_linear_weight_kt(weight_t);
    let weight_buf = if packed_bf16 {
        backend.cached_bf16_packed_weight_buffer_kt(weight_t)?
    } else {
        backend.cached_f32_weight_buffer_kt(weight_t)?
    };
    let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
    let tokens = kiln_vulkan_kernel::kernels::dispatch_linear_decode_sample_batch_bytes(
        vk_device,
        &x_data,
        &weight_buf,
        packed_bf16,
        batch,
        hidden,
        out_dim,
        history_rows,
        history_indices,
        history_counts,
        repetition_penalties,
        presence_penalties,
        frequency_penalties,
        temperatures,
        top_k,
        top_p,
        min_p,
        seeds,
    )
    .context("fused linear_decode_sample_batch dispatch failed")?;
    Ok(Some(tokens))
}

pub(super) fn linear_decode_argmax_batch(
    backend: &VulkanBackend,
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> Result<Option<Vec<u32>>> {
    // kt guards read directly off the kt args before the bridge.
    if !backend.has_vulkan()
        || !backend.linear_decode_enabled
        || !backend.linear_argmax_batch_enabled
        || x.dtype() != kiln_tensor::DType::F32
    {
        return Ok(None);
    }
    if !matches!(x.device(), kiln_tensor::Device::Cpu)
        || !matches!(weight_t.device(), kiln_tensor::Device::Cpu)
    {
        return Ok(None);
    }
    // (#1082) Fully kt-native: lm_head weight keyed on the stable kt id.
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return Ok(None);
    };
    if batch == 0 || seq_len != 1 {
        return Ok(None);
    }
    let Ok((weight_hidden, out_dim)) = weight_t.dims2() else {
        return Ok(None);
    };
    if weight_hidden != hidden {
        return Ok(None);
    }

    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
    let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
    let tokens = if backend.use_bf16_packed_linear_weight_kt(weight_t) {
        let weight_buf = backend.cached_bf16_packed_weight_buffer_kt(weight_t)?;
        kiln_vulkan_kernel::kernels::dispatch_linear_decode_argmax_batched_cached_bf16_weights_bytes(
            vk_device,
            &x_data,
            &weight_buf,
            batch,
            hidden,
            out_dim,
        )
    } else {
        let weight_buf = backend.cached_f32_weight_buffer_kt(weight_t)?;
        kiln_vulkan_kernel::kernels::dispatch_linear_decode_argmax_batched_cached_bytes(
            vk_device,
            &x_data,
            &weight_buf,
            batch,
            hidden,
            out_dim,
        )
    }
    .context("linear_decode_argmax_batch kernel failed")?;
    Ok(Some(tokens))
}
