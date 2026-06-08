//! Metal normalization and rotary embedding helpers.
//!
//! This module owns command encoding for BF16 RMSNorm and paired Q/K rotary
//! embedding. The Metal backend facade re-exports these helpers for forward
//! code that still calls through `backend::metal::*`.

use anyhow::{Context, Result};

use super::metal_config::*;
use super::metal_core::{kt_metal, kt_metal_alloc};
use super::metal_pipeline::*;
use kiln_tensor::metal_types::buffer_o_kt;

pub(crate) fn metal_rms_norm_supports(
    x: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
) -> bool {
    if metal_rms_norm_disabled() {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(weight.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if x.dtype() != kiln_tensor::DType::BF16 || weight.dtype() != kiln_tensor::DType::BF16 {
        return false;
    }
    let Some(hidden) = x.dims().last().copied() else {
        return false;
    };
    x.rank() >= 1 && weight.dims() == [hidden] && hidden <= 8192
}

pub(crate) fn metal_rotary_embedding_supports(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    cos: &kiln_tensor::Tensor,
    sin: &kiln_tensor::Tensor,
    head_dim: usize,
    rotary_dim: usize,
) -> bool {
    if !matches!(q.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k.device(), kiln_tensor::Device::Metal(_))
        || !matches!(cos.device(), kiln_tensor::Device::Metal(_))
        || !matches!(sin.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::BF16
        || k.dtype() != kiln_tensor::DType::BF16
        || cos.dtype() != kiln_tensor::DType::F32
        || sin.dtype() != kiln_tensor::DType::F32
    {
        return false;
    }
    if !q.is_contiguous() || !k.is_contiguous() || !cos.is_contiguous() || !sin.is_contiguous() {
        return false;
    }
    let Ok((batch, seq_len, q_heads, q_head_dim)) = q.dims4() else {
        return false;
    };
    let Ok(k_dims) = k.dims4() else {
        return false;
    };
    let half_rotary = rotary_dim / 2;
    let table_batch_stride = metal_rotary_table_batch_stride(cos, sin, batch, seq_len, half_rotary);
    let Some(total_q) = batch
        .checked_mul(seq_len)
        .and_then(|n| n.checked_mul(q_heads))
        .and_then(|n| n.checked_mul(head_dim))
    else {
        return false;
    };
    let Some(total_k) = batch
        .checked_mul(seq_len)
        .and_then(|n| n.checked_mul(k_dims.2))
        .and_then(|n| n.checked_mul(head_dim))
    else {
        return false;
    };
    k_dims.0 == batch
        && k_dims.1 == seq_len
        && k_dims.3 == head_dim
        && q_head_dim == head_dim
        && rotary_dim > 0
        && rotary_dim <= head_dim
        && rotary_dim % 2 == 0
        && table_batch_stride.is_some()
        && batch <= u32::MAX as usize
        && seq_len <= u32::MAX as usize
        && q_heads <= u32::MAX as usize
        && k_dims.2 <= u32::MAX as usize
        && head_dim <= u32::MAX as usize
        && rotary_dim <= u32::MAX as usize
        && total_q <= u32::MAX as usize
        && total_k <= u32::MAX as usize
        && total_q <= (u32::MAX as usize).saturating_sub(total_k)
}

fn metal_rotary_table_batch_stride(
    cos: &kiln_tensor::Tensor,
    sin: &kiln_tensor::Tensor,
    batch: usize,
    seq_len: usize,
    half_rotary: usize,
) -> Option<usize> {
    if cos.dims() != sin.dims() {
        return None;
    }
    match cos.dims() {
        [t, r] if (*t, *r) == (seq_len, half_rotary) => Some(0),
        [b, t, r] if (*b, *t, *r) == (batch, seq_len, half_rotary) => Some(seq_len),
        _ => None,
    }
}

pub(crate) fn metal_rotary_embedding_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    cos: &kiln_tensor::Tensor,
    sin: &kiln_tensor::Tensor,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<(kiln_tensor::Tensor, kiln_tensor::Tensor)> {
    anyhow::ensure!(
        metal_rotary_embedding_supports(q, k, cos, sin, head_dim, rotary_dim),
        "metal rotary qk unsupported shape"
    );
    let (batch, seq_len, q_heads, _) = q.dims4()?;
    let (_, _, k_heads, _) = k.dims4()?;
    let table_batch_stride =
        metal_rotary_table_batch_stride(cos, sin, batch, seq_len, rotary_dim / 2)
            .context("metal rotary qk unsupported position table shape")?;
    let q_shape = q.dims().to_vec();
    let k_shape = k.dims().to_vec();
    let q_metal = kt_metal(&q)?;
    let k_metal = kt_metal(&k)?;
    // SAFETY: the kernel dispatch writes every Q output element exactly once.
    let q_out = kt_metal_alloc(q_metal, kiln_tensor::DType::BF16, q_shape.as_slice())?;
    // SAFETY: the kernel dispatch writes every K output element exactly once.
    let k_out = kt_metal_alloc(k_metal, kiln_tensor::DType::BF16, k_shape.as_slice())?;

    let companion = q_metal.companion()?;
    let pipeline = metal_rotary_qk_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_rotary_qk_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let cos_metal = kt_metal(&cos)?;
        let sin_metal = kt_metal(&sin)?;
        let q_out_metal = kt_metal(&q_out)?;
        let k_out_metal = kt_metal(&k_out)?;

        // #1082 Step 4 embedding-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let cos_buf = buffer_o_kt(cos_metal.buffer().as_ref(), cos.layout(), cos.dtype());
        let sin_buf = buffer_o_kt(sin_metal.buffer().as_ref(), sin.layout(), sin.dtype());
        let q_out_buf = buffer_o_kt(q_out_metal.buffer().as_ref(), q_out.layout(), q_out.dtype());
        let k_out_buf = buffer_o_kt(k_out_metal.buffer().as_ref(), k_out.layout(), k_out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(cos_buf.buffer), cos_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(sin_buf.buffer), sin_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(q_out_buf.buffer), q_out_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(k_out_buf.buffer), k_out_buf.offset_in_bytes);

        let batch_u32 = batch as u32;
        let seq_len_u32 = seq_len as u32;
        let q_heads_u32 = q_heads as u32;
        let k_heads_u32 = k_heads as u32;
        let head_dim_u32 = head_dim as u32;
        let rotary_dim_u32 = rotary_dim as u32;
        let total_q = batch * seq_len * q_heads * head_dim;
        let total_k = batch * seq_len * k_heads * head_dim;
        let total = total_q + total_k;
        let total_q_u32 = total_q as u32;
        let total_u32 = total as u32;
        let table_batch_stride_u32 = table_batch_stride as u32;
        encoder.set_bytes(6, &batch_u32);
        encoder.set_bytes(7, &seq_len_u32);
        encoder.set_bytes(8, &q_heads_u32);
        encoder.set_bytes(9, &k_heads_u32);
        encoder.set_bytes(10, &head_dim_u32);
        encoder.set_bytes(11, &rotary_dim_u32);
        encoder.set_bytes(12, &total_q_u32);
        encoder.set_bytes(13, &total_u32);
        encoder.set_bytes(14, &table_batch_stride_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: total,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok((q_out, k_out))
}

pub(crate) fn metal_rms_norm_bf16(
    x: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    eps: f32,
) -> Result<kiln_tensor::Tensor> {
    let x_dims = x.dims().to_vec();
    let hidden = *x_dims
        .last()
        .context("metal rmsnorm requires rank >= 1 input")?;
    anyhow::ensure!(hidden <= 8192, "metal rmsnorm hidden dim > 8192");
    let rows: usize = x_dims[..x_dims.len() - 1].iter().product();
    anyhow::ensure!(
        rows <= u32::MAX as usize && hidden <= u32::MAX as usize,
        "metal rmsnorm shape too large"
    );

    let x = x.contiguous()?;
    let weight = weight.contiguous()?;

    let x_metal = kt_metal(&x)?;
    // The kernel writes every hidden element for every row.
    let out = kt_metal_alloc(x_metal, kiln_tensor::DType::BF16, &x_dims)?;

    if rows == 0 {
        return Ok(out);
    }

    let companion = x_metal.companion()?;
    let pipeline = metal_rms_norm_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_rmsnorm_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let w_metal = kt_metal(&weight)?;
        let out_metal = kt_metal(&out)?;

        // #1082 candle-free: buffers + layout + dtype come straight off the
        // kt MetalStorage / kt Tensor — no candle storage_and_layout bridge.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let w_buf = buffer_o_kt(w_metal.buffer().as_ref(), weight.layout(), weight.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let rows_u32 = rows as u32;
        let hidden_u32 = hidden as u32;
        let threads = hidden.next_power_of_two().clamp(32, 1024);
        let threads_u32 = threads as u32;
        encoder.set_bytes(3, &rows_u32);
        encoder.set_bytes(4, &hidden_u32);
        encoder.set_bytes(5, &eps);
        encoder.set_bytes(6, &threads_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: threads,
            height: rows,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}
