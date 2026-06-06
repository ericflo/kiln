//! Metal causal conv1d operation helpers.
//!
//! This module owns the plain causal conv1d prefill/update support gates and
//! dispatch helpers. GDN-specific fused QKV/conv splitting stays with the GDN
//! path until that operation family is split separately.

use anyhow::Result;

use super::metal_core::{kt_metal, kt_metal_alloc};
use super::metal_pipeline::{metal_conv1d_prefill_pipeline, metal_conv1d_update_pipeline};
use kiln_tensor::metal_types::buffer_o_kt;

pub(crate) fn metal_conv1d_prefill_supports(
    x: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    conv_state: &kiln_tensor::Tensor,
    kernel_size: usize,
) -> bool {
    if kernel_size != 4 {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_)) {
        return false;
    }
    if x.dtype() != kiln_tensor::DType::BF16
        || weight.dtype() != kiln_tensor::DType::BF16
        || conv_state.dtype() != kiln_tensor::DType::F32
    {
        return false;
    }
    let Ok((batch, channels, seq_len)) = x.dims3() else {
        return false;
    };
    if seq_len <= 1 {
        return false;
    }
    let weight_ok = match weight.rank() {
        3 => weight
            .dims3()
            .is_ok_and(|(c, one, k)| c == channels && one == 1 && k == kernel_size),
        2 => weight
            .dims2()
            .is_ok_and(|(c, k)| c == channels && k == kernel_size),
        _ => false,
    };
    if !weight_ok {
        return false;
    }
    conv_state
        .dims3()
        .is_ok_and(|(b, c, k)| (b, c, k) == (batch, channels, kernel_size - 1))
}

pub(crate) fn metal_conv1d_update_supports(
    x: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    conv_state: &kiln_tensor::Tensor,
    kernel_size: usize,
) -> bool {
    if kernel_size != 4 {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_)) {
        return false;
    }
    if x.dtype() != kiln_tensor::DType::BF16
        || weight.dtype() != kiln_tensor::DType::BF16
        || conv_state.dtype() != kiln_tensor::DType::F32
    {
        return false;
    }
    let Ok((batch, channels, seq_len)) = x.dims3() else {
        return false;
    };
    if seq_len != 1 {
        return false;
    }
    let weight_ok = match weight.rank() {
        3 => weight
            .dims3()
            .is_ok_and(|(c, one, k)| c == channels && one == 1 && k == kernel_size),
        2 => weight
            .dims2()
            .is_ok_and(|(c, k)| c == channels && k == kernel_size),
        _ => false,
    };
    if !weight_ok {
        return false;
    }
    conv_state
        .dims3()
        .is_ok_and(|(b, c, k)| (b, c, k) == (batch, channels, kernel_size - 1))
}

pub(crate) fn metal_causal_conv1d_prefill_bf16_f32_k4(
    x: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    conv_state: &mut kiln_tensor::Tensor,
    kernel_size: usize,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(kernel_size == 4, "metal conv1d prefill only supports K=4");
    let (batch, channels, seq_len) = x.dims3()?;
    anyhow::ensure!(seq_len > 1, "metal conv1d prefill requires seq_len > 1");

    let x = x.contiguous()?;
    let weight = match weight.rank() {
        3 => weight.reshape((channels, kernel_size))?,
        2 => weight.clone(),
        r => anyhow::bail!("metal conv1d prefill weight rank must be 2 or 3, got {r}"),
    }
    .contiguous()?;
    if !conv_state.is_contiguous() {
        *conv_state = conv_state.contiguous()?;
    }
    // The conv prefill kernel writes every batch/channel/time element.
    let x_metal = kt_metal(&x)?;
    let out = kt_metal_alloc(
        x_metal,
        kiln_tensor::DType::F32,
        &[batch, channels, seq_len],
    )?;

    let companion = x_metal.companion()?;
    let pipeline = metal_conv1d_prefill_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_causal_conv1d_prefill_bf16_f32_k4");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let w_metal = kt_metal(&weight)?;
        let s_metal = kt_metal(&conv_state)?;
        let o_metal = kt_metal(&out)?;

        // #1082 Step 4 conv1d-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let w_buf = buffer_o_kt(w_metal.buffer().as_ref(), weight.layout(), weight.dtype());
        let s_buf = buffer_o_kt(
            s_metal.buffer().as_ref(),
            conv_state.layout(),
            conv_state.dtype(),
        );
        let o_buf = buffer_o_kt(o_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(s_buf.buffer), s_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(o_buf.buffer), o_buf.offset_in_bytes);

        let batch_u32 = batch as u32;
        let channels_u32 = channels as u32;
        let seq_len_u32 = seq_len as u32;
        let threads = seq_len.next_power_of_two().clamp(32, 256);
        let threads_u32 = threads as u32;
        encoder.set_bytes(4, &batch_u32);
        encoder.set_bytes(5, &channels_u32);
        encoder.set_bytes(6, &seq_len_u32);
        encoder.set_bytes(7, &threads_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch * channels,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

pub(crate) fn metal_causal_conv1d_update_bf16_f32_k4(
    x: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    conv_state: &mut kiln_tensor::Tensor,
    kernel_size: usize,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(kernel_size == 4, "metal conv1d update only supports K=4");
    let (batch, channels, seq_len) = x.dims3()?;
    anyhow::ensure!(seq_len == 1, "metal conv1d update requires seq_len == 1");

    let x = x.contiguous()?;
    let weight = match weight.rank() {
        3 => weight.reshape((channels, kernel_size))?,
        2 => weight.clone(),
        r => anyhow::bail!("metal conv1d update weight rank must be 2 or 3, got {r}"),
    }
    .contiguous()?;
    if !conv_state.is_contiguous() {
        *conv_state = conv_state.contiguous()?;
    }
    // The conv update kernel writes every batch/channel element.
    let x_metal = kt_metal(&x)?;
    let out = kt_metal_alloc(x_metal, kiln_tensor::DType::F32, &[batch, channels, 1usize])?;

    let companion = x_metal.companion()?;
    let pipeline = metal_conv1d_update_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_causal_conv1d_update_bf16_f32_k4");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let w_metal = kt_metal(&weight)?;
        let s_metal = kt_metal(&conv_state)?;
        let o_metal = kt_metal(&out)?;

        // #1082 Step 4 conv1d-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let w_buf = buffer_o_kt(w_metal.buffer().as_ref(), weight.layout(), weight.dtype());
        let s_buf = buffer_o_kt(
            s_metal.buffer().as_ref(),
            conv_state.layout(),
            conv_state.dtype(),
        );
        let o_buf = buffer_o_kt(o_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(s_buf.buffer), s_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(o_buf.buffer), o_buf.offset_in_bytes);

        let batch_u32 = batch as u32;
        let channels_u32 = channels as u32;
        encoder.set_bytes(4, &batch_u32);
        encoder.set_bytes(5, &channels_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: batch * channels,
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

    Ok(out)
}
