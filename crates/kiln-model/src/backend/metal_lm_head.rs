//! Metal lm-head and sampling kernels.
//!
//! This module keeps the decode lm-head, argmax, and sampled-token operation
//! family out of the broader Metal backend runtime. The MSL source and pipeline
//! cache still live in `metal.rs` until the shared pipeline/source layer is
//! split as its own Phase 7 boundary.

use anyhow::{Context, Result};

use super::metal::{
    metal_lm_head_argmax_batch_pipeline, metal_lm_head_argmax_pipeline,
    metal_lm_head_argmax_reduce_batch_pipeline, metal_lm_head_argmax_reduce_pipeline,
    metal_lm_head_pipeline, metal_lm_head_sample_pipeline,
    metal_lm_head_sample_reduce_pipeline,
};
use super::metal_config::{
    METAL_LM_HEAD_SAMPLE_TOP_K_MAX, metal_lm_head_argmax_disabled,
    metal_lm_head_argmax_gpu_reduce_disabled, metal_lm_head_argmax_rows_disabled,
    metal_lm_head_sample_disabled,
};
use super::metal_core::{kt_metal, kt_metal_alloc};
use kiln_tensor::metal_types::buffer_o_kt;

pub(crate) fn metal_lm_head_supports(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> bool {
    if !matches!(x.dtype(), kiln_tensor::DType::BF16)
        || !matches!(weight_t.dtype(), kiln_tensor::DType::BF16)
    {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(weight_t.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !x.is_contiguous() || !weight_t.is_contiguous() {
        return false;
    }
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return false;
    };
    let Ok((weight_hidden, vocab)) = weight_t.dims2() else {
        return false;
    };
    batch == 1
        && seq_len == 1
        && hidden == weight_hidden
        && hidden <= u32::MAX as usize
        && vocab <= u32::MAX as usize
}

pub(crate) fn metal_lm_head_argmax_supports(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> bool {
    if metal_lm_head_argmax_disabled() {
        return false;
    }
    if !metal_lm_head_supports(x, weight_t) {
        return false;
    }
    let Ok((_, vocab)) = weight_t.dims2() else {
        return false;
    };
    let num_groups = vocab.div_ceil(256);
    // The final reduction is intentionally bounded to one threadgroup for the
    // Qwen3.5-4B vocab path; larger vocabs fall back to materialized logits.
    num_groups > 0 && num_groups <= 1024
}

pub(crate) fn metal_lm_head_argmax_rows_supports(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> bool {
    if metal_lm_head_argmax_rows_disabled() {
        return false;
    }
    if !matches!(x.dtype(), kiln_tensor::DType::BF16)
        || !matches!(weight_t.dtype(), kiln_tensor::DType::BF16)
    {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(weight_t.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !x.is_contiguous() || !weight_t.is_contiguous() {
        return false;
    }
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return false;
    };
    let Ok((weight_hidden, vocab)) = weight_t.dims2() else {
        return false;
    };
    let num_groups = vocab.div_ceil(256);
    batch > 0
        && seq_len == 1
        && hidden == weight_hidden
        && hidden <= u32::MAX as usize
        && vocab <= u32::MAX as usize
        && num_groups > 0
        && num_groups <= 1024
}

pub(crate) fn metal_lm_head_sample_supports(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
    top_k: u32,
    temperature: f32,
    history_len: usize,
) -> bool {
    if metal_lm_head_sample_disabled()
        || top_k == 0
        || top_k > METAL_LM_HEAD_SAMPLE_TOP_K_MAX
        || !temperature.is_finite()
        || temperature <= 0.0
        || history_len > u32::MAX as usize
    {
        return false;
    }
    if !metal_lm_head_supports(x, weight_t) {
        return false;
    }
    let Ok((_, vocab)) = weight_t.dims2() else {
        return false;
    };
    let num_groups = vocab.div_ceil(256);
    num_groups > 0 && num_groups <= 1024
}

pub(crate) fn metal_lm_head_bf16(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_lm_head_supports(x, weight_t),
        "metal lm head supports only BF16 [1,1,H] x [H,V] on Metal"
    );
    let (_, _, hidden) = x.dims3()?;
    let (_, vocab) = weight_t.dims2()?;

    let x_metal = kt_metal(x)?;
    // The kernel writes every vocab element.
    let out = kt_metal_alloc(x_metal, kiln_tensor::DType::BF16, &[1usize, 1usize, vocab])?;

    let companion = x_metal.companion()?;
    let pipeline = metal_lm_head_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_lm_head_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let w_metal = kt_metal(weight_t)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 lm_head-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let w_buf = buffer_o_kt(
            w_metal.buffer().as_ref(),
            weight_t.layout(),
            weight_t.dtype(),
        );
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let hidden_u32 = hidden as u32;
        let vocab_u32 = vocab as u32;
        encoder.set_bytes(3, &hidden_u32);
        encoder.set_bytes(4, &vocab_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: vocab,
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

pub(crate) fn metal_lm_head_argmax_bf16(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> Result<u32> {
    anyhow::ensure!(
        metal_lm_head_argmax_supports(x, weight_t),
        "metal lm head argmax supports only BF16 [1,1,H] x [H,V] on Metal with <= 262144 vocab"
    );
    let (_, _, hidden) = x.dims3()?;
    let (_, vocab) = weight_t.dims2()?;

    let chunk_width = 256usize;
    let num_groups = vocab.div_ceil(chunk_width);
    let x_metal = kt_metal(x)?;
    let partial_scores = kt_metal_alloc(x_metal, kiln_tensor::DType::F32, &[num_groups])?;
    let partial_indices = kt_metal_alloc(x_metal, kiln_tensor::DType::F32, &[num_groups])?;
    let final_index = if metal_lm_head_argmax_gpu_reduce_disabled() {
        None
    } else {
        Some(kt_metal_alloc(x_metal, kiln_tensor::DType::F32, &[1usize])?)
    };

    let companion = x_metal.companion()?;
    let pipeline = metal_lm_head_argmax_pipeline(&*companion)?;
    let reduce_pipeline = if final_index.is_some() {
        Some(metal_lm_head_argmax_reduce_pipeline(&*companion)?)
    } else {
        None
    };
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_lm_head_argmax_chunks_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let w_metal = kt_metal(weight_t)?;
        let ps_metal = kt_metal(&partial_scores)?;
        let pi_metal = kt_metal(&partial_indices)?;
        let final_metal = match final_index.as_ref() {
            Some(t) => Some((kt_metal(t)?, t)),
            None => None,
        };

        // #1082 Step 4 lm_head-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let w_buf = buffer_o_kt(
            w_metal.buffer().as_ref(),
            weight_t.layout(),
            weight_t.dtype(),
        );
        let ps_buf = buffer_o_kt(
            ps_metal.buffer().as_ref(),
            partial_scores.layout(),
            partial_scores.dtype(),
        );
        let pi_buf = buffer_o_kt(
            pi_metal.buffer().as_ref(),
            partial_indices.layout(),
            partial_indices.dtype(),
        );
        let final_buf = final_metal.map(|(storage, tensor)| {
            buffer_o_kt(
                storage.buffer().as_ref(),
                tensor.layout(),
                kiln_tensor::DType::F32,
            )
        });

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(ps_buf.buffer), ps_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(pi_buf.buffer), pi_buf.offset_in_bytes);

        let hidden_u32 = hidden as u32;
        let vocab_u32 = vocab as u32;
        encoder.set_bytes(4, &hidden_u32);
        encoder.set_bytes(5, &vocab_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: num_groups,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: chunk_width,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);

        if let (Some(reduce_pipeline), Some(final_buf)) = (&reduce_pipeline, final_buf) {
            encoder.set_label("kiln_lm_head_argmax_reduce_f32");
            encoder.set_compute_pipeline_state(reduce_pipeline);
            encoder.set_buffer(0, Some(ps_buf.buffer), ps_buf.offset_in_bytes);
            encoder.set_buffer(1, Some(pi_buf.buffer), pi_buf.offset_in_bytes);
            encoder.set_buffer(2, Some(final_buf.buffer), final_buf.offset_in_bytes);

            let num_groups_u32 = num_groups as u32;
            encoder.set_bytes(3, &num_groups_u32);

            let reduce_threadgroups = objc2_metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            };
            let reduce_threads = objc2_metal::MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_thread_groups(reduce_threadgroups, reduce_threads);
        }
    }

    // Commit the argmax dispatch before the tiny synchronous readback. The
    // default path reduces chunk winners on-GPU and reads only one scalar.
    drop(encoder);

    if let Some(final_index) = final_index {
        let token = final_index
            .to_vec1::<f32>()
            .context("read metal lm head argmax final index")?
            .into_iter()
            .next()
            .context("metal lm head argmax final index missing")?;
        return Ok(token as u32);
    }

    let scores = partial_scores
        .to_vec1::<f32>()
        .context("read metal lm head argmax partial scores")?;
    let indices = partial_indices
        .to_vec1::<f32>()
        .context("read metal lm head argmax partial indices")?;

    let mut best_score = f32::NEG_INFINITY;
    let mut best_idx = 0u32;
    for (&score, &idx_f) in scores.iter().zip(indices.iter()) {
        let idx = idx_f as u32;
        if score > best_score || (score == best_score && idx < best_idx) {
            best_score = score;
            best_idx = idx;
        }
    }
    Ok(best_idx)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_lm_head_sample_bf16(
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
) -> Result<u32> {
    anyhow::ensure!(
        history_indices.len() == history_counts.len(),
        "metal lm head sample history index/count length mismatch ({} vs {})",
        history_indices.len(),
        history_counts.len()
    );
    anyhow::ensure!(
        metal_lm_head_sample_supports(x, weight_t, top_k, temperature, history_indices.len()),
        "metal lm head sample supports BF16 [1,1,H] x [H,V] on Metal with top_k in 1..={}",
        METAL_LM_HEAD_SAMPLE_TOP_K_MAX
    );
    let (_, _, hidden) = x.dims3()?;
    let (_, vocab) = weight_t.dims2()?;

    let effective_top_k = (top_k as usize).min(vocab).max(1);
    let chunk_width = 256usize;
    let num_groups = vocab.div_ceil(chunk_width);
    let x_metal = kt_metal(x)?;
    let partial_scores = kt_metal_alloc(
        x_metal,
        kiln_tensor::DType::F32,
        &[num_groups, effective_top_k],
    )?;
    let partial_indices = kt_metal_alloc(
        x_metal,
        kiln_tensor::DType::F32,
        &[num_groups, effective_top_k],
    )?;
    let final_index = kt_metal_alloc(x_metal, kiln_tensor::DType::F32, &[1usize])?;

    let device = x.device();
    let history_indices_tensor = if history_indices.is_empty() {
        kiln_tensor::Tensor::from_vec_on(device, vec![0u32], vec![1])?
    } else {
        kiln_tensor::Tensor::from_vec_on(
            device,
            history_indices.to_vec(),
            vec![history_indices.len()],
        )?
    };
    let history_counts_tensor = if history_counts.is_empty() {
        kiln_tensor::Tensor::from_vec_on(device, vec![0u32], vec![1])?
    } else {
        kiln_tensor::Tensor::from_vec_on(
            device,
            history_counts.to_vec(),
            vec![history_counts.len()],
        )?
    };

    let companion = x_metal.companion()?;
    let pipeline = metal_lm_head_sample_pipeline(&*companion)?;
    let reduce_pipeline = metal_lm_head_sample_reduce_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_lm_head_sample_topk_chunks_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let w_metal = kt_metal(weight_t)?;
        let hist_idx_metal = kt_metal(&history_indices_tensor)?;
        let hist_count_metal = kt_metal(&history_counts_tensor)?;
        let ps_metal = kt_metal(&partial_scores)?;
        let pi_metal = kt_metal(&partial_indices)?;
        let final_metal = kt_metal(&final_index)?;

        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let w_buf = buffer_o_kt(
            w_metal.buffer().as_ref(),
            weight_t.layout(),
            weight_t.dtype(),
        );
        let hist_idx_buf = buffer_o_kt(
            hist_idx_metal.buffer().as_ref(),
            history_indices_tensor.layout(),
            history_indices_tensor.dtype(),
        );
        let hist_count_buf = buffer_o_kt(
            hist_count_metal.buffer().as_ref(),
            history_counts_tensor.layout(),
            history_counts_tensor.dtype(),
        );
        let ps_buf = buffer_o_kt(
            ps_metal.buffer().as_ref(),
            partial_scores.layout(),
            partial_scores.dtype(),
        );
        let pi_buf = buffer_o_kt(
            pi_metal.buffer().as_ref(),
            partial_indices.layout(),
            partial_indices.dtype(),
        );
        let final_buf = buffer_o_kt(
            final_metal.buffer().as_ref(),
            final_index.layout(),
            final_index.dtype(),
        );

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(hist_idx_buf.buffer), hist_idx_buf.offset_in_bytes);
        encoder.set_buffer(
            3,
            Some(hist_count_buf.buffer),
            hist_count_buf.offset_in_bytes,
        );
        encoder.set_buffer(4, Some(ps_buf.buffer), ps_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(pi_buf.buffer), pi_buf.offset_in_bytes);

        let hidden_u32 = hidden as u32;
        let vocab_u32 = vocab as u32;
        let history_len_u32 = history_indices.len() as u32;
        let inv_temperature = 1.0f32 / temperature;
        let effective_top_k_u32 = effective_top_k as u32;
        encoder.set_bytes(6, &hidden_u32);
        encoder.set_bytes(7, &vocab_u32);
        encoder.set_bytes(8, &history_len_u32);
        encoder.set_bytes(9, &repetition_penalty);
        encoder.set_bytes(10, &presence_penalty);
        encoder.set_bytes(11, &frequency_penalty);
        encoder.set_bytes(12, &inv_temperature);
        encoder.set_bytes(13, &effective_top_k_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: num_groups,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: chunk_width,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);

        encoder.set_label("kiln_lm_head_sample_reduce_f32");
        encoder.set_compute_pipeline_state(&reduce_pipeline);
        encoder.set_buffer(0, Some(ps_buf.buffer), ps_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(pi_buf.buffer), pi_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(final_buf.buffer), final_buf.offset_in_bytes);

        let num_groups_u32 = num_groups as u32;
        let seed_lo = seed as u32;
        let seed_hi = (seed >> 32) as u32;
        encoder.set_bytes(3, &num_groups_u32);
        encoder.set_bytes(4, &effective_top_k_u32);
        encoder.set_bytes(5, &top_p);
        encoder.set_bytes(6, &min_p);
        encoder.set_bytes(7, &seed_lo);
        encoder.set_bytes(8, &seed_hi);

        let reduce_threadgroups = objc2_metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let reduce_threads = objc2_metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(reduce_threadgroups, reduce_threads);
    }

    drop(encoder);

    let token = final_index
        .to_vec1::<f32>()
        .context("read metal lm head sampled final index")?
        .into_iter()
        .next()
        .context("metal lm head sampled final index missing")?;
    Ok(token as u32)
}

pub(crate) fn metal_lm_head_argmax_rows_bf16(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> Result<Vec<u32>> {
    anyhow::ensure!(
        metal_lm_head_argmax_rows_supports(x, weight_t),
        "metal lm head row argmax supports only BF16 [B,1,H] x [H,V] on Metal with <= 262144 vocab"
    );
    let (batch, _, hidden) = x.dims3()?;
    let (_, vocab) = weight_t.dims2()?;

    let chunk_width = 256usize;
    let num_groups = vocab.div_ceil(chunk_width);
    let x_metal = kt_metal(x)?;
    let partial_scores = kt_metal_alloc(x_metal, kiln_tensor::DType::F32, &[batch, num_groups])?;
    let partial_indices = kt_metal_alloc(x_metal, kiln_tensor::DType::F32, &[batch, num_groups])?;
    let final_indices = if metal_lm_head_argmax_gpu_reduce_disabled() {
        None
    } else {
        Some(kt_metal_alloc(x_metal, kiln_tensor::DType::F32, &[batch])?)
    };

    let companion = x_metal.companion()?;
    let pipeline = metal_lm_head_argmax_batch_pipeline(&*companion)?;
    let reduce_pipeline = if final_indices.is_some() {
        Some(metal_lm_head_argmax_reduce_batch_pipeline(&*companion)?)
    } else {
        None
    };
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_lm_head_argmax_chunks_batch_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let w_metal = kt_metal(weight_t)?;
        let ps_metal = kt_metal(&partial_scores)?;
        let pi_metal = kt_metal(&partial_indices)?;
        let final_metal = match final_indices.as_ref() {
            Some(t) => Some((kt_metal(t)?, t)),
            None => None,
        };

        // #1082 Step 4 lm_head-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let w_buf = buffer_o_kt(
            w_metal.buffer().as_ref(),
            weight_t.layout(),
            weight_t.dtype(),
        );
        let ps_buf = buffer_o_kt(
            ps_metal.buffer().as_ref(),
            partial_scores.layout(),
            partial_scores.dtype(),
        );
        let pi_buf = buffer_o_kt(
            pi_metal.buffer().as_ref(),
            partial_indices.layout(),
            partial_indices.dtype(),
        );
        let final_buf = final_metal.map(|(storage, tensor)| {
            buffer_o_kt(
                storage.buffer().as_ref(),
                tensor.layout(),
                kiln_tensor::DType::F32,
            )
        });

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(ps_buf.buffer), ps_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(pi_buf.buffer), pi_buf.offset_in_bytes);

        let hidden_u32 = hidden as u32;
        let vocab_u32 = vocab as u32;
        let num_groups_u32 = num_groups as u32;
        encoder.set_bytes(4, &hidden_u32);
        encoder.set_bytes(5, &vocab_u32);
        encoder.set_bytes(6, &num_groups_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: num_groups,
            height: batch,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: chunk_width,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);

        if let (Some(reduce_pipeline), Some(final_buf)) = (&reduce_pipeline, final_buf) {
            encoder.set_label("kiln_lm_head_argmax_reduce_batch_f32");
            encoder.set_compute_pipeline_state(reduce_pipeline);
            encoder.set_buffer(0, Some(ps_buf.buffer), ps_buf.offset_in_bytes);
            encoder.set_buffer(1, Some(pi_buf.buffer), pi_buf.offset_in_bytes);
            encoder.set_buffer(2, Some(final_buf.buffer), final_buf.offset_in_bytes);
            encoder.set_bytes(3, &num_groups_u32);

            let reduce_threadgroups = objc2_metal::MTLSize {
                width: batch,
                height: 1,
                depth: 1,
            };
            let reduce_threads = objc2_metal::MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_thread_groups(reduce_threadgroups, reduce_threads);
        }
    }

    drop(encoder);

    if let Some(final_indices) = final_indices {
        return Ok(final_indices
            .to_vec1::<f32>()
            .context("read metal lm head row argmax final indices")?
            .into_iter()
            .map(|idx| idx as u32)
            .collect());
    }

    let scores = partial_scores
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read metal lm head row argmax partial scores")?;
    let indices = partial_indices
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read metal lm head row argmax partial indices")?;
    let mut out = Vec::with_capacity(batch);
    for row in 0..batch {
        let row_start = row * num_groups;
        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx = 0u32;
        for group in 0..num_groups {
            let offset = row_start + group;
            let score = scores[offset];
            let idx = indices[offset] as u32;
            if score > best_score || (score == best_score && idx < best_idx) {
                best_score = score;
                best_idx = idx;
            }
        }
        out.push(best_idx);
    }
    Ok(out)
}
