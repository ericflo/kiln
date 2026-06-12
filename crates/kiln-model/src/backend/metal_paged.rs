//! Metal paged-attention, paged-KV, and paged decode ICB helpers.
//!
//! This module owns the operation-family command encoding for token-major KV
//! writes, head-major KV reads, contiguous paged attention decode, and the ICB
//! graph capture helpers built from those primitives. Pipeline construction
//! remains in `metal_pipeline` and the backend trait glue remains in `metal`.

use anyhow::Result;

use super::metal_config::{
    metal_paged_attn_decode_contiguous_disabled, metal_paged_kv_write_token_major_disabled,
};
use super::metal_core::{kt_metal, kt_metal_alloc};
use super::metal_icb::{
    MetalGraphResourceRef, MetalPagedAttnDecodeDynSeqlenIcbArgs,
    MetalPagedAttnDecodeDynSeqlenScalars, MetalPagedDecodeIcbGraph,
    MetalPagedKvWriteTokenMajorBatchIcbArgs, MetalPagedKvWriteTokenMajorIcbArgs,
    MetalSingleTokenPagedDecodeIcbGraph, merge_metal_graph_resources,
    metal_paged_decode_replay_state,
};
use super::metal_pipeline::{
    metal_paged_attn_decode_contiguous_batch_dyn_seqlen_pipeline,
    metal_paged_attn_decode_contiguous_batch_dyn_seqlen_pipeline_indirect,
    metal_paged_attn_decode_contiguous_batch_pipeline, metal_paged_attn_decode_contiguous_pipeline,
    metal_paged_kv_head_major_read_append_token_major_pipeline,
    metal_paged_kv_head_major_read_pipeline, metal_paged_kv_write_token_major_batch_pipeline,
    metal_paged_kv_write_token_major_batch_pipeline_indirect,
    metal_paged_kv_write_token_major_pipeline, metal_paged_kv_write_token_major_pipeline_indirect,
};
use kiln_tensor::metal_types::{
    BufferOffset, IndirectCommandBufferDescriptor, IndirectComputeCommand, IndirectDispatchKind,
    MTLResourceOptions, buffer_o_kt,
};

#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn metal_record_paged_decode_icb_graph(
    q: &kiln_tensor::Tensor,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    block_table: &kiln_tensor::Tensor,
    seqused_k: &kiln_tensor::Tensor,
    out: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    slots: &kiln_tensor::Tensor,
    max_seqlen_k: usize,
    page_block_size: usize,
    softmax_scale: f32,
) -> Result<MetalPagedDecodeIcbGraph> {
    let q_metal = kt_metal(q)?;
    let companion = q_metal.companion()?;
    let (batch, _, q_heads, _) = q.dims4()?;
    let (total_slots, kv_heads, head_dim) = k_pool.dims3()?;
    let (_, max_blocks_per_seq) = block_table.dims2()?;

    anyhow::ensure!(batch > 0, "Metal paged decode ICB graph requires batch > 0");
    anyhow::ensure!(
        slots.dims1()? == batch,
        "Metal paged decode ICB graph slots length must match batch"
    );

    let descriptor = IndirectCommandBufferDescriptor {
        max_kernel_buffer_bind_count: 14,
        dispatch_kind: IndirectDispatchKind::ThreadgroupsAndThreads,
        ..Default::default()
    };
    let icb = companion
        .device()
        .new_indirect_command_buffer(descriptor, 2, MTLResourceOptions::StorageModePrivate)
        .map_err(|e| anyhow::anyhow!("create Metal paged decode ICB: {e:?}"))?;
    icb.reset(0, 2);

    let kv_args = MetalPagedKvWriteTokenMajorBatchIcbArgs::new(
        &companion,
        batch as u32,
        kv_heads as u32,
        head_dim as u32,
        total_slots as u32,
    )?;
    let attn_args = MetalPagedAttnDecodeDynSeqlenIcbArgs::new(
        &companion,
        MetalPagedAttnDecodeDynSeqlenScalars {
            batch: batch as u32,
            max_blocks_per_seq: max_blocks_per_seq as u32,
            max_seqlen_k: max_seqlen_k as u32,
            page_block_size: page_block_size as u32,
            q_heads: q_heads as u32,
            kv_heads: kv_heads as u32,
            softmax_scale,
            total_slots: total_slots as u32,
        },
    )?;

    let kv_resources = metal_record_paged_kv_write_token_major_batch_bf16_icb(
        &icb.compute_command_at(0),
        &kv_args,
        k_pool,
        v_pool,
        slots,
        k,
        v,
    )?;
    let attn_resources = metal_record_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256_icb(
        &icb.compute_command_at(1),
        &attn_args,
        q,
        k_pool,
        v_pool,
        block_table,
        seqused_k,
        out,
        max_seqlen_k,
        page_block_size,
    )?;

    let resources = merge_metal_graph_resources(kv_resources.into_iter().chain(attn_resources))?;
    let captured = kiln_graph_metal::MetalCapturedGraph::from_indirect_commands_with_resources(
        (*companion).clone(),
        icb,
        2,
        0,
        resources,
    )
    .map_err(|e| anyhow::anyhow!("capture Metal paged decode ICB graph: {e}"))?;

    Ok(MetalPagedDecodeIcbGraph {
        captured,
        attn_args,
        replay_state: metal_paged_decode_replay_state(
            q,
            k_pool,
            v_pool,
            block_table,
            seqused_k,
            out,
            k,
            v,
            slots,
            max_seqlen_k,
            page_block_size,
        )?,
    })
}

#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn metal_record_single_token_paged_decode_icb_graph(
    q: &kiln_tensor::Tensor,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    block_table: &kiln_tensor::Tensor,
    seqused_k: &kiln_tensor::Tensor,
    out: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    slot: usize,
    max_seqlen_k: usize,
    page_block_size: usize,
    softmax_scale: f32,
) -> Result<MetalSingleTokenPagedDecodeIcbGraph> {
    let q_metal = kt_metal(q)?;
    let companion = q_metal.companion()?;
    let (_, kv_heads, head_dim) = k_pool.dims3()?;
    let (batch, _, q_heads, _) = q.dims4()?;
    let (total_slots, _, _) = k_pool.dims3()?;
    let (_, max_blocks_per_seq) = block_table.dims2()?;

    anyhow::ensure!(batch == 1, "Metal single-token ICB graph requires batch=1");
    anyhow::ensure!(slot <= u32::MAX as usize, "Metal ICB KV slot exceeds u32");

    let descriptor = IndirectCommandBufferDescriptor {
        max_kernel_buffer_bind_count: 14,
        dispatch_kind: IndirectDispatchKind::Threadgroups,
        ..Default::default()
    };
    let icb = companion
        .device()
        .new_indirect_command_buffer(descriptor, 2, MTLResourceOptions::StorageModePrivate)
        .map_err(|e| anyhow::anyhow!("create Metal paged decode ICB: {e:?}"))?;
    icb.reset(0, 2);

    let kv_args = MetalPagedKvWriteTokenMajorIcbArgs::new(
        &companion,
        slot as u32,
        kv_heads as u32,
        head_dim as u32,
    )?;
    let attn_args = MetalPagedAttnDecodeDynSeqlenIcbArgs::new(
        &companion,
        MetalPagedAttnDecodeDynSeqlenScalars {
            batch: batch as u32,
            max_blocks_per_seq: max_blocks_per_seq as u32,
            max_seqlen_k: max_seqlen_k as u32,
            page_block_size: page_block_size as u32,
            q_heads: q_heads as u32,
            kv_heads: kv_heads as u32,
            softmax_scale,
            total_slots: total_slots as u32,
        },
    )?;

    let kv_resources = metal_record_paged_kv_write_token_major_bf16_icb(
        &icb.compute_command_at(0),
        &kv_args,
        k_pool,
        v_pool,
        slot,
        k,
        v,
    )?;
    let attn_resources = metal_record_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256_icb(
        &icb.compute_command_at(1),
        &attn_args,
        q,
        k_pool,
        v_pool,
        block_table,
        seqused_k,
        out,
        max_seqlen_k,
        page_block_size,
    )?;

    let resources = merge_metal_graph_resources(kv_resources.into_iter().chain(attn_resources))?;
    let captured = kiln_graph_metal::MetalCapturedGraph::from_indirect_commands_with_resources(
        (*companion).clone(),
        icb,
        2,
        0,
        resources,
    )
    .map_err(|e| anyhow::anyhow!("capture Metal single-token paged decode ICB graph: {e}"))?;

    Ok(MetalSingleTokenPagedDecodeIcbGraph {
        captured,
        kv_args,
        attn_args,
    })
}

pub(super) fn metal_paged_kv_head_major_read_supports(
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    start_slot: usize,
    seq_len: usize,
) -> bool {
    if seq_len == 0
        || k_pool.dtype() != kiln_tensor::DType::BF16
        || v_pool.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    if !matches!(k_pool.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v_pool.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !k_pool.is_contiguous() || !v_pool.is_contiguous() {
        return false;
    }
    let Ok((total_slots, heads, head_dim)) = k_pool.dims3() else {
        return false;
    };
    let Ok(v_dims) = v_pool.dims3() else {
        return false;
    };
    let Some(total) = seq_len
        .checked_mul(heads)
        .and_then(|n| n.checked_mul(head_dim))
    else {
        return false;
    };
    v_dims == (total_slots, heads, head_dim)
        && start_slot <= total_slots
        && seq_len <= total_slots.saturating_sub(start_slot)
        && total <= u32::MAX as usize
        && seq_len <= u32::MAX as usize
        && heads <= u32::MAX as usize
        && head_dim <= u32::MAX as usize
        && start_slot <= u32::MAX as usize
}

pub(super) fn metal_paged_kv_head_major_read_bf16(
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    start_slot: usize,
    seq_len: usize,
) -> Result<(kiln_tensor::Tensor, kiln_tensor::Tensor)> {
    anyhow::ensure!(
        metal_paged_kv_head_major_read_supports(k_pool, v_pool, start_slot, seq_len),
        "metal paged kv head-major read unsupported shape"
    );
    let (_, heads, head_dim) = k_pool.dims3()?;
    let out_shape = (1usize, heads, seq_len, head_dim);
    let k_pool_metal = kt_metal(k_pool)?;
    let v_pool_metal = kt_metal(v_pool)?;
    // SAFETY: the kernel dispatch covers exactly every element in `out_shape`.
    let k_out = kt_metal_alloc(
        k_pool_metal,
        kiln_tensor::DType::BF16,
        &[out_shape.0, out_shape.1, out_shape.2, out_shape.3],
    )?;
    // SAFETY: the kernel dispatch covers exactly every element in `out_shape`.
    let v_out = kt_metal_alloc(
        v_pool_metal,
        kiln_tensor::DType::BF16,
        &[out_shape.0, out_shape.1, out_shape.2, out_shape.3],
    )?;

    let companion = k_pool_metal.companion()?;
    let pipeline = metal_paged_kv_head_major_read_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_paged_kv_head_major_read_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let ko_metal = kt_metal(&k_out)?;
        let vo_metal = kt_metal(&v_out)?;

        // #1082 candle-free: buffers + layout + dtype come straight off the
        // kt MetalStorage / kt Tensor — no candle storage_and_layout bridge.
        let k_buf = buffer_o_kt(
            k_pool_metal.buffer().as_ref(),
            k_pool.layout(),
            k_pool.dtype(),
        );
        let v_buf = buffer_o_kt(
            v_pool_metal.buffer().as_ref(),
            v_pool.layout(),
            v_pool.dtype(),
        );
        let ko_buf = buffer_o_kt(ko_metal.buffer().as_ref(), k_out.layout(), k_out.dtype());
        let vo_buf = buffer_o_kt(vo_metal.buffer().as_ref(), v_out.layout(), v_out.dtype());

        encoder.set_buffer(0, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(ko_buf.buffer), ko_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(vo_buf.buffer), vo_buf.offset_in_bytes);

        let start_slot_u32 = start_slot as u32;
        let seq_len_u32 = seq_len as u32;
        let heads_u32 = heads as u32;
        let head_dim_u32 = head_dim as u32;
        encoder.set_bytes(4, &start_slot_u32);
        encoder.set_bytes(5, &seq_len_u32);
        encoder.set_bytes(6, &heads_u32);
        encoder.set_bytes(7, &head_dim_u32);

        let total = seq_len * heads * head_dim;
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

    Ok((k_out, v_out))
}

pub(super) fn metal_paged_attn_decode_contiguous_supports(
    q: &kiln_tensor::Tensor,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    start_slot: usize,
    seq_len: usize,
) -> bool {
    if metal_paged_attn_decode_contiguous_disabled() {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::BF16
        || k_pool.dtype() != kiln_tensor::DType::BF16
        || v_pool.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    if !matches!(q.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k_pool.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v_pool.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !q.is_contiguous() || !k_pool.is_contiguous() || !v_pool.is_contiguous() {
        return false;
    }
    let Ok((batch, q_heads, q_len, head_dim)) = q.dims4() else {
        return false;
    };
    let Ok((total_slots, kv_heads, k_head_dim)) = k_pool.dims3() else {
        return false;
    };
    let Ok(v_dims) = v_pool.dims3() else {
        return false;
    };
    let Some(end_slot) = start_slot.checked_add(seq_len) else {
        return false;
    };
    batch == 1
        && q_heads == 16
        && kv_heads == 4
        && q_len == 1
        && head_dim == 256
        && k_head_dim == head_dim
        && v_dims == (total_slots, kv_heads, head_dim)
        && seq_len > 0
        && end_slot <= total_slots
        && start_slot <= u32::MAX as usize
        && seq_len <= u32::MAX as usize
        && q_heads <= u32::MAX as usize
        && kv_heads <= u32::MAX as usize
}

pub(super) fn metal_paged_attn_decode_contiguous_bf16_d256(
    q: &kiln_tensor::Tensor,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    start_slot: usize,
    seq_len: usize,
    softmax_scale: f32,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_paged_attn_decode_contiguous_supports(q, k_pool, v_pool, start_slot, seq_len),
        "metal contiguous paged decode attention unsupported shape"
    );
    let (_, q_heads, _, head_dim) = q.dims4()?;
    let q_metal = kt_metal(q)?;
    // SAFETY: the kernel writes one contiguous [1, 1, q_heads * head_dim] output.
    let out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[1usize, 1usize, q_heads * head_dim],
    )?;

    let companion = q_metal.companion()?;
    let pipeline = metal_paged_attn_decode_contiguous_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_paged_attn_decode_contiguous_bf16_d256");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(k_pool)?;
        let v_metal = kt_metal(v_pool)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 paged_kv-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k_pool.layout(), k_pool.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v_pool.layout(), v_pool.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let start_slot_u32 = start_slot as u32;
        let seq_len_u32 = seq_len as u32;
        let q_heads_u32 = q_heads as u32;
        let kv_heads_u32 = 4u32;
        encoder.set_bytes(4, &start_slot_u32);
        encoder.set_bytes(5, &seq_len_u32);
        encoder.set_bytes(6, &q_heads_u32);
        encoder.set_bytes(7, &kv_heads_u32);
        encoder.set_bytes(8, &softmax_scale);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: 1,
            height: q_heads,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

#[allow(dead_code)]
pub(super) fn metal_paged_attn_decode_contiguous_batch_supports(
    q: &kiln_tensor::Tensor,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    start_slots: &kiln_tensor::Tensor,
    seq_len: usize,
) -> bool {
    if metal_paged_attn_decode_contiguous_disabled() {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::BF16
        || k_pool.dtype() != kiln_tensor::DType::BF16
        || v_pool.dtype() != kiln_tensor::DType::BF16
        || start_slots.dtype() != kiln_tensor::DType::U32
    {
        return false;
    }
    if !matches!(q.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k_pool.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v_pool.device(), kiln_tensor::Device::Metal(_))
        || !matches!(start_slots.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !q.is_contiguous()
        || !k_pool.is_contiguous()
        || !v_pool.is_contiguous()
        || !start_slots.is_contiguous()
    {
        return false;
    }
    let Ok((batch, q_heads, q_len, head_dim)) = q.dims4() else {
        return false;
    };
    let Ok((total_slots, kv_heads, k_head_dim)) = k_pool.dims3() else {
        return false;
    };
    let Ok(v_dims) = v_pool.dims3() else {
        return false;
    };
    let Ok(slot_count) = start_slots.dims1() else {
        return false;
    };
    batch > 0
        && q_heads == 16
        && kv_heads == 4
        && q_len == 1
        && head_dim == 256
        && k_head_dim == head_dim
        && v_dims == (total_slots, kv_heads, head_dim)
        && slot_count == batch
        && seq_len > 0
        && batch <= u32::MAX as usize
        && seq_len <= u32::MAX as usize
        && q_heads <= u32::MAX as usize
        && kv_heads <= u32::MAX as usize
        && total_slots <= u32::MAX as usize
}

#[allow(dead_code)]
pub(super) fn metal_paged_attn_decode_contiguous_batch_bf16_d256(
    q: &kiln_tensor::Tensor,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    start_slots: &kiln_tensor::Tensor,
    seq_len: usize,
    softmax_scale: f32,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_paged_attn_decode_contiguous_batch_supports(q, k_pool, v_pool, start_slots, seq_len),
        "metal contiguous paged batch decode attention unsupported shape"
    );
    let (batch, q_heads, _, head_dim) = q.dims4()?;
    let (total_slots, _, _) = k_pool.dims3()?;
    let q_metal = kt_metal(q)?;
    let out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[batch, 1usize, q_heads * head_dim],
    )?;

    let companion = q_metal.companion()?;
    let pipeline = metal_paged_attn_decode_contiguous_batch_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_paged_attn_decode_contiguous_batch_bf16_d256");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(k_pool)?;
        let v_metal = kt_metal(v_pool)?;
        let out_metal = kt_metal(&out)?;
        let slot_metal = kt_metal(start_slots)?;

        // #1082 Step 4 paged_kv-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k_pool.layout(), k_pool.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v_pool.layout(), v_pool.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());
        let slot_buf = buffer_o_kt(
            slot_metal.buffer().as_ref(),
            start_slots.layout(),
            start_slots.dtype(),
        );

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(out_buf.buffer), out_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(slot_buf.buffer), slot_buf.offset_in_bytes);

        let batch_u32 = batch as u32;
        let seq_len_u32 = seq_len as u32;
        let q_heads_u32 = q_heads as u32;
        let kv_heads_u32 = 4u32;
        let total_slots_u32 = total_slots as u32;
        encoder.set_bytes(5, &batch_u32);
        encoder.set_bytes(6, &seq_len_u32);
        encoder.set_bytes(7, &q_heads_u32);
        encoder.set_bytes(8, &kv_heads_u32);
        encoder.set_bytes(9, &softmax_scale);
        encoder.set_bytes(10, &total_slots_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch,
            height: q_heads,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

#[allow(dead_code)]
pub(super) fn metal_paged_attn_decode_contiguous_batch_dyn_seqlen_supports(
    q: &kiln_tensor::Tensor,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    block_table: &kiln_tensor::Tensor,
    seqused_k: &kiln_tensor::Tensor,
    max_seqlen_k: usize,
    page_block_size: usize,
) -> bool {
    if metal_paged_attn_decode_contiguous_disabled() {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::BF16
        || k_pool.dtype() != kiln_tensor::DType::BF16
        || v_pool.dtype() != kiln_tensor::DType::BF16
        || block_table.dtype() != kiln_tensor::DType::U32
        || seqused_k.dtype() != kiln_tensor::DType::U32
    {
        return false;
    }
    if !matches!(q.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k_pool.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v_pool.device(), kiln_tensor::Device::Metal(_))
        || !matches!(block_table.device(), kiln_tensor::Device::Metal(_))
        || !matches!(seqused_k.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !q.is_contiguous()
        || !k_pool.is_contiguous()
        || !v_pool.is_contiguous()
        || !block_table.is_contiguous()
        || !seqused_k.is_contiguous()
    {
        return false;
    }
    let Ok((batch, q_len, q_heads, head_dim)) = q.dims4() else {
        return false;
    };
    let Ok((total_slots, kv_heads, k_head_dim)) = k_pool.dims3() else {
        return false;
    };
    let Ok(v_dims) = v_pool.dims3() else {
        return false;
    };
    let Ok((table_batch, max_blocks_per_seq)) = block_table.dims2() else {
        return false;
    };
    let Ok(seq_rows) = seqused_k.dims1() else {
        return false;
    };
    batch > 0
        && q_len == 1
        && q_heads == 16
        && kv_heads == 4
        && head_dim == 256
        && k_head_dim == head_dim
        && v_dims == (total_slots, kv_heads, head_dim)
        && table_batch == batch
        && seq_rows == batch
        && max_blocks_per_seq > 0
        && max_seqlen_k > 0
        && page_block_size > 0
        && max_blocks_per_seq <= u32::MAX as usize
        && max_seqlen_k <= u32::MAX as usize
        && page_block_size <= u32::MAX as usize
        && batch <= u32::MAX as usize
        && q_heads <= u32::MAX as usize
        && kv_heads <= u32::MAX as usize
        && total_slots <= u32::MAX as usize
}

#[allow(dead_code)]
pub(super) fn metal_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256(
    q: &kiln_tensor::Tensor,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    block_table: &kiln_tensor::Tensor,
    seqused_k: &kiln_tensor::Tensor,
    max_seqlen_k: usize,
    page_block_size: usize,
    softmax_scale: f32,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_paged_attn_decode_contiguous_batch_dyn_seqlen_supports(
            q,
            k_pool,
            v_pool,
            block_table,
            seqused_k,
            max_seqlen_k,
            page_block_size,
        ),
        "metal dyn-seqlen paged batch decode attention unsupported shape"
    );
    let (batch, _, q_heads, head_dim) = q.dims4()?;
    let q_metal = kt_metal(q)?;
    let out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[batch, 1usize, q_heads, head_dim],
    )?;

    metal_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256_into(
        q,
        k_pool,
        v_pool,
        block_table,
        seqused_k,
        &out,
        max_seqlen_k,
        page_block_size,
        softmax_scale,
    )?;

    Ok(out)
}

#[allow(dead_code)]
pub(super) fn metal_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256_into(
    q: &kiln_tensor::Tensor,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    block_table: &kiln_tensor::Tensor,
    seqused_k: &kiln_tensor::Tensor,
    out: &kiln_tensor::Tensor,
    max_seqlen_k: usize,
    page_block_size: usize,
    softmax_scale: f32,
) -> Result<()> {
    anyhow::ensure!(
        metal_paged_attn_decode_contiguous_batch_dyn_seqlen_supports(
            q,
            k_pool,
            v_pool,
            block_table,
            seqused_k,
            max_seqlen_k,
            page_block_size,
        ),
        "metal dyn-seqlen paged batch decode attention unsupported shape"
    );
    let (batch, _, q_heads, head_dim) = q.dims4()?;
    let (total_slots, _, _) = k_pool.dims3()?;
    let (_, max_blocks_per_seq) = block_table.dims2()?;
    anyhow::ensure!(
        out.dtype() == kiln_tensor::DType::BF16
            && matches!(out.device(), kiln_tensor::Device::Metal(_))
            && out.is_contiguous()
            && out.dims() == [batch, 1usize, q_heads, head_dim],
        "metal dyn-seqlen paged batch decode graph output must be contiguous BF16 [batch,1,q_heads,head_dim] on Metal"
    );
    let q_metal = kt_metal(q)?;
    let companion = q_metal.companion()?;
    let pipeline = metal_paged_attn_decode_contiguous_batch_dyn_seqlen_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(k_pool)?;
        let v_metal = kt_metal(v_pool)?;
        let out_metal = kt_metal(&out)?;
        let table_metal = kt_metal(block_table)?;
        let seq_metal = kt_metal(seqused_k)?;

        // #1082 Step 4 paged_kv-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k_pool.layout(), k_pool.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v_pool.layout(), v_pool.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());
        let table_buf = buffer_o_kt(
            table_metal.buffer().as_ref(),
            block_table.layout(),
            block_table.dtype(),
        );
        let seq_buf = buffer_o_kt(
            seq_metal.buffer().as_ref(),
            seqused_k.layout(),
            seqused_k.dtype(),
        );

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(out_buf.buffer), out_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(table_buf.buffer), table_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(seq_buf.buffer), seq_buf.offset_in_bytes);

        let batch_u32 = batch as u32;
        let max_blocks_u32 = max_blocks_per_seq as u32;
        let max_seqlen_u32 = max_seqlen_k as u32;
        let page_block_size_u32 = page_block_size as u32;
        let q_heads_u32 = q_heads as u32;
        let kv_heads_u32 = 4u32;
        let total_slots_u32 = total_slots as u32;
        encoder.set_bytes(6, &batch_u32);
        encoder.set_bytes(7, &max_blocks_u32);
        encoder.set_bytes(8, &max_seqlen_u32);
        encoder.set_bytes(9, &page_block_size_u32);
        encoder.set_bytes(10, &q_heads_u32);
        encoder.set_bytes(11, &kv_heads_u32);
        encoder.set_bytes(12, &softmax_scale);
        encoder.set_bytes(13, &total_slots_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch,
            height: q_heads,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(())
}
fn push_read_resource(resources: &mut Vec<MetalGraphResourceRef>, buf: &BufferOffset<'_>) {
    resources.push(MetalGraphResourceRef::read(buf.buffer));
}

fn push_write_resource(resources: &mut Vec<MetalGraphResourceRef>, buf: &BufferOffset<'_>) {
    resources.push(MetalGraphResourceRef::write(buf.buffer));
}

#[allow(dead_code)]
pub(crate) fn metal_record_paged_kv_write_token_major_bf16_icb(
    command: &IndirectComputeCommand,
    args: &MetalPagedKvWriteTokenMajorIcbArgs,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    slot: usize,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
) -> Result<Vec<MetalGraphResourceRef>> {
    anyhow::ensure!(
        metal_paged_kv_write_token_major_supports(k_pool, v_pool, slot, k, v),
        "metal paged kv token-major write ICB unsupported shape"
    );
    let (_, heads, head_dim) = k_pool.dims3()?;
    let kp_metal = kt_metal(k_pool)?;
    let companion = kp_metal.companion()?;
    let pipeline = metal_paged_kv_write_token_major_pipeline_indirect(&*companion)?;
    command.set_compute_pipeline_state(&pipeline);

    let ks_metal = kt_metal(k)?;
    let vs_metal = kt_metal(v)?;
    let vp_metal = kt_metal(v_pool)?;

    let ks_buf = buffer_o_kt(ks_metal.buffer().as_ref(), k.layout(), k.dtype());
    let vs_buf = buffer_o_kt(vs_metal.buffer().as_ref(), v.layout(), v.dtype());
    let kp_buf = buffer_o_kt(kp_metal.buffer().as_ref(), k_pool.layout(), k_pool.dtype());
    let vp_buf = buffer_o_kt(vp_metal.buffer().as_ref(), v_pool.layout(), v_pool.dtype());

    command.set_kernel_buffer(0, ks_buf.buffer, ks_buf.offset_in_bytes);
    command.set_kernel_buffer(1, vs_buf.buffer, vs_buf.offset_in_bytes);
    command.set_kernel_buffer(2, kp_buf.buffer, kp_buf.offset_in_bytes);
    command.set_kernel_buffer(3, vp_buf.buffer, vp_buf.offset_in_bytes);
    command.set_kernel_buffer(4, args.slot.buffer(), 0);
    command.set_kernel_buffer(5, args.heads.buffer(), 0);
    command.set_kernel_buffer(6, args.head_dim.buffer(), 0);

    let total = heads * head_dim;
    let threadgroups_per_grid = objc2_metal::MTLSize {
        width: total.div_ceil(256),
        height: 1,
        depth: 1,
    };
    let threads_per_threadgroup = objc2_metal::MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    command.dispatch_threadgroups(threadgroups_per_grid, threads_per_threadgroup);
    command.set_barrier();

    let mut resources = Vec::with_capacity(7);
    push_read_resource(&mut resources, &ks_buf);
    push_read_resource(&mut resources, &vs_buf);
    push_write_resource(&mut resources, &kp_buf);
    push_write_resource(&mut resources, &vp_buf);
    resources.extend(args.scalar_resources());
    Ok(resources)
}

#[allow(dead_code)]
pub(crate) fn metal_record_paged_kv_write_token_major_batch_bf16_icb(
    command: &IndirectComputeCommand,
    args: &MetalPagedKvWriteTokenMajorBatchIcbArgs,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    slots: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
) -> Result<Vec<MetalGraphResourceRef>> {
    anyhow::ensure!(
        metal_paged_kv_write_token_major_batch_supports(k_pool, v_pool, slots, k, v),
        "metal paged kv token-major batch write ICB unsupported shape"
    );
    let (_, heads, head_dim) = k_pool.dims3()?;
    let (batch, _, _, _) = k.dims4()?;
    let kp_metal = kt_metal(k_pool)?;
    let companion = kp_metal.companion()?;
    let pipeline = metal_paged_kv_write_token_major_batch_pipeline_indirect(&*companion)?;
    command.set_compute_pipeline_state(&pipeline);

    let ks_metal = kt_metal(k)?;
    let vs_metal = kt_metal(v)?;
    let vp_metal = kt_metal(v_pool)?;
    let slot_metal = kt_metal(slots)?;

    let ks_buf = buffer_o_kt(ks_metal.buffer().as_ref(), k.layout(), k.dtype());
    let vs_buf = buffer_o_kt(vs_metal.buffer().as_ref(), v.layout(), v.dtype());
    let kp_buf = buffer_o_kt(kp_metal.buffer().as_ref(), k_pool.layout(), k_pool.dtype());
    let vp_buf = buffer_o_kt(vp_metal.buffer().as_ref(), v_pool.layout(), v_pool.dtype());
    let slot_buf = buffer_o_kt(slot_metal.buffer().as_ref(), slots.layout(), slots.dtype());

    command.set_kernel_buffer(0, ks_buf.buffer, ks_buf.offset_in_bytes);
    command.set_kernel_buffer(1, vs_buf.buffer, vs_buf.offset_in_bytes);
    command.set_kernel_buffer(2, kp_buf.buffer, kp_buf.offset_in_bytes);
    command.set_kernel_buffer(3, vp_buf.buffer, vp_buf.offset_in_bytes);
    command.set_kernel_buffer(4, slot_buf.buffer, slot_buf.offset_in_bytes);
    command.set_kernel_buffer(5, args.batch.buffer(), 0);
    command.set_kernel_buffer(6, args.heads.buffer(), 0);
    command.set_kernel_buffer(7, args.head_dim.buffer(), 0);
    command.set_kernel_buffer(8, args.total_slots.buffer(), 0);

    let total = batch * heads * head_dim;
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
    command.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    command.set_barrier();

    let mut resources = Vec::with_capacity(9);
    push_read_resource(&mut resources, &ks_buf);
    push_read_resource(&mut resources, &vs_buf);
    push_write_resource(&mut resources, &kp_buf);
    push_write_resource(&mut resources, &vp_buf);
    push_read_resource(&mut resources, &slot_buf);
    resources.extend(args.scalar_resources());
    Ok(resources)
}

#[allow(dead_code)]
pub(crate) fn metal_record_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256_icb(
    command: &IndirectComputeCommand,
    args: &MetalPagedAttnDecodeDynSeqlenIcbArgs,
    q: &kiln_tensor::Tensor,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    block_table: &kiln_tensor::Tensor,
    seqused_k: &kiln_tensor::Tensor,
    out: &kiln_tensor::Tensor,
    max_seqlen_k: usize,
    page_block_size: usize,
) -> Result<Vec<MetalGraphResourceRef>> {
    anyhow::ensure!(
        metal_paged_attn_decode_contiguous_batch_dyn_seqlen_supports(
            q,
            k_pool,
            v_pool,
            block_table,
            seqused_k,
            max_seqlen_k,
            page_block_size,
        ),
        "metal dyn-seqlen paged batch decode attention ICB unsupported shape"
    );
    let (batch, _, q_heads, head_dim) = q.dims4()?;
    anyhow::ensure!(
        out.dtype() == kiln_tensor::DType::BF16
            && matches!(out.device(), kiln_tensor::Device::Metal(_))
            && out.is_contiguous()
            && out.dims() == [batch, 1usize, q_heads, head_dim],
        "metal dyn-seqlen paged batch decode ICB output must be contiguous BF16 [batch,1,q_heads,head_dim] on Metal"
    );

    let q_metal = kt_metal(q)?;
    let companion = q_metal.companion()?;
    let pipeline =
        metal_paged_attn_decode_contiguous_batch_dyn_seqlen_pipeline_indirect(&*companion)?;
    command.set_compute_pipeline_state(&pipeline);

    let k_metal = kt_metal(k_pool)?;
    let v_metal = kt_metal(v_pool)?;
    let out_metal = kt_metal(out)?;
    let table_metal = kt_metal(block_table)?;
    let seq_metal = kt_metal(seqused_k)?;

    let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
    let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k_pool.layout(), k_pool.dtype());
    let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v_pool.layout(), v_pool.dtype());
    let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());
    let table_buf = buffer_o_kt(
        table_metal.buffer().as_ref(),
        block_table.layout(),
        block_table.dtype(),
    );
    let seq_buf = buffer_o_kt(
        seq_metal.buffer().as_ref(),
        seqused_k.layout(),
        seqused_k.dtype(),
    );

    command.set_kernel_buffer(0, q_buf.buffer, q_buf.offset_in_bytes);
    command.set_kernel_buffer(1, k_buf.buffer, k_buf.offset_in_bytes);
    command.set_kernel_buffer(2, v_buf.buffer, v_buf.offset_in_bytes);
    command.set_kernel_buffer(3, out_buf.buffer, out_buf.offset_in_bytes);
    command.set_kernel_buffer(4, table_buf.buffer, table_buf.offset_in_bytes);
    command.set_kernel_buffer(5, seq_buf.buffer, seq_buf.offset_in_bytes);
    command.set_kernel_buffer(6, args.batch.buffer(), 0);
    command.set_kernel_buffer(7, args.max_blocks_per_seq.buffer(), 0);
    command.set_kernel_buffer(8, args.max_seqlen_k.buffer(), 0);
    command.set_kernel_buffer(9, args.page_block_size.buffer(), 0);
    command.set_kernel_buffer(10, args.q_heads.buffer(), 0);
    command.set_kernel_buffer(11, args.kv_heads.buffer(), 0);
    command.set_kernel_buffer(12, args.softmax_scale.buffer(), 0);
    command.set_kernel_buffer(13, args.total_slots.buffer(), 0);

    let threadgroups_per_grid = objc2_metal::MTLSize {
        width: batch,
        height: q_heads,
        depth: 1,
    };
    let threads_per_threadgroup = objc2_metal::MTLSize {
        width: 1024,
        height: 1,
        depth: 1,
    };
    command.dispatch_threadgroups(threadgroups_per_grid, threads_per_threadgroup);
    command.set_barrier();

    let mut resources = Vec::with_capacity(14);
    push_read_resource(&mut resources, &q_buf);
    push_read_resource(&mut resources, &k_buf);
    push_read_resource(&mut resources, &v_buf);
    push_write_resource(&mut resources, &out_buf);
    push_read_resource(&mut resources, &table_buf);
    push_read_resource(&mut resources, &seq_buf);
    resources.extend(args.scalar_resources());
    Ok(resources)
}

pub(super) fn metal_paged_kv_head_major_read_append_token_major_supports(
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    start_slot: usize,
    prefix_len: usize,
    k_tail: &kiln_tensor::Tensor,
    v_tail: &kiln_tensor::Tensor,
) -> bool {
    if prefix_len == 0 {
        return false;
    }
    if !metal_paged_kv_head_major_read_supports(k_pool, v_pool, start_slot, prefix_len) {
        return false;
    }
    if k_tail.dtype() != kiln_tensor::DType::BF16 || v_tail.dtype() != kiln_tensor::DType::BF16 {
        return false;
    }
    if !matches!(k_tail.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v_tail.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !k_tail.is_contiguous() || !v_tail.is_contiguous() {
        return false;
    }
    let Ok((batch, tail_len, heads, head_dim)) = k_tail.dims4() else {
        return false;
    };
    let Ok(v_dims) = v_tail.dims4() else {
        return false;
    };
    let Ok((_, pool_heads, pool_head_dim)) = k_pool.dims3() else {
        return false;
    };
    let Some(total_len) = prefix_len.checked_add(tail_len) else {
        return false;
    };
    let Some(total) = total_len
        .checked_mul(heads)
        .and_then(|n| n.checked_mul(head_dim))
    else {
        return false;
    };
    batch == 1
        && v_dims == (batch, tail_len, heads, head_dim)
        && heads == pool_heads
        && head_dim == pool_head_dim
        && total_len <= u32::MAX as usize
        && tail_len <= u32::MAX as usize
        && heads <= u32::MAX as usize
        && head_dim <= u32::MAX as usize
        && total <= u32::MAX as usize
}

pub(super) fn metal_paged_kv_head_major_read_append_token_major_bf16(
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    start_slot: usize,
    prefix_len: usize,
    k_tail: &kiln_tensor::Tensor,
    v_tail: &kiln_tensor::Tensor,
) -> Result<(kiln_tensor::Tensor, kiln_tensor::Tensor)> {
    anyhow::ensure!(
        metal_paged_kv_head_major_read_append_token_major_supports(
            k_pool, v_pool, start_slot, prefix_len, k_tail, v_tail,
        ),
        "metal paged kv head-major read+append unsupported shape"
    );
    let (_, tail_len, heads, head_dim) = k_tail.dims4()?;
    let total_len = prefix_len + tail_len;
    let out_shape = (1usize, heads, total_len, head_dim);
    let k_pool_metal = kt_metal(k_pool)?;
    let v_pool_metal = kt_metal(v_pool)?;
    // SAFETY: the kernel dispatch covers exactly every element in `out_shape`.
    let k_out = kt_metal_alloc(
        k_pool_metal,
        kiln_tensor::DType::BF16,
        &[out_shape.0, out_shape.1, out_shape.2, out_shape.3],
    )?;
    // SAFETY: the kernel dispatch covers exactly every element in `out_shape`.
    let v_out = kt_metal_alloc(
        v_pool_metal,
        kiln_tensor::DType::BF16,
        &[out_shape.0, out_shape.1, out_shape.2, out_shape.3],
    )?;

    let companion = k_pool_metal.companion()?;
    let pipeline = metal_paged_kv_head_major_read_append_token_major_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_paged_kv_head_major_read_append_token_major_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let kt_metal_buf = kt_metal(k_tail)?;
        let vt_metal = kt_metal(v_tail)?;
        let ko_metal = kt_metal(&k_out)?;
        let vo_metal = kt_metal(&v_out)?;

        // #1082 candle-free: buffers + layout + dtype come straight off the
        // kt MetalStorage / kt Tensor — no candle storage_and_layout bridge.
        let k_buf = buffer_o_kt(
            k_pool_metal.buffer().as_ref(),
            k_pool.layout(),
            k_pool.dtype(),
        );
        let v_buf = buffer_o_kt(
            v_pool_metal.buffer().as_ref(),
            v_pool.layout(),
            v_pool.dtype(),
        );
        let kt_buf = buffer_o_kt(
            kt_metal_buf.buffer().as_ref(),
            k_tail.layout(),
            k_tail.dtype(),
        );
        let vt_buf = buffer_o_kt(vt_metal.buffer().as_ref(), v_tail.layout(), v_tail.dtype());
        let ko_buf = buffer_o_kt(ko_metal.buffer().as_ref(), k_out.layout(), k_out.dtype());
        let vo_buf = buffer_o_kt(vo_metal.buffer().as_ref(), v_out.layout(), v_out.dtype());

        encoder.set_buffer(0, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(kt_buf.buffer), kt_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(vt_buf.buffer), vt_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(ko_buf.buffer), ko_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(vo_buf.buffer), vo_buf.offset_in_bytes);

        let start_slot_u32 = start_slot as u32;
        let prefix_len_u32 = prefix_len as u32;
        let tail_len_u32 = tail_len as u32;
        let heads_u32 = heads as u32;
        let head_dim_u32 = head_dim as u32;
        encoder.set_bytes(6, &start_slot_u32);
        encoder.set_bytes(7, &prefix_len_u32);
        encoder.set_bytes(8, &tail_len_u32);
        encoder.set_bytes(9, &heads_u32);
        encoder.set_bytes(10, &head_dim_u32);

        let total = total_len * heads * head_dim;
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

    Ok((k_out, v_out))
}

pub(crate) fn metal_paged_kv_write_token_major_supports(
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    slot: usize,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
) -> bool {
    if metal_paged_kv_write_token_major_disabled() {
        return false;
    }
    if k_pool.dtype() != kiln_tensor::DType::BF16
        || v_pool.dtype() != kiln_tensor::DType::BF16
        || k.dtype() != kiln_tensor::DType::BF16
        || v.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    if !matches!(k_pool.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v_pool.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !k_pool.is_contiguous()
        || !v_pool.is_contiguous()
        || !k.is_contiguous()
        || !v.is_contiguous()
    {
        return false;
    }
    let Ok((total_slots, pool_heads, pool_head_dim)) = k_pool.dims3() else {
        return false;
    };
    let Ok(v_pool_dims) = v_pool.dims3() else {
        return false;
    };
    let Ok((batch, seq_len, heads, head_dim)) = k.dims4() else {
        return false;
    };
    let Ok(v_dims) = v.dims4() else {
        return false;
    };
    let Some(total) = heads.checked_mul(head_dim) else {
        return false;
    };

    batch == 1
        && seq_len == 1
        && v_pool_dims == (total_slots, pool_heads, pool_head_dim)
        && v_dims == (batch, seq_len, heads, head_dim)
        && heads == pool_heads
        && head_dim == pool_head_dim
        && slot < total_slots
        && slot <= u32::MAX as usize
        && heads <= u32::MAX as usize
        && head_dim <= u32::MAX as usize
        && total <= u32::MAX as usize
}

pub(crate) fn metal_paged_kv_write_token_major_bf16(
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    slot: usize,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
) -> Result<()> {
    anyhow::ensure!(
        metal_paged_kv_write_token_major_supports(k_pool, v_pool, slot, k, v),
        "metal paged kv token-major write unsupported shape"
    );
    let (_, heads, head_dim) = k_pool.dims3()?;
    let kp_metal = kt_metal(k_pool)?;
    let companion = kp_metal.companion()?;
    let pipeline = metal_paged_kv_write_token_major_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_paged_kv_write_token_major_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let ks_metal = kt_metal(k)?;
        let vs_metal = kt_metal(v)?;
        let vp_metal = kt_metal(v_pool)?;

        // #1082 candle-free: buffers + layout + dtype come straight off the
        // kt MetalStorage / kt Tensor — no candle storage_and_layout bridge.
        let ks_buf = buffer_o_kt(ks_metal.buffer().as_ref(), k.layout(), k.dtype());
        let vs_buf = buffer_o_kt(vs_metal.buffer().as_ref(), v.layout(), v.dtype());
        let kp_buf = buffer_o_kt(kp_metal.buffer().as_ref(), k_pool.layout(), k_pool.dtype());
        let vp_buf = buffer_o_kt(vp_metal.buffer().as_ref(), v_pool.layout(), v_pool.dtype());

        encoder.set_buffer(0, Some(ks_buf.buffer), ks_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(vs_buf.buffer), vs_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(kp_buf.buffer), kp_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(vp_buf.buffer), vp_buf.offset_in_bytes);

        let slot_u32 = slot as u32;
        let heads_u32 = heads as u32;
        let head_dim_u32 = head_dim as u32;
        encoder.set_bytes(4, &slot_u32);
        encoder.set_bytes(5, &heads_u32);
        encoder.set_bytes(6, &head_dim_u32);

        let total = heads * head_dim;
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

    Ok(())
}

#[allow(dead_code)]
#[allow(dead_code)]
pub(crate) fn metal_paged_kv_write_token_major_batch_supports(
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    slots: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
) -> bool {
    if metal_paged_kv_write_token_major_disabled() {
        return false;
    }
    if k_pool.dtype() != kiln_tensor::DType::BF16
        || v_pool.dtype() != kiln_tensor::DType::BF16
        || slots.dtype() != kiln_tensor::DType::U32
        || k.dtype() != kiln_tensor::DType::BF16
        || v.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    if !matches!(k_pool.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v_pool.device(), kiln_tensor::Device::Metal(_))
        || !matches!(slots.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !k_pool.is_contiguous()
        || !v_pool.is_contiguous()
        || !slots.is_contiguous()
        || !k.is_contiguous()
        || !v.is_contiguous()
    {
        return false;
    }
    let Ok((total_slots, pool_heads, pool_head_dim)) = k_pool.dims3() else {
        return false;
    };
    let Ok(v_pool_dims) = v_pool.dims3() else {
        return false;
    };
    let Ok((batch, seq_len, heads, head_dim)) = k.dims4() else {
        return false;
    };
    let Ok(v_dims) = v.dims4() else {
        return false;
    };
    let Ok(slot_count) = slots.dims1() else {
        return false;
    };
    let Some(row_stride) = heads.checked_mul(head_dim) else {
        return false;
    };
    let Some(total) = batch.checked_mul(row_stride) else {
        return false;
    };

    batch > 0
        && total_slots > 0
        && seq_len == 1
        && slot_count == batch
        && v_pool_dims == (total_slots, pool_heads, pool_head_dim)
        && v_dims == (batch, seq_len, heads, head_dim)
        && heads == pool_heads
        && head_dim == pool_head_dim
        && batch <= u32::MAX as usize
        && heads <= u32::MAX as usize
        && head_dim <= u32::MAX as usize
        && total_slots <= u32::MAX as usize
        && row_stride <= u32::MAX as usize
        && total <= u32::MAX as usize
}

#[allow(dead_code)]
#[allow(dead_code)]
pub(crate) fn metal_paged_kv_write_token_major_batch_bf16(
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    slots: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
) -> Result<()> {
    anyhow::ensure!(
        metal_paged_kv_write_token_major_batch_supports(k_pool, v_pool, slots, k, v),
        "metal paged kv token-major batch write unsupported shape"
    );
    let (total_slots, heads, head_dim) = k_pool.dims3()?;
    let (batch, _, _, _) = k.dims4()?;
    let kp_metal = kt_metal(k_pool)?;
    let companion = kp_metal.companion()?;
    let pipeline = metal_paged_kv_write_token_major_batch_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_paged_kv_write_token_major_batch_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let ks_metal = kt_metal(k)?;
        let vs_metal = kt_metal(v)?;
        let vp_metal = kt_metal(v_pool)?;
        let slot_metal = kt_metal(slots)?;

        // #1082 candle-free: buffers + layout + dtype come straight off the
        // kt MetalStorage / kt Tensor — no candle storage_and_layout bridge.
        let ks_buf = buffer_o_kt(ks_metal.buffer().as_ref(), k.layout(), k.dtype());
        let vs_buf = buffer_o_kt(vs_metal.buffer().as_ref(), v.layout(), v.dtype());
        let kp_buf = buffer_o_kt(kp_metal.buffer().as_ref(), k_pool.layout(), k_pool.dtype());
        let vp_buf = buffer_o_kt(vp_metal.buffer().as_ref(), v_pool.layout(), v_pool.dtype());
        let slot_buf = buffer_o_kt(slot_metal.buffer().as_ref(), slots.layout(), slots.dtype());

        encoder.set_buffer(0, Some(ks_buf.buffer), ks_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(vs_buf.buffer), vs_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(kp_buf.buffer), kp_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(vp_buf.buffer), vp_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(slot_buf.buffer), slot_buf.offset_in_bytes);

        let batch_u32 = batch as u32;
        let heads_u32 = heads as u32;
        let head_dim_u32 = head_dim as u32;
        let total_slots_u32 = total_slots as u32;
        encoder.set_bytes(5, &batch_u32);
        encoder.set_bytes(6, &heads_u32);
        encoder.set_bytes(7, &head_dim_u32);
        encoder.set_bytes(8, &total_slots_u32);

        let total = batch * heads * head_dim;
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

    Ok(())
}

#[cfg(test)]
mod metal_icb_decode_tests {
    use super::*;
    use kiln_tensor::{Device, Tensor};

    fn metal_device() -> Option<Device> {
        kiln_tensor::primary_metal_companion(0)
            .ok()
            .map(|_| Device::Metal(0))
    }

    fn pattern_bf16(n: usize, seed: u64) -> Vec<half::bf16> {
        let mut out = Vec::with_capacity(n);
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for i in 0..n {
            s = s
                .wrapping_add(0xA076_1D64_78BD_642F)
                .wrapping_mul(0xE703_7ED1_A0B4_28DB);
            let raw = ((s >> 40) as u32 % 1024) as f32 / 4096.0 - 0.125;
            let trend = (i % 17) as f32 * 0.0007;
            out.push(half::bf16::from_f32(raw + trend));
        }
        out
    }

    fn zeroed_bf16(n: usize) -> Vec<half::bf16> {
        vec![half::bf16::ZERO; n]
    }

    fn max_abs_diff_bf16(a: &[half::bf16], b: &[half::bf16]) -> f32 {
        assert_eq!(
            a.len(),
            b.len(),
            "length mismatch {} vs {}",
            a.len(),
            b.len()
        );
        a.iter()
            .zip(b)
            .map(|(x, y)| (x.to_f32() - y.to_f32()).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn single_token_paged_decode_icb_matches_eager_and_updates_slot() -> Result<()> {
        let Some(dev) = metal_device() else {
            eprintln!(
                "Metal unavailable, skipping single_token_paged_decode_icb_matches_eager_and_updates_slot"
            );
            return Ok(());
        };

        let total_slots = 4usize;
        let kv_heads = 4usize;
        let q_heads = 16usize;
        let head_dim = 256usize;
        let pool_elems = total_slots * kv_heads * head_dim;
        let kv_elems = kv_heads * head_dim;
        let q_elems = q_heads * head_dim;
        let out_elems = q_heads * head_dim;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let mut k_pool_host = zeroed_bf16(pool_elems);
        let mut v_pool_host = zeroed_bf16(pool_elems);
        let prefix_k = pattern_bf16(2 * kv_elems, 10);
        let prefix_v = pattern_bf16(2 * kv_elems, 11);
        k_pool_host[..2 * kv_elems].copy_from_slice(&prefix_k);
        v_pool_host[..2 * kv_elems].copy_from_slice(&prefix_v);

        let q = Tensor::from_vec_on(
            dev,
            pattern_bf16(q_elems, 12),
            vec![1, 1, q_heads, head_dim],
        )?;
        let k = Tensor::from_vec_on(
            dev,
            pattern_bf16(kv_elems, 13),
            vec![1, 1, kv_heads, head_dim],
        )?;
        let v = Tensor::from_vec_on(
            dev,
            pattern_bf16(kv_elems, 14),
            vec![1, 1, kv_heads, head_dim],
        )?;
        let k_pool_eager = Tensor::from_vec_on(
            dev,
            k_pool_host.clone(),
            vec![total_slots, kv_heads, head_dim],
        )?;
        let v_pool_eager = Tensor::from_vec_on(
            dev,
            v_pool_host.clone(),
            vec![total_slots, kv_heads, head_dim],
        )?;
        let k_pool_icb =
            Tensor::from_vec_on(dev, k_pool_host, vec![total_slots, kv_heads, head_dim])?;
        let v_pool_icb =
            Tensor::from_vec_on(dev, v_pool_host, vec![total_slots, kv_heads, head_dim])?;
        let block_table = Tensor::from_vec_on(dev, vec![0u32, 1, 2], vec![1, 3])?;
        let seqused_k = Tensor::from_vec_on(dev, vec![3u32], vec![1])?;
        let out_icb =
            Tensor::from_vec_on(dev, zeroed_bf16(out_elems), vec![1, 1, q_heads, head_dim])?;

        let graph = metal_record_single_token_paged_decode_icb_graph(
            &q,
            &k_pool_icb,
            &v_pool_icb,
            &block_table,
            &seqused_k,
            &out_icb,
            &k,
            &v,
            2,
            3,
            1,
            scale,
        )?;

        metal_paged_kv_write_token_major_bf16(&k_pool_eager, &v_pool_eager, 2, &k, &v)?;
        let eager = metal_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256(
            &q,
            &k_pool_eager,
            &v_pool_eager,
            &block_table,
            &seqused_k,
            3,
            1,
            scale,
        )?;
        graph.replay(2, 3, scale)?;

        let eager_0 = eager.to_vec::<half::bf16>()?;
        let icb_0 = out_icb.to_vec::<half::bf16>()?;
        assert_eq!(
            eager_0, icb_0,
            "first ICB replay must be bit-identical to eager Metal decode"
        );

        let next_k = pattern_bf16(kv_elems, 20);
        let next_v = pattern_bf16(kv_elems, 21);
        kiln_tensor::metal_write_host_in_place(&k, &next_k)?;
        kiln_tensor::metal_write_host_in_place(&v, &next_v)?;
        kiln_tensor::metal_write_host_in_place(&block_table, &[0u32, 1, 3])?;

        metal_paged_kv_write_token_major_bf16(&k_pool_eager, &v_pool_eager, 3, &k, &v)?;
        let eager = metal_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256(
            &q,
            &k_pool_eager,
            &v_pool_eager,
            &block_table,
            &seqused_k,
            3,
            1,
            scale,
        )?;
        graph.replay(3, 3, scale)?;

        let eager_1 = eager.to_vec::<half::bf16>()?;
        let icb_1 = out_icb.to_vec::<half::bf16>()?;
        assert_eq!(
            eager_1, icb_1,
            "ICB replay after stable-buffer and slot updates must match eager"
        );
        assert_eq!(graph.replay_count(), 2);
        assert!(
            max_abs_diff_bf16(&icb_0, &icb_1) > 0.0,
            "second replay should observe refreshed K/V and metadata"
        );

        Ok(())
    }

    #[test]
    fn batched_paged_decode_icb_matches_eager_and_updates_slots() -> Result<()> {
        let Some(dev) = metal_device() else {
            eprintln!(
                "Metal unavailable, skipping batched_paged_decode_icb_matches_eager_and_updates_slots"
            );
            return Ok(());
        };

        let batch = 2usize;
        let total_slots = 8usize;
        let kv_heads = 4usize;
        let q_heads = 16usize;
        let head_dim = 256usize;
        let pool_row = kv_heads * head_dim;
        let pool_elems = total_slots * pool_row;
        let kv_elems = batch * pool_row;
        let q_elems = batch * q_heads * head_dim;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let mut k_pool_host = zeroed_bf16(pool_elems);
        let mut v_pool_host = zeroed_bf16(pool_elems);
        for row in 0..batch {
            let block_base = row * 4;
            for prefix_idx in 0..2 {
                let slot = block_base + prefix_idx;
                let dst = slot * pool_row;
                let seed = 100 + (row * 10 + prefix_idx) as u64;
                k_pool_host[dst..dst + pool_row].copy_from_slice(&pattern_bf16(pool_row, seed));
                v_pool_host[dst..dst + pool_row].copy_from_slice(&pattern_bf16(pool_row, seed + 1));
            }
        }

        let q = Tensor::from_vec_on(
            dev,
            pattern_bf16(q_elems, 12),
            vec![batch, 1, q_heads, head_dim],
        )?;
        let k = Tensor::from_vec_on(
            dev,
            pattern_bf16(kv_elems, 13),
            vec![batch, 1, kv_heads, head_dim],
        )?;
        let v = Tensor::from_vec_on(
            dev,
            pattern_bf16(kv_elems, 14),
            vec![batch, 1, kv_heads, head_dim],
        )?;
        let k_pool_eager = Tensor::from_vec_on(
            dev,
            k_pool_host.clone(),
            vec![total_slots, kv_heads, head_dim],
        )?;
        let v_pool_eager = Tensor::from_vec_on(
            dev,
            v_pool_host.clone(),
            vec![total_slots, kv_heads, head_dim],
        )?;
        let k_pool_icb =
            Tensor::from_vec_on(dev, k_pool_host, vec![total_slots, kv_heads, head_dim])?;
        let v_pool_icb =
            Tensor::from_vec_on(dev, v_pool_host, vec![total_slots, kv_heads, head_dim])?;
        let block_table = Tensor::from_vec_on(dev, vec![0u32, 1, 2, 4, 5, 6], vec![batch, 3])?;
        let seqused_k = Tensor::from_vec_on(dev, vec![3u32, 3], vec![batch])?;
        let slots = Tensor::from_vec_on(dev, vec![2u32, 6], vec![batch])?;
        let out_icb =
            Tensor::from_vec_on(dev, zeroed_bf16(q_elems), vec![batch, 1, q_heads, head_dim])?;

        let graph = metal_record_paged_decode_icb_graph(
            &q,
            &k_pool_icb,
            &v_pool_icb,
            &block_table,
            &seqused_k,
            &out_icb,
            &k,
            &v,
            &slots,
            3,
            1,
            scale,
        )?;

        metal_paged_kv_write_token_major_batch_bf16(&k_pool_eager, &v_pool_eager, &slots, &k, &v)?;
        let eager = metal_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256(
            &q,
            &k_pool_eager,
            &v_pool_eager,
            &block_table,
            &seqused_k,
            3,
            1,
            scale,
        )?;
        graph.replay(3, scale)?;

        let eager_0 = eager.to_vec::<half::bf16>()?;
        let icb_0 = out_icb.to_vec::<half::bf16>()?;
        assert_eq!(
            eager_0, icb_0,
            "first batched ICB replay must be bit-identical to eager Metal decode"
        );

        let next_k = pattern_bf16(kv_elems, 20);
        let next_v = pattern_bf16(kv_elems, 21);
        kiln_tensor::metal_write_host_in_place(&k, &next_k)?;
        kiln_tensor::metal_write_host_in_place(&v, &next_v)?;
        kiln_tensor::metal_write_host_in_place(&block_table, &[0u32, 1, 3, 4, 5, 7])?;
        kiln_tensor::metal_write_host_in_place(&slots, &[3u32, 7])?;

        metal_paged_kv_write_token_major_batch_bf16(&k_pool_eager, &v_pool_eager, &slots, &k, &v)?;
        let eager = metal_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256(
            &q,
            &k_pool_eager,
            &v_pool_eager,
            &block_table,
            &seqused_k,
            3,
            1,
            scale,
        )?;
        graph.replay(3, scale)?;

        let eager_1 = eager.to_vec::<half::bf16>()?;
        let icb_1 = out_icb.to_vec::<half::bf16>()?;
        assert_eq!(
            eager_1, icb_1,
            "batched ICB replay after stable slot updates must match eager"
        );
        assert_eq!(graph.replay_count(), 2);
        assert!(
            max_abs_diff_bf16(&icb_0, &icb_1) > 0.0,
            "second batched replay should observe refreshed K/V and metadata"
        );

        Ok(())
    }
}
