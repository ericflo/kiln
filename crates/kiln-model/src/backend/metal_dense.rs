//! Metal dense projection and elementwise fusion helpers.
//!
//! This module owns command encoding for MLP gate/up fusion, SiLU multiply,
//! attention gate fusion, transposed cooperative GEMV, fused QKV projection,
//! and LoRA decode delta application. The Metal backend facade re-exports the
//! public helpers used by forward code.

use anyhow::Result;

use super::metal_config::*;
use super::metal_core::{kt_metal, kt_metal_alloc};
use super::metal_pipeline::*;
use kiln_tensor::metal_types::buffer_o_kt;

pub(crate) fn metal_mlp_gate_up_supports(
    x: &kiln_tensor::Tensor,
    gate_t: &kiln_tensor::Tensor,
    up_t: &kiln_tensor::Tensor,
) -> bool {
    if x.dtype() != kiln_tensor::DType::BF16
        || gate_t.dtype() != kiln_tensor::DType::BF16
        || up_t.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(gate_t.device(), kiln_tensor::Device::Metal(_))
        || !matches!(up_t.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !x.is_contiguous() || !gate_t.is_contiguous() || !up_t.is_contiguous() {
        return false;
    }
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return false;
    };
    let Ok((gate_hidden, intermediate)) = gate_t.dims2() else {
        return false;
    };
    let Ok((up_hidden, up_intermediate)) = up_t.dims2() else {
        return false;
    };
    let Some(rows) = batch.checked_mul(seq_len) else {
        return false;
    };
    let Some(total) = rows.checked_mul(intermediate) else {
        return false;
    };

    rows > 0
        && seq_len == 1
        && hidden == gate_hidden
        && hidden == up_hidden
        && intermediate == up_intermediate
        && hidden <= u32::MAX as usize
        && intermediate <= u32::MAX as usize
        && total <= u32::MAX as usize
}

pub(crate) fn metal_mlp_gate_up_bf16(
    x: &kiln_tensor::Tensor,
    gate_t: &kiln_tensor::Tensor,
    up_t: &kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_mlp_gate_up_supports(x, gate_t, up_t),
        "metal mlp gate/up supports only BF16 [B,1,H] x [H,I] on Metal"
    );
    let (batch, seq_len, hidden) = x.dims3()?;
    let (_, intermediate) = gate_t.dims2()?;
    let rows = batch * seq_len;
    let row_group_size = if rows == 3
        && !metal_mlp_gate_up_row_pair_disabled()
        && !metal_mlp_gate_up_row_triple_disabled()
    {
        3
    } else if rows >= 5
        && !metal_mlp_gate_up_row_pair_disabled()
        && !metal_mlp_gate_up_row_quad_disabled()
    {
        4
    } else if rows > 1 && !metal_mlp_gate_up_row_pair_disabled() {
        2
    } else {
        1
    };
    let row_groups = rows.div_ceil(row_group_size);
    let total = row_groups * intermediate.div_ceil(2);

    let x_metal = kt_metal(x)?;
    // The kernel writes every row/intermediate element.
    let out = kt_metal_alloc(
        x_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, intermediate],
    )?;

    let companion = x_metal.companion()?;
    let encoder = companion.command_encoder()?;

    {
        let gate_metal = kt_metal(gate_t)?;
        let up_metal = kt_metal(up_t)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 mlp-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let gate_buf = buffer_o_kt(
            gate_metal.buffer().as_ref(),
            gate_t.layout(),
            gate_t.dtype(),
        );
        let up_buf = buffer_o_kt(up_metal.buffer().as_ref(), up_t.layout(), up_t.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(gate_buf.buffer), gate_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(up_buf.buffer), up_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let hidden_u32 = hidden as u32;
        let intermediate_u32 = intermediate as u32;

        let serial_vector_safe = rows == 1
            && !metal_mlp_gate_up_serial_vector_load_disabled()
            && intermediate % 2 == 0
            && gate_buf.offset_in_bytes % 4 == 0
            && up_buf.offset_in_bytes % 4 == 0;
        let serial_dedicated = serial_vector_safe && !metal_mlp_gate_up_serial_dedicated_disabled();
        if serial_dedicated {
            let pipeline = metal_mlp_gate_up_serial_pipeline(&*companion)?;
            encoder.set_label("kiln_mlp_gate_up_serial_bf16");
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_bytes(4, &hidden_u32);
            encoder.set_bytes(5, &intermediate_u32);
        } else {
            let pipeline = metal_mlp_gate_up_pipeline(&*companion)?;
            encoder.set_label("kiln_mlp_gate_up_bf16");
            encoder.set_compute_pipeline_state(&pipeline);
            let rows_u32 = rows as u32;
            let row_pair_mode_u32 = if row_group_size == 1 {
                if serial_vector_safe { 6 } else { 0 }
            } else if row_group_size == 3
                && intermediate % 2 == 0
                && gate_buf.offset_in_bytes % 4 == 0
                && up_buf.offset_in_bytes % 4 == 0
            {
                7
            } else if row_group_size == 4
                && !metal_mlp_gate_up_row_quad_vector_load_disabled()
                && intermediate % 2 == 0
                && gate_buf.offset_in_bytes % 4 == 0
                && up_buf.offset_in_bytes % 4 == 0
            {
                5
            } else {
                row_group_size as u32
            };
            encoder.set_bytes(4, &rows_u32);
            encoder.set_bytes(5, &hidden_u32);
            encoder.set_bytes(6, &intermediate_u32);
            encoder.set_bytes(7, &row_pair_mode_u32);
        }

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

    Ok(out)
}

pub(crate) fn metal_mlp_silu_mul_supports(
    gate: &kiln_tensor::Tensor,
    up: &kiln_tensor::Tensor,
) -> bool {
    if metal_mlp_silu_mul_disabled() {
        return false;
    }
    if gate.dtype() != kiln_tensor::DType::BF16 || up.dtype() != kiln_tensor::DType::BF16 {
        return false;
    }
    if !matches!(gate.device(), kiln_tensor::Device::Metal(_))
        || !matches!(up.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !gate.is_contiguous() || !up.is_contiguous() || gate.shape() != up.shape() {
        return false;
    }
    gate.elem_count() > 0 && gate.elem_count() <= u32::MAX as usize
}

pub(crate) fn metal_mlp_silu_mul_bf16(
    gate: &kiln_tensor::Tensor,
    up: &kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_mlp_silu_mul_supports(gate, up),
        "metal mlp silu*mul supports only matching contiguous BF16 Metal tensors"
    );
    let total = gate.elem_count();
    let gate = gate.contiguous()?;
    let up = up.contiguous()?;
    let gate_metal = kt_metal(&gate)?;
    let out = kt_metal_alloc(gate_metal, kiln_tensor::DType::BF16, gate.dims())?;

    let companion = gate_metal.companion()?;
    let pipeline = metal_mlp_silu_mul_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_mlp_silu_mul_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let up_metal = kt_metal(&up)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 mlp-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let gate_buf = buffer_o_kt(gate_metal.buffer().as_ref(), gate.layout(), gate.dtype());
        let up_buf = buffer_o_kt(up_metal.buffer().as_ref(), up.layout(), up.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(gate_buf.buffer), gate_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(up_buf.buffer), up_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let total_u32 = total as u32;
        encoder.set_bytes(3, &total_u32);

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

    Ok(out)
}

pub(crate) fn metal_attn_gate_sigmoid_mul_supports(
    x: &kiln_tensor::Tensor,
    gate: &kiln_tensor::Tensor,
) -> bool {
    if metal_attn_gate_fusion_disabled() {
        return false;
    }
    if x.dtype() != kiln_tensor::DType::BF16 || gate.dtype() != kiln_tensor::DType::BF16 {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(gate.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !x.is_contiguous() || !gate.is_contiguous() {
        return false;
    }
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return false;
    };
    let Ok((gate_batch, gate_seq_len, gate_hidden)) = gate.dims3() else {
        return false;
    };
    let Some(rows) = batch.checked_mul(seq_len) else {
        return false;
    };
    let Some(total) = rows.checked_mul(hidden) else {
        return false;
    };

    batch > 0
        && seq_len == 1
        && gate_batch == batch
        && gate_seq_len == seq_len
        && gate_hidden == hidden
        && total <= u32::MAX as usize
}

pub(crate) fn metal_attn_gate_sigmoid_mul_bf16(
    x: &kiln_tensor::Tensor,
    gate: &kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_attn_gate_sigmoid_mul_supports(x, gate),
        "metal attn gate sigmoid/mul supports only BF16 [B,1,H] tensors on Metal"
    );
    let (batch, seq_len, hidden) = x.dims3()?;
    let total = batch * seq_len * hidden;

    // The kernel writes every hidden element exactly once.
    let x_metal = kt_metal(&x)?;
    let out = kt_metal_alloc(x_metal, kiln_tensor::DType::BF16, &[batch, seq_len, hidden])?;

    let companion = x_metal.companion()?;
    let pipeline = metal_attn_gate_sigmoid_mul_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_attn_gate_sigmoid_mul_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let gate_metal = kt_metal(&gate)?;
        let out_metal = kt_metal(&out)?;

        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let gate_buf = buffer_o_kt(gate_metal.buffer().as_ref(), gate.layout(), gate.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(gate_buf.buffer), gate_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let total_u32 = total as u32;
        encoder.set_bytes(3, &total_u32);

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

    Ok(out)
}

pub(crate) fn metal_transposed_coop_gemv_supports(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> bool {
    if metal_transposed_coop_gemv_disabled() {
        return false;
    }
    if x.dtype() != kiln_tensor::DType::BF16 || weight_t.dtype() != kiln_tensor::DType::BF16 {
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
    let Ok((batch, seq_len, input_dim)) = x.dims3() else {
        return false;
    };
    let Ok((weight_input_dim, output_dim)) = weight_t.dims2() else {
        return false;
    };

    batch == 1
        && seq_len == 1
        && input_dim > 0
        && output_dim > 0
        && input_dim == weight_input_dim
        && input_dim <= u32::MAX as usize
        && output_dim <= u32::MAX as usize
}

pub(crate) fn metal_transposed_coop_gemv_decode_batch_supports(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> bool {
    if metal_transposed_coop_gemv_disabled() {
        return false;
    }
    if x.dtype() != kiln_tensor::DType::BF16 || weight_t.dtype() != kiln_tensor::DType::BF16 {
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
    let Ok((batch, seq_len, input_dim)) = x.dims3() else {
        return false;
    };
    let Ok((weight_input_dim, output_dim)) = weight_t.dims2() else {
        return false;
    };
    let Some(total) = batch.checked_mul(output_dim) else {
        return false;
    };

    batch > 1
        && seq_len == 1
        && input_dim > 0
        && output_dim > 0
        && input_dim == weight_input_dim
        && input_dim <= u32::MAX as usize
        && output_dim <= u32::MAX as usize
        && batch <= u32::MAX as usize
        && total <= u32::MAX as usize
}

pub(crate) fn metal_transposed_coop_gemv_bf16(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    if metal_transposed_coop_gemv_decode_batch_supports(x, weight_t) {
        return metal_transposed_coop_gemv_batch_bf16(x, weight_t);
    }

    let (_, _, input_dim) = x.dims3()?;
    let (_, output_dim) = weight_t.dims2()?;
    metal_transposed_coop_gemv_bf16_with_tile(
        x,
        weight_t,
        metal_transposed_coop_gemv_select_tile(input_dim, output_dim),
    )
}

pub(super) fn metal_transposed_coop_gemv_bf16_with_tile(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
    tile: MetalTransposedCoopGemvTile,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_transposed_coop_gemv_supports(x, weight_t),
        "metal transposed coop GEMV supports only BF16 [1,1,K] x [K,N] on Metal"
    );
    let (_, _, input_dim) = x.dims3()?;
    let (_, output_dim) = weight_t.dims2()?;

    let x_metal = kt_metal(&x)?;
    // The kernel writes every output channel exactly once.
    let out = kt_metal_alloc(
        x_metal,
        kiln_tensor::DType::BF16,
        &[1usize, 1usize, output_dim],
    )?;

    let companion = x_metal.companion()?;
    let pipeline = metal_transposed_coop_gemv_pipeline(&*companion, tile)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label(tile.label());
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let w_metal = kt_metal(&weight_t)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 5 gemv/matmul-family: `buffer_o` → `buffer_o_kt`.
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

        let input_dim_u32 = input_dim as u32;
        let output_dim_u32 = output_dim as u32;
        encoder.set_bytes(3, &input_dim_u32);
        encoder.set_bytes(4, &output_dim_u32);

        let cols_per_threadgroup = tile.tile_cols() * METAL_TRANSPOSED_COOP_GEMV_SIMDGROUPS;
        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: output_dim.div_ceil(cols_per_threadgroup),
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: METAL_TRANSPOSED_COOP_GEMV_THREADS,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

pub(super) fn metal_transposed_coop_gemv_batch_bf16(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_transposed_coop_gemv_decode_batch_supports(x, weight_t),
        "metal batch transposed coop GEMV supports only BF16 [B,1,K] x [K,N] with B > 1 on Metal"
    );
    let (batch, _, input_dim) = x.dims3()?;
    let (_, output_dim) = weight_t.dims2()?;
    let row_grouping_enabled = batch > 1 && !metal_transposed_coop_gemv_row_pair_disabled();
    let row_quad_enabled =
        row_grouping_enabled && batch >= 3 && !metal_transposed_coop_gemv_row_quad_disabled();
    let row_triple_tile8_enabled = row_quad_enabled
        && batch == 3
        && !metal_transposed_coop_gemv_row_quad_tile8_disabled()
        && !metal_transposed_coop_gemv_row_triple_tile8_disabled();
    let row_quad_tile8_enabled = row_quad_enabled
        && !row_triple_tile8_enabled
        && !metal_transposed_coop_gemv_row_quad_tile8_disabled();
    let row_group_size = if row_triple_tile8_enabled {
        3usize
    } else if row_quad_enabled {
        4usize
    } else if row_grouping_enabled {
        2usize
    } else {
        1usize
    };
    let row_groups = batch.div_ceil(row_group_size);

    let x_metal = kt_metal(&x)?;
    // The kernel writes every batch/output channel exactly once.
    let out = kt_metal_alloc(
        x_metal,
        kiln_tensor::DType::BF16,
        &[batch, 1usize, output_dim],
    )?;

    let companion = x_metal.companion()?;
    let pipeline = if row_triple_tile8_enabled {
        metal_transposed_coop_gemv_batch_row_triple_tile8_pipeline(&*companion)?
    } else if row_quad_tile8_enabled {
        metal_transposed_coop_gemv_batch_row_quad_tile8_pipeline(&*companion)?
    } else {
        metal_transposed_coop_gemv_batch_pipeline(&*companion)?
    };
    let encoder = companion.command_encoder()?;
    encoder.set_label(if row_triple_tile8_enabled {
        "kiln_transposed_coop_gemv8_batch_row_triple_tile8_bf16"
    } else if row_quad_tile8_enabled {
        "kiln_transposed_coop_gemv8_batch_row_quad_tile8_bf16"
    } else {
        "kiln_transposed_coop_gemv8_batch_bf16"
    });
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let w_metal = kt_metal(&weight_t)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 5 gemv/matmul-family: `buffer_o` → `buffer_o_kt`.
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

        let input_dim_u32 = input_dim as u32;
        let output_dim_u32 = output_dim as u32;
        encoder.set_bytes(3, &input_dim_u32);
        encoder.set_bytes(4, &output_dim_u32);
        if row_quad_tile8_enabled {
            let batch_u32 = batch as u32;
            encoder.set_bytes(5, &batch_u32);
        } else if !row_triple_tile8_enabled {
            let row_pair_mode_u32 = if row_group_size > 1 { batch as u32 } else { 0 };
            let row_group_size_u32 = row_group_size as u32;
            encoder.set_bytes(5, &row_pair_mode_u32);
            encoder.set_bytes(6, &row_group_size_u32);
        }

        let tile_cols = if row_triple_tile8_enabled || row_quad_tile8_enabled {
            METAL_TRANSPOSED_COOP_GEMV_TILE8_COLS
        } else if row_quad_enabled {
            METAL_TRANSPOSED_COOP_GEMV_TILE4_COLS
        } else {
            METAL_TRANSPOSED_COOP_GEMV_TILE8_COLS
        };
        let cols_per_threadgroup = tile_cols * METAL_TRANSPOSED_COOP_GEMV_SIMDGROUPS;
        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: output_dim.div_ceil(cols_per_threadgroup),
            height: row_groups,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: METAL_TRANSPOSED_COOP_GEMV_THREADS,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

pub(crate) fn metal_fused_qkv_transposed_coop_gemv_supports(
    x: &kiln_tensor::Tensor,
    q_t: &kiln_tensor::Tensor,
    k_t: &kiln_tensor::Tensor,
    v_t: &kiln_tensor::Tensor,
) -> bool {
    if metal_fused_qkv_proj_disabled() {
        return false;
    }
    if !metal_transposed_coop_gemv_supports(x, q_t)
        || !metal_transposed_coop_gemv_supports(x, k_t)
        || !metal_transposed_coop_gemv_supports(x, v_t)
    {
        return false;
    }

    let Ok((_, _, input_dim)) = x.dims3() else {
        return false;
    };
    let Ok((q_input_dim, _)) = q_t.dims2() else {
        return false;
    };
    let Ok((k_input_dim, _)) = k_t.dims2() else {
        return false;
    };
    let Ok((v_input_dim, _)) = v_t.dims2() else {
        return false;
    };
    input_dim == q_input_dim && input_dim == k_input_dim && input_dim == v_input_dim
}

pub(crate) fn metal_fused_qkv_transposed_coop_gemv_bf16(
    x: &kiln_tensor::Tensor,
    q_t: &kiln_tensor::Tensor,
    k_t: &kiln_tensor::Tensor,
    v_t: &kiln_tensor::Tensor,
) -> Result<(
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
)> {
    anyhow::ensure!(
        metal_fused_qkv_transposed_coop_gemv_supports(x, q_t, k_t, v_t),
        "metal fused QKV projection supports only BF16 [1,1,K] x [K,Nq/Nk/Nv] on Metal"
    );
    let (_, _, input_dim) = x.dims3()?;
    let (_, q_output_dim) = q_t.dims2()?;
    let (_, k_output_dim) = k_t.dims2()?;
    let (_, v_output_dim) = v_t.dims2()?;

    let total_output_dim = q_output_dim + k_output_dim + v_output_dim;
    // The kernel writes each projection output independently with the existing
    // tile8 cooperative GEMV mapping. Back the three result views with one
    // allocation to avoid repeated small Metal buffer allocations in decode.
    let x_metal = kt_metal(&x)?;
    let fused_out = kt_metal_alloc(
        x_metal,
        kiln_tensor::DType::BF16,
        &[1usize, 1usize, total_output_dim],
    )?;
    let q_out = fused_out.narrow(2, 0, q_output_dim)?;
    let k_out = fused_out.narrow(2, q_output_dim, k_output_dim)?;
    let v_out = fused_out.narrow(2, q_output_dim + k_output_dim, v_output_dim)?;

    let companion = x_metal.companion()?;
    let pipeline = metal_fused_qkv_transposed_coop_gemv_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_fused_qkv_transposed_coop_gemv8_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let q_metal = kt_metal(&q_t)?;
        let k_metal = kt_metal(&k_t)?;
        let v_metal = kt_metal(&v_t)?;
        let q_out_metal = kt_metal(&q_out)?;
        let k_out_metal = kt_metal(&k_out)?;
        let v_out_metal = kt_metal(&v_out)?;

        // #1082 Step 5 gemv/matmul-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q_t.layout(), q_t.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k_t.layout(), k_t.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v_t.layout(), v_t.dtype());
        let q_out_buf = buffer_o_kt(q_out_metal.buffer().as_ref(), q_out.layout(), q_out.dtype());
        let k_out_buf = buffer_o_kt(k_out_metal.buffer().as_ref(), k_out.layout(), k_out.dtype());
        let v_out_buf = buffer_o_kt(v_out_metal.buffer().as_ref(), v_out.layout(), v_out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(q_out_buf.buffer), q_out_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(k_out_buf.buffer), k_out_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(v_out_buf.buffer), v_out_buf.offset_in_bytes);

        let input_dim_u32 = input_dim as u32;
        let q_output_dim_u32 = q_output_dim as u32;
        let k_output_dim_u32 = k_output_dim as u32;
        let v_output_dim_u32 = v_output_dim as u32;
        encoder.set_bytes(7, &input_dim_u32);
        encoder.set_bytes(8, &q_output_dim_u32);
        encoder.set_bytes(9, &k_output_dim_u32);
        encoder.set_bytes(10, &v_output_dim_u32);

        let cols_per_threadgroup =
            METAL_TRANSPOSED_COOP_GEMV_TILE8_COLS * METAL_TRANSPOSED_COOP_GEMV_SIMDGROUPS;
        let q_groups = q_output_dim.div_ceil(cols_per_threadgroup);
        let k_groups = k_output_dim.div_ceil(cols_per_threadgroup);
        let v_groups = v_output_dim.div_ceil(cols_per_threadgroup);
        let total_groups = q_groups + k_groups + v_groups;
        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: total_groups,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: METAL_TRANSPOSED_COOP_GEMV_THREADS,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok((q_out, k_out, v_out))
}

pub(crate) fn metal_lora_add_decode_supports(
    base: &kiln_tensor::Tensor,
    x: &kiln_tensor::Tensor,
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
) -> bool {
    if metal_lora_delta_decode_disabled() {
        return false;
    }
    if base.dtype() != kiln_tensor::DType::BF16
        || x.dtype() != kiln_tensor::DType::BF16
        || a.dtype() != kiln_tensor::DType::BF16
        || b.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    if !matches!(base.device(), kiln_tensor::Device::Metal(_))
        || !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(a.device(), kiln_tensor::Device::Metal(_))
        || !matches!(b.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !base.is_contiguous() || !x.is_contiguous() || !a.is_contiguous() || !b.is_contiguous() {
        return false;
    }

    let Ok((batch, seq_len, input_dim)) = x.dims3() else {
        return false;
    };
    let Ok((base_batch, base_seq_len, output_dim)) = base.dims3() else {
        return false;
    };
    let Ok((rank, a_input_dim)) = a.dims2() else {
        return false;
    };
    let Ok((b_output_dim, b_rank)) = b.dims2() else {
        return false;
    };
    let Some(total_output) = batch.checked_mul(output_dim) else {
        return false;
    };
    let Some(hidden_total) = batch.checked_mul(rank) else {
        return false;
    };

    batch > 0
        && seq_len == 1
        && base_batch == batch
        && base_seq_len == 1
        && input_dim > 0
        && output_dim > 0
        && input_dim >= 1024
        && output_dim >= 1024
        && rank > 0
        && a_input_dim == input_dim
        && b_output_dim == output_dim
        && b_rank == rank
        && batch <= u32::MAX as usize
        && input_dim <= u32::MAX as usize
        && output_dim <= u32::MAX as usize
        && rank <= u32::MAX as usize
        && total_output <= u32::MAX as usize
        && hidden_total <= u32::MAX as usize
}

pub(crate) fn metal_lora_add_decode_bf16(
    base: &kiln_tensor::Tensor,
    x: &kiln_tensor::Tensor,
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    scale: f32,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_lora_add_decode_supports(base, x, a, b),
        "metal LoRA decode add supports only contiguous BF16 Metal base/x/A/B decode tensors"
    );
    let (batch, _, input_dim) = x.dims3()?;
    let (_, _, output_dim) = base.dims3()?;
    let (rank, _) = a.dims2()?;

    let x_metal = kt_metal(&x)?;
    let base_metal = kt_metal(&base)?;
    let hidden = kt_metal_alloc(x_metal, kiln_tensor::DType::BF16, &[batch, rank])?;
    let out = kt_metal_alloc(
        base_metal,
        kiln_tensor::DType::BF16,
        &[batch, 1usize, output_dim],
    )?;

    let companion = x_metal.companion()?;
    let encoder = companion.command_encoder()?;

    {
        let pipeline = metal_lora_hidden_decode_pipeline(&*companion)?;
        encoder.set_label("kiln_lora_hidden_decode_bf16");
        encoder.set_compute_pipeline_state(&pipeline);

        let a_metal = kt_metal(&a)?;
        let hidden_metal = kt_metal(&hidden)?;

        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let a_buf = buffer_o_kt(a_metal.buffer().as_ref(), a.layout(), a.dtype());
        let hidden_buf = buffer_o_kt(
            hidden_metal.buffer().as_ref(),
            hidden.layout(),
            hidden.dtype(),
        );

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(hidden_buf.buffer), hidden_buf.offset_in_bytes);

        let batch_u32 = batch as u32;
        let input_dim_u32 = input_dim as u32;
        let rank_u32 = rank as u32;
        encoder.set_bytes(3, &batch_u32);
        encoder.set_bytes(4, &input_dim_u32);
        encoder.set_bytes(5, &rank_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: rank,
            height: batch,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    {
        let pipeline = metal_lora_add_decode_pipeline(&*companion)?;
        encoder.set_label("kiln_lora_add_decode_bf16");
        encoder.set_compute_pipeline_state(&pipeline);

        let hidden_metal = kt_metal(&hidden)?;
        let b_metal = kt_metal(&b)?;
        let base_metal = kt_metal(&base)?;
        let out_metal = kt_metal(&out)?;

        let hidden_buf = buffer_o_kt(
            hidden_metal.buffer().as_ref(),
            hidden.layout(),
            hidden.dtype(),
        );
        let b_buf = buffer_o_kt(b_metal.buffer().as_ref(), b.layout(), b.dtype());
        let base_buf = buffer_o_kt(base_metal.buffer().as_ref(), base.layout(), base.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(hidden_buf.buffer), hidden_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(b_buf.buffer), b_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(base_buf.buffer), base_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_u32 = batch as u32;
        let output_dim_u32 = output_dim as u32;
        let rank_u32 = rank as u32;
        encoder.set_bytes(4, &scale);
        encoder.set_bytes(5, &batch_u32);
        encoder.set_bytes(6, &output_dim_u32);
        encoder.set_bytes(7, &rank_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: batch * output_dim,
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

    drop(encoder);
    Ok(out)
}
