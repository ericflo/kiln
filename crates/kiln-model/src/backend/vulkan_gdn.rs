//! Vulkan Gated DeltaNet operation helpers.
//!
//! This owns Vulkan's kt-facing GDN support gates and dispatch hooks while
//! `backend/vulkan.rs` remains the `BackendRuntime` facade.

use anyhow::{Context, Result};
use std::sync::Arc;

use super::vulkan::VulkanBackend;
use super::vulkan_residency::{
    get_recurrent_state_resident_buffer, insert_recurrent_state_resident_buffer,
    recurrent_state_resident_scope_active,
};
use super::vulkan_tensor_bridge::{
    kt_tensor_from_f32_bytes, kt_tensor_to_f32_bytes_with_shape,
    upload_gdn_chunkwise_activations_from_cpu_bytes_vk,
    upload_gdn_chunkwise_inputs_from_cpu_bytes_vk, vk_f32_tensors_to_cpu_tensors_batched_vk,
};

fn fused_gdn_resident_state_enabled() -> bool {
    kiln_vulkan_kernel::kernels::vulkan_kernel_policy().gdn_decode_fused_resident_state_enabled
}

pub(super) fn supports_gdn_forward_substitution(backend: &VulkanBackend) -> bool {
    // solve_tri is experimental: shared-memory layout not yet validated
    // against CPU parity, and may exceed maxComputeSharedMemorySize on many
    // GPUs. Disabled by the portable Vulkan policy.
    backend.has_vulkan() && backend.gdn_forward_sub_enabled
}

pub(super) fn supports_gdn_recurrent_step(backend: &VulkanBackend) -> bool {
    backend.has_vulkan() && backend.gdn_enabled
}

pub(super) fn supports_gdn_recurrent_prefill_native_head_last(backend: &VulkanBackend) -> bool {
    backend.has_vulkan() && backend.gdn_recurrent_unexpanded_qk_enabled
}

pub(super) fn supports_gdn_recurrent_qk_norm_prefill_native_head_last(
    backend: &VulkanBackend,
) -> bool {
    backend.has_vulkan() && backend.gdn_recurrent_qk_norm_unexpanded_enabled
}

pub(super) fn supports_gdn_chunk_prep(backend: &VulkanBackend) -> bool {
    backend.has_vulkan() && backend.gdn_enabled
}

pub(super) fn supports_gdn_chunk_scan(backend: &VulkanBackend) -> bool {
    backend.has_vulkan() && backend.gdn_enabled
}

pub(super) fn supports_gdn_full_chunk_forward(backend: &VulkanBackend) -> bool {
    backend.has_vulkan() && backend.gdn_full_chunk_forward_enabled
}

pub(super) fn supports_gdn_gates(backend: &VulkanBackend) -> bool {
    backend.has_vulkan() && backend.gdn_gates_enabled
}

pub(super) fn supports_gdn_gated_rms_norm(backend: &VulkanBackend) -> bool {
    backend.has_vulkan() && backend.gdn_gated_rms_norm_enabled
}

pub(super) fn gdn_in_proj_decode(
    backend: &VulkanBackend,
    x: &kiln_tensor::Tensor,
    in_proj_qkv_t: &kiln_tensor::Tensor,
    in_proj_z_t: &kiln_tensor::Tensor,
    in_proj_a_t: &kiln_tensor::Tensor,
    in_proj_b_t: &kiln_tensor::Tensor,
) -> Result<
    Option<(
        kiln_tensor::Tensor,
        kiln_tensor::Tensor,
        kiln_tensor::Tensor,
        kiln_tensor::Tensor,
    )>,
> {
    // kt guards read directly off the kt args before the bridge.
    if !backend.has_vulkan() || !backend.gdn_enabled || x.dtype() != kiln_tensor::DType::F32 {
        return Ok(None);
    }
    if !matches!(x.device(), kiln_tensor::Device::Cpu)
        || !matches!(in_proj_qkv_t.device(), kiln_tensor::Device::Cpu)
        || !matches!(in_proj_z_t.device(), kiln_tensor::Device::Cpu)
        || !matches!(in_proj_a_t.device(), kiln_tensor::Device::Cpu)
        || !matches!(in_proj_b_t.device(), kiln_tensor::Device::Cpu)
    {
        return Ok(None);
    }
    // (#1082) Fully kt-native: shapes off kt, weight buffers keyed on the
    // stable kt id (upload once), x bytes + outputs straight from/to kt.
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return Ok(None);
    };
    if seq_len != 1 && !backend.gdn_prefill_in_proj_enabled {
        return Ok(None);
    }

    let Ok((qkv_hidden, qkv_dim)) = in_proj_qkv_t.dims2() else {
        return Ok(None);
    };
    let Ok((z_hidden, z_dim)) = in_proj_z_t.dims2() else {
        return Ok(None);
    };
    let Ok((a_hidden, a_dim)) = in_proj_a_t.dims2() else {
        return Ok(None);
    };
    let Ok((b_hidden, b_dim)) = in_proj_b_t.dims2() else {
        return Ok(None);
    };
    if qkv_hidden != hidden || z_hidden != hidden || a_hidden != hidden || b_hidden != hidden {
        return Ok(None);
    }

    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
    let row_count = batch * seq_len;
    let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
    let use_bf16 = backend.bf16_packed_gdn_in_proj_weights_enabled
        && in_proj_qkv_t.dtype() == kiln_tensor::DType::BF16
        && in_proj_z_t.dtype() == kiln_tensor::DType::BF16
        && in_proj_a_t.dtype() == kiln_tensor::DType::BF16
        && in_proj_b_t.dtype() == kiln_tensor::DType::BF16;
    let (qkv_b, z_b, a_b, b_b) = if use_bf16 {
        let qkv_buf = backend.cached_bf16_packed_weight_buffer_kt(in_proj_qkv_t)?;
        let z_buf = backend.cached_bf16_packed_weight_buffer_kt(in_proj_z_t)?;
        let a_buf = backend.cached_bf16_packed_weight_buffer_kt(in_proj_a_t)?;
        let b_buf = backend.cached_bf16_packed_weight_buffer_kt(in_proj_b_t)?;
        kiln_vulkan_kernel::kernels::dispatch_gdn_in_proj_decode_cached_bf16_weights_bytes(
            vk_device, &x_data, row_count, &qkv_buf, &z_buf, &a_buf, &b_buf, hidden, qkv_dim,
            z_dim, a_dim, b_dim,
        )
        .context("gdn_in_proj_decode kernel failed")?
    } else {
        let qkv_buf = backend.cached_f32_weight_buffer_kt(in_proj_qkv_t)?;
        let z_buf = backend.cached_f32_weight_buffer_kt(in_proj_z_t)?;
        let a_buf = backend.cached_f32_weight_buffer_kt(in_proj_a_t)?;
        let b_buf = backend.cached_f32_weight_buffer_kt(in_proj_b_t)?;
        kiln_vulkan_kernel::kernels::dispatch_gdn_in_proj_decode_cached_bytes(
            vk_device, &x_data, row_count, &qkv_buf, &z_buf, &a_buf, &b_buf, hidden, qkv_dim,
            z_dim, a_dim, b_dim,
        )
        .context("gdn_in_proj_decode kernel failed")?
    };
    Ok(Some((
        kt_tensor_from_f32_bytes(&qkv_b, &[batch, seq_len, qkv_dim], kiln_tensor::DType::F32)?,
        kt_tensor_from_f32_bytes(&z_b, &[batch, seq_len, z_dim], kiln_tensor::DType::F32)?,
        kt_tensor_from_f32_bytes(&a_b, &[batch, seq_len, a_dim], kiln_tensor::DType::F32)?,
        kt_tensor_from_f32_bytes(&b_b, &[batch, seq_len, b_dim], kiln_tensor::DType::F32)?,
    )))
}

pub(super) fn gdn_forward_substitution(
    backend: &VulkanBackend,
    a_strict: &kiln_tensor::Tensor,
    v_prime: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    // kt guards read directly off the kt args before the bridge.
    if !backend.has_vulkan() || !backend.gdn_enabled {
        return Ok(None);
    }
    if a_strict.dtype() != kiln_tensor::DType::BF16 && a_strict.dtype() != kiln_tensor::DType::F32 {
        return Ok(None);
    }
    // (#1082) kt-native: byte extraction reads straight from kt storage.
    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

    let v_dims = v_prime.dims();
    let (batch, heads, chunk, dv) = (v_dims[0], v_dims[1], v_dims[2], v_dims[3]);
    let a_strict_bytes = kt_tensor_to_f32_bytes_with_shape(a_strict)?.0;
    let v_prime_bytes = kt_tensor_to_f32_bytes_with_shape(v_prime)?.0;
    let beta_bytes = kt_tensor_to_f32_bytes_with_shape(beta)?.0;
    let out_data = kiln_vulkan_kernel::kernels::dispatch_gdn_forward_substitution_bytes(
        vk_device,
        &a_strict_bytes,
        &v_prime_bytes,
        &beta_bytes,
        batch,
        heads,
        chunk,
        dv,
    )
    .context("gdn_forward_substitution kernel failed")?;
    let out = kt_tensor_from_f32_bytes(
        &out_data,
        &[batch, heads, chunk, dv],
        kiln_tensor::DType::F32,
    )?;
    Ok(Some(out))
}

pub(super) fn gdn_solve_tri_transpose(
    backend: &VulkanBackend,
    a_strict: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    dw: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !backend.has_vulkan() || !backend.gdn_enabled {
        return Ok(None);
    }
    if a_strict.dtype() != kiln_tensor::DType::F32
        || beta.dtype() != kiln_tensor::DType::F32
        || dw.dtype() != kiln_tensor::DType::F32
    {
        return Ok(None);
    }
    let Some(vk_device) = backend.vulkan_device() else {
        return Ok(None);
    };

    let load = |t: &kiln_tensor::Tensor| -> Result<kiln_vulkan_kernel::vk_tensor::VkTensor> {
        let (bytes, shape) = kt_tensor_to_f32_bytes_with_shape(t)?;
        let data: &[f32] = bytemuck::cast_slice(&bytes);
        kiln_vulkan_kernel::vk_tensor::VkTensor::from_f32_slice_recycled(
            data,
            shape,
            vk_device.clone(),
        )
    };

    let a_vk = load(a_strict)?;
    let beta_vk = load(beta)?;
    let dw_vk = load(dw)?;
    let (batch, heads, chunk, dv) = dw.dims4()?;
    let out_vk = kiln_vulkan_kernel::vk_ops::gdn_chunk_bwd::vk_solve_tri_transpose_no_grad(
        &a_vk, &beta_vk, &dw_vk, batch, heads, chunk, dv,
    )
    .context("vk_solve_tri_transpose_no_grad")?;
    let mut out =
        vk_f32_tensors_to_cpu_tensors_batched_vk(&[(&out_vk, "gdn_solve_tri_transpose")])?;
    Ok(out.pop())
}

pub(super) fn gdn_chunkwise_forward(
    backend: &VulkanBackend,
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    g: &kiln_tensor::Tensor,
    state_kt: &mut kiln_tensor::Tensor,
    chunk_size: usize,
) -> Result<Option<kiln_tensor::Tensor>> {
    // Proper Vulkan GDN prefill: run the chunkwise scan on the GPU in
    // parallel (`vk_gdn_chunkwise_forward_no_grad`) instead of the CPU
    // chunkwise (raw kt matmuls on CPU-host tensors). F32 only on Vulkan
    // (activations are F32). kt-native: extract f32 straight from kt
    // storage, no candle bridge. (#1082)
    if !backend.has_vulkan() || !backend.gdn_enabled {
        return Ok(None);
    }
    if q.dtype() != kiln_tensor::DType::F32 || state_kt.dtype() != kiln_tensor::DType::F32 {
        return Ok(None);
    }
    let policy = kiln_vulkan_kernel::kernels::vulkan_kernel_policy();
    if !policy.gdn_chunkwise_forward_enabled {
        return Ok(None);
    }
    let Some(vk_device) = backend.vulkan_device() else {
        return Ok(None);
    };

    let state_shape = state_kt.shape().to_vec();
    let load = |t: &kiln_tensor::Tensor| -> Result<kiln_vulkan_kernel::vk_tensor::VkTensor> {
        let shape = t.shape().to_vec();
        let data = t
            .flatten_all()
            .map_err(|e| anyhow::anyhow!("gdn_chunkwise_forward: flatten: {e}"))?
            .to_vec1::<f32>()
            .map_err(|e| anyhow::anyhow!("gdn_chunkwise_forward: to_vec1 f32: {e}"))?;
        kiln_vulkan_kernel::vk_tensor::VkTensor::from_f32_slice_recycled(
            &data,
            shape,
            vk_device.clone(),
        )
    };
    let resident_scope_active = recurrent_state_resident_scope_active();
    let state_id = state_kt.id();
    let resident_state = resident_scope_active.then(|| {
        get_recurrent_state_resident_buffer(&backend.recurrent_state_resident_registry, state_id)
    });
    let resident_state = resident_state.flatten();
    let (q_vk, k_vk, v_vk, beta_vk, g_vk, mut state_vk) = if let Some(state_buffer) = resident_state
    {
        let (q_vk, k_vk, v_vk, beta_vk, g_vk) = if let Some([q_vk, k_vk, v_vk, beta_vk, g_vk]) =
            upload_gdn_chunkwise_activations_from_cpu_bytes_vk(vk_device, q, k, v, beta, g)?
        {
            (q_vk, k_vk, v_vk, beta_vk, g_vk)
        } else {
            (load(q)?, load(k)?, load(v)?, load(beta)?, load(g)?)
        };
        let state_vk = kiln_vulkan_kernel::vk_tensor::VkTensor::from_buffer(
            state_buffer,
            state_shape.clone(),
            kiln_vulkan_kernel::vk_tensor::VkDType::F32,
            vk_device.clone(),
        );
        (q_vk, k_vk, v_vk, beta_vk, g_vk, state_vk)
    } else if let Some([q_vk, k_vk, v_vk, beta_vk, g_vk, state_vk]) =
        upload_gdn_chunkwise_inputs_from_cpu_bytes_vk(vk_device, q, k, v, beta, g, state_kt)?
    {
        (q_vk, k_vk, v_vk, beta_vk, g_vk, state_vk)
    } else {
        (
            load(q)?,
            load(k)?,
            load(v)?,
            load(beta)?,
            load(g)?,
            load(state_kt)?,
        )
    };

    let out_vk = if !policy.gdn_chunkwise_single_submit_enabled {
        if policy.gdn_chunkwise_fallback_enabled {
            tracing::warn!("single-submit Vulkan GDN chunkwise prefill disabled; falling back");
            kiln_vulkan_kernel::vk_ops::gdn_chunkwise::vk_gdn_chunkwise_forward_no_grad(
                &q_vk,
                &k_vk,
                &v_vk,
                &beta_vk,
                &g_vk,
                &mut state_vk,
                chunk_size,
            )
            .context("vk_gdn_chunkwise_forward_no_grad fallback")?
        } else {
            anyhow::bail!("single-submit Vulkan GDN chunkwise prefill disabled; fallback disabled");
        }
    } else {
        match kiln_vulkan_kernel::vk_ops::gdn_chunkwise::vk_gdn_chunkwise_forward_no_grad_single_submit(
            &q_vk,
            &k_vk,
            &v_vk,
            &beta_vk,
            &g_vk,
            &mut state_vk,
            chunk_size,
        ) {
            Ok(out) => out,
            Err(err) => {
                if policy.gdn_chunkwise_fallback_enabled {
                    tracing::warn!(
                        error = %err,
                        "single-submit Vulkan GDN chunkwise prefill failed; falling back"
                    );
                    kiln_vulkan_kernel::vk_ops::gdn_chunkwise::vk_gdn_chunkwise_forward_no_grad(
                        &q_vk,
                        &k_vk,
                        &v_vk,
                        &beta_vk,
                        &g_vk,
                        &mut state_vk,
                        chunk_size,
                    )
                    .context("vk_gdn_chunkwise_forward_no_grad fallback")?
                } else {
                    return Err(err)
                        .context("single-submit Vulkan GDN chunkwise prefill failed; fallback disabled");
                }
            }
        }
    };

    if resident_scope_active {
        anyhow::ensure!(
            state_vk.shape() == state_shape.as_slice(),
            "gdn_chunkwise_forward: resident state shape mismatch: got {:?}, expected {:?}",
            state_vk.shape(),
            state_shape
        );
        let mut outputs =
            vk_f32_tensors_to_cpu_tensors_batched_vk(&[(&out_vk, "gdn_chunkwise_forward output")])?;
        let out_kt = outputs
            .pop()
            .context("gdn_chunkwise_forward: resident output readback was empty")?;
        insert_recurrent_state_resident_buffer(
            &backend.recurrent_state_resident_registry,
            state_id,
            Arc::clone(state_vk.buffer()),
        );
        return Ok(Some(out_kt));
    }

    // Outside a resumable scope, read back output + updated state together.
    let [out_kt, new_state]: [kiln_tensor::Tensor; 2] =
        vk_f32_tensors_to_cpu_tensors_batched_vk(&[
            (&out_vk, "gdn_chunkwise_forward output"),
            (&state_vk, "gdn_chunkwise_forward state"),
        ])?
        .try_into()
        .map_err(|readbacks: Vec<_>| {
            anyhow::anyhow!(
                "gdn_chunkwise_forward: read back {} tensors, expected 2",
                readbacks.len()
            )
        })?;
    anyhow::ensure!(
        new_state.shape() == state_shape.as_slice(),
        "gdn_chunkwise_forward: state shape mismatch after readback: got {:?}, expected {:?}",
        new_state.shape(),
        state_shape
    );
    *state_kt = new_state;
    Ok(Some(out_kt))
}

pub(super) fn gdn_chunk_prep(
    backend: &VulkanBackend,
    g: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    kkt: &kiln_tensor::Tensor,
    qkt: &kiln_tensor::Tensor,
    ks_entry: &kiln_tensor::Tensor,
    q_s: &kiln_tensor::Tensor,
) -> Result<
    Option<(
        kiln_tensor::Tensor,
        kiln_tensor::Tensor,
        kiln_tensor::Tensor,
        kiln_tensor::Tensor,
        kiln_tensor::Tensor,
        kiln_tensor::Tensor,
    )>,
> {
    // kt guards read directly off the kt args before the bridge.
    if !backend.has_vulkan() || !backend.gdn_enabled {
        return Ok(None);
    }
    if g.dtype() != kiln_tensor::DType::BF16 {
        return Ok(None);
    }
    // (#1082) kt-native: byte extraction + reconstruction run on kt args.
    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

    let g_data = kt_tensor_to_f32_bytes_with_shape(g)?.0;
    let v_data = kt_tensor_to_f32_bytes_with_shape(v)?.0;
    let kkt_data = kt_tensor_to_f32_bytes_with_shape(kkt)?.0;
    let qkt_data = kt_tensor_to_f32_bytes_with_shape(qkt)?.0;
    let ks_entry_data = kt_tensor_to_f32_bytes_with_shape(ks_entry)?.0;
    let q_s_data = kt_tensor_to_f32_bytes_with_shape(q_s)?.0;
    let g_dims = g.dims();
    let (batch, heads, chunk) = (g_dims[0], g_dims[1], g_dims[2]);
    let dv = v.dims()[3];
    let (a_strict_b, b_mask_b, v_prime_b, q_s_scaled_b, decay_last_col_b, p_last_b) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_chunk_prep_bytes(
            vk_device,
            &g_data,
            &v_data,
            &kkt_data,
            &qkt_data,
            &ks_entry_data,
            &q_s_data,
            batch,
            heads,
            chunk,
            dv,
        )
        .context("gdn_chunk_prep kernel failed")?;
    let cc_shape = [batch, heads, chunk, chunk];
    let cv_shape = [batch, heads, chunk, dv];
    let decay_shape = [batch, heads, chunk];
    let p_last_shape = [batch, heads];
    Ok(Some((
        kt_tensor_from_f32_bytes(&a_strict_b, &cc_shape, kiln_tensor::DType::BF16)?,
        kt_tensor_from_f32_bytes(&b_mask_b, &cc_shape, kiln_tensor::DType::BF16)?,
        kt_tensor_from_f32_bytes(&v_prime_b, &cv_shape, kiln_tensor::DType::BF16)?,
        kt_tensor_from_f32_bytes(&q_s_scaled_b, &cv_shape, kiln_tensor::DType::BF16)?,
        kt_tensor_from_f32_bytes(&decay_last_col_b, &decay_shape, kiln_tensor::DType::BF16)?,
        kt_tensor_from_f32_bytes(&p_last_b, &p_last_shape, kiln_tensor::DType::BF16)?,
    )))
}

pub(super) fn gdn_chunk_scan(
    backend: &VulkanBackend,
    a_strict: &kiln_tensor::Tensor,
    b_mask: &kiln_tensor::Tensor,
    v_prime: &kiln_tensor::Tensor,
    q_s_scaled: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    decay_last_col: &kiln_tensor::Tensor,
) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
    // kt guards read directly off the kt args before the bridge.
    if !backend.has_vulkan() || !backend.gdn_enabled {
        return Ok(None);
    }
    if a_strict.dtype() != kiln_tensor::DType::BF16 {
        return Ok(None);
    }
    // (#1082) kt-native: byte extraction + reconstruction run on kt args.
    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

    let a_strict_data = kt_tensor_to_f32_bytes_with_shape(a_strict)?.0;
    let b_mask_data = kt_tensor_to_f32_bytes_with_shape(b_mask)?.0;
    let v_prime_data = kt_tensor_to_f32_bytes_with_shape(v_prime)?.0;
    let q_s_scaled_data = kt_tensor_to_f32_bytes_with_shape(q_s_scaled)?.0;
    let beta_data = kt_tensor_to_f32_bytes_with_shape(beta)?.0;
    let decay_last_col_data = kt_tensor_to_f32_bytes_with_shape(decay_last_col)?.0;
    let v_prime_dims = v_prime.dims();
    let (batch, heads, chunk, dv) = (
        v_prime_dims[0],
        v_prime_dims[1],
        v_prime_dims[2],
        v_prime_dims[3],
    );
    let (out_data, p_out_data) = kiln_vulkan_kernel::kernels::dispatch_gdn_chunk_scan_bytes(
        vk_device,
        &a_strict_data,
        &b_mask_data,
        &v_prime_data,
        &q_s_scaled_data,
        &beta_data,
        &decay_last_col_data,
        batch,
        heads,
        chunk,
        dv,
    )
    .context("gdn_chunk_scan kernel failed")?;
    let out_tensor = kt_tensor_from_f32_bytes(
        &out_data,
        &[batch, heads, chunk, dv],
        kiln_tensor::DType::BF16,
    )?;
    let p_out_tensor = kt_tensor_from_f32_bytes(
        &p_out_data,
        &[batch, heads, chunk, dv],
        kiln_tensor::DType::BF16,
    )?;
    Ok(Some((out_tensor, p_out_tensor)))
}

pub(super) fn gdn_full_chunk_forward(
    backend: &VulkanBackend,
    g: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    kkt: &kiln_tensor::Tensor,
    qkt: &kiln_tensor::Tensor,
    ks_entry: &kiln_tensor::Tensor,
    q_s: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    k_t: &kiln_tensor::Tensor,
    state_kt: &mut kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    // kt guards read directly off the kt args before the bridge.
    if !backend.has_vulkan() || !backend.gdn_enabled {
        return Ok(None);
    }
    if g.dtype() != kiln_tensor::DType::BF16 {
        return Ok(None);
    }
    // (#1082) kt-native: all args are already kt; `state_kt` is mutated in
    // place at the return below.
    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

    let g_data = kt_tensor_to_f32_bytes_with_shape(g)?.0;
    let v_data = kt_tensor_to_f32_bytes_with_shape(v)?.0;
    let kkt_data = kt_tensor_to_f32_bytes_with_shape(kkt)?.0;
    let qkt_data = kt_tensor_to_f32_bytes_with_shape(qkt)?.0;
    let ks_entry_data = kt_tensor_to_f32_bytes_with_shape(ks_entry)?.0;
    let q_s_data = kt_tensor_to_f32_bytes_with_shape(q_s)?.0;
    let beta_data = kt_tensor_to_f32_bytes_with_shape(beta)?.0;
    let k_t_data = kt_tensor_to_f32_bytes_with_shape(k_t)?.0;
    let state_data = kt_tensor_to_f32_bytes_with_shape(state_kt)?.0;
    let g_dims = g.dims();
    let (batch, heads, chunk) = (g_dims[0], g_dims[1], g_dims[2]);
    let dv = v.dims()[3];
    let dk = k_t.dims()[2];
    let state_dims = state_kt.dims().to_vec();
    let (out_data, new_state_data) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_full_chunk_forward_bytes(
            vk_device,
            &g_data,
            &v_data,
            &kkt_data,
            &qkt_data,
            &ks_entry_data,
            &q_s_data,
            &beta_data,
            &k_t_data,
            &state_data,
            batch,
            heads,
            chunk,
            dk,
            dv,
        )
        .context("gdn_full_chunk_forward kernel failed")?;
    let out = kt_tensor_from_f32_bytes(
        &out_data,
        &[batch, heads, chunk, dv],
        kiln_tensor::DType::BF16,
    )?;
    *state_kt = kt_tensor_from_f32_bytes(&new_state_data, &state_dims, kiln_tensor::DType::BF16)?;
    Ok(Some(out))
}

pub(super) fn gdn_decode_gates_recurrent_rmsnorm(
    backend: &VulkanBackend,
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
    state_kt: &mut kiln_tensor::Tensor,
    z: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    eps: f64,
) -> Result<Option<kiln_tensor::Tensor>> {
    // kt guards read directly off the kt args before the bridge.
    if !backend.has_vulkan() || !backend.gdn_enabled || q.dtype() != kiln_tensor::DType::F32 {
        return Ok(None);
    }
    if !matches!(q.device(), kiln_tensor::Device::Cpu)
        || !matches!(k.device(), kiln_tensor::Device::Cpu)
        || !matches!(v.device(), kiln_tensor::Device::Cpu)
        || !matches!(a.device(), kiln_tensor::Device::Cpu)
        || !matches!(b.device(), kiln_tensor::Device::Cpu)
        || !matches!(a_log.device(), kiln_tensor::Device::Cpu)
        || !matches!(dt_bias.device(), kiln_tensor::Device::Cpu)
        || !matches!(state_kt.device(), kiln_tensor::Device::Cpu)
        || !matches!(z.device(), kiln_tensor::Device::Cpu)
        || !matches!(weight.device(), kiln_tensor::Device::Cpu)
    {
        return Ok(None);
    }
    // (#1082) kt-native: all args are already kt. `state_kt` is mutated in
    // place at each return that may have updated the recurrent state.
    let Ok((batch, seq_len, nv, dk)) = q.dims4() else {
        return Ok(None);
    };
    let Ok((k_batch, k_seq, k_nv, k_dk)) = k.dims4() else {
        return Ok(None);
    };
    let Ok((v_batch, v_seq, v_nv, dv)) = v.dims4() else {
        return Ok(None);
    };
    let Ok((z_batch, z_seq, z_nv, z_dv)) = z.dims4() else {
        return Ok(None);
    };
    let Ok((state_batch, state_nv, state_dk, state_dv)) = state_kt.dims4() else {
        return Ok(None);
    };
    if batch == 1 && !backend.gdn_decode_fused_enabled {
        return Ok(None);
    }
    if seq_len != 1
        || k_batch != batch
        || k_seq != 1
        || v_batch != batch
        || v_seq != 1
        || z_batch != batch
        || z_seq != 1
        || k_nv != nv
        || v_nv != nv
        || z_nv != nv
        || k_dk != dk
        || state_batch != batch
        || state_nv != nv
        || state_dk != dk
        || state_dv != dv
        || z_dv != dv
        || dv > 256
    {
        return Ok(None);
    }
    if a.dims() != [batch, 1, nv]
        || b.dims() != [batch, 1, nv]
        || a_log.dims() != [nv]
        || dt_bias.dims() != [nv]
        || weight.dims() != [dv]
    {
        return Ok(None);
    }

    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
    let skip_state_readback = crate::forward::vulkan_skip_gdn_state_readback_active();
    if batch > 1 && fused_gdn_resident_state_enabled() && recurrent_state_resident_scope_active() {
        let state_id = state_kt.id();
        let resident_state = get_recurrent_state_resident_buffer(
            &backend.recurrent_state_resident_registry,
            state_id,
        );
        let (batch_d, _, nv, dk) = q.dims4()?;
        let dv = v.dims4()?.3;
        let q_dtype = q.dtype();
        let q_b = kt_tensor_to_f32_bytes_with_shape(q)?.0;
        let k_b = kt_tensor_to_f32_bytes_with_shape(k)?.0;
        let v_b = kt_tensor_to_f32_bytes_with_shape(v)?.0;
        let a_b = kt_tensor_to_f32_bytes_with_shape(a)?.0;
        let b_b = kt_tensor_to_f32_bytes_with_shape(b)?.0;
        let a_log_b = kt_tensor_to_f32_bytes_with_shape(a_log)?.0;
        let dt_bias_b = kt_tensor_to_f32_bytes_with_shape(dt_bias)?.0;
        let z_b = kt_tensor_to_f32_bytes_with_shape(z)?.0;
        let weight_b = kt_tensor_to_f32_bytes_with_shape(weight)?.0;
        let state_b = if resident_state.is_none() {
            Some(kt_tensor_to_f32_bytes_with_shape(state_kt)?.0)
        } else {
            None
        };
        let (out_data, resident_state) =
            kiln_vulkan_kernel::kernels::dispatch_gdn_decode_gates_recurrent_rmsnorm_resident_state_bytes(
                vk_device,
                &q_b, &k_b, &v_b, &a_b, &b_b, &a_log_b, &dt_bias_b,
                state_b.as_deref(),
                &z_b, &weight_b,
                batch_d, nv, dk, dv,
                eps as f32,
                resident_state,
            )
            .context("gdn_decode_gates_recurrent_rmsnorm resident-state kernel failed")?;
        let out = kt_tensor_from_f32_bytes(&out_data, &[batch_d, 1, nv, dv], q_dtype)?;
        insert_recurrent_state_resident_buffer(
            &backend.recurrent_state_resident_registry,
            state_id,
            resident_state,
        );
        return Ok(Some(out));
    }
    let (batch, _, nv, dk) = q.dims4()?;
    let dv = v.dims4()?.3;
    let q_dtype = q.dtype();
    let state_dtype = state_kt.dtype();
    let state_dims = state_kt.dims().to_vec();
    let input_tensors: [&kiln_tensor::Tensor; 10] =
        [q, k, v, a, b, a_log, dt_bias, &*state_kt, z, weight];
    let mut input_data: Vec<Vec<u8>> = Vec::with_capacity(input_tensors.len());
    for tensor in &input_tensors {
        input_data.push(kt_tensor_to_f32_bytes_with_shape(tensor)?.0);
    }
    let (out_data, new_state_data) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_decode_gates_recurrent_rmsnorm_bytes(
            vk_device,
            &input_data,
            batch,
            nv,
            dk,
            dv,
            eps as f32,
            skip_state_readback,
        )
        .context("gdn_decode_gates_recurrent_rmsnorm kernel failed")?;
    let out = kt_tensor_from_f32_bytes(&out_data, &[batch, 1, nv, dv], q_dtype)?;
    if !skip_state_readback {
        if let Some(sd) = new_state_data {
            *state_kt = kt_tensor_from_f32_bytes(&sd, &state_dims, state_dtype)?;
        }
    }
    Ok(Some(out))
}

pub(super) fn gdn_recurrent_step(
    backend: &VulkanBackend,
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    g: &kiln_tensor::Tensor,
    state_kt: &mut kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    // kt guards read directly off the kt args before the bridge.
    if !backend.has_vulkan() || !backend.gdn_enabled {
        return Ok(None);
    }
    if !matches!(
        q.dtype(),
        kiln_tensor::DType::BF16 | kiln_tensor::DType::F32
    ) {
        return Ok(None);
    }
    // (#1082) kt-native: all args are already kt; `state_kt` is mutated in
    // place. The recurrent-state resident cache keys on the kt `TensorId`.
    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

    if recurrent_state_resident_scope_active() {
        let state_id = state_kt.id();
        let resident_state = get_recurrent_state_resident_buffer(
            &backend.recurrent_state_resident_registry,
            state_id,
        );

        let q_data = kt_tensor_to_f32_bytes_with_shape(q)?.0;
        let k_data = kt_tensor_to_f32_bytes_with_shape(k)?.0;
        let v_data = kt_tensor_to_f32_bytes_with_shape(v)?.0;
        let beta_data = kt_tensor_to_f32_bytes_with_shape(beta)?.0;
        let g_data = kt_tensor_to_f32_bytes_with_shape(g)?.0;
        let state_data_owned = if resident_state.is_none() {
            Some(kt_tensor_to_f32_bytes_with_shape(state_kt)?.0)
        } else {
            None
        };
        let q_dims = q.dims();
        let (batch, heads, dk) = (q_dims[0], q_dims[1], q_dims[2]);
        let dv = v.dims()[2];
        let q_dtype = q.dtype();
        let (out_data, resident_state) =
            kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step_resident_state_bytes(
                vk_device,
                &q_data,
                &k_data,
                &v_data,
                &beta_data,
                &g_data,
                state_data_owned.as_deref(),
                batch,
                heads,
                dk,
                dv,
                resident_state,
            )
            .context("gdn_recurrent_step resident-state kernel failed")?;
        let out = kt_tensor_from_f32_bytes(&out_data, &[batch, heads, dv], q_dtype)?;

        insert_recurrent_state_resident_buffer(
            &backend.recurrent_state_resident_registry,
            state_id,
            resident_state,
        );
        return Ok(Some(out));
    }

    let skip_state_readback = crate::forward::vulkan_skip_gdn_state_readback_active();
    let q_dims = q.dims();
    let (batch, heads, dk) = (q_dims[0], q_dims[1], q_dims[2]);
    let dv = v.dims()[2];
    let q_dtype = q.dtype();
    let state_dtype = state_kt.dtype();
    let state_dims = state_kt.dims().to_vec();
    let q_data = kt_tensor_to_f32_bytes_with_shape(q)?.0;
    let k_data = kt_tensor_to_f32_bytes_with_shape(k)?.0;
    let v_data = kt_tensor_to_f32_bytes_with_shape(v)?.0;
    let beta_data = kt_tensor_to_f32_bytes_with_shape(beta)?.0;
    let g_data = kt_tensor_to_f32_bytes_with_shape(g)?.0;
    let state_data = kt_tensor_to_f32_bytes_with_shape(state_kt)?.0;
    let (out_data, new_state_data) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step_with_options_bytes(
            vk_device,
            &q_data,
            &k_data,
            &v_data,
            &beta_data,
            &g_data,
            &state_data,
            batch,
            heads,
            dk,
            dv,
            skip_state_readback,
        )
        .context("gdn_recurrent_step kernel failed")?;
    let out = kt_tensor_from_f32_bytes(&out_data, &[batch, heads, dv], q_dtype)?;
    if let Some(sd) = new_state_data {
        *state_kt = kt_tensor_from_f32_bytes(&sd, &state_dims, state_dtype)?;
    }
    Ok(Some(out))
}

pub(super) fn gdn_recurrent_prefill_native_head_last(
    backend: &VulkanBackend,
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    g: &kiln_tensor::Tensor,
    state_kt: &mut kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    // kt guards read directly off the kt args before the bridge.
    if !backend.has_vulkan()
        || !backend.gdn_recurrent_unexpanded_qk_enabled
        || !matches!(
            q.dtype(),
            kiln_tensor::DType::BF16 | kiln_tensor::DType::F32
        )
    {
        return Ok(None);
    }
    if !matches!(q.device(), kiln_tensor::Device::Cpu)
        || !matches!(k.device(), kiln_tensor::Device::Cpu)
        || !matches!(v.device(), kiln_tensor::Device::Cpu)
        || !matches!(beta.device(), kiln_tensor::Device::Cpu)
        || !matches!(g.device(), kiln_tensor::Device::Cpu)
        || !matches!(state_kt.device(), kiln_tensor::Device::Cpu)
    {
        return Ok(None);
    }
    // (#1082) kt-native: all args are already kt; `state_kt` is mutated in
    // place. The recurrent-state resident cache keys on the kt `TensorId`.
    let Ok((batch, seq_len, q_heads, dk)) = q.dims4() else {
        return Ok(None);
    };
    let Ok((k_batch, k_seq_len, k_heads, k_dk)) = k.dims4() else {
        return Ok(None);
    };
    let Ok((v_batch, v_seq_len, heads, dv)) = v.dims4() else {
        return Ok(None);
    };
    let Ok((beta_batch, beta_seq_len, beta_heads)) = beta.dims3() else {
        return Ok(None);
    };
    let Ok((g_batch, g_seq_len, g_heads)) = g.dims3() else {
        return Ok(None);
    };
    let Ok((state_batch, state_heads, state_dk, state_dv)) = state_kt.dims4() else {
        return Ok(None);
    };
    if seq_len != 1
        || k_batch != batch
        || k_seq_len != seq_len
        || k_heads != q_heads
        || k_dk != dk
        || v_batch != batch
        || v_seq_len != seq_len
        || beta_batch != batch
        || beta_seq_len != seq_len
        || beta_heads != heads
        || g_batch != batch
        || g_seq_len != seq_len
        || g_heads != heads
        || state_batch != batch
        || state_heads != heads
        || state_dk != dk
        || state_dv != dv
        || q_heads == 0
        || heads % q_heads != 0
    {
        return Ok(None);
    }

    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
    if recurrent_state_resident_scope_active() && state_kt.dtype() == q.dtype() {
        let state_id = state_kt.id();
        let resident_state = get_recurrent_state_resident_buffer(
            &backend.recurrent_state_resident_registry,
            state_id,
        );
        let q_data = kt_tensor_to_f32_bytes_with_shape(q)?.0;
        let k_data = kt_tensor_to_f32_bytes_with_shape(k)?.0;
        let v_data = kt_tensor_to_f32_bytes_with_shape(v)?.0;
        let beta_data = kt_tensor_to_f32_bytes_with_shape(beta)?.0;
        let g_data = kt_tensor_to_f32_bytes_with_shape(g)?.0;
        let state_data_owned = if resident_state.is_none() {
            Some(kt_tensor_to_f32_bytes_with_shape(state_kt)?.0)
        } else {
            None
        };
        let (batch, seq_len, q_heads, dk) = q.dims4()?;
        let (_, _, heads, dv) = v.dims4()?;
        let q_dtype = q.dtype();
        let (out_data, resident_state) =
            kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step_native_head_last_resident_state_bytes(
                vk_device,
                &q_data, &k_data, &v_data, &beta_data, &g_data,
                state_data_owned.as_deref(),
                batch, seq_len, q_heads, heads, dk, dv,
                resident_state,
            )
            .context("gdn_recurrent_step native-head resident-state Vulkan kernel failed")?;
        // `out_data` is the un-unsqueezed [batch, heads, dv] layout.
        // Reconstruct the kt tensor and re-unsqueeze to match prior public shape.
        let out_no_seq = kt_tensor_from_f32_bytes(&out_data, &[batch, heads, dv], q_dtype)?;
        let out = out_no_seq.unsqueeze(1)?;
        insert_recurrent_state_resident_buffer(
            &backend.recurrent_state_resident_registry,
            state_id,
            resident_state,
        );
        return Ok(Some(out));
    }
    let skip_state_readback = crate::forward::vulkan_skip_gdn_state_readback_active();
    let (batch, _seq, q_heads, dk) = q.dims4()?;
    let (_, _, heads, dv) = v.dims4()?;
    let q_dtype = q.dtype();
    let state_dtype = state_kt.dtype();
    let state_dims = state_kt.dims().to_vec();
    let q_data = kt_tensor_to_f32_bytes_with_shape(q)?.0;
    let k_data = kt_tensor_to_f32_bytes_with_shape(k)?.0;
    let v_data = kt_tensor_to_f32_bytes_with_shape(v)?.0;
    let beta_data = kt_tensor_to_f32_bytes_with_shape(beta)?.0;
    let g_data = kt_tensor_to_f32_bytes_with_shape(g)?.0;
    let state_data = kt_tensor_to_f32_bytes_with_shape(state_kt)?.0;
    let (out_data, new_state_data) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step_native_head_last_with_options_bytes(
            vk_device,
            &q_data,
            &k_data,
            &v_data,
            &beta_data,
            &g_data,
            &state_data,
            batch,
            q_heads,
            heads,
            dk,
            dv,
            skip_state_readback,
        )
        .context("gdn_recurrent_step native-head Vulkan kernel failed")?;
    let out = kt_tensor_from_f32_bytes(&out_data, &[batch, heads, dv], q_dtype)?.unsqueeze(1)?;
    if let Some(sd) = new_state_data {
        *state_kt = kt_tensor_from_f32_bytes(&sd, &state_dims, state_dtype)?;
    }
    Ok(Some(out))
}

pub(super) fn gdn_recurrent_qk_norm_prefill_native_head_last(
    backend: &VulkanBackend,
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    g: &kiln_tensor::Tensor,
    state_kt: &mut kiln_tensor::Tensor,
    q_scale: f64,
    qk_eps: f64,
) -> Result<Option<kiln_tensor::Tensor>> {
    // kt guards read directly off the kt args before the bridge.
    if !backend.has_vulkan()
        || !backend.gdn_recurrent_qk_norm_unexpanded_enabled
        || !matches!(
            q.dtype(),
            kiln_tensor::DType::F32 | kiln_tensor::DType::BF16
        )
    {
        return Ok(None);
    }
    if !matches!(q.device(), kiln_tensor::Device::Cpu)
        || !matches!(k.device(), kiln_tensor::Device::Cpu)
        || !matches!(v.device(), kiln_tensor::Device::Cpu)
        || !matches!(beta.device(), kiln_tensor::Device::Cpu)
        || !matches!(g.device(), kiln_tensor::Device::Cpu)
        || !matches!(state_kt.device(), kiln_tensor::Device::Cpu)
    {
        return Ok(None);
    }
    // (#1082) kt-native: all args are already kt; `state_kt` is mutated in
    // place at the return below.
    let Ok((_, _, _, dk)) = q.dims4() else {
        return Ok(None);
    };
    let expected_scale = 1.0 / (dk as f64).sqrt();
    if (q_scale - expected_scale).abs() > 1e-6 || (qk_eps - 1e-6).abs() > 1e-12 {
        return Ok(None);
    }
    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
    let skip_state_readback = crate::forward::vulkan_skip_gdn_state_readback_active();
    let (batch, _seq, q_heads, dk) = q.dims4()?;
    let (_, _, heads, dv) = v.dims4()?;
    let state_dtype = state_kt.dtype();
    let state_dims = state_kt.dims().to_vec();
    let q_data = kt_tensor_to_f32_bytes_with_shape(q)?.0;
    let k_data = kt_tensor_to_f32_bytes_with_shape(k)?.0;
    let v_data = kt_tensor_to_f32_bytes_with_shape(v)?.0;
    let beta_data = kt_tensor_to_f32_bytes_with_shape(beta)?.0;
    let g_data = kt_tensor_to_f32_bytes_with_shape(g)?.0;
    let state_data = kt_tensor_to_f32_bytes_with_shape(state_kt)?.0;
    let (out_data, new_state_data) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_qk_norm_step_native_head_last_with_options_bytes(
            vk_device,
            &q_data,
            &k_data,
            &v_data,
            &beta_data,
            &g_data,
            &state_data,
            batch,
            q_heads,
            heads,
            dk,
            dv,
            skip_state_readback,
        )
        .context("gdn_recurrent_qk_norm native-head Vulkan kernel failed")?;
    let out =
        kt_tensor_from_f32_bytes(&out_data, &[batch, heads, dv], state_dtype)?.unsqueeze(1)?;
    if let Some(sd) = new_state_data {
        *state_kt = kt_tensor_from_f32_bytes(&sd, &state_dims, state_dtype)?;
    }
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ResidencyBackend;

    fn tensor(data: Vec<f32>, shape: impl Into<kiln_tensor::Shape>) -> Result<kiln_tensor::Tensor> {
        kiln_tensor::Tensor::from_vec(data, shape)
            .map_err(|error| anyhow::anyhow!("create test tensor: {error}"))
    }

    fn values(tensor: &kiln_tensor::Tensor) -> Result<Vec<f32>> {
        tensor
            .flatten_all()
            .map_err(|error| anyhow::anyhow!("flatten test tensor: {error}"))?
            .to_vec1::<f32>()
            .map_err(|error| anyhow::anyhow!("read test tensor: {error}"))
    }

    fn assert_close(label: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
        let max_error = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error <= 5e-4, "{label} max error {max_error}");
    }

    fn inputs(seed: usize, seq_len: usize) -> Result<[kiln_tensor::Tensor; 5]> {
        let mk = |count: usize, scale: f32, offset: f32| {
            (0..count)
                .map(|idx| {
                    let value = ((idx + seed * 17) % 23) as f32 / 23.0;
                    offset + value * scale
                })
                .collect::<Vec<_>>()
        };
        Ok([
            tensor(mk(seq_len * 2, 0.2, -0.1), (1, 1, seq_len, 2))?,
            tensor(mk(seq_len * 2, 0.18, -0.09), (1, 1, seq_len, 2))?,
            tensor(mk(seq_len * 2, 0.24, -0.12), (1, 1, seq_len, 2))?,
            tensor(mk(seq_len, 0.3, 0.35), (1, 1, seq_len))?,
            tensor(mk(seq_len, 0.02, -0.04), (1, 1, seq_len))?,
        ])
    }

    fn resident_backend() -> Arc<VulkanBackend> {
        let mut backend = VulkanBackend::new(kiln_tensor::Device::Cpu);
        backend.prefill_recurrent_state_residency_enabled = true;
        Arc::new(backend)
    }

    #[test]
    fn production_prefill_recurrent_state_residency_is_quarantined() {
        let mut backend = VulkanBackend::new(kiln_tensor::Device::Cpu);
        backend.recurrent_state_residency_enabled = false;

        assert!(!backend.prefill_recurrent_state_residency_enabled);
        assert!(!ResidencyBackend::runtime_enter_gdn_recurrent_resident_state_scope(&backend));
        assert!(!ResidencyBackend::runtime_enter_gdn_prefill_resident_state_scope(&backend, 17,));
        assert!(
            !ResidencyBackend::runtime_enter_gdn_prefill_resident_state_layer_scope(&backend, 0,)
        );
    }

    #[test]
    fn chunkwise_prefill_resident_state_uses_stable_owner_across_handles_and_threads() -> Result<()>
    {
        let backend = resident_backend();
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping resident chunkwise test");
            return Ok(());
        }
        let [q1, k1, v1, beta1, g1] = inputs(1, 5)?;
        let [q2, k2, v2, beta2, g2] = inputs(2, 3)?;
        let initial = vec![0.0f32; 4];

        let mut baseline_state = tensor(initial.clone(), (1, 1, 2, 2))?;
        let baseline_out1 =
            gdn_chunkwise_forward(&backend, &q1, &k1, &v1, &beta1, &g1, &mut baseline_state, 4)?
                .context("baseline first chunk declined")?;
        let baseline_out2 =
            gdn_chunkwise_forward(&backend, &q2, &k2, &v2, &beta2, &g2, &mut baseline_state, 4)?
                .context("baseline second chunk declined")?;
        let q_all = kiln_tensor::Tensor::cat(&[&q1, &q2], 2)?;
        let k_all = kiln_tensor::Tensor::cat(&[&k1, &k2], 2)?;
        let v_all = kiln_tensor::Tensor::cat(&[&v1, &v2], 2)?;
        let beta_all = kiln_tensor::Tensor::cat(&[&beta1, &beta2], 2)?;
        let g_all = kiln_tensor::Tensor::cat(&[&g1, &g2], 2)?;
        let mut monolithic_state = tensor(initial.clone(), (1, 1, 2, 2))?;
        let monolithic_out = gdn_chunkwise_forward(
            &backend,
            &q_all,
            &k_all,
            &v_all,
            &beta_all,
            &g_all,
            &mut monolithic_state,
            4,
        )?
        .context("monolithic baseline declined")?;
        let split_out = kiln_tensor::Tensor::cat(&[&baseline_out1, &baseline_out2], 2)?;
        assert_close(
            "split versus monolithic output",
            &values(&split_out)?,
            &values(&monolithic_out)?,
        );
        assert_close(
            "split versus monolithic state",
            &values(&baseline_state)?,
            &values(&monolithic_state)?,
        );
        let mut quantized_state =
            tensor(initial.clone(), (1, 1, 2, 2))?.to_dtype(kiln_tensor::DType::BF16)?;
        let mut quantized_work = quantized_state.to_dtype(kiln_tensor::DType::F32)?;
        let quantized_out1 =
            gdn_chunkwise_forward(&backend, &q1, &k1, &v1, &beta1, &g1, &mut quantized_work, 4)?
                .context("quantized first chunk declined")?;
        quantized_state = quantized_work.to_dtype(kiln_tensor::DType::BF16)?;
        let mut quantized_work = quantized_state.to_dtype(kiln_tensor::DType::F32)?;
        let quantized_out2 =
            gdn_chunkwise_forward(&backend, &q2, &k2, &v2, &beta2, &g2, &mut quantized_work, 4)?
                .context("quantized second chunk declined")?;
        quantized_state = quantized_work.to_dtype(kiln_tensor::DType::BF16)?;
        let owner_id = 17;
        let layer_idx = 0;
        let mut resident_state =
            tensor(initial.clone(), (1, 1, 2, 2))?.to_dtype(kiln_tensor::DType::BF16)?;
        let stable_id = resident_state.id();
        // Model the inference wrapper's BF16 -> F32 normalization. The work
        // handle has a different TensorId from the persistent state slot.
        let mut work_state = resident_state.to_dtype(kiln_tensor::DType::F32)?;
        assert_ne!(work_state.id(), stable_id);
        assert!(
            ResidencyBackend::runtime_enter_gdn_prefill_resident_state_scope(&*backend, owner_id,)
        );
        assert!(
            ResidencyBackend::runtime_enter_gdn_prefill_resident_state_layer_scope(
                &*backend, layer_idx,
            )
        );
        let resident_out1 =
            gdn_chunkwise_forward(&backend, &q1, &k1, &v1, &beta1, &g1, &mut work_state, 4)?
                .context("resident first chunk declined")?;
        let stale_external_state = work_state.to_dtype(kiln_tensor::DType::BF16)?;
        assert!(
            ResidencyBackend::runtime_rekey_gdn_recurrent_resident_state(
                &*backend,
                &work_state,
                &stale_external_state,
            )?
        );
        assert!(
            ResidencyBackend::runtime_rekey_gdn_recurrent_resident_state(
                &*backend,
                &stale_external_state,
                &resident_state,
            )?
        );
        assert!(!ResidencyBackend::runtime_has_gdn_recurrent_resident_state(
            &*backend,
            &work_state,
        ));
        assert!(ResidencyBackend::runtime_has_gdn_recurrent_resident_state(
            &*backend,
            &resident_state,
        ));
        ResidencyBackend::runtime_exit_gdn_prefill_resident_state_layer_scope(&*backend);
        ResidencyBackend::runtime_apply_gdn_prefill_resident_state_boundary(
            &*backend,
            owner_id,
            layer_idx,
            &mut resident_state,
        )?;
        ResidencyBackend::runtime_exit_gdn_prefill_resident_state_scope(&*backend);

        // Resume with an unrelated, stale zero handle on another worker. The
        // request/layer key must find the first chunk's device state without a
        // TensorId transfer at this boundary.
        let resume_backend = Arc::clone(&backend);
        let (resident_out2, resumed_state) = std::thread::spawn(move || {
            let persistent_state =
                tensor(initial, (1, 1, 2, 2))?.to_dtype(kiln_tensor::DType::BF16)?;
            let mut next_work_state = persistent_state.to_dtype(kiln_tensor::DType::F32)?;
            ResidencyBackend::runtime_enter_gdn_prefill_resident_state_scope(
                &*resume_backend,
                owner_id,
            );
            ResidencyBackend::runtime_enter_gdn_prefill_resident_state_layer_scope(
                &*resume_backend,
                layer_idx,
            );
            let out = gdn_chunkwise_forward(
                &resume_backend,
                &q2,
                &k2,
                &v2,
                &beta2,
                &g2,
                &mut next_work_state,
                4,
            )?
            .context("resident second chunk declined")?;
            let stale_external_state = next_work_state.to_dtype(kiln_tensor::DType::BF16)?;
            assert!(
                ResidencyBackend::runtime_rekey_gdn_recurrent_resident_state(
                    &*resume_backend,
                    &next_work_state,
                    &stale_external_state,
                )?
            );
            ResidencyBackend::runtime_exit_gdn_prefill_resident_state_layer_scope(&*resume_backend);
            ResidencyBackend::runtime_exit_gdn_prefill_resident_state_scope(&*resume_backend);
            Ok::<_, anyhow::Error>((out, stale_external_state))
        })
        .join()
        .map_err(|_| anyhow::anyhow!("cross-thread resident resume panicked"))??;

        assert_eq!(resident_state.id(), stable_id);
        assert!(!ResidencyBackend::runtime_has_gdn_recurrent_resident_state(
            &*backend,
            &resident_state,
        ));
        assert!(ResidencyBackend::runtime_has_gdn_recurrent_resident_state(
            &*backend,
            &resumed_state,
        ));
        let resident_stats =
            ResidencyBackend::runtime_gdn_recurrent_state_residency_stats(&*backend);
        assert_eq!(resident_stats.entry_count, 1);
        assert!(resident_stats.buffer_bytes >= 4 * std::mem::size_of::<f32>() as u64);
        assert!(resident_stats.allocation_bytes >= resident_stats.buffer_bytes);
        assert_eq!(
            values(&resident_out1)?,
            values(&quantized_out1)?,
            "first output must be bit-exact"
        );
        assert_eq!(
            values(&resident_out2)?,
            values(&quantized_out2)?,
            "second output must be bit-exact"
        );

        let materialize_backend = Arc::clone(&backend);
        let resident_state = std::thread::spawn(move || -> Result<kiln_tensor::Tensor> {
            ResidencyBackend::runtime_materialize_gdn_prefill_resident_state(
                &*materialize_backend,
                owner_id,
                layer_idx,
                &mut resident_state,
            )?;
            Ok(resident_state)
        })
        .join()
        .map_err(|_| anyhow::anyhow!("cross-thread resident materialization panicked"))??;
        assert!(!ResidencyBackend::runtime_has_gdn_recurrent_resident_state(
            &*backend,
            &resident_state,
        ));
        assert_eq!(
            ResidencyBackend::runtime_gdn_recurrent_state_residency_stats(&*backend),
            crate::backend::GdnRecurrentStateResidencyStats::default()
        );
        assert_eq!(
            values(&resident_state.to_dtype(kiln_tensor::DType::F32)?)?,
            values(&quantized_state.to_dtype(kiln_tensor::DType::F32)?)?,
            "final BF16 state must be bit-exact"
        );

        ResidencyBackend::runtime_evict_gdn_recurrent_resident_state(&*backend, &resident_state);
        Ok(())
    }

    #[test]
    fn prefill_owner_eviction_is_exact() -> Result<()> {
        let mut backend = VulkanBackend::new(kiln_tensor::Device::Cpu);
        backend.prefill_recurrent_state_residency_enabled = true;
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping resident owner eviction test");
            return Ok(());
        }
        let [q, k, v, beta, g] = inputs(3, 3)?;
        let mut state_a = tensor(vec![0.0; 4], (1, 1, 2, 2))?;
        let mut state_b = tensor(vec![0.0; 4], (1, 1, 2, 2))?;

        for (owner_id, state) in [(101, &mut state_a), (202, &mut state_b)] {
            assert!(
                ResidencyBackend::runtime_enter_gdn_prefill_resident_state_scope(
                    &backend, owner_id,
                )
            );
            assert!(
                ResidencyBackend::runtime_enter_gdn_prefill_resident_state_layer_scope(&backend, 0,)
            );
            gdn_chunkwise_forward(&backend, &q, &k, &v, &beta, &g, state, 4)?
                .context("resident owner-eviction chunk declined")?;
            ResidencyBackend::runtime_exit_gdn_prefill_resident_state_layer_scope(&backend);
            ResidencyBackend::runtime_exit_gdn_prefill_resident_state_scope(&backend);
        }
        assert_eq!(
            ResidencyBackend::runtime_gdn_recurrent_state_residency_stats(&backend).entry_count,
            2
        );

        ResidencyBackend::runtime_evict_gdn_prefill_resident_state_owner(&backend, 101);
        assert_eq!(
            ResidencyBackend::runtime_gdn_recurrent_state_residency_stats(&backend).entry_count,
            1
        );
        assert!(!ResidencyBackend::runtime_has_gdn_recurrent_resident_state(
            &backend, &state_a,
        ));
        assert!(ResidencyBackend::runtime_has_gdn_recurrent_resident_state(
            &backend, &state_b,
        ));

        ResidencyBackend::runtime_evict_gdn_prefill_resident_state_owner(&backend, 202);
        assert_eq!(
            ResidencyBackend::runtime_gdn_recurrent_state_residency_stats(&backend),
            crate::backend::GdnRecurrentStateResidencyStats::default()
        );
        Ok(())
    }
}

pub(super) fn gdn_gates(
    backend: &VulkanBackend,
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
    // kt guards read directly off the kt args before the bridge.
    if !supports_gdn_gates(backend) {
        return Ok(None);
    }
    if !matches!(
        a.dtype(),
        kiln_tensor::DType::BF16 | kiln_tensor::DType::F32
    ) {
        return Ok(None);
    }
    // (#1082) kt-native: weight buffers keyed on the stable kt id; byte
    // extraction + reconstruction run on the kt args.
    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
    let nv = a_log.elem_count();
    if dt_bias.elem_count() != nv {
        return Ok(None);
    }
    let a_log_buf = backend.cached_f32_weight_buffer_kt(a_log)?;
    let dt_bias_buf = backend.cached_f32_weight_buffer_kt(dt_bias)?;

    // Output shape matches input shape [B, T, nv]
    let out_shape = a.dims().to_vec();
    let a_data = kt_tensor_to_f32_bytes_with_shape(a)?.0;
    let b_data = kt_tensor_to_f32_bytes_with_shape(b)?.0;
    let output_dtype = a.dtype();
    let (beta_b, g_b) = kiln_vulkan_kernel::kernels::dispatch_gdn_gates_cached_bytes(
        vk_device,
        &a_data,
        &b_data,
        &a_log_buf,
        &dt_bias_buf,
        nv,
        &out_shape,
    )
    .context("gdn_gates kernel failed")?;
    let beta = kt_tensor_from_f32_bytes(&beta_b, &out_shape, output_dtype)?;
    let g = kt_tensor_from_f32_bytes(&g_b, &out_shape, output_dtype)?;
    Ok(Some((beta, g)))
}

pub(super) fn gdn_gated_rms_norm(
    backend: &VulkanBackend,
    x: &kiln_tensor::Tensor,
    z: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    eps: f64,
) -> Result<Option<kiln_tensor::Tensor>> {
    // kt guards read directly off the kt args before the bridge.
    if !supports_gdn_gated_rms_norm(backend) {
        return Ok(None);
    }
    if !matches!(
        x.dtype(),
        kiln_tensor::DType::BF16 | kiln_tensor::DType::F32
    ) {
        return Ok(None);
    }
    // (#1082) kt-native: weight buffer keyed on the stable kt id; byte
    // extraction + reconstruction run on the kt args.
    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
    let hidden = weight.elem_count();
    if hidden == 0 || x.elem_count() % hidden != 0 {
        return Ok(None);
    }
    let weight_buf = backend.cached_f32_weight_buffer_kt(weight)?;

    // Output shape matches x shape
    let out_shape = x.dims().to_vec();
    let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
    let z_data = kt_tensor_to_f32_bytes_with_shape(z)?.0;
    let output_dtype = x.dtype();
    let out_data = kiln_vulkan_kernel::kernels::dispatch_gdn_gated_rms_norm_cached_bytes(
        vk_device,
        &x_data,
        &z_data,
        &weight_buf,
        hidden,
        eps as f32,
        &out_shape,
    )
    .context("gdn_gated_rms_norm kernel failed")?;
    let out = kt_tensor_from_f32_bytes(&out_data, &out_shape, output_dtype)?;
    Ok(Some(out))
}
