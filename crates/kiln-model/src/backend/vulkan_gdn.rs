//! Vulkan Gated DeltaNet operation helpers.
//!
//! This owns Vulkan's kt-facing GDN support gates and dispatch hooks while
//! `backend/vulkan.rs` remains the `BackendRuntime` facade.

use anyhow::{Context, Result};
use std::sync::OnceLock;

use super::vulkan::VulkanBackend;
use super::vulkan_residency::{
    get_recurrent_state_resident_buffer, insert_recurrent_state_resident_buffer,
    recurrent_state_resident_scope_active,
};
use super::vulkan_tensor_bridge::{
    kt_tensor_from_f32_bytes, kt_tensor_to_f32_bytes_with_shape,
    upload_gdn_chunkwise_inputs_from_cpu_bytes_vk, vk_f32_tensors_to_cpu_tensors_batched_vk,
};

fn fused_gdn_resident_state_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_DISABLE_VULKAN_GDN_DECODE_FUSED_RESIDENT_STATE").is_err()
    })
}

pub(super) fn supports_gdn_forward_substitution(backend: &VulkanBackend) -> bool {
    // solve_tri is experimental: shared-memory layout not yet validated
    // against CPU parity, and may exceed maxComputeSharedMemorySize on many
    // GPUs. Opt-in only via KILN_ENABLE_VULKAN_GDN_FORWARD_SUB.
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
    if std::env::var("KILN_DISABLE_VULKAN_GDN_CHUNKWISE_FORWARD").is_ok() {
        return Ok(None);
    }
    let Some(vk_device) = backend.vulkan_device() else {
        return Ok(None);
    };

    let state_shape = state_kt.shape().to_vec();
    let (q_vk, k_vk, v_vk, beta_vk, g_vk, mut state_vk) =
        if let Some([q_vk, k_vk, v_vk, beta_vk, g_vk, state_vk]) =
            upload_gdn_chunkwise_inputs_from_cpu_bytes_vk(vk_device, q, k, v, beta, g, state_kt)?
        {
            (q_vk, k_vk, v_vk, beta_vk, g_vk, state_vk)
        } else {
            let load =
                |t: &kiln_tensor::Tensor| -> Result<kiln_vulkan_kernel::vk_tensor::VkTensor> {
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
            (
                load(q)?,
                load(k)?,
                load(v)?,
                load(beta)?,
                load(g)?,
                load(state_kt)?,
            )
        };

    let out_vk = if std::env::var("KILN_DISABLE_VULKAN_GDN_CHUNKWISE_SINGLE_SUBMIT").is_ok() {
        if kiln_core::env_flag::env_flag("KILN_VULKAN_GDN_CHUNKWISE_FALLBACK", false) {
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
                if kiln_core::env_flag::env_flag("KILN_VULKAN_GDN_CHUNKWISE_FALLBACK", false) {
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

    // Read back output + the updated state together, then rebuild CPU-host
    // kt tensors without decoding through an intermediate Vec<f32>.
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
        let resident_state = get_recurrent_state_resident_buffer(state_id);
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
        insert_recurrent_state_resident_buffer(state_id, resident_state);
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

    if backend.recurrent_state_residency_enabled && recurrent_state_resident_scope_active() {
        let state_id = state_kt.id();
        let resident_state = get_recurrent_state_resident_buffer(state_id);

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

        insert_recurrent_state_resident_buffer(state_id, resident_state);
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
    if backend.recurrent_state_residency_enabled
        && recurrent_state_resident_scope_active()
        && state_kt.dtype() == q.dtype()
    {
        let state_id = state_kt.id();
        let resident_state = get_recurrent_state_resident_buffer(state_id);
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
        insert_recurrent_state_resident_buffer(state_id, resident_state);
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
