//! Vulkan Gated DeltaNet operation helpers.
//!
//! This starts the GDN operation-family split with the small gate and gated
//! RMSNorm dispatch hooks. The larger recurrent/chunkwise paths stay in the
//! `BackendRuntime` facade until they can move in narrower slices.

use anyhow::{Context, Result};

use super::vulkan::VulkanBackend;
use super::vulkan_residency::{
    get_recurrent_state_resident_buffer, insert_recurrent_state_resident_buffer,
    recurrent_state_resident_scope_active,
};
use super::vulkan_tensor_bridge::{kt_tensor_from_f32_bytes, kt_tensor_to_f32_bytes_with_shape};

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
    if a_strict.dtype() != kiln_tensor::DType::BF16 {
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
