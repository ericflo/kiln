//! Metal backend: candle's fused SDPA for the attention hot path, portable
//! fallback for GDN and paged-decode.
//!
//! candle-metal ships `candle_nn::ops::sdpa` — an MLX-style fused scaled-dot-
//! product attention kernel with native GQA, BF16, and head dims
//! {32, 64, 72, 80, 96, 128, 256, 512}. For typical transformer head sizes
//! this replaces the vendored CUDA FlashAttention-2 call on Apple Silicon.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

use super::BackendRuntime;

const DISABLE_METAL_CONV1D_PREFILL: &str = "KILN_DISABLE_METAL_CONV1D_PREFILL";
const DISABLE_GDN_KERNEL: &str = "KILN_DISABLE_GDN_KERNEL";
const DISABLE_METAL_GDN_RECURRENT: &str = "KILN_DISABLE_METAL_GDN_RECURRENT";

#[derive(Debug)]
pub struct MetalBackend {
    device: Device,
}

impl MetalBackend {
    pub fn new(device: Device) -> Self {
        debug_assert!(
            matches!(device, Device::Metal(_)),
            "MetalBackend created on non-Metal device"
        );
        Self { device }
    }
}

impl BackendRuntime for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn supports_flash_attn_prefill(&self) -> bool {
        true
    }

    fn supports_flash_attn_prefill_head_major(&self) -> bool {
        true
    }

    fn supports_flash_attn_paged_decode(&self) -> bool {
        true
    }

    fn supports_causal_conv1d_prefill(&self) -> bool {
        !metal_conv1d_prefill_disabled()
    }

    fn supports_gdn_recurrent_step(&self) -> bool {
        !metal_gdn_recurrent_disabled()
    }

    fn flash_attn_prefill(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<Tensor>> {
        // Decline (caller falls back to the portable path) when candle's SDPA
        // can't handle the shape/dtype. Cheaper than surfacing a kernel error
        // from inside the fused path.
        if !matches!(q.dtype(), DType::BF16 | DType::F16 | DType::F32) {
            return Ok(None);
        }
        if !metal_sdpa_supports_head_dim(q.dim(candle_core::D::Minus1)?) {
            return Ok(None);
        }

        let q_t = q.transpose(1, 2)?.contiguous()?;
        let k_t = k.transpose(1, 2)?.contiguous()?;
        let v_t = v.transpose(1, 2)?.contiguous()?;

        // sdpa(q, k, v, mask, do_causal, scale, softcapping). softcapping=1.0
        // disables it; kiln's prefill path is always causal.
        let out = candle_nn::ops::sdpa(&q_t, &k_t, &v_t, None, causal, softmax_scale, 1.0)
            .context("candle-metal sdpa failed")?;

        let out = out.transpose(1, 2)?.contiguous()?;
        Ok(Some(out))
    }

    fn flash_attn_prefill_head_major(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<Tensor>> {
        if !matches!(q.dtype(), DType::BF16 | DType::F16 | DType::F32) {
            return Ok(None);
        }
        if !metal_sdpa_supports_head_dim(q.dim(candle_core::D::Minus1)?) {
            return Ok(None);
        }

        let out = candle_nn::ops::sdpa(q, k, v, None, causal, softmax_scale, 1.0)
            .context("candle-metal head-major sdpa failed")?;
        Ok(Some(out))
    }

    /// Gather K/V from the paged pool via `index_select` on the block table,
    /// then call candle's vectorized SDPA (single-query path). The gather
    /// replaces the slow materializing `paged_cache.read` +
    /// naive-softmax+matmul fallback — same result, one fused kernel.
    fn flash_attn_paged_decode(
        &self,
        q: &Tensor,
        k_pool: &Tensor,
        v_pool: &Tensor,
        block_table: &Tensor,
        total_seqlen_k: usize,
        page_block_size: usize,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<Tensor>> {
        // Gate on everything SDPA can handle. Pool dtype matches q dtype by
        // construction (both come from the same forward config), so only q
        // needs checking.
        if !matches!(q.dtype(), DType::BF16 | DType::F16 | DType::F32) {
            return Ok(None);
        }
        let head_dim = q.dim(candle_core::D::Minus1)?;
        if !metal_sdpa_supports_head_dim(head_dim) {
            return Ok(None);
        }

        let (batch, q_len, num_heads, _) = q.dims4()?;
        if batch != 1 || q_len != 1 {
            // Multi-sequence paged decode would need a per-sequence gather.
            // Stay on the fallback until the scheduler exercises it.
            return Ok(None);
        }

        let (total_slots, num_kv_heads, _) = k_pool.dims3()?;
        if total_slots % page_block_size != 0 {
            return Ok(None);
        }
        let num_blocks = total_slots / page_block_size;
        let max_blocks_per_seq = block_table.dim(1)?;

        // [num_blocks, block_size, num_kv_heads, head_dim] so index_select on
        // dim 0 gathers a full logical block's slots per physical block id.
        let k_blocks = k_pool.reshape((num_blocks, page_block_size, num_kv_heads, head_dim))?;
        let v_blocks = v_pool.reshape((num_blocks, page_block_size, num_kv_heads, head_dim))?;

        // The block_table is identical across all 8 full-attention layers in
        // a decode step, but the trait forces us to re-flatten it per call.
        // Threading a pre-flattened handle through the trait would save
        // ~8× redundant flattens per token; defer until the signature can
        // grow a cache parameter.
        let block_ids = block_table.flatten_all()?;

        let k_gathered = k_blocks.index_select(&block_ids, 0)?;
        let v_gathered = v_blocks.index_select(&block_ids, 0)?;

        // [max_blocks_per_seq * block_size, num_kv_heads, head_dim] then
        // narrow to the live KV length.
        let total_gathered = max_blocks_per_seq * page_block_size;
        let k_flat = k_gathered.reshape((total_gathered, num_kv_heads, head_dim))?;
        let v_flat = v_gathered.reshape((total_gathered, num_kv_heads, head_dim))?;
        let k_live = k_flat.narrow(0, 0, total_seqlen_k)?;
        let v_live = v_flat.narrow(0, 0, total_seqlen_k)?;

        // SDPA needs [batch, num_heads, seq, head_dim]. Q arrives as
        // [1, 1, num_heads, head_dim]; K/V are [total_seqlen_k, num_kv_heads, head_dim].
        // SDPA handles GQA internally when num_heads % num_kv_heads == 0.
        let q_sdpa = q.transpose(1, 2)?.contiguous()?; // [1, num_heads, 1, head_dim]
        let k_sdpa = k_live.unsqueeze(0)?.transpose(1, 2)?.contiguous()?; // [1, num_kv_heads, total_seqlen_k, head_dim]
        let v_sdpa = v_live.unsqueeze(0)?.transpose(1, 2)?.contiguous()?;

        let out = candle_nn::ops::sdpa(&q_sdpa, &k_sdpa, &v_sdpa, None, causal, softmax_scale, 1.0)
            .context("candle-metal paged sdpa failed")?;

        // Back to [1, 1, num_heads, head_dim].
        let out = out.transpose(1, 2)?.contiguous()?;
        debug_assert_eq!(out.dims(), &[1, 1, num_heads, head_dim]);
        Ok(Some(out))
    }

    fn causal_conv1d_prefill(
        &self,
        x: &Tensor,
        weight: &Tensor,
        conv_state: &mut Tensor,
        kernel_size: usize,
    ) -> Result<Option<Tensor>> {
        if metal_conv1d_prefill_disabled()
            || !metal_conv1d_prefill_supports(x, weight, conv_state, kernel_size)
        {
            return Ok(None);
        }
        let out = metal_causal_conv1d_prefill_bf16_f32_k4(x, weight, conv_state, kernel_size)
            .context("metal causal_conv1d_prefill kernel failed")?;
        Ok(Some(out))
    }

    fn gdn_recurrent_step(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        beta: &Tensor,
        g: &Tensor,
        state: &mut Tensor,
    ) -> Result<Option<Tensor>> {
        if metal_gdn_recurrent_disabled() || !metal_gdn_recurrent_supports(q, k, v, beta, g, state)
        {
            return Ok(None);
        }
        let out = metal_gdn_recurrent_bf16(q, k, v, beta, g, state)
            .context("metal gdn_recurrent_step kernel failed")?;
        Ok(Some(out))
    }
}

/// Mirrors the head-dim whitelist in candle-nn 0.10.2's
/// `Sdpa::custom_op3` (see `ops.rs`). Drifts silently if the upstream
/// list grows — the fallback path absorbs the mismatch (correct, just
/// slower). Re-check this on candle bumps.
fn metal_sdpa_supports_head_dim(head_dim: usize) -> bool {
    matches!(head_dim, 32 | 64 | 72 | 80 | 96 | 128 | 256 | 512)
}

fn metal_conv1d_prefill_disabled() -> bool {
    env_truthy(DISABLE_METAL_CONV1D_PREFILL)
}

fn metal_gdn_recurrent_disabled() -> bool {
    env_truthy(DISABLE_GDN_KERNEL) || env_truthy(DISABLE_METAL_GDN_RECURRENT)
}

fn env_truthy(var: &str) -> bool {
    matches!(
        std::env::var(var)
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

fn metal_conv1d_prefill_supports(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &Tensor,
    kernel_size: usize,
) -> bool {
    if kernel_size != 4 {
        return false;
    }
    if !matches!(x.device(), Device::Metal(_)) {
        return false;
    }
    if x.dtype() != DType::BF16 || weight.dtype() != DType::BF16 || conv_state.dtype() != DType::F32
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

fn metal_gdn_recurrent_supports(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    state: &Tensor,
) -> bool {
    if !matches!(q.device(), Device::Metal(_))
        || !matches!(k.device(), Device::Metal(_))
        || !matches!(v.device(), Device::Metal(_))
        || !matches!(beta.device(), Device::Metal(_))
        || !matches!(g.device(), Device::Metal(_))
        || !matches!(state.device(), Device::Metal(_))
    {
        return false;
    }
    if q.dtype() != DType::BF16
        || k.dtype() != DType::BF16
        || v.dtype() != DType::BF16
        || beta.dtype() != DType::BF16
        || g.dtype() != DType::BF16
        || state.dtype() != DType::BF16
    {
        return false;
    }
    let Ok((batch, heads, dk)) = q.dims3() else {
        return false;
    };
    let Ok((b_k, h_k, dk_k)) = k.dims3() else {
        return false;
    };
    let Ok((b_v, h_v, dv)) = v.dims3() else {
        return false;
    };
    let Ok((b_b, h_b)) = beta.dims2() else {
        return false;
    };
    let Ok((b_g, h_g)) = g.dims2() else {
        return false;
    };
    let Ok((b_s, h_s, dk_s, dv_s)) = state.dims4() else {
        return false;
    };
    (b_k, h_k, dk_k) == (batch, heads, dk)
        && (b_v, h_v) == (batch, heads)
        && (b_b, h_b) == (batch, heads)
        && (b_g, h_g) == (batch, heads)
        && (b_s, h_s, dk_s, dv_s) == (batch, heads, dk, dv)
        && dk <= 256
        && dv <= 1024
}

const METAL_GDN_RECURRENT_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gdn_recurrent_bf16(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device const bfloat* beta [[buffer(3)]],
    device const bfloat* g [[buffer(4)]],
    device bfloat* state [[buffer(5)]],
    device bfloat* out [[buffer(6)]],
    constant uint& batch_heads [[buffer(7)]],
    constant uint& dk [[buffer(8)]],
    constant uint& dv [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = batch_heads * dv;
    if (gid >= total) {
        return;
    }

    const uint bh = gid / dv;
    const uint col = gid - bh * dv;
    const uint qk_base = bh * dk;
    const uint v_base = bh * dv;
    const uint state_base = bh * dk * dv;

    const float decay = exp(static_cast<float>(g[bh]));
    const float beta_t = static_cast<float>(beta[bh]);

    float v_pred = 0.0f;
    for (uint i = 0; i < dk; ++i) {
        const float k_i = static_cast<float>(k[qk_base + i]);
        const float s_i = static_cast<float>(state[state_base + i * dv + col]);
        v_pred += k_i * (decay * s_i);
    }

    const float v_t = static_cast<float>(v[v_base + col]);
    const float delta = beta_t * (v_t - v_pred);

    float out_acc = 0.0f;
    for (uint i = 0; i < dk; ++i) {
        const float q_i = static_cast<float>(q[qk_base + i]);
        const float k_i = static_cast<float>(k[qk_base + i]);
        const uint state_idx = state_base + i * dv + col;
        const float old_s = static_cast<float>(state[state_idx]);
        const float new_s = decay * old_s + k_i * delta;
        state[state_idx] = static_cast<bfloat>(new_s);
        out_acc += q_i * new_s;
    }

    out[v_base + col] = static_cast<bfloat>(out_acc);
}
"#;

fn metal_gdn_recurrent_pipeline(
    device: &candle_core::metal_backend::MetalDevice,
) -> Result<candle_metal_kernels::metal::ComputePipeline> {
    use candle_core::metal_backend::DeviceId;
    use candle_metal_kernels::metal::ComputePipeline;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    static PIPELINES: OnceLock<Mutex<HashMap<DeviceId, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn recurrent pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = device
        .device()
        .new_library_with_source(METAL_GDN_RECURRENT_KERNEL, None)
        .map_err(|e| anyhow::anyhow!("compile metal gdn recurrent library: {e:?}"))?;
    let function = library
        .get_function("kiln_gdn_recurrent_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn recurrent function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn recurrent pipeline: {e:?}"))?;
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gdn_recurrent_bf16(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    let (batch, heads, dk) = q.dims3()?;
    let dv = v.dim(2)?;
    let batch_heads = batch * heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize && dk <= u32::MAX as usize && dv <= u32::MAX as usize,
        "metal gdn recurrent shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let beta = beta.contiguous()?;
    let g = g.contiguous()?;
    if !state.is_contiguous() {
        *state = state.contiguous()?;
    }
    let out = Tensor::zeros((batch, heads, dv), DType::BF16, q.device())?;

    let Device::Metal(device) = q.device() else {
        anyhow::bail!("metal gdn recurrent requires a Metal tensor");
    };
    let pipeline = metal_gdn_recurrent_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_gdn_recurrent_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (q_storage, q_layout) = q.storage_and_layout();
        let (k_storage, k_layout) = k.storage_and_layout();
        let (v_storage, v_layout) = v.storage_and_layout();
        let (beta_storage, beta_layout) = beta.storage_and_layout();
        let (g_storage, g_layout) = g.storage_and_layout();
        let (state_storage, state_layout) = state.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let q_metal = match &*q_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn recurrent q must be on Metal"),
        };
        let k_metal = match &*k_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn recurrent k must be on Metal"),
        };
        let v_metal = match &*v_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn recurrent v must be on Metal"),
        };
        let beta_metal = match &*beta_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn recurrent beta must be on Metal"),
        };
        let g_metal = match &*g_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn recurrent g must be on Metal"),
        };
        let state_metal = match &*state_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn recurrent state must be on Metal"),
        };
        let out_metal = match &*out_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn recurrent out must be on Metal"),
        };

        let q_buf = candle_core::metal_backend::buffer_o(q_metal.buffer(), &q_layout, q.dtype());
        let k_buf = candle_core::metal_backend::buffer_o(k_metal.buffer(), &k_layout, k.dtype());
        let v_buf = candle_core::metal_backend::buffer_o(v_metal.buffer(), &v_layout, v.dtype());
        let beta_buf =
            candle_core::metal_backend::buffer_o(beta_metal.buffer(), &beta_layout, beta.dtype());
        let g_buf = candle_core::metal_backend::buffer_o(g_metal.buffer(), &g_layout, g.dtype());
        let state_buf = candle_core::metal_backend::buffer_o(
            state_metal.buffer(),
            &state_layout,
            state.dtype(),
        );
        let out_buf =
            candle_core::metal_backend::buffer_o(out_metal.buffer(), &out_layout, out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(g_buf.buffer), g_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        encoder.set_bytes(7, &batch_heads_u32);
        encoder.set_bytes(8, &dk_u32);
        encoder.set_bytes(9, &dv_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: batch_heads * dv,
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

const METAL_CONV1D_PREFILL_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_causal_conv1d_prefill_bf16_f32_k4(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device float* conv_state [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_channels = batch * channels;
    if (gid >= total_channels) {
        return;
    }

    const uint b = gid / channels;
    const uint c = gid - b * channels;
    const uint x_base = (b * channels + c) * seq_len;
    const uint state_base = (b * channels + c) * 3;
    const uint weight_base = c * 4;

    for (uint t = 0; t < seq_len; ++t) {
        float acc = 0.0f;
        for (uint j = 0; j < 4; ++j) {
            const uint padded_idx = t + j;
            float v;
            if (padded_idx < 3) {
                v = conv_state[state_base + padded_idx];
            } else {
                v = static_cast<float>(x[x_base + padded_idx - 3]);
            }
            acc += v * static_cast<float>(weight[weight_base + j]);
        }
        out[x_base + t] = acc / (1.0f + exp(-acc));
    }

    if (seq_len >= 3) {
        conv_state[state_base + 0] = static_cast<float>(x[x_base + seq_len - 3]);
        conv_state[state_base + 1] = static_cast<float>(x[x_base + seq_len - 2]);
        conv_state[state_base + 2] = static_cast<float>(x[x_base + seq_len - 1]);
    } else if (seq_len == 2) {
        conv_state[state_base + 0] = conv_state[state_base + 2];
        conv_state[state_base + 1] = static_cast<float>(x[x_base + 0]);
        conv_state[state_base + 2] = static_cast<float>(x[x_base + 1]);
    } else if (seq_len == 1) {
        conv_state[state_base + 0] = conv_state[state_base + 1];
        conv_state[state_base + 1] = conv_state[state_base + 2];
        conv_state[state_base + 2] = static_cast<float>(x[x_base]);
    }
}
"#;

fn metal_conv1d_prefill_pipeline(
    device: &candle_core::metal_backend::MetalDevice,
) -> Result<candle_metal_kernels::metal::ComputePipeline> {
    use candle_core::metal_backend::DeviceId;
    use candle_metal_kernels::metal::ComputePipeline;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    static PIPELINES: OnceLock<Mutex<HashMap<DeviceId, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal conv1d prefill pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = device
        .device()
        .new_library_with_source(METAL_CONV1D_PREFILL_KERNEL, None)
        .map_err(|e| anyhow::anyhow!("compile metal conv1d prefill library: {e:?}"))?;
    let function = library
        .get_function("kiln_causal_conv1d_prefill_bf16_f32_k4", None)
        .map_err(|e| anyhow::anyhow!("load metal conv1d prefill function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal conv1d prefill pipeline: {e:?}"))?;
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

fn metal_causal_conv1d_prefill_bf16_f32_k4(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &mut Tensor,
    kernel_size: usize,
) -> Result<Tensor> {
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
    let out = Tensor::zeros((batch, channels, seq_len), DType::F32, x.device())?;

    let Device::Metal(device) = x.device() else {
        anyhow::bail!("metal conv1d prefill requires a Metal tensor");
    };
    let pipeline = metal_conv1d_prefill_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_causal_conv1d_prefill_bf16_f32_k4");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (w_storage, w_layout) = weight.storage_and_layout();
        let (s_storage, s_layout) = conv_state.storage_and_layout();
        let (o_storage, o_layout) = out.storage_and_layout();

        let x_metal = match &*x_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal conv1d prefill x must be on Metal"),
        };
        let w_metal = match &*w_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal conv1d prefill weight must be on Metal"),
        };
        let s_metal = match &*s_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal conv1d prefill state must be on Metal"),
        };
        let o_metal = match &*o_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal conv1d prefill output must be on Metal"),
        };

        let x_buf = candle_core::metal_backend::buffer_o(x_metal.buffer(), &x_layout, x.dtype());
        let w_buf =
            candle_core::metal_backend::buffer_o(w_metal.buffer(), &w_layout, weight.dtype());
        let s_buf =
            candle_core::metal_backend::buffer_o(s_metal.buffer(), &s_layout, conv_state.dtype());
        let o_buf = candle_core::metal_backend::buffer_o(o_metal.buffer(), &o_layout, out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(s_buf.buffer), s_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(o_buf.buffer), o_buf.offset_in_bytes);

        let batch_u32 = batch as u32;
        let channels_u32 = channels as u32;
        let seq_len_u32 = seq_len as u32;
        encoder.set_bytes(4, &batch_u32);
        encoder.set_bytes(5, &channels_u32);
        encoder.set_bytes(6, &seq_len_u32);

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

/// Test helper: try to initialize a Metal device, returning `None` if Metal
/// isn't available OR if candle-metal's `MetalDevice::new` panics (observed on
/// GitHub's macos-14 runners, where the CI sandbox can produce an empty device
/// list and candle 0.10.2's `swap_remove` panics instead of returning `Err`).
#[doc(hidden)]
pub fn try_new_metal() -> Option<Device> {
    let result = std::panic::catch_unwind(|| Device::new_metal(0));
    match result {
        Ok(Ok(d)) => Some(d),
        Ok(Err(e)) => {
            eprintln!("Metal unavailable: {e}");
            None
        }
        Err(_) => {
            eprintln!("Metal device init panicked (likely CI sandbox with no Metal access)");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::D;

    /// Parity: `MetalBackend::flash_attn_paged_decode` output matches a
    /// direct materialize+SDPA reference computation on the same inputs.
    /// Validates the paged gather (index_select + narrow) logic.
    #[test]
    fn test_paged_decode_parity_with_direct_sdpa() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let num_heads = 4;
        let num_kv_heads = 1;
        let head_dim = 128;
        let block_size = 16;
        let num_blocks = 8;
        let total_slots = num_blocks * block_size;
        let max_blocks_per_seq = 4;
        let total_seqlen_k = 50; // covers 4 full blocks (64 slots) but only 50 valid.

        // Shuffled physical block table — exercises the gather, not just
        // sequential blocks.
        let block_ids: [u32; 4] = [3, 7, 0, 5];
        let block_table =
            Tensor::new(block_ids.as_slice(), &device)?.reshape((1usize, max_blocks_per_seq))?;

        // Fill the pool with distinctive per-slot values so the gather's
        // correctness is visible in the output. Each slot's values are
        // `slot_idx * 0.0001 + head_dim_offset * 0.000001`.
        let k_pool_data: Vec<f32> = (0..total_slots * num_kv_heads * head_dim)
            .map(|i| (i as f32) * 0.0001)
            .collect();
        let v_pool_data: Vec<f32> = (0..total_slots * num_kv_heads * head_dim)
            .map(|i| (i as f32) * 0.0001 + 1.0)
            .collect();
        let k_pool =
            Tensor::from_slice(&k_pool_data, (total_slots, num_kv_heads, head_dim), &device)?;
        let v_pool =
            Tensor::from_slice(&v_pool_data, (total_slots, num_kv_heads, head_dim), &device)?;

        let q = Tensor::randn(0.0f32, 0.02, (1, 1, num_heads, head_dim), &device)?;
        let softmax_scale = 1.0 / (head_dim as f32).sqrt();
        let backend = MetalBackend::new(device.clone());

        let out_paged = backend
            .flash_attn_paged_decode(
                &q,
                &k_pool,
                &v_pool,
                &block_table,
                total_seqlen_k,
                block_size,
                softmax_scale,
                true,
            )?
            .expect("backend should handle this shape");

        // Reference: manually gather K/V the same way and call SDPA.
        let k_blocks = k_pool.reshape((num_blocks, block_size, num_kv_heads, head_dim))?;
        let v_blocks = v_pool.reshape((num_blocks, block_size, num_kv_heads, head_dim))?;
        let ids = block_table.flatten_all()?;
        let k_gathered = k_blocks
            .index_select(&ids, 0)?
            .reshape((max_blocks_per_seq * block_size, num_kv_heads, head_dim))?
            .narrow(0, 0, total_seqlen_k)?;
        let v_gathered = v_blocks
            .index_select(&ids, 0)?
            .reshape((max_blocks_per_seq * block_size, num_kv_heads, head_dim))?
            .narrow(0, 0, total_seqlen_k)?;

        let q_ref = q.transpose(1, 2)?.contiguous()?;
        let k_ref = k_gathered.unsqueeze(0)?.transpose(1, 2)?.contiguous()?;
        let v_ref = v_gathered.unsqueeze(0)?.transpose(1, 2)?.contiguous()?;
        let ref_out = candle_nn::ops::sdpa(&q_ref, &k_ref, &v_ref, None, true, softmax_scale, 1.0)?;
        let ref_out = ref_out.transpose(1, 2)?.contiguous()?;

        assert_eq!(out_paged.dims(), ref_out.dims());
        let diff = (&out_paged - &ref_out)?
            .abs()?
            .flatten_all()?
            .max(D::Minus1)?
            .to_scalar::<f32>()?;
        assert!(
            diff < 1e-5,
            "paged vs direct SDPA diverge: max abs diff = {diff}"
        );

        Ok(())
    }

    /// Non-SDPA head_dim should decline cleanly so the caller falls back.
    #[test]
    fn test_paged_decode_declines_on_unsupported_head_dim() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };
        let head_dim = 4; // not in whitelist
        let total_slots = 16;
        let k_pool = Tensor::zeros((total_slots, 1, head_dim), DType::F32, &device)?;
        let v_pool = Tensor::zeros((total_slots, 1, head_dim), DType::F32, &device)?;
        let block_table = Tensor::new(&[0u32, 0, 0, 0][..], &device)?.reshape((1usize, 4))?;
        let q = Tensor::zeros((1, 1, 2, head_dim), DType::F32, &device)?;

        let backend = MetalBackend::new(device);
        let out =
            backend.flash_attn_paged_decode(&q, &k_pool, &v_pool, &block_table, 4, 4, 1.0, true)?;
        assert!(out.is_none(), "should decline unsupported head_dim");
        Ok(())
    }
}
