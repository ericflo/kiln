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
const DISABLE_FUSED_CONV1D: &str = "KILN_DISABLE_FUSED_CONV1D";
const DISABLE_METAL_FUSED_CONV1D: &str = "KILN_DISABLE_METAL_FUSED_CONV1D";
const DISABLE_GDN_KERNEL: &str = "KILN_DISABLE_GDN_KERNEL";
const DISABLE_FUSED_GDN_GATES: &str = "KILN_DISABLE_FUSED_GDN_GATES";
const DISABLE_METAL_GDN_GATES: &str = "KILN_DISABLE_METAL_GDN_GATES";
const DISABLE_METAL_GDN_FORWARD_SUBSTITUTION: &str =
    "KILN_DISABLE_METAL_GDN_FORWARD_SUBSTITUTION";
const DISABLE_METAL_GDN_RECURRENT: &str = "KILN_DISABLE_METAL_GDN_RECURRENT";
const DISABLE_METAL_GATED_RMSNORM: &str = "KILN_DISABLE_METAL_GATED_RMSNORM";
const DISABLE_METAL_GDN_QK_NORM: &str = "KILN_DISABLE_METAL_GDN_QK_NORM";
const DISABLE_RMSNORM_KERNEL: &str = "KILN_DISABLE_RMSNORM_KERNEL";
const DISABLE_METAL_RMSNORM: &str = "KILN_DISABLE_METAL_RMSNORM";

#[derive(Debug)]
pub struct MetalBackend {
    device: Device,
    /// Cached at construction to keep env-var reads off per-token support gates.
    disable: MetalKernelDisables,
}

#[derive(Debug, Clone, Copy)]
struct MetalKernelDisables {
    conv1d_prefill: bool,
    conv1d_update: bool,
    gdn_forward_substitution: bool,
    gdn_recurrent: bool,
    gdn_gates: bool,
    gated_rms_norm: bool,
}

impl MetalKernelDisables {
    fn from_env() -> Self {
        let gdn_kernel = env_truthy(DISABLE_GDN_KERNEL);
        Self {
            conv1d_prefill: env_truthy(DISABLE_METAL_CONV1D_PREFILL),
            conv1d_update: env_present(DISABLE_FUSED_CONV1D)
                || env_truthy(DISABLE_METAL_FUSED_CONV1D),
            gdn_forward_substitution: gdn_kernel
                || env_truthy(DISABLE_METAL_GDN_FORWARD_SUBSTITUTION),
            gdn_recurrent: gdn_kernel || env_truthy(DISABLE_METAL_GDN_RECURRENT),
            gdn_gates: env_present(DISABLE_FUSED_GDN_GATES)
                || env_truthy(DISABLE_METAL_GDN_GATES),
            gated_rms_norm: env_truthy(DISABLE_METAL_GATED_RMSNORM),
        }
    }
}

impl MetalBackend {
    pub fn new(device: Device) -> Self {
        debug_assert!(
            matches!(device, Device::Metal(_)),
            "MetalBackend created on non-Metal device"
        );
        Self {
            device,
            disable: MetalKernelDisables::from_env(),
        }
    }
}

/// Compile Kiln's custom Metal library and compute pipelines ahead of the
/// first forward pass. Candle kernels still compile lazily inside Candle, but
/// this removes Kiln-owned pipeline setup from the first prewarm/request.
pub fn precompile_custom_kernels(device: &Device) -> Result<()> {
    let Device::Metal(metal_device) = device else {
        return Ok(());
    };

    metal_shared_library(metal_device)?;
    metal_rms_norm_pipeline(metal_device)?;
    metal_gdn_qk_norm_pipeline(metal_device)?;
    metal_gdn_gates_pipeline(metal_device)?;
    metal_gated_rms_norm_pipeline(metal_device)?;
    metal_gdn_recurrent_pipeline(metal_device)?;
    metal_gdn_forward_substitution_pipeline(metal_device)?;
    metal_gdn_chunk_prep_pipeline(metal_device)?;
    metal_conv1d_prefill_pipeline(metal_device)?;
    metal_conv1d_update_pipeline(metal_device)?;
    metal_lm_head_pipeline(metal_device)?;
    Ok(())
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
        !self.disable.conv1d_prefill
    }

    fn supports_causal_conv1d_update(&self) -> bool {
        !self.disable.conv1d_update
    }

    fn supports_gdn_forward_substitution(&self) -> bool {
        !self.disable.gdn_forward_substitution
    }

    fn supports_gdn_recurrent_step(&self) -> bool {
        !self.disable.gdn_recurrent
    }

    fn supports_gdn_chunk_prep(&self) -> bool {
        !self.disable.gdn_forward_substitution
    }

    fn supports_gdn_gates(&self) -> bool {
        !self.disable.gdn_gates
    }

    fn supports_gdn_gated_rms_norm(&self) -> bool {
        !self.disable.gated_rms_norm
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
        if self.disable.conv1d_prefill
            || !metal_conv1d_prefill_supports(x, weight, conv_state, kernel_size)
        {
            return Ok(None);
        }
        let out = metal_causal_conv1d_prefill_bf16_f32_k4(x, weight, conv_state, kernel_size)
            .context("metal causal_conv1d_prefill kernel failed")?;
        Ok(Some(out))
    }

    fn causal_conv1d_update(
        &self,
        x: &Tensor,
        weight: &Tensor,
        conv_state: &mut Tensor,
        kernel_size: usize,
    ) -> Result<Option<Tensor>> {
        if self.disable.conv1d_update
            || !metal_conv1d_update_supports(x, weight, conv_state, kernel_size)
        {
            return Ok(None);
        }
        let out = metal_causal_conv1d_update_bf16_f32_k4(x, weight, conv_state, kernel_size)
            .context("metal causal_conv1d_update kernel failed")?;
        Ok(Some(out))
    }

    fn gdn_forward_substitution(
        &self,
        a_strict: &Tensor,
        v_prime: &Tensor,
        beta: &Tensor,
    ) -> Result<Option<Tensor>> {
        if self.disable.gdn_forward_substitution
            || !metal_gdn_forward_substitution_supports(a_strict, v_prime, beta)
        {
            return Ok(None);
        }
        let out = metal_gdn_forward_substitution_bf16(a_strict, v_prime, beta)
            .context("metal gdn_forward_substitution kernel failed")?;
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
        if self.disable.gdn_recurrent || !metal_gdn_recurrent_supports(q, k, v, beta, g, state) {
            return Ok(None);
        }
        let out = metal_gdn_recurrent_bf16(q, k, v, beta, g, state)
            .context("metal gdn_recurrent_step kernel failed")?;
        Ok(Some(out))
    }

    fn gdn_chunk_prep(
        &self,
        g: &Tensor,
        v: &Tensor,
        kkt: &Tensor,
        qkt: &Tensor,
        ks_entry: &Tensor,
        q_s: &Tensor,
    ) -> Result<Option<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)>> {
        if self.disable.gdn_forward_substitution
            || !metal_gdn_chunk_prep_supports(g, v, kkt, qkt, ks_entry, q_s)
        {
            return Ok(None);
        }
        let out = metal_gdn_chunk_prep_bf16(g, v, kkt, qkt, ks_entry, q_s)
            .context("metal gdn_chunk_prep kernel failed")?;
        Ok(Some(out))
    }

    fn gdn_gates(
        &self,
        a: &Tensor,
        b: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
    ) -> Result<Option<(Tensor, Tensor)>> {
        if self.disable.gdn_gates || !metal_gdn_gates_supports(a, b, a_log, dt_bias) {
            return Ok(None);
        }
        let out = metal_gdn_gates_bf16(a, b, a_log, dt_bias)
            .context("metal gdn_gates kernel failed")?;
        Ok(Some(out))
    }

    fn gdn_gated_rms_norm(
        &self,
        x: &Tensor,
        z: &Tensor,
        weight: &Tensor,
        eps: f64,
    ) -> Result<Option<Tensor>> {
        if self.disable.gated_rms_norm || !metal_gated_rms_norm_supports(x, z, weight) {
            return Ok(None);
        }
        let out = metal_gated_rms_norm_bf16(x, z, weight, eps as f32)
            .context("metal gated_rms_norm kernel failed")?;
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

fn metal_gdn_qk_norm_disabled() -> bool {
    env_truthy(DISABLE_METAL_GDN_QK_NORM)
}

fn metal_rms_norm_disabled() -> bool {
    env_present(DISABLE_RMSNORM_KERNEL) || env_truthy(DISABLE_METAL_RMSNORM)
}

fn env_present(var: &str) -> bool {
    std::env::var(var).is_ok()
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

fn metal_conv1d_update_supports(
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

fn metal_gdn_gates_supports(a: &Tensor, b: &Tensor, a_log: &Tensor, dt_bias: &Tensor) -> bool {
    if !matches!(a.device(), Device::Metal(_))
        || !matches!(b.device(), Device::Metal(_))
        || !matches!(a_log.device(), Device::Metal(_))
        || !matches!(dt_bias.device(), Device::Metal(_))
    {
        return false;
    }
    if a.dtype() != DType::BF16
        || b.dtype() != DType::BF16
        || a_log.dtype() != DType::BF16
        || dt_bias.dtype() != DType::BF16
    {
        return false;
    }
    if a.shape() != b.shape() {
        return false;
    }
    let Some(&nv) = a.dims().last() else {
        return false;
    };
    if nv == 0 || nv > 256 {
        return false;
    }
    if a_log.dims() != [nv] || dt_bias.dims() != [nv] {
        return false;
    }
    a.elem_count() > 0
}

fn metal_gdn_forward_substitution_supports(
    a_strict: &Tensor,
    v_prime: &Tensor,
    beta: &Tensor,
) -> bool {
    if !matches!(a_strict.device(), Device::Metal(_))
        || !matches!(v_prime.device(), Device::Metal(_))
        || !matches!(beta.device(), Device::Metal(_))
    {
        return false;
    }
    if a_strict.dtype() != DType::BF16
        || v_prime.dtype() != DType::BF16
        || beta.dtype() != DType::BF16
    {
        return false;
    }
    let Ok((batch, heads, chunk, chunk_cols)) = a_strict.dims4() else {
        return false;
    };
    let Ok((b_v, h_v, c_v, dv)) = v_prime.dims4() else {
        return false;
    };
    let Ok((b_b, h_b, c_b)) = beta.dims3() else {
        return false;
    };

    chunk == chunk_cols
        && (b_v, h_v, c_v) == (batch, heads, chunk)
        && (b_b, h_b, c_b) == (batch, heads, chunk)
        && chunk > 0
        && chunk <= 64
        && dv > 0
        && dv <= 128
}

fn metal_gdn_chunk_prep_supports(
    g: &Tensor,
    v: &Tensor,
    kkt: &Tensor,
    qkt: &Tensor,
    ks_entry: &Tensor,
    q_s: &Tensor,
) -> bool {
    if !matches!(g.device(), Device::Metal(_))
        || !matches!(v.device(), Device::Metal(_))
        || !matches!(kkt.device(), Device::Metal(_))
        || !matches!(qkt.device(), Device::Metal(_))
        || !matches!(ks_entry.device(), Device::Metal(_))
        || !matches!(q_s.device(), Device::Metal(_))
    {
        return false;
    }
    if g.dtype() != DType::BF16
        || v.dtype() != DType::BF16
        || kkt.dtype() != DType::BF16
        || qkt.dtype() != DType::BF16
        || ks_entry.dtype() != DType::BF16
        || q_s.dtype() != DType::BF16
    {
        return false;
    }
    let Ok((batch, heads, chunk)) = g.dims3() else {
        return false;
    };
    // Full chunks only: this keeps speculative multi-token verification on
    // the already-stable Candle path while accelerating long prompt prefill.
    if chunk != 64 {
        return false;
    }
    let Ok((b_v, h_v, c_v, dv)) = v.dims4() else {
        return false;
    };
    if (b_v, h_v, c_v) != (batch, heads, chunk) || dv == 0 || dv > 128 {
        return false;
    }
    let Ok((b_kkt, h_kkt, c_kkt, c2_kkt)) = kkt.dims4() else {
        return false;
    };
    if (b_kkt, h_kkt, c_kkt, c2_kkt) != (batch, heads, chunk, chunk) {
        return false;
    }
    let Ok((b_qkt, h_qkt, c_qkt, c2_qkt)) = qkt.dims4() else {
        return false;
    };
    if (b_qkt, h_qkt, c_qkt, c2_qkt) != (batch, heads, chunk, chunk) {
        return false;
    }
    let Ok((b_ks, h_ks, c_ks, dv_ks)) = ks_entry.dims4() else {
        return false;
    };
    if (b_ks, h_ks, c_ks, dv_ks) != (batch, heads, chunk, dv) {
        return false;
    }
    let Ok((b_qs, h_qs, c_qs, dv_qs)) = q_s.dims4() else {
        return false;
    };
    (b_qs, h_qs, c_qs, dv_qs) == (batch, heads, chunk, dv)
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

fn metal_gated_rms_norm_supports(x: &Tensor, z: &Tensor, weight: &Tensor) -> bool {
    if !matches!(x.device(), Device::Metal(_))
        || !matches!(z.device(), Device::Metal(_))
        || !matches!(weight.device(), Device::Metal(_))
    {
        return false;
    }
    if x.dtype() != DType::BF16 || z.dtype() != DType::BF16 || weight.dtype() != DType::BF16 {
        return false;
    }
    let Ok((batch, seq_len, heads, hidden)) = x.dims4() else {
        return false;
    };
    let Ok((z_batch, z_seq_len, z_heads, z_hidden)) = z.dims4() else {
        return false;
    };
    if (z_batch, z_seq_len, z_heads, z_hidden) != (batch, seq_len, heads, hidden) {
        return false;
    }
    weight.dims() == &[hidden] && hidden <= 1024
}

pub(crate) fn metal_rms_norm_supports(x: &Tensor, weight: &Tensor) -> bool {
    if metal_rms_norm_disabled() {
        return false;
    }
    if !matches!(x.device(), Device::Metal(_)) || !matches!(weight.device(), Device::Metal(_)) {
        return false;
    }
    if x.dtype() != DType::BF16 || weight.dtype() != DType::BF16 {
        return false;
    }
    let Some(hidden) = x.dims().last().copied() else {
        return false;
    };
    x.rank() >= 1 && weight.dims() == &[hidden] && hidden <= 8192
}

pub(crate) fn metal_gdn_qk_norm_supports(q: &Tensor, k: &Tensor) -> bool {
    if metal_gdn_qk_norm_disabled() {
        return false;
    }
    if !matches!(q.device(), Device::Metal(_)) || !matches!(k.device(), Device::Metal(_)) {
        return false;
    }
    if q.dtype() != DType::F32 || k.dtype() != DType::F32 {
        return false;
    }
    let Some(hidden) = q.dims().last().copied() else {
        return false;
    };
    q.rank() >= 1 && q.dims() == k.dims() && hidden <= 8192
}

const METAL_RMSNORM_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_rmsnorm_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    constant uint& threadgroup_width [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float scratch[1024];

    const uint row = gid.y;
    if (row >= rows) {
        return;
    }

    const uint base = row * hidden;
    float sum_sq = 0.0f;
    for (uint col = tid; col < hidden; col += threadgroup_width) {
        const float xv = static_cast<float>(x[base + col]);
        sum_sq += xv * xv;
    }
    scratch[tid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threadgroup_width / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms_inv = rsqrt((scratch[0] / static_cast<float>(hidden)) + eps);
    for (uint col = tid; col < hidden; col += threadgroup_width) {
        const float xv = static_cast<float>(x[base + col]);
        const float scale = 1.0f + static_cast<float>(weight[col]);
        out[base + col] = static_cast<bfloat>(xv * rms_inv * scale);
    }
}
"#;

const METAL_GDN_QK_NORM_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gdn_qk_norm_f32_bf16(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device bfloat* q_out [[buffer(2)]],
    device bfloat* k_out [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant float& q_scale [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant uint& threadgroup_width [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float q_scratch[1024];
    threadgroup float k_scratch[1024];

    const uint row = gid.y;
    if (row >= rows) {
        return;
    }

    const uint base = row * hidden;
    float q_sum_sq = 0.0f;
    float k_sum_sq = 0.0f;
    for (uint col = tid; col < hidden; col += threadgroup_width) {
        const float qv = q[base + col];
        const float kv = k[base + col];
        q_sum_sq += qv * qv;
        k_sum_sq += kv * kv;
    }
    q_scratch[tid] = q_sum_sq;
    k_scratch[tid] = k_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threadgroup_width / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            q_scratch[tid] += q_scratch[tid + stride];
            k_scratch[tid] += k_scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float q_inv = rsqrt(q_scratch[0] + eps);
    const float k_inv = rsqrt(k_scratch[0] + eps);
    for (uint col = tid; col < hidden; col += threadgroup_width) {
        const uint idx = base + col;
        q_out[idx] = static_cast<bfloat>(q[idx] * q_inv * q_scale);
        k_out[idx] = static_cast<bfloat>(k[idx] * k_inv);
    }
}
"#;

const METAL_LM_HEAD_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_lm_head_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight_t [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& hidden [[buffer(3)]],
    constant uint& vocab [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= vocab) {
        return;
    }

    float acc = 0.0f;
    for (uint i = 0; i < hidden; ++i) {
        acc += static_cast<float>(x[i]) * static_cast<float>(weight_t[i * vocab + gid]);
    }
    out[gid] = static_cast<bfloat>(acc);
}
"#;

fn metal_shared_library(
    device: &candle_core::metal_backend::MetalDevice,
) -> Result<candle_metal_kernels::metal::Library> {
    use candle_core::metal_backend::DeviceId;
    use candle_metal_kernels::metal::Library;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    static LIBRARIES: OnceLock<Mutex<HashMap<DeviceId, Library>>> = OnceLock::new();
    let cache = LIBRARIES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal shared library cache poisoned"))?;
    if let Some(library) = cache.get(&device.id()) {
        return Ok(library.clone());
    }

    let shared_source = [
        METAL_RMSNORM_KERNEL,
        METAL_GDN_QK_NORM_KERNEL,
        METAL_GDN_GATES_KERNEL,
        METAL_GATED_RMSNORM_KERNEL,
        METAL_GDN_RECURRENT_KERNEL,
        METAL_CONV1D_PREFILL_KERNEL,
        METAL_CONV1D_UPDATE_KERNEL,
        METAL_LM_HEAD_KERNEL,
    ]
    .join("");
    let library = device
        .device()
        .new_library_with_source(&shared_source, None)
        .map_err(|e| anyhow::anyhow!("compile metal shared library: {e:?}"))?;
    cache.insert(device.id(), library.clone());
    Ok(library)
}

fn metal_rms_norm_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal rmsnorm pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_rmsnorm_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal rmsnorm function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal rmsnorm pipeline: {e:?}"))?;
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gdn_qk_norm_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal gdn qk norm pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_qk_norm_f32_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn qk norm function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn qk norm pipeline: {e:?}"))?;
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

fn metal_lm_head_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal lm head pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_lm_head_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal lm head function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal lm head pipeline: {e:?}"))?;
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

pub(crate) fn metal_lm_head_supports(x: &Tensor, weight_t: &Tensor) -> bool {
    if !matches!(x.dtype(), DType::BF16) || !matches!(weight_t.dtype(), DType::BF16) {
        return false;
    }
    if !matches!(x.device(), Device::Metal(_)) || !matches!(weight_t.device(), Device::Metal(_)) {
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

pub(crate) fn metal_lm_head_bf16(x: &Tensor, weight_t: &Tensor) -> Result<Tensor> {
    anyhow::ensure!(
        metal_lm_head_supports(x, weight_t),
        "metal lm head supports only BF16 [1,1,H] x [H,V] on Metal"
    );
    let (_, _, hidden) = x.dims3()?;
    let (_, vocab) = weight_t.dims2()?;

    let out = Tensor::zeros((1usize, 1usize, vocab), DType::BF16, x.device())?;

    let Device::Metal(device) = x.device() else {
        anyhow::bail!("metal lm head requires Metal tensors");
    };
    let pipeline = metal_lm_head_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_lm_head_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (w_storage, w_layout) = weight_t.storage_and_layout();
        let (o_storage, o_layout) = out.storage_and_layout();

        let x_metal = match &*x_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal lm head x must be on Metal"),
        };
        let w_metal = match &*w_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal lm head weight must be on Metal"),
        };
        let out_metal = match &*o_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal lm head out must be on Metal"),
        };

        let x_buf = candle_core::metal_backend::buffer_o(x_metal.buffer(), &x_layout, x.dtype());
        let w_buf =
            candle_core::metal_backend::buffer_o(w_metal.buffer(), &w_layout, weight_t.dtype());
        let out_buf =
            candle_core::metal_backend::buffer_o(out_metal.buffer(), &o_layout, out.dtype());

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

pub(crate) fn metal_rms_norm_bf16(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
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
    let out = Tensor::zeros(x_dims.as_slice(), DType::BF16, x.device())?;

    if rows == 0 {
        return Ok(out);
    }

    let Device::Metal(device) = x.device() else {
        anyhow::bail!("metal rmsnorm requires a Metal tensor");
    };
    let pipeline = metal_rms_norm_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_rmsnorm_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (w_storage, w_layout) = weight.storage_and_layout();
        let (o_storage, o_layout) = out.storage_and_layout();

        let x_metal = match &*x_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal rmsnorm x must be on Metal"),
        };
        let w_metal = match &*w_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal rmsnorm weight must be on Metal"),
        };
        let out_metal = match &*o_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal rmsnorm out must be on Metal"),
        };

        let x_buf = candle_core::metal_backend::buffer_o(x_metal.buffer(), &x_layout, x.dtype());
        let w_buf =
            candle_core::metal_backend::buffer_o(w_metal.buffer(), &w_layout, weight.dtype());
        let out_buf =
            candle_core::metal_backend::buffer_o(out_metal.buffer(), &o_layout, out.dtype());

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

pub(crate) fn metal_gdn_qk_norm_f32_bf16(
    q: &Tensor,
    k: &Tensor,
    q_scale: f32,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    let dims = q.dims().to_vec();
    let hidden = *dims
        .last()
        .context("metal gdn qk norm requires rank >= 1 input")?;
    anyhow::ensure!(q.dims() == k.dims(), "metal gdn qk norm shape mismatch");
    anyhow::ensure!(hidden <= 8192, "metal gdn qk norm hidden dim > 8192");
    let rows: usize = dims[..dims.len() - 1].iter().product();
    anyhow::ensure!(
        rows <= u32::MAX as usize && hidden <= u32::MAX as usize,
        "metal gdn qk norm shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let q_out = Tensor::zeros(dims.as_slice(), DType::BF16, q.device())?;
    let k_out = Tensor::zeros(dims.as_slice(), DType::BF16, q.device())?;

    if rows == 0 {
        return Ok((q_out, k_out));
    }

    let Device::Metal(device) = q.device() else {
        anyhow::bail!("metal gdn qk norm requires Metal tensors");
    };
    let pipeline = metal_gdn_qk_norm_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_gdn_qk_norm_f32_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (q_storage, q_layout) = q.storage_and_layout();
        let (k_storage, k_layout) = k.storage_and_layout();
        let (qo_storage, qo_layout) = q_out.storage_and_layout();
        let (ko_storage, ko_layout) = k_out.storage_and_layout();

        let q_metal = match &*q_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn qk norm q must be on Metal"),
        };
        let k_metal = match &*k_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn qk norm k must be on Metal"),
        };
        let qo_metal = match &*qo_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn qk norm q_out must be on Metal"),
        };
        let ko_metal = match &*ko_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn qk norm k_out must be on Metal"),
        };

        let q_buf = candle_core::metal_backend::buffer_o(q_metal.buffer(), &q_layout, q.dtype());
        let k_buf = candle_core::metal_backend::buffer_o(k_metal.buffer(), &k_layout, k.dtype());
        let qo_buf =
            candle_core::metal_backend::buffer_o(qo_metal.buffer(), &qo_layout, q_out.dtype());
        let ko_buf =
            candle_core::metal_backend::buffer_o(ko_metal.buffer(), &ko_layout, k_out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(qo_buf.buffer), qo_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(ko_buf.buffer), ko_buf.offset_in_bytes);

        let rows_u32 = rows as u32;
        let hidden_u32 = hidden as u32;
        let threads = hidden.next_power_of_two().clamp(32, 1024);
        let threads_u32 = threads as u32;
        encoder.set_bytes(4, &rows_u32);
        encoder.set_bytes(5, &hidden_u32);
        encoder.set_bytes(6, &q_scale);
        encoder.set_bytes(7, &eps);
        encoder.set_bytes(8, &threads_u32);

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

    Ok((q_out, k_out))
}

const METAL_GDN_GATES_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float kiln_stable_sigmoid(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + exp(-x));
    }
    const float e = exp(x);
    return e / (1.0f + e);
}

inline float kiln_stable_softplus(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return exp(x);
    }
    return log(1.0f + exp(x));
}

kernel void kiln_gdn_gates_bf16(
    device const bfloat* a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device const bfloat* a_log [[buffer(2)]],
    device const bfloat* dt_bias [[buffer(3)]],
    device bfloat* beta_out [[buffer(4)]],
    device bfloat* g_out [[buffer(5)]],
    constant uint& nv [[buffer(6)]],
    constant uint& total [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) {
        return;
    }

    const uint h = gid % nv;
    const float a_val = static_cast<float>(a[gid]);
    const float b_val = static_cast<float>(b[gid]);
    const float a_log_val = static_cast<float>(a_log[h]);
    const float dt_bias_val = static_cast<float>(dt_bias[h]);

    const float beta = kiln_stable_sigmoid(b_val);
    const float sp = kiln_stable_softplus(a_val + dt_bias_val);
    const float g = sp * -exp(a_log_val);

    beta_out[gid] = static_cast<bfloat>(beta);
    g_out[gid] = static_cast<bfloat>(g);
}
"#;

fn metal_gdn_gates_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal gdn_gates pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_gates_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn_gates function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn_gates pipeline: {e:?}"))?;
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gdn_gates_bf16(
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let shape = a.dims().to_vec();
    let nv = *shape
        .last()
        .ok_or_else(|| anyhow::anyhow!("metal gdn_gates requires at least rank-1 input"))?;
    let total = a.elem_count();
    anyhow::ensure!(
        total <= u32::MAX as usize,
        "metal gdn_gates input too large"
    );
    anyhow::ensure!(nv <= u32::MAX as usize, "metal gdn_gates nv too large");

    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let a_log = a_log.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let beta = Tensor::zeros(shape.clone(), DType::BF16, a.device())?;
    let g = Tensor::zeros(shape, DType::BF16, a.device())?;

    let Device::Metal(device) = a.device() else {
        anyhow::bail!("metal gdn_gates requires a Metal tensor");
    };
    let pipeline = metal_gdn_gates_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_gdn_gates_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (a_storage, a_layout) = a.storage_and_layout();
        let (b_storage, b_layout) = b.storage_and_layout();
        let (al_storage, al_layout) = a_log.storage_and_layout();
        let (dt_storage, dt_layout) = dt_bias.storage_and_layout();
        let (beta_storage, beta_layout) = beta.storage_and_layout();
        let (g_storage, g_layout) = g.storage_and_layout();

        let a_metal = match &*a_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn_gates a must be on Metal"),
        };
        let b_metal = match &*b_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn_gates b must be on Metal"),
        };
        let al_metal = match &*al_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn_gates a_log must be on Metal"),
        };
        let dt_metal = match &*dt_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn_gates dt_bias must be on Metal"),
        };
        let beta_metal = match &*beta_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn_gates beta output must be on Metal"),
        };
        let g_metal = match &*g_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn_gates g output must be on Metal"),
        };

        let a_buf = candle_core::metal_backend::buffer_o(a_metal.buffer(), &a_layout, a.dtype());
        let b_buf = candle_core::metal_backend::buffer_o(b_metal.buffer(), &b_layout, b.dtype());
        let al_buf =
            candle_core::metal_backend::buffer_o(al_metal.buffer(), &al_layout, a_log.dtype());
        let dt_buf =
            candle_core::metal_backend::buffer_o(dt_metal.buffer(), &dt_layout, dt_bias.dtype());
        let beta_buf =
            candle_core::metal_backend::buffer_o(beta_metal.buffer(), &beta_layout, beta.dtype());
        let g_buf = candle_core::metal_backend::buffer_o(g_metal.buffer(), &g_layout, g.dtype());

        encoder.set_buffer(0, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(b_buf.buffer), b_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(al_buf.buffer), al_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(dt_buf.buffer), dt_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(g_buf.buffer), g_buf.offset_in_bytes);

        let nv_u32 = nv as u32;
        let total_u32 = total as u32;
        encoder.set_bytes(6, &nv_u32);
        encoder.set_bytes(7, &total_u32);

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

    Ok((beta, g))
}

const METAL_GATED_RMSNORM_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gated_rmsnorm_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* z [[buffer(1)]],
    device const bfloat* weight [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    constant uint& threadgroup_width [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float scratch[1024];

    const uint row = gid.y;
    if (row >= rows) {
        return;
    }

    const uint base = row * hidden;
    float sum_sq = 0.0f;
    if (tid < hidden) {
        const float xv = static_cast<float>(x[base + tid]);
        sum_sq = xv * xv;
    }
    scratch[tid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threadgroup_width / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < hidden) {
        const float rms_inv = rsqrt((scratch[0] / static_cast<float>(hidden)) + eps);
        const float xv = static_cast<float>(x[base + tid]);
        const float zv = static_cast<float>(z[base + tid]);
        const float gate = zv / (1.0f + exp(-zv));
        out[base + tid] = static_cast<bfloat>(xv * rms_inv * static_cast<float>(weight[tid]) * gate);
    }
}
"#;

fn metal_gated_rms_norm_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal gated rmsnorm pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gated_rmsnorm_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gated rmsnorm function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gated rmsnorm pipeline: {e:?}"))?;
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gated_rms_norm_bf16(x: &Tensor, z: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let (batch, seq_len, heads, hidden) = x.dims4()?;
    let rows = batch
        .checked_mul(seq_len)
        .and_then(|v| v.checked_mul(heads))
        .context("metal gated rmsnorm row count overflow")?;
    anyhow::ensure!(
        rows <= u32::MAX as usize && hidden <= u32::MAX as usize,
        "metal gated rmsnorm shape too large"
    );
    anyhow::ensure!(hidden <= 1024, "metal gated rmsnorm hidden dim > 1024");

    let x = x.contiguous()?;
    let z = z.contiguous()?;
    let weight = weight.contiguous()?;
    let out = Tensor::zeros((batch, seq_len, heads, hidden), DType::BF16, x.device())?;

    if rows == 0 {
        return Ok(out);
    }

    let Device::Metal(device) = x.device() else {
        anyhow::bail!("metal gated rmsnorm requires a Metal tensor");
    };
    let pipeline = metal_gated_rms_norm_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_gated_rmsnorm_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (z_storage, z_layout) = z.storage_and_layout();
        let (w_storage, w_layout) = weight.storage_and_layout();
        let (o_storage, o_layout) = out.storage_and_layout();

        let x_metal = match &*x_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gated rmsnorm x must be on Metal"),
        };
        let z_metal = match &*z_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gated rmsnorm z must be on Metal"),
        };
        let w_metal = match &*w_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gated rmsnorm weight must be on Metal"),
        };
        let out_metal = match &*o_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gated rmsnorm out must be on Metal"),
        };

        let x_buf = candle_core::metal_backend::buffer_o(x_metal.buffer(), &x_layout, x.dtype());
        let z_buf = candle_core::metal_backend::buffer_o(z_metal.buffer(), &z_layout, z.dtype());
        let w_buf =
            candle_core::metal_backend::buffer_o(w_metal.buffer(), &w_layout, weight.dtype());
        let out_buf =
            candle_core::metal_backend::buffer_o(out_metal.buffer(), &o_layout, out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(z_buf.buffer), z_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let rows_u32 = rows as u32;
        let hidden_u32 = hidden as u32;
        let threads = hidden.next_power_of_two().clamp(32, 1024);
        let threads_u32 = threads as u32;
        encoder.set_bytes(4, &rows_u32);
        encoder.set_bytes(5, &hidden_u32);
        encoder.set_bytes(6, &eps);
        encoder.set_bytes(7, &threads_u32);

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

kernel void kiln_gdn_forward_substitution_bf16(
    device const bfloat* a_strict [[buffer(0)]],
    device const bfloat* v_prime [[buffer(1)]],
    device const bfloat* beta [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant uint& batch_heads [[buffer(4)]],
    constant uint& chunk_size [[buffer(5)]],
    constant uint& dv [[buffer(6)]],
    uint bh [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (bh >= batch_heads) {
        return;
    }

    // Conservative Qwen3.5 envelope: C <= 64, dv <= 128. Static threadgroup
    // storage keeps the kernel simple and under Apple Silicon's common 32 KiB
    // per-threadgroup memory budget: (64*64 + 64*128) bf16 = 24 KiB.
    threadgroup bfloat sA[4096];
    threadgroup bfloat sW[8192];

    const uint a_base = bh * chunk_size * chunk_size;
    const uint v_base = bh * chunk_size * dv;
    const uint beta_base = bh * chunk_size;
    const uint total_a = chunk_size * chunk_size;

    for (uint i = tid; i < total_a; i += 128) {
        sA[i] = a_strict[a_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 0; t < chunk_size; ++t) {
        const float beta_t = static_cast<float>(beta[beta_base + t]);

        for (uint d = tid; d < dv; d += 128) {
            float acc = 0.0f;
            for (uint i = 0; i < t; ++i) {
                const float a = static_cast<float>(sA[t * chunk_size + i]);
                const float w = static_cast<float>(sW[i * dv + d]);
                acc += a * w;
            }

            const uint row_col = t * dv + d;
            const float vp = static_cast<float>(v_prime[v_base + row_col]);
            const bfloat w = static_cast<bfloat>(beta_t * (vp - acc));
            sW[row_col] = w;
            out[v_base + row_col] = w;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void kiln_gdn_chunk_prep_bf16(
    device const bfloat* g [[buffer(0)]],
    device const bfloat* v [[buffer(1)]],
    device const bfloat* kkt [[buffer(2)]],
    device const bfloat* qkt [[buffer(3)]],
    device const bfloat* ks_entry [[buffer(4)]],
    device const bfloat* q_s [[buffer(5)]],
    device bfloat* a_strict [[buffer(6)]],
    device bfloat* b_mask [[buffer(7)]],
    device bfloat* v_prime [[buffer(8)]],
    device bfloat* q_s_scaled [[buffer(9)]],
    device bfloat* decay_last_col [[buffer(10)]],
    device bfloat* p_last [[buffer(11)]],
    constant uint& batch_heads [[buffer(12)]],
    constant uint& chunk_size [[buffer(13)]],
    constant uint& dv [[buffer(14)]],
    uint bh [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (bh >= batch_heads) {
        return;
    }

    threadgroup float sBigG[64];
    threadgroup bfloat sP[64];

    const uint g_base = bh * chunk_size;
    const uint cdv_base = bh * chunk_size * dv;
    const uint cc_base = bh * chunk_size * chunk_size;

    if (tid < chunk_size) {
        sBigG[tid] = static_cast<float>(g[g_base + tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float acc = 0.0f;
        for (uint i = 0; i < 64; ++i) {
            acc += sBigG[i];
            sBigG[i] = acc;
        }
        for (uint i = 0; i < 64; ++i) {
            sP[i] = static_cast<bfloat>(exp(sBigG[i]));
        }
        p_last[bh] = sP[chunk_size - 1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint cc = chunk_size * chunk_size;
    for (uint idx = tid; idx < cc; idx += 128) {
        const uint t = idx / chunk_size;
        const uint i = idx - t * chunk_size;
        const bfloat decay_bf = static_cast<bfloat>(exp(sBigG[t] - sBigG[i]));
        const float decay = static_cast<float>(decay_bf);
        const float kkt_val = static_cast<float>(kkt[cc_base + idx]);
        const float qkt_val = static_cast<float>(qkt[cc_base + idx]);
        a_strict[cc_base + idx] =
            (i < t) ? static_cast<bfloat>(kkt_val * decay) : static_cast<bfloat>(0.0f);
        b_mask[cc_base + idx] =
            (i <= t) ? static_cast<bfloat>(qkt_val * decay) : static_cast<bfloat>(0.0f);
    }

    const uint cdv = chunk_size * dv;
    for (uint idx = tid; idx < cdv; idx += 128) {
        const uint t = idx / dv;
        const float p = static_cast<float>(sP[t]);
        const float v_val = static_cast<float>(v[cdv_base + idx]);
        const float ks_val = static_cast<float>(ks_entry[cdv_base + idx]);
        const float qs_val = static_cast<float>(q_s[cdv_base + idx]);
        v_prime[cdv_base + idx] = static_cast<bfloat>(v_val - ks_val * p);
        q_s_scaled[cdv_base + idx] = static_cast<bfloat>(qs_val * p);
    }

    if (tid < chunk_size) {
        const float decay = exp(sBigG[chunk_size - 1] - sBigG[tid]);
        decay_last_col[g_base + tid] = static_cast<bfloat>(decay);
    }
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

    let library = metal_shared_library(device)?;
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

fn metal_gdn_forward_substitution_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal gdn forward-substitution pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_forward_substitution_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn forward-substitution function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn forward-substitution pipeline: {e:?}"))?;
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gdn_chunk_prep_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal gdn chunk-prep pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_chunk_prep_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn chunk-prep function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn chunk-prep pipeline: {e:?}"))?;
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gdn_forward_substitution_bf16(
    a_strict: &Tensor,
    v_prime: &Tensor,
    beta: &Tensor,
) -> Result<Tensor> {
    let (batch, heads, chunk_size, _) = a_strict.dims4()?;
    let dv = v_prime.dim(3)?;
    let batch_heads = batch * heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && chunk_size <= u32::MAX as usize
            && dv <= u32::MAX as usize,
        "metal gdn forward-substitution shape too large"
    );

    let a_strict = a_strict.contiguous()?;
    let v_prime = v_prime.contiguous()?;
    let beta = beta.contiguous()?;
    let out = Tensor::zeros(
        (batch, heads, chunk_size, dv),
        DType::BF16,
        a_strict.device(),
    )?;

    let Device::Metal(device) = a_strict.device() else {
        anyhow::bail!("metal gdn forward-substitution requires a Metal tensor");
    };
    let pipeline = metal_gdn_forward_substitution_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_gdn_forward_substitution_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (a_storage, a_layout) = a_strict.storage_and_layout();
        let (v_storage, v_layout) = v_prime.storage_and_layout();
        let (beta_storage, beta_layout) = beta.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let a_metal = match &*a_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn forward-substitution a_strict must be on Metal"),
        };
        let v_metal = match &*v_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn forward-substitution v_prime must be on Metal"),
        };
        let beta_metal = match &*beta_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn forward-substitution beta must be on Metal"),
        };
        let out_metal = match &*out_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn forward-substitution out must be on Metal"),
        };

        let a_buf =
            candle_core::metal_backend::buffer_o(a_metal.buffer(), &a_layout, a_strict.dtype());
        let v_buf =
            candle_core::metal_backend::buffer_o(v_metal.buffer(), &v_layout, v_prime.dtype());
        let beta_buf =
            candle_core::metal_backend::buffer_o(beta_metal.buffer(), &beta_layout, beta.dtype());
        let out_buf =
            candle_core::metal_backend::buffer_o(out_metal.buffer(), &out_layout, out.dtype());

        encoder.set_buffer(0, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let chunk_size_u32 = chunk_size as u32;
        let dv_u32 = dv as u32;
        encoder.set_bytes(4, &batch_heads_u32);
        encoder.set_bytes(5, &chunk_size_u32);
        encoder.set_bytes(6, &dv_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: batch_heads,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threads_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

fn metal_gdn_chunk_prep_bf16(
    g: &Tensor,
    v: &Tensor,
    kkt: &Tensor,
    qkt: &Tensor,
    ks_entry: &Tensor,
    q_s: &Tensor,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let (batch, heads, chunk_size) = g.dims3()?;
    let dv = v.dim(3)?;
    let batch_heads = batch * heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && chunk_size <= u32::MAX as usize
            && dv <= u32::MAX as usize,
        "metal gdn chunk-prep shape too large"
    );

    let g = g.contiguous()?;
    let v = v.contiguous()?;
    let kkt = kkt.contiguous()?;
    let qkt = qkt.contiguous()?;
    let ks_entry = ks_entry.contiguous()?;
    let q_s = q_s.contiguous()?;
    let device_ref = g.device();
    let a_strict = Tensor::zeros((batch, heads, chunk_size, chunk_size), DType::BF16, device_ref)?;
    let b_mask = Tensor::zeros((batch, heads, chunk_size, chunk_size), DType::BF16, device_ref)?;
    let v_prime = Tensor::zeros((batch, heads, chunk_size, dv), DType::BF16, device_ref)?;
    let q_s_scaled = Tensor::zeros((batch, heads, chunk_size, dv), DType::BF16, device_ref)?;
    let decay_last_col = Tensor::zeros((batch, heads, chunk_size), DType::BF16, device_ref)?;
    let p_last = Tensor::zeros((batch, heads), DType::BF16, device_ref)?;

    let Device::Metal(device) = g.device() else {
        anyhow::bail!("metal gdn chunk-prep requires a Metal tensor");
    };
    let pipeline = metal_gdn_chunk_prep_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_gdn_chunk_prep_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (g_storage, g_layout) = g.storage_and_layout();
        let (v_storage, v_layout) = v.storage_and_layout();
        let (kkt_storage, kkt_layout) = kkt.storage_and_layout();
        let (qkt_storage, qkt_layout) = qkt.storage_and_layout();
        let (ks_storage, ks_layout) = ks_entry.storage_and_layout();
        let (qs_storage, qs_layout) = q_s.storage_and_layout();
        let (a_storage, a_layout) = a_strict.storage_and_layout();
        let (b_storage, b_layout) = b_mask.storage_and_layout();
        let (vp_storage, vp_layout) = v_prime.storage_and_layout();
        let (qss_storage, qss_layout) = q_s_scaled.storage_and_layout();
        let (dl_storage, dl_layout) = decay_last_col.storage_and_layout();
        let (pl_storage, pl_layout) = p_last.storage_and_layout();

        let g_metal = match &*g_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn chunk-prep g must be on Metal"),
        };
        let v_metal = match &*v_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn chunk-prep v must be on Metal"),
        };
        let kkt_metal = match &*kkt_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn chunk-prep kkt must be on Metal"),
        };
        let qkt_metal = match &*qkt_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn chunk-prep qkt must be on Metal"),
        };
        let ks_metal = match &*ks_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn chunk-prep ks_entry must be on Metal"),
        };
        let qs_metal = match &*qs_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn chunk-prep q_s must be on Metal"),
        };
        let a_metal = match &*a_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn chunk-prep a_strict must be on Metal"),
        };
        let b_metal = match &*b_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn chunk-prep b_mask must be on Metal"),
        };
        let vp_metal = match &*vp_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn chunk-prep v_prime must be on Metal"),
        };
        let qss_metal = match &*qss_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn chunk-prep q_s_scaled must be on Metal"),
        };
        let dl_metal = match &*dl_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn chunk-prep decay_last_col must be on Metal"),
        };
        let pl_metal = match &*pl_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn chunk-prep p_last must be on Metal"),
        };

        let g_buf = candle_core::metal_backend::buffer_o(g_metal.buffer(), &g_layout, g.dtype());
        let v_buf = candle_core::metal_backend::buffer_o(v_metal.buffer(), &v_layout, v.dtype());
        let kkt_buf =
            candle_core::metal_backend::buffer_o(kkt_metal.buffer(), &kkt_layout, kkt.dtype());
        let qkt_buf =
            candle_core::metal_backend::buffer_o(qkt_metal.buffer(), &qkt_layout, qkt.dtype());
        let ks_buf = candle_core::metal_backend::buffer_o(
            ks_metal.buffer(),
            &ks_layout,
            ks_entry.dtype(),
        );
        let qs_buf =
            candle_core::metal_backend::buffer_o(qs_metal.buffer(), &qs_layout, q_s.dtype());
        let a_buf = candle_core::metal_backend::buffer_o(
            a_metal.buffer(),
            &a_layout,
            a_strict.dtype(),
        );
        let b_buf =
            candle_core::metal_backend::buffer_o(b_metal.buffer(), &b_layout, b_mask.dtype());
        let vp_buf =
            candle_core::metal_backend::buffer_o(vp_metal.buffer(), &vp_layout, v_prime.dtype());
        let qss_buf = candle_core::metal_backend::buffer_o(
            qss_metal.buffer(),
            &qss_layout,
            q_s_scaled.dtype(),
        );
        let dl_buf = candle_core::metal_backend::buffer_o(
            dl_metal.buffer(),
            &dl_layout,
            decay_last_col.dtype(),
        );
        let pl_buf =
            candle_core::metal_backend::buffer_o(pl_metal.buffer(), &pl_layout, p_last.dtype());

        encoder.set_buffer(0, Some(g_buf.buffer), g_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(kkt_buf.buffer), kkt_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(qkt_buf.buffer), qkt_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(ks_buf.buffer), ks_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(qs_buf.buffer), qs_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(7, Some(b_buf.buffer), b_buf.offset_in_bytes);
        encoder.set_buffer(8, Some(vp_buf.buffer), vp_buf.offset_in_bytes);
        encoder.set_buffer(9, Some(qss_buf.buffer), qss_buf.offset_in_bytes);
        encoder.set_buffer(10, Some(dl_buf.buffer), dl_buf.offset_in_bytes);
        encoder.set_buffer(11, Some(pl_buf.buffer), pl_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let chunk_size_u32 = chunk_size as u32;
        let dv_u32 = dv as u32;
        encoder.set_bytes(12, &batch_heads_u32);
        encoder.set_bytes(13, &chunk_size_u32);
        encoder.set_bytes(14, &dv_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok((a_strict, b_mask, v_prime, q_s_scaled, decay_last_col, p_last))
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
    constant uint& threadgroup_width [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
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

    threadgroup float s_state[3];
    if (tid < 3) {
        s_state[tid] = conv_state[state_base + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = tid; t < seq_len; t += threadgroup_width) {
        float acc = 0.0f;
        for (uint j = 0; j < 4; ++j) {
            const uint padded_idx = t + j;
            float v;
            if (padded_idx < 3) {
                v = s_state[padded_idx];
            } else {
                v = static_cast<float>(x[x_base + padded_idx - 3]);
            }
            acc += v * static_cast<float>(weight[weight_base + j]);
        }
        out[x_base + t] = acc / (1.0f + exp(-acc));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        if (seq_len >= 3) {
            conv_state[state_base + 0] = static_cast<float>(x[x_base + seq_len - 3]);
            conv_state[state_base + 1] = static_cast<float>(x[x_base + seq_len - 2]);
            conv_state[state_base + 2] = static_cast<float>(x[x_base + seq_len - 1]);
        } else if (seq_len == 2) {
            conv_state[state_base + 0] = s_state[2];
            conv_state[state_base + 1] = static_cast<float>(x[x_base + 0]);
            conv_state[state_base + 2] = static_cast<float>(x[x_base + 1]);
        } else if (seq_len == 1) {
            conv_state[state_base + 0] = s_state[1];
            conv_state[state_base + 1] = s_state[2];
            conv_state[state_base + 2] = static_cast<float>(x[x_base]);
        }
    }
}
"#;

const METAL_CONV1D_UPDATE_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_causal_conv1d_update_bf16_f32_k4(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device float* conv_state [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = batch * channels;
    if (gid >= total) {
        return;
    }

    const uint c = gid % channels;
    const uint state_base = gid * 3;
    const uint weight_base = c * 4;

    const float s0 = conv_state[state_base + 0];
    const float s1 = conv_state[state_base + 1];
    const float s2 = conv_state[state_base + 2];
    const float x0 = static_cast<float>(x[gid]);

    const float acc =
        s0 * static_cast<float>(weight[weight_base + 0]) +
        s1 * static_cast<float>(weight[weight_base + 1]) +
        s2 * static_cast<float>(weight[weight_base + 2]) +
        x0 * static_cast<float>(weight[weight_base + 3]);

    out[gid] = acc / (1.0f + exp(-acc));
    conv_state[state_base + 0] = s1;
    conv_state[state_base + 1] = s2;
    conv_state[state_base + 2] = x0;
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

    let library = metal_shared_library(device)?;
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

fn metal_conv1d_update_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal conv1d update pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_causal_conv1d_update_bf16_f32_k4", None)
        .map_err(|e| anyhow::anyhow!("load metal conv1d update function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal conv1d update pipeline: {e:?}"))?;
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

fn metal_causal_conv1d_update_bf16_f32_k4(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &mut Tensor,
    kernel_size: usize,
) -> Result<Tensor> {
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
    let out = Tensor::zeros((batch, channels, 1usize), DType::F32, x.device())?;

    let Device::Metal(device) = x.device() else {
        anyhow::bail!("metal conv1d update requires a Metal tensor");
    };
    let pipeline = metal_conv1d_update_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_causal_conv1d_update_bf16_f32_k4");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (w_storage, w_layout) = weight.storage_and_layout();
        let (s_storage, s_layout) = conv_state.storage_and_layout();
        let (o_storage, o_layout) = out.storage_and_layout();

        let x_metal = match &*x_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal conv1d update x must be on Metal"),
        };
        let w_metal = match &*w_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal conv1d update weight must be on Metal"),
        };
        let s_metal = match &*s_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal conv1d update state must be on Metal"),
        };
        let o_metal = match &*o_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal conv1d update output must be on Metal"),
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

    #[test]
    fn test_precompile_custom_kernels_smoke() -> Result<()> {
        let Some(device) = try_new_metal() else {
            eprintln!("Metal unavailable, skipping test_precompile_custom_kernels_smoke");
            return Ok(());
        };

        precompile_custom_kernels(&device)
    }

    fn gdn_qk_norm_reference(
        q: &Tensor,
        k: &Tensor,
        q_scale: f64,
        eps: f64,
    ) -> Result<(Tensor, Tensor)> {
        let q_sum = q.sqr()?.sum_keepdim(D::Minus1)?;
        let q_norm = (q_sum + eps)?.sqrt()?;
        let q_out = (q.broadcast_div(&q_norm)? * q_scale)?.to_dtype(DType::BF16)?;

        let k_sum = k.sqr()?.sum_keepdim(D::Minus1)?;
        let k_norm = (k_sum + eps)?.sqrt()?;
        let k_out = k.broadcast_div(&k_norm)?.to_dtype(DType::BF16)?;

        Ok((q_out, k_out))
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        Ok((a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .max(D::Minus1)?
            .to_scalar::<f32>()?)
    }

    fn mean_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        Ok((a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .mean(D::Minus1)?
            .to_scalar::<f32>()?)
    }

    fn gdn_chunk_prep_reference(
        g: &Tensor,
        v: &Tensor,
        kkt: &Tensor,
        qkt: &Tensor,
        ks_entry: &Tensor,
        q_s: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let chunk = g.dim(2)?;
        let g_f32 = g.to_dtype(DType::F32)?;
        let big_g = g_f32.cumsum(D::Minus1)?;
        let big_g_col = big_g.unsqueeze(3)?;
        let big_g_row = big_g.unsqueeze(2)?;
        let decay = big_g_col.broadcast_sub(&big_g_row)?.exp()?.to_dtype(DType::BF16)?;
        let p = big_g.exp()?.to_dtype(DType::BF16)?;
        let p_col = p.unsqueeze(3)?;

        let strict_mask = Tensor::from_slice(
            &(0..(chunk * chunk))
                .map(|idx| {
                    let t = idx / chunk;
                    let i = idx % chunk;
                    if i < t { 1.0f32 } else { 0.0f32 }
                })
                .collect::<Vec<_>>(),
            (chunk, chunk),
            g.device(),
        )?
        .to_dtype(DType::BF16)?;
        let causal_mask = Tensor::from_slice(
            &(0..(chunk * chunk))
                .map(|idx| {
                    let t = idx / chunk;
                    let i = idx % chunk;
                    if i <= t { 1.0f32 } else { 0.0f32 }
                })
                .collect::<Vec<_>>(),
            (chunk, chunk),
            g.device(),
        )?
        .to_dtype(DType::BF16)?;

        let v_prime = (v - ks_entry.broadcast_mul(&p_col)?)?;
        let a_strict = kkt
            .broadcast_mul(&decay)?
            .broadcast_mul(&strict_mask)?
            .contiguous()?;
        let b_mask = qkt
            .broadcast_mul(&decay)?
            .broadcast_mul(&causal_mask)?
            .contiguous()?;
        let q_s_scaled = q_s.broadcast_mul(&p_col)?;
        let g_last = big_g.narrow(2, chunk - 1, 1)?;
        let decay_last_col = g_last.broadcast_sub(&big_g)?.exp()?.to_dtype(DType::BF16)?;
        let p_last = g_last.squeeze(2)?.exp()?.to_dtype(DType::BF16)?;

        Ok((
            a_strict,
            b_mask,
            v_prime,
            q_s_scaled,
            decay_last_col,
            p_last,
        ))
    }

    #[test]
    fn test_lm_head_matches_broadcast_matmul() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let hidden = 128usize;
        let vocab = 257usize;
        let x_data: Vec<f32> = (0..hidden)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.03125)
            .collect();
        let weight_data: Vec<f32> = (0..(hidden * vocab))
            .map(|i| ((i % 29) as f32 - 14.0) * 0.015625)
            .collect();

        let x = Tensor::from_slice(&x_data, (1usize, 1usize, hidden), &device)?
            .to_dtype(DType::BF16)?;
        let weight_t =
            Tensor::from_slice(&weight_data, (hidden, vocab), &device)?.to_dtype(DType::BF16)?;

        assert!(metal_lm_head_supports(&x, &weight_t));
        let reference = x.broadcast_matmul(&weight_t)?;
        let fused = metal_lm_head_bf16(&x, &weight_t)?;

        assert_eq!(fused.dims(), &[1usize, 1usize, vocab]);
        assert_eq!(fused.dtype(), DType::BF16);

        let max = max_abs_diff(&reference, &fused)?;
        let mean = mean_abs_diff(&reference, &fused)?;
        assert!(
            max < 2e-2,
            "Metal LM-head max_abs_diff={max:e} exceeds tolerance"
        );
        assert!(
            mean < 2e-3,
            "Metal LM-head mean_abs_diff={mean:e} exceeds tolerance"
        );

        Ok(())
    }

    fn gdn_gates_reference(
        a: &Tensor,
        b: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let beta = (b.neg()?.exp()? + 1.0)?.recip()?.to_dtype(DType::BF16)?;
        let a_biased = a
            .to_dtype(DType::F32)?
            .broadcast_add(&dt_bias.to_dtype(DType::F32)?)?;
        let zeros = Tensor::zeros_like(&a_biased)?;
        let relu_x = a_biased.maximum(&zeros)?;
        let relu_neg_x = a_biased.neg()?.maximum(&zeros)?;
        let neg_abs = (relu_x.clone() + relu_neg_x)?.neg()?;
        let log_term = (neg_abs.exp()? + 1.0)?.log()?;
        let sp = (relu_x + log_term)?;
        let neg_decay = a_log.to_dtype(DType::F32)?.exp()?.neg()?;
        let g = sp.broadcast_mul(&neg_decay)?.to_dtype(DType::BF16)?;
        Ok((beta, g))
    }

    fn assert_gdn_gates_matches_reference(
        batch: usize,
        seq_len: usize,
        nv: usize,
        device: &Device,
    ) -> Result<()> {
        let total = batch * seq_len * nv;
        let a_data: Vec<f32> = (0..total)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.25)
            .collect();
        let b_data: Vec<f32> = (0..total)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.2)
            .collect();
        let a_log_data: Vec<f32> = (0..nv)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.075)
            .collect();
        let dt_bias_data: Vec<f32> = (0..nv)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.08)
            .collect();

        let a =
            Tensor::from_slice(&a_data, (batch, seq_len, nv), device)?.to_dtype(DType::BF16)?;
        let b =
            Tensor::from_slice(&b_data, (batch, seq_len, nv), device)?.to_dtype(DType::BF16)?;
        let a_log = Tensor::from_slice(&a_log_data, nv, device)?.to_dtype(DType::BF16)?;
        let dt_bias = Tensor::from_slice(&dt_bias_data, nv, device)?.to_dtype(DType::BF16)?;

        assert!(metal_gdn_gates_supports(&a, &b, &a_log, &dt_bias));
        let (beta_ref, g_ref) = gdn_gates_reference(&a, &b, &a_log, &dt_bias)?;
        let (beta_fused, g_fused) = metal_gdn_gates_bf16(&a, &b, &a_log, &dt_bias)?;

        assert_eq!(beta_fused.dims(), &[batch, seq_len, nv]);
        assert_eq!(g_fused.dims(), &[batch, seq_len, nv]);
        assert_eq!(beta_fused.dtype(), DType::BF16);
        assert_eq!(g_fused.dtype(), DType::BF16);

        let beta_max = max_abs_diff(&beta_ref, &beta_fused)?;
        let beta_mean = mean_abs_diff(&beta_ref, &beta_fused)?;
        let g_max = max_abs_diff(&g_ref, &g_fused)?;
        let g_mean = mean_abs_diff(&g_ref, &g_fused)?;
        assert!(
            beta_max < 2e-2,
            "GDN beta parity failed: max_abs_diff={beta_max}"
        );
        assert!(
            beta_mean < 5e-3,
            "GDN beta parity failed: mean_abs_diff={beta_mean}"
        );
        assert!(
            g_max < 2e-2,
            "GDN g parity failed: max_abs_diff={g_max}"
        );
        assert!(
            g_mean < 5e-3,
            "GDN g parity failed: mean_abs_diff={g_mean}"
        );

        Ok(())
    }

    #[test]
    fn test_gdn_gates_matches_fallback_decode_shape() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        assert_gdn_gates_matches_reference(1, 1, 32, &device)
    }

    #[test]
    fn test_gdn_gates_matches_fallback_prefill_shape() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        assert_gdn_gates_matches_reference(1, 64, 32, &device)
    }

    #[test]
    fn test_gdn_chunk_prep_matches_fallback_prefill_shape() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let batch = 1usize;
        let heads = 32usize;
        let chunk = 64usize;
        let dv = 128usize;
        let c_elems = batch * heads * chunk;
        let cc_elems = batch * heads * chunk * chunk;
        let cdv_elems = batch * heads * chunk * dv;

        let g_data: Vec<f32> = (0..c_elems)
            .map(|idx| -0.15 + ((idx % 19) as f32) * 0.004)
            .collect();
        let v_data: Vec<f32> = (0..cdv_elems)
            .map(|idx| ((idx % 43) as f32 - 21.0) * 0.025)
            .collect();
        let kkt_data: Vec<f32> = (0..cc_elems)
            .map(|idx| ((idx % 31) as f32 - 15.0) * 0.0125)
            .collect();
        let qkt_data: Vec<f32> = (0..cc_elems)
            .map(|idx| ((idx % 37) as f32 - 18.0) * 0.01)
            .collect();
        let ks_data: Vec<f32> = (0..cdv_elems)
            .map(|idx| ((idx % 47) as f32 - 23.0) * 0.0125)
            .collect();
        let qs_data: Vec<f32> = (0..cdv_elems)
            .map(|idx| ((idx % 53) as f32 - 26.0) * 0.01)
            .collect();

        let g =
            Tensor::from_slice(&g_data, (batch, heads, chunk), &device)?.to_dtype(DType::BF16)?;
        let v = Tensor::from_slice(&v_data, (batch, heads, chunk, dv), &device)?
            .to_dtype(DType::BF16)?;
        let kkt = Tensor::from_slice(&kkt_data, (batch, heads, chunk, chunk), &device)?
            .to_dtype(DType::BF16)?;
        let qkt = Tensor::from_slice(&qkt_data, (batch, heads, chunk, chunk), &device)?
            .to_dtype(DType::BF16)?;
        let ks_entry = Tensor::from_slice(&ks_data, (batch, heads, chunk, dv), &device)?
            .to_dtype(DType::BF16)?;
        let q_s = Tensor::from_slice(&qs_data, (batch, heads, chunk, dv), &device)?
            .to_dtype(DType::BF16)?;

        assert!(metal_gdn_chunk_prep_supports(
            &g, &v, &kkt, &qkt, &ks_entry, &q_s
        ));

        let (a_ref, b_ref, vp_ref, qss_ref, decay_ref, plast_ref) =
            gdn_chunk_prep_reference(&g, &v, &kkt, &qkt, &ks_entry, &q_s)?;
        let (a_fused, b_fused, vp_fused, qss_fused, decay_fused, plast_fused) =
            metal_gdn_chunk_prep_bf16(&g, &v, &kkt, &qkt, &ks_entry, &q_s)?;

        let checks = [
            ("a_strict", &a_ref, &a_fused),
            ("b_mask", &b_ref, &b_fused),
            ("v_prime", &vp_ref, &vp_fused),
            ("q_s_scaled", &qss_ref, &qss_fused),
            ("decay_last_col", &decay_ref, &decay_fused),
            ("p_last", &plast_ref, &plast_fused),
        ];
        for (name, want, got) in checks {
            let max = max_abs_diff(want, got)?;
            let mean = mean_abs_diff(want, got)?;
            assert!(
                max < 1e-2,
                "GDN chunk-prep {name} max_abs_diff={max:e} exceeds tolerance"
            );
            assert!(
                mean < 1e-3,
                "GDN chunk-prep {name} mean_abs_diff={mean:e} exceeds tolerance"
            );
        }

        Ok(())
    }

    #[test]
    fn test_gdn_qk_norm_matches_fallback_decode_shape() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let batch = 1usize;
        let seq_len = 1usize;
        let heads = 32usize;
        let hidden = 128usize;
        let q_scale = 1.0 / (hidden as f64).sqrt();
        let eps = 1e-6;

        let q = Tensor::randn(0.0f32, 0.5, (batch, seq_len, heads, hidden), &device)?;
        let k = Tensor::randn(0.0f32, 0.5, (batch, seq_len, heads, hidden), &device)?;
        assert!(metal_gdn_qk_norm_supports(&q, &k));

        let (q_ref, k_ref) = gdn_qk_norm_reference(&q, &k, q_scale, eps)?;
        let (q_fused, k_fused) = metal_gdn_qk_norm_f32_bf16(&q, &k, q_scale as f32, eps as f32)?;

        assert_eq!(q_fused.dims(), &[batch, seq_len, heads, hidden]);
        assert_eq!(k_fused.dims(), &[batch, seq_len, heads, hidden]);

        let q_diff = max_abs_diff(&q_ref, &q_fused)?;
        let k_diff = max_abs_diff(&k_ref, &k_fused)?;
        let q_mean = mean_abs_diff(&q_ref, &q_fused)?;
        let k_mean = mean_abs_diff(&k_ref, &k_fused)?;
        assert!(
            q_diff < 1e-2,
            "GDN Q norm parity failed: max_abs_diff={q_diff}"
        );
        assert!(
            k_diff < 1e-2,
            "GDN K norm parity failed: max_abs_diff={k_diff}"
        );
        assert!(
            q_mean < 1e-3,
            "GDN Q norm parity failed: mean_abs_diff={q_mean}"
        );
        assert!(
            k_mean < 1e-3,
            "GDN K norm parity failed: mean_abs_diff={k_mean}"
        );

        Ok(())
    }

    #[test]
    fn test_gdn_qk_norm_matches_fallback_prefill_shape() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let batch = 1usize;
        let seq_len = 512usize;
        let heads = 32usize;
        let hidden = 128usize;
        let q_scale = 1.0 / (hidden as f64).sqrt();
        let eps = 1e-6;

        let q = Tensor::randn(0.0f32, 0.7, (batch, seq_len, heads, hidden), &device)?;
        let k = Tensor::randn(0.0f32, 0.7, (batch, seq_len, heads, hidden), &device)?;

        let (q_ref, k_ref) = gdn_qk_norm_reference(&q, &k, q_scale, eps)?;
        let (q_fused, k_fused) = metal_gdn_qk_norm_f32_bf16(&q, &k, q_scale as f32, eps as f32)?;

        let q_diff = max_abs_diff(&q_ref, &q_fused)?;
        let k_diff = max_abs_diff(&k_ref, &k_fused)?;
        let q_mean = mean_abs_diff(&q_ref, &q_fused)?;
        let k_mean = mean_abs_diff(&k_ref, &k_fused)?;
        assert!(
            q_diff < 1e-2,
            "prefill GDN Q norm parity failed: max_abs_diff={q_diff}"
        );
        assert!(
            k_diff < 1e-2,
            "prefill GDN K norm parity failed: max_abs_diff={k_diff}"
        );
        assert!(
            q_mean < 1e-3,
            "prefill GDN Q norm parity failed: mean_abs_diff={q_mean}"
        );
        assert!(
            k_mean < 1e-3,
            "prefill GDN K norm parity failed: mean_abs_diff={k_mean}"
        );

        Ok(())
    }

    #[test]
    fn test_gdn_qk_norm_known_values_and_zeros() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let hidden = 128usize;
        let q_scale = 1.0 / (hidden as f32).sqrt();
        let eps = 1e-6f32;

        let ones = Tensor::ones((1usize, 1usize, 1usize, hidden), DType::F32, &device)?;
        let (q_ones, k_ones) = metal_gdn_qk_norm_f32_bf16(&ones, &ones, q_scale, eps)?;
        let q_expected = Tensor::new(
            &[1.0f32 / hidden as f32],
            &device,
        )?
        .to_dtype(DType::BF16)?
        .broadcast_as((1usize, 1usize, 1usize, hidden))?;
        let k_expected = Tensor::new(
            &[1.0f32 / ((hidden as f32) + eps).sqrt()],
            &device,
        )?
        .to_dtype(DType::BF16)?
        .broadcast_as((1usize, 1usize, 1usize, hidden))?;
        assert!(max_abs_diff(&q_ones, &q_expected)? < 1e-4);
        assert!(max_abs_diff(&k_ones, &k_expected)? < 1e-4);

        let zeros = Tensor::zeros((1usize, 1usize, 1usize, hidden), DType::F32, &device)?;
        let (q_zero, k_zero) = metal_gdn_qk_norm_f32_bf16(&zeros, &zeros, q_scale, eps)?;
        assert_eq!(q_zero.to_dtype(DType::F32)?.max_all()?.to_scalar::<f32>()?, 0.0);
        assert_eq!(k_zero.to_dtype(DType::F32)?.max_all()?.to_scalar::<f32>()?, 0.0);

        Ok(())
    }

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
