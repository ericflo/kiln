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
const DISABLE_METAL_GDN_FORWARD_SUBSTITUTION: &str = "KILN_DISABLE_METAL_GDN_FORWARD_SUBSTITUTION";
const DISABLE_METAL_GDN_RECURRENT: &str = "KILN_DISABLE_METAL_GDN_RECURRENT";
const DISABLE_METAL_GDN_DECODE_GATES_RECURRENT: &str =
    "KILN_DISABLE_METAL_GDN_DECODE_GATES_RECURRENT";
const DISABLE_METAL_GDN_DECODE_GATES_RECURRENT_RMSNORM: &str =
    "KILN_DISABLE_METAL_GDN_DECODE_GATES_RECURRENT_RMSNORM";
const DISABLE_METAL_GATED_RMSNORM: &str = "KILN_DISABLE_METAL_GATED_RMSNORM";
const DISABLE_METAL_GDN_QK_NORM: &str = "KILN_DISABLE_METAL_GDN_QK_NORM";
const DISABLE_METAL_GDN_QKV_CONV_NORM: &str = "KILN_DISABLE_METAL_GDN_QKV_CONV_NORM";
const DISABLE_METAL_GDN_PREFILL_QKV_CONV_SPLIT: &str =
    "KILN_DISABLE_METAL_GDN_PREFILL_QKV_CONV_SPLIT";
const DISABLE_RMSNORM_KERNEL: &str = "KILN_DISABLE_RMSNORM_KERNEL";
const DISABLE_METAL_RMSNORM: &str = "KILN_DISABLE_METAL_RMSNORM";
const DISABLE_METAL_MLP_GATE_UP_FUSION: &str = "KILN_DISABLE_METAL_MLP_GATE_UP_FUSION";
const DISABLE_METAL_ATTN_GATE_FUSION: &str = "KILN_DISABLE_METAL_ATTN_GATE_FUSION";
const DISABLE_METAL_FUSED_QKV_PROJ: &str = "KILN_DISABLE_METAL_FUSED_QKV_PROJ";
const DISABLE_METAL_GDN_IN_PROJ_FUSION: &str = "KILN_DISABLE_METAL_GDN_IN_PROJ_FUSION";
const ENABLE_METAL_LM_HEAD_ARGMAX: &str = "KILN_ENABLE_METAL_LM_HEAD_ARGMAX";
const DISABLE_METAL_LM_HEAD_ARGMAX: &str = "KILN_DISABLE_METAL_LM_HEAD_ARGMAX";
const DISABLE_METAL_LM_HEAD_ARGMAX_GPU_REDUCE: &str =
    "KILN_DISABLE_METAL_LM_HEAD_ARGMAX_GPU_REDUCE";
const DISABLE_METAL_PAGED_ATTN_DECODE_CONTIGUOUS: &str =
    "KILN_DISABLE_METAL_PAGED_ATTN_DECODE_CONTIGUOUS";
const DISABLE_METAL_PAGED_KV_WRITE_TOKEN_MAJOR: &str =
    "KILN_DISABLE_METAL_PAGED_KV_WRITE_TOKEN_MAJOR";
const DISABLE_METAL_TRANSPOSED_COOP_GEMV: &str = "KILN_DISABLE_METAL_TRANSPOSED_COOP_GEMV";
const DISABLE_METAL_TRANSPOSED_COOP_GEMV_TILE8: &str =
    "KILN_DISABLE_METAL_TRANSPOSED_COOP_GEMV_TILE8";
const METAL_TRANSPOSED_COOP_GEMV_TILE4_COLS: usize = 4;
const METAL_TRANSPOSED_COOP_GEMV_TILE8_COLS: usize = 8;
const METAL_TRANSPOSED_COOP_GEMV_SIMDGROUPS: usize = 4;
const METAL_TRANSPOSED_COOP_GEMV_THREADS: usize = 32 * METAL_TRANSPOSED_COOP_GEMV_SIMDGROUPS;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MetalTransposedCoopGemvTile {
    Tile4,
    Tile8,
}

impl MetalTransposedCoopGemvTile {
    fn function_name(self) -> &'static str {
        match self {
            Self::Tile4 => "kiln_transposed_coop_gemv4_bf16",
            Self::Tile8 => "kiln_transposed_coop_gemv8_bf16",
        }
    }

    fn label(self) -> &'static str {
        self.function_name()
    }

    fn tile_cols(self) -> usize {
        match self {
            Self::Tile4 => METAL_TRANSPOSED_COOP_GEMV_TILE4_COLS,
            Self::Tile8 => METAL_TRANSPOSED_COOP_GEMV_TILE8_COLS,
        }
    }
}

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
    gdn_in_proj: bool,
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
            gdn_gates: env_present(DISABLE_FUSED_GDN_GATES) || env_truthy(DISABLE_METAL_GDN_GATES),
            gated_rms_norm: env_truthy(DISABLE_METAL_GATED_RMSNORM),
            gdn_in_proj: gdn_kernel || env_truthy(DISABLE_METAL_GDN_IN_PROJ_FUSION),
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
    metal_rotary_qk_pipeline(metal_device)?;
    metal_gdn_qk_norm_pipeline(metal_device)?;
    metal_gdn_qk_norm_gqa_pipeline(metal_device)?;
    metal_gdn_decode_qkv_conv_norm_pipeline(metal_device)?;
    metal_gdn_prefill_qkv_conv_split_pipeline(metal_device)?;
    metal_gdn_gates_pipeline(metal_device)?;
    metal_gdn_decode_gates_recurrent_pipeline(metal_device)?;
    metal_gdn_decode_gates_recurrent_rmsnorm_pipeline(metal_device)?;
    metal_gated_rms_norm_pipeline(metal_device)?;
    metal_gdn_in_proj_pipeline(metal_device)?;
    metal_gdn_recurrent_pipeline(metal_device)?;
    metal_gdn_recurrent_prefill_head_last_pipeline(metal_device)?;
    metal_gdn_forward_substitution_pipeline(metal_device)?;
    metal_gdn_chunk_prep_pipeline(metal_device)?;
    metal_gdn_full_chunk_forward_pipeline(metal_device)?;
    metal_conv1d_prefill_pipeline(metal_device)?;
    metal_conv1d_update_pipeline(metal_device)?;
    metal_lm_head_pipeline(metal_device)?;
    if !metal_lm_head_argmax_disabled() {
        metal_lm_head_argmax_pipeline(metal_device)?;
        if !metal_lm_head_argmax_gpu_reduce_disabled() {
            metal_lm_head_argmax_reduce_pipeline(metal_device)?;
        }
    }
    if !metal_mlp_gate_up_fusion_disabled() {
        metal_mlp_gate_up_pipeline(metal_device)?;
    }
    if !metal_attn_gate_fusion_disabled() {
        metal_attn_gate_sigmoid_mul_pipeline(metal_device)?;
    }
    if !metal_transposed_coop_gemv_disabled() {
        let default_tile = metal_transposed_coop_gemv_default_tile();
        metal_transposed_coop_gemv_pipeline(metal_device, default_tile)?;
        metal_transposed_coop_gemv_batch_pipeline(metal_device)?;
        if default_tile != MetalTransposedCoopGemvTile::Tile4 {
            metal_transposed_coop_gemv_pipeline(metal_device, MetalTransposedCoopGemvTile::Tile4)?;
        }
        if !metal_fused_qkv_proj_disabled() {
            metal_fused_qkv_transposed_coop_gemv_pipeline(metal_device)?;
        }
    }
    metal_paged_kv_head_major_read_pipeline(metal_device)?;
    metal_paged_kv_head_major_read_append_token_major_pipeline(metal_device)?;
    if !metal_paged_attn_decode_contiguous_disabled() {
        metal_paged_attn_decode_contiguous_pipeline(metal_device)?;
        metal_paged_attn_decode_contiguous_batch_pipeline(metal_device)?;
    }
    if !metal_paged_kv_write_token_major_disabled() {
        metal_paged_kv_write_token_major_pipeline(metal_device)?;
        metal_paged_kv_write_token_major_batch_pipeline(metal_device)?;
    }
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
        std::env::var("KILN_DISABLE_METAL_SDPA").is_err()
    }

    fn supports_flash_attn_prefill_head_major(&self) -> bool {
        std::env::var("KILN_DISABLE_METAL_SDPA").is_err()
    }

    // Note: keep `supports_*` returning true so the planner picks the SDPA
    // path; the per-call gate inside the kernel functions then decides
    // whether the *specific* shape is safe and silently falls back to the
    // naive softmax+matmul path when it isn't.

    fn supports_flash_attn_paged_decode(&self) -> bool {
        true
    }

    fn flash_attn_paged_decode_contiguous(
        &self,
        q: &Tensor,
        k_pool: &Tensor,
        v_pool: &Tensor,
        start_slot: usize,
        total_seqlen_k: usize,
        softmax_scale: f32,
    ) -> Result<Option<Tensor>> {
        if !metal_paged_attn_decode_contiguous_supports(
            q,
            k_pool,
            v_pool,
            start_slot,
            total_seqlen_k,
        ) {
            return Ok(None);
        }
        let out = metal_paged_attn_decode_contiguous_bf16_d256(
            q,
            k_pool,
            v_pool,
            start_slot,
            total_seqlen_k,
            softmax_scale,
        )
        .context("metal contiguous paged decode attention failed")?;
        Ok(Some(out))
    }

    fn supports_paged_kv_head_major_read(&self) -> bool {
        true
    }

    fn supports_paged_kv_head_major_read_append_token_major(&self) -> bool {
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

    fn supports_gdn_full_chunk_forward(&self) -> bool {
        !self.disable.gdn_forward_substitution
    }

    fn supports_gdn_full_chunk_forward_head_last(&self) -> bool {
        !self.disable.gdn_forward_substitution
    }

    fn supports_gdn_recurrent_prefill_head_last(&self) -> bool {
        !self.disable.gdn_recurrent
    }

    fn supports_gdn_recurrent_prefill_native_head_last(&self) -> bool {
        !self.disable.gdn_recurrent
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
        if std::env::var("KILN_DISABLE_METAL_SDPA").is_ok() {
            return Ok(None);
        }
        // Decline (caller falls back to the portable path) when candle's SDPA
        // can't handle the shape/dtype. Cheaper than surfacing a kernel error
        // from inside the fused path.
        if !matches!(q.dtype(), DType::BF16 | DType::F16 | DType::F32) {
            return Ok(None);
        }
        let head_dim = q.dim(candle_core::D::Minus1)?;
        if !metal_sdpa_supports_head_dim(head_dim) {
            return Ok(None);
        }
        let q_seq = q.dim(2)?;
        if !metal_sdpa_full_safe_for_q_seq(head_dim, q_seq) {
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
        if std::env::var("KILN_DISABLE_METAL_SDPA").is_ok() {
            return Ok(None);
        }
        if !matches!(q.dtype(), DType::BF16 | DType::F16 | DType::F32) {
            return Ok(None);
        }
        let head_dim = q.dim(candle_core::D::Minus1)?;
        if !metal_sdpa_supports_head_dim(head_dim) {
            return Ok(None);
        }
        let q_seq = q.dim(2)?;
        if !metal_sdpa_full_safe_for_q_seq(head_dim, q_seq) {
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

    fn paged_kv_head_major_read(
        &self,
        k_pool: &Tensor,
        v_pool: &Tensor,
        start_slot: usize,
        seq_len: usize,
    ) -> Result<Option<(Tensor, Tensor)>> {
        if !metal_paged_kv_head_major_read_supports(k_pool, v_pool, start_slot, seq_len) {
            return Ok(None);
        }
        metal_paged_kv_head_major_read_bf16(k_pool, v_pool, start_slot, seq_len)
            .map(Some)
            .context("metal paged_kv_head_major_read failed")
    }

    fn paged_kv_head_major_read_append_token_major(
        &self,
        k_pool: &Tensor,
        v_pool: &Tensor,
        start_slot: usize,
        prefix_len: usize,
        k_tail: &Tensor,
        v_tail: &Tensor,
    ) -> Result<Option<(Tensor, Tensor)>> {
        if !metal_paged_kv_head_major_read_append_token_major_supports(
            k_pool, v_pool, start_slot, prefix_len, k_tail, v_tail,
        ) {
            return Ok(None);
        }
        metal_paged_kv_head_major_read_append_token_major_bf16(
            k_pool, v_pool, start_slot, prefix_len, k_tail, v_tail,
        )
        .map(Some)
        .context("metal paged_kv_head_major_read_append_token_major failed")
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

    fn gdn_full_chunk_forward(
        &self,
        g: &Tensor,
        v: &Tensor,
        kkt: &Tensor,
        qkt: &Tensor,
        ks_entry: &Tensor,
        q_s: &Tensor,
        beta: &Tensor,
        k_t: &Tensor,
        state: &mut Tensor,
    ) -> Result<Option<Tensor>> {
        if self.disable.gdn_forward_substitution
            || !metal_gdn_full_chunk_forward_supports(
                g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state,
            )
        {
            return Ok(None);
        }
        let out =
            metal_gdn_full_chunk_forward_bf16(g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state)
                .context("metal gdn_full_chunk_forward kernel failed")?;
        Ok(Some(out))
    }

    #[allow(clippy::too_many_arguments)]
    fn gdn_full_chunk_forward_head_last_into(
        &self,
        g: &Tensor,
        v: &Tensor,
        kkt: &Tensor,
        qkt: &Tensor,
        ks_entry: &Tensor,
        q_s: &Tensor,
        beta: &Tensor,
        k_t: &Tensor,
        state: &mut Tensor,
        out: &Tensor,
        t_start: usize,
        seq_len: usize,
    ) -> Result<bool> {
        if self.disable.gdn_forward_substitution
            || !metal_gdn_full_chunk_forward_head_last_supports(
                g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state, out, t_start, seq_len,
            )
        {
            return Ok(false);
        }
        metal_gdn_full_chunk_forward_head_last_into_bf16(
            g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state, out, t_start, seq_len,
        )
        .context("metal gdn_full_chunk_forward_head_last_into kernel failed")?;
        Ok(true)
    }

    fn gdn_recurrent_prefill_head_last(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        beta: &Tensor,
        g: &Tensor,
        state: &mut Tensor,
    ) -> Result<Option<Tensor>> {
        if self.disable.gdn_recurrent
            || !metal_gdn_recurrent_prefill_head_last_supports(q, k, v, beta, g, state)
        {
            return Ok(None);
        }
        let out = metal_gdn_recurrent_prefill_head_last_bf16(q, k, v, beta, g, state)
            .context("metal gdn_recurrent_prefill_head_last kernel failed")?;
        Ok(Some(out))
    }

    fn gdn_recurrent_prefill_native_head_last(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        beta: &Tensor,
        g: &Tensor,
        state: &mut Tensor,
    ) -> Result<Option<Tensor>> {
        if self.disable.gdn_recurrent
            || !metal_gdn_recurrent_prefill_native_head_last_supports(q, k, v, beta, g, state)
        {
            return Ok(None);
        }
        let out = metal_gdn_recurrent_prefill_native_head_last_bf16(q, k, v, beta, g, state)
            .context("metal gdn_recurrent_prefill_native_head_last kernel failed")?;
        Ok(Some(out))
    }

    fn gdn_in_proj_decode(
        &self,
        x: &Tensor,
        in_proj_qkv_t: &Tensor,
        in_proj_z_t: &Tensor,
        in_proj_a_t: &Tensor,
        in_proj_b_t: &Tensor,
    ) -> Result<Option<(Tensor, Tensor, Tensor, Tensor)>> {
        if self.disable.gdn_in_proj
            || !metal_gdn_in_proj_decode_supports(
                x,
                in_proj_qkv_t,
                in_proj_z_t,
                in_proj_a_t,
                in_proj_b_t,
            )
        {
            return Ok(None);
        }
        let out =
            metal_gdn_in_proj_decode_bf16(x, in_proj_qkv_t, in_proj_z_t, in_proj_a_t, in_proj_b_t)
                .context("metal gdn_in_proj_decode kernel failed")?;
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
        let out =
            metal_gdn_gates_bf16(a, b, a_log, dt_bias).context("metal gdn_gates kernel failed")?;
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

/// candle-metal-kernels 0.10.2's `call_sdpa_full` (steel attention) silently
/// returns an all-NaN output buffer for many BF16 prefill shapes on the
/// Qwen3.5-4B (head_dim=256) layout. Reproduced empirically by varying
/// q_seq while holding all other model state constant (greedy, prefix cache
/// off, no streaming prefill, no fused mlp/qkv/attn-gate kernels):
///
///   ptoks=120 → NaN     ptoks=266 → NaN
///   ptoks=138 → finite  ptoks=330 → NaN
///   ptoks=170 → finite  ptoks=410 → finite
///   ptoks=210 → finite  ptoks=522 → NaN
///                       ptoks=610 → NaN
///
/// There is no clean alignment / `q_seq mod bq` rule — the failure depends
/// on internal kernel state we cannot inspect from outside candle.
/// Short q_seq <= 8 (the "vector" kernel path) is unaffected, and a
/// previously-suspected `8 < q_seq < bq` boundary turned out to be only
/// one slice of a larger NaN surface.
///
/// Until upstream candle fixes the kernel, decline the SDPA full path for
/// any q_seq > 8 and let the caller fall back to the naive softmax+matmul
/// prefill (which is bit-exact, just ~5–10× slower per prefill — typically
/// <1 s extra at chat-context sizes, and the per-token decode hot path is
/// untouched). Set `KILN_ENABLE_METAL_SDPA_FULL=1` to opt back in for
/// benchmarking once the upstream fix lands.
fn metal_sdpa_full_safe_for_q_seq(_head_dim: usize, q_seq: usize) -> bool {
    if q_seq <= 8 {
        return true;
    }
    matches!(
        std::env::var("KILN_ENABLE_METAL_SDPA_FULL").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

fn metal_gdn_qk_norm_disabled() -> bool {
    env_truthy(DISABLE_METAL_GDN_QK_NORM)
}

fn metal_gdn_qkv_conv_norm_disabled() -> bool {
    env_present(DISABLE_FUSED_CONV1D)
        || env_truthy(DISABLE_METAL_FUSED_CONV1D)
        || env_truthy(DISABLE_METAL_GDN_QKV_CONV_NORM)
}

fn metal_gdn_prefill_qkv_conv_split_disabled() -> bool {
    env_present(DISABLE_FUSED_CONV1D)
        || env_truthy(DISABLE_METAL_CONV1D_PREFILL)
        || env_truthy(DISABLE_METAL_FUSED_CONV1D)
        || env_truthy(DISABLE_METAL_GDN_PREFILL_QKV_CONV_SPLIT)
}

fn metal_gdn_gates_disabled() -> bool {
    env_present(DISABLE_FUSED_GDN_GATES) || env_truthy(DISABLE_METAL_GDN_GATES)
}

fn metal_gdn_recurrent_disabled() -> bool {
    env_truthy(DISABLE_GDN_KERNEL) || env_truthy(DISABLE_METAL_GDN_RECURRENT)
}

fn metal_gdn_decode_gates_recurrent_disabled() -> bool {
    metal_gdn_gates_disabled()
        || metal_gdn_recurrent_disabled()
        || env_truthy(DISABLE_METAL_GDN_DECODE_GATES_RECURRENT)
}

fn metal_gdn_decode_gates_recurrent_rmsnorm_disabled() -> bool {
    metal_gdn_decode_gates_recurrent_disabled()
        || env_truthy(DISABLE_METAL_GDN_DECODE_GATES_RECURRENT_RMSNORM)
        || env_truthy(DISABLE_METAL_GATED_RMSNORM)
}

fn metal_rms_norm_disabled() -> bool {
    env_present(DISABLE_RMSNORM_KERNEL) || env_truthy(DISABLE_METAL_RMSNORM)
}

pub(crate) fn metal_mlp_gate_up_fusion_disabled() -> bool {
    env_truthy(DISABLE_METAL_MLP_GATE_UP_FUSION)
}

fn metal_attn_gate_fusion_disabled() -> bool {
    env_truthy(DISABLE_METAL_ATTN_GATE_FUSION)
}

fn metal_fused_qkv_proj_disabled() -> bool {
    env_truthy(DISABLE_METAL_FUSED_QKV_PROJ) || metal_transposed_coop_gemv_tile8_disabled()
}

pub(crate) fn metal_lm_head_argmax_disabled() -> bool {
    // On the Qwen3.5-4B macOS desktop path, Candle's materialized last-row
    // projection plus argmax is faster than this custom chunk/reduce kernel.
    // Keep the kernel available for tuning, but require explicit opt-in.
    env_truthy(DISABLE_METAL_LM_HEAD_ARGMAX) || !env_truthy(ENABLE_METAL_LM_HEAD_ARGMAX)
}

fn metal_lm_head_argmax_gpu_reduce_disabled() -> bool {
    env_truthy(DISABLE_METAL_LM_HEAD_ARGMAX_GPU_REDUCE)
}

fn metal_paged_attn_decode_contiguous_disabled() -> bool {
    env_truthy(DISABLE_METAL_PAGED_ATTN_DECODE_CONTIGUOUS)
}

fn metal_paged_kv_write_token_major_disabled() -> bool {
    env_truthy(DISABLE_METAL_PAGED_KV_WRITE_TOKEN_MAJOR)
}

pub(crate) fn metal_transposed_coop_gemv_disabled() -> bool {
    env_truthy(DISABLE_METAL_TRANSPOSED_COOP_GEMV)
}

fn metal_transposed_coop_gemv_tile8_disabled() -> bool {
    env_truthy(DISABLE_METAL_TRANSPOSED_COOP_GEMV_TILE8)
}

fn metal_transposed_coop_gemv_default_tile() -> MetalTransposedCoopGemvTile {
    if metal_transposed_coop_gemv_tile8_disabled() {
        MetalTransposedCoopGemvTile::Tile4
    } else {
        MetalTransposedCoopGemvTile::Tile8
    }
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
        || a_log.dtype() != DType::F32
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

fn metal_gdn_full_chunk_forward_supports(
    g: &Tensor,
    v: &Tensor,
    kkt: &Tensor,
    qkt: &Tensor,
    ks_entry: &Tensor,
    q_s: &Tensor,
    beta: &Tensor,
    k_t: &Tensor,
    state: &Tensor,
) -> bool {
    if !metal_gdn_chunk_prep_supports(g, v, kkt, qkt, ks_entry, q_s)
        || !matches!(beta.device(), Device::Metal(_))
        || !matches!(k_t.device(), Device::Metal(_))
        || !matches!(state.device(), Device::Metal(_))
    {
        return false;
    }
    if beta.dtype() != DType::BF16 || k_t.dtype() != DType::BF16 || state.dtype() != DType::BF16 {
        return false;
    }
    let Ok((batch, heads, chunk)) = g.dims3() else {
        return false;
    };
    let Ok((_, _, _, dv)) = v.dims4() else {
        return false;
    };
    let Ok((b_beta, h_beta, c_beta)) = beta.dims3() else {
        return false;
    };
    let Ok((b_kt, h_kt, dk, c_kt)) = k_t.dims4() else {
        return false;
    };
    let Ok((b_state, h_state, dk_state, dv_state)) = state.dims4() else {
        return false;
    };
    (b_beta, h_beta, c_beta) == (batch, heads, chunk)
        && (b_kt, h_kt, c_kt) == (batch, heads, chunk)
        && (b_state, h_state, dk_state, dv_state) == (batch, heads, dk, dv)
        && chunk == 64
        && dk > 0
        && dk <= 128
        && dv > 0
        && dv <= 128
        && state.is_contiguous()
}

fn metal_gdn_full_chunk_forward_strided_inputs_support(
    g: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    k_t: &Tensor,
    heads: usize,
) -> bool {
    fn flat_batch_head_ok(stride: &[usize], heads: usize) -> bool {
        stride.len() >= 2
            && stride[1] > 0
            && heads
                .checked_mul(stride[1])
                .is_some_and(|expected| stride[0] == expected)
    }

    fn stride_u32_ok(stride: usize) -> bool {
        stride > 0 && stride <= u32::MAX as usize
    }

    let (_g_storage, g_layout) = g.storage_and_layout();
    let (_v_storage, v_layout) = v.storage_and_layout();
    let (_beta_storage, beta_layout) = beta.storage_and_layout();
    let (_kt_storage, kt_layout) = k_t.storage_and_layout();
    let g_stride = g_layout.stride();
    let v_stride = v_layout.stride();
    let beta_stride = beta_layout.stride();
    let kt_stride = kt_layout.stride();

    if g_stride.len() != 3 || v_stride.len() != 4 || beta_stride.len() != 3 || kt_stride.len() != 4
    {
        return false;
    }
    if !flat_batch_head_ok(g_stride, heads)
        || !flat_batch_head_ok(v_stride, heads)
        || !flat_batch_head_ok(beta_stride, heads)
        || !flat_batch_head_ok(kt_stride, heads)
    {
        return false;
    }

    // This path only needs to support a time-window narrow. Keep the value
    // dimension contiguous so the per-value lane remains coalesced.
    v_stride[3] == 1
        && [
            g_stride[1],
            g_stride[2],
            v_stride[1],
            v_stride[2],
            v_stride[3],
            beta_stride[1],
            beta_stride[2],
            kt_stride[1],
            kt_stride[2],
            kt_stride[3],
        ]
        .into_iter()
        .all(stride_u32_ok)
}

#[allow(clippy::too_many_arguments)]
fn metal_gdn_full_chunk_forward_head_last_supports(
    g: &Tensor,
    v: &Tensor,
    kkt: &Tensor,
    qkt: &Tensor,
    ks_entry: &Tensor,
    q_s: &Tensor,
    beta: &Tensor,
    k_t: &Tensor,
    state: &Tensor,
    out: &Tensor,
    t_start: usize,
    seq_len: usize,
) -> bool {
    if !metal_gdn_full_chunk_forward_supports(g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state)
        || !matches!(out.device(), Device::Metal(_))
        || out.dtype() != DType::BF16
        || !out.is_contiguous()
    {
        return false;
    }
    let Ok((batch, heads, chunk)) = g.dims3() else {
        return false;
    };
    let Ok((_, _, _, dv)) = v.dims4() else {
        return false;
    };
    out.dims4()
        .is_ok_and(|dims| dims == (batch, seq_len, heads, dv))
        && chunk == 64
        && t_start <= seq_len
        && t_start + chunk <= seq_len
        && metal_gdn_full_chunk_forward_strided_inputs_support(g, v, beta, k_t, heads)
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

const METAL_GDN_RECURRENT_PREFILL_MAX_SEQ_LEN: usize = 2048;

fn metal_gdn_recurrent_prefill_head_last_supports(
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
    let Ok((batch, q_heads, seq_len, dk)) = q.dims4() else {
        return false;
    };
    let Ok((b_k, h_k, t_k, dk_k)) = k.dims4() else {
        return false;
    };
    let Ok((b_v, v_heads, t_v, dv)) = v.dims4() else {
        return false;
    };
    let Ok((b_beta, h_beta, t_beta)) = beta.dims3() else {
        return false;
    };
    let Ok((b_g, h_g, t_g)) = g.dims3() else {
        return false;
    };
    let Ok((b_state, h_state, dk_state, dv_state)) = state.dims4() else {
        return false;
    };
    (b_k, h_k, t_k, dk_k) == (batch, q_heads, seq_len, dk)
        && (b_v, t_v) == (batch, seq_len)
        && (b_beta, h_beta, t_beta) == (batch, v_heads, seq_len)
        && (b_g, h_g, t_g) == (batch, v_heads, seq_len)
        && (b_state, h_state, dk_state, dv_state) == (batch, v_heads, dk, dv)
        && v_heads >= q_heads
        && v_heads % q_heads == 0
        && seq_len > 1
        && seq_len <= METAL_GDN_RECURRENT_PREFILL_MAX_SEQ_LEN
        && dk == 128
        && dv > 0
        && dv <= 128
        && state.is_contiguous()
}

fn metal_gdn_recurrent_prefill_native_head_last_supports(
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
    let Ok((batch, seq_len, q_heads, dk)) = q.dims4() else {
        return false;
    };
    let Ok((b_k, t_k, h_k, dk_k)) = k.dims4() else {
        return false;
    };
    let Ok((b_v, t_v, value_heads, dv)) = v.dims4() else {
        return false;
    };
    let Ok((b_beta, t_beta, h_beta)) = beta.dims3() else {
        return false;
    };
    let Ok((b_g, t_g, h_g)) = g.dims3() else {
        return false;
    };
    let Ok((b_state, h_state, dk_state, dv_state)) = state.dims4() else {
        return false;
    };
    (b_k, t_k, h_k, dk_k) == (batch, seq_len, q_heads, dk)
        && (b_v, t_v) == (batch, seq_len)
        && (b_beta, t_beta, h_beta) == (batch, seq_len, value_heads)
        && (b_g, t_g, h_g) == (batch, seq_len, value_heads)
        && (b_state, h_state, dk_state, dv_state) == (batch, value_heads, dk, dv)
        && value_heads >= q_heads
        && value_heads % q_heads == 0
        && seq_len >= 1
        && seq_len <= METAL_GDN_RECURRENT_PREFILL_MAX_SEQ_LEN
        && dk == 128
        && dv > 0
        && dv <= 128
        && state.is_contiguous()
}

pub(crate) fn metal_gdn_decode_gates_recurrent_supports(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    state: &Tensor,
) -> bool {
    if metal_gdn_decode_gates_recurrent_disabled() || !matches!(q.device(), Device::Metal(_)) {
        return false;
    }
    if !matches!(k.device(), Device::Metal(_))
        || !matches!(v.device(), Device::Metal(_))
        || !matches!(a.device(), Device::Metal(_))
        || !matches!(b.device(), Device::Metal(_))
        || !matches!(a_log.device(), Device::Metal(_))
        || !matches!(dt_bias.device(), Device::Metal(_))
        || !matches!(state.device(), Device::Metal(_))
    {
        return false;
    }
    if q.dtype() != DType::BF16
        || k.dtype() != DType::BF16
        || v.dtype() != DType::BF16
        || a.dtype() != DType::BF16
        || b.dtype() != DType::BF16
        || a_log.dtype() != DType::F32
        || dt_bias.dtype() != DType::BF16
        || state.dtype() != DType::BF16
    {
        return false;
    }
    let Ok((batch, seq_len, q_heads, dk)) = q.dims4() else {
        return false;
    };
    let Ok((b_k, t_k, h_k, dk_k)) = k.dims4() else {
        return false;
    };
    let Ok((b_v, t_v, value_heads, dv)) = v.dims4() else {
        return false;
    };
    let Ok((b_a, t_a, h_a)) = a.dims3() else {
        return false;
    };
    let Ok((b_b, t_b, h_b)) = b.dims3() else {
        return false;
    };
    let Ok((b_state, h_state, dk_state, dv_state)) = state.dims4() else {
        return false;
    };
    let Some(batch_heads) = batch.checked_mul(value_heads) else {
        return false;
    };
    batch > 0
        && seq_len == 1
        && (b_k, t_k, h_k, dk_k) == (batch, seq_len, q_heads, dk)
        && (b_v, t_v) == (batch, seq_len)
        && (b_a, t_a, h_a) == (batch, seq_len, value_heads)
        && (b_b, t_b, h_b) == (batch, seq_len, value_heads)
        && a_log.dims() == [value_heads]
        && dt_bias.dims() == [value_heads]
        && (b_state, h_state, dk_state, dv_state) == (batch, value_heads, dk, dv)
        && q_heads > 0
        && value_heads > q_heads
        && value_heads % q_heads == 0
        && dk == 128
        && dv == 128
        && batch_heads <= u32::MAX as usize
        && q_heads <= u32::MAX as usize
        && value_heads <= u32::MAX as usize
        && state.is_contiguous()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_decode_gates_recurrent_rmsnorm_supports(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    state: &Tensor,
    z: &Tensor,
    weight: &Tensor,
) -> bool {
    if metal_gdn_decode_gates_recurrent_rmsnorm_disabled() {
        return false;
    }
    if !metal_gdn_decode_gates_recurrent_supports(q, k, v, a, b, a_log, dt_bias, state) {
        return false;
    }
    metal_gated_rms_norm_supports(v, z, weight)
}

fn metal_gated_rms_norm_supports(x: &Tensor, z: &Tensor, weight: &Tensor) -> bool {
    if !matches!(x.device(), Device::Metal(_))
        || !matches!(z.device(), Device::Metal(_))
        || !matches!(weight.device(), Device::Metal(_))
    {
        return false;
    }
    if x.dtype() != DType::BF16 || z.dtype() != DType::BF16 || weight.dtype() != DType::F32 {
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

pub(crate) fn metal_gdn_qk_norm_gqa_supports(q: &Tensor, k: &Tensor, nv: usize) -> bool {
    if metal_gdn_qk_norm_disabled() {
        return false;
    }
    if !matches!(q.device(), Device::Metal(_)) || !matches!(k.device(), Device::Metal(_)) {
        return false;
    }
    if q.dtype() != DType::F32 || k.dtype() != DType::F32 {
        return false;
    }
    let Ok((_, _, nk, hidden)) = q.dims4() else {
        return false;
    };
    q.dims() == k.dims()
        && nk > 0
        && nv > nk
        && nv % nk == 0
        && hidden <= 8192
        && nv <= u32::MAX as usize
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_decode_qkv_conv_norm_supports(
    mixed_qkv: &Tensor,
    weight: &Tensor,
    conv_state: &Tensor,
    kernel_size: usize,
    nk: usize,
    dk: usize,
    nv: usize,
    dv: usize,
) -> bool {
    if metal_gdn_qkv_conv_norm_disabled() || metal_gdn_qk_norm_disabled() {
        return false;
    }
    if kernel_size != 4 || dk != 128 || dv != 128 {
        return false;
    }
    if !matches!(mixed_qkv.device(), Device::Metal(_))
        || !matches!(weight.device(), Device::Metal(_))
        || !matches!(conv_state.device(), Device::Metal(_))
    {
        return false;
    }
    if mixed_qkv.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || conv_state.dtype() != DType::F32
    {
        return false;
    }
    let Ok((batch, seq_len, channels)) = mixed_qkv.dims3() else {
        return false;
    };
    let qk_dim = nk.saturating_mul(dk);
    let v_dim = nv.saturating_mul(dv);
    let Some(expected_channels) = qk_dim.checked_mul(2).and_then(|n| n.checked_add(v_dim)) else {
        return false;
    };
    let weight_ok = match weight.rank() {
        3 => weight
            .dims3()
            .is_ok_and(|(c, one, k)| c == channels && one == 1 && k == kernel_size),
        2 => weight
            .dims2()
            .is_ok_and(|(c, k)| c == channels && k == kernel_size),
        _ => false,
    };
    let Some(rows) = nk
        .checked_add(nk)
        .and_then(|n| n.checked_add(nv))
        .and_then(|n| n.checked_mul(batch))
    else {
        return false;
    };
    batch > 0
        && seq_len == 1
        && channels == expected_channels
        && nk > 0
        && nv > nk
        && nv % nk == 0
        && weight_ok
        && conv_state
            .dims3()
            .is_ok_and(|(b, c, k)| (b, c, k) == (batch, channels, kernel_size - 1))
        && channels <= u32::MAX as usize
        && nk <= u32::MAX as usize
        && nv <= u32::MAX as usize
        && rows <= u32::MAX as usize
}

pub(crate) fn metal_rotary_embedding_supports(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    head_dim: usize,
    rotary_dim: usize,
) -> bool {
    if !matches!(q.device(), Device::Metal(_))
        || !matches!(k.device(), Device::Metal(_))
        || !matches!(cos.device(), Device::Metal(_))
        || !matches!(sin.device(), Device::Metal(_))
    {
        return false;
    }
    if q.dtype() != DType::BF16
        || k.dtype() != DType::BF16
        || cos.dtype() != DType::F32
        || sin.dtype() != DType::F32
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
    cos: &Tensor,
    sin: &Tensor,
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

const METAL_ROTARY_QK_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_rotary_qk_bf16(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const float* cos [[buffer(2)]],
    device const float* sin [[buffer(3)]],
    device bfloat* q_out [[buffer(4)]],
    device bfloat* k_out [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& q_heads [[buffer(8)]],
    constant uint& k_heads [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]],
    constant uint& rotary_dim [[buffer(11)]],
    constant uint& total_q [[buffer(12)]],
    constant uint& total [[buffer(13)]],
    constant uint& table_batch_stride [[buffer(14)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) {
        return;
    }

    const bool is_q = gid < total_q;
    const uint local = is_q ? gid : gid - total_q;
    const uint heads = is_q ? q_heads : k_heads;
    device const bfloat* src = is_q ? q : k;
    device bfloat* dst = is_q ? q_out : k_out;

    const uint d = local % head_dim;
    const uint h = (local / head_dim) % heads;
    const uint t = (local / (head_dim * heads)) % seq_len;
    const uint b = local / (head_dim * heads * seq_len);
    if (b >= batch) {
        return;
    }

    if (d >= rotary_dim) {
        dst[local] = src[local];
        return;
    }

    const uint half_rotary = rotary_dim / 2;
    const bool first_half = d < half_rotary;
    const uint pair_d = first_half ? d + half_rotary : d - half_rotary;
    const uint pair_idx = ((b * seq_len + t) * heads + h) * head_dim + pair_d;
    const uint table_t = table_batch_stride == 0 ? t : b * table_batch_stride + t;
    const uint table_idx = table_t * half_rotary + (first_half ? d : pair_d);
    const float x = static_cast<float>(src[local]);
    const float y = static_cast<float>(src[pair_idx]);
    const float c = cos[table_idx];
    const float s = sin[table_idx];
    const float rotated = first_half ? (x * c - y * s) : (y * s + x * c);
    dst[local] = static_cast<bfloat>(rotated);
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

kernel void kiln_gdn_qk_norm_gqa_f32_bf16(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device bfloat* q_out [[buffer(2)]],
    device bfloat* k_out [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& nk [[buffer(5)]],
    constant uint& nv [[buffer(6)]],
    constant uint& hidden [[buffer(7)]],
    constant uint& gqa_ratio [[buffer(8)]],
    constant float& q_scale [[buffer(9)]],
    constant float& eps [[buffer(10)]],
    constant uint& threadgroup_width [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float q_scratch[1024];
    threadgroup float k_scratch[1024];

    const uint row = gid.y;
    if (row >= rows) {
        return;
    }

    const uint src_head = row % nk;
    const uint bt = row / nk;
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
    const uint dst_head_base = src_head * gqa_ratio;
    for (uint col = tid; col < hidden; col += threadgroup_width) {
        const float q_norm = q[base + col] * q_inv * q_scale;
        const float k_norm = k[base + col] * k_inv;
        for (uint rep = 0; rep < gqa_ratio; ++rep) {
            const uint dst_head = dst_head_base + rep;
            const uint dst_idx = ((bt * nv + dst_head) * hidden) + col;
            q_out[dst_idx] = static_cast<bfloat>(q_norm);
            k_out[dst_idx] = static_cast<bfloat>(k_norm);
        }
    }
}
"#;

const METAL_GDN_DECODE_QKV_CONV_NORM_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gdn_decode_qkv_conv_norm_bf16(
    device const bfloat* mixed_qkv [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device float* conv_state [[buffer(2)]],
    device bfloat* q_out [[buffer(3)]],
    device bfloat* k_out [[buffer(4)]],
    device bfloat* v_out [[buffer(5)]],
    constant uint& nk [[buffer(6)]],
    constant uint& nv [[buffer(7)]],
    constant float& q_scale [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    uint2 tgroup [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr uint D = 128;
    threadgroup float values[D];
    threadgroup float sum_scratch[D];

    const uint row = tgroup.x;
    const uint batch_idx = tgroup.y;
    const uint local_row = row;
    const uint qk_dim = nk * D;
    const uint channels = qk_dim + qk_dim + nv * D;

    uint channel = 0;
    bool is_q = false;
    bool is_k = false;
    bool is_v = false;
    uint src_head = 0;
    uint v_head = 0;

    if (local_row < nk) {
        is_q = true;
        src_head = local_row;
        channel = src_head * D + tid;
    } else if (local_row < nk + nk) {
        is_k = true;
        src_head = local_row - nk;
        channel = qk_dim + src_head * D + tid;
    } else {
        is_v = true;
        v_head = local_row - nk - nk;
        channel = qk_dim + qk_dim + v_head * D + tid;
    }

    const uint token_idx = batch_idx * channels + channel;
    const uint state_base = (batch_idx * channels + channel) * 3;
    const uint weight_base = channel * 4;

    const float s0 = conv_state[state_base + 0];
    const float s1 = conv_state[state_base + 1];
    const float s2 = conv_state[state_base + 2];
    const float x0 = static_cast<float>(mixed_qkv[token_idx]);
    const float acc =
        s0 * static_cast<float>(weight[weight_base + 0]) +
        s1 * static_cast<float>(weight[weight_base + 1]) +
        s2 * static_cast<float>(weight[weight_base + 2]) +
        x0 * static_cast<float>(weight[weight_base + 3]);
    const float y = acc / (1.0f + exp(-acc));

    conv_state[state_base + 0] = s1;
    conv_state[state_base + 1] = s2;
    conv_state[state_base + 2] = x0;

    if (is_v) {
        const uint out_idx = (batch_idx * nv + v_head) * D + tid;
        v_out[out_idx] = static_cast<bfloat>(y);
        return;
    }

    values[tid] = y;
    sum_scratch[tid] = y * y;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = D / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_scratch[tid] += sum_scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv = rsqrt(sum_scratch[0] + eps);
    const float norm = values[tid] * inv * (is_q ? q_scale : 1.0f);
    const uint dst_idx = (batch_idx * nk + src_head) * D + tid;
    if (is_q) {
        q_out[dst_idx] = static_cast<bfloat>(norm);
    } else if (is_k) {
        k_out[dst_idx] = static_cast<bfloat>(norm);
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

kernel void kiln_lm_head_argmax_chunks_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight_t [[buffer(1)]],
    device float* partial_scores [[buffer(2)]],
    device float* partial_indices [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant uint& vocab [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]
) {
    threadgroup float scores[256];
    threadgroup float indices[256];

    const uint col = group * 256 + tid;
    float score = -INFINITY;
    float index = 0.0f;
    if (col < vocab) {
        float acc = 0.0f;
        for (uint i = 0; i < hidden; ++i) {
            acc += static_cast<float>(x[i]) * static_cast<float>(weight_t[i * vocab + col]);
        }
        score = static_cast<float>(static_cast<bfloat>(acc));
        index = static_cast<float>(col);
    }
    scores[tid] = score;
    indices[tid] = index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_score = scores[tid + stride];
            const float other_index = indices[tid + stride];
            if (other_score > scores[tid] ||
                (other_score == scores[tid] && other_index < indices[tid])) {
                scores[tid] = other_score;
                indices[tid] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        partial_scores[group] = scores[0];
        partial_indices[group] = indices[0];
    }
}

kernel void kiln_lm_head_argmax_reduce_f32(
    device const float* partial_scores [[buffer(0)]],
    device const float* partial_indices [[buffer(1)]],
    device float* final_index [[buffer(2)]],
    constant uint& num_groups [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float scores[1024];
    threadgroup float indices[1024];

    float score = -INFINITY;
    float index = 0.0f;
    if (tid < num_groups) {
        score = partial_scores[tid];
        index = partial_indices[tid];
    }
    scores[tid] = score;
    indices[tid] = index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_score = scores[tid + stride];
            const float other_index = indices[tid + stride];
            if (other_score > scores[tid] ||
                (other_score == scores[tid] && other_index < indices[tid])) {
                scores[tid] = other_score;
                indices[tid] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        final_index[0] = indices[0];
    }
}
"#;

const METAL_MLP_GATE_UP_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_mlp_gate_up_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* gate_t [[buffer(1)]],
    device const bfloat* up_t [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant uint& intermediate [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint cols2 = (intermediate + 1) >> 1;
    const uint total = rows * cols2;
    if (gid >= total) {
        return;
    }

    const uint row = gid / cols2;
    const uint col0 = (gid - row * cols2) << 1;
    const uint col1 = col0 + 1;
    const bool has_col1 = col1 < intermediate;
    const uint x_base = row * hidden;
    float gate_acc0 = 0.0f;
    float up_acc0 = 0.0f;
    float gate_acc1 = 0.0f;
    float up_acc1 = 0.0f;
    for (uint i = 0; i < hidden; ++i) {
        const float xv = static_cast<float>(x[x_base + i]);
        const uint w_idx0 = i * intermediate + col0;
        gate_acc0 += xv * static_cast<float>(gate_t[w_idx0]);
        up_acc0 += xv * static_cast<float>(up_t[w_idx0]);
        if (has_col1) {
            const uint w_idx1 = w_idx0 + 1;
            gate_acc1 += xv * static_cast<float>(gate_t[w_idx1]);
            up_acc1 += xv * static_cast<float>(up_t[w_idx1]);
        }
    }

    const uint out_base = row * intermediate;
    const float gate_sigmoid0 = 1.0f / (1.0f + exp(-gate_acc0));
    out[out_base + col0] = static_cast<bfloat>((gate_acc0 * gate_sigmoid0) * up_acc0);
    if (has_col1) {
        const float gate_sigmoid1 = 1.0f / (1.0f + exp(-gate_acc1));
        out[out_base + col1] = static_cast<bfloat>((gate_acc1 * gate_sigmoid1) * up_acc1);
    }
}
"#;

const METAL_ATTN_GATE_SIGMOID_MUL_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_attn_gate_sigmoid_mul_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* gate [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& total [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) {
        return;
    }

    const float gate_sigmoid = 1.0f / (1.0f + exp(-static_cast<float>(gate[gid])));
    out[gid] = static_cast<bfloat>(
        static_cast<float>(x[gid]) * static_cast<float>(static_cast<bfloat>(gate_sigmoid))
    );
}
"#;

const METAL_TRANSPOSED_COOP_GEMV_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_transposed_coop_gemv4_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight_t [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& input_dim [[buffer(3)]],
    constant uint& output_dim [[buffer(4)]],
    uint tgroup [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint col_base = (tgroup * 4 + simd_group) * 4;
    if (col_base >= output_dim) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const bool full_tile = col_base + 3 < output_dim;
    const bool vector_load_safe = full_tile && (output_dim % 4 == 0);

    for (uint row = lane; row < input_dim; row += 32) {
        const float xv = static_cast<float>(x[row]);
        const uint weight_base = row * output_dim + col_base;
        if (vector_load_safe) {
            device const bfloat4* w4_ptr = (device const bfloat4*)(weight_t + weight_base);
            const bfloat4 w = *w4_ptr;
            acc0 += xv * static_cast<float>(w[0]);
            acc1 += xv * static_cast<float>(w[1]);
            acc2 += xv * static_cast<float>(w[2]);
            acc3 += xv * static_cast<float>(w[3]);
        } else {
            acc0 += xv * static_cast<float>(weight_t[weight_base + 0]);
            if (col_base + 1 < output_dim) {
                acc1 += xv * static_cast<float>(weight_t[weight_base + 1]);
            }
            if (col_base + 2 < output_dim) {
                acc2 += xv * static_cast<float>(weight_t[weight_base + 2]);
            }
            if (col_base + 3 < output_dim) {
                acc3 += xv * static_cast<float>(weight_t[weight_base + 3]);
            }
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    if (lane == 0) {
        out[col_base + 0] = static_cast<bfloat>(acc0);
        if (col_base + 1 < output_dim) {
            out[col_base + 1] = static_cast<bfloat>(acc1);
        }
        if (col_base + 2 < output_dim) {
            out[col_base + 2] = static_cast<bfloat>(acc2);
        }
        if (col_base + 3 < output_dim) {
            out[col_base + 3] = static_cast<bfloat>(acc3);
        }
    }
}

kernel void kiln_transposed_coop_gemv8_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight_t [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& input_dim [[buffer(3)]],
    constant uint& output_dim [[buffer(4)]],
    uint tgroup [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint col_base = (tgroup * 4 + simd_group) * 8;
    if (col_base >= output_dim) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    const bool full_tile = col_base + 7 < output_dim;
    const bool vector_load_safe = full_tile && (output_dim % 4 == 0);

    for (uint row = lane; row < input_dim; row += 32) {
        const float xv = static_cast<float>(x[row]);
        const uint weight_base = row * output_dim + col_base;
        if (vector_load_safe) {
            device const bfloat4* w4_ptr = (device const bfloat4*)(weight_t + weight_base);
            const bfloat4 w0 = w4_ptr[0];
            const bfloat4 w1 = w4_ptr[1];
            acc0 += xv * static_cast<float>(w0[0]);
            acc1 += xv * static_cast<float>(w0[1]);
            acc2 += xv * static_cast<float>(w0[2]);
            acc3 += xv * static_cast<float>(w0[3]);
            acc4 += xv * static_cast<float>(w1[0]);
            acc5 += xv * static_cast<float>(w1[1]);
            acc6 += xv * static_cast<float>(w1[2]);
            acc7 += xv * static_cast<float>(w1[3]);
        } else {
            acc0 += xv * static_cast<float>(weight_t[weight_base + 0]);
            if (col_base + 1 < output_dim) {
                acc1 += xv * static_cast<float>(weight_t[weight_base + 1]);
            }
            if (col_base + 2 < output_dim) {
                acc2 += xv * static_cast<float>(weight_t[weight_base + 2]);
            }
            if (col_base + 3 < output_dim) {
                acc3 += xv * static_cast<float>(weight_t[weight_base + 3]);
            }
            if (col_base + 4 < output_dim) {
                acc4 += xv * static_cast<float>(weight_t[weight_base + 4]);
            }
            if (col_base + 5 < output_dim) {
                acc5 += xv * static_cast<float>(weight_t[weight_base + 5]);
            }
            if (col_base + 6 < output_dim) {
                acc6 += xv * static_cast<float>(weight_t[weight_base + 6]);
            }
            if (col_base + 7 < output_dim) {
                acc7 += xv * static_cast<float>(weight_t[weight_base + 7]);
            }
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4);
    acc5 = simd_sum(acc5);
    acc6 = simd_sum(acc6);
    acc7 = simd_sum(acc7);

    if (lane == 0) {
        out[col_base + 0] = static_cast<bfloat>(acc0);
        if (col_base + 1 < output_dim) {
            out[col_base + 1] = static_cast<bfloat>(acc1);
        }
        if (col_base + 2 < output_dim) {
            out[col_base + 2] = static_cast<bfloat>(acc2);
        }
        if (col_base + 3 < output_dim) {
            out[col_base + 3] = static_cast<bfloat>(acc3);
        }
        if (col_base + 4 < output_dim) {
            out[col_base + 4] = static_cast<bfloat>(acc4);
        }
        if (col_base + 5 < output_dim) {
            out[col_base + 5] = static_cast<bfloat>(acc5);
        }
        if (col_base + 6 < output_dim) {
            out[col_base + 6] = static_cast<bfloat>(acc6);
        }
        if (col_base + 7 < output_dim) {
            out[col_base + 7] = static_cast<bfloat>(acc7);
        }
    }
}

kernel void kiln_transposed_coop_gemv8_batch_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight_t [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& input_dim [[buffer(3)]],
    constant uint& output_dim [[buffer(4)]],
    uint2 tgroup [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint batch_idx = tgroup.y;
    const uint col_base = (tgroup.x * 4 + simd_group) * 8;
    if (col_base >= output_dim) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    const bool full_tile = col_base + 7 < output_dim;
    const bool vector_load_safe = full_tile && (output_dim % 4 == 0);
    const uint x_base = batch_idx * input_dim;

    for (uint row = lane; row < input_dim; row += 32) {
        const float xv = static_cast<float>(x[x_base + row]);
        const uint weight_base = row * output_dim + col_base;
        if (vector_load_safe) {
            device const bfloat4* w4_ptr = (device const bfloat4*)(weight_t + weight_base);
            const bfloat4 w0 = w4_ptr[0];
            const bfloat4 w1 = w4_ptr[1];
            acc0 += xv * static_cast<float>(w0[0]);
            acc1 += xv * static_cast<float>(w0[1]);
            acc2 += xv * static_cast<float>(w0[2]);
            acc3 += xv * static_cast<float>(w0[3]);
            acc4 += xv * static_cast<float>(w1[0]);
            acc5 += xv * static_cast<float>(w1[1]);
            acc6 += xv * static_cast<float>(w1[2]);
            acc7 += xv * static_cast<float>(w1[3]);
        } else {
            acc0 += xv * static_cast<float>(weight_t[weight_base + 0]);
            if (col_base + 1 < output_dim) {
                acc1 += xv * static_cast<float>(weight_t[weight_base + 1]);
            }
            if (col_base + 2 < output_dim) {
                acc2 += xv * static_cast<float>(weight_t[weight_base + 2]);
            }
            if (col_base + 3 < output_dim) {
                acc3 += xv * static_cast<float>(weight_t[weight_base + 3]);
            }
            if (col_base + 4 < output_dim) {
                acc4 += xv * static_cast<float>(weight_t[weight_base + 4]);
            }
            if (col_base + 5 < output_dim) {
                acc5 += xv * static_cast<float>(weight_t[weight_base + 5]);
            }
            if (col_base + 6 < output_dim) {
                acc6 += xv * static_cast<float>(weight_t[weight_base + 6]);
            }
            if (col_base + 7 < output_dim) {
                acc7 += xv * static_cast<float>(weight_t[weight_base + 7]);
            }
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4);
    acc5 = simd_sum(acc5);
    acc6 = simd_sum(acc6);
    acc7 = simd_sum(acc7);

    if (lane == 0) {
        const uint out_base = batch_idx * output_dim;
        out[out_base + col_base + 0] = static_cast<bfloat>(acc0);
        if (col_base + 1 < output_dim) {
            out[out_base + col_base + 1] = static_cast<bfloat>(acc1);
        }
        if (col_base + 2 < output_dim) {
            out[out_base + col_base + 2] = static_cast<bfloat>(acc2);
        }
        if (col_base + 3 < output_dim) {
            out[out_base + col_base + 3] = static_cast<bfloat>(acc3);
        }
        if (col_base + 4 < output_dim) {
            out[out_base + col_base + 4] = static_cast<bfloat>(acc4);
        }
        if (col_base + 5 < output_dim) {
            out[out_base + col_base + 5] = static_cast<bfloat>(acc5);
        }
        if (col_base + 6 < output_dim) {
            out[out_base + col_base + 6] = static_cast<bfloat>(acc6);
        }
        if (col_base + 7 < output_dim) {
            out[out_base + col_base + 7] = static_cast<bfloat>(acc7);
        }
    }
}
"#;

const METAL_FUSED_QKV_TRANSPOSED_COOP_GEMV_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_fused_qkv_transposed_coop_gemv8_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* q_t [[buffer(1)]],
    device const bfloat* k_t [[buffer(2)]],
    device const bfloat* v_t [[buffer(3)]],
    device bfloat* q_out [[buffer(4)]],
    device bfloat* k_out [[buffer(5)]],
    device bfloat* v_out [[buffer(6)]],
    constant uint& input_dim [[buffer(7)]],
    constant uint& q_output_dim [[buffer(8)]],
    constant uint& k_output_dim [[buffer(9)]],
    constant uint& v_output_dim [[buffer(10)]],
    uint2 tgroup [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint TILE_COLS = 8;
    constexpr uint SIMD_GROUPS = 4;
    constexpr uint COLS_PER_TGROUP = TILE_COLS * SIMD_GROUPS;

    const uint q_groups = (q_output_dim + COLS_PER_TGROUP - 1) / COLS_PER_TGROUP;
    const uint k_groups = (k_output_dim + COLS_PER_TGROUP - 1) / COLS_PER_TGROUP;
    const uint v_groups = (v_output_dim + COLS_PER_TGROUP - 1) / COLS_PER_TGROUP;
    const uint group = tgroup.x;

    device const bfloat* weight_t = q_t;
    device bfloat* out = q_out;
    uint output_dim = q_output_dim;
    uint group_in_proj = group;
    if (group < q_groups) {
        weight_t = q_t;
        out = q_out;
        output_dim = q_output_dim;
    } else if (group < q_groups + k_groups) {
        weight_t = k_t;
        out = k_out;
        output_dim = k_output_dim;
        group_in_proj = group - q_groups;
    } else if (group < q_groups + k_groups + v_groups) {
        weight_t = v_t;
        out = v_out;
        output_dim = v_output_dim;
        group_in_proj = group - q_groups - k_groups;
    } else {
        return;
    }

    const uint col_base = group_in_proj * COLS_PER_TGROUP + simd_group * TILE_COLS;
    if (col_base >= output_dim) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    const bool full_tile = col_base + 7 < output_dim;
    const bool vector_load_safe = full_tile && (output_dim % 4 == 0);

    for (uint row = lane; row < input_dim; row += 32) {
        const float xv = static_cast<float>(x[row]);
        const uint weight_base = row * output_dim + col_base;
        if (vector_load_safe) {
            device const bfloat4* w4_ptr = (device const bfloat4*)(weight_t + weight_base);
            const bfloat4 w0 = w4_ptr[0];
            const bfloat4 w1 = w4_ptr[1];
            acc0 += xv * static_cast<float>(w0[0]);
            acc1 += xv * static_cast<float>(w0[1]);
            acc2 += xv * static_cast<float>(w0[2]);
            acc3 += xv * static_cast<float>(w0[3]);
            acc4 += xv * static_cast<float>(w1[0]);
            acc5 += xv * static_cast<float>(w1[1]);
            acc6 += xv * static_cast<float>(w1[2]);
            acc7 += xv * static_cast<float>(w1[3]);
        } else {
            acc0 += xv * static_cast<float>(weight_t[weight_base + 0]);
            if (col_base + 1 < output_dim) {
                acc1 += xv * static_cast<float>(weight_t[weight_base + 1]);
            }
            if (col_base + 2 < output_dim) {
                acc2 += xv * static_cast<float>(weight_t[weight_base + 2]);
            }
            if (col_base + 3 < output_dim) {
                acc3 += xv * static_cast<float>(weight_t[weight_base + 3]);
            }
            if (col_base + 4 < output_dim) {
                acc4 += xv * static_cast<float>(weight_t[weight_base + 4]);
            }
            if (col_base + 5 < output_dim) {
                acc5 += xv * static_cast<float>(weight_t[weight_base + 5]);
            }
            if (col_base + 6 < output_dim) {
                acc6 += xv * static_cast<float>(weight_t[weight_base + 6]);
            }
            if (col_base + 7 < output_dim) {
                acc7 += xv * static_cast<float>(weight_t[weight_base + 7]);
            }
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4);
    acc5 = simd_sum(acc5);
    acc6 = simd_sum(acc6);
    acc7 = simd_sum(acc7);

    if (lane == 0) {
        out[col_base + 0] = static_cast<bfloat>(acc0);
        if (col_base + 1 < output_dim) {
            out[col_base + 1] = static_cast<bfloat>(acc1);
        }
        if (col_base + 2 < output_dim) {
            out[col_base + 2] = static_cast<bfloat>(acc2);
        }
        if (col_base + 3 < output_dim) {
            out[col_base + 3] = static_cast<bfloat>(acc3);
        }
        if (col_base + 4 < output_dim) {
            out[col_base + 4] = static_cast<bfloat>(acc4);
        }
        if (col_base + 5 < output_dim) {
            out[col_base + 5] = static_cast<bfloat>(acc5);
        }
        if (col_base + 6 < output_dim) {
            out[col_base + 6] = static_cast<bfloat>(acc6);
        }
        if (col_base + 7 < output_dim) {
            out[col_base + 7] = static_cast<bfloat>(acc7);
        }
    }
}
"#;

const METAL_GDN_IN_PROJ_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gdn_in_proj_decode_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* qkv_t [[buffer(1)]],
    device const bfloat* z_t [[buffer(2)]],
    device const bfloat* a_t [[buffer(3)]],
    device const bfloat* b_t [[buffer(4)]],
    device bfloat* qkv_out [[buffer(5)]],
    device bfloat* z_out [[buffer(6)]],
    device bfloat* a_out [[buffer(7)]],
    device bfloat* b_out [[buffer(8)]],
    constant uint& hidden [[buffer(9)]],
    constant uint& qkv_dim [[buffer(10)]],
    constant uint& z_dim [[buffer(11)]],
    constant uint& nv [[buffer(12)]],
    constant uint& batch [[buffer(13)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = qkv_dim + z_dim + (nv * 2);
    if (gid >= total * batch) {
        return;
    }
    const uint batch_idx = gid / total;
    const uint local_gid = gid - batch_idx * total;
    const uint x_base = batch_idx * hidden;

    float acc = 0.0f;
    if (local_gid < qkv_dim) {
        const uint col = local_gid;
        for (uint i = 0; i < hidden; ++i) {
            acc += static_cast<float>(x[x_base + i]) * static_cast<float>(qkv_t[i * qkv_dim + col]);
        }
        qkv_out[batch_idx * qkv_dim + col] = static_cast<bfloat>(acc);
    } else if (local_gid < qkv_dim + z_dim) {
        const uint col = local_gid - qkv_dim;
        for (uint i = 0; i < hidden; ++i) {
            acc += static_cast<float>(x[x_base + i]) * static_cast<float>(z_t[i * z_dim + col]);
        }
        z_out[batch_idx * z_dim + col] = static_cast<bfloat>(acc);
    } else if (local_gid < qkv_dim + z_dim + nv) {
        const uint col = local_gid - qkv_dim - z_dim;
        for (uint i = 0; i < hidden; ++i) {
            acc += static_cast<float>(x[x_base + i]) * static_cast<float>(a_t[i * nv + col]);
        }
        a_out[batch_idx * nv + col] = static_cast<bfloat>(acc);
    } else {
        const uint col = local_gid - qkv_dim - z_dim - nv;
        for (uint i = 0; i < hidden; ++i) {
            acc += static_cast<float>(x[x_base + i]) * static_cast<float>(b_t[i * nv + col]);
        }
        b_out[batch_idx * nv + col] = static_cast<bfloat>(acc);
    }
}
"#;

const METAL_PAGED_KV_HEAD_MAJOR_READ_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_paged_kv_head_major_read_bf16(
    device const bfloat* k_pool [[buffer(0)]],
    device const bfloat* v_pool [[buffer(1)]],
    device bfloat* k_out [[buffer(2)]],
    device bfloat* v_out [[buffer(3)]],
    constant uint& start_slot [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = seq_len * heads * head_dim;
    if (gid >= total) {
        return;
    }

    const uint d = gid % head_dim;
    const uint h = (gid / head_dim) % heads;
    const uint t = gid / (head_dim * heads);
    const uint pool_idx = ((start_slot + t) * heads + h) * head_dim + d;
    const uint out_idx = (h * seq_len + t) * head_dim + d;

    k_out[out_idx] = k_pool[pool_idx];
    v_out[out_idx] = v_pool[pool_idx];
}
"#;

const METAL_PAGED_KV_HEAD_MAJOR_READ_APPEND_TOKEN_MAJOR_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_paged_kv_head_major_read_append_token_major_bf16(
    device const bfloat* k_pool [[buffer(0)]],
    device const bfloat* v_pool [[buffer(1)]],
    device const bfloat* k_tail [[buffer(2)]],
    device const bfloat* v_tail [[buffer(3)]],
    device bfloat* k_out [[buffer(4)]],
    device bfloat* v_out [[buffer(5)]],
    constant uint& start_slot [[buffer(6)]],
    constant uint& prefix_len [[buffer(7)]],
    constant uint& tail_len [[buffer(8)]],
    constant uint& heads [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_len = prefix_len + tail_len;
    const uint total = total_len * heads * head_dim;
    if (gid >= total) {
        return;
    }

    const uint d = gid % head_dim;
    const uint h = (gid / head_dim) % heads;
    const uint t = gid / (head_dim * heads);
    const uint out_idx = (h * total_len + t) * head_dim + d;

    if (t < prefix_len) {
        const uint pool_idx = ((start_slot + t) * heads + h) * head_dim + d;
        k_out[out_idx] = k_pool[pool_idx];
        v_out[out_idx] = v_pool[pool_idx];
    } else {
        const uint tail_t = t - prefix_len;
        const uint tail_idx = (tail_t * heads + h) * head_dim + d;
        k_out[out_idx] = k_tail[tail_idx];
        v_out[out_idx] = v_tail[tail_idx];
    }
}
"#;

const METAL_PAGED_ATTN_DECODE_CONTIGUOUS_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_paged_attn_decode_contiguous_bf16_d256(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k_pool [[buffer(1)]],
    device const bfloat* v_pool [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant uint& start_slot [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& q_heads [[buffer(6)]],
    constant uint& kv_heads [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    constexpr uint D = 256;
    constexpr uint BN = 32;
    constexpr uint BD = 32;
    constexpr uint EPT = D / BD;
    constexpr uint QWEN_HEADS_PER_KV = 4;

    const uint head_idx = tid.y;
    if (head_idx >= q_heads) {
        return;
    }

    const uint kv_head_idx = head_idx / QWEN_HEADS_PER_KV;
    if (kv_head_idx >= kv_heads) {
        return;
    }

    thread float q_frag[EPT];
    thread float k_frag[EPT];
    thread float o_frag[EPT];
    threadgroup float outputs[BN * BD];
    threadgroup float max_scores[BN];
    threadgroup float sum_exp_scores[BN];

    device const bfloat* q_ptr = q + head_idx * D + simd_lid * EPT;
    device const bfloat* k_ptr =
        k_pool + ((start_slot + simd_gid) * kv_heads + kv_head_idx) * D + simd_lid * EPT;
    device const bfloat* v_ptr =
        v_pool + ((start_slot + simd_gid) * kv_heads + kv_head_idx) * D + simd_lid * EPT;
    device bfloat* out_ptr = out + head_idx * D + simd_gid * EPT;

    for (uint i = 0; i < EPT; ++i) {
        q_frag[i] = scale * static_cast<float>(q_ptr[i]);
        o_frag[i] = 0.0f;
    }

    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;

    for (uint t = simd_gid; t < seq_len; t += BN) {
        for (uint i = 0; i < EPT; ++i) {
            k_frag[i] = static_cast<float>(k_ptr[i]);
        }

        float score = 0.0f;
        for (uint i = 0; i < EPT; ++i) {
            score += q_frag[i] * k_frag[i];
        }
        score = simd_sum(score);

        const float new_max = max(max_score, score);
        const float factor = fast::exp(max_score - new_max);
        const float exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (uint i = 0; i < EPT; ++i) {
            o_frag[i] = o_frag[i] * factor + exp_score * static_cast<float>(v_ptr[i]);
        }

        k_ptr += BN * kv_heads * D;
        v_ptr += BN * kv_heads * D;
    }

    if (simd_lid == 0) {
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float partial_max = max_scores[simd_lid];
    const float global_max = simd_max(partial_max);
    const float partial_factor = fast::exp(partial_max - global_max);
    const float denom = simd_sum(sum_exp_scores[simd_lid] * partial_factor);

    for (uint i = 0; i < EPT; ++i) {
        outputs[simd_lid * BD + simd_gid] = o_frag[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o_frag[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * partial_factor) / denom;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (uint i = 0; i < EPT; ++i) {
            out_ptr[i] = static_cast<bfloat>(o_frag[i]);
        }
    }
}

kernel void kiln_paged_attn_decode_contiguous_batch_bf16_d256(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k_pool [[buffer(1)]],
    device const bfloat* v_pool [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    device const uint* start_slots [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    constant uint& q_heads [[buffer(7)]],
    constant uint& kv_heads [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant uint& total_slots [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    constexpr uint D = 256;
    constexpr uint BN = 32;
    constexpr uint BD = 32;
    constexpr uint EPT = D / BD;
    constexpr uint QWEN_HEADS_PER_KV = 4;

    const uint batch_idx = tid.x;
    const uint head_idx = tid.y;
    if (batch_idx >= batch || head_idx >= q_heads) {
        return;
    }

    const uint kv_head_idx = head_idx / QWEN_HEADS_PER_KV;
    if (kv_head_idx >= kv_heads) {
        return;
    }

    device bfloat* out_ptr =
        out + (batch_idx * q_heads + head_idx) * D + simd_gid * EPT;
    const uint start_slot = start_slots[batch_idx];
    if (start_slot >= total_slots || seq_len == 0 || seq_len > total_slots - start_slot) {
        if (simd_lid == 0) {
            for (uint i = 0; i < EPT; ++i) {
                out_ptr[i] = static_cast<bfloat>(0.0f);
            }
        }
        return;
    }

    thread float q_frag[EPT];
    thread float k_frag[EPT];
    thread float o_frag[EPT];
    threadgroup float outputs[BN * BD];
    threadgroup float max_scores[BN];
    threadgroup float sum_exp_scores[BN];

    device const bfloat* q_ptr =
        q + (batch_idx * q_heads + head_idx) * D + simd_lid * EPT;
    device const bfloat* k_ptr =
        k_pool + ((start_slot + simd_gid) * kv_heads + kv_head_idx) * D + simd_lid * EPT;
    device const bfloat* v_ptr =
        v_pool + ((start_slot + simd_gid) * kv_heads + kv_head_idx) * D + simd_lid * EPT;

    for (uint i = 0; i < EPT; ++i) {
        q_frag[i] = scale * static_cast<float>(q_ptr[i]);
        o_frag[i] = 0.0f;
    }

    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;

    for (uint t = simd_gid; t < seq_len; t += BN) {
        for (uint i = 0; i < EPT; ++i) {
            k_frag[i] = static_cast<float>(k_ptr[i]);
        }

        float score = 0.0f;
        for (uint i = 0; i < EPT; ++i) {
            score += q_frag[i] * k_frag[i];
        }
        score = simd_sum(score);

        const float new_max = max(max_score, score);
        const float factor = fast::exp(max_score - new_max);
        const float exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (uint i = 0; i < EPT; ++i) {
            o_frag[i] = o_frag[i] * factor + exp_score * static_cast<float>(v_ptr[i]);
        }

        k_ptr += BN * kv_heads * D;
        v_ptr += BN * kv_heads * D;
    }

    if (simd_lid == 0) {
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float partial_max = max_scores[simd_lid];
    const float global_max = simd_max(partial_max);
    const float partial_factor = fast::exp(partial_max - global_max);
    const float denom = simd_sum(sum_exp_scores[simd_lid] * partial_factor);

    for (uint i = 0; i < EPT; ++i) {
        outputs[simd_lid * BD + simd_gid] = o_frag[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o_frag[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * partial_factor) / denom;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (uint i = 0; i < EPT; ++i) {
            out_ptr[i] = static_cast<bfloat>(o_frag[i]);
        }
    }
}
"#;

const METAL_PAGED_KV_WRITE_TOKEN_MAJOR_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_paged_kv_write_token_major_bf16(
    device const bfloat* k_src [[buffer(0)]],
    device const bfloat* v_src [[buffer(1)]],
    device bfloat* k_pool [[buffer(2)]],
    device bfloat* v_pool [[buffer(3)]],
    constant uint& slot [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = heads * head_dim;
    if (gid >= total) {
        return;
    }

    const uint pool_idx = slot * total + gid;
    k_pool[pool_idx] = k_src[gid];
    v_pool[pool_idx] = v_src[gid];
}

kernel void kiln_paged_kv_write_token_major_batch_bf16(
    device const bfloat* k_src [[buffer(0)]],
    device const bfloat* v_src [[buffer(1)]],
    device bfloat* k_pool [[buffer(2)]],
    device bfloat* v_pool [[buffer(3)]],
    device const uint* slots [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& total_slots [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint row_stride = heads * head_dim;
    const uint total = batch * row_stride;
    if (gid >= total) {
        return;
    }

    const uint batch_idx = gid / row_stride;
    const uint local = gid - batch_idx * row_stride;
    const uint slot = slots[batch_idx];
    if (slot >= total_slots) {
        return;
    }
    const uint pool_idx = slot * row_stride + local;
    k_pool[pool_idx] = k_src[gid];
    v_pool[pool_idx] = v_src[gid];
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
        METAL_ROTARY_QK_KERNEL,
        METAL_GDN_QK_NORM_KERNEL,
        METAL_GDN_DECODE_QKV_CONV_NORM_KERNEL,
        METAL_GDN_GATES_KERNEL,
        METAL_GDN_DECODE_GATES_RECURRENT_KERNEL,
        METAL_GDN_DECODE_GATES_RECURRENT_RMSNORM_KERNEL,
        METAL_GATED_RMSNORM_KERNEL,
        METAL_GDN_RECURRENT_KERNEL,
        METAL_GDN_RECURRENT_PREFILL_HEAD_LAST_KERNEL,
        METAL_GDN_FULL_CHUNK_FORWARD_KERNEL,
        METAL_CONV1D_PREFILL_KERNEL,
        METAL_CONV1D_UPDATE_KERNEL,
        METAL_LM_HEAD_KERNEL,
        METAL_MLP_GATE_UP_KERNEL,
        METAL_ATTN_GATE_SIGMOID_MUL_KERNEL,
        METAL_TRANSPOSED_COOP_GEMV_KERNEL,
        METAL_FUSED_QKV_TRANSPOSED_COOP_GEMV_KERNEL,
        METAL_GDN_IN_PROJ_KERNEL,
        METAL_PAGED_KV_HEAD_MAJOR_READ_KERNEL,
        METAL_PAGED_KV_HEAD_MAJOR_READ_APPEND_TOKEN_MAJOR_KERNEL,
        METAL_PAGED_ATTN_DECODE_CONTIGUOUS_KERNEL,
        METAL_PAGED_KV_WRITE_TOKEN_MAJOR_KERNEL,
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

fn metal_rotary_qk_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal rotary qk pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_rotary_qk_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal rotary qk function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal rotary qk pipeline: {e:?}"))?;
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

fn metal_gdn_qk_norm_gqa_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal gdn qk norm gqa pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_qk_norm_gqa_f32_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn qk norm gqa function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn qk norm gqa pipeline: {e:?}"))?;
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gdn_decode_qkv_conv_norm_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal gdn decode qkv conv/norm pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_decode_qkv_conv_norm_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn decode qkv conv/norm function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn decode qkv conv/norm pipeline: {e:?}"))?;
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

fn metal_lm_head_argmax_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal lm head argmax pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_lm_head_argmax_chunks_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal lm head argmax function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal lm head argmax pipeline: {e:?}"))?;
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

fn metal_lm_head_argmax_reduce_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal lm head argmax reduce pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_lm_head_argmax_reduce_f32", None)
        .map_err(|e| anyhow::anyhow!("load metal lm head argmax reduce function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal lm head argmax reduce pipeline: {e:?}"))?;
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

fn metal_mlp_gate_up_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal mlp gate/up pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_mlp_gate_up_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal mlp gate/up function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal mlp gate/up pipeline: {e:?}"))?;
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

fn metal_attn_gate_sigmoid_mul_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal attn gate sigmoid/mul pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_attn_gate_sigmoid_mul_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal attn gate sigmoid/mul function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal attn gate sigmoid/mul pipeline: {e:?}"))?;
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

fn metal_transposed_coop_gemv_pipeline(
    device: &candle_core::metal_backend::MetalDevice,
    tile: MetalTransposedCoopGemvTile,
) -> Result<candle_metal_kernels::metal::ComputePipeline> {
    use candle_core::metal_backend::DeviceId;
    use candle_metal_kernels::metal::ComputePipeline;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    static PIPELINES: OnceLock<
        Mutex<HashMap<(DeviceId, MetalTransposedCoopGemvTile), ComputePipeline>>,
    > = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal transposed coop GEMV pipeline cache poisoned"))?;
    let key = (device.id(), tile);
    if let Some(pipeline) = cache.get(&key) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function(tile.function_name(), None)
        .map_err(|e| anyhow::anyhow!("load metal transposed coop GEMV function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal transposed coop GEMV pipeline: {e:?}"))?;
    cache.insert(key, pipeline.clone());
    Ok(pipeline)
}

fn metal_transposed_coop_gemv_batch_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal batch transposed coop GEMV cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_transposed_coop_gemv8_batch_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal batch transposed coop GEMV function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal batch transposed coop GEMV pipeline: {e:?}"))?;
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

fn metal_fused_qkv_transposed_coop_gemv_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal fused QKV projection pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_fused_qkv_transposed_coop_gemv8_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal fused QKV projection function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal fused QKV projection pipeline: {e:?}"))?;
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gdn_in_proj_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal gdn in-proj pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_in_proj_decode_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn in-proj function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn in-proj pipeline: {e:?}"))?;
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

fn metal_paged_kv_head_major_read_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal paged kv read pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_paged_kv_head_major_read_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal paged kv read function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal paged kv read pipeline: {e:?}"))?;
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

fn metal_paged_kv_head_major_read_append_token_major_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal paged kv read+append pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function(
            "kiln_paged_kv_head_major_read_append_token_major_bf16",
            None,
        )
        .map_err(|e| anyhow::anyhow!("load metal paged kv read+append function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal paged kv read+append pipeline: {e:?}"))?;
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

fn metal_paged_attn_decode_contiguous_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal contiguous paged decode attention cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_paged_attn_decode_contiguous_bf16_d256", None)
        .map_err(|e| anyhow::anyhow!("load metal contiguous paged decode attention: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal contiguous paged decode attention: {e:?}"))?;
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

fn metal_paged_attn_decode_contiguous_batch_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal contiguous paged batch decode pipeline poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_paged_attn_decode_contiguous_batch_bf16_d256", None)
        .map_err(|e| {
            anyhow::anyhow!("load metal contiguous paged batch decode attention: {e:?}")
        })?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| {
            anyhow::anyhow!("build metal contiguous paged batch decode attention: {e:?}")
        })?;
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

fn metal_paged_kv_write_token_major_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal paged kv write pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_paged_kv_write_token_major_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal paged kv write function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal paged kv write pipeline: {e:?}"))?;
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

fn metal_paged_kv_write_token_major_batch_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal paged kv batch write pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_paged_kv_write_token_major_batch_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal paged kv batch write function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal paged kv batch write pipeline: {e:?}"))?;
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

pub(crate) fn metal_lm_head_argmax_supports(x: &Tensor, weight_t: &Tensor) -> bool {
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

pub(crate) fn metal_lm_head_bf16(x: &Tensor, weight_t: &Tensor) -> Result<Tensor> {
    anyhow::ensure!(
        metal_lm_head_supports(x, weight_t),
        "metal lm head supports only BF16 [1,1,H] x [H,V] on Metal"
    );
    let (_, _, hidden) = x.dims3()?;
    let (_, vocab) = weight_t.dims2()?;

    // The kernel writes every vocab element.
    let out = unsafe { Tensor::empty((1usize, 1usize, vocab), DType::BF16, x.device())? };

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

pub(crate) fn metal_lm_head_argmax_bf16(x: &Tensor, weight_t: &Tensor) -> Result<u32> {
    anyhow::ensure!(
        metal_lm_head_argmax_supports(x, weight_t),
        "metal lm head argmax supports only BF16 [1,1,H] x [H,V] on Metal with <= 262144 vocab"
    );
    let (_, _, hidden) = x.dims3()?;
    let (_, vocab) = weight_t.dims2()?;

    let chunk_width = 256usize;
    let num_groups = vocab.div_ceil(chunk_width);
    let partial_scores = unsafe { Tensor::empty((num_groups,), DType::F32, x.device())? };
    let partial_indices = unsafe { Tensor::empty((num_groups,), DType::F32, x.device())? };
    let final_index = if metal_lm_head_argmax_gpu_reduce_disabled() {
        None
    } else {
        Some(unsafe { Tensor::empty((1usize,), DType::F32, x.device())? })
    };

    let Device::Metal(device) = x.device() else {
        anyhow::bail!("metal lm head argmax requires Metal tensors");
    };
    let pipeline = metal_lm_head_argmax_pipeline(device)?;
    let reduce_pipeline = if final_index.is_some() {
        Some(metal_lm_head_argmax_reduce_pipeline(device)?)
    } else {
        None
    };
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_lm_head_argmax_chunks_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (w_storage, w_layout) = weight_t.storage_and_layout();
        let (ps_storage, ps_layout) = partial_scores.storage_and_layout();
        let (pi_storage, pi_layout) = partial_indices.storage_and_layout();
        let final_storage_and_layout = final_index.as_ref().map(Tensor::storage_and_layout);

        let x_metal = match &*x_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal lm head argmax x must be on Metal"),
        };
        let w_metal = match &*w_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal lm head argmax weight must be on Metal"),
        };
        let ps_metal = match &*ps_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal lm head argmax partial scores must be on Metal"),
        };
        let pi_metal = match &*pi_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal lm head argmax partial indices must be on Metal"),
        };
        let final_metal = match final_storage_and_layout.as_ref().map(|(s, l)| (&**s, l)) {
            Some((candle_core::Storage::Metal(s), layout)) => Some((s, layout)),
            Some(_) => anyhow::bail!("metal lm head argmax final index must be on Metal"),
            None => None,
        };

        let x_buf = candle_core::metal_backend::buffer_o(x_metal.buffer(), &x_layout, x.dtype());
        let w_buf =
            candle_core::metal_backend::buffer_o(w_metal.buffer(), &w_layout, weight_t.dtype());
        let ps_buf = candle_core::metal_backend::buffer_o(
            ps_metal.buffer(),
            &ps_layout,
            partial_scores.dtype(),
        );
        let pi_buf = candle_core::metal_backend::buffer_o(
            pi_metal.buffer(),
            &pi_layout,
            partial_indices.dtype(),
        );
        let final_buf = final_metal.map(|(storage, layout)| {
            candle_core::metal_backend::buffer_o(storage.buffer(), layout, DType::F32)
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

pub(crate) fn metal_mlp_gate_up_supports(x: &Tensor, gate_t: &Tensor, up_t: &Tensor) -> bool {
    if x.dtype() != DType::BF16 || gate_t.dtype() != DType::BF16 || up_t.dtype() != DType::BF16 {
        return false;
    }
    if !matches!(x.device(), Device::Metal(_))
        || !matches!(gate_t.device(), Device::Metal(_))
        || !matches!(up_t.device(), Device::Metal(_))
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

pub(crate) fn metal_mlp_gate_up_bf16(x: &Tensor, gate_t: &Tensor, up_t: &Tensor) -> Result<Tensor> {
    anyhow::ensure!(
        metal_mlp_gate_up_supports(x, gate_t, up_t),
        "metal mlp gate/up supports only BF16 [B,1,H] x [H,I] on Metal"
    );
    let (batch, seq_len, hidden) = x.dims3()?;
    let (_, intermediate) = gate_t.dims2()?;
    let rows = batch * seq_len;
    let total = rows * intermediate.div_ceil(2);

    // The kernel writes every row/intermediate element.
    let out = unsafe { Tensor::empty((batch, seq_len, intermediate), DType::BF16, x.device())? };

    let Device::Metal(device) = x.device() else {
        anyhow::bail!("metal mlp gate/up requires Metal tensors");
    };
    let pipeline = metal_mlp_gate_up_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_mlp_gate_up_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (gate_storage, gate_layout) = gate_t.storage_and_layout();
        let (up_storage, up_layout) = up_t.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let x_metal = match &*x_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal mlp gate/up x must be on Metal"),
        };
        let gate_metal = match &*gate_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal mlp gate/up gate_t must be on Metal"),
        };
        let up_metal = match &*up_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal mlp gate/up up_t must be on Metal"),
        };
        let out_metal = match &*out_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal mlp gate/up out must be on Metal"),
        };

        let x_buf = candle_core::metal_backend::buffer_o(x_metal.buffer(), &x_layout, x.dtype());
        let gate_buf =
            candle_core::metal_backend::buffer_o(gate_metal.buffer(), &gate_layout, gate_t.dtype());
        let up_buf =
            candle_core::metal_backend::buffer_o(up_metal.buffer(), &up_layout, up_t.dtype());
        let out_buf =
            candle_core::metal_backend::buffer_o(out_metal.buffer(), &out_layout, out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(gate_buf.buffer), gate_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(up_buf.buffer), up_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let rows_u32 = rows as u32;
        let hidden_u32 = hidden as u32;
        let intermediate_u32 = intermediate as u32;
        encoder.set_bytes(4, &rows_u32);
        encoder.set_bytes(5, &hidden_u32);
        encoder.set_bytes(6, &intermediate_u32);

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

pub(crate) fn metal_attn_gate_sigmoid_mul_supports(x: &Tensor, gate: &Tensor) -> bool {
    if metal_attn_gate_fusion_disabled() {
        return false;
    }
    if x.dtype() != DType::BF16 || gate.dtype() != DType::BF16 {
        return false;
    }
    if !matches!(x.device(), Device::Metal(_)) || !matches!(gate.device(), Device::Metal(_)) {
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

pub(crate) fn metal_attn_gate_sigmoid_mul_bf16(x: &Tensor, gate: &Tensor) -> Result<Tensor> {
    anyhow::ensure!(
        metal_attn_gate_sigmoid_mul_supports(x, gate),
        "metal attn gate sigmoid/mul supports only BF16 [B,1,H] tensors on Metal"
    );
    let (batch, seq_len, hidden) = x.dims3()?;
    let total = batch * seq_len * hidden;

    // The kernel writes every hidden element exactly once.
    let out = unsafe { Tensor::empty((batch, seq_len, hidden), DType::BF16, x.device())? };

    let Device::Metal(device) = x.device() else {
        anyhow::bail!("metal attn gate sigmoid/mul requires Metal tensors");
    };
    let pipeline = metal_attn_gate_sigmoid_mul_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_attn_gate_sigmoid_mul_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (gate_storage, gate_layout) = gate.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let x_metal = match &*x_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal attn gate sigmoid/mul x must be on Metal"),
        };
        let gate_metal = match &*gate_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal attn gate sigmoid/mul gate must be on Metal"),
        };
        let out_metal = match &*out_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal attn gate sigmoid/mul out must be on Metal"),
        };

        let x_buf = candle_core::metal_backend::buffer_o(x_metal.buffer(), &x_layout, x.dtype());
        let gate_buf =
            candle_core::metal_backend::buffer_o(gate_metal.buffer(), &gate_layout, gate.dtype());
        let out_buf =
            candle_core::metal_backend::buffer_o(out_metal.buffer(), &out_layout, out.dtype());

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

pub(crate) fn metal_transposed_coop_gemv_supports(x: &Tensor, weight_t: &Tensor) -> bool {
    if metal_transposed_coop_gemv_disabled() {
        return false;
    }
    if x.dtype() != DType::BF16 || weight_t.dtype() != DType::BF16 {
        return false;
    }
    if !matches!(x.device(), Device::Metal(_)) || !matches!(weight_t.device(), Device::Metal(_)) {
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
    x: &Tensor,
    weight_t: &Tensor,
) -> bool {
    if metal_transposed_coop_gemv_disabled() {
        return false;
    }
    if x.dtype() != DType::BF16 || weight_t.dtype() != DType::BF16 {
        return false;
    }
    if !matches!(x.device(), Device::Metal(_)) || !matches!(weight_t.device(), Device::Metal(_)) {
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

pub(crate) fn metal_transposed_coop_gemv_bf16(x: &Tensor, weight_t: &Tensor) -> Result<Tensor> {
    if metal_transposed_coop_gemv_decode_batch_supports(x, weight_t) {
        return metal_transposed_coop_gemv_batch_bf16(x, weight_t);
    }

    metal_transposed_coop_gemv_bf16_with_tile(
        x,
        weight_t,
        metal_transposed_coop_gemv_default_tile(),
    )
}

fn metal_transposed_coop_gemv_bf16_with_tile(
    x: &Tensor,
    weight_t: &Tensor,
    tile: MetalTransposedCoopGemvTile,
) -> Result<Tensor> {
    anyhow::ensure!(
        metal_transposed_coop_gemv_supports(x, weight_t),
        "metal transposed coop GEMV supports only BF16 [1,1,K] x [K,N] on Metal"
    );
    let (_, _, input_dim) = x.dims3()?;
    let (_, output_dim) = weight_t.dims2()?;

    // The kernel writes every output channel exactly once.
    let out = unsafe { Tensor::empty((1usize, 1usize, output_dim), DType::BF16, x.device())? };

    let Device::Metal(device) = x.device() else {
        anyhow::bail!("metal transposed coop GEMV requires Metal tensors");
    };
    let pipeline = metal_transposed_coop_gemv_pipeline(device, tile)?;
    let encoder = device.command_encoder()?;
    encoder.set_label(tile.label());
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (w_storage, w_layout) = weight_t.storage_and_layout();
        let (o_storage, o_layout) = out.storage_and_layout();

        let x_metal = match &*x_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal transposed coop GEMV x must be on Metal"),
        };
        let w_metal = match &*w_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal transposed coop GEMV weight_t must be on Metal"),
        };
        let out_metal = match &*o_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal transposed coop GEMV out must be on Metal"),
        };

        let x_buf = candle_core::metal_backend::buffer_o(x_metal.buffer(), &x_layout, x.dtype());
        let w_buf =
            candle_core::metal_backend::buffer_o(w_metal.buffer(), &w_layout, weight_t.dtype());
        let out_buf =
            candle_core::metal_backend::buffer_o(out_metal.buffer(), &o_layout, out.dtype());

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

fn metal_transposed_coop_gemv_batch_bf16(x: &Tensor, weight_t: &Tensor) -> Result<Tensor> {
    anyhow::ensure!(
        metal_transposed_coop_gemv_decode_batch_supports(x, weight_t),
        "metal batch transposed coop GEMV supports only BF16 [B,1,K] x [K,N] with B > 1 on Metal"
    );
    let (batch, _, input_dim) = x.dims3()?;
    let (_, output_dim) = weight_t.dims2()?;

    // The kernel writes every batch/output channel exactly once.
    let out = unsafe { Tensor::empty((batch, 1usize, output_dim), DType::BF16, x.device())? };

    let Device::Metal(device) = x.device() else {
        anyhow::bail!("metal batch transposed coop GEMV requires Metal tensors");
    };
    let pipeline = metal_transposed_coop_gemv_batch_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_transposed_coop_gemv8_batch_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (w_storage, w_layout) = weight_t.storage_and_layout();
        let (o_storage, o_layout) = out.storage_and_layout();

        let x_metal = match &*x_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal batch transposed coop GEMV x must be on Metal"),
        };
        let w_metal = match &*w_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal batch transposed coop GEMV weight_t must be on Metal"),
        };
        let out_metal = match &*o_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal batch transposed coop GEMV out must be on Metal"),
        };

        let x_buf = candle_core::metal_backend::buffer_o(x_metal.buffer(), &x_layout, x.dtype());
        let w_buf =
            candle_core::metal_backend::buffer_o(w_metal.buffer(), &w_layout, weight_t.dtype());
        let out_buf =
            candle_core::metal_backend::buffer_o(out_metal.buffer(), &o_layout, out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let input_dim_u32 = input_dim as u32;
        let output_dim_u32 = output_dim as u32;
        encoder.set_bytes(3, &input_dim_u32);
        encoder.set_bytes(4, &output_dim_u32);

        let cols_per_threadgroup =
            METAL_TRANSPOSED_COOP_GEMV_TILE8_COLS * METAL_TRANSPOSED_COOP_GEMV_SIMDGROUPS;
        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: output_dim.div_ceil(cols_per_threadgroup),
            height: batch,
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
    x: &Tensor,
    q_t: &Tensor,
    k_t: &Tensor,
    v_t: &Tensor,
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
    x: &Tensor,
    q_t: &Tensor,
    k_t: &Tensor,
    v_t: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
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
    let fused_out =
        unsafe { Tensor::empty((1usize, 1usize, total_output_dim), DType::BF16, x.device())? };
    let q_out = fused_out.narrow(2, 0, q_output_dim)?;
    let k_out = fused_out.narrow(2, q_output_dim, k_output_dim)?;
    let v_out = fused_out.narrow(2, q_output_dim + k_output_dim, v_output_dim)?;

    let Device::Metal(device) = x.device() else {
        anyhow::bail!("metal fused QKV projection requires Metal tensors");
    };
    let pipeline = metal_fused_qkv_transposed_coop_gemv_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_fused_qkv_transposed_coop_gemv8_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (q_storage, q_layout) = q_t.storage_and_layout();
        let (k_storage, k_layout) = k_t.storage_and_layout();
        let (v_storage, v_layout) = v_t.storage_and_layout();
        let (q_out_storage, q_out_layout) = q_out.storage_and_layout();
        let (k_out_storage, k_out_layout) = k_out.storage_and_layout();
        let (v_out_storage, v_out_layout) = v_out.storage_and_layout();

        let x_metal = match &*x_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal fused QKV projection x must be on Metal"),
        };
        let q_metal = match &*q_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal fused QKV projection q_t must be on Metal"),
        };
        let k_metal = match &*k_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal fused QKV projection k_t must be on Metal"),
        };
        let v_metal = match &*v_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal fused QKV projection v_t must be on Metal"),
        };
        let q_out_metal = match &*q_out_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal fused QKV projection q_out must be on Metal"),
        };
        let k_out_metal = match &*k_out_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal fused QKV projection k_out must be on Metal"),
        };
        let v_out_metal = match &*v_out_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal fused QKV projection v_out must be on Metal"),
        };

        let x_buf = candle_core::metal_backend::buffer_o(x_metal.buffer(), &x_layout, x.dtype());
        let q_buf = candle_core::metal_backend::buffer_o(q_metal.buffer(), &q_layout, q_t.dtype());
        let k_buf = candle_core::metal_backend::buffer_o(k_metal.buffer(), &k_layout, k_t.dtype());
        let v_buf = candle_core::metal_backend::buffer_o(v_metal.buffer(), &v_layout, v_t.dtype());
        let q_out_buf = candle_core::metal_backend::buffer_o(
            q_out_metal.buffer(),
            &q_out_layout,
            q_out.dtype(),
        );
        let k_out_buf = candle_core::metal_backend::buffer_o(
            k_out_metal.buffer(),
            &k_out_layout,
            k_out.dtype(),
        );
        let v_out_buf = candle_core::metal_backend::buffer_o(
            v_out_metal.buffer(),
            &v_out_layout,
            v_out.dtype(),
        );

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

fn metal_gdn_in_proj_decode_supports(
    x: &Tensor,
    qkv_t: &Tensor,
    z_t: &Tensor,
    a_t: &Tensor,
    b_t: &Tensor,
) -> bool {
    if x.dtype() != DType::BF16
        || qkv_t.dtype() != DType::BF16
        || z_t.dtype() != DType::BF16
        || a_t.dtype() != DType::BF16
        || b_t.dtype() != DType::BF16
    {
        return false;
    }
    if !matches!(x.device(), Device::Metal(_))
        || !matches!(qkv_t.device(), Device::Metal(_))
        || !matches!(z_t.device(), Device::Metal(_))
        || !matches!(a_t.device(), Device::Metal(_))
        || !matches!(b_t.device(), Device::Metal(_))
    {
        return false;
    }
    if !x.is_contiguous()
        || !qkv_t.is_contiguous()
        || !z_t.is_contiguous()
        || !a_t.is_contiguous()
        || !b_t.is_contiguous()
    {
        return false;
    }
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return false;
    };
    let Ok((qkv_hidden, qkv_dim)) = qkv_t.dims2() else {
        return false;
    };
    let Ok((z_hidden, z_dim)) = z_t.dims2() else {
        return false;
    };
    let Ok((a_hidden, nv)) = a_t.dims2() else {
        return false;
    };
    let Ok((b_hidden, b_nv)) = b_t.dims2() else {
        return false;
    };
    let Some(total) = qkv_dim
        .checked_add(z_dim)
        .and_then(|n| n.checked_add(nv))
        .and_then(|n| n.checked_add(b_nv))
    else {
        return false;
    };
    let Some(dispatch_total) = total.checked_mul(batch) else {
        return false;
    };

    batch > 0
        && seq_len == 1
        && hidden == qkv_hidden
        && hidden == z_hidden
        && hidden == a_hidden
        && hidden == b_hidden
        && nv == b_nv
        && hidden <= u32::MAX as usize
        && qkv_dim <= u32::MAX as usize
        && z_dim <= u32::MAX as usize
        && nv <= u32::MAX as usize
        && total <= u32::MAX as usize
        && dispatch_total <= u32::MAX as usize
}

fn metal_gdn_in_proj_decode_bf16(
    x: &Tensor,
    qkv_t: &Tensor,
    z_t: &Tensor,
    a_t: &Tensor,
    b_t: &Tensor,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    anyhow::ensure!(
        metal_gdn_in_proj_decode_supports(x, qkv_t, z_t, a_t, b_t),
        "metal gdn in-proj supports only BF16 [B,1,H] x [H,*] on Metal"
    );
    let (batch, _, hidden) = x.dims3()?;
    let (_, qkv_dim) = qkv_t.dims2()?;
    let (_, z_dim) = z_t.dims2()?;
    let (_, nv) = a_t.dims2()?;
    let total = qkv_dim + z_dim + (nv * 2);
    let dispatch_total = batch * total;

    // The kernel writes every output element exactly once. Keep bs=1 on one
    // backing allocation, but use separate batch outputs so each `[B,1,N]`
    // tensor remains contiguous for the following fused decode kernels.
    let (qkv_out, z_out, a_out, b_out) = if batch == 1 {
        let proj_out = unsafe { Tensor::empty((1usize, 1usize, total), DType::BF16, x.device())? };
        (
            proj_out.narrow(2, 0, qkv_dim)?,
            proj_out.narrow(2, qkv_dim, z_dim)?,
            proj_out.narrow(2, qkv_dim + z_dim, nv)?,
            proj_out.narrow(2, qkv_dim + z_dim + nv, nv)?,
        )
    } else {
        (
            unsafe { Tensor::empty((batch, 1usize, qkv_dim), DType::BF16, x.device())? },
            unsafe { Tensor::empty((batch, 1usize, z_dim), DType::BF16, x.device())? },
            unsafe { Tensor::empty((batch, 1usize, nv), DType::BF16, x.device())? },
            unsafe { Tensor::empty((batch, 1usize, nv), DType::BF16, x.device())? },
        )
    };

    let Device::Metal(device) = x.device() else {
        anyhow::bail!("metal gdn in-proj requires Metal tensors");
    };
    let pipeline = metal_gdn_in_proj_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_gdn_in_proj_decode_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (qkv_storage, qkv_layout) = qkv_t.storage_and_layout();
        let (z_storage, z_layout) = z_t.storage_and_layout();
        let (a_storage, a_layout) = a_t.storage_and_layout();
        let (b_storage, b_layout) = b_t.storage_and_layout();
        let (qkv_o_storage, qkv_o_layout) = qkv_out.storage_and_layout();
        let (z_o_storage, z_o_layout) = z_out.storage_and_layout();
        let (a_o_storage, a_o_layout) = a_out.storage_and_layout();
        let (b_o_storage, b_o_layout) = b_out.storage_and_layout();

        let x_metal = match &*x_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn in-proj x must be on Metal"),
        };
        let qkv_metal = match &*qkv_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn in-proj qkv_t must be on Metal"),
        };
        let z_metal = match &*z_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn in-proj z_t must be on Metal"),
        };
        let a_metal = match &*a_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn in-proj a_t must be on Metal"),
        };
        let b_metal = match &*b_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn in-proj b_t must be on Metal"),
        };
        let qkv_o_metal = match &*qkv_o_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn in-proj qkv_out must be on Metal"),
        };
        let z_o_metal = match &*z_o_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn in-proj z_out must be on Metal"),
        };
        let a_o_metal = match &*a_o_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn in-proj a_out must be on Metal"),
        };
        let b_o_metal = match &*b_o_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn in-proj b_out must be on Metal"),
        };

        let x_buf = candle_core::metal_backend::buffer_o(x_metal.buffer(), &x_layout, x.dtype());
        let qkv_buf =
            candle_core::metal_backend::buffer_o(qkv_metal.buffer(), &qkv_layout, qkv_t.dtype());
        let z_buf = candle_core::metal_backend::buffer_o(z_metal.buffer(), &z_layout, z_t.dtype());
        let a_buf = candle_core::metal_backend::buffer_o(a_metal.buffer(), &a_layout, a_t.dtype());
        let b_buf = candle_core::metal_backend::buffer_o(b_metal.buffer(), &b_layout, b_t.dtype());
        let qkv_o_buf = candle_core::metal_backend::buffer_o(
            qkv_o_metal.buffer(),
            &qkv_o_layout,
            qkv_out.dtype(),
        );
        let z_o_buf =
            candle_core::metal_backend::buffer_o(z_o_metal.buffer(), &z_o_layout, z_out.dtype());
        let a_o_buf =
            candle_core::metal_backend::buffer_o(a_o_metal.buffer(), &a_o_layout, a_out.dtype());
        let b_o_buf =
            candle_core::metal_backend::buffer_o(b_o_metal.buffer(), &b_o_layout, b_out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(qkv_buf.buffer), qkv_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(z_buf.buffer), z_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(b_buf.buffer), b_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(qkv_o_buf.buffer), qkv_o_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(z_o_buf.buffer), z_o_buf.offset_in_bytes);
        encoder.set_buffer(7, Some(a_o_buf.buffer), a_o_buf.offset_in_bytes);
        encoder.set_buffer(8, Some(b_o_buf.buffer), b_o_buf.offset_in_bytes);

        let hidden_u32 = hidden as u32;
        let qkv_dim_u32 = qkv_dim as u32;
        let z_dim_u32 = z_dim as u32;
        let nv_u32 = nv as u32;
        let batch_u32 = batch as u32;
        encoder.set_bytes(9, &hidden_u32);
        encoder.set_bytes(10, &qkv_dim_u32);
        encoder.set_bytes(11, &z_dim_u32);
        encoder.set_bytes(12, &nv_u32);
        encoder.set_bytes(13, &batch_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: dispatch_total,
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

    Ok((qkv_out, z_out, a_out, b_out))
}

pub(crate) fn metal_rotary_embedding_bf16(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<(Tensor, Tensor)> {
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
    // SAFETY: the kernel dispatch writes every Q output element exactly once.
    let q_out = unsafe { Tensor::empty(q_shape.as_slice(), DType::BF16, q.device())? };
    // SAFETY: the kernel dispatch writes every K output element exactly once.
    let k_out = unsafe { Tensor::empty(k_shape.as_slice(), DType::BF16, k.device())? };

    let Device::Metal(device) = q.device() else {
        anyhow::bail!("metal rotary qk requires Metal tensors");
    };
    let pipeline = metal_rotary_qk_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_rotary_qk_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (q_storage, q_layout) = q.storage_and_layout();
        let (k_storage, k_layout) = k.storage_and_layout();
        let (c_storage, c_layout) = cos.storage_and_layout();
        let (s_storage, s_layout) = sin.storage_and_layout();
        let (qo_storage, qo_layout) = q_out.storage_and_layout();
        let (ko_storage, ko_layout) = k_out.storage_and_layout();

        let q_metal = match &*q_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal rotary q must be on Metal"),
        };
        let k_metal = match &*k_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal rotary k must be on Metal"),
        };
        let cos_metal = match &*c_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal rotary cos must be on Metal"),
        };
        let sin_metal = match &*s_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal rotary sin must be on Metal"),
        };
        let q_out_metal = match &*qo_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal rotary q_out must be on Metal"),
        };
        let k_out_metal = match &*ko_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal rotary k_out must be on Metal"),
        };

        let q_buf = candle_core::metal_backend::buffer_o(q_metal.buffer(), &q_layout, q.dtype());
        let k_buf = candle_core::metal_backend::buffer_o(k_metal.buffer(), &k_layout, k.dtype());
        let cos_buf =
            candle_core::metal_backend::buffer_o(cos_metal.buffer(), &c_layout, cos.dtype());
        let sin_buf =
            candle_core::metal_backend::buffer_o(sin_metal.buffer(), &s_layout, sin.dtype());
        let q_out_buf =
            candle_core::metal_backend::buffer_o(q_out_metal.buffer(), &qo_layout, q_out.dtype());
        let k_out_buf =
            candle_core::metal_backend::buffer_o(k_out_metal.buffer(), &ko_layout, k_out.dtype());

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

fn metal_paged_kv_head_major_read_supports(
    k_pool: &Tensor,
    v_pool: &Tensor,
    start_slot: usize,
    seq_len: usize,
) -> bool {
    if seq_len == 0 || k_pool.dtype() != DType::BF16 || v_pool.dtype() != DType::BF16 {
        return false;
    }
    if !matches!(k_pool.device(), Device::Metal(_)) || !matches!(v_pool.device(), Device::Metal(_))
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

fn metal_paged_kv_head_major_read_bf16(
    k_pool: &Tensor,
    v_pool: &Tensor,
    start_slot: usize,
    seq_len: usize,
) -> Result<(Tensor, Tensor)> {
    anyhow::ensure!(
        metal_paged_kv_head_major_read_supports(k_pool, v_pool, start_slot, seq_len),
        "metal paged kv head-major read unsupported shape"
    );
    let (_, heads, head_dim) = k_pool.dims3()?;
    let out_shape = (1usize, heads, seq_len, head_dim);
    // SAFETY: the kernel dispatch covers exactly every element in `out_shape`.
    let k_out = unsafe { Tensor::empty(out_shape, DType::BF16, k_pool.device())? };
    // SAFETY: the kernel dispatch covers exactly every element in `out_shape`.
    let v_out = unsafe { Tensor::empty(out_shape, DType::BF16, v_pool.device())? };

    let Device::Metal(device) = k_pool.device() else {
        anyhow::bail!("metal paged kv read requires Metal tensors");
    };
    let pipeline = metal_paged_kv_head_major_read_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_paged_kv_head_major_read_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (k_storage, k_layout) = k_pool.storage_and_layout();
        let (v_storage, v_layout) = v_pool.storage_and_layout();
        let (ko_storage, ko_layout) = k_out.storage_and_layout();
        let (vo_storage, vo_layout) = v_out.storage_and_layout();

        let k_metal = match &*k_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal paged kv read k_pool must be on Metal"),
        };
        let v_metal = match &*v_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal paged kv read v_pool must be on Metal"),
        };
        let ko_metal = match &*ko_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal paged kv read k_out must be on Metal"),
        };
        let vo_metal = match &*vo_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal paged kv read v_out must be on Metal"),
        };

        let k_buf =
            candle_core::metal_backend::buffer_o(k_metal.buffer(), &k_layout, k_pool.dtype());
        let v_buf =
            candle_core::metal_backend::buffer_o(v_metal.buffer(), &v_layout, v_pool.dtype());
        let ko_buf =
            candle_core::metal_backend::buffer_o(ko_metal.buffer(), &ko_layout, k_out.dtype());
        let vo_buf =
            candle_core::metal_backend::buffer_o(vo_metal.buffer(), &vo_layout, v_out.dtype());

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

fn metal_paged_attn_decode_contiguous_supports(
    q: &Tensor,
    k_pool: &Tensor,
    v_pool: &Tensor,
    start_slot: usize,
    seq_len: usize,
) -> bool {
    if metal_paged_attn_decode_contiguous_disabled() {
        return false;
    }
    if q.dtype() != DType::BF16 || k_pool.dtype() != DType::BF16 || v_pool.dtype() != DType::BF16 {
        return false;
    }
    if !matches!(q.device(), Device::Metal(_))
        || !matches!(k_pool.device(), Device::Metal(_))
        || !matches!(v_pool.device(), Device::Metal(_))
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

fn metal_paged_attn_decode_contiguous_bf16_d256(
    q: &Tensor,
    k_pool: &Tensor,
    v_pool: &Tensor,
    start_slot: usize,
    seq_len: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    anyhow::ensure!(
        metal_paged_attn_decode_contiguous_supports(q, k_pool, v_pool, start_slot, seq_len),
        "metal contiguous paged decode attention unsupported shape"
    );
    let (_, q_heads, _, head_dim) = q.dims4()?;
    // SAFETY: the kernel writes one contiguous [1, 1, q_heads * head_dim] output.
    let out = unsafe {
        Tensor::empty(
            (1usize, 1usize, q_heads * head_dim),
            DType::BF16,
            q.device(),
        )?
    };

    let Device::Metal(device) = q.device() else {
        anyhow::bail!("metal contiguous paged decode attention requires Metal tensors");
    };
    let pipeline = metal_paged_attn_decode_contiguous_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_paged_attn_decode_contiguous_bf16_d256");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (q_storage, q_layout) = q.storage_and_layout();
        let (k_storage, k_layout) = k_pool.storage_and_layout();
        let (v_storage, v_layout) = v_pool.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let q_metal = match &*q_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal contiguous paged attention q must be on Metal"),
        };
        let k_metal = match &*k_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal contiguous paged attention k_pool must be on Metal"),
        };
        let v_metal = match &*v_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal contiguous paged attention v_pool must be on Metal"),
        };
        let out_metal = match &*out_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal contiguous paged attention out must be on Metal"),
        };

        let q_buf = candle_core::metal_backend::buffer_o(q_metal.buffer(), &q_layout, q.dtype());
        let k_buf =
            candle_core::metal_backend::buffer_o(k_metal.buffer(), &k_layout, k_pool.dtype());
        let v_buf =
            candle_core::metal_backend::buffer_o(v_metal.buffer(), &v_layout, v_pool.dtype());
        let out_buf =
            candle_core::metal_backend::buffer_o(out_metal.buffer(), &out_layout, out.dtype());

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
fn metal_paged_attn_decode_contiguous_batch_supports(
    q: &Tensor,
    k_pool: &Tensor,
    v_pool: &Tensor,
    start_slots: &Tensor,
    seq_len: usize,
) -> bool {
    if metal_paged_attn_decode_contiguous_disabled() {
        return false;
    }
    if q.dtype() != DType::BF16
        || k_pool.dtype() != DType::BF16
        || v_pool.dtype() != DType::BF16
        || start_slots.dtype() != DType::U32
    {
        return false;
    }
    if !matches!(q.device(), Device::Metal(_))
        || !matches!(k_pool.device(), Device::Metal(_))
        || !matches!(v_pool.device(), Device::Metal(_))
        || !matches!(start_slots.device(), Device::Metal(_))
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
fn metal_paged_attn_decode_contiguous_batch_bf16_d256(
    q: &Tensor,
    k_pool: &Tensor,
    v_pool: &Tensor,
    start_slots: &Tensor,
    seq_len: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    anyhow::ensure!(
        metal_paged_attn_decode_contiguous_batch_supports(q, k_pool, v_pool, start_slots, seq_len),
        "metal contiguous paged batch decode attention unsupported shape"
    );
    let (batch, q_heads, _, head_dim) = q.dims4()?;
    let (total_slots, _, _) = k_pool.dims3()?;
    let out =
        unsafe { Tensor::empty((batch, 1usize, q_heads * head_dim), DType::BF16, q.device())? };

    let Device::Metal(device) = q.device() else {
        anyhow::bail!("metal contiguous paged batch decode attention requires Metal tensors");
    };
    let pipeline = metal_paged_attn_decode_contiguous_batch_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_paged_attn_decode_contiguous_batch_bf16_d256");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (q_storage, q_layout) = q.storage_and_layout();
        let (k_storage, k_layout) = k_pool.storage_and_layout();
        let (v_storage, v_layout) = v_pool.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();
        let (slot_storage, slot_layout) = start_slots.storage_and_layout();

        let q_metal = match &*q_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal contiguous paged batch attention q must be on Metal"),
        };
        let k_metal = match &*k_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal contiguous paged batch attention k_pool must be on Metal"),
        };
        let v_metal = match &*v_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal contiguous paged batch attention v_pool must be on Metal"),
        };
        let out_metal = match &*out_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal contiguous paged batch attention out must be on Metal"),
        };
        let slot_metal = match &*slot_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal contiguous paged batch attention slots must be on Metal"),
        };

        let q_buf = candle_core::metal_backend::buffer_o(q_metal.buffer(), &q_layout, q.dtype());
        let k_buf =
            candle_core::metal_backend::buffer_o(k_metal.buffer(), &k_layout, k_pool.dtype());
        let v_buf =
            candle_core::metal_backend::buffer_o(v_metal.buffer(), &v_layout, v_pool.dtype());
        let out_buf =
            candle_core::metal_backend::buffer_o(out_metal.buffer(), &out_layout, out.dtype());
        let slot_buf = candle_core::metal_backend::buffer_o(
            slot_metal.buffer(),
            &slot_layout,
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

fn metal_paged_kv_head_major_read_append_token_major_supports(
    k_pool: &Tensor,
    v_pool: &Tensor,
    start_slot: usize,
    prefix_len: usize,
    k_tail: &Tensor,
    v_tail: &Tensor,
) -> bool {
    if prefix_len == 0 {
        return false;
    }
    if !metal_paged_kv_head_major_read_supports(k_pool, v_pool, start_slot, prefix_len) {
        return false;
    }
    if k_tail.dtype() != DType::BF16 || v_tail.dtype() != DType::BF16 {
        return false;
    }
    if !matches!(k_tail.device(), Device::Metal(_)) || !matches!(v_tail.device(), Device::Metal(_))
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

fn metal_paged_kv_head_major_read_append_token_major_bf16(
    k_pool: &Tensor,
    v_pool: &Tensor,
    start_slot: usize,
    prefix_len: usize,
    k_tail: &Tensor,
    v_tail: &Tensor,
) -> Result<(Tensor, Tensor)> {
    anyhow::ensure!(
        metal_paged_kv_head_major_read_append_token_major_supports(
            k_pool, v_pool, start_slot, prefix_len, k_tail, v_tail,
        ),
        "metal paged kv head-major read+append unsupported shape"
    );
    let (_, tail_len, heads, head_dim) = k_tail.dims4()?;
    let total_len = prefix_len + tail_len;
    let out_shape = (1usize, heads, total_len, head_dim);
    // SAFETY: the kernel dispatch covers exactly every element in `out_shape`.
    let k_out = unsafe { Tensor::empty(out_shape, DType::BF16, k_pool.device())? };
    // SAFETY: the kernel dispatch covers exactly every element in `out_shape`.
    let v_out = unsafe { Tensor::empty(out_shape, DType::BF16, v_pool.device())? };

    let Device::Metal(device) = k_pool.device() else {
        anyhow::bail!("metal paged kv read+append requires Metal tensors");
    };
    let pipeline = metal_paged_kv_head_major_read_append_token_major_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_paged_kv_head_major_read_append_token_major_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (k_storage, k_layout) = k_pool.storage_and_layout();
        let (v_storage, v_layout) = v_pool.storage_and_layout();
        let (kt_storage, kt_layout) = k_tail.storage_and_layout();
        let (vt_storage, vt_layout) = v_tail.storage_and_layout();
        let (ko_storage, ko_layout) = k_out.storage_and_layout();
        let (vo_storage, vo_layout) = v_out.storage_and_layout();

        let k_metal = match &*k_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal paged kv read+append k_pool must be on Metal"),
        };
        let v_metal = match &*v_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal paged kv read+append v_pool must be on Metal"),
        };
        let kt_metal = match &*kt_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal paged kv read+append k_tail must be on Metal"),
        };
        let vt_metal = match &*vt_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal paged kv read+append v_tail must be on Metal"),
        };
        let ko_metal = match &*ko_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal paged kv read+append k_out must be on Metal"),
        };
        let vo_metal = match &*vo_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal paged kv read+append v_out must be on Metal"),
        };

        let k_buf =
            candle_core::metal_backend::buffer_o(k_metal.buffer(), &k_layout, k_pool.dtype());
        let v_buf =
            candle_core::metal_backend::buffer_o(v_metal.buffer(), &v_layout, v_pool.dtype());
        let kt_buf =
            candle_core::metal_backend::buffer_o(kt_metal.buffer(), &kt_layout, k_tail.dtype());
        let vt_buf =
            candle_core::metal_backend::buffer_o(vt_metal.buffer(), &vt_layout, v_tail.dtype());
        let ko_buf =
            candle_core::metal_backend::buffer_o(ko_metal.buffer(), &ko_layout, k_out.dtype());
        let vo_buf =
            candle_core::metal_backend::buffer_o(vo_metal.buffer(), &vo_layout, v_out.dtype());

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
    k_pool: &Tensor,
    v_pool: &Tensor,
    slot: usize,
    k: &Tensor,
    v: &Tensor,
) -> bool {
    if metal_paged_kv_write_token_major_disabled() {
        return false;
    }
    if k_pool.dtype() != DType::BF16
        || v_pool.dtype() != DType::BF16
        || k.dtype() != DType::BF16
        || v.dtype() != DType::BF16
    {
        return false;
    }
    if !matches!(k_pool.device(), Device::Metal(_))
        || !matches!(v_pool.device(), Device::Metal(_))
        || !matches!(k.device(), Device::Metal(_))
        || !matches!(v.device(), Device::Metal(_))
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
    k_pool: &mut Tensor,
    v_pool: &mut Tensor,
    slot: usize,
    k: &Tensor,
    v: &Tensor,
) -> Result<()> {
    anyhow::ensure!(
        metal_paged_kv_write_token_major_supports(k_pool, v_pool, slot, k, v),
        "metal paged kv token-major write unsupported shape"
    );
    let (_, heads, head_dim) = k_pool.dims3()?;
    let Device::Metal(device) = k_pool.device() else {
        anyhow::bail!("metal paged kv write requires Metal tensors");
    };
    let pipeline = metal_paged_kv_write_token_major_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_paged_kv_write_token_major_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (ks_storage, ks_layout) = k.storage_and_layout();
        let (vs_storage, vs_layout) = v.storage_and_layout();
        let (kp_storage, kp_layout) = k_pool.storage_and_layout();
        let (vp_storage, vp_layout) = v_pool.storage_and_layout();

        let ks_metal = match &*ks_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal paged kv write k source must be on Metal"),
        };
        let vs_metal = match &*vs_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal paged kv write v source must be on Metal"),
        };
        let kp_metal = match &*kp_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal paged kv write k pool must be on Metal"),
        };
        let vp_metal = match &*vp_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal paged kv write v pool must be on Metal"),
        };

        let ks_buf = candle_core::metal_backend::buffer_o(ks_metal.buffer(), &ks_layout, k.dtype());
        let vs_buf = candle_core::metal_backend::buffer_o(vs_metal.buffer(), &vs_layout, v.dtype());
        let kp_buf =
            candle_core::metal_backend::buffer_o(kp_metal.buffer(), &kp_layout, k_pool.dtype());
        let vp_buf =
            candle_core::metal_backend::buffer_o(vp_metal.buffer(), &vp_layout, v_pool.dtype());

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
pub(crate) fn metal_paged_kv_write_token_major_batch_supports(
    k_pool: &Tensor,
    v_pool: &Tensor,
    slots: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> bool {
    if metal_paged_kv_write_token_major_disabled() {
        return false;
    }
    if k_pool.dtype() != DType::BF16
        || v_pool.dtype() != DType::BF16
        || slots.dtype() != DType::U32
        || k.dtype() != DType::BF16
        || v.dtype() != DType::BF16
    {
        return false;
    }
    if !matches!(k_pool.device(), Device::Metal(_))
        || !matches!(v_pool.device(), Device::Metal(_))
        || !matches!(slots.device(), Device::Metal(_))
        || !matches!(k.device(), Device::Metal(_))
        || !matches!(v.device(), Device::Metal(_))
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
pub(crate) fn metal_paged_kv_write_token_major_batch_bf16(
    k_pool: &mut Tensor,
    v_pool: &mut Tensor,
    slots: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<()> {
    anyhow::ensure!(
        metal_paged_kv_write_token_major_batch_supports(k_pool, v_pool, slots, k, v),
        "metal paged kv token-major batch write unsupported shape"
    );
    let (total_slots, heads, head_dim) = k_pool.dims3()?;
    let (batch, _, _, _) = k.dims4()?;
    let Device::Metal(device) = k_pool.device() else {
        anyhow::bail!("metal paged kv batch write requires Metal tensors");
    };
    let pipeline = metal_paged_kv_write_token_major_batch_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_paged_kv_write_token_major_batch_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (ks_storage, ks_layout) = k.storage_and_layout();
        let (vs_storage, vs_layout) = v.storage_and_layout();
        let (kp_storage, kp_layout) = k_pool.storage_and_layout();
        let (vp_storage, vp_layout) = v_pool.storage_and_layout();
        let (slot_storage, slot_layout) = slots.storage_and_layout();

        let ks_metal = match &*ks_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal paged kv batch write k source must be on Metal"),
        };
        let vs_metal = match &*vs_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal paged kv batch write v source must be on Metal"),
        };
        let kp_metal = match &*kp_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal paged kv batch write k pool must be on Metal"),
        };
        let vp_metal = match &*vp_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal paged kv batch write v pool must be on Metal"),
        };
        let slot_metal = match &*slot_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal paged kv batch write slots must be on Metal"),
        };

        let ks_buf = candle_core::metal_backend::buffer_o(ks_metal.buffer(), &ks_layout, k.dtype());
        let vs_buf = candle_core::metal_backend::buffer_o(vs_metal.buffer(), &vs_layout, v.dtype());
        let kp_buf =
            candle_core::metal_backend::buffer_o(kp_metal.buffer(), &kp_layout, k_pool.dtype());
        let vp_buf =
            candle_core::metal_backend::buffer_o(vp_metal.buffer(), &vp_layout, v_pool.dtype());
        let slot_buf =
            candle_core::metal_backend::buffer_o(slot_metal.buffer(), &slot_layout, slots.dtype());

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
    // The kernel writes every hidden element for every row.
    let out = unsafe { Tensor::empty(x_dims.as_slice(), DType::BF16, x.device())? };

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
    // The kernel writes every Q/K element for every row.
    let q_out = unsafe { Tensor::empty(dims.as_slice(), DType::BF16, q.device())? };
    let k_out = unsafe { Tensor::empty(dims.as_slice(), DType::BF16, q.device())? };

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

pub(crate) fn metal_gdn_qk_norm_gqa_f32_bf16(
    q: &Tensor,
    k: &Tensor,
    nv: usize,
    q_scale: f32,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    anyhow::ensure!(
        metal_gdn_qk_norm_gqa_supports(q, k, nv),
        "metal gdn qk norm gqa unsupported shape"
    );
    let (batch, seq_len, nk, hidden) = q.dims4()?;
    let gqa_ratio = nv / nk;
    let rows = batch * seq_len * nk;
    anyhow::ensure!(
        rows <= u32::MAX as usize
            && nk <= u32::MAX as usize
            && hidden <= u32::MAX as usize
            && gqa_ratio <= u32::MAX as usize,
        "metal gdn qk norm gqa shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    // Each source head writes all replicated value-head outputs.
    let q_out = unsafe { Tensor::empty((batch, seq_len, nv, hidden), DType::BF16, q.device())? };
    let k_out = unsafe { Tensor::empty((batch, seq_len, nv, hidden), DType::BF16, q.device())? };

    if rows == 0 {
        return Ok((q_out, k_out));
    }

    let Device::Metal(device) = q.device() else {
        anyhow::bail!("metal gdn qk norm gqa requires Metal tensors");
    };
    let pipeline = metal_gdn_qk_norm_gqa_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_gdn_qk_norm_gqa_f32_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (q_storage, q_layout) = q.storage_and_layout();
        let (k_storage, k_layout) = k.storage_and_layout();
        let (qo_storage, qo_layout) = q_out.storage_and_layout();
        let (ko_storage, ko_layout) = k_out.storage_and_layout();

        let q_metal = match &*q_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn qk norm gqa q must be on Metal"),
        };
        let k_metal = match &*k_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn qk norm gqa k must be on Metal"),
        };
        let qo_metal = match &*qo_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn qk norm gqa q_out must be on Metal"),
        };
        let ko_metal = match &*ko_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn qk norm gqa k_out must be on Metal"),
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
        let nk_u32 = nk as u32;
        let nv_u32 = nv as u32;
        let hidden_u32 = hidden as u32;
        let gqa_ratio_u32 = gqa_ratio as u32;
        let threads = hidden.next_power_of_two().clamp(32, 1024);
        let threads_u32 = threads as u32;
        encoder.set_bytes(4, &rows_u32);
        encoder.set_bytes(5, &nk_u32);
        encoder.set_bytes(6, &nv_u32);
        encoder.set_bytes(7, &hidden_u32);
        encoder.set_bytes(8, &gqa_ratio_u32);
        encoder.set_bytes(9, &q_scale);
        encoder.set_bytes(10, &eps);
        encoder.set_bytes(11, &threads_u32);

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

#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_decode_qkv_conv_norm_bf16(
    mixed_qkv: &Tensor,
    weight: &Tensor,
    conv_state: &mut Tensor,
    kernel_size: usize,
    nk: usize,
    dk: usize,
    nv: usize,
    dv: usize,
    q_scale: f32,
    eps: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    anyhow::ensure!(
        metal_gdn_decode_qkv_conv_norm_supports(
            mixed_qkv,
            weight,
            conv_state,
            kernel_size,
            nk,
            dk,
            nv,
            dv
        ),
        "metal gdn decode qkv conv/norm unsupported shape"
    );
    let (batch, _, channels) = mixed_qkv.dims3()?;
    let rows_per_batch = nk + nk + nv;
    let rows = batch * rows_per_batch;
    anyhow::ensure!(
        rows <= u32::MAX as usize,
        "metal gdn decode qkv conv/norm shape too large"
    );

    let mixed_qkv = mixed_qkv.contiguous()?;
    let weight = match weight.rank() {
        3 => weight.reshape((channels, kernel_size))?,
        2 => weight.clone(),
        r => anyhow::bail!("metal gdn decode qkv conv/norm weight rank must be 2 or 3, got {r}"),
    }
    .contiguous()?;
    if !conv_state.is_contiguous() {
        *conv_state = conv_state.contiguous()?;
    }

    // The kernel writes every unexpanded Q/K and V element, and updates each
    // convolution state channel exactly once.
    let q_out = unsafe { Tensor::empty((batch, 1usize, nk, dk), DType::BF16, mixed_qkv.device())? };
    let k_out = unsafe { Tensor::empty((batch, 1usize, nk, dk), DType::BF16, mixed_qkv.device())? };
    let v_out = unsafe { Tensor::empty((batch, 1usize, nv, dv), DType::BF16, mixed_qkv.device())? };

    let Device::Metal(device) = mixed_qkv.device() else {
        anyhow::bail!("metal gdn decode qkv conv/norm requires Metal tensors");
    };
    let pipeline = metal_gdn_decode_qkv_conv_norm_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_gdn_decode_qkv_conv_norm_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (x_storage, x_layout) = mixed_qkv.storage_and_layout();
        let (w_storage, w_layout) = weight.storage_and_layout();
        let (s_storage, s_layout) = conv_state.storage_and_layout();
        let (qo_storage, qo_layout) = q_out.storage_and_layout();
        let (ko_storage, ko_layout) = k_out.storage_and_layout();
        let (vo_storage, vo_layout) = v_out.storage_and_layout();

        let x_metal = match &*x_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn decode qkv conv/norm mixed_qkv must be on Metal"),
        };
        let w_metal = match &*w_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn decode qkv conv/norm weight must be on Metal"),
        };
        let s_metal = match &*s_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn decode qkv conv/norm state must be on Metal"),
        };
        let qo_metal = match &*qo_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn decode qkv conv/norm q_out must be on Metal"),
        };
        let ko_metal = match &*ko_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn decode qkv conv/norm k_out must be on Metal"),
        };
        let vo_metal = match &*vo_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn decode qkv conv/norm v_out must be on Metal"),
        };

        let x_buf =
            candle_core::metal_backend::buffer_o(x_metal.buffer(), &x_layout, mixed_qkv.dtype());
        let w_buf =
            candle_core::metal_backend::buffer_o(w_metal.buffer(), &w_layout, weight.dtype());
        let s_buf =
            candle_core::metal_backend::buffer_o(s_metal.buffer(), &s_layout, conv_state.dtype());
        let qo_buf =
            candle_core::metal_backend::buffer_o(qo_metal.buffer(), &qo_layout, q_out.dtype());
        let ko_buf =
            candle_core::metal_backend::buffer_o(ko_metal.buffer(), &ko_layout, k_out.dtype());
        let vo_buf =
            candle_core::metal_backend::buffer_o(vo_metal.buffer(), &vo_layout, v_out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(s_buf.buffer), s_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(qo_buf.buffer), qo_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(ko_buf.buffer), ko_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(vo_buf.buffer), vo_buf.offset_in_bytes);

        let nk_u32 = nk as u32;
        let nv_u32 = nv as u32;
        encoder.set_bytes(6, &nk_u32);
        encoder.set_bytes(7, &nv_u32);
        encoder.set_bytes(8, &q_scale);
        encoder.set_bytes(9, &eps);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: rows_per_batch,
            height: batch,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok((q_out, k_out, v_out))
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
    device const float* a_log [[buffer(2)]],
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

const METAL_GDN_DECODE_GATES_RECURRENT_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gdn_decode_gates_recurrent_bf16(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device const bfloat* a [[buffer(3)]],
    device const bfloat* b [[buffer(4)]],
    device const float* a_log [[buffer(5)]],
    device const bfloat* dt_bias [[buffer(6)]],
    device bfloat* state [[buffer(7)]],
    device bfloat* out [[buffer(8)]],
    constant uint& batch_heads [[buffer(9)]],
    constant uint& dk [[buffer(10)]],
    constant uint& dv [[buffer(11)]],
    constant uint& value_heads [[buffer(12)]],
    constant uint& q_heads [[buffer(13)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr uint NSG = 4;
    constexpr uint LANES = 32;
    if (gid >= batch_heads * dv || tid >= LANES || dk != 128 || dv != 128) {
        return;
    }

    const uint bh = gid / dv;
    const uint d = gid - bh * dv;
    const uint batch_idx = bh / value_heads;
    const uint head_idx = bh - batch_idx * value_heads;
    const uint q_group = value_heads / q_heads;
    const uint q_head_idx = head_idx / q_group;
    const uint qk_base = (batch_idx * q_heads + q_head_idx) * dk;
    const uint v_base = (batch_idx * value_heads + head_idx) * dv;
    const uint gate_idx = batch_idx * value_heads + head_idx;
    const uint state_base = bh * dk * dv;

    const float beta = static_cast<float>(static_cast<bfloat>(
        kiln_stable_sigmoid(static_cast<float>(b[gate_idx]))
    ));
    const float g = static_cast<float>(static_cast<bfloat>(
        kiln_stable_softplus(static_cast<float>(a[gate_idx]) + static_cast<float>(dt_bias[head_idx])) *
        -exp(static_cast<float>(a_log[head_idx]))
    ));
    const float decay = static_cast<float>(static_cast<bfloat>(exp(g)));

    float ls[NSG];
    for (uint j = 0; j < NSG; ++j) {
        const uint is = tid * NSG + j;
        ls[j] = static_cast<float>(state[state_base + is * dv + d]);
    }

    float s_k = 0.0f;
    for (uint j = 0; j < NSG; ++j) {
        const uint is = tid * NSG + j;
        const float decayed = static_cast<float>(static_cast<bfloat>(ls[j] * decay));
        ls[j] = decayed;
        s_k += decayed * static_cast<float>(k[qk_base + is]);
    }
    s_k = simd_sum(s_k);

    const float delta = static_cast<float>(static_cast<bfloat>(
        (static_cast<float>(v[v_base + d]) - s_k) * beta
    ));

    float y = 0.0f;
    for (uint j = 0; j < NSG; ++j) {
        const uint is = tid * NSG + j;
        const float new_s = static_cast<float>(static_cast<bfloat>(
            ls[j] + static_cast<float>(k[qk_base + is]) * delta
        ));
        ls[j] = new_s;
        y += new_s * static_cast<float>(q[qk_base + is]);
        state[state_base + is * dv + d] = static_cast<bfloat>(new_s);
    }
    y = simd_sum(y);

    if (tid == 0) {
        out[v_base + d] = static_cast<bfloat>(y);
    }
}
"#;

const METAL_GDN_DECODE_GATES_RECURRENT_RMSNORM_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gdn_decode_gates_recurrent_rmsnorm_bf16(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device const bfloat* a [[buffer(3)]],
    device const bfloat* b [[buffer(4)]],
    device const float* a_log [[buffer(5)]],
    device const bfloat* dt_bias [[buffer(6)]],
    device bfloat* state [[buffer(7)]],
    device const bfloat* z [[buffer(8)]],
    device const float* weight [[buffer(9)]],
    device bfloat* out [[buffer(10)]],
    constant uint& batch_heads [[buffer(11)]],
    constant uint& dk [[buffer(12)]],
    constant uint& dv [[buffer(13)]],
    constant uint& value_heads [[buffer(14)]],
    constant uint& q_heads [[buffer(15)]],
    constant float& eps [[buffer(16)]],
    uint bh [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float scratch[128];
    if (bh >= batch_heads || tid >= dv || dk != 128 || dv != 128) {
        return;
    }

    const uint d = tid;
    const uint batch_idx = bh / value_heads;
    const uint head_idx = bh - batch_idx * value_heads;
    const uint q_group = value_heads / q_heads;
    const uint q_head_idx = head_idx / q_group;
    const uint qk_base = (batch_idx * q_heads + q_head_idx) * dk;
    const uint v_base = (batch_idx * value_heads + head_idx) * dv;
    const uint gate_idx = batch_idx * value_heads + head_idx;
    const uint state_base = bh * dk * dv;

    const float beta = static_cast<float>(static_cast<bfloat>(
        kiln_stable_sigmoid(static_cast<float>(b[gate_idx]))
    ));
    const float g = static_cast<float>(static_cast<bfloat>(
        kiln_stable_softplus(static_cast<float>(a[gate_idx]) + static_cast<float>(dt_bias[head_idx])) *
        -exp(static_cast<float>(a_log[head_idx]))
    ));
    const float decay = static_cast<float>(static_cast<bfloat>(exp(g)));

    float s_k = 0.0f;
    for (uint i = 0; i < dk; ++i) {
        const uint state_idx = state_base + i * dv + d;
        const float decayed = static_cast<float>(static_cast<bfloat>(
            static_cast<float>(state[state_idx]) * decay
        ));
        state[state_idx] = static_cast<bfloat>(decayed);
        s_k += decayed * static_cast<float>(k[qk_base + i]);
    }
    const float delta = static_cast<float>(static_cast<bfloat>(
        (static_cast<float>(v[v_base + d]) - s_k) * beta
    ));

    float y = 0.0f;
    for (uint i = 0; i < dk; ++i) {
        const uint state_idx = state_base + i * dv + d;
        const float new_s = static_cast<float>(static_cast<bfloat>(
            static_cast<float>(state[state_idx]) + static_cast<float>(k[qk_base + i]) * delta
        ));
        state[state_idx] = static_cast<bfloat>(new_s);
        y += new_s * static_cast<float>(q[qk_base + i]);
    }

    scratch[tid] = y * y;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms_inv = rsqrt((scratch[0] / static_cast<float>(dv)) + eps);
    const float zv = static_cast<float>(z[v_base + d]);
    const float gate = zv / (1.0f + exp(-zv));
    out[v_base + d] = static_cast<bfloat>(
        y * rms_inv * static_cast<float>(weight[d]) * gate
    );
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

fn metal_gdn_decode_gates_recurrent_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal gdn decode gates+recurrent pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_decode_gates_recurrent_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn decode gates+recurrent function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn decode gates+recurrent pipeline: {e:?}"))?;
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gdn_decode_gates_recurrent_rmsnorm_pipeline(
    device: &candle_core::metal_backend::MetalDevice,
) -> Result<candle_metal_kernels::metal::ComputePipeline> {
    use candle_core::metal_backend::DeviceId;
    use candle_metal_kernels::metal::ComputePipeline;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    static PIPELINES: OnceLock<Mutex<HashMap<DeviceId, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().map_err(|_| {
        anyhow::anyhow!("metal gdn decode gates+recurrent+rmsnorm pipeline cache poisoned")
    })?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_decode_gates_recurrent_rmsnorm_bf16", None)
        .map_err(|e| {
            anyhow::anyhow!("load metal gdn decode gates+recurrent+rmsnorm function: {e:?}")
        })?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| {
            anyhow::anyhow!("build metal gdn decode gates+recurrent+rmsnorm pipeline: {e:?}")
        })?;
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
    // The gates kernel writes every beta/g element.
    let beta = unsafe { Tensor::empty(shape.clone(), DType::BF16, a.device())? };
    let g = unsafe { Tensor::empty(shape, DType::BF16, a.device())? };

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

#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_decode_gates_recurrent_bf16(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    anyhow::ensure!(
        metal_gdn_decode_gates_recurrent_supports(q, k, v, a, b, a_log, dt_bias, state),
        "metal gdn decode gates+recurrent unsupported shape"
    );
    let (batch, seq_len, q_heads, dk) = q.dims4()?;
    let (_, _, value_heads, dv) = v.dims4()?;
    let batch_heads = batch * value_heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && dk <= u32::MAX as usize
            && dv <= u32::MAX as usize
            && value_heads <= u32::MAX as usize
            && q_heads <= u32::MAX as usize,
        "metal gdn decode gates+recurrent shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let a_log = a_log.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let out = unsafe { Tensor::empty((batch, seq_len, value_heads, dv), DType::BF16, q.device())? };

    let Device::Metal(device) = q.device() else {
        anyhow::bail!("metal gdn decode gates+recurrent requires a Metal tensor");
    };
    let pipeline = metal_gdn_decode_gates_recurrent_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_gdn_decode_gates_recurrent_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (q_storage, q_layout) = q.storage_and_layout();
        let (k_storage, k_layout) = k.storage_and_layout();
        let (v_storage, v_layout) = v.storage_and_layout();
        let (a_storage, a_layout) = a.storage_and_layout();
        let (b_storage, b_layout) = b.storage_and_layout();
        let (al_storage, al_layout) = a_log.storage_and_layout();
        let (dt_storage, dt_layout) = dt_bias.storage_and_layout();
        let (state_storage, state_layout) = state.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let q_metal = match &*q_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn decode gates+recurrent q must be on Metal"),
        };
        let k_metal = match &*k_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn decode gates+recurrent k must be on Metal"),
        };
        let v_metal = match &*v_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn decode gates+recurrent v must be on Metal"),
        };
        let a_metal = match &*a_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn decode gates+recurrent a must be on Metal"),
        };
        let b_metal = match &*b_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn decode gates+recurrent b must be on Metal"),
        };
        let al_metal = match &*al_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn decode gates+recurrent a_log must be on Metal"),
        };
        let dt_metal = match &*dt_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn decode gates+recurrent dt_bias must be on Metal"),
        };
        let state_metal = match &*state_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn decode gates+recurrent state must be on Metal"),
        };
        let out_metal = match &*out_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn decode gates+recurrent out must be on Metal"),
        };

        let q_buf = candle_core::metal_backend::buffer_o(q_metal.buffer(), &q_layout, q.dtype());
        let k_buf = candle_core::metal_backend::buffer_o(k_metal.buffer(), &k_layout, k.dtype());
        let v_buf = candle_core::metal_backend::buffer_o(v_metal.buffer(), &v_layout, v.dtype());
        let a_buf = candle_core::metal_backend::buffer_o(a_metal.buffer(), &a_layout, a.dtype());
        let b_buf = candle_core::metal_backend::buffer_o(b_metal.buffer(), &b_layout, b.dtype());
        let al_buf =
            candle_core::metal_backend::buffer_o(al_metal.buffer(), &al_layout, a_log.dtype());
        let dt_buf =
            candle_core::metal_backend::buffer_o(dt_metal.buffer(), &dt_layout, dt_bias.dtype());
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
        encoder.set_buffer(3, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(b_buf.buffer), b_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(al_buf.buffer), al_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(dt_buf.buffer), dt_buf.offset_in_bytes);
        encoder.set_buffer(7, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(8, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let value_heads_u32 = value_heads as u32;
        let q_heads_u32 = q_heads as u32;
        encoder.set_bytes(9, &batch_heads_u32);
        encoder.set_bytes(10, &dk_u32);
        encoder.set_bytes(11, &dv_u32);
        encoder.set_bytes(12, &value_heads_u32);
        encoder.set_bytes(13, &q_heads_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads * dv,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_decode_gates_recurrent_rmsnorm_bf16(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    state: &mut Tensor,
    z: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    anyhow::ensure!(
        metal_gdn_decode_gates_recurrent_rmsnorm_supports(
            q, k, v, a, b, a_log, dt_bias, state, z, weight
        ),
        "metal gdn decode gates+recurrent+rmsnorm unsupported shape"
    );
    let (batch, seq_len, q_heads, dk) = q.dims4()?;
    let (_, _, value_heads, dv) = v.dims4()?;
    let batch_heads = batch * value_heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && dk <= u32::MAX as usize
            && dv <= u32::MAX as usize
            && value_heads <= u32::MAX as usize
            && q_heads <= u32::MAX as usize,
        "metal gdn decode gates+recurrent+rmsnorm shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let a_log = a_log.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let z = z.contiguous()?;
    let weight = weight.contiguous()?;
    let out = unsafe { Tensor::empty((batch, seq_len, value_heads, dv), DType::BF16, q.device())? };

    let Device::Metal(device) = q.device() else {
        anyhow::bail!("metal gdn decode gates+recurrent+rmsnorm requires a Metal tensor");
    };
    let pipeline = metal_gdn_decode_gates_recurrent_rmsnorm_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_gdn_decode_gates_recurrent_rmsnorm_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (q_storage, q_layout) = q.storage_and_layout();
        let (k_storage, k_layout) = k.storage_and_layout();
        let (v_storage, v_layout) = v.storage_and_layout();
        let (a_storage, a_layout) = a.storage_and_layout();
        let (b_storage, b_layout) = b.storage_and_layout();
        let (al_storage, al_layout) = a_log.storage_and_layout();
        let (dt_storage, dt_layout) = dt_bias.storage_and_layout();
        let (state_storage, state_layout) = state.storage_and_layout();
        let (z_storage, z_layout) = z.storage_and_layout();
        let (w_storage, w_layout) = weight.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let q_metal = match &*q_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn decode gates+recurrent+rmsnorm q must be on Metal"),
        };
        let k_metal = match &*k_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn decode gates+recurrent+rmsnorm k must be on Metal"),
        };
        let v_metal = match &*v_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn decode gates+recurrent+rmsnorm v must be on Metal"),
        };
        let a_metal = match &*a_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn decode gates+recurrent+rmsnorm a must be on Metal"),
        };
        let b_metal = match &*b_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn decode gates+recurrent+rmsnorm b must be on Metal"),
        };
        let al_metal = match &*al_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn decode gates+recurrent+rmsnorm a_log must be on Metal"),
        };
        let dt_metal = match &*dt_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn decode gates+recurrent+rmsnorm dt_bias must be on Metal"),
        };
        let state_metal = match &*state_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn decode gates+recurrent+rmsnorm state must be on Metal"),
        };
        let z_metal = match &*z_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn decode gates+recurrent+rmsnorm z must be on Metal"),
        };
        let w_metal = match &*w_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn decode gates+recurrent+rmsnorm weight must be on Metal"),
        };
        let out_metal = match &*out_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn decode gates+recurrent+rmsnorm out must be on Metal"),
        };

        let q_buf = candle_core::metal_backend::buffer_o(q_metal.buffer(), &q_layout, q.dtype());
        let k_buf = candle_core::metal_backend::buffer_o(k_metal.buffer(), &k_layout, k.dtype());
        let v_buf = candle_core::metal_backend::buffer_o(v_metal.buffer(), &v_layout, v.dtype());
        let a_buf = candle_core::metal_backend::buffer_o(a_metal.buffer(), &a_layout, a.dtype());
        let b_buf = candle_core::metal_backend::buffer_o(b_metal.buffer(), &b_layout, b.dtype());
        let al_buf =
            candle_core::metal_backend::buffer_o(al_metal.buffer(), &al_layout, a_log.dtype());
        let dt_buf =
            candle_core::metal_backend::buffer_o(dt_metal.buffer(), &dt_layout, dt_bias.dtype());
        let state_buf = candle_core::metal_backend::buffer_o(
            state_metal.buffer(),
            &state_layout,
            state.dtype(),
        );
        let z_buf = candle_core::metal_backend::buffer_o(z_metal.buffer(), &z_layout, z.dtype());
        let w_buf =
            candle_core::metal_backend::buffer_o(w_metal.buffer(), &w_layout, weight.dtype());
        let out_buf =
            candle_core::metal_backend::buffer_o(out_metal.buffer(), &out_layout, out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(b_buf.buffer), b_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(al_buf.buffer), al_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(dt_buf.buffer), dt_buf.offset_in_bytes);
        encoder.set_buffer(7, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(8, Some(z_buf.buffer), z_buf.offset_in_bytes);
        encoder.set_buffer(9, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(10, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let value_heads_u32 = value_heads as u32;
        let q_heads_u32 = q_heads as u32;
        encoder.set_bytes(11, &batch_heads_u32);
        encoder.set_bytes(12, &dk_u32);
        encoder.set_bytes(13, &dv_u32);
        encoder.set_bytes(14, &value_heads_u32);
        encoder.set_bytes(15, &q_heads_u32);
        encoder.set_bytes(16, &eps);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: dv,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

const METAL_GATED_RMSNORM_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gated_rmsnorm_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* z [[buffer(1)]],
    device const float* weight [[buffer(2)]],
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
    // The kernel writes every hidden element for every row.
    let out = unsafe { Tensor::empty((batch, seq_len, heads, hidden), DType::BF16, x.device())? };

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

const METAL_GDN_RECURRENT_PREFILL_HEAD_LAST_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gdn_recurrent_prefill_head_last_bf16(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device const bfloat* beta [[buffer(3)]],
    device const bfloat* g [[buffer(4)]],
    device bfloat* state [[buffer(5)]],
    device bfloat* out [[buffer(6)]],
    constant uint& batch_heads [[buffer(7)]],
    constant uint& seq_len [[buffer(8)]],
    constant uint& dk [[buffer(9)]],
    constant uint& dv [[buffer(10)]],
    constant uint& value_heads [[buffer(11)]],
    constant uint& q_heads [[buffer(12)]],
    constant uint& input_mode [[buffer(13)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr uint NSG = 4;
    constexpr uint LANES = 32;
    if (gid >= batch_heads * dv || tid >= LANES || dk != 128) {
        return;
    }

    const uint bh = gid / dv;
    const uint d = gid - bh * dv;
    const uint batch_idx = bh / value_heads;
    const uint head_idx = bh - batch_idx * value_heads;
    const uint q_group = value_heads / q_heads;
    const uint q_head_idx = head_idx / q_group;
    const uint qk_base = (batch_idx * q_heads + q_head_idx) * seq_len * dk;
    const uint v_base = bh * seq_len * dv;
    const uint gate_base = bh * seq_len;
    const uint state_base = bh * dk * dv;

    float ls[NSG];
    for (uint j = 0; j < NSG; ++j) {
        const uint is = tid * NSG + j;
        ls[j] = static_cast<float>(state[state_base + is * dv + d]);
    }

    for (uint t = 0; t < seq_len; ++t) {
        const uint qk_t = (input_mode == 0)
            ? qk_base + t * dk
            : ((batch_idx * seq_len + t) * q_heads + q_head_idx) * dk;
        const uint v_t = (input_mode == 0)
            ? v_base + t * dv
            : ((batch_idx * seq_len + t) * value_heads + head_idx) * dv;
        const uint gate_t = (input_mode == 0)
            ? gate_base + t
            : (batch_idx * seq_len + t) * value_heads + head_idx;
        const float decay = static_cast<float>(static_cast<bfloat>(
            exp(static_cast<float>(g[gate_t]))
        ));

        float s_k = 0.0f;
        for (uint j = 0; j < NSG; ++j) {
            const uint is = tid * NSG + j;
            const float decayed = static_cast<float>(static_cast<bfloat>(ls[j] * decay));
            ls[j] = decayed;
            s_k += decayed * static_cast<float>(k[qk_t + is]);
        }
        s_k = simd_sum(s_k);

        const float delta = static_cast<float>(static_cast<bfloat>(
            (static_cast<float>(v[v_t + d]) - s_k) *
            static_cast<float>(beta[gate_t])
        ));

        float y = 0.0f;
        for (uint j = 0; j < NSG; ++j) {
            const uint is = tid * NSG + j;
            const float new_s = static_cast<float>(static_cast<bfloat>(
                ls[j] + static_cast<float>(k[qk_t + is]) * delta
            ));
            ls[j] = new_s;
            y += new_s * static_cast<float>(q[qk_t + is]);
        }
        y = simd_sum(y);

        if (tid == 0) {
            const uint out_idx = ((batch_idx * seq_len + t) * value_heads + head_idx) * dv + d;
            out[out_idx] = static_cast<bfloat>(y);
        }
    }

    for (uint j = 0; j < NSG; ++j) {
        const uint is = tid * NSG + j;
        state[state_base + is * dv + d] = static_cast<bfloat>(ls[j]);
    }
}
"#;

const METAL_GDN_FULL_CHUNK_FORWARD_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gdn_full_chunk_forward_bf16(
    device const bfloat* g [[buffer(0)]],
    device const bfloat* v [[buffer(1)]],
    device const bfloat* kkt [[buffer(2)]],
    device const bfloat* qkt [[buffer(3)]],
    device const bfloat* ks_entry [[buffer(4)]],
    device const bfloat* q_s [[buffer(5)]],
    device const bfloat* beta [[buffer(6)]],
    device const bfloat* k_t [[buffer(7)]],
    device bfloat* state [[buffer(8)]],
    device bfloat* out [[buffer(9)]],
    constant uint& batch_heads [[buffer(10)]],
    constant uint& dk [[buffer(11)]],
    constant uint& dv [[buffer(12)]],
    constant uint& output_mode [[buffer(13)]],
    constant uint& t_start [[buffer(14)]],
    constant uint& seq_len [[buffer(15)]],
    constant uint& heads [[buffer(16)]],
    constant uint& g_bh_stride [[buffer(17)]],
    constant uint& g_t_stride [[buffer(18)]],
    constant uint& v_bh_stride [[buffer(19)]],
    constant uint& v_t_stride [[buffer(20)]],
    constant uint& v_d_stride [[buffer(21)]],
    constant uint& beta_bh_stride [[buffer(22)]],
    constant uint& beta_t_stride [[buffer(23)]],
    constant uint& kt_bh_stride [[buffer(24)]],
    constant uint& kt_k_stride [[buffer(25)]],
    constant uint& kt_t_stride [[buffer(26)]],
    uint bh [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr uint C = 64;
    constexpr uint MAX_DV = 128;
    if (bh >= batch_heads) {
        return;
    }

    threadgroup bfloat sArow[64];
    threadgroup bfloat sBrow[64];
    threadgroup bfloat sW[8192];
    threadgroup float sBigG[64];
    threadgroup float sP[64];
    threadgroup float sDecayLast[64];
    threadgroup float sPLast;

    const uint g_strided_base = bh * g_bh_stride;
    const uint v_strided_base = bh * v_bh_stride;
    const uint beta_base = bh * beta_bh_stride;
    const uint cdv_base = bh * C * dv;
    const uint cc_base = bh * C * C;
    const uint kt_strided_base = bh * kt_bh_stride;
    const uint state_base = bh * dk * dv;

    for (uint i = tid; i < C; i += 128) {
        sBigG[i] = static_cast<float>(g[g_strided_base + i * g_t_stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float acc = 0.0f;
        for (uint i = 0; i < C; ++i) {
            acc += sBigG[i];
            sBigG[i] = acc;
        }
        sPLast = exp(sBigG[C - 1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < C; i += 128) {
        sP[i] = exp(sBigG[i]);
        sDecayLast[i] = exp(sBigG[C - 1] - sBigG[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 0; t < C; ++t) {
        for (uint i = tid; i < C; i += 128) {
            const uint ti = t * C + i;
            const float decay = exp(sBigG[t] - sBigG[i]);
            const float a_val = (i < t) ? static_cast<float>(kkt[cc_base + ti]) * decay : 0.0f;
            const float b_val = (i <= t) ? static_cast<float>(qkt[cc_base + ti]) * decay : 0.0f;
            sArow[i] = static_cast<bfloat>(a_val);
            sBrow[i] = static_cast<bfloat>(b_val);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float beta_t = static_cast<float>(beta[beta_base + t * beta_t_stride]);
        const float p_t = static_cast<float>(static_cast<bfloat>(sP[t]));

        if (tid < dv) {
            float acc_a = 0.0f;
            for (uint i = 0; i < t; ++i) {
                acc_a += static_cast<float>(sArow[i]) *
                         static_cast<float>(sW[i * MAX_DV + tid]);
            }

            const uint td = t * dv + tid;
            const float vp = static_cast<float>(static_cast<bfloat>(
                static_cast<float>(v[v_strided_base + t * v_t_stride + tid * v_d_stride]) -
                static_cast<float>(ks_entry[cdv_base + td]) * p_t
            ));
            const float w_val = beta_t * (vp - acc_a);
            sW[t * MAX_DV + tid] = static_cast<bfloat>(w_val);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < dv) {
            float acc_b = 0.0f;
            for (uint i = 0; i <= t; ++i) {
                acc_b += static_cast<float>(sBrow[i]) *
                         static_cast<float>(sW[i * MAX_DV + tid]);
            }

            const uint td = t * dv + tid;
            const float qss = static_cast<float>(static_cast<bfloat>(
                static_cast<float>(q_s[cdv_base + td]) * p_t
            ));
            const bfloat out_val = static_cast<bfloat>(qss + acc_b);
            if (output_mode == 0) {
                out[cdv_base + td] = out_val;
            } else {
                const uint batch_idx = bh / heads;
                const uint head_idx = bh - batch_idx * heads;
                const uint out_t = t_start + t;
                const uint out_idx = ((batch_idx * seq_len + out_t) * heads + head_idx) * dv + tid;
                out[out_idx] = out_val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < dv) {
        const float p_last = static_cast<float>(static_cast<bfloat>(sPLast));
        for (uint k_idx = 0; k_idx < dk; ++k_idx) {
            float delta = 0.0f;
            for (uint t = 0; t < C; ++t) {
                const float kt = static_cast<float>(
                    k_t[kt_strided_base + k_idx * kt_k_stride + t * kt_t_stride]
                );
                const float w = static_cast<float>(sW[t * MAX_DV + tid]);
                const float decay_last = static_cast<float>(static_cast<bfloat>(sDecayLast[t]));
                const float w_weighted = static_cast<float>(static_cast<bfloat>(w * decay_last));
                delta += kt * w_weighted;
            }
            const uint state_idx = state_base + k_idx * dv + tid;
            const float prev = static_cast<float>(state[state_idx]);
            state[state_idx] = static_cast<bfloat>(prev * p_last + delta);
        }
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

fn metal_gdn_recurrent_prefill_head_last_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal gdn recurrent prefill pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_recurrent_prefill_head_last_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn recurrent prefill function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn recurrent prefill pipeline: {e:?}"))?;
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

fn metal_gdn_full_chunk_forward_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal gdn full-chunk pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_full_chunk_forward_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn full-chunk function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn full-chunk pipeline: {e:?}"))?;
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
    // The kernel writes every chunk/value element.
    let out = unsafe {
        Tensor::empty(
            (batch, heads, chunk_size, dv),
            DType::BF16,
            a_strict.device(),
        )?
    };

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
    // The prep kernel fills each temporary completely before any consumer sees it.
    let a_strict = unsafe {
        Tensor::empty(
            (batch, heads, chunk_size, chunk_size),
            DType::BF16,
            device_ref,
        )?
    };
    let b_mask = unsafe {
        Tensor::empty(
            (batch, heads, chunk_size, chunk_size),
            DType::BF16,
            device_ref,
        )?
    };
    let v_prime =
        unsafe { Tensor::empty((batch, heads, chunk_size, dv), DType::BF16, device_ref)? };
    let q_s_scaled =
        unsafe { Tensor::empty((batch, heads, chunk_size, dv), DType::BF16, device_ref)? };
    let decay_last_col =
        unsafe { Tensor::empty((batch, heads, chunk_size), DType::BF16, device_ref)? };
    let p_last = unsafe { Tensor::empty((batch, heads), DType::BF16, device_ref)? };

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
        let ks_buf =
            candle_core::metal_backend::buffer_o(ks_metal.buffer(), &ks_layout, ks_entry.dtype());
        let qs_buf =
            candle_core::metal_backend::buffer_o(qs_metal.buffer(), &qs_layout, q_s.dtype());
        let a_buf =
            candle_core::metal_backend::buffer_o(a_metal.buffer(), &a_layout, a_strict.dtype());
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

    Ok((
        a_strict,
        b_mask,
        v_prime,
        q_s_scaled,
        decay_last_col,
        p_last,
    ))
}

fn metal_gdn_full_chunk_forward_bf16(
    g: &Tensor,
    v: &Tensor,
    kkt: &Tensor,
    qkt: &Tensor,
    ks_entry: &Tensor,
    q_s: &Tensor,
    beta: &Tensor,
    k_t: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    anyhow::ensure!(
        metal_gdn_full_chunk_forward_supports(g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state),
        "metal gdn full-chunk unsupported shape"
    );
    let (batch, heads, chunk_size) = g.dims3()?;
    let (_, _, dk, _) = k_t.dims4()?;
    let dv = v.dim(3)?;
    let batch_heads = batch * heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize && dk <= u32::MAX as usize && dv <= u32::MAX as usize,
        "metal gdn full-chunk shape too large"
    );

    let g = g.contiguous()?;
    let v = v.contiguous()?;
    let kkt = kkt.contiguous()?;
    let qkt = qkt.contiguous()?;
    let ks_entry = ks_entry.contiguous()?;
    let q_s = q_s.contiguous()?;
    let beta = beta.contiguous()?;
    let k_t = k_t.contiguous()?;
    // The full-chunk kernel writes every output token/head/value element.
    let out = unsafe { Tensor::empty((batch, heads, chunk_size, dv), DType::BF16, g.device())? };

    let Device::Metal(device) = g.device() else {
        anyhow::bail!("metal gdn full-chunk requires a Metal tensor");
    };
    let pipeline = metal_gdn_full_chunk_forward_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_gdn_full_chunk_forward_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (g_storage, g_layout) = g.storage_and_layout();
        let (v_storage, v_layout) = v.storage_and_layout();
        let (kkt_storage, kkt_layout) = kkt.storage_and_layout();
        let (qkt_storage, qkt_layout) = qkt.storage_and_layout();
        let (ks_storage, ks_layout) = ks_entry.storage_and_layout();
        let (qs_storage, qs_layout) = q_s.storage_and_layout();
        let (beta_storage, beta_layout) = beta.storage_and_layout();
        let (kt_storage, kt_layout) = k_t.storage_and_layout();
        let (state_storage, state_layout) = state.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let g_metal = match &*g_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn full-chunk g must be on Metal"),
        };
        let v_metal = match &*v_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn full-chunk v must be on Metal"),
        };
        let kkt_metal = match &*kkt_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn full-chunk kkt must be on Metal"),
        };
        let qkt_metal = match &*qkt_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn full-chunk qkt must be on Metal"),
        };
        let ks_metal = match &*ks_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn full-chunk ks_entry must be on Metal"),
        };
        let qs_metal = match &*qs_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn full-chunk q_s must be on Metal"),
        };
        let beta_metal = match &*beta_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn full-chunk beta must be on Metal"),
        };
        let kt_metal = match &*kt_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn full-chunk k_t must be on Metal"),
        };
        let state_metal = match &*state_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn full-chunk state must be on Metal"),
        };
        let out_metal = match &*out_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn full-chunk out must be on Metal"),
        };

        let g_buf = candle_core::metal_backend::buffer_o(g_metal.buffer(), &g_layout, g.dtype());
        let v_buf = candle_core::metal_backend::buffer_o(v_metal.buffer(), &v_layout, v.dtype());
        let kkt_buf =
            candle_core::metal_backend::buffer_o(kkt_metal.buffer(), &kkt_layout, kkt.dtype());
        let qkt_buf =
            candle_core::metal_backend::buffer_o(qkt_metal.buffer(), &qkt_layout, qkt.dtype());
        let ks_buf =
            candle_core::metal_backend::buffer_o(ks_metal.buffer(), &ks_layout, ks_entry.dtype());
        let qs_buf =
            candle_core::metal_backend::buffer_o(qs_metal.buffer(), &qs_layout, q_s.dtype());
        let beta_buf =
            candle_core::metal_backend::buffer_o(beta_metal.buffer(), &beta_layout, beta.dtype());
        let kt_buf =
            candle_core::metal_backend::buffer_o(kt_metal.buffer(), &kt_layout, k_t.dtype());
        let state_buf = candle_core::metal_backend::buffer_o(
            state_metal.buffer(),
            &state_layout,
            state.dtype(),
        );
        let out_buf =
            candle_core::metal_backend::buffer_o(out_metal.buffer(), &out_layout, out.dtype());

        encoder.set_buffer(0, Some(g_buf.buffer), g_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(kkt_buf.buffer), kkt_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(qkt_buf.buffer), qkt_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(ks_buf.buffer), ks_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(qs_buf.buffer), qs_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(7, Some(kt_buf.buffer), kt_buf.offset_in_bytes);
        encoder.set_buffer(8, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(9, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let output_mode_u32 = 0u32;
        let t_start_u32 = 0u32;
        let seq_len_u32 = chunk_size as u32;
        let heads_u32 = heads as u32;
        let g_stride = g_layout.stride();
        let v_stride = v_layout.stride();
        let beta_stride = beta_layout.stride();
        let kt_stride = kt_layout.stride();
        let g_bh_stride_u32 = g_stride[1] as u32;
        let g_t_stride_u32 = g_stride[2] as u32;
        let v_bh_stride_u32 = v_stride[1] as u32;
        let v_t_stride_u32 = v_stride[2] as u32;
        let v_d_stride_u32 = v_stride[3] as u32;
        let beta_bh_stride_u32 = beta_stride[1] as u32;
        let beta_t_stride_u32 = beta_stride[2] as u32;
        let kt_bh_stride_u32 = kt_stride[1] as u32;
        let kt_k_stride_u32 = kt_stride[2] as u32;
        let kt_t_stride_u32 = kt_stride[3] as u32;
        encoder.set_bytes(10, &batch_heads_u32);
        encoder.set_bytes(11, &dk_u32);
        encoder.set_bytes(12, &dv_u32);
        encoder.set_bytes(13, &output_mode_u32);
        encoder.set_bytes(14, &t_start_u32);
        encoder.set_bytes(15, &seq_len_u32);
        encoder.set_bytes(16, &heads_u32);
        encoder.set_bytes(17, &g_bh_stride_u32);
        encoder.set_bytes(18, &g_t_stride_u32);
        encoder.set_bytes(19, &v_bh_stride_u32);
        encoder.set_bytes(20, &v_t_stride_u32);
        encoder.set_bytes(21, &v_d_stride_u32);
        encoder.set_bytes(22, &beta_bh_stride_u32);
        encoder.set_bytes(23, &beta_t_stride_u32);
        encoder.set_bytes(24, &kt_bh_stride_u32);
        encoder.set_bytes(25, &kt_k_stride_u32);
        encoder.set_bytes(26, &kt_t_stride_u32);

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

    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn metal_gdn_full_chunk_forward_head_last_into_bf16(
    g: &Tensor,
    v: &Tensor,
    kkt: &Tensor,
    qkt: &Tensor,
    ks_entry: &Tensor,
    q_s: &Tensor,
    beta: &Tensor,
    k_t: &Tensor,
    state: &mut Tensor,
    out: &Tensor,
    t_start: usize,
    seq_len: usize,
) -> Result<()> {
    anyhow::ensure!(
        metal_gdn_full_chunk_forward_head_last_supports(
            g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state, out, t_start, seq_len,
        ),
        "metal gdn full-chunk head-last unsupported shape"
    );
    let (batch, heads, _) = g.dims3()?;
    let (_, _, dk, _) = k_t.dims4()?;
    let dv = v.dim(3)?;
    let batch_heads = batch * heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && dk <= u32::MAX as usize
            && dv <= u32::MAX as usize
            && t_start <= u32::MAX as usize
            && seq_len <= u32::MAX as usize
            && heads <= u32::MAX as usize,
        "metal gdn full-chunk head-last shape too large"
    );

    let kkt = kkt.contiguous()?;
    let qkt = qkt.contiguous()?;
    let ks_entry = ks_entry.contiguous()?;
    let q_s = q_s.contiguous()?;

    let Device::Metal(device) = g.device() else {
        anyhow::bail!("metal gdn full-chunk head-last requires a Metal tensor");
    };
    let pipeline = metal_gdn_full_chunk_forward_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_gdn_full_chunk_forward_head_last_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (g_storage, g_layout) = g.storage_and_layout();
        let (v_storage, v_layout) = v.storage_and_layout();
        let (kkt_storage, kkt_layout) = kkt.storage_and_layout();
        let (qkt_storage, qkt_layout) = qkt.storage_and_layout();
        let (ks_storage, ks_layout) = ks_entry.storage_and_layout();
        let (qs_storage, qs_layout) = q_s.storage_and_layout();
        let (beta_storage, beta_layout) = beta.storage_and_layout();
        let (kt_storage, kt_layout) = k_t.storage_and_layout();
        let (state_storage, state_layout) = state.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let g_metal = match &*g_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn full-chunk head-last g must be on Metal"),
        };
        let v_metal = match &*v_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn full-chunk head-last v must be on Metal"),
        };
        let kkt_metal = match &*kkt_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn full-chunk head-last kkt must be on Metal"),
        };
        let qkt_metal = match &*qkt_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn full-chunk head-last qkt must be on Metal"),
        };
        let ks_metal = match &*ks_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn full-chunk head-last ks_entry must be on Metal"),
        };
        let qs_metal = match &*qs_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn full-chunk head-last q_s must be on Metal"),
        };
        let beta_metal = match &*beta_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn full-chunk head-last beta must be on Metal"),
        };
        let kt_metal = match &*kt_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn full-chunk head-last k_t must be on Metal"),
        };
        let state_metal = match &*state_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn full-chunk head-last state must be on Metal"),
        };
        let out_metal = match &*out_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn full-chunk head-last out must be on Metal"),
        };

        let g_buf = candle_core::metal_backend::buffer_o(g_metal.buffer(), &g_layout, g.dtype());
        let v_buf = candle_core::metal_backend::buffer_o(v_metal.buffer(), &v_layout, v.dtype());
        let kkt_buf =
            candle_core::metal_backend::buffer_o(kkt_metal.buffer(), &kkt_layout, kkt.dtype());
        let qkt_buf =
            candle_core::metal_backend::buffer_o(qkt_metal.buffer(), &qkt_layout, qkt.dtype());
        let ks_buf =
            candle_core::metal_backend::buffer_o(ks_metal.buffer(), &ks_layout, ks_entry.dtype());
        let qs_buf =
            candle_core::metal_backend::buffer_o(qs_metal.buffer(), &qs_layout, q_s.dtype());
        let beta_buf =
            candle_core::metal_backend::buffer_o(beta_metal.buffer(), &beta_layout, beta.dtype());
        let kt_buf =
            candle_core::metal_backend::buffer_o(kt_metal.buffer(), &kt_layout, k_t.dtype());
        let state_buf = candle_core::metal_backend::buffer_o(
            state_metal.buffer(),
            &state_layout,
            state.dtype(),
        );
        let out_buf =
            candle_core::metal_backend::buffer_o(out_metal.buffer(), &out_layout, out.dtype());

        encoder.set_buffer(0, Some(g_buf.buffer), g_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(kkt_buf.buffer), kkt_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(qkt_buf.buffer), qkt_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(ks_buf.buffer), ks_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(qs_buf.buffer), qs_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(7, Some(kt_buf.buffer), kt_buf.offset_in_bytes);
        encoder.set_buffer(8, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(9, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let output_mode_u32 = 1u32;
        let t_start_u32 = t_start as u32;
        let seq_len_u32 = seq_len as u32;
        let heads_u32 = heads as u32;
        let g_stride = g_layout.stride();
        let v_stride = v_layout.stride();
        let beta_stride = beta_layout.stride();
        let kt_stride = kt_layout.stride();
        let g_bh_stride_u32 = g_stride[1] as u32;
        let g_t_stride_u32 = g_stride[2] as u32;
        let v_bh_stride_u32 = v_stride[1] as u32;
        let v_t_stride_u32 = v_stride[2] as u32;
        let v_d_stride_u32 = v_stride[3] as u32;
        let beta_bh_stride_u32 = beta_stride[1] as u32;
        let beta_t_stride_u32 = beta_stride[2] as u32;
        let kt_bh_stride_u32 = kt_stride[1] as u32;
        let kt_k_stride_u32 = kt_stride[2] as u32;
        let kt_t_stride_u32 = kt_stride[3] as u32;
        encoder.set_bytes(10, &batch_heads_u32);
        encoder.set_bytes(11, &dk_u32);
        encoder.set_bytes(12, &dv_u32);
        encoder.set_bytes(13, &output_mode_u32);
        encoder.set_bytes(14, &t_start_u32);
        encoder.set_bytes(15, &seq_len_u32);
        encoder.set_bytes(16, &heads_u32);
        encoder.set_bytes(17, &g_bh_stride_u32);
        encoder.set_bytes(18, &g_t_stride_u32);
        encoder.set_bytes(19, &v_bh_stride_u32);
        encoder.set_bytes(20, &v_t_stride_u32);
        encoder.set_bytes(21, &v_d_stride_u32);
        encoder.set_bytes(22, &beta_bh_stride_u32);
        encoder.set_bytes(23, &beta_t_stride_u32);
        encoder.set_bytes(24, &kt_bh_stride_u32);
        encoder.set_bytes(25, &kt_k_stride_u32);
        encoder.set_bytes(26, &kt_t_stride_u32);

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

    Ok(())
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
    // The recurrent kernel writes every batch/head/value element.
    let out = unsafe { Tensor::empty((batch, heads, dv), DType::BF16, q.device())? };

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

fn metal_gdn_recurrent_prefill_head_last_bf16(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    anyhow::ensure!(
        metal_gdn_recurrent_prefill_head_last_supports(q, k, v, beta, g, state),
        "metal gdn recurrent prefill unsupported shape"
    );
    let (batch, q_heads, seq_len, dk) = q.dims4()?;
    let (_, value_heads, _, _) = v.dims4()?;
    let dv = v.dim(3)?;
    let batch_heads = batch * value_heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && seq_len <= u32::MAX as usize
            && dk <= u32::MAX as usize
            && dv <= u32::MAX as usize
            && value_heads <= u32::MAX as usize
            && q_heads <= u32::MAX as usize,
        "metal gdn recurrent prefill shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let beta = beta.contiguous()?;
    let g = g.contiguous()?;
    if !state.is_contiguous() {
        *state = state.contiguous()?;
    }
    // SAFETY: the kernel dispatch covers every (batch, token, value-head, dv)
    // output element exactly once via `gid=batch_head*dv+d` and the token loop.
    let out = unsafe { Tensor::empty((batch, seq_len, value_heads, dv), DType::BF16, q.device())? };

    let Device::Metal(device) = q.device() else {
        anyhow::bail!("metal gdn recurrent prefill requires a Metal tensor");
    };
    let pipeline = metal_gdn_recurrent_prefill_head_last_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_gdn_recurrent_prefill_head_last_bf16");
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
            _ => anyhow::bail!("metal gdn recurrent prefill q must be on Metal"),
        };
        let k_metal = match &*k_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn recurrent prefill k must be on Metal"),
        };
        let v_metal = match &*v_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn recurrent prefill v must be on Metal"),
        };
        let beta_metal = match &*beta_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn recurrent prefill beta must be on Metal"),
        };
        let g_metal = match &*g_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn recurrent prefill g must be on Metal"),
        };
        let state_metal = match &*state_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn recurrent prefill state must be on Metal"),
        };
        let out_metal = match &*out_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn recurrent prefill out must be on Metal"),
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
        let seq_len_u32 = seq_len as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let value_heads_u32 = value_heads as u32;
        let q_heads_u32 = q_heads as u32;
        let input_mode_u32 = 0u32;
        encoder.set_bytes(7, &batch_heads_u32);
        encoder.set_bytes(8, &seq_len_u32);
        encoder.set_bytes(9, &dk_u32);
        encoder.set_bytes(10, &dv_u32);
        encoder.set_bytes(11, &value_heads_u32);
        encoder.set_bytes(12, &q_heads_u32);
        encoder.set_bytes(13, &input_mode_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads * dv,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

fn metal_gdn_recurrent_prefill_native_head_last_bf16(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    anyhow::ensure!(
        metal_gdn_recurrent_prefill_native_head_last_supports(q, k, v, beta, g, state),
        "metal gdn recurrent native prefill unsupported shape"
    );
    let (batch, seq_len, q_heads, dk) = q.dims4()?;
    let (_, _, value_heads, dv) = v.dims4()?;
    let batch_heads = batch * value_heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && seq_len <= u32::MAX as usize
            && dk <= u32::MAX as usize
            && dv <= u32::MAX as usize
            && value_heads <= u32::MAX as usize
            && q_heads <= u32::MAX as usize,
        "metal gdn recurrent native prefill shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let beta = beta.contiguous()?;
    let g = g.contiguous()?;
    if !state.is_contiguous() {
        *state = state.contiguous()?;
    }
    // SAFETY: the kernel dispatch covers every (batch, token, value-head, dv)
    // output element exactly once via `gid=batch_head*dv+d` and the token loop.
    let out = unsafe { Tensor::empty((batch, seq_len, value_heads, dv), DType::BF16, q.device())? };

    let Device::Metal(device) = q.device() else {
        anyhow::bail!("metal gdn recurrent native prefill requires a Metal tensor");
    };
    let pipeline = metal_gdn_recurrent_prefill_head_last_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_gdn_recurrent_prefill_native_head_last_bf16");
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
            _ => anyhow::bail!("metal gdn recurrent native prefill q must be on Metal"),
        };
        let k_metal = match &*k_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn recurrent native prefill k must be on Metal"),
        };
        let v_metal = match &*v_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn recurrent native prefill v must be on Metal"),
        };
        let beta_metal = match &*beta_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn recurrent native prefill beta must be on Metal"),
        };
        let g_metal = match &*g_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn recurrent native prefill g must be on Metal"),
        };
        let state_metal = match &*state_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn recurrent native prefill state must be on Metal"),
        };
        let out_metal = match &*out_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn recurrent native prefill out must be on Metal"),
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
        let seq_len_u32 = seq_len as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let value_heads_u32 = value_heads as u32;
        let q_heads_u32 = q_heads as u32;
        let input_mode_u32 = 1u32;
        encoder.set_bytes(7, &batch_heads_u32);
        encoder.set_bytes(8, &seq_len_u32);
        encoder.set_bytes(9, &dk_u32);
        encoder.set_bytes(10, &dv_u32);
        encoder.set_bytes(11, &value_heads_u32);
        encoder.set_bytes(12, &q_heads_u32);
        encoder.set_bytes(13, &input_mode_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads * dv,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
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

kernel void kiln_gdn_prefill_qkv_conv_split_bf16_f32_k4(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device float* conv_state [[buffer(2)]],
    device float* q_out [[buffer(3)]],
    device float* k_out [[buffer(4)]],
    device bfloat* v_out [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& channels [[buffer(8)]],
    constant uint& qk_dim [[buffer(9)]],
    constant uint& v_dim [[buffer(10)]],
    constant uint& threadgroup_width [[buffer(11)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint total_channels = batch * channels;
    if (gid >= total_channels) {
        return;
    }

    const uint b = gid / channels;
    const uint c = gid - b * channels;
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
            float xv;
            if (padded_idx < 3) {
                xv = s_state[padded_idx];
            } else {
                const uint x_idx = (b * seq_len + (padded_idx - 3)) * channels + c;
                xv = static_cast<float>(x[x_idx]);
            }
            acc += xv * static_cast<float>(weight[weight_base + j]);
        }

        const float y = acc / (1.0f + exp(-acc));
        if (c < qk_dim) {
            q_out[(b * seq_len + t) * qk_dim + c] = y;
        } else if (c < qk_dim * 2) {
            k_out[(b * seq_len + t) * qk_dim + (c - qk_dim)] = y;
        } else {
            v_out[(b * seq_len + t) * v_dim + (c - qk_dim * 2)] = static_cast<bfloat>(y);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        if (seq_len >= 3) {
            conv_state[state_base + 0] =
                static_cast<float>(x[(b * seq_len + (seq_len - 3)) * channels + c]);
            conv_state[state_base + 1] =
                static_cast<float>(x[(b * seq_len + (seq_len - 2)) * channels + c]);
            conv_state[state_base + 2] =
                static_cast<float>(x[(b * seq_len + (seq_len - 1)) * channels + c]);
        } else if (seq_len == 2) {
            conv_state[state_base + 0] = s_state[2];
            conv_state[state_base + 1] = static_cast<float>(x[(b * seq_len) * channels + c]);
            conv_state[state_base + 2] = static_cast<float>(x[(b * seq_len + 1) * channels + c]);
        } else if (seq_len == 1) {
            conv_state[state_base + 0] = s_state[1];
            conv_state[state_base + 1] = s_state[2];
            conv_state[state_base + 2] = static_cast<float>(x[(b * seq_len) * channels + c]);
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

fn metal_gdn_prefill_qkv_conv_split_pipeline(
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
        .map_err(|_| anyhow::anyhow!("metal gdn prefill qkv conv-split pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_prefill_qkv_conv_split_bf16_f32_k4", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn prefill qkv conv-split function: {e:?}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn prefill qkv conv-split pipeline: {e:?}"))?;
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_prefill_qkv_conv_split_supports(
    mixed_qkv: &Tensor,
    weight: &Tensor,
    conv_state: &Tensor,
    kernel_size: usize,
    nk: usize,
    dk: usize,
    nv: usize,
    dv: usize,
) -> bool {
    if metal_gdn_prefill_qkv_conv_split_disabled() {
        return false;
    }
    if mixed_qkv.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || conv_state.dtype() != DType::F32
    {
        return false;
    }
    if !matches!(mixed_qkv.device(), Device::Metal(_))
        || !matches!(weight.device(), Device::Metal(_))
        || !matches!(conv_state.device(), Device::Metal(_))
    {
        return false;
    }
    if !mixed_qkv.is_contiguous() || !weight.is_contiguous() || !conv_state.is_contiguous() {
        return false;
    }
    let Ok((batch, seq_len, channels)) = mixed_qkv.dims3() else {
        return false;
    };
    let Ok((s_batch, s_channels, s_width)) = conv_state.dims3() else {
        return false;
    };
    let qk_dim = nk.saturating_mul(dk);
    let v_dim = nv.saturating_mul(dv);
    let Some(expected_channels) = qk_dim.checked_mul(2).and_then(|n| n.checked_add(v_dim)) else {
        return false;
    };
    let weight_ok = match weight.rank() {
        2 => weight.dims() == [channels, kernel_size],
        3 => weight.dims() == [channels, 1, kernel_size],
        _ => false,
    };
    batch <= u32::MAX as usize
        && seq_len > 1
        && seq_len <= u32::MAX as usize
        && channels == expected_channels
        && channels <= u32::MAX as usize
        && qk_dim <= u32::MAX as usize
        && v_dim <= u32::MAX as usize
        && kernel_size == 4
        && (s_batch, s_channels, s_width) == (batch, channels, 3)
        && weight_ok
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_prefill_qkv_conv_split_bf16_f32_k4(
    mixed_qkv: &Tensor,
    weight: &Tensor,
    conv_state: &mut Tensor,
    kernel_size: usize,
    nk: usize,
    dk: usize,
    nv: usize,
    dv: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    anyhow::ensure!(
        metal_gdn_prefill_qkv_conv_split_supports(
            mixed_qkv,
            weight,
            conv_state,
            kernel_size,
            nk,
            dk,
            nv,
            dv,
        ),
        "metal gdn prefill qkv conv-split unsupported shape"
    );
    let (batch, seq_len, channels) = mixed_qkv.dims3()?;
    let qk_dim = nk * dk;
    let v_dim = nv * dv;

    let weight = match weight.rank() {
        3 => weight.reshape((channels, kernel_size))?,
        2 => weight.clone(),
        r => anyhow::bail!("metal gdn prefill qkv conv-split weight rank must be 2 or 3, got {r}"),
    }
    .contiguous()?;
    if !conv_state.is_contiguous() {
        *conv_state = conv_state.contiguous()?;
    }
    let q = unsafe { Tensor::empty((batch, seq_len, nk, dk), DType::F32, mixed_qkv.device())? };
    let k = unsafe { Tensor::empty((batch, seq_len, nk, dk), DType::F32, mixed_qkv.device())? };
    let v = unsafe { Tensor::empty((batch, seq_len, nv, dv), DType::BF16, mixed_qkv.device())? };

    let Device::Metal(device) = mixed_qkv.device() else {
        anyhow::bail!("metal gdn prefill qkv conv-split requires a Metal tensor");
    };
    let pipeline = metal_gdn_prefill_qkv_conv_split_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_gdn_prefill_qkv_conv_split_bf16_f32_k4");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (x_storage, x_layout) = mixed_qkv.storage_and_layout();
        let (w_storage, w_layout) = weight.storage_and_layout();
        let (s_storage, s_layout) = conv_state.storage_and_layout();
        let (q_storage, q_layout) = q.storage_and_layout();
        let (k_storage, k_layout) = k.storage_and_layout();
        let (v_storage, v_layout) = v.storage_and_layout();

        let x_metal = match &*x_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn prefill qkv conv-split x must be on Metal"),
        };
        let w_metal = match &*w_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn prefill qkv conv-split weight must be on Metal"),
        };
        let s_metal = match &*s_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn prefill qkv conv-split state must be on Metal"),
        };
        let q_metal = match &*q_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn prefill qkv conv-split q output must be on Metal"),
        };
        let k_metal = match &*k_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn prefill qkv conv-split k output must be on Metal"),
        };
        let v_metal = match &*v_storage {
            candle_core::Storage::Metal(s) => s,
            _ => anyhow::bail!("metal gdn prefill qkv conv-split v output must be on Metal"),
        };

        let x_buf =
            candle_core::metal_backend::buffer_o(x_metal.buffer(), &x_layout, mixed_qkv.dtype());
        let w_buf =
            candle_core::metal_backend::buffer_o(w_metal.buffer(), &w_layout, weight.dtype());
        let s_buf =
            candle_core::metal_backend::buffer_o(s_metal.buffer(), &s_layout, conv_state.dtype());
        let q_buf = candle_core::metal_backend::buffer_o(q_metal.buffer(), &q_layout, q.dtype());
        let k_buf = candle_core::metal_backend::buffer_o(k_metal.buffer(), &k_layout, k.dtype());
        let v_buf = candle_core::metal_backend::buffer_o(v_metal.buffer(), &v_layout, v.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(s_buf.buffer), s_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(v_buf.buffer), v_buf.offset_in_bytes);

        let batch_u32 = batch as u32;
        let seq_len_u32 = seq_len as u32;
        let channels_u32 = channels as u32;
        let qk_dim_u32 = qk_dim as u32;
        let v_dim_u32 = v_dim as u32;
        let threads = seq_len.next_power_of_two().clamp(32, 256);
        let threads_u32 = threads as u32;
        encoder.set_bytes(6, &batch_u32);
        encoder.set_bytes(7, &seq_len_u32);
        encoder.set_bytes(8, &channels_u32);
        encoder.set_bytes(9, &qk_dim_u32);
        encoder.set_bytes(10, &v_dim_u32);
        encoder.set_bytes(11, &threads_u32);

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

    Ok((q, k, v))
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
    // The conv prefill kernel writes every batch/channel/time element.
    let out = unsafe { Tensor::empty((batch, channels, seq_len), DType::F32, x.device())? };

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
    // The conv update kernel writes every batch/channel element.
    let out = unsafe { Tensor::empty((batch, channels, 1usize), DType::F32, x.device())? };

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
    use std::time::Instant;

    const QWEN35_HIDDEN: usize = 2560;
    const QWEN35_INTERMEDIATE: usize = 9216;
    const QWEN35_ATTN_QKV_OUT: usize = 4096;

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

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(default)
    }

    fn patterned_bf16_2d(
        rows: usize,
        cols: usize,
        device: &Device,
        modulus: usize,
        scale: f32,
    ) -> Result<Tensor> {
        let data: Vec<f32> = (0..(rows * cols))
            .map(|i| ((i % modulus) as f32 - (modulus / 2) as f32) * scale)
            .collect();
        Ok(Tensor::from_slice(&data, (rows, cols), device)?
            .to_dtype(DType::BF16)?
            .contiguous()?)
    }

    fn patterned_bf16_x(hidden: usize, device: &Device) -> Result<Tensor> {
        let data: Vec<f32> = (0..hidden)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.001953125)
            .collect();
        Ok(Tensor::from_slice(&data, (1usize, 1usize, hidden), device)?
            .to_dtype(DType::BF16)?
            .contiguous()?)
    }

    fn patterned_bf16_decode_batch(batch: usize, hidden: usize, device: &Device) -> Result<Tensor> {
        let data: Vec<f32> = (0..(batch * hidden))
            .map(|i| ((i % 31) as f32 - 15.0) * 0.001953125)
            .collect();
        Ok(Tensor::from_slice(&data, (batch, 1usize, hidden), device)?
            .to_dtype(DType::BF16)?
            .contiguous()?)
    }

    fn patterned_bf16_4d(
        batch: usize,
        seq_len: usize,
        heads: usize,
        head_dim: usize,
        device: &Device,
        modulus: usize,
        scale: f32,
    ) -> Result<Tensor> {
        let elems = batch * seq_len * heads * head_dim;
        let data: Vec<f32> = (0..elems)
            .map(|i| ((i % modulus) as f32 - (modulus / 2) as f32) * scale)
            .collect();
        Ok(
            Tensor::from_slice(&data, (batch, seq_len, heads, head_dim), device)?
                .to_dtype(DType::BF16)?
                .contiguous()?,
        )
    }

    fn patterned_rotary_tables_3d(
        batch: usize,
        seq_len: usize,
        half_rotary: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let elems = batch * seq_len * half_rotary;
        let mut cos_data = Vec::with_capacity(elems);
        let mut sin_data = Vec::with_capacity(elems);
        for b in 0..batch {
            for t in 0..seq_len {
                for d in 0..half_rotary {
                    let angle = ((b * 29 + t * 7 + d) as f32 + 0.5) * 0.03125;
                    cos_data.push(angle.cos());
                    sin_data.push(angle.sin());
                }
            }
        }
        Ok((
            Tensor::from_slice(&cos_data, (batch, seq_len, half_rotary), device)?.contiguous()?,
            Tensor::from_slice(&sin_data, (batch, seq_len, half_rotary), device)?.contiguous()?,
        ))
    }

    fn mlp_gate_up_reference(x: &Tensor, gate_t: &Tensor, up_t: &Tensor) -> Result<Tensor> {
        let gate = x.broadcast_matmul(gate_t)?;
        let gate_sig = (gate.neg()?.exp()? + 1.0)?.recip()?;
        let gate = (gate * gate_sig)?;
        let up = x.broadcast_matmul(up_t)?;
        Ok((gate * up)?)
    }

    fn attn_gate_reference(x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let gate_sig = (gate.neg()?.exp()? + 1.0)?.recip()?;
        Ok((x * gate_sig)?)
    }

    fn paged_attn_decode_contiguous_rowwise(
        q: &Tensor,
        k_pool: &Tensor,
        v_pool: &Tensor,
        start_slots: &[usize],
        seq_len: usize,
        scale: f32,
    ) -> Result<Tensor> {
        let (batch, _, _, _) = q.dims4()?;
        anyhow::ensure!(
            start_slots.len() == batch,
            "rowwise contiguous decode start slot count mismatch"
        );
        let mut rows = Vec::with_capacity(batch);
        for (idx, &start_slot) in start_slots.iter().enumerate() {
            let q_row = q.narrow(0, idx, 1)?.contiguous()?;
            rows.push(metal_paged_attn_decode_contiguous_bf16_d256(
                &q_row, k_pool, v_pool, start_slot, seq_len, scale,
            )?);
        }
        let row_refs: Vec<&Tensor> = rows.iter().collect();
        Tensor::cat(&row_refs, 0).context("stack rowwise contiguous decode outputs")
    }

    fn rotary_qk_rowwise(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        head_dim: usize,
        rotary_dim: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (batch, _, _, _) = q.dims4()?;
        anyhow::ensure!(
            cos.dims().len() == 3 && sin.dims().len() == 3,
            "rowwise rotary qk expects 3D per-batch tables"
        );
        let mut q_rows = Vec::with_capacity(batch);
        let mut k_rows = Vec::with_capacity(batch);
        for idx in 0..batch {
            let q_row = q.narrow(0, idx, 1)?.contiguous()?;
            let k_row = k.narrow(0, idx, 1)?.contiguous()?;
            let cos_row = cos.narrow(0, idx, 1)?.squeeze(0)?.contiguous()?;
            let sin_row = sin.narrow(0, idx, 1)?.squeeze(0)?.contiguous()?;
            let (q_rot, k_rot) = metal_rotary_embedding_bf16(
                &q_row, &k_row, &cos_row, &sin_row, head_dim, rotary_dim,
            )?;
            q_rows.push(q_rot);
            k_rows.push(k_rot);
        }
        let q_refs: Vec<&Tensor> = q_rows.iter().collect();
        let k_refs: Vec<&Tensor> = k_rows.iter().collect();
        Ok((Tensor::cat(&q_refs, 0)?, Tensor::cat(&k_refs, 0)?))
    }

    fn bench_metal_tensor_op<F>(
        device: &Device,
        warmup: usize,
        iters: usize,
        mut op: F,
    ) -> Result<f64>
    where
        F: FnMut() -> Result<Tensor>,
    {
        let mut last = None;
        for _ in 0..warmup {
            last = Some(op()?);
        }
        device.synchronize()?;
        std::hint::black_box(last.as_ref().map(Tensor::dims));
        drop(last);

        let start = Instant::now();
        let mut last = None;
        for _ in 0..iters {
            last = Some(op()?);
        }
        device.synchronize()?;
        std::hint::black_box(last.as_ref().map(Tensor::dims));
        Ok(start.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64)
    }

    fn bench_metal_unit_op<F>(
        device: &Device,
        warmup: usize,
        iters: usize,
        mut op: F,
    ) -> Result<f64>
    where
        F: FnMut() -> Result<()>,
    {
        for _ in 0..warmup {
            op()?;
        }
        device.synchronize()?;

        let start = Instant::now();
        for _ in 0..iters {
            op()?;
        }
        device.synchronize()?;
        Ok(start.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64)
    }

    fn bench_metal_pair_op<F>(
        device: &Device,
        warmup: usize,
        iters: usize,
        mut op: F,
    ) -> Result<f64>
    where
        F: FnMut() -> Result<(Tensor, Tensor)>,
    {
        let mut last = None;
        for _ in 0..warmup {
            last = Some(op()?);
        }
        device.synchronize()?;
        if let Some((a, b)) = &last {
            std::hint::black_box((a.dims(), b.dims()));
        }
        drop(last);

        let start = Instant::now();
        let mut last = None;
        for _ in 0..iters {
            last = Some(op()?);
        }
        device.synchronize()?;
        if let Some((a, b)) = &last {
            std::hint::black_box((a.dims(), b.dims()));
        }
        Ok(start.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64)
    }

    fn bench_metal_qkv_op<F>(device: &Device, warmup: usize, iters: usize, mut op: F) -> Result<f64>
    where
        F: FnMut() -> Result<(Tensor, Tensor, Tensor)>,
    {
        let mut last = None;
        for _ in 0..warmup {
            last = Some(op()?);
        }
        device.synchronize()?;
        if let Some((q, k, v)) = &last {
            std::hint::black_box((q.dims(), k.dims(), v.dims()));
        }
        drop(last);

        let start = Instant::now();
        let mut last = None;
        for _ in 0..iters {
            last = Some(op()?);
        }
        device.synchronize()?;
        if let Some((q, k, v)) = &last {
            std::hint::black_box((q.dims(), k.dims(), v.dims()));
        }
        Ok(start.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64)
    }

    fn bench_metal_gdn_in_proj_op<F>(
        device: &Device,
        warmup: usize,
        iters: usize,
        mut op: F,
    ) -> Result<f64>
    where
        F: FnMut() -> Result<(Tensor, Tensor, Tensor, Tensor)>,
    {
        let mut last = None;
        for _ in 0..warmup {
            last = Some(op()?);
        }
        device.synchronize()?;
        if let Some((qkv, z, a, b)) = &last {
            std::hint::black_box((qkv.dims(), z.dims(), a.dims(), b.dims()));
        }
        drop(last);

        let start = Instant::now();
        let mut last = None;
        for _ in 0..iters {
            last = Some(op()?);
        }
        device.synchronize()?;
        if let Some((qkv, z, a, b)) = &last {
            std::hint::black_box((qkv.dims(), z.dims(), a.dims(), b.dims()));
        }
        Ok(start.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64)
    }

    fn gdn_in_proj_broadcast_reference(
        x: &Tensor,
        qkv_t: &Tensor,
        z_t: &Tensor,
        a_t: &Tensor,
        b_t: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        Ok((
            x.broadcast_matmul(qkv_t)?,
            x.broadcast_matmul(z_t)?,
            x.broadcast_matmul(a_t)?,
            x.broadcast_matmul(b_t)?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn gdn_decode_qkv_conv_norm_split_reference(
        mixed_qkv: &Tensor,
        weight: &Tensor,
        conv_state: &mut Tensor,
        kernel_size: usize,
        nk: usize,
        dk: usize,
        nv: usize,
        dv: usize,
        q_scale: f64,
        eps: f64,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, seq_len, _) = mixed_qkv.dims3()?;
        let qk_dim = nk * dk;
        let v_dim = nv * dv;
        let conv = metal_causal_conv1d_update_bf16_f32_k4(
            &mixed_qkv.transpose(1, 2)?.contiguous()?,
            weight,
            conv_state,
            kernel_size,
        )?
        .transpose(1, 2)?;
        let q = conv
            .narrow(2, 0, qk_dim)?
            .reshape((batch, seq_len, nk, dk))?;
        let k = conv
            .narrow(2, qk_dim, qk_dim)?
            .reshape((batch, seq_len, nk, dk))?;
        let v = conv
            .narrow(2, 2 * qk_dim, v_dim)?
            .reshape((batch, seq_len, nv, dv))?
            .to_dtype(DType::BF16)?;
        let (q, k) = gdn_qk_norm_reference(&q, &k, q_scale, eps)?;
        Ok((q, k, v))
    }

    #[allow(clippy::too_many_arguments)]
    fn gdn_decode_qkv_conv_norm_row_fused_reference(
        mixed_qkv: &Tensor,
        weight: &Tensor,
        state_data: &[f32],
        kernel_size: usize,
        nk: usize,
        dk: usize,
        nv: usize,
        dv: usize,
        q_scale: f32,
        eps: f32,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (batch, _, channels) = mixed_qkv.dims3()?;
        let state_width = kernel_size - 1;
        anyhow::ensure!(
            state_data.len() == batch * channels * state_width,
            "row fused reference state data shape mismatch"
        );
        let mut q_rows = Vec::with_capacity(batch);
        let mut k_rows = Vec::with_capacity(batch);
        let mut v_rows = Vec::with_capacity(batch);
        let mut state_rows = Vec::with_capacity(batch);

        for b in 0..batch {
            let row = mixed_qkv.narrow(0, b, 1)?.contiguous()?;
            let state_start = b * channels * state_width;
            let state_end = state_start + channels * state_width;
            let mut state_row = Tensor::from_slice(
                &state_data[state_start..state_end],
                (1usize, channels, state_width),
                mixed_qkv.device(),
            )?
            .contiguous()?;
            let (q, k, v) = metal_gdn_decode_qkv_conv_norm_bf16(
                &row,
                weight,
                &mut state_row,
                kernel_size,
                nk,
                dk,
                nv,
                dv,
                q_scale,
                eps,
            )?;
            q_rows.push(q);
            k_rows.push(k);
            v_rows.push(v);
            state_rows.push(state_row);
        }

        let q_refs: Vec<&Tensor> = q_rows.iter().collect();
        let k_refs: Vec<&Tensor> = k_rows.iter().collect();
        let v_refs: Vec<&Tensor> = v_rows.iter().collect();
        let state_refs: Vec<&Tensor> = state_rows.iter().collect();
        Ok((
            Tensor::cat(&q_refs, 0)?,
            Tensor::cat(&k_refs, 0)?,
            Tensor::cat(&v_refs, 0)?,
            Tensor::cat(&state_refs, 0)?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn gdn_decode_gates_recurrent_rmsnorm_split_reference(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        a: &Tensor,
        b: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
        state: &mut Tensor,
        z: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        let (beta, g) = metal_gdn_gates_bf16(a, b, a_log, dt_bias)?;
        let out = metal_gdn_recurrent_prefill_native_head_last_bf16(q, k, v, &beta, &g, state)?;
        metal_gated_rms_norm_bf16(&out, z, weight, eps)
    }

    fn bench_transposed_coop_projection_case(
        device: &Device,
        name: &str,
        input_dim: usize,
        output_dim: usize,
        warmup: usize,
        iters: usize,
    ) -> Result<()> {
        let x = patterned_bf16_x(input_dim, device)?;
        let weight_t = patterned_bf16_2d(input_dim, output_dim, device, 37, 0.0009765625)?;
        device.synchronize()?;

        assert!(metal_transposed_coop_gemv_supports(&x, &weight_t));
        let reference = x.broadcast_matmul(&weight_t)?;
        let tile4 = metal_transposed_coop_gemv_bf16_with_tile(
            &x,
            &weight_t,
            MetalTransposedCoopGemvTile::Tile4,
        )?;
        let tile8 = metal_transposed_coop_gemv_bf16_with_tile(
            &x,
            &weight_t,
            MetalTransposedCoopGemvTile::Tile8,
        )?;
        device.synchronize()?;

        assert_eq!(reference.dims(), &[1usize, 1usize, output_dim]);
        assert_eq!(tile4.dims(), &[1usize, 1usize, output_dim]);
        assert_eq!(tile8.dims(), &[1usize, 1usize, output_dim]);
        assert_eq!(tile4.dtype(), DType::BF16);
        assert_eq!(tile8.dtype(), DType::BF16);

        let tile4_max = max_abs_diff(&reference, &tile4)?;
        let tile4_mean = mean_abs_diff(&reference, &tile4)?;
        let tile8_max = max_abs_diff(&reference, &tile8)?;
        let tile8_mean = mean_abs_diff(&reference, &tile8)?;
        assert!(
            tile4_max < 1.5e-1,
            "{name} transposed coop GEMV tile4 max_abs_diff={tile4_max:e} exceeds tolerance"
        );
        assert!(
            tile4_mean < 2.5e-2,
            "{name} transposed coop GEMV tile4 mean_abs_diff={tile4_mean:e} exceeds tolerance"
        );
        assert!(
            tile8_max < 1.5e-1,
            "{name} transposed coop GEMV tile8 max_abs_diff={tile8_max:e} exceeds tolerance"
        );
        assert!(
            tile8_mean < 2.5e-2,
            "{name} transposed coop GEMV tile8 mean_abs_diff={tile8_mean:e} exceeds tolerance"
        );

        let broadcast_us = bench_metal_tensor_op(device, warmup, iters, || {
            x.broadcast_matmul(&weight_t)
                .context("bench broadcast_matmul transposed projection")
        })?;
        let tile4_us = bench_metal_tensor_op(device, warmup, iters, || {
            metal_transposed_coop_gemv_bf16_with_tile(
                &x,
                &weight_t,
                MetalTransposedCoopGemvTile::Tile4,
            )
            .context("bench transposed coop GEMV tile4 projection")
        })?;
        let tile8_us = bench_metal_tensor_op(device, warmup, iters, || {
            metal_transposed_coop_gemv_bf16_with_tile(
                &x,
                &weight_t,
                MetalTransposedCoopGemvTile::Tile8,
            )
            .context("bench transposed coop GEMV tile8 projection")
        })?;

        eprintln!(
            "synthetic Metal Qwen3.5 {name} BF16 transposed GEMV: x=[1,1,{input_dim}] \
             weight_t=[{input_dim},{output_dim}] simdgroups={} warmup={warmup} iters={iters} \
             broadcast_matmul={broadcast_us:.3} us tile4={tile4_us:.3} us tile8={tile8_us:.3} us \
             tile8_vs_tile4={:.3}x tile4_speedup={:.3}x tile8_speedup={:.3}x \
             tile4_max_abs_diff={tile4_max:.6e} tile4_mean_abs_diff={tile4_mean:.6e} \
             tile8_max_abs_diff={tile8_max:.6e} tile8_mean_abs_diff={tile8_mean:.6e}",
            METAL_TRANSPOSED_COOP_GEMV_SIMDGROUPS,
            tile4_us / tile8_us,
            broadcast_us / tile4_us,
            broadcast_us / tile8_us,
        );

        Ok(())
    }

    #[test]
    fn test_metal_rotary_embedding_batched_position_tables_match_rowwise() -> Result<()> {
        let Some(device) = try_new_metal() else {
            eprintln!(
                "Metal unavailable, skipping test_metal_rotary_embedding_batched_position_tables_match_rowwise"
            );
            return Ok(());
        };

        let batch = 3usize;
        let seq_len = 1usize;
        let q_heads = 4usize;
        let k_heads = 2usize;
        let head_dim = 16usize;
        let rotary_dim = 8usize;
        let q = patterned_bf16_4d(batch, seq_len, q_heads, head_dim, &device, 37, 0.03125)?;
        let k = patterned_bf16_4d(batch, seq_len, k_heads, head_dim, &device, 41, 0.03125)?;
        let (cos, sin) = patterned_rotary_tables_3d(batch, seq_len, rotary_dim / 2, &device)?;
        device.synchronize()?;

        assert!(metal_rotary_embedding_supports(
            &q, &k, &cos, &sin, head_dim, rotary_dim
        ));
        let (q_batched, k_batched) =
            metal_rotary_embedding_bf16(&q, &k, &cos, &sin, head_dim, rotary_dim)?;
        let (q_rowwise, k_rowwise) = rotary_qk_rowwise(&q, &k, &cos, &sin, head_dim, rotary_dim)?;
        device.synchronize()?;

        assert_eq!(q_batched.dims(), q.dims());
        assert_eq!(k_batched.dims(), k.dims());
        let q_max = max_abs_diff(&q_rowwise, &q_batched)?;
        let k_max = max_abs_diff(&k_rowwise, &k_batched)?;
        assert!(
            q_max < 1e-6 && k_max < 1e-6,
            "batched rotary position tables differ from rowwise q={q_max:e} k={k_max:e}"
        );

        Ok(())
    }

    #[test]
    #[ignore]
    fn bench_rotary_embedding_batched_position_tables_synthetic() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };
        let warmup = env_usize("KILN_METAL_ROTARY_BATCH_TABLE_BENCH_WARMUP", 20);
        let iters = env_usize("KILN_METAL_ROTARY_BATCH_TABLE_BENCH_ITERS", 200);
        let seq_len = 1usize;
        let q_heads = 16usize;
        let k_heads = 4usize;
        let head_dim = 256usize;
        let rotary_dim = 64usize;

        for batch in [1usize, 2, 4, 8] {
            let q = patterned_bf16_4d(batch, seq_len, q_heads, head_dim, &device, 97, 0.0005)?;
            let k = patterned_bf16_4d(batch, seq_len, k_heads, head_dim, &device, 89, 0.0007)?;
            let (cos, sin) = patterned_rotary_tables_3d(batch, seq_len, rotary_dim / 2, &device)?;
            device.synchronize()?;

            assert!(metal_rotary_embedding_supports(
                &q, &k, &cos, &sin, head_dim, rotary_dim
            ));
            let (q_rowwise, k_rowwise) =
                rotary_qk_rowwise(&q, &k, &cos, &sin, head_dim, rotary_dim)?;
            let (q_batched, k_batched) =
                metal_rotary_embedding_bf16(&q, &k, &cos, &sin, head_dim, rotary_dim)?;
            device.synchronize()?;

            let q_max = max_abs_diff(&q_rowwise, &q_batched)?;
            let k_max = max_abs_diff(&k_rowwise, &k_batched)?;
            assert!(
                q_max < 1e-6 && k_max < 1e-6,
                "Qwen3.5 rotary batch={batch} rowwise max_abs_diff q={q_max:e} k={k_max:e}"
            );

            let rowwise_us = bench_metal_pair_op(&device, warmup, iters, || {
                rotary_qk_rowwise(&q, &k, &cos, &sin, head_dim, rotary_dim)
                    .context("bench rowwise rotary qk per-batch tables")
            })?;
            let batched_us = bench_metal_pair_op(&device, warmup, iters, || {
                metal_rotary_embedding_bf16(&q, &k, &cos, &sin, head_dim, rotary_dim)
                    .context("bench batched rotary qk per-batch tables")
            })?;
            let physical_tokens = batch;

            eprintln!(
                "synthetic Metal Qwen3.5 rotary QK batched position tables decode BF16: \
                 batch={batch} physical_tokens={physical_tokens} \
                 q=[{batch},{seq_len},{q_heads},{head_dim}] \
                 k=[{batch},{seq_len},{k_heads},{head_dim}] \
                 cos/sin=[{batch},{seq_len},{}] warmup={warmup} iters={iters} \
                 rowwise={rowwise_us:.3} us batched={batched_us:.3} us speedup={:.3}x \
                 q_max_abs_diff={q_max:.6e} k_max_abs_diff={k_max:.6e}",
                rotary_dim / 2,
                rowwise_us / batched_us,
            );
        }

        Ok(())
    }

    #[test]
    #[ignore]
    fn bench_attn_gate_decode_batch_synthetic() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };
        let warmup = env_usize("KILN_METAL_ATTN_GATE_BATCH_BENCH_WARMUP", 5);
        let iters = env_usize("KILN_METAL_ATTN_GATE_BATCH_BENCH_ITERS", 20);
        let hidden = 4096usize;

        for batch in [1usize, 2, 4, 8] {
            let x_data: Vec<f32> = (0..batch * hidden)
                .map(|i| ((i % 29) as f32 - 14.0) * 0.03125)
                .collect();
            let gate_data: Vec<f32> = (0..batch * hidden)
                .map(|i| ((i % 37) as f32 - 18.0) * 0.0625)
                .collect();
            let x = Tensor::from_slice(&x_data, (batch, 1usize, hidden), &device)?
                .to_dtype(DType::BF16)?
                .contiguous()?;
            let gate = Tensor::from_slice(&gate_data, (batch, 1usize, hidden), &device)?
                .to_dtype(DType::BF16)?
                .contiguous()?;
            device.synchronize()?;
            assert!(metal_attn_gate_sigmoid_mul_supports(&x, &gate));

            let reference = attn_gate_reference(&x, &gate)?;
            let fused = metal_attn_gate_sigmoid_mul_bf16(&x, &gate)?;
            device.synchronize()?;
            let max = max_abs_diff(&reference, &fused)?;
            let mean = mean_abs_diff(&reference, &fused)?;
            assert!(
                max < 2e-2,
                "Qwen3.5 attention gate batch={batch} max_abs_diff={max:e} exceeds tolerance"
            );
            assert!(
                mean < 2e-3,
                "Qwen3.5 attention gate batch={batch} mean_abs_diff={mean:e} exceeds tolerance"
            );

            let fallback_us = bench_metal_tensor_op(&device, warmup, iters, || {
                attn_gate_reference(&x, &gate).context("bench fallback attention gate")
            })?;
            let fused_us = bench_metal_tensor_op(&device, warmup, iters, || {
                metal_attn_gate_sigmoid_mul_bf16(&x, &gate).context("bench fused attention gate")
            })?;
            let physical_tokens = batch;

            eprintln!(
                "synthetic Metal Qwen3.5 attention gate decode batch BF16: batch={batch} \
                 physical_tokens={physical_tokens} x/gate=[{batch},1,{hidden}] \
                 warmup={warmup} iters={iters} fallback={fallback_us:.3} us \
                 fused={fused_us:.3} us speedup={:.3}x max_abs_diff={max:.6e} \
                 mean_abs_diff={mean:.6e}",
                fallback_us / fused_us,
            );
        }

        Ok(())
    }

    #[test]
    #[ignore]
    fn bench_mlp_gate_up_decode_batch_synthetic() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };
        let warmup = env_usize("KILN_METAL_MLP_GATE_UP_BATCH_BENCH_WARMUP", 5);
        let iters = env_usize("KILN_METAL_MLP_GATE_UP_BATCH_BENCH_ITERS", 20);
        let hidden = 2560usize;
        let intermediate = 9216usize;
        let gate_t = patterned_bf16_2d(hidden, intermediate, &device, 37, 0.0009765625)?;
        let up_t = patterned_bf16_2d(hidden, intermediate, &device, 43, 0.0009765625)?;
        device.synchronize()?;

        for batch in [1usize, 2, 4, 8] {
            let x = patterned_bf16_decode_batch(batch, hidden, &device)?;
            device.synchronize()?;
            assert!(metal_mlp_gate_up_supports(&x, &gate_t, &up_t));

            let reference = mlp_gate_up_reference(&x, &gate_t, &up_t)?;
            let fused = metal_mlp_gate_up_bf16(&x, &gate_t, &up_t)?;
            device.synchronize()?;
            let max = max_abs_diff(&reference, &fused)?;
            let mean = mean_abs_diff(&reference, &fused)?;
            assert!(
                max < 2e-2,
                "Qwen3.5 MLP gate/up batch={batch} max_abs_diff={max:e} exceeds tolerance"
            );
            assert!(
                mean < 2e-3,
                "Qwen3.5 MLP gate/up batch={batch} mean_abs_diff={mean:e} exceeds tolerance"
            );

            let fallback_us = bench_metal_tensor_op(&device, warmup, iters, || {
                mlp_gate_up_reference(&x, &gate_t, &up_t).context("bench fallback MLP gate/up")
            })?;
            let fused_us = bench_metal_tensor_op(&device, warmup, iters, || {
                metal_mlp_gate_up_bf16(&x, &gate_t, &up_t).context("bench fused MLP gate/up")
            })?;
            let physical_tokens = batch;

            eprintln!(
                "synthetic Metal Qwen3.5 MLP gate/up decode batch BF16: batch={batch} \
                 physical_tokens={physical_tokens} x=[{batch},1,{hidden}] \
                 gate_t/up_t=[{hidden},{intermediate}] warmup={warmup} iters={iters} \
                 fallback={fallback_us:.3} us fused={fused_us:.3} us \
                 speedup={:.3}x max_abs_diff={max:.6e} mean_abs_diff={mean:.6e}",
                fallback_us / fused_us,
            );
        }

        Ok(())
    }

    #[test]
    #[ignore]
    fn bench_transposed_coop_gemv_decode_batch_synthetic() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };
        let warmup = env_usize("KILN_METAL_BATCH_GEMV_BENCH_WARMUP", 5);
        let iters = env_usize("KILN_METAL_BATCH_GEMV_BENCH_ITERS", 20);
        let input_dim = 9216usize;
        let output_dim = 2560usize;
        let weight_t = patterned_bf16_2d(input_dim, output_dim, &device, 37, 0.0009765625)?;
        device.synchronize()?;

        for batch in [2usize, 4, 8] {
            let x = patterned_bf16_decode_batch(batch, input_dim, &device)?;
            device.synchronize()?;
            assert!(metal_transposed_coop_gemv_decode_batch_supports(
                &x, &weight_t
            ));

            let reference = x.broadcast_matmul(&weight_t)?;
            let fused = metal_transposed_coop_gemv_bf16(&x, &weight_t)?;
            device.synchronize()?;
            let max = max_abs_diff(&reference, &fused)?;
            let mean = mean_abs_diff(&reference, &fused)?;
            assert!(
                max < 1.5e-1,
                "Qwen3.5 batch transposed coop GEMV batch={batch} max_abs_diff={max:e} exceeds tolerance"
            );
            assert!(
                mean < 2.5e-2,
                "Qwen3.5 batch transposed coop GEMV batch={batch} mean_abs_diff={mean:e} exceeds tolerance"
            );

            let fallback_us = bench_metal_tensor_op(&device, warmup, iters, || {
                x.broadcast_matmul(&weight_t)
                    .context("bench fallback batch transposed GEMV")
            })?;
            let fused_us = bench_metal_tensor_op(&device, warmup, iters, || {
                metal_transposed_coop_gemv_bf16(&x, &weight_t)
                    .context("bench fused batch transposed coop GEMV")
            })?;
            let physical_tokens = batch;

            eprintln!(
                "synthetic Metal Qwen3.5 transposed GEMV decode batch BF16: batch={batch} \
                 physical_tokens={physical_tokens} x=[{batch},1,{input_dim}] \
                 weight_t=[{input_dim},{output_dim}] warmup={warmup} iters={iters} \
                 fallback={fallback_us:.3} us fused_tile8={fused_us:.3} us \
                 speedup={:.3}x max_abs_diff={max:.6e} mean_abs_diff={mean:.6e}",
                fallback_us / fused_us,
            );
        }

        Ok(())
    }

    #[test]
    #[ignore]
    fn bench_lm_head_decode_batch_synthetic() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };
        let warmup = env_usize("KILN_METAL_LM_HEAD_BATCH_BENCH_WARMUP", 1);
        let iters = env_usize("KILN_METAL_LM_HEAD_BATCH_BENCH_ITERS", 3);
        let hidden = 2560usize;
        let vocab = env_usize("KILN_METAL_LM_HEAD_BATCH_BENCH_VOCAB", 248_320);
        let weight_t = Tensor::zeros((hidden, vocab), DType::BF16, &device)?;
        device.synchronize()?;

        for batch in [2usize, 4, 8] {
            let x = patterned_bf16_decode_batch(batch, hidden, &device)?;
            device.synchronize()?;
            assert!(metal_transposed_coop_gemv_decode_batch_supports(
                &x, &weight_t
            ));

            let fused = metal_transposed_coop_gemv_bf16(&x, &weight_t)?;
            device.synchronize()?;
            let reference = x.broadcast_matmul(&weight_t);
            let fallback_check = match reference {
                Ok(reference) => {
                    let max = max_abs_diff(&reference, &fused)?;
                    let mean = mean_abs_diff(&reference, &fused)?;
                    assert!(
                        max < 2e-2,
                        "Qwen3.5 LM-head batch={batch} max_abs_diff={max:e} exceeds tolerance"
                    );
                    assert!(
                        mean < 2e-3,
                        "Qwen3.5 LM-head batch={batch} mean_abs_diff={mean:e} exceeds tolerance"
                    );
                    Some((max, mean))
                }
                Err(err) => {
                    eprintln!(
                        "synthetic Metal Qwen3.5 LM-head decode batch BF16: batch={batch} \
                         fallback_reference_failed={err:#}"
                    );
                    None
                }
            };

            let fused_us = bench_metal_tensor_op(&device, warmup, iters, || {
                metal_transposed_coop_gemv_bf16(&x, &weight_t).context("bench batch LM-head GEMV")
            })?;
            let physical_tokens = batch;
            match fallback_check {
                Some((max, mean)) => {
                    let fallback_us = bench_metal_tensor_op(&device, warmup, iters, || {
                        x.broadcast_matmul(&weight_t)
                            .context("bench fallback LM-head broadcast_matmul")
                    })?;
                    eprintln!(
                        "synthetic Metal Qwen3.5 LM-head decode batch BF16: batch={batch} \
                         physical_tokens={physical_tokens} x=[{batch},1,{hidden}] \
                         weight_t=[{hidden},{vocab}] warmup={warmup} iters={iters} \
                         fallback={fallback_us:.3} us fused={fused_us:.3} us \
                         speedup={:.3}x max_abs_diff={max:.6e} mean_abs_diff={mean:.6e}",
                        fallback_us / fused_us,
                    );
                }
                None => {
                    eprintln!(
                        "synthetic Metal Qwen3.5 LM-head decode batch BF16: batch={batch} \
                         physical_tokens={physical_tokens} x=[{batch},1,{hidden}] \
                         weight_t=[{hidden},{vocab}] warmup={warmup} iters={iters} \
                         fallback=failed fused={fused_us:.3} us"
                    );
                }
            }
        }

        Ok(())
    }

    #[test]
    #[ignore]
    fn bench_paged_kv_write_token_major_decode_batch_synthetic() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };
        let warmup = env_usize("KILN_METAL_PAGED_KV_WRITE_BATCH_BENCH_WARMUP", 20);
        let iters = env_usize("KILN_METAL_PAGED_KV_WRITE_BATCH_BENCH_ITERS", 200);
        let total_slots = env_usize("KILN_METAL_PAGED_KV_WRITE_BATCH_BENCH_SLOTS", 4096);
        let heads = 4usize;
        let head_dim = 256usize;
        let row_stride = heads * head_dim;

        for batch in [1usize, 2, 4, 8] {
            let elems = batch * row_stride;
            let k_data: Vec<f32> = (0..elems)
                .map(|i| ((i % 37) as f32 - 18.0) * 0.015625)
                .collect();
            let v_data: Vec<f32> = (0..elems)
                .map(|i| ((i % 31) as f32 - 15.0) * 0.03125)
                .collect();
            let slots_data: Vec<u32> = (0..batch)
                .map(|idx| ((idx * 17 + 5) % total_slots) as u32)
                .collect();
            let k = Tensor::from_slice(&k_data, (batch, 1usize, heads, head_dim), &device)?
                .to_dtype(DType::BF16)?
                .contiguous()?;
            let v = Tensor::from_slice(&v_data, (batch, 1usize, heads, head_dim), &device)?
                .to_dtype(DType::BF16)?
                .contiguous()?;
            let slots = Tensor::from_slice(&slots_data, batch, &device)?.contiguous()?;
            let k_rows: Vec<Tensor> = (0..batch)
                .map(|idx| k.narrow(0, idx, 1).and_then(|row| row.contiguous()))
                .collect::<candle_core::Result<Vec<_>>>()?;
            let v_rows: Vec<Tensor> = (0..batch)
                .map(|idx| v.narrow(0, idx, 1).and_then(|row| row.contiguous()))
                .collect::<candle_core::Result<Vec<_>>>()?;
            let rowwise_slots: Vec<usize> = slots_data.iter().map(|&slot| slot as usize).collect();
            let mut k_pool_row =
                Tensor::zeros((total_slots, heads, head_dim), DType::BF16, &device)?;
            let mut v_pool_row =
                Tensor::zeros((total_slots, heads, head_dim), DType::BF16, &device)?;
            let mut k_pool_batch =
                Tensor::zeros((total_slots, heads, head_dim), DType::BF16, &device)?;
            let mut v_pool_batch =
                Tensor::zeros((total_slots, heads, head_dim), DType::BF16, &device)?;
            device.synchronize()?;

            assert!(metal_paged_kv_write_token_major_batch_supports(
                &k_pool_batch,
                &v_pool_batch,
                &slots,
                &k,
                &v
            ));
            for idx in 0..batch {
                assert!(metal_paged_kv_write_token_major_supports(
                    &k_pool_row,
                    &v_pool_row,
                    rowwise_slots[idx],
                    &k_rows[idx],
                    &v_rows[idx],
                ));
                metal_paged_kv_write_token_major_bf16(
                    &mut k_pool_row,
                    &mut v_pool_row,
                    rowwise_slots[idx],
                    &k_rows[idx],
                    &v_rows[idx],
                )?;
            }
            metal_paged_kv_write_token_major_batch_bf16(
                &mut k_pool_batch,
                &mut v_pool_batch,
                &slots,
                &k,
                &v,
            )?;
            device.synchronize()?;

            let k_max = max_abs_diff(&k_pool_row, &k_pool_batch)?;
            let v_max = max_abs_diff(&v_pool_row, &v_pool_batch)?;
            assert!(
                k_max < 1e-6 && v_max < 1e-6,
                "Qwen3.5 paged KV batch write batch={batch} max_abs_diff k={k_max:e} v={v_max:e} exceeds tolerance"
            );

            let rowwise_us = bench_metal_unit_op(&device, warmup, iters, || {
                for idx in 0..batch {
                    metal_paged_kv_write_token_major_bf16(
                        &mut k_pool_row,
                        &mut v_pool_row,
                        rowwise_slots[idx],
                        &k_rows[idx],
                        &v_rows[idx],
                    )?;
                }
                Ok(())
            })?;
            let batch_us = bench_metal_unit_op(&device, warmup, iters, || {
                metal_paged_kv_write_token_major_batch_bf16(
                    &mut k_pool_batch,
                    &mut v_pool_batch,
                    &slots,
                    &k,
                    &v,
                )
            })?;
            let physical_tokens = batch;

            eprintln!(
                "synthetic Metal Qwen3.5 paged KV token-major write decode batch BF16: \
                 batch={batch} physical_tokens={physical_tokens} \
                 k/v=[{batch},1,{heads},{head_dim}] pools=[{total_slots},{heads},{head_dim}] \
                 warmup={warmup} iters={iters} rowwise={rowwise_us:.3} us \
                 batched={batch_us:.3} us speedup={:.3}x \
                 k_max_abs_diff={k_max:.6e} v_max_abs_diff={v_max:.6e}",
                rowwise_us / batch_us,
            );
        }

        Ok(())
    }

    #[test]
    #[ignore]
    fn bench_paged_attn_decode_contiguous_batch_synthetic() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };
        let warmup = env_usize("KILN_METAL_PAGED_ATTN_CONTIG_BATCH_BENCH_WARMUP", 3);
        let iters = env_usize("KILN_METAL_PAGED_ATTN_CONTIG_BATCH_BENCH_ITERS", 10);
        let total_slots = env_usize("KILN_METAL_PAGED_ATTN_CONTIG_BATCH_BENCH_SLOTS", 4096);
        let seq_len = env_usize("KILN_METAL_PAGED_ATTN_CONTIG_BATCH_BENCH_SEQ_LEN", 512);
        let q_heads = 16usize;
        let kv_heads = 4usize;
        let head_dim = 256usize;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let kv_elems = total_slots * kv_heads * head_dim;
        let k_data: Vec<f32> = (0..kv_elems)
            .map(|i| ((i % 131) as f32 - 65.0) * 0.0004)
            .collect();
        let v_data: Vec<f32> = (0..kv_elems)
            .map(|i| ((i % 127) as f32 - 63.0) * 0.0006)
            .collect();
        let k_pool = Tensor::from_slice(&k_data, (total_slots, kv_heads, head_dim), &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        let v_pool = Tensor::from_slice(&v_data, (total_slots, kv_heads, head_dim), &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;

        for batch in [1usize, 2, 4, 8] {
            let q_elems = batch * q_heads * head_dim;
            let q_data: Vec<f32> = (0..q_elems)
                .map(|i| ((i % 97) as f32 - 48.0) * 0.0005)
                .collect();
            let start_slots_data: Vec<u32> =
                (0..batch).map(|idx| (17 + idx * 384) as u32).collect();
            let start_slots_usize: Vec<usize> =
                start_slots_data.iter().map(|&slot| slot as usize).collect();
            for &start_slot in &start_slots_usize {
                anyhow::ensure!(
                    start_slot + seq_len <= total_slots,
                    "bench start slot out of range"
                );
            }
            let q = Tensor::from_slice(&q_data, (batch, q_heads, 1usize, head_dim), &device)?
                .to_dtype(DType::BF16)?
                .contiguous()?;
            let start_slots =
                Tensor::from_slice(&start_slots_data, batch, &device)?.contiguous()?;
            device.synchronize()?;

            assert!(metal_paged_attn_decode_contiguous_batch_supports(
                &q,
                &k_pool,
                &v_pool,
                &start_slots,
                seq_len
            ));
            let rowwise = paged_attn_decode_contiguous_rowwise(
                &q,
                &k_pool,
                &v_pool,
                &start_slots_usize,
                seq_len,
                scale,
            )?;
            let batched = metal_paged_attn_decode_contiguous_batch_bf16_d256(
                &q,
                &k_pool,
                &v_pool,
                &start_slots,
                seq_len,
                scale,
            )?;
            device.synchronize()?;

            let max = max_abs_diff(&rowwise, &batched)?;
            let mean = mean_abs_diff(&rowwise, &batched)?;
            assert!(
                max < 1e-6,
                "Qwen3.5 contiguous paged decode batch={batch} max_abs_diff={max:e}"
            );

            let rowwise_us = bench_metal_tensor_op(&device, warmup, iters, || {
                paged_attn_decode_contiguous_rowwise(
                    &q,
                    &k_pool,
                    &v_pool,
                    &start_slots_usize,
                    seq_len,
                    scale,
                )
                .context("bench rowwise contiguous paged decode attention")
            })?;
            let batched_us = bench_metal_tensor_op(&device, warmup, iters, || {
                metal_paged_attn_decode_contiguous_batch_bf16_d256(
                    &q,
                    &k_pool,
                    &v_pool,
                    &start_slots,
                    seq_len,
                    scale,
                )
                .context("bench batched contiguous paged decode attention")
            })?;
            let physical_tokens = batch;

            eprintln!(
                "synthetic Metal Qwen3.5 contiguous paged decode attention batch BF16: \
                 batch={batch} physical_tokens={physical_tokens} \
                 q=[{batch},{q_heads},1,{head_dim}] pools=[{total_slots},{kv_heads},{head_dim}] \
                 seq_len={seq_len} warmup={warmup} iters={iters} \
                 rowwise={rowwise_us:.3} us batched={batched_us:.3} us speedup={:.3}x \
                 max_abs_diff={max:.6e} mean_abs_diff={mean:.6e}",
                rowwise_us / batched_us,
            );
        }

        Ok(())
    }

    #[test]
    #[ignore]
    fn bench_gdn_decode_qkv_conv_norm_decode_batch_synthetic() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };
        let warmup = env_usize("KILN_METAL_GDN_QKV_CONV_NORM_BATCH_BENCH_WARMUP", 5);
        let iters = env_usize("KILN_METAL_GDN_QKV_CONV_NORM_BATCH_BENCH_ITERS", 20);
        let nk = 16usize;
        let dk = 128usize;
        let nv = 32usize;
        let dv = 128usize;
        let kernel_size = 4usize;
        let qk_dim = nk * dk;
        let v_dim = nv * dv;
        let channels = qk_dim * 2 + v_dim;
        let scale = 1.0f32 / (dk as f32).sqrt();
        let weight = patterned_bf16_2d(channels, kernel_size, &device, 37, 0.0009765625)?
            .reshape((channels, 1usize, kernel_size))?
            .contiguous()?;
        device.synchronize()?;

        for batch in [1usize, 2, 4, 8] {
            let mixed_qkv = patterned_bf16_decode_batch(batch, channels, &device)?;
            let state_data: Vec<f32> = (0..(batch * channels * (kernel_size - 1)))
                .map(|i| ((i % 43) as f32 - 21.0) * 0.0009765625)
                .collect();
            let state_init =
                Tensor::from_slice(&state_data, (batch, channels, kernel_size - 1), &device)?
                    .contiguous()?;
            device.synchronize()?;
            assert!(metal_gdn_decode_qkv_conv_norm_supports(
                &mixed_qkv,
                &weight,
                &state_init,
                kernel_size,
                nk,
                dk,
                nv,
                dv
            ));

            let (q_ref, k_ref, v_ref, state_ref) = gdn_decode_qkv_conv_norm_row_fused_reference(
                &mixed_qkv,
                &weight,
                &state_data,
                kernel_size,
                nk,
                dk,
                nv,
                dv,
                scale,
                1e-6,
            )?;
            let mut state_fused =
                Tensor::from_slice(&state_data, (batch, channels, kernel_size - 1), &device)?
                    .contiguous()?;
            let (q_fused, k_fused, v_fused) = metal_gdn_decode_qkv_conv_norm_bf16(
                &mixed_qkv,
                &weight,
                &mut state_fused,
                kernel_size,
                nk,
                dk,
                nv,
                dv,
                scale,
                1e-6,
            )?;
            device.synchronize()?;

            let q_max = max_abs_diff(&q_ref, &q_fused)?;
            let k_max = max_abs_diff(&k_ref, &k_fused)?;
            let v_max = max_abs_diff(&v_ref, &v_fused)?;
            let state_max = max_abs_diff(&state_ref, &state_fused)?;
            assert!(
                q_max < 1e-6 && k_max < 1e-6,
                "Qwen3.5 GDN qkv-conv/norm batch={batch} q/k row-fused max_abs_diff q={q_max:e} k={k_max:e} exceeds tolerance"
            );
            assert!(
                v_max < 1e-6 && state_max < 1e-6,
                "Qwen3.5 GDN qkv-conv/norm batch={batch} v/state row-fused max_abs_diff v={v_max:e} state={state_max:e} exceeds tolerance"
            );

            let mut split_state =
                Tensor::from_slice(&state_data, (batch, channels, kernel_size - 1), &device)?
                    .contiguous()?;
            let split_us = bench_metal_qkv_op(&device, warmup, iters, || {
                gdn_decode_qkv_conv_norm_split_reference(
                    &mixed_qkv,
                    &weight,
                    &mut split_state,
                    kernel_size,
                    nk,
                    dk,
                    nv,
                    dv,
                    scale as f64,
                    1e-6,
                )
                .context("bench split GDN qkv conv/norm")
            })?;
            let mut fused_state =
                Tensor::from_slice(&state_data, (batch, channels, kernel_size - 1), &device)?
                    .contiguous()?;
            let fused_us = bench_metal_qkv_op(&device, warmup, iters, || {
                metal_gdn_decode_qkv_conv_norm_bf16(
                    &mixed_qkv,
                    &weight,
                    &mut fused_state,
                    kernel_size,
                    nk,
                    dk,
                    nv,
                    dv,
                    scale,
                    1e-6,
                )
                .context("bench fused GDN qkv conv/norm")
            })?;
            let physical_tokens = batch;

            eprintln!(
                "synthetic Metal Qwen3.5 GDN qkv-conv/norm decode batch BF16: batch={batch} \
                 physical_tokens={physical_tokens} mixed_qkv=[{batch},1,{channels}] \
                 weight=[{channels},1,{kernel_size}] warmup={warmup} iters={iters} \
                 split={split_us:.3} us fused={fused_us:.3} us speedup={:.3}x \
                 row_q_max_abs_diff={q_max:.6e} row_k_max_abs_diff={k_max:.6e} \
                 row_v_max_abs_diff={v_max:.6e} row_state_max_abs_diff={state_max:.6e}",
                split_us / fused_us,
            );
        }

        Ok(())
    }

    #[test]
    #[ignore]
    fn bench_gdn_in_proj_decode_batch_synthetic() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };
        let warmup = env_usize("KILN_METAL_GDN_IN_PROJ_BATCH_BENCH_WARMUP", 3);
        let iters = env_usize("KILN_METAL_GDN_IN_PROJ_BATCH_BENCH_ITERS", 10);
        let hidden = QWEN35_HIDDEN;
        let qkv_dim = 8192usize;
        let z_dim = 4096usize;
        let nv = 32usize;
        let qkv_t = patterned_bf16_2d(hidden, qkv_dim, &device, 37, 0.0009765625)?;
        let z_t = patterned_bf16_2d(hidden, z_dim, &device, 43, 0.0009765625)?;
        let a_t = patterned_bf16_2d(hidden, nv, &device, 29, 0.00390625)?;
        let b_t = patterned_bf16_2d(hidden, nv, &device, 31, 0.00390625)?;
        device.synchronize()?;

        for batch in [1usize, 2, 4, 8] {
            let x = patterned_bf16_decode_batch(batch, hidden, &device)?;
            assert!(metal_gdn_in_proj_decode_supports(
                &x, &qkv_t, &z_t, &a_t, &b_t
            ));
            let (qkv_ref, z_ref, a_ref, b_ref) =
                gdn_in_proj_broadcast_reference(&x, &qkv_t, &z_t, &a_t, &b_t)?;
            let (qkv_fused, z_fused, a_fused, b_fused) =
                metal_gdn_in_proj_decode_bf16(&x, &qkv_t, &z_t, &a_t, &b_t)?;
            device.synchronize()?;

            assert_eq!(qkv_fused.dims(), &[batch, 1usize, qkv_dim]);
            assert_eq!(z_fused.dims(), &[batch, 1usize, z_dim]);
            assert_eq!(a_fused.dims(), &[batch, 1usize, nv]);
            assert_eq!(b_fused.dims(), &[batch, 1usize, nv]);
            assert!(qkv_fused.is_contiguous());
            assert!(z_fused.is_contiguous());
            assert!(a_fused.is_contiguous());
            assert!(b_fused.is_contiguous());

            let qkv_max = max_abs_diff(&qkv_ref, &qkv_fused)?;
            let z_max = max_abs_diff(&z_ref, &z_fused)?;
            let a_max = max_abs_diff(&a_ref, &a_fused)?;
            let b_max = max_abs_diff(&b_ref, &b_fused)?;
            let max_diff = qkv_max.max(z_max).max(a_max).max(b_max);
            assert!(
                max_diff < 2e-2,
                "Qwen3.5 GDN in-proj batch={batch} max_abs_diff={max_diff:e} exceeds tolerance"
            );

            let broadcast_us = bench_metal_gdn_in_proj_op(&device, warmup, iters, || {
                gdn_in_proj_broadcast_reference(&x, &qkv_t, &z_t, &a_t, &b_t)
                    .context("bench broadcast GDN in-proj")
            })?;
            let fused_us = bench_metal_gdn_in_proj_op(&device, warmup, iters, || {
                metal_gdn_in_proj_decode_bf16(&x, &qkv_t, &z_t, &a_t, &b_t)
                    .context("bench fused GDN in-proj")
            })?;
            let physical_tokens = batch;

            eprintln!(
                "synthetic Metal Qwen3.5 GDN in-proj decode batch BF16: batch={batch} \
                 physical_tokens={physical_tokens} x=[{batch},1,{hidden}] \
                 qkv_t=[{hidden},{qkv_dim}] z_t=[{hidden},{z_dim}] a/b_t=[{hidden},{nv}] \
                 warmup={warmup} iters={iters} broadcast={broadcast_us:.3} us \
                 fused={fused_us:.3} us speedup={:.3}x max_abs_diff={max_diff:.6e}",
                broadcast_us / fused_us,
            );
        }

        Ok(())
    }

    #[test]
    #[ignore]
    fn bench_gdn_decode_gates_recurrent_rmsnorm_decode_batch_synthetic() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };
        let warmup = env_usize(
            "KILN_METAL_GDN_GATES_RECURRENT_RMSNORM_BATCH_BENCH_WARMUP",
            5,
        );
        let iters = env_usize(
            "KILN_METAL_GDN_GATES_RECURRENT_RMSNORM_BATCH_BENCH_ITERS",
            20,
        );
        let q_heads = 16usize;
        let value_heads = 32usize;
        let dk = 128usize;
        let dv = 128usize;
        let seq_len = 1usize;
        let a_log_data: Vec<f32> = (0..value_heads)
            .map(|idx| ((idx % 11) as f32 - 5.0) * 0.04)
            .collect();
        let dt_bias_data: Vec<f32> = (0..value_heads)
            .map(|idx| ((idx % 13) as f32 - 6.0) * 0.05)
            .collect();
        let weight_data: Vec<f32> = (0..dv)
            .map(|idx| 0.9 + ((idx % 17) as f32) * 0.01)
            .collect();
        let a_log = Tensor::from_slice(&a_log_data, value_heads, &device)?.contiguous()?;
        let dt_bias = Tensor::from_slice(&dt_bias_data, value_heads, &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        let weight = Tensor::from_slice(&weight_data, dv, &device)?.contiguous()?;
        device.synchronize()?;

        for batch in [1usize, 2, 4, 8] {
            let qk_elems = batch * seq_len * q_heads * dk;
            let v_elems = batch * seq_len * value_heads * dv;
            let gate_elems = batch * seq_len * value_heads;
            let state_elems = batch * value_heads * dk * dv;
            let q_data: Vec<f32> = (0..qk_elems)
                .map(|idx| ((idx % 31) as f32 - 15.0) * 0.00390625)
                .collect();
            let k_data: Vec<f32> = (0..qk_elems)
                .map(|idx| ((idx % 29) as f32 - 14.0) * 0.00390625)
                .collect();
            let v_data: Vec<f32> = (0..v_elems)
                .map(|idx| ((idx % 37) as f32 - 18.0) * 0.005859375)
                .collect();
            let a_data: Vec<f32> = (0..gate_elems)
                .map(|idx| ((idx % 17) as f32 - 8.0) * 0.08)
                .collect();
            let b_data: Vec<f32> = (0..gate_elems)
                .map(|idx| ((idx % 19) as f32 - 9.0) * 0.07)
                .collect();
            let z_data: Vec<f32> = (0..v_elems)
                .map(|idx| ((idx % 23) as f32 - 11.0) * 0.01)
                .collect();
            let state_data: Vec<f32> = (0..state_elems)
                .map(|idx| ((idx % 41) as f32 - 20.0) * 0.003)
                .collect();

            let q = Tensor::from_slice(&q_data, (batch, seq_len, q_heads, dk), &device)?
                .to_dtype(DType::BF16)?
                .contiguous()?;
            let k = Tensor::from_slice(&k_data, (batch, seq_len, q_heads, dk), &device)?
                .to_dtype(DType::BF16)?
                .contiguous()?;
            let v = Tensor::from_slice(&v_data, (batch, seq_len, value_heads, dv), &device)?
                .to_dtype(DType::BF16)?
                .contiguous()?;
            let a = Tensor::from_slice(&a_data, (batch, seq_len, value_heads), &device)?
                .to_dtype(DType::BF16)?
                .contiguous()?;
            let b = Tensor::from_slice(&b_data, (batch, seq_len, value_heads), &device)?
                .to_dtype(DType::BF16)?
                .contiguous()?;
            let z = Tensor::from_slice(&z_data, (batch, seq_len, value_heads, dv), &device)?
                .to_dtype(DType::BF16)?
                .contiguous()?;

            let mut state_split =
                Tensor::from_slice(&state_data, (batch, value_heads, dk, dv), &device)?
                    .to_dtype(DType::BF16)?
                    .contiguous()?;
            assert!(metal_gdn_decode_gates_recurrent_supports(
                &q,
                &k,
                &v,
                &a,
                &b,
                &a_log,
                &dt_bias,
                &state_split
            ));
            assert!(metal_gdn_decode_gates_recurrent_rmsnorm_supports(
                &q,
                &k,
                &v,
                &a,
                &b,
                &a_log,
                &dt_bias,
                &state_split,
                &z,
                &weight
            ));

            let out_split = gdn_decode_gates_recurrent_rmsnorm_split_reference(
                &q,
                &k,
                &v,
                &a,
                &b,
                &a_log,
                &dt_bias,
                &mut state_split,
                &z,
                &weight,
                1e-6,
            )?;
            let mut state_fused =
                Tensor::from_slice(&state_data, (batch, value_heads, dk, dv), &device)?
                    .to_dtype(DType::BF16)?
                    .contiguous()?;
            let out_fused = metal_gdn_decode_gates_recurrent_rmsnorm_bf16(
                &q,
                &k,
                &v,
                &a,
                &b,
                &a_log,
                &dt_bias,
                &mut state_fused,
                &z,
                &weight,
                1e-6,
            )?;
            device.synchronize()?;

            let out_max = max_abs_diff(&out_split, &out_fused)?;
            let state_max = max_abs_diff(&state_split, &state_fused)?;
            assert!(
                out_max < 5e-2,
                "Qwen3.5 GDN gates+recurrent+rmsnorm batch={batch} out max_abs_diff={out_max:e} exceeds tolerance"
            );
            assert!(
                state_max < 3e-2,
                "Qwen3.5 GDN gates+recurrent+rmsnorm batch={batch} state max_abs_diff={state_max:e} exceeds tolerance"
            );

            let mut split_state =
                Tensor::from_slice(&state_data, (batch, value_heads, dk, dv), &device)?
                    .to_dtype(DType::BF16)?
                    .contiguous()?;
            let split_us = bench_metal_tensor_op(&device, warmup, iters, || {
                gdn_decode_gates_recurrent_rmsnorm_split_reference(
                    &q,
                    &k,
                    &v,
                    &a,
                    &b,
                    &a_log,
                    &dt_bias,
                    &mut split_state,
                    &z,
                    &weight,
                    1e-6,
                )
                .context("bench split GDN gates+recurrent+rmsnorm")
            })?;
            let mut fused_state =
                Tensor::from_slice(&state_data, (batch, value_heads, dk, dv), &device)?
                    .to_dtype(DType::BF16)?
                    .contiguous()?;
            let fused_us = bench_metal_tensor_op(&device, warmup, iters, || {
                metal_gdn_decode_gates_recurrent_rmsnorm_bf16(
                    &q,
                    &k,
                    &v,
                    &a,
                    &b,
                    &a_log,
                    &dt_bias,
                    &mut fused_state,
                    &z,
                    &weight,
                    1e-6,
                )
                .context("bench fused GDN gates+recurrent+rmsnorm")
            })?;
            let physical_tokens = batch;

            eprintln!(
                "synthetic Metal Qwen3.5 GDN gates+recurrent+rmsnorm decode batch BF16: \
                 batch={batch} physical_tokens={physical_tokens} q=[{batch},1,{q_heads},{dk}] \
                 v=[{batch},1,{value_heads},{dv}] state=[{batch},{value_heads},{dk},{dv}] \
                 warmup={warmup} iters={iters} split={split_us:.3} us fused={fused_us:.3} us \
                 speedup={:.3}x out_max_abs_diff={out_max:.6e} state_max_abs_diff={state_max:.6e}",
                split_us / fused_us,
            );
        }

        Ok(())
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
        let decay = big_g_col
            .broadcast_sub(&big_g_row)?
            .exp()?
            .to_dtype(DType::BF16)?;
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

    #[test]
    fn test_lm_head_argmax_matches_materialized_logits() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let enable_prev = std::env::var_os(ENABLE_METAL_LM_HEAD_ARGMAX);
        let disable_prev = std::env::var_os(DISABLE_METAL_LM_HEAD_ARGMAX);
        unsafe {
            std::env::set_var(ENABLE_METAL_LM_HEAD_ARGMAX, "1");
            std::env::remove_var(DISABLE_METAL_LM_HEAD_ARGMAX);
        }

        let result = (|| -> Result<()> {
            let hidden = 128usize;
            let vocab = 257usize;
            let x_data: Vec<f32> = (0..hidden)
                .map(|i| ((i % 23) as f32 - 11.0) * 0.0234375)
                .collect();
            let weight_data: Vec<f32> = (0..(hidden * vocab))
                .map(|i| ((i % 31) as f32 - 15.0) * 0.01953125)
                .collect();

            let x = Tensor::from_slice(&x_data, (1usize, 1usize, hidden), &device)?
                .to_dtype(DType::BF16)?;
            let weight_t = Tensor::from_slice(&weight_data, (hidden, vocab), &device)?
                .to_dtype(DType::BF16)?;

            assert!(metal_lm_head_argmax_supports(&x, &weight_t));
            let logits = metal_lm_head_bf16(&x, &weight_t)?;
            let reference = crate::sampling::greedy_sample(&logits)?;
            let fused = metal_lm_head_argmax_bf16(&x, &weight_t)?;

            assert_eq!(fused, reference);
            Ok(())
        })();

        unsafe {
            match enable_prev {
                Some(value) => std::env::set_var(ENABLE_METAL_LM_HEAD_ARGMAX, value),
                None => std::env::remove_var(ENABLE_METAL_LM_HEAD_ARGMAX),
            }
            match disable_prev {
                Some(value) => std::env::set_var(DISABLE_METAL_LM_HEAD_ARGMAX, value),
                None => std::env::remove_var(DISABLE_METAL_LM_HEAD_ARGMAX),
            }
        }

        result
    }

    #[test]
    fn test_mlp_gate_up_matches_reference() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let hidden = 64usize;
        let intermediate = 97usize;
        let x_data: Vec<f32> = (0..hidden)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.03125)
            .collect();
        let gate_data: Vec<f32> = (0..(hidden * intermediate))
            .map(|i| ((i % 23) as f32 - 11.0) * 0.0078125)
            .collect();
        let up_data: Vec<f32> = (0..(hidden * intermediate))
            .map(|i| ((i % 31) as f32 - 15.0) * 0.0078125)
            .collect();

        let x = Tensor::from_slice(&x_data, (1usize, 1usize, hidden), &device)?
            .to_dtype(DType::BF16)?;
        let gate_t = Tensor::from_slice(&gate_data, (hidden, intermediate), &device)?
            .to_dtype(DType::BF16)?;
        let up_t =
            Tensor::from_slice(&up_data, (hidden, intermediate), &device)?.to_dtype(DType::BF16)?;

        assert!(metal_mlp_gate_up_supports(&x, &gate_t, &up_t));
        let fused = metal_mlp_gate_up_bf16(&x, &gate_t, &up_t)?;

        let gate = x.broadcast_matmul(&gate_t)?;
        let gate_sig = (gate.neg()?.exp()? + 1.0)?.recip()?;
        let gate = (gate * gate_sig)?;
        let up = x.broadcast_matmul(&up_t)?;
        let reference = (gate * up)?;

        assert_eq!(fused.dims(), &[1usize, 1usize, intermediate]);
        assert_eq!(fused.dtype(), DType::BF16);

        let max = max_abs_diff(&reference, &fused)?;
        let mean = mean_abs_diff(&reference, &fused)?;
        assert!(
            max < 2e-2,
            "Metal MLP gate/up max_abs_diff={max:e} exceeds tolerance"
        );
        assert!(
            mean < 2e-3,
            "Metal MLP gate/up mean_abs_diff={mean:e} exceeds tolerance"
        );

        Ok(())
    }

    #[test]
    fn test_mlp_gate_up_decode_batch_matches_reference() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let batch = 4usize;
        let hidden = 64usize;
        let intermediate = 97usize;
        let x = patterned_bf16_decode_batch(batch, hidden, &device)?;
        let gate_t = patterned_bf16_2d(hidden, intermediate, &device, 23, 0.0078125)?;
        let up_t = patterned_bf16_2d(hidden, intermediate, &device, 31, 0.0078125)?;

        assert!(metal_mlp_gate_up_supports(&x, &gate_t, &up_t));
        let fused = metal_mlp_gate_up_bf16(&x, &gate_t, &up_t)?;
        let reference = mlp_gate_up_reference(&x, &gate_t, &up_t)?;

        assert_eq!(fused.dims(), &[batch, 1usize, intermediate]);
        assert_eq!(fused.dtype(), DType::BF16);

        let max = max_abs_diff(&reference, &fused)?;
        let mean = mean_abs_diff(&reference, &fused)?;
        assert!(
            max < 2e-2,
            "Metal MLP gate/up batch max_abs_diff={max:e} exceeds tolerance"
        );
        assert!(
            mean < 2e-3,
            "Metal MLP gate/up batch mean_abs_diff={mean:e} exceeds tolerance"
        );

        Ok(())
    }

    #[test]
    fn test_attn_gate_sigmoid_mul_matches_reference() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let batch = 4usize;
        let hidden = 257usize;
        let x_data: Vec<f32> = (0..batch * hidden)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.03125)
            .collect();
        let gate_data: Vec<f32> = (0..batch * hidden)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.0625)
            .collect();

        let x = Tensor::from_slice(&x_data, (batch, 1usize, hidden), &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        let gate = Tensor::from_slice(&gate_data, (batch, 1usize, hidden), &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;

        assert!(metal_attn_gate_sigmoid_mul_supports(&x, &gate));
        let fused = metal_attn_gate_sigmoid_mul_bf16(&x, &gate)?;

        let reference = attn_gate_reference(&x, &gate)?;

        assert_eq!(fused.dims(), &[batch, 1usize, hidden]);
        assert_eq!(fused.dtype(), DType::BF16);

        let max = max_abs_diff(&reference, &fused)?;
        let mean = mean_abs_diff(&reference, &fused)?;
        assert!(
            max < 2e-2,
            "Metal attn gate sigmoid/mul max_abs_diff={max:e} exceeds tolerance"
        );
        assert!(
            mean < 2e-3,
            "Metal attn gate sigmoid/mul mean_abs_diff={mean:e} exceeds tolerance"
        );

        Ok(())
    }

    #[test]
    fn test_transposed_coop_gemv_matches_broadcast_matmul() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let input_dim = 128usize;
        let output_dim = 133usize;
        let x_data: Vec<f32> = (0..input_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.03125)
            .collect();
        let weight_data: Vec<f32> = (0..(input_dim * output_dim))
            .map(|i| ((i % 29) as f32 - 14.0) * 0.0078125)
            .collect();

        let x = Tensor::from_slice(&x_data, (1usize, 1usize, input_dim), &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        let weight_t = Tensor::from_slice(&weight_data, (input_dim, output_dim), &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;

        assert!(metal_transposed_coop_gemv_supports(&x, &weight_t));
        let reference = x.broadcast_matmul(&weight_t)?;
        let tile4 = metal_transposed_coop_gemv_bf16_with_tile(
            &x,
            &weight_t,
            MetalTransposedCoopGemvTile::Tile4,
        )?;
        let tile8 = metal_transposed_coop_gemv_bf16_with_tile(
            &x,
            &weight_t,
            MetalTransposedCoopGemvTile::Tile8,
        )?;

        assert_eq!(tile4.dims(), &[1usize, 1usize, output_dim]);
        assert_eq!(tile8.dims(), &[1usize, 1usize, output_dim]);
        assert_eq!(tile4.dtype(), DType::BF16);
        assert_eq!(tile8.dtype(), DType::BF16);

        let tile4_max = max_abs_diff(&reference, &tile4)?;
        let tile4_mean = mean_abs_diff(&reference, &tile4)?;
        let tile8_max = max_abs_diff(&reference, &tile8)?;
        let tile8_mean = mean_abs_diff(&reference, &tile8)?;
        assert!(
            tile4_max < 2e-2,
            "Metal transposed coop GEMV tile4 max_abs_diff={tile4_max:e} exceeds tolerance"
        );
        assert!(
            tile4_mean < 3e-3,
            "Metal transposed coop GEMV tile4 mean_abs_diff={tile4_mean:e} exceeds tolerance"
        );
        assert!(
            tile8_max < 2e-2,
            "Metal transposed coop GEMV tile8 max_abs_diff={tile8_max:e} exceeds tolerance"
        );
        assert!(
            tile8_mean < 3e-3,
            "Metal transposed coop GEMV tile8 mean_abs_diff={tile8_mean:e} exceeds tolerance"
        );

        Ok(())
    }

    #[test]
    fn test_transposed_coop_gemv_decode_batch_matches_broadcast_matmul() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let batch = 4usize;
        let input_dim = 128usize;
        let output_dim = 133usize;
        let x = patterned_bf16_decode_batch(batch, input_dim, &device)?;
        let weight_t = patterned_bf16_2d(input_dim, output_dim, &device, 29, 0.0078125)?;

        assert!(metal_transposed_coop_gemv_decode_batch_supports(
            &x, &weight_t
        ));
        let reference = x.broadcast_matmul(&weight_t)?;
        let fused = metal_transposed_coop_gemv_bf16(&x, &weight_t)?;

        assert_eq!(fused.dims(), &[batch, 1usize, output_dim]);
        assert_eq!(fused.dtype(), DType::BF16);

        let max = max_abs_diff(&reference, &fused)?;
        let mean = mean_abs_diff(&reference, &fused)?;
        assert!(
            max < 2e-2,
            "Metal batch transposed coop GEMV max_abs_diff={max:e} exceeds tolerance"
        );
        assert!(
            mean < 3e-3,
            "Metal batch transposed coop GEMV mean_abs_diff={mean:e} exceeds tolerance"
        );

        Ok(())
    }

    #[test]
    fn test_fused_qkv_transposed_coop_gemv_matches_broadcast_matmul() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let input_dim = 128usize;
        let q_output_dim = 65usize;
        let k_output_dim = 37usize;
        let v_output_dim = 41usize;
        let x_data: Vec<f32> = (0..input_dim)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.03125)
            .collect();
        let make_weight = |output_dim: usize, modulus: usize, scale: f32| -> Vec<f32> {
            (0..(input_dim * output_dim))
                .map(|i| ((i % modulus) as f32 - (modulus / 2) as f32) * scale)
                .collect()
        };

        let x = Tensor::from_slice(&x_data, (1usize, 1usize, input_dim), &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        let q_t = Tensor::from_slice(
            &make_weight(q_output_dim, 29, 0.0078125),
            (input_dim, q_output_dim),
            &device,
        )?
        .to_dtype(DType::BF16)?
        .contiguous()?;
        let k_t = Tensor::from_slice(
            &make_weight(k_output_dim, 31, 0.0068359375),
            (input_dim, k_output_dim),
            &device,
        )?
        .to_dtype(DType::BF16)?
        .contiguous()?;
        let v_t = Tensor::from_slice(
            &make_weight(v_output_dim, 37, 0.005859375),
            (input_dim, v_output_dim),
            &device,
        )?
        .to_dtype(DType::BF16)?
        .contiguous()?;

        assert!(metal_fused_qkv_transposed_coop_gemv_supports(
            &x, &q_t, &k_t, &v_t
        ));
        let q_reference = x.broadcast_matmul(&q_t)?;
        let k_reference = x.broadcast_matmul(&k_t)?;
        let v_reference = x.broadcast_matmul(&v_t)?;
        let (q, k, v) = metal_fused_qkv_transposed_coop_gemv_bf16(&x, &q_t, &k_t, &v_t)?;

        assert_eq!(q.dims(), &[1usize, 1usize, q_output_dim]);
        assert_eq!(k.dims(), &[1usize, 1usize, k_output_dim]);
        assert_eq!(v.dims(), &[1usize, 1usize, v_output_dim]);
        assert_eq!(q.dtype(), DType::BF16);
        assert_eq!(k.dtype(), DType::BF16);
        assert_eq!(v.dtype(), DType::BF16);

        let q_max = max_abs_diff(&q_reference, &q)?;
        let k_max = max_abs_diff(&k_reference, &k)?;
        let v_max = max_abs_diff(&v_reference, &v)?;
        let q_mean = mean_abs_diff(&q_reference, &q)?;
        let k_mean = mean_abs_diff(&k_reference, &k)?;
        let v_mean = mean_abs_diff(&v_reference, &v)?;
        assert!(
            q_max < 2e-2 && k_max < 2e-2 && v_max < 2e-2,
            "Metal fused QKV max_abs_diff q={q_max:e} k={k_max:e} v={v_max:e} exceeds tolerance"
        );
        assert!(
            q_mean < 3e-3 && k_mean < 3e-3 && v_mean < 3e-3,
            "Metal fused QKV mean_abs_diff q={q_mean:e} k={k_mean:e} v={v_mean:e} exceeds tolerance"
        );

        Ok(())
    }

    #[test]
    fn test_transposed_coop_gemv_tile8_env_falls_back_to_tile4() {
        unsafe {
            std::env::remove_var(DISABLE_METAL_TRANSPOSED_COOP_GEMV_TILE8);
        }
        assert_eq!(
            metal_transposed_coop_gemv_default_tile(),
            MetalTransposedCoopGemvTile::Tile8
        );

        unsafe {
            std::env::set_var(DISABLE_METAL_TRANSPOSED_COOP_GEMV_TILE8, "1");
        }
        assert_eq!(
            metal_transposed_coop_gemv_default_tile(),
            MetalTransposedCoopGemvTile::Tile4
        );
        unsafe {
            std::env::remove_var(DISABLE_METAL_TRANSPOSED_COOP_GEMV_TILE8);
        }
    }

    #[test]
    #[ignore = "synthetic Metal microbench; run explicitly with --ignored --nocapture"]
    fn bench_transposed_coop_gemv_qwen35_synthetic() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let warmup = env_usize("KILN_METAL_TRANSPOSED_COOP_BENCH_WARMUP", 5);
        let iters = env_usize("KILN_METAL_TRANSPOSED_COOP_BENCH_ITERS", 20);

        bench_transposed_coop_projection_case(
            &device,
            "mlp_gate_or_up",
            QWEN35_HIDDEN,
            QWEN35_INTERMEDIATE,
            warmup,
            iters,
        )?;
        bench_transposed_coop_projection_case(
            &device,
            "down_proj",
            QWEN35_INTERMEDIATE,
            QWEN35_HIDDEN,
            warmup,
            iters,
        )?;
        bench_transposed_coop_projection_case(
            &device,
            "attn_output",
            QWEN35_HIDDEN,
            QWEN35_HIDDEN,
            warmup,
            iters,
        )?;
        bench_transposed_coop_projection_case(
            &device,
            "attn_qkv_like",
            QWEN35_HIDDEN,
            QWEN35_ATTN_QKV_OUT,
            warmup,
            iters,
        )?;

        Ok(())
    }

    #[test]
    fn test_gdn_in_proj_decode_matches_broadcast_matmul() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let batch = 4usize;
        let hidden = 64usize;
        let qkv_dim = 131usize;
        let z_dim = 97usize;
        let nv = 17usize;
        let x_data: Vec<f32> = (0..(batch * hidden))
            .map(|i| ((i % 17) as f32 - 8.0) * 0.03125)
            .collect();
        let qkv_data: Vec<f32> = (0..(hidden * qkv_dim))
            .map(|i| ((i % 29) as f32 - 14.0) * 0.0078125)
            .collect();
        let z_data: Vec<f32> = (0..(hidden * z_dim))
            .map(|i| ((i % 31) as f32 - 15.0) * 0.0078125)
            .collect();
        let a_data: Vec<f32> = (0..(hidden * nv))
            .map(|i| ((i % 23) as f32 - 11.0) * 0.015625)
            .collect();
        let b_data: Vec<f32> = (0..(hidden * nv))
            .map(|i| ((i % 19) as f32 - 9.0) * 0.015625)
            .collect();

        let x =
            Tensor::from_slice(&x_data, (batch, 1usize, hidden), &device)?.to_dtype(DType::BF16)?;
        let qkv_t =
            Tensor::from_slice(&qkv_data, (hidden, qkv_dim), &device)?.to_dtype(DType::BF16)?;
        let z_t = Tensor::from_slice(&z_data, (hidden, z_dim), &device)?.to_dtype(DType::BF16)?;
        let a_t = Tensor::from_slice(&a_data, (hidden, nv), &device)?.to_dtype(DType::BF16)?;
        let b_t = Tensor::from_slice(&b_data, (hidden, nv), &device)?.to_dtype(DType::BF16)?;

        assert!(metal_gdn_in_proj_decode_supports(
            &x, &qkv_t, &z_t, &a_t, &b_t
        ));
        let (qkv_fused, z_fused, a_fused, b_fused) =
            metal_gdn_in_proj_decode_bf16(&x, &qkv_t, &z_t, &a_t, &b_t)?;

        let qkv_ref = x.broadcast_matmul(&qkv_t)?;
        let z_ref = x.broadcast_matmul(&z_t)?;
        let a_ref = x.broadcast_matmul(&a_t)?;
        let b_ref = x.broadcast_matmul(&b_t)?;

        for (name, got, want) in [
            ("qkv", &qkv_fused, &qkv_ref),
            ("z", &z_fused, &z_ref),
            ("a", &a_fused, &a_ref),
            ("b", &b_fused, &b_ref),
        ] {
            assert_eq!(got.dtype(), DType::BF16);
            assert!(
                got.is_contiguous(),
                "GDN in-proj {name} output must stay contiguous"
            );
            let max = max_abs_diff(got, want)?;
            let mean = mean_abs_diff(got, want)?;
            assert!(
                max < 2e-2,
                "GDN in-proj {name} max_abs_diff={max:e} exceeds tolerance"
            );
            assert!(
                mean < 2e-3,
                "GDN in-proj {name} mean_abs_diff={mean:e} exceeds tolerance"
            );
        }

        let x_prefill =
            Tensor::zeros((batch, 2usize, hidden), DType::BF16, &device)?.contiguous()?;
        assert!(!metal_gdn_in_proj_decode_supports(
            &x_prefill, &qkv_t, &z_t, &a_t, &b_t
        ));

        Ok(())
    }

    #[test]
    fn test_gdn_decode_qkv_conv_norm_matches_split_reference() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let nk = 2usize;
        let nv = 4usize;
        let dk = 128usize;
        let dv = 128usize;
        let kernel_size = 4usize;
        let qk_dim = nk * dk;
        let v_dim = nv * dv;
        let channels = qk_dim * 2 + v_dim;
        let scale = 1.0f32 / (dk as f32).sqrt();

        let mixed_data: Vec<f32> = (0..channels)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.03125)
            .collect();
        let weight_data: Vec<f32> = (0..(channels * kernel_size))
            .map(|i| ((i % 17) as f32 - 8.0) * 0.015625)
            .collect();
        let state_data: Vec<f32> = (0..(channels * (kernel_size - 1)))
            .map(|i| ((i % 23) as f32 - 11.0) * 0.0078125)
            .collect();

        let mixed_qkv = Tensor::from_slice(&mixed_data, (1usize, 1usize, channels), &device)?
            .to_dtype(DType::BF16)?;
        let weight = Tensor::from_slice(&weight_data, (channels, 1usize, kernel_size), &device)?
            .to_dtype(DType::BF16)?;
        let mut state_ref =
            Tensor::from_slice(&state_data, (1usize, channels, kernel_size - 1), &device)?;
        let mut state_fused =
            Tensor::from_slice(&state_data, (1usize, channels, kernel_size - 1), &device)?;

        assert!(metal_gdn_decode_qkv_conv_norm_supports(
            &mixed_qkv,
            &weight,
            &state_fused,
            kernel_size,
            nk,
            dk,
            nv,
            dv
        ));

        let conv_ref = metal_causal_conv1d_update_bf16_f32_k4(
            &mixed_qkv.transpose(1, 2)?.contiguous()?,
            &weight,
            &mut state_ref,
            kernel_size,
        )?
        .transpose(1, 2)?;
        let q_ref = conv_ref
            .narrow(2, 0, qk_dim)?
            .reshape((1usize, 1usize, nk, dk))?;
        let k_ref = conv_ref
            .narrow(2, qk_dim, qk_dim)?
            .reshape((1usize, 1usize, nk, dk))?;
        let v_ref = conv_ref
            .narrow(2, 2 * qk_dim, v_dim)?
            .reshape((1usize, 1usize, nv, dv))?
            .to_dtype(DType::BF16)?;
        let (q_ref, k_ref) = gdn_qk_norm_reference(&q_ref, &k_ref, scale as f64, 1e-6)?;

        let (q_fused, k_fused, v_fused) = metal_gdn_decode_qkv_conv_norm_bf16(
            &mixed_qkv,
            &weight,
            &mut state_fused,
            kernel_size,
            nk,
            dk,
            nv,
            dv,
            scale,
            1e-6,
        )?;

        assert_eq!(q_fused.dims(), &[1usize, 1usize, nk, dk]);
        assert_eq!(k_fused.dims(), &[1usize, 1usize, nk, dk]);
        assert_eq!(v_fused.dims(), &[1usize, 1usize, nv, dv]);
        assert_eq!(q_fused.dtype(), DType::BF16);
        assert_eq!(k_fused.dtype(), DType::BF16);
        assert_eq!(v_fused.dtype(), DType::BF16);

        let q_max = max_abs_diff(&q_ref, &q_fused)?;
        let k_max = max_abs_diff(&k_ref, &k_fused)?;
        let v_max = max_abs_diff(&v_ref, &v_fused)?;
        let state_max = max_abs_diff(&state_ref, &state_fused)?;
        assert!(q_max < 2e-2, "fused q max_abs_diff={q_max:e}");
        assert!(k_max < 2e-2, "fused k max_abs_diff={k_max:e}");
        assert!(v_max < 1e-6, "fused v max_abs_diff={v_max:e}");
        assert!(
            state_max < 1e-6,
            "fused conv state max_abs_diff={state_max:e}"
        );

        let prefill_mixed = Tensor::zeros((1usize, 2usize, channels), DType::BF16, &device)?;
        assert!(!metal_gdn_decode_qkv_conv_norm_supports(
            &prefill_mixed,
            &weight,
            &state_fused,
            kernel_size,
            nk,
            dk,
            nv,
            dv
        ));

        Ok(())
    }

    #[test]
    fn test_gdn_decode_qkv_conv_norm_decode_batch_matches_row_fused_reference() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let batch = 4usize;
        let nk = 2usize;
        let nv = 4usize;
        let dk = 128usize;
        let dv = 128usize;
        let kernel_size = 4usize;
        let qk_dim = nk * dk;
        let v_dim = nv * dv;
        let channels = qk_dim * 2 + v_dim;
        let scale = 1.0f32 / (dk as f32).sqrt();

        let mixed_data: Vec<f32> = (0..(batch * channels))
            .map(|i| ((i % 31) as f32 - 15.0) * 0.03125)
            .collect();
        let weight_data: Vec<f32> = (0..(channels * kernel_size))
            .map(|i| ((i % 17) as f32 - 8.0) * 0.015625)
            .collect();
        let state_data: Vec<f32> = (0..(batch * channels * (kernel_size - 1)))
            .map(|i| ((i % 23) as f32 - 11.0) * 0.0078125)
            .collect();

        let mixed_qkv = Tensor::from_slice(&mixed_data, (batch, 1usize, channels), &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        let weight = Tensor::from_slice(&weight_data, (channels, 1usize, kernel_size), &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        let mut state_fused =
            Tensor::from_slice(&state_data, (batch, channels, kernel_size - 1), &device)?
                .contiguous()?;

        assert!(metal_gdn_decode_qkv_conv_norm_supports(
            &mixed_qkv,
            &weight,
            &state_fused,
            kernel_size,
            nk,
            dk,
            nv,
            dv
        ));

        let (q_ref, k_ref, v_ref, state_ref) = gdn_decode_qkv_conv_norm_row_fused_reference(
            &mixed_qkv,
            &weight,
            &state_data,
            kernel_size,
            nk,
            dk,
            nv,
            dv,
            scale,
            1e-6,
        )?;
        let (q_fused, k_fused, v_fused) = metal_gdn_decode_qkv_conv_norm_bf16(
            &mixed_qkv,
            &weight,
            &mut state_fused,
            kernel_size,
            nk,
            dk,
            nv,
            dv,
            scale,
            1e-6,
        )?;

        assert_eq!(q_fused.dims(), &[batch, 1usize, nk, dk]);
        assert_eq!(k_fused.dims(), &[batch, 1usize, nk, dk]);
        assert_eq!(v_fused.dims(), &[batch, 1usize, nv, dv]);
        assert_eq!(q_fused.dtype(), DType::BF16);
        assert_eq!(k_fused.dtype(), DType::BF16);
        assert_eq!(v_fused.dtype(), DType::BF16);

        let q_max = max_abs_diff(&q_ref, &q_fused)?;
        let k_max = max_abs_diff(&k_ref, &k_fused)?;
        let v_max = max_abs_diff(&v_ref, &v_fused)?;
        let state_max = max_abs_diff(&state_ref, &state_fused)?;
        assert!(
            q_max < 1e-6,
            "fused batch q row-fused max_abs_diff={q_max:e}"
        );
        assert!(
            k_max < 1e-6,
            "fused batch k row-fused max_abs_diff={k_max:e}"
        );
        assert!(v_max < 1e-6, "fused batch v max_abs_diff={v_max:e}");
        assert!(
            state_max < 1e-6,
            "fused batch conv state max_abs_diff={state_max:e}"
        );

        Ok(())
    }

    #[test]
    fn test_gdn_prefill_qkv_conv_split_matches_channel_major_conv() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let batch = 4usize;
        let nk = 2usize;
        let dk = 128usize;
        let nv = 4usize;
        let dv = 16usize;
        let qk_dim = nk * dk;
        let v_dim = nv * dv;
        let channels = 2 * qk_dim + v_dim;
        let kernel_size = 4usize;
        for seq_len in [2usize, 3, 16] {
            let x_elems = batch * seq_len * channels;
            let w_elems = channels * kernel_size;
            let s_elems = batch * channels * (kernel_size - 1);
            let x_data: Vec<f32> = (0..x_elems)
                .map(|idx| ((idx % 67) as f32 - 33.0) * 0.003)
                .collect();
            let w_data: Vec<f32> = (0..w_elems)
                .map(|idx| ((idx % 29) as f32 - 14.0) * 0.002)
                .collect();
            let s_data: Vec<f32> = (0..s_elems)
                .map(|idx| ((idx % 31) as f32 - 15.0) * 0.004)
                .collect();

            let x = Tensor::from_slice(&x_data, (batch, seq_len, channels), &device)?
                .to_dtype(DType::BF16)?
                .contiguous()?;
            let weight = Tensor::from_slice(&w_data, (channels, 1usize, kernel_size), &device)?
                .to_dtype(DType::BF16)?
                .contiguous()?;
            let state_init =
                Tensor::from_slice(&s_data, (batch, channels, kernel_size - 1), &device)?
                    .contiguous()?;

            let x_ct = x.transpose(1, 2)?.contiguous()?;
            let mut state_ref = state_init.clone();
            let conv_ref =
                metal_causal_conv1d_prefill_bf16_f32_k4(&x_ct, &weight, &mut state_ref, 4)?
                    .transpose(1, 2)?
                    .contiguous()?;
            let q_ref = conv_ref
                .narrow(2, 0, qk_dim)?
                .reshape((batch, seq_len, nk, dk))?;
            let k_ref = conv_ref
                .narrow(2, qk_dim, qk_dim)?
                .reshape((batch, seq_len, nk, dk))?;
            let v_ref = conv_ref
                .narrow(2, 2 * qk_dim, v_dim)?
                .reshape((batch, seq_len, nv, dv))?
                .to_dtype(DType::BF16)?;

            let mut state_split = state_init.clone();
            assert!(metal_gdn_prefill_qkv_conv_split_supports(
                &x,
                &weight,
                &state_split,
                kernel_size,
                nk,
                dk,
                nv,
                dv
            ));
            let (q, k, v) = metal_gdn_prefill_qkv_conv_split_bf16_f32_k4(
                &x,
                &weight,
                &mut state_split,
                kernel_size,
                nk,
                dk,
                nv,
                dv,
            )?;

            assert_eq!(q.dims(), &[batch, seq_len, nk, dk]);
            assert_eq!(k.dims(), &[batch, seq_len, nk, dk]);
            assert_eq!(v.dims(), &[batch, seq_len, nv, dv]);
            assert_eq!(q.dtype(), DType::F32);
            assert_eq!(k.dtype(), DType::F32);
            assert_eq!(v.dtype(), DType::BF16);
            let q_max = max_abs_diff(&q_ref, &q)?;
            let k_max = max_abs_diff(&k_ref, &k)?;
            let v_max = max_abs_diff(&v_ref, &v)?;
            let state_max = max_abs_diff(&state_ref, &state_split)?;
            eprintln!(
                "gdn prefill qkv conv-split seq_len={seq_len}: q_max={q_max:e} k_max={k_max:e} v_max={v_max:e} state_max={state_max:e}"
            );
            assert!(q_max < 5e-3, "q conv-split diverged at seq_len={seq_len}");
            assert!(k_max < 5e-3, "k conv-split diverged at seq_len={seq_len}");
            assert!(v_max < 5e-3, "v conv-split diverged at seq_len={seq_len}");
            assert!(
                state_max < 1e-6,
                "state conv-split diverged at seq_len={seq_len}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_paged_kv_head_major_read_matches_reference() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let total_slots = 20usize;
        let heads = 2usize;
        let head_dim = 16usize;
        let start_slot = 5usize;
        let seq_len = 7usize;
        let elems = total_slots * heads * head_dim;
        let k_data: Vec<f32> = (0..elems)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.015625)
            .collect();
        let v_data: Vec<f32> = (0..elems)
            .map(|i| ((i % 89) as f32 - 44.0) * 0.03125)
            .collect();
        let k_pool = Tensor::from_slice(&k_data, (total_slots, heads, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let v_pool = Tensor::from_slice(&v_data, (total_slots, heads, head_dim), &device)?
            .to_dtype(DType::BF16)?;

        assert!(metal_paged_kv_head_major_read_supports(
            &k_pool, &v_pool, start_slot, seq_len
        ));
        let (k_fast, v_fast) =
            metal_paged_kv_head_major_read_bf16(&k_pool, &v_pool, start_slot, seq_len)?;
        let k_ref = k_pool
            .narrow(0, start_slot, seq_len)?
            .transpose(0, 1)?
            .contiguous()?
            .unsqueeze(0)?;
        let v_ref = v_pool
            .narrow(0, start_slot, seq_len)?
            .transpose(0, 1)?
            .contiguous()?
            .unsqueeze(0)?;

        assert_eq!(k_fast.dims(), &[1usize, heads, seq_len, head_dim]);
        assert_eq!(v_fast.dims(), &[1usize, heads, seq_len, head_dim]);
        assert!(
            max_abs_diff(&k_fast, &k_ref)? < 1e-6,
            "K fast read diverged from reference"
        );
        assert!(
            max_abs_diff(&v_fast, &v_ref)? < 1e-6,
            "V fast read diverged from reference"
        );

        Ok(())
    }

    #[test]
    fn test_paged_attn_decode_contiguous_matches_candle_sdpa() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let total_slots = 768usize;
        let start_slot = 11usize;
        let seq_len = 512usize;
        let q_heads = 16usize;
        let kv_heads = 4usize;
        let head_dim = 256usize;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let q_data: Vec<f32> = (0..q_heads * head_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.0005)
            .collect();
        let kv_elems = total_slots * kv_heads * head_dim;
        let k_data: Vec<f32> = (0..kv_elems)
            .map(|i| ((i % 131) as f32 - 65.0) * 0.0004)
            .collect();
        let v_data: Vec<f32> = (0..kv_elems)
            .map(|i| ((i % 127) as f32 - 63.0) * 0.0006)
            .collect();

        let q = Tensor::from_slice(&q_data, (1usize, q_heads, 1usize, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let k_pool = Tensor::from_slice(&k_data, (total_slots, kv_heads, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let v_pool = Tensor::from_slice(&v_data, (total_slots, kv_heads, head_dim), &device)?
            .to_dtype(DType::BF16)?;

        assert!(metal_paged_attn_decode_contiguous_supports(
            &q, &k_pool, &v_pool, start_slot, seq_len
        ));
        let fast = metal_paged_attn_decode_contiguous_bf16_d256(
            &q, &k_pool, &v_pool, start_slot, seq_len, scale,
        )?;

        let k_ref = k_pool
            .narrow(0, start_slot, seq_len)?
            .transpose(0, 1)?
            .contiguous()?
            .unsqueeze(0)?;
        let v_ref = v_pool
            .narrow(0, start_slot, seq_len)?
            .transpose(0, 1)?
            .contiguous()?
            .unsqueeze(0)?;
        let reference = candle_nn::ops::sdpa(&q, &k_ref, &v_ref, None, true, scale, 1.0)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((1usize, 1usize, q_heads * head_dim))?;

        assert_eq!(fast.dims(), reference.dims());
        let max = max_abs_diff(&fast, &reference)?;
        let mean = mean_abs_diff(&fast, &reference)?;
        assert!(
            max < 2e-2,
            "contiguous paged decode attention max_abs_diff={max:e}"
        );
        assert!(
            mean < 3e-3,
            "contiguous paged decode attention mean_abs_diff={mean:e}"
        );

        Ok(())
    }

    #[test]
    fn test_paged_attn_decode_contiguous_batch_matches_rowwise() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let batch = 4usize;
        let total_slots = 1024usize;
        let seq_len = 128usize;
        let q_heads = 16usize;
        let kv_heads = 4usize;
        let head_dim = 256usize;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let start_slots_data = vec![11u32, 181, 353, 577];
        let start_slots_usize: Vec<usize> =
            start_slots_data.iter().map(|&slot| slot as usize).collect();

        let q_data: Vec<f32> = (0..batch * q_heads * head_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.0005)
            .collect();
        let kv_elems = total_slots * kv_heads * head_dim;
        let k_data: Vec<f32> = (0..kv_elems)
            .map(|i| ((i % 131) as f32 - 65.0) * 0.0004)
            .collect();
        let v_data: Vec<f32> = (0..kv_elems)
            .map(|i| ((i % 127) as f32 - 63.0) * 0.0006)
            .collect();

        let q = Tensor::from_slice(&q_data, (batch, q_heads, 1usize, head_dim), &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        let k_pool = Tensor::from_slice(&k_data, (total_slots, kv_heads, head_dim), &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        let v_pool = Tensor::from_slice(&v_data, (total_slots, kv_heads, head_dim), &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        let start_slots = Tensor::from_slice(&start_slots_data, batch, &device)?.contiguous()?;

        assert!(metal_paged_attn_decode_contiguous_batch_supports(
            &q,
            &k_pool,
            &v_pool,
            &start_slots,
            seq_len
        ));
        let rowwise = paged_attn_decode_contiguous_rowwise(
            &q,
            &k_pool,
            &v_pool,
            &start_slots_usize,
            seq_len,
            scale,
        )?;
        let batched = metal_paged_attn_decode_contiguous_batch_bf16_d256(
            &q,
            &k_pool,
            &v_pool,
            &start_slots,
            seq_len,
            scale,
        )?;
        device.synchronize()?;

        assert_eq!(batched.dims(), &[batch, 1usize, q_heads * head_dim]);
        let max = max_abs_diff(&rowwise, &batched)?;
        let mean = mean_abs_diff(&rowwise, &batched)?;
        assert!(
            max < 1e-6,
            "contiguous paged decode batch max_abs_diff={max:e}"
        );
        assert!(
            mean < 1e-7,
            "contiguous paged decode batch mean_abs_diff={mean:e}"
        );

        Ok(())
    }

    #[test]
    fn test_paged_kv_head_major_read_append_token_major_matches_reference() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let total_slots = 24usize;
        let heads = 2usize;
        let head_dim = 16usize;
        let start_slot = 3usize;
        let prefix_len = 9usize;
        let tail_len = 5usize;
        let elems = total_slots * heads * head_dim;
        let k_data: Vec<f32> = (0..elems)
            .map(|i| ((i % 101) as f32 - 50.0) * 0.015625)
            .collect();
        let v_data: Vec<f32> = (0..elems)
            .map(|i| ((i % 83) as f32 - 41.0) * 0.03125)
            .collect();
        let k_pool = Tensor::from_slice(&k_data, (total_slots, heads, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let v_pool = Tensor::from_slice(&v_data, (total_slots, heads, head_dim), &device)?
            .to_dtype(DType::BF16)?;

        let tail_elems = tail_len * heads * head_dim;
        let k_tail_data: Vec<f32> = (0..tail_elems)
            .map(|i| ((i % 67) as f32 - 33.0) * 0.0078125)
            .collect();
        let v_tail_data: Vec<f32> = (0..tail_elems)
            .map(|i| ((i % 59) as f32 - 29.0) * 0.015625)
            .collect();
        let k_tail =
            Tensor::from_slice(&k_tail_data, (1usize, tail_len, heads, head_dim), &device)?
                .to_dtype(DType::BF16)?;
        let v_tail =
            Tensor::from_slice(&v_tail_data, (1usize, tail_len, heads, head_dim), &device)?
                .to_dtype(DType::BF16)?;

        assert!(metal_paged_kv_head_major_read_append_token_major_supports(
            &k_pool, &v_pool, start_slot, prefix_len, &k_tail, &v_tail
        ));
        let (k_fast, v_fast) = metal_paged_kv_head_major_read_append_token_major_bf16(
            &k_pool, &v_pool, start_slot, prefix_len, &k_tail, &v_tail,
        )?;

        let prefix_k = k_pool
            .narrow(0, start_slot, prefix_len)?
            .transpose(0, 1)?
            .contiguous()?
            .unsqueeze(0)?;
        let prefix_v = v_pool
            .narrow(0, start_slot, prefix_len)?
            .transpose(0, 1)?
            .contiguous()?
            .unsqueeze(0)?;
        let current_k = k_tail.transpose(1, 2)?.contiguous()?;
        let current_v = v_tail.transpose(1, 2)?.contiguous()?;
        let k_ref = Tensor::cat(&[&prefix_k, &current_k], 2)?;
        let v_ref = Tensor::cat(&[&prefix_v, &current_v], 2)?;

        assert_eq!(
            k_fast.dims(),
            &[1usize, heads, prefix_len + tail_len, head_dim]
        );
        assert_eq!(
            v_fast.dims(),
            &[1usize, heads, prefix_len + tail_len, head_dim]
        );
        assert!(
            max_abs_diff(&k_fast, &k_ref)? < 1e-6,
            "K fast read+append diverged from reference"
        );
        assert!(
            max_abs_diff(&v_fast, &v_ref)? < 1e-6,
            "V fast read+append diverged from reference"
        );

        Ok(())
    }

    #[test]
    fn test_paged_kv_write_token_major_matches_reference() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let total_slots = 8usize;
        let heads = 2usize;
        let head_dim = 16usize;
        let slot = 5usize;
        let elems = heads * head_dim;
        let k_data: Vec<f32> = (0..elems)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.015625)
            .collect();
        let v_data: Vec<f32> = (0..elems)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.03125)
            .collect();
        let k = Tensor::from_slice(&k_data, (1usize, 1usize, heads, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let v = Tensor::from_slice(&v_data, (1usize, 1usize, heads, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let mut k_pool = Tensor::zeros((total_slots, heads, head_dim), DType::BF16, &device)?;
        let mut v_pool = Tensor::zeros((total_slots, heads, head_dim), DType::BF16, &device)?;

        assert!(metal_paged_kv_write_token_major_supports(
            &k_pool, &v_pool, slot, &k, &v
        ));
        metal_paged_kv_write_token_major_bf16(&mut k_pool, &mut v_pool, slot, &k, &v)?;

        let k_written = k_pool.narrow(0, slot, 1)?;
        let v_written = v_pool.narrow(0, slot, 1)?;
        let k_ref = k.squeeze(0)?.contiguous()?;
        let v_ref = v.squeeze(0)?.contiguous()?;

        assert!(
            max_abs_diff(&k_written, &k_ref)? < 1e-6,
            "K fast write diverged from source"
        );
        assert!(
            max_abs_diff(&v_written, &v_ref)? < 1e-6,
            "V fast write diverged from source"
        );

        Ok(())
    }

    #[test]
    fn test_paged_kv_write_token_major_batch_matches_reference() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let total_slots = 16usize;
        let heads = 2usize;
        let head_dim = 16usize;
        let batch = 4usize;
        let elems = batch * heads * head_dim;
        let slots_data = vec![5u32, 2, 11, 7];
        let k_data: Vec<f32> = (0..elems)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.015625)
            .collect();
        let v_data: Vec<f32> = (0..elems)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.03125)
            .collect();
        let k = Tensor::from_slice(&k_data, (batch, 1usize, heads, head_dim), &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        let v = Tensor::from_slice(&v_data, (batch, 1usize, heads, head_dim), &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        let slots = Tensor::from_slice(&slots_data, batch, &device)?.contiguous()?;
        let mut k_pool = Tensor::zeros((total_slots, heads, head_dim), DType::BF16, &device)?;
        let mut v_pool = Tensor::zeros((total_slots, heads, head_dim), DType::BF16, &device)?;

        assert!(metal_paged_kv_write_token_major_batch_supports(
            &k_pool, &v_pool, &slots, &k, &v
        ));
        metal_paged_kv_write_token_major_batch_bf16(&mut k_pool, &mut v_pool, &slots, &k, &v)?;
        device.synchronize()?;

        for (batch_idx, &slot) in slots_data.iter().enumerate() {
            let slot = slot as usize;
            let k_written = k_pool.narrow(0, slot, 1)?;
            let v_written = v_pool.narrow(0, slot, 1)?;
            let k_ref = k.narrow(0, batch_idx, 1)?.squeeze(0)?.contiguous()?;
            let v_ref = v.narrow(0, batch_idx, 1)?.squeeze(0)?.contiguous()?;

            assert!(
                max_abs_diff(&k_written, &k_ref)? < 1e-6,
                "K batch write slot={slot} diverged from source row={batch_idx}"
            );
            assert!(
                max_abs_diff(&v_written, &v_ref)? < 1e-6,
                "V batch write slot={slot} diverged from source row={batch_idx}"
            );
        }

        let zero_ref = Tensor::zeros((1usize, heads, head_dim), DType::BF16, &device)?;
        assert!(
            max_abs_diff(&k_pool.narrow(0, 0, 1)?, &zero_ref)? < 1e-6,
            "K batch write changed an untouched slot"
        );
        assert!(
            max_abs_diff(&v_pool.narrow(0, 0, 1)?, &zero_ref)? < 1e-6,
            "V batch write changed an untouched slot"
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
        let a_data: Vec<f32> = (0..total).map(|i| ((i % 17) as f32 - 8.0) * 0.25).collect();
        let b_data: Vec<f32> = (0..total).map(|i| ((i % 19) as f32 - 9.0) * 0.2).collect();
        let a_log_data: Vec<f32> = (0..nv).map(|i| ((i % 11) as f32 - 5.0) * 0.075).collect();
        let dt_bias_data: Vec<f32> = (0..nv).map(|i| ((i % 13) as f32 - 6.0) * 0.08).collect();

        let a = Tensor::from_slice(&a_data, (batch, seq_len, nv), device)?.to_dtype(DType::BF16)?;
        let b = Tensor::from_slice(&b_data, (batch, seq_len, nv), device)?.to_dtype(DType::BF16)?;
        let a_log = Tensor::from_slice(&a_log_data, nv, device)?;
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
        assert!(g_max < 2e-2, "GDN g parity failed: max_abs_diff={g_max}");
        assert!(g_mean < 5e-3, "GDN g parity failed: mean_abs_diff={g_mean}");

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
    fn test_gdn_full_chunk_forward_matches_fallback_prefill_shape() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let batch = 1usize;
        let heads = 2usize;
        let chunk = 64usize;
        let dk = 64usize;
        let dv = 64usize;
        let c_elems = batch * heads * chunk;
        let cc_elems = batch * heads * chunk * chunk;
        let cdv_elems = batch * heads * chunk * dv;
        let ktd_elems = batch * heads * dk * chunk;
        let state_elems = batch * heads * dk * dv;

        let g_data: Vec<f32> = (0..c_elems)
            .map(|idx| -0.08 + ((idx % 17) as f32) * 0.002)
            .collect();
        let v_data: Vec<f32> = (0..cdv_elems)
            .map(|idx| ((idx % 43) as f32 - 21.0) * 0.0125)
            .collect();
        let kkt_data: Vec<f32> = (0..cc_elems)
            .map(|idx| ((idx % 31) as f32 - 15.0) * 0.004)
            .collect();
        let qkt_data: Vec<f32> = (0..cc_elems)
            .map(|idx| ((idx % 37) as f32 - 18.0) * 0.004)
            .collect();
        let ks_data: Vec<f32> = (0..cdv_elems)
            .map(|idx| ((idx % 47) as f32 - 23.0) * 0.006)
            .collect();
        let qs_data: Vec<f32> = (0..cdv_elems)
            .map(|idx| ((idx % 53) as f32 - 26.0) * 0.006)
            .collect();
        let beta_data: Vec<f32> = (0..c_elems)
            .map(|idx| 0.15 + ((idx % 11) as f32) * 0.01)
            .collect();
        let kt_data: Vec<f32> = (0..ktd_elems)
            .map(|idx| ((idx % 29) as f32 - 14.0) * 0.006)
            .collect();
        let state_data: Vec<f32> = (0..state_elems)
            .map(|idx| ((idx % 41) as f32 - 20.0) * 0.004)
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
        let beta = Tensor::from_slice(&beta_data, (batch, heads, chunk), &device)?
            .to_dtype(DType::BF16)?;
        let k_t = Tensor::from_slice(&kt_data, (batch, heads, dk, chunk), &device)?
            .to_dtype(DType::BF16)?;
        let state_entry = Tensor::from_slice(&state_data, (batch, heads, dk, dv), &device)?
            .to_dtype(DType::BF16)?;
        let mut state_fused = Tensor::from_slice(&state_data, (batch, heads, dk, dv), &device)?
            .to_dtype(DType::BF16)?;
        let mut state_head_last = Tensor::from_slice(&state_data, (batch, heads, dk, dv), &device)?
            .to_dtype(DType::BF16)?;
        let mut state_head_last_strided =
            Tensor::from_slice(&state_data, (batch, heads, dk, dv), &device)?
                .to_dtype(DType::BF16)?;

        assert!(metal_gdn_full_chunk_forward_supports(
            &g,
            &v,
            &kkt,
            &qkt,
            &ks_entry,
            &q_s,
            &beta,
            &k_t,
            &state_fused
        ));

        let (a_strict, b_mask, v_prime, q_s_scaled, decay_last_col, p_last) =
            gdn_chunk_prep_reference(&g, &v, &kkt, &qkt, &ks_entry, &q_s)?;
        let w = metal_gdn_forward_substitution_bf16(&a_strict, &v_prime, &beta)?;
        let out_ref = (q_s_scaled + b_mask.matmul(&w)?)?;
        let decay_last_col_u = decay_last_col.unsqueeze(3)?;
        let w_weighted = w.broadcast_mul(&decay_last_col_u)?.contiguous()?;
        let p_last_u = p_last.unsqueeze(2)?.unsqueeze(3)?;
        let state_ref = (state_entry.broadcast_mul(&p_last_u)? + k_t.matmul(&w_weighted)?)?;

        let out_fused = metal_gdn_full_chunk_forward_bf16(
            &g,
            &v,
            &kkt,
            &qkt,
            &ks_entry,
            &q_s,
            &beta,
            &k_t,
            &mut state_fused,
        )?;
        let seq_len = chunk * 2;
        let t_start = chunk;
        let out_head_last = Tensor::zeros((batch, seq_len, heads, dv), DType::BF16, &device)?;
        assert!(metal_gdn_full_chunk_forward_head_last_supports(
            &g,
            &v,
            &kkt,
            &qkt,
            &ks_entry,
            &q_s,
            &beta,
            &k_t,
            &state_head_last,
            &out_head_last,
            t_start,
            seq_len
        ));
        metal_gdn_full_chunk_forward_head_last_into_bf16(
            &g,
            &v,
            &kkt,
            &qkt,
            &ks_entry,
            &q_s,
            &beta,
            &k_t,
            &mut state_head_last,
            &out_head_last,
            t_start,
            seq_len,
        )?;
        let out_head_last_chunk = out_head_last.narrow(1, t_start, chunk)?.transpose(1, 2)?;

        let mut g_full_data = vec![0.0f32; batch * heads * seq_len];
        let mut v_full_data = vec![0.0f32; batch * heads * seq_len * dv];
        let mut beta_full_data = vec![0.0f32; batch * heads * seq_len];
        let mut k_full_data = vec![0.0f32; batch * heads * seq_len * dk];
        for b in 0..batch {
            for h in 0..heads {
                for t in 0..chunk {
                    let compact_c = (b * heads + h) * chunk + t;
                    let full_t = t_start + t;
                    let full_c = (b * heads + h) * seq_len + full_t;
                    g_full_data[full_c] = g_data[compact_c];
                    beta_full_data[full_c] = beta_data[compact_c];
                    for d in 0..dv {
                        let compact = ((b * heads + h) * chunk + t) * dv + d;
                        let full = ((b * heads + h) * seq_len + full_t) * dv + d;
                        v_full_data[full] = v_data[compact];
                    }
                    for k_idx in 0..dk {
                        let kt_compact = ((b * heads + h) * dk + k_idx) * chunk + t;
                        let k_full = ((b * heads + h) * seq_len + full_t) * dk + k_idx;
                        k_full_data[k_full] = kt_data[kt_compact];
                    }
                }
            }
        }
        let g_full = Tensor::from_slice(&g_full_data, (batch, heads, seq_len), &device)?
            .to_dtype(DType::BF16)?;
        let v_full = Tensor::from_slice(&v_full_data, (batch, heads, seq_len, dv), &device)?
            .to_dtype(DType::BF16)?;
        let beta_full = Tensor::from_slice(&beta_full_data, (batch, heads, seq_len), &device)?
            .to_dtype(DType::BF16)?;
        let k_full = Tensor::from_slice(&k_full_data, (batch, heads, seq_len, dk), &device)?
            .to_dtype(DType::BF16)?;
        let g_view = g_full.narrow(2, t_start, chunk)?;
        let v_view = v_full.narrow(2, t_start, chunk)?;
        let beta_view = beta_full.narrow(2, t_start, chunk)?;
        let k_t_view = k_full.narrow(2, t_start, chunk)?.transpose(2, 3)?;
        assert!(!v_view.is_contiguous());
        assert!(!beta_view.is_contiguous());
        assert!(!k_t_view.is_contiguous());

        let out_head_last_strided =
            Tensor::zeros((batch, seq_len, heads, dv), DType::BF16, &device)?;
        assert!(metal_gdn_full_chunk_forward_head_last_supports(
            &g_view,
            &v_view,
            &kkt,
            &qkt,
            &ks_entry,
            &q_s,
            &beta_view,
            &k_t_view,
            &state_head_last_strided,
            &out_head_last_strided,
            t_start,
            seq_len
        ));
        metal_gdn_full_chunk_forward_head_last_into_bf16(
            &g_view,
            &v_view,
            &kkt,
            &qkt,
            &ks_entry,
            &q_s,
            &beta_view,
            &k_t_view,
            &mut state_head_last_strided,
            &out_head_last_strided,
            t_start,
            seq_len,
        )?;
        let out_head_last_strided_chunk = out_head_last_strided
            .narrow(1, t_start, chunk)?
            .transpose(1, 2)?;

        let out_max = max_abs_diff(&out_ref, &out_fused)?;
        let out_mean = mean_abs_diff(&out_ref, &out_fused)?;
        let state_max = max_abs_diff(&state_ref, &state_fused)?;
        let state_mean = mean_abs_diff(&state_ref, &state_fused)?;
        let head_last_out_max = max_abs_diff(&out_ref, &out_head_last_chunk)?;
        let head_last_out_mean = mean_abs_diff(&out_ref, &out_head_last_chunk)?;
        let head_last_state_max = max_abs_diff(&state_ref, &state_head_last)?;
        let head_last_state_mean = mean_abs_diff(&state_ref, &state_head_last)?;
        let strided_head_last_out_max = max_abs_diff(&out_ref, &out_head_last_strided_chunk)?;
        let strided_head_last_out_mean = mean_abs_diff(&out_ref, &out_head_last_strided_chunk)?;
        let strided_head_last_state_max = max_abs_diff(&state_ref, &state_head_last_strided)?;
        let strided_head_last_state_mean = mean_abs_diff(&state_ref, &state_head_last_strided)?;
        assert!(
            out_max < 2e-2,
            "GDN full-chunk out max_abs_diff={out_max:e} exceeds tolerance"
        );
        assert!(
            out_mean < 2e-3,
            "GDN full-chunk out mean_abs_diff={out_mean:e} exceeds tolerance"
        );
        assert!(
            state_max < 2e-2,
            "GDN full-chunk state max_abs_diff={state_max:e} exceeds tolerance"
        );
        assert!(
            state_mean < 2e-3,
            "GDN full-chunk state mean_abs_diff={state_mean:e} exceeds tolerance"
        );
        assert!(
            head_last_out_max < 2e-2,
            "GDN full-chunk head-last out max_abs_diff={head_last_out_max:e} exceeds tolerance"
        );
        assert!(
            head_last_out_mean < 2e-3,
            "GDN full-chunk head-last out mean_abs_diff={head_last_out_mean:e} exceeds tolerance"
        );
        assert!(
            head_last_state_max < 2e-2,
            "GDN full-chunk head-last state max_abs_diff={head_last_state_max:e} exceeds tolerance"
        );
        assert!(
            head_last_state_mean < 2e-3,
            "GDN full-chunk head-last state mean_abs_diff={head_last_state_mean:e} exceeds tolerance"
        );
        assert!(
            strided_head_last_out_max < 2e-2,
            "GDN full-chunk strided head-last out max_abs_diff={strided_head_last_out_max:e} exceeds tolerance"
        );
        assert!(
            strided_head_last_out_mean < 2e-3,
            "GDN full-chunk strided head-last out mean_abs_diff={strided_head_last_out_mean:e} exceeds tolerance"
        );
        assert!(
            strided_head_last_state_max < 2e-2,
            "GDN full-chunk strided head-last state max_abs_diff={strided_head_last_state_max:e} exceeds tolerance"
        );
        assert!(
            strided_head_last_state_mean < 2e-3,
            "GDN full-chunk strided head-last state mean_abs_diff={strided_head_last_state_mean:e} exceeds tolerance"
        );

        Ok(())
    }

    #[test]
    fn test_gdn_recurrent_prefill_head_last_matches_sequential() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let batch = 1usize;
        let q_heads = 1usize;
        let value_heads = 2usize;
        let gqa_ratio = value_heads / q_heads;
        let seq_len = 8usize;
        let dk = 128usize;
        let dv = 16usize;
        let qk_elems = batch * q_heads * seq_len * dk;
        let v_elems = batch * value_heads * seq_len * dv;
        let gate_elems = batch * value_heads * seq_len;
        let state_elems = batch * value_heads * dk * dv;

        let q_data: Vec<f32> = (0..qk_elems)
            .map(|idx| ((idx % 31) as f32 - 15.0) * 0.004)
            .collect();
        let k_data: Vec<f32> = (0..qk_elems)
            .map(|idx| ((idx % 29) as f32 - 14.0) * 0.004)
            .collect();
        let v_data: Vec<f32> = (0..v_elems)
            .map(|idx| ((idx % 37) as f32 - 18.0) * 0.006)
            .collect();
        let beta_data: Vec<f32> = (0..gate_elems)
            .map(|idx| 0.2 + ((idx % 7) as f32) * 0.03)
            .collect();
        let g_data: Vec<f32> = (0..gate_elems)
            .map(|idx| -0.08 + ((idx % 11) as f32) * 0.003)
            .collect();
        let state_data: Vec<f32> = (0..state_elems)
            .map(|idx| ((idx % 41) as f32 - 20.0) * 0.003)
            .collect();

        let q = Tensor::from_slice(&q_data, (batch, q_heads, seq_len, dk), &device)?
            .to_dtype(DType::BF16)?;
        let k = Tensor::from_slice(&k_data, (batch, q_heads, seq_len, dk), &device)?
            .to_dtype(DType::BF16)?;
        let v = Tensor::from_slice(&v_data, (batch, value_heads, seq_len, dv), &device)?
            .to_dtype(DType::BF16)?;
        let beta = Tensor::from_slice(&beta_data, (batch, value_heads, seq_len), &device)?
            .to_dtype(DType::BF16)?;
        let g = Tensor::from_slice(&g_data, (batch, value_heads, seq_len), &device)?
            .to_dtype(DType::BF16)?;
        let mut state_ref = Tensor::from_slice(&state_data, (batch, value_heads, dk, dv), &device)?
            .to_dtype(DType::BF16)?;
        let mut state_fused =
            Tensor::from_slice(&state_data, (batch, value_heads, dk, dv), &device)?
                .to_dtype(DType::BF16)?;
        let mut state_native =
            Tensor::from_slice(&state_data, (batch, value_heads, dk, dv), &device)?
                .to_dtype(DType::BF16)?;
        let q_ref = q
            .unsqueeze(2)?
            .expand(&[batch, q_heads, gqa_ratio, seq_len, dk])?
            .contiguous()?
            .reshape((batch, value_heads, seq_len, dk))?;
        let k_ref = k
            .unsqueeze(2)?
            .expand(&[batch, q_heads, gqa_ratio, seq_len, dk])?
            .contiguous()?
            .reshape((batch, value_heads, seq_len, dk))?;

        let mut out_chunks = Vec::with_capacity(seq_len);
        let mut state_ref_first = None;
        for t in 0..seq_len {
            let q_t = q_ref.narrow(2, t, 1)?.contiguous()?;
            let k_t = k_ref.narrow(2, t, 1)?.contiguous()?;
            let v_t = v.narrow(2, t, 1)?.contiguous()?;
            let beta_t = beta.narrow(2, t, 1)?.contiguous()?;
            let g_t = g.narrow(2, t, 1)?.contiguous()?;
            let p = g_t.to_dtype(DType::F32)?.exp()?.to_dtype(DType::BF16)?;
            let p_u = p.unsqueeze(3)?;
            let k_t_mat = k_t.transpose(2, 3)?.contiguous()?;
            let ks_entry = k_t.matmul(&state_ref)?;
            let q_s = q_t.matmul(&state_ref)?;
            let v_prime = (v_t - ks_entry.broadcast_mul(&p_u)?)?;
            let w = v_prime.broadcast_mul(&beta_t.unsqueeze(3)?)?;
            let qk = q_t.matmul(&k_t_mat)?;
            let out_t = (q_s.broadcast_mul(&p_u)? + qk.matmul(&w)?)?;
            let state_scaled = state_ref.broadcast_mul(&p_u)?;
            let delta_state = k_t_mat.matmul(&w)?;
            state_ref = (state_scaled + delta_state)?;
            if t == 0 {
                state_ref_first = Some(state_ref.clone());
            }
            out_chunks.push(out_t);
        }
        let out_ref = Tensor::cat(&out_chunks, 2)?.transpose(1, 2)?.contiguous()?;

        assert!(metal_gdn_recurrent_prefill_head_last_supports(
            &q,
            &k,
            &v,
            &beta,
            &g,
            &state_fused
        ));
        let out_fused =
            metal_gdn_recurrent_prefill_head_last_bf16(&q, &k, &v, &beta, &g, &mut state_fused)?;

        let out_max = max_abs_diff(&out_ref, &out_fused)?;
        let out_mean = mean_abs_diff(&out_ref, &out_fused)?;
        let state_max = max_abs_diff(&state_ref, &state_fused)?;
        let state_mean = mean_abs_diff(&state_ref, &state_fused)?;
        assert!(
            out_max < 3e-2,
            "GDN recurrent prefill out max_abs_diff={out_max:e} exceeds tolerance"
        );
        assert!(
            out_mean < 3e-3,
            "GDN recurrent prefill out mean_abs_diff={out_mean:e} exceeds tolerance"
        );
        assert!(
            state_max < 3e-2,
            "GDN recurrent prefill state max_abs_diff={state_max:e} exceeds tolerance"
        );
        assert!(
            state_mean < 3e-3,
            "GDN recurrent prefill state mean_abs_diff={state_mean:e} exceeds tolerance"
        );

        let q_native = q.transpose(1, 2)?.contiguous()?;
        let k_native = k.transpose(1, 2)?.contiguous()?;
        let v_native = v.transpose(1, 2)?.contiguous()?;
        let beta_native = beta.transpose(1, 2)?.contiguous()?;
        let g_native = g.transpose(1, 2)?.contiguous()?;
        assert!(metal_gdn_recurrent_prefill_native_head_last_supports(
            &q_native,
            &k_native,
            &v_native,
            &beta_native,
            &g_native,
            &state_native
        ));
        let out_native = metal_gdn_recurrent_prefill_native_head_last_bf16(
            &q_native,
            &k_native,
            &v_native,
            &beta_native,
            &g_native,
            &mut state_native,
        )?;
        let native_out_max = max_abs_diff(&out_ref, &out_native)?;
        let native_out_mean = mean_abs_diff(&out_ref, &out_native)?;
        let native_state_max = max_abs_diff(&state_ref, &state_native)?;
        let native_state_mean = mean_abs_diff(&state_ref, &state_native)?;
        assert!(
            native_out_max < 3e-2,
            "GDN recurrent native prefill out max_abs_diff={native_out_max:e} exceeds tolerance"
        );
        assert!(
            native_out_mean < 3e-3,
            "GDN recurrent native prefill out mean_abs_diff={native_out_mean:e} exceeds tolerance"
        );
        assert!(
            native_state_max < 3e-2,
            "GDN recurrent native prefill state max_abs_diff={native_state_max:e} exceeds tolerance"
        );
        assert!(
            native_state_mean < 3e-3,
            "GDN recurrent native prefill state mean_abs_diff={native_state_mean:e} exceeds tolerance"
        );

        let q_native_one = q_native.narrow(1, 0, 1)?.contiguous()?;
        let k_native_one = k_native.narrow(1, 0, 1)?.contiguous()?;
        let v_native_one = v_native.narrow(1, 0, 1)?.contiguous()?;
        let beta_native_one = beta_native.narrow(1, 0, 1)?.contiguous()?;
        let g_native_one = g_native.narrow(1, 0, 1)?.contiguous()?;
        let mut state_native_one =
            Tensor::from_slice(&state_data, (batch, value_heads, dk, dv), &device)?
                .to_dtype(DType::BF16)?;
        assert!(metal_gdn_recurrent_prefill_native_head_last_supports(
            &q_native_one,
            &k_native_one,
            &v_native_one,
            &beta_native_one,
            &g_native_one,
            &state_native_one
        ));
        let out_native_one = metal_gdn_recurrent_prefill_native_head_last_bf16(
            &q_native_one,
            &k_native_one,
            &v_native_one,
            &beta_native_one,
            &g_native_one,
            &mut state_native_one,
        )?;
        let out_ref_one = out_ref.narrow(1, 0, 1)?;
        let state_ref_first = state_ref_first.expect("first recurrent state reference");
        let native_one_out_max = max_abs_diff(&out_ref_one, &out_native_one)?;
        let native_one_state_max = max_abs_diff(&state_ref_first, &state_native_one)?;
        assert!(
            native_one_out_max < 3e-2,
            "GDN recurrent native decode out max_abs_diff={native_one_out_max:e} exceeds tolerance"
        );
        assert!(
            native_one_state_max < 3e-2,
            "GDN recurrent native decode state max_abs_diff={native_one_state_max:e} exceeds tolerance"
        );

        Ok(())
    }

    #[test]
    fn test_gdn_decode_gates_recurrent_matches_split_reference() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let batch = 1usize;
        let seq_len = 1usize;
        let q_heads = 1usize;
        let value_heads = 2usize;
        let dk = 128usize;
        let dv = 128usize;
        let qk_elems = batch * seq_len * q_heads * dk;
        let v_elems = batch * seq_len * value_heads * dv;
        let gate_elems = batch * seq_len * value_heads;
        let state_elems = batch * value_heads * dk * dv;

        let q_data: Vec<f32> = (0..qk_elems)
            .map(|idx| ((idx % 31) as f32 - 15.0) * 0.004)
            .collect();
        let k_data: Vec<f32> = (0..qk_elems)
            .map(|idx| ((idx % 29) as f32 - 14.0) * 0.004)
            .collect();
        let v_data: Vec<f32> = (0..v_elems)
            .map(|idx| ((idx % 37) as f32 - 18.0) * 0.006)
            .collect();
        let a_data: Vec<f32> = (0..gate_elems)
            .map(|idx| ((idx % 17) as f32 - 8.0) * 0.08)
            .collect();
        let b_data: Vec<f32> = (0..gate_elems)
            .map(|idx| ((idx % 19) as f32 - 9.0) * 0.07)
            .collect();
        let a_log_data: Vec<f32> = (0..value_heads)
            .map(|idx| ((idx % 11) as f32 - 5.0) * 0.04)
            .collect();
        let dt_bias_data: Vec<f32> = (0..value_heads)
            .map(|idx| ((idx % 13) as f32 - 6.0) * 0.05)
            .collect();
        let state_data: Vec<f32> = (0..state_elems)
            .map(|idx| ((idx % 41) as f32 - 20.0) * 0.003)
            .collect();
        let z_data: Vec<f32> = (0..v_elems)
            .map(|idx| ((idx % 23) as f32 - 11.0) * 0.01)
            .collect();
        let weight_data: Vec<f32> = (0..dv)
            .map(|idx| 0.9 + ((idx % 17) as f32) * 0.01)
            .collect();

        let q = Tensor::from_slice(&q_data, (batch, seq_len, q_heads, dk), &device)?
            .to_dtype(DType::BF16)?;
        let k = Tensor::from_slice(&k_data, (batch, seq_len, q_heads, dk), &device)?
            .to_dtype(DType::BF16)?;
        let v = Tensor::from_slice(&v_data, (batch, seq_len, value_heads, dv), &device)?
            .to_dtype(DType::BF16)?;
        let a = Tensor::from_slice(&a_data, (batch, seq_len, value_heads), &device)?
            .to_dtype(DType::BF16)?;
        let b = Tensor::from_slice(&b_data, (batch, seq_len, value_heads), &device)?
            .to_dtype(DType::BF16)?;
        let a_log = Tensor::from_slice(&a_log_data, value_heads, &device)?;
        let dt_bias =
            Tensor::from_slice(&dt_bias_data, value_heads, &device)?.to_dtype(DType::BF16)?;
        let z = Tensor::from_slice(&z_data, (batch, seq_len, value_heads, dv), &device)?
            .to_dtype(DType::BF16)?;
        let weight = Tensor::from_slice(&weight_data, dv, &device)?;
        let mut state_ref = Tensor::from_slice(&state_data, (batch, value_heads, dk, dv), &device)?
            .to_dtype(DType::BF16)?;
        let mut state_fused =
            Tensor::from_slice(&state_data, (batch, value_heads, dk, dv), &device)?
                .to_dtype(DType::BF16)?;
        let mut state_fused_norm =
            Tensor::from_slice(&state_data, (batch, value_heads, dk, dv), &device)?
                .to_dtype(DType::BF16)?;

        let (beta, g) = gdn_gates_reference(&a, &b, &a_log, &dt_bias)?;
        let out_ref = metal_gdn_recurrent_prefill_native_head_last_bf16(
            &q,
            &k,
            &v,
            &beta,
            &g,
            &mut state_ref,
        )?;

        assert!(metal_gdn_decode_gates_recurrent_supports(
            &q,
            &k,
            &v,
            &a,
            &b,
            &a_log,
            &dt_bias,
            &state_fused
        ));
        let out_fused = metal_gdn_decode_gates_recurrent_bf16(
            &q,
            &k,
            &v,
            &a,
            &b,
            &a_log,
            &dt_bias,
            &mut state_fused,
        )?;

        let out_max = max_abs_diff(&out_ref, &out_fused)?;
        let out_mean = mean_abs_diff(&out_ref, &out_fused)?;
        let state_max = max_abs_diff(&state_ref, &state_fused)?;
        let state_mean = mean_abs_diff(&state_ref, &state_fused)?;
        assert!(
            out_max < 3e-2,
            "GDN decode gates+recurrent out max_abs_diff={out_max:e} exceeds tolerance"
        );
        assert!(
            out_mean < 3e-3,
            "GDN decode gates+recurrent out mean_abs_diff={out_mean:e} exceeds tolerance"
        );
        assert!(
            state_max < 3e-2,
            "GDN decode gates+recurrent state max_abs_diff={state_max:e} exceeds tolerance"
        );
        assert!(
            state_mean < 3e-3,
            "GDN decode gates+recurrent state mean_abs_diff={state_mean:e} exceeds tolerance"
        );

        assert!(metal_gdn_decode_gates_recurrent_rmsnorm_supports(
            &q,
            &k,
            &v,
            &a,
            &b,
            &a_log,
            &dt_bias,
            &state_fused_norm,
            &z,
            &weight
        ));
        let out_norm_ref = metal_gated_rms_norm_bf16(&out_ref, &z, &weight, 1e-6)?;
        let out_norm_fused = metal_gdn_decode_gates_recurrent_rmsnorm_bf16(
            &q,
            &k,
            &v,
            &a,
            &b,
            &a_log,
            &dt_bias,
            &mut state_fused_norm,
            &z,
            &weight,
            1e-6,
        )?;

        let norm_out_max = max_abs_diff(&out_norm_ref, &out_norm_fused)?;
        let norm_out_mean = mean_abs_diff(&out_norm_ref, &out_norm_fused)?;
        let norm_state_max = max_abs_diff(&state_ref, &state_fused_norm)?;
        let norm_state_mean = mean_abs_diff(&state_ref, &state_fused_norm)?;
        assert!(
            norm_out_max < 5e-2,
            "GDN decode gates+recurrent+rmsnorm out max_abs_diff={norm_out_max:e} exceeds tolerance"
        );
        assert!(
            norm_out_mean < 5e-3,
            "GDN decode gates+recurrent+rmsnorm out mean_abs_diff={norm_out_mean:e} exceeds tolerance"
        );
        assert!(
            norm_state_max < 3e-2,
            "GDN decode gates+recurrent+rmsnorm state max_abs_diff={norm_state_max:e} exceeds tolerance"
        );
        assert!(
            norm_state_mean < 3e-3,
            "GDN decode gates+recurrent+rmsnorm state mean_abs_diff={norm_state_mean:e} exceeds tolerance"
        );

        let q_prefill = Tensor::zeros((batch, 2usize, q_heads, dk), DType::BF16, &device)?;
        assert!(!metal_gdn_decode_gates_recurrent_supports(
            &q_prefill,
            &k,
            &v,
            &a,
            &b,
            &a_log,
            &dt_bias,
            &state_fused
        ));

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
    fn test_gdn_qk_norm_gqa_matches_expanded_reference() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };

        let batch = 1usize;
        let seq_len = 128usize;
        let nk = 4usize;
        let gqa_ratio = 4usize;
        let nv = nk * gqa_ratio;
        let hidden = 128usize;
        let q_scale = 1.0 / (hidden as f64).sqrt();
        let eps = 1e-6;

        let q = Tensor::randn(0.0f32, 0.7, (batch, seq_len, nk, hidden), &device)?;
        let k = Tensor::randn(0.0f32, 0.7, (batch, seq_len, nk, hidden), &device)?;
        assert!(metal_gdn_qk_norm_gqa_supports(&q, &k, nv));

        let q_expanded = q
            .unsqueeze(3)?
            .expand(&[batch, seq_len, nk, gqa_ratio, hidden])?
            .contiguous()?
            .reshape((batch, seq_len, nv, hidden))?;
        let k_expanded = k
            .unsqueeze(3)?
            .expand(&[batch, seq_len, nk, gqa_ratio, hidden])?
            .contiguous()?
            .reshape((batch, seq_len, nv, hidden))?;
        let (q_ref, k_ref) = gdn_qk_norm_reference(&q_expanded, &k_expanded, q_scale, eps)?;
        let (q_fused, k_fused) =
            metal_gdn_qk_norm_gqa_f32_bf16(&q, &k, nv, q_scale as f32, eps as f32)?;

        assert_eq!(q_fused.dims(), &[batch, seq_len, nv, hidden]);
        assert_eq!(k_fused.dims(), &[batch, seq_len, nv, hidden]);
        assert_eq!(q_fused.dtype(), DType::BF16);
        assert_eq!(k_fused.dtype(), DType::BF16);

        let q_diff = max_abs_diff(&q_ref, &q_fused)?;
        let k_diff = max_abs_diff(&k_ref, &k_fused)?;
        let q_mean = mean_abs_diff(&q_ref, &q_fused)?;
        let k_mean = mean_abs_diff(&k_ref, &k_fused)?;
        assert!(
            q_diff < 1e-2,
            "GQA GDN Q norm parity failed: max_abs_diff={q_diff}"
        );
        assert!(
            k_diff < 1e-2,
            "GQA GDN K norm parity failed: max_abs_diff={k_diff}"
        );
        assert!(
            q_mean < 1e-3,
            "GQA GDN Q norm parity failed: mean_abs_diff={q_mean}"
        );
        assert!(
            k_mean < 1e-3,
            "GQA GDN K norm parity failed: mean_abs_diff={k_mean}"
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
        let q_expected = Tensor::new(&[1.0f32 / hidden as f32], &device)?
            .to_dtype(DType::BF16)?
            .broadcast_as((1usize, 1usize, 1usize, hidden))?;
        let k_expected = Tensor::new(&[1.0f32 / ((hidden as f32) + eps).sqrt()], &device)?
            .to_dtype(DType::BF16)?
            .broadcast_as((1usize, 1usize, 1usize, hidden))?;
        assert!(max_abs_diff(&q_ones, &q_expected)? < 1e-4);
        assert!(max_abs_diff(&k_ones, &k_expected)? < 1e-4);

        let zeros = Tensor::zeros((1usize, 1usize, 1usize, hidden), DType::F32, &device)?;
        let (q_zero, k_zero) = metal_gdn_qk_norm_f32_bf16(&zeros, &zeros, q_scale, eps)?;
        assert_eq!(
            q_zero.to_dtype(DType::F32)?.max_all()?.to_scalar::<f32>()?,
            0.0
        );
        assert_eq!(
            k_zero.to_dtype(DType::F32)?.max_all()?.to_scalar::<f32>()?,
            0.0
        );

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
