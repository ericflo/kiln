//! CUDA backend: FlashAttention-2 and Gated DeltaNet fused kernels.
//!
//! Wraps the vendored `kiln-flash-attn` and `kiln-gdn-kernel` crates.
//! `Ok(None)` responses route the caller to the portable candle path.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use super::{BackendRuntime, TrainingCapabilities};
use crate::lora_loader::{LoraProjectionWeights, compute_lora_delta};

static CUDA_RESIDENT_TENSOR_IDS: OnceLock<Mutex<HashSet<candle_core::TensorId>>> = OnceLock::new();

fn with_cuda_resident_ids<R>(f: impl FnOnce(&mut HashSet<candle_core::TensorId>) -> R) -> R {
    let registry = CUDA_RESIDENT_TENSOR_IDS.get_or_init(|| Mutex::new(HashSet::new()));
    let mut guard = registry
        .lock()
        .expect("CUDA resident TensorId registry mutex poisoned");
    f(&mut guard)
}

#[derive(Debug)]
pub struct CudaBackend {
    device: Device,
    /// Cached at construction: reading env vars per decode step × 24 GDN layers
    /// shows up in decode NVTX captures. Env vars don't change at runtime.
    gdn_enabled: bool,
    /// Same pattern: cache the env-var read. The fused gates kernel is
    /// gated behind its own kill switch so it can be disabled independently.
    gdn_gates_enabled: bool,
    /// Kill switch for the fused GDN gated RMSNorm kernel (decode/prefill
    /// kiln/gdn/gated_norm region).
    gdn_gated_rms_norm_enabled: bool,
    /// Experimental fused native-MTP decode GDN gates + recurrent update.
    /// Opt-in only until output parity is proven.
    gdn_decode_fused_enabled: bool,
    /// CUDA fused decode supports native GQA Q/K heads; this avoids expanding
    /// Q/K to value_heads before the fused recurrent decode kernel.
    gdn_decode_unexpanded_qk_enabled: bool,
    /// Fuses GDN decode Q/K L2-normalization into the gates+recurrent kernel,
    /// avoiding the separate tiny qk_norm launch in the single-token path.
    gdn_decode_qk_norm_recurrent_enabled: bool,
    /// Fuses GDN decode Q/K L2-normalization, gates, recurrent update, and
    /// gated RMSNorm into one single-token CUDA launch.
    gdn_decode_qk_norm_recurrent_rmsnorm_enabled: bool,
    /// Kill switch for the fused causal_conv1d_update kernel (decode
    /// kiln/gdn/conv region). When off, forward.rs falls back to the
    /// candle to_f32/cat/sum/narrow chain.
    fused_conv1d_enabled: bool,
    /// Forward-only CUDA LoRA delta/add for decode. Training declines because
    /// tracked LoRA tensors need autograd.
    lora_decode_add_enabled: bool,
}

impl CudaBackend {
    pub fn new(device: Device) -> Self {
        debug_assert!(device.is_cuda(), "CudaBackend created on non-CUDA device");
        let gdn_enabled = std::env::var("KILN_DISABLE_GDN_KERNEL").is_err();
        let gdn_gates_enabled =
            gdn_enabled && std::env::var("KILN_DISABLE_FUSED_GDN_GATES").is_err();
        let gdn_gated_rms_norm_enabled =
            gdn_enabled && std::env::var("KILN_DISABLE_FUSED_GDN_GATED_RMS_NORM").is_err();
        let fused_conv1d_enabled = std::env::var("KILN_DISABLE_FUSED_CONV1D").is_err();
        let gdn_decode_fused_enabled = gdn_gates_enabled
            && gdn_gated_rms_norm_enabled
            && std::env::var("KILN_DISABLE_FUSED_GDN_DECODE").is_err();
        let gdn_decode_unexpanded_qk_enabled = gdn_decode_fused_enabled
            && std::env::var("KILN_DISABLE_GDN_DECODE_UNEXPANDED_QK").is_err();
        let gdn_decode_qk_norm_recurrent_enabled = gdn_decode_unexpanded_qk_enabled
            && std::env::var("KILN_DISABLE_CUDA_GDN_DECODE_QK_NORM_RECURRENT").is_err();
        let gdn_decode_qk_norm_recurrent_rmsnorm_enabled = gdn_decode_qk_norm_recurrent_enabled
            && std::env::var("KILN_DISABLE_CUDA_GDN_DECODE_QK_NORM_RECURRENT_RMSNORM").is_err();
        let lora_decode_add_enabled = std::env::var("KILN_DISABLE_CUDA_LORA_DECODE_ADD").is_err();
        Self {
            device,
            gdn_enabled,
            gdn_gates_enabled,
            gdn_gated_rms_norm_enabled,
            gdn_decode_fused_enabled,
            gdn_decode_unexpanded_qk_enabled,
            gdn_decode_qk_norm_recurrent_enabled,
            gdn_decode_qk_norm_recurrent_rmsnorm_enabled,
            fused_conv1d_enabled,
            lora_decode_add_enabled,
        }
    }

    pub fn training_capabilities_static() -> TrainingCapabilities {
        TrainingCapabilities {
            projection_training: "backend-routed candle CUDA autograd",
            flce_loss: "FLCE CustomOp on CUDA tensors; no full logits by default",
            rmsnorm_training: "CUDA CustomOp2 behind 47 GiB autograd VRAM gate",
            resident_activation: "TensorId lifecycle registry; candle CUDA tensors are canonical",
            lora_delta_training: "registered candle CUDA autograd; fused lora_decode_add declines tracked tensors",
            sgd_step: "candle CUDA Var::set fallback",
            adamw_step: "candle CUDA Var::set fallback",
            native_training: "not implemented",
        }
    }
}

impl BackendRuntime for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn training_capabilities(&self) -> TrainingCapabilities {
        Self::training_capabilities_static()
    }

    fn supports_resident_activation(&self) -> bool {
        true
    }

    fn register_resident_activation(&self, tensor: &Tensor) -> Result<()> {
        with_cuda_resident_ids(|ids| {
            ids.insert(tensor.id());
        });
        Ok(())
    }

    fn evict_resident_activation(&self, tensor: &Tensor) {
        with_cuda_resident_ids(|ids| {
            ids.remove(&tensor.id());
        });
    }

    fn update_resident_activation(&self, tensor: &Tensor) -> Result<()> {
        with_cuda_resident_ids(|ids| {
            ids.insert(tensor.id());
        });
        Ok(())
    }

    fn has_resident_activation(&self, tensor: &Tensor) -> bool {
        with_cuda_resident_ids(|ids| ids.contains(&tensor.id()))
    }

    fn supports_flash_attn_prefill(&self) -> bool {
        true
    }

    fn supports_flash_attn_paged_decode(&self) -> bool {
        true
    }

    fn supports_gdn_forward_substitution(&self) -> bool {
        self.gdn_enabled
    }

    fn supports_gdn_recurrent_step(&self) -> bool {
        self.gdn_enabled
    }

    fn supports_gdn_chunk_prep(&self) -> bool {
        self.gdn_enabled
    }

    fn supports_gdn_chunk_scan(&self) -> bool {
        self.gdn_enabled
    }

    fn supports_gdn_full_chunk_forward(&self) -> bool {
        self.gdn_enabled
    }

    fn supports_gdn_decode_gates_recurrent_unexpanded_qk(&self) -> bool {
        self.gdn_decode_unexpanded_qk_enabled
    }

    fn supports_gdn_decode_qk_norm_gates_recurrent(&self) -> bool {
        self.gdn_decode_qk_norm_recurrent_enabled
    }

    fn flash_attn_prefill(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<Tensor>> {
        // The vendored CUDA kernel hard-errors on non-BF16. Decline here so
        // the caller falls back to the portable path instead of bubbling a
        // hard error up for non-BF16 test configs.
        if q.dtype() != DType::BF16 {
            return Ok(None);
        }
        let out = kiln_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
            .context("flash_attn kernel failed")?;
        Ok(Some(out))
    }

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
        if q.dtype() != DType::BF16 {
            return Ok(None);
        }
        let out = kiln_flash_attn::flash_attn_paged_decode(
            q,
            k_pool,
            v_pool,
            block_table,
            total_seqlen_k,
            page_block_size,
            softmax_scale,
            causal,
        )
        .context("flash_attn_paged_decode kernel failed")?;
        Ok(Some(out))
    }

    fn flash_attn_paged_decode_contiguous_batch_dyn_seqlen(
        &self,
        q: &Tensor,
        k_pool: &Tensor,
        v_pool: &Tensor,
        block_table: &Tensor,
        seqused_k: &Tensor,
        max_seqlen_k: usize,
        page_block_size: usize,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<Tensor>> {
        if q.dtype() != DType::BF16 {
            return Ok(None);
        }
        let out = kiln_flash_attn::flash_attn_paged_decode_dyn_seqlen(
            q,
            k_pool,
            v_pool,
            block_table,
            seqused_k,
            None,
            max_seqlen_k,
            page_block_size,
            softmax_scale,
            causal,
        )
        .context("flash_attn_paged_decode_dyn_seqlen kernel failed")?;
        Ok(Some(out))
    }

    fn gdn_forward_substitution(
        &self,
        a_strict: &Tensor,
        v_prime: &Tensor,
        beta: &Tensor,
    ) -> Result<Option<Tensor>> {
        if a_strict.dtype() != DType::BF16 {
            return Ok(None);
        }
        let out = kiln_gdn_kernel::gdn_forward_substitution(a_strict, v_prime, beta)?;
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
        if q.dtype() != DType::BF16 {
            return Ok(None);
        }
        let out = kiln_gdn_kernel::gdn_recurrent_forward(q, k, v, beta, g, state)?;
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
        if !kiln_gdn_kernel::gdn_chunk_prep_supports(g, v, kkt, qkt, ks_entry, q_s) {
            return Ok(None);
        }
        let out = kiln_gdn_kernel::gdn_chunk_prep(g, v, kkt, qkt, ks_entry, q_s)
            .context("gdn_chunk_prep kernel failed")?;
        Ok(Some(out))
    }

    fn gdn_chunk_scan(
        &self,
        a_strict: &Tensor,
        b_mask: &Tensor,
        v_prime: &Tensor,
        q_s_scaled: &Tensor,
        beta: &Tensor,
        decay_last_col: &Tensor,
    ) -> Result<Option<(Tensor, Tensor)>> {
        if !kiln_gdn_kernel::gdn_chunk_scan_supports(
            a_strict,
            b_mask,
            v_prime,
            q_s_scaled,
            beta,
            decay_last_col,
        ) {
            return Ok(None);
        }
        let out = kiln_gdn_kernel::gdn_chunk_scan(
            a_strict,
            b_mask,
            v_prime,
            q_s_scaled,
            beta,
            decay_last_col,
        )
        .context("gdn_chunk_scan kernel failed")?;
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
        if !kiln_gdn_kernel::gdn_full_chunk_forward_supports(
            g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state,
        ) {
            return Ok(None);
        }
        let out = kiln_gdn_kernel::gdn_full_chunk_forward(
            g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state,
        )
        .context("gdn_full_chunk_forward kernel failed")?;
        Ok(Some(out))
    }

    #[allow(clippy::too_many_arguments)]
    fn gdn_decode_gates_recurrent(
        &self,
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
        eps: f64,
    ) -> Result<Option<Tensor>> {
        if !self.gdn_decode_fused_enabled {
            return Ok(None);
        }
        if !kiln_gdn_kernel::gdn_decode_gates_recurrent_supports(
            q, k, v, a, b, a_log, dt_bias, state, z, weight,
        ) {
            tracing::debug!(
                q_shape = ?q.shape(), q_dtype = ?q.dtype(),
                k_shape = ?k.shape(), k_dtype = ?k.dtype(),
                v_shape = ?v.shape(), v_dtype = ?v.dtype(),
                a_shape = ?a.shape(), a_dtype = ?a.dtype(),
                b_shape = ?b.shape(), b_dtype = ?b.dtype(),
                a_log_shape = ?a_log.shape(), a_log_dtype = ?a_log.dtype(),
                dt_bias_shape = ?dt_bias.shape(), dt_bias_dtype = ?dt_bias.dtype(),
                state_shape = ?state.shape(), state_dtype = ?state.dtype(), state_contiguous = state.is_contiguous(),
                z_shape = ?z.shape(), z_dtype = ?z.dtype(),
                weight_shape = ?weight.shape(), weight_dtype = ?weight.dtype(),
                "CUDA gdn_decode_gates_recurrent declined; using split decode path"
            );
            return Ok(None);
        }
        let out = kiln_gdn_kernel::gdn_decode_gates_recurrent(
            q, k, v, a, b, a_log, dt_bias, state, z, weight, eps as f32,
        )
        .context("gdn_decode_gates_recurrent kernel failed")?;
        Ok(Some(out))
    }

    #[allow(clippy::too_many_arguments)]
    fn gdn_decode_qk_norm_gates_recurrent(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        a: &Tensor,
        b: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
        state: &mut Tensor,
        q_scale: f64,
        qk_eps: f64,
    ) -> Result<Option<Tensor>> {
        if !self.gdn_decode_qk_norm_recurrent_enabled {
            return Ok(None);
        }
        if !kiln_gdn_kernel::gdn_decode_qk_norm_gates_recurrent_supports(
            q, k, v, a, b, a_log, dt_bias, state,
        ) {
            tracing::debug!(
                q_shape = ?q.shape(), q_dtype = ?q.dtype(),
                k_shape = ?k.shape(), k_dtype = ?k.dtype(),
                v_shape = ?v.shape(), v_dtype = ?v.dtype(),
                a_shape = ?a.shape(), a_dtype = ?a.dtype(),
                b_shape = ?b.shape(), b_dtype = ?b.dtype(),
                a_log_shape = ?a_log.shape(), a_log_dtype = ?a_log.dtype(),
                dt_bias_shape = ?dt_bias.shape(), dt_bias_dtype = ?dt_bias.dtype(),
                state_shape = ?state.shape(), state_dtype = ?state.dtype(), state_contiguous = state.is_contiguous(),
                "CUDA gdn_decode_qk_norm_gates_recurrent declined; using split qk_norm path"
            );
            return Ok(None);
        }
        let out = kiln_gdn_kernel::gdn_decode_qk_norm_gates_recurrent(
            q,
            k,
            v,
            a,
            b,
            a_log,
            dt_bias,
            state,
            q_scale as f32,
            qk_eps as f32,
        )
        .context("gdn_decode_qk_norm_gates_recurrent kernel failed")?;
        Ok(Some(out))
    }

    #[allow(clippy::too_many_arguments)]
    fn gdn_decode_qk_norm_gates_recurrent_rmsnorm(
        &self,
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
        q_scale: f64,
        qk_eps: f64,
        rms_eps: f64,
    ) -> Result<Option<Tensor>> {
        if !self.gdn_decode_qk_norm_recurrent_rmsnorm_enabled {
            return Ok(None);
        }
        if !kiln_gdn_kernel::gdn_decode_qk_norm_gates_recurrent_rmsnorm_supports(
            q, k, v, a, b, a_log, dt_bias, state, z, weight,
        ) {
            tracing::debug!(
                q_shape = ?q.shape(), q_dtype = ?q.dtype(),
                k_shape = ?k.shape(), k_dtype = ?k.dtype(),
                v_shape = ?v.shape(), v_dtype = ?v.dtype(),
                a_shape = ?a.shape(), a_dtype = ?a.dtype(),
                b_shape = ?b.shape(), b_dtype = ?b.dtype(),
                a_log_shape = ?a_log.shape(), a_log_dtype = ?a_log.dtype(),
                dt_bias_shape = ?dt_bias.shape(), dt_bias_dtype = ?dt_bias.dtype(),
                state_shape = ?state.shape(), state_dtype = ?state.dtype(), state_contiguous = state.is_contiguous(),
                z_shape = ?z.shape(), z_dtype = ?z.dtype(),
                weight_shape = ?weight.shape(), weight_dtype = ?weight.dtype(),
                "CUDA gdn_decode_qk_norm_gates_recurrent_rmsnorm declined; using split gated_norm path"
            );
            return Ok(None);
        }
        let out = kiln_gdn_kernel::gdn_decode_qk_norm_gates_recurrent_rmsnorm(
            q,
            k,
            v,
            a,
            b,
            a_log,
            dt_bias,
            state,
            z,
            weight,
            q_scale as f32,
            qk_eps as f32,
            rms_eps as f32,
        )
        .context("gdn_decode_qk_norm_gates_recurrent_rmsnorm kernel failed")?;
        Ok(Some(out))
    }

    fn supports_gdn_gates(&self) -> bool {
        self.gdn_gates_enabled
    }

    fn gdn_gates(
        &self,
        a: &Tensor,
        b: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
    ) -> Result<Option<(Tensor, Tensor)>> {
        let dims = a.dims();
        let is_t1_decode = dims.len() >= 2 && dims[dims.len() - 2] == 1;
        if !is_t1_decode && std::env::var("KILN_DISABLE_CUDA_GDN_PREFILL_GATES").is_ok() {
            tracing::debug!(
                a_shape = ?a.shape(),
                a_log_dtype = ?a_log.dtype(),
                dt_bias_dtype = ?dt_bias.dtype(),
                "CUDA prefill gdn_gates disabled; using Candle fallback"
            );
            return Ok(None);
        }
        if let Some(reason) = kiln_gdn_kernel::gdn_gates_decline_reason(a, b, a_log, dt_bias) {
            tracing::debug!(
                reason,
                a_shape = ?a.shape(),
                b_shape = ?b.shape(),
                a_log_shape = ?a_log.shape(),
                dt_bias_shape = ?dt_bias.shape(),
                a_dtype = ?a.dtype(),
                b_dtype = ?b.dtype(),
                a_log_dtype = ?a_log.dtype(),
                dt_bias_dtype = ?dt_bias.dtype(),
                "CUDA gdn_gates declined; using Candle fallback"
            );
            return Ok(None);
        }
        let (beta, g) =
            kiln_gdn_kernel::gdn_gates(a, b, a_log, dt_bias).context("gdn_gates kernel failed")?;
        Ok(Some((beta, g)))
    }

    fn supports_gdn_gated_rms_norm(&self) -> bool {
        self.gdn_gated_rms_norm_enabled
    }

    fn lora_decode_add(
        &self,
        base: &Tensor,
        x: &Tensor,
        a: &Tensor,
        b: &Tensor,
        scale: f32,
    ) -> Result<Option<Tensor>> {
        if !self.lora_decode_add_enabled
            || base.track_op()
            || x.track_op()
            || a.track_op()
            || b.track_op()
            || !kiln_rmsnorm_kernel::supports_lora_decode_add(base, x, a, b)
        {
            return Ok(None);
        }
        let out = kiln_rmsnorm_kernel::lora_decode_add(base, x, a, b, scale)
            .context("cuda lora_decode_add kernel failed")?;
        Ok(Some(out))
    }

    fn linear_prefill_apply(&self, x: &Tensor, weight_t: &Tensor) -> Result<Option<Tensor>> {
        if !matches!(x.device(), Device::Cuda(_))
            || !matches!(weight_t.device(), Device::Cuda(_))
            || x.dims().is_empty()
            || weight_t.dims().len() != 2
            || *x.dims().last().unwrap() != weight_t.dims()[0]
        {
            return Ok(None);
        }

        static FIRST_CUDA_LINEAR_PREFILL_LOGGED: OnceLock<()> = OnceLock::new();
        FIRST_CUDA_LINEAR_PREFILL_LOGGED.get_or_init(|| {
            tracing::info!(
                x_shape = ?x.dims(),
                weight_t_shape = ?weight_t.dims(),
                tracked = x.track_op() || weight_t.track_op(),
                "CudaBackend::linear_prefill_apply first call (candle CUDA autograd)"
            );
        });

        Ok(Some(x.broadcast_matmul(weight_t)?))
    }

    fn lora_delta_resident(
        &self,
        x: &Tensor,
        a: &Tensor,
        b: &Tensor,
        scale: f32,
    ) -> Result<Option<Tensor>> {
        if !matches!(x.device(), Device::Cuda(_))
            || !matches!(a.device(), Device::Cuda(_))
            || !matches!(b.device(), Device::Cuda(_))
            || !self.has_resident_activation(a)
            || !self.has_resident_activation(b)
        {
            return Ok(None);
        }

        let proj = LoraProjectionWeights {
            a: a.clone(),
            b: b.clone(),
        };
        let delta = compute_lora_delta(x, &proj, scale)
            .context("cuda registered LoRA delta via candle CUDA autograd failed")?;

        static FIRST_CUDA_LORA_DELTA_LOGGED: OnceLock<()> = OnceLock::new();
        FIRST_CUDA_LORA_DELTA_LOGGED.get_or_init(|| {
            tracing::info!(
                x_shape = ?x.dims(),
                a_shape = ?a.dims(),
                b_shape = ?b.dims(),
                scale,
                "CudaBackend::lora_delta_resident first call (candle CUDA autograd)"
            );
        });

        Ok(Some(delta))
    }

    fn gdn_gated_rms_norm(
        &self,
        x: &Tensor,
        z: &Tensor,
        weight: &Tensor,
        eps: f64,
    ) -> Result<Option<Tensor>> {
        if !self.gdn_gated_rms_norm_enabled {
            return Ok(None);
        }
        if !kiln_gdn_kernel::gdn_gated_rms_norm_supports(x, z, weight) {
            return Ok(None);
        }
        let out = kiln_gdn_kernel::gdn_gated_rms_norm(x, z, weight, eps as f32)
            .context("gdn_gated_rms_norm kernel failed")?;
        Ok(Some(out))
    }

    fn supports_causal_conv1d_update(&self) -> bool {
        self.fused_conv1d_enabled
    }

    fn supports_causal_conv1d_prefill(&self) -> bool {
        self.fused_conv1d_enabled
    }

    fn causal_conv1d_update(
        &self,
        x: &Tensor,
        weight: &Tensor,
        conv_state: &mut Tensor,
        kernel_size: usize,
    ) -> Result<Option<Tensor>> {
        if !self.fused_conv1d_enabled {
            return Ok(None);
        }
        if !kiln_conv1d_kernel::supports(x, weight, conv_state, kernel_size) {
            return Ok(None);
        }
        let out = kiln_conv1d_kernel::causal_conv1d_update(x, weight, conv_state, kernel_size)
            .context("causal_conv1d_update kernel failed")?;
        Ok(Some(out))
    }

    fn causal_conv1d_prefill(
        &self,
        x: &Tensor,
        weight: &Tensor,
        conv_state: &mut Tensor,
        kernel_size: usize,
    ) -> Result<Option<Tensor>> {
        if !self.fused_conv1d_enabled {
            return Ok(None);
        }
        if !kiln_conv1d_kernel::supports_prefill(x, weight, conv_state, kernel_size) {
            return Ok(None);
        }
        let out = kiln_conv1d_kernel::causal_conv1d_prefill(x, weight, conv_state, kernel_size)
            .context("causal_conv1d_prefill kernel failed")?;
        Ok(Some(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_backend() -> CudaBackend {
        CudaBackend {
            device: Device::Cpu,
            gdn_enabled: false,
            gdn_gates_enabled: false,
            gdn_gated_rms_norm_enabled: false,
            gdn_decode_fused_enabled: false,
            gdn_decode_unexpanded_qk_enabled: false,
            gdn_decode_qk_norm_recurrent_enabled: false,
            gdn_decode_qk_norm_recurrent_rmsnorm_enabled: false,
            fused_conv1d_enabled: false,
            lora_decode_add_enabled: false,
        }
    }

    #[test]
    fn cuda_resident_activation_registry_lifecycle() -> Result<()> {
        let backend = test_backend();
        let tensor = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;

        assert!(backend.supports_resident_activation());
        assert!(!backend.has_resident_activation(&tensor));

        backend.register_resident_activation(&tensor)?;
        assert!(backend.has_resident_activation(&tensor));

        backend.evict_resident_activation(&tensor);
        assert!(!backend.has_resident_activation(&tensor));

        backend.update_resident_activation(&tensor)?;
        assert!(backend.has_resident_activation(&tensor));

        Ok(())
    }

    #[test]
    fn cuda_linear_prefill_apply_matches_candle_cuda_matmul() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_linear_prefill_apply_matches_candle_cuda_matmul: {err}");
                return Ok(());
            }
        };
        let backend = CudaBackend::new(device.clone());

        let x = Tensor::from_slice(&[1.0f32, -2.0, 0.5, 3.0, 4.0, -1.0], (2, 3), &device)?;
        let w = Tensor::from_slice(
            &[0.5f32, 1.0, -1.5, 2.0, -0.25, 0.75, 1.25, -0.5, 2.0, -1.0, 0.0, 0.5],
            (3, 4),
            &device,
        )?;

        let routed = backend
            .linear_prefill_apply(&x, &w)?
            .expect("CUDA linear_prefill_apply should accept CUDA tensors");
        let expected = x.broadcast_matmul(&w)?;
        assert_eq!(routed.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
        Ok(())
    }

    #[test]
    fn cuda_registered_lora_delta_matches_candle_cuda_reference() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_registered_lora_delta_matches_candle_cuda_reference: {err}");
                return Ok(());
            }
        };
        let backend = CudaBackend::new(device.clone());

        let x = Tensor::from_slice(&[0.5f32, -1.0, 2.0, 1.5, 0.25, -0.75], (2, 3), &device)?;
        let a = Tensor::from_slice(&[0.25f32, -0.5, 1.0, 1.5, 0.0, -1.0], (2, 3), &device)?;
        let b = Tensor::from_slice(&[1.0f32, -0.25, 0.5, 0.75, -1.0, 0.25, 0.0, 1.5], (4, 2), &device)?;
        let scale = 0.5;

        assert!(backend.lora_delta_resident(&x, &a, &b, scale)?.is_none());
        backend.register_resident_activation(&a)?;
        backend.register_resident_activation(&b)?;

        let routed = backend
            .lora_delta_resident(&x, &a, &b, scale)?
            .expect("registered CUDA LoRA delta should engage");
        let expected = compute_lora_delta(
            &x,
            &LoraProjectionWeights {
                a: a.clone(),
                b: b.clone(),
            },
            scale,
        )?;

        assert_eq!(routed.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
        Ok(())
    }
}
