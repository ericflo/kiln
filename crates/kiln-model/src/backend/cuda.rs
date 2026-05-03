//! CUDA backend: FlashAttention-2 and Gated DeltaNet fused kernels.
//!
//! Wraps the vendored `kiln-flash-attn` and `kiln-gdn-kernel` crates.
//! `Ok(None)` responses route the caller to the portable candle path.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

use super::BackendRuntime;

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
    /// Kill switch for the fused causal_conv1d_update kernel (decode
    /// kiln/gdn/conv region). When off, forward.rs falls back to the
    /// candle to_f32/cat/sum/narrow chain.
    fused_conv1d_enabled: bool,
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
            && std::env::var("KILN_ENABLE_FUSED_GDN_DECODE").is_ok()
            && std::env::var("KILN_DISABLE_FUSED_GDN_DECODE").is_err();
        Self {
            device,
            gdn_enabled,
            gdn_gates_enabled,
            gdn_gated_rms_norm_enabled,
            gdn_decode_fused_enabled,
            fused_conv1d_enabled,
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
            return Ok(None);
        }
        let out = kiln_gdn_kernel::gdn_decode_gates_recurrent(
            q, k, v, a, b, a_log, dt_bias, state, z, weight, eps as f32,
        )
        .context("gdn_decode_gates_recurrent kernel failed")?;
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
        if !is_t1_decode {
            tracing::debug!(
                a_shape = ?a.shape(),
                a_log_dtype = ?a_log.dtype(),
                dt_bias_dtype = ?dt_bias.dtype(),
                "CUDA gdn_gates fused path is decode-only; using Candle fallback"
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
