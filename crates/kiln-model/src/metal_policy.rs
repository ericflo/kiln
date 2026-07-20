use anyhow::Result;
use std::sync::OnceLock;

static METAL_KERNEL_POLICY: OnceLock<MetalKernelPolicy> = OnceLock::new();

/// Complete process-lifetime Metal backend-kernel policy.
///
/// The server installs one closed profile before creating a Metal device.
/// Embedded callers that do not install a policy receive the historical
/// native defaults when the Metal backend is first constructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalKernelPolicy {
    pub(crate) sdpa: bool,
    pub(crate) sdpa_full: bool,
    pub(crate) conv1d_prefill: bool,
    pub(crate) conv1d_update: bool,
    pub(crate) gdn_forward_substitution: bool,
    pub(crate) gdn_recurrent: bool,
    pub(crate) gdn_gates: bool,
    pub(crate) gated_rms_norm: bool,
    pub(crate) gdn_in_proj: bool,
    pub(crate) gdn_qk_norm: bool,
    pub(crate) gdn_qkv_conv_norm: bool,
    pub(crate) gdn_prefill_qkv_conv_split: bool,
    pub(crate) gdn_in_proj_row_pair: bool,
    pub(crate) gdn_in_proj_row_quad: bool,
    pub(crate) gdn_in_proj_row_triple: bool,
    pub(crate) gdn_in_proj_serial_vector_load: bool,
    pub(crate) gdn_in_proj_serial_x2_load: bool,
    pub(crate) gdn_prefill_decay_recurrent: bool,
    pub(crate) gdn_prefill_ab_in_proj: bool,
    pub(crate) gdn_decode_gates_recurrent: bool,
    pub(crate) gdn_decode_gates_recurrent_rmsnorm: bool,
    pub(crate) rms_norm: bool,
    pub(crate) mlp_gate_up_fusion: bool,
    pub(crate) mlp_gate_up_row_pair: bool,
    pub(crate) mlp_gate_up_row_quad: bool,
    pub(crate) mlp_gate_up_row_triple: bool,
    pub(crate) mlp_gate_up_row_quad_vector_load: bool,
    pub(crate) mlp_gate_up_serial_vector_load: bool,
    pub(crate) mlp_gate_up_serial_dedicated: bool,
    pub(crate) mlp_silu_mul: bool,
    pub(crate) attn_gate_fusion: bool,
    pub(crate) fused_qkv_proj: bool,
    pub(crate) lora_delta_decode: bool,
    pub(crate) lm_head_argmax: bool,
    pub(crate) lm_head_argmax_rows: bool,
    pub(crate) lm_head_argmax_gpu_reduce: bool,
    pub(crate) lm_head_sample: bool,
    pub(crate) paged_attn_decode_contiguous: bool,
    pub(crate) paged_kv_write_token_major: bool,
    pub(crate) transposed_coop_gemv: bool,
    pub(crate) transposed_coop_gemv_tile8: bool,
    pub(crate) transposed_coop_gemv_tile16: bool,
    pub(crate) transposed_coop_gemv_row_pair: bool,
    pub(crate) transposed_coop_gemv_row_quad: bool,
    pub(crate) transposed_coop_gemv_row_quad_tile8: bool,
    pub(crate) transposed_coop_gemv_row_triple_tile8: bool,
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
impl MetalKernelPolicy {
    /// Metal routes that were active with no legacy environment overrides.
    pub const fn native_default() -> Self {
        Self {
            sdpa: true,
            sdpa_full: true,
            conv1d_prefill: true,
            conv1d_update: true,
            gdn_forward_substitution: true,
            gdn_recurrent: true,
            gdn_gates: true,
            gated_rms_norm: true,
            gdn_in_proj: true,
            gdn_qk_norm: true,
            gdn_qkv_conv_norm: true,
            gdn_prefill_qkv_conv_split: true,
            gdn_in_proj_row_pair: true,
            gdn_in_proj_row_quad: true,
            gdn_in_proj_row_triple: true,
            gdn_in_proj_serial_vector_load: true,
            gdn_in_proj_serial_x2_load: true,
            gdn_prefill_decay_recurrent: true,
            gdn_prefill_ab_in_proj: true,
            gdn_decode_gates_recurrent: true,
            gdn_decode_gates_recurrent_rmsnorm: true,
            rms_norm: true,
            mlp_gate_up_fusion: true,
            mlp_gate_up_row_pair: true,
            mlp_gate_up_row_quad: true,
            mlp_gate_up_row_triple: true,
            mlp_gate_up_row_quad_vector_load: true,
            mlp_gate_up_serial_vector_load: true,
            mlp_gate_up_serial_dedicated: true,
            mlp_silu_mul: true,
            attn_gate_fusion: true,
            fused_qkv_proj: true,
            lora_delta_decode: true,
            // The custom chunk/reduce path was historically explicit opt-in.
            lm_head_argmax: false,
            lm_head_argmax_rows: true,
            lm_head_argmax_gpu_reduce: true,
            lm_head_sample: true,
            paged_attn_decode_contiguous: true,
            paged_kv_write_token_major: true,
            transposed_coop_gemv: true,
            transposed_coop_gemv_tile8: true,
            transposed_coop_gemv_tile16: true,
            transposed_coop_gemv_row_pair: true,
            transposed_coop_gemv_row_quad: true,
            transposed_coop_gemv_row_quad_tile8: true,
            transposed_coop_gemv_row_triple_tile8: true,
        }
    }

    /// Decline every governed Metal route and use its portable fallback.
    pub const fn portable_fallback() -> Self {
        Self {
            sdpa: false,
            sdpa_full: false,
            conv1d_prefill: false,
            conv1d_update: false,
            gdn_forward_substitution: false,
            gdn_recurrent: false,
            gdn_gates: false,
            gated_rms_norm: false,
            gdn_in_proj: false,
            gdn_qk_norm: false,
            gdn_qkv_conv_norm: false,
            gdn_prefill_qkv_conv_split: false,
            gdn_in_proj_row_pair: false,
            gdn_in_proj_row_quad: false,
            gdn_in_proj_row_triple: false,
            gdn_in_proj_serial_vector_load: false,
            gdn_in_proj_serial_x2_load: false,
            gdn_prefill_decay_recurrent: false,
            gdn_prefill_ab_in_proj: false,
            gdn_decode_gates_recurrent: false,
            gdn_decode_gates_recurrent_rmsnorm: false,
            rms_norm: false,
            mlp_gate_up_fusion: false,
            mlp_gate_up_row_pair: false,
            mlp_gate_up_row_quad: false,
            mlp_gate_up_row_triple: false,
            mlp_gate_up_row_quad_vector_load: false,
            mlp_gate_up_serial_vector_load: false,
            mlp_gate_up_serial_dedicated: false,
            mlp_silu_mul: false,
            attn_gate_fusion: false,
            fused_qkv_proj: false,
            lora_delta_decode: false,
            lm_head_argmax: false,
            lm_head_argmax_rows: false,
            lm_head_argmax_gpu_reduce: false,
            lm_head_sample: false,
            paged_attn_decode_contiguous: false,
            paged_kv_write_token_major: false,
            transposed_coop_gemv: false,
            transposed_coop_gemv_tile8: false,
            transposed_coop_gemv_tile16: false,
            transposed_coop_gemv_row_pair: false,
            transposed_coop_gemv_row_quad: false,
            transposed_coop_gemv_row_quad_tile8: false,
            transposed_coop_gemv_row_triple_tile8: false,
        }
    }

    #[cfg(test)]
    const fn routes(self) -> [bool; 46] {
        [
            self.sdpa,
            self.sdpa_full,
            self.conv1d_prefill,
            self.conv1d_update,
            self.gdn_forward_substitution,
            self.gdn_recurrent,
            self.gdn_gates,
            self.gated_rms_norm,
            self.gdn_in_proj,
            self.gdn_qk_norm,
            self.gdn_qkv_conv_norm,
            self.gdn_prefill_qkv_conv_split,
            self.gdn_in_proj_row_pair,
            self.gdn_in_proj_row_quad,
            self.gdn_in_proj_row_triple,
            self.gdn_in_proj_serial_vector_load,
            self.gdn_in_proj_serial_x2_load,
            self.gdn_prefill_decay_recurrent,
            self.gdn_prefill_ab_in_proj,
            self.gdn_decode_gates_recurrent,
            self.gdn_decode_gates_recurrent_rmsnorm,
            self.rms_norm,
            self.mlp_gate_up_fusion,
            self.mlp_gate_up_row_pair,
            self.mlp_gate_up_row_quad,
            self.mlp_gate_up_row_triple,
            self.mlp_gate_up_row_quad_vector_load,
            self.mlp_gate_up_serial_vector_load,
            self.mlp_gate_up_serial_dedicated,
            self.mlp_silu_mul,
            self.attn_gate_fusion,
            self.fused_qkv_proj,
            self.lora_delta_decode,
            self.lm_head_argmax,
            self.lm_head_argmax_rows,
            self.lm_head_argmax_gpu_reduce,
            self.lm_head_sample,
            self.paged_attn_decode_contiguous,
            self.paged_kv_write_token_major,
            self.transposed_coop_gemv,
            self.transposed_coop_gemv_tile8,
            self.transposed_coop_gemv_tile16,
            self.transposed_coop_gemv_row_pair,
            self.transposed_coop_gemv_row_quad,
            self.transposed_coop_gemv_row_quad_tile8,
            self.transposed_coop_gemv_row_triple_tile8,
        ]
    }
}

impl Default for MetalKernelPolicy {
    fn default() -> Self {
        Self::native_default()
    }
}

/// Install process-lifetime Metal kernel policy. Reinstalling the same value
/// is idempotent; conflicting installation fails instead of changing routes.
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub fn install_metal_kernel_policy(policy: MetalKernelPolicy) -> Result<()> {
    match METAL_KERNEL_POLICY.set(policy) {
        Ok(()) => Ok(()),
        Err(policy) if METAL_KERNEL_POLICY.get() == Some(&policy) => Ok(()),
        Err(_) => anyhow::bail!("Metal kernel policy was already installed with a different value"),
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(crate) fn current_metal_kernel_policy() -> MetalKernelPolicy {
    *METAL_KERNEL_POLICY.get_or_init(MetalKernelPolicy::default)
}

#[cfg(test)]
mod tests {
    use super::MetalKernelPolicy;

    #[test]
    fn profiles_cover_every_backend_route_and_preserve_argmax_default() {
        let routes = MetalKernelPolicy::native_default().routes();
        assert_eq!(routes.len(), 46);
        assert_eq!(routes.iter().filter(|enabled| **enabled).count(), 45);
        assert!(!routes[33], "custom LM-head argmax remains default-off");
        assert_eq!(MetalKernelPolicy::portable_fallback().routes(), [false; 46]);
    }
}
