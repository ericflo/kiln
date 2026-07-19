use anyhow::Result;
use std::sync::OnceLock;

static ROCM_KERNEL_POLICY: OnceLock<RocmKernelPolicy> = OnceLock::new();

/// Complete process-lifetime ROCm model-kernel policy.
///
/// The server maps its closed profile vocabulary to one of these policies
/// before device creation. Embedders that do not install a policy receive the
/// qualified production defaults on first ROCm use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RocmKernelPolicy {
    pub(crate) full_attn_qkv_in_proj: bool,
    pub(crate) gdn_ab_in_proj: bool,
    pub(crate) gdn_prefill_ab_in_proj: bool,
    pub(crate) gdn_prefill_gates: bool,
    pub(crate) head_major_prefill: bool,
    pub(crate) gdn: bool,
    pub(crate) gdn_gates: bool,
    pub(crate) gdn_gated_rms_norm: bool,
    pub(crate) gdn_decode_fused: bool,
    pub(crate) gdn_decode_unexpanded_qk: bool,
    pub(crate) gdn_decode_qk_norm_recurrent: bool,
    pub(crate) gdn_decode_qk_norm_recurrent_rmsnorm: bool,
    pub(crate) fused_conv1d: bool,
    pub(crate) lora_decode_add: bool,
    pub(crate) gdn_full_chunk_forward_multiblock: bool,
    pub(crate) attn_decode_qkv_prep: bool,
    pub(crate) fused_paged_decode: bool,
    pub(crate) paged_decode_dyn_seqlen_batch: bool,
    pub(crate) fused_mlp_silu_mul: bool,
    pub(crate) fused_mlp_gate_up_prefill: bool,
    pub(crate) fused_rmsnorm: bool,
    pub(crate) long_flash_attn: bool,
    pub(crate) native_rectangular_causal_flash: bool,
    pub(crate) w8_projection: bool,
    pub(crate) w8a8_projection: bool,
    pub(crate) w8_swiglu: bool,
    pub(crate) w8_sampled_lm_head: bool,
    pub(crate) w8a8_sampled_lm_head: bool,
    pub(crate) gqa_sdpa_f32_materialized: bool,
    pub(crate) split_q_gate_f32_output: bool,
    pub(crate) training_mlp_chunking: bool,
    pub(crate) training_mlp_chunk_tokens: usize,
    pub(crate) split_q_gate_training: bool,
    pub(crate) split_q_gate_output_chunk_features: usize,
    pub(crate) split_q_gate_row_tile_tokens: usize,
}

#[cfg_attr(not(feature = "rocm"), allow(dead_code))]
impl RocmKernelPolicy {
    /// Qualified Strix Halo production policy.
    pub const fn qualified() -> Self {
        Self {
            full_attn_qkv_in_proj: true,
            gdn_ab_in_proj: true,
            gdn_prefill_ab_in_proj: true,
            gdn_prefill_gates: true,
            head_major_prefill: true,
            gdn: true,
            gdn_gates: true,
            gdn_gated_rms_norm: true,
            gdn_decode_fused: true,
            gdn_decode_unexpanded_qk: true,
            gdn_decode_qk_norm_recurrent: true,
            gdn_decode_qk_norm_recurrent_rmsnorm: true,
            fused_conv1d: true,
            lora_decode_add: true,
            gdn_full_chunk_forward_multiblock: false,
            attn_decode_qkv_prep: true,
            fused_paged_decode: true,
            paged_decode_dyn_seqlen_batch: true,
            fused_mlp_silu_mul: true,
            fused_mlp_gate_up_prefill: true,
            fused_rmsnorm: true,
            long_flash_attn: true,
            native_rectangular_causal_flash: true,
            w8_projection: true,
            w8a8_projection: true,
            w8_swiglu: true,
            w8_sampled_lm_head: true,
            w8a8_sampled_lm_head: true,
            gqa_sdpa_f32_materialized: true,
            split_q_gate_f32_output: true,
            training_mlp_chunking: true,
            training_mlp_chunk_tokens: 512,
            split_q_gate_training: true,
            split_q_gate_output_chunk_features: 1024,
            split_q_gate_row_tile_tokens: 512,
        }
    }

    /// Reference-oriented policy that declines every accelerated model route
    /// while retaining bounded training chunking and split-projection safety.
    pub const fn portable_fallback() -> Self {
        Self {
            full_attn_qkv_in_proj: false,
            gdn_ab_in_proj: false,
            gdn_prefill_ab_in_proj: false,
            gdn_prefill_gates: false,
            head_major_prefill: false,
            gdn: false,
            gdn_gates: false,
            gdn_gated_rms_norm: false,
            gdn_decode_fused: false,
            gdn_decode_unexpanded_qk: false,
            gdn_decode_qk_norm_recurrent: false,
            gdn_decode_qk_norm_recurrent_rmsnorm: false,
            fused_conv1d: false,
            lora_decode_add: false,
            gdn_full_chunk_forward_multiblock: false,
            attn_decode_qkv_prep: false,
            fused_paged_decode: false,
            paged_decode_dyn_seqlen_batch: false,
            fused_mlp_silu_mul: false,
            fused_mlp_gate_up_prefill: false,
            fused_rmsnorm: false,
            long_flash_attn: false,
            native_rectangular_causal_flash: false,
            w8_projection: false,
            w8a8_projection: false,
            w8_swiglu: false,
            w8_sampled_lm_head: false,
            w8a8_sampled_lm_head: false,
            gqa_sdpa_f32_materialized: false,
            split_q_gate_f32_output: false,
            training_mlp_chunking: true,
            training_mlp_chunk_tokens: 512,
            split_q_gate_training: true,
            split_q_gate_output_chunk_features: 1024,
            split_q_gate_row_tile_tokens: 512,
        }
    }

    /// Diagnostic policy that declines the complete GDN/recurrent model-route
    /// family while preserving every other qualified model route.
    pub const fn gdn_fallback() -> Self {
        Self {
            gdn_ab_in_proj: false,
            gdn_prefill_ab_in_proj: false,
            gdn_prefill_gates: false,
            gdn: false,
            gdn_gates: false,
            gdn_gated_rms_norm: false,
            gdn_decode_fused: false,
            gdn_decode_unexpanded_qk: false,
            gdn_decode_qk_norm_recurrent: false,
            gdn_decode_qk_norm_recurrent_rmsnorm: false,
            fused_conv1d: false,
            gdn_full_chunk_forward_multiblock: false,
            ..Self::qualified()
        }
    }

    /// Diagnostic inverse of [`Self::gdn_fallback`]: retain only the
    /// qualified GDN/recurrent model-route family and decline every other
    /// accelerated model route.
    pub const fn non_gdn_fallback() -> Self {
        Self {
            gdn_ab_in_proj: true,
            gdn_prefill_ab_in_proj: true,
            gdn_prefill_gates: true,
            gdn: true,
            gdn_gates: true,
            gdn_gated_rms_norm: true,
            gdn_decode_fused: true,
            gdn_decode_unexpanded_qk: true,
            gdn_decode_qk_norm_recurrent: true,
            gdn_decode_qk_norm_recurrent_rmsnorm: true,
            fused_conv1d: true,
            gdn_full_chunk_forward_multiblock: false,
            ..Self::portable_fallback()
        }
    }

    /// Diagnostic policy that declines fused RMSNorm and fused MLP dispatch
    /// while preserving every other qualified model route.
    pub const fn fused_norm_mlp_fallback() -> Self {
        Self {
            fused_mlp_silu_mul: false,
            fused_mlp_gate_up_prefill: false,
            fused_rmsnorm: false,
            ..Self::qualified()
        }
    }

    /// Diagnostic inverse of [`Self::fused_norm_mlp_fallback`]: enable only
    /// fused RMSNorm and fused MLP dispatch on the portable model policy.
    pub const fn fused_norm_mlp_only() -> Self {
        Self {
            fused_mlp_silu_mul: true,
            fused_mlp_gate_up_prefill: true,
            fused_rmsnorm: true,
            ..Self::portable_fallback()
        }
    }

    /// Diagnostic policy that changes only the split q/gate F32-output
    /// projection route from the qualified model policy.
    pub const fn split_q_gate_fallback() -> Self {
        Self {
            split_q_gate_f32_output: false,
            ..Self::qualified()
        }
    }

    /// Diagnostic inverse of [`Self::split_q_gate_fallback`]: enable only the
    /// split q/gate F32-output projection route on the portable model policy.
    pub const fn split_q_gate_only() -> Self {
        Self {
            split_q_gate_f32_output: true,
            ..Self::portable_fallback()
        }
    }

    /// Qualified policy plus the unqualified multi-block GDN prefill route.
    pub const fn experimental_multiblock() -> Self {
        Self {
            gdn_full_chunk_forward_multiblock: true,
            ..Self::qualified()
        }
    }

    #[cfg(test)]
    const fn accelerated_routes(self) -> [bool; 30] {
        [
            self.full_attn_qkv_in_proj,
            self.gdn_ab_in_proj,
            self.gdn_prefill_ab_in_proj,
            self.gdn_prefill_gates,
            self.head_major_prefill,
            self.gdn,
            self.gdn_gates,
            self.gdn_gated_rms_norm,
            self.gdn_decode_fused,
            self.gdn_decode_unexpanded_qk,
            self.gdn_decode_qk_norm_recurrent,
            self.gdn_decode_qk_norm_recurrent_rmsnorm,
            self.fused_conv1d,
            self.lora_decode_add,
            self.gdn_full_chunk_forward_multiblock,
            self.attn_decode_qkv_prep,
            self.fused_paged_decode,
            self.paged_decode_dyn_seqlen_batch,
            self.fused_mlp_silu_mul,
            self.fused_mlp_gate_up_prefill,
            self.fused_rmsnorm,
            self.long_flash_attn,
            self.native_rectangular_causal_flash,
            self.w8_projection,
            self.w8a8_projection,
            self.w8_swiglu,
            self.w8_sampled_lm_head,
            self.w8a8_sampled_lm_head,
            self.gqa_sdpa_f32_materialized,
            self.split_q_gate_f32_output,
        ]
    }
}

impl Default for RocmKernelPolicy {
    fn default() -> Self {
        Self::qualified()
    }
}

/// Install process-lifetime ROCm kernel policy. Reinstalling the same value is
/// idempotent; conflicting values fail instead of changing live dispatch.
#[cfg_attr(not(feature = "rocm"), allow(dead_code))]
pub fn install_rocm_kernel_policy(policy: RocmKernelPolicy) -> Result<()> {
    match ROCM_KERNEL_POLICY.set(policy) {
        Ok(()) => Ok(()),
        Err(policy) if ROCM_KERNEL_POLICY.get() == Some(&policy) => Ok(()),
        Err(_) => anyhow::bail!("ROCm kernel policy was already installed with a different value"),
    }
}

pub(crate) fn current_rocm_kernel_policy() -> RocmKernelPolicy {
    *ROCM_KERNEL_POLICY.get_or_init(RocmKernelPolicy::default)
}

#[cfg(test)]
mod tests {
    use super::RocmKernelPolicy;

    #[test]
    fn profiles_cover_accelerated_routes_and_training_safeguards() {
        let qualified = RocmKernelPolicy::qualified();
        let fallback = RocmKernelPolicy::portable_fallback();
        let gdn_fallback = RocmKernelPolicy::gdn_fallback();
        let non_gdn_fallback = RocmKernelPolicy::non_gdn_fallback();
        let fused_norm_mlp_fallback = RocmKernelPolicy::fused_norm_mlp_fallback();
        let fused_norm_mlp_only = RocmKernelPolicy::fused_norm_mlp_only();
        let split_q_gate_fallback = RocmKernelPolicy::split_q_gate_fallback();
        let split_q_gate_only = RocmKernelPolicy::split_q_gate_only();
        let experimental = RocmKernelPolicy::experimental_multiblock();

        assert_eq!(
            qualified.accelerated_routes(),
            [
                true, true, true, true, true, true, true, true, true, true, true, true, true, true,
                false, true, true, true, true, true, true, true, true, true, true, true, true,
                true, true, true,
            ]
        );
        assert_eq!(fallback.accelerated_routes(), [false; 30]);
        assert_eq!(
            gdn_fallback.accelerated_routes(),
            [
                true, false, false, false, true, false, false, false, false, false, false, false,
                false, true, false, true, true, true, true, true, true, true, true, true, true,
                true, true, true, true, true,
            ]
        );
        assert_eq!(
            non_gdn_fallback.accelerated_routes(),
            [
                false, true, true, true, false, true, true, true, true, true, true, true, true,
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false,
            ]
        );
        assert_eq!(
            fused_norm_mlp_fallback.accelerated_routes(),
            [
                true, true, true, true, true, true, true, true, true, true, true, true, true, true,
                false, true, true, true, false, false, false, true, true, true, true, true, true,
                true, true, true,
            ]
        );
        assert_eq!(
            fused_norm_mlp_only.accelerated_routes(),
            [
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, true, true, true, false, false, false,
                false, false, false, false, false, false,
            ]
        );
        assert_eq!(
            split_q_gate_fallback.accelerated_routes(),
            [
                true, true, true, true, true, true, true, true, true, true, true, true, true, true,
                false, true, true, true, true, true, true, true, true, true, true, true, true,
                true, true, false,
            ]
        );
        assert_eq!(
            split_q_gate_only.accelerated_routes(),
            [
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, true,
            ]
        );
        assert_eq!(experimental.accelerated_routes(), [true; 30]);

        for policy in [
            qualified,
            fallback,
            gdn_fallback,
            non_gdn_fallback,
            fused_norm_mlp_fallback,
            fused_norm_mlp_only,
            split_q_gate_fallback,
            split_q_gate_only,
            experimental,
        ] {
            assert!(policy.training_mlp_chunking);
            assert_eq!(policy.training_mlp_chunk_tokens, 512);
            assert!(policy.split_q_gate_training);
            assert_eq!(policy.split_q_gate_output_chunk_features, 1024);
            assert_eq!(policy.split_q_gate_row_tile_tokens, 512);
        }
    }
}
