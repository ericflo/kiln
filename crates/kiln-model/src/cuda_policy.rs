use anyhow::Result;
use std::sync::OnceLock;

static CUDA_KERNEL_POLICY: OnceLock<CudaKernelPolicy> = OnceLock::new();

/// Complete process-lifetime CUDA backend-kernel policy.
///
/// The server maps its closed startup profile to one of these policies before
/// creating a CUDA device. Embedders that do not install a policy receive the
/// native defaults on first CUDA backend construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaKernelPolicy {
    pub(crate) full_attn_qkv_in_proj: bool,
    pub(crate) gdn_ab_in_proj: bool,
    pub(crate) gdn_prefill_ab_in_proj: bool,
    pub(crate) gdn_prefill_gates: bool,
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
    pub(crate) fused_paged_decode: bool,
    pub(crate) fused_rotary_qk: bool,
    pub(crate) attn_decode_qkv_prep: bool,
    pub(crate) fused_mlp_silu_mul: bool,
    pub(crate) fused_mlp_gate_up_prefill: bool,
    pub(crate) fused_attn_sigmoid_mul: bool,
    pub(crate) fused_rmsnorm: bool,
    pub(crate) rmsnorm_backward: bool,
    pub(crate) fused_l2_qk_norm: bool,
    pub(crate) paged_decode_dyn_seqlen_batch: bool,
    pub(crate) gdn_chunk_pre_permute: bool,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl CudaKernelPolicy {
    /// Device-neutral subset of the CUDA routes enabled before policy
    /// consolidation.
    pub const fn native_default() -> Self {
        Self {
            full_attn_qkv_in_proj: true,
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
            lora_decode_add: true,
            // This route is correct but has only been performance-qualified on
            // one 76-SM laptop GPU. Keep the generic default device-neutral.
            gdn_full_chunk_forward_multiblock: false,
            fused_paged_decode: true,
            fused_rotary_qk: true,
            attn_decode_qkv_prep: true,
            fused_mlp_silu_mul: true,
            fused_mlp_gate_up_prefill: true,
            fused_attn_sigmoid_mul: true,
            fused_rmsnorm: true,
            rmsnorm_backward: true,
            fused_l2_qk_norm: true,
            paged_decode_dyn_seqlen_batch: true,
            gdn_chunk_pre_permute: true,
        }
    }

    /// Decline every governed CUDA route and use its portable fallback.
    pub const fn portable_fallback() -> Self {
        Self {
            full_attn_qkv_in_proj: false,
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
            lora_decode_add: false,
            gdn_full_chunk_forward_multiblock: false,
            fused_paged_decode: false,
            fused_rotary_qk: false,
            attn_decode_qkv_prep: false,
            fused_mlp_silu_mul: false,
            fused_mlp_gate_up_prefill: false,
            fused_attn_sigmoid_mul: false,
            fused_rmsnorm: false,
            rmsnorm_backward: false,
            fused_l2_qk_norm: false,
            paged_decode_dyn_seqlen_batch: false,
            gdn_chunk_pre_permute: false,
        }
    }

    #[cfg(test)]
    const fn routes(self) -> [bool; 25] {
        [
            self.full_attn_qkv_in_proj,
            self.gdn_ab_in_proj,
            self.gdn_prefill_ab_in_proj,
            self.gdn_prefill_gates,
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
            self.fused_paged_decode,
            self.fused_rotary_qk,
            self.attn_decode_qkv_prep,
            self.fused_mlp_silu_mul,
            self.fused_mlp_gate_up_prefill,
            self.fused_attn_sigmoid_mul,
            self.fused_rmsnorm,
            self.rmsnorm_backward,
            self.fused_l2_qk_norm,
            self.paged_decode_dyn_seqlen_batch,
            self.gdn_chunk_pre_permute,
        ]
    }
}

impl Default for CudaKernelPolicy {
    fn default() -> Self {
        Self::native_default()
    }
}

/// Install process-lifetime CUDA kernel policy. Reinstalling the same value is
/// idempotent; conflicting installation fails instead of changing live routes.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub fn install_cuda_kernel_policy(policy: CudaKernelPolicy) -> Result<()> {
    match CUDA_KERNEL_POLICY.set(policy) {
        Ok(()) => Ok(()),
        Err(policy) if CUDA_KERNEL_POLICY.get() == Some(&policy) => Ok(()),
        Err(_) => anyhow::bail!("CUDA kernel policy was already installed with a different value"),
    }
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn current_cuda_kernel_policy() -> CudaKernelPolicy {
    *CUDA_KERNEL_POLICY.get_or_init(CudaKernelPolicy::default)
}

#[cfg(test)]
mod tests {
    use super::CudaKernelPolicy;

    #[test]
    fn profiles_cover_every_backend_route() {
        let mut native_routes = [true; 25];
        native_routes[13] = false;
        assert_eq!(CudaKernelPolicy::native_default().routes(), native_routes);
        assert_eq!(CudaKernelPolicy::portable_fallback().routes(), [false; 25]);
    }
}
