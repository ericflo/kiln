//! Metal backend: candle's fused SDPA for the attention hot path, portable
//! fallback for GDN and paged-decode.
//!
//! The chokepoint-routed `sdpa` symbol (imported at module level from the
//! kt-side re-export) is an MLX-style fused scaled-dot-product attention
//! kernel with native GQA, BF16, and head dims {32, 64, 72, 80, 96, 128,
//! 256, 512}. For typical transformer head sizes this replaces the vendored
//! CUDA FlashAttention-2 call on Apple Silicon.

pub(crate) use super::metal_config::metal_mlp_gate_up_fusion_disabled;
use super::metal_config::*;
pub(crate) use super::metal_dense::{
    metal_attn_gate_sigmoid_mul_bf16, metal_attn_gate_sigmoid_mul_supports,
    metal_fused_qkv_transposed_coop_gemv_bf16, metal_fused_qkv_transposed_coop_gemv_supports,
    metal_lora_add_decode_bf16, metal_lora_add_decode_supports, metal_mlp_gate_up_bf16,
    metal_mlp_gate_up_supports, metal_mlp_silu_mul_bf16, metal_mlp_silu_mul_supports,
    metal_transposed_coop_gemv_bf16, metal_transposed_coop_gemv_decode_batch_supports,
    metal_transposed_coop_gemv_supports,
};
pub(crate) use super::metal_gdn::{
    metal_gdn_decode_gates_recurrent_bf16, metal_gdn_decode_gates_recurrent_rmsnorm_bf16,
    metal_gdn_decode_gates_recurrent_rmsnorm_supports, metal_gdn_decode_gates_recurrent_supports,
    metal_gdn_decode_qkv_conv_norm_bf16, metal_gdn_decode_qkv_conv_norm_supports,
    metal_gdn_gates_decay_ab_bf16, metal_gdn_gates_decay_ab_supports, metal_gdn_gates_decay_bf16,
    metal_gdn_gates_decay_supports, metal_gdn_prefill_ab_in_proj_bf16,
    metal_gdn_prefill_ab_in_proj_supports, metal_gdn_prefill_qkv_conv_split_bf16_f32_k4,
    metal_gdn_prefill_qkv_conv_split_supports, metal_gdn_qk_norm_f32_bf16,
    metal_gdn_qk_norm_gqa_f32_bf16, metal_gdn_qk_norm_gqa_supports, metal_gdn_qk_norm_supports,
    metal_gdn_recurrent_prefill_native_head_last_decay_bf16,
    metal_gdn_recurrent_prefill_native_head_last_decay_supports,
};
pub(crate) use super::metal_icb::{MetalPagedDecodeIcbGraph, MetalSingleTokenPagedDecodeIcbGraph};
pub(crate) use super::metal_lm_head::{
    metal_lm_head_argmax_bf16, metal_lm_head_argmax_rows_bf16, metal_lm_head_argmax_rows_supports,
    metal_lm_head_argmax_supports, metal_lm_head_bf16, metal_lm_head_sample_bf16,
    metal_lm_head_sample_supports, metal_lm_head_supports,
};
pub(crate) use super::metal_norm::{
    metal_rms_norm_bf16, metal_rms_norm_supports, metal_rotary_embedding_bf16,
    metal_rotary_embedding_supports,
};
pub(crate) use super::metal_paged::{
    metal_paged_kv_write_token_major_batch_bf16, metal_paged_kv_write_token_major_batch_supports,
    metal_paged_kv_write_token_major_bf16, metal_paged_kv_write_token_major_supports,
    metal_record_paged_decode_icb_graph, metal_record_single_token_paged_decode_icb_graph,
};
pub use super::metal_precompile::precompile_custom_kernels;
use super::{TrainingCapabilities, metal_training};

// Phase 7 #1082: module-level imports for the kt-metal chokepoint types,
// hoisted from ~92 per-function `use` statements so that the chokepoint
// surface in this file is centralized at a single import location. Future
// substrate swaps (e.g. candle → objc2-metal) touch this single import
// block instead of hundreds of scattered fully-qualified references.
#[derive(Debug)]
pub struct MetalBackend {
    /// The kt Metal device this backend dispatches on. (#1082: the
    /// formerly-retained candle `device` field is gone — every trait
    /// method is kt-native, so no candle handle is held.)
    pub(super) device_kt: kiln_tensor::Device,
    /// Cached at construction to keep env-var reads off per-token support gates.
    pub(super) disable: MetalKernelDisables,
    pub(super) resident_activation_registry: super::metal_residency::ResidentActivationRegistry,
}

impl MetalBackend {
    pub fn new(device: kiln_tensor::Device) -> Self {
        debug_assert!(
            matches!(device, kiln_tensor::Device::Metal(_)),
            "MetalBackend created on non-Metal device"
        );
        Self {
            device_kt: device,
            disable: MetalKernelDisables::from_env(),
            resident_activation_registry: super::metal_residency::new_resident_activation_registry(
            ),
        }
    }

    pub fn training_capabilities_static() -> TrainingCapabilities {
        metal_training::training_capabilities_static()
    }
}

/// Test/helper: try to initialize a kt Metal device, returning `None` if Metal
/// isn't available or if device discovery panics in a sandboxed runner.
#[doc(hidden)]
pub fn try_new_metal() -> Option<kiln_tensor::Device> {
    let result = std::panic::catch_unwind(|| kiln_tensor::primary_metal_companion(0));
    match result {
        Ok(Ok(_)) => Some(kiln_tensor::Device::Metal(0)),
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
