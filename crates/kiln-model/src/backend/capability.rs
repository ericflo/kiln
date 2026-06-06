//! Typed capability descriptors for backend diagnostics.
//!
//! This is the Phase 0/1 bridge from bool-only `supports_*` predicates toward
//! request-shaped capability queries. The snapshot intentionally reads existing
//! `BackendRuntime` methods; it is descriptive only and does not change
//! dispatch behavior.

use super::{BackendRuntime, TrainingCapabilities};

/// Backend answer for a capability query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Support {
    Native,
    NativeWithConstraints,
    HostFallbackAllowed,
    Declined,
    Unsupported,
    DisabledByEnv,
    RequiresFeature,
}

impl Support {
    pub const fn from_supports_predicate(supported: bool) -> Self {
        if supported {
            Support::NativeWithConstraints
        } else {
            Support::Declined
        }
    }
}

/// Snapshot of the existing backend capability predicates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendCapabilitySnapshot {
    pub backend: &'static str,
    pub device: kiln_tensor::Device,
    pub training: TrainingCapabilities,
    pub resident_decode: Support,
    pub resident_activation: Support,
    pub flash_attn_prefill: Support,
    pub flash_attn_paged_decode: Support,
    pub paged_kv_head_major_read: Support,
    pub gdn_recurrent_step: Support,
    pub causal_conv1d_update: Support,
    pub linear_decode_argmax: Support,
}

impl BackendCapabilitySnapshot {
    pub fn from_backend(backend: &dyn BackendRuntime) -> Self {
        Self {
            backend: backend.name(),
            device: backend.device(),
            training: backend.training_capabilities(),
            resident_decode: Support::from_supports_predicate(
                backend.supports_resident_decode(),
            ),
            resident_activation: Support::from_supports_predicate(
                backend.supports_resident_activation(),
            ),
            flash_attn_prefill: Support::from_supports_predicate(
                backend.supports_flash_attn_prefill(),
            ),
            flash_attn_paged_decode: Support::from_supports_predicate(
                backend.supports_flash_attn_paged_decode(),
            ),
            paged_kv_head_major_read: Support::from_supports_predicate(
                backend.supports_paged_kv_head_major_read(),
            ),
            gdn_recurrent_step: Support::from_supports_predicate(
                backend.supports_gdn_recurrent_step(),
            ),
            causal_conv1d_update: Support::from_supports_predicate(
                backend.supports_causal_conv1d_update(),
            ),
            linear_decode_argmax: Support::from_supports_predicate(
                backend.supports_linear_decode_argmax(),
            ),
        }
    }
}
