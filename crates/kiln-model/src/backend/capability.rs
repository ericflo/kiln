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

/// Attention operation family being queried.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionRequestKind {
    FlashPrefill,
    FlashPrefillHeadMajor,
    FlashPagedDecode,
}

/// Request descriptor for attention capability queries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionRequest {
    pub kind: AttentionRequestKind,
    pub q_dtype: kiln_tensor::DType,
    pub k_dtype: kiln_tensor::DType,
    pub v_dtype: kiln_tensor::DType,
    pub batch: usize,
    pub seq_len: usize,
    pub head_dim: usize,
    pub replay_safe: bool,
}

impl AttentionRequest {
    pub const fn flash_prefill(
        q_dtype: kiln_tensor::DType,
        k_dtype: kiln_tensor::DType,
        v_dtype: kiln_tensor::DType,
        batch: usize,
        seq_len: usize,
        head_dim: usize,
        replay_safe: bool,
    ) -> Self {
        Self {
            kind: AttentionRequestKind::FlashPrefill,
            q_dtype,
            k_dtype,
            v_dtype,
            batch,
            seq_len,
            head_dim,
            replay_safe,
        }
    }
}

/// Linear/decode operation family being queried.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LinearRequestKind {
    DecodeArgmax,
    DecodeArgmaxBatch,
    DecodeSample,
    DecodeSampleBatch,
}

/// Request descriptor for linear/lm-head capability queries.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearRequest {
    pub kind: LinearRequestKind,
    pub input_dtype: kiln_tensor::DType,
    pub weight_dtype: kiln_tensor::DType,
    pub output_dtype: kiln_tensor::DType,
    pub batch: usize,
    pub top_k: Vec<u32>,
    pub temperatures: Vec<f32>,
    pub replay_safe: bool,
}

impl LinearRequest {
    pub fn decode_argmax(
        input_dtype: kiln_tensor::DType,
        weight_dtype: kiln_tensor::DType,
        output_dtype: kiln_tensor::DType,
        batch: usize,
        replay_safe: bool,
    ) -> Self {
        Self {
            kind: if batch > 1 {
                LinearRequestKind::DecodeArgmaxBatch
            } else {
                LinearRequestKind::DecodeArgmax
            },
            input_dtype,
            weight_dtype,
            output_dtype,
            batch,
            top_k: Vec::new(),
            temperatures: Vec::new(),
            replay_safe,
        }
    }

    pub fn decode_sample(
        input_dtype: kiln_tensor::DType,
        weight_dtype: kiln_tensor::DType,
        output_dtype: kiln_tensor::DType,
        top_k: Vec<u32>,
        temperatures: Vec<f32>,
        replay_safe: bool,
    ) -> Self {
        let batch = top_k.len().max(1);
        Self {
            kind: if batch > 1 {
                LinearRequestKind::DecodeSampleBatch
            } else {
                LinearRequestKind::DecodeSample
            },
            input_dtype,
            weight_dtype,
            output_dtype,
            batch,
            top_k,
            temperatures,
            replay_safe,
        }
    }
}

/// Replay operation family being queried.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReplayRequestKind {
    ResidentDecode,
    PagedDecodeGraphOutputs,
}

/// Request descriptor for replay/capture capability queries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayRequest {
    pub kind: ReplayRequestKind,
    pub max_hidden: usize,
    pub max_intermediate: usize,
    pub max_batch: usize,
}

impl ReplayRequest {
    pub const fn resident_decode(
        max_hidden: usize,
        max_intermediate: usize,
        max_batch: usize,
    ) -> Self {
        Self {
            kind: ReplayRequestKind::ResidentDecode,
            max_hidden,
            max_intermediate,
            max_batch,
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
    pub fn from_backend<T: BackendRuntime + ?Sized>(backend: &T) -> Self {
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

/// Request-shaped capability query surface backed by the current runtime.
///
/// This is the compatibility bridge for the target architecture: call sites can
/// start asking request-shaped questions while existing backends keep their
/// current bool predicates and shape gates.
pub trait BackendCapabilityQueries: BackendRuntime {
    fn capability_snapshot(&self) -> BackendCapabilitySnapshot {
        BackendCapabilitySnapshot::from_backend(self)
    }

    fn supports_attention_request(&self, req: &AttentionRequest) -> Support {
        Support::from_supports_predicate(match req.kind {
            AttentionRequestKind::FlashPrefill => self.supports_flash_attn_prefill(),
            AttentionRequestKind::FlashPrefillHeadMajor => {
                self.supports_flash_attn_prefill_head_major()
            }
            AttentionRequestKind::FlashPagedDecode => self.supports_flash_attn_paged_decode(),
        })
    }

    fn supports_linear_request(&self, req: &LinearRequest) -> Support {
        Support::from_supports_predicate(match req.kind {
            LinearRequestKind::DecodeArgmax => self.supports_linear_decode_argmax(),
            LinearRequestKind::DecodeArgmaxBatch => self.supports_linear_decode_argmax_batch(),
            LinearRequestKind::DecodeSample => req
                .top_k
                .first()
                .copied()
                .map(|top_k| self.supports_linear_decode_sample(top_k))
                .unwrap_or(false),
            LinearRequestKind::DecodeSampleBatch => {
                self.supports_linear_decode_sample_batch(&req.top_k, &req.temperatures)
            }
        })
    }

    fn supports_replay_request(&self, req: &ReplayRequest) -> Support {
        Support::from_supports_predicate(match req.kind {
            ReplayRequestKind::ResidentDecode => self.supports_resident_decode(),
            ReplayRequestKind::PagedDecodeGraphOutputs => self.supports_flash_attn_paged_decode(),
        })
    }
}

impl<T: BackendRuntime + ?Sized> BackendCapabilityQueries for T {}
