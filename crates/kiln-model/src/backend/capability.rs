//! Typed capability descriptors for backend diagnostics.
//!
//! This is the Phase 0/1 bridge from bool-only `supports_*` predicates toward
//! request-shaped capability queries. The snapshot intentionally reads existing
//! `BackendRuntime` methods; it is descriptive only and does not change
//! dispatch behavior.

use kiln_graph::ReplayKey;

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

/// Accumulation precision requested by a matmul.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulAccumulation {
    F32,
}

/// Logical storage layout for a matmul operand or output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulOperandLayout {
    RowMajor,
    ColMajor,
}

/// Fused matmul tail requested by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulEpilogue {
    Identity,
    Bias,
    Relu,
    Gelu,
    Silu,
    BiasSilu,
    BiasGelu,
}

/// Batch shape collapsed into the logical matmul request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulBatchPolicy {
    Single,
    Batched { batches: usize },
}

impl MatmulBatchPolicy {
    fn from_leading_shape(shape: &[usize]) -> Self {
        let batches = shape.iter().product::<usize>().max(1);
        if batches == 1 {
            Self::Single
        } else {
            Self::Batched { batches }
        }
    }
}

/// Engine-facing matmul capability request.
///
/// This intentionally carries the richer shape/dtype/layout/replay vocabulary
/// from the unification plan, while staying compatible with the narrower
/// BLAS-crate `m/n/k + dtype + layout + epilogue` request direction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatmulRequest {
    pub lhs_shape: Vec<usize>,
    pub rhs_shape: Vec<usize>,
    pub lhs_dtype: kiln_tensor::DType,
    pub rhs_dtype: kiln_tensor::DType,
    pub out_dtype: kiln_tensor::DType,
    pub accumulation: MatmulAccumulation,
    pub lhs_layout: MatmulOperandLayout,
    pub rhs_layout: MatmulOperandLayout,
    pub out_layout: MatmulOperandLayout,
    pub batch: MatmulBatchPolicy,
    pub epilogue: MatmulEpilogue,
    pub replay_safe: bool,
}

impl MatmulRequest {
    pub fn plain(
        lhs_shape: Vec<usize>,
        rhs_shape: Vec<usize>,
        dtype: kiln_tensor::DType,
        replay_safe: bool,
    ) -> Self {
        let batch = lhs_shape
            .len()
            .checked_sub(2)
            .map(|rank_prefix| {
                MatmulBatchPolicy::from_leading_shape(&lhs_shape[..rank_prefix])
            })
            .unwrap_or(MatmulBatchPolicy::Single);
        Self {
            lhs_shape,
            rhs_shape,
            lhs_dtype: dtype,
            rhs_dtype: dtype,
            out_dtype: dtype,
            accumulation: MatmulAccumulation::F32,
            lhs_layout: MatmulOperandLayout::RowMajor,
            rhs_layout: MatmulOperandLayout::RowMajor,
            out_layout: MatmulOperandLayout::RowMajor,
            batch,
            epilogue: MatmulEpilogue::Identity,
            replay_safe,
        }
    }

    pub fn with_epilogue(mut self, epilogue: MatmulEpilogue) -> Self {
        self.epilogue = epilogue;
        self
    }

    pub fn logical_mnk(&self) -> Option<(usize, usize, usize)> {
        if !self.has_compatible_shapes() {
            return None;
        }
        let rank = self.lhs_shape.len();
        Some((
            self.lhs_shape[rank - 2],
            self.rhs_shape[rank - 1],
            self.lhs_shape[rank - 1],
        ))
    }

    pub fn rank(&self) -> Option<usize> {
        if self.lhs_shape.len() == self.rhs_shape.len() {
            Some(self.lhs_shape.len())
        } else {
            None
        }
    }

    fn has_compatible_shapes(&self) -> bool {
        let Some(rank) = self.rank() else {
            return false;
        };
        if rank < 2 {
            return false;
        }
        if self.lhs_shape[..rank - 2] != self.rhs_shape[..rank - 2] {
            return false;
        }
        if self.lhs_shape[rank - 1] != self.rhs_shape[rank - 2] {
            return false;
        }
        self.batch
            == MatmulBatchPolicy::from_leading_shape(&self.lhs_shape[..rank - 2])
    }

    fn has_supported_dtype_contract(&self) -> bool {
        self.lhs_dtype == self.rhs_dtype
            && self.lhs_dtype == self.out_dtype
            && self.accumulation == MatmulAccumulation::F32
            && matches!(
                self.lhs_dtype,
                kiln_tensor::DType::F32 | kiln_tensor::DType::BF16 | kiln_tensor::DType::F16
            )
    }

    fn is_row_major_output(&self) -> bool {
        self.out_layout == MatmulOperandLayout::RowMajor
    }

    fn is_row_major_input(&self) -> bool {
        self.lhs_layout == MatmulOperandLayout::RowMajor
            && self.rhs_layout == MatmulOperandLayout::RowMajor
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

impl ReplayRequestKind {
    pub const fn operation_name(self) -> &'static str {
        match self {
            Self::ResidentDecode => "resident_decode",
            Self::PagedDecodeGraphOutputs => "paged_decode_graph_outputs",
        }
    }
}

/// Request descriptor for replay/capture capability queries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayRequest {
    pub kind: ReplayRequestKind,
    pub max_hidden: usize,
    pub max_intermediate: usize,
    pub max_batch: usize,
    pub dtype: Option<kiln_tensor::DType>,
    pub replay_safe: bool,
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
            dtype: None,
            replay_safe: true,
        }
    }

    pub const fn paged_decode_graph_outputs(
        max_hidden: usize,
        max_intermediate: usize,
        max_batch: usize,
    ) -> Self {
        Self {
            kind: ReplayRequestKind::PagedDecodeGraphOutputs,
            max_hidden,
            max_intermediate,
            max_batch,
            dtype: None,
            replay_safe: true,
        }
    }

    pub fn with_dtype(mut self, dtype: kiln_tensor::DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    pub fn with_replay_safe(mut self, replay_safe: bool) -> Self {
        self.replay_safe = replay_safe;
        self
    }

    pub fn shape_key(&self) -> Vec<usize> {
        vec![self.max_hidden, self.max_intermediate, self.max_batch]
    }

    pub fn replay_key(&self, backend: kiln_tensor::Backend) -> ReplayKey {
        ReplayKey::new(
            backend,
            self.kind.operation_name(),
            self.shape_key(),
            self.dtype,
            self.max_batch,
            self.replay_safe,
        )
    }

    pub const fn has_valid_bounds(&self) -> bool {
        self.max_hidden > 0 && self.max_intermediate > 0 && self.max_batch > 0
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

    fn supports_matmul_request(&self, req: &MatmulRequest) -> Support {
        if !req.has_compatible_shapes()
            || !req.has_supported_dtype_contract()
            || !req.is_row_major_output()
            || !req.is_row_major_input()
            || !matches!(req.epilogue, MatmulEpilogue::Identity | MatmulEpilogue::Bias)
        {
            return Support::Unsupported;
        }

        let Some(rank) = req.rank() else {
            return Support::Unsupported;
        };
        let native = match self.name() {
            "cpu" => matches!(req.epilogue, MatmulEpilogue::Identity | MatmulEpilogue::Bias),
            "cuda" | "rocm" => match req.epilogue {
                MatmulEpilogue::Identity => true,
                MatmulEpilogue::Bias => rank == 2,
                _ => false,
            },
            "metal" => {
                req.lhs_dtype == kiln_tensor::DType::BF16
                    && matches!(req.epilogue, MatmulEpilogue::Identity)
            }
            "vulkan" => {
                matches!(req.epilogue, MatmulEpilogue::Identity)
                    && (req.lhs_dtype == kiln_tensor::DType::F32
                        || req.lhs_dtype == kiln_tensor::DType::BF16 && rank > 2)
            }
            _ => false,
        };

        if native {
            Support::NativeWithConstraints
        } else {
            Support::HostFallbackAllowed
        }
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
        if !req.replay_safe || !req.has_valid_bounds() {
            return Support::Unsupported;
        }

        Support::from_supports_predicate(match req.kind {
            ReplayRequestKind::ResidentDecode => self.supports_resident_decode(),
            ReplayRequestKind::PagedDecodeGraphOutputs => self.supports_flash_attn_paged_decode(),
        })
    }
}

impl<T: BackendRuntime + ?Sized> BackendCapabilityQueries for T {}
