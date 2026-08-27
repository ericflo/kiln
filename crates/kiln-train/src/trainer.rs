//! In-process LoRA SFT and GRPO training.
//!
//! Trains LoRA adapter weights directly on the already-loaded model's GPU
//! tensors in one process. This facade preserves the public training surface;
//! method, checkpoint, optimizer, and reporting behavior live in owned modules.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use kiln_core::block::BlockTable;
use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_flce_kernel::DEFAULT_CHUNK_SIZE;
#[cfg(feature = "vulkan")]
use kiln_model::backend::GrpoLossRoute;
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
use kiln_model::backend::TrainingTapeRoute;
use kiln_model::backend::{
    self, BackendIdentity, BackendRuntime, ExternalYieldBackend, FinalRmsNormBackwardRoute,
    GrpoKlAuxiliaryRoute, OptimizerBackend, ResidencyBackend, SftFlceLossRoute,
    TrainingLossBackend, TrainingPrecisionPolicy,
};
use kiln_model::forward::{
    GpuAttentionWeights, GpuWeights, LinearAttentionState, StreamingPrefillExecutionPolicy,
    model_forward_kt_with_policy, model_forward_no_head_with_policy,
    model_forward_paged_normed_hidden, model_forward_segment_with_policy, rms_norm,
};
// GPU-feature kt-tape composites: consumed only by the cuda/metal/vulkan/rocm
// forward paths (trainer/forward_backward.rs); trainer/tests/mod.rs imports its
// own copies for the non-GPU test build, so this stays cfg-gated.
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
use kiln_model::forward::{model_forward_embed, model_forward_final_norm, model_forward_head};
use kiln_model::lora_loader::{LoraLayerWeights, LoraProjectionWeights, LoraWeights};
use kiln_model::sampling::{greedy_sample, try_topk_on_device};
use kiln_model::{
    BackendCapabilityQueries, PagedKvCacheKt, TrainingOptimizerRequest, TrainingOptimizerRounding,
    TrainingOptimizerSupport,
};

use crate::cd_types::*;
use crate::replay::{
    self, BaseModel, Lineage, OutcomeRecord, OutcomeStatus, ParentLora, ReplayKind, ReplayLog,
    RequestRecord,
};
use crate::{
    AdvantageMode, BehaviorPolicy, ChatMessage, GrpoConfig, GrpoGroup, IsLevel, KlEstimator,
    KlReferencePolicy, LossAggregation, Optimizer, RewardFilterOnEmpty, SftConfig, SftExample,
    TurnKind,
};
use kiln_optim::{
    AdamW as KtAdamW, AdamWHyperparameters as KtAdamWHyperparameters,
    AdamWMoments as KtHostAdamWMoments, MomentLocation as KtMomentLocation, Muon as KtMuon,
    MuonState as KtHostMuonState, OptimStep, StochasticRoundingPolicy,
};
use kiln_param::{AmpPolicy as KtAmpPolicy, ForwardStorage as KtForwardStorage, Parameter};
use kiln_tensor::{DType as KtDType, Tensor as KtTensor, TensorId as KtTensorId};

mod provenance;
pub use provenance::*;
mod tensor_support;
pub(crate) use tensor_support::*;
mod lora_parameters;
pub use lora_parameters::*;
mod optimizer_state;
pub use optimizer_state::*;
mod training_support;
pub use training_support::*;
mod checkpointing;
pub(crate) use checkpointing::*;
mod reporting;
pub(crate) use reporting::*;
mod sft;
pub use sft::*;
mod grpo;
pub use grpo::*;
mod grpo_jsonl;
pub use grpo_jsonl::*;
mod grpo_step;
pub use grpo_step::*;
mod reference_policy;
pub(crate) use reference_policy::*;
mod sft_data;
pub use sft_data::*;
mod optimizers;
pub use optimizers::*;
mod checkpoint_execution;
pub use checkpoint_execution::*;
mod forward_backward;
pub use forward_backward::*;

// (#1082 CP-4) `pub(crate)` so the OPD tape-authoritative test in `opd.rs`'s
// own `#[cfg(test)] mod tests` can reuse the BF16 tiny-model fixtures
// (`tiny_config_bf16` / `tiny_weights_bf16`) instead of duplicating them —
// single source of truth for the BF16 CUDA fixture. Still `#[cfg(test)]`, so
// it carries no cost in non-test builds.
#[cfg(test)]
pub(crate) mod tests;
