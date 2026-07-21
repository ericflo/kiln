//! In-process LoRA SFT and GRPO training using candle autograd.
//!
//! Trains LoRA adapter weights directly on the already-loaded model's GPU
//! tensors. No Python sidecar, no second model copy, single process.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
// NOTE(#1082): trainer.rs has zero `use candle_*` imports at module top
// and exactly one direct candle path remaining: the `impl candle CustomOp1
// for InjectTensorGradient` block near the bottom of this file (the trait
// path is fully spelled out there because traits cannot be type-aliased
// on stable Rust without `trait_alias`). Every other candle reference
// resolves through
// `crate::cd_types`, the per-crate candle facade that holds the type
// aliases (`Tensor` / `Var` / `Device` / `DType` / `Shape` /
// `GradStore` / `TensorId` / `D` / `CdResult`), the `cd_bail!` macro,
// the safetensors I/O shims (`safetensors_load_file` /
// `safetensors_save_file`), and the generic constructor helpers
// (`tensor_new` / `tensor_from_vec`).
//
// Historical reductions:
//   1. Dropped the cuda-gated `use {CudaStorage, backend::BackendStorage};`.
//   2. Dropped `use {CpuStorage, CustomOp1, DType, Device, Layout, Shape,
//      Tensor, Var};` — every reference is now inline-qualified or
//      resolved through `cd_types`.
//   3. Moved every candle path (type aliases, constructor helpers,
//      safetensors I/O, `cd_bail!`) into `crate::cd_types`. Brought the
//      file from ~590 direct candle references down to 1 — only the
//      `CustomOp1` trait impl, which cannot be type-aliased on stable
//      Rust without `trait_alias`.
//   4. (#1082) Pruned `cd_types::{CpuStorage, CudaStorage, Layout}`:
//      every production use of these types already imports the
//      kt-native counterpart (`kiln_tensor::{CpuStorage, CudaStorage,
//      Layout}`) directly, so the candle facade no longer needs to
//      re-expose them. First pilot step in the
//      `cd_types::* -> kiln_tensor::*` migration.
//
// The candle_core crate dep itself stays because:
//   * `Var` is the canonical trainable parameter type used
//     throughout SFT/GRPO autograd
//   * `Tensor` is the autograd-tracked tensor consumed by `loss.backward()`
//   * `backprop` provides the `GradStore` API
//   * `safetensors::{save, load}` is the adapter on-disk format
//   * `CustomOp1` is the trait used by `InjectTensorGradient`
//
// Migrating off candle autograd is the larger Phase 7 task tracked by
// the kt-typed OPD/FLCE/RMSNorm forward+backward landings.
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use kiln_core::block::BlockTable;
use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
// (#1082) FLCE candle-typed surface relocated to `crate::flce_candle_shim`
// so `kiln-flce-kernel` could drop candle-core (3rd kernel-crate candle
// drop). The kernel crate keeps only the pure-kt `kt_api` + `kt_tape`.
// (#1082) candle FLCE shim removed; DEFAULT_CHUNK_SIZE now sourced from the
// kt-native kernel crate (same value). The candle `FlceMatmulProvider`/
// `FlceProvider`/`fused_linear_cross_entropy*` opt-in (KILN_CUDA_FLCE) is gone —
// FLCE is kt-native via `kiln_flce_kernel::kt_api::fused_linear_cross_entropy_phase_b_kt`.
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
    GDN_CHUNK_SIZE, GpuAttentionWeights, GpuWeights, GqaAttentionPrepared, LinearAttentionState,
    StreamingPrefillExecutionPolicy, gdn_attention_in_projections, gdn_attention_input_norm,
    gdn_attention_residual_block, gdn_gated_norm_from_recurrent, gdn_out_proj_from_gated_norm,
    gdn_qkv_from_mixed_training, gdn_recurrent_backward_no_grad, gdn_recurrent_forward_from_parts,
    gqa_attention_apply_output_gate, gqa_attention_core_prefill, gqa_attention_kv_prefill,
    gqa_attention_output_projection, gqa_attention_pre_o, gqa_attention_pre_o_chunked_prefill,
    gqa_attention_prepare_prefill, gqa_attention_q_gate_prefill, model_forward_embed,
    model_forward_final_norm, model_forward_head, model_forward_kt_with_policy,
    model_forward_no_head_with_policy, model_forward_paged_normed_hidden,
    model_forward_segment_with_policy, rms_norm, swiglu_ffn, transformer_mlp_down_from_gated,
    transformer_mlp_gated_hidden,
};
use kiln_model::lora_loader::{LoraLayerWeights, LoraProjectionWeights, LoraWeights};
use kiln_model::sampling::{greedy_sample, try_topk_on_device};
use kiln_model::{
    BackendCapabilityQueries, PagedKvCacheKt, TrainingOptimizerRequest, TrainingOptimizerRounding,
    TrainingOptimizerSupport,
};

use crate::replay::{
    self, BaseModel, Lineage, OutcomeRecord, OutcomeStatus, ParentLora, ReplayKind, ReplayLog,
    RequestRecord,
};
use crate::{
    AdvantageMode, BehaviorPolicy, ChatMessage, GrpoConfig, GrpoGroup, IsLevel, KlEstimator,
    KlReferencePolicy, LossAggregation, Optimizer, RewardFilterOnEmpty, SftConfig, SftExample,
    TurnKind,
};

/// Per-job context the HTTP layer hands the trainer so the accepted request
/// and its parent lineage can be audited from the on-disk artifacts.
///
/// `request_id` is the same UUID the queue uses for the job; `request_body`
/// is the verbatim deserialized request the HTTP handler accepted. The
/// trainer is responsible for resolving the effective seed (using
/// `config.seed.unwrap_or_else(|| rand::random())` so every run records a
/// concrete number), opening the parent lineage if `config.base_adapter` is
/// set, appending the replay record before stepping the optimizer, writing
/// `lineage.json`, and finally appending the outcome record.
#[derive(Debug, Clone)]
pub struct ReplayContext {
    pub request_id: String,
    pub kind: ReplayKind,
    pub request_body: serde_json::Value,
    pub base_model: BaseModel,
}

/// Build a default `BaseModel` description for the only model kiln supports.
///
/// `id` is fixed to `Qwen/Qwen3.5-4B`; `revision` is left unset; the config
/// digest is a SHA-256 of the JSON-serialized `ModelConfig` so lineage
/// verification can detect mismatched architectures even when `id` matches.
pub fn default_base_model(config: &ModelConfig) -> BaseModel {
    let digest = serde_json::to_string(config).ok().map(|s| {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(s.as_bytes());
        let hex: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        format!("sha256:{hex}")
    });
    BaseModel {
        id: "Qwen/Qwen3.5-4B".to_string(),
        revision: None,
        config_digest: digest,
    }
}

/// State threaded through a training run so the request can be appended
/// before the optimizer step and an outcome can be appended afterward.
pub struct ReplayState {
    log: ReplayLog,
    lineage: Lineage,
    request_id: String,
    started_at: std::time::Instant,
}

/// Reserved directory used by staged server training to pin the adapter being
/// rewritten. It is intentionally hidden from adapter registry scans.
pub const STARTING_ADAPTER_SNAPSHOT_DIR: &str = ".starting-adapter";

/// Open the replay log + lineage for a training run *before* the optimizer
/// step runs. Returns the effective seed so the trainer can apply it
/// consistently to RNG sources used during init.
///
/// Writes the request record (durable, fsynced) and `lineage.json` before
/// returning so a crash mid-step still leaves a recoverable trail.
pub fn open_replay_state(
    ctx: &ReplayContext,
    config_seed: Option<u64>,
    parent_adapter: Option<&str>,
    adapter_dir: &Path,
    adapter_name: &str,
) -> Result<(ReplayState, u64)> {
    open_replay_state_to(
        ctx,
        config_seed,
        parent_adapter,
        adapter_dir,
        adapter_dir,
        adapter_name,
    )
}

/// Staged-output variant of [`open_replay_state`]. Parent lineage resolves
/// from the prepared starting snapshot when rewriting that same adapter, or
/// from the durable registry otherwise. New replay state remains beneath
/// `output_adapter_dir` until the caller publishes it.
pub fn open_replay_state_to(
    ctx: &ReplayContext,
    config_seed: Option<u64>,
    parent_adapter: Option<&str>,
    adapter_dir: &Path,
    output_adapter_dir: &Path,
    adapter_name: &str,
) -> Result<(ReplayState, u64)> {
    let seed = config_seed.unwrap_or_else(|| rand::random());
    let kiln_commit = replay::kiln_commit();
    let submitted_at = chrono::Utc::now().to_rfc3339();
    let request = RequestRecord {
        request_id: ctx.request_id.clone(),
        kind: ctx.kind,
        request_body: ctx.request_body.clone(),
        seed,
        kiln_commit: kiln_commit.clone(),
        submitted_at: submitted_at.clone(),
    };

    let parent_lora = match parent_adapter {
        Some(name) => {
            let parent_dir = resolve_base_adapter_dir_from_roots(
                name,
                adapter_dir,
                output_adapter_dir,
                adapter_name,
            );
            let parent_lineage = replay::read_lineage(&parent_dir)
                .with_context(|| format!("reading parent lineage at {}", parent_dir.display()))?;
            Some(ParentLora {
                name: name.to_string(),
                replay_hash: parent_lineage.replay_hash,
            })
        }
        None => None,
    };

    let output_dir = output_adapter_dir.join(adapter_name);
    let log = ReplayLog::new(&output_dir)?;
    log.append_request(&request)?;

    let parent_hash = parent_lora.as_ref().map(|p| p.replay_hash.as_str());
    let replay_hash = replay::compute_replay_hash(parent_hash, &ctx.base_model, &[&request])?;

    let lineage = Lineage {
        schema_version: replay::LINEAGE_SCHEMA_VERSION,
        adapter_name: adapter_name.to_string(),
        base_model: ctx.base_model.clone(),
        parent_lora,
        kiln_commit,
        created_at: submitted_at,
        replay_hash,
    };
    replay::write_lineage(&output_dir, &lineage)?;

    Ok((
        ReplayState {
            log,
            lineage,
            request_id: ctx.request_id.clone(),
            started_at: std::time::Instant::now(),
        },
        seed,
    ))
}

/// Append an outcome record after the optimizer step finishes (or fails).
///
/// `result` is `Ok(final_loss)` on success, `Err(message)` on failure.
pub fn close_replay_state(state: ReplayState, result: Result<f64, String>) -> Result<()> {
    let elapsed = state.started_at.elapsed().as_secs_f64();
    let outcome = match result {
        Ok(loss) => OutcomeRecord {
            request_id: state.request_id,
            status: OutcomeStatus::Completed,
            final_loss: Some(loss),
            elapsed_secs: Some(elapsed),
            error: None,
        },
        Err(msg) => OutcomeRecord {
            request_id: state.request_id,
            status: OutcomeStatus::Failed,
            final_loss: None,
            elapsed_secs: Some(elapsed),
            error: Some(msg),
        },
    };
    state.log.append_outcome(&outcome)?;
    let _ = state.lineage; // lineage already written; kept for diagnostics if extended
    Ok(())
}

// ---------------------------------------------------------------------------
// (#1082) Small candle helpers that consolidate the most frequently-repeated
// inline-qualified `*` patterns in this file. Each helper takes
// and returns candle types (autograd-tracked); the win is purely textual:
// one helper-side `*` reference replaces N caller-side ones.
//
// These are NOT a `use candle_*` import — they are free functions /
// extension traits in the trainer module, so the audit invariant
// "trainer.rs has zero `use candle_*` imports at module top" still holds.
// Every candle prefix here is inline-qualified inside the helper body.
//
// Reduction accounting (post-bb1210dc, baseline 598 lines containing
// `*`):
//   * `TensorCastExt::to_f32_dtype` consolidates the most common pattern
//     in the file — `.to_dtype(DType::F32)?` chained onto a
//     tensor. Each rewritten call site loses one `*`
//     reference (the `DType::F32` qualifier). Net win: roughly `N - 5`
//     lines for N migrated sites.
//   * `zeros_f32_on` consolidates the
//     `Tensor::zeros(shape, DType::F32, device)`
//     constructor. Each rewritten call site loses two refs
//     (`Tensor::zeros` and `DType::F32`).
//   * `cpu_device` consolidates `Device::Cpu`.
// ---------------------------------------------------------------------------

/// Extension trait that consolidates the
/// `.to_dtype(DType::F32)?` cast — the single most common
/// inline-qualified pattern in this file. Method form keeps call sites
/// chainable.
trait TensorCastExt {
    fn to_f32_dtype(&self) -> Result<Tensor>;
}

impl TensorCastExt for Tensor {
    #[inline]
    fn to_f32_dtype(&self) -> Result<Tensor> {
        // Note: this body explicitly calls `to_dtype` to avoid infinite
        // recursion via the extension-trait method we are defining here.
        Ok(Tensor::to_dtype(self, DType::F32)?)
    }
}

/// Allocate a zero-filled F32 tensor on `device`. Consolidates the
/// `Tensor::zeros(shape, DType::F32, device)`
/// constructor.
#[inline]
fn zeros_f32_on<S: Into<Shape>>(shape: S, device: &Device) -> Result<Tensor> {
    Ok(Tensor::zeros(shape, DType::F32, device)?)
}

/// Return a candle CPU device. Consolidates `Device::Cpu`
/// (~70 sites pre-consolidation, mostly `let device = Device::Cpu;`
/// in `#[cfg(test)]` blocks).
#[inline]
fn cpu_device() -> Device {
    Device::Cpu
}

/// The reduction-axis marker for "the last dimension" — passed to
/// `Tensor::sum_keepdim` / `max_keepdim` / `mean_keepdim` / `log_sum_exp`
/// in this file. Consolidates `D::Minus1` (~21 sites
/// pre-consolidation).
const LAST_DIM: D = D::Minus1;

// `tensor_new` and `tensor_from_vec` (which require candle `NdArray`
// and `WithDType` generic bounds) have moved to `crate::cd_types` so
// this file holds zero direct candle paths for the generic constructor
// helpers. (#1082)

// (#1082) `var_from_tensor` (candle `Var::from_tensor`) removed: the
// trainable LoRA params are `kiln_param::Parameter` now, built via
// `lora_parameter_from_kt` below. The kt tape is the sole grad producer
// (no candle autograd `Var` tracking).

// ---------------------------------------------------------------------------
// (#1082) Type aliases for the most-repeated candle generic-parameter
// patterns in this file. These are NOT `use candle_*` imports — they are
// `type` aliases local to this module, so the audit invariant "trainer.rs
// has zero `use candle_*` imports at module top" still holds. The aliases
// keep all candle types fully spelled out at the alias definition site;
// every callsite that previously embedded two `*` references
// (e.g. `HashMap<TensorId, Tensor>`) collapses to
// one alias name (`GradMap`), netting out one candle reference per site.
// ---------------------------------------------------------------------------

/// (#1082) Map from a LoRA `Parameter`'s kt `TensorId` to its accumulated
/// kt gradient `Tensor`. Was `HashMap<candle TensorId, candle Tensor>`;
/// now fully kt-native (keys = `Parameter::tensor_id()`, values =
/// `kiln_tensor::Tensor`). Used by GRPO token-level cross-completion grad
/// accumulation.
type GradMap = std::collections::HashMap<KtTensorId, KtTensor>;

/// Concatenate a slice of `&Tensor` refs along `dim`. Consolidates the
/// Tensor::cat call site (~7 sites in the segment-level gradient +
/// activation stitching paths).
#[inline]
fn cat_tensors(refs: &[&Tensor], dim: usize) -> Result<Tensor> {
    Ok(Tensor::cat(refs, dim)?)
}

/// Allocate a zero-filled tensor with caller-supplied dtype + device.
/// Consolidates the Tensor::zeros constructor (~8 sites in segment / tile /
/// boundary-state init paths where the dtype is not statically F32).
#[inline]
fn zeros_dtype_on<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Tensor> {
    Ok(Tensor::zeros(shape, dtype, device)?)
}

/// Allocate a ones-filled tensor with caller-supplied dtype + device.
/// Consolidates the Tensor::ones constructor (~5 sites in q_norm/k_norm
/// init + gradient-test fixtures).
#[inline]
fn ones_dtype_on<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Tensor> {
    Ok(Tensor::ones(shape, dtype, device)?)
}

/// (#1082) Allocate a zero-filled LoRA `Parameter` (the LoRA-B init —
/// B=zeros so the initial LoRA contribution is zero). Replaces the candle
/// `Var::zeros` constructor. The AdamW moment allocation that also used
/// `Var::zeros` is gone (`kiln_optim::AdamW` owns its own moments keyed by
/// `Parameter::tensor_id()`).
fn lora_param_zeros(shape: (usize, usize), dtype: DType, device: &Device) -> Result<Parameter> {
    let n = shape.0 * shape.1;
    let data = vec![0.0f32; n];
    let master = build_lora_master_kt(&data, &[shape.0, shape.1], dtype, device)
        .context("lora_param_zeros: build kt LoRA-B master")?;
    Ok(lora_parameter_from_kt(master))
}

#[inline]
fn training_precision_policy_for_device(device: &Device) -> TrainingPrecisionPolicy {
    backend::training_precision_policy_for_device_kt(*device)
}

#[inline]
pub(crate) fn training_precision_policy_for_backend(
    backend: &dyn BackendRuntime,
) -> TrainingPrecisionPolicy {
    TrainingLossBackend::runtime_training_precision_policy(backend)
}

/// Validate the exact optimizer kind, derived LoRA dtype, and immutable write
/// policy before any resident model, trainable parameter, or optimizer-state
/// allocation. The per-step fallback guard remains necessary for dynamic
/// residency/dispatch failures.
pub(crate) fn ensure_training_optimizer_supported(
    workload: &str,
    backend: &dyn BackendRuntime,
    optimizer: Optimizer,
    base_weight_dtype: kiln_tensor::DType,
    lora_rank: usize,
) -> Result<TrainingOptimizerRequest> {
    let capabilities = BackendCapabilityQueries::backend_capabilities(backend);
    capabilities
        .training
        .resolve_optimizer_request(
            optimizer.kind(),
            base_weight_dtype,
            TrainingOptimizerRounding::RoundToNearest,
            lora_rank,
        )
        .map_err(|error| {
            anyhow::anyhow!(
                "{workload} optimizer is unsupported by backend `{}`: {error}",
                BackendIdentity::runtime_name(backend)
            )
        })
}

/// Cheap public-entry validation for optimizer hyperparameters and the exact
/// backend/dtype/rank tuple. Call this before source inspection or governor
/// initialization; execution paths repeat the capability check before their
/// first resident allocation so capability drift still fails closed.
pub(crate) fn ensure_training_optimizer_device_supported(
    workload: &str,
    weights: &GpuWeights,
    runtime_device: Device,
    optimizer: Optimizer,
    lora_rank: usize,
) -> Result<()> {
    optimizer
        .validate_hyperparameters()
        .with_context(|| format!("{workload}: invalid optimizer configuration"))?;
    TrainingOptimizerSupport::for_device(runtime_device)
        .resolve_optimizer_request(
            training_precision_policy_for_device(&runtime_device),
            optimizer.kind(),
            weights.embed_tokens.dtype(),
            TrainingOptimizerRounding::RoundToNearest,
            lora_rank,
        )
        .with_context(|| {
            format!(
                "{workload} optimizer is unsupported for configured runtime device {runtime_device}"
            )
        })?;
    Ok(())
}

pub(crate) fn ensure_training_optimizer_entry_supported(
    workload: &str,
    weights: &GpuWeights,
    runtime: &crate::TrainingRuntimeContext,
    optimizer: Optimizer,
    lora_rank: usize,
) -> Result<Device> {
    let runtime_device = training_device_for_weights(weights, runtime)
        .with_context(|| format!("{workload}: resolve runtime device"))?;
    ensure_training_optimizer_device_supported(
        workload,
        weights,
        runtime_device,
        optimizer,
        lora_rank,
    )?;
    Ok(runtime_device)
}

fn training_activation_bytes_per_elem_for_policy(
    weights: &GpuWeights,
    policy: TrainingPrecisionPolicy,
    has_linear_attention: bool,
) -> usize {
    const GDN_TAPE_EFFECTIVE_BYTES_PER_ELEM: usize = 10;

    if policy.uses_f32_activations_for_mixed_base_weights() {
        // Backends with mixed BF16 base weights and F32 training activations
        // keep hidden activations in F32 for the tape path.
        return 4;
    }
    let base = match weights.embed_tokens.dtype() {
        kiln_tensor::DType::BF16 | kiln_tensor::DType::F16 => 2,
        kiln_tensor::DType::F32 => 4,
        _ => 4,
    };
    if has_linear_attention
        || weights
            .layers
            .iter()
            .any(|layer| matches!(layer.attention, GpuAttentionWeights::Linear(_)))
    {
        // GDN replay records q/k/v, gate, recurrent, qk-norm, and gated-norm
        // tensors in addition to the hidden stream. Use an intentionally
        // inflated effective width so very long contexts prefer one-layer replay
        // scopes on tight VRAM. Long-context SFT spools checkpoint boundaries
        // off-device, so the extra segment count does not pin every boundary on
        // the GPU.
        base.max(GDN_TAPE_EFFECTIVE_BYTES_PER_ELEM)
    } else {
        base
    }
}

pub(crate) fn training_activation_bytes_per_elem_for_backend(
    weights: &GpuWeights,
    backend: &dyn BackendRuntime,
) -> usize {
    training_activation_bytes_per_elem_for_policy(
        weights,
        training_precision_policy_for_backend(backend),
        false,
    )
}

#[cfg(test)]
pub(crate) fn training_activation_bytes_per_elem(weights: &GpuWeights, device: &Device) -> usize {
    training_activation_bytes_per_elem_for_policy(
        weights,
        training_precision_policy_for_device(device),
        false,
    )
}

fn model_config_has_linear_attention(model_config: &ModelConfig) -> bool {
    model_config.num_full_attention_layers < model_config.num_layers
}

#[inline]
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
fn final_rmsnorm_backward_route_for_backend(
    backend: &dyn BackendRuntime,
) -> FinalRmsNormBackwardRoute {
    TrainingLossBackend::runtime_final_rmsnorm_backward_route(backend)
}

#[inline]
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
fn grpo_kl_auxiliary_route_for_backend(backend: &dyn BackendRuntime) -> GrpoKlAuxiliaryRoute {
    TrainingLossBackend::runtime_grpo_kl_auxiliary_route(backend)
}
// ---------------------------------------------------------------------------
// (#1082) The candle facade — type aliases, generic constructor helpers,
// safetensors I/O shims, and the `cd_bail!` macro — has been extracted to
// `crate::cd_types`. That keeps every direct candle path out of this
// file (except the one `impl` block near the bottom for `CustomOp1`,
// whose trait impl must live next to its struct).
//
// The wildcard re-import below brings every `pub(crate)` item from
// `cd_types` (type aliases like `Tensor` / `Var` / `Device` / `DType`
// / `Shape` / `GradStore` / `TensorId` / `D` / `CdResult`; the
// constructor helpers `tensor_new` and `tensor_from_vec`; and the
// safetensors shims) into scope so the ~16k call sites in this file
// keep working unchanged. The macro is brought in explicitly so the
// `cd_bail!(...)` ergonomic at the InjectTensorGradient site keeps
// resolving.
//
// NOTE: `use crate::cd_types::*` is not a `use candle_*` import — it is
// a wildcard re-export of items local to this crate, so the audit
// invariant "trainer.rs has zero `use candle_*` imports at module top"
// still holds.
// ---------------------------------------------------------------------------
use crate::cd_types::*;
// `cd_bail!` macro was used by the old `InjectTensorGradient::bwd`
// impl; that impl was deleted as part of the CP-4 caller flip
// (see comment block above the `full_attention_single_layer_tiled_mlp_reverse`
// function). No other call sites remain in this file. (#1082)

// (#1082) kt-native parameter + optimizer types. The LoRA params are
// `kiln_param::Parameter` (one stable kt `TensorId`, BF16 master + LoRA
// forward storage) and the optimizer is `kiln_optim::AdamW` (`OptimStep`
// keyed by `Parameter::tensor_id()`). These REPLACE the candle `Var` +
// `AdamWMoments{m,v:Var}` + `OptimizerState` machinery.
use kiln_optim::{
    AdamW as KtAdamW, AdamWHyperparameters as KtAdamWHyperparameters,
    AdamWMoments as KtHostAdamWMoments, MomentLocation as KtMomentLocation, Muon as KtMuon,
    MuonState as KtHostMuonState, OptimStep, StochasticRoundingPolicy,
};
use kiln_param::{AmpPolicy as KtAmpPolicy, ForwardStorage as KtForwardStorage, Parameter};
// kt tensor types used directly for LoRA param construction + grads.
use kiln_tensor::{DType as KtDType, Tensor as KtTensor};
// (#1082) `KtTensorId` was the `cd_types` alias for the kt tensor id, removed
// when `cd_types::TensorId` itself became `kiln_tensor::TensorId`. The kt
// `GradStore` is keyed on `kiln_tensor::TensorId`, so the grad-insert and
// grad-map sites here name it via this explicit alias.
use kiln_tensor::TensorId as KtTensorId;

/// AMP policy for a trainable LoRA parameter, derived from the backend-selected
/// LoRA storage dtype. Vulkan and the portable reference use F32 LoRA tensors;
/// stamping those parameters with the historical hard-coded BF16 policy made
/// the first host optimizer step silently narrow their master to BF16.
#[inline]
fn lora_amp_policy(dtype: KtDType) -> KtAmpPolicy {
    match dtype {
        KtDType::F32 => KtAmpPolicy::fp32_reference(),
        KtDType::BF16 => KtAmpPolicy::qwen3p5_4b_default(),
        KtDType::F16 => KtAmpPolicy {
            forward_compute_dtype: KtDType::F16,
            backward_compute_dtype: KtDType::F16,
            master_dtype: KtDType::F16,
            accumulation_dtype: KtDType::F32,
        },
        _ => unreachable!("LoRA parameters require F32, BF16, or F16 storage, got {dtype}"),
    }
}

/// (#1082) Build a trainable LoRA `Parameter` from a kt master tensor.
/// The forward storage IS the master (LoRA A/B are dense BF16, no
/// quantization), so `forward_storage().primary_tensor()` and
/// `backward_storage()` share the same kt tensor. The `Parameter`'s
/// stable kt `tensor_id` becomes the tape grad key + the optimizer
/// moment key.
#[inline]
fn lora_parameter_from_kt(master: KtTensor) -> Parameter {
    let policy = lora_amp_policy(master.dtype());
    Parameter::trainable(KtForwardStorage::Plain(master.clone()), master, policy)
}

/// Sample a Kaiming-uniform LoRA-A initialization.
///
/// When `rng` is `Some`, the values are drawn from the supplied RNG so the
/// init is byte-deterministic across runs; this is the path used when the
/// caller passes `seed: Some(_)`. When `rng` is `None`, we fall back to
/// `Var::rand_f64`, which uses the device-global RNG (seeded earlier with
/// `device.set_seed` on backends that support it).
// (#1082) Now returns a kt-native LoRA `Parameter` (Kaiming-uniform A,
// BF16 master) instead of a candle `Var`. The A values are drawn on the
// host (deterministic when `rng` is `Some`) and uploaded to a kt CUDA
// tensor via the bridge so the param's primary kt tensor lives on
// `device` — exactly where the kt tape forward + the resident-activation
// registry expect it. `dtype` is BF16 in production.
fn kaiming_uniform_a(
    rng: Option<&mut StdRng>,
    bound: f64,
    shape: (usize, usize),
    dtype: DType,
    device: &Device,
) -> Result<Parameter> {
    let bound_f32 = bound as f32;
    let n = shape.0 * shape.1;
    let data: Vec<f32> = match rng {
        Some(rng) => (0..n)
            .map(|_| rng.random_range(-bound_f32..bound_f32))
            .collect(),
        None => {
            // Deterministic-init contract: callers that pass `seed:
            // Some(_)` always hand us an `rng`. The `None` path (device
            // RNG) is the non-reproducible fallback; draw from a
            // thread RNG so we stay candle-free (candle `Var::rand_f64`
            // is gone with the autograd `Var`).
            let mut trng = StdRng::seed_from_u64(rand::random());
            (0..n)
                .map(|_| trng.random_range(-bound_f32..bound_f32))
                .collect()
        }
    };
    // Build the A master directly as a kt CUDA tensor on `device`.
    let master = build_lora_master_kt(&data, &[shape.0, shape.1], dtype, device)
        .context("kaiming_uniform_a: build kt LoRA-A master")?;
    Ok(lora_parameter_from_kt(master))
}

/// (#1082) Upload f32 host values to a kt tensor on `device`, cast to
/// `dtype` (BF16 in production). Lands directly on the requested device
/// via the candle-free `Tensor::from_vec_on` host->device upload — the
/// host->kt CUDA upload helper the old candle bridge was waiting for now
/// exists in kiln-tensor, so this is fully kt-native (no candle hop).
fn build_lora_master_kt(
    data: &[f32],
    shape: &[usize],
    dtype: DType,
    device: &Device,
) -> Result<KtTensor> {
    // Land the f32 host data on `device` (CPU direct, CUDA via H2D copy),
    // then cast to the requested dtype (BF16 in production).
    Tensor::from_vec_on(*device, data.to_vec(), shape.to_vec())?
        .to_dtype(dtype)
        .map_err(|e| anyhow::anyhow!("build_lora_master_kt: to_dtype: {e}"))
}

/// Convert our ChatMessage to the core tokenizer's ChatMessage.
fn to_core_messages(msgs: &[ChatMessage]) -> Vec<kiln_core::tokenizer::ChatMessage> {
    msgs.to_vec()
}

/// Which linear projections to train LoRA on.
const DEFAULT_TARGET_MODULES: &[&str] = crate::adapter_shape::TRAINABLE_TARGET_MODULES;
const ADAPTER_SMOKE_TEST_PROMPTS: &[&str] = &[
    "In one short sentence, name a primary color:",
    "Complete this sentence with a brief answer: The capital of France is",
    "Return a compact JSON tool call for a weather lookup in Paris:",
];
const ADAPTER_SMOKE_TEST_MAX_NEW_TOKENS: usize = 4;

struct AdapterSmokeGeneration {
    output: String,
    output_tokens: usize,
    elapsed_ms: u64,
}

/// Trainable LoRA parameters as kt-native `kiln_param::Parameter`s (#1082).
///
/// Each `Parameter` holds a BF16 master kt tensor and a stable kt
/// `TensorId`. The trainer threads each param's primary kt tensor into
/// the kt tape forward (via [`Self::as_lora_weights`] →
/// `LoraProjectionWeights`); the tape backward then yields a grad keyed
/// by `Parameter::tensor_id()`, which the kt optimizer
/// (`kiln_optim::AdamW`) consumes directly. NO candle autograd `Var`,
/// NO `loss.backward()`.
pub struct TrainableLoraParams {
    /// Per-layer, per-module (A, B) parameter pairs.
    /// Indexed as: layers[layer_idx].module_name -> (Param_A, Param_B)
    pub layers: Vec<TrainableLoraLayerParams>,
    /// LoRA pairs for the native MTP draft block (MTP training plan
    /// PR-B). `None` unless the post-SFT MTP alignment phase initialized
    /// them. Deliberately EXCLUDED from `all_params`/`all_params_mut` —
    /// the main training phase must not see parameters that its forward
    /// graph never touches; the alignment phase drives these through
    /// [`Self::mtp_params`]/[`Self::mtp_params_mut`].
    pub mtp: Option<TrainableLoraLayerParams>,
    pub rank: usize,
    pub alpha: f32,
    pub scale: f32,
}

/// Trainable LoRA A/B pairs for one transformer layer.
#[derive(Default)]
pub struct TrainableLoraLayerParams {
    pub q_proj: Option<(Parameter, Parameter)>,
    pub k_proj: Option<(Parameter, Parameter)>,
    pub v_proj: Option<(Parameter, Parameter)>,
    pub o_proj: Option<(Parameter, Parameter)>,
    pub in_proj_qkv: Option<(Parameter, Parameter)>,
    pub in_proj_z: Option<(Parameter, Parameter)>,
    pub gdn_out_proj: Option<(Parameter, Parameter)>,
    pub gate_proj: Option<(Parameter, Parameter)>,
    pub up_proj: Option<(Parameter, Parameter)>,
    pub down_proj: Option<(Parameter, Parameter)>,
}

struct LoraParamRef<'a> {
    layer_idx: usize,
    module: &'static str,
    matrix: &'static str,
    param: &'a Parameter,
}

fn push_lora_param_pair<'a>(
    params: &mut Vec<LoraParamRef<'a>>,
    layer_idx: usize,
    module: &'static str,
    pair: &'a Option<(Parameter, Parameter)>,
) {
    if let Some((a, b)) = pair {
        params.push(LoraParamRef {
            layer_idx,
            module,
            matrix: "A",
            param: a,
        });
        params.push(LoraParamRef {
            layer_idx,
            module,
            matrix: "B",
            param: b,
        });
    }
}

impl TrainableLoraParams {
    /// Initialize fresh LoRA parameters with Kaiming-uniform A and zero B.
    ///
    /// This matches the standard LoRA initialization:
    /// - A: Kaiming uniform (so the product A*B starts near zero)
    /// - B: zeros (so initial LoRA contribution is zero)
    ///
    /// Equivalent to `initialize_seeded(.., None)` — the device-global RNG
    /// drives A initialization. Tests, benches, and any caller that does not
    /// need byte-for-byte reproducibility should use this entry point.
    pub fn initialize(
        config: &ModelConfig,
        weights: &GpuWeights,
        rank: usize,
        alpha: f32,
        device: &Device,
    ) -> Result<Self> {
        Self::initialize_seeded(config, weights, rank, alpha, device, None)
    }

    /// Like [`initialize`], but uses a deterministic RNG seeded with `seed`
    /// to draw A. Used by the SFT/GRPO training loops so an adapter
    /// initialized with the same seed against the same base weights produces
    /// byte-identical LoRA-A tensors on every run, even on backends like the
    /// candle CPU device whose `set_seed` is a no-op.
    ///
    /// `seed: None` falls back to the device-global RNG (preserves the
    /// pre-replay behavior).
    pub fn initialize_seeded(
        config: &ModelConfig,
        weights: &GpuWeights,
        rank: usize,
        alpha: f32,
        device: &Device,
        seed: Option<u64>,
    ) -> Result<Self> {
        Self::initialize_seeded_with_precision_policy(
            config,
            weights,
            rank,
            alpha,
            device,
            seed,
            training_precision_policy_for_device(device),
        )
    }

    pub fn initialize_seeded_with_precision_policy(
        config: &ModelConfig,
        weights: &GpuWeights,
        rank: usize,
        alpha: f32,
        device: &Device,
        seed: Option<u64>,
        precision_policy: TrainingPrecisionPolicy,
    ) -> Result<Self> {
        // (#1082) kt `Device` has no `set_seed` (candle's was a no-op on CPU
        // anyway); the seeded `StdRng` below is what actually delivers
        // byte-for-byte determinism for LoRA-A. LoRA-B is plain zeros and
        // never touches a device RNG, so nothing else here needs seeding.
        let mut rng = seed.map(StdRng::seed_from_u64);

        let scale = alpha / rank as f32;
        let num_layers = config.num_layers;
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;

        // (#1082) LoRA-param dtype follows the backend-owned training precision
        // policy. CUDA/ROCm/Metal track the base dtype, while Vulkan keeps LoRA
        // parameters F32 to match its F32 activation policy.
        //
        // (#1443 step 2) On Vulkan the ACTIVATION dtype is F32 regardless of the
        // base WEIGHT dtype — the mixed-precision design keeps base projection
        // weights BF16 (the VRAM win) but runs F32 activations through the
        // F32-only Vulkan rmsnorm/softmax kernels, and the embedding output is
        // cast BF16→F32 at the head of the forward. So on Vulkan LoRA A/B are
        // ALWAYS F32 (matching the F32 activations / LoRA delta path) even on a
        // BF16 base; otherwise a BF16 LoRA on a BF16 base would mismatch the F32
        // `x2d` in `try_tape_lora_linear_kt`'s LoRA branch and decline. CUDA/Metal
        // keep `embed_tokens.dtype()` (BF16 activations end-to-end) unchanged.
        let lora_dtype =
            precision_policy.lora_parameter_dtype_for_base_weight(weights.embed_tokens.dtype());

        // Kaiming uniform bound: sqrt(1 / in_features) for A
        let bound_hidden = (1.0 / hidden as f64).sqrt();
        let bound_intermediate = (1.0 / intermediate as f64).sqrt();

        let mut layers = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let mut layer_params = TrainableLoraLayerParams::default();

            // Determine actual dimensions from the weight tensors
            let layer_weights = &weights.layers[layer_idx];

            for &module in DEFAULT_TARGET_MODULES {
                let (in_features, out_features, bound) = match module {
                    "q_proj" | "k_proj" | "v_proj" | "o_proj" => {
                        // Read the transposed weight's shape. Post-Phase 4.x
                        // residency, this tensor is a `broadcast_as` view
                        // that preserves the original `[hidden, out_dim]`
                        // dims while sharing 2 bytes of storage — so
                        // `.dims()` still returns the right shape and
                        // we don't have to mirror Qwen3.5-specific quirks
                        // (e.g. attn_output_gate doubling q_proj out_dim)
                        // here.
                        let w_t = match &layer_weights.attention {
                            kiln_model::forward::GpuAttentionWeights::Full(full) => match module {
                                "q_proj" => &full.q_proj_t,
                                "k_proj" => &full.k_proj_t,
                                "v_proj" => &full.v_proj_t,
                                "o_proj" => &full.o_proj_t,
                                _ => unreachable!(),
                            },
                            // Linear attention layers don't have q/k/v/o_proj
                            kiln_model::forward::GpuAttentionWeights::Linear(_) => {
                                continue;
                            }
                        };
                        let dims = w_t.dims();
                        anyhow::ensure!(
                            dims.len() == 2,
                            "expected rank-2 {module}_t for layer {layer_idx}, got {:?}",
                            dims
                        );
                        // Transposed weight is [in_features, out_features].
                        (dims[0], dims[1], bound_hidden)
                    }
                    "in_proj_qkv" | "in_proj_z" | "out_proj" => {
                        let w_t = match &layer_weights.attention {
                            kiln_model::forward::GpuAttentionWeights::Linear(linear) => {
                                match module {
                                    "in_proj_qkv" => &linear.in_proj_qkv_t,
                                    "in_proj_z" => &linear.in_proj_z_t,
                                    "out_proj" => &linear.out_proj_t,
                                    _ => unreachable!(),
                                }
                            }
                            // Full-attention layers use o_proj; these names are
                            // reserved for GDN/LinearAttention PEFT adapters.
                            kiln_model::forward::GpuAttentionWeights::Full(_) => {
                                continue;
                            }
                        };
                        let dims = w_t.dims();
                        anyhow::ensure!(
                            dims.len() == 2,
                            "expected rank-2 {module}_t for layer {layer_idx}, got {:?}",
                            dims
                        );
                        let in_features = dims[0];
                        let out_features = dims[1];
                        let bound = (1.0 / in_features as f64).sqrt();
                        (in_features, out_features, bound)
                    }
                    "gate_proj" => (hidden, intermediate, bound_hidden),
                    "up_proj" => (hidden, intermediate, bound_hidden),
                    "down_proj" => (intermediate, hidden, bound_intermediate),
                    _ => continue,
                };

                // A: [rank, in_features] — Kaiming uniform
                // Phase 10: BF16 storage + FP32-accumulate via tensor cores (audit
                // docs/audits/PHASE10_LORA_PRECISION_STUDY.md §5). (#1082) The
                // dtype now follows the base (`lora_dtype`): BF16 base ⇒ BF16
                // (unchanged); F32 base (Vulkan-only) ⇒ F32 so the tape recorder
                // matches the F32 activations.
                let a =
                    kaiming_uniform_a(rng.as_mut(), bound, (rank, in_features), lora_dtype, device)
                        .with_context(|| format!("init LoRA A for layer {layer_idx} {module}"))?;

                // B: [out_features, rank] — zeros
                let b = lora_param_zeros((out_features, rank), lora_dtype, device)
                    .with_context(|| format!("init LoRA B for layer {layer_idx} {module}"))?;

                match module {
                    "q_proj" => layer_params.q_proj = Some((a, b)),
                    "k_proj" => layer_params.k_proj = Some((a, b)),
                    "v_proj" => layer_params.v_proj = Some((a, b)),
                    "o_proj" => layer_params.o_proj = Some((a, b)),
                    "in_proj_qkv" => layer_params.in_proj_qkv = Some((a, b)),
                    "in_proj_z" => layer_params.in_proj_z = Some((a, b)),
                    "out_proj" => layer_params.gdn_out_proj = Some((a, b)),
                    "gate_proj" => layer_params.gate_proj = Some((a, b)),
                    "up_proj" => layer_params.up_proj = Some((a, b)),
                    "down_proj" => layer_params.down_proj = Some((a, b)),
                    _ => {}
                }
            }

            layers.push(layer_params);
        }

        Ok(Self {
            layers,
            mtp: None,
            rank,
            alpha,
            scale,
        })
    }

    /// Initialize LoRA A/B pairs for the native MTP draft block's seven
    /// modules (q/k/v/o + gate/up/down), shaped from the checkpoint's
    /// actual `mtp.*` tensors. Returns `Ok(false)` (no-op) when the
    /// checkpoint ships no MTP tensors. Same Kaiming-A / zero-B init and
    /// precision policy as the main layers. (MTP training plan PR-B.)
    pub fn initialize_mtp_seeded(
        &mut self,
        weights: &GpuWeights,
        device: &Device,
        seed: Option<u64>,
    ) -> Result<bool> {
        if weights.mtp.is_none() {
            return Ok(false);
        }
        let mtp = weights
            .mtp_weights()
            .context("initialize_mtp_seeded: materializing mtp.* tensors")?;
        let full = match &mtp.layer.attention {
            kiln_model::forward::GpuAttentionWeights::Full(full) => full,
            kiln_model::forward::GpuAttentionWeights::Linear(_) => {
                anyhow::bail!(
                    "initialize_mtp_seeded: MTP layer is linear-attention — the loader \
                     guarantees full attention; checkpoint is malformed"
                )
            }
        };
        let mut rng = seed.map(StdRng::seed_from_u64);
        let lora_dtype = training_precision_policy_for_device(device)
            .lora_parameter_dtype_for_base_weight(weights.embed_tokens.dtype());
        let rank = self.rank;

        let mut pairs = TrainableLoraLayerParams::default();
        let mut make_pair = |w_t: &kiln_tensor::Tensor,
                             module: &str|
         -> Result<(Parameter, Parameter)> {
            let dims = w_t.dims();
            anyhow::ensure!(
                dims.len() == 2,
                "initialize_mtp_seeded: expected rank-2 {module}_t, got {dims:?}"
            );
            let (in_features, out_features) = (dims[0], dims[1]);
            let bound = (1.0 / in_features as f64).sqrt();
            let a = kaiming_uniform_a(rng.as_mut(), bound, (rank, in_features), lora_dtype, device)
                .with_context(|| format!("init MTP LoRA A for {module}"))?;
            let b = lora_param_zeros((out_features, rank), lora_dtype, device)
                .with_context(|| format!("init MTP LoRA B for {module}"))?;
            Ok((a, b))
        };
        pairs.q_proj = Some(make_pair(&full.q_proj_t, "q_proj")?);
        pairs.k_proj = Some(make_pair(&full.k_proj_t, "k_proj")?);
        pairs.v_proj = Some(make_pair(&full.v_proj_t, "v_proj")?);
        pairs.o_proj = Some(make_pair(&full.o_proj_t, "o_proj")?);
        pairs.gate_proj = Some(make_pair(&mtp.layer.mlp.gate_proj_t, "gate_proj")?);
        pairs.up_proj = Some(make_pair(&mtp.layer.mlp.up_proj_t, "up_proj")?);
        pairs.down_proj = Some(make_pair(&mtp.layer.mlp.down_proj_t, "down_proj")?);
        self.mtp = Some(pairs);
        Ok(true)
    }

    /// MTP draft-block params for the alignment phase's grad lookup +
    /// optimizer (empty when [`Self::mtp`] is `None`).
    pub fn mtp_params(&self) -> Vec<&Parameter> {
        let mut out = Vec::new();
        if let Some(mtp) = &self.mtp {
            for pair in [
                &mtp.q_proj,
                &mtp.k_proj,
                &mtp.v_proj,
                &mtp.o_proj,
                &mtp.gate_proj,
                &mtp.up_proj,
                &mtp.down_proj,
            ]
            .into_iter()
            .flatten()
            {
                out.push(&pair.0);
                out.push(&pair.1);
            }
        }
        out
    }

    /// Mutable variant for the alignment phase's optimizer step.
    pub fn mtp_params_mut(&mut self) -> Vec<&mut Parameter> {
        let mut out: Vec<&mut Parameter> = Vec::new();
        if let Some(mtp) = self.mtp.as_mut() {
            for pair in [
                &mut mtp.q_proj,
                &mut mtp.k_proj,
                &mut mtp.v_proj,
                &mut mtp.o_proj,
                &mut mtp.gate_proj,
                &mut mtp.up_proj,
                &mut mtp.down_proj,
            ] {
                if let Some((a, b)) = pair.as_mut() {
                    out.push(a);
                    out.push(b);
                }
            }
        }
        out
    }

    /// Phase 4.1: register every LoRA `Var` (A and B for all modules
    /// across all layers) in the backend's resident activation
    /// registry. After this call, the trainer's training-time forward
    /// path dispatches the LoRA delta on-device via
    /// `lora_delta_resident` (which wraps the dispatch in
    /// `VulkanLoraOp` — a CustomOp3 with analytic backward), and the
    /// trainer's `apply_sgd_update` prefers the on-device
    /// `dispatch_sgd_step` path that writes to the registry buffer
    /// in-place.
    ///
    /// Caller invokes this once after [`initialize_seeded`], typically
    /// from `sft_train` / `grpo_train`. Test code that doesn't
    /// exercise the registry path skips this call — the trainer's
    /// existing fall-through logic handles the not-resident case
    /// transparently.
    ///
    /// Memory cost: one DMA upload per Var (~16 MB total for
    /// Qwen3.5-4B at rank=8 when GDN targets are present). On
    /// non-resident-supporting backends (CPU/Metal/CUDA today) the
    /// hook is a no-op.
    ///
    /// Lifecycle: each `apply_sgd_update` keeps the registry buffer
    /// in sync with the candle Var storage (or vice versa, depending
    /// on whether the on-device or CPU SGD path fired). The
    /// matching [`Self::evict_from_backend`] runs at training
    /// completion to release registry entries before the trainer
    /// returns.
    pub fn register_with_backend(&self, backend: &dyn BackendRuntime) -> Result<()> {
        if !ResidencyBackend::runtime_supports_resident_activation(backend) {
            return Ok(());
        }
        for param in self.all_params() {
            ResidencyBackend::runtime_register_resident_activation(
                backend,
                param.forward_storage().primary_tensor(),
            )?;
        }
        for param in self.mtp_params() {
            ResidencyBackend::runtime_register_resident_activation(
                backend,
                param.forward_storage().primary_tensor(),
            )?;
        }
        Ok(())
    }

    /// Inverse of [`register_with_backend`]: evict every LoRA param
    /// from the resident activation registry. Caller invokes this
    /// after the training loop completes (or per-step if Phase 4.1
    /// step 2 makes the registry the data-of-record and the trainer
    /// re-registers per step).
    pub fn evict_from_backend(&self, backend: &dyn BackendRuntime) {
        if !ResidencyBackend::runtime_supports_resident_activation(backend) {
            return;
        }
        for param in self.all_params() {
            ResidencyBackend::runtime_evict_resident_activation(
                backend,
                param.forward_storage().primary_tensor(),
            );
        }
        for param in self.mtp_params() {
            ResidencyBackend::runtime_evict_resident_activation(
                backend,
                param.forward_storage().primary_tensor(),
            );
        }
    }

    /// Pull every LoRA param's current value from the registry buffer
    /// back into its kt master storage.
    ///
    /// The on-device SGD and AdamW dispatch paths leave the kt master
    /// stale (the registry buffer is the source of truth between
    /// training steps). Callers that need the current master —
    /// `save_peft`, checkpoint writes — invoke this first. The refresh
    /// swaps the param's forward + backward storage to the resolved kt
    /// tensor while preserving `Parameter::tensor_id()` (anti-pattern 11).
    ///
    /// No-op on backends without resident-activation support. Returns
    /// the number of params synced for telemetry.
    pub fn sync_to_master(&mut self, backend: &dyn BackendRuntime) -> Result<usize> {
        if !ResidencyBackend::runtime_supports_resident_activation(backend) {
            return Ok(0);
        }
        let mut synced = 0;
        fn sync_one(
            backend: &dyn BackendRuntime,
            param: &mut Parameter,
            synced: &mut usize,
        ) -> Result<()> {
            let primary = param.forward_storage().primary_tensor().clone();
            if !ResidencyBackend::runtime_has_resident_activation(backend, &primary) {
                return Ok(());
            }
            let dims: Vec<usize> = primary.dims().to_vec();
            let dtype = primary.dtype();
            if let Some(resolved) = ResidencyBackend::runtime_resolve_resident_activation(
                backend, &primary, &dims, dtype,
            )? {
                let resolved = if resolved.device() == primary.device() {
                    resolved
                } else {
                    resolved
                        .to_device(primary.device())
                        .context("realign resolved LoRA parameter to its owner device")?
                };
                param
                    .replace_plain_trainable_tensor(resolved)
                    .map_err(|error| anyhow::anyhow!("sync LoRA parameter identity: {error}"))?;
                *synced += 1;
            }
            Ok(())
        }
        for param in self.all_params_mut() {
            sync_one(backend, param, &mut synced)?;
        }
        // The MTP draft-block pairs (alignment phase) sync too — save_peft
        // serializes them under the mtp.* keys.
        for param in self.mtp_params_mut() {
            sync_one(backend, param, &mut synced)?;
        }
        Ok(synced)
    }

    /// (#1082) Allocate AdamW optimizer state.
    ///
    /// CORRECTNESS (C1, candle-drop): the on-device CUDA AdamW kernel
    /// (`dispatch_adamw_step`) reads/writes the first/second-moment buffers
    /// **in place**. It therefore needs two *real* per-parameter device
    /// tensors `m`/`v` — distinct from the param. The candle-drop interim
    /// passed `&primary` twice in place of `m`/`v`, which aliased the moments
    /// onto the param (corrupting the weight and keeping NO Adam state). This
    /// restores the pre-flip design (`feaf2e99`'s `AdamWMoments{m,v}`) in
    /// kt-native form: a zero-init device `m`/`v` per LoRA param, matching the
    /// param master's shape/dtype/device, keyed by `Parameter::tensor_id()`.
    ///
    /// The CPU `kiln_optim::AdamW` instance is retained as the genuine
    /// non-resident host fallback (it owns its own host-side moments + grad
    /// dtype checks); the on-device path never touches it.
    ///
    /// `lr`/`beta1`/`beta2`/`eps`/`weight_decay` come from the trainer config
    /// (constant lr across steps — no scheduler). `device` is the param
    /// device; the moments are allocated on the *param's* device
    /// (`primary_tensor().device()`) so the CUDA gate
    /// (`cuda_optimizer_tensors_supported_for_kt`) sees four same-device
    /// same-dtype contiguous tensors.
    pub fn allocate_adamw_state(
        &self,
        lr: f64,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        _device: &Device,
    ) -> Result<OptimizerState> {
        let hp = KtAdamWHyperparameters {
            lr: lr as f32,
            beta1,
            beta2,
            eps,
            weight_decay,
        };
        let mut moments: HashMap<KtTensorId, KtAdamWMoments> = HashMap::new();
        for param in self.all_params() {
            let primary = param.forward_storage().primary_tensor();
            let dims: Vec<usize> = primary.dims().to_vec();
            let dtype = primary.dtype();
            // Allocate on the param's own device (CUDA → on-device zeros via
            // `Tensor::zeros_on` → `cuda_zeros_ctx`, NOT zeros_cpu) so the
            // dispatch gate's same-device/same-dtype/contiguous checks pass
            // and the kernel updates m/v in VRAM without a host round-trip.
            let m = KtTensor::zeros_on(primary.device(), dims.clone(), dtype)
                .with_context(|| "allocating AdamW first-moment tensor")?;
            let v = KtTensor::zeros_on(primary.device(), dims, dtype)
                .with_context(|| "allocating AdamW second-moment tensor")?;
            moments.insert(param.tensor_id(), KtAdamWMoments { m, v });
        }
        Ok(OptimizerState::AdamW {
            adamw: KtAdamW::new(hp),
            moments,
            host_authoritative: HashSet::new(),
            step: 0,
        })
    }

    /// (#1082) Allocate kt-native Muon optimizer state: one zero-init
    /// device momentum tensor per LoRA `Parameter` (same shape+dtype as
    /// the param master, on the param's own device so the on-device
    /// Newton-Schulz kernel's same-device/same-dtype/contiguous gate
    /// passes), plus the CPU reference `kiln_optim::Muon` host fallback.
    pub fn allocate_muon_state(
        &self,
        lr: f64,
        momentum: f32,
        nesterov: bool,
        ns_iters: u32,
        weight_decay: f32,
        _device: &Device,
    ) -> Result<OptimizerState> {
        let mut momenta: HashMap<KtTensorId, KtMuonMomentum> = HashMap::new();
        for param in self.all_params() {
            let primary = param.forward_storage().primary_tensor();
            let dims: Vec<usize> = primary.dims().to_vec();
            let dtype = primary.dtype();
            let m = KtTensor::zeros_on(primary.device(), dims, dtype)
                .with_context(|| "allocating Muon momentum tensor")?;
            momenta.insert(param.tensor_id(), KtMuonMomentum { m });
        }
        Ok(OptimizerState::Muon {
            muon: KtMuon::new(lr as f32, momentum, nesterov, ns_iters, weight_decay),
            momenta,
            host_authoritative: HashSet::new(),
            step: 0,
        })
    }
}

// (#1082) C1 fix: the kt-native restoration of the pre-flip candle
// `AdamWMoments{m,v:Var}`. The on-device CUDA AdamW kernel updates
// param/m/v in place, so it needs real per-param device moment tensors —
// not the param aliased onto itself. `KtAdamWMoments` holds those two
// device tensors; `OptimizerState.moments` maps `Parameter::tensor_id()`
// → `KtAdamWMoments`. The `KtAdamW` instance is the host (non-resident)
// fallback only.

/// (#1082) AdamW per-parameter first/second-moment device tensors.
///
/// `m` and `v` are zero-init kt tensors of the same shape+dtype as the
/// LoRA param master, allocated on the param's device. The on-device
/// CUDA AdamW kernel (`dispatch_adamw_step`) reads+writes both in place
/// each step (decoupled WD on the param, biased moments on m/v). Restores
/// the pre-flip candle `AdamWMoments{m: Var, v: Var}` in kt form.
pub struct KtAdamWMoments {
    pub m: KtTensor,
    pub v: KtTensor,
}

/// (#1082) Muon per-parameter momentum device tensor.
///
/// `m` is a zero-init kt tensor of the same shape+dtype as the LoRA
/// param master, allocated on the param's device. The on-device Muon
/// kernel (`runtime_dispatch_muon_step`) reads+writes it in place each
/// step (heavy-ball momentum), then orthogonalizes the look-ahead and
/// updates the param. Unlike AdamW there is no second moment — Muon's
/// state is a single momentum buffer per parameter.
pub struct KtMuonMomentum {
    pub m: KtTensor,
}

/// (#1082) kt-native optimizer state. One variant per stateful
/// optimizer; SGD is stateless and passes `None`.
///
/// - [`OptimizerState::AdamW`]: per-param device `m`/`v` (the real Adam
///   state on the resident/device path) + the CPU reference
///   `kiln_optim::AdamW` host fallback + a global 1-indexed `step`
///   counter (standard AdamW bias correction).
/// - [`OptimizerState::Muon`]: per-param device momentum `m` (the
///   heavy-ball state the on-device Newton-Schulz kernel updates) + the
///   CPU reference `kiln_optim::Muon` host fallback + a global `step`
///   counter (used only as a stochastic-rounding decorrelator on the
///   host path; Muon needs no bias correction).
///
/// The wrapper keeps the trainer's `opt_state: Option<&mut OptimizerState>`
/// signatures unchanged across all dispatch sites.
pub enum OptimizerState {
    AdamW {
        adamw: KtAdamW,
        moments: HashMap<KtTensorId, KtAdamWMoments>,
        host_authoritative: HashSet<KtTensorId>,
        step: u32,
    },
    Muon {
        muon: KtMuon,
        momenta: HashMap<KtTensorId, KtMuonMomentum>,
        host_authoritative: HashSet<KtTensorId>,
        step: u32,
    },
}

impl OptimizerState {
    /// Register every per-param device state tensor as a resident
    /// activation so the on-device kernel's `has_resident_activation`
    /// gate passes (otherwise it returns `false` → host fallback). For
    /// AdamW that is `m`+`v`; for Muon the single momentum `m`.
    ///
    /// No-op on backends without resident-activation support (the host
    /// `kiln_optim` references handle those).
    pub fn register_with_backend(&self, backend: &dyn BackendRuntime) -> Result<()> {
        if !ResidencyBackend::runtime_supports_resident_activation(backend) {
            return Ok(());
        }
        match self {
            OptimizerState::AdamW { moments, .. } => {
                for m in moments.values() {
                    ResidencyBackend::runtime_register_resident_activation(backend, &m.m)?;
                    ResidencyBackend::runtime_register_resident_activation(backend, &m.v)?;
                }
            }
            OptimizerState::Muon { momenta, .. } => {
                for mom in momenta.values() {
                    ResidencyBackend::runtime_register_resident_activation(backend, &mom.m)?;
                }
            }
        }
        Ok(())
    }

    /// Inverse of [`Self::register_with_backend`]: release every state
    /// tensor from the resident registry at training completion.
    pub fn evict_from_backend(&self, backend: &dyn BackendRuntime) {
        if !ResidencyBackend::runtime_supports_resident_activation(backend) {
            return;
        }
        match self {
            OptimizerState::AdamW { moments, .. } => {
                for m in moments.values() {
                    ResidencyBackend::runtime_evict_resident_activation(backend, &m.m);
                    ResidencyBackend::runtime_evict_resident_activation(backend, &m.v);
                }
            }
            OptimizerState::Muon { momenta, .. } => {
                for mom in momenta.values() {
                    ResidencyBackend::runtime_evict_resident_activation(backend, &mom.m);
                }
            }
        }
    }

    /// The global 1-indexed optimizer step counter (shared by both
    /// stateful variants).
    pub fn step_count(&self) -> u32 {
        match self {
            OptimizerState::AdamW { step, .. } => *step,
            OptimizerState::Muon { step, .. } => *step,
        }
    }

    fn checkpoint_rounding_policy(&self) -> StochasticRoundingPolicy {
        match self {
            OptimizerState::AdamW { adamw, .. } => adamw.rounding_policy(),
            OptimizerState::Muon { muon, .. } => muon.rounding_policy(),
        }
    }

    fn checkpoint_state_dtype(&self) -> Result<KtDType> {
        match self {
            OptimizerState::AdamW { moments, .. } => moments
                .values()
                .next()
                .map(|state| state.m.dtype())
                .context("AdamW checkpoint state has no parameter moments"),
            OptimizerState::Muon { momenta, .. } => momenta
                .values()
                .next()
                .map(|state| state.m.dtype())
                .context("Muon checkpoint state has no parameter momentum"),
        }
    }

    /// AdamW per-param moment map, if this is AdamW state (diagnostic /
    /// test accessor).
    pub fn adamw_moments(&self) -> Option<&HashMap<KtTensorId, KtAdamWMoments>> {
        match self {
            OptimizerState::AdamW { moments, .. } => Some(moments),
            OptimizerState::Muon { .. } => None,
        }
    }

    /// Muon per-param momentum map, if this is Muon state (diagnostic /
    /// test accessor).
    pub fn muon_momenta(&self) -> Option<&HashMap<KtTensorId, KtMuonMomentum>> {
        match self {
            OptimizerState::Muon { momenta, .. } => Some(momenta),
            OptimizerState::AdamW { .. } => None,
        }
    }

    /// Pull resident device optimizer buffers into their kt tensor owners.
    /// Host fallback state already lives in `adamw`/`muon` and needs no sync.
    pub fn sync_to_master(&mut self, backend: &dyn BackendRuntime) -> Result<usize> {
        if !ResidencyBackend::runtime_supports_resident_activation(backend) {
            return Ok(0);
        }
        fn sync_one(backend: &dyn BackendRuntime, tensor: &mut KtTensor) -> Result<bool> {
            if !ResidencyBackend::runtime_has_resident_activation(backend, tensor) {
                return Ok(false);
            }
            let id = tensor.id();
            let dims = tensor.dims().to_vec();
            let dtype = tensor.dtype();
            if let Some(resolved) = ResidencyBackend::runtime_resolve_resident_activation(
                backend, tensor, &dims, dtype,
            )? {
                let resolved = if resolved.device() == tensor.device() {
                    resolved
                } else {
                    resolved
                        .to_device(tensor.device())
                        .context("realign resolved optimizer state to its owner device")?
                };
                *tensor = checkpoint_tensor_with_id(resolved, id, "optimizer resident sync")?;
                Ok(true)
            } else {
                Ok(false)
            }
        }

        let mut synced = 0;
        match self {
            OptimizerState::AdamW { moments, .. } => {
                for state in moments.values_mut() {
                    synced += usize::from(sync_one(backend, &mut state.m)?);
                    synced += usize::from(sync_one(backend, &mut state.v)?);
                }
            }
            OptimizerState::Muon { momenta, .. } => {
                for state in momenta.values_mut() {
                    synced += usize::from(sync_one(backend, &mut state.m)?);
                }
            }
        }
        Ok(synced)
    }

    /// Capture optimizer tensors by stable parameter name into CPU storage.
    /// Device buffers and CPU fallback state share one F32 safetensors
    /// representation; per-param counters are U32 scalar tensors.
    pub(crate) fn capture_checkpoint_state(
        &mut self,
        params: &TrainableLoraParams,
        backend: &dyn BackendRuntime,
    ) -> Result<CheckpointTensorSnapshot> {
        self.sync_to_master(backend)?;
        let mut owned: Vec<(String, KtTensor)> = Vec::new();
        match self {
            OptimizerState::AdamW {
                adamw,
                moments,
                host_authoritative,
                step,
            } => {
                for (key, param) in params.checkpoint_params() {
                    let id = param.tensor_id();
                    let shape = param.forward_storage().primary_tensor().dims().to_vec();
                    let (m, v, param_step) = if host_authoritative.contains(&id) {
                        let host = adamw.moments(id).with_context(|| {
                            format!("checkpoint AdamW authoritative host moments missing for {key}")
                        })?;
                        anyhow::ensure!(
                            host.m.len() == param.forward_storage().primary_tensor().elem_count()
                                && host.v.len()
                                    == param.forward_storage().primary_tensor().elem_count(),
                            "checkpoint AdamW host moment shape drift for {key}"
                        );
                        (
                            KtTensor::from_vec_on(
                                kiln_tensor::Device::Cpu,
                                host.m.clone(),
                                shape.clone(),
                            )?,
                            KtTensor::from_vec_on(kiln_tensor::Device::Cpu, host.v.clone(), shape)?,
                            u32::try_from(host.step).with_context(|| {
                                format!("checkpoint AdamW per-param step overflow for {key}")
                            })?,
                        )
                    } else {
                        let state = moments.get(&id).with_context(|| {
                            format!("checkpoint AdamW device moments missing for {key}")
                        })?;
                        (
                            checkpoint_tensor_to_cpu_f32(&state.m, &format!("{key}.adamw.m"))?,
                            checkpoint_tensor_to_cpu_f32(&state.v, &format!("{key}.adamw.v"))?,
                            *step,
                        )
                    };
                    checkpoint_ensure_finite_f32(&m, &format!("{key}.adamw.m"))?;
                    checkpoint_ensure_finite_f32(&v, &format!("{key}.adamw.v"))?;
                    owned.push((format!("{key}.adamw.m"), m));
                    owned.push((format!("{key}.adamw.v"), v));
                    owned.push((
                        format!("{key}.adamw.step"),
                        KtTensor::from_vec_on(kiln_tensor::Device::Cpu, vec![param_step], vec![1])?,
                    ));
                }
            }
            OptimizerState::Muon {
                muon,
                momenta,
                host_authoritative,
                step,
            } => {
                for (key, param) in params.checkpoint_params() {
                    let id = param.tensor_id();
                    let shape = param.forward_storage().primary_tensor().dims().to_vec();
                    let (momentum, param_step) = if host_authoritative.contains(&id) {
                        let host = muon.momentum_for(id).with_context(|| {
                            format!("checkpoint Muon authoritative host momentum missing for {key}")
                        })?;
                        anyhow::ensure!(
                            host.m.len() == param.forward_storage().primary_tensor().elem_count(),
                            "checkpoint Muon host momentum shape drift for {key}"
                        );
                        (
                            KtTensor::from_vec_on(kiln_tensor::Device::Cpu, host.m.clone(), shape)?,
                            u32::try_from(host.step).with_context(|| {
                                format!("checkpoint Muon per-param step overflow for {key}")
                            })?,
                        )
                    } else {
                        let state = momenta.get(&id).with_context(|| {
                            format!("checkpoint Muon device momentum missing for {key}")
                        })?;
                        (
                            checkpoint_tensor_to_cpu_f32(
                                &state.m,
                                &format!("{key}.muon.momentum"),
                            )?,
                            *step,
                        )
                    };
                    checkpoint_ensure_finite_f32(&momentum, &format!("{key}.muon.momentum"))?;
                    owned.push((format!("{key}.muon.momentum"), momentum));
                    owned.push((
                        format!("{key}.muon.step"),
                        KtTensor::from_vec_on(kiln_tensor::Device::Cpu, vec![param_step], vec![1])?,
                    ));
                }
            }
        }
        CheckpointTensorSnapshot::new(owned, "optimizer")
    }

    /// Save optimizer state directly. Production loop checkpointing uses the
    /// split capture/publish path below so filesystem latency never extends
    /// the serving GPU write section; this wrapper remains useful to codecs
    /// and focused tests.
    pub fn save_checkpoint_state(
        &mut self,
        params: &TrainableLoraParams,
        backend: &dyn BackendRuntime,
        path: &Path,
    ) -> Result<()> {
        self.capture_checkpoint_state(params, backend)?.save(path)
    }

    /// Restore optimizer tensors into both the device-owned buffers and the
    /// CPU fallback optimizer. Populating both prevents a post-resume routing
    /// change from silently resetting momentum.
    pub fn load_checkpoint_state(
        &mut self,
        params: &TrainableLoraParams,
        path: &Path,
        expected_step: u32,
    ) -> Result<()> {
        let mut loaded = kiln_tensor::safetensors::load_cpu(path)
            .map_err(|error| anyhow::anyhow!("load checkpoint optimizer state: {error}"))?;
        let suffixes: &[&str] = match self {
            OptimizerState::AdamW { .. } => &["adamw.m", "adamw.v", "adamw.step"],
            OptimizerState::Muon { .. } => &["muon.momentum", "muon.step"],
        };
        let expected: BTreeSet<_> = params
            .checkpoint_param_keys()
            .into_iter()
            .flat_map(|key| suffixes.iter().map(move |suffix| format!("{key}.{suffix}")))
            .collect();
        let actual: BTreeSet<_> = loaded.keys().cloned().collect();
        anyhow::ensure!(
            actual == expected,
            "checkpoint optimizer tensor set mismatch: expected {expected:?}, found {actual:?}"
        );

        match self {
            OptimizerState::AdamW {
                adamw,
                moments,
                host_authoritative,
                step,
            } => {
                host_authoritative.clear();
                for (key, param) in params.checkpoint_params() {
                    let id = param.tensor_id();
                    let m_key = format!("{key}.adamw.m");
                    let v_key = format!("{key}.adamw.v");
                    let step_key = format!("{key}.adamw.step");
                    let m = loaded.remove(&m_key).expect("validated AdamW m must exist");
                    let v = loaded.remove(&v_key).expect("validated AdamW v must exist");
                    checkpoint_validate_f32_state_shape(&m, param, &m_key)?;
                    checkpoint_validate_f32_state_shape(&v, param, &v_key)?;
                    checkpoint_ensure_finite_f32(&m, &m_key)?;
                    checkpoint_ensure_finite_f32(&v, &v_key)?;
                    let param_step = checkpoint_read_step(
                        &loaded
                            .remove(&step_key)
                            .expect("validated AdamW step must exist"),
                        &step_key,
                    )?;
                    anyhow::ensure!(
                        param_step <= expected_step,
                        "checkpoint AdamW step {param_step} for {key} exceeds global step {expected_step}"
                    );
                    let state = moments.get_mut(&id).with_context(|| {
                        format!("checkpoint AdamW destination moments missing for {key}")
                    })?;
                    state.m = checkpoint_restore_state_tensor(&m, &state.m, &m_key)?;
                    state.v = checkpoint_restore_state_tensor(&v, &state.v, &v_key)?;
                    adamw.restore_moments(
                        id,
                        KtHostAdamWMoments {
                            m: m.to_vec::<f32>()?,
                            v: v.to_vec::<f32>()?,
                            step: u64::from(param_step),
                            location: KtMomentLocation::Device,
                        },
                    )?;
                }
                *step = expected_step;
            }
            OptimizerState::Muon {
                muon,
                momenta,
                host_authoritative,
                step,
            } => {
                host_authoritative.clear();
                for (key, param) in params.checkpoint_params() {
                    let id = param.tensor_id();
                    let momentum_key = format!("{key}.muon.momentum");
                    let step_key = format!("{key}.muon.step");
                    let momentum = loaded
                        .remove(&momentum_key)
                        .expect("validated Muon momentum must exist");
                    checkpoint_validate_f32_state_shape(&momentum, param, &momentum_key)?;
                    checkpoint_ensure_finite_f32(&momentum, &momentum_key)?;
                    let param_step = checkpoint_read_step(
                        &loaded
                            .remove(&step_key)
                            .expect("validated Muon step must exist"),
                        &step_key,
                    )?;
                    anyhow::ensure!(
                        param_step <= expected_step,
                        "checkpoint Muon step {param_step} for {key} exceeds global step {expected_step}"
                    );
                    let state = momenta.get_mut(&id).with_context(|| {
                        format!("checkpoint Muon destination momentum missing for {key}")
                    })?;
                    state.m = checkpoint_restore_state_tensor(&momentum, &state.m, &momentum_key)?;
                    muon.restore_momentum(
                        id,
                        KtHostMuonState {
                            m: momentum.to_vec::<f32>()?,
                            step: u64::from(param_step),
                        },
                    )?;
                }
                *step = expected_step;
            }
        }
        Ok(())
    }
}

fn checkpoint_tensor_to_cpu_f32(tensor: &KtTensor, label: &str) -> Result<KtTensor> {
    tensor
        .to_dtype(KtDType::F32)
        .and_then(|tensor| tensor.to_device(kiln_tensor::Device::Cpu))
        .and_then(|tensor| tensor.contiguous())
        .map_err(|error| anyhow::anyhow!("checkpoint optimizer tensor {label}: {error}"))
}

#[derive(Debug)]
pub(crate) struct CheckpointTensorSnapshot {
    kind: &'static str,
    tensors: Vec<(String, KtTensor)>,
}

impl CheckpointTensorSnapshot {
    fn new(tensors: Vec<(String, KtTensor)>, kind: &'static str) -> Result<Self> {
        let unique_names: BTreeSet<_> = tensors.iter().map(|(name, _)| name.as_str()).collect();
        anyhow::ensure!(
            unique_names.len() == tensors.len(),
            "checkpoint {kind} snapshot contains duplicate tensor names"
        );
        Ok(Self { kind, tensors })
    }

    pub(crate) fn save(&self, path: &Path) -> Result<()> {
        let tensors: HashMap<&str, &KtTensor> = self
            .tensors
            .iter()
            .map(|(key, tensor)| (key.as_str(), tensor))
            .collect();
        kiln_tensor::safetensors::save_cpu(&tensors, path)
            .map_err(|error| anyhow::anyhow!("save checkpoint {} state: {error}", self.kind))
    }
}

fn checkpoint_ensure_finite_f32(tensor: &KtTensor, label: &str) -> Result<()> {
    anyhow::ensure!(
        tensor.dtype() == KtDType::F32,
        "checkpoint optimizer tensor {label} must be F32, found {}",
        tensor.dtype()
    );
    anyhow::ensure!(
        tensor
            .to_vec::<f32>()?
            .iter()
            .all(|value| value.is_finite()),
        "checkpoint optimizer tensor {label} contains non-finite values"
    );
    Ok(())
}

fn checkpoint_ensure_finite_tensor(tensor: &KtTensor, label: &str) -> Result<()> {
    let values = tensor
        .to_dtype(KtDType::F32)
        .and_then(|tensor| tensor.to_device(kiln_tensor::Device::Cpu))
        .and_then(|tensor| tensor.contiguous())
        .and_then(|tensor| tensor.to_vec::<f32>())
        .map_err(|error| anyhow::anyhow!("read checkpoint tensor {label}: {error}"))?;
    anyhow::ensure!(
        values.iter().all(|value| value.is_finite()),
        "checkpoint tensor {label} contains non-finite values"
    );
    Ok(())
}

fn checkpoint_validate_f32_state_shape(
    tensor: &KtTensor,
    param: &Parameter,
    label: &str,
) -> Result<()> {
    checkpoint_ensure_finite_f32(tensor, label)?;
    let expected = param.forward_storage().primary_tensor().dims();
    anyhow::ensure!(
        tensor.dims() == expected,
        "checkpoint optimizer tensor {label} shape mismatch: expected {expected:?}, found {:?}",
        tensor.dims()
    );
    Ok(())
}

fn checkpoint_restore_state_tensor(
    source_f32: &KtTensor,
    destination: &KtTensor,
    label: &str,
) -> Result<KtTensor> {
    let restored = source_f32
        .to_dtype(destination.dtype())
        .and_then(|tensor| tensor.to_device(destination.device()))
        .map_err(|error| anyhow::anyhow!("restore checkpoint optimizer tensor {label}: {error}"))?;
    checkpoint_tensor_with_id(restored, destination.id(), label)
}

fn checkpoint_tensor_with_id(tensor: KtTensor, id: KtTensorId, label: &str) -> Result<KtTensor> {
    KtTensor::from_parts(tensor.storage().clone(), tensor.layout().clone(), id).map_err(|error| {
        anyhow::anyhow!("preserve checkpoint tensor identity for {label}: {error}")
    })
}

fn checkpoint_read_step(tensor: &KtTensor, label: &str) -> Result<u32> {
    anyhow::ensure!(
        tensor.dtype() == KtDType::U32 && tensor.dims() == [1],
        "checkpoint optimizer step tensor {label} must be U32[1], found {}{:?}",
        tensor.dtype(),
        tensor.dims()
    );
    Ok(tensor.to_vec::<u32>()?[0])
}

/// (#1082) Build `Option<OptimizerState>` from the configured optimizer:
/// `None` for SGD (stateless), `Some(KtAdamW-backed state)` for AdamW.
/// Consolidates the three identical production blocks that previously
/// `match`ed `config.optimizer` + pre-allocated candle moment `Var`s.
pub(crate) fn make_opt_state(
    params: &TrainableLoraParams,
    optimizer: Optimizer,
    lr: f64,
    device: &Device,
) -> Result<Option<OptimizerState>> {
    match optimizer {
        Optimizer::Sgd => Ok(None),
        Optimizer::AdamW {
            beta1,
            beta2,
            eps,
            weight_decay,
        } => Ok(Some(params.allocate_adamw_state(
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            device,
        )?)),
        Optimizer::Muon {
            momentum,
            nesterov,
            ns_iters,
            weight_decay,
        } => Ok(Some(params.allocate_muon_state(
            lr,
            momentum,
            nesterov,
            ns_iters,
            weight_decay,
            device,
        )?)),
    }
}

fn checkpoint_parameter_key(layer_idx: usize, module: &str, matrix: &str) -> String {
    let sub = if matches!(
        module,
        "q_proj" | "k_proj" | "v_proj" | "o_proj" | "in_proj_qkv" | "in_proj_z" | "out_proj"
    ) {
        "self_attn"
    } else {
        "mlp"
    };
    format!("base_model.model.model.layers.{layer_idx}.{sub}.{module}.lora_{matrix}.weight")
}

impl TrainableLoraParams {
    /// Convert trainable params to a `LoraWeights` for use with the forward pass.
    ///
    /// The returned `LoraWeights` holds tensors that are backed by our Vars,
    /// so autograd tracks all operations through them.
    pub fn as_lora_weights(&self) -> LoraWeights {
        let layers: Vec<LoraLayerWeights> = self
            .layers
            .iter()
            .map(|lp| {
                // (#1082) `LoraProjectionWeights.a/.b` are kt `Tensor` now;
                // thread each param's primary kt tensor (the BF16 LoRA
                // master) straight in. The tape forward records ops over
                // these kt tensors, so the backward grad keys on
                // `Parameter::tensor_id()` == `a/.b.id()`.
                let make_proj =
                    |pair: &Option<(Parameter, Parameter)>| -> Option<LoraProjectionWeights> {
                        pair.as_ref().map(|(a, b)| LoraProjectionWeights {
                            a: a.forward_storage().primary_tensor().clone(),
                            b: b.forward_storage().primary_tensor().clone(),
                        })
                    };
                LoraLayerWeights {
                    q_proj: make_proj(&lp.q_proj),
                    k_proj: make_proj(&lp.k_proj),
                    v_proj: make_proj(&lp.v_proj),
                    o_proj: make_proj(&lp.o_proj),
                    in_proj_qkv: make_proj(&lp.in_proj_qkv),
                    in_proj_z: make_proj(&lp.in_proj_z),
                    gdn_out_proj: make_proj(&lp.gdn_out_proj),
                    gate_proj: make_proj(&lp.gate_proj),
                    up_proj: make_proj(&lp.up_proj),
                    down_proj: make_proj(&lp.down_proj),
                    ..Default::default()
                }
            })
            .collect();

        let make_proj_view =
            |pair: &Option<(Parameter, Parameter)>| -> Option<LoraProjectionWeights> {
                pair.as_ref().map(|(a, b)| LoraProjectionWeights {
                    a: a.forward_storage().primary_tensor().clone(),
                    b: b.forward_storage().primary_tensor().clone(),
                })
            };
        let mtp = self.mtp.as_ref().map(|mp| LoraLayerWeights {
            q_proj: make_proj_view(&mp.q_proj),
            k_proj: make_proj_view(&mp.k_proj),
            v_proj: make_proj_view(&mp.v_proj),
            o_proj: make_proj_view(&mp.o_proj),
            gate_proj: make_proj_view(&mp.gate_proj),
            up_proj: make_proj_view(&mp.up_proj),
            down_proj: make_proj_view(&mp.down_proj),
            ..Default::default()
        });

        LoraWeights {
            layers,
            mtp,
            rank: self.rank,
            alpha: self.alpha,
            scale: self.scale,
            source_identity: None,
        }
    }

    /// Collect all LoRA `Parameter` references for grad lookup + updates.
    pub fn all_params(&self) -> Vec<&Parameter> {
        self.all_params_with_modules()
            .into_iter()
            .map(|entry| entry.param)
            .collect()
    }

    fn all_params_with_modules(&self) -> Vec<LoraParamRef<'_>> {
        let mut params = Vec::new();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            push_lora_param_pair(&mut params, layer_idx, "q_proj", &layer.q_proj);
            push_lora_param_pair(&mut params, layer_idx, "k_proj", &layer.k_proj);
            push_lora_param_pair(&mut params, layer_idx, "v_proj", &layer.v_proj);
            push_lora_param_pair(&mut params, layer_idx, "o_proj", &layer.o_proj);
            push_lora_param_pair(&mut params, layer_idx, "in_proj_qkv", &layer.in_proj_qkv);
            push_lora_param_pair(&mut params, layer_idx, "in_proj_z", &layer.in_proj_z);
            push_lora_param_pair(&mut params, layer_idx, "out_proj", &layer.gdn_out_proj);
            push_lora_param_pair(&mut params, layer_idx, "gate_proj", &layer.gate_proj);
            push_lora_param_pair(&mut params, layer_idx, "up_proj", &layer.up_proj);
            push_lora_param_pair(&mut params, layer_idx, "down_proj", &layer.down_proj);
        }
        params
    }

    /// Mutable variant — the optimizer step + `sync_to_master` mutate
    /// each `Parameter`'s storage in place (preserving `tensor_id`).
    /// Same traversal order as [`all_params`].
    pub fn all_params_mut(&mut self) -> Vec<&mut Parameter> {
        let mut out: Vec<&mut Parameter> = Vec::new();
        for layer in &mut self.layers {
            for pair in [
                &mut layer.q_proj,
                &mut layer.k_proj,
                &mut layer.v_proj,
                &mut layer.o_proj,
                &mut layer.in_proj_qkv,
                &mut layer.in_proj_z,
                &mut layer.gdn_out_proj,
                &mut layer.gate_proj,
                &mut layer.up_proj,
                &mut layer.down_proj,
            ] {
                if let Some((a, b)) = pair.as_mut() {
                    out.push(a);
                    out.push(b);
                }
            }
        }
        out
    }

    /// Stable PEFT-compatible names for the main-loop trainable parameters.
    /// Tensor IDs are process-local and must never appear in durable optimizer
    /// state, so checkpoint save/restore joins state through this ordering.
    fn checkpoint_param_keys(&self) -> Vec<String> {
        self.all_params_with_modules()
            .into_iter()
            .map(|entry| checkpoint_parameter_key(entry.layer_idx, entry.module, entry.matrix))
            .collect()
    }

    fn checkpoint_params(&self) -> Vec<(String, &Parameter)> {
        self.checkpoint_param_keys()
            .into_iter()
            .zip(self.all_params())
            .collect()
    }

    fn checkpoint_params_mut(&mut self) -> Vec<(String, &mut Parameter)> {
        let keys = self.checkpoint_param_keys();
        keys.into_iter().zip(self.all_params_mut()).collect()
    }

    /// Capture exact main-loop adapter parameters into CPU storage without
    /// PEFT receipts/config. The enclosing checkpoint writer owns atomicity
    /// and checksums.
    pub(crate) fn capture_checkpoint_parameters(&self) -> Result<CheckpointTensorSnapshot> {
        let mut owned = Vec::with_capacity(self.all_params().len());
        for (key, param) in self.checkpoint_params() {
            let tensor = param
                .forward_storage()
                .primary_tensor()
                .to_device(kiln_tensor::Device::Cpu)
                .and_then(|tensor| tensor.contiguous())
                .map_err(|error| {
                    anyhow::anyhow!("checkpoint adapter parameter {key}: to CPU: {error}")
                })?;
            owned.push((key, tensor));
        }
        CheckpointTensorSnapshot::new(owned, "adapter parameter")
    }

    /// Save adapter parameters directly. Production loop checkpointing uses
    /// a coordinated CPU snapshot and publishes it after releasing the GPU
    /// lock; this wrapper remains useful to codecs and focused tests.
    pub fn save_checkpoint_parameters(&self, path: &Path) -> Result<()> {
        self.capture_checkpoint_parameters()?.save(path)
    }

    /// Restore exact main-loop adapter parameters by stable name. Missing,
    /// extra, shape-drifted, or dtype-drifted tensors fail before mutation.
    pub fn load_checkpoint_parameters(&mut self, path: &Path) -> Result<()> {
        let mut loaded = kiln_tensor::safetensors::load_cpu(path)
            .map_err(|error| anyhow::anyhow!("load checkpoint adapter parameters: {error}"))?;
        let expected: BTreeSet<_> = self.checkpoint_param_keys().into_iter().collect();
        let actual: BTreeSet<_> = loaded.keys().cloned().collect();
        anyhow::ensure!(
            actual == expected,
            "checkpoint adapter parameter set mismatch: expected {expected:?}, found {actual:?}"
        );

        // Validate the entire file before replacing the first live parameter.
        for (key, param) in self.checkpoint_params() {
            let tensor = loaded
                .get(&key)
                .with_context(|| format!("checkpoint adapter parameter {key} missing"))?;
            let current = param.forward_storage().primary_tensor();
            anyhow::ensure!(
                tensor.dims() == current.dims(),
                "checkpoint adapter parameter {key} shape mismatch: expected {:?}, found {:?}",
                current.dims(),
                tensor.dims()
            );
            anyhow::ensure!(
                tensor.dtype() == current.dtype(),
                "checkpoint adapter parameter {key} dtype mismatch: expected {}, found {}",
                current.dtype(),
                tensor.dtype()
            );
            checkpoint_ensure_finite_tensor(tensor, &key)?;
        }

        for (key, param) in self.checkpoint_params_mut() {
            let current_device = param.forward_storage().primary_tensor().device();
            let tensor = loaded
                .remove(&key)
                .expect("validated checkpoint parameter must exist")
                .to_device(current_device)
                .map_err(|error| {
                    anyhow::anyhow!("checkpoint adapter parameter {key}: to device: {error}")
                })?;
            param
                .replace_plain_trainable_tensor(tensor)
                .map_err(|error| {
                    anyhow::anyhow!("restore checkpoint adapter parameter {key}: {error}")
                })?;
        }
        Ok(())
    }

    /// Load a previously-saved PEFT adapter into the existing Vars,
    /// replacing the seeded-init values.
    ///
    /// Reads `<adapter_dir>/adapter_model.safetensors` and copies each
    /// tensor into the matching Var via `Var::set`. The adapter's rank,
    /// alpha, and target_modules must match this `TrainableLoraParams`
    /// instance — those are passed at `initialize_seeded` time and not
    /// reconfigurable here.
    ///
    /// Used by Phase 3 verifier-free chaining: take a strong Phase 2
    /// adapter, run `--no-policy-loss` from those weights, save a new
    /// adapter that's a verifier-free continuation. Without this, the
    /// `--base-adapter` CLI flag is effectively a lineage label.
    ///
    /// Returns the number of tensors loaded. Training entry points call
    /// `validate_base_adapter_compatibility` before this method so missing,
    /// extra, rank-mismatched, or shape-mismatched tensors fail before
    /// optimizer setup instead of leaving seeded-init gaps.
    // (#1082) `&mut self` now: loading replaces each `Parameter`'s
    // forward + backward storage (preserving `tensor_id`) rather than
    // calling candle `Var::set`. Safetensors load stays a candle island
    // (the `safetensors_load_file` shim is candle); the loaded candle
    // tensor is bridged to kt and installed.
    // // (#1082) bridge — safetensors I/O is still a candle island.
    pub fn load_from_safetensors(&mut self, adapter_dir: &Path, device: &Device) -> Result<usize> {
        let st_path = adapter_dir.join("adapter_model.safetensors");
        // (#1082) kt-native safetensors load — `kiln_tensor::safetensors::load_cpu`
        // returns CPU kt tensors; each is moved to the training device and
        // installed directly. No candle: was a candle `safetensors::load` + a
        // per-tensor candle->kt borrow.
        let tensors = kiln_tensor::safetensors::load_cpu(&st_path)
            .with_context(|| format!("loading adapter safetensors from {}", st_path.display()))?;

        let install = |param: &mut Parameter, t: &KtTensor, key: &str| -> Result<()> {
            let kt = t
                .to_device(*device)
                .map_err(|e| anyhow::anyhow!("load adapter {key}: to device: {e}"))?;
            param
                .replace_plain_trainable_tensor(kt)
                .map_err(|error| anyhow::anyhow!("load PEFT adapter parameter {key}: {error}"))?;
            Ok(())
        };

        let mut loaded = 0usize;
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let mut load_proj = |name: &str,
                                 pair: &mut Option<(Parameter, Parameter)>,
                                 is_attn: bool|
             -> Result<()> {
                if let Some((a, b)) = pair.as_mut() {
                    let sub = if is_attn { "self_attn" } else { "mlp" };
                    let prefix = format!("base_model.model.model.layers.{layer_idx}.{sub}.{name}");
                    let a_key = format!("{prefix}.lora_A.weight");
                    let b_key = format!("{prefix}.lora_B.weight");
                    if let Some(a_t) = tensors.get(&a_key) {
                        install(a, a_t, &a_key)?;
                        loaded += 1;
                    }
                    if let Some(b_t) = tensors.get(&b_key) {
                        install(b, b_t, &b_key)?;
                        loaded += 1;
                    }
                }
                Ok(())
            };

            load_proj("q_proj", &mut layer.q_proj, true)?;
            load_proj("k_proj", &mut layer.k_proj, true)?;
            load_proj("v_proj", &mut layer.v_proj, true)?;
            load_proj("o_proj", &mut layer.o_proj, true)?;
            load_proj("in_proj_qkv", &mut layer.in_proj_qkv, true)?;
            load_proj("in_proj_z", &mut layer.in_proj_z, true)?;
            load_proj("out_proj", &mut layer.gdn_out_proj, true)?;
            load_proj("gate_proj", &mut layer.gate_proj, false)?;
            load_proj("up_proj", &mut layer.up_proj, false)?;
            load_proj("down_proj", &mut layer.down_proj, false)?;
        }

        tracing::info!(
            path = %adapter_dir.display(),
            num_tensors = loaded,
            "loaded base adapter into TrainableLoraParams"
        );
        Ok(loaded)
    }

    /// Save the trained adapter in PEFT-compatible format.
    ///
    /// Creates `adapter_config.json` and `adapter_model.safetensors` that can
    /// be loaded by the existing `LoraWeights::load()` method.
    pub fn save_peft(&self, output_dir: &Path, _num_layers: usize) -> Result<PathBuf> {
        std::fs::create_dir_all(output_dir)
            .with_context(|| format!("creating adapter dir: {}", output_dir.display()))?;

        // Write adapter_config.json
        let config = serde_json::json!({
            "r": self.rank,
            "lora_alpha": self.alpha,
            "target_modules": crate::adapter_shape::TRAINABLE_TARGET_MODULES,
            "task_type": "CAUSAL_LM",
            "bias": "none",
            "peft_type": "LORA",
        });
        let config_path = output_dir.join("adapter_config.json");
        std::fs::write(&config_path, serde_json::to_string_pretty(&config)?)?;

        // Collect all LoRA tensors for safetensors serialization.
        // (#1082) kt-native: read each `Parameter`'s primary kt tensor, move it
        // to CPU (contiguous) for the writer, and serialize via
        // `kiln_tensor::safetensors::save_cpu`. No candle: was a per-tensor
        // kt->candle copy + a candle writer.
        let mut owned: Vec<(String, KtTensor)> = Vec::new();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let mut save_proj =
                |name: &str, pair: &Option<(Parameter, Parameter)>, is_attn: bool| -> Result<()> {
                    if let Some((a, b)) = pair {
                        let sub = if is_attn { "self_attn" } else { "mlp" };
                        let prefix =
                            format!("base_model.model.model.layers.{layer_idx}.{sub}.{name}");
                        let to_cpu = |kt: &KtTensor, key: &str| -> Result<KtTensor> {
                            kt.to_device(kiln_tensor::Device::Cpu)
                                .and_then(|t| t.contiguous())
                                .map_err(|e| anyhow::anyhow!("save adapter {key}: to cpu: {e}"))
                        };
                        let a_key = format!("{prefix}.lora_A.weight");
                        let b_key = format!("{prefix}.lora_B.weight");
                        let a_cpu = to_cpu(a.forward_storage().primary_tensor(), &a_key)?;
                        let b_cpu = to_cpu(b.forward_storage().primary_tensor(), &b_key)?;
                        owned.push((a_key, a_cpu));
                        owned.push((b_key, b_cpu));
                    }
                    Ok(())
                };

            save_proj("q_proj", &layer.q_proj, true)?;
            save_proj("k_proj", &layer.k_proj, true)?;
            save_proj("v_proj", &layer.v_proj, true)?;
            save_proj("o_proj", &layer.o_proj, true)?;
            save_proj("in_proj_qkv", &layer.in_proj_qkv, true)?;
            save_proj("in_proj_z", &layer.in_proj_z, true)?;
            save_proj("out_proj", &layer.gdn_out_proj, true)?;
            save_proj("gate_proj", &layer.gate_proj, false)?;
            save_proj("up_proj", &layer.up_proj, false)?;
            save_proj("down_proj", &layer.down_proj, false)?;
        }

        // MTP draft-block LoRA (MTP training plan PR-B). Keyed under
        // `...mtp.layers.0...` — the loader parses these into
        // `LoraWeights.mtp` (and `parse_peft_key.is_mtp` keeps them from
        // aliasing main layer 0).
        if let Some(mtp) = &self.mtp {
            let mut save_mtp =
                |name: &str, pair: &Option<(Parameter, Parameter)>, is_attn: bool| -> Result<()> {
                    if let Some((a, b)) = pair {
                        let sub = if is_attn { "self_attn" } else { "mlp" };
                        let prefix = format!("base_model.model.model.mtp.layers.0.{sub}.{name}");
                        let to_cpu = |kt: &KtTensor, key: &str| -> Result<KtTensor> {
                            kt.to_device(kiln_tensor::Device::Cpu)
                                .and_then(|t| t.contiguous())
                                .map_err(|e| anyhow::anyhow!("save adapter {key}: to cpu: {e}"))
                        };
                        let a_key = format!("{prefix}.lora_A.weight");
                        let b_key = format!("{prefix}.lora_B.weight");
                        let a_cpu = to_cpu(a.forward_storage().primary_tensor(), &a_key)?;
                        let b_cpu = to_cpu(b.forward_storage().primary_tensor(), &b_key)?;
                        owned.push((a_key, a_cpu));
                        owned.push((b_key, b_cpu));
                    }
                    Ok(())
                };
            save_mtp("q_proj", &mtp.q_proj, true)?;
            save_mtp("k_proj", &mtp.k_proj, true)?;
            save_mtp("v_proj", &mtp.v_proj, true)?;
            save_mtp("o_proj", &mtp.o_proj, true)?;
            save_mtp("gate_proj", &mtp.gate_proj, false)?;
            save_mtp("up_proj", &mtp.up_proj, false)?;
            save_mtp("down_proj", &mtp.down_proj, false)?;
        }

        let st_path = output_dir.join("adapter_model.safetensors");
        let save_map: std::collections::HashMap<&str, &KtTensor> =
            owned.iter().map(|(k, v)| (k.as_str(), v)).collect();
        kiln_tensor::safetensors::save_cpu(&save_map, &st_path)
            .with_context(|| format!("saving safetensors to {}", st_path.display()))?;
        let adapter_name = output_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("adapter");
        crate::adapter_output::write_adapter_output_receipt(output_dir, adapter_name, None)
            .with_context(|| {
                format!("writing adapter output receipt to {}", output_dir.display())
            })?;

        tracing::info!(
            path = %output_dir.display(),
            num_tensors = owned.len(),
            "saved PEFT adapter"
        );

        Ok(output_dir.to_path_buf())
    }
}

/// Flow-control verdict a progress callback returns. `Stop` requests a
/// cooperative cancellation at the next step boundary — the run aborts
/// with a "training cancelled by user" error and the receipt records
/// failure_reason "cancelled".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainControl {
    Continue,
    Stop,
}

/// Progress callback for training. Returns a [`TrainControl`] verdict —
/// the per-step call site doubles as the cancellation point, so a running
/// job can be stopped without threading a separate flag through every
/// train loop.
pub type ProgressCallback = Box<dyn Fn(TrainingProgress) -> TrainControl + Send>;

/// Training progress update.
#[derive(Debug, Clone)]
pub struct TrainingProgress {
    pub epoch: usize,
    pub total_epochs: usize,
    pub step: usize,
    pub total_steps: usize,
    pub loss: f64,
    /// Overall progress as a fraction [0, 1].
    pub progress: f32,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GrpoBenchmarkTimings {
    pub tokenize_ms: f64,
    pub mask_build_ms: f64,
    pub reference_forward_ms: f64,
    pub policy_forward_ms: f64,
    pub backward_ms: f64,
    pub optimizer_ms: f64,
    #[serde(default)]
    pub gpu_writer_wait_ms: f64,
    #[serde(default)]
    pub gpu_writer_held_ms: f64,
    #[serde(default)]
    pub gpu_writer_acquisitions: u64,
}

impl GrpoBenchmarkTimings {
    fn add_tokenize(&mut self, elapsed: Duration) {
        self.tokenize_ms += elapsed.as_secs_f64() * 1000.0;
    }

    fn add_mask_build(&mut self, elapsed: Duration) {
        self.mask_build_ms += elapsed.as_secs_f64() * 1000.0;
    }

    fn add_reference_forward(&mut self, elapsed: Duration) {
        self.reference_forward_ms += elapsed.as_secs_f64() * 1000.0;
    }

    fn add_policy_forward(&mut self, elapsed: Duration) {
        self.policy_forward_ms += elapsed.as_secs_f64() * 1000.0;
    }

    fn add_backward(&mut self, elapsed: Duration) {
        self.backward_ms += elapsed.as_secs_f64() * 1000.0;
    }

    fn add_optimizer(&mut self, elapsed: Duration) {
        self.optimizer_ms += elapsed.as_secs_f64() * 1000.0;
    }

    fn to_receipt(&self) -> crate::train_receipt::TrainingPhaseTimingsReceipt {
        crate::train_receipt::TrainingPhaseTimingsReceipt {
            tokenize_ms: self.tokenize_ms,
            mask_build_ms: self.mask_build_ms,
            reference_forward_ms: self.reference_forward_ms,
            policy_forward_ms: self.policy_forward_ms,
            backward_ms: self.backward_ms,
            optimizer_ms: self.optimizer_ms,
            gpu_writer_wait_ms: self.gpu_writer_wait_ms,
            gpu_writer_held_ms: self.gpu_writer_held_ms,
            gpu_writer_acquisitions: self.gpu_writer_acquisitions,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GrpoBenchmarkReport {
    pub completions: usize,
    pub min_seq_len: usize,
    pub max_seq_len: usize,
    pub total_tokens: u64,
    pub action_tokens: u64,
    pub env_tokens: u64,
    pub context_tokens: u64,
    pub loss: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub policy_audit: Option<crate::train_receipt::GrpoPolicyAuditReceipt>,
    pub timings: GrpoBenchmarkTimings,
    pub total_ms: f64,
    pub tokens_per_sec: f64,
}

#[derive(Debug, Clone)]
pub struct GrpoDryRunReport {
    pub adapter_dir: PathBuf,
    pub receipt_path: PathBuf,
    pub base_adapter_dir: Option<PathBuf>,
    pub alpha_over_rank: Option<f32>,
    pub data: crate::train_receipt::DataStatsReceipt,
    pub rewards: crate::train_receipt::RewardStatsReceipt,
    pub token_counts: crate::train_receipt::TokenCountReceipt,
    pub dynamic_groups_filtered: usize,
}

/// Build a progress bar for a training step/group loop.
///
/// Returns `None` when stderr is not a TTY so log files, server-mode tracing,
/// and CI runs stay clean. The structured `tracing::info!` lines and the
/// `progress_cb` HTTP-status callback remain the source of truth for
/// non-interactive runs; the bar is purely additive UX for interactive
/// `kiln train` invocations, where SFT and GRPO loops often run
/// hundreds–thousands of iterations with no other visual feedback between
/// every-10-step log lines.
///
/// `label` is the per-loop prefix shown before the bar (e.g. `"sft training"`
/// or `"grpo training"`).
fn make_step_progress(total_steps: usize, label: &str) -> Option<indicatif::ProgressBar> {
    if !console::Term::stderr().features().is_attended() {
        return None;
    }
    let pb = indicatif::ProgressBar::new(total_steps as u64);
    let template = format!(
        "  {label} {{bar:40.cyan/blue}} {{pos:>5}}/{{len:5}} step ({{elapsed}}) loss={{msg}}"
    );
    pb.set_style(
        indicatif::ProgressStyle::with_template(&template)
            .expect("static progress template is valid")
            .progress_chars("##-"),
    );
    Some(pb)
}

fn resolve_and_validate_base_adapter(
    base_adapter: Option<&str>,
    adapter_dir: &Path,
    model_config: &ModelConfig,
    lora_rank: usize,
    allow_adapter_shape_conversion: bool,
) -> Result<Option<PathBuf>> {
    let Some(base_name) = base_adapter else {
        return Ok(None);
    };
    let base_dir = crate::adapter_shape::resolve_base_adapter_dir(base_name, adapter_dir);
    let compatibility = crate::adapter_shape::validate_base_adapter_compatibility(
        &base_dir,
        model_config,
        lora_rank,
        allow_adapter_shape_conversion,
    )
    .with_context(|| {
        format!(
            "validate base adapter {} before optimizer setup",
            base_dir.display()
        )
    })?;
    tracing::info!(
        base = %base_dir.display(),
        rank = compatibility.rank,
        tensor_count = compatibility.tensor_count,
        "validated base adapter compatibility"
    );
    Ok(Some(base_dir))
}

pub(crate) fn resolve_base_adapter_dir_from_roots(
    base_name: &str,
    adapter_dir: &Path,
    output_adapter_dir: &Path,
    output_adapter_name: &str,
) -> PathBuf {
    if base_name == output_adapter_name {
        let starting_snapshot = output_adapter_dir.join(STARTING_ADAPTER_SNAPSHOT_DIR);
        if starting_snapshot.is_dir() {
            return starting_snapshot;
        }
    }
    let staged = output_adapter_dir.join(base_name);
    if base_name != output_adapter_name && staged.is_dir() {
        staged
    } else {
        crate::adapter_shape::resolve_base_adapter_dir(base_name, adapter_dir)
    }
}

pub(crate) fn resolve_and_validate_base_adapter_from_roots(
    base_adapter: Option<&str>,
    adapter_dir: &Path,
    output_adapter_dir: &Path,
    output_adapter_name: &str,
    model_config: &ModelConfig,
    lora_rank: usize,
    allow_adapter_shape_conversion: bool,
) -> Result<Option<PathBuf>> {
    let Some(base_name) = base_adapter else {
        return Ok(None);
    };
    let base_dir = resolve_base_adapter_dir_from_roots(
        base_name,
        adapter_dir,
        output_adapter_dir,
        output_adapter_name,
    );
    let compatibility = crate::adapter_shape::validate_base_adapter_compatibility(
        &base_dir,
        model_config,
        lora_rank,
        allow_adapter_shape_conversion,
    )
    .with_context(|| {
        format!(
            "validate base adapter {} before optimizer setup",
            base_dir.display()
        )
    })?;
    tracing::info!(
        base = %base_dir.display(),
        rank = compatibility.rank,
        tensor_count = compatibility.tensor_count,
        "validated base adapter compatibility"
    );
    Ok(Some(base_dir))
}

/// Deterministic per-epoch permutation of `0..n` (Fisher-Yates seeded by
/// `seed` + epoch). SFT previously replayed the dataset in identical order
/// every epoch at batch size 1, so late examples always saw the
/// freshest weights and inter-example gradient correlation repeated
/// epoch over epoch.
fn epoch_order(seed: u64, epoch: usize, n: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    // splitmix-style epoch mix so epoch streams are decorrelated even for
    // adjacent epoch numbers.
    let mixed = seed ^ (epoch as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut rng = StdRng::seed_from_u64(mixed);
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        order.swap(i, j);
    }
    order
}

const SFT_CHECKPOINT_LOOP_STATE_SCHEMA_VERSION: u32 = 1;
const SFT_CHECKPOINT_LOOP_STATE_TYPE: &str = "kiln.sft-loop-state.v1";
const SFT_CHECKPOINT_ADAPTER_FILE: &str = "adapter.safetensors";
const SFT_CHECKPOINT_OPTIMIZER_FILE: &str = "optimizer.safetensors";
const SFT_CHECKPOINT_LOOP_STATE_FILE: &str = "sft_loop_state.json";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct SftCheckpointLoopState {
    schema_version: u32,
    state_type: String,
    global_step: u64,
    epoch_index: u64,
    cursor_in_epoch: u64,
    loss_history: Vec<f64>,
    last_loss: f64,
    current_epoch_loss_sum: f64,
    current_epoch_items: u64,
    first_epoch_loss: Option<f64>,
    best_epoch_loss: Option<f64>,
    lora_grad_norms: crate::train_receipt::LoraGradNormAccumulator,
}

impl SftCheckpointLoopState {
    #[allow(clippy::too_many_arguments)]
    fn capture(
        global_step: usize,
        epoch_index: usize,
        cursor_in_epoch: usize,
        loss_history: &[f64],
        last_loss: f64,
        current_epoch_loss_sum: f64,
        current_epoch_items: usize,
        first_epoch_loss: Option<f64>,
        best_epoch_loss: f64,
        lora_grad_norms: &crate::train_receipt::LoraGradNormAccumulator,
    ) -> Self {
        Self {
            schema_version: SFT_CHECKPOINT_LOOP_STATE_SCHEMA_VERSION,
            state_type: SFT_CHECKPOINT_LOOP_STATE_TYPE.to_string(),
            global_step: global_step as u64,
            epoch_index: epoch_index as u64,
            cursor_in_epoch: cursor_in_epoch as u64,
            loss_history: loss_history.to_vec(),
            last_loss,
            current_epoch_loss_sum,
            current_epoch_items: current_epoch_items as u64,
            first_epoch_loss,
            best_epoch_loss: best_epoch_loss.is_finite().then_some(best_epoch_loss),
            lora_grad_norms: lora_grad_norms.clone(),
        }
    }

    fn validate(&self, progress: &crate::checkpoint::TrainingCheckpointProgress) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == SFT_CHECKPOINT_LOOP_STATE_SCHEMA_VERSION
                && self.state_type == SFT_CHECKPOINT_LOOP_STATE_TYPE,
            "unsupported SFT checkpoint loop-state contract"
        );
        anyhow::ensure!(
            self.global_step == progress.global_step
                && self.epoch_index == progress.epoch_index
                && self.cursor_in_epoch == progress.cursor_in_epoch,
            "SFT checkpoint loop state disagrees with manifest progress"
        );
        anyhow::ensure!(
            self.loss_history.len() as u64 == self.global_step,
            "SFT checkpoint loss-history length {} does not match global step {}",
            self.loss_history.len(),
            self.global_step
        );
        anyhow::ensure!(
            self.loss_history.iter().all(|loss| loss.is_finite()),
            "SFT checkpoint loss history contains a non-finite value"
        );
        anyhow::ensure!(
            self.last_loss.is_finite()
                && self.current_epoch_loss_sum.is_finite()
                && self.first_epoch_loss.is_none_or(f64::is_finite)
                && self.best_epoch_loss.is_none_or(f64::is_finite),
            "SFT checkpoint loop state contains a non-finite scalar"
        );
        anyhow::ensure!(
            self.loss_history.last().copied() == Some(self.last_loss),
            "SFT checkpoint last_loss does not match loss history"
        );
        anyhow::ensure!(
            self.current_epoch_items == self.cursor_in_epoch,
            "SFT checkpoint current-epoch item count does not match cursor"
        );
        anyhow::ensure!(
            self.first_epoch_loss.is_some() == (self.epoch_index > 0)
                && self.best_epoch_loss.is_some() == (self.epoch_index > 0),
            "SFT checkpoint completed-epoch loss state is inconsistent"
        );
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct SftCheckpointDescriptor {
    adapter_name: String,
    effective_config: serde_json::Value,
    precision_policy: crate::checkpoint::TrainingCheckpointPrecision,
    data: crate::checkpoint::TrainingCheckpointData,
    init_seed: u64,
    shuffle_seed: u64,
    optimizer: Optimizer,
    learning_rate: f64,
    total_steps: usize,
    base_model_weights_sha256: Option<String>,
    auxiliary_state: serde_json::Value,
}

#[derive(Debug)]
struct SftCheckpointSnapshot {
    target: PathBuf,
    manifest: crate::checkpoint::TrainingCheckpointManifest,
    artifacts: Vec<crate::checkpoint::CheckpointArtifact>,
    adapter_parameters: CheckpointTensorSnapshot,
    optimizer_state: Option<CheckpointTensorSnapshot>,
    loop_state_bytes: Vec<u8>,
}

impl SftCheckpointSnapshot {
    fn publish(self) -> Result<PathBuf> {
        let Self {
            target,
            manifest,
            artifacts,
            adapter_parameters,
            optimizer_state,
            loop_state_bytes,
        } = self;
        crate::checkpoint::write_training_checkpoint_atomic(
            &target,
            manifest,
            &artifacts,
            move |staging| {
                adapter_parameters.save(&staging.join(SFT_CHECKPOINT_ADAPTER_FILE))?;
                if let Some(state) = optimizer_state.as_ref() {
                    state.save(&staging.join(SFT_CHECKPOINT_OPTIMIZER_FILE))?;
                }
                std::fs::write(
                    staging.join(SFT_CHECKPOINT_LOOP_STATE_FILE),
                    &loop_state_bytes,
                )
                .context("write SFT checkpoint loop state")?;
                Ok(())
            },
        )
    }
}

impl SftCheckpointDescriptor {
    fn optimizer_state_file(&self) -> Option<String> {
        (!matches!(self.optimizer, Optimizer::Sgd))
            .then(|| SFT_CHECKPOINT_OPTIMIZER_FILE.to_string())
    }

    fn optimizer_manifest(
        &self,
        step: u64,
    ) -> Result<crate::checkpoint::TrainingCheckpointOptimizer> {
        let kind = match self.optimizer {
            Optimizer::Sgd => "sgd",
            Optimizer::AdamW { .. } => "adam_w",
            Optimizer::Muon { .. } => "muon",
        };
        let hyperparameters = canonical_checkpoint_json_value(serde_json::json!({
            "learning_rate": self.learning_rate,
            "optimizer": serde_json::to_value(self.optimizer)
                .context("serialize SFT checkpoint optimizer")?,
        }))?;
        Ok(crate::checkpoint::TrainingCheckpointOptimizer {
            kind: kind.to_string(),
            step,
            hyperparameters,
            state_file: self.optimizer_state_file(),
        })
    }

    fn scheduler_manifest(&self, step: u64) -> crate::checkpoint::TrainingCheckpointScheduler {
        crate::checkpoint::TrainingCheckpointScheduler {
            kind: "constant".to_string(),
            step,
            state: serde_json::json!({
                "training_profile": crate::NATIVE_SFT_PROFILE_V1,
                "learning_rate": self.learning_rate,
                "microbatch_conversations": 1,
                "gradient_accumulation_steps": 1,
                "warmup_steps": 0,
                "gradient_clipping": "none",
            }),
        }
    }

    fn rng_states(
        &self,
        epoch_index: u64,
    ) -> BTreeMap<String, crate::checkpoint::TrainingCheckpointRngState> {
        BTreeMap::from([
            (
                "epoch-order".to_string(),
                crate::checkpoint::TrainingCheckpointRngState {
                    algorithm: "kiln.epoch-order.v1".to_string(),
                    seed: self.shuffle_seed,
                    position: epoch_index,
                    state_file: None,
                },
            ),
            (
                "lora-init".to_string(),
                crate::checkpoint::TrainingCheckpointRngState {
                    algorithm: "kiln.seeded-lora-init.v1".to_string(),
                    seed: self.init_seed,
                    position: 0,
                    state_file: None,
                },
            ),
        ])
    }

    fn manifest(
        &self,
        progress: crate::checkpoint::TrainingCheckpointProgress,
    ) -> Result<crate::checkpoint::TrainingCheckpointManifest> {
        let step = progress.global_step;
        let optimizer_state = self.optimizer_state_file();
        Ok(crate::checkpoint::TrainingCheckpointManifest::new(
            format!("sft-step-{step:08}"),
            crate::checkpoint::TrainingKind::Sft,
            &self.adapter_name,
            self.effective_config.clone(),
            self.precision_policy.clone(),
            progress.clone(),
            self.data.clone(),
            self.rng_states(progress.epoch_index),
            self.optimizer_manifest(step)?,
            self.scheduler_manifest(step),
            crate::checkpoint::TrainingCheckpointStateFiles {
                adapter_parameters: SFT_CHECKPOINT_ADAPTER_FILE.to_string(),
                optimizer_state,
                reference_state: None,
                ema_state: None,
                reward_normalization_state: None,
                loss_history: Some(SFT_CHECKPOINT_LOOP_STATE_FILE.to_string()),
            },
            self.auxiliary_state.clone(),
        ))
    }

    fn validate_resume(
        &self,
        checkpoint: &crate::checkpoint::ValidatedTrainingCheckpoint,
        loop_state: &SftCheckpointLoopState,
    ) -> Result<()> {
        let manifest = &checkpoint.manifest;
        anyhow::ensure!(
            manifest.training_kind == crate::checkpoint::TrainingKind::Sft,
            "resume checkpoint is {:?}, not SFT",
            manifest.training_kind
        );
        anyhow::ensure!(
            manifest.adapter_name == self.adapter_name,
            "resume checkpoint adapter {:?} does not match output adapter {:?}",
            manifest.adapter_name,
            self.adapter_name
        );
        anyhow::ensure!(
            manifest.effective_config == self.effective_config,
            "resume checkpoint effective SFT configuration differs from this request: checkpoint={}, request={}",
            manifest.effective_config,
            self.effective_config
        );
        anyhow::ensure!(
            manifest.precision_policy == self.precision_policy,
            "resume checkpoint precision policy differs from this runtime"
        );
        anyhow::ensure!(
            manifest.data == self.data,
            "resume checkpoint training data identity differs from this request"
        );
        anyhow::ensure!(
            manifest.progress.total_steps == self.total_steps as u64,
            "resume checkpoint total step count {} differs from this run {}",
            manifest.progress.total_steps,
            self.total_steps
        );
        anyhow::ensure!(
            manifest.optimizer == self.optimizer_manifest(manifest.progress.global_step)?,
            "resume checkpoint optimizer contract differs from this request"
        );
        anyhow::ensure!(
            manifest.scheduler == self.scheduler_manifest(manifest.progress.global_step),
            "resume checkpoint scheduler contract differs from this request"
        );
        anyhow::ensure!(
            manifest.rng_states == self.rng_states(manifest.progress.epoch_index),
            "resume checkpoint RNG streams differ from this request"
        );
        crate::checkpoint::validate_checkpoint_base_weight_resume_binding(
            &manifest.auxiliary_state,
            &self.auxiliary_state,
        )?;
        crate::checkpoint::validate_checkpoint_execution_resume_binding(
            &manifest.auxiliary_state,
            &self.auxiliary_state,
        )?;
        anyhow::ensure!(
            manifest.auxiliary_state == self.auxiliary_state,
            "resume checkpoint model/tokenizer/runtime identity differs from this run"
        );
        let epochs = self.total_steps as u64 / self.data.item_count;
        anyhow::ensure!(
            manifest.progress.epoch_index < epochs,
            "resume checkpoint epoch index {} is outside {epochs} configured epochs",
            manifest.progress.epoch_index
        );
        let expected_step = manifest
            .progress
            .epoch_index
            .checked_mul(self.data.item_count)
            .and_then(|base| base.checked_add(manifest.progress.cursor_in_epoch))
            .context("resume checkpoint progress overflow")?;
        anyhow::ensure!(
            expected_step == manifest.progress.global_step,
            "resume checkpoint cursor implies step {expected_step}, not {}",
            manifest.progress.global_step
        );
        let expected_order: Vec<u64> = epoch_order(
            self.shuffle_seed,
            manifest.progress.epoch_index as usize,
            self.data.item_count as usize,
        )
        .into_iter()
        .map(|index| index as u64)
        .collect();
        anyhow::ensure!(
            manifest.progress.data_order == expected_order,
            "resume checkpoint data order does not match its seeded epoch order"
        );
        loop_state.validate(&manifest.progress)
    }

    #[allow(clippy::too_many_arguments)]
    fn capture(
        &self,
        output_root: &Path,
        backend: &dyn BackendRuntime,
        params: &mut TrainableLoraParams,
        opt_state: &mut Option<OptimizerState>,
        epoch_index: usize,
        cursor_in_epoch: usize,
        data_order: &[usize],
        loop_state: &SftCheckpointLoopState,
    ) -> Result<SftCheckpointSnapshot> {
        anyhow::ensure!(
            self.base_model_weights_sha256.is_some(),
            "exact SFT checkpointing requires base-model weights loaded with a content identity"
        );
        crate::checkpoint::validated_checkpoint_base_weight_manifest(&self.auxiliary_state)?;
        crate::checkpoint::validated_checkpoint_execution_provenance(&self.auxiliary_state)?;
        let progress = crate::checkpoint::TrainingCheckpointProgress {
            global_step: loop_state.global_step,
            total_steps: self.total_steps as u64,
            epoch_index: epoch_index as u64,
            cursor_in_epoch: cursor_in_epoch as u64,
            data_order: data_order.iter().map(|&index| index as u64).collect(),
        };
        loop_state.validate(&progress)?;
        let manifest = self.manifest(progress)?;
        let target = output_root.join(format!(
            "{}-checkpoint-step-{:08}.kiln-checkpoint",
            self.adapter_name, loop_state.global_step
        ));
        params.sync_to_master(backend)?;
        let adapter_parameters = params.capture_checkpoint_parameters()?;
        let optimizer_state = opt_state
            .as_mut()
            .map(|state| state.capture_checkpoint_state(params, backend))
            .transpose()?;

        let mut artifacts = vec![
            crate::checkpoint::CheckpointArtifact {
                relative_path: SFT_CHECKPOINT_ADAPTER_FILE.to_string(),
                role: crate::checkpoint::CheckpointFileRole::AdapterParameters,
            },
            crate::checkpoint::CheckpointArtifact {
                relative_path: SFT_CHECKPOINT_LOOP_STATE_FILE.to_string(),
                role: crate::checkpoint::CheckpointFileRole::LossHistory,
            },
        ];
        if opt_state.is_some() {
            artifacts.push(crate::checkpoint::CheckpointArtifact {
                relative_path: SFT_CHECKPOINT_OPTIMIZER_FILE.to_string(),
                role: crate::checkpoint::CheckpointFileRole::OptimizerState,
            });
        }
        let loop_state_bytes =
            serde_json::to_vec_pretty(loop_state).context("serialize SFT checkpoint loop state")?;
        Ok(SftCheckpointSnapshot {
            target,
            manifest,
            artifacts,
            adapter_parameters,
            optimizer_state,
            loop_state_bytes,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn save(
        &self,
        output_root: &Path,
        backend: &dyn BackendRuntime,
        params: &mut TrainableLoraParams,
        opt_state: &mut Option<OptimizerState>,
        epoch_index: usize,
        cursor_in_epoch: usize,
        data_order: &[usize],
        loop_state: &SftCheckpointLoopState,
        gpu_step_coordination: Option<&GpuStepCoordination>,
    ) -> Result<PathBuf> {
        let wait_started = Instant::now();
        let checkpoint_gpu = gpu_step_coordination
            .map(GpuStepCoordination::blocking_write)
            .transpose()
            .context("acquire healthy backend for SFT checkpoint snapshot")?;
        let gpu_wait_ms = wait_started.elapsed().as_millis() as u64;
        let snapshot_started = Instant::now();
        let snapshot = self.capture(
            output_root,
            backend,
            params,
            opt_state,
            epoch_index,
            cursor_in_epoch,
            data_order,
            loop_state,
        )?;
        let device_snapshot_ms = snapshot_started.elapsed().as_millis() as u64;
        drop(checkpoint_gpu);

        let publish_started = Instant::now();
        let path = snapshot.publish()?;
        let publish_ms = publish_started.elapsed().as_millis() as u64;
        tracing::info!(
            checkpoint = %path.display(),
            gpu_wait_ms,
            device_snapshot_ms,
            publish_ms,
            "published coordinated SFT checkpoint"
        );
        Ok(path)
    }
}

fn checkpoint_dtype_name(dtype: KtDType) -> String {
    dtype.to_string().to_ascii_lowercase()
}

pub(crate) fn training_checkpoint_precision(
    params: &TrainableLoraParams,
    opt_state: Option<&OptimizerState>,
) -> Result<crate::checkpoint::TrainingCheckpointPrecision> {
    let parameter = params
        .all_params()
        .into_iter()
        .next()
        .context("SFT checkpoint has no trainable parameters")?;
    let amp = parameter.amp_policy();
    let (optimizer_state_dtype, rounding) = match opt_state {
        Some(state) => {
            let policy = state.checkpoint_rounding_policy();
            let rounding = match policy {
                StochasticRoundingPolicy::RoundToNearest => {
                    serde_json::json!({"mode": "round_to_nearest"})
                }
                StochasticRoundingPolicy::Stochastic { seed } => {
                    serde_json::json!({"mode": "stochastic", "seed": seed})
                }
                _ => serde_json::json!({"mode": policy.name()}),
            };
            (
                checkpoint_dtype_name(state.checkpoint_state_dtype()?),
                rounding,
            )
        }
        None => (
            "none".to_string(),
            serde_json::json!({"mode": "round_to_nearest"}),
        ),
    };
    Ok(crate::checkpoint::TrainingCheckpointPrecision {
        parameter_dtype: checkpoint_dtype_name(amp.master_dtype),
        optimizer_state_dtype,
        activation_dtype: checkpoint_dtype_name(amp.forward_compute_dtype),
        gradient_dtype: checkpoint_dtype_name(amp.backward_compute_dtype),
        stochastic_rounding: rounding,
    })
}

pub(crate) fn training_precision_for_receipt_best_effort(
    params: &TrainableLoraParams,
    opt_state: Option<&OptimizerState>,
) -> Option<crate::checkpoint::TrainingCheckpointPrecision> {
    match training_checkpoint_precision(params, opt_state) {
        Ok(precision) => Some(precision),
        Err(error) => {
            tracing::warn!(error = %format!("{error:#}"), "could not record concrete training precision in receipt");
            None
        }
    }
}

fn sft_checkpoint_effective_config(
    config: &SftConfig,
    learning_rate: f64,
    effective_seed: u64,
) -> Result<serde_json::Value> {
    let mut value = serde_json::to_value(config).context("serialize effective SFT config")?;
    let object = value
        .as_object_mut()
        .context("serialized SFT config is not an object")?;
    object.remove("resume_checkpoint");
    object.insert(
        "learning_rate".to_string(),
        serde_json::json!(learning_rate),
    );
    object.insert("seed".to_string(), serde_json::json!(effective_seed));
    canonical_checkpoint_json_value(value)
}

pub(crate) fn canonical_checkpoint_json_value(
    value: serde_json::Value,
) -> Result<serde_json::Value> {
    let encoded = serde_json::to_vec(&value).context("encode canonical checkpoint JSON")?;
    serde_json::from_slice(&encoded).context("decode canonical checkpoint JSON")
}

fn sft_checkpoint_auxiliary_state(
    model_config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    precision_policy: TrainingPrecisionPolicy,
    valid_indices: &[usize],
    base_model_weights_sha256: Option<&str>,
    base_weight_shard_manifest: Option<&kiln_core::model_provenance::BaseWeightShardManifest>,
    execution_provenance: Option<&kiln_core::execution_provenance::ExecutionProvenanceV1>,
    backend_runtime: &str,
    gradient_checkpoint_plan_sha256: &str,
    ingestion_receipt_sha256: &str,
    training_runtime_planning_identity: &serde_json::Value,
) -> Result<serde_json::Value> {
    let hashes =
        kiln_core::config_hashes::ConfigHashes::from_model_tokenizer(model_config, tokenizer, None);
    let valid_indices_sha256 = crate::train_receipt::sha256_json_serializable(&valid_indices)
        .context("hash SFT valid-example index set")?;
    Ok(serde_json::json!({
        "loop_state_type": SFT_CHECKPOINT_LOOP_STATE_TYPE,
        "model_config_sha256": hashes.model_config_hash,
        "tokenizer_config_sha256": hashes.tokenizer_config_hash,
        "chat_template_sha256": hashes.chat_template_hash,
        "training_chat_template_sha256": hashes.training_chat_template_hash,
        "base_model_weights_sha256": base_model_weights_sha256,
        "base_weight_shard_manifest": base_weight_shard_manifest,
        "execution_provenance": execution_provenance,
        "backend_runtime": backend_runtime,
        "kiln_train_version": env!("CARGO_PKG_VERSION"),
        "gradient_checkpoint_plan_sha256": gradient_checkpoint_plan_sha256,
        "ingestion_receipt_sha256": ingestion_receipt_sha256,
        "training_precision_policy": precision_policy.name,
        "training_runtime_planning_identity": training_runtime_planning_identity,
        "valid_indices_sha256": valid_indices_sha256,
    }))
}

pub(crate) fn checkpoint_sha256_hex(prefixed: Option<&str>, label: &str) -> Result<String> {
    let value = prefixed.with_context(|| format!("compute {label} SHA-256"))?;
    value
        .strip_prefix("sha256:")
        .map(ToOwned::to_owned)
        .with_context(|| format!("{label} SHA-256 lacks sha256: prefix"))
}

pub(crate) fn validate_exact_training_provenance(weights: &GpuWeights) -> Result<()> {
    let aggregate = weights
        .source_content_sha256
        .as_deref()
        .context("exact checkpointing requires a loader-owned base-model content identity")?;
    let manifest = weights
        .base_weight_shard_manifest
        .as_ref()
        .context("exact checkpointing requires a loader-owned base-weight shard manifest")?;
    manifest
        .validate()
        .context("validate resident base-weight shard manifest")?;
    anyhow::ensure!(
        aggregate == manifest.aggregate_sha256,
        "resident base-model aggregate {aggregate} differs from its shard manifest {}",
        manifest.aggregate_sha256
    );
    let execution_provenance = weights
        .execution_provenance
        .as_ref()
        .context("exact checkpointing requires a startup-owned execution provenance record")?;
    execution_provenance
        .validate()
        .context("validate resident execution provenance")?;
    Ok(())
}

fn load_sft_checkpoint_loop_state(
    checkpoint: &crate::checkpoint::ValidatedTrainingCheckpoint,
) -> Result<SftCheckpointLoopState> {
    let relative = checkpoint
        .manifest
        .state_files
        .loss_history
        .as_deref()
        .context("SFT resume checkpoint has no loop-state file")?;
    anyhow::ensure!(
        relative == SFT_CHECKPOINT_LOOP_STATE_FILE,
        "unsupported SFT loop-state artifact {relative:?}"
    );
    let path = checkpoint.artifact_path(relative)?;
    let bytes = std::fs::read(&path)
        .with_context(|| format!("read SFT checkpoint loop state {}", path.display()))?;
    serde_json::from_slice(&bytes).context("parse strict SFT checkpoint loop state")
}

const GRPO_CHECKPOINT_LOOP_STATE_SCHEMA_VERSION: u32 = 1;
const GRPO_CHECKPOINT_LOOP_STATE_TYPE: &str = "kiln.grpo-loop-state.v1";
const GRPO_CHECKPOINT_ADAPTER_FILE: &str = "adapter.safetensors";
const GRPO_CHECKPOINT_OPTIMIZER_FILE: &str = "optimizer.safetensors";
const GRPO_CHECKPOINT_REFERENCE_FILE: &str = "reference.safetensors";
const GRPO_CHECKPOINT_LOOP_STATE_FILE: &str = "grpo_loop_state.json";

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum GrpoCheckpointRoute {
    Inline,
    Jsonl,
}

impl GrpoCheckpointRoute {
    fn source_kind(self) -> &'static str {
        match self {
            Self::Inline => "inline-grpo-trainable-order-v1",
            Self::Jsonl => "jsonl-grpo-trainable-order-v1",
        }
    }
}

/// CPU-owned state required to continue GRPO at the next optimizer-group
/// boundary. Tensor state lives in the adjacent safetensors artifacts; this
/// strict JSON owns cursors and receipt accumulators that would otherwise be
/// silently reset after a restart.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct GrpoCheckpointLoopState {
    schema_version: u32,
    state_type: String,
    route: GrpoCheckpointRoute,
    global_step: u64,
    /// Exact byte offset of the next unread JSONL line. Inline runs have no
    /// source-file cursor and therefore store `None`.
    source_byte_offset: Option<u64>,
    /// Number of physical JSONL lines already consumed. This restores line
    /// attribution after seeking, including blank lines.
    source_lines_consumed: Option<u64>,
    processed_completions: u64,
    loss_history: Vec<f64>,
    last_loss: Option<f64>,
    data_stats: crate::train_receipt::DataStatsReceipt,
    token_counts: crate::train_receipt::TokenCountReceipt,
    dynamic_groups_filtered: u64,
    echo_metrics: crate::train_receipt::EchoActivityMetrics,
    lora_grad_norms: crate::train_receipt::LoraGradNormAccumulator,
    policy_audit: crate::train_receipt::GrpoPolicyAuditAccumulator,
    phase_timings: GrpoBenchmarkTimings,
    gpu_writer_timings: GrpoGpuWriterTimings,
    /// Present exactly when the KL reference is an EMA snapshot.
    ema_groups_since_refresh: Option<u64>,
}

impl GrpoCheckpointLoopState {
    #[allow(clippy::too_many_arguments)]
    fn capture(
        route: GrpoCheckpointRoute,
        global_step: usize,
        source_byte_offset: Option<u64>,
        source_lines_consumed: Option<u64>,
        processed_completions: usize,
        loss_history: &[f64],
        data_stats: &crate::train_receipt::DataStatsReceipt,
        token_counts: &crate::train_receipt::TokenCountReceipt,
        dynamic_groups_filtered: usize,
        echo_metrics: &crate::train_receipt::EchoActivityMetrics,
        lora_grad_norms: &crate::train_receipt::LoraGradNormAccumulator,
        policy_audit: &crate::train_receipt::GrpoPolicyAuditAccumulator,
        phase_timings: &GrpoBenchmarkTimings,
        gpu_writer_timings: &GrpoGpuWriterTimings,
        ema_ref_state: Option<&EmaReferenceState>,
    ) -> Self {
        Self {
            schema_version: GRPO_CHECKPOINT_LOOP_STATE_SCHEMA_VERSION,
            state_type: GRPO_CHECKPOINT_LOOP_STATE_TYPE.to_string(),
            route,
            global_step: global_step as u64,
            source_byte_offset,
            source_lines_consumed,
            processed_completions: processed_completions as u64,
            loss_history: loss_history.to_vec(),
            last_loss: loss_history.last().copied(),
            data_stats: data_stats.clone(),
            token_counts: token_counts.clone(),
            dynamic_groups_filtered: dynamic_groups_filtered as u64,
            echo_metrics: echo_metrics.clone(),
            lora_grad_norms: lora_grad_norms.clone(),
            policy_audit: policy_audit.clone(),
            phase_timings: phase_timings.clone(),
            gpu_writer_timings: gpu_writer_timings.clone(),
            ema_groups_since_refresh: ema_ref_state.map(|state| state.groups_since_refresh as u64),
        }
    }

    fn validate(&self, progress: &crate::checkpoint::TrainingCheckpointProgress) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == GRPO_CHECKPOINT_LOOP_STATE_SCHEMA_VERSION
                && self.state_type == GRPO_CHECKPOINT_LOOP_STATE_TYPE,
            "unsupported GRPO checkpoint loop-state contract"
        );
        anyhow::ensure!(
            progress.epoch_index == 0
                && self.global_step == progress.global_step
                && self.global_step == progress.cursor_in_epoch,
            "GRPO checkpoint loop state disagrees with manifest progress"
        );
        anyhow::ensure!(
            self.loss_history.len() as u64 == self.global_step,
            "GRPO checkpoint loss-history length {} does not match global step {}",
            self.loss_history.len(),
            self.global_step
        );
        anyhow::ensure!(
            self.loss_history.iter().all(|loss| loss.is_finite()),
            "GRPO checkpoint loss history contains a non-finite value"
        );
        match (self.loss_history.last().copied(), self.last_loss) {
            (None, None) => {}
            (Some(expected), Some(actual)) if expected == actual && actual.is_finite() => {}
            _ => anyhow::bail!("GRPO checkpoint last_loss does not match loss history"),
        }
        anyhow::ensure!(
            self.data_stats.groups_trained as u64 == self.global_step,
            "GRPO checkpoint trained-group count does not match global step"
        );
        anyhow::ensure!(
            self.data_stats.completions_trained as u64 == self.processed_completions,
            "GRPO checkpoint trained-completion count does not match loop state"
        );
        anyhow::ensure!(
            self.dynamic_groups_filtered as usize <= self.data_stats.groups_filtered,
            "GRPO checkpoint dynamic-filter count exceeds all filtered groups"
        );
        match self.route {
            GrpoCheckpointRoute::Inline => anyhow::ensure!(
                self.source_byte_offset.is_none() && self.source_lines_consumed.is_none(),
                "inline GRPO checkpoint unexpectedly contains a JSONL cursor"
            ),
            GrpoCheckpointRoute::Jsonl => anyhow::ensure!(
                self.source_byte_offset.is_some() && self.source_lines_consumed.is_some(),
                "JSONL GRPO checkpoint is missing its exact source cursor"
            ),
        }
        let timing_values = [
            self.phase_timings.tokenize_ms,
            self.phase_timings.mask_build_ms,
            self.phase_timings.reference_forward_ms,
            self.phase_timings.policy_forward_ms,
            self.phase_timings.backward_ms,
            self.phase_timings.optimizer_ms,
            self.phase_timings.gpu_writer_wait_ms,
            self.phase_timings.gpu_writer_held_ms,
            self.gpu_writer_timings.wait_ms,
            self.gpu_writer_timings.held_ms,
        ];
        anyhow::ensure!(
            timing_values
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0),
            "GRPO checkpoint contains an invalid phase timing"
        );
        anyhow::ensure!(
            self.echo_metrics.initial_env_ce.is_none_or(f64::is_finite)
                && self.echo_metrics.final_env_ce.is_none_or(f64::is_finite),
            "GRPO checkpoint contains a non-finite ECHO measurement"
        );
        anyhow::ensure!(
            (self.echo_metrics.measurements == 0)
                == (self.echo_metrics.initial_env_ce.is_none()
                    && self.echo_metrics.final_env_ce.is_none()),
            "GRPO checkpoint ECHO accumulator is inconsistent"
        );
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct GrpoCheckpointDescriptor {
    route: GrpoCheckpointRoute,
    adapter_name: String,
    effective_config: serde_json::Value,
    precision_policy: crate::checkpoint::TrainingCheckpointPrecision,
    data: crate::checkpoint::TrainingCheckpointData,
    init_seed: u64,
    optimizer: Optimizer,
    learning_rate: f64,
    total_steps: usize,
    base_model_weights_sha256: Option<String>,
    auxiliary_state: serde_json::Value,
    ema_refresh_every: Option<usize>,
}

#[derive(Debug)]
struct GrpoCheckpointSnapshot {
    target: PathBuf,
    manifest: crate::checkpoint::TrainingCheckpointManifest,
    artifacts: Vec<crate::checkpoint::CheckpointArtifact>,
    adapter_parameters: CheckpointTensorSnapshot,
    optimizer_state: Option<CheckpointTensorSnapshot>,
    reference_state: Option<CheckpointTensorSnapshot>,
    loop_state_bytes: Vec<u8>,
}

impl GrpoCheckpointSnapshot {
    fn replace_loop_state(&mut self, loop_state: &GrpoCheckpointLoopState) -> Result<()> {
        self.loop_state_bytes = serde_json::to_vec_pretty(loop_state)
            .context("serialize GRPO checkpoint loop state")?;
        Ok(())
    }

    fn publish(self) -> Result<PathBuf> {
        let Self {
            target,
            manifest,
            artifacts,
            adapter_parameters,
            optimizer_state,
            reference_state,
            loop_state_bytes,
        } = self;
        crate::checkpoint::write_training_checkpoint_atomic(
            &target,
            manifest,
            &artifacts,
            move |staging| {
                adapter_parameters.save(&staging.join(GRPO_CHECKPOINT_ADAPTER_FILE))?;
                if let Some(state) = optimizer_state.as_ref() {
                    state.save(&staging.join(GRPO_CHECKPOINT_OPTIMIZER_FILE))?;
                }
                if let Some(state) = reference_state.as_ref() {
                    state.save(&staging.join(GRPO_CHECKPOINT_REFERENCE_FILE))?;
                }
                std::fs::write(
                    staging.join(GRPO_CHECKPOINT_LOOP_STATE_FILE),
                    &loop_state_bytes,
                )
                .context("write GRPO checkpoint loop state")?;
                Ok(())
            },
        )
    }
}

impl GrpoCheckpointDescriptor {
    fn optimizer_state_file(&self) -> Option<String> {
        (!matches!(self.optimizer, Optimizer::Sgd))
            .then(|| GRPO_CHECKPOINT_OPTIMIZER_FILE.to_string())
    }

    fn reference_state_file(&self) -> Option<String> {
        self.ema_refresh_every
            .map(|_| GRPO_CHECKPOINT_REFERENCE_FILE.to_string())
    }

    fn optimizer_manifest(
        &self,
        step: u64,
    ) -> Result<crate::checkpoint::TrainingCheckpointOptimizer> {
        let kind = match self.optimizer {
            Optimizer::Sgd => "sgd",
            Optimizer::AdamW { .. } => "adam_w",
            Optimizer::Muon { .. } => "muon",
        };
        let hyperparameters = canonical_checkpoint_json_value(serde_json::json!({
            "learning_rate": self.learning_rate,
            "optimizer": serde_json::to_value(self.optimizer)
                .context("serialize GRPO checkpoint optimizer")?,
        }))?;
        Ok(crate::checkpoint::TrainingCheckpointOptimizer {
            kind: kind.to_string(),
            step,
            hyperparameters,
            state_file: self.optimizer_state_file(),
        })
    }

    fn scheduler_manifest(&self, step: u64) -> crate::checkpoint::TrainingCheckpointScheduler {
        crate::checkpoint::TrainingCheckpointScheduler {
            kind: "constant".to_string(),
            step,
            state: serde_json::json!({"learning_rate": self.learning_rate}),
        }
    }

    fn rng_states(
        &self,
        step: u64,
    ) -> BTreeMap<String, crate::checkpoint::TrainingCheckpointRngState> {
        let mut states = BTreeMap::from([(
            "lora-init".to_string(),
            crate::checkpoint::TrainingCheckpointRngState {
                algorithm: "kiln.seeded-lora-init.v1".to_string(),
                seed: self.init_seed,
                position: 0,
                state_file: None,
            },
        )]);
        let rounding = &self.precision_policy.stochastic_rounding;
        if rounding.get("mode").and_then(serde_json::Value::as_str) == Some("stochastic") {
            if let Some(seed) = rounding.get("seed").and_then(serde_json::Value::as_u64) {
                states.insert(
                    "optimizer-rounding".to_string(),
                    crate::checkpoint::TrainingCheckpointRngState {
                        algorithm: "kiln.optimizer-stochastic-rounding.v1".to_string(),
                        seed,
                        position: step,
                        state_file: None,
                    },
                );
            }
        }
        states
    }

    fn data_order(&self) -> Vec<u64> {
        (0..self.total_steps as u64).collect()
    }

    fn state_files(&self) -> crate::checkpoint::TrainingCheckpointStateFiles {
        crate::checkpoint::TrainingCheckpointStateFiles {
            adapter_parameters: GRPO_CHECKPOINT_ADAPTER_FILE.to_string(),
            optimizer_state: self.optimizer_state_file(),
            reference_state: self.reference_state_file(),
            ema_state: None,
            reward_normalization_state: None,
            loss_history: Some(GRPO_CHECKPOINT_LOOP_STATE_FILE.to_string()),
        }
    }

    fn progress(
        &self,
        loop_state: &GrpoCheckpointLoopState,
    ) -> crate::checkpoint::TrainingCheckpointProgress {
        crate::checkpoint::TrainingCheckpointProgress {
            global_step: loop_state.global_step,
            total_steps: self.total_steps as u64,
            epoch_index: 0,
            cursor_in_epoch: loop_state.global_step,
            data_order: self.data_order(),
        }
    }

    fn manifest(
        &self,
        progress: crate::checkpoint::TrainingCheckpointProgress,
    ) -> Result<crate::checkpoint::TrainingCheckpointManifest> {
        let step = progress.global_step;
        Ok(crate::checkpoint::TrainingCheckpointManifest::new(
            format!("grpo-step-{step:08}"),
            crate::checkpoint::TrainingKind::Grpo,
            &self.adapter_name,
            self.effective_config.clone(),
            self.precision_policy.clone(),
            progress,
            self.data.clone(),
            self.rng_states(step),
            self.optimizer_manifest(step)?,
            self.scheduler_manifest(step),
            self.state_files(),
            self.auxiliary_state.clone(),
        ))
    }

    fn validate_resume(
        &self,
        checkpoint: &crate::checkpoint::ValidatedTrainingCheckpoint,
        loop_state: &GrpoCheckpointLoopState,
    ) -> Result<()> {
        let manifest = &checkpoint.manifest;
        anyhow::ensure!(
            manifest.training_kind == crate::checkpoint::TrainingKind::Grpo,
            "resume checkpoint is {:?}, not GRPO",
            manifest.training_kind
        );
        anyhow::ensure!(
            manifest.adapter_name == self.adapter_name,
            "resume checkpoint adapter {:?} does not match output adapter {:?}",
            manifest.adapter_name,
            self.adapter_name
        );
        anyhow::ensure!(
            manifest.effective_config == self.effective_config,
            "resume checkpoint effective GRPO configuration differs from this request: checkpoint={}, request={}",
            manifest.effective_config,
            self.effective_config
        );
        anyhow::ensure!(
            manifest.precision_policy == self.precision_policy,
            "resume checkpoint precision policy differs from this runtime"
        );
        anyhow::ensure!(
            manifest.data == self.data,
            "resume checkpoint GRPO data identity differs from this request"
        );
        anyhow::ensure!(
            manifest.progress.total_steps == self.total_steps as u64
                && manifest.progress.data_order == self.data_order(),
            "resume checkpoint GRPO trainable order differs from this run"
        );
        anyhow::ensure!(
            manifest.optimizer == self.optimizer_manifest(manifest.progress.global_step)?,
            "resume checkpoint optimizer contract differs from this request"
        );
        anyhow::ensure!(
            manifest.scheduler == self.scheduler_manifest(manifest.progress.global_step),
            "resume checkpoint scheduler contract differs from this request"
        );
        anyhow::ensure!(
            manifest.rng_states == self.rng_states(manifest.progress.global_step),
            "resume checkpoint RNG streams differ from this request"
        );
        anyhow::ensure!(
            manifest.state_files == self.state_files(),
            "resume checkpoint GRPO artifact contract differs from this runtime"
        );
        crate::checkpoint::validate_checkpoint_base_weight_resume_binding(
            &manifest.auxiliary_state,
            &self.auxiliary_state,
        )?;
        crate::checkpoint::validate_checkpoint_execution_resume_binding(
            &manifest.auxiliary_state,
            &self.auxiliary_state,
        )?;
        anyhow::ensure!(
            manifest.auxiliary_state == self.auxiliary_state,
            "resume checkpoint model/tokenizer/runtime identity differs from this run"
        );
        anyhow::ensure!(
            loop_state.route == self.route,
            "resume checkpoint GRPO route differs from this request"
        );
        match (self.ema_refresh_every, loop_state.ema_groups_since_refresh) {
            (None, None) => {}
            (Some(refresh_every), Some(position)) => anyhow::ensure!(
                position < refresh_every as u64,
                "resume checkpoint EMA refresh cursor {position} exceeds cadence {refresh_every}"
            ),
            _ => anyhow::bail!("resume checkpoint EMA metadata differs from this request"),
        }
        loop_state.validate(&manifest.progress)
    }

    fn capture(
        &self,
        output_root: &Path,
        backend: &dyn BackendRuntime,
        params: &mut TrainableLoraParams,
        opt_state: &mut Option<OptimizerState>,
        ema_ref_state: Option<&EmaReferenceState>,
        loop_state: &GrpoCheckpointLoopState,
    ) -> Result<GrpoCheckpointSnapshot> {
        anyhow::ensure!(
            self.base_model_weights_sha256.is_some(),
            "exact GRPO checkpointing requires base-model weights loaded with a content identity"
        );
        crate::checkpoint::validated_checkpoint_base_weight_manifest(&self.auxiliary_state)?;
        crate::checkpoint::validated_checkpoint_execution_provenance(&self.auxiliary_state)?;
        anyhow::ensure!(
            self.ema_refresh_every.is_some() == ema_ref_state.is_some(),
            "GRPO checkpoint EMA tensor state differs from its manifest contract"
        );
        match (&self.optimizer, opt_state.as_ref()) {
            (Optimizer::Sgd, None) => {}
            (Optimizer::Sgd, Some(_)) => {
                anyhow::bail!("SGD GRPO checkpoint unexpectedly has optimizer state")
            }
            (_, Some(state)) => anyhow::ensure!(
                u64::from(state.step_count()) == loop_state.global_step,
                "GRPO optimizer step {} differs from loop step {}",
                state.step_count(),
                loop_state.global_step
            ),
            (_, None) => anyhow::bail!("stateful GRPO optimizer has no checkpoint state"),
        }
        match (ema_ref_state, loop_state.ema_groups_since_refresh) {
            (None, None) => {}
            (Some(state), Some(position)) => anyhow::ensure!(
                state.groups_since_refresh as u64 == position,
                "GRPO EMA tensor state cursor differs from loop state"
            ),
            _ => anyhow::bail!("GRPO checkpoint EMA cursor is inconsistent"),
        }
        let progress = self.progress(loop_state);
        loop_state.validate(&progress)?;
        let manifest = self.manifest(progress)?;
        let target = output_root.join(format!(
            "{}-checkpoint-step-{:08}.kiln-checkpoint",
            self.adapter_name, loop_state.global_step
        ));
        params.sync_to_master(backend)?;
        let adapter_parameters = params.capture_checkpoint_parameters()?;
        let optimizer_state = opt_state
            .as_mut()
            .map(|state| state.capture_checkpoint_state(params, backend))
            .transpose()?;
        let reference_state = ema_ref_state
            .map(|state| capture_lora_reference_checkpoint(&state.snapshot))
            .transpose()?;

        let mut artifacts = vec![
            crate::checkpoint::CheckpointArtifact {
                relative_path: GRPO_CHECKPOINT_ADAPTER_FILE.to_string(),
                role: crate::checkpoint::CheckpointFileRole::AdapterParameters,
            },
            crate::checkpoint::CheckpointArtifact {
                relative_path: GRPO_CHECKPOINT_LOOP_STATE_FILE.to_string(),
                role: crate::checkpoint::CheckpointFileRole::LossHistory,
            },
        ];
        if optimizer_state.is_some() {
            artifacts.push(crate::checkpoint::CheckpointArtifact {
                relative_path: GRPO_CHECKPOINT_OPTIMIZER_FILE.to_string(),
                role: crate::checkpoint::CheckpointFileRole::OptimizerState,
            });
        }
        if reference_state.is_some() {
            artifacts.push(crate::checkpoint::CheckpointArtifact {
                relative_path: GRPO_CHECKPOINT_REFERENCE_FILE.to_string(),
                role: crate::checkpoint::CheckpointFileRole::ReferenceState,
            });
        }
        let loop_state_bytes = serde_json::to_vec_pretty(loop_state)
            .context("serialize GRPO checkpoint loop state")?;
        Ok(GrpoCheckpointSnapshot {
            target,
            manifest,
            artifacts,
            adapter_parameters,
            optimizer_state,
            reference_state,
            loop_state_bytes,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn save(
        &self,
        output_root: &Path,
        backend: &dyn BackendRuntime,
        params: &mut TrainableLoraParams,
        opt_state: &mut Option<OptimizerState>,
        ema_ref_state: Option<&EmaReferenceState>,
        loop_state: &mut GrpoCheckpointLoopState,
        gpu_step_coordination: Option<&GpuStepCoordination>,
        gpu_writer_timings: &mut GrpoGpuWriterTimings,
        phase: &'static str,
    ) -> Result<PathBuf> {
        let mut snapshot = run_coordinated_grpo_gpu_phase(
            gpu_step_coordination,
            backend,
            gpu_writer_timings,
            phase,
            || {
                self.capture(
                    output_root,
                    backend,
                    params,
                    opt_state,
                    ema_ref_state,
                    loop_state,
                )
            },
        )?;
        // Capture the acquisition/wait update produced by the snapshot phase
        // itself. Tensor copying is already complete, so re-encoding this CPU
        // metadata does not extend writer ownership.
        loop_state.gpu_writer_timings = gpu_writer_timings.clone();
        snapshot.replace_loop_state(loop_state)?;
        let publish_started = Instant::now();
        let path = snapshot.publish()?;
        tracing::info!(
            checkpoint = %path.display(),
            publish_ms = publish_started.elapsed().as_millis() as u64,
            "published exact GRPO checkpoint"
        );
        Ok(path)
    }
}

fn load_grpo_checkpoint_loop_state(
    checkpoint: &crate::checkpoint::ValidatedTrainingCheckpoint,
) -> Result<GrpoCheckpointLoopState> {
    let relative = checkpoint
        .manifest
        .state_files
        .loss_history
        .as_deref()
        .context("GRPO resume checkpoint has no loop-state file")?;
    anyhow::ensure!(
        relative == GRPO_CHECKPOINT_LOOP_STATE_FILE,
        "unsupported GRPO loop-state artifact {relative:?}"
    );
    let path = checkpoint.artifact_path(relative)?;
    let bytes = std::fs::read(&path)
        .with_context(|| format!("read GRPO checkpoint loop state {}", path.display()))?;
    serde_json::from_slice(&bytes).context("parse strict GRPO checkpoint loop state")
}

fn grpo_checkpoint_effective_config(
    config: &GrpoConfig,
    learning_rate: f64,
    effective_seed: u64,
) -> Result<serde_json::Value> {
    let mut value = serde_json::to_value(config).context("serialize effective GRPO config")?;
    let object = value
        .as_object_mut()
        .context("serialized GRPO config is not an object")?;
    object.remove("resume_checkpoint");
    object.insert(
        "learning_rate".to_string(),
        serde_json::json!(learning_rate),
    );
    object.insert("seed".to_string(), serde_json::json!(effective_seed));
    canonical_checkpoint_json_value(value)
}

#[allow(clippy::too_many_arguments)]
fn grpo_checkpoint_auxiliary_state(
    route: GrpoCheckpointRoute,
    model_config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    precision_policy: TrainingPrecisionPolicy,
    base_model_weights_sha256: Option<&str>,
    base_weight_shard_manifest: Option<&kiln_core::model_provenance::BaseWeightShardManifest>,
    execution_provenance: Option<&kiln_core::execution_provenance::ExecutionProvenanceV1>,
    backend_runtime: &str,
    trainable_order_sha256: &str,
    gradient_checkpoint_plan_sha256: &str,
    training_runtime_planning_identity: &serde_json::Value,
) -> serde_json::Value {
    let hashes =
        kiln_core::config_hashes::ConfigHashes::from_model_tokenizer(model_config, tokenizer, None);
    serde_json::json!({
        "loop_state_type": GRPO_CHECKPOINT_LOOP_STATE_TYPE,
        "route": route,
        "model_config_sha256": hashes.model_config_hash,
        "tokenizer_config_sha256": hashes.tokenizer_config_hash,
        "chat_template_sha256": hashes.chat_template_hash,
        "base_model_weights_sha256": base_model_weights_sha256,
        "base_weight_shard_manifest": base_weight_shard_manifest,
        "execution_provenance": execution_provenance,
        "backend_runtime": backend_runtime,
        "kiln_train_version": env!("CARGO_PKG_VERSION"),
        "trainable_order_sha256": trainable_order_sha256,
        "gradient_checkpoint_plan_sha256": gradient_checkpoint_plan_sha256,
        "training_precision_policy": precision_policy.name,
        "training_runtime_planning_identity": training_runtime_planning_identity,
    })
}

fn sft_hyperparameters(
    config: &SftConfig,
    effective_seed: Option<u64>,
    alpha_over_rank: Option<f32>,
) -> crate::train_receipt::HyperparameterReceipt {
    crate::train_receipt::HyperparameterReceipt {
        mode: "sft".to_string(),
        rank: config.lora_rank,
        alpha: config.lora_alpha,
        alpha_over_rank,
        // Receipts record the RESOLVED learning rate — the value the
        // optimizer actually stepped with — not the Option.
        learning_rate: config.effective_learning_rate(),
        epochs: config.epochs,
        seed: effective_seed,
        shuffle: true,
    }
}

fn grpo_hyperparameters(
    config: &GrpoConfig,
    effective_seed: Option<u64>,
    alpha_over_rank: Option<f32>,
) -> crate::train_receipt::HyperparameterReceipt {
    crate::train_receipt::HyperparameterReceipt {
        mode: "grpo".to_string(),
        rank: config.lora_rank,
        alpha: config.lora_alpha,
        alpha_over_rank,
        // Resolved value, as in `sft_hyperparameters`.
        learning_rate: config.effective_learning_rate(),
        epochs: 1,
        seed: effective_seed,
        shuffle: false,
    }
}

fn grpo_echo_receipt(config: &GrpoConfig) -> crate::train_receipt::EchoReceipt {
    match config.loss.echo.as_ref() {
        // The env-CE term is live again (resurrection PR2 #1512): `enabled`
        // records whether the term is armed for this run (λ≠0). Whether it
        // actually FIRED shows in initial/final_env_ce, filled in by the
        // EchoActivityMetrics from per-step observations — a run whose
        // rollouts carry no env tokens keeps env_ce: None and the standing
        // warn_echo_enabled_without_env_tokens diagnostic.
        Some(echo) => crate::train_receipt::EchoReceipt {
            enabled: config.loss.echo_enabled(),
            lambda: Some(echo.lambda),
            env_mask_mode: serde_json::to_value(echo.env_mask_mode)
                .ok()
                .and_then(|v| v.as_str().map(ToString::to_string)),
            warning_filter: Some(echo.warning_filter),
            initial_env_ce: None,
            final_env_ce: None,
            dropped_reason: None,
        },
        None => crate::train_receipt::EchoReceipt::disabled(),
    }
}

fn grpo_settings_receipt(
    config: &GrpoConfig,
    dynamic_groups_filtered: usize,
) -> crate::train_receipt::GrpoReceipt {
    crate::train_receipt::GrpoReceipt {
        kl_coeff: config.kl_coeff,
        clip_epsilon: config.clip_epsilon,
        clip_eps_high: config.clip_eps_high,
        cispo_max_weight: config.cispo_max_weight,
        dynamic_sampling: config.dynamic_sampling,
        dynamic_groups_filtered,
        advantage_mode: serde_json::to_value(config.advantage_mode)
            .unwrap_or(serde_json::Value::Null),
        loss_aggregation: serde_json::to_value(config.loss_aggregation)
            .unwrap_or(serde_json::Value::Null),
        kl_estimator: serde_json::to_value(config.kl_estimator).unwrap_or(serde_json::Value::Null),
        is_level: serde_json::to_value(config.is_level).unwrap_or(serde_json::Value::Null),
        behavior_policy: serde_json::to_value(config.behavior_policy)
            .unwrap_or(serde_json::Value::Null),
        kl_reference_policy: serde_json::to_value(&config.kl_reference_policy)
            .unwrap_or(serde_json::Value::Null),
        entropy_aware_kl_quantile: config.entropy_aware_kl_quantile,
        policy_audit: None,
    }
}

#[derive(Debug, Clone)]
struct RewardFilterInputGroup {
    id: String,
    source_index: usize,
    source_line: Option<usize>,
    reward_variance: f64,
}

#[derive(Debug, Clone)]
struct RewardFilterPlan {
    kept_source_indices: Vec<usize>,
    kept_source_lines: Vec<usize>,
    skip_training: bool,
    failure_reason: Option<String>,
    sidecar_path: PathBuf,
    groups_kept: usize,
    groups_dropped: usize,
}

impl RewardFilterPlan {
    fn keeps_source_index(&self, source_index: usize) -> bool {
        self.kept_source_indices
            .binary_search(&source_index)
            .is_ok()
    }

    fn keeps_source_line(&self, line_no: usize) -> bool {
        self.kept_source_lines.binary_search(&line_no).is_ok()
    }
}

fn reward_filter_enabled(config: &GrpoConfig) -> bool {
    config.reward_filter_var_min.is_some() || config.reward_filter_var_max.is_some()
}

fn validate_reward_filter_config(config: &GrpoConfig) -> Result<()> {
    if let Some(var_min) = config.reward_filter_var_min {
        anyhow::ensure!(
            var_min.is_finite() && var_min >= 0.0,
            "reward filter --filter-var-min must be a finite non-negative float"
        );
    }
    if let Some(var_max) = config.reward_filter_var_max {
        anyhow::ensure!(
            var_max.is_finite() && var_max >= 0.0,
            "reward filter --filter-var-max must be a finite non-negative float"
        );
    }
    if let (Some(var_min), Some(var_max)) =
        (config.reward_filter_var_min, config.reward_filter_var_max)
    {
        anyhow::ensure!(
            var_min <= var_max,
            "reward filter --filter-var-min must be <= --filter-var-max"
        );
    }
    anyhow::ensure!(
        config.reward_filter_min_groups > 0,
        "reward filter --min-groups must be at least 1"
    );
    Ok(())
}

fn reward_filter_group_matches(
    variance: f64,
    var_min: Option<f64>,
    var_max: Option<f64>,
) -> (bool, Option<String>) {
    if let Some(min) = var_min {
        if variance < min {
            return (false, Some(format!("variance_below_min:{min}")));
        }
    }
    if let Some(max) = var_max {
        if variance > max {
            return (false, Some(format!("variance_above_max:{max}")));
        }
    }
    (true, None)
}

fn reward_filter_variance(rewards: &[f64]) -> f64 {
    if rewards.is_empty() {
        return 0.0;
    }
    let mean = rewards.iter().sum::<f64>() / rewards.len() as f64;
    rewards
        .iter()
        .map(|reward| {
            let centered = *reward - mean;
            centered * centered
        })
        .sum::<f64>()
        / rewards.len() as f64
}

#[derive(Debug)]
struct StreamedRewardStatsAccumulator {
    count: usize,
    mean: f64,
    sum_squared_deviation: f64,
    min: f64,
    max: f64,
    group_count: usize,
    all_pass_group_count: usize,
    all_fail_group_count: usize,
    degenerate_group_count: usize,
    variance_histogram_counts: [usize; 6],
}

impl Default for StreamedRewardStatsAccumulator {
    fn default() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            sum_squared_deviation: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            group_count: 0,
            all_pass_group_count: 0,
            all_fail_group_count: 0,
            degenerate_group_count: 0,
            variance_histogram_counts: [0; 6],
        }
    }
}

impl StreamedRewardStatsAccumulator {
    fn observe_group<'a, I>(&mut self, rewards: I, all_pass_threshold: f64) -> f64
    where
        I: IntoIterator<Item = &'a f64>,
    {
        let mut group_count = 0usize;
        let mut group_mean = 0.0;
        let mut group_squared_deviation = 0.0;
        let mut all_pass = true;
        let mut all_fail = true;
        for reward in rewards {
            let reward = *reward;
            group_count += 1;
            let group_delta = reward - group_mean;
            group_mean += group_delta / group_count as f64;
            group_squared_deviation += group_delta * (reward - group_mean);

            self.count += 1;
            let delta = reward - self.mean;
            self.mean += delta / self.count as f64;
            self.sum_squared_deviation += delta * (reward - self.mean);
            self.min = self.min.min(reward);
            self.max = self.max.max(reward);
            all_pass &= reward >= all_pass_threshold;
            all_fail &= reward <= 0.0;
        }
        if group_count == 0 {
            return 0.0;
        }
        self.group_count += 1;
        self.all_pass_group_count += if all_pass { 1 } else { 0 };
        self.all_fail_group_count += if all_fail { 1 } else { 0 };
        let variance = (group_squared_deviation / group_count as f64).max(0.0);
        if variance <= crate::train_receipt::REWARD_DEGENERATE_GROUP_VARIANCE_EPSILON {
            self.degenerate_group_count += 1;
        }
        let bucket = if variance == 0.0 {
            0
        } else if variance <= 1e-6 {
            1
        } else if variance <= 0.01 {
            2
        } else if variance <= 0.25 {
            3
        } else if variance <= 1.0 {
            4
        } else {
            5
        };
        self.variance_histogram_counts[bucket] += 1;
        variance
    }

    fn finish(self) -> crate::train_receipt::RewardStatsReceipt {
        if self.count == 0 {
            return crate::train_receipt::RewardStatsReceipt::default();
        }
        let specs = [
            ("zero", Some(0.0), Some(0.0)),
            ("tiny", Some(f64::MIN_POSITIVE), Some(1e-6)),
            ("low", Some(1e-6), Some(0.01)),
            ("medium", Some(0.01), Some(0.25)),
            ("high", Some(0.25), Some(1.0)),
            ("extreme", Some(1.0), None),
        ];
        crate::train_receipt::RewardStatsReceipt {
            count: self.count,
            mean: Some(self.mean),
            stdev: Some(
                (self.sum_squared_deviation / self.count as f64)
                    .max(0.0)
                    .sqrt(),
            ),
            min: Some(self.min),
            max: Some(self.max),
            group_count: self.group_count,
            all_pass_group_count: self.all_pass_group_count,
            all_fail_group_count: self.all_fail_group_count,
            degenerate_group_count: self.degenerate_group_count,
            group_variance_histogram: specs
                .into_iter()
                .zip(self.variance_histogram_counts)
                .map(|((label, min_inclusive, max_inclusive), count)| {
                    crate::train_receipt::HistogramBucket {
                        label: label.to_string(),
                        min_inclusive,
                        max_inclusive,
                        count,
                    }
                })
                .collect(),
        }
    }
}

fn reward_filter_on_empty_label(mode: RewardFilterOnEmpty) -> &'static str {
    match mode {
        RewardFilterOnEmpty::Fail => "fail",
        RewardFilterOnEmpty::TrainAll => "train-all",
        RewardFilterOnEmpty::Skip => "skip",
    }
}

fn build_reward_filter_plan(
    config: &GrpoConfig,
    output_dir: &Path,
    source: &str,
    groups: Vec<RewardFilterInputGroup>,
) -> Result<Option<RewardFilterPlan>> {
    if !reward_filter_enabled(config) {
        return Ok(None);
    }
    validate_reward_filter_config(config)?;

    let mut candidate_kept_count = 0usize;
    let mut decisions = Vec::new();
    for group in &groups {
        let variance = group.reward_variance;
        let (matched_filter, reject_reason) = reward_filter_group_matches(
            variance,
            config.reward_filter_var_min,
            config.reward_filter_var_max,
        );
        if matched_filter {
            candidate_kept_count = candidate_kept_count.saturating_add(1);
        }
        decisions.push(crate::train_receipt::RewardFilterGroupDecisionReceipt {
            id: group.id.clone(),
            source_index: group.source_index,
            source_line: group.source_line,
            reward_variance: variance,
            matched_filter,
            kept: matched_filter,
            reject_reason,
        });
    }

    let empty_filter_triggered = candidate_kept_count < config.reward_filter_min_groups;
    let on_empty = config.reward_filter_on_empty;
    let empty_filter_action = if empty_filter_triggered {
        reward_filter_on_empty_label(on_empty)
    } else {
        "use-filter"
    };

    let mut kept_ids = Vec::new();
    let mut dropped_ids = Vec::new();
    let mut kept_indices = Vec::new();
    let mut kept_lines = Vec::new();
    let mut skip_training = false;
    let mut failure_reason = None;

    if empty_filter_triggered {
        match on_empty {
            RewardFilterOnEmpty::Fail => {
                dropped_ids = groups.iter().map(|group| group.id.clone()).collect();
                for decision in &mut decisions {
                    decision.kept = false;
                    decision
                        .reject_reason
                        .get_or_insert_with(|| "below_min_groups".to_string());
                }
                failure_reason = Some(format!(
                    "reward variance filter kept {} group(s), below --min-groups {}; --on-empty-filter=fail",
                    candidate_kept_count, config.reward_filter_min_groups
                ));
            }
            RewardFilterOnEmpty::TrainAll => {
                kept_ids = groups.iter().map(|group| group.id.clone()).collect();
                for group in &groups {
                    kept_indices.push(group.source_index);
                    if let Some(line) = group.source_line {
                        kept_lines.push(line);
                    }
                }
                for decision in &mut decisions {
                    decision.kept = true;
                    decision.reject_reason = None;
                }
            }
            RewardFilterOnEmpty::Skip => {
                skip_training = true;
                dropped_ids = groups.iter().map(|group| group.id.clone()).collect();
                for decision in &mut decisions {
                    decision.kept = false;
                    decision
                        .reject_reason
                        .get_or_insert_with(|| "below_min_groups".to_string());
                }
            }
        }
    } else {
        for (group, decision) in groups.iter().zip(&decisions) {
            if decision.matched_filter {
                kept_ids.push(group.id.clone());
                kept_indices.push(group.source_index);
                if let Some(line) = group.source_line {
                    kept_lines.push(line);
                }
            } else {
                dropped_ids.push(group.id.clone());
            }
        }
    }

    let sidecar = crate::train_receipt::RewardFilterSidecar {
        schema_version: 1,
        sidecar_type: "kiln_reward_filter_groups".to_string(),
        source: source.to_string(),
        var_min: config.reward_filter_var_min,
        var_max: config.reward_filter_var_max,
        min_groups: config.reward_filter_min_groups,
        on_empty_filter: reward_filter_on_empty_label(on_empty).to_string(),
        empty_filter_triggered,
        empty_filter_action: empty_filter_action.to_string(),
        groups_read: groups.len(),
        groups_kept: kept_ids.len(),
        groups_dropped: dropped_ids.len(),
        kept_group_ids: kept_ids,
        dropped_group_ids: dropped_ids,
        groups: decisions,
    };
    let sidecar_path = crate::train_receipt::write_reward_filter_sidecar(output_dir, &sidecar)?;
    kept_indices.sort_unstable();
    kept_indices.dedup();
    kept_lines.sort_unstable();
    kept_lines.dedup();
    Ok(Some(RewardFilterPlan {
        groups_kept: sidecar.groups_kept,
        groups_dropped: sidecar.groups_dropped,
        kept_source_indices: kept_indices,
        kept_source_lines: kept_lines,
        skip_training,
        failure_reason,
        sidecar_path,
    }))
}

fn record_reward_filter_plan(
    data_stats: &mut crate::train_receipt::DataStatsReceipt,
    plan: &RewardFilterPlan,
) {
    data_stats.reward_groups_kept = plan.groups_kept;
    data_stats.reward_groups_filtered = plan.groups_dropped;
    data_stats.reward_filter_sidecar = Some(plan.sidecar_path.display().to_string());
}

fn run_adapter_smoke_test_best_effort(
    adapter_name: &str,
    backend: &dyn BackendRuntime,
    weights: &GpuWeights,
    model_config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    params: &TrainableLoraParams,
    configured_prompts: Option<&[String]>,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> crate::train_receipt::AdapterSmokeTestReceipt {
    let receipt = run_adapter_smoke_test(
        backend,
        weights,
        model_config,
        tokenizer,
        params,
        configured_prompts,
        streaming_prefill,
    )
    .unwrap_or_else(|err| {
        crate::train_receipt::failed_adapter_smoke_test_receipt(format!("{err:#}"))
    });
    if receipt.passed {
        tracing::info!(
            adapter = adapter_name,
            prompts = receipt.prompts.len(),
            "adapter smoke test passed"
        );
    } else {
        for warning in &receipt.warnings {
            tracing::warn!(
                adapter = adapter_name,
                warning = %warning,
                "adapter smoke test warning"
            );
        }
    }
    receipt
}

fn run_adapter_smoke_test(
    backend: &dyn BackendRuntime,
    weights: &GpuWeights,
    model_config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    params: &TrainableLoraParams,
    configured_prompts: Option<&[String]>,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<crate::train_receipt::AdapterSmokeTestReceipt> {
    let lora = lora_weights_detached(params);
    let smoke_prompts = adapter_smoke_test_prompts(configured_prompts)?;
    let mut prompts = Vec::with_capacity(smoke_prompts.len());

    for prompt in &smoke_prompts {
        let prompt_ids = tokenizer
            .encode(prompt)
            .map_err(|err| anyhow::anyhow!("{err}"))
            .with_context(|| format!("tokenize adapter smoke prompt {prompt:?}"))?;
        anyhow::ensure!(
            !prompt_ids.is_empty(),
            "adapter smoke prompt tokenized to zero tokens: {prompt:?}"
        );

        let base_logits = adapter_smoke_forward_logits(
            backend,
            &prompt_ids,
            weights,
            model_config,
            None,
            streaming_prefill,
        )
        .with_context(|| format!("base forward for adapter smoke prompt {prompt:?}"))?;
        let adapter_logits = adapter_smoke_forward_logits(
            backend,
            &prompt_ids,
            weights,
            model_config,
            Some(&lora),
            streaming_prefill,
        )
        .with_context(|| format!("adapter forward for adapter smoke prompt {prompt:?}"))?;
        let (finite_logits, logit_delta_l2) =
            adapter_smoke_logit_delta_l2(&base_logits, &adapter_logits)
                .with_context(|| format!("compare adapter smoke logits for {prompt:?}"))?;

        let base_generation = adapter_smoke_greedy_generate(
            backend,
            weights,
            model_config,
            tokenizer,
            prompt,
            None,
            streaming_prefill,
        )
        .with_context(|| format!("base generation for adapter smoke prompt {prompt:?}"))?;
        let adapter_generation = adapter_smoke_greedy_generate(
            backend,
            weights,
            model_config,
            tokenizer,
            prompt,
            Some(&lora),
            streaming_prefill,
        )
        .with_context(|| format!("adapter generation for adapter smoke prompt {prompt:?}"))?;

        prompts.push(crate::train_receipt::AdapterSmokePromptReceipt {
            prompt: prompt.to_string(),
            finite_logits,
            logit_delta_l2,
            generated_text_different: base_generation.output != adapter_generation.output,
            base_output: base_generation.output,
            adapter_output_chars: adapter_generation.output.chars().count(),
            adapter_output: adapter_generation.output,
            adapter_output_tokens: adapter_generation.output_tokens,
            base_generation_ms: base_generation.elapsed_ms,
            adapter_generation_ms: adapter_generation.elapsed_ms,
        });
    }

    Ok(crate::train_receipt::build_adapter_smoke_test_receipt(
        prompts,
    ))
}

fn adapter_smoke_test_prompts(configured: Option<&[String]>) -> Result<Vec<String>> {
    let prompts = match configured {
        Some(prompts) => prompts.to_vec(),
        None => ADAPTER_SMOKE_TEST_PROMPTS
            .iter()
            .map(|prompt| (*prompt).to_string())
            .collect(),
    };
    anyhow::ensure!(
        !prompts.is_empty(),
        "adapter_smoke_prompts must contain at least one prompt"
    );
    for (index, prompt) in prompts.iter().enumerate() {
        anyhow::ensure!(
            !prompt.trim().is_empty(),
            "adapter_smoke_prompts[{index}] must not be blank"
        );
    }
    Ok(prompts)
}

fn adapter_smoke_linear_state(
    backend: &dyn BackendRuntime,
    weights: &GpuWeights,
    model_config: &ModelConfig,
) -> Result<LinearAttentionState> {
    // (#1082) `Tensor::device()` returns an owned kt `Device` (Copy); the
    // constructor wants `&Device`, so bind to a local and borrow.
    let kt_device = weights.embed_tokens.device();
    LinearAttentionState::new_with_batch_for_inference_runtime(model_config, 1, &kt_device, backend)
}

fn adapter_smoke_forward_logits(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    model_config: &ModelConfig,
    lora: Option<&LoraWeights>,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<Tensor> {
    let mut linear_state = adapter_smoke_linear_state(backend, weights, model_config)?;
    model_forward_kt_with_policy(
        backend,
        token_ids,
        weights,
        model_config,
        None,
        Some(&mut linear_state),
        lora,
        streaming_prefill,
    )
}

fn adapter_smoke_logit_delta_l2(
    base_logits: &Tensor,
    adapter_logits: &Tensor,
) -> Result<(bool, Option<f64>)> {
    let base = adapter_smoke_last_logits(base_logits)?;
    let adapter = adapter_smoke_last_logits(adapter_logits)?;
    anyhow::ensure!(
        base.len() == adapter.len(),
        "base and adapter logits have different vocab sizes: {} vs {}",
        base.len(),
        adapter.len()
    );

    let finite_logits = base
        .iter()
        .chain(adapter.iter())
        .all(|value| value.is_finite());
    if !finite_logits {
        return Ok((false, None));
    }

    let sum_sq = base
        .iter()
        .zip(adapter.iter())
        .map(|(base, adapter)| {
            let delta = *adapter as f64 - *base as f64;
            delta * delta
        })
        .sum::<f64>();
    let l2 = sum_sq.sqrt();
    Ok((l2.is_finite(), l2.is_finite().then_some(l2)))
}

fn adapter_smoke_last_logits(logits: &Tensor) -> Result<Vec<f32>> {
    let dims = logits.dims();
    anyhow::ensure!(
        dims.len() >= 2,
        "adapter smoke logits must have at least 2 dimensions, got {dims:?}"
    );
    let seq_dim = dims.len() - 2;
    let seq_len = dims[seq_dim];
    anyhow::ensure!(
        seq_len > 0,
        "adapter smoke logits have zero sequence length"
    );
    Ok(logits
        .narrow(seq_dim, seq_len - 1, 1)?
        .squeeze(seq_dim)?
        .flatten_all()?
        .to_f32_dtype()?
        .to_vec1::<f32>()?)
}

fn adapter_smoke_greedy_generate(
    backend: &dyn BackendRuntime,
    weights: &GpuWeights,
    model_config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    prompt: &str,
    lora: Option<&LoraWeights>,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<AdapterSmokeGeneration> {
    let mut context = tokenizer
        .encode(prompt)
        .map_err(|err| anyhow::anyhow!("{err}"))
        .with_context(|| format!("tokenize adapter smoke generation prompt {prompt:?}"))?;
    anyhow::ensure!(
        !context.is_empty(),
        "adapter smoke generation prompt tokenized to zero tokens: {prompt:?}"
    );

    let started = Instant::now();
    let mut generated = Vec::with_capacity(ADAPTER_SMOKE_TEST_MAX_NEW_TOKENS);
    for _ in 0..ADAPTER_SMOKE_TEST_MAX_NEW_TOKENS {
        let logits = adapter_smoke_forward_logits(
            backend,
            &context,
            weights,
            model_config,
            lora,
            streaming_prefill,
        )?;
        let token = greedy_sample(&logits)?;
        generated.push(token);
        context.push(token);
    }

    let output = tokenizer
        .decode(&generated)
        .map_err(|err| anyhow::anyhow!("{err}"))
        .context("decode adapter smoke generated tokens")?;
    let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    Ok(AdapterSmokeGeneration {
        output,
        output_tokens: generated.len(),
        elapsed_ms,
    })
}

#[allow(clippy::too_many_arguments)]
fn write_sft_train_receipt_best_effort(
    adapter_name: &str,
    model_config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    base_weight_shard_manifest: Option<&kiln_core::model_provenance::BaseWeightShardManifest>,
    execution_provenance: Option<&kiln_core::execution_provenance::ExecutionProvenanceV1>,
    training_precision: Option<crate::checkpoint::TrainingCheckpointPrecision>,
    sft_loss_route: SftFlceLossRoute,
    config: &SftConfig,
    effective_seed: Option<u64>,
    alpha_over_rank: Option<f32>,
    base_adapter_dir: Option<&Path>,
    output_dir: &Path,
    training_data_sha256: Option<String>,
    ingestion: &crate::sft_ingestion::SftIngestionReceipt,
    data: crate::train_receipt::DataStatsReceipt,
    token_counts: crate::train_receipt::TokenCountReceipt,
    wall_clock_ms: u64,
    lora_grad_norms: Vec<crate::train_receipt::LoraGradNormSummary>,
    adapter_smoke_test: Option<crate::train_receipt::AdapterSmokeTestReceipt>,
    status_error: Option<String>,
) {
    let mut receipt = crate::train_receipt::TrainReceipt::new(
        adapter_name,
        "sft",
        model_config,
        tokenizer,
        sft_hyperparameters(config, effective_seed, alpha_over_rank),
        serde_json::to_value(config).unwrap_or(serde_json::Value::Null),
    );
    receipt.model.base_weight_shard_manifest = base_weight_shard_manifest.cloned();
    receipt.runtime.execution_provenance = execution_provenance.cloned();
    receipt.runtime.training_precision = training_precision;
    receipt.runtime.sft_loss_route = Some(sft_loss_route);
    receipt.training_data = crate::train_receipt::TrainingDataReceipt {
        source: ingestion.source.clone(),
        path: ingestion.source_locator.clone(),
        sha256: training_data_sha256,
    };
    receipt.adapters.base = crate::train_receipt::adapter_file_receipt(base_adapter_dir);
    receipt.adapters.output = crate::train_receipt::adapter_file_receipt(Some(output_dir));
    receipt.data = data;
    receipt.token_counts = token_counts;
    receipt.runtime.wall_clock_ms = wall_clock_ms;
    receipt.lora_grad_norms = lora_grad_norms;
    receipt.adapter_smoke_test = adapter_smoke_test;
    crate::train_receipt::log_training_token_counts("sft", &receipt.token_counts);
    if status_error.is_none() {
        receipt.lora_delta_norms =
            crate::train_receipt::lora_delta_norm_summary_from_adapter(
                output_dir,
                alpha_over_rank.unwrap_or(0.0) as f64,
            )
            .unwrap_or_else(|err| {
                tracing::warn!(adapter = adapter_name, error = %err, "failed to summarize LoRA delta norms for train receipt");
                Vec::new()
            });
        crate::train_receipt::warn_lora_delta_norms(
            "sft",
            adapter_name,
            &receipt.lora_delta_norms,
            alpha_over_rank.unwrap_or(0.0) as f64,
        );
    }
    if let Some(err) = status_error {
        receipt = receipt.mark_failed(err);
    }
    if let Err(err) = receipt.write_to_adapter_dir(output_dir) {
        tracing::warn!(adapter = adapter_name, error = %err, "failed to write SFT train receipt");
    }
}

#[allow(clippy::too_many_arguments)]
fn build_grpo_train_receipt(
    adapter_name: &str,
    model_config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    base_weight_shard_manifest: Option<&kiln_core::model_provenance::BaseWeightShardManifest>,
    execution_provenance: Option<&kiln_core::execution_provenance::ExecutionProvenanceV1>,
    training_precision: Option<crate::checkpoint::TrainingCheckpointPrecision>,
    config: &GrpoConfig,
    effective_seed: Option<u64>,
    alpha_over_rank: Option<f32>,
    base_adapter_dir: Option<&Path>,
    output_dir: &Path,
    training_data: crate::train_receipt::TrainingDataReceipt,
    data: crate::train_receipt::DataStatsReceipt,
    rewards: crate::train_receipt::RewardStatsReceipt,
    token_counts: crate::train_receipt::TokenCountReceipt,
    phase_timings: crate::train_receipt::TrainingPhaseTimingsReceipt,
    echo_metrics: crate::train_receipt::EchoActivityMetrics,
    wall_clock_ms: u64,
    dynamic_groups_filtered: usize,
    adapter_smoke_test: Option<crate::train_receipt::AdapterSmokeTestReceipt>,
    lora_grad_norms: Vec<crate::train_receipt::LoraGradNormSummary>,
    policy_audit: Option<crate::train_receipt::GrpoPolicyAuditReceipt>,
    status_error: Option<String>,
) -> crate::train_receipt::TrainReceipt {
    let mut receipt = crate::train_receipt::TrainReceipt::new(
        adapter_name,
        "grpo",
        model_config,
        tokenizer,
        grpo_hyperparameters(config, effective_seed, alpha_over_rank),
        serde_json::to_value(config).unwrap_or(serde_json::Value::Null),
    );
    receipt.model.base_weight_shard_manifest = base_weight_shard_manifest.cloned();
    receipt.runtime.execution_provenance = execution_provenance.cloned();
    receipt.runtime.training_precision = training_precision;
    receipt.training_data = training_data;
    receipt.adapters.base = crate::train_receipt::adapter_file_receipt(base_adapter_dir);
    receipt.adapters.output = crate::train_receipt::adapter_file_receipt(Some(output_dir));
    let mut grpo = grpo_settings_receipt(config, dynamic_groups_filtered);
    grpo.policy_audit = policy_audit;
    receipt.grpo = Some(grpo);
    receipt.echo = grpo_echo_receipt(config);
    echo_metrics.apply_to_echo_receipt(&mut receipt.echo);
    receipt.no_policy_loss = config.loss.no_policy_loss;
    receipt.data = data;
    receipt.rewards = rewards;
    receipt.token_counts = token_counts;
    receipt.phase_timings = phase_timings;
    receipt.runtime.wall_clock_ms = wall_clock_ms;
    receipt.adapter_smoke_test = adapter_smoke_test;
    receipt.lora_grad_norms = lora_grad_norms;
    crate::train_receipt::log_training_token_counts("grpo", &receipt.token_counts);
    crate::train_receipt::warn_echo_enabled_without_env_tokens(
        "grpo",
        config.loss.echo_enabled(),
        &receipt.token_counts,
    );
    crate::train_receipt::warn_reward_diagnostics(
        "grpo",
        adapter_name,
        &receipt.rewards,
        config.reward_saturation_threshold,
        config.reward_low_variance_threshold,
    );
    if status_error.is_none() {
        receipt.lora_delta_norms =
            crate::train_receipt::lora_delta_norm_summary_from_adapter(
                output_dir,
                alpha_over_rank.unwrap_or(0.0) as f64,
            )
            .unwrap_or_else(|err| {
                tracing::warn!(adapter = adapter_name, error = %err, "failed to summarize LoRA delta norms for train receipt");
                Vec::new()
            });
        crate::train_receipt::warn_lora_delta_norms(
            "grpo",
            adapter_name,
            &receipt.lora_delta_norms,
            alpha_over_rank.unwrap_or(0.0) as f64,
        );
    }
    if let Some(err) = status_error {
        receipt = receipt.mark_failed(err);
    }
    receipt
}

#[allow(clippy::too_many_arguments)]
fn write_grpo_train_receipt_best_effort(
    adapter_name: &str,
    model_config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    base_weight_shard_manifest: Option<&kiln_core::model_provenance::BaseWeightShardManifest>,
    execution_provenance: Option<&kiln_core::execution_provenance::ExecutionProvenanceV1>,
    training_precision: Option<crate::checkpoint::TrainingCheckpointPrecision>,
    config: &GrpoConfig,
    effective_seed: Option<u64>,
    alpha_over_rank: Option<f32>,
    base_adapter_dir: Option<&Path>,
    output_dir: &Path,
    training_data: crate::train_receipt::TrainingDataReceipt,
    data: crate::train_receipt::DataStatsReceipt,
    rewards: crate::train_receipt::RewardStatsReceipt,
    token_counts: crate::train_receipt::TokenCountReceipt,
    phase_timings: crate::train_receipt::TrainingPhaseTimingsReceipt,
    echo_metrics: crate::train_receipt::EchoActivityMetrics,
    wall_clock_ms: u64,
    dynamic_groups_filtered: usize,
    adapter_smoke_test: Option<crate::train_receipt::AdapterSmokeTestReceipt>,
    lora_grad_norms: Vec<crate::train_receipt::LoraGradNormSummary>,
    policy_audit: Option<crate::train_receipt::GrpoPolicyAuditReceipt>,
    status_error: Option<String>,
) {
    if let Some(audit) = policy_audit.as_ref() {
        tracing::info!(
            schema = %audit.schema,
            ratio_scope = audit
                .importance_sampling
                .ratio_scope
                .as_deref()
                .unwrap_or("none"),
            action_tokens = audit.importance_sampling.action_tokens,
            ratio_observations = audit.importance_sampling.ratio_observations,
            mean_ratio = ?audit.importance_sampling.mean_ratio,
            outside_clip_fraction = ?audit.importance_sampling.outside_clip_fraction,
            kl_tokens = audit.kl_reference.token_observations,
            mean_kl_estimator = ?audit.kl_reference.mean_estimator,
            mean_masked_kl_estimator = ?audit.kl_reference.mean_masked_estimator,
            recorded_completions = audit.recorded_provenance.completion_count,
            behavior_sources = audit.recorded_provenance.unique_behavior_sources,
            behavior_source_manifest_sha256 = audit
                .recorded_provenance
                .behavior_source_manifest_sha256
                .as_deref()
                .unwrap_or("none"),
            "GRPO policy audit"
        );
    }
    let receipt = build_grpo_train_receipt(
        adapter_name,
        model_config,
        tokenizer,
        base_weight_shard_manifest,
        execution_provenance,
        training_precision,
        config,
        effective_seed,
        alpha_over_rank,
        base_adapter_dir,
        output_dir,
        training_data,
        data,
        rewards,
        token_counts,
        phase_timings,
        echo_metrics,
        wall_clock_ms,
        dynamic_groups_filtered,
        adapter_smoke_test,
        lora_grad_norms,
        policy_audit,
        status_error,
    );
    if let Err(err) = receipt.write_to_adapter_dir(output_dir) {
        tracing::warn!(adapter = adapter_name, error = %err, "failed to write GRPO train receipt");
    }
}

fn finish_grpo_policy_audit<T>(
    training_result: &mut Result<T>,
    accumulator: crate::train_receipt::GrpoPolicyAuditAccumulator,
) -> Option<crate::train_receipt::GrpoPolicyAuditReceipt> {
    match accumulator.finish().context("finalize GRPO policy audit") {
        Ok(receipt) => Some(receipt),
        Err(error) => {
            if training_result.is_ok() {
                *training_result = Err(error);
            } else {
                tracing::warn!(error = %error, "failed to finalize partial GRPO policy audit");
            }
            None
        }
    }
}

/// Run SFT training on the provided examples using the already-loaded model.
///
/// This runs in the calling thread (blocking). The caller should spawn this
/// on a background thread to avoid blocking inference.
///
/// When `replay_ctx` is `Some`, the trainer writes a `replay.jsonl` request
/// record (with the resolved seed) and `lineage.json` into the adapter
/// directory *before* the optimizer step, then appends an outcome record
/// when training completes or fails. When `None`, no replay artifacts are
/// written — used by tests and benches that don't need replay.
///
/// Returns the path to the saved adapter directory.
/// Post-SFT MTP alignment phase (MTP training plan PR-B).
///
/// Trains LoRA on the native MTP draft block so the draft keeps
/// predicting what the freshly-tuned model would say — every LoRA step
/// moves the served distribution away from the frozen pretrained draft
/// head, so speculative-decode acceptance decays exactly in proportion
/// to personalization unless the draft trains too.
///
/// Per example: one detached no-head forward of the TUNED model gives
/// post-final-norm hiddens h (the same tensor `mtp_forward_step` consumes
/// as `h_prev` at serve time); the MTP block then trains under the kt
/// tape on `fused_t = fc(concat(norm_e(emb(tok_{t+1})), norm_h(h_t)))`
/// with the production FLCE root over the tied head — fed the shifted
/// `ids[1..]` / `mask[1..]`, which makes row t's label `ids[t+2]`: the
/// MTP objective, with zero new loss machinery. Only the seven
/// draft-block LoRA pairs receive gradients (the hiddens are detached;
/// fc / norms / tied head are frozen).
///
/// Returns `(examples_trained, initial_ce, final_ce)`; `None` when the
/// checkpoint has no MTP tensors or the phase is disabled.
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
#[allow(clippy::too_many_arguments)]
fn run_mtp_alignment_phase(
    backend: &dyn BackendRuntime,
    weights: &GpuWeights,
    model_config: &ModelConfig,
    params: &mut TrainableLoraParams,
    examples: &[SftExample],
    valid_indices: &[usize],
    tokenizer: &KilnTokenizer,
    config: &SftConfig,
    device: &Device,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<Option<(usize, Option<f64>, Option<f64>)>> {
    let enabled = config.train_mtp.unwrap_or(true);
    if !enabled || weights.mtp.is_none() {
        return Ok(None);
    }
    if !params.initialize_mtp_seeded(weights, device, Some(0x4D54_5042))? {
        return Ok(None);
    }
    let mtp = weights
        .mtp_weights()
        .context("mtp alignment: materializing mtp.* tensors")?;
    let mtp_full_attention = matches!(
        mtp.layer.attention,
        kiln_model::forward::GpuAttentionWeights::Full(_)
    );
    anyhow::ensure!(
        mtp_full_attention,
        "mtp alignment: MTP layer must be full attention"
    );

    // Move the draft-block pairs into a one-layer TrainableLoraParams so
    // the existing optimizer machinery (make_opt_state /
    // optimizer_step_dispatch, both keyed on `all_params`) drives EXACTLY
    // these seven pairs. Ownership moves — no clones — so Parameter
    // identity (tensor_id, registry residency) is preserved; the pairs
    // move back into `params.mtp` at the end for save_peft.
    let taken = params
        .mtp
        .take()
        .expect("initialize_mtp_seeded just populated params.mtp");
    let mut mtp_train = TrainableLoraParams {
        layers: vec![taken],
        mtp: None,
        rank: params.rank,
        alpha: params.alpha,
        scale: params.scale,
    };
    mtp_train
        .register_with_backend(backend)
        .context("mtp alignment: registering draft-block LoRA params with resident backend")?;

    // Serving view of the trained MAIN adapter (applied to the hiddens
    // forward) and the draft-block LoRA view (applied inside the block;
    // shares the SAME kt tensors the optimizer updates).
    let lora_view = params.as_lora_weights();
    let mtp_lora_view = {
        let mut v = mtp_train.as_lora_weights();
        v.layers.remove(0)
    };
    let lora_scale = params.scale;

    let learning_rate = config.effective_learning_rate();
    let mut opt_state = make_opt_state(&mtp_train, config.optimizer, learning_rate, device)?;
    if let Some(state) = opt_state.as_ref() {
        state.register_with_backend(backend)?;
    }
    let mut initial_ce: Option<f64> = None;
    let mut final_ce: Option<f64> = None;
    let mut trained = 0usize;
    let phase_started = Instant::now();

    for &idx in valid_indices {
        let ex = &examples[idx];
        let (input_ids, label_mask) = match tokenize_for_training(ex, tokenizer) {
            Ok(pair) => pair,
            Err(_) => continue,
        };
        let seq_len = input_ids.len();
        // The +2 objective needs at least one supervised label at t+2.
        if seq_len < 3 || !label_mask.get(2..).is_some_and(|m| m.iter().any(|&v| v)) {
            continue;
        }

        // 1) Detached hiddens from the TUNED model — outside any tape scope.
        let mut linear_state = LinearAttentionState::new(model_config, device)?;
        let hidden = model_forward_no_head_with_policy(
            backend,
            &input_ids,
            weights,
            model_config,
            Some(&mut linear_state),
            Some(&lora_view),
            streaming_prefill,
        )
        .context("mtp alignment: no-head hiddens forward")?
        .detach();

        // 2) MTP block forward + FLCE root under the tape-authoritative scope.
        let shifted_ids: Vec<u32> = input_ids[1..].to_vec();
        let shifted_mask: Vec<bool> = label_mask[1..].to_vec();
        let positions: Vec<u32> = (0..seq_len as u32 - 1).collect();
        let result = kiln_kt_bridge::tape_bridge::with_tape_authoritative_scope_kt(
            kiln_autograd::TapeOptions {
                detect_anomaly: config.detect_anomaly,
            },
            || {
                let to_err = |e: anyhow::Error| kiln_kt_bridge::BridgeError::new(format!("{e:#}"));
                // emb rows for tok_{1..T} — frozen embedding, plain index_select.
                let idx_t = kiln_tensor::Tensor::from_vec_on(
                    *device,
                    shifted_ids.clone(),
                    vec![seq_len - 1],
                )
                .map_err(|e| to_err(anyhow::anyhow!("mtp alignment: idx tensor: {e}")))?;
                let emb = weights
                    .embed_tokens
                    .index_select(&idx_t, 0)
                    .map_err(|e| to_err(anyhow::anyhow!("mtp alignment: emb select: {e}")))?;
                let norm_e = kiln_model::forward::rms_norm(
                    &emb.unsqueeze(0).map_err(|e| {
                        to_err(anyhow::anyhow!("mtp alignment: emb unsqueeze: {e}"))
                    })?,
                    &mtp.pre_fc_norm_embedding,
                    model_config.rms_norm_eps,
                )
                .map_err(to_err)?;
                let h_rows = hidden
                    .narrow(1, 0, seq_len - 1)
                    .map_err(|e| to_err(anyhow::anyhow!("mtp alignment: hidden narrow: {e}")))?;
                let norm_h = kiln_model::forward::rms_norm(
                    &h_rows,
                    &mtp.pre_fc_norm_hidden,
                    model_config.rms_norm_eps,
                )
                .map_err(to_err)?;
                let concat = kiln_tensor::ops::concat(&[&norm_e, &norm_h], 2)
                    .map_err(|e| to_err(anyhow::anyhow!("mtp alignment: concat: {e}")))?;
                let fc_t = mtp
                    .fc_t
                    .to_dtype(concat.dtype())
                    .map_err(|e| to_err(anyhow::anyhow!("mtp alignment: fc cast: {e}")))?;
                let fused = concat
                    .squeeze(0)
                    .and_then(|c2| c2.matmul(&fc_t))
                    .and_then(|f2| f2.unsqueeze(0))
                    .map_err(|e| to_err(anyhow::anyhow!("mtp alignment: fc matmul: {e}")))?;
                let block_out = kiln_model::forward::transformer_block_with_policy(
                    backend,
                    &fused,
                    &mtp.layer,
                    model_config,
                    &positions,
                    model_config.num_attention_heads,
                    model_config.num_kv_heads,
                    model_config.head_dim,
                    model_config.rotary_dim(),
                    &weights.rotary_inv_freq,
                    model_config.rms_norm_eps,
                    None,
                    0,
                    Some((&mtp_lora_view, lora_scale)),
                    streaming_prefill,
                )
                .map_err(to_err)?;
                let normed = kiln_model::forward::rms_norm(
                    &block_out,
                    &mtp.final_layernorm,
                    model_config.rms_norm_eps,
                )
                .map_err(to_err)?;
                let loss = kiln_autograd::with_active_tape(|tape| {
                    kiln_flce_kernel::fused_linear_cross_entropy_phase_b_unit_grad_via_kt_tape(
                        &normed,
                        &weights.embed_tokens_t,
                        &shifted_ids,
                        &shifted_mask,
                        DEFAULT_CHUNK_SIZE,
                        tape,
                    )
                })
                .ok_or_else(|| {
                    kiln_kt_bridge::BridgeError::new("mtp alignment: no active kt tape".to_string())
                })?
                .map_err(|e| {
                    kiln_kt_bridge::BridgeError::new(format!("mtp alignment FLCE: {e}"))
                })?;
                let loss_val = loss
                    .to_dtype(kiln_tensor::DType::F32)
                    .and_then(|t| t.to_scalar::<f32>())
                    .map_err(|e| {
                        kiln_kt_bridge::BridgeError::new(format!("mtp alignment loss read: {e}"))
                    })? as f64;
                Ok((loss_val, loss))
            },
        );
        let (loss_val, _loss_kt, grads_by_candle_raw) = match result {
            Ok(triple) => triple,
            Err(e) => anyhow::bail!("mtp alignment step failed: {e}"),
        };

        let mut grads = kiln_autograd::GradStore::new();
        for (key_raw, kt_grad) in grads_by_candle_raw {
            let Some(param_raw) =
                kiln_kt_bridge::tape_bridge::decode_kt_param_deposit(key_raw as u64)
            else {
                continue;
            };
            grads.insert(KtTensorId::from_raw(param_raw), kt_grad);
        }
        anyhow::ensure!(
            !grads.is_empty(),
            "mtp alignment: tape backward produced no MTP LoRA grads — the draft \
             block's lora-linear did not record (report this; the adapter would \
             silently ship an untrained draft head)"
        );

        optimizer_step_dispatch(
            backend,
            &mut mtp_train,
            &GradSource::Kt(grads),
            learning_rate,
            config.optimizer,
            opt_state.as_mut(),
        )?;

        if initial_ce.is_none() {
            initial_ce = Some(loss_val);
        }
        final_ce = Some(loss_val);
        trained += 1;
    }

    if let Some(state) = opt_state.as_ref() {
        state.evict_from_backend(backend);
    }

    // Return the trained pairs to params.mtp — save_peft serializes them
    // under the mtp.* keys.
    params.mtp = Some(mtp_train.layers.remove(0));

    tracing::info!(
        examples = trained,
        initial_ce = ?initial_ce,
        final_ce = ?final_ce,
        elapsed_ms = phase_started.elapsed().as_millis() as u64,
        "MTP alignment phase complete"
    );
    Ok(Some((trained, initial_ce, final_ce)))
}

/// Resolve the training device against the immutable runtime binding.
/// Production training requires the runtime device to match the resident
/// model-weight device exactly. In particular, CPU-host weights are not
/// promoted into the incomplete hybrid Vulkan training substrate; every
/// device mismatch fails closed before LoRA or optimizer allocation.
fn training_device_for_weights(
    weights: &GpuWeights,
    runtime: &crate::TrainingRuntimeContext,
) -> Result<Device> {
    runtime.resolve_device_for_weights(weights.embed_tokens.device())
}

/// Construct the backend named by the immutable training runtime.
///
/// `kiln-model` retains CPU-as-Vulkan autodetection for compatibility
/// inference. Training cannot use that shortcut: an explicit CPU runtime stays
/// CPU, and an accelerated runtime is accepted only when the resident weight
/// device matches it exactly.
pub(crate) fn training_backend_for_device(
    device: Device,
) -> Result<std::sync::Arc<dyn BackendRuntime>> {
    backend::for_explicit_device_kt(device)
        .with_context(|| format!("initialize exact native training backend for {device}"))
}

/// Confirm training will use the already resident serving weights.
///
/// Runtime device resolution rejects mismatches before this point. Keep this
/// second gate at the former upload boundary so a future bypass cannot silently
/// start an unqualified multi-GiB full-model copy.
fn resident_training_weights(
    weights: &GpuWeights,
    training_device: &Device,
) -> Result<Option<GpuWeights>> {
    if weights.embed_tokens.device() == *training_device {
        return Ok(None);
    }
    if weights.embed_tokens.device() == Device::Cpu && matches!(training_device, Device::Vulkan(_))
    {
        anyhow::bail!(
            "native Vulkan training is unavailable for CPU-host serving weights: the full-model resident Vulkan training substrate is not production-qualified"
        );
    }
    anyhow::bail!(
        "training device {} does not match resident model weight device {}; full-model training uploads are disabled",
        training_device.short_name(),
        weights.embed_tokens.device().short_name(),
    )
}

/// Per-step GPU coordination that remains interruptible by the serving
/// backend's process-lifetime quarantine latch.
///
/// A quarantined inference request may intentionally retain its read owner
/// because dropping unknown device state is unsafe. A bare blocking write
/// would therefore strand SFT forever between steps. Polling acquisition lets
/// the trainer return the quarantine error while preserving that owner.
#[derive(Clone)]
pub struct GpuStepCoordination {
    lock: std::sync::Arc<tokio::sync::RwLock<()>>,
    backend_health: kiln_model::BackendHealthHandle,
}

impl GpuStepCoordination {
    pub fn new(
        lock: std::sync::Arc<tokio::sync::RwLock<()>>,
        backend_health: kiln_model::BackendHealthHandle,
    ) -> Self {
        Self {
            lock,
            backend_health,
        }
    }

    fn blocking_write(&self) -> Result<tokio::sync::OwnedRwLockWriteGuard<()>> {
        loop {
            self.backend_health.ensure_healthy()?;
            if let Ok(guard) = self.lock.clone().try_write_owned() {
                self.backend_health.ensure_healthy()?;
                return Ok(guard);
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    fn blocking_gpu_phase<T>(
        &self,
        backend: &dyn BackendRuntime,
        workload: &'static str,
        phase: &'static str,
        operation: impl FnOnce() -> Result<T>,
    ) -> Result<CoordinatedGpuPhase<T>> {
        let wait_started = Instant::now();
        let guard = self
            .blocking_write()
            .with_context(|| format!("acquire healthy backend for {workload} {phase}"))?;
        let wait_ms = wait_started.elapsed().as_secs_f64() * 1000.0;

        let held_started = Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation));
        let sync_result = ExternalYieldBackend::runtime_synchronize_external_yield(backend)
            .with_context(|| format!("synchronize backend after {workload} {phase}"));
        let held_ms = held_started.elapsed().as_secs_f64() * 1000.0;
        match (&result, &sync_result) {
            (Err(_), settlement) => {
                let sync_suffix = settlement
                    .as_ref()
                    .err()
                    .map(|error| format!("; settlement also failed: {error:#}"))
                    .unwrap_or_default();
                self.backend_health
                    .quarantine(format!("{workload} {phase} panicked{sync_suffix}"));
            }
            (Ok(_), Err(sync_error)) => self.backend_health.quarantine(format!(
                "{workload} {phase} external-yield synchronization failed: {sync_error:#}"
            )),
            (Ok(_), Ok(())) => {}
        }
        drop(guard);
        tracing::debug!(
            workload,
            phase,
            wait_ms,
            held_ms,
            "completed coordinated training GPU phase"
        );

        match result {
            Ok(operation_result) => match sync_result {
                Ok(()) => operation_result.map(|value| CoordinatedGpuPhase {
                    value,
                    wait_ms,
                    held_ms,
                }),
                Err(sync_error) => match operation_result {
                    Ok(_) => Err(sync_error),
                    Err(operation_error) => Err(anyhow::anyhow!(
                        "{workload} {phase} failed ({operation_error:#}) and backend settlement also failed ({sync_error:#})"
                    )),
                },
            },
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    /// Run one bounded training GPU phase, settle the backend before releasing
    /// serving ownership, and quarantine the process if settlement is unknown.
    pub fn run_gpu_phase<T>(
        &self,
        backend: &dyn BackendRuntime,
        workload: &'static str,
        phase: &'static str,
        operation: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        self.blocking_gpu_phase(backend, workload, phase, operation)
            .map(|outcome| outcome.value)
    }
}

struct CoordinatedGpuPhase<T> {
    value: T,
    wait_ms: f64,
    held_ms: f64,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct GrpoGpuWriterTimings {
    wait_ms: f64,
    held_ms: f64,
    acquisitions: u64,
}

impl GrpoGpuWriterTimings {
    fn apply_to(&self, timings: &mut GrpoBenchmarkTimings) {
        timings.gpu_writer_wait_ms = self.wait_ms;
        timings.gpu_writer_held_ms = self.held_ms;
        timings.gpu_writer_acquisitions = self.acquisitions;
    }
}

fn run_coordinated_grpo_gpu_phase<T>(
    coordination: Option<&GpuStepCoordination>,
    backend: &dyn BackendRuntime,
    timings: &mut GrpoGpuWriterTimings,
    phase: &'static str,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let Some(coordination) = coordination else {
        return operation();
    };

    let outcome = coordination.blocking_gpu_phase(backend, "GRPO", phase, operation)?;
    timings.wait_ms += outcome.wait_ms;
    timings.held_ms += outcome.held_ms;
    timings.acquisitions = timings.acquisitions.saturating_add(1);
    Ok(outcome.value)
}

pub fn sft_train(
    examples: &[SftExample],
    config: &SftConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    replay_ctx: Option<ReplayContext>,
    gpu_step_coordination: Option<GpuStepCoordination>,
) -> Result<PathBuf> {
    sft_train_to(
        examples,
        config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        adapter_dir,
        adapter_name,
        progress_cb,
        replay_ctx,
        gpu_step_coordination,
    )
}

/// Train against adapters in `adapter_dir` while writing all new artifacts to
/// `output_adapter_dir`. The server uses this to keep an in-progress rewrite
/// invisible until its revision-barrier commit.
#[allow(clippy::too_many_arguments)]
pub fn sft_train_to(
    examples: &[SftExample],
    config: &SftConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    output_adapter_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    replay_ctx: Option<ReplayContext>,
    gpu_step_coordination: Option<GpuStepCoordination>,
) -> Result<PathBuf> {
    sft_train_to_with_checkpoint_root(
        examples,
        config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        output_adapter_dir,
        output_adapter_dir,
        adapter_name,
        progress_cb,
        replay_ctx,
        gpu_step_coordination,
    )
}

/// Staged-output SFT with a separate durable checkpoint root. Server training
/// uses this entry point so a process crash cannot discard already-published
/// resumable checkpoints with the temporary final-adapter staging tree.
#[allow(clippy::too_many_arguments)]
pub fn sft_train_to_with_checkpoint_root(
    examples: &[SftExample],
    config: &SftConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    output_adapter_dir: &Path,
    checkpoint_output_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    replay_ctx: Option<ReplayContext>,
    gpu_step_coordination: Option<GpuStepCoordination>,
) -> Result<PathBuf> {
    config
        .validate_native_contract()
        .context("validate native SFT profile before row admission")?;
    ensure_training_optimizer_device_supported(
        "SFT",
        weights,
        weights.embed_tokens.device(),
        config.optimizer,
        config.lora_rank,
    )?;
    let runtime =
        crate::standalone_training_runtime_for_weight_device(weights.embed_tokens.device())?;
    ensure_training_optimizer_entry_supported(
        "SFT",
        weights,
        &runtime,
        config.optimizer,
        config.lora_rank,
    )?;
    let prepared = crate::sft_ingestion::prepare_sft_examples(
        examples.iter().cloned(),
        tokenizer,
        config.invalid_row_policy,
        "rust_api",
        None,
    )?;
    sft_train_to_with_checkpoint_root_and_ingestion_with_runtime(
        &prepared.examples,
        &prepared.ingestion,
        config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        output_adapter_dir,
        checkpoint_output_dir,
        adapter_name,
        progress_cb,
        replay_ctx,
        gpu_step_coordination,
        &runtime,
    )
}

/// Standalone convenience wrapper for already-admitted SFT rows.
///
/// Server callers should use
/// [`sft_train_to_with_checkpoint_root_and_ingestion_with_runtime`] so the run
/// remains bound to their process-lifetime memory configuration.
#[allow(clippy::too_many_arguments)]
pub fn sft_train_to_with_checkpoint_root_and_ingestion(
    examples: &[SftExample],
    ingestion: &crate::sft_ingestion::SftIngestionReceipt,
    config: &SftConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    output_adapter_dir: &Path,
    checkpoint_output_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    replay_ctx: Option<ReplayContext>,
    gpu_step_coordination: Option<GpuStepCoordination>,
) -> Result<PathBuf> {
    ensure_training_optimizer_device_supported(
        "SFT",
        weights,
        weights.embed_tokens.device(),
        config.optimizer,
        config.lora_rank,
    )?;
    let runtime =
        crate::standalone_training_runtime_for_weight_device(weights.embed_tokens.device())?;
    sft_train_to_with_checkpoint_root_and_ingestion_with_runtime(
        examples,
        ingestion,
        config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        output_adapter_dir,
        checkpoint_output_dir,
        adapter_name,
        progress_cb,
        replay_ctx,
        gpu_step_coordination,
        &runtime,
    )
}

/// Server-owned SFT entry point with immutable process-lifetime runtime inputs.
#[allow(clippy::too_many_arguments)]
pub fn sft_train_to_with_checkpoint_root_and_ingestion_with_runtime(
    examples: &[SftExample],
    ingestion: &crate::sft_ingestion::SftIngestionReceipt,
    config: &SftConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    output_adapter_dir: &Path,
    checkpoint_output_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    replay_ctx: Option<ReplayContext>,
    gpu_step_coordination: Option<GpuStepCoordination>,
    runtime: &crate::TrainingRuntimeContext,
) -> Result<PathBuf> {
    let runtime_device = ensure_training_optimizer_entry_supported(
        "SFT",
        weights,
        runtime,
        config.optimizer,
        config.lora_rank,
    )?;
    crate::ensure_memory_governor_for_runtime(runtime_device, runtime)
        .context("initialize SFT memory governor")?;
    config
        .validate_native_contract()
        .context("validate native SFT profile")?;
    anyhow::ensure!(
        ingestion.invalid_row_policy == config.invalid_row_policy,
        "SFT ingestion policy {} differs from trainer config {}",
        ingestion.invalid_row_policy,
        config.invalid_row_policy
    );
    crate::sft_ingestion::verify_prepared_sft_examples(examples, tokenizer, ingestion)
        .context("verify admitted SFT rows before training")?;
    sft_train_prepared_to_with_checkpoint_root(
        examples,
        ingestion,
        config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        output_adapter_dir,
        checkpoint_output_dir,
        adapter_name,
        progress_cb,
        replay_ctx,
        gpu_step_coordination,
        runtime,
    )
}

#[allow(clippy::too_many_arguments)]
fn sft_train_prepared_to_with_checkpoint_root(
    examples: &[SftExample],
    ingestion: &crate::sft_ingestion::SftIngestionReceipt,
    config: &SftConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    output_adapter_dir: &Path,
    checkpoint_output_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    replay_ctx: Option<ReplayContext>,
    gpu_step_coordination: Option<GpuStepCoordination>,
    runtime: &crate::TrainingRuntimeContext,
) -> Result<PathBuf> {
    let run_started = Instant::now();
    anyhow::ensure!(
        config.checkpoint_interval != Some(0),
        "SFT checkpoint_interval must be greater than zero"
    );
    let output_dir = output_adapter_dir.join(adapter_name);
    let training_data_sha256 = Some(ingestion.kept_corpus_sha256.clone());
    let ingestion_receipt_sha256 = crate::train_receipt::sha256_json_serializable(ingestion)
        .context("hash SFT ingestion receipt")?;
    let mut data_stats = crate::train_receipt::DataStatsReceipt {
        examples_read: ingestion.rows_read,
        examples_filtered: ingestion.rows_rejected,
        sft_ingestion: Some(ingestion.clone()),
        ..Default::default()
    };
    let mut token_counts = crate::train_receipt::TokenCountReceipt::default();
    let mut lora_grad_norms = crate::train_receipt::LoraGradNormAccumulator::default();
    let requested_base_adapter_dir = config.base_adapter.as_deref().map(|name| {
        resolve_base_adapter_dir_from_roots(name, adapter_dir, output_adapter_dir, adapter_name)
    });

    // (#1082) `embed_tokens.device()` is a kt Device; the SFT path is now
    // kt-native end-to-end (kt `Parameter`s, kt AdamW state, kt tape
    // forward/backward), so keep `device` kt downstream. The only candle
    // touch left is the safetensors adapter I/O, which bridges the kt device
    // to candle locally inside `load_from_safetensors`/`save_peft`.
    let device = training_device_for_weights(weights, runtime)?;
    let backend = training_backend_for_device(device)?;
    let training_precision_policy = training_precision_policy_for_backend(backend.as_ref());
    ensure_training_optimizer_supported(
        "SFT",
        backend.as_ref(),
        config.optimizer,
        weights.embed_tokens.dtype(),
        config.lora_rank,
    )?;
    let backend_loss_route = TrainingLossBackend::runtime_sft_flce_loss_route(backend.as_ref());
    let sft_loss_route = runtime
        .admitted_sft_loss_route()
        .unwrap_or(backend_loss_route);
    anyhow::ensure!(
        sft_loss_route == backend_loss_route,
        "SFT loss route changed after admission: admitted `{}`, execution backend reports `{}`",
        sft_loss_route.as_str(),
        backend_loss_route.as_str(),
    );
    let bound_runtime = runtime.with_admitted_sft_loss_route(sft_loss_route);
    let runtime = &bound_runtime;
    // Training-session residency: upload a one-device copy of the weights
    // when the substrate needs it (Vulkan hybrid). Shadow `weights` so the
    // whole body trains against the resident copy; it drops at return. Route
    // drift has already failed closed before this potentially large copy.
    let resident_weights = resident_training_weights(weights, &device)?;
    let weights = resident_weights.as_ref().unwrap_or(weights);
    let checkpoint_boundary_policy = runtime.checkpoint_boundary_policy();
    let streaming_prefill = runtime.resolved_streaming_prefill_policy(device);
    let training_runtime_planning_identity =
        runtime.checkpoint_planning_identity_for_device(device);

    let learning_rate = config.effective_learning_rate();
    if let Some(explicit) = config.learning_rate {
        if let Some(warning) = crate::learning_rate_band_warning(
            explicit,
            crate::resolve_learning_rate(&config.optimizer, crate::TrainMode::Sft),
        ) {
            tracing::warn!(optimizer = ?config.optimizer, "SFT {warning}");
        }
    }

    tracing::info!(
        num_examples = examples.len(),
        training_profile = %config.training_profile,
        epochs = config.epochs,
        lr = learning_rate,
        rank = config.lora_rank,
        alpha = config.lora_alpha,
        adapter_name,
        "starting SFT training"
    );

    let resume_checkpoint = config
        .resume_checkpoint
        .as_deref()
        .map(Path::new)
        .map(crate::checkpoint::load_training_checkpoint)
        .transpose()
        .context("load SFT resume checkpoint")?;
    if config.checkpoint_interval.is_some() || resume_checkpoint.is_some() {
        validate_exact_training_provenance(weights)?;
    }
    let resume_init_seed = resume_checkpoint
        .as_ref()
        .map(|checkpoint| {
            let state = checkpoint
                .manifest
                .rng_states
                .get("lora-init")
                .context("SFT resume checkpoint has no lora-init RNG state")?;
            anyhow::ensure!(
                state.algorithm == "kiln.seeded-lora-init.v1" && state.position == 0,
                "unsupported SFT lora-init RNG state"
            );
            Ok(state.seed)
        })
        .transpose()?;
    if let (Some(requested), Some(restored)) = (config.seed, resume_init_seed) {
        anyhow::ensure!(
            requested == restored,
            "SFT resume seed {restored} differs from requested seed {requested}"
        );
    }
    let requested_effective_seed = resume_init_seed.or(config.seed);

    let alpha_over_rank = match crate::lora_scaling::validate_lora_scaling(
        config.lora_rank,
        config.lora_alpha,
        config.allow_high_lora_scale,
    ) {
        Ok(value) => value,
        Err(err) => {
            let message = format!("{err:#}");
            write_sft_train_receipt_best_effort(
                adapter_name,
                model_config,
                tokenizer,
                weights.base_weight_shard_manifest.as_ref(),
                weights.execution_provenance.as_ref(),
                None,
                sft_loss_route,
                config,
                config.seed,
                None,
                requested_base_adapter_dir.as_deref(),
                &output_dir,
                training_data_sha256,
                ingestion,
                data_stats,
                token_counts,
                run_started.elapsed().as_millis() as u64,
                Vec::new(),
                None,
                Some(message),
            );
            return Err(crate::train_receipt::annotate_training_error(err));
        }
    };
    tracing::info!(
        alpha_over_rank,
        allow_high_lora_scale = config.allow_high_lora_scale,
        "validated LoRA scaling"
    );

    // Open replay state (writes request record + lineage.json *before* the
    // optimizer step, so a crash mid-step still leaves a recoverable trail)
    // and resolve the effective seed.
    let replay_parent_adapter = resume_checkpoint
        .is_none()
        .then_some(config.base_adapter.as_deref())
        .flatten();
    let (replay_state, effective_seed) = match replay_ctx.as_ref() {
        Some(ctx) => {
            let (state, seed) = open_replay_state_to(
                ctx,
                requested_effective_seed,
                replay_parent_adapter,
                adapter_dir,
                output_adapter_dir,
                adapter_name,
            )?;
            (Some(state), Some(seed))
        }
        None => (
            None,
            Some(requested_effective_seed.unwrap_or_else(rand::random)),
        ),
    };
    let effective_seed_value = effective_seed.expect("SFT always resolves an effective seed");
    let effective_checkpoint_config =
        sft_checkpoint_effective_config(config, learning_rate, effective_seed_value)?;
    let training_data_checkpoint_sha256 =
        checkpoint_sha256_hex(training_data_sha256.as_deref(), "SFT training data")?;
    let resume_loop_state = resume_checkpoint
        .as_ref()
        .map(load_sft_checkpoint_loop_state)
        .transpose()?;
    if let Some(checkpoint) = resume_checkpoint.as_ref() {
        anyhow::ensure!(
            checkpoint.manifest.training_kind == crate::checkpoint::TrainingKind::Sft,
            "resume checkpoint is not an SFT checkpoint"
        );
        anyhow::ensure!(
            checkpoint.manifest.adapter_name == adapter_name,
            "resume checkpoint adapter {:?} does not match {:?}",
            checkpoint.manifest.adapter_name,
            adapter_name
        );
        anyhow::ensure!(
            checkpoint.manifest.effective_config == effective_checkpoint_config,
            "resume checkpoint effective SFT configuration differs from this request: checkpoint={}, request={}",
            checkpoint.manifest.effective_config,
            effective_checkpoint_config
        );
        anyhow::ensure!(
            checkpoint.manifest.data.content_sha256 == training_data_checkpoint_sha256,
            "resume checkpoint training data hash differs from this request"
        );
    }

    let base_adapter_result = if resume_checkpoint.is_some() {
        Ok(None)
    } else {
        resolve_and_validate_base_adapter_from_roots(
            config.base_adapter.as_deref(),
            adapter_dir,
            output_adapter_dir,
            adapter_name,
            model_config,
            config.lora_rank,
            config.allow_adapter_shape_conversion,
        )
    };
    let base_adapter_dir = match base_adapter_result {
        Ok(value) => value,
        Err(err) => {
            let message = format!("{err:#}");
            write_sft_train_receipt_best_effort(
                adapter_name,
                model_config,
                tokenizer,
                weights.base_weight_shard_manifest.as_ref(),
                weights.execution_provenance.as_ref(),
                None,
                sft_loss_route,
                config,
                effective_seed,
                Some(alpha_over_rank),
                requested_base_adapter_dir.as_deref(),
                &output_dir,
                training_data_sha256,
                ingestion,
                data_stats,
                token_counts,
                run_started.elapsed().as_millis() as u64,
                Vec::new(),
                None,
                Some(message),
            );
            return Err(crate::train_receipt::annotate_training_error(err));
        }
    };

    // Initialize trainable LoRA parameters
    let mut params = TrainableLoraParams::initialize_seeded_with_precision_policy(
        model_config,
        weights,
        config.lora_rank,
        config.lora_alpha,
        &device,
        effective_seed,
        training_precision_policy,
    )?;

    tracing::info!(
        num_vars = params.all_params().len(),
        "initialized trainable LoRA parameters"
    );
    let lora_grad_index = LoraGradNormIndex::new(&params);

    if let Some(checkpoint) = resume_checkpoint.as_ref() {
        let adapter_path =
            checkpoint.artifact_path(&checkpoint.manifest.state_files.adapter_parameters)?;
        params.load_checkpoint_parameters(&adapter_path)?;
        tracing::info!(
            checkpoint = %checkpoint.root.display(),
            step = checkpoint.manifest.progress.global_step,
            "restored exact SFT adapter parameters"
        );
    } else if let Some(base_dir) = base_adapter_dir.as_deref() {
        let n_loaded = params.load_from_safetensors(base_dir, &device)?;
        tracing::info!(
            base = %base_dir.display(),
            num_tensors = n_loaded,
            "loaded base adapter — continuing SFT from those weights"
        );
    }

    // Allocate AdamW state if selected; SGD has no per-param state.
    // Register the per-param `m`/`v` device moment tensors alongside the
    // LoRA params so the on-device AdamW kernel's
    // `has_resident_activation(m/v)` gate passes (C1 fix — without this the
    // device path declines and a no-op interim corrupted the param).
    let mut opt_state = make_opt_state(&params, config.optimizer, learning_rate, &device)?;
    if let Some(checkpoint) = resume_checkpoint.as_ref() {
        let state_path = checkpoint
            .manifest
            .state_files
            .optimizer_state
            .as_deref()
            .map(|relative| checkpoint.artifact_path(relative))
            .transpose()?;
        match (opt_state.as_mut(), state_path) {
            (Some(state), Some(path)) => {
                let step = u32::try_from(checkpoint.manifest.progress.global_step)
                    .context("SFT resume optimizer step exceeds u32")?;
                state.load_checkpoint_state(&params, &path, step)?;
            }
            (None, None) => {}
            (Some(_), None) => {
                anyhow::bail!("stateful SFT optimizer checkpoint has no optimizer artifact")
            }
            (None, Some(_)) => {
                anyhow::bail!("SGD SFT checkpoint unexpectedly contains optimizer state")
            }
        }
    }

    // Register only after checkpoint restoration. Registry identity is
    // process-local, so loading into already-registered tensors would leave
    // the restored host object and the resident device object out of sync.
    params.register_with_backend(&*backend)?;
    if let Some(state) = opt_state.as_ref() {
        state.register_with_backend(&*backend)?;
    }

    // Run the actual training body inside a closure so we can write the
    // outcome record (success or failure) before returning to the caller.
    let mut train_body = || -> Result<(PathBuf, f64)> {
        // Validate examples without retaining every tokenized long-context
        // payload at once. The step loop tokenizes the current example on
        // demand so full-file SFT jobs don't pin all input_ids/label masks for
        // the entire run.
        let mut valid_indices = Vec::new();
        let mut one_epoch_counts = crate::train_receipt::TokenCountReceipt::default();
        let mut max_seq_len_tokens: usize = 0;
        let mut valid_seq_lens = Vec::new();
        for (idx, ex) in examples.iter().enumerate() {
            match tokenize_for_training(ex, tokenizer) {
                Ok((input_ids, label_mask)) => {
                    let action_tokens = label_mask.iter().filter(|&&mask| mask).count() as u64;
                    one_epoch_counts.action_tokens =
                        one_epoch_counts.action_tokens.saturating_add(action_tokens);
                    one_epoch_counts.context_tokens =
                        one_epoch_counts.context_tokens.saturating_add(
                            input_ids.len().saturating_sub(action_tokens as usize) as u64,
                        );
                    if input_ids.len() > max_seq_len_tokens {
                        max_seq_len_tokens = input_ids.len();
                    }
                    valid_indices.push(idx);
                    valid_seq_lens.push(input_ids.len());
                }
                Err(e) => anyhow::bail!(
                    "admitted SFT row {} failed repeat tokenization before training: {e:#}",
                    idx + 1
                ),
            }
        }

        if valid_indices.is_empty() {
            anyhow::bail!("no valid training examples after tokenization");
        }
        anyhow::ensure!(
            valid_indices.len() == examples.len(),
            "not every admitted SFT row reached the training set"
        );
        data_stats.examples_trained = valid_indices.len().saturating_mul(config.epochs);
        token_counts.action_tokens = one_epoch_counts
            .action_tokens
            .saturating_mul(config.epochs as u64);
        token_counts.env_tokens = 0;
        token_counts.context_tokens = one_epoch_counts
            .context_tokens
            .saturating_mul(config.epochs as u64);

        // Resolve checkpointing at each step using the current example's
        // actual sequence length. The server preflight stamps the maximum
        // segment count needed for admission; treating that as a job-wide
        // fixed value makes a single very long row slow down every shorter
        // row in the same upload.
        let activation_bytes_per_elem = training_activation_bytes_per_elem_for_policy(
            weights,
            training_precision_policy,
            model_config_has_linear_attention(model_config),
        );
        tracing::info!(
            max_seq_len_tokens,
            preflight_max_segments = ?config.grad_checkpoint_segments,
            activation_bytes_per_elem,
            "SFT gradient checkpointing will resolve per example"
        );

        let total_steps = config
            .epochs
            .checked_mul(valid_indices.len())
            .context("SFT optimizer-step count overflow")?;
        let shuffle_seed = match resume_checkpoint.as_ref() {
            Some(checkpoint) => {
                let state = checkpoint
                    .manifest
                    .rng_states
                    .get("epoch-order")
                    .context("SFT resume checkpoint has no epoch-order RNG state")?;
                anyhow::ensure!(
                    state.algorithm == "kiln.epoch-order.v1" && state.state_file.is_none(),
                    "unsupported SFT epoch-order RNG state"
                );
                state.seed
            }
            None => effective_seed_value,
        };
        let mut has_checkpointed_step = false;
        let gradient_checkpoint_plan: Vec<_> = valid_seq_lens
            .iter()
            .map(|&seq_len| {
                let config_for_step = checkpoint_config_for_training_step(
                    weights,
                    &device,
                    config.grad_checkpoint_segments,
                    model_config.num_layers,
                    seq_len,
                    model_config.hidden_size,
                    model_config.intermediate_size,
                    model_config.vocab_size,
                    2,
                    activation_bytes_per_elem,
                    runtime,
                );
                let boundaries = checkpoint_segments_for_config(
                    weights,
                    &device,
                    seq_len,
                    config_for_step,
                    streaming_prefill,
                );
                has_checkpointed_step |= boundaries.is_some();
                serde_json::json!({
                    "seq_len": seq_len,
                    "enabled": config_for_step.enabled,
                    "num_segments": config_for_step.num_segments,
                    "auto_configured": config_for_step.auto_configured,
                    "boundaries": boundaries,
                })
            })
            .collect();
        ensure_sft_loss_route_supports_checkpointing(sft_loss_route, has_checkpointed_step)?;
        let gradient_checkpoint_plan_sha256 =
            crate::train_receipt::sha256_json_serializable(&gradient_checkpoint_plan)
                .context("hash SFT gradient-checkpoint plan")?;
        let checkpoint_descriptor = SftCheckpointDescriptor {
            adapter_name: adapter_name.to_string(),
            effective_config: effective_checkpoint_config.clone(),
            precision_policy: training_checkpoint_precision(&params, opt_state.as_ref())?,
            data: crate::checkpoint::TrainingCheckpointData {
                source_kind: "sft-valid-example-order-v1".to_string(),
                content_sha256: training_data_checkpoint_sha256.clone(),
                item_count: valid_indices.len() as u64,
            },
            init_seed: effective_seed_value,
            shuffle_seed,
            optimizer: config.optimizer,
            learning_rate,
            total_steps,
            base_model_weights_sha256: weights.source_content_sha256.clone(),
            auxiliary_state: sft_checkpoint_auxiliary_state(
                model_config,
                tokenizer,
                training_precision_policy,
                &valid_indices,
                weights.source_content_sha256.as_deref(),
                weights.base_weight_shard_manifest.as_ref(),
                weights.execution_provenance.as_ref(),
                BackendIdentity::runtime_name(backend.as_ref()),
                &gradient_checkpoint_plan_sha256,
                &ingestion_receipt_sha256,
                &training_runtime_planning_identity,
            )?,
        };
        if let (Some(checkpoint), Some(loop_state)) =
            (resume_checkpoint.as_ref(), resume_loop_state.as_ref())
        {
            checkpoint_descriptor.validate_resume(checkpoint, loop_state)?;
        }

        let mut global_step = resume_loop_state
            .as_ref()
            .map_or(0, |state| state.global_step as usize);
        let mut loss_history = resume_loop_state
            .as_ref()
            .map_or_else(Vec::new, |state| state.loss_history.clone());
        let mut last_loss = resume_loop_state
            .as_ref()
            .map_or(0.0, |state| state.last_loss);
        let mut first_epoch_loss = resume_loop_state
            .as_ref()
            .and_then(|state| state.first_epoch_loss);
        let mut best_epoch_loss = resume_loop_state
            .as_ref()
            .and_then(|state| state.best_epoch_loss)
            .unwrap_or(f64::INFINITY);
        if let Some(state) = resume_loop_state.as_ref() {
            lora_grad_norms = state.lora_grad_norms.clone();
        }
        let start_epoch = resume_loop_state
            .as_ref()
            .map_or(0, |state| state.epoch_index as usize);
        let start_cursor = resume_loop_state
            .as_ref()
            .map_or(0, |state| state.cursor_in_epoch as usize);
        let mut last_ckpt_log_key: Option<(bool, usize)> = None;
        let mut last_saved_step = resume_loop_state
            .as_ref()
            .map(|state| state.global_step as usize);
        const SFT_DIVERGENCE_RATIO: f64 = 8.0;
        const SFT_DIVERGENCE_MIN_INCREASE: f64 = 5.0;

        let pb = make_step_progress(total_steps, "sft training");
        if let Some(pb) = &pb {
            pb.set_position(global_step as u64);
        }

        for epoch in start_epoch..config.epochs {
            let order = epoch_order(shuffle_seed, epoch, valid_indices.len());
            let cursor_start = (epoch == start_epoch).then_some(start_cursor).unwrap_or(0);
            let mut epoch_loss = if epoch == start_epoch {
                resume_loop_state
                    .as_ref()
                    .map_or(0.0, |state| state.current_epoch_loss_sum)
            } else {
                0.0
            };
            let mut epoch_items = if epoch == start_epoch {
                resume_loop_state
                    .as_ref()
                    .map_or(0, |state| state.current_epoch_items as usize)
            } else {
                0
            };
            anyhow::ensure!(
                cursor_start <= order.len() && epoch_items == cursor_start,
                "SFT resume cursor is outside the current epoch"
            );
            let mut checkpoint_after_epoch = false;

            for (cursor, &order_idx) in order.iter().enumerate().skip(cursor_start) {
                let ex_idx = valid_indices[order_idx];
                let (input_ids, label_mask) =
                    tokenize_for_training(&examples[ex_idx], tokenizer)
                        .with_context(|| format!("retokenize SFT example {ex_idx}"))?;
                let ckpt_config = checkpoint_config_for_training_step(
                    weights,
                    &device,
                    config.grad_checkpoint_segments,
                    model_config.num_layers,
                    input_ids.len(),
                    model_config.hidden_size,
                    model_config.intermediate_size,
                    model_config.vocab_size,
                    2, // BF16 base weights (canonical kiln inference dtype)
                    activation_bytes_per_elem,
                    runtime,
                );
                let segments = checkpoint_segments_for_config(
                    weights,
                    &device,
                    input_ids.len(),
                    ckpt_config,
                    streaming_prefill,
                );
                let ckpt_log_key = (ckpt_config.enabled, ckpt_config.num_segments);
                if last_ckpt_log_key != Some(ckpt_log_key) {
                    if let Some(ref segs) = segments {
                        tracing::info!(
                            seq_len = input_ids.len(),
                            num_segments = segs.len(),
                            preflight_max_segments = ?config.grad_checkpoint_segments,
                            boundaries = ?segs,
                            "SFT gradient checkpointing enabled for step shape"
                        );
                    } else {
                        tracing::info!(
                            seq_len = input_ids.len(),
                            preflight_max_segments = ?config.grad_checkpoint_segments,
                            "SFT gradient checkpointing disabled for step shape"
                        );
                    }
                    last_ckpt_log_key = Some(ckpt_log_key);
                }
                let loss_val;

                // Per-STEP GPU coordination (the state.rs contract the
                // job-long server guard violated): hold the write lock
                // only across this step's forward/backward/optimizer so
                // in-flight inference streams interleave between steps
                // instead of freezing mid-token for the whole job.
                // Tokenization above runs lock-free (CPU work).
                let _step_gpu = match gpu_step_coordination.as_ref() {
                    Some(coordination) => Some(
                        coordination
                            .blocking_write()
                            .context("acquire healthy backend for SFT step")?,
                    ),
                    None => None,
                };

                // (#1082 candle-drop) The SFT forward/backward is now UNCONDITIONALLY
                // kt tape-authoritative — the candle checkpointed reverse + candle
                // `loss.backward()` paths are deleted, and the candle FLCE provider
                // opt-in (KILN_CUDA_FLCE) is removed (FLCE is kt-native).
                // `standard_forward_backward` and
                // `checkpointed_forward_backward_tape_authoritative_kt` both return
                // `GradSource::Kt`, consumed kt-native by the dispatchers.
                let grads: GradSource = if let Some(ref segs) = segments {
                    #[cfg(any(
                        feature = "cuda",
                        feature = "metal",
                        feature = "vulkan",
                        feature = "rocm"
                    ))]
                    {
                        let (lv, kt_grads) = checkpointed_forward_backward_tape_authoritative_kt(
                            &*backend,
                            sft_loss_route,
                            &input_ids,
                            weights,
                            model_config,
                            &params,
                            &label_mask,
                            segs,
                            &device,
                            config.detect_anomaly,
                            checkpoint_boundary_policy,
                            streaming_prefill,
                        )?;
                        loss_val = lv;
                        GradSource::Kt(kt_grads)
                    }
                    #[cfg(not(any(
                        feature = "cuda",
                        feature = "metal",
                        feature = "vulkan",
                        feature = "rocm"
                    )))]
                    {
                        // Non-GPU build: the kt tape adapters don't record on a
                        // CPU candle device, so checkpointed kt-tape backward is a
                        // GPU-only path (CUDA/Metal/Vulkan). The CPU smoke test uses
                        // the non-checkpointed `standard_forward_backward` path;
                        // reaching here means a CPU run requested checkpointing, which
                        // the candle-drop endgame does not support yet.
                        let _ = (segs, checkpoint_boundary_policy);
                        anyhow::bail!(
                            "gradient checkpointing requires a GPU feature (`cuda`, \
                             `metal`, or `vulkan`); the kt-tape checkpointed reverse \
                             is GPU-only post candle-drop)"
                        );
                    }
                } else {
                    let (lv, g) = standard_forward_backward_with_policy_and_loss_route(
                        &*backend,
                        sft_loss_route,
                        &input_ids,
                        weights,
                        model_config,
                        &params,
                        &label_mask,
                        &device,
                        config.detect_anomaly,
                        streaming_prefill,
                    )?;
                    loss_val = lv;
                    g
                };
                anyhow::ensure!(
                    loss_val.is_finite(),
                    "SFT loss became non-finite at epoch {} step {}: {loss_val}",
                    epoch + 1,
                    global_step + 1
                );
                observe_lora_grad_norms_dispatch(&mut lora_grad_norms, &params, &grads)?;
                optimizer_step_dispatch(
                    &*backend,
                    &mut params,
                    &grads,
                    learning_rate,
                    config.optimizer,
                    opt_state.as_mut(),
                )?;
                drop(_step_gpu);

                epoch_loss += loss_val;
                epoch_items += 1;
                last_loss = loss_val;
                loss_history.push(loss_val);

                global_step += 1;

                let checkpoint_due = config.checkpoint_interval.is_some_and(|interval| {
                    interval > 0 && global_step % interval == 0 && global_step < total_steps
                });
                if checkpoint_due {
                    if cursor + 1 == order.len() {
                        checkpoint_after_epoch = true;
                    } else {
                        let loop_state = SftCheckpointLoopState::capture(
                            global_step,
                            epoch,
                            cursor + 1,
                            &loss_history,
                            last_loss,
                            epoch_loss,
                            epoch_items,
                            first_epoch_loss,
                            best_epoch_loss,
                            &lora_grad_norms,
                        );
                        let path = checkpoint_descriptor.save(
                            checkpoint_output_dir,
                            &*backend,
                            &mut params,
                            &mut opt_state,
                            epoch,
                            cursor + 1,
                            &order,
                            &loop_state,
                            gpu_step_coordination.as_ref(),
                        )?;
                        last_saved_step = Some(global_step);
                        tracing::info!(
                            step = global_step,
                            checkpoint = %path.display(),
                            "saved resumable SFT checkpoint"
                        );
                    }
                }

                if let Some(ref cb) = progress_cb {
                    let control = cb(TrainingProgress {
                        epoch: epoch + 1,
                        total_epochs: config.epochs,
                        step: global_step,
                        total_steps,
                        loss: loss_val,
                        progress: global_step as f32 / total_steps as f32,
                    });
                    if control == TrainControl::Stop && global_step < total_steps {
                        if last_saved_step != Some(global_step) {
                            let loop_state = SftCheckpointLoopState::capture(
                                global_step,
                                epoch,
                                cursor + 1,
                                &loss_history,
                                last_loss,
                                epoch_loss,
                                epoch_items,
                                first_epoch_loss,
                                best_epoch_loss,
                                &lora_grad_norms,
                            );
                            let path = checkpoint_descriptor.save(
                                checkpoint_output_dir,
                                &*backend,
                                &mut params,
                                &mut opt_state,
                                epoch,
                                cursor + 1,
                                &order,
                                &loop_state,
                                gpu_step_coordination.as_ref(),
                            )?;
                            tracing::info!(
                                step = global_step,
                                checkpoint = %path.display(),
                                "saved resumable SFT checkpoint before cancellation"
                            );
                        }
                        anyhow::bail!(
                            "training cancelled by user (stop requested at step boundary)"
                        );
                    }
                }

                if global_step % 10 == 0 || global_step == total_steps {
                    tracing::info!(
                        epoch = epoch + 1,
                        step = global_step,
                        total_steps,
                        loss = format!("{loss_val:.6}"),
                        "training step"
                    );
                }

                if let Some(pb) = &pb {
                    pb.set_message(format!("{loss_val:.6}"));
                    pb.inc(1);
                }
            }

            anyhow::ensure!(
                epoch_items == valid_indices.len(),
                "SFT epoch {} completed with {epoch_items} items, expected {}",
                epoch + 1,
                valid_indices.len()
            );
            let avg_loss = epoch_loss / epoch_items as f64;
            anyhow::ensure!(
                avg_loss.is_finite(),
                "SFT epoch {} average loss became non-finite: {avg_loss}",
                epoch + 1
            );
            let first_loss = *first_epoch_loss.get_or_insert(avg_loss);
            if epoch > 0
                && avg_loss > first_loss * SFT_DIVERGENCE_RATIO
                && avg_loss - best_epoch_loss > SFT_DIVERGENCE_MIN_INCREASE
            {
                anyhow::bail!(
                    "SFT loss diverged at epoch {}: avg_loss={avg_loss:.6}, \
                     first_epoch_loss={first_loss:.6}, best_epoch_loss={best_epoch_loss:.6}",
                    epoch + 1
                );
            }
            best_epoch_loss = best_epoch_loss.min(avg_loss);
            tracing::info!(
                epoch = epoch + 1,
                avg_loss = format!("{avg_loss:.6}"),
                "epoch complete"
            );
            if checkpoint_after_epoch && global_step < total_steps {
                let next_epoch = epoch + 1;
                let next_order = epoch_order(shuffle_seed, next_epoch, valid_indices.len());
                let loop_state = SftCheckpointLoopState::capture(
                    global_step,
                    next_epoch,
                    0,
                    &loss_history,
                    last_loss,
                    0.0,
                    0,
                    first_epoch_loss,
                    best_epoch_loss,
                    &lora_grad_norms,
                );
                let path = checkpoint_descriptor.save(
                    checkpoint_output_dir,
                    &*backend,
                    &mut params,
                    &mut opt_state,
                    next_epoch,
                    0,
                    &next_order,
                    &loop_state,
                    gpu_step_coordination.as_ref(),
                )?;
                last_saved_step = Some(global_step);
                tracing::info!(
                    step = global_step,
                    checkpoint = %path.display(),
                    "saved resumable SFT checkpoint at epoch boundary"
                );
            }
        }

        if let Some(pb) = pb {
            pb.finish_and_clear();
        }
        anyhow::ensure!(
            global_step == total_steps && loss_history.len() == total_steps,
            "SFT loop completed with inconsistent progress ({global_step}/{total_steps}, {} losses)",
            loss_history.len()
        );

        // MTP alignment phase (PR-B): train the native draft block's LoRA
        // against the freshly-tuned model so speculative decoding keeps its
        // acceptance rate under this adapter. Auto-on when the checkpoint
        // ships mtp.* tensors; config.train_mtp = false opts out. Soft-fail:
        // a draft-head alignment problem must not lose the main adapter.
        #[cfg(any(
            feature = "cuda",
            feature = "metal",
            feature = "vulkan",
            feature = "rocm"
        ))]
        match run_mtp_alignment_phase(
            &*backend,
            weights,
            model_config,
            &mut params,
            examples,
            &valid_indices,
            tokenizer,
            config,
            &device,
            streaming_prefill,
        ) {
            Ok(Some((mtp_examples, mtp_initial_ce, mtp_final_ce))) => {
                tracing::info!(
                    examples = mtp_examples,
                    initial_ce = ?mtp_initial_ce,
                    final_ce = ?mtp_final_ce,
                    "MTP draft-block LoRA trained alongside the adapter"
                );
            }
            Ok(None) => {}
            Err(e) => {
                tracing::warn!(
                    error = %format!("{e:#}"),
                    "MTP alignment phase failed — saving the adapter WITHOUT \
                     a trained draft head (spec decode falls back to the base \
                     draft for this adapter)"
                );
                params.mtp = None;
            }
        }

        // Pull current Var values from registry into candle CPU
        // storage before final save_peft (the on-device optimizer
        // path leaves candle storage stale between steps).
        let final_snapshot_wait_started = Instant::now();
        let final_snapshot_gpu = gpu_step_coordination
            .as_ref()
            .map(GpuStepCoordination::blocking_write)
            .transpose()
            .context("acquire healthy backend for final SFT adapter snapshot")?;
        let final_snapshot_gpu_wait_ms = final_snapshot_wait_started.elapsed().as_millis() as u64;
        let final_snapshot_started = Instant::now();
        let synced = params
            .sync_to_master(&*backend)
            .context("capture final SFT adapter state from resident backend")?;
        let final_device_snapshot_ms = final_snapshot_started.elapsed().as_millis() as u64;
        drop(final_snapshot_gpu);
        tracing::info!(
            synced,
            final_snapshot_gpu_wait_ms,
            final_device_snapshot_ms,
            "captured final SFT adapter state before publication"
        );

        // Safetensors/config/receipt I/O consumes only the captured master
        // state and therefore cannot hold serving behind the GPU writer.
        params.save_peft(&output_dir, model_config.num_layers)?;

        tracing::info!(
            adapter = adapter_name,
            path = %output_dir.display(),
            final_loss = format!("{last_loss:.6}"),
            "SFT training complete"
        );

        Ok((output_dir.clone(), last_loss))
    };

    let result = train_body();
    drop(train_body);
    let adapter_smoke_test = if config.adapter_smoke_test && result.is_ok() {
        Some(run_adapter_smoke_test_best_effort(
            adapter_name,
            &*backend,
            weights,
            model_config,
            tokenizer,
            &params,
            config.adapter_smoke_prompts.as_deref(),
            streaming_prefill,
        ))
    } else {
        None
    };
    // Phase 4.1 cleanup: evict the LoRA Vars from the registry so a
    // long-running server doesn't accumulate stale entries from past
    // training jobs (each job creates fresh Vars with new TensorIds).
    // The eviction happens regardless of whether training succeeded
    // or failed.
    if let Some(state) = opt_state.as_ref() {
        state.evict_from_backend(&*backend);
    }
    params.evict_from_backend(&*backend);
    if let Some(state) = replay_state {
        let outcome = match &result {
            Ok((_, loss)) => Ok(*loss),
            Err(e) => Err(format!("{e:#}")),
        };
        if let Err(e) = close_replay_state(state, outcome) {
            tracing::warn!(error = %e, "failed to append SFT replay outcome record");
        }
    }
    let status_error = result.as_ref().err().map(|err| format!("{err:#}"));
    write_sft_train_receipt_best_effort(
        adapter_name,
        model_config,
        tokenizer,
        weights.base_weight_shard_manifest.as_ref(),
        weights.execution_provenance.as_ref(),
        training_precision_for_receipt_best_effort(&params, opt_state.as_ref()),
        sft_loss_route,
        config,
        effective_seed,
        Some(alpha_over_rank),
        base_adapter_dir.as_deref(),
        &output_dir,
        training_data_sha256,
        ingestion,
        data_stats,
        token_counts,
        run_started.elapsed().as_millis() as u64,
        lora_grad_norms.finish(),
        adapter_smoke_test,
        status_error,
    );
    result
        .map(|(dir, _)| dir)
        .map_err(crate::train_receipt::annotate_training_error)
}

/// Run GRPO training on the provided groups using the already-loaded model.
///
/// GRPO (Group Relative Policy Optimization) trains LoRA adapters by:
/// 1. Computing log-probs under the current policy (base + LoRA) for each completion
/// 2. Computing reference log-probs under the base model (no LoRA) — KL anchor
/// 3. Computing advantages from rewards normalized within each group
/// 4. Optimizing a clipped importance-sampling objective with KL penalty
///
/// Returns the path to the saved adapter directory.
pub fn grpo_train(
    groups: &[GrpoGroup],
    config: &GrpoConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    replay_ctx: Option<ReplayContext>,
) -> Result<PathBuf> {
    grpo_train_to(
        groups,
        config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        adapter_dir,
        adapter_name,
        progress_cb,
        replay_ctx,
    )
}

/// Staged-output variant of [`grpo_train`].
#[allow(clippy::too_many_arguments)]
pub fn grpo_train_to(
    groups: &[GrpoGroup],
    config: &GrpoConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    output_adapter_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    replay_ctx: Option<ReplayContext>,
) -> Result<PathBuf> {
    grpo_train_to_with_coordination(
        groups,
        config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        output_adapter_dir,
        adapter_name,
        progress_cb,
        replay_ctx,
        None,
    )
}

/// Staged-output GRPO with bounded server GPU ownership. Direct callers should
/// normally use [`grpo_train_to`]; the server supplies coordination so
/// inference can run between optimizer groups and checkpoint snapshots.
#[allow(clippy::too_many_arguments)]
pub fn grpo_train_to_with_coordination(
    groups: &[GrpoGroup],
    config: &GrpoConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    output_adapter_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    replay_ctx: Option<ReplayContext>,
    gpu_step_coordination: Option<GpuStepCoordination>,
) -> Result<PathBuf> {
    grpo_train_to_with_checkpoint_root(
        groups,
        config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        output_adapter_dir,
        output_adapter_dir,
        adapter_name,
        progress_cb,
        replay_ctx,
        gpu_step_coordination,
    )
}

/// Standalone staged-output GRPO with a separate durable checkpoint root.
///
/// Server callers should use [`grpo_train_to_with_checkpoint_root_and_runtime`]
/// to bind every per-group plan to their process-lifetime memory configuration.
#[allow(clippy::too_many_arguments)]
pub fn grpo_train_to_with_checkpoint_root(
    groups: &[GrpoGroup],
    config: &GrpoConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    output_adapter_dir: &Path,
    checkpoint_output_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    replay_ctx: Option<ReplayContext>,
    gpu_step_coordination: Option<GpuStepCoordination>,
) -> Result<PathBuf> {
    ensure_training_optimizer_device_supported(
        "GRPO",
        weights,
        weights.embed_tokens.device(),
        config.optimizer,
        config.lora_rank,
    )?;
    let runtime =
        crate::standalone_training_runtime_for_weight_device(weights.embed_tokens.device())?;
    grpo_train_to_with_checkpoint_root_and_runtime(
        groups,
        config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        output_adapter_dir,
        checkpoint_output_dir,
        adapter_name,
        progress_cb,
        replay_ctx,
        gpu_step_coordination,
        &runtime,
    )
}

/// Server-owned inline GRPO entry point with immutable runtime inputs.
#[allow(clippy::too_many_arguments)]
pub fn grpo_train_to_with_checkpoint_root_and_runtime(
    groups: &[GrpoGroup],
    config: &GrpoConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    output_adapter_dir: &Path,
    checkpoint_output_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    replay_ctx: Option<ReplayContext>,
    gpu_step_coordination: Option<GpuStepCoordination>,
    runtime: &crate::TrainingRuntimeContext,
) -> Result<PathBuf> {
    let runtime_device = ensure_training_optimizer_entry_supported(
        "GRPO",
        weights,
        runtime,
        config.optimizer,
        config.lora_rank,
    )?;
    crate::ensure_memory_governor_for_runtime(runtime_device, runtime)
        .context("initialize GRPO memory governor")?;
    let run_started = Instant::now();
    anyhow::ensure!(
        config.checkpoint_interval != Some(0),
        "GRPO checkpoint_interval must be greater than zero"
    );
    // Fail fast on loss compositions the kt-tape path cannot train —
    // BEFORE any forward pass. The old order discovered this per-step,
    // after the rollout + reference forwards had already burned GPU time.
    let has_env_tokens = groups.iter().any(|g| {
        g.completions.iter().any(|c| {
            c.trajectory
                .iter()
                .any(|seg| seg.kind == crate::trajectory::TurnKind::Observation)
        })
    });
    config
        .loss
        .validate_for_kt_tape(has_env_tokens)
        .map_err(|e| anyhow::anyhow!("GRPO loss config: {e}"))?;
    config
        .validate_policy_config()
        .map_err(|e| anyhow::anyhow!("GRPO policy config: {e}"))?;

    let output_dir = output_adapter_dir.join(adapter_name);
    let training_data_sha256 = crate::train_receipt::sha256_json_serializable(&groups);
    let training_data_checkpoint_sha256 =
        checkpoint_sha256_hex(training_data_sha256.as_deref(), "GRPO training data")?;
    let requested_base_adapter_dir = config.base_adapter.as_deref().map(|name| {
        resolve_base_adapter_dir_from_roots(name, adapter_dir, output_adapter_dir, adapter_name)
    });
    let resume_checkpoint = config
        .resume_checkpoint
        .as_deref()
        .map(Path::new)
        .map(crate::checkpoint::load_training_checkpoint)
        .transpose()
        .context("load GRPO resume checkpoint")?;
    let resume_loop_state = resume_checkpoint
        .as_ref()
        .map(load_grpo_checkpoint_loop_state)
        .transpose()?;
    if let Some(checkpoint) = resume_checkpoint.as_ref() {
        anyhow::ensure!(
            checkpoint.manifest.training_kind == crate::checkpoint::TrainingKind::Grpo,
            "resume checkpoint is not a GRPO checkpoint"
        );
        anyhow::ensure!(
            checkpoint.manifest.adapter_name == adapter_name,
            "resume checkpoint adapter {:?} does not match {:?}",
            checkpoint.manifest.adapter_name,
            adapter_name
        );
        anyhow::ensure!(
            checkpoint.manifest.data.source_kind == GrpoCheckpointRoute::Inline.source_kind()
                && checkpoint.manifest.data.content_sha256 == training_data_checkpoint_sha256,
            "resume checkpoint inline GRPO data identity differs from this request"
        );
        anyhow::ensure!(
            resume_loop_state
                .as_ref()
                .is_some_and(|state| state.route == GrpoCheckpointRoute::Inline),
            "resume checkpoint was not produced by inline GRPO"
        );
    }
    if config.checkpoint_interval.is_some() || resume_checkpoint.is_some() {
        validate_exact_training_provenance(weights)?;
    }
    let resume_init_seed = resume_checkpoint
        .as_ref()
        .map(|checkpoint| {
            let state = checkpoint
                .manifest
                .rng_states
                .get("lora-init")
                .context("GRPO resume checkpoint has no lora-init RNG state")?;
            anyhow::ensure!(
                state.algorithm == "kiln.seeded-lora-init.v1" && state.position == 0,
                "unsupported GRPO lora-init RNG state"
            );
            Ok(state.seed)
        })
        .transpose()?;
    if let (Some(requested), Some(restored)) = (config.seed, resume_init_seed) {
        anyhow::ensure!(
            requested == restored,
            "GRPO resume seed {restored} differs from requested seed {requested}"
        );
    }
    let requested_effective_seed = resume_init_seed.or(config.seed);
    let mut gpu_writer_timings = resume_loop_state
        .as_ref()
        .map_or_else(GrpoGpuWriterTimings::default, |state| {
            state.gpu_writer_timings.clone()
        });
    // (#1082) `embed_tokens.device()` is a kt Device; the GRPO body is now
    // kt-native (kt `Parameter`s, kt AdamW state, kt tape forward/backward),
    // so keep `device` kt downstream. The only candle touch is safetensors
    // adapter I/O, which bridges kt->candle locally inside save/load.
    let device = training_device_for_weights(weights, runtime)?;
    let backend = training_backend_for_device(device)?;
    ensure_tape_forward_backward_supported("GRPO", weights, backend.as_ref())?;
    let training_precision_policy = training_precision_policy_for_backend(backend.as_ref());
    ensure_training_optimizer_supported(
        "GRPO",
        backend.as_ref(),
        config.optimizer,
        weights.embed_tokens.dtype(),
        config.lora_rank,
    )?;
    // Training-session residency: upload a one-device copy of the weights
    // when the substrate needs it (Vulkan hybrid). Shadow `weights` so the
    // whole body trains against the resident copy; it drops at return.
    let resident_weights = run_coordinated_grpo_gpu_phase(
        gpu_step_coordination.as_ref(),
        &*backend,
        &mut gpu_writer_timings,
        "resident model setup",
        || resident_training_weights(weights, &device),
    )?;
    let weights = resident_weights.as_ref().unwrap_or(weights);
    let streaming_prefill = runtime.resolved_streaming_prefill_policy(device);
    let training_runtime_planning_identity =
        runtime.checkpoint_planning_identity_for_device(device);

    let total_completions: usize = groups.iter().map(|g| g.completions.len()).sum();
    let mut data_stats = crate::train_receipt::DataStatsReceipt {
        groups_read: groups.len(),
        completions_read: total_completions,
        ..Default::default()
    };
    let mut token_counts = resume_loop_state
        .as_ref()
        .map_or_else(crate::train_receipt::TokenCountReceipt::default, |state| {
            state.token_counts.clone()
        });
    let mut echo_metrics = resume_loop_state.as_ref().map_or_else(
        crate::train_receipt::EchoActivityMetrics::default,
        |state| state.echo_metrics.clone(),
    );
    let mut reward_stats = crate::train_receipt::RewardStatsReceipt::default();
    let mut lora_grad_norms = resume_loop_state.as_ref().map_or_else(
        crate::train_receipt::LoraGradNormAccumulator::default,
        |state| state.lora_grad_norms.clone(),
    );
    let mut policy_audit = resume_loop_state.as_ref().map_or_else(
        crate::train_receipt::GrpoPolicyAuditAccumulator::default,
        |state| state.policy_audit.clone(),
    );
    let mut phase_timings = resume_loop_state
        .as_ref()
        .map_or_else(GrpoBenchmarkTimings::default, |state| {
            state.phase_timings.clone()
        });
    let mut dynamic_groups_filtered = resume_loop_state
        .as_ref()
        .map_or(0, |state| state.dynamic_groups_filtered as usize);
    let learning_rate = config.effective_learning_rate();
    if let Some(explicit) = config.learning_rate {
        if let Some(warning) = crate::learning_rate_band_warning(
            explicit,
            crate::resolve_learning_rate(&config.optimizer, crate::TrainMode::Grpo),
        ) {
            tracing::warn!(optimizer = ?config.optimizer, "GRPO {warning}");
        }
    }
    tracing::info!(
        num_groups = groups.len(),
        total_completions,
        total_input_groups = groups.len(),
        total_input_completions = total_completions,
        lr = learning_rate,
        kl_coeff = config.kl_coeff,
        clip_epsilon = config.clip_epsilon,
        rank = config.lora_rank,
        alpha = config.lora_alpha,
        adapter_name,
        "starting GRPO training"
    );
    tracing::info!(
        groups = groups.len(),
        completions = total_completions,
        "GRPO data loaded"
    );

    let alpha_over_rank = match crate::lora_scaling::validate_lora_scaling(
        config.lora_rank,
        config.lora_alpha,
        config.allow_high_lora_scale,
    ) {
        Ok(value) => value,
        Err(err) => {
            let message = format!("{err:#}");
            write_grpo_train_receipt_best_effort(
                adapter_name,
                model_config,
                tokenizer,
                weights.base_weight_shard_manifest.as_ref(),
                weights.execution_provenance.as_ref(),
                None,
                config,
                config.seed,
                None,
                requested_base_adapter_dir.as_deref(),
                &output_dir,
                crate::train_receipt::TrainingDataReceipt {
                    source: "inline_grpo_groups".to_string(),
                    path: None,
                    sha256: training_data_sha256,
                },
                data_stats,
                reward_stats,
                token_counts,
                crate::train_receipt::TrainingPhaseTimingsReceipt::default(),
                crate::train_receipt::EchoActivityMetrics::default(),
                run_started.elapsed().as_millis() as u64,
                dynamic_groups_filtered,
                None,
                Vec::new(),
                None,
                Some(message),
            );
            return Err(crate::train_receipt::annotate_training_error(err));
        }
    };
    tracing::info!(
        alpha_over_rank,
        allow_high_lora_scale = config.allow_high_lora_scale,
        "validated LoRA scaling"
    );

    // Open replay state (writes request record + lineage.json *before* the
    // optimizer step) and resolve the effective seed.
    let replay_parent_adapter = resume_checkpoint
        .is_none()
        .then_some(config.base_adapter.as_deref())
        .flatten();
    let (replay_state, effective_seed) = match replay_ctx.as_ref() {
        Some(ctx) => {
            let (state, seed) = open_replay_state_to(
                ctx,
                requested_effective_seed,
                replay_parent_adapter,
                adapter_dir,
                output_adapter_dir,
                adapter_name,
            )?;
            (Some(state), Some(seed))
        }
        None => (
            None,
            Some(requested_effective_seed.unwrap_or_else(rand::random)),
        ),
    };
    let effective_seed_value = effective_seed.expect("GRPO always resolves an effective seed");
    let effective_checkpoint_config =
        grpo_checkpoint_effective_config(config, learning_rate, effective_seed_value)?;
    if let Some(checkpoint) = resume_checkpoint.as_ref() {
        anyhow::ensure!(
            checkpoint.manifest.effective_config == effective_checkpoint_config,
            "resume checkpoint effective GRPO configuration differs from this request: checkpoint={}, request={}",
            checkpoint.manifest.effective_config,
            effective_checkpoint_config
        );
    }

    let base_adapter_result = if resume_checkpoint.is_some() {
        Ok(None)
    } else {
        resolve_and_validate_base_adapter_from_roots(
            config.base_adapter.as_deref(),
            adapter_dir,
            output_adapter_dir,
            adapter_name,
            model_config,
            config.lora_rank,
            config.allow_adapter_shape_conversion,
        )
    };
    let base_adapter_dir = match base_adapter_result {
        Ok(value) => value,
        Err(err) => {
            let message = format!("{err:#}");
            write_grpo_train_receipt_best_effort(
                adapter_name,
                model_config,
                tokenizer,
                weights.base_weight_shard_manifest.as_ref(),
                weights.execution_provenance.as_ref(),
                None,
                config,
                effective_seed,
                Some(alpha_over_rank),
                requested_base_adapter_dir.as_deref(),
                &output_dir,
                crate::train_receipt::TrainingDataReceipt {
                    source: "inline_grpo_groups".to_string(),
                    path: None,
                    sha256: training_data_sha256,
                },
                data_stats,
                reward_stats,
                token_counts,
                crate::train_receipt::TrainingPhaseTimingsReceipt::default(),
                crate::train_receipt::EchoActivityMetrics::default(),
                run_started.elapsed().as_millis() as u64,
                dynamic_groups_filtered,
                None,
                Vec::new(),
                None,
                Some(message),
            );
            return Err(crate::train_receipt::annotate_training_error(err));
        }
    };

    // Initialization, optional base-adapter upload, and registry admission all
    // mutate backend state. Keep them in one explicit setup phase, then release
    // serving before reward filtering and tokenization.
    let (mut params, mut opt_state) = run_coordinated_grpo_gpu_phase(
        gpu_step_coordination.as_ref(),
        &*backend,
        &mut gpu_writer_timings,
        "adapter and optimizer setup",
        || {
            let mut params = TrainableLoraParams::initialize_seeded_with_precision_policy(
                model_config,
                weights,
                config.lora_rank,
                config.lora_alpha,
                &device,
                Some(effective_seed_value),
                training_precision_policy,
            )?;

            if let Some(checkpoint) = resume_checkpoint.as_ref() {
                let adapter_path = checkpoint
                    .artifact_path(&checkpoint.manifest.state_files.adapter_parameters)?;
                params.load_checkpoint_parameters(&adapter_path)?;
                tracing::info!(
                    checkpoint = %checkpoint.root.display(),
                    step = checkpoint.manifest.progress.global_step,
                    "restored exact GRPO adapter parameters"
                );
            } else if let Some(base_dir) = base_adapter_dir.as_deref() {
                let n_loaded = params.load_from_safetensors(base_dir, &device)?;
                tracing::info!(
                    base = %base_dir.display(),
                    num_tensors = n_loaded,
                    "loaded base adapter — continuing GRPO from those weights"
                );
            }

            let mut opt_state = make_opt_state(&params, config.optimizer, learning_rate, &device)?;
            if let Some(checkpoint) = resume_checkpoint.as_ref() {
                let state_path = checkpoint
                    .manifest
                    .state_files
                    .optimizer_state
                    .as_deref()
                    .map(|relative| checkpoint.artifact_path(relative))
                    .transpose()?;
                match (opt_state.as_mut(), state_path) {
                    (Some(state), Some(path)) => {
                        let step = u32::try_from(checkpoint.manifest.progress.global_step)
                            .context("GRPO resume optimizer step exceeds u32")?;
                        state.load_checkpoint_state(&params, &path, step)?;
                    }
                    (None, None) => {}
                    (Some(_), None) => anyhow::bail!(
                        "stateful GRPO optimizer checkpoint has no optimizer artifact"
                    ),
                    (None, Some(_)) => {
                        anyhow::bail!("SGD GRPO checkpoint unexpectedly contains optimizer state")
                    }
                }
            }
            // Registry identity is process-local. Restore host tensors before
            // registration so resident copies cannot retain seeded state.
            params.register_with_backend(&*backend)?;
            if let Some(state) = opt_state.as_ref() {
                state.register_with_backend(&*backend)?;
            }
            Ok((params, opt_state))
        },
    )?;

    tracing::info!(
        num_vars = params.all_params().len(),
        "initialized trainable LoRA parameters"
    );
    let lora_grad_index = LoraGradNormIndex::new(&params);

    let mut train_body = || -> Result<(PathBuf, f64)> {
        let dynamic_sampling = config.dynamic_sampling;
        let mut dynamic_dropped: usize = 0;
        let mut tokenization_failed: usize = 0;
        let input_reward_groups: Vec<Vec<f64>> = groups
            .iter()
            .map(|group| {
                group
                    .completions
                    .iter()
                    .map(|completion| completion.reward)
                    .collect()
            })
            .collect();
        reward_stats = crate::train_receipt::reward_stats_from_groups_with_threshold(
            input_reward_groups.iter().map(Vec::as_slice),
            config.reward_saturation_threshold,
        );
        crate::train_receipt::warn_reward_diagnostics(
            "grpo_startup",
            adapter_name,
            &reward_stats,
            config.reward_saturation_threshold,
            config.reward_low_variance_threshold,
        );
        let reward_filter_plan = build_reward_filter_plan(
            config,
            &output_dir,
            "inline_grpo_groups",
            groups
                .iter()
                .enumerate()
                .map(|(idx, group)| RewardFilterInputGroup {
                    id: format!("group:{}", idx + 1),
                    source_index: idx + 1,
                    source_line: None,
                    reward_variance: reward_filter_variance(
                        &group
                            .completions
                            .iter()
                            .map(|completion| completion.reward)
                            .collect::<Vec<_>>(),
                    ),
                })
                .collect(),
        )?;
        if let Some(plan) = reward_filter_plan.as_ref() {
            record_reward_filter_plan(&mut data_stats, plan);
            data_stats.groups_filtered = data_stats
                .groups_filtered
                .saturating_add(plan.groups_dropped);
            tracing::info!(
                kept = plan.groups_kept,
                dropped = plan.groups_dropped,
                sidecar = %plan.sidecar_path.display(),
                "GRPO reward variance filter applied"
            );
            if let Some(reason) = plan.failure_reason.as_ref() {
                anyhow::bail!("{reason}");
            }
            if plan.skip_training {
                params.save_peft(&output_dir, model_config.num_layers)?;
                tracing::info!(
                    adapter = adapter_name,
                    path = %output_dir.display(),
                    "GRPO reward variance filter skipped training"
                );
                return Ok((output_dir.clone(), 0.0));
            }
        }

        // Tokenize all completions: for each group, tokenize prompt + each completion.
        // When dynamic_sampling is enabled (DAPO, arXiv:2503.14476), groups whose
        // completions all share the same reward are dropped before tokenization —
        // their advantage vector is uniformly zero and would contribute no
        // policy-gradient signal anyway.
        tracing::info!(
            groups = groups.len(),
            completions = total_completions,
            dynamic_sampling,
            "GRPO tokenize start"
        );
        let tokenize_all_started = Instant::now();
        let mut tokenized_groups: Vec<TokenizedGrpoGroup> = Vec::new();
        let mut trainable_source_indices: Vec<u64> = Vec::new();
        for (idx, group) in groups.iter().enumerate() {
            let source_index = idx + 1;
            if let Some(plan) = reward_filter_plan.as_ref() {
                if !plan.keeps_source_index(source_index) {
                    continue;
                }
            }
            if dynamic_sampling && is_degenerate_grpo_group(group) {
                dynamic_dropped += 1;
                continue;
            }
            let mask_cfg = crate::trajectory_mask::MaskConfig::from_grpo_config(config);
            match tokenize_grpo_group_timed(group, tokenizer, &mask_cfg, Some(&mut phase_timings)) {
                Ok(tgroup) => {
                    validate_tokenized_behavior_policy(&tgroup, config.behavior_policy)
                        .with_context(|| {
                            format!("validate GRPO group {source_index} behavior provenance")
                        })?;
                    tokenized_groups.push(tgroup);
                    trainable_source_indices.push(idx as u64);
                }
                Err(e) => {
                    if config.behavior_policy == BehaviorPolicy::Recorded {
                        return Err(e).with_context(|| {
                            format!(
                                "tokenize GRPO group {source_index} with required recorded behavior provenance"
                            )
                        });
                    }
                    tokenization_failed += 1;
                    tracing::warn!("skipping GRPO group: {e}");
                }
            }
        }
        if let Some(state) = resume_loop_state.as_ref() {
            anyhow::ensure!(
                state.dynamic_groups_filtered as usize == dynamic_dropped,
                "resume checkpoint dynamic-sampling selection differs from this request"
            );
        }
        dynamic_groups_filtered = dynamic_dropped;
        data_stats.groups_filtered = data_stats
            .reward_groups_filtered
            .saturating_add(dynamic_dropped)
            .saturating_add(tokenization_failed);
        let planned_token_counts = token_counts_for_grpo_groups(&tokenized_groups);
        let planned_completions: usize = tokenized_groups
            .iter()
            .map(|group| group.completions.len())
            .sum();
        if let Some(state) = resume_loop_state.as_ref() {
            let current_reward_filter_sidecar = data_stats.reward_filter_sidecar.clone();
            anyhow::ensure!(
                state.data_stats.groups_read == data_stats.groups_read
                    && state.data_stats.completions_read == data_stats.completions_read
                    && state.data_stats.groups_filtered == data_stats.groups_filtered
                    && state.data_stats.reward_groups_filtered == data_stats.reward_groups_filtered
                    && state.data_stats.reward_groups_kept == data_stats.reward_groups_kept,
                "resume checkpoint GRPO filtering statistics differ from this request"
            );
            data_stats = state.data_stats.clone();
            data_stats.reward_filter_sidecar = current_reward_filter_sidecar;
        }
        tracing::info!(
            groups = tokenized_groups.len(),
            completions = planned_completions,
            action_tokens = planned_token_counts.action_tokens,
            env_tokens = planned_token_counts.env_tokens,
            context_tokens = planned_token_counts.context_tokens,
            elapsed_ms = tokenize_all_started.elapsed().as_millis() as u64,
            "GRPO tokenize end"
        );
        crate::train_receipt::warn_echo_enabled_without_env_tokens(
            "grpo",
            config.loss.echo_enabled(),
            &planned_token_counts,
        );

        if dynamic_dropped > 0 {
            tracing::info!(
                dropped = dynamic_dropped,
                total = groups.len(),
                "GRPO dynamic sampling: dropped degenerate groups (all rewards equal)"
            );
        }

        if tokenized_groups.is_empty() {
            anyhow::bail!("no valid GRPO groups after tokenization");
        }

        // Compute the max seq_len across every completion in every group
        // so the auto-tuner sizes checkpointing against the longest path,
        // not the average.
        let max_seq_len_tokens: usize = tokenized_groups
            .iter()
            .flat_map(|g| g.completions.iter())
            .map(|c| c.input_ids.len())
            .max()
            .unwrap_or(0);

        // Resolve checkpointing per group from the actual longest
        // completion in that group. The submission preflight covers the
        // worst-case group for admission, but using that segment count for
        // every group needlessly slows shorter groups.
        let activation_bytes_per_elem = training_activation_bytes_per_elem_for_policy(
            weights,
            training_precision_policy,
            model_config_has_linear_attention(model_config),
        );
        tracing::info!(
            max_seq_len_tokens,
            preflight_max_segments = ?config.grad_checkpoint_segments,
            activation_bytes_per_elem,
            "GRPO gradient checkpointing will resolve per group"
        );

        let total_steps = tokenized_groups.len();
        let gradient_checkpoint_plan: Vec<_> = tokenized_groups
            .iter()
            .zip(&trainable_source_indices)
            .map(|(group, source_index)| {
                let max_seq_len = group
                    .completions
                    .iter()
                    .map(|completion| completion.input_ids.len())
                    .max()
                    .unwrap_or(0);
                let resolved = checkpoint_config_for_training_step(
                    weights,
                    &device,
                    config.grad_checkpoint_segments,
                    model_config.num_layers,
                    max_seq_len,
                    model_config.hidden_size,
                    model_config.intermediate_size,
                    model_config.vocab_size,
                    2,
                    activation_bytes_per_elem,
                    runtime,
                );
                let boundaries = checkpoint_segments_for_config(
                    weights,
                    &device,
                    max_seq_len,
                    resolved,
                    streaming_prefill,
                );
                serde_json::json!({
                    "source_index": source_index,
                    "max_seq_len": max_seq_len,
                    "enabled": resolved.enabled,
                    "num_segments": resolved.num_segments,
                    "auto_configured": resolved.auto_configured,
                    "boundaries": boundaries,
                })
            })
            .collect();
        let trainable_order_sha256 =
            crate::train_receipt::sha256_json_serializable(&trainable_source_indices)
                .context("hash inline GRPO trainable order")?;
        let gradient_checkpoint_plan_sha256 =
            crate::train_receipt::sha256_json_serializable(&gradient_checkpoint_plan)
                .context("hash inline GRPO gradient-checkpoint plan")?;
        let ema_refresh_every = if config.kl_penalty_enabled() {
            match &config.kl_reference_policy {
                KlReferencePolicy::Ema { refresh_every, .. } => Some(*refresh_every),
                _ => None,
            }
        } else {
            None
        };
        let checkpoint_descriptor = GrpoCheckpointDescriptor {
            route: GrpoCheckpointRoute::Inline,
            adapter_name: adapter_name.to_string(),
            effective_config: effective_checkpoint_config.clone(),
            precision_policy: training_checkpoint_precision(&params, opt_state.as_ref())?,
            data: crate::checkpoint::TrainingCheckpointData {
                source_kind: GrpoCheckpointRoute::Inline.source_kind().to_string(),
                content_sha256: training_data_checkpoint_sha256.clone(),
                item_count: total_steps as u64,
            },
            init_seed: effective_seed_value,
            optimizer: config.optimizer,
            learning_rate,
            total_steps,
            base_model_weights_sha256: weights.source_content_sha256.clone(),
            auxiliary_state: grpo_checkpoint_auxiliary_state(
                GrpoCheckpointRoute::Inline,
                model_config,
                tokenizer,
                training_precision_policy,
                weights.source_content_sha256.as_deref(),
                weights.base_weight_shard_manifest.as_ref(),
                weights.execution_provenance.as_ref(),
                BackendIdentity::runtime_name(backend.as_ref()),
                &trainable_order_sha256,
                &gradient_checkpoint_plan_sha256,
                &training_runtime_planning_identity,
            ),
            ema_refresh_every,
        };
        if let (Some(checkpoint), Some(loop_state)) =
            (resume_checkpoint.as_ref(), resume_loop_state.as_ref())
        {
            checkpoint_descriptor.validate_resume(checkpoint, loop_state)?;
        }

        let mut global_step = resume_loop_state
            .as_ref()
            .map_or(0, |state| state.global_step as usize);
        let mut processed_completions = resume_loop_state
            .as_ref()
            .map_or(0, |state| state.processed_completions as usize);
        let mut loss_history = resume_loop_state
            .as_ref()
            .map_or_else(Vec::new, |state| state.loss_history.clone());
        let mut last_loss = resume_loop_state
            .as_ref()
            .and_then(|state| state.last_loss)
            .unwrap_or(0.0);
        let mut last_saved_step = resume_loop_state
            .as_ref()
            .map(|state| state.global_step as usize);
        anyhow::ensure!(
            global_step <= total_steps,
            "GRPO resume cursor {global_step} exceeds {total_steps} trainable groups"
        );
        let expected_processed_completions: usize = tokenized_groups
            .iter()
            .take(global_step)
            .map(|group| group.completions.len())
            .sum();
        let expected_token_counts = token_counts_for_grpo_groups(&tokenized_groups[..global_step]);
        anyhow::ensure!(
            processed_completions == expected_processed_completions
                && token_counts == expected_token_counts,
            "GRPO resume diagnostics do not match the committed trainable prefix"
        );
        let mut last_ckpt_log_key: Option<(bool, usize)> = None;

        let pb = make_step_progress(total_steps, "grpo training");
        if let Some(pb) = &pb {
            pb.set_position(global_step as u64);
        }

        // Phase 3b: maintain an EMA-snapshot LoRA when
        // `KlReferencePolicy::Ema` is configured. Initialized eagerly to a
        // deepcopy of the (post-init, pre-train) LoRA so the very first
        // group's reference forward already runs against a frozen snapshot
        // rather than the live policy.
        let mut ema_ref_state = if config.kl_penalty_enabled() {
            match &config.kl_reference_policy {
                KlReferencePolicy::Ema {
                    decay,
                    refresh_every,
                } => {
                    let (snapshot, groups_since_refresh) =
                        if let (Some(checkpoint), Some(loop_state)) =
                            (resume_checkpoint.as_ref(), resume_loop_state.as_ref())
                        {
                            let relative = checkpoint
                                .manifest
                                .state_files
                                .reference_state
                                .as_deref()
                                .context("EMA GRPO resume checkpoint has no reference state")?;
                            let path = checkpoint.artifact_path(relative)?;
                            (
                                load_lora_reference_checkpoint(&path, &params, &device)?,
                                loop_state
                                    .ema_groups_since_refresh
                                    .context("EMA GRPO resume checkpoint has no cadence cursor")?
                                    as usize,
                            )
                        } else {
                            (
                                run_coordinated_grpo_gpu_phase(
                                    gpu_step_coordination.as_ref(),
                                    &*backend,
                                    &mut gpu_writer_timings,
                                    "initial EMA reference snapshot",
                                    || {
                                        lora_snapshot_capture_or_blend(
                                            &params, None, *decay, &device,
                                        )
                                        .context("initial EMA reference snapshot")
                                    },
                                )?,
                                0,
                            )
                        };
                    Some(EmaReferenceState {
                        snapshot,
                        groups_since_refresh,
                        refresh_every: *refresh_every,
                        decay: *decay,
                    })
                }
                _ => None,
            }
        } else {
            None
        };

        for (group_idx, tgroup) in tokenized_groups.iter().enumerate().skip(global_step) {
            let num_completions = tgroup.completions.len();
            let group_counts = token_counts_for_grpo_groups(std::slice::from_ref(tgroup));
            let group_max_seq_len = tgroup
                .completions
                .iter()
                .map(|completion| completion.input_ids.len())
                .max()
                .unwrap_or(0);
            let ckpt_config = checkpoint_config_for_training_step(
                weights,
                &device,
                config.grad_checkpoint_segments,
                model_config.num_layers,
                group_max_seq_len,
                model_config.hidden_size,
                model_config.intermediate_size,
                model_config.vocab_size,
                2, // BF16 base weights
                activation_bytes_per_elem,
                runtime,
            );
            let segments = checkpoint_segments_for_config(
                weights,
                &device,
                group_max_seq_len,
                ckpt_config,
                streaming_prefill,
            );
            let ckpt_log_key = (ckpt_config.enabled, ckpt_config.num_segments);
            if last_ckpt_log_key != Some(ckpt_log_key) {
                if let Some(ref segs) = segments {
                    tracing::info!(
                        group = group_idx + 1,
                        max_seq_len = group_max_seq_len,
                        num_segments = segs.len(),
                        preflight_max_segments = ?config.grad_checkpoint_segments,
                        boundaries = ?segs,
                        "GRPO gradient checkpointing enabled for group shape"
                    );
                } else {
                    tracing::info!(
                        group = group_idx + 1,
                        max_seq_len = group_max_seq_len,
                        preflight_max_segments = ?config.grad_checkpoint_segments,
                        "GRPO gradient checkpointing disabled for group shape"
                    );
                }
                last_ckpt_log_key = Some(ckpt_log_key);
            }
            let step_report = run_coordinated_grpo_gpu_phase(
                gpu_step_coordination.as_ref(),
                &*backend,
                &mut gpu_writer_timings,
                "optimizer group",
                || {
                    let step_report = train_tokenized_grpo_group_with_grad_norms(
                        &*backend,
                        tgroup,
                        weights,
                        model_config,
                        &mut params,
                        config,
                        segments.as_deref(),
                        &device,
                        opt_state.as_mut(),
                        &mut lora_grad_norms,
                        &lora_grad_index,
                        &mut policy_audit,
                        ema_ref_state.as_ref().map(|s| &s.snapshot),
                        Some(&mut phase_timings),
                        streaming_prefill,
                    )?;

                    // Refresh while the same writer is held: both the policy
                    // update and the frozen reference transition form one
                    // exact optimizer-group boundary.
                    if let Some(state) = ema_ref_state.as_mut() {
                        state.groups_since_refresh += 1;
                        if state.groups_since_refresh >= state.refresh_every {
                            params
                                .sync_to_master(&*backend)
                                .context("sync policy before EMA reference refresh")?;
                            state.snapshot = lora_snapshot_capture_or_blend(
                                &params,
                                Some(&state.snapshot),
                                state.decay,
                                &device,
                            )
                            .context("EMA reference snapshot refresh")?;
                            state.groups_since_refresh = 0;
                            tracing::debug!(
                                group = group_idx + 1,
                                refresh_every = state.refresh_every,
                                decay = state.decay,
                                "GRPO EMA reference snapshot refreshed"
                            );
                        }
                    }
                    Ok(step_report)
                },
            )?;
            let avg_group_loss = step_report.loss;
            anyhow::ensure!(
                avg_group_loss.is_finite(),
                "GRPO loss became non-finite at group {}: {avg_group_loss}",
                group_idx + 1
            );
            echo_metrics.observe_env_ce(step_report.echo_env_ce);
            last_loss = avg_group_loss;
            loss_history.push(avg_group_loss);
            global_step += 1;
            processed_completions = processed_completions.saturating_add(num_completions);
            token_counts.add_from(&group_counts);
            data_stats.groups_trained = global_step;
            data_stats.completions_trained = processed_completions;

            let checkpoint_due = config
                .checkpoint_interval
                .is_some_and(|interval| global_step % interval == 0 && global_step < total_steps);
            if checkpoint_due {
                let mut loop_state = GrpoCheckpointLoopState::capture(
                    GrpoCheckpointRoute::Inline,
                    global_step,
                    None,
                    None,
                    processed_completions,
                    &loss_history,
                    &data_stats,
                    &token_counts,
                    dynamic_groups_filtered,
                    &echo_metrics,
                    &lora_grad_norms,
                    &policy_audit,
                    &phase_timings,
                    &gpu_writer_timings,
                    ema_ref_state.as_ref(),
                );
                let path = checkpoint_descriptor.save(
                    checkpoint_output_dir,
                    &*backend,
                    &mut params,
                    &mut opt_state,
                    ema_ref_state.as_ref(),
                    &mut loop_state,
                    gpu_step_coordination.as_ref(),
                    &mut gpu_writer_timings,
                    "checkpoint device snapshot",
                )?;
                last_saved_step = Some(global_step);
                tracing::info!(
                    step = global_step,
                    checkpoint = %path.display(),
                    "saved exact GRPO training checkpoint"
                );
            }

            if let Some(ref cb) = progress_cb {
                let control = cb(TrainingProgress {
                    epoch: 1,
                    total_epochs: 1,
                    step: global_step,
                    total_steps,
                    loss: avg_group_loss,
                    progress: global_step as f32 / total_steps as f32,
                });
                if control == TrainControl::Stop && global_step < total_steps {
                    if last_saved_step != Some(global_step) {
                        let mut loop_state = GrpoCheckpointLoopState::capture(
                            GrpoCheckpointRoute::Inline,
                            global_step,
                            None,
                            None,
                            processed_completions,
                            &loss_history,
                            &data_stats,
                            &token_counts,
                            dynamic_groups_filtered,
                            &echo_metrics,
                            &lora_grad_norms,
                            &policy_audit,
                            &phase_timings,
                            &gpu_writer_timings,
                            ema_ref_state.as_ref(),
                        );
                        let path = checkpoint_descriptor.save(
                            checkpoint_output_dir,
                            &*backend,
                            &mut params,
                            &mut opt_state,
                            ema_ref_state.as_ref(),
                            &mut loop_state,
                            gpu_step_coordination.as_ref(),
                            &mut gpu_writer_timings,
                            "cancellation checkpoint device snapshot",
                        )?;
                        tracing::info!(
                            step = global_step,
                            checkpoint = %path.display(),
                            "saved exact GRPO checkpoint before cancellation"
                        );
                    }
                    anyhow::bail!("training cancelled by user (stop requested at step boundary)");
                }
            }

            tracing::info!(
                group = group_idx + 1,
                total_groups = total_steps,
                num_completions,
                action_tokens = group_counts.action_tokens,
                env_tokens = group_counts.env_tokens,
                loss = format!("{avg_group_loss:.6}"),
                "GRPO group step"
            );
            if let Some(echo_env_ce) = step_report.echo_env_ce {
                tracing::info!(
                    group = group_idx + 1,
                    total_groups = total_steps,
                    action_tokens = group_counts.action_tokens,
                    env_tokens = group_counts.env_tokens,
                    echo_env_ce,
                    "GRPO ECHO group metrics"
                );
            }

            if let Some(pb) = &pb {
                pb.set_message(format!("{avg_group_loss:.6}"));
                pb.inc(1);
            }
        }

        anyhow::ensure!(
            global_step == total_steps
                && loss_history.len() == total_steps
                && processed_completions == planned_completions
                && token_counts == planned_token_counts,
            "GRPO loop completed with inconsistent progress or diagnostics"
        );

        if let Some(pb) = pb {
            pb.finish_and_clear();
        }

        // Pull current Var values from registry into candle CPU
        // storage before final save_peft.
        let synced = run_coordinated_grpo_gpu_phase(
            gpu_step_coordination.as_ref(),
            &*backend,
            &mut gpu_writer_timings,
            "final adapter snapshot",
            || {
                params
                    .sync_to_master(&*backend)
                    .context("capture final GRPO adapter state")
            },
        )?;
        tracing::debug!(synced, "synced LoRA Vars to candle before GRPO save");

        // Save the trained adapter
        params.save_peft(&output_dir, model_config.num_layers)?;

        tracing::info!(
            adapter = adapter_name,
            path = %output_dir.display(),
            final_loss = format!("{last_loss:.6}"),
            "GRPO training complete"
        );

        Ok((output_dir.clone(), last_loss))
    };

    let mut result = train_body();
    drop(train_body);
    let policy_audit = finish_grpo_policy_audit(&mut result, policy_audit);
    let mut adapter_smoke_test = None;
    let cleanup_result = run_coordinated_grpo_gpu_phase(
        gpu_step_coordination.as_ref(),
        &*backend,
        &mut gpu_writer_timings,
        "adapter smoke test and cleanup",
        || {
            if config.adapter_smoke_test && result.is_ok() {
                adapter_smoke_test = Some(run_adapter_smoke_test_best_effort(
                    adapter_name,
                    &*backend,
                    weights,
                    model_config,
                    tokenizer,
                    &params,
                    config.adapter_smoke_prompts.as_deref(),
                    streaming_prefill,
                ));
            }
            // Registry eviction is backend mutation too; keep it within the
            // final bounded phase even after a failed training step.
            if let Some(state) = opt_state.as_ref() {
                state.evict_from_backend(&*backend);
            }
            params.evict_from_backend(&*backend);
            Ok(())
        },
    );
    if let Err(error) = cleanup_result {
        if result.is_ok() {
            result = Err(error.context("complete coordinated GRPO cleanup"));
        } else {
            tracing::warn!(error = %format!("{error:#}"), "GRPO cleanup could not acquire healthy backend");
        }
    }
    if let Some(state) = replay_state {
        let outcome = match &result {
            Ok((_, loss)) => Ok(*loss),
            Err(e) => Err(format!("{e:#}")),
        };
        if let Err(e) = close_replay_state(state, outcome) {
            tracing::warn!(error = %e, "failed to append GRPO replay outcome record");
        }
    }
    gpu_writer_timings.apply_to(&mut phase_timings);
    let status_error = result.as_ref().err().map(|err| format!("{err:#}"));
    write_grpo_train_receipt_best_effort(
        adapter_name,
        model_config,
        tokenizer,
        weights.base_weight_shard_manifest.as_ref(),
        weights.execution_provenance.as_ref(),
        training_precision_for_receipt_best_effort(&params, opt_state.as_ref()),
        config,
        effective_seed,
        Some(alpha_over_rank),
        base_adapter_dir.as_deref(),
        &output_dir,
        crate::train_receipt::TrainingDataReceipt {
            source: "inline_grpo_groups".to_string(),
            path: None,
            sha256: training_data_sha256,
        },
        data_stats,
        reward_stats,
        token_counts,
        phase_timings.to_receipt(),
        echo_metrics,
        run_started.elapsed().as_millis() as u64,
        dynamic_groups_filtered,
        adapter_smoke_test,
        lora_grad_norms.finish(),
        policy_audit,
        status_error,
    );
    result
        .map(|(dir, _)| dir)
        .map_err(crate::train_receipt::annotate_training_error)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BoundedGrpoJsonlScanStats {
    total_bytes: u64,
    total_lines: usize,
    groups: usize,
    completions: usize,
    max_row_bytes: u64,
}

fn scan_pinned_grpo_jsonl<F>(
    dataset_source: &PinnedGrpoJsonlSource,
    model_num_layers: usize,
    filter_enabled: bool,
    phase: &str,
    mut visit_group: F,
) -> Result<BoundedGrpoJsonlScanStats>
where
    F: FnMut(usize, usize, &GrpoGroup) -> Result<()>,
{
    use std::io::{BufRead as _, BufReader, Read as _};

    let dataset_path = dataset_source.display_path();
    let total_bytes = dataset_source.len()?;
    let file = dataset_source.reader_from_start()?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut line_no = 0usize;
    let mut bytes_read = 0u64;
    let mut groups = 0usize;
    let mut completions = 0usize;
    let mut max_row_bytes = 0u64;

    loop {
        line.clear();
        let read = (&mut reader)
            .take(MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES + 1)
            .read_line(&mut line)
            .with_context(|| {
                format!(
                    "read GRPO JSONL dataset {} line {} during {phase}",
                    dataset_path.display(),
                    line_no.saturating_add(1)
                )
            })?;
        if read == 0 {
            break;
        }
        line_no = line_no
            .checked_add(1)
            .with_context(|| format!("GRPO JSONL line count overflow during {phase}"))?;
        anyhow::ensure!(
            line.len() as u64 <= MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES,
            "GRPO JSONL line {line_no} exceeds the {} byte streamed preflight row limit during {phase}",
            MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES
        );
        max_row_bytes = max_row_bytes.max(line.len() as u64);
        bytes_read = bytes_read
            .checked_add(read as u64)
            .with_context(|| format!("GRPO JSONL byte count overflow during {phase}"))?;
        anyhow::ensure!(
            bytes_read <= total_bytes,
            "GRPO JSONL dataset {} grew while scanning during {phase}",
            dataset_path.display()
        );
        streamed_grpo_preflight_host_bytes(
            groups,
            completions,
            max_row_bytes,
            model_num_layers,
            filter_enabled,
        )
        .with_context(|| {
            format!("bound GRPO JSONL host memory before line {line_no} during {phase}")
        })?;

        let Some(group) = parse_grpo_jsonl_group_line(&line, line_no)? else {
            continue;
        };
        validate_grpo_trajectory_roles(&group, line_no)?;
        anyhow::ensure!(
            !group.completions.is_empty()
                && group.completions.len() <= crate::HF_TRL_GRPO_MAX_COMPLETIONS_PER_GROUP,
            "GRPO JSONL line {line_no} must contain 1..={} completions",
            crate::HF_TRL_GRPO_MAX_COMPLETIONS_PER_GROUP
        );
        groups = groups
            .checked_add(1)
            .with_context(|| format!("GRPO JSONL group count overflow during {phase}"))?;
        completions = completions
            .checked_add(group.completions.len())
            .with_context(|| format!("GRPO JSONL completion count overflow during {phase}"))?;
        streamed_grpo_preflight_host_bytes(
            groups,
            completions,
            max_row_bytes,
            model_num_layers,
            filter_enabled,
        )
        .with_context(|| format!("bound GRPO JSONL metadata at line {line_no} during {phase}"))?;
        visit_group(line_no, groups, &group)?;
    }

    anyhow::ensure!(
        bytes_read == total_bytes,
        "GRPO JSONL dataset {} changed length during {phase}: expected {total_bytes}, read {bytes_read}",
        dataset_path.display()
    );
    Ok(BoundedGrpoJsonlScanStats {
        total_bytes,
        total_lines: line_no,
        groups,
        completions,
        max_row_bytes,
    })
}

/// First-pass reward metadata for dry-run validation. Global variance is
/// supplied by a second disk pass so the legacy materialized receipt's fold
/// order remains byte-for-byte stable without retaining every reward.
#[derive(Debug)]
struct DryRunRewardStatsAccumulator {
    count: usize,
    sum: f64,
    min: f64,
    max: f64,
    group_count: usize,
    all_pass_group_count: usize,
    all_fail_group_count: usize,
    degenerate_group_count: usize,
    variance_histogram_counts: [usize; 6],
}

impl Default for DryRunRewardStatsAccumulator {
    fn default() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            group_count: 0,
            all_pass_group_count: 0,
            all_fail_group_count: 0,
            degenerate_group_count: 0,
            variance_histogram_counts: [0; 6],
        }
    }
}

impl DryRunRewardStatsAccumulator {
    fn observe_group(&mut self, group: &GrpoGroup, all_pass_threshold: f64) -> Result<f64> {
        let group_count = group.completions.len();
        anyhow::ensure!(group_count > 0, "GRPO reward group must not be empty");
        let group_mean = group
            .completions
            .iter()
            .map(|completion| completion.reward)
            .sum::<f64>()
            / group_count as f64;
        let group_variance = group
            .completions
            .iter()
            .map(|completion| {
                let centered = completion.reward - group_mean;
                centered * centered
            })
            .sum::<f64>()
            / group_count as f64;

        self.group_count = self
            .group_count
            .checked_add(1)
            .context("GRPO dry-run reward group count overflow")?;
        if group_variance <= crate::train_receipt::REWARD_DEGENERATE_GROUP_VARIANCE_EPSILON {
            self.degenerate_group_count = self
                .degenerate_group_count
                .checked_add(1)
                .context("GRPO dry-run degenerate group count overflow")?;
        }
        if group
            .completions
            .iter()
            .all(|completion| completion.reward >= all_pass_threshold)
        {
            self.all_pass_group_count = self
                .all_pass_group_count
                .checked_add(1)
                .context("GRPO dry-run all-pass group count overflow")?;
        }
        if group
            .completions
            .iter()
            .all(|completion| completion.reward <= 0.0)
        {
            self.all_fail_group_count = self
                .all_fail_group_count
                .checked_add(1)
                .context("GRPO dry-run all-fail group count overflow")?;
        }
        let histogram_bucket = if group_variance == 0.0 {
            Some(0)
        } else if group_variance > f64::MIN_POSITIVE && group_variance <= 1e-6 {
            Some(1)
        } else if group_variance > 1e-6 && group_variance <= 0.01 {
            Some(2)
        } else if group_variance > 0.01 && group_variance <= 0.25 {
            Some(3)
        } else if group_variance > 0.25 && group_variance <= 1.0 {
            Some(4)
        } else if group_variance > 1.0 {
            Some(5)
        } else {
            None
        };
        if let Some(bucket) = histogram_bucket {
            self.variance_histogram_counts[bucket] = self.variance_histogram_counts[bucket]
                .checked_add(1)
                .context("GRPO dry-run reward histogram count overflow")?;
        }
        for completion in &group.completions {
            self.count = self
                .count
                .checked_add(1)
                .context("GRPO dry-run reward count overflow")?;
            self.sum += completion.reward;
            self.min = self.min.min(completion.reward);
            self.max = self.max.max(completion.reward);
        }
        Ok(group_variance)
    }

    fn mean(&self) -> Option<f64> {
        (self.count > 0).then(|| self.sum / self.count as f64)
    }

    fn finish(self, squared_deviation_sum: f64) -> crate::train_receipt::RewardStatsReceipt {
        if self.count == 0 {
            return crate::train_receipt::RewardStatsReceipt::default();
        }
        let specs = [
            ("zero", Some(0.0), Some(0.0)),
            ("tiny", Some(f64::MIN_POSITIVE), Some(1e-6)),
            ("low", Some(1e-6), Some(0.01)),
            ("medium", Some(0.01), Some(0.25)),
            ("high", Some(0.25), Some(1.0)),
            ("extreme", Some(1.0), None),
        ];
        crate::train_receipt::RewardStatsReceipt {
            count: self.count,
            mean: Some(self.sum / self.count as f64),
            stdev: Some((squared_deviation_sum / self.count as f64).sqrt()),
            min: Some(self.min),
            max: Some(self.max),
            group_count: self.group_count,
            all_pass_group_count: self.all_pass_group_count,
            all_fail_group_count: self.all_fail_group_count,
            degenerate_group_count: self.degenerate_group_count,
            group_variance_histogram: specs
                .into_iter()
                .zip(self.variance_histogram_counts)
                .map(|((label, min_inclusive, max_inclusive), count)| {
                    crate::train_receipt::HistogramBucket {
                        label: label.to_string(),
                        min_inclusive,
                        max_inclusive,
                        count,
                    }
                })
                .collect(),
        }
    }
}

/// Validate a streamed GRPO JSONL dataset and training configuration without
/// loading model weights or running forward/backward.
pub fn grpo_dry_run_jsonl(
    dataset_path: &Path,
    config: &GrpoConfig,
    model_config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    adapter_name: &str,
    allow_empty_after_filter: bool,
) -> Result<GrpoDryRunReport> {
    grpo_dry_run_jsonl_with_pass_hook(
        dataset_path,
        config,
        model_config,
        tokenizer,
        adapter_dir,
        adapter_name,
        allow_empty_after_filter,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn grpo_dry_run_jsonl_with_pass_hook(
    dataset_path: &Path,
    config: &GrpoConfig,
    model_config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    adapter_name: &str,
    allow_empty_after_filter: bool,
    mut after_first_pass: Option<&mut dyn FnMut() -> Result<()>>,
) -> Result<GrpoDryRunReport> {
    let run_started = Instant::now();
    let output_dir = adapter_dir.join(adapter_name);
    let receipt_path = output_dir.join(crate::train_receipt::TRAIN_RECEIPT_FILENAME);
    let mut training_data = crate::train_receipt::TrainingDataReceipt {
        source: "jsonl_grpo_groups_dry_run".to_string(),
        path: Some(dataset_path.display().to_string()),
        sha256: None,
    };
    let dataset_source = PinnedGrpoJsonlSource::open(dataset_path);
    let source_sha256 = dataset_source
        .as_ref()
        .map_err(|error| format!("{error:#}"))
        .and_then(|source| source.sha256().map_err(|error| format!("{error:#}")));
    training_data.sha256 = source_sha256.as_ref().ok().cloned();
    let requested_base_adapter_dir = config
        .base_adapter
        .as_deref()
        .map(|name| crate::adapter_shape::resolve_base_adapter_dir(name, adapter_dir));
    let mut data_stats = crate::train_receipt::DataStatsReceipt::default();
    let mut token_counts = crate::train_receipt::TokenCountReceipt::default();
    let mut reward_stats = crate::train_receipt::RewardStatsReceipt::default();
    let mut phase_timings = GrpoBenchmarkTimings::default();
    let mut dynamic_groups_filtered = 0usize;
    let mut alpha_over_rank = None;
    let mut base_adapter_dir = None;

    let result = (|| -> Result<GrpoDryRunReport> {
        config
            .validate_policy_config()
            .map_err(|e| anyhow::anyhow!("GRPO policy config: {e}"))?;
        let ratio = crate::lora_scaling::validate_lora_scaling(
            config.lora_rank,
            config.lora_alpha,
            config.allow_high_lora_scale,
        )?;
        alpha_over_rank = Some(ratio);
        base_adapter_dir = resolve_and_validate_base_adapter(
            config.base_adapter.as_deref(),
            adapter_dir,
            model_config,
            config.lora_rank,
            config.allow_adapter_shape_conversion,
        )?;

        let dataset_source = dataset_source
            .as_ref()
            .map_err(|error| anyhow::anyhow!("{error:#}"))?;
        let source_sha256 = source_sha256
            .as_ref()
            .map_err(|error| anyhow::anyhow!("hash GRPO JSONL dataset: {error}"))?;
        let filter_enabled = reward_filter_enabled(config);
        let mut reward_accumulator = DryRunRewardStatsAccumulator::default();
        let mut reward_filter_inputs = Vec::new();
        let first_scan = scan_pinned_grpo_jsonl(
            dataset_source,
            model_config.num_layers,
            filter_enabled,
            "dry-run reward preflight",
            |line_no, source_index, group| {
                data_stats.groups_read = source_index;
                data_stats.completions_read = data_stats
                    .completions_read
                    .checked_add(group.completions.len())
                    .context("GRPO dry-run completion count overflow")?;
                let reward_variance =
                    reward_accumulator.observe_group(group, config.reward_saturation_threshold)?;
                if filter_enabled {
                    reward_filter_inputs
                        .try_reserve(1)
                        .context("reserve bounded GRPO dry-run reward filter input")?;
                    reward_filter_inputs.push(RewardFilterInputGroup {
                        id: format!("line:{line_no}"),
                        source_index,
                        source_line: Some(line_no),
                        reward_variance,
                    });
                }
                Ok(())
            },
        )?;
        anyhow::ensure!(
            first_scan.groups == data_stats.groups_read
                && first_scan.completions == data_stats.completions_read,
            "GRPO dry-run reward preflight count mismatch"
        );
        if let Some(hook) = after_first_pass.take() {
            hook()?;
        }
        anyhow::ensure!(
            dataset_source.sha256()? == source_sha256.as_str(),
            "GRPO JSONL dataset changed after dry-run reward preflight"
        );

        let reward_mean = reward_accumulator.mean();
        let mut squared_deviation_sum = 0.0;
        let variance_scan = scan_pinned_grpo_jsonl(
            dataset_source,
            model_config.num_layers,
            filter_enabled,
            "dry-run reward variance pass",
            |_line_no, _source_index, group| {
                if let Some(mean) = reward_mean {
                    for completion in &group.completions {
                        let centered = completion.reward - mean;
                        squared_deviation_sum += centered * centered;
                    }
                }
                Ok(())
            },
        )?;
        anyhow::ensure!(
            variance_scan == first_scan && dataset_source.sha256()? == source_sha256.as_str(),
            "GRPO JSONL dataset changed during dry-run reward variance pass"
        );
        reward_stats = reward_accumulator.finish(squared_deviation_sum);
        crate::train_receipt::warn_reward_diagnostics(
            "grpo_dry_run",
            adapter_name,
            &reward_stats,
            config.reward_saturation_threshold,
            config.reward_low_variance_threshold,
        );
        let reward_filter_plan = build_reward_filter_plan(
            config,
            &output_dir,
            "jsonl_grpo_groups_dry_run",
            reward_filter_inputs,
        )?;
        if let Some(plan) = reward_filter_plan.as_ref() {
            record_reward_filter_plan(&mut data_stats, plan);
            data_stats.groups_filtered = data_stats
                .groups_filtered
                .checked_add(plan.groups_dropped)
                .context("GRPO dry-run filtered group count overflow")?;
            tracing::info!(
                kept = plan.groups_kept,
                dropped = plan.groups_dropped,
                sidecar = %plan.sidecar_path.display(),
                "GRPO dry-run reward variance filter applied"
            );
            if let Some(reason) = plan.failure_reason.as_ref() {
                anyhow::bail!("{reason}");
            }
        }

        let mut processed_groups = 0usize;
        let mut processed_completions = 0usize;
        let validation_scan = scan_pinned_grpo_jsonl(
            dataset_source,
            model_config.num_layers,
            filter_enabled,
            "dry-run token and mask validation",
            |line_no, _source_index, group| {
                if let Some(plan) = reward_filter_plan.as_ref() {
                    if !plan.keeps_source_line(line_no) || plan.skip_training {
                        return Ok(());
                    }
                }
                if config.dynamic_sampling && is_degenerate_grpo_group(group) {
                    dynamic_groups_filtered = dynamic_groups_filtered
                        .checked_add(1)
                        .context("GRPO dry-run dynamic filter count overflow")?;
                    data_stats.groups_filtered = data_stats
                        .groups_filtered
                        .checked_add(1)
                        .context("GRPO dry-run filtered group count overflow")?;
                    return Ok(());
                }

                let group_idx = processed_groups
                    .checked_add(1)
                    .context("GRPO dry-run processed group count overflow")?;
                let mask_cfg = crate::trajectory_mask::MaskConfig::from_grpo_config(config);
                let tgroup = tokenize_grpo_group_timed(
                    group,
                    tokenizer,
                    &mask_cfg,
                    Some(&mut phase_timings),
                )
                .with_context(|| {
                    format!("tokenize GRPO dry-run group {group_idx} at line {line_no}")
                })?;
                validate_tokenized_behavior_policy(&tgroup, config.behavior_policy).with_context(
                    || format!("validate GRPO dry-run group {group_idx} behavior provenance"),
                )?;
                validate_grpo_dry_run_masks(&tgroup, group_idx, line_no)?;
                let group_counts = token_counts_for_grpo_groups(std::slice::from_ref(&tgroup));
                token_counts.add_from(&group_counts);
                processed_groups = group_idx;
                processed_completions = processed_completions
                    .checked_add(tgroup.completions.len())
                    .context("GRPO dry-run processed completion count overflow")?;
                Ok(())
            },
        )?;
        anyhow::ensure!(
            validation_scan == first_scan && dataset_source.sha256()? == source_sha256.as_str(),
            "GRPO JSONL dataset changed during dry-run token validation"
        );

        data_stats.groups_trained = processed_groups;
        data_stats.completions_trained = processed_completions;
        let reward_filter_skipped = reward_filter_plan
            .as_ref()
            .is_some_and(|plan| plan.skip_training);
        if processed_groups == 0 && !allow_empty_after_filter && !reward_filter_skipped {
            anyhow::bail!(
                "GRPO dry run: zero valid GRPO groups after filtering in {}; pass --allow-empty-dry-run to permit this",
                dataset_path.display()
            );
        }
        if processed_groups > 0 {
            anyhow::ensure!(
                processed_completions > 0,
                "GRPO dry run: no valid GRPO completions in {}",
                dataset_path.display()
            );
            anyhow::ensure!(
                token_counts.action_tokens > 0,
                "GRPO dry run: dataset has no action tokens after mask construction"
            );
        }

        Ok(GrpoDryRunReport {
            adapter_dir: output_dir.clone(),
            receipt_path: receipt_path.clone(),
            base_adapter_dir: base_adapter_dir.clone(),
            alpha_over_rank,
            data: data_stats.clone(),
            rewards: reward_stats.clone(),
            token_counts: token_counts.clone(),
            dynamic_groups_filtered,
        })
    })();

    let status_error = result.as_ref().err().map(|err| format!("{err:#}"));
    let receipt = build_grpo_train_receipt(
        adapter_name,
        model_config,
        tokenizer,
        None,
        None,
        None,
        config,
        config.seed,
        alpha_over_rank,
        base_adapter_dir
            .as_deref()
            .or(requested_base_adapter_dir.as_deref()),
        &output_dir,
        training_data,
        data_stats,
        reward_stats,
        token_counts,
        phase_timings.to_receipt(),
        crate::train_receipt::EchoActivityMetrics::default(),
        run_started.elapsed().as_millis() as u64,
        dynamic_groups_filtered,
        None,
        Vec::new(),
        None,
        status_error,
    );
    let receipt_write = receipt
        .write_to_adapter_dir(&output_dir)
        .with_context(|| format!("write GRPO dry-run receipt {}", receipt_path.display()));

    match (result, receipt_write) {
        (Ok(report), Ok(_)) => Ok(report),
        (Ok(_), Err(err)) => Err(err),
        (Err(err), Ok(_)) => Err(crate::train_receipt::annotate_training_error(err)),
        (Err(err), Err(write_err)) => {
            tracing::warn!(
                adapter = adapter_name,
                error = %write_err,
                "failed to write GRPO dry-run receipt after validation failure"
            );
            Err(crate::train_receipt::annotate_training_error(err))
        }
    }
}

/// Maximum host-memory charge for the streamed GRPO preflight itself. Server
/// admission additionally charges the immutable disk snapshot against its
/// process-wide prepared-data cap.
pub const MAX_STREAMED_GRPO_PREFLIGHT_HOST_BYTES: u64 = 256 * 1024 * 1024;
pub const MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES: u64 = 16 * 1024 * 1024;
pub const MAX_STREAMED_GRPO_PREFLIGHT_GROUPS: usize = 1_000_000;
pub const MAX_STREAMED_GRPO_PREFLIGHT_COMPLETIONS: usize = 16_000_000;

/// Conservative host peak for streamed GRPO planning.
///
/// The charge covers the compact trainable entry, reward/filter decisions and
/// sidecar serialization overlap, incremental identity hashing, one row's JSON
/// plus tokenization transients, and one group's checkpoint-boundary scratch.
/// Every operation is checked so adversarial counts fail before allocation.
pub fn streamed_grpo_preflight_host_bytes(
    groups: usize,
    completions: usize,
    max_row_bytes: u64,
    model_num_layers: usize,
    reward_filter_enabled: bool,
) -> Result<u64> {
    const BASE_BYTES: u64 = 256 * 1024;
    const TRAINABLE_PLAN_BYTES_PER_GROUP: u64 = 384;
    const FILTER_AND_SIDECAR_BYTES_PER_GROUP: u64 = 1_536;
    const COMPLETION_DIAGNOSTIC_BYTES: u64 = 8;
    const MAX_ROW_TRANSIENT_MULTIPLIER: u64 = 12;
    const CHECKPOINT_SCRATCH_BYTES_PER_LAYER: u64 = 32;

    anyhow::ensure!(
        groups <= MAX_STREAMED_GRPO_PREFLIGHT_GROUPS,
        "streamed GRPO preflight has {groups} groups; maximum is {MAX_STREAMED_GRPO_PREFLIGHT_GROUPS}"
    );
    anyhow::ensure!(
        completions <= MAX_STREAMED_GRPO_PREFLIGHT_COMPLETIONS,
        "streamed GRPO preflight has {completions} completions; maximum is {MAX_STREAMED_GRPO_PREFLIGHT_COMPLETIONS}"
    );
    anyhow::ensure!(
        max_row_bytes <= MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES,
        "streamed GRPO preflight row has {max_row_bytes} bytes; maximum is {}",
        MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES
    );

    let per_group = TRAINABLE_PLAN_BYTES_PER_GROUP
        .checked_add(if reward_filter_enabled {
            FILTER_AND_SIDECAR_BYTES_PER_GROUP
        } else {
            0
        })
        .context("streamed GRPO per-group preflight charge overflow")?;
    let group_bytes = u64::try_from(groups)
        .context("streamed GRPO group count exceeds u64")?
        .checked_mul(per_group)
        .context("streamed GRPO group-plan charge overflow")?;
    let completion_bytes = u64::try_from(completions)
        .context("streamed GRPO completion count exceeds u64")?
        .checked_mul(COMPLETION_DIAGNOSTIC_BYTES)
        .context("streamed GRPO completion charge overflow")?;
    let row_bytes = max_row_bytes
        .checked_mul(MAX_ROW_TRANSIENT_MULTIPLIER)
        .context("streamed GRPO row-transient charge overflow")?;
    let checkpoint_bytes = u64::try_from(model_num_layers.max(1))
        .context("streamed GRPO model layer count exceeds u64")?
        .checked_mul(CHECKPOINT_SCRATCH_BYTES_PER_LAYER)
        .context("streamed GRPO checkpoint scratch charge overflow")?;
    let total = BASE_BYTES
        .checked_add(group_bytes)
        .and_then(|bytes| bytes.checked_add(completion_bytes))
        .and_then(|bytes| bytes.checked_add(row_bytes))
        .and_then(|bytes| bytes.checked_add(checkpoint_bytes))
        .context("streamed GRPO preflight host-memory charge overflow")?;
    anyhow::ensure!(
        total <= MAX_STREAMED_GRPO_PREFLIGHT_HOST_BYTES,
        "streamed GRPO preflight projects {total} host bytes; maximum is {MAX_STREAMED_GRPO_PREFLIGHT_HOST_BYTES}"
    );
    Ok(total)
}

/// Disk-backed GRPO source pinned to one open file identity.
///
/// Path-based entry points construct this immediately after opening their
/// input. Server callers can instead pass an already verified handle, so later
/// preflight, resume, epoch, and receipt reads cannot be redirected by an
/// atomic pathname replacement. Reader clones keep the corpus streamed from
/// disk; the source never materializes the whole JSONL in memory.
#[derive(Debug)]
pub struct PinnedGrpoJsonlSource {
    file: std::fs::File,
    display_path: PathBuf,
    // `File::try_clone` shares the cursor on Unix. The streamed implementation
    // drops each phase reader before rewinding the next one; making this type
    // !Sync prevents concurrent callers from violating that order.
    _not_sync: std::marker::PhantomData<std::cell::Cell<()>>,
}

impl PinnedGrpoJsonlSource {
    pub fn open(path: &Path) -> Result<Self> {
        #[cfg(unix)]
        let file = {
            use std::os::unix::fs::OpenOptionsExt as _;

            std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
                .open(path)
        };
        #[cfg(not(unix))]
        let file = std::fs::File::open(path);
        let file = file.with_context(|| format!("open GRPO JSONL dataset {}", path.display()))?;
        Self::from_file(file, path.to_path_buf())
    }

    pub fn from_file(file: std::fs::File, display_path: PathBuf) -> Result<Self> {
        let metadata = file
            .metadata()
            .with_context(|| format!("stat GRPO JSONL dataset {}", display_path.display()))?;
        anyhow::ensure!(
            metadata.is_file(),
            "GRPO JSONL dataset {} is not a regular file",
            display_path.display()
        );
        anyhow::ensure!(
            metadata.len() <= crate::HF_TRL_GRPO_MAX_DATASET_BYTES,
            "GRPO JSONL dataset {} has {} bytes; maximum is {}",
            display_path.display(),
            metadata.len(),
            crate::HF_TRL_GRPO_MAX_DATASET_BYTES
        );
        Ok(Self {
            file,
            display_path,
            _not_sync: std::marker::PhantomData,
        })
    }

    pub fn display_path(&self) -> &Path {
        &self.display_path
    }

    pub fn len(&self) -> Result<u64> {
        self.file
            .metadata()
            .with_context(|| format!("stat GRPO JSONL dataset {}", self.display_path.display()))
            .map(|metadata| metadata.len())
    }

    pub fn metadata(&self) -> Result<std::fs::Metadata> {
        self.file
            .metadata()
            .with_context(|| format!("stat GRPO JSONL dataset {}", self.display_path.display()))
    }

    pub fn sha256(&self) -> Result<String> {
        use sha2::{Digest, Sha256};
        use std::io::Read as _;

        let mut file = self.reader_from_start()?;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let read = file.read(&mut buffer).with_context(|| {
                format!(
                    "read GRPO JSONL dataset {} for sha256",
                    self.display_path.display()
                )
            })?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        let digest: [u8; 32] = hasher.finalize().into();
        Ok(crate::train_receipt::format_sha256_digest(&digest))
    }

    fn reader_from_start(&self) -> Result<std::fs::File> {
        use std::io::{Seek as _, SeekFrom};

        let mut file = self.file.try_clone().with_context(|| {
            format!(
                "clone pinned GRPO JSONL handle {}",
                self.display_path.display()
            )
        })?;
        file.seek(SeekFrom::Start(0)).with_context(|| {
            format!(
                "rewind pinned GRPO JSONL handle {}",
                self.display_path.display()
            )
        })?;
        Ok(file)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GrpoJsonlGradientCheckpointPlan {
    config: CheckpointConfig,
    boundaries_sha256: String,
}

struct Sha256Writer<'a>(&'a mut sha2::Sha256);

impl std::io::Write for Sha256Writer<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        use sha2::Digest as _;
        self.0.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

struct StreamingJsonArraySha256 {
    hasher: sha2::Sha256,
    has_items: bool,
}

impl StreamingJsonArraySha256 {
    fn new() -> Self {
        use sha2::Digest as _;
        let mut hasher = sha2::Sha256::new();
        hasher.update(b"[");
        Self {
            hasher,
            has_items: false,
        }
    }

    fn push<T: serde::Serialize>(&mut self, value: &T) -> Result<()> {
        use sha2::Digest as _;
        if self.has_items {
            self.hasher.update(b",");
        }
        serde_json::to_writer(Sha256Writer(&mut self.hasher), value)
            .context("serialize streamed GRPO preflight identity item")?;
        self.has_items = true;
        Ok(())
    }

    fn finish(mut self) -> String {
        use sha2::Digest as _;
        self.hasher.update(b"]");
        let digest: [u8; 32] = self.hasher.finalize().into();
        crate::train_receipt::format_sha256_digest(&digest)
    }
}

#[derive(serde::Serialize)]
struct GrpoJsonlOrderIdentity<'a> {
    source_index: usize,
    source_line: usize,
    byte_offset: u64,
    next_byte_offset: u64,
    line_sha256: &'a str,
    completions: usize,
    token_counts: &'a crate::train_receipt::TokenCountReceipt,
    max_seq_len: usize,
}

#[derive(serde::Serialize)]
struct GrpoJsonlGradientIdentity<'a> {
    source_index: usize,
    source_line: usize,
    max_seq_len: usize,
    enabled: bool,
    num_segments: usize,
    auto_configured: bool,
    boundaries: &'a Option<Vec<(usize, usize)>>,
}

#[derive(Debug, Clone)]
struct GrpoJsonlTrainablePlanEntry {
    source_index: usize,
    source_line: usize,
    byte_offset: u64,
    next_byte_offset: u64,
    line_sha256: String,
    completions: usize,
    token_counts: crate::train_receipt::TokenCountReceipt,
    max_seq_len: usize,
    gradient_checkpoint: GrpoJsonlGradientCheckpointPlan,
}

#[derive(Debug)]
struct GrpoJsonlPreflightPlan {
    total_bytes: u64,
    total_lines: usize,
    trainable: Vec<GrpoJsonlTrainablePlanEntry>,
    planned_completions: usize,
    planned_token_counts: crate::train_receipt::TokenCountReceipt,
    data_stats: crate::train_receipt::DataStatsReceipt,
    reward_stats: crate::train_receipt::RewardStatsReceipt,
    dynamic_groups_filtered: usize,
    trainable_order_sha256: String,
    gradient_checkpoint_plan_sha256: String,
    skip_training: bool,
}

impl GrpoJsonlPreflightPlan {
    fn expected_cursor(&self, global_step: usize) -> Result<(u64, usize)> {
        anyhow::ensure!(
            global_step <= self.trainable.len(),
            "streamed GRPO resume cursor {global_step} exceeds {} trainable groups",
            self.trainable.len()
        );
        Ok(if global_step == 0 {
            (0, 0)
        } else {
            let previous = &self.trainable[global_step - 1];
            (previous.next_byte_offset, previous.source_line)
        })
    }

    fn prefix_diagnostics(
        &self,
        global_step: usize,
    ) -> Result<(usize, crate::train_receipt::TokenCountReceipt)> {
        anyhow::ensure!(
            global_step <= self.trainable.len(),
            "streamed GRPO diagnostic prefix {global_step} exceeds {} groups",
            self.trainable.len()
        );
        let mut completions = 0usize;
        let mut token_counts = crate::train_receipt::TokenCountReceipt::default();
        for entry in &self.trainable[..global_step] {
            completions = completions.saturating_add(entry.completions);
            token_counts.add_from(&entry.token_counts);
        }
        Ok((completions, token_counts))
    }
}

fn grpo_checkpoint_static_data_stats(
    mut stats: crate::train_receipt::DataStatsReceipt,
) -> crate::train_receipt::DataStatsReceipt {
    stats.groups_trained = 0;
    stats.completions_trained = 0;
    // This is a publication location, not training state. A resumed server
    // job intentionally uses a new staging directory and rewrites the same
    // deterministic sidecar there.
    stats.reward_filter_sidecar = None;
    stats
}

#[allow(clippy::too_many_arguments)]
fn build_grpo_jsonl_preflight_plan(
    dataset_source: &PinnedGrpoJsonlSource,
    config: &GrpoConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    output_dir: &Path,
    adapter_name: &str,
    device: &Device,
    activation_bytes_per_elem: usize,
    runtime: &crate::TrainingRuntimeContext,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<GrpoJsonlPreflightPlan> {
    use std::io::{BufRead, BufReader, Read as _};

    let dataset_path = dataset_source.display_path();
    let file = dataset_source.reader_from_start()?;
    let total_bytes = dataset_source.len()?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut line_no = 0usize;
    let mut bytes_read = 0u64;
    let mut source_index = 0usize;
    let mut max_row_bytes = 0u64;
    let mut data_stats = crate::train_receipt::DataStatsReceipt::default();
    let filter_enabled = reward_filter_enabled(config);
    let mut reward_stats_accumulator = StreamedRewardStatsAccumulator::default();
    let mut reward_filter_inputs = Vec::new();

    loop {
        line.clear();
        let read = (&mut reader)
            .take(MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES + 1)
            .read_line(&mut line)
            .with_context(|| {
                format!(
                    "read GRPO JSONL dataset {} line {} during preflight",
                    dataset_path.display(),
                    line_no + 1
                )
            })?;
        if read == 0 {
            break;
        }
        line_no = line_no
            .checked_add(1)
            .context("streamed GRPO preflight line count overflow")?;
        anyhow::ensure!(
            line.len() as u64 <= MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES,
            "GRPO JSONL line {line_no} exceeds the {} byte streamed preflight row limit",
            MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES
        );
        max_row_bytes = max_row_bytes.max(line.len() as u64);
        bytes_read = bytes_read
            .checked_add(read as u64)
            .context("streamed GRPO preflight byte count overflow")?;
        streamed_grpo_preflight_host_bytes(
            data_stats.groups_read,
            data_stats.completions_read,
            max_row_bytes,
            model_config.num_layers,
            filter_enabled,
        )
        .with_context(|| format!("bound streamed GRPO preflight before parsing line {line_no}"))?;
        let Some(group) = parse_grpo_jsonl_group_line(&line, line_no)? else {
            continue;
        };
        validate_grpo_trajectory_roles(&group, line_no)?;
        anyhow::ensure!(
            !group.completions.is_empty()
                && group.completions.len() <= crate::HF_TRL_GRPO_MAX_COMPLETIONS_PER_GROUP,
            "GRPO JSONL line {line_no} must contain 1..={} completions",
            crate::HF_TRL_GRPO_MAX_COMPLETIONS_PER_GROUP
        );
        source_index = source_index
            .checked_add(1)
            .context("streamed GRPO source index overflow")?;
        data_stats.groups_read = data_stats
            .groups_read
            .checked_add(1)
            .context("streamed GRPO group count overflow")?;
        data_stats.completions_read = data_stats
            .completions_read
            .checked_add(group.completions.len())
            .context("streamed GRPO completion count overflow")?;
        streamed_grpo_preflight_host_bytes(
            data_stats.groups_read,
            data_stats.completions_read,
            max_row_bytes,
            model_config.num_layers,
            filter_enabled,
        )
        .with_context(|| format!("bound streamed GRPO preflight metadata at line {line_no}"))?;
        let reward_variance = reward_stats_accumulator.observe_group(
            group
                .completions
                .iter()
                .map(|completion| &completion.reward),
            config.reward_saturation_threshold,
        );
        if filter_enabled {
            reward_filter_inputs.push(RewardFilterInputGroup {
                id: format!("line:{line_no}"),
                source_index,
                source_line: Some(line_no),
                reward_variance,
            });
        }
    }
    anyhow::ensure!(
        bytes_read == total_bytes,
        "GRPO JSONL dataset length changed during preflight: expected {total_bytes}, read {bytes_read}"
    );

    let preflight_host_bytes = streamed_grpo_preflight_host_bytes(
        data_stats.groups_read,
        data_stats.completions_read,
        max_row_bytes,
        model_config.num_layers,
        filter_enabled,
    )?;
    tracing::debug!(
        preflight_host_bytes,
        groups = data_stats.groups_read,
        completions = data_stats.completions_read,
        max_row_bytes,
        "bounded streamed GRPO preflight host plan"
    );
    let reward_stats = reward_stats_accumulator.finish();
    crate::train_receipt::warn_reward_diagnostics(
        "streamed_grpo_startup",
        adapter_name,
        &reward_stats,
        config.reward_saturation_threshold,
        config.reward_low_variance_threshold,
    );
    let reward_filter_plan = build_reward_filter_plan(
        config,
        output_dir,
        "jsonl_grpo_groups",
        reward_filter_inputs,
    )?;
    if let Some(plan) = reward_filter_plan.as_ref() {
        record_reward_filter_plan(&mut data_stats, plan);
        data_stats.groups_filtered = plan.groups_dropped;
        tracing::info!(
            kept = plan.groups_kept,
            dropped = plan.groups_dropped,
            sidecar = %plan.sidecar_path.display(),
            "streamed GRPO reward variance filter applied"
        );
        if let Some(reason) = plan.failure_reason.as_ref() {
            anyhow::bail!("{reason}");
        }
    }
    let skip_training = reward_filter_plan
        .as_ref()
        .is_some_and(|plan| plan.skip_training);
    if skip_training {
        return Ok(GrpoJsonlPreflightPlan {
            total_bytes,
            total_lines: line_no,
            trainable: Vec::new(),
            planned_completions: 0,
            planned_token_counts: crate::train_receipt::TokenCountReceipt::default(),
            data_stats,
            reward_stats,
            dynamic_groups_filtered: 0,
            trainable_order_sha256: StreamingJsonArraySha256::new().finish(),
            gradient_checkpoint_plan_sha256: StreamingJsonArraySha256::new().finish(),
            skip_training,
        });
    }

    drop(reader);
    let file = dataset_source.reader_from_start()?;
    let mut reader = BufReader::new(file);
    let mut line_no = 0usize;
    let mut bytes_read = 0u64;
    let mut source_index = 0usize;
    let mut dynamic_groups_filtered = 0usize;
    let mut trainable = Vec::new();
    trainable
        .try_reserve_exact(data_stats.groups_read)
        .context("reserve bounded streamed GRPO trainable plan")?;
    let mut planned_completions = 0usize;
    let mut planned_token_counts = crate::train_receipt::TokenCountReceipt::default();
    let mut order_identity = StreamingJsonArraySha256::new();
    let mut gradient_identity = StreamingJsonArraySha256::new();

    loop {
        line.clear();
        let byte_offset = bytes_read;
        let read = (&mut reader)
            .take(MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES + 1)
            .read_line(&mut line)
            .with_context(|| {
                format!(
                    "read GRPO JSONL dataset {} line {} during trainable preflight",
                    dataset_path.display(),
                    line_no + 1
                )
            })?;
        if read == 0 {
            break;
        }
        line_no = line_no
            .checked_add(1)
            .context("streamed GRPO trainable-pass line count overflow")?;
        anyhow::ensure!(
            line.len() as u64 <= MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES,
            "GRPO JSONL line {line_no} exceeds the {} byte streamed preflight row limit",
            MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES
        );
        bytes_read = bytes_read
            .checked_add(read as u64)
            .context("streamed GRPO trainable-pass byte count overflow")?;
        let Some(group) = parse_grpo_jsonl_group_line(&line, line_no)? else {
            continue;
        };
        source_index = source_index
            .checked_add(1)
            .context("streamed GRPO source index overflow")?;
        if reward_filter_plan
            .as_ref()
            .is_some_and(|plan| !plan.keeps_source_line(line_no))
        {
            continue;
        }
        if config.dynamic_sampling && is_degenerate_grpo_group(&group) {
            dynamic_groups_filtered = dynamic_groups_filtered.saturating_add(1);
            continue;
        }

        let mask_cfg = crate::trajectory_mask::MaskConfig::from_grpo_config(config);
        let tokenized = tokenize_grpo_group_timed(&group, tokenizer, &mask_cfg, None)
            .with_context(|| {
                format!("preflight GRPO JSONL group {source_index} at line {line_no}")
            })?;
        validate_tokenized_behavior_policy(&tokenized, config.behavior_policy).with_context(
            || {
                format!(
                    "validate preflight GRPO JSONL group {source_index} at line {line_no} behavior provenance"
                )
            },
        )?;
        let token_counts = token_counts_for_grpo_groups(std::slice::from_ref(&tokenized));
        let completions = tokenized.completions.len();
        let max_seq_len = tokenized
            .completions
            .iter()
            .map(|completion| completion.input_ids.len())
            .max()
            .unwrap_or(0);
        let checkpoint_config = checkpoint_config_for_training_step(
            weights,
            device,
            config.grad_checkpoint_segments,
            model_config.num_layers,
            max_seq_len,
            model_config.hidden_size,
            model_config.intermediate_size,
            model_config.vocab_size,
            2,
            activation_bytes_per_elem,
            runtime,
        );
        let boundaries = checkpoint_segments_for_config(
            weights,
            device,
            max_seq_len,
            checkpoint_config,
            streaming_prefill,
        );
        let line_sha256 = crate::train_receipt::sha256_bytes(line.as_bytes());
        order_identity.push(&GrpoJsonlOrderIdentity {
            source_index,
            source_line: line_no,
            byte_offset,
            next_byte_offset: bytes_read,
            line_sha256: &line_sha256,
            completions,
            token_counts: &token_counts,
            max_seq_len,
        })?;
        gradient_identity.push(&GrpoJsonlGradientIdentity {
            source_index,
            source_line: line_no,
            max_seq_len,
            enabled: checkpoint_config.enabled,
            num_segments: checkpoint_config.num_segments,
            auto_configured: checkpoint_config.auto_configured,
            boundaries: &boundaries,
        })?;
        let boundaries_sha256 = crate::train_receipt::sha256_json_serializable(&boundaries)
            .context("hash streamed GRPO checkpoint boundaries")?;
        planned_completions = planned_completions
            .checked_add(completions)
            .context("streamed GRPO planned completion count overflow")?;
        planned_token_counts.add_from(&token_counts);
        trainable.push(GrpoJsonlTrainablePlanEntry {
            source_index,
            source_line: line_no,
            byte_offset,
            next_byte_offset: bytes_read,
            line_sha256,
            completions,
            token_counts,
            max_seq_len,
            gradient_checkpoint: GrpoJsonlGradientCheckpointPlan {
                config: checkpoint_config,
                boundaries_sha256,
            },
        });
    }
    anyhow::ensure!(
        bytes_read == total_bytes,
        "GRPO JSONL dataset length changed between preflight passes: expected {total_bytes}, read {bytes_read}"
    );
    anyhow::ensure!(
        !trainable.is_empty(),
        "grpo_train_jsonl: no valid GRPO groups in {}",
        dataset_path.display()
    );
    anyhow::ensure!(
        planned_completions > 0 && planned_token_counts.action_tokens > 0,
        "grpo_train_jsonl: trainable groups contain no completions or action tokens"
    );
    data_stats.groups_filtered = data_stats
        .reward_groups_filtered
        .saturating_add(dynamic_groups_filtered);

    Ok(GrpoJsonlPreflightPlan {
        total_bytes,
        total_lines: line_no,
        trainable,
        planned_completions,
        planned_token_counts,
        data_stats,
        reward_stats,
        dynamic_groups_filtered,
        trainable_order_sha256: order_identity.finish(),
        gradient_checkpoint_plan_sha256: gradient_identity.finish(),
        skip_training,
    })
}

/// Stream GRPO training from a JSONL dataset path through the kt-native route.
///
/// Each non-empty line must be one [`GrpoGroup`]. Unlike [`grpo_train`], this
/// path does not retain all parsed or tokenized groups before training.
pub fn grpo_train_jsonl(
    dataset_path: &Path,
    config: &GrpoConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    replay_ctx: Option<ReplayContext>,
) -> Result<PathBuf> {
    grpo_train_jsonl_to(
        dataset_path,
        config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        adapter_dir,
        adapter_name,
        progress_cb,
        replay_ctx,
    )
}

/// Staged-output variant of [`grpo_train_jsonl`].
#[allow(clippy::too_many_arguments)]
pub fn grpo_train_jsonl_to(
    dataset_path: &Path,
    config: &GrpoConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    output_adapter_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    replay_ctx: Option<ReplayContext>,
) -> Result<PathBuf> {
    grpo_train_jsonl_to_with_coordination(
        dataset_path,
        config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        output_adapter_dir,
        adapter_name,
        progress_cb,
        replay_ctx,
        None,
    )
}

/// Streaming staged-output GRPO with bounded server GPU ownership.
#[allow(clippy::too_many_arguments)]
pub fn grpo_train_jsonl_to_with_coordination(
    dataset_path: &Path,
    config: &GrpoConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    output_adapter_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    replay_ctx: Option<ReplayContext>,
    gpu_step_coordination: Option<GpuStepCoordination>,
) -> Result<PathBuf> {
    grpo_train_jsonl_to_with_checkpoint_root(
        dataset_path,
        config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        output_adapter_dir,
        output_adapter_dir,
        adapter_name,
        progress_cb,
        replay_ctx,
        gpu_step_coordination,
    )
}

/// Standalone streamed GRPO with a separate durable checkpoint root.
///
/// Server callers should use
/// [`grpo_train_jsonl_to_with_checkpoint_root_and_runtime`] to bind preflight
/// and execution to the same process-lifetime memory configuration.
#[allow(clippy::too_many_arguments)]
pub fn grpo_train_jsonl_to_with_checkpoint_root(
    dataset_path: &Path,
    config: &GrpoConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    output_adapter_dir: &Path,
    checkpoint_output_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    replay_ctx: Option<ReplayContext>,
    gpu_step_coordination: Option<GpuStepCoordination>,
) -> Result<PathBuf> {
    ensure_training_optimizer_device_supported(
        "streamed GRPO",
        weights,
        weights.embed_tokens.device(),
        config.optimizer,
        config.lora_rank,
    )?;
    let runtime =
        crate::standalone_training_runtime_for_weight_device(weights.embed_tokens.device())?;
    grpo_train_jsonl_to_with_checkpoint_root_and_runtime(
        dataset_path,
        config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        output_adapter_dir,
        checkpoint_output_dir,
        adapter_name,
        progress_cb,
        replay_ctx,
        gpu_step_coordination,
        &runtime,
    )
}

/// Server-owned streamed GRPO entry point with immutable runtime inputs.
#[allow(clippy::too_many_arguments)]
pub fn grpo_train_jsonl_to_with_checkpoint_root_and_runtime(
    dataset_path: &Path,
    config: &GrpoConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    output_adapter_dir: &Path,
    checkpoint_output_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    replay_ctx: Option<ReplayContext>,
    gpu_step_coordination: Option<GpuStepCoordination>,
    runtime: &crate::TrainingRuntimeContext,
) -> Result<PathBuf> {
    ensure_training_optimizer_entry_supported(
        "streamed GRPO",
        weights,
        runtime,
        config.optimizer,
        config.lora_rank,
    )?;
    let dataset_source = PinnedGrpoJsonlSource::open(dataset_path)?;
    grpo_train_pinned_jsonl_to_with_checkpoint_root_and_runtime(
        &dataset_source,
        config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        output_adapter_dir,
        checkpoint_output_dir,
        adapter_name,
        progress_cb,
        replay_ctx,
        gpu_step_coordination,
        runtime,
    )
}

/// Streamed GRPO entry point for a caller-pinned file identity.
#[allow(clippy::too_many_arguments)]
pub fn grpo_train_pinned_jsonl_to_with_checkpoint_root_and_runtime(
    dataset_source: &PinnedGrpoJsonlSource,
    config: &GrpoConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    output_adapter_dir: &Path,
    checkpoint_output_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    replay_ctx: Option<ReplayContext>,
    gpu_step_coordination: Option<GpuStepCoordination>,
    runtime: &crate::TrainingRuntimeContext,
) -> Result<PathBuf> {
    use std::io::{BufRead, BufReader, Seek, SeekFrom};

    let dataset_path = dataset_source.display_path();
    let runtime_device = ensure_training_optimizer_entry_supported(
        "streamed GRPO",
        weights,
        runtime,
        config.optimizer,
        config.lora_rank,
    )?;
    crate::ensure_memory_governor_for_runtime(runtime_device, runtime)
        .context("initialize streamed GRPO memory governor")?;
    let run_started = Instant::now();
    anyhow::ensure!(
        config.checkpoint_interval != Some(0),
        "GRPO checkpoint_interval must be greater than zero"
    );
    // Fail fast on compositions the kt-tape path cannot train. The
    // streaming path can't cheaply pre-scan every group for Observation
    // segments, so `no_policy_loss` / reserved-OPD reject here and the
    // echo+env case rejects per-group at mask-construction time, before
    // that group's forward (plus in the dry-run gate below).
    config
        .loss
        .validate_for_kt_tape(false)
        .map_err(|e| anyhow::anyhow!("GRPO loss config: {e}"))?;
    config
        .validate_policy_config()
        .map_err(|e| anyhow::anyhow!("GRPO policy config: {e}"))?;
    let output_dir = output_adapter_dir.join(adapter_name);
    let training_data_sha256 = dataset_source
        .sha256()
        .with_context(|| format!("hash GRPO JSONL dataset {}", dataset_path.display()))?;
    let training_data_checkpoint_sha256 =
        checkpoint_sha256_hex(Some(&training_data_sha256), "GRPO JSONL training data")?;
    let training_data = crate::train_receipt::TrainingDataReceipt {
        source: "jsonl_grpo_groups".to_string(),
        path: Some(dataset_path.display().to_string()),
        sha256: Some(training_data_sha256.clone()),
    };
    let requested_base_adapter_dir = config.base_adapter.as_deref().map(|name| {
        resolve_base_adapter_dir_from_roots(name, adapter_dir, output_adapter_dir, adapter_name)
    });
    let resume_checkpoint = config
        .resume_checkpoint
        .as_deref()
        .map(Path::new)
        .map(crate::checkpoint::load_training_checkpoint)
        .transpose()
        .context("load streamed GRPO resume checkpoint")?;
    let resume_loop_state = resume_checkpoint
        .as_ref()
        .map(load_grpo_checkpoint_loop_state)
        .transpose()?;
    if let Some(checkpoint) = resume_checkpoint.as_ref() {
        anyhow::ensure!(
            checkpoint.manifest.training_kind == crate::checkpoint::TrainingKind::Grpo,
            "resume checkpoint is not a GRPO checkpoint"
        );
        anyhow::ensure!(
            checkpoint.manifest.adapter_name == adapter_name,
            "resume checkpoint adapter {:?} does not match {:?}",
            checkpoint.manifest.adapter_name,
            adapter_name
        );
        anyhow::ensure!(
            checkpoint.manifest.data.source_kind == GrpoCheckpointRoute::Jsonl.source_kind()
                && checkpoint.manifest.data.content_sha256 == training_data_checkpoint_sha256,
            "resume checkpoint streamed GRPO data identity differs from this request"
        );
        anyhow::ensure!(
            resume_loop_state
                .as_ref()
                .is_some_and(|state| state.route == GrpoCheckpointRoute::Jsonl),
            "resume checkpoint was not produced by streamed JSONL GRPO"
        );
    }
    if config.checkpoint_interval.is_some() || resume_checkpoint.is_some() {
        validate_exact_training_provenance(weights)?;
    }
    let resume_init_seed = resume_checkpoint
        .as_ref()
        .map(|checkpoint| {
            let state = checkpoint
                .manifest
                .rng_states
                .get("lora-init")
                .context("streamed GRPO resume checkpoint has no lora-init RNG state")?;
            anyhow::ensure!(
                state.algorithm == "kiln.seeded-lora-init.v1" && state.position == 0,
                "unsupported streamed GRPO lora-init RNG state"
            );
            Ok(state.seed)
        })
        .transpose()?;
    if let (Some(requested), Some(restored)) = (config.seed, resume_init_seed) {
        anyhow::ensure!(
            requested == restored,
            "streamed GRPO resume seed {restored} differs from requested seed {requested}"
        );
    }
    let effective_seed_value = resume_init_seed
        .or(config.seed)
        .unwrap_or_else(rand::random);
    let learning_rate = config.effective_learning_rate();
    let effective_checkpoint_config =
        grpo_checkpoint_effective_config(config, learning_rate, effective_seed_value)?;
    if let Some(checkpoint) = resume_checkpoint.as_ref() {
        anyhow::ensure!(
            checkpoint.manifest.effective_config == effective_checkpoint_config,
            "resume checkpoint effective GRPO configuration differs from this request: checkpoint={}, request={}",
            checkpoint.manifest.effective_config,
            effective_checkpoint_config
        );
    }

    // (#1082) `embed_tokens.device()` is a kt Device; the OPD/GRPO body is now
    // kt-native (kt `Parameter`s, kt AdamW state, kt tape forward/backward), so
    // keep `device` kt downstream. The only candle touch is safetensors adapter
    // I/O, which bridges kt->candle locally inside save/load.
    let device = training_device_for_weights(weights, runtime)?;
    let backend = training_backend_for_device(device)?;
    ensure_tape_forward_backward_supported("streamed GRPO", weights, backend.as_ref())?;
    let training_precision_policy = training_precision_policy_for_backend(backend.as_ref());
    ensure_training_optimizer_supported(
        "streamed GRPO",
        backend.as_ref(),
        config.optimizer,
        weights.embed_tokens.dtype(),
        config.lora_rank,
    )?;
    let streaming_prefill = runtime.resolved_streaming_prefill_policy(device);
    let training_runtime_planning_identity =
        runtime.checkpoint_planning_identity_for_device(device);
    let activation_bytes_per_elem = training_activation_bytes_per_elem_for_policy(
        weights,
        training_precision_policy,
        model_config_has_linear_attention(model_config),
    );
    if let Some(explicit) = config.learning_rate {
        if let Some(warning) = crate::learning_rate_band_warning(
            explicit,
            crate::resolve_learning_rate(&config.optimizer, crate::TrainMode::Grpo),
        ) {
            tracing::warn!(optimizer = ?config.optimizer, "GRPO {warning}");
        }
    }
    tracing::info!(
        dataset = %dataset_path.display(),
        lr = learning_rate,
        kl_coeff = config.kl_coeff,
        clip_epsilon = config.clip_epsilon,
        rank = config.lora_rank,
        alpha = config.lora_alpha,
        adapter_name,
        "starting streamed GRPO training"
    );

    let alpha_over_rank = match crate::lora_scaling::validate_lora_scaling(
        config.lora_rank,
        config.lora_alpha,
        config.allow_high_lora_scale,
    ) {
        Ok(value) => value,
        Err(err) => {
            let message = format!("{err:#}");
            write_grpo_train_receipt_best_effort(
                adapter_name,
                model_config,
                tokenizer,
                weights.base_weight_shard_manifest.as_ref(),
                weights.execution_provenance.as_ref(),
                None,
                config,
                Some(effective_seed_value),
                None,
                requested_base_adapter_dir.as_deref(),
                &output_dir,
                training_data.clone(),
                crate::train_receipt::DataStatsReceipt::default(),
                crate::train_receipt::RewardStatsReceipt::default(),
                crate::train_receipt::TokenCountReceipt::default(),
                crate::train_receipt::TrainingPhaseTimingsReceipt::default(),
                crate::train_receipt::EchoActivityMetrics::default(),
                run_started.elapsed().as_millis() as u64,
                0,
                None,
                Vec::new(),
                None,
                Some(message),
            );
            return Err(crate::train_receipt::annotate_training_error(err));
        }
    };
    tracing::info!(
        alpha_over_rank,
        allow_high_lora_scale = config.allow_high_lora_scale,
        "validated LoRA scaling"
    );

    let base_adapter_result = if resume_checkpoint.is_some() {
        Ok(None)
    } else {
        resolve_and_validate_base_adapter_from_roots(
            config.base_adapter.as_deref(),
            adapter_dir,
            output_adapter_dir,
            adapter_name,
            model_config,
            config.lora_rank,
            config.allow_adapter_shape_conversion,
        )
    };
    let base_adapter_dir = match base_adapter_result {
        Ok(value) => value,
        Err(err) => {
            let message = format!("{err:#}");
            write_grpo_train_receipt_best_effort(
                adapter_name,
                model_config,
                tokenizer,
                weights.base_weight_shard_manifest.as_ref(),
                weights.execution_provenance.as_ref(),
                None,
                config,
                Some(effective_seed_value),
                Some(alpha_over_rank),
                requested_base_adapter_dir.as_deref(),
                &output_dir,
                training_data.clone(),
                crate::train_receipt::DataStatsReceipt::default(),
                crate::train_receipt::RewardStatsReceipt::default(),
                crate::train_receipt::TokenCountReceipt::default(),
                crate::train_receipt::TrainingPhaseTimingsReceipt::default(),
                crate::train_receipt::EchoActivityMetrics::default(),
                run_started.elapsed().as_millis() as u64,
                0,
                None,
                Vec::new(),
                None,
                Some(message),
            );
            return Err(crate::train_receipt::annotate_training_error(err));
        }
    };

    let preflight = match build_grpo_jsonl_preflight_plan(
        dataset_source,
        config,
        model_config,
        weights,
        tokenizer,
        &output_dir,
        adapter_name,
        &device,
        activation_bytes_per_elem,
        runtime,
        streaming_prefill,
    ) {
        Ok(plan) => plan,
        Err(err) => {
            let message = format!("{err:#}");
            write_grpo_train_receipt_best_effort(
                adapter_name,
                model_config,
                tokenizer,
                weights.base_weight_shard_manifest.as_ref(),
                weights.execution_provenance.as_ref(),
                None,
                config,
                Some(effective_seed_value),
                Some(alpha_over_rank),
                base_adapter_dir
                    .as_deref()
                    .or(requested_base_adapter_dir.as_deref()),
                &output_dir,
                training_data.clone(),
                crate::train_receipt::DataStatsReceipt::default(),
                crate::train_receipt::RewardStatsReceipt::default(),
                crate::train_receipt::TokenCountReceipt::default(),
                crate::train_receipt::TrainingPhaseTimingsReceipt::default(),
                crate::train_receipt::EchoActivityMetrics::default(),
                run_started.elapsed().as_millis() as u64,
                0,
                None,
                Vec::new(),
                None,
                Some(message),
            );
            return Err(crate::train_receipt::annotate_training_error(err));
        }
    };
    let post_preflight_sha256 = dataset_source
        .sha256()
        .with_context(|| format!("rehash GRPO JSONL dataset {}", dataset_path.display()))?;
    anyhow::ensure!(
        post_preflight_sha256 == training_data_sha256,
        "GRPO JSONL dataset changed while constructing the exact trainable plan"
    );
    anyhow::ensure!(
        !(preflight.skip_training && resume_checkpoint.is_some()),
        "a streamed GRPO resume checkpoint cannot target a filter-skipped run"
    );

    let current_reward_filter_sidecar = preflight.data_stats.reward_filter_sidecar.clone();
    let mut data_stats = preflight.data_stats.clone();
    let mut token_counts = crate::train_receipt::TokenCountReceipt::default();
    let mut echo_metrics = crate::train_receipt::EchoActivityMetrics::default();
    let reward_stats = preflight.reward_stats.clone();
    let mut lora_grad_norms = crate::train_receipt::LoraGradNormAccumulator::default();
    let mut policy_audit = crate::train_receipt::GrpoPolicyAuditAccumulator::default();
    let mut phase_timings = GrpoBenchmarkTimings::default();
    let mut gpu_writer_timings = GrpoGpuWriterTimings::default();
    let mut dynamic_groups_filtered = preflight.dynamic_groups_filtered;
    if let Some(state) = resume_loop_state.as_ref() {
        anyhow::ensure!(
            grpo_checkpoint_static_data_stats(state.data_stats.clone())
                == grpo_checkpoint_static_data_stats(preflight.data_stats.clone())
                && state.dynamic_groups_filtered as usize == preflight.dynamic_groups_filtered,
            "streamed GRPO resume filtering statistics differ from the current preflight"
        );
        data_stats = state.data_stats.clone();
        data_stats.reward_filter_sidecar = current_reward_filter_sidecar;
        token_counts = state.token_counts.clone();
        echo_metrics = state.echo_metrics.clone();
        lora_grad_norms = state.lora_grad_norms.clone();
        policy_audit = state.policy_audit.clone();
        phase_timings = state.phase_timings.clone();
        gpu_writer_timings = state.gpu_writer_timings.clone();
        dynamic_groups_filtered = state.dynamic_groups_filtered as usize;
    }

    let replay_parent_adapter = resume_checkpoint
        .is_none()
        .then_some(config.base_adapter.as_deref())
        .flatten();
    let (replay_state, effective_seed) = match replay_ctx.as_ref() {
        Some(ctx) => {
            let (state, seed) = open_replay_state_to(
                ctx,
                Some(effective_seed_value),
                replay_parent_adapter,
                adapter_dir,
                output_adapter_dir,
                adapter_name,
            )?;
            anyhow::ensure!(
                seed == effective_seed_value,
                "streamed GRPO replay seed drifted"
            );
            (Some(state), Some(seed))
        }
        None => (None, Some(effective_seed_value)),
    };

    // Upload only after the checkpoint and CPU-only trainable plan have both
    // been validated. This keeps malformed resume requests out of GPU work.
    let resident_weights = run_coordinated_grpo_gpu_phase(
        gpu_step_coordination.as_ref(),
        &*backend,
        &mut gpu_writer_timings,
        "streamed resident model setup",
        || resident_training_weights(weights, &device),
    )?;
    let weights = resident_weights.as_ref().unwrap_or(weights);

    let (mut params, mut opt_state) = run_coordinated_grpo_gpu_phase(
        gpu_step_coordination.as_ref(),
        &*backend,
        &mut gpu_writer_timings,
        "streamed adapter and optimizer setup",
        || {
            let mut params = TrainableLoraParams::initialize_seeded_with_precision_policy(
                model_config,
                weights,
                config.lora_rank,
                config.lora_alpha,
                &device,
                Some(effective_seed_value),
                training_precision_policy,
            )?;
            if let Some(checkpoint) = resume_checkpoint.as_ref() {
                let adapter_path = checkpoint
                    .artifact_path(&checkpoint.manifest.state_files.adapter_parameters)?;
                params.load_checkpoint_parameters(&adapter_path)?;
                tracing::info!(
                    checkpoint = %checkpoint.root.display(),
                    step = checkpoint.manifest.progress.global_step,
                    "restored exact streamed GRPO adapter parameters"
                );
            } else if let Some(base_dir) = base_adapter_dir.as_deref() {
                let n_loaded = params.load_from_safetensors(base_dir, &device)?;
                tracing::info!(
                    base = %base_dir.display(),
                    num_tensors = n_loaded,
                    "loaded base adapter — continuing streamed GRPO from those weights"
                );
            }
            let mut opt_state = make_opt_state(&params, config.optimizer, learning_rate, &device)?;
            if let Some(checkpoint) = resume_checkpoint.as_ref() {
                let state_path = checkpoint
                    .manifest
                    .state_files
                    .optimizer_state
                    .as_deref()
                    .map(|relative| checkpoint.artifact_path(relative))
                    .transpose()?;
                match (opt_state.as_mut(), state_path) {
                    (Some(state), Some(path)) => {
                        let step = u32::try_from(checkpoint.manifest.progress.global_step)
                            .context("streamed GRPO resume optimizer step exceeds u32")?;
                        state.load_checkpoint_state(&params, &path, step)?;
                    }
                    (None, None) => {}
                    (Some(_), None) => anyhow::bail!(
                        "stateful streamed GRPO optimizer checkpoint has no optimizer artifact"
                    ),
                    (None, Some(_)) => anyhow::bail!(
                        "SGD streamed GRPO checkpoint unexpectedly contains optimizer state"
                    ),
                }
            }
            params.register_with_backend(&*backend)?;
            if let Some(state) = opt_state.as_ref() {
                state.register_with_backend(&*backend)?;
            }
            Ok((params, opt_state))
        },
    )?;

    tracing::info!(
        num_vars = params.all_params().len(),
        "initialized streamed GRPO trainable LoRA parameters"
    );
    let lora_grad_index = LoraGradNormIndex::new(&params);

    let ema_refresh_every = if config.kl_penalty_enabled() {
        match &config.kl_reference_policy {
            KlReferencePolicy::Ema { refresh_every, .. } => Some(*refresh_every),
            _ => None,
        }
    } else {
        None
    };
    let checkpoint_descriptor = if preflight.skip_training {
        None
    } else {
        Some(GrpoCheckpointDescriptor {
            route: GrpoCheckpointRoute::Jsonl,
            adapter_name: adapter_name.to_string(),
            effective_config: effective_checkpoint_config.clone(),
            precision_policy: training_checkpoint_precision(&params, opt_state.as_ref())?,
            data: crate::checkpoint::TrainingCheckpointData {
                source_kind: GrpoCheckpointRoute::Jsonl.source_kind().to_string(),
                content_sha256: training_data_checkpoint_sha256.clone(),
                item_count: preflight.trainable.len() as u64,
            },
            init_seed: effective_seed_value,
            optimizer: config.optimizer,
            learning_rate,
            total_steps: preflight.trainable.len(),
            base_model_weights_sha256: weights.source_content_sha256.clone(),
            auxiliary_state: grpo_checkpoint_auxiliary_state(
                GrpoCheckpointRoute::Jsonl,
                model_config,
                tokenizer,
                training_precision_policy,
                weights.source_content_sha256.as_deref(),
                weights.base_weight_shard_manifest.as_ref(),
                weights.execution_provenance.as_ref(),
                BackendIdentity::runtime_name(backend.as_ref()),
                &preflight.trainable_order_sha256,
                &preflight.gradient_checkpoint_plan_sha256,
                &training_runtime_planning_identity,
            ),
            ema_refresh_every,
        })
    };
    if let (Some(descriptor), Some(checkpoint), Some(loop_state)) = (
        checkpoint_descriptor.as_ref(),
        resume_checkpoint.as_ref(),
        resume_loop_state.as_ref(),
    ) {
        descriptor.validate_resume(checkpoint, loop_state)?;
    }

    let mut train_body = || -> Result<(PathBuf, f64)> {
        tracing::info!(
            preflight_max_segments = ?config.grad_checkpoint_segments,
            activation_bytes_per_elem,
            trainable_groups = preflight.trainable.len(),
            total_bytes = preflight.total_bytes,
            "streamed GRPO exact trainable plan validated"
        );

        if preflight.skip_training {
            data_stats.groups_trained = 0;
            data_stats.completions_trained = 0;
            params.save_peft(&output_dir, model_config.num_layers)?;
            tracing::info!(
                adapter = adapter_name,
                path = %output_dir.display(),
                "streamed GRPO reward variance filter skipped training"
            );
            return Ok((output_dir.clone(), 0.0));
        }
        let checkpoint_descriptor = checkpoint_descriptor
            .as_ref()
            .context("streamed GRPO trainable plan has no checkpoint descriptor")?;
        let total_steps = preflight.trainable.len();
        let mut global_step = resume_loop_state
            .as_ref()
            .map_or(0, |state| state.global_step as usize);
        let mut processed_completions = resume_loop_state
            .as_ref()
            .map_or(0, |state| state.processed_completions as usize);
        let mut loss_history = resume_loop_state
            .as_ref()
            .map_or_else(Vec::new, |state| state.loss_history.clone());
        let mut last_loss = resume_loop_state
            .as_ref()
            .and_then(|state| state.last_loss)
            .unwrap_or(0.0);
        let mut last_saved_step = resume_loop_state
            .as_ref()
            .map(|state| state.global_step as usize);
        let (expected_byte_offset, expected_lines_consumed) =
            preflight.expected_cursor(global_step)?;
        let (expected_completions, expected_token_counts) =
            preflight.prefix_diagnostics(global_step)?;
        if let Some(state) = resume_loop_state.as_ref() {
            anyhow::ensure!(
                state.source_byte_offset == Some(expected_byte_offset)
                    && state.source_lines_consumed == Some(expected_lines_consumed as u64),
                "streamed GRPO resume source cursor differs from the exact trainable prefix"
            );
        }
        anyhow::ensure!(
            processed_completions == expected_completions
                && token_counts == expected_token_counts
                && data_stats.groups_trained == global_step
                && data_stats.completions_trained == processed_completions,
            "streamed GRPO resume diagnostics do not match the committed trainable prefix"
        );

        let mut file = dataset_source.reader_from_start()?;
        file.seek(SeekFrom::Start(expected_byte_offset))
            .with_context(|| {
                format!(
                    "seek GRPO JSONL dataset {} to byte {expected_byte_offset}",
                    dataset_path.display()
                )
            })?;
        tracing::info!(
            dataset = %dataset_path.display(),
            total_bytes = preflight.total_bytes,
            byte_offset = expected_byte_offset,
            lines_consumed = expected_lines_consumed,
            global_step,
            total_steps,
            "streamed GRPO data positioned at exact resume cursor"
        );
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        let mut bytes_read = expected_byte_offset;
        let mut line_no = expected_lines_consumed;
        let mut last_ckpt_log_key: Option<(bool, usize)> = None;

        // Phase 3b: maintain an EMA-snapshot LoRA when
        // `KlReferencePolicy::Ema` is configured (see `grpo_train` for the
        // identical pattern; streaming JSONL just iterates one group at a
        // time).
        let mut ema_ref_state = if config.kl_penalty_enabled() {
            match &config.kl_reference_policy {
                KlReferencePolicy::Ema {
                    decay,
                    refresh_every,
                } => {
                    let (snapshot, groups_since_refresh) =
                        if let (Some(checkpoint), Some(loop_state)) =
                            (resume_checkpoint.as_ref(), resume_loop_state.as_ref())
                        {
                            let relative = checkpoint
                                .manifest
                                .state_files
                                .reference_state
                                .as_deref()
                                .context(
                                    "EMA streamed GRPO resume checkpoint has no reference state",
                                )?;
                            let path = checkpoint.artifact_path(relative)?;
                            (
                                load_lora_reference_checkpoint(&path, &params, &device)?,
                                loop_state.ema_groups_since_refresh.context(
                                    "EMA streamed GRPO resume checkpoint has no cadence cursor",
                                )? as usize,
                            )
                        } else {
                            (
                                run_coordinated_grpo_gpu_phase(
                                    gpu_step_coordination.as_ref(),
                                    &*backend,
                                    &mut gpu_writer_timings,
                                    "streamed initial EMA reference snapshot",
                                    || {
                                        lora_snapshot_capture_or_blend(
                                            &params, None, *decay, &device,
                                        )
                                        .context("initial EMA reference snapshot")
                                    },
                                )?,
                                0,
                            )
                        };
                    Some(EmaReferenceState {
                        snapshot,
                        groups_since_refresh,
                        refresh_every: *refresh_every,
                        decay: *decay,
                    })
                }
                _ => None,
            }
        } else {
            None
        };

        loop {
            line.clear();
            let byte_offset = bytes_read;
            let read = reader.read_line(&mut line).with_context(|| {
                format!(
                    "read GRPO JSONL dataset {} line {}",
                    dataset_path.display(),
                    line_no + 1
                )
            })?;
            if read == 0 {
                break;
            }
            line_no = line_no.saturating_add(1);
            bytes_read = bytes_read
                .checked_add(read as u64)
                .context("streamed GRPO training byte count overflow")?;
            let Some(entry) = preflight.trainable.get(global_step) else {
                // Consume trailing blank/filtered lines so the final file
                // cursor and hash still cover the complete source.
                continue;
            };
            if line_no < entry.source_line {
                continue;
            }
            anyhow::ensure!(
                line_no == entry.source_line
                    && byte_offset == entry.byte_offset
                    && bytes_read == entry.next_byte_offset,
                "streamed GRPO source cursor drifted before trainable group {}: expected line {} bytes {}..{}, found line {} bytes {}..{}",
                global_step + 1,
                entry.source_line,
                entry.byte_offset,
                entry.next_byte_offset,
                line_no,
                byte_offset,
                bytes_read
            );
            anyhow::ensure!(
                crate::train_receipt::sha256_bytes(line.as_bytes()) == entry.line_sha256,
                "streamed GRPO trainable line {} changed after preflight",
                entry.source_line
            );
            let group = parse_grpo_jsonl_group_line(&line, line_no)?
                .context("planned streamed GRPO line became blank")?;
            validate_grpo_trajectory_roles(&group, line_no)?;
            let group_number = global_step + 1;
            tracing::info!(
                group = group_number,
                source_index = entry.source_index,
                line = line_no,
                line_bytes = read,
                byte_offset,
                "streamed GRPO tokenize start"
            );
            let tokenize_start = Instant::now();
            let mask_cfg = crate::trajectory_mask::MaskConfig::from_grpo_config(config);
            let tgroup =
                tokenize_grpo_group_timed(&group, tokenizer, &mask_cfg, Some(&mut phase_timings))
                    .with_context(|| {
                    format!(
                        "tokenize GRPO JSONL group {} at line {}",
                        group_number, line_no
                    )
                })?;
            validate_tokenized_behavior_policy(&tgroup, config.behavior_policy).with_context(
                || {
                    format!(
                        "validate GRPO JSONL group {} at line {} behavior provenance",
                        group_number, line_no
                    )
                },
            )?;
            let group_counts = token_counts_for_grpo_groups(std::slice::from_ref(&tgroup));
            tracing::info!(
                group = group_number,
                completions = tgroup.completions.len(),
                action_tokens = group_counts.action_tokens,
                env_tokens = group_counts.env_tokens,
                context_tokens = group_counts.context_tokens,
                elapsed_ms = tokenize_start.elapsed().as_millis() as u64,
                "streamed GRPO tokenize end"
            );

            let group_max_seq_len = tgroup
                .completions
                .iter()
                .map(|completion| completion.input_ids.len())
                .max()
                .unwrap_or(0);
            anyhow::ensure!(
                tgroup.completions.len() == entry.completions
                    && group_counts == entry.token_counts
                    && group_max_seq_len == entry.max_seq_len,
                "streamed GRPO tokenization drifted from preflight at line {}",
                line_no
            );
            let ckpt_config = checkpoint_config_for_training_step(
                weights,
                &device,
                config.grad_checkpoint_segments,
                model_config.num_layers,
                group_max_seq_len,
                model_config.hidden_size,
                model_config.intermediate_size,
                model_config.vocab_size,
                2, // BF16 base weights
                activation_bytes_per_elem,
                runtime,
            );
            let segments = checkpoint_segments_for_config(
                weights,
                &device,
                group_max_seq_len,
                ckpt_config,
                streaming_prefill,
            );
            let segments_sha256 = crate::train_receipt::sha256_json_serializable(&segments)
                .context("hash streamed GRPO runtime checkpoint boundaries")?;
            anyhow::ensure!(
                ckpt_config == entry.gradient_checkpoint.config
                    && segments_sha256 == entry.gradient_checkpoint.boundaries_sha256,
                "streamed GRPO gradient-checkpoint plan drifted at line {}",
                line_no
            );
            let ckpt_log_key = (ckpt_config.enabled, ckpt_config.num_segments);
            if last_ckpt_log_key != Some(ckpt_log_key) {
                if let Some(ref segs) = segments {
                    tracing::info!(
                        group = group_number,
                        max_seq_len = group_max_seq_len,
                        num_segments = segs.len(),
                        preflight_max_segments = ?config.grad_checkpoint_segments,
                        boundaries = ?segs,
                        "streamed GRPO gradient checkpointing enabled for group shape"
                    );
                } else {
                    tracing::info!(
                        group = group_number,
                        max_seq_len = group_max_seq_len,
                        preflight_max_segments = ?config.grad_checkpoint_segments,
                        "streamed GRPO gradient checkpointing disabled for group shape"
                    );
                }
                last_ckpt_log_key = Some(ckpt_log_key);
            }

            let step_report = run_coordinated_grpo_gpu_phase(
                gpu_step_coordination.as_ref(),
                &*backend,
                &mut gpu_writer_timings,
                "streamed optimizer group",
                || {
                    let step_report = train_tokenized_grpo_group_with_grad_norms(
                        &*backend,
                        &tgroup,
                        weights,
                        model_config,
                        &mut params,
                        config,
                        segments.as_deref(),
                        &device,
                        opt_state.as_mut(),
                        &mut lora_grad_norms,
                        &lora_grad_index,
                        &mut policy_audit,
                        ema_ref_state.as_ref().map(|s| &s.snapshot),
                        Some(&mut phase_timings),
                        streaming_prefill,
                    )?;
                    if let Some(state) = ema_ref_state.as_mut() {
                        state.groups_since_refresh += 1;
                        if state.groups_since_refresh >= state.refresh_every {
                            params
                                .sync_to_master(&*backend)
                                .context("sync streamed policy before EMA reference refresh")?;
                            state.snapshot = lora_snapshot_capture_or_blend(
                                &params,
                                Some(&state.snapshot),
                                state.decay,
                                &device,
                            )
                            .context("EMA reference snapshot refresh")?;
                            state.groups_since_refresh = 0;
                            tracing::debug!(
                                group = group_number,
                                refresh_every = state.refresh_every,
                                decay = state.decay,
                                "streamed GRPO EMA reference snapshot refreshed"
                            );
                        }
                    }
                    Ok(step_report)
                },
            )?;
            let avg_group_loss = step_report.loss;
            echo_metrics.observe_env_ce(step_report.echo_env_ce);
            anyhow::ensure!(
                avg_group_loss.is_finite(),
                "grpo_train_jsonl: non-finite loss {avg_group_loss} at group {group_number}"
            );
            last_loss = avg_group_loss;
            loss_history.push(avg_group_loss);
            global_step = global_step.saturating_add(1);
            processed_completions = processed_completions.saturating_add(entry.completions);
            token_counts.add_from(&group_counts);
            data_stats.groups_trained = global_step;
            data_stats.completions_trained = processed_completions;

            let checkpoint_due = config
                .checkpoint_interval
                .is_some_and(|interval| global_step % interval == 0 && global_step < total_steps);
            if checkpoint_due {
                let mut loop_state = GrpoCheckpointLoopState::capture(
                    GrpoCheckpointRoute::Jsonl,
                    global_step,
                    Some(bytes_read),
                    Some(line_no as u64),
                    processed_completions,
                    &loss_history,
                    &data_stats,
                    &token_counts,
                    dynamic_groups_filtered,
                    &echo_metrics,
                    &lora_grad_norms,
                    &policy_audit,
                    &phase_timings,
                    &gpu_writer_timings,
                    ema_ref_state.as_ref(),
                );
                let path = checkpoint_descriptor.save(
                    checkpoint_output_dir,
                    &*backend,
                    &mut params,
                    &mut opt_state,
                    ema_ref_state.as_ref(),
                    &mut loop_state,
                    gpu_step_coordination.as_ref(),
                    &mut gpu_writer_timings,
                    "streamed checkpoint device snapshot",
                )?;
                last_saved_step = Some(global_step);
                tracing::info!(
                    step = global_step,
                    checkpoint = %path.display(),
                    "saved exact streamed GRPO training checkpoint"
                );
            }

            let (step, progress_total_steps, progress) =
                jsonl_byte_progress(preflight.total_bytes, bytes_read);
            if let Some(ref cb) = progress_cb {
                let control = cb(TrainingProgress {
                    epoch: 1,
                    total_epochs: 1,
                    step,
                    total_steps: progress_total_steps,
                    loss: avg_group_loss,
                    progress,
                });
                if control == TrainControl::Stop && global_step < total_steps {
                    if last_saved_step != Some(global_step) {
                        let mut loop_state = GrpoCheckpointLoopState::capture(
                            GrpoCheckpointRoute::Jsonl,
                            global_step,
                            Some(bytes_read),
                            Some(line_no as u64),
                            processed_completions,
                            &loss_history,
                            &data_stats,
                            &token_counts,
                            dynamic_groups_filtered,
                            &echo_metrics,
                            &lora_grad_norms,
                            &policy_audit,
                            &phase_timings,
                            &gpu_writer_timings,
                            ema_ref_state.as_ref(),
                        );
                        let path = checkpoint_descriptor.save(
                            checkpoint_output_dir,
                            &*backend,
                            &mut params,
                            &mut opt_state,
                            ema_ref_state.as_ref(),
                            &mut loop_state,
                            gpu_step_coordination.as_ref(),
                            &mut gpu_writer_timings,
                            "streamed cancellation checkpoint device snapshot",
                        )?;
                        tracing::info!(
                            step = global_step,
                            checkpoint = %path.display(),
                            "saved exact streamed GRPO checkpoint before cancellation"
                        );
                    }
                    anyhow::bail!("training cancelled by user (stop requested at step boundary)");
                }
            }

            tracing::info!(
                group = global_step,
                completions_seen = processed_completions,
                action_tokens = group_counts.action_tokens,
                env_tokens = group_counts.env_tokens,
                byte_offset = bytes_read,
                total_bytes = preflight.total_bytes,
                loss = format!("{avg_group_loss:.6}"),
                "streamed GRPO group step"
            );
            if let Some(echo_env_ce) = step_report.echo_env_ce {
                tracing::info!(
                    group = global_step,
                    completions_seen = processed_completions,
                    action_tokens = group_counts.action_tokens,
                    env_tokens = group_counts.env_tokens,
                    echo_env_ce,
                    "streamed GRPO ECHO group metrics"
                );
            }
        }

        anyhow::ensure!(
            global_step == total_steps
                && loss_history.len() == total_steps
                && processed_completions == preflight.planned_completions
                && token_counts == preflight.planned_token_counts
                && bytes_read == preflight.total_bytes
                && line_no == preflight.total_lines,
            "streamed GRPO completed with inconsistent progress, diagnostics, or source cursor"
        );
        drop(reader);
        let final_training_data_sha256 = dataset_source
            .sha256()
            .with_context(|| format!("rehash GRPO JSONL dataset {}", dataset_path.display()))?;
        anyhow::ensure!(
            final_training_data_sha256 == training_data_sha256,
            "GRPO JSONL dataset changed during training"
        );
        crate::train_receipt::warn_echo_enabled_without_env_tokens(
            "streamed_grpo",
            config.loss.echo_enabled(),
            &token_counts,
        );

        let synced = run_coordinated_grpo_gpu_phase(
            gpu_step_coordination.as_ref(),
            &*backend,
            &mut gpu_writer_timings,
            "streamed final adapter snapshot",
            || {
                params
                    .sync_to_master(&*backend)
                    .context("capture final streamed GRPO adapter state")
            },
        )?;
        tracing::debug!(
            synced,
            "synced LoRA Vars to candle before streamed GRPO save"
        );

        params.save_peft(&output_dir, model_config.num_layers)?;

        tracing::info!(
            adapter = adapter_name,
            path = %output_dir.display(),
            final_loss = format!("{last_loss:.6}"),
            processed_groups = global_step,
            processed_completions,
            "streamed GRPO training complete"
        );

        Ok((output_dir.clone(), last_loss))
    };

    let mut result = train_body();
    drop(train_body);
    let policy_audit = finish_grpo_policy_audit(&mut result, policy_audit);
    let mut adapter_smoke_test = None;
    let cleanup_result = run_coordinated_grpo_gpu_phase(
        gpu_step_coordination.as_ref(),
        &*backend,
        &mut gpu_writer_timings,
        "streamed adapter smoke test and cleanup",
        || {
            if config.adapter_smoke_test && result.is_ok() {
                adapter_smoke_test = Some(run_adapter_smoke_test_best_effort(
                    adapter_name,
                    &*backend,
                    weights,
                    model_config,
                    tokenizer,
                    &params,
                    config.adapter_smoke_prompts.as_deref(),
                    streaming_prefill,
                ));
            }
            if let Some(state) = opt_state.as_ref() {
                state.evict_from_backend(&*backend);
            }
            params.evict_from_backend(&*backend);
            Ok(())
        },
    );
    if let Err(error) = cleanup_result {
        if result.is_ok() {
            result = Err(error.context("complete coordinated streamed GRPO cleanup"));
        } else {
            tracing::warn!(error = %format!("{error:#}"), "streamed GRPO cleanup could not acquire healthy backend");
        }
    }
    if let Some(state) = replay_state {
        let outcome = match &result {
            Ok((_, loss)) => Ok(*loss),
            Err(e) => Err(format!("{e:#}")),
        };
        if let Err(e) = close_replay_state(state, outcome) {
            tracing::warn!(error = %e, "failed to append streamed GRPO replay outcome record");
        }
    }
    gpu_writer_timings.apply_to(&mut phase_timings);
    let status_error = result.as_ref().err().map(|err| format!("{err:#}"));
    write_grpo_train_receipt_best_effort(
        adapter_name,
        model_config,
        tokenizer,
        weights.base_weight_shard_manifest.as_ref(),
        weights.execution_provenance.as_ref(),
        training_precision_for_receipt_best_effort(&params, opt_state.as_ref()),
        config,
        effective_seed,
        Some(alpha_over_rank),
        base_adapter_dir.as_deref(),
        &output_dir,
        training_data,
        data_stats,
        reward_stats,
        token_counts,
        phase_timings.to_receipt(),
        echo_metrics,
        run_started.elapsed().as_millis() as u64,
        dynamic_groups_filtered,
        adapter_smoke_test,
        lora_grad_norms.finish(),
        policy_audit,
        status_error,
    );
    result
        .map(|(dir, _)| dir)
        .map_err(crate::train_receipt::annotate_training_error)
}

/// Tokenized data for a single completion within a GRPO group.
///
/// Carries two parallel masks:
/// - `action_mask` — true at policy-gradient target positions (assistant
///   tokens). Equivalent to the pre-ECHO `completion_mask` for legacy
///   single-turn rollouts.
/// - `env_mask` — true at environment-observation target positions (tool
///   results). All-false when the rollout has no trajectory or when the
///   trajectory is single-turn Action-only. ECHO's env-CE consumes this.
struct TokenizedGrpoCompletion {
    /// Full input_ids: prompt + completion tokens.
    input_ids: Vec<u32>,
    /// Exact end of the prompt prefix, independent of the first sampled
    /// action position. Forced controller tokens may appear after this point.
    prompt_token_count: usize,
    /// Mask of positions the model generated (assistant turns).
    /// Targets of the GRPO policy-gradient objective.
    action_mask: Vec<bool>,
    /// Mask of positions the environment produced (tool-result turns).
    /// Targets of ECHO's env-CE auxiliary loss. All-false for legacy
    /// single-turn rollouts.
    env_mask: Vec<bool>,
    /// Total observation length |O| for paper §3.1 length normalization
    /// in the ECHO term. Counts every Observation token regardless of the
    /// warning_filter trim — `env_mask` may be a strict subset of |O|.
    total_obs_len: usize,
    /// Behavior-policy log-probabilities in sampled-action order. `None`
    /// means the rollout was admitted only under an explicit
    /// no-importance-correction policy.
    recorded_behavior_log_probs: Option<Vec<f32>>,
    /// Content-addressed rollout source identity without the provenance
    /// record's potentially long token arrays.
    recorded_behavior_source: Option<crate::train_receipt::GrpoRecordedBehaviorSourceObservation>,
}

/// A tokenized GRPO group ready for training.
struct TokenizedGrpoGroup {
    completions: Vec<TokenizedGrpoCompletion>,
    rewards: Vec<f64>,
}

fn validate_tokenized_behavior_policy(
    group: &TokenizedGrpoGroup,
    behavior_policy: BehaviorPolicy,
) -> Result<()> {
    for (completion_idx, completion) in group.completions.iter().enumerate() {
        anyhow::ensure!(
            completion.prompt_token_count > 0
                && completion.prompt_token_count <= completion.input_ids.len(),
            "GRPO completion {completion_idx} has invalid prompt_token_count {} for {} input tokens",
            completion.prompt_token_count,
            completion.input_ids.len()
        );
        let sampled_tokens = completion
            .action_mask
            .get(1..)
            .map_or(0, |mask| mask.iter().filter(|&&active| active).count());
        if behavior_policy == BehaviorPolicy::Recorded {
            let log_probs = completion.recorded_behavior_log_probs.as_ref().with_context(|| {
                format!(
                    "GRPO completion {completion_idx} is missing exact rollout provenance required by behavior_policy=recorded"
                )
            })?;
            anyhow::ensure!(
                log_probs.len() == sampled_tokens,
                "GRPO completion {completion_idx} has {} recorded behavior log-probabilities for {sampled_tokens} sampled action tokens",
                log_probs.len()
            );
            anyhow::ensure!(
                log_probs
                    .iter()
                    .all(|value| value.is_finite() && *value <= 1e-6),
                "GRPO completion {completion_idx} contains an invalid recorded behavior log-probability"
            );
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct GrpoGroupStepReport {
    loss: f64,
    echo_env_ce: Option<f64>,
}

fn token_counts_for_grpo_groups(
    groups: &[TokenizedGrpoGroup],
) -> crate::train_receipt::TokenCountReceipt {
    let mut counts = crate::train_receipt::TokenCountReceipt::default();
    for group in groups {
        for completion in &group.completions {
            let action = completion
                .action_mask
                .iter()
                .filter(|&&active| active)
                .count() as u64;
            let env = completion.env_mask.iter().filter(|&&active| active).count() as u64;
            let env_before = (completion.total_obs_len as u64).max(env);
            counts.observe_completion(completion.input_ids.len(), action, env, env_before);
        }
    }
    counts
}

fn grpo_benchmark_report_from_tokenized(
    tgroup: &TokenizedGrpoGroup,
    timings: GrpoBenchmarkTimings,
    loss: Option<f64>,
    policy_audit: Option<crate::train_receipt::GrpoPolicyAuditReceipt>,
    elapsed: Duration,
) -> GrpoBenchmarkReport {
    let counts = token_counts_for_grpo_groups(std::slice::from_ref(tgroup));
    let min_seq_len = tgroup
        .completions
        .iter()
        .map(|completion| completion.input_ids.len())
        .min()
        .unwrap_or(0);
    let max_seq_len = tgroup
        .completions
        .iter()
        .map(|completion| completion.input_ids.len())
        .max()
        .unwrap_or(0);
    let total_tokens = counts
        .action_tokens
        .saturating_add(counts.env_tokens)
        .saturating_add(counts.context_tokens);
    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let tokens_per_sec = if total_ms > 0.0 {
        total_tokens as f64 / (total_ms / 1000.0)
    } else {
        0.0
    };
    GrpoBenchmarkReport {
        completions: tgroup.completions.len(),
        min_seq_len,
        max_seq_len,
        total_tokens,
        action_tokens: counts.action_tokens,
        env_tokens: counts.env_tokens,
        context_tokens: counts.context_tokens,
        loss,
        policy_audit,
        timings,
        total_ms,
        tokens_per_sec,
    }
}

pub fn grpo_benchmark_tokenization(
    group: &GrpoGroup,
    tokenizer: &KilnTokenizer,
) -> Result<GrpoBenchmarkReport> {
    let started = Instant::now();
    let mut timings = GrpoBenchmarkTimings::default();
    let mask_cfg = crate::trajectory_mask::MaskConfig::default();
    let tgroup = tokenize_grpo_group_timed(group, tokenizer, &mask_cfg, Some(&mut timings))?;
    Ok(grpo_benchmark_report_from_tokenized(
        &tgroup,
        timings,
        None,
        None,
        started.elapsed(),
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn grpo_benchmark_training_step(
    backend: &dyn BackendRuntime,
    group: &GrpoGroup,
    weights: &GpuWeights,
    model_config: &ModelConfig,
    // (#1082) `&mut` — the GRPO step mutates each LoRA `Parameter` in place.
    params: &mut TrainableLoraParams,
    config: &GrpoConfig,
    segments: Option<&[(usize, usize)]>,
    device: &Device,
    tokenizer: &KilnTokenizer,
    opt_state: Option<&mut OptimizerState>,
) -> Result<GrpoBenchmarkReport> {
    grpo_benchmark_training_step_with_policy(
        backend,
        group,
        weights,
        model_config,
        params,
        config,
        segments,
        device,
        tokenizer,
        opt_state,
        StreamingPrefillExecutionPolicy::for_device(*device),
    )
}

/// Explicit-policy variant of [`grpo_benchmark_training_step`].
#[allow(clippy::too_many_arguments)]
pub fn grpo_benchmark_training_step_with_policy(
    backend: &dyn BackendRuntime,
    group: &GrpoGroup,
    weights: &GpuWeights,
    model_config: &ModelConfig,
    params: &mut TrainableLoraParams,
    config: &GrpoConfig,
    segments: Option<&[(usize, usize)]>,
    device: &Device,
    tokenizer: &KilnTokenizer,
    opt_state: Option<&mut OptimizerState>,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<GrpoBenchmarkReport> {
    let started = Instant::now();
    let mut timings = GrpoBenchmarkTimings::default();
    let mask_cfg = crate::trajectory_mask::MaskConfig::from_grpo_config(config);
    let tgroup = tokenize_grpo_group_timed(group, tokenizer, &mask_cfg, Some(&mut timings))?;
    let mut grad_norms = crate::train_receipt::LoraGradNormAccumulator::default();
    let mut policy_audit = crate::train_receipt::GrpoPolicyAuditAccumulator::default();
    let lora_grad_index = LoraGradNormIndex::new(params);
    let step_report = train_tokenized_grpo_group_with_grad_norms(
        backend,
        &tgroup,
        weights,
        model_config,
        params,
        config,
        segments,
        device,
        opt_state,
        &mut grad_norms,
        &lora_grad_index,
        &mut policy_audit,
        None,
        Some(&mut timings),
        streaming_prefill,
    )?;
    let policy_audit = policy_audit
        .finish()
        .context("finish GRPO benchmark policy audit")?;
    Ok(grpo_benchmark_report_from_tokenized(
        &tgroup,
        timings,
        Some(step_report.loss),
        Some(policy_audit),
        started.elapsed(),
    ))
}

fn validate_grpo_trajectory_roles(group: &GrpoGroup, line_no: usize) -> Result<()> {
    for (rollout_idx, rollout) in group.completions.iter().enumerate() {
        for (segment_idx, segment) in rollout.trajectory.iter().enumerate() {
            let role = segment.role.trim();
            anyhow::ensure!(
                !role.is_empty(),
                "malformed trajectory role at line {line_no}, completion {rollout_idx}, segment {segment_idx}: role must be non-empty"
            );
            match segment.kind {
                TurnKind::Action => anyhow::ensure!(
                    role.eq_ignore_ascii_case("assistant"),
                    "malformed trajectory role at line {line_no}, completion {rollout_idx}, segment {segment_idx}: Action segment must use role \"assistant\", got {:?}",
                    segment.role
                ),
                TurnKind::Observation => anyhow::ensure!(
                    role.eq_ignore_ascii_case("tool"),
                    "malformed trajectory role at line {line_no}, completion {rollout_idx}, segment {segment_idx}: Observation segment must use role \"tool\", got {:?}",
                    segment.role
                ),
                TurnKind::Context => {}
            }
        }
    }
    Ok(())
}

fn validate_grpo_dry_run_masks(
    group: &TokenizedGrpoGroup,
    group_idx: usize,
    line_no: usize,
) -> Result<()> {
    for (completion_idx, completion) in group.completions.iter().enumerate() {
        anyhow::ensure!(
            completion.action_mask.len() == completion.input_ids.len(),
            "GRPO dry run: group {group_idx} line {line_no} completion {completion_idx} action_mask length {} does not match input_ids length {}",
            completion.action_mask.len(),
            completion.input_ids.len()
        );
        anyhow::ensure!(
            completion.env_mask.len() == completion.input_ids.len(),
            "GRPO dry run: group {group_idx} line {line_no} completion {completion_idx} env_mask length {} does not match input_ids length {}",
            completion.env_mask.len(),
            completion.input_ids.len()
        );
        let action_tokens = completion
            .action_mask
            .iter()
            .filter(|&&active| active)
            .count();
        anyhow::ensure!(
            action_tokens > 0,
            "GRPO dry run: group {group_idx} line {line_no} completion {completion_idx} has empty action_mask"
        );
    }
    Ok(())
}

fn parse_grpo_jsonl_group_line(line: &str, line_no: usize) -> Result<Option<GrpoGroup>> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    serde_json::from_str::<GrpoGroup>(trimmed)
        .map(Some)
        .with_context(|| format!("parse GRPO JSONL group at line {line_no}"))
}

fn jsonl_byte_progress(total_bytes: u64, offset: u64) -> (usize, usize, f32) {
    let total = total_bytes.max(1);
    let clamped = offset.min(total);
    let total_steps = total.min(usize::MAX as u64).max(1) as usize;
    let step = clamped.min(usize::MAX as u64).max(1) as usize;
    let progress = (clamped as f64 / total as f64).min(0.999) as f32;
    (step, total_steps, progress)
}

/// Page size used by the GRPO shared-prompt-prefix paged cache. Matches the
/// production server / bench setting so the same FA fast paths fire (#1082:
/// 16 -> 64 so each FA2 kBlockN=64 tile is one page; keeps parity with
/// `DEFAULT_BLOCK_SIZE`).
const GRPO_REF_PAGED_BLOCK_SIZE: usize = 64;

fn grpo_shared_prefix_tile_tokens(
    streaming_prefill: StreamingPrefillExecutionPolicy,
    seq_len: usize,
) -> Result<Option<usize>> {
    if !streaming_prefill.enabled_for(seq_len) {
        return Ok(None);
    }
    let tile_tokens = streaming_prefill.base_tile_tokens_for(seq_len);
    anyhow::ensure!(
        tile_tokens > 0,
        "GRPO shared-prefix streaming tile size must be greater than zero"
    );
    Ok((tile_tokens < seq_len).then_some(tile_tokens))
}

#[allow(clippy::too_many_arguments)]
fn model_forward_paged_normed_hidden_with_policy(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    model_config: &ModelConfig,
    paged_cache: &PagedKvCacheKt,
    block_table: &BlockTable,
    start_pos: usize,
    mut linear_state: Option<&mut LinearAttentionState>,
    ema_ref_lora: Option<&LoraWeights>,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<Tensor> {
    anyhow::ensure!(
        !token_ids.is_empty(),
        "GRPO shared-prefix paged forward requires at least one token"
    );
    let Some(tile_tokens) = grpo_shared_prefix_tile_tokens(streaming_prefill, token_ids.len())?
    else {
        return model_forward_paged_normed_hidden(
            backend,
            token_ids,
            weights,
            model_config,
            paged_cache,
            block_table,
            start_pos,
            linear_state,
            ema_ref_lora,
        );
    };

    let mut tile_hidden = Vec::with_capacity(token_ids.len().div_ceil(tile_tokens));
    let mut cursor = 0usize;
    while cursor < token_ids.len() {
        let end = (cursor + tile_tokens).min(token_ids.len());
        tile_hidden.push(
            model_forward_paged_normed_hidden(
                backend,
                &token_ids[cursor..end],
                weights,
                model_config,
                paged_cache,
                block_table,
                start_pos + cursor,
                linear_state.as_deref_mut(),
                ema_ref_lora,
            )
            .with_context(|| {
                format!(
                    "GRPO shared-prefix streaming tile [{cursor}, {end}) of {}",
                    token_ids.len()
                )
            })?,
        );
        cursor = end;
    }
    let refs: Vec<&Tensor> = tile_hidden.iter().collect();
    cat_tensors(&refs, 1).context("GRPO shared-prefix: concatenate streaming hidden tiles")
}

/// Compute the reference-policy log probs for every completion in a GRPO
/// group, sharing the prompt-prefix forward across all completions.
///
/// All completions in a GRPO group share an identical prompt prefix
/// (tokenize_grpo_group computes `prompt_ids` once and reuses it). The legacy
/// path ran `model_forward_no_head` over `[prompt | completion]` once per
/// completion (4× per group at default settings), redoing the
/// O(prompt_len²) full-attention and O(prompt_len) GDN work each time.
///
/// This helper runs the prompt forward exactly once via the paged path,
/// snapshots the GDN linear state at `prompt_len`, then forwards only each
/// completion's tokens with `start_pos == prompt_len`. The paged cache
/// transparently feeds the prompt's K/V into the full-attention layers as
/// "prefix history", so FlashAttention prefill-with-prefix runs at
/// O(comp_len × prompt_len) instead of O(prompt_len²) per completion. The
/// GDN linear state is restored from the post-prompt snapshot before each
/// completion so its recurrent state starts from the correct point.
///
/// The shared-prefix path requires no gradient (this is the reference
/// forward), so the paged inference kernels are used directly. Total
/// reference-forward attention work drops from `n_comp × (P + C)²` to
/// `P² + n_comp × C × (P + C)` — roughly a `n_comp×` speedup when
/// `C << P`, which is the production regime for pi-compaction.
///
/// All returned log-prob tensors are detached (the ratio computation in
/// `grpo_loss` only needs the policy side to track gradients).
fn compute_ref_log_probs_shared_prefix(
    backend: &dyn BackendRuntime,
    tgroup: &TokenizedGrpoGroup,
    weights: &GpuWeights,
    model_config: &ModelConfig,
    ema_ref_lora: Option<&LoraWeights>,
    device: &Device,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<Vec<Tensor>> {
    if tgroup.completions.is_empty() {
        return Ok(Vec::new());
    }

    let first = &tgroup.completions[0];
    let prompt_len = first.prompt_token_count;
    if prompt_len < 1 {
        anyhow::bail!("GRPO shared-prefix ref forward requires prompt_len >= 1, got {prompt_len}");
    }

    // Validate the prefix invariant — every completion must share the same
    // prompt prefix or the shared-prefix path is unsound.
    for (idx, comp) in tgroup.completions.iter().enumerate() {
        let comp_prompt_len = comp.prompt_token_count;
        anyhow::ensure!(
            comp_prompt_len == prompt_len,
            "GRPO completions have different prompt lengths ({prompt_len} vs {comp_prompt_len} for completion {idx})"
        );
        anyhow::ensure!(
            comp.input_ids.len() >= prompt_len,
            "completion {idx} input_ids shorter than prompt_len {prompt_len}"
        );
        anyhow::ensure!(
            comp.input_ids[..prompt_len] == first.input_ids[..prompt_len],
            "completion {idx} prompt token ids differ from completion 0",
        );
    }

    let prompt_ids: &[u32] = &first.input_ids[..prompt_len];
    let max_total = tgroup
        .completions
        .iter()
        .map(|c| c.input_ids.len())
        .max()
        .unwrap_or(prompt_len)
        .max(prompt_len);

    let dtype = match model_config.dtype {
        kiln_core::config::DType::BF16 => DType::BF16,
        kiln_core::config::DType::FP16 => DType::F16,
        kiln_core::config::DType::FP32 => DType::F32,
    };

    let num_blocks = (max_total + GRPO_REF_PAGED_BLOCK_SIZE - 1) / GRPO_REF_PAGED_BLOCK_SIZE;
    // (#1082) The candle `PagedKvCache::new` took a candle device; its kt twin
    // `PagedKvCacheKt::new` allocates its pools on the model's runtime `Device`.
    // `device` is a kt `Device` (Copy) — pass it through so the pools land on
    // the same device as the model's tensors (CPU model → CPU pools, etc.).
    let paged_cache = PagedKvCacheKt::new(
        model_config.num_full_attention_layers,
        num_blocks,
        GRPO_REF_PAGED_BLOCK_SIZE,
        model_config.num_kv_heads,
        model_config.head_dim,
        dtype,
        *device,
    )
    .context("GRPO shared-prefix: build PagedKvCacheKt")?;
    let mut block_table = BlockTable::new();
    for i in 0..num_blocks as u32 {
        block_table.push(i);
    }

    let mut linear_state = LinearAttentionState::new(model_config, device)
        .context("GRPO shared-prefix: build LinearAttentionState")?;

    // Phase 1: prompt forward — populates the paged cache for positions
    // [0..prompt_len) and advances the GDN linear state past the prompt.
    let prompt_hidden = model_forward_paged_normed_hidden_with_policy(
        backend,
        prompt_ids,
        weights,
        model_config,
        &paged_cache,
        &block_table,
        0,
        Some(&mut linear_state),
        ema_ref_lora,
        streaming_prefill,
    )
    .context("GRPO shared-prefix: prompt forward")?;

    // The position that predicts the first completion token (input_ids[prompt_len])
    // is prompt_len - 1. Capture its normed hidden state as a detached, stable
    // owning tensor so the rest of the prompt_hidden allocation can be freed.
    // (#1082) `prompt_hidden` is kt (kt-flipped `model_forward_paged_normed_hidden`);
    // the downstream GRPO ref log-prob math (`cat_tensors`,
    // `chunked_log_probs_for_completion`) is now kt-native too, so keep it kt.
    let last_prompt_hidden = prompt_hidden
        .narrow(1, prompt_len - 1, 1)
        .context("GRPO shared-prefix: narrow last prompt hidden")?
        .contiguous()
        .context("GRPO shared-prefix: contiguous last prompt hidden")?;
    drop(prompt_hidden);

    // Snapshot the GDN linear state at end-of-prompt so each completion can
    // restore from this point before running its own forward.
    let linear_snap = linear_state
        .snapshot()
        .context("GRPO shared-prefix: snapshot linear state")?;

    let mut ref_log_probs_per_comp = Vec::with_capacity(tgroup.completions.len());
    for (comp_idx, comp) in tgroup.completions.iter().enumerate() {
        let full_len = comp.input_ids.len();
        let comp_len = full_len - prompt_len;
        if comp_len == 0 {
            // No completion tokens — placeholder zero tensor matches the legacy
            // path's behaviour for empty active completions.
            ref_log_probs_per_comp.push(zeros_f32_on(1, device)?.detach());
            continue;
        }

        // Restore GDN state to end-of-prompt. The paged-cache full-attn K/V
        // for positions [0..prompt_len) is preserved (writes only target
        // start_pos..start_pos+seq_len, never the prefix), so the cache is
        // implicitly reset by passing start_pos = prompt_len. Each completion
        // overwrites the cache slots at [prompt_len..prompt_len+comp_len), but
        // that region is throw-away — we never read another completion's K/V.
        linear_state.restore_from(&linear_snap).with_context(|| {
            format!("GRPO shared-prefix: restore linear state for completion {comp_idx}")
        })?;

        let completion_ids = &comp.input_ids[prompt_len..];

        let comp_hidden = {
            let kt = model_forward_paged_normed_hidden_with_policy(
                backend,
                completion_ids,
                weights,
                model_config,
                &paged_cache,
                &block_table,
                prompt_len,
                Some(&mut linear_state),
                ema_ref_lora,
                streaming_prefill,
            )
            .with_context(|| format!("GRPO shared-prefix: completion {comp_idx} forward"))?;
            // (#1082) kt forward output flows straight into the kt log-prob path.
            kt
        };

        // Build the "active hidden" tensor: aligned with the completion tokens
        // we want to compute log-probs for. Following the legacy convention in
        // `selected_log_probs_from_normed_hidden_chunked`, hidden[i] is the
        // normed pre-LM-head state that predicts the token at position i+1.
        //
        //   active_hidden[0]            = last prompt hidden (predicts input_ids[prompt_len])
        //   active_hidden[1..comp_len]  = comp_hidden[0..comp_len-1]
        //                                 (predicts input_ids[prompt_len+1..full_len])
        //
        // Shape: [1, comp_len, hidden_size]. comp_hidden[comp_len-1] is dropped
        // because there's no token after the last completion token to predict.
        let active_hidden = if comp_len == 1 {
            last_prompt_hidden.clone()
        } else {
            let comp_prefix = comp_hidden.narrow(1, 0, comp_len - 1).with_context(|| {
                format!("GRPO shared-prefix: narrow comp prefix completion {comp_idx}")
            })?;
            cat_tensors(&[&last_prompt_hidden, &comp_prefix], 1).with_context(|| {
                format!("GRPO shared-prefix: concat active hidden completion {comp_idx}")
            })?
        };
        drop(comp_hidden);

        // (#1082) `embed_tokens_t` and the chunked log-prob helper are both
        // kt now; pass the kt head weight straight through.
        let log_probs = chunked_log_probs_for_completion(
            &active_hidden,
            &weights.embed_tokens_t,
            completion_ids,
            DEFAULT_CHUNK_SIZE,
            device,
        )
        .with_context(|| format!("GRPO shared-prefix: chunked log-probs completion {comp_idx}"))?;

        ref_log_probs_per_comp.push(log_probs.detach());
    }

    Ok(ref_log_probs_per_comp)
}

/// Compute per-target-token log probs from a pre-shifted normed-hidden tensor.
///
/// `active_hidden` is `[1, n_targets, hidden_size]` and is assumed to be the
/// post-final-RMSNorm hidden state at exactly the positions that need a
/// log-prob (one row per target token). `target_ids` is the actual token id
/// each row predicts. This is the chunked-softmax core that
/// [`selected_log_probs_from_normed_hidden_chunked`] also uses, but without
/// the position-selection / shift bookkeeping (the caller has already aligned
/// rows with targets).
fn chunked_log_probs_for_completion(
    active_hidden: &Tensor,
    head_t: &Tensor,
    target_ids: &[u32],
    chunk_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let n_targets = target_ids.len();
    if n_targets == 0 {
        return zeros_f32_on(1, device).map_err(Into::into);
    }
    if chunk_size == 0 {
        anyhow::bail!("chunked_log_probs_for_completion chunk_size must be > 0");
    }

    let dims = active_hidden.dims();
    if dims.len() != 3 || dims[0] != 1 || dims[1] != n_targets {
        anyhow::bail!(
            "active_hidden must have shape [1, n_targets={n_targets}, hidden_size], got {:?}",
            dims
        );
    }
    let hidden_size = dims[2];
    if head_t.dims().len() != 2 || head_t.dims()[0] != hidden_size {
        anyhow::bail!(
            "head_t must have shape [hidden_size, vocab_size], got {:?}",
            head_t.dims()
        );
    }

    let hidden_2d = active_hidden.squeeze(0)?.to_f32_dtype()?;
    let head_t_f32 = head_t.to_f32_dtype()?;
    let vocab_size = head_t_f32.dim(1)?;
    if vocab_size == 0 {
        anyhow::bail!("head_t vocab dimension is zero");
    }

    let mut running_max: Option<Tensor> = None;
    let mut running_sumexp: Option<Tensor> = None;
    let mut correct_logits: Option<Tensor> = None;
    let mut chunk_start = 0usize;
    while chunk_start < vocab_size {
        let chunk_len = chunk_size.min(vocab_size - chunk_start);
        let chunk_end = chunk_start + chunk_len;
        {
            let head_chunk = head_t_f32.narrow(1, chunk_start, chunk_len)?.contiguous()?;
            let logits_chunk = hidden_2d.matmul(&head_chunk)?;
            let chunk_max = logits_chunk.max_keepdim(LAST_DIM)?;
            let (new_max, new_sumexp) = match (running_max.as_ref(), running_sumexp.as_ref()) {
                (None, None) => {
                    let shifted =
                        (&logits_chunk - chunk_max.broadcast_as(logits_chunk.shape())?)?;
                    let chunk_sumexp = shifted.exp()?.sum_keepdim(LAST_DIM)?;
                    (chunk_max.detach(), chunk_sumexp.detach())
                }
                (Some(prev_max), Some(prev_sumexp)) => {
                    let new_max = prev_max.maximum(&chunk_max)?;
                    let prev_scale = (prev_max - &new_max)?.exp()?;
                    let scaled_prev = prev_sumexp.broadcast_mul(&prev_scale)?;
                    let shifted = (&logits_chunk - new_max.broadcast_as(logits_chunk.shape())?)?;
                    let chunk_sumexp = shifted.exp()?.sum_keepdim(LAST_DIM)?;
                    let new_sumexp = (scaled_prev + chunk_sumexp)?;
                    (new_max.detach(), new_sumexp.detach())
                }
                _ => unreachable!("running max/sumexp are set together"),
            };
            running_max = Some(new_max);
            running_sumexp = Some(new_sumexp);

            let chunk_correct = selected_logits_from_chunk_sparse(
                &logits_chunk,
                target_ids,
                chunk_start,
                chunk_len,
                vocab_size,
                device,
                "chunked_log_probs_for_completion",
            )?;
            correct_logits = Some(match correct_logits.as_ref() {
                Some(prev) => (prev + chunk_correct)?.detach(),
                None => chunk_correct.detach(),
            });
        }
        synchronize_tail_chunk("synchronize chunked_log_probs_for_completion")?;
        chunk_start = chunk_end;
    }

    let running_max = running_max.context("vocab_size was zero")?;
    let running_sumexp = running_sumexp.context("vocab_size was zero")?;
    let correct_logits = correct_logits.context("vocab_size was zero")?;
    let log_sum_exp = (running_max + running_sumexp.log()?)?;
    Ok((correct_logits - log_sum_exp)?.squeeze(1)?)
}

fn observe_grpo_policy_audit_completion(
    policy_audit: &mut crate::train_receipt::GrpoPolicyAuditAccumulator,
    policy_log_probs: &Tensor,
    behavior_log_probs: Option<&[f32]>,
    kl_reference_log_probs: Option<&Tensor>,
    loss_params: GrpoLossParams,
    behavior_source: Option<&crate::train_receipt::GrpoRecordedBehaviorSourceObservation>,
) -> Result<()> {
    let policy_log_probs_host = policy_log_probs
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_device(cpu_device())?
        .to_vec1::<f32>()?;
    let kl_reference_log_probs_host = kl_reference_log_probs
        .map(|reference| {
            reference
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_device(cpu_device())?
                .to_vec1::<f32>()
        })
        .transpose()?;
    policy_audit.observe_policy_values(
        &policy_log_probs_host,
        behavior_log_probs,
        kl_reference_log_probs_host.as_deref(),
        loss_params.is_level,
        loss_params.clip_low,
        loss_params.clip_high,
        loss_params.kl_estimator,
        loss_params.entropy_aware_kl_quantile,
    )?;
    if let Some(source) = behavior_source {
        policy_audit.observe_recorded_behavior_source(source);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn train_tokenized_grpo_group_with_grad_norms(
    backend: &dyn BackendRuntime,
    tgroup: &TokenizedGrpoGroup,
    weights: &GpuWeights,
    model_config: &ModelConfig,
    // (#1082) `&mut` — the optimizer step mutates each LoRA `Parameter`'s
    // kt master in place.
    params: &mut TrainableLoraParams,
    config: &GrpoConfig,
    segments: Option<&[(usize, usize)]>,
    device: &Device,
    opt_state: Option<&mut OptimizerState>,
    grad_norms: &mut crate::train_receipt::LoraGradNormAccumulator,
    lora_grad_index: &LoraGradNormIndex,
    policy_audit: &mut crate::train_receipt::GrpoPolicyAuditAccumulator,
    // Optional EMA-snapshot LoRA used as the KL reference when
    // `config.kl_reference_policy == KlReferencePolicy::Ema`. None means the
    // KL-reference forward runs without LoRA (`BasePerStep`) or is skipped.
    ema_ref_lora: Option<&LoraWeights>,
    mut timings: Option<&mut GrpoBenchmarkTimings>,
    streaming_prefill_policy: StreamingPrefillExecutionPolicy,
) -> Result<GrpoGroupStepReport> {
    validate_tokenized_behavior_policy(tgroup, config.behavior_policy)
        .context("validate GRPO behavior-policy provenance")?;
    let skip_kl_reference = !config.kl_penalty_enabled()
        || matches!(config.kl_reference_policy, KlReferencePolicy::None);

    // Learning-rate resolution is request-local and independent of runtime
    // execution policy.
    let learning_rate = config.effective_learning_rate();
    let advantages = compute_advantages(&tgroup.rewards, config.advantage_mode);
    let mut group_loss_sum = 0.0;
    let mut opt_state = opt_state;

    // Active token counts per completion (matches the next-token-shift convention
    // used by token_log_probs and the analytic tail: action_mask[1..]).
    let per_comp_active: Vec<usize> = tgroup
        .completions
        .iter()
        .map(|c| {
            c.action_mask
                .get(1..)
                .map_or(0, |m| m.iter().filter(|&&v| v).count())
        })
        .collect();
    let group_total_active: usize = per_comp_active.iter().sum();
    if group_total_active == 0 {
        return Ok(GrpoGroupStepReport {
            loss: 0.0,
            echo_env_ce: None,
        });
    }
    ensure_tape_forward_backward_supported("GRPO group step", weights, backend)?;
    let group_counts = token_counts_for_grpo_groups(std::slice::from_ref(tgroup));
    let group_max_seq_len = tgroup
        .completions
        .iter()
        .map(|completion| completion.input_ids.len())
        .max()
        .unwrap_or(0);
    let checkpoint_segments = segments.map_or(0, |segs| segs.len());
    let streaming_tile_tokens = streaming_prefill_policy.base_tile_tokens();
    let streaming_prefill = streaming_prefill_policy.enabled_for(group_max_seq_len);

    let token_level = matches!(config.loss_aggregation, LossAggregation::TokenLevel);
    let mut group_accum: GradMap = HashMap::new();
    let mut group_echo_ce_sum = 0.0f64;
    let mut group_echo_ce_weight = 0usize;

    // Shared-prefix optimization: when the reference policy is active and the
    // group has more than one completion, run the prompt forward exactly once
    // (paged path) and reuse its K/V + GDN state across all completions. The
    // legacy per-completion `model_forward_no_head` loop is kept as the
    // fallback below when (a) reference is skipped, or (b) the group has a
    // single completion (no sharing to be had), or (c) the shared-prefix path
    // is explicitly disabled.
    let use_shared_prefix = !skip_kl_reference
        && tgroup.completions.len() > 1
        // The paged shared-prefix reference path still mixes a host
        // broadcast temporary into a fully resident Vulkan graph. Keep the
        // exact per-completion path until paged Vulkan reference parity is
        // independently qualified.
        && !matches!(device, Device::Vulkan(_))
        && config.shared_prefix_reference;
    let shared_prefix_log_probs: Option<Vec<Tensor>> = if use_shared_prefix {
        let started = Instant::now();
        tracing::info!(
            completions = tgroup.completions.len(),
            max_seq_len = group_max_seq_len,
            action_tokens = group_counts.action_tokens,
            env_tokens = group_counts.env_tokens,
            checkpoint_segments,
            streaming_prefill,
            streaming_tile_tokens,
            "GRPO ref forward start"
        );
        let log_probs = compute_ref_log_probs_shared_prefix(
            backend,
            tgroup,
            weights,
            model_config,
            ema_ref_lora,
            device,
            streaming_prefill_policy,
        )
        .context("GRPO shared-prefix reference forward")?;
        let elapsed = started.elapsed();
        if let Some(t) = timings.as_deref_mut() {
            t.add_reference_forward(elapsed);
        }
        tracing::info!(
            n_completions = tgroup.completions.len(),
            max_seq_len = group_max_seq_len,
            action_tokens = group_counts.action_tokens,
            env_tokens = group_counts.env_tokens,
            checkpoint_segments,
            streaming_prefill,
            streaming_tile_tokens,
            elapsed_ms = elapsed.as_millis() as u64,
            "GRPO ref forward end"
        );
        Some(log_probs)
    } else {
        None
    };

    for (comp_idx, comp) in tgroup.completions.iter().enumerate() {
        let num_active = per_comp_active[comp_idx];
        if num_active == 0 {
            continue;
        }
        let loss_normalizer = if token_level {
            1.0 / group_total_active as f64
        } else {
            1.0 / num_active as f64
        };
        let loss_params =
            GrpoLossParams::from_config(config, advantages[comp_idx], loss_normalizer);
        let comp_env_count = comp
            .env_mask
            .get(1..)
            .map_or(0, |m| m.iter().filter(|&&v| v).count());
        let mut comp_echo_env_ce: Option<f64> = None;

        let kl_reference_log_probs = if skip_kl_reference {
            // KL is disabled, so the placeholder is never inspected by the
            // loss. Behavior-policy probabilities are prepared separately.
            zeros_f32_on(num_active, device)?.detach()
        } else if let Some(shared) = shared_prefix_log_probs.as_ref() {
            // The shared-prefix output is one log-prob per completion-span
            // position (predicting input_ids[prompt_len + i] for
            // i in 0..comp_len). For trajectory-aware rollouts that
            // include Observation segments, we need only the Action
            // positions to match policy_log_probs's shape; for legacy
            // single-turn rollouts the action_mask is true at every
            // completion-span position and this filter is a no-op.
            let span = &shared[comp_idx];
            let comp_prompt_len = comp.prompt_token_count;
            let active_indices: Vec<u32> = (0..span.dim(0)?)
                .filter(|&i| {
                    comp.action_mask
                        .get(comp_prompt_len + i)
                        .copied()
                        .unwrap_or(false)
                })
                .map(|i| i as u32)
                .collect();
            if active_indices.len() == span.dim(0)? {
                // Legacy fast-path: every span position is active. No need
                // to allocate an indices tensor or do an index_select.
                span.clone()
            } else if active_indices.is_empty() {
                // Defensive: shouldn't happen (num_active > 0 was checked
                // above), but handle it cleanly.
                zeros_f32_on(num_active, device)?.detach()
            } else {
                let n_idx = active_indices.len();
                let indices = Tensor::from_vec_on(*device, active_indices, vec![n_idx])?;
                span.index_select(&indices, 0)?.detach()
            }
        } else {
            let ref_started = Instant::now();
            tracing::info!(
                comp_idx,
                seq_len = comp.input_ids.len(),
                action_tokens = num_active,
                env_tokens = comp_env_count,
                checkpoint_segments,
                streaming_prefill = streaming_prefill_policy.enabled_for(comp.input_ids.len()),
                streaming_tile_tokens,
                "GRPO ref forward start"
            );
            let mut ref_linear_state = LinearAttentionState::new(model_config, device)?;
            // BasePerStep (None ema_ref_lora) → base model (no LoRA).
            // Ema (Some(snapshot)) → frozen snapshot of the LoRA from a
            // prior training point.
            // (#1082) `model_forward_no_head` and
            // `selected_log_probs_from_normed_hidden_chunked` are both kt-native;
            // the kt hidden + kt `embed_tokens_t` head weight flow through
            // directly (no candle bridge).
            let ref_hidden = model_forward_no_head_with_policy(
                backend,
                &comp.input_ids,
                weights,
                model_config,
                Some(&mut ref_linear_state),
                ema_ref_lora,
                streaming_prefill_policy,
            )
            .context("GRPO reference forward pass")?
            .contiguous()
            .context("GRPO ref hidden contiguous")?;
            let ref_log_probs = selected_log_probs_from_normed_hidden_chunked(
                &ref_hidden,
                &weights.embed_tokens_t,
                &comp.input_ids,
                &comp.action_mask,
                DEFAULT_CHUNK_SIZE,
            )?
            .detach();
            if let Some(t) = timings.as_deref_mut() {
                t.add_reference_forward(ref_started.elapsed());
            }
            tracing::info!(
                comp_idx,
                seq_len = comp.input_ids.len(),
                action_tokens = num_active,
                env_tokens = comp_env_count,
                checkpoint_segments,
                streaming_prefill = streaming_prefill_policy.enabled_for(comp.input_ids.len()),
                streaming_tile_tokens,
                elapsed_ms = ref_started.elapsed().as_millis() as u64,
                "GRPO ref forward end"
            );
            ref_log_probs
        };

        let behavior_log_probs = match config.behavior_policy {
            BehaviorPolicy::NoImportanceCorrection => zeros_f32_on(num_active, device)?.detach(),
            BehaviorPolicy::Recorded => {
                let values = comp.recorded_behavior_log_probs.as_ref().with_context(|| {
                    format!("completion {comp_idx} is missing behavior-policy log-probabilities")
                })?;
                Tensor::from_vec_on(*device, values.clone(), vec![num_active])?.detach()
            }
        };

        // (#1082 candle-drop) GRPO per-completion step is now UNCONDITIONALLY
        // kt tape-authoritative. The candle gradient-checkpointed GRPO reverse
        // (`checkpointed_grpo_forward_backward` + analytic ECHO tail), the
        // candle tape-bridge producer, and the inline candle `loss.backward()`
        // path are all DELETED. ECHO env-CE has no kt tape root, so an
        // ECHO-active GRPO step is not supported on the kt-only path (the
        // candle ECHO term was a candle-authoritative feature dropped in the
        // candle drop). `grpo_step_forward_backward_tape_authoritative_kt`
        // returns `GradSource::Kt`, consumed kt-native by the dispatchers.
        let loss_val: f64;
        let (grads, policy_log_probs): (GradSource, Tensor) = {
            #[cfg(any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "rocm"
            ))]
            {
                // ECHO env-CE spec (resurrection PR2): built when the term
                // is enabled and this completion actually has env rows; the
                // fused loss roots add λ·env_CE to the value and the matching
                // constant-coefficient rows to the gradient.
                let echo_env_spec =
                    if config.loss.echo_enabled() && comp_env_count > 0 && comp.total_obs_len > 0 {
                        Some(crate::grpo_tape_shim::EchoEnvSpec {
                            env_mask: comp.env_mask.clone(),
                            total_obs_len: comp.total_obs_len,
                            lambda: config.loss.echo_lambda(),
                        })
                    } else {
                        None
                    };
                let (lv, env_ce, kt_grads, policy_log_probs) = if let Some(segs) = segments {
                    let step_started = Instant::now();
                    let out = checkpointed_grpo_forward_backward_tape_authoritative_kt(
                        backend,
                        &comp.input_ids,
                        weights,
                        model_config,
                        params,
                        &comp.action_mask,
                        &behavior_log_probs,
                        &kl_reference_log_probs,
                        loss_params,
                        segs,
                        device,
                        echo_env_spec.as_ref(),
                        config.loss.no_policy_loss,
                        config.detect_anomaly,
                        streaming_prefill_policy,
                    )?;
                    let step_elapsed = step_started.elapsed();
                    if let Some(t) = timings.as_deref_mut() {
                        t.add_backward(step_elapsed);
                    }
                    tracing::info!(
                        comp_idx,
                        seq_len = comp.input_ids.len(),
                        action_tokens = num_active,
                        env_tokens = comp_env_count,
                        checkpoint_segments,
                        streaming_prefill =
                            streaming_prefill_policy.enabled_for(comp.input_ids.len()),
                        streaming_tile_tokens,
                        elapsed_ms = step_elapsed.as_millis() as u64,
                        "GRPO step end (checkpointed tape-authoritative kt)"
                    );
                    out
                } else {
                    grpo_step_forward_backward_tape_authoritative_kt(
                        backend,
                        &comp.input_ids,
                        weights,
                        model_config,
                        params,
                        &comp.action_mask,
                        &behavior_log_probs,
                        &kl_reference_log_probs,
                        loss_params,
                        device,
                        comp_idx,
                        num_active,
                        comp_env_count,
                        streaming_tile_tokens,
                        checkpoint_segments,
                        timings.as_deref_mut(),
                        echo_env_spec.as_ref(),
                        config.loss.no_policy_loss,
                        config.detect_anomaly,
                        streaming_prefill_policy,
                    )?
                };
                loss_val = lv;
                comp_echo_env_ce = env_ce;
                (GradSource::Kt(kt_grads), policy_log_probs)
            }
            #[cfg(not(any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "rocm"
            )))]
            {
                // The group-entry capability check already bailed without a GPU
                // backend feature. This arm keeps `loss_val` definitely assigned.
                let _ = (
                    &behavior_log_probs,
                    &kl_reference_log_probs,
                    num_active,
                    comp_env_count,
                    comp_idx,
                );
                unreachable!("GRPO kt path requires a GPU backend feature");
            }
        };
        anyhow::ensure!(
            policy_log_probs.elem_count() == num_active,
            "GRPO loss returned {} selected policy log-probabilities for {num_active} active tokens",
            policy_log_probs.elem_count()
        );
        let behavior_log_probs_host = match config.behavior_policy {
            BehaviorPolicy::NoImportanceCorrection => None,
            BehaviorPolicy::Recorded => comp.recorded_behavior_log_probs.as_deref(),
        };
        observe_grpo_policy_audit_completion(
            policy_audit,
            &policy_log_probs,
            behavior_log_probs_host,
            (!skip_kl_reference).then_some(&kl_reference_log_probs),
            loss_params,
            comp.recorded_behavior_source.as_ref(),
        )
        .with_context(|| format!("record GRPO policy metrics for completion {comp_idx}"))?;
        if token_level {
            // Cross-completion grad accumulation into the kt `GradMap`
            // (keyed by `Parameter::tensor_id()`).
            accumulate_grads_dispatch(&mut group_accum, &grads, params)?;
        } else {
            observe_lora_grad_norms_dispatch(grad_norms, params, &grads)?;
            let optimizer_started = Instant::now();
            tracing::info!(
                comp_idx,
                seq_len = comp.input_ids.len(),
                action_tokens = num_active,
                env_tokens = comp_env_count,
                optimizer = ?config.optimizer,
                "GRPO optimizer start"
            );
            optimizer_step_dispatch(
                backend,
                params,
                &grads,
                learning_rate,
                config.optimizer,
                opt_state.as_deref_mut(),
            )?;
            if let Some(t) = timings.as_deref_mut() {
                t.add_optimizer(optimizer_started.elapsed());
            }
            tracing::info!(
                comp_idx,
                seq_len = comp.input_ids.len(),
                action_tokens = num_active,
                env_tokens = comp_env_count,
                optimizer = ?config.optimizer,
                elapsed_ms = optimizer_started.elapsed().as_millis() as u64,
                "GRPO optimizer end"
            );
        }

        group_loss_sum += loss_val;
        if let Some(env_ce) = comp_echo_env_ce {
            if comp.total_obs_len > 0 {
                group_echo_ce_sum += env_ce * comp.total_obs_len as f64;
                group_echo_ce_weight = group_echo_ce_weight.saturating_add(comp.total_obs_len);
            }
        }
    }

    if token_level && !group_accum.is_empty() {
        observe_lora_grad_norms_from_map(grad_norms, lora_grad_index, &group_accum)?;
        let optimizer_started = Instant::now();
        tracing::info!(
            completions = tgroup.completions.len(),
            max_seq_len = group_max_seq_len,
            action_tokens = group_counts.action_tokens,
            env_tokens = group_counts.env_tokens,
            optimizer = ?config.optimizer,
            "GRPO optimizer start"
        );
        optimizer_step_from_map(
            backend,
            params,
            &group_accum,
            learning_rate,
            config.optimizer,
            opt_state.as_deref_mut(),
        )?;
        if let Some(t) = timings.as_deref_mut() {
            t.add_optimizer(optimizer_started.elapsed());
        }
        tracing::info!(
            completions = tgroup.completions.len(),
            max_seq_len = group_max_seq_len,
            action_tokens = group_counts.action_tokens,
            env_tokens = group_counts.env_tokens,
            optimizer = ?config.optimizer,
            elapsed_ms = optimizer_started.elapsed().as_millis() as u64,
            "GRPO optimizer end"
        );
    }

    let loss = if tgroup.completions.is_empty() {
        0.0
    } else if token_level {
        // Per-completion loss_val is already its share of the group-level mean
        // (each was scaled by 1/group_total_active). Sum across completions
        // gives the true group-level mean.
        group_loss_sum
    } else {
        group_loss_sum / tgroup.completions.len() as f64
    };
    let echo_env_ce = if group_echo_ce_weight > 0 {
        Some(group_echo_ce_sum / group_echo_ce_weight as f64)
    } else {
        None
    };
    Ok(GrpoGroupStepReport { loss, echo_env_ce })
}

// (#1082) `merge_grad_maps` removed: its sole caller was the candle
// gradient-checkpointed GRPO path (`checkpointed_grpo_forward_backward`),
// which was deleted in the candle drop. The kt-only GRPO token-level path
// accumulates directly via `accumulate_grads_dispatch`.

struct LoraGradNormIndex {
    // (#1082) keyed by each LoRA `Parameter::tensor_id()` (kt).
    modules_by_param: HashMap<KtTensorId, &'static str>,
}

impl LoraGradNormIndex {
    fn new(params: &TrainableLoraParams) -> Self {
        Self {
            modules_by_param: params
                .all_params_with_modules()
                .into_iter()
                .map(|entry| (entry.param.tensor_id(), entry.module))
                .collect(),
        }
    }
}

fn observe_lora_grad_norms_from_map(
    accumulator: &mut crate::train_receipt::LoraGradNormAccumulator,
    index: &LoraGradNormIndex,
    grads: &GradMap,
) -> Result<()> {
    let mut sum_sq_by_module: BTreeMap<&'static str, f64> = BTreeMap::new();
    for (id, grad) in grads {
        if let Some(module) = index.modules_by_param.get(id).copied() {
            accumulate_lora_grad_sum_sq(&mut sum_sq_by_module, module, grad)?;
        }
    }
    observe_lora_grad_module_norms(accumulator, sum_sq_by_module);
    Ok(())
}

/// (#1082) kt-native LoRA grad-norm observer: reads each LoRA
/// `Parameter`'s gradient from a kt-native [`kiln_autograd::GradStore`]
/// (keyed by `Parameter::tensor_id()`) and accumulates its squared L2
/// norm per module. The per-param norm is computed KT-NATIVELY via
/// `train_receipt::tensor_l2_norm_kt` (cast-to-F32 on-device + single D2H
/// scalar readback) — NO full-tensor kt->candle grad copy.
pub(crate) fn observe_lora_grad_norms_from_kt_grad_store(
    accumulator: &mut crate::train_receipt::LoraGradNormAccumulator,
    params: &TrainableLoraParams,
    grads: &kiln_autograd::GradStore,
) -> Result<()> {
    let mut sum_sq_by_module: BTreeMap<&'static str, f64> = BTreeMap::new();
    for entry in params.all_params_with_modules() {
        if let Some(kt_grad) = grads.get(entry.param.tensor_id()) {
            let norm = crate::train_receipt::tensor_l2_norm_kt(kt_grad).with_context(|| {
                format!("compute LoRA grad l2 norm (kt) for module {}", entry.module)
            })?;
            if norm.is_finite() {
                *sum_sq_by_module.entry(entry.module).or_insert(0.0) += norm * norm;
            } else {
                let value_summary = summarize_sft_debug_values(kt_grad)
                    .map(|(_, summary)| summary)
                    .unwrap_or_else(|e| format!("stats_error={e:#}"));
                tracing::warn!(
                    layer = entry.layer_idx,
                    module = entry.module,
                    matrix = entry.matrix,
                    tensor_id = entry.param.tensor_id().as_raw(),
                    dtype = ?kt_grad.dtype(),
                    shape = ?kt_grad.shape(),
                    device = %kt_grad.device(),
                    value_summary,
                    "skipping non-finite LoRA grad norm sample (kt)"
                );
            }
        }
    }
    observe_lora_grad_module_norms(accumulator, sum_sq_by_module);
    Ok(())
}

fn validate_lora_gradient_metadata(
    context: &str,
    leaf: &str,
    expected_shape: &[usize],
    expected_dtype: KtDType,
    expected_device: kiln_tensor::Device,
    observed_shape: &[usize],
    observed_dtype: KtDType,
    observed_device: kiln_tensor::Device,
) -> Result<()> {
    anyhow::ensure!(
        observed_shape == expected_shape,
        "{context}: LoRA gradient shape mismatch for {leaf}: expected={expected_shape:?} observed={observed_shape:?}"
    );
    anyhow::ensure!(
        observed_dtype == expected_dtype,
        "{context}: LoRA gradient dtype mismatch for {leaf}: expected={expected_dtype:?} observed={observed_dtype:?}"
    );
    anyhow::ensure!(
        observed_device == expected_device,
        "{context}: LoRA gradient device mismatch for {leaf}: expected={expected_device} observed={observed_device}"
    );
    Ok(())
}

fn validate_lora_gradient_tensor(
    context: &str,
    entry: &LoraParamRef<'_>,
    grad: &KtTensor,
    check_finite_values: bool,
) -> Result<()> {
    let id = entry.param.tensor_id();
    let leaf = format!(
        "layer={} module={} matrix={} tensor_id={}",
        entry.layer_idx, entry.module, entry.matrix, id
    );
    let master = entry.param.backward_storage().ok_or_else(|| {
        anyhow::anyhow!("{context}: configured trainable LoRA leaf has no master storage: {leaf}")
    })?;
    validate_lora_gradient_metadata(
        context,
        &leaf,
        master.shape(),
        entry.param.amp_policy().backward_compute_dtype,
        master.device(),
        grad.shape(),
        grad.dtype(),
        grad.device(),
    )?;
    if !check_finite_values {
        return Ok(());
    }

    let fast_finite = grad
        .all_finite()
        .with_context(|| format!("{context}: finite scan failed for LoRA gradient {leaf}"))?;
    if fast_finite {
        return Ok(());
    }

    let (cpu_finite, value_summary) = summarize_sft_debug_values(grad)
        .with_context(|| format!("{context}: CPU-confirming finite scan failed for {leaf}"))?;
    if cpu_finite {
        tracing::warn!(
            layer = entry.layer_idx,
            module = entry.module,
            matrix = entry.matrix,
            tensor_id = id.as_raw(),
            dtype = ?grad.dtype(),
            shape = ?grad.shape(),
            device = %grad.device(),
            value_summary,
            "{context}: backend finite reducer reported a non-finite LoRA gradient but CPU confirmation was finite"
        );
        return Ok(());
    }

    anyhow::bail!(
        "{context}: non-finite LoRA gradient {leaf} dtype={:?} shape={:?} device={} {}",
        grad.dtype(),
        grad.shape(),
        grad.device(),
        value_summary
    )
}

#[derive(Clone, Copy)]
enum ExpectedLoraGradientSet {
    WholeAdapter,
    CheckpointLayerRange,
}

fn validate_exact_lora_gradients<'p, 'g>(
    expected_entries: impl IntoIterator<Item = LoraParamRef<'p>>,
    observed_gradients: impl IntoIterator<Item = (KtTensorId, &'g KtTensor)>,
    context: &str,
    expected_set: ExpectedLoraGradientSet,
    check_finite_values: bool,
) -> Result<()> {
    let mut expected = BTreeMap::new();
    for entry in expected_entries {
        let id = entry.param.tensor_id();
        if let Some(previous) = expected.insert(id, entry) {
            anyhow::bail!(
                "{context}: duplicate configured LoRA tensor_id={id}: first=layer={} module={} matrix={} second=layer={} module={} matrix={}",
                previous.layer_idx,
                previous.module,
                previous.matrix,
                expected[&id].layer_idx,
                expected[&id].module,
                expected[&id].matrix
            );
        }
    }
    if matches!(expected_set, ExpectedLoraGradientSet::WholeAdapter) {
        anyhow::ensure!(
            !expected.is_empty(),
            "{context}: configured trainable LoRA leaf set is empty"
        );
    }
    let observed: BTreeMap<_, _> = observed_gradients.into_iter().collect();

    let missing = expected
        .iter()
        .filter(|(id, _)| !observed.contains_key(id))
        .map(|(id, entry)| {
            format!(
                "layer={} module={} matrix={} tensor_id={id}",
                entry.layer_idx, entry.module, entry.matrix
            )
        })
        .collect::<Vec<_>>();
    let unknown = observed
        .keys()
        .filter(|id| !expected.contains_key(id))
        .map(|id| format!("tensor_id={id}"))
        .collect::<Vec<_>>();
    anyhow::ensure!(
        missing.is_empty() && unknown.is_empty(),
        "{context}: exact LoRA gradient identity mismatch: configured={} observed={} missing=[{}] unknown=[{}]",
        expected.len(),
        observed.len(),
        missing.join(", "),
        unknown.join(", ")
    );

    for (id, entry) in &expected {
        let grad = observed
            .get(id)
            .expect("exact LoRA gradient identity check established membership");
        validate_lora_gradient_tensor(context, entry, grad, check_finite_values)?;
    }
    Ok(())
}

pub(crate) fn validate_exact_lora_grad_store(
    params: &TrainableLoraParams,
    grads: &kiln_autograd::GradStore,
    context: &str,
) -> Result<()> {
    validate_exact_lora_gradients(
        params.all_params_with_modules(),
        grads.iter().map(|(id, grad)| (*id, grad)),
        context,
        ExpectedLoraGradientSet::WholeAdapter,
        true,
    )
}

fn validate_exact_lora_grad_store_metadata(
    params: &TrainableLoraParams,
    grads: &kiln_autograd::GradStore,
    context: &str,
) -> Result<()> {
    validate_exact_lora_gradients(
        params.all_params_with_modules(),
        grads.iter().map(|(id, grad)| (*id, grad)),
        context,
        ExpectedLoraGradientSet::WholeAdapter,
        false,
    )
}

fn validate_exact_lora_grad_map(
    params: &TrainableLoraParams,
    grads: &GradMap,
    context: &str,
) -> Result<()> {
    validate_exact_lora_gradients(
        params.all_params_with_modules(),
        grads.iter().map(|(id, grad)| (*id, grad)),
        context,
        ExpectedLoraGradientSet::WholeAdapter,
        true,
    )
}

fn validate_exact_lora_grad_map_metadata(
    params: &TrainableLoraParams,
    grads: &GradMap,
    context: &str,
) -> Result<()> {
    validate_exact_lora_gradients(
        params.all_params_with_modules(),
        grads.iter().map(|(id, grad)| (*id, grad)),
        context,
        ExpectedLoraGradientSet::WholeAdapter,
        false,
    )
}

pub(crate) fn merge_checkpoint_lora_grad_segment(
    params: &TrainableLoraParams,
    accumulated: &mut kiln_autograd::GradStore,
    segment: kiln_autograd::GradStore,
    start_layer: usize,
    end_layer: usize,
    context: &str,
) -> Result<()> {
    anyhow::ensure!(
        start_layer < end_layer && end_layer <= params.layers.len(),
        "{context}: invalid checkpoint layer range {start_layer}..{end_layer} for {} layers",
        params.layers.len()
    );
    validate_exact_lora_gradients(
        params
            .all_params_with_modules()
            .into_iter()
            .filter(|entry| entry.layer_idx >= start_layer && entry.layer_idx < end_layer),
        segment.iter().map(|(id, grad)| (*id, grad)),
        context,
        ExpectedLoraGradientSet::CheckpointLayerRange,
        false,
    )?;
    let segment = segment.into_inner();
    let mut duplicate_ids = segment
        .keys()
        .copied()
        .filter(|id| accumulated.contains(*id))
        .collect::<Vec<_>>();
    duplicate_ids.sort_unstable();
    anyhow::ensure!(
        duplicate_ids.is_empty(),
        "{context}: duplicate checkpoint LoRA gradient tensor IDs across layer segments: [{}]",
        duplicate_ids
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );
    for (id, grad) in segment {
        accumulated.insert(id, grad);
    }
    Ok(())
}

fn accumulate_lora_grad_sum_sq(
    sum_sq_by_module: &mut BTreeMap<&'static str, f64>,
    module: &'static str,
    grad: &KtTensor,
) -> Result<()> {
    // (#1082) kt grad now; norm computed kt-natively.
    let norm = crate::train_receipt::tensor_l2_norm_kt(grad)
        .with_context(|| format!("compute LoRA grad l2 norm for module {module}"))?;
    if norm.is_finite() {
        *sum_sq_by_module.entry(module).or_insert(0.0) += norm * norm;
    } else {
        tracing::warn!(module, "skipping non-finite LoRA grad norm sample");
    }
    Ok(())
}

fn observe_lora_grad_module_norms(
    accumulator: &mut crate::train_receipt::LoraGradNormAccumulator,
    sum_sq_by_module: BTreeMap<&'static str, f64>,
) {
    for (module, sum_sq) in sum_sq_by_module {
        accumulator.observe(module, sum_sq.sqrt());
    }
}

/// Tokenize a GRPO group: prompt messages + each completion text.
///
/// When a rollout carries a populated `trajectory` field, this routes
/// through `crate::trajectory_mask::build_masks_from_trajectory` so the
/// resulting `TokenizedGrpoCompletion` carries proper `action_mask` and
/// `env_mask` separations. ECHO consumes both masks; the legacy GRPO
/// policy-gradient path consumes `action_mask` (aliased to
/// `completion_mask` for back-compat).
///
/// When a rollout has no trajectory (legacy single-string `text` only),
/// behaviour is bit-identical to the pre-ECHO path: `action_mask` is "true
/// after the prompt" and `env_mask` is all-false.
fn tokenize_grpo_group(group: &GrpoGroup, tokenizer: &KilnTokenizer) -> Result<TokenizedGrpoGroup> {
    let mask_cfg = crate::trajectory_mask::MaskConfig::default();
    tokenize_grpo_group_timed(group, tokenizer, &mask_cfg, None)
}

/// Validate one GRPO group's policy/provenance contract without loading model
/// weights or touching an accelerator. API admission uses this for recorded
/// behavior data so a doomed job cannot sit in the training queue.
pub fn validate_grpo_group_policy_data(
    group: &GrpoGroup,
    config: &GrpoConfig,
    tokenizer: &KilnTokenizer,
) -> Result<()> {
    validate_grpo_group_policy_data_and_max_seq_len(group, config, tokenizer, 1).map(|_| ())
}

/// Validate one streamed GRPO row and return its exact longest tokenized
/// completion length. Server admission uses this while scanning the complete
/// JSONL source, so its memory plan is based on the same tokenizer and masks as
/// the trainer rather than a character-count estimate.
pub fn validate_grpo_group_policy_data_and_max_seq_len(
    group: &GrpoGroup,
    config: &GrpoConfig,
    tokenizer: &KilnTokenizer,
    source_line: usize,
) -> Result<usize> {
    config
        .validate_policy_config()
        .map_err(|error| anyhow::anyhow!("GRPO policy config: {error}"))?;
    validate_grpo_trajectory_roles(group, source_line)?;
    let has_env_tokens = group.completions.iter().any(|completion| {
        completion
            .trajectory
            .iter()
            .any(|segment| segment.kind == TurnKind::Observation)
    });
    config
        .loss
        .validate_for_kt_tape(has_env_tokens)
        .map_err(|error| anyhow::anyhow!("GRPO loss config: {error}"))?;
    let mask_cfg = crate::trajectory_mask::MaskConfig::from_grpo_config(config);
    let tokenized = tokenize_grpo_group_timed(group, tokenizer, &mask_cfg, None)?;
    validate_tokenized_behavior_policy(&tokenized, config.behavior_policy)?;
    validate_grpo_dry_run_masks(&tokenized, source_line, source_line)?;
    Ok(tokenized
        .completions
        .iter()
        .map(|completion| completion.input_ids.len())
        .max()
        .unwrap_or(0))
}

fn tokenize_grpo_group_timed(
    group: &GrpoGroup,
    tokenizer: &KilnTokenizer,
    mask_cfg: &crate::trajectory_mask::MaskConfig,
    mut timings: Option<&mut GrpoBenchmarkTimings>,
) -> Result<TokenizedGrpoGroup> {
    if group.completions.is_empty() {
        anyhow::bail!("GRPO group has no completions");
    }

    let prompt_messages = to_core_messages(&group.messages);

    // Tokenize the prompt (without any assistant response). Used by the
    // legacy single-string path to find where the completion begins; the
    // trajectory-aware path computes its own boundaries via the mask
    // builder so it doesn't need this.
    let prompt_tokenize_started = Instant::now();
    let prompt_text = tokenizer
        .apply_chat_template(&prompt_messages)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let prompt_ids = tokenizer
        .encode(&prompt_text)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    if let Some(t) = timings.as_deref_mut() {
        t.add_tokenize(prompt_tokenize_started.elapsed());
    }

    let mut raw_rewards = Vec::with_capacity(group.completions.len());
    let mut full_message_batches = Vec::with_capacity(group.completions.len());

    // Pre-built per-completion (input_ids, action_mask, env_mask,
    // total_obs_len) for the trajectory-aware path, indexed parallel to
    // `full_message_batches`. `None` means "use the legacy single-string
    // path for this rollout".
    let mut prebuilt: Vec<Option<crate::trajectory_mask::MaskedRollout>> =
        Vec::with_capacity(group.completions.len());

    for scored in &group.completions {
        if scored.has_trajectory() {
            // Trajectory-aware path: build masks from the explicit
            // segment structure. The MaskedRollout is the canonical
            // output; full_message_batches gets a stub so the indices
            // stay aligned with `full_id_batches` below.
            let (masked, mask_timings) = crate::trajectory_mask::build_masks_from_trajectory_timed(
                &scored.trajectory,
                &group.messages,
                tokenizer,
                mask_cfg,
            )?;
            if let Some(t) = timings.as_deref_mut() {
                t.tokenize_ms += mask_timings.tokenize_ms;
                t.mask_build_ms += mask_timings.mask_build_ms;
            }
            prebuilt.push(Some(masked));
            // Placeholder; not used when prebuilt is Some.
            full_message_batches.push(prompt_messages.clone());
        } else if scored.provenance.is_some() {
            // Exact provenance owns the model sequence. Keep a cheap prompt
            // placeholder in the parallel batch; re-rendering completed text
            // is not equivalent to the sequence inference consumed because
            // chat templates append generation prefixes.
            full_message_batches.push(prompt_messages.clone());
            prebuilt.push(None);
        } else {
            // Legacy single-string path: assemble [prompt + assistant
            // completion] and tokenize as one chat batch (cheap because
            // we batch all completions in one apply_chat_template_batch
            // call).
            let mut full_messages = prompt_messages.clone();
            full_messages.push(kiln_core::tokenizer::ChatMessage {
                role: "assistant".to_string(),
                content: scored.text.clone(),
                ..Default::default()
            });
            full_message_batches.push(full_messages);
            prebuilt.push(None);
        }
        raw_rewards.push(scored.reward);
    }

    let batch_tokenize_started = Instant::now();
    let full_texts = tokenizer
        .apply_chat_template_batch(&full_message_batches)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let full_id_batches = tokenizer
        .encode_batch(&full_texts)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    if let Some(t) = timings.as_deref_mut() {
        t.add_tokenize(batch_tokenize_started.elapsed());
    }
    let mut completions = Vec::with_capacity(full_id_batches.len());
    let mut rewards = Vec::with_capacity(full_id_batches.len());
    for (completion_idx, ((full_ids, reward), pre)) in full_id_batches
        .into_iter()
        .zip(raw_rewards.into_iter())
        .zip(prebuilt.into_iter())
        .enumerate()
    {
        if let Some(provenance) = group.completions[completion_idx].provenance.as_ref() {
            provenance.validate().map_err(|error| {
                anyhow::anyhow!(
                    "completion {completion_idx} has invalid rollout provenance: {error}"
                )
            })?;

            let vocab_sha256 = tokenizer.vocab_identity_sha256();
            let config_sha256 = tokenizer
                .tokenizer_config_sha256()
                .map_err(|error| anyhow::anyhow!("hash tokenizer config: {error}"))?;
            let chat_template_sha256 = tokenizer.chat_template_sha256().with_context(|| {
                format!(
                    "completion {completion_idx} has recorded provenance but the training tokenizer has no chat template"
                )
            })?;
            anyhow::ensure!(
                provenance.tokenizer.vocab_sha256 == vocab_sha256,
                "completion {completion_idx} rollout tokenizer vocabulary identity mismatch: provenance={}, training={vocab_sha256}",
                provenance.tokenizer.vocab_sha256
            );
            anyhow::ensure!(
                provenance.tokenizer.config_sha256 == config_sha256,
                "completion {completion_idx} rollout tokenizer config identity mismatch: provenance={}, training={config_sha256}",
                provenance.tokenizer.config_sha256
            );
            anyhow::ensure!(
                provenance.tokenizer.chat_template_sha256 == chat_template_sha256,
                "completion {completion_idx} rollout chat-template identity mismatch: provenance={}, training={chat_template_sha256}",
                provenance.tokenizer.chat_template_sha256
            );

            let prompt_messages_sha256 = crate::rollout_prompt_messages_sha256(&group.messages)
                .map_err(anyhow::Error::msg)?;
            let scored_payload_sha256 =
                crate::scored_rollout_payload_sha256(&group.completions[completion_idx])
                    .map_err(anyhow::Error::msg)?;
            anyhow::ensure!(
                provenance.prompt_messages_sha256 == prompt_messages_sha256,
                "completion {completion_idx} prompt messages differ from rollout provenance"
            );
            anyhow::ensure!(
                provenance.scored_payload_sha256 == scored_payload_sha256,
                "completion {completion_idx} scored text/trajectory differs from rollout provenance"
            );

            let recorded_prompt_text = tokenizer
                .apply_chat_template_full_with_options(
                    &prompt_messages,
                    (!provenance.template_invocation.tools.is_empty())
                        .then_some(provenance.template_invocation.tools.as_slice()),
                    provenance.template_invocation.tool_choice.as_ref(),
                    kiln_core::tokenizer::ChatTemplateOptions {
                        template_kwargs: provenance
                            .template_invocation
                            .template_kwargs
                            .clone(),
                    },
                )
                .map_err(|error| {
                    anyhow::anyhow!(
                        "completion {completion_idx} could not replay its recorded chat-template invocation: {error}"
                    )
                })?;
            let recorded_prompt_ids = tokenizer.encode(&recorded_prompt_text).map_err(|error| {
                anyhow::anyhow!(
                    "completion {completion_idx} could not tokenize its replayed rollout prompt: {error}"
                )
            })?;
            anyhow::ensure!(
                provenance.prompt_token_count == recorded_prompt_ids.len(),
                "completion {completion_idx} rollout prompt boundary {} differs from the rendered prompt length {}",
                provenance.prompt_token_count,
                recorded_prompt_ids.len()
            );
            anyhow::ensure!(
                provenance.input_token_ids[..provenance.prompt_token_count] == recorded_prompt_ids,
                "completion {completion_idx} rollout input prefix differs from the rendered prompt tokens"
            );

            let (rendered_trajectory_ids, trajectory_action_mask, env_mask, total_obs_len) =
                if let Some(masked) = pre {
                    let total_obs_len = masked.total_obs_len();
                    let crate::trajectory_mask::MaskedRollout {
                        input_ids,
                        action_mask,
                        env_mask,
                        segment_spans: _,
                    } = masked;
                    (Some(input_ids), Some(action_mask), env_mask, total_obs_len)
                } else {
                    (None, None, vec![false; provenance.input_token_ids.len()], 0)
                };

            if let Some(rendered_input_ids) = rendered_trajectory_ids.as_ref() {
                anyhow::ensure!(
                    rendered_input_ids == &provenance.input_token_ids,
                    "completion {completion_idx} trajectory rendering differs from its exact rollout token sequence; exact observation masks cannot be recovered safely"
                );
            }
            let rendered_action_indices =
                if let Some(expected_action_mask) = trajectory_action_mask.as_ref() {
                    expected_action_mask
                        .iter()
                        .enumerate()
                        .filter_map(|(index, &active)| active.then_some(index))
                        .collect::<Vec<_>>()
                } else {
                    (provenance.prompt_token_count..provenance.input_token_ids.len())
                        .collect::<Vec<_>>()
                };
            let provenance_action_indices = provenance
                .action_tokens
                .iter()
                .map(|token| token.sequence_index)
                .collect::<Vec<_>>();
            anyhow::ensure!(
                rendered_action_indices == provenance_action_indices,
                "completion {completion_idx} scored payload action positions differ from rollout provenance"
            );

            let mut action_mask = vec![false; provenance.input_token_ids.len()];
            let mut behavior_log_probs = Vec::new();
            for action in &provenance.action_tokens {
                if action.source == crate::RolloutActionTokenSourceV1::Sampled {
                    action_mask[action.sequence_index] = true;
                    let logprob = action.behavior_logprob.with_context(|| {
                        format!(
                            "completion {completion_idx} sampled token {} is missing behavior_logprob",
                            action.sequence_index
                        )
                    })? as f32;
                    anyhow::ensure!(
                        logprob.is_finite(),
                        "completion {completion_idx} behavior_logprob at token {} cannot be represented as f32",
                        action.sequence_index
                    );
                    behavior_log_probs.push(logprob);
                }
            }
            anyhow::ensure!(
                action_mask
                    .iter()
                    .zip(env_mask.iter())
                    .all(|(&action, &env)| !(action && env)),
                "completion {completion_idx} has overlapping sampled-action and environment masks"
            );
            completions.push(TokenizedGrpoCompletion {
                input_ids: provenance.input_token_ids.clone(),
                prompt_token_count: provenance.prompt_token_count,
                action_mask,
                env_mask,
                total_obs_len,
                recorded_behavior_log_probs: Some(behavior_log_probs),
                recorded_behavior_source: Some(
                    crate::train_receipt::GrpoRecordedBehaviorSourceObservation::from_provenance(
                        provenance,
                    )
                    .with_context(|| {
                        format!(
                            "build completion {completion_idx} GRPO behavior-source observation"
                        )
                    })?,
                ),
            });
            rewards.push(reward);
            continue;
        }

        // Trajectory-aware path: prebuilt MaskedRollout overrides the
        // batch-tokenized full_ids. The mask builder rendered the
        // conversation itself, so its input_ids are authoritative for
        // the trajectory case.
        if let Some(masked) = pre {
            if masked.input_ids.len() < 2 {
                tracing::warn!(
                    "skipping trajectory completion: too short ({} tokens)",
                    masked.input_ids.len()
                );
                continue;
            }
            let total_obs_len = masked.total_obs_len();
            let crate::trajectory_mask::MaskedRollout {
                input_ids,
                action_mask,
                env_mask,
                segment_spans: _,
            } = masked;
            anyhow::ensure!(
                input_ids.len() >= prompt_ids.len() && input_ids[..prompt_ids.len()] == prompt_ids,
                "trajectory completion {completion_idx} does not preserve the rendered group prompt token prefix"
            );
            completions.push(TokenizedGrpoCompletion {
                input_ids,
                prompt_token_count: prompt_ids.len(),
                action_mask,
                env_mask,
                total_obs_len,
                recorded_behavior_log_probs: None,
                recorded_behavior_source: None,
            });
            rewards.push(reward);
            continue;
        }

        if full_ids.len() < 2 {
            tracing::warn!("skipping completion: too short ({} tokens)", full_ids.len());
            continue;
        }

        // Legacy single-string path: tokens after the prompt are
        // action tokens; there are no observation tokens.
        let mask_started = Instant::now();
        tracing::info!(
            seq_len = full_ids.len(),
            prompt_tokens = prompt_ids.len(),
            completion_tokens = full_ids.len().saturating_sub(prompt_ids.len()),
            "GRPO mask build start"
        );
        let mut action_mask = vec![false; full_ids.len()];
        for i in prompt_ids.len()..full_ids.len() {
            action_mask[i] = true;
        }
        let env_mask = vec![false; full_ids.len()];
        let mask_elapsed = mask_started.elapsed();
        if let Some(t) = timings.as_deref_mut() {
            t.add_mask_build(mask_elapsed);
        }
        tracing::info!(
            seq_len = full_ids.len(),
            action_tokens = action_mask.iter().filter(|&&active| active).count(),
            env_tokens = 0usize,
            elapsed_ms = mask_elapsed.as_secs_f64() * 1000.0,
            "GRPO mask build end"
        );

        completions.push(TokenizedGrpoCompletion {
            input_ids: full_ids,
            prompt_token_count: prompt_ids.len(),
            action_mask,
            env_mask,
            total_obs_len: 0,
            recorded_behavior_log_probs: None,
            recorded_behavior_source: None,
        });
        rewards.push(reward);
    }

    if completions.is_empty() {
        anyhow::bail!("no valid completions in GRPO group after tokenization");
    }

    Ok(TokenizedGrpoGroup {
        completions,
        rewards,
    })
}

/// Compute group-normalized advantages from rewards.
///
/// advantage_i = (reward_i - mean(rewards)) / (std(rewards) + 1e-8)
/// Deep-copy a tensor so the result's backing storage is independent of the
/// input's (which may be a [`Var`]'s storage that subsequent optimizer steps
/// can replace). Goes via host-side `f32` round-trip; restores the original
/// dtype on the way out.
///
/// Used by [`lora_snapshot_capture_or_blend`] to materialize a reference
/// LoRA that won't silently track future policy updates.
fn deepcopy_tensor_for_snapshot(t: &Tensor, snapshot_device: &Device) -> Result<Tensor> {
    let dtype = t.dtype();
    let shape = t.dims().to_vec();
    let host: Vec<f32> = t
        .to_f32_dtype()?
        .flatten_all()?
        .to_device(cpu_device())?
        .to_vec1::<f32>()
        .context("snapshot: read tensor to host f32 vec")?;
    // (#1082) kt-native rebuild on the source device (no candle constructor).
    let rebuilt = Tensor::from_vec_on(snapshot_device.clone(), host, shape)?;
    if dtype == DType::F32 {
        Ok(rebuilt.detach())
    } else {
        Ok(rebuilt.to_dtype(dtype)?.detach())
    }
}

/// EMA blend two tensors: `new = decay * old + (1 - decay) * current`. The
/// result has the same dtype as `old` and is independent of either input's
/// storage (the affine + add chain materializes a fresh tensor).
fn ema_blend_tensor(
    old: &Tensor,
    current: &Tensor,
    decay: f32,
    snapshot_device: &Device,
) -> Result<Tensor> {
    let dtype = old.dtype();
    let a = old
        .to_device(snapshot_device.clone())?
        .to_f32_dtype()?
        .affine(decay as f64, 0.0)?;
    let b = current
        .to_device(snapshot_device.clone())?
        .to_f32_dtype()?
        .affine((1.0 - decay) as f64, 0.0)?;
    let blended = (a + b)?;
    let out = if dtype == DType::F32 {
        blended
    } else {
        blended.to_dtype(dtype)?
    };
    Ok(out.detach())
}

/// Per-projection helper used by [`lora_snapshot_capture_or_blend`]. Given a
/// current trainable Var pair and an optional prior snapshot projection,
/// produces a fresh `LoraProjectionWeights` whose tensors are EMA-blended
/// from the snapshot toward the current params (or a pure deepcopy of
/// current if no prior snapshot exists).
fn snapshot_projection(
    cur: &Option<(Parameter, Parameter)>,
    prior: Option<&LoraProjectionWeights>,
    decay: f32,
    snapshot_device: &Device,
) -> Result<Option<LoraProjectionWeights>> {
    let Some((cur_a, cur_b)) = cur else {
        return Ok(None);
    };
    // (#1082) The EMA blend / deepcopy helpers (`ema_blend_tensor` /
    // `deepcopy_tensor_for_snapshot`) are now kt-native, and the param's
    // primary tensor + `LoraProjectionWeights.a/.b` are kt, so the whole
    // snapshot blend runs in kt with no candle bridge.
    let cur_a_kt = cur_a.forward_storage().primary_tensor();
    let cur_b_kt = cur_b.forward_storage().primary_tensor();
    let (a, b) = match prior {
        Some(prior) => (
            ema_blend_tensor(&prior.a, cur_a_kt, decay, snapshot_device)?,
            ema_blend_tensor(&prior.b, cur_b_kt, decay, snapshot_device)?,
        ),
        None => (
            deepcopy_tensor_for_snapshot(cur_a_kt, snapshot_device)?,
            deepcopy_tensor_for_snapshot(cur_b_kt, snapshot_device)?,
        ),
    };
    anyhow::ensure!(
        a.device() == *snapshot_device && b.device() == *snapshot_device,
        "GRPO EMA snapshot landed on {}/{} instead of {}",
        a.device(),
        b.device(),
        snapshot_device
    );
    Ok(Some(LoraProjectionWeights { a, b }))
}

/// Capture an EMA snapshot of the current LoRA params, blending with a prior
/// snapshot if provided.
///
/// * `prior = None`: a fresh deepcopy of `current` becomes the snapshot.
/// * `prior = Some(snap)`: returns `decay * snap + (1 - decay) * current`,
///   blended per-tensor.
///
/// Returned `LoraWeights` is fully owned (no aliasing of `current`'s Var
/// storage) and safe to pass as the reference into `model_forward_no_head`
/// across subsequent optimizer steps on `current`.
///
/// Used by [`KlReferencePolicy::Ema`] in `grpo_train` and `grpo_train_jsonl`.
fn lora_snapshot_capture_or_blend(
    current: &TrainableLoraParams,
    prior: Option<&LoraWeights>,
    decay: f32,
    snapshot_device: &Device,
) -> Result<LoraWeights> {
    let layers = current
        .layers
        .iter()
        .enumerate()
        .map(|(layer_idx, lp)| {
            let snap_layer = prior.and_then(|p| p.layers.get(layer_idx));
            // For each named projection, blend or deepcopy.
            let mk = |which: fn(&LoraLayerWeights) -> Option<&LoraProjectionWeights>,
                      cur: &Option<(Parameter, Parameter)>|
             -> Result<Option<LoraProjectionWeights>> {
                snapshot_projection(cur, snap_layer.and_then(which), decay, snapshot_device)
            };
            Ok::<LoraLayerWeights, anyhow::Error>(LoraLayerWeights {
                q_proj: mk(|l| l.q_proj.as_ref(), &lp.q_proj)?,
                k_proj: mk(|l| l.k_proj.as_ref(), &lp.k_proj)?,
                v_proj: mk(|l| l.v_proj.as_ref(), &lp.v_proj)?,
                o_proj: mk(|l| l.o_proj.as_ref(), &lp.o_proj)?,
                in_proj_qkv: mk(|l| l.in_proj_qkv.as_ref(), &lp.in_proj_qkv)?,
                in_proj_z: mk(|l| l.in_proj_z.as_ref(), &lp.in_proj_z)?,
                gdn_out_proj: mk(|l| l.gdn_out_proj.as_ref(), &lp.gdn_out_proj)?,
                gate_proj: mk(|l| l.gate_proj.as_ref(), &lp.gate_proj)?,
                up_proj: mk(|l| l.up_proj.as_ref(), &lp.up_proj)?,
                down_proj: mk(|l| l.down_proj.as_ref(), &lp.down_proj)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(LoraWeights {
        layers,
        mtp: None,
        rank: current.rank,
        alpha: current.alpha,
        scale: current.scale,
        source_identity: None,
    })
}

fn capture_lora_reference_checkpoint(snapshot: &LoraWeights) -> Result<CheckpointTensorSnapshot> {
    anyhow::ensure!(
        snapshot.mtp.is_none(),
        "GRPO EMA reference checkpoint must not contain MTP weights"
    );
    let mut tensors = Vec::new();
    for (layer_idx, layer) in snapshot.layers.iter().enumerate() {
        for (module, projection) in [
            ("q_proj", &layer.q_proj),
            ("k_proj", &layer.k_proj),
            ("v_proj", &layer.v_proj),
            ("o_proj", &layer.o_proj),
            ("in_proj_qkv", &layer.in_proj_qkv),
            ("in_proj_z", &layer.in_proj_z),
            ("out_proj", &layer.gdn_out_proj),
            ("gate_proj", &layer.gate_proj),
            ("up_proj", &layer.up_proj),
            ("down_proj", &layer.down_proj),
        ] {
            let Some(projection) = projection else {
                continue;
            };
            for (matrix, tensor) in [("A", &projection.a), ("B", &projection.b)] {
                let key = checkpoint_parameter_key(layer_idx, module, matrix);
                let tensor = tensor
                    .to_device(kiln_tensor::Device::Cpu)
                    .and_then(|tensor| tensor.contiguous())
                    .map_err(|error| {
                        anyhow::anyhow!("capture GRPO EMA reference tensor {key}: {error}")
                    })?;
                checkpoint_ensure_finite_tensor(&tensor, &key)?;
                tensors.push((key, tensor));
            }
        }
    }
    CheckpointTensorSnapshot::new(tensors, "GRPO EMA reference")
}

fn restore_lora_reference_tensor(
    loaded: &mut HashMap<String, KtTensor>,
    key: &str,
    current: &KtTensor,
    snapshot_device: &Device,
) -> Result<KtTensor> {
    let tensor = loaded
        .remove(key)
        .with_context(|| format!("GRPO EMA reference tensor {key} missing"))?;
    anyhow::ensure!(
        tensor.dims() == current.dims(),
        "GRPO EMA reference tensor {key} shape mismatch: expected {:?}, found {:?}",
        current.dims(),
        tensor.dims()
    );
    anyhow::ensure!(
        tensor.dtype() == current.dtype(),
        "GRPO EMA reference tensor {key} dtype mismatch: expected {}, found {}",
        current.dtype(),
        tensor.dtype()
    );
    checkpoint_ensure_finite_tensor(&tensor, key)?;
    tensor
        .to_device(snapshot_device.clone())
        .and_then(|tensor| tensor.contiguous())
        .map_err(|error| anyhow::anyhow!("restore GRPO EMA reference tensor {key}: {error}"))
}

fn restore_lora_reference_projection(
    loaded: &mut HashMap<String, KtTensor>,
    layer_idx: usize,
    module: &str,
    current: &Option<(Parameter, Parameter)>,
    snapshot_device: &Device,
) -> Result<Option<LoraProjectionWeights>> {
    let Some((current_a, current_b)) = current else {
        return Ok(None);
    };
    let a_key = checkpoint_parameter_key(layer_idx, module, "A");
    let b_key = checkpoint_parameter_key(layer_idx, module, "B");
    Ok(Some(LoraProjectionWeights {
        a: restore_lora_reference_tensor(
            loaded,
            &a_key,
            current_a.forward_storage().primary_tensor(),
            snapshot_device,
        )?,
        b: restore_lora_reference_tensor(
            loaded,
            &b_key,
            current_b.forward_storage().primary_tensor(),
            snapshot_device,
        )?,
    }))
}

fn load_lora_reference_checkpoint(
    path: &Path,
    current: &TrainableLoraParams,
    snapshot_device: &Device,
) -> Result<LoraWeights> {
    let mut loaded = kiln_tensor::safetensors::load_cpu(path)
        .map_err(|error| anyhow::anyhow!("load GRPO EMA reference checkpoint: {error}"))?;
    let expected: BTreeSet<_> = current.checkpoint_param_keys().into_iter().collect();
    let actual: BTreeSet<_> = loaded.keys().cloned().collect();
    anyhow::ensure!(
        actual == expected,
        "GRPO EMA reference tensor set mismatch: expected {expected:?}, found {actual:?}"
    );

    let layers = current
        .layers
        .iter()
        .enumerate()
        .map(|(layer_idx, layer)| {
            Ok::<_, anyhow::Error>(LoraLayerWeights {
                q_proj: restore_lora_reference_projection(
                    &mut loaded,
                    layer_idx,
                    "q_proj",
                    &layer.q_proj,
                    snapshot_device,
                )?,
                k_proj: restore_lora_reference_projection(
                    &mut loaded,
                    layer_idx,
                    "k_proj",
                    &layer.k_proj,
                    snapshot_device,
                )?,
                v_proj: restore_lora_reference_projection(
                    &mut loaded,
                    layer_idx,
                    "v_proj",
                    &layer.v_proj,
                    snapshot_device,
                )?,
                o_proj: restore_lora_reference_projection(
                    &mut loaded,
                    layer_idx,
                    "o_proj",
                    &layer.o_proj,
                    snapshot_device,
                )?,
                in_proj_qkv: restore_lora_reference_projection(
                    &mut loaded,
                    layer_idx,
                    "in_proj_qkv",
                    &layer.in_proj_qkv,
                    snapshot_device,
                )?,
                in_proj_z: restore_lora_reference_projection(
                    &mut loaded,
                    layer_idx,
                    "in_proj_z",
                    &layer.in_proj_z,
                    snapshot_device,
                )?,
                gdn_out_proj: restore_lora_reference_projection(
                    &mut loaded,
                    layer_idx,
                    "out_proj",
                    &layer.gdn_out_proj,
                    snapshot_device,
                )?,
                gate_proj: restore_lora_reference_projection(
                    &mut loaded,
                    layer_idx,
                    "gate_proj",
                    &layer.gate_proj,
                    snapshot_device,
                )?,
                up_proj: restore_lora_reference_projection(
                    &mut loaded,
                    layer_idx,
                    "up_proj",
                    &layer.up_proj,
                    snapshot_device,
                )?,
                down_proj: restore_lora_reference_projection(
                    &mut loaded,
                    layer_idx,
                    "down_proj",
                    &layer.down_proj,
                    snapshot_device,
                )?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    anyhow::ensure!(loaded.is_empty(), "unconsumed GRPO EMA reference tensors");
    Ok(LoraWeights {
        layers,
        mtp: None,
        rank: current.rank,
        alpha: current.alpha,
        scale: current.scale,
        source_identity: None,
    })
}

/// State threaded through a GRPO run to support `KlReferencePolicy::Ema`.
///
/// Captures the most recent snapshot of the LoRA params and the number of
/// completed groups since the last refresh. When `groups_since_refresh
/// >= refresh_every`, the outer caller calls
/// [`lora_snapshot_capture_or_blend`] with the current params and decay,
/// then resets the counter.
struct EmaReferenceState {
    snapshot: LoraWeights,
    groups_since_refresh: usize,
    refresh_every: usize,
    decay: f32,
}

/// Returns true when every completion in `group` carries the same reward.
/// Such a group produces a uniformly-zero advantage vector under any of the
/// supported [`AdvantageMode`]s and contributes no policy-gradient signal,
/// only a spurious KL update. Dropped by Dynamic Sampling
/// (DAPO, arXiv:2503.14476) when `GrpoConfig::dynamic_sampling` is true.
fn is_degenerate_grpo_group(group: &GrpoGroup) -> bool {
    let mut rewards = group.completions.iter().map(|c| c.reward);
    let Some(first) = rewards.next() else {
        return true;
    };
    rewards.all(|r| r == first)
}

fn compute_advantages(rewards: &[f64], mode: AdvantageMode) -> Vec<f64> {
    let n = rewards.len() as f64;
    if n <= 1.0 {
        return vec![0.0; rewards.len()];
    }
    let mean = rewards.iter().sum::<f64>() / n;
    let centered: Vec<f64> = rewards.iter().map(|r| r - mean).collect();
    match mode {
        AdvantageMode::DrGrpo => centered,
        AdvantageMode::Vanilla => {
            let var = centered.iter().map(|c| c * c).sum::<f64>() / n;
            let std = var.sqrt();
            centered.into_iter().map(|c| c / (std + 1e-8)).collect()
        }
    }
}

/// Compute per-token log-probabilities for the tokens indicated by the mask.
///
/// Returns a 1-D tensor of log-probs for only the masked (completion) positions.
/// Uses the next-token prediction convention: logits[i] predicts token[i+1].
// `pub(crate)` so the GRPO tape-authoritative loss-root shim
// (`crate::grpo_tape_shim`) can recompute the EXACT same policy log-probs
// inside its candle-autograd backward composite (#1082 CP-4).
pub(crate) fn token_log_probs(
    logits: &Tensor,
    input_ids: &[u32],
    mask: &[bool],
    device: &Device,
) -> Result<Tensor> {
    let seq_len = input_ids.len();
    let logits = logits.squeeze(0)?; // [seq_len, vocab_size]

    // Next-token prediction: logits[i] predicts input_ids[i+1]
    // So for completion token at position j, use logits[j-1]
    let shift_logits = logits.narrow(0, 0, seq_len - 1)?; // [seq_len-1, vocab_size]
    let shift_labels: Vec<u32> = input_ids[1..].to_vec();
    let shift_mask: Vec<bool> = mask[1..].to_vec();

    // Find active positions (completion tokens)
    let active_positions: Vec<usize> = shift_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i) } else { None })
        .collect();

    if active_positions.is_empty() {
        // Return a zero tensor if no completion tokens
        return zeros_f32_on(1, device).map_err(Into::into);
    }

    // Gather active logits
    let active_idx_u32: Vec<u32> = active_positions.iter().map(|&i| i as u32).collect();
    let n_active_idx = active_idx_u32.len();
    let indices = Tensor::from_vec_on(*device, active_idx_u32, vec![n_active_idx])?;
    let active_logits = shift_logits.index_select(&indices, 0)?; // [num_active, vocab_size]

    let active_labels: Vec<u32> = active_positions.iter().map(|&i| shift_labels[i]).collect();

    // log_softmax denominator (CUDA-capable reduce).
    let active_logits_f32 = active_logits.to_f32_dtype()?;
    let log_sum_exp = active_logits_f32.log_sum_exp(LAST_DIM)?; // [num_active]

    // correct_logits[a] = active_logits[a, label_a]. (#1082) kt `gather` is
    // CPU-only (gather.rs requires both indices AND x be CpuStorage), whereas
    // the candle gather it replaced ran on CUDA — so a direct `.gather` here
    // breaks the CUDA GRPO path. Select via a FLAT `index_select` instead, which
    // is CUDA-capable (the `shift_logits.index_select` above already established
    // that on-device U32 indices work) and stays on-device (a CPU round-trip of
    // the [num_active, vocab=248320] active logits would be prohibitive). Flatten
    // [num_active, vocab] -> [num_active*vocab] and index at a*vocab + label_a.
    let vocab_size = *active_logits_f32
        .dims()
        .last()
        .expect("active_logits_f32 has a last dim");
    let flat_idx: Vec<u32> = active_labels
        .iter()
        .enumerate()
        .map(|(a, &lbl)| (a * vocab_size + lbl as usize) as u32)
        .collect();
    let flat_indices = Tensor::from_vec_on(*device, flat_idx, vec![n_active_idx])?;
    let correct_logits = active_logits_f32
        .contiguous()?
        .flatten_all()? // [num_active*vocab]
        .index_select(&flat_indices, 0)?; // [num_active]

    // log_prob = logit - log_sum_exp
    let log_probs = (correct_logits - log_sum_exp)?;

    Ok(log_probs)
}

/// Select one logit per row from a chunked `[rows, chunk_len]` logits tile.
///
/// This is the chunked analogue of [`token_log_probs`]' flat-index
/// `index_select`: it avoids materializing a dense `[rows, chunk_len]`
/// one-hot tensor just to pick sparse target columns. Labels outside this
/// chunk contribute zero, so summing the returned chunks recovers the selected
/// full-vocab logits.
pub(crate) fn selected_logits_from_chunk_sparse(
    logits_chunk: &Tensor,
    target_ids: &[u32],
    chunk_start: usize,
    chunk_len: usize,
    vocab_size: usize,
    device: &Device,
    caller: &str,
) -> Result<Tensor> {
    let num_rows = target_ids.len();
    let dims = logits_chunk.dims();
    anyhow::ensure!(
        dims == [num_rows, chunk_len],
        "{caller}: logits_chunk shape {dims:?} != [{num_rows}, {chunk_len}]"
    );

    let mut row_indices = Vec::new();
    let mut flat_indices = Vec::new();
    for (row_idx, &label) in target_ids.iter().enumerate() {
        let label = label as usize;
        if label >= vocab_size {
            anyhow::bail!("{caller}: label {label} is outside vocab size {vocab_size}");
        }
        if label >= chunk_start && label < chunk_start + chunk_len {
            let rel = label - chunk_start;
            let flat = row_idx
                .checked_mul(chunk_len)
                .and_then(|base| base.checked_add(rel))
                .ok_or_else(|| anyhow::anyhow!("{caller}: flat selected-logit index overflow"))?;
            row_indices.push(
                u32::try_from(row_idx)
                    .with_context(|| format!("{caller}: row index {row_idx} exceeds u32 range"))?,
            );
            flat_indices.push(
                u32::try_from(flat)
                    .with_context(|| format!("{caller}: flat index {flat} exceeds u32 range"))?,
            );
        }
    }

    if flat_indices.is_empty() {
        return Tensor::zeros(vec![num_rows, 1], DType::F32, *device).map_err(Into::into);
    }

    let n_selected = flat_indices.len();
    let flat_idx = Tensor::from_vec_on(*device, flat_indices, vec![n_selected])?;
    let selected = logits_chunk
        .contiguous()?
        .flatten_all()?
        .index_select(&flat_idx, 0)?;
    let row_idx = Tensor::from_vec_on(*device, row_indices, vec![n_selected])?;
    let selected_rows = kiln_tensor::ops::scatter_add(&selected, 0, &row_idx, num_rows)?;
    selected_rows.unsqueeze(1).map_err(Into::into)
}

/// Compute selected next-token log-probs from post-final-RMSNorm hidden states
/// without materializing the full `[seq_len, vocab_size]` logits tensor.
fn selected_log_probs_from_normed_hidden_chunked(
    normed_hidden: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    mask: &[bool],
    chunk_size: usize,
) -> Result<Tensor> {
    let device = normed_hidden.device();
    let seq_len = input_ids.len();
    if seq_len < 2 {
        anyhow::bail!("selected log-probs require at least 2 tokens");
    }
    if mask.len() != seq_len {
        anyhow::bail!(
            "selected log-prob mask length {} does not match input length {}",
            mask.len(),
            seq_len
        );
    }
    if chunk_size == 0 {
        anyhow::bail!("selected log-prob chunk_size must be > 0");
    }

    let dims = normed_hidden.dims();
    if dims.len() != 3 || dims[0] != 1 || dims[1] != seq_len {
        anyhow::bail!(
            "normed_hidden must have shape [1, seq_len, hidden_size], got {:?}",
            dims
        );
    }
    let hidden_size = dims[2];
    if head_t.dims().len() != 2 || head_t.dims()[0] != hidden_size {
        anyhow::bail!(
            "head_t must have shape [hidden_size, vocab_size], got {:?}",
            head_t.dims()
        );
    }

    let active_positions: Vec<u32> = mask[1..]
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
        .collect();
    if active_positions.is_empty() {
        return zeros_f32_on(1, &device).map_err(Into::into);
    }
    let active_labels: Vec<u32> = active_positions
        .iter()
        .map(|&i| input_ids[i as usize + 1])
        .collect();

    let hidden_2d = normed_hidden.squeeze(0)?;
    let shift_hidden = hidden_2d.narrow(0, 0, seq_len - 1)?;
    let n_pos = active_positions.len();
    let active_indices = Tensor::from_vec_on(device, active_positions.clone(), vec![n_pos])?;
    let active_hidden = shift_hidden
        .index_select(&active_indices, 0)?
        .to_f32_dtype()?;

    let head_t_f32 = head_t.to_f32_dtype()?;
    let vocab_size = head_t_f32.dim(1)?;
    if vocab_size == 0 {
        anyhow::bail!("head_t vocab dimension is zero");
    }

    let mut running_max: Option<Tensor> = None;
    let mut running_sumexp: Option<Tensor> = None;
    let mut correct_logits: Option<Tensor> = None;
    let mut chunk_start = 0usize;
    while chunk_start < vocab_size {
        let chunk_len = chunk_size.min(vocab_size - chunk_start);
        let chunk_end = chunk_start + chunk_len;
        {
            let head_chunk = head_t_f32.narrow(1, chunk_start, chunk_len)?.contiguous()?;
            let logits_chunk = active_hidden.matmul(&head_chunk)?;
            let chunk_max = logits_chunk.max_keepdim(LAST_DIM)?;
            let (new_max, new_sumexp) = match (running_max.as_ref(), running_sumexp.as_ref()) {
                (None, None) => {
                    let shifted =
                        (&logits_chunk - chunk_max.broadcast_as(logits_chunk.shape())?)?;
                    let chunk_sumexp = shifted.exp()?.sum_keepdim(LAST_DIM)?;
                    (chunk_max.detach(), chunk_sumexp.detach())
                }
                (Some(prev_max), Some(prev_sumexp)) => {
                    let new_max = prev_max.maximum(&chunk_max)?;
                    let prev_scale = (prev_max - &new_max)?.exp()?;
                    let scaled_prev = prev_sumexp.broadcast_mul(&prev_scale)?;
                    let shifted = (&logits_chunk - new_max.broadcast_as(logits_chunk.shape())?)?;
                    let chunk_sumexp = shifted.exp()?.sum_keepdim(LAST_DIM)?;
                    let new_sumexp = (scaled_prev + chunk_sumexp)?;
                    (new_max.detach(), new_sumexp.detach())
                }
                _ => unreachable!("running max/sumexp are set together"),
            };
            running_max = Some(new_max);
            running_sumexp = Some(new_sumexp);

            let chunk_correct = selected_logits_from_chunk_sparse(
                &logits_chunk,
                &active_labels,
                chunk_start,
                chunk_len,
                vocab_size,
                &device,
                "selected_log_probs_from_normed_hidden_chunked",
            )?;
            correct_logits = Some(match correct_logits.as_ref() {
                Some(prev) => (prev + chunk_correct)?.detach(),
                None => chunk_correct.detach(),
            });
        }
        synchronize_tail_chunk("synchronize selected log-prob chunk")?;
        chunk_start = chunk_end;
    }

    let running_max = running_max.context("vocab_size was zero")?;
    let running_sumexp = running_sumexp.context("vocab_size was zero")?;
    let correct_logits = correct_logits.context("vocab_size was zero")?;
    let log_sum_exp = (running_max + running_sumexp.log()?)?;
    Ok((correct_logits - log_sum_exp)?.squeeze(1)?)
}

/// Tokenize a training example into (input_ids, label_mask).
///
/// The label_mask indicates which tokens are part of assistant responses
/// (true = compute loss here, false = ignore).
pub fn tokenize_for_training(
    example: &SftExample,
    tokenizer: &KilnTokenizer,
) -> Result<(Vec<u32>, Vec<bool>)> {
    let core_messages = to_core_messages(&example.messages);

    // Build the full conversation text using the chat template
    let (full_text, template_assistant_spans) = tokenizer
        .apply_chat_template_for_training_with_spans(&core_messages)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let (input_ids, offsets) = tokenizer
        .encode_with_offsets(&full_text)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    if input_ids.is_empty() {
        anyhow::bail!("empty tokenization result");
    }

    let assistant_count = core_messages
        .iter()
        .filter(|message| message.role == "assistant")
        .count();
    let mut label_mask = if let Some(spans) = template_assistant_spans {
        anyhow::ensure!(
            spans.len() == assistant_count,
            "training template returned {} assistant spans for {assistant_count} assistant messages",
            spans.len()
        );
        let mut mask = vec![false; input_ids.len()];
        for (start, end) in spans {
            mark_offsets_overlapping_span(&mut mask, &offsets, start, end);
        }
        mask
    } else {
        label_mask_from_rendered_assistant_spans(
            &full_text,
            &offsets,
            input_ids.len(),
            assistant_count,
        )
        .unwrap_or_else(|| vec![false; input_ids.len()])
    };
    // ChatML/Qwen-style templates are handled directly from the single rendered
    // full example. This avoids prefix renders that are not stable when
    // templates append generation prompts or rewrite post-tool turns.
    if !label_mask.iter().any(|&marked| marked) {
        let mut prefix_messages: Vec<kiln_core::tokenizer::ChatMessage> = Vec::new();
        for msg in &core_messages {
            if msg.role == "assistant" {
                let before_text = if prefix_messages.is_empty() {
                    String::new()
                } else {
                    tokenizer
                        .apply_chat_template_for_training(&prefix_messages)
                        .map_err(|e| anyhow::anyhow!("{e}"))?
                };

                prefix_messages.push(msg.clone());
                let prefix_text = tokenizer
                    .apply_chat_template_for_training(&prefix_messages)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;

                if !full_text.starts_with(&prefix_text) || before_text.len() > prefix_text.len() {
                    label_mask = label_mask_by_prefix_tokenization(
                        input_ids.len(),
                        &core_messages,
                        tokenizer,
                    )?;
                    break;
                }

                let start = before_text.len();
                let end = prefix_text.len().min(full_text.len());
                for (i, &(token_start, token_end)) in offsets.iter().enumerate() {
                    if token_start == token_end {
                        continue;
                    }
                    if token_start < end && token_end > start {
                        label_mask[i] = true;
                    }
                }
            } else {
                prefix_messages.push(msg.clone());
            }
        }
    }

    // For next-token prediction, we need at least 2 tokens
    if input_ids.len() < 2 {
        anyhow::bail!("example too short ({} tokens)", input_ids.len());
    }
    if !has_supervised_shifted_labels(&label_mask) {
        anyhow::bail!("example has no supervised assistant tokens after next-token shift");
    }

    Ok((input_ids, label_mask))
}

fn label_mask_from_rendered_assistant_spans(
    full_text: &str,
    offsets: &[(usize, usize)],
    input_len: usize,
    expected_assistant_spans: usize,
) -> Option<Vec<bool>> {
    const ASSISTANT_START: &str = "<|im_start|>assistant\n";
    const MESSAGE_END: &str = "<|im_end|>";

    if expected_assistant_spans == 0 {
        return Some(vec![false; input_len]);
    }

    let mut label_mask = vec![false; input_len];
    let mut search_from = 0usize;
    let mut found = 0usize;

    while let Some(relative_start) = full_text[search_from..].find(ASSISTANT_START) {
        let start = search_from + relative_start;
        let content_start = start + ASSISTANT_START.len();
        let Some(relative_end) = full_text[content_start..].find(MESSAGE_END) else {
            break;
        };
        let mut end = content_start + relative_end + MESSAGE_END.len();
        if full_text[end..].starts_with('\n') {
            end += 1;
        }

        // TRL's Qwen3.5 training template opens its generation span after the
        // assistant role header and closes it after the message terminator.
        mark_offsets_overlapping_span(&mut label_mask, offsets, content_start, end);
        found += 1;
        search_from = end;
    }

    (found == expected_assistant_spans).then_some(label_mask)
}

fn mark_offsets_overlapping_span(
    label_mask: &mut [bool],
    offsets: &[(usize, usize)],
    start: usize,
    end: usize,
) {
    for (index, &(token_start, token_end)) in offsets.iter().enumerate() {
        if index >= label_mask.len() || token_start == token_end {
            continue;
        }
        if token_start < end && token_end > start {
            label_mask[index] = true;
        }
    }
}

fn label_mask_by_prefix_tokenization(
    input_len: usize,
    core_messages: &[kiln_core::tokenizer::ChatMessage],
    tokenizer: &KilnTokenizer,
) -> Result<Vec<bool>> {
    let mut label_mask = vec![false; input_len];
    let mut prefix_messages: Vec<kiln_core::tokenizer::ChatMessage> = Vec::new();
    for msg in core_messages {
        prefix_messages.push(msg.clone());
        if msg.role == "assistant" {
            let prefix_text = tokenizer
                .apply_chat_template_for_training(&prefix_messages)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let prefix_ids = tokenizer
                .encode(&prefix_text)
                .map_err(|e| anyhow::anyhow!("{e}"))?;

            let before_messages: Vec<_> = prefix_messages[..prefix_messages.len() - 1].to_vec();
            let before_text = if before_messages.is_empty() {
                String::new()
            } else {
                tokenizer
                    .apply_chat_template_for_training(&before_messages)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
            };
            let before_ids = if before_text.is_empty() {
                Vec::new()
            } else {
                tokenizer
                    .encode(&before_text)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
            };

            let start = before_ids.len();
            let end = prefix_ids.len().min(input_len);
            for i in start..end {
                label_mask[i] = true;
            }
        }
    }
    Ok(label_mask)
}

fn has_supervised_shifted_labels(label_mask: &[bool]) -> bool {
    label_mask.get(1..).is_some_and(|m| m.iter().any(|&v| v))
}

/// Compute cross-entropy loss on masked positions.
///
/// `logits`: [1, seq_len, vocab_size] — model output
/// `input_ids`: token IDs (used as labels, shifted by 1)
/// `label_mask`: which positions to include in the loss
///
/// SFT next-token cross-entropy loss VALUE (scalar `f64`), kt-native.
///
/// (#1082 candle-drop) This is a value-only reader: it returns the scalar loss
/// for logging / the gradient-checkpoint final-boundary readback. The
/// *differentiable* CE root is `try_tape_cross_entropy_from_logits_kt` recorded
/// DIRECTLY by the SFT/GRPO/OPD `with_tape_authoritative_scope_kt` closures — it
/// does NOT go through here. The old candle `[1, T, V]` bridge + candle
/// log-sum-exp/gather composite + the candle `try_tape_cross_entropy_cuda`
/// adapter are deleted; unsupported backend/dtype combinations are rejected by
/// the backend tape route plus `TrainingPrecisionPolicy`. The kt CE math itself
/// is covered by `tape_forward_parity`
/// (`tape_forward_cross_entropy_matches_reference`,
/// `tape_backward_cross_entropy_matches_analytic_gradient`).
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
fn cross_entropy_loss(
    logits: &KtTensor,
    input_ids: &[u32],
    label_mask: &[bool],
    _device: &Device,
) -> Result<f64> {
    let loss_kt = kiln_model::tape_forward::try_tape_cross_entropy_from_logits_kt(
        logits, input_ids, label_mask,
    )?
    .ok_or_else(|| {
        anyhow::anyhow!(
            "cross_entropy_loss: kt CE-from-logits declined (requires CUDA BF16 [1, T, V] \
             logits; F32/CPU cross-entropy was dropped in the candle drop, #1082)"
        )
    })?;
    Ok(loss_kt.to_scalar::<f32>()? as f64)
}

/// Analytic SFT tail seed: `d loss / d hidden` for final RMSNorm + tied
/// LM-head + next-token cross-entropy.
///
/// This mirrors [`cross_entropy_loss`] / FLCE shifted-label semantics while
/// chunking over vocab so the full `[T, V]` logits tensor is never
/// materialized. The returned tensor is F32 with shape `[1, T, H]`; inactive
/// shifted-label rows and the final sequence row are zero.
fn synchronize_tail_chunk(_context: &'static str) -> Result<()> {
    // (#1082) kt `Device` has no per-device `synchronize()` (candle-only API);
    // the old chunk-tail sync point is retained as a named no-op for caller
    // structure without branching on backend identity here.
    Ok(())
}

fn analytic_sft_tail_grad_pre_final_norm(
    final_rmsnorm_backward_route: FinalRmsNormBackwardRoute,
    hidden: &Tensor,
    final_norm_weight: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    rms_norm_eps: f64,
    chunk_size: usize,
) -> Result<Tensor> {
    validate_analytic_sft_tail_grad_inputs(
        hidden,
        None,
        final_norm_weight,
        head_t,
        input_ids,
        label_mask,
        chunk_size,
    )?;
    let normed = rms_norm(hidden, final_norm_weight, rms_norm_eps)
        .context("analytic SFT tail final RMSNorm")?;
    analytic_sft_tail_grad_from_validated_normed_pre_final_norm(
        final_rmsnorm_backward_route,
        hidden,
        &normed,
        final_norm_weight,
        head_t,
        input_ids,
        label_mask,
        rms_norm_eps,
        chunk_size,
        None,
    )
}

fn analytic_sft_tail_grad_from_normed_pre_final_norm(
    final_rmsnorm_backward_route: FinalRmsNormBackwardRoute,
    hidden: &Tensor,
    normed: &Tensor,
    final_norm_weight: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    rms_norm_eps: f64,
    chunk_size: usize,
) -> Result<Tensor> {
    validate_analytic_sft_tail_grad_inputs(
        hidden,
        Some(normed),
        final_norm_weight,
        head_t,
        input_ids,
        label_mask,
        chunk_size,
    )?;
    analytic_sft_tail_grad_from_validated_normed_pre_final_norm(
        final_rmsnorm_backward_route,
        hidden,
        normed,
        final_norm_weight,
        head_t,
        input_ids,
        label_mask,
        rms_norm_eps,
        chunk_size,
        None,
    )
}

fn analytic_sft_tail_grad_from_normed_pre_final_norm_with_flce_metadata(
    final_rmsnorm_backward_route: FinalRmsNormBackwardRoute,
    hidden: &Tensor,
    normed: &Tensor,
    final_norm_weight: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    rms_norm_eps: f64,
    chunk_size: usize,
    active_metadata: Option<&kiln_flce_kernel::kt_api::FlceActiveMetadata>,
) -> Result<Tensor> {
    validate_analytic_sft_tail_grad_inputs(
        hidden,
        Some(normed),
        final_norm_weight,
        head_t,
        input_ids,
        label_mask,
        chunk_size,
    )?;
    analytic_sft_tail_grad_from_validated_normed_pre_final_norm(
        final_rmsnorm_backward_route,
        hidden,
        normed,
        final_norm_weight,
        head_t,
        input_ids,
        label_mask,
        rms_norm_eps,
        chunk_size,
        active_metadata,
    )
}

fn analytic_sft_tail_grad_from_validated_normed_pre_final_norm(
    final_rmsnorm_backward_route: FinalRmsNormBackwardRoute,
    hidden: &Tensor,
    normed: &Tensor,
    final_norm_weight: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    rms_norm_eps: f64,
    chunk_size: usize,
    active_metadata: Option<&kiln_flce_kernel::kt_api::FlceActiveMetadata>,
) -> Result<Tensor> {
    let grad_normed = if let Some(active_metadata) = active_metadata {
        kiln_flce_kernel::kt_api::fused_linear_cross_entropy_phase_b_backward_unit_grad_with_metadata_kt(
            normed, head_t, input_ids, label_mask, chunk_size, active_metadata,
        )
    } else {
        kiln_flce_kernel::kt_api::fused_linear_cross_entropy_phase_b_backward_unit_grad_kt(
            normed, head_t, input_ids, label_mask, chunk_size,
        )
    }
    .map_err(|e| anyhow::anyhow!("analytic SFT tail FLCE hidden gradient: {e}"))?;
    rms_norm_backward_pre_final_norm(
        final_rmsnorm_backward_route,
        hidden,
        final_norm_weight,
        &grad_normed,
        rms_norm_eps,
    )
    .context("analytic SFT tail final RMSNorm backward")
}

fn validate_analytic_sft_tail_grad_inputs(
    hidden: &Tensor,
    normed: Option<&Tensor>,
    final_norm_weight: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    chunk_size: usize,
) -> Result<()> {
    let seq_len = input_ids.len();
    if seq_len < 2 {
        anyhow::bail!("analytic SFT tail gradient requires at least 2 tokens");
    }
    if chunk_size == 0 {
        anyhow::bail!("analytic SFT tail gradient chunk_size must be > 0");
    }
    if label_mask.len() != seq_len {
        anyhow::bail!(
            "label_mask length {} does not match input_ids length {}",
            label_mask.len(),
            seq_len
        );
    }

    let dims = hidden.dims();
    if dims.len() != 3 || dims[0] != 1 || dims[1] != seq_len {
        anyhow::bail!(
            "hidden must have shape [1, seq_len, hidden_size], got {:?} for seq_len {}",
            dims,
            seq_len
        );
    }
    if let Some(normed) = normed {
        if normed.dims() != hidden.dims() {
            anyhow::bail!(
                "normed hidden shape {:?} does not match hidden shape {:?}",
                normed.dims(),
                hidden.dims()
            );
        }
    }
    let hidden_size = dims[2];
    if final_norm_weight.dims() != [hidden_size] {
        anyhow::bail!(
            "final_norm_weight shape {:?} does not match hidden size {}",
            final_norm_weight.dims(),
            hidden_size
        );
    }
    if head_t.dims().len() != 2 || head_t.dims()[0] != hidden_size {
        anyhow::bail!(
            "head_t must have shape [hidden_size, vocab_size], got {:?}",
            head_t.dims()
        );
    }

    Ok(())
}

pub(crate) fn rms_norm_backward_pre_final_norm(
    _final_rmsnorm_backward_route: FinalRmsNormBackwardRoute,
    hidden: &Tensor,
    final_norm_weight: &Tensor,
    grad_normed: &Tensor,
    rms_norm_eps: f64,
) -> Result<Tensor> {
    let dims = hidden.dims().to_vec();
    anyhow::ensure!(
        dims.len() == 3,
        "rms_norm_backward_pre_final_norm: hidden must be [batch, seq, hidden], got {dims:?}"
    );
    anyhow::ensure!(
        grad_normed.dims() == hidden.dims(),
        "rms_norm_backward_pre_final_norm: grad_normed shape {:?} != hidden shape {:?}",
        grad_normed.dims(),
        hidden.dims()
    );
    let hidden_size = dims[2];
    anyhow::ensure!(
        final_norm_weight.dims() == [hidden_size],
        "rms_norm_backward_pre_final_norm: final_norm_weight shape {:?} != hidden size {hidden_size}",
        final_norm_weight.dims()
    );

    #[cfg(any(feature = "cuda", feature = "rocm"))]
    {
        let same_device = final_norm_weight.device() == hidden.device()
            && grad_normed.device() == hidden.device();
        let non_empty_rows = dims[0] > 0 && dims[1] > 0;
        let fused_envelope = matches!(
            _final_rmsnorm_backward_route,
            FinalRmsNormBackwardRoute::CudaRocmFusedTail
        ) && same_device
            && non_empty_rows
            && kiln_rmsnorm_kernel::supports_rmsnorm_kt(hidden, final_norm_weight)
            && grad_normed.dtype() == KtDType::BF16
            && grad_normed.is_contiguous();

        if fused_envelope {
            let grad_hidden = kiln_rmsnorm_kernel::fused_rmsnorm_backward_dx_kt(
                hidden,
                final_norm_weight,
                grad_normed,
                rms_norm_eps as f32,
            )
            .map_err(|e| anyhow::anyhow!("fused final RMSNorm backward: {e}"))?;
            return Ok(grad_hidden.detach());
        }
    }

    let hidden_f32 = hidden.to_f32_dtype()?;
    let grad_normed_f32 = grad_normed.to_f32_dtype()?;
    let norm_weight = final_norm_weight.to_f32_dtype()?;
    let norm_weight_plus_one = (norm_weight.ones_like()? + norm_weight)?;
    let variance = hidden_f32.sqr()?.mean_keepdim(LAST_DIM)?;
    let rms_inv = (variance + rms_norm_eps)?.sqrt()?.recip()?;

    // Qwen RMSNorm: y = x * rsqrt(mean(x^2) + eps) * (1 + w).
    // Given dL/dy, the pre-norm gradient is:
    // u * r - x * r^3 / H * sum(u * x), where u = dL/dy * (1 + w).
    let u = grad_normed_f32.broadcast_mul(&norm_weight_plus_one)?;
    let dot = (&u * &hidden_f32)?.sum_keepdim(LAST_DIM)?;
    let rms_inv_sq = rms_inv.sqr()?;
    let rms_inv_cubed = rms_inv_sq.broadcast_mul(&rms_inv)?;
    let correction_scale = rms_inv_cubed.affine(1.0f64 / hidden_size as f64, 0.0)?;
    let correction = hidden_f32.broadcast_mul(&dot.broadcast_mul(&correction_scale)?)?;
    Ok((u.broadcast_mul(&rms_inv)? - correction)?.detach())
}

fn synchronize_training_tensor_ready(label: &str, tensor: &Tensor) -> Result<()> {
    match tensor.device() {
        Device::Cpu => Ok(()),
        #[cfg(feature = "cuda")]
        Device::Cuda(idx) => kiln_tensor::cuda_synchronize_default_stream(idx)
            .with_context(|| format!("{label}: synchronize CUDA tensor readiness")),
        #[cfg(feature = "rocm")]
        Device::Rocm(idx) => {
            if kiln_tensor::rocm_capture_arena_active() {
                Ok(())
            } else {
                kiln_tensor::rocm_synchronize_default_stream(idx)
                    .with_context(|| format!("{label}: synchronize ROCm tensor readiness"))
            }
        }
        #[cfg(feature = "metal")]
        Device::Metal(idx) => kiln_tensor::primary_metal_companion(idx)
            .and_then(|companion| companion.wait_until_completed())
            .with_context(|| format!("{label}: synchronize Metal tensor readiness")),
        #[cfg(feature = "vulkan")]
        Device::Vulkan(idx) => kiln_tensor::vulkan_synchronize_queue(idx)
            .with_context(|| format!("{label}: synchronize Vulkan tensor readiness")),
        _ => Ok(()),
    }
}

fn summarize_sft_debug_values(tensor: &Tensor) -> Result<(bool, String)> {
    let host = tensor
        .to_device(Device::Cpu)
        .context("copy SFT debug tensor to CPU")?
        .to_dtype(DType::F32)
        .context("cast SFT debug tensor to f32")?
        .contiguous()
        .context("make SFT debug CPU tensor contiguous")?;
    let values = host
        .to_vec::<f32>()
        .context("read SFT debug CPU tensor values")?;
    let mut first_bad: Option<(usize, f32)> = None;
    let mut max_abs = 0.0f32;
    let mut max_abs_idx = 0usize;
    for (idx, value) in values.iter().copied().enumerate() {
        if value.is_finite() {
            let abs = value.abs();
            if abs > max_abs {
                max_abs = abs;
                max_abs_idx = idx;
            }
        } else if first_bad.is_none() {
            first_bad = Some((idx, value));
        }
    }
    let shape = tensor.shape();
    let coord = |mut idx: usize| -> Vec<usize> {
        let mut out = vec![0usize; shape.len()];
        for axis in (0..shape.len()).rev() {
            let dim = shape[axis].max(1);
            out[axis] = idx % dim;
            idx /= dim;
        }
        out
    };
    let (bad_idx, bad_value) = first_bad.unwrap_or((usize::MAX, f32::NAN));
    let summary = format!(
        "first_bad_flat={} first_bad_coord={:?} first_bad_value={} max_finite_abs={} max_finite_abs_flat={} max_finite_abs_coord={:?}",
        bad_idx,
        if bad_idx == usize::MAX {
            Vec::new()
        } else {
            coord(bad_idx)
        },
        bad_value,
        max_abs,
        max_abs_idx,
        coord(max_abs_idx)
    );
    Ok((first_bad.is_none(), summary))
}

fn dtype_size_bytes(dtype: DType) -> usize {
    match dtype {
        DType::BF16 | DType::F16 => 2,
        DType::F32 => 4,
        DType::U8 => 1,
        DType::U32 => 4,
        DType::I64 => 8,
        _ => 4,
    }
}

struct StoredCheckpointBoundaries {
    tensors: std::cell::RefCell<Vec<Option<Tensor>>>,
    resident_device_storage: bool,
    anchor_stride: usize,
}

impl StoredCheckpointBoundaries {
    fn new(num_segments: usize, resident_device_storage: bool, anchor_stride: usize) -> Self {
        Self {
            tensors: std::cell::RefCell::new(vec![None; num_segments + 1]),
            resident_device_storage,
            anchor_stride: anchor_stride.max(1),
        }
    }

    fn should_store(&self, boundary_idx: usize) -> bool {
        boundary_idx == 0 || boundary_idx % self.anchor_stride == 0
    }

    fn anchor_for_boundary(&self, boundary_idx: usize) -> usize {
        (boundary_idx / self.anchor_stride) * self.anchor_stride
    }

    // Long-context checkpoint boundaries are too large to retain at every
    // segment boundary. Keep sparse anchors in process memory and replay from
    // the nearest anchor on demand.
    fn save(&self, boundary_idx: usize, tensor: &Tensor) -> Result<()> {
        if !self.should_store(boundary_idx) {
            return Ok(());
        }
        let stored = if self.resident_device_storage {
            tensor
                .contiguous()
                .map_err(|e| anyhow::anyhow!("checkpoint boundary save: contiguous: {e}"))?
        } else {
            tensor
                .to_device(kiln_tensor::Device::Cpu)
                .and_then(|t| t.contiguous())
                .map_err(|e| anyhow::anyhow!("checkpoint boundary save: to cpu: {e}"))?
        };
        let mut tensors = self.tensors.borrow_mut();
        let slot = tensors.get_mut(boundary_idx).ok_or_else(|| {
            anyhow::anyhow!("checkpoint boundary index {boundary_idx} out of storage range")
        })?;
        *slot = Some(stored);
        Ok(())
    }

    fn load_stored(&self, boundary_idx: usize, device: &Device) -> Result<Option<Tensor>> {
        let tensors = self.tensors.borrow();
        let Some(slot) = tensors.get(boundary_idx) else {
            anyhow::bail!("checkpoint boundary index {boundary_idx} out of spool range");
        };
        let Some(hidden) = slot.as_ref() else {
            return Ok(None);
        };
        if self.resident_device_storage {
            return Ok(Some(hidden.clone()));
        }
        Ok(Some(hidden.to_device(*device).map_err(|e| {
            anyhow::anyhow!("checkpoint boundary load: move to device: {e}")
        })?))
    }

    fn load(&self, boundary_idx: usize, device: &Device) -> Result<Tensor> {
        self.load_stored(boundary_idx, device)?.ok_or_else(|| {
            anyhow::anyhow!("checkpoint boundary {boundary_idx} missing hidden tensor")
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn load_or_recompute_checkpoint_boundary(
    spool: &StoredCheckpointBoundaries,
    boundary_idx: usize,
    backend: &dyn BackendRuntime,
    weights: &GpuWeights,
    model_config: &ModelConfig,
    positions: &[u32],
    segments: &[(usize, usize)],
    lora_detached: &LoraWeights,
    device: &Device,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<Tensor> {
    anyhow::ensure!(
        boundary_idx <= segments.len(),
        "checkpoint boundary {boundary_idx} out of range for {} segments",
        segments.len()
    );
    if let Some(stored) = spool.load_stored(boundary_idx, device)? {
        return Ok(stored);
    }

    let anchor_idx = spool.anchor_for_boundary(boundary_idx);
    let mut current = spool.load(anchor_idx, device)?;
    let mut linear_state = LinearAttentionState::new(model_config, device)?;
    for replay_idx in anchor_idx..boundary_idx {
        let (start, end) = segments[replay_idx];
        current = model_forward_segment_with_policy(
            backend,
            current,
            weights,
            model_config,
            positions,
            start,
            end,
            Some(&mut linear_state),
            Some(lora_detached),
            streaming_prefill,
        )
        .with_context(|| {
            format!("checkpoint boundary replay segment {replay_idx} layers {start}..{end}")
        })?
        .detach();
    }
    Ok(current)
}

/// Gradient output from one tape-authoritative training step.
///
/// The active tape scope is mandatory. Gradients are keyed by each configured
/// LoRA [`Parameter::tensor_id`] and remain kt-native through observation,
/// accumulation, exact-contract validation, and optimizer dispatch.
pub enum GradSource {
    /// kt-native gradients (the SOLE grad producer post-#1082). Keyed by
    /// `Parameter::tensor_id()`; values are `kiln_tensor::Tensor`. The
    /// candle `Candle(GradStore)` variant is GONE — every candle
    /// `loss.backward()` producer was deleted in the candle drop.
    Kt(kiln_autograd::GradStore),
}

impl GradSource {
    /// Number of parameters that received a gradient.
    pub fn num_grad_ids(&self) -> usize {
        match self {
            GradSource::Kt(kt) => kt.len(),
        }
    }

    /// Borrow the underlying kt `GradStore`.
    pub fn kt(&self) -> &kiln_autograd::GradStore {
        match self {
            GradSource::Kt(kt) => kt,
        }
    }

    /// Owned kt grad for `param`, or `None` if the store has no grad for
    /// it. Used by diagnostic / convergence-gate sites. (#1082)
    pub fn grad_for(&self, param: &Parameter) -> Option<KtTensor> {
        match self {
            GradSource::Kt(kt) => kt.get(param.tensor_id()).cloned(),
        }
    }
}

/// (#1082) Optimizer-step dispatcher over [`GradSource`] — kt-native only.
pub fn optimizer_step_dispatch(
    backend: &dyn BackendRuntime,
    params: &mut TrainableLoraParams,
    grads: &GradSource,
    lr: f64,
    optimizer: Optimizer,
    opt_state: Option<&mut OptimizerState>,
) -> Result<()> {
    match grads {
        GradSource::Kt(kt) => {
            optimizer_step_from_kt_grad_store(backend, params, kt, lr, optimizer, opt_state)
        }
    }
}

fn ensure_training_optimizer_fallback_allowed(
    backend: &dyn BackendRuntime,
    device: kiln_tensor::Device,
    optimizer_name: &'static str,
) -> Result<()> {
    let policy = BackendCapabilityQueries::backend_capabilities(backend)
        .fallback
        .training_optimizer;
    if policy.allows_fallback() {
        return Ok(());
    }
    anyhow::bail!(
        "{optimizer_name} optimizer fallback policy {:?} for {} training hot path on {}; \
         native optimizer dispatch is required and no runtime fallback override is supported",
        policy,
        BackendIdentity::runtime_name(backend),
        device.short_name()
    )
}

/// (#1082) LoRA grad-norm observer dispatcher — kt-native only.
pub fn observe_lora_grad_norms_dispatch(
    accumulator: &mut crate::train_receipt::LoraGradNormAccumulator,
    params: &TrainableLoraParams,
    grads: &GradSource,
) -> Result<()> {
    match grads {
        GradSource::Kt(kt) => observe_lora_grad_norms_from_kt_grad_store(accumulator, params, kt),
    }
}

/// Dispatch the configured optimizer against an exact kt gradient set.
/// Stateful optimizers increment their step only after the gradient contract
/// has accepted every configured leaf.
/// (#1082) Apply one kt-native SGD update (param = param - lr*grad) to a
/// single LoRA `Parameter`, preferring the on-device registry path when
/// param + grad are both resident (the backend trait takes kt tensors).
///
/// On-device path: register the kt grad → `OptimizerBackend` writes the param
/// buffer in place → evict the grad. The `Parameter`'s master is
/// left stale; `sync_to_master` pulls the registry back before save.
///
/// CPU fallback: compute `param - lr*grad` kt-natively and install it via
/// `replace_backward_storage` + `replace_forward_storage` (preserving
/// `tensor_id`).
fn apply_sgd_update_kt(
    backend: &dyn BackendRuntime,
    param: &mut Parameter,
    grad: &KtTensor,
    lr: f64,
    resident_activation: bool,
) -> Result<()> {
    let primary = param.forward_storage().primary_tensor().clone();
    if resident_activation && ResidencyBackend::runtime_has_resident_activation(backend, &primary) {
        ResidencyBackend::runtime_register_resident_activation(backend, grad)?;
        let dispatched =
            match OptimizerBackend::runtime_dispatch_sgd_step(backend, &primary, grad, lr as f32) {
                Ok(b) => b,
                Err(e) => {
                    ResidencyBackend::runtime_evict_resident_activation(backend, grad);
                    return Err(e);
                }
            };
        if dispatched {
            ResidencyBackend::runtime_evict_resident_activation(backend, grad);
            return Ok(());
        }
        ResidencyBackend::runtime_evict_resident_activation(backend, grad);
    }
    ensure_training_optimizer_fallback_allowed(backend, primary.device(), "SGD")?;
    // CPU/host fallback: master = master - lr*grad, kt-native (F32
    // accumulate then back to param dtype, mirroring the old candle math).
    let dtype = primary.dtype();
    let master_f32 = primary
        .to_dtype(KtDType::F32)
        .map_err(|e| anyhow::anyhow!("apply_sgd_update_kt: master to f32: {e}"))?;
    let grad_f32 = grad
        .to_dtype(KtDType::F32)
        .map_err(|e| anyhow::anyhow!("apply_sgd_update_kt: grad to f32: {e}"))?;
    let scaled = kiln_tensor::ops::mul_scalar(&grad_f32, lr as f32)
        .map_err(|e| anyhow::anyhow!("apply_sgd_update_kt: grad*lr: {e}"))?;
    let updated_f32 = kiln_tensor::ops::sub(&master_f32, &scaled)
        .map_err(|e| anyhow::anyhow!("apply_sgd_update_kt: master-update: {e}"))?;
    let updated = updated_f32
        .to_dtype(dtype)
        .map_err(|e| anyhow::anyhow!("apply_sgd_update_kt: back to {dtype:?}: {e}"))?;
    param
        .replace_plain_trainable_tensor(updated)
        .map_err(|error| anyhow::anyhow!("apply_sgd_update_kt: preserve identity: {error}"))?;
    if resident_activation {
        ResidencyBackend::runtime_update_resident_activation(
            backend,
            param.forward_storage().primary_tensor(),
        )?;
    }
    Ok(())
}

/// (#1082) Apply one AdamW step to a single LoRA `Parameter`.
///
/// On-device path (resident): when the param **and** its `m`/`v` device
/// moment tensors are all resident, dispatch the CUDA AdamW kernel which
/// updates **param, m, and v in place** in one launch. This is the
/// production path (BF16 CUDA, LoRA params resident). The `m`/`v` passed
/// are the REAL per-param device moments from `OptimizerState.moments`
/// (NOT the param aliased onto itself — that was the C1 corruption bug).
/// The forward storage shares the master tensor (LoRA A/B are plain dense
/// BF16, forward primary == master), so the in-place param update is
/// immediately visible to the next forward; no refresh needed.
///
/// Host fallback (non-resident): drive the CPU reference
/// `kiln_optim::AdamW` (`OptimStep::step`), which owns its own host-side
/// moments keyed by `Parameter::tensor_id()` and installs the new master
/// via `replace_backward_storage` (preserving `tensor_id`). The forward
/// storage is refreshed from the new master.
///
/// `lr`/`beta1`/`beta2`/`eps`/`weight_decay` are threaded directly from
/// the optimizer config (no more `ADAMW_ACTIVE_HP` thread-local shim —
/// that hack existed only because the moments were host-side and the
/// device path had no real hp source). `step` is the global 1-indexed
/// step counter (shared by all params for standard AdamW bias correction).
///
/// `grad` must match the param's AMP `backward_compute_dtype` (BF16 in
/// production). The exact gradient contract checks this before any optimizer
/// state or parameter is mutated.
#[allow(clippy::too_many_arguments)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OptimizerStateAuthority {
    Device,
    Host,
}

fn apply_adamw_update_kt(
    backend: &dyn BackendRuntime,
    param: &mut Parameter,
    adamw: &mut KtAdamW,
    moments: Option<&KtAdamWMoments>,
    grad: &KtTensor,
    lr: f64,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: u32,
    resident_activation: bool,
) -> Result<OptimizerStateAuthority> {
    let primary = param.forward_storage().primary_tensor().clone();
    // On-device registry path: param + grad + the REAL per-param m/v must
    // all be resident, then the CUDA kernel updates param/m/v in place.
    if let Some(moments) = moments {
        if resident_activation
            && ResidencyBackend::runtime_has_resident_activation(backend, &primary)
            && ResidencyBackend::runtime_has_resident_activation(backend, &moments.m)
            && ResidencyBackend::runtime_has_resident_activation(backend, &moments.v)
        {
            ResidencyBackend::runtime_register_resident_activation(backend, grad)?;
            let dispatched = match OptimizerBackend::runtime_dispatch_adamw_step(
                backend,
                &primary,
                grad,
                &moments.m,
                &moments.v,
                lr as f32,
                beta1,
                beta2,
                eps,
                weight_decay,
                step,
            ) {
                Ok(b) => b,
                Err(e) => {
                    ResidencyBackend::runtime_evict_resident_activation(backend, grad);
                    return Err(e);
                }
            };
            if dispatched {
                // The kernel updated param/m/v in place. Forward primary IS
                // the master for LoRA params, so the update is already live;
                // re-assert residency of the param buffer for the next fwd.
                ResidencyBackend::runtime_evict_resident_activation(backend, grad);
                ResidencyBackend::runtime_update_resident_activation(backend, &primary)?;
                return Ok(OptimizerStateAuthority::Device);
            }
            ResidencyBackend::runtime_evict_resident_activation(backend, grad);
        }
    }

    ensure_training_optimizer_fallback_allowed(backend, primary.device(), "AdamW")?;
    // Host fallback: drive the CPU reference `kiln_optim::AdamW`. The exact
    // gradient contract already established the policy dtype, shape, and
    // device, so this boundary must not coerce an invalid gradient.
    adamw
        .step(param, grad)
        .map_err(|e| anyhow::anyhow!("apply_adamw_update_kt: kiln_optim AdamW step: {e}"))?;
    // `AdamW::step` swaps the master via `replace_backward_storage`
    // (preserving tensor_id). Refresh the forward storage from the new
    // master so the next forward reads the updated weights.
    if let Some(new_master) = param.backward_storage().cloned() {
        param
            .replace_plain_trainable_tensor(new_master)
            .map_err(|error| anyhow::anyhow!("AdamW preserve parameter identity: {error}"))?;
    }
    if resident_activation {
        ResidencyBackend::runtime_update_resident_activation(
            backend,
            param.forward_storage().primary_tensor(),
        )?;
    }
    Ok(OptimizerStateAuthority::Host)
}

/// (#1082) Apply one Muon step to a single LoRA `Parameter`.
///
/// On-device path (resident): when the param **and** its per-param
/// momentum device tensor are both resident, dispatch the fused Muon
/// kernel (`runtime_dispatch_muon_step`) which updates **param and
/// momentum in place** in one launch — heavy-ball momentum, then (for
/// rank-2 matrices) Newton-Schulz orthogonalization of the (Nesterov)
/// look-ahead with the RMS-matching scale, then the decoupled-weight-
/// decay descent step. The forward storage shares the master tensor
/// (LoRA A/B are plain dense BF16, forward primary == master), so the
/// in-place update is immediately visible to the next forward.
///
/// Host fallback (non-resident): drive the CPU reference
/// `kiln_optim::Muon` (`OptimStep::step`), which owns its own host-side
/// momentum keyed by `Parameter::tensor_id()` and installs the new
/// master via `replace_backward_storage` (preserving `tensor_id`). The
/// forward storage is refreshed from the new master.
///
/// `lr` and the Muon hyperparameters are threaded directly from the
/// optimizer config each step (honouring the LR schedule).
#[allow(clippy::too_many_arguments)]
fn apply_muon_update_kt(
    backend: &dyn BackendRuntime,
    param: &mut Parameter,
    muon: &mut KtMuon,
    momentum_state: Option<&KtMuonMomentum>,
    grad: &KtTensor,
    lr: f64,
    momentum: f32,
    nesterov: bool,
    ns_iters: u32,
    weight_decay: f32,
    resident_activation: bool,
) -> Result<OptimizerStateAuthority> {
    let primary = param.forward_storage().primary_tensor().clone();
    // On-device registry path: param + grad + the per-param momentum
    // must all be resident, then the kernel updates param/momentum in
    // place.
    if let Some(momentum_state) = momentum_state {
        if resident_activation
            && ResidencyBackend::runtime_has_resident_activation(backend, &primary)
            && ResidencyBackend::runtime_has_resident_activation(backend, &momentum_state.m)
        {
            ResidencyBackend::runtime_register_resident_activation(backend, grad)?;
            let dispatched = match OptimizerBackend::runtime_dispatch_muon_step(
                backend,
                &primary,
                grad,
                &momentum_state.m,
                lr as f32,
                momentum,
                nesterov,
                ns_iters,
                weight_decay,
            ) {
                Ok(b) => b,
                Err(e) => {
                    ResidencyBackend::runtime_evict_resident_activation(backend, grad);
                    return Err(e);
                }
            };
            if dispatched {
                // The kernel updated param/momentum in place. Forward
                // primary IS the master for LoRA params, so the update
                // is already live; re-assert residency for the next fwd.
                ResidencyBackend::runtime_evict_resident_activation(backend, grad);
                ResidencyBackend::runtime_update_resident_activation(backend, &primary)?;
                return Ok(OptimizerStateAuthority::Device);
            }
            ResidencyBackend::runtime_evict_resident_activation(backend, grad);
        }
    }

    ensure_training_optimizer_fallback_allowed(backend, primary.device(), "Muon")?;
    // Host fallback: drive the CPU reference `kiln_optim::Muon`. Thread
    // the scheduled lr + config hyperparameters in so the host path
    // honours the LR schedule and keeps the optimizer config the single
    // source of truth. The host Muon owns its own host-side momentum +
    // step counter keyed by `tensor_id`.
    muon.lr = lr as f32;
    muon.momentum = momentum;
    muon.nesterov = nesterov;
    muon.ns_iters = ns_iters;
    muon.weight_decay = weight_decay;
    // The exact gradient contract has already checked dtype/shape/device;
    // silently casting here would turn a producer defect into a plausible step.
    muon.step(param, grad)
        .map_err(|e| anyhow::anyhow!("apply_muon_update_kt: kiln_optim Muon step: {e}"))?;
    // `Muon::step` swaps the master via `replace_backward_storage`
    // (preserving tensor_id). Refresh the forward storage from the new
    // master so the next forward reads the updated weights.
    if let Some(new_master) = param.backward_storage().cloned() {
        param
            .replace_plain_trainable_tensor(new_master)
            .map_err(|error| anyhow::anyhow!("Muon preserve parameter identity: {error}"))?;
    }
    if resident_activation {
        ResidencyBackend::runtime_update_resident_activation(
            backend,
            param.forward_storage().primary_tensor(),
        )?;
    }
    Ok(OptimizerStateAuthority::Host)
}

/// (#1082) Accumulate kt gradients from a kt-native [`kiln_autograd::GradStore`]
/// (keyed by `Parameter::tensor_id()`) into `dst` (a kt `GradMap`).
/// The source must contain exactly one shape/dtype/device-valid gradient for
/// every configured LoRA leaf. Entries are created on the first source and
/// summed thereafter; gradients stay on-device. The optimizer boundary scans
/// the final accumulated values once before any mutation.
pub(crate) fn accumulate_grads(
    dst: &mut GradMap,
    src: &kiln_autograd::GradStore,
    params: &TrainableLoraParams,
) -> Result<()> {
    validate_exact_lora_grad_store_metadata(params, src, "accumulate_grads source")?;
    for (id, grad) in src.iter() {
        if let Some(existing) = dst.get(id) {
            let summed = kiln_tensor::ops::add(existing, grad)
                .map_err(|e| anyhow::anyhow!("accumulate_grads: kt add: {e}"))?;
            dst.insert(*id, summed);
        } else {
            dst.insert(*id, grad.clone());
        }
    }
    // Adding exact sources preserves the destination id set and tensor
    // metadata. Keep this post-merge guard cheap; the final optimizer boundary
    // performs the one required finite-value scan of the accumulated result.
    validate_exact_lora_grad_map_metadata(params, dst, "accumulate_grads result")?;
    Ok(())
}

/// (#1082) [`accumulate_grads`] dispatcher over [`GradSource`] for the
/// GRPO token-level aggregation boundary — kt-native only now. Routes the
/// kt `GradStore` straight into the kt `GradMap` keyed by
/// `Parameter::tensor_id()`.
fn accumulate_grads_dispatch(
    dst: &mut GradMap,
    src: &GradSource,
    params: &TrainableLoraParams,
) -> Result<()> {
    match src {
        GradSource::Kt(kt) => accumulate_grads(dst, kt, params),
    }
}

/// (#1082) SGD update from an accumulated kt gradient map (keyed by
/// `Parameter::tensor_id()`).
fn sgd_step_from_map(
    backend: &dyn BackendRuntime,
    params: &mut TrainableLoraParams,
    grads: &GradMap,
    lr: f64,
) -> Result<()> {
    let resident_activation = ResidencyBackend::runtime_supports_resident_activation(backend);
    for param in params.all_params_mut() {
        let id = param.tensor_id();
        let grad = grads.get(&id).ok_or_else(|| {
            anyhow::anyhow!("sgd_step_from_map: exact gradient contract lost tensor_id={id}")
        })?;
        apply_sgd_update_kt(backend, param, grad, lr, resident_activation)?;
    }
    Ok(())
}

/// (#1082) Configured-optimizer dispatch from an accumulated kt gradient
/// map (keyed by `Parameter::tensor_id()`). Drives `kiln_optim::AdamW`
/// (or kt SGD) per param.
pub(crate) fn optimizer_step_from_map(
    backend: &dyn BackendRuntime,
    params: &mut TrainableLoraParams,
    grads: &GradMap,
    lr: f64,
    optimizer: Optimizer,
    opt_state: Option<&mut OptimizerState>,
) -> Result<()> {
    validate_exact_lora_grad_map(params, grads, "optimizer_step_from_map")?;
    match optimizer {
        Optimizer::Sgd => sgd_step_from_map(backend, params, grads, lr),
        Optimizer::AdamW {
            beta1,
            beta2,
            eps,
            weight_decay,
        } => {
            let state = opt_state.ok_or_else(|| {
                anyhow::anyhow!("optimizer_step_from_map: AdamW requires OptimizerState")
            })?;
            let resident_activation =
                ResidencyBackend::runtime_supports_resident_activation(backend);
            // Global 1-indexed step counter (shared by all params), bumped
            // once per optimizer step for AdamW bias correction. Disjoint
            // borrows of `adamw` (mut, host fallback) vs `moments` (shared,
            // device m/v) via the match binding.
            match state {
                OptimizerState::AdamW {
                    adamw,
                    moments,
                    host_authoritative,
                    step,
                } => {
                    *step = step.saturating_add(1);
                    let step = *step;
                    for param in params.all_params_mut() {
                        let id = param.tensor_id();
                        let grad = grads.get(&id).ok_or_else(|| {
                            anyhow::anyhow!(
                                "optimizer_step_from_map: exact gradient contract lost tensor_id={id}"
                            )
                        })?;
                        let m = moments.get(&id);
                        let authority = apply_adamw_update_kt(
                            backend,
                            param,
                            adamw,
                            m,
                            grad,
                            lr,
                            beta1,
                            beta2,
                            eps,
                            weight_decay,
                            step,
                            resident_activation,
                        )?;
                        match authority {
                            OptimizerStateAuthority::Device => {
                                host_authoritative.remove(&id);
                            }
                            OptimizerStateAuthority::Host => {
                                host_authoritative.insert(id);
                            }
                        }
                    }
                    Ok(())
                }
                _ => anyhow::bail!(
                    "optimizer_step_from_map: AdamW optimizer requires AdamW OptimizerState"
                ),
            }
        }
        Optimizer::Muon {
            momentum,
            nesterov,
            ns_iters,
            weight_decay,
        } => {
            let state = opt_state.ok_or_else(|| {
                anyhow::anyhow!("optimizer_step_from_map: Muon requires OptimizerState")
            })?;
            let resident_activation =
                ResidencyBackend::runtime_supports_resident_activation(backend);
            match state {
                OptimizerState::Muon {
                    muon,
                    momenta,
                    host_authoritative,
                    step,
                } => {
                    *step = step.saturating_add(1);
                    for param in params.all_params_mut() {
                        let id = param.tensor_id();
                        let grad = grads.get(&id).ok_or_else(|| {
                            anyhow::anyhow!(
                                "optimizer_step_from_map: exact gradient contract lost tensor_id={id}"
                            )
                        })?;
                        let mom = momenta.get(&id);
                        let authority = apply_muon_update_kt(
                            backend,
                            param,
                            muon,
                            mom,
                            grad,
                            lr,
                            momentum,
                            nesterov,
                            ns_iters,
                            weight_decay,
                            resident_activation,
                        )?;
                        match authority {
                            OptimizerStateAuthority::Device => {
                                host_authoritative.remove(&id);
                            }
                            OptimizerStateAuthority::Host => {
                                host_authoritative.insert(id);
                            }
                        }
                    }
                    Ok(())
                }
                _ => anyhow::bail!(
                    "optimizer_step_from_map: Muon optimizer requires Muon OptimizerState"
                ),
            }
        }
    }
}

/// (#1082) kt-native-grad consumer — the SOLE optimizer consumer post
/// candle-drop. Reads gradients from a kt-native
/// [`kiln_autograd::GradStore`] (keyed by `Parameter::tensor_id()`,
/// values `kiln_tensor::Tensor`), produced by
/// [`standard_forward_backward_tape_authoritative_kt`] /
/// [`grpo_step_forward_backward_tape_authoritative_kt`] / the
/// checkpointed kt producer.
///
/// For each LoRA `Parameter` it looks the grad up by `tensor_id()` and
/// steps the param kt-natively: SGD via [`apply_sgd_update_kt`], AdamW via
/// `kiln_optim::AdamW` (`OptimStep::step`) inside [`apply_adamw_update_kt`].
/// NO candle grad copy, NO candle `Var` master — the LoRA `Parameter`'s kt
/// master is updated in place (preserving `tensor_id`).
pub(crate) fn optimizer_step_from_kt_grad_store(
    backend: &dyn BackendRuntime,
    params: &mut TrainableLoraParams,
    grads: &kiln_autograd::GradStore,
    lr: f64,
    optimizer: Optimizer,
    opt_state: Option<&mut OptimizerState>,
) -> Result<()> {
    validate_exact_lora_grad_store(params, grads, "optimizer_step_from_kt_grad_store")?;
    match optimizer {
        Optimizer::Sgd => {
            let resident_activation =
                ResidencyBackend::runtime_supports_resident_activation(backend);
            for param in params.all_params_mut() {
                let id = param.tensor_id();
                let kt_grad = grads.get(id).ok_or_else(|| {
                    anyhow::anyhow!(
                        "optimizer_step_from_kt_grad_store: exact gradient contract lost tensor_id={id}"
                    )
                })?;
                apply_sgd_update_kt(backend, param, kt_grad, lr, resident_activation)?;
            }
            Ok(())
        }
        Optimizer::AdamW {
            beta1,
            beta2,
            eps,
            weight_decay,
        } => {
            let state = opt_state.ok_or_else(|| {
                anyhow::anyhow!("optimizer_step_from_kt_grad_store: AdamW requires OptimizerState")
            })?;
            let resident_activation =
                ResidencyBackend::runtime_supports_resident_activation(backend);
            // Global 1-indexed step counter (shared by all params), bumped
            // once per optimizer step for AdamW bias correction. Disjoint
            // borrows of `adamw` (mut, host fallback) vs `moments` (shared,
            // device m/v) via the match binding.
            match state {
                OptimizerState::AdamW {
                    adamw,
                    moments,
                    host_authoritative,
                    step,
                } => {
                    *step = step.saturating_add(1);
                    let step = *step;
                    for param in params.all_params_mut() {
                        let id = param.tensor_id();
                        let kt_grad = grads.get(id).ok_or_else(|| {
                            anyhow::anyhow!(
                                "optimizer_step_from_kt_grad_store: exact gradient contract lost tensor_id={id}"
                            )
                        })?;
                        let m = moments.get(&id);
                        let authority = apply_adamw_update_kt(
                            backend,
                            param,
                            adamw,
                            m,
                            kt_grad,
                            lr,
                            beta1,
                            beta2,
                            eps,
                            weight_decay,
                            step,
                            resident_activation,
                        )?;
                        match authority {
                            OptimizerStateAuthority::Device => {
                                host_authoritative.remove(&id);
                            }
                            OptimizerStateAuthority::Host => {
                                host_authoritative.insert(id);
                            }
                        }
                    }
                    Ok(())
                }
                _ => anyhow::bail!(
                    "optimizer_step_from_kt_grad_store: AdamW optimizer requires AdamW OptimizerState"
                ),
            }
        }
        Optimizer::Muon {
            momentum,
            nesterov,
            ns_iters,
            weight_decay,
        } => {
            let state = opt_state.ok_or_else(|| {
                anyhow::anyhow!("optimizer_step_from_kt_grad_store: Muon requires OptimizerState")
            })?;
            let resident_activation =
                ResidencyBackend::runtime_supports_resident_activation(backend);
            match state {
                OptimizerState::Muon {
                    muon,
                    momenta,
                    host_authoritative,
                    step,
                } => {
                    *step = step.saturating_add(1);
                    for param in params.all_params_mut() {
                        let id = param.tensor_id();
                        let kt_grad = grads.get(id).ok_or_else(|| {
                            anyhow::anyhow!(
                                "optimizer_step_from_kt_grad_store: exact gradient contract lost tensor_id={id}"
                            )
                        })?;
                        let mom = momenta.get(&id);
                        let authority = apply_muon_update_kt(
                            backend,
                            param,
                            muon,
                            mom,
                            kt_grad,
                            lr,
                            momentum,
                            nesterov,
                            ns_iters,
                            weight_decay,
                            resident_activation,
                        )?;
                        match authority {
                            OptimizerStateAuthority::Device => {
                                host_authoritative.remove(&id);
                            }
                            OptimizerStateAuthority::Host => {
                                host_authoritative.insert(id);
                            }
                        }
                    }
                    Ok(())
                }
                _ => anyhow::bail!(
                    "optimizer_step_from_kt_grad_store: Muon optimizer requires Muon OptimizerState"
                ),
            }
        }
    }
}

/// Gradient checkpointing configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointConfig {
    /// Number of segments to split layers into.
    pub num_segments: usize,
    /// Whether checkpointing is enabled.
    pub enabled: bool,
    /// Whether num_segments was auto-configured from VRAM detection.
    pub auto_configured: bool,
}

impl CheckpointConfig {
    pub fn from_resolved_segments(num_layers: usize, num_segments: usize) -> Self {
        let num_segments = num_segments.min(num_layers).max(1);
        Self {
            num_segments,
            enabled: num_segments > 1,
            auto_configured: true,
        }
    }

    /// Standalone constructor with VRAM-aware automatic defaults.
    ///
    /// This is the *VRAM-only* path. Callers that know the workload's
    /// `max_seq_len` should prefer [`CheckpointConfig::auto_for_workload`],
    /// which can additionally choose to *disable* checkpointing when the
    /// activation tape comfortably fits in available VRAM (typical on big
    /// GPUs with short prompts).
    pub fn standalone(num_layers: usize) -> Self {
        let runtime = crate::TrainingRuntimeContext::standalone();
        Self::from_runtime(num_layers, &runtime)
    }

    /// Standalone compatibility with an already-resolved capacity.
    pub fn standalone_with_vram(num_layers: usize, vram: &kiln_memory::vram::GpuVramInfo) -> Self {
        let runtime = crate::TrainingRuntimeContext::standalone_with_effective_vram(*vram);
        Self::from_runtime(num_layers, &runtime)
    }

    /// Deprecated compatibility name. This function does not read environment.
    #[deprecated(note = "use CheckpointConfig::standalone or CheckpointConfig::from_runtime")]
    pub fn from_env(num_layers: usize) -> Self {
        Self::standalone(num_layers)
    }

    /// Deprecated compatibility name. This function does not read environment.
    #[deprecated(note = "use CheckpointConfig::standalone_with_vram or from_runtime")]
    pub fn from_env_with_vram(num_layers: usize, vram: &kiln_memory::vram::GpuVramInfo) -> Self {
        Self::standalone_with_vram(num_layers, vram)
    }

    /// Resolve a VRAM-only checkpoint configuration from immutable inputs.
    pub fn from_runtime(num_layers: usize, runtime: &crate::TrainingRuntimeContext) -> Self {
        use crate::GradientCheckpointPolicy;

        let vram = runtime.effective_vram();
        match runtime.gradient_checkpoint_policy() {
            GradientCheckpointPolicy::ExplicitSegments { segments } => {
                let mut config = Self::from_resolved_segments(num_layers, segments.get());
                config.auto_configured = false;
                return config;
            }
            GradientCheckpointPolicy::Disabled {
                segments: Some(segments),
            } => {
                return Self {
                    num_segments: segments.get().min(num_layers).max(1),
                    enabled: false,
                    auto_configured: false,
                };
            }
            GradientCheckpointPolicy::Auto
            | GradientCheckpointPolicy::Disabled { segments: None } => {}
        }

        // VRAM-aware auto-configuration
        let num_segments = kiln_memory::vram::recommended_checkpoint_segments(vram)
            .unwrap_or(4) // conservative fallback when capacity is unknown
            .min(num_layers)
            .max(1);

        let auto_configured = vram.source != kiln_memory::vram::VramSource::None;

        if auto_configured {
            tracing::info!(
                num_segments,
                vram_gb = vram.total_bytes as f64 / 1e9,
                source = %vram.source,
                "auto-configured gradient checkpoint segments for detected VRAM"
            );
        }

        Self {
            num_segments,
            enabled: !runtime.gradient_checkpoint_policy().is_disabled(),
            auto_configured,
        }
    }

    /// Create config with **VRAM + workload-shape** auto-tuning. Preferred over
    /// [`CheckpointConfig::standalone`] for trainer call sites that have the
    /// `max_seq_len` available after tokenization.
    ///
    /// Standalone wrappers detect physical capacity once into a
    /// [`crate::TrainingRuntimeContext`]. Runtime-aware callers receive the
    /// server's resolved typed configuration.
    ///
    /// In auto mode this calls [`kiln_memory::vram::recommended_checkpoint_plan`]
    ///   which can *disable* checkpointing entirely when the activation tape
    ///   comfortably fits in available VRAM. On A6000 + Qwen3.5-4B, this
    ///   skips checkpointing for sequences up to ~12K tokens and only
    ///   engages it (with the right number of segments) for longer contexts.
    ///
    /// `bytes_per_base_param` is used to estimate base-model footprint —
    /// pass 2 for BF16 (canonical kiln inference dtype) or 4 for F32.
    #[allow(clippy::too_many_arguments)]
    pub fn auto_for_workload(
        num_layers: usize,
        max_seq_len_tokens: usize,
        hidden_size: usize,
        intermediate_size: usize,
        vocab_size: usize,
        bytes_per_base_param: usize,
    ) -> Self {
        let runtime = crate::TrainingRuntimeContext::standalone();
        Self::auto_for_workload_with_activation_bytes_and_runtime(
            num_layers,
            max_seq_len_tokens,
            hidden_size,
            intermediate_size,
            vocab_size,
            bytes_per_base_param,
            4,
            &runtime,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn auto_for_workload_with_vram(
        num_layers: usize,
        max_seq_len_tokens: usize,
        hidden_size: usize,
        intermediate_size: usize,
        vocab_size: usize,
        bytes_per_base_param: usize,
        vram: &kiln_memory::vram::GpuVramInfo,
    ) -> Self {
        let runtime = crate::TrainingRuntimeContext::standalone_with_effective_vram(*vram);
        Self::auto_for_workload_with_activation_bytes_and_runtime(
            num_layers,
            max_seq_len_tokens,
            hidden_size,
            intermediate_size,
            vocab_size,
            bytes_per_base_param,
            4,
            &runtime,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn auto_for_workload_with_activation_bytes(
        num_layers: usize,
        max_seq_len_tokens: usize,
        hidden_size: usize,
        intermediate_size: usize,
        vocab_size: usize,
        bytes_per_base_param: usize,
        activation_bytes_per_elem: usize,
    ) -> Self {
        let runtime = crate::TrainingRuntimeContext::standalone();
        Self::auto_for_workload_with_activation_bytes_and_runtime(
            num_layers,
            max_seq_len_tokens,
            hidden_size,
            intermediate_size,
            vocab_size,
            bytes_per_base_param,
            activation_bytes_per_elem,
            &runtime,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn auto_for_workload_with_activation_bytes_and_vram(
        num_layers: usize,
        max_seq_len_tokens: usize,
        hidden_size: usize,
        intermediate_size: usize,
        vocab_size: usize,
        bytes_per_base_param: usize,
        activation_bytes_per_elem: usize,
        vram: &kiln_memory::vram::GpuVramInfo,
    ) -> Self {
        let runtime = crate::TrainingRuntimeContext::standalone_with_effective_vram(*vram);
        Self::auto_for_workload_with_activation_bytes_and_runtime(
            num_layers,
            max_seq_len_tokens,
            hidden_size,
            intermediate_size,
            vocab_size,
            bytes_per_base_param,
            activation_bytes_per_elem,
            &runtime,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn auto_for_workload_with_activation_bytes_and_runtime(
        num_layers: usize,
        max_seq_len_tokens: usize,
        hidden_size: usize,
        intermediate_size: usize,
        vocab_size: usize,
        bytes_per_base_param: usize,
        activation_bytes_per_elem: usize,
        runtime: &crate::TrainingRuntimeContext,
    ) -> Self {
        match runtime.gradient_checkpoint_policy() {
            crate::GradientCheckpointPolicy::Auto => {}
            crate::GradientCheckpointPolicy::ExplicitSegments { segments } => {
                let mut config = Self::from_resolved_segments(num_layers, segments.get());
                config.auto_configured = false;
                return config;
            }
            crate::GradientCheckpointPolicy::Disabled { segments } => {
                return Self {
                    num_segments: segments.map_or(1, |value| value.get().min(num_layers).max(1)),
                    enabled: false,
                    auto_configured: false,
                };
            }
        }

        let vram = runtime.effective_vram();

        let base_bytes = kiln_memory::vram::estimate_base_model_bytes(
            num_layers,
            hidden_size,
            intermediate_size,
            vocab_size,
            bytes_per_base_param,
        );

        match kiln_memory::vram::recommended_checkpoint_plan_with_activation_bytes(
            vram,
            num_layers,
            max_seq_len_tokens,
            hidden_size,
            base_bytes,
            activation_bytes_per_elem,
        ) {
            None => Self::from_runtime(num_layers, runtime),
            Some(kiln_memory::vram::CheckpointPlan::Disabled {
                max_act_gib,
                available_gib,
            }) => {
                tracing::info!(
                    max_seq_len_tokens,
                    activation_bytes_per_elem,
                    activation_tape_gib = format!("{max_act_gib:.2}"),
                    available_gib = format!("{available_gib:.2}"),
                    vram_total_gb = vram.total_bytes as f64 / 1e9,
                    vram_source = %vram.source,
                    "auto-tuned: gradient checkpointing DISABLED — activation tape fits comfortably in available VRAM"
                );
                Self {
                    num_segments: 1,
                    enabled: false,
                    auto_configured: true,
                }
            }
            Some(kiln_memory::vram::CheckpointPlan::Enabled {
                num_segments,
                max_act_gib,
                per_segment_gib,
                available_gib,
            }) => {
                tracing::info!(
                    num_segments,
                    max_seq_len_tokens,
                    activation_bytes_per_elem,
                    activation_tape_gib = format!("{max_act_gib:.2}"),
                    per_segment_gib = format!("{per_segment_gib:.2}"),
                    available_gib = format!("{available_gib:.2}"),
                    vram_total_gb = vram.total_bytes as f64 / 1e9,
                    vram_source = %vram.source,
                    "auto-tuned: gradient checkpointing engaged for workload shape"
                );
                Self {
                    num_segments,
                    enabled: true,
                    auto_configured: true,
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn checkpoint_config_for_training_step(
    weights: &GpuWeights,
    device: &Device,
    preflight_resolved_segments: Option<usize>,
    num_layers: usize,
    seq_len_tokens: usize,
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    bytes_per_base_param: usize,
    activation_bytes_per_elem: usize,
    runtime: &crate::TrainingRuntimeContext,
) -> CheckpointConfig {
    match runtime.gradient_checkpoint_policy() {
        crate::GradientCheckpointPolicy::Auto => {}
        crate::GradientCheckpointPolicy::ExplicitSegments { segments } => {
            let mut config = CheckpointConfig::from_resolved_segments(
                num_layers,
                preflight_resolved_segments.unwrap_or(segments.get()),
            );
            config.auto_configured = false;
            return config;
        }
        crate::GradientCheckpointPolicy::Disabled { segments } => {
            return CheckpointConfig {
                num_segments: segments
                    .map(std::num::NonZeroUsize::get)
                    .or(preflight_resolved_segments)
                    .unwrap_or(1)
                    .min(num_layers)
                    .max(1),
                enabled: false,
                auto_configured: false,
            };
        }
    }

    if let Some(resolved_segments) = preflight_resolved_segments {
        // Server admission resolves against live memory after the model and KV
        // cache are resident. That exact plan is stricter than replanning from
        // the immutable startup capacity and must remain authoritative.
        return CheckpointConfig::from_resolved_segments(num_layers, resolved_segments);
    }

    let mut cfg = CheckpointConfig::auto_for_workload_with_activation_bytes_and_runtime(
        num_layers,
        seq_len_tokens,
        hidden_size,
        intermediate_size,
        vocab_size,
        bytes_per_base_param,
        activation_bytes_per_elem,
        runtime,
    );

    if let Some(num_segments) =
        long_context_full_attention_forced_checkpoint_segments(weights, device, seq_len_tokens)
        && (!cfg.enabled || cfg.num_segments < num_segments)
    {
        tracing::info!(
            seq_len_tokens,
            num_segments,
            "auto-tuned: gradient checkpointing engaged for long-context full-attention tape pressure"
        );
        cfg.enabled = num_segments > 1;
        cfg.num_segments = num_segments;
        cfg.auto_configured = true;
    }

    cfg
}

fn long_context_full_attention_forced_checkpoint_segments(
    weights: &GpuWeights,
    device: &Device,
    seq_len_tokens: usize,
) -> Option<usize> {
    const MIN_TOKENS: usize = 8 * 1024;

    if seq_len_tokens < MIN_TOKENS {
        return None;
    }
    if !matches!(
        device,
        Device::Cuda(_) | Device::Rocm(_) | Device::Metal(_) | Device::Vulkan(_)
    ) {
        return None;
    }
    let full_attention_layers = weights
        .layers
        .iter()
        .filter(|layer| matches!(layer.attention, GpuAttentionWeights::Full(_)))
        .count();
    if full_attention_layers == 0 {
        return None;
    }

    Some(weights.layers.len().max(1))
}

fn checkpoint_segments_for_config(
    weights: &GpuWeights,
    device: &Device,
    seq_len_tokens: usize,
    ckpt_config: CheckpointConfig,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Option<Vec<(usize, usize)>> {
    if !ckpt_config.enabled {
        return None;
    }
    let mut boundaries = compute_segment_boundaries(weights.layers.len(), ckpt_config.num_segments);
    if ckpt_config.auto_configured
        && (materialized_full_attention_checkpoint_refinement_needed(
            weights,
            device,
            seq_len_tokens,
            streaming_prefill,
        ) || rocm_online_full_attention_checkpoint_refinement_needed(
            weights,
            device,
            seq_len_tokens,
            streaming_prefill,
        ))
    {
        let refined = refine_segments_for_materialized_full_attention(weights, &boundaries);
        if refined.len() > boundaries.len() {
            tracing::info!(
                seq_len = seq_len_tokens,
                original_segments = boundaries.len(),
                refined_segments = refined.len(),
                "refined gradient checkpoint boundaries for materialized full-attention replay"
            );
            boundaries = refined;
        }
    }
    Some(boundaries)
}

fn rocm_online_full_attention_checkpoint_refinement_needed(
    weights: &GpuWeights,
    device: &Device,
    seq_len_tokens: usize,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> bool {
    const MIN_TOKENS: usize = 8 * 1024;

    if seq_len_tokens < MIN_TOKENS || !streaming_prefill.enabled_for(seq_len_tokens) {
        return false;
    }
    if !matches!(device, Device::Rocm(_)) {
        return false;
    }
    weights
        .layers
        .iter()
        .filter(|layer| matches!(layer.attention, GpuAttentionWeights::Full(_)))
        .count()
        > 1
}

fn materialized_full_attention_checkpoint_refinement_needed(
    weights: &GpuWeights,
    device: &Device,
    seq_len_tokens: usize,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> bool {
    if !streaming_prefill.enabled_for(seq_len_tokens) {
        return false;
    }
    if !matches!(device, Device::Metal(_) | Device::Vulkan(_)) {
        return false;
    }
    weights
        .layers
        .iter()
        .filter(|layer| matches!(layer.attention, GpuAttentionWeights::Full(_)))
        .count()
        > 1
}

fn refine_segments_for_materialized_full_attention(
    weights: &GpuWeights,
    boundaries: &[(usize, usize)],
) -> Vec<(usize, usize)> {
    let mut refined = Vec::with_capacity(boundaries.len());
    for &(start, end) in boundaries {
        if start >= end {
            continue;
        }
        let mut seg_start = start;
        let mut full_attn_in_segment = 0usize;
        for layer_idx in start..end {
            if matches!(
                weights.layers[layer_idx].attention,
                GpuAttentionWeights::Full(_)
            ) {
                if full_attn_in_segment > 0 {
                    refined.push((seg_start, layer_idx));
                    seg_start = layer_idx;
                    full_attn_in_segment = 0;
                }
                full_attn_in_segment += 1;
            }
        }
        if seg_start < end {
            refined.push((seg_start, end));
        }
    }
    refined
}

/// Compute segment boundaries for gradient checkpointing.
///
/// Returns a list of `(start_layer, end_layer)` pairs that partition
/// `[0..num_layers)` into `num_segments` roughly-equal segments.
pub(crate) fn compute_segment_boundaries(
    num_layers: usize,
    num_segments: usize,
) -> Vec<(usize, usize)> {
    let seg_size = num_layers / num_segments;
    let remainder = num_layers % num_segments;
    let mut boundaries = Vec::with_capacity(num_segments);
    let mut start = 0;
    for i in 0..num_segments {
        let extra = if i < remainder { 1 } else { 0 };
        let end = start + seg_size + extra;
        boundaries.push((start, end));
        start = end;
    }
    boundaries
}

/// Returns true when every transformer layer in `weights` uses linear (GDN)
/// attention — i.e., the model has **no** full-attention layers anywhere.
///
/// The training-time time-axis tile path
/// ([`tiled_segment_recompute_and_backward`]) thread `LinearAttentionState`
/// across tiles to keep GDN forward bit-exact, but full-attention layers have
/// no analogous KV-cache thread at training time (training does not allocate
/// a paged KV cache). Within a tile a full-attention layer would attend only
/// inside the tile and produce different logits, breaking both per-tile loss
/// and any LoRA gradient that flows through it.
///
/// Per-segment iteration also runs **later** segments detached on the tile's
/// output, so even a segment that is itself GDN-only would dispatch into
/// later full-attention layers under tiling — which would also break parity.
/// The cleanest correctness invariant is therefore "no full-attention layers
/// anywhere in the model".
#[allow(dead_code)]
fn model_is_gdn_only(weights: &GpuWeights) -> bool {
    weights
        .layers
        .iter()
        .all(|l| matches!(l.attention, GpuAttentionWeights::Linear(_)))
}

/// Build a [`LoraWeights`] view whose `a` / `b` projections are **detached**
/// from the LoRA Vars' autograd graph.
///
/// Used by [`layer_pair_tiled_segment_recompute_and_backward`] for forwards
/// whose backward should NOT produce LoRA gradients — specifically, the
/// tail forward (whose only useful output is the gradient at the segment-
/// output Var) and the block-boundary forward in Step 2 (which only
/// computes activation VALUES). Without this, those backward passes would
/// produce LoRA gradients that would then be discarded — wasted compute,
/// and a correctness hazard if the discard is forgotten.
pub(crate) fn lora_weights_detached(params: &TrainableLoraParams) -> LoraWeights {
    let layers: Vec<LoraLayerWeights> = params
        .layers
        .iter()
        .map(|lp| {
            // (#1082) kt `Tensor::detach()` — the detached forward LoRA used by
            // the checkpointed Step-1 boundary forward (no grad recording).
            let make_proj =
                |pair: &Option<(Parameter, Parameter)>| -> Option<LoraProjectionWeights> {
                    pair.as_ref().map(|(a, b)| LoraProjectionWeights {
                        a: a.forward_storage().primary_tensor().detach(),
                        b: b.forward_storage().primary_tensor().detach(),
                    })
                };
            LoraLayerWeights {
                q_proj: make_proj(&lp.q_proj),
                k_proj: make_proj(&lp.k_proj),
                v_proj: make_proj(&lp.v_proj),
                o_proj: make_proj(&lp.o_proj),
                in_proj_qkv: make_proj(&lp.in_proj_qkv),
                in_proj_z: make_proj(&lp.in_proj_z),
                gdn_out_proj: make_proj(&lp.gdn_out_proj),
                gate_proj: make_proj(&lp.gate_proj),
                up_proj: make_proj(&lp.up_proj),
                down_proj: make_proj(&lp.down_proj),
                ..Default::default()
            }
        })
        .collect();

    LoraWeights {
        layers,
        mtp: None,
        rank: params.rank,
        alpha: params.alpha,
        scale: params.scale,
        source_identity: None,
    }
}

/// Attention kind of a single transformer layer for the layer-pair tiled path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttnKind {
    Gdn,
    FullAttn,
}

fn attn_kind_at(weights: &GpuWeights, layer_idx: usize) -> AttnKind {
    match &weights.layers[layer_idx].attention {
        GpuAttentionWeights::Linear(_) => AttnKind::Gdn,
        GpuAttentionWeights::Full(_) => AttnKind::FullAttn,
    }
}

/// Partition `[seg_start, seg_end)` into maximal contiguous runs of the same
/// attention kind. Each entry is `(kind, layer_range)` where `layer_range` is
/// a sub-range of the segment with all layers of the same kind.
///
/// Used by the layer-pair tiled path to process GDN sub-blocks (time-tile)
/// and full-attention sub-blocks (monolithic) sequentially within one
/// segment-recompute pass.
fn partition_segment_layers_by_attn_type(
    weights: &GpuWeights,
    seg_start: usize,
    seg_end: usize,
) -> Vec<(AttnKind, std::ops::Range<usize>)> {
    debug_assert!(seg_start < seg_end);
    let mut blocks: Vec<(AttnKind, std::ops::Range<usize>)> = Vec::new();
    let mut block_start = seg_start;
    let mut current_kind = attn_kind_at(weights, seg_start);
    for i in (seg_start + 1)..seg_end {
        let kind = attn_kind_at(weights, i);
        if kind != current_kind {
            blocks.push((current_kind, block_start..i));
            block_start = i;
            current_kind = kind;
        }
    }
    blocks.push((current_kind, block_start..seg_end));
    blocks
}

/// Determine whether a time-axis tile path applies for this training step.
///
/// Returns `Some(tile_size)` when:
/// 1. The injected streaming-prefill policy is enabled at this `seq_len`.
/// 2. The tile size is a positive multiple of `GDN_CHUNK_SIZE` (enforced by
///    typed startup validation) and strictly less than `seq_len`.
///
/// Caller routes between two implementations based on
/// [`model_is_gdn_only`]:
/// * GDN-only models use [`tiled_segment_recompute_and_backward`], which is
///   bit-exact against monolithic and skips gradient injection (cheaper).
/// * Hybrid GDN + full-attn models use
///   [`layer_pair_tiled_segment_recompute_and_backward`], which partitions
///   each segment into contiguous-attention-type blocks and processes them
///   with gradient injection so the tiled path can fire on production
///   models like Qwen3.5-4B (24 GDN + 8 full-attn).
#[allow(dead_code)]
fn tiled_training_tile_size(
    weights: &GpuWeights,
    seq_len: usize,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Option<usize> {
    let _ = weights; // signature retained for callers; gating moved to the dispatcher.
    if !streaming_prefill.enabled_for(seq_len) {
        return None;
    }
    let tile = streaming_prefill.base_tile_tokens();
    if tile == 0 || tile % GDN_CHUNK_SIZE != 0 || tile >= seq_len {
        return None;
    }
    Some(tile)
}

// (#1082) Deleted five orphaned residues of the removed exact_gdn tiled-reverse
// machinery — all had zero callers after the candle-drop:
//   * `profile_exact_gdn_reverse_tiles`
//   * `exact_gdn_split_recurrent_backward_enabled`
//   * `finish_exact_gdn_reverse_tile_stage` (its only call was to the already
//     deleted `synchronize_checkpoint_boundary`)
//   * `exact_gdn_reverse_tile_size`
//   * `exact_gdn_backward_tile_tokens_for`

// `InjectTensorGradient` (struct + impl candle_core::CustomOp1) was
// deleted as part of the #1082 CP-4 step 2-3 caller flip. All 6 call
// sites in `full_attention_single_layer_tiled_mlp_reverse` now use
// `kiln_kt_bridge::inject_grad_shim::inject_gradient_via_shim` which
// produces a bit-equivalent candle Tensor (the shim's `bwd` returns
// the precomputed `upstream`, byte-for-byte matching the previous
// in-trainer impl). With this deletion, `kiln-train::trainer` has
// zero production `candle_core::CustomOp1` impls and the crate's
// `candle-core` dep can move to `[dev-dependencies]`. See commits
// e2f8723c (substrate revision), 07afd64a (IO mapping removal),
// a6531830 (shim hoist), and the InjectTensorGradient flip
// commit itself. (#1082)

/// (#1082) Whether the kt tape grad-delivery path supports this base model's
/// dtype on this device. The decisive dtype is the **activation** dtype, which
/// follows the BASE model weights (`embed_tokens` dtype) — NOT the LoRA Vars,
/// which now FOLLOW the base dtype (see `initialize_seeded`).
///
/// BF16 is supported by the kt tape adapters. F32 is supported only when the
/// backend-owned precision policy declares F32 activations for mixed base
/// weights. Other dtypes fail before training work begins.
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
fn base_dtype_supports_tape_for_policy(
    weights: &GpuWeights,
    policy: TrainingPrecisionPolicy,
) -> bool {
    // (#1082) `embed_tokens.dtype()` is now kt `DType`.
    match weights.embed_tokens.dtype() {
        kiln_tensor::DType::BF16 => true,
        kiln_tensor::DType::F32 => policy.uses_f32_activations_for_mixed_base_weights(),
        _ => false,
    }
}

fn ensure_sft_loss_route_supports_checkpointing(
    route: SftFlceLossRoute,
    checkpointed: bool,
) -> Result<()> {
    anyhow::ensure!(
        !checkpointed || route != SftFlceLossRoute::FullLogits,
        "checkpointed SFT does not support loss route `{}`: its loss-value path \
         requires an active kt tape, while checkpoint tails run outside segment \
         tapes; disable gradient checkpointing or use a backend with a \
         checkpoint-compatible SFT loss route",
        route.as_str()
    );
    Ok(())
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
fn ensure_tape_forward_backward_supported(
    workload: &str,
    weights: &GpuWeights,
    backend: &dyn BackendRuntime,
) -> Result<()> {
    let route = TrainingLossBackend::runtime_tape_forward_backward_route(backend);
    anyhow::ensure!(
        matches!(route, TrainingTapeRoute::KtTapeAuthoritative),
        "{workload}: kt tape-authoritative training is required, but the backend \
         advertises tape route `{}`",
        route.as_str()
    );
    let precision_policy = training_precision_policy_for_backend(backend);
    anyhow::ensure!(
        base_dtype_supports_tape_for_policy(weights, precision_policy),
        "{workload}: base activation dtype {:?} is incompatible with backend \
         training precision policy `{}` for kt tape-authoritative training",
        weights.embed_tokens.dtype(),
        precision_policy.name
    );
    Ok(())
}

#[cfg(not(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
)))]
fn ensure_tape_forward_backward_supported(
    workload: &str,
    _weights: &GpuWeights,
    _backend: &dyn BackendRuntime,
) -> Result<()> {
    anyhow::bail!(
        "{workload}: kt tape-authoritative training requires a CUDA, ROCm, Metal, \
         or Vulkan backend build"
    )
}

/// (#1082 Increment-0 PR2) kt-native sibling of
/// [`standard_forward_backward_tape_authoritative`]: delivers the
/// tape-authoritative LoRA gradients into a kt-native
/// [`kiln_autograd::GradStore`] keyed by [`KtTensorId`], WITHOUT the candle
/// `loss.backward()` GradStore-container hack and WITHOUT the per-grad
/// `kt -> candle` copy.
///
/// The kt grads produced by `with_tape_authoritative_scope` are the
/// authoritative output; they are inserted as-is (the candle `loss` is used
/// only for the `loss_val` scalar readback). The optimizer bridges each grad
/// to candle at its per-Var boundary (`optimizer_step_from_kt_grad_store`,
/// Inc-0 PR3) until the optimizer itself goes kt-native via `kiln-optim`.
///
/// This is the perf-correct grad-delivery path AND the structural gate for the
/// forward.rs type-flip: it removes the dependency on a candle `loss` existing
/// to call `.backward()` on (post-flip `model_forward` returns kt, so there is
/// no candle loss to instantiate a candle `GradStore` from). The grad keys
/// match the PR1 `KtTensorId`-keyed `OptimizerState.moments`
/// (`KtTensorId::from_raw(var.id().as_raw() as u64)` ==
/// `cd_tensor_id_to_kt(var.id())`), so PR3's consumer looks moments up by the
/// same key. (#1082 Inc-0 PR4) NOW WIRED IN: `standard_forward_backward`'s
/// tape-authoritative CUDA branch calls this and returns `GradSource::Kt`, so
/// the SFT loop + the CP-4 gates exercise this kt-native path.
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
fn standard_forward_backward_tape_authoritative_kt(
    backend: &dyn BackendRuntime,
    sft_loss_route: SftFlceLossRoute,
    input_ids: &[u32],
    weights: &GpuWeights,
    model_config: &ModelConfig,
    params: &TrainableLoraParams,
    label_mask: &[bool],
    device: &Device,
    detect_anomaly: bool,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<(f64, kiln_autograd::GradStore)> {
    let lora_weights = params.as_lora_weights();
    let mut linear_state = LinearAttentionState::new(model_config, device)?;
    ensure_sft_loss_route_supports_checkpointing(sft_loss_route, false)?;

    let (loss_val, _loss_kt, grads_by_candle_raw) =
        kiln_kt_bridge::tape_bridge::with_tape_authoritative_scope_kt(
            kiln_autograd::TapeOptions { detect_anomaly },
            || {
            let loss_kt = match sft_loss_route {
                SftFlceLossRoute::KtTapeFlce => {
                    let normed = model_forward_no_head_with_policy(
                        backend,
                        input_ids,
                        weights,
                        model_config,
                        Some(&mut linear_state),
                        Some(&lora_weights),
                        streaming_prefill,
                    )
                    .context("tape-authoritative(kt) no-head forward")
                    .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))?;
                    // Default SFT records the kt FLCE loss root against final normed hidden
                    // instead of materializing `[1, T, V]` logits. The frozen tied head
                    // receives no gradient; the FLCE tape node returns `dhidden`, keeping
                    // the LoRA path connected through `model_forward_no_head`.
                    let loss = kiln_autograd::with_active_tape(|tape| {
                        kiln_flce_kernel::fused_linear_cross_entropy_phase_b_unit_grad_via_kt_tape(
                            &normed,
                            &weights.embed_tokens_t,
                            input_ids,
                            label_mask,
                            DEFAULT_CHUNK_SIZE,
                            tape,
                        )
                    })
                    .ok_or_else(|| {
                        kiln_kt_bridge::BridgeError::new(
                            "tape-authoritative(kt) SFT FLCE: no active kt tape".to_string(),
                        )
                    })?
                    .map_err(|e| {
                        kiln_kt_bridge::BridgeError::new(format!(
                            "tape-authoritative(kt) SFT FLCE kt-tape: {e}"
                        ))
                    })?;
                    loss
                }
                SftFlceLossRoute::VulkanActiveRows => {
                    #[cfg(feature = "vulkan")]
                    {
                        let normed = model_forward_no_head_with_policy(
                            backend,
                            input_ids,
                            weights,
                            model_config,
                            Some(&mut linear_state),
                            Some(&lora_weights),
                            streaming_prefill,
                        )
                        .context("tape-authoritative(kt) no-head Vulkan forward")
                        .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))?;
                        // Vulkan has its own fused FLCE shaders over active rows
                        // and canonical tied weight [V, H], so route the SFT root
                        // there instead of materializing [1, T, V] logits.
                        crate::sft_tape_shim::try_tape_sft_flce_vulkan_kt(
                            &normed,
                            &weights.embed_tokens,
                            input_ids,
                            label_mask,
                        )
                        .context("tape-authoritative(kt) Vulkan SFT FLCE")
                        .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))?
                        .ok_or_else(|| {
                            kiln_kt_bridge::BridgeError::new(
                                "tape-authoritative(kt) Vulkan SFT FLCE returned None".to_string(),
                            )
                        })?
                    }
                    #[cfg(not(feature = "vulkan"))]
                    {
                        return Err(kiln_kt_bridge::BridgeError::new(
                            "backend requested Vulkan SFT FLCE without the vulkan feature"
                                .to_string(),
                        ));
                    }
                }
                SftFlceLossRoute::FullLogits => {
                    let logits = model_forward_kt_with_policy(
                        backend,
                        input_ids,
                        weights,
                        model_config,
                        None,
                        Some(&mut linear_state),
                        Some(&lora_weights),
                        streaming_prefill,
                    )
                    .context("tape-authoritative(kt) fallback full forward")
                    .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))?;
                    kiln_model::tape_forward::try_tape_cross_entropy_from_logits_kt(
                        &logits,
                        input_ids,
                        label_mask,
                    )
                    .context("tape-authoritative(kt) cross_entropy_from_logits_kt")
                    .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))?
                    .ok_or_else(|| {
                        kiln_kt_bridge::BridgeError::new(
                            "tape-authoritative(kt) SFT: cross_entropy_from_logits_kt returned None \
                             (kt CE envelope declined — expected [1, T, V] CUDA logits)"
                                .to_string(),
                        )
                    })?
                }
            };
            let loss_val = loss_kt
                .to_scalar::<f32>()
                .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("loss_kt.to_scalar: {e}")))?
                as f64;
            Ok((loss_val, loss_kt))
            },
        )
        .map_err(|e| anyhow::anyhow!("tape-authoritative(kt) backward: {e}"))?;

    // (#1082) Build a kt-native GradStore from the tape grads, keyed by each
    // LoRA `Parameter::tensor_id()`. The tape's `out` map mixes candle-keyed
    // deposits (frozen base/activation/norm tensors via `register_input_mapping`)
    // and kt-param deposits (LoRA leaves via `register_input_mapping_kt`, which
    // namespace-tags the key with `KT_PARAM_DEPOSIT_TAG`). Decode each key: only
    // tagged entries are genuine LoRA-param grads — `decode_kt_param_deposit`
    // strips the tag and yields the param's kt id, so a candle id that happens to
    // equal a param id (independent counters, both start at 1) is rejected. This
    // is the read side of the #1082 collision fix (a frozen RMSNorm `[hidden]`
    // grad was aliasing the `in_proj_z` LoRA-B `[out, rank]` slot → AdamW shape
    // mismatch `[32] != [32, 4]`).
    let mut grads = kiln_autograd::GradStore::new();
    for (key_raw, kt_grad) in grads_by_candle_raw {
        let Some(param_raw) = kiln_kt_bridge::tape_bridge::decode_kt_param_deposit(key_raw as u64)
        else {
            continue;
        };
        grads.insert(KtTensorId::from_raw(param_raw), kt_grad);
    }
    Ok((loss_val, grads))
}

/// Gradient-checkpointed SFT forward/backward via the kt autograd tape (#1082).
///
/// The kt-native replacement for the legacy candle gradient-checkpointing
/// reverse (`checkpointed_forward_backward`), which the forward.rs candle→kt
/// flip grad-severed: candle `.backward()` can no longer trace through the now
/// kt-internal `model_forward_segment` (the kt↔candle copy bridge breaks the
/// autograd lineage). This routes each checkpoint segment's backward through
/// the kt tape instead — the same validated grad producer as the monolithic
/// `standard_forward_backward_tape_authoritative_kt`, just applied per segment
/// so only one segment's activations are resident at a time (the whole point of
/// gradient checkpointing).
///
/// Flow (mirrors `checkpointed_forward_backward` Steps 1-2, replaces Step 3):
///  1. One detached forward → kt boundary activations (one per segment start +
///     the final pre-final-norm hidden). No tape recording (memory-bounded).
///  2. Loss at the final boundary + the analytic tail seed `d(loss)/d(hidden)`
///     through final-RMSNorm + tied LM-head + masked next-token cross-entropy
///     (a candle island; bridged to kt to seed the tape).
///  3. Walk segments in reverse: re-run each segment's forward UNDER A FRESH
///     thread-local tape (recording only that segment), seed the tape backward
///     at the segment output with the upstream grad, read out (a) the LoRA `Var`
///     grads for that segment and (b) the segment-INPUT grad to chain into the
///     previous segment. The fresh-tape-per-segment design bounds memory.
///
/// Returns the LoRA grads as a kt-native `kiln_autograd::GradStore` (keyed by
/// `KtTensorId`), consumed directly by `optimizer_step_from_kt_grad_store` — no
/// candle `loss.backward()` and no kt→candle grad copy.
///
/// The dispatch below uses backend training capabilities plus precision policy
/// for tape eligibility, then keeps local loss-shape exclusions such as ECHO.
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
#[allow(clippy::too_many_arguments)]
fn checkpointed_forward_backward_tape_authoritative_kt(
    backend: &dyn BackendRuntime,
    sft_loss_route: SftFlceLossRoute,
    input_ids: &[u32],
    weights: &GpuWeights,
    model_config: &ModelConfig,
    params: &TrainableLoraParams,
    label_mask: &[bool],
    segments: &[(usize, usize)],
    device: &Device,
    detect_anomaly: bool,
    checkpoint_boundary_policy: crate::CheckpointBoundaryPolicy,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<(f64, kiln_autograd::GradStore)> {
    let num_segments = segments.len();
    anyhow::ensure!(
        num_segments > 0,
        "checkpointed (kt-tape) SFT requires at least one segment"
    );
    anyhow::ensure!(
        input_ids.len() == label_mask.len(),
        "input_ids/label_mask length mismatch: {} vs {}",
        input_ids.len(),
        label_mask.len()
    );
    anyhow::ensure!(
        has_supervised_shifted_labels(label_mask),
        "checkpointed (kt-tape) SFT called with no supervised shifted-label positions"
    );
    ensure_tape_forward_backward_supported("checkpointed SFT", weights, backend)?;
    ensure_sft_loss_route_supports_checkpointing(sft_loss_route, true)?;

    let positions: Vec<u32> = (0..input_ids.len()).map(|p| p as u32).collect();
    let lora_detached = lora_weights_detached(params);
    let lora_weights = params.as_lora_weights();

    // Step 1: detached forward → kt boundary activations (one per segment start
    // + the final pre-final-norm hidden). NOT under a tape scope, so nothing is
    // recorded — only the boundary tensors are kept (the checkpointing memory
    // profile). A single threaded `LinearAttentionState` is fine: each GDN
    // layer's recurrence is internal to its own full-sequence pass.
    let (embed_hidden, _) = model_forward_embed(input_ids, weights)?;
    let spool_boundaries = if checkpoint_boundary_policy.recompute_for(input_ids.len()) {
        let resident_device_storage =
            ResidencyBackend::runtime_supports_resident_activation(backend);
        let anchor_stride = checkpoint_boundary_policy.anchor_stride_for_shape(
            input_ids.len(),
            num_segments,
            model_config.hidden_size,
            dtype_size_bytes(embed_hidden.dtype()),
        );
        Some(StoredCheckpointBoundaries::new(
            num_segments,
            resident_device_storage,
            anchor_stride,
        ))
    } else {
        None
    };
    let mut boundaries: Vec<Option<kiln_tensor::Tensor>> = Vec::with_capacity(num_segments + 1);
    let mut boundary_dtypes: Vec<DType> = Vec::with_capacity(num_segments + 1);
    let mut current = embed_hidden.detach();
    synchronize_training_tensor_ready("embed_hidden", &current)?;
    boundary_dtypes.push(current.dtype());
    if let Some(spool) = spool_boundaries.as_ref() {
        spool.save(0, &current)?;
        boundaries.push(None);
    } else {
        boundaries.push(Some(current.clone()));
    }
    {
        let mut linear_state = LinearAttentionState::new(model_config, device)?;
        for (seg_idx, &(start, end)) in segments.iter().enumerate() {
            current = model_forward_segment_with_policy(
                backend,
                current,
                weights,
                model_config,
                &positions,
                start,
                end,
                Some(&mut linear_state),
                Some(&lora_detached),
                streaming_prefill,
            )?
            .detach();
            let boundary_label = format!("boundary_segment[{seg_idx}] layers {start}..{end}");
            synchronize_training_tensor_ready(&boundary_label, &current)?;
            boundary_dtypes.push(current.dtype());
            if let Some(spool) = spool_boundaries.as_ref() {
                spool.save(seg_idx + 1, &current)?;
                boundaries.push(None);
            } else {
                boundaries.push(Some(current.clone()));
            }
        }
    }
    let final_hidden_kt = current.clone();

    // Step 2: real loss at the final boundary + the exact FLCE/RMSNorm tail
    // seed. When the CUDA FLCE loss-value path computes final-norm output, the
    // analytic tail reuses that `normed` hidden for the FLCE backward seed
    // instead of recomputing the same [1, T, H] RMSNorm. The tail then applies
    // the shared final-RMSNorm backward to return d(loss)/d(pre-final-norm
    // hidden) as kt [1, T, H] (BF16 on the fused GPU path, F32 on the composite
    // fallback) — exactly the upstream grad to seed the LAST segment's backward
    // (its output IS that hidden). The loss value here is ONLY consumed as a
    // scalar `loss_val`; the gradient comes entirely from this tail seed,
    // outside the per-segment tape scopes.
    let mut normed_for_tail = None;
    let mut flce_active_metadata_for_tail = None;
    let tail_grad_override: Option<Tensor>;
    let loss_val = match sft_loss_route {
        SftFlceLossRoute::KtTapeFlce => {
            tail_grad_override = None;
            // (#1082 H-FLCE / candle-drop) FLCE loss-VALUE via the kt-native forward
            // `fused_linear_cross_entropy_phase_b_kt` — taking the kt `normed` hidden
            // and the kt `embed_tokens_t` head DIRECTLY (no candle `cd_out` copy, no
            // ~780MB/step `embed_tokens_t` kt->candle copy, no candle device bridge).
            // Only the resulting scalar crosses back to host. The same `normed`
            // tensor is retained for the FLCE/RMSNorm tail seed below.
            // The candle FLCE provider opt-in (`KILN_CUDA_FLCE`) was removed in the
            // candle drop — this is now the sole FLCE path.
            synchronize_training_tensor_ready("tail_pre_final_norm_hidden", &final_hidden_kt)?;
            let normed = model_forward_final_norm(&final_hidden_kt, weights, model_config)?;
            synchronize_training_tensor_ready("tail_final_norm", &normed)?;
            let (loss_kt, active_metadata) =
                kiln_flce_kernel::kt_api::fused_linear_cross_entropy_phase_b_with_metadata_kt(
                    &normed,
                    &weights.embed_tokens_t,
                    input_ids,
                    label_mask,
                    DEFAULT_CHUNK_SIZE,
                )
                .map_err(|e| {
                    anyhow::anyhow!(
                        "ckpt-kt kt-native fused linear cross-entropy (final boundary): {e}"
                    )
                })?;
            synchronize_training_tensor_ready("tail_flce_loss_scalar", &loss_kt)?;
            let loss_val = loss_kt.to_scalar::<f32>()? as f64;
            flce_active_metadata_for_tail = active_metadata;
            normed_for_tail = Some(normed);
            loss_val
        }
        SftFlceLossRoute::VulkanActiveRows => {
            #[cfg(feature = "vulkan")]
            {
                synchronize_training_tensor_ready("tail_pre_final_norm_hidden", &final_hidden_kt)?;
                let normed = model_forward_final_norm(&final_hidden_kt, weights, model_config)?;
                synchronize_training_tensor_ready("tail_final_norm", &normed)?;
                let (loss_kt, grad_normed) =
                    crate::sft_tape_shim::vulkan_sft_flce_loss_and_grad_kt(
                        &normed,
                        &weights.embed_tokens,
                        input_ids,
                        label_mask,
                    )
                    .map_err(|e| anyhow::anyhow!("ckpt-kt Vulkan fused SFT FLCE tail: {e}"))?;
                synchronize_training_tensor_ready("tail_vulkan_flce_loss_scalar", &loss_kt)?;
                let loss_val = loss_kt.to_scalar::<f32>()? as f64;
                tail_grad_override = Some(
                    rms_norm_backward_pre_final_norm(
                        final_rmsnorm_backward_route_for_backend(backend),
                        &final_hidden_kt,
                        &weights.final_norm,
                        &grad_normed,
                        model_config.rms_norm_eps,
                    )
                    .context("ckpt-kt Vulkan final RMSNorm backward")?,
                );
                loss_val
            }
            #[cfg(not(feature = "vulkan"))]
            {
                anyhow::bail!("backend requested Vulkan SFT FLCE without the vulkan feature");
            }
        }
        SftFlceLossRoute::FullLogits => {
            anyhow::bail!(
                "checkpointed SFT reached unsupported loss route `{}` after its entry guard",
                sft_loss_route.as_str()
            )
        }
    };
    anyhow::ensure!(
        loss_val.is_finite(),
        "SFT loss became non-finite before backward: loss={loss_val} route={} seq_len={} segments={}",
        sft_loss_route.as_str(),
        input_ids.len(),
        num_segments
    );
    let tail_grad = if let Some(tail_grad) = tail_grad_override {
        Ok(tail_grad)
    } else {
        match normed_for_tail.as_ref() {
            Some(normed) => analytic_sft_tail_grad_from_normed_pre_final_norm_with_flce_metadata(
                final_rmsnorm_backward_route_for_backend(backend),
                &final_hidden_kt,
                normed,
                &weights.final_norm,
                &weights.embed_tokens_t,
                input_ids,
                label_mask,
                model_config.rms_norm_eps,
                DEFAULT_CHUNK_SIZE,
                flce_active_metadata_for_tail.as_ref(),
            ),
            None => analytic_sft_tail_grad_pre_final_norm(
                final_rmsnorm_backward_route_for_backend(backend),
                &final_hidden_kt,
                &weights.final_norm,
                &weights.embed_tokens_t,
                input_ids,
                label_mask,
                model_config.rms_norm_eps,
                DEFAULT_CHUNK_SIZE,
            ),
        }
    };
    let mut upstream_grad = tail_grad
        .context("ckpt-kt FLCE/RMSNorm SFT tail gradient")?
        .detach();
    drop(final_hidden_kt);
    drop(current);
    // Step 3: reverse pass over segments via the kt tape. Each segment is
    // re-run under its OWN fresh tape (memory bounded to one segment), seeded at
    // its output with the upstream grad; we read the LoRA Var grads and the
    // segment-input grad (to chain) out of the walk.
    // (#1082) keyed by `Parameter::tensor_id()`.
    let mut grads = kiln_autograd::GradStore::new();
    for seg_idx in (0..num_segments).rev() {
        let (start, end) = segments[seg_idx];
        let seg_input = if let Some(spool) = spool_boundaries.as_ref() {
            load_or_recompute_checkpoint_boundary(
                spool,
                seg_idx,
                backend,
                weights,
                model_config,
                &positions,
                segments,
                &lora_detached,
                device,
                streaming_prefill,
            )?
        } else {
            boundaries[seg_idx]
                .as_ref()
                .context("ckpt-kt: missing in-memory checkpoint boundary")?
                .clone()
        };
        let seg_input_id = seg_input.id();
        // Match the seed dtype to the segment output (the model hidden dtype);
        // the analytic tail is F32 and chained grads may differ.
        let seg_output_dtype = boundary_dtypes[seg_idx + 1];
        let seed = upstream_grad
            .to_dtype(seg_output_dtype)
            .map_err(|e| anyhow::anyhow!("ckpt-kt: seed dtype cast (segment {seg_idx}): {e}"))?;
        let positions_ref = &positions;
        let lora_ref = &lora_weights;
        let (kt_grads, candle_grads) =
            kiln_kt_bridge::tape_bridge::with_tape_segment_backward_scope(
                kiln_autograd::TapeOptions { detect_anomaly },
                seed,
                || {
                    // Fresh recurrence state per segment (GDN recurrence is internal
                    // to each layer's full-sequence pass — see Step 1 note).
                    let mut seg_ls = LinearAttentionState::new(model_config, device)
                        .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))?;
                    model_forward_segment_with_policy(
                        backend,
                        seg_input,
                        weights,
                        model_config,
                        positions_ref,
                        start,
                        end,
                        Some(&mut seg_ls),
                        Some(lora_ref),
                        streaming_prefill,
                    )
                    .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))
                },
            )
            .map_err(|e| anyhow::anyhow!("ckpt-kt: segment {seg_idx} tape backward: {e}"))?;

        // Decode every tagged parameter deposit into a segment-local store.
        // The exact segment contract below rejects missing leaves, deposits for
        // another layer range, and any unknown tagged parameter before merge.
        let mut segment_grads = kiln_autograd::GradStore::new();
        for (candle_raw, g) in candle_grads {
            let Some(param_raw) =
                kiln_kt_bridge::tape_bridge::decode_kt_param_deposit(candle_raw as u64)
            else {
                continue;
            };
            segment_grads.insert(KtTensorId::from_raw(param_raw), g);
        }
        let grad_context =
            format!("checkpointed SFT segment {seg_idx} layers {start}..{end} gradient contract");
        merge_checkpoint_lora_grad_segment(
            params,
            &mut grads,
            segment_grads,
            start,
            end,
            &grad_context,
        )?;

        // Chain the upstream grad into the previous (earlier) segment.
        if seg_idx > 0 {
            upstream_grad = kt_grads.get(seg_input_id).cloned().ok_or_else(|| {
                anyhow::anyhow!(
                    "ckpt-kt: tape backward produced no input gradient for segment {seg_idx}"
                )
            })?;
        }
    }

    Ok((loss_val, grads))
}

pub fn standard_forward_backward(
    backend: &dyn BackendRuntime,
    input_ids: &[u32],
    weights: &GpuWeights,
    model_config: &ModelConfig,
    params: &TrainableLoraParams,
    label_mask: &[bool],
    device: &Device,
) -> Result<(f64, GradSource)> {
    standard_forward_backward_with_policy(
        backend,
        input_ids,
        weights,
        model_config,
        params,
        label_mask,
        device,
        StreamingPrefillExecutionPolicy::for_device(*device),
    )
}

/// Explicit-policy variant of [`standard_forward_backward`].
#[allow(clippy::too_many_arguments)]
pub fn standard_forward_backward_with_policy(
    backend: &dyn BackendRuntime,
    input_ids: &[u32],
    weights: &GpuWeights,
    model_config: &ModelConfig,
    params: &TrainableLoraParams,
    label_mask: &[bool],
    device: &Device,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<(f64, GradSource)> {
    standard_forward_backward_with_policy_and_loss_route(
        backend,
        TrainingLossBackend::runtime_sft_flce_loss_route(backend),
        input_ids,
        weights,
        model_config,
        params,
        label_mask,
        device,
        false,
        streaming_prefill,
    )
}

#[allow(clippy::too_many_arguments)]
fn standard_forward_backward_with_policy_and_loss_route(
    backend: &dyn BackendRuntime,
    sft_loss_route: SftFlceLossRoute,
    input_ids: &[u32],
    weights: &GpuWeights,
    model_config: &ModelConfig,
    params: &TrainableLoraParams,
    label_mask: &[bool],
    device: &Device,
    detect_anomaly: bool,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<(f64, GradSource)> {
    // (#1082 candle-drop) The SFT forward/backward is now UNCONDITIONALLY
    // kt tape-authoritative when the backend capability and precision policy
    // allow it. The candle producers
    // (`standard_forward_backward_tape_authoritative` F32-hack,
    // `standard_forward_backward_via_tape_bridge`, the inline candle
    // `loss.backward()` path) are all DELETED. Unsupported backend/dtype
    // combinations fail before the kt step is attempted.
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    {
        ensure_tape_forward_backward_supported("standard_forward_backward", weights, backend)?;
        let (loss_val, kt_grads) = standard_forward_backward_tape_authoritative_kt(
            backend,
            sft_loss_route,
            input_ids,
            weights,
            model_config,
            params,
            label_mask,
            device,
            detect_anomaly,
            streaming_prefill,
        )?;
        Ok((loss_val, GradSource::Kt(kt_grads)))
    }
    #[cfg(not(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    )))]
    {
        let _ = (
            backend,
            sft_loss_route,
            input_ids,
            weights,
            model_config,
            params,
            label_mask,
            device,
            detect_anomaly,
            streaming_prefill,
        );
        anyhow::bail!(
            "standard_forward_backward: SFT training requires a GPU backend feature \
             post candle-drop because the candle `loss.backward()` path was removed."
        )
    }
}

/// kt-native sibling of [`grpo_step_forward_backward_tape_authoritative`]
/// (#1082 Inc-0 PR5). Identical GRPO policy-gradient tape-authoritative
/// forward/backward, but delivers the LoRA grads in a kt-native
/// [`kiln_autograd::GradStore`] (keyed by [`KtTensorId`]) DIRECTLY from the
/// tape — NO candle `loss.backward()` GradStore-container hack and NO per-grad
/// `kt -> candle` copy. This is the exact GRPO analogue of the SFT kt producer
/// [`standard_forward_backward_tape_authoritative_kt`].
///
/// As with SFT, this is the perf-correct grad-delivery path AND the structural
/// gate for the forward.rs type-flip: it removes the dependency on a candle
/// `loss` existing to call `.backward()` on (post-flip `model_forward` returns
/// kt, so there is no candle loss to seed a candle `GradStore` from). The grad
/// keys match the PR1 `KtTensorId`-keyed `OptimizerState.moments`
/// (`KtTensorId::from_raw(var.id().as_raw() as u64)` ==
/// `cd_tensor_id_to_kt(var.id())`), so the kt consumers
/// ([`optimizer_step_from_kt_grad_store`],
/// [`observe_lora_grad_norms_from_kt_grad_store`], and the kt accumulate path)
/// look grads/moments up by the same key.
///
/// The backend-selected loss route must record inside the active tape scope;
/// there is no environment opt-out or alternate autograd producer. ECHO, when
/// configured, is composed into that same tape-rooted loss.
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
#[allow(clippy::too_many_arguments)]
fn grpo_step_forward_backward_tape_authoritative_kt(
    backend: &dyn BackendRuntime,
    input_ids: &[u32],
    weights: &GpuWeights,
    model_config: &ModelConfig,
    params: &TrainableLoraParams,
    action_mask: &[bool],
    behavior_log_probs: &Tensor,
    kl_reference_log_probs: &Tensor,
    loss_params: GrpoLossParams,
    device: &Device,
    comp_idx: usize,
    num_active: usize,
    comp_env_count: usize,
    streaming_tile_tokens: usize,
    checkpoint_segments: usize,
    mut timings: Option<&mut GrpoBenchmarkTimings>,
    echo_env: Option<&crate::grpo_tape_shim::EchoEnvSpec>,
    no_pg: bool,
    detect_anomaly: bool,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<(
    f64,
    Option<f64>,
    kiln_autograd::GradStore,
    kiln_tensor::Tensor,
)> {
    let lora_weights = params.as_lora_weights();
    let mut linear_state = LinearAttentionState::new(model_config, device)?;
    let step_started = Instant::now();

    let ((loss_val, env_ce, policy_log_probs), _loss_kt, grads_by_candle_raw) =
        kiln_kt_bridge::tape_bridge::with_tape_authoritative_scope_kt(
            kiln_autograd::TapeOptions { detect_anomaly },
            || {
                // Single policy forward through final RMSNorm, without materializing
                // `[1, T, V]` logits. The GRPO loss root chunks the frozen tied head
                // internally and records `dL/d(normed_hidden)` directly.
                let policy_hidden = model_forward_no_head_with_policy(
                    backend,
                    input_ids,
                    weights,
                    model_config,
                    Some(&mut linear_state),
                    Some(&lora_weights),
                    streaming_prefill,
                )
                .context("GRPO tape-authoritative(kt) no-head policy forward")
                .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))?;

                // The Vulkan fused active-rows root carries no env rows — an
                // ECHO-active step takes the KtComposite root below instead.
                #[cfg(feature = "vulkan")]
                let mut loss_opt = match TrainingLossBackend::runtime_grpo_loss_route(backend) {
                    GrpoLossRoute::VulkanActiveRows if echo_env.is_none() => {
                        crate::grpo_tape_shim::try_tape_grpo_pg_loss_from_normed_hidden_vulkan_kt(
                            &policy_hidden,
                            &weights.embed_tokens,
                            input_ids,
                            action_mask,
                            behavior_log_probs,
                            kl_reference_log_probs,
                            loss_params,
                        )
                        .context("GRPO tape-authoritative(kt) Vulkan fused scalar loss")
                        .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))?
                        .map(|(loss, policy_log_probs)| (loss, None, policy_log_probs))
                    }
                    _ => None,
                };
                #[cfg(not(feature = "vulkan"))]
                let mut loss_opt = None;
                if loss_opt.is_none() {
                    loss_opt = crate::grpo_tape_shim::try_tape_grpo_pg_loss_from_normed_hidden_kt(
                        &policy_hidden,
                        &weights.embed_tokens_t,
                        input_ids,
                        action_mask,
                        behavior_log_probs,
                        kl_reference_log_probs,
                        loss_params,
                        grpo_kl_auxiliary_route_for_backend(backend),
                        device,
                        DEFAULT_CHUNK_SIZE,
                        echo_env,
                        no_pg,
                    )
                    .context("GRPO tape-authoritative(kt) scalar loss")
                    .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))?;
                }

                let (loss, env_ce, policy_log_probs) = match loss_opt {
                    Some(values) => values,
                    None => {
                        return Err(kiln_kt_bridge::BridgeError::new(
                            "GRPO tape-authoritative(kt): the selected loss route did not record a \
                         scalar root (an active tape scope is mandatory; the active set may be \
                         empty or the hidden/head tensors may be outside the route envelope)",
                        ));
                    }
                };
                let loss_val = loss.to_scalar::<f32>().map_err(|e| {
                    kiln_kt_bridge::BridgeError::new(format!("GRPO(kt) loss.to_scalar: {e}"))
                })? as f64;
                Ok(((loss_val, env_ce, policy_log_probs), loss))
            },
        )
        .map_err(|e| anyhow::anyhow!("GRPO tape-authoritative(kt) backward: {e}"))?;

    // Build a kt-native GradStore DIRECTLY from the tape grads. No
    // `loss.backward()` container hack (`GradStore::new()` on kiln_autograd is
    // public, unlike candle's) and no `kt -> candle` grad copy: the kt grads are
    // inserted as-is, keyed by each LoRA Var's id bridged into the kt id space
    // (matching the PR1 `KtTensorId`-keyed moments). Identical shape to the SFT
    // kt producer.
    // (#1082) keyed by `Parameter::tensor_id()` (== the LoRA primary kt
    // tensor id the tape adapter registered as the candle-input key).
    let mut grads = kiln_autograd::GradStore::new();
    for (key_raw, kt_grad) in grads_by_candle_raw {
        let Some(param_raw) = kiln_kt_bridge::tape_bridge::decode_kt_param_deposit(key_raw as u64)
        else {
            continue;
        };
        grads.insert(KtTensorId::from_raw(param_raw), kt_grad);
    }

    let step_elapsed = step_started.elapsed();
    if let Some(t) = timings.as_deref_mut() {
        // The tape walk owns the backward internally so we can't break the step
        // into policy_forward / backward here; bucket the full wall-clock against
        // the backward timer so the GRPO benchmark accounting still totals
        // correctly when this path is exercised.
        t.add_backward(step_elapsed);
    }
    tracing::info!(
        comp_idx,
        seq_len = input_ids.len(),
        action_tokens = num_active,
        env_tokens = comp_env_count,
        checkpoint_segments,
        streaming_prefill = streaming_prefill.enabled_for(input_ids.len()),
        streaming_tile_tokens,
        elapsed_ms = step_elapsed.as_millis() as u64,
        "GRPO step end (tape-authoritative kt)"
    );

    Ok((loss_val, env_ce, grads, policy_log_probs))
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
#[allow(clippy::too_many_arguments)]
fn checkpointed_grpo_forward_backward_tape_authoritative_kt(
    backend: &dyn BackendRuntime,
    input_ids: &[u32],
    weights: &GpuWeights,
    model_config: &ModelConfig,
    params: &TrainableLoraParams,
    action_mask: &[bool],
    behavior_log_probs: &Tensor,
    kl_reference_log_probs: &Tensor,
    loss_params: GrpoLossParams,
    segments: &[(usize, usize)],
    device: &Device,
    echo_env: Option<&crate::grpo_tape_shim::EchoEnvSpec>,
    no_pg: bool,
    detect_anomaly: bool,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<(
    f64,
    Option<f64>,
    kiln_autograd::GradStore,
    kiln_tensor::Tensor,
)> {
    let num_segments = segments.len();
    anyhow::ensure!(
        num_segments > 0,
        "checkpointed GRPO requires at least one segment"
    );
    anyhow::ensure!(
        input_ids.len() == action_mask.len(),
        "input_ids/action_mask length mismatch: {} vs {}",
        input_ids.len(),
        action_mask.len()
    );
    anyhow::ensure!(
        action_mask.get(1..).is_some_and(|m| m.iter().any(|&v| v)),
        "checkpointed GRPO called with no active shifted action positions"
    );

    let positions: Vec<u32> = (0..input_ids.len()).map(|p| p as u32).collect();
    let lora_detached = lora_weights_detached(params);
    let lora_weights = params.as_lora_weights();

    let (embed_hidden, _) = model_forward_embed(input_ids, weights)?;
    let mut boundaries: Vec<Tensor> =
        Vec::with_capacity(crate::retained_checkpoint_boundary_count(num_segments));
    let mut current = embed_hidden.detach();
    boundaries.push(current.clone());
    {
        let mut linear_state = LinearAttentionState::new(model_config, device)?;
        for &(start, end) in segments {
            current = model_forward_segment_with_policy(
                backend,
                current,
                weights,
                model_config,
                &positions,
                start,
                end,
                Some(&mut linear_state),
                Some(&lora_detached),
                streaming_prefill,
            )?
            .detach();
            boundaries.push(current.clone());
        }
    }
    let final_hidden = boundaries
        .last()
        .context("checkpointed GRPO: missing final checkpoint boundary")?
        .clone();

    let normed = model_forward_final_norm(&final_hidden, weights, model_config)
        .context("checkpointed GRPO final norm")?;
    // The Vulkan fused active-rows tail carries no env rows — ECHO-active
    // steps take the KtComposite tail instead.
    #[cfg(feature = "vulkan")]
    let fused_vulkan_tail = if echo_env.is_some() {
        None
    } else {
        match TrainingLossBackend::runtime_grpo_loss_route(backend) {
            GrpoLossRoute::VulkanActiveRows => {
                crate::grpo_tape_shim::vulkan_grpo_pg_loss_from_normed_hidden_loss_and_grad_kt(
                    &normed,
                    &weights.embed_tokens,
                    input_ids,
                    action_mask,
                    behavior_log_probs,
                    kl_reference_log_probs,
                    loss_params,
                )
                .context("checkpointed GRPO Vulkan fused tail loss/gradient")?
            }
            GrpoLossRoute::KtComposite => None,
        }
    };
    #[cfg(not(feature = "vulkan"))]
    let fused_vulkan_tail = None;
    let (loss_kt, grad_normed, env_ce, policy_log_probs) = match fused_vulkan_tail {
        Some((loss, grad, policy_log_probs)) => (loss, grad, None, policy_log_probs),
        None => crate::grpo_tape_shim::grpo_pg_loss_from_normed_hidden_loss_and_grad_kt(
            &normed,
            &weights.embed_tokens_t,
            input_ids,
            action_mask,
            behavior_log_probs,
            kl_reference_log_probs,
            loss_params,
            grpo_kl_auxiliary_route_for_backend(backend),
            1.0,
            device,
            DEFAULT_CHUNK_SIZE,
            echo_env,
            no_pg,
        )
        .context("checkpointed GRPO tail loss/gradient")?,
    };
    let loss_val = loss_kt.to_scalar::<f32>()? as f64;
    let mut upstream_grad = rms_norm_backward_pre_final_norm(
        final_rmsnorm_backward_route_for_backend(backend),
        &final_hidden,
        &weights.final_norm,
        &grad_normed,
        model_config.rms_norm_eps,
    )
    .context("checkpointed GRPO final RMSNorm backward")?
    .detach();

    let mut grads = kiln_autograd::GradStore::new();
    for seg_idx in (0..num_segments).rev() {
        let (start, end) = segments[seg_idx];
        let seg_input = boundaries[seg_idx].clone();
        let seg_input_id = seg_input.id();
        let seg_output_dtype = boundaries[seg_idx + 1].dtype();
        let seed = upstream_grad.to_dtype(seg_output_dtype).map_err(|e| {
            anyhow::anyhow!("checkpointed GRPO: seed dtype cast (segment {seg_idx}): {e}")
        })?;
        let positions_ref = &positions;
        let lora_ref = &lora_weights;
        let (kt_grads, candle_grads) =
            kiln_kt_bridge::tape_bridge::with_tape_segment_backward_scope(
                kiln_autograd::TapeOptions { detect_anomaly },
                seed,
                || {
                    let mut seg_ls = LinearAttentionState::new(model_config, device)
                        .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))?;
                    model_forward_segment_with_policy(
                        backend,
                        seg_input,
                        weights,
                        model_config,
                        positions_ref,
                        start,
                        end,
                        Some(&mut seg_ls),
                        Some(lora_ref),
                        streaming_prefill,
                    )
                    .map_err(|e| kiln_kt_bridge::BridgeError::new(format!("{e:#}")))
                },
            )
            .map_err(|e| {
                anyhow::anyhow!("checkpointed GRPO: segment {seg_idx} tape backward: {e}")
            })?;

        let mut segment_grads = kiln_autograd::GradStore::new();
        for (candle_raw, g) in candle_grads {
            let Some(param_raw) =
                kiln_kt_bridge::tape_bridge::decode_kt_param_deposit(candle_raw as u64)
            else {
                continue;
            };
            segment_grads.insert(KtTensorId::from_raw(param_raw), g);
        }
        let grad_context =
            format!("checkpointed GRPO segment {seg_idx} layers {start}..{end} gradient contract");
        merge_checkpoint_lora_grad_segment(
            params,
            &mut grads,
            segment_grads,
            start,
            end,
            &grad_context,
        )?;

        if seg_idx > 0 {
            upstream_grad = kt_grads.get(seg_input_id).cloned().ok_or_else(|| {
                anyhow::anyhow!(
                    "checkpointed GRPO: tape backward produced no input gradient for segment {seg_idx}"
                )
            })?;
        }
    }

    Ok((loss_val, env_ce, grads, policy_log_probs))
}

/// Bundled parameters for the GRPO surrogate / KL loss.
///
/// `loss_normalizer` is the scalar applied to the *sum* of per-token loss
/// contributions before backward. For `LossAggregation::PerSample` it is
/// `1 / num_active_tokens` for the current completion (recovering the
/// historical kiln per-completion mean). For `LossAggregation::TokenLevel`
/// it is `1 / group_total_active_tokens` so the per-token contributions sum
/// across the entire group to a DAPO-style token-level mean.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GrpoLossParams {
    pub advantage: f64,
    pub clip_low: f64,
    /// Additive PPO upper epsilon for token/sequence GRPO; absolute upper
    /// importance-weight cap for CISPO.
    pub clip_high: f64,
    pub kl_coeff: f64,
    pub kl_estimator: KlEstimator,
    pub loss_normalizer: f64,
    /// Importance-sampling level (Phase 2). `Token` is the historical
    /// per-token PPO surrogate; `Sequence` computes the IS ratio at the
    /// sequence level (GSPO, arXiv:2507.18071); `Cispo` clips the IS
    /// weight rather than the surrogate (arXiv:2506.13585).
    pub is_level: IsLevel,
    /// When true, the IS ratio is forced to 1.0 and the surrogate reduces to
    /// `advantage` per token. KL selection remains independent.
    pub reinforce: bool,
    /// Phase 3c — entropy-aware KL quantile. `None` = full-token KL; when
    /// `Some(q)`, tokens whose `-policy_log_prob` is below the q-quantile
    /// across this loss-instance's active tokens get zero KL contribution
    /// (and zero KL gradient). Approximates the Cui et al. selective-KL
    /// idea (arXiv:2506.01939).
    pub entropy_aware_kl_quantile: Option<f32>,
}

impl GrpoLossParams {
    fn from_config(config: &GrpoConfig, advantage: f64, loss_normalizer: f64) -> Self {
        let (clip_low, ppo_clip_high) = config.clip_bounds();
        let clip_high = if matches!(config.is_level, IsLevel::Cispo) {
            config.cispo_max_weight
        } else {
            ppo_clip_high
        };
        let reinforce = matches!(
            config.behavior_policy,
            BehaviorPolicy::NoImportanceCorrection
        );
        let kl_estimator = config.kl_estimator;
        // Entropy-aware KL only makes sense when KL is actually being
        // applied; gate it off otherwise so the quantile compute doesn't
        // run for nothing.
        let entropy_aware_kl_quantile = if matches!(kl_estimator, KlEstimator::None) {
            None
        } else {
            config.entropy_aware_kl_quantile
        };
        Self {
            advantage,
            clip_low,
            clip_high,
            kl_coeff: config.kl_coeff,
            kl_estimator,
            loss_normalizer,
            is_level: config.is_level,
            reinforce,
            entropy_aware_kl_quantile,
        }
    }
}

fn entropy_aware_kl_threshold_from_policy_log_probs(
    grpo_kl_auxiliary_route: GrpoKlAuxiliaryRoute,
    policy_log_probs: &Tensor,
    q: f32,
    num_active: usize,
) -> Result<f32> {
    anyhow::ensure!(
        num_active > 0,
        "entropy-aware KL threshold requires at least one active token"
    );
    let idx = ((q as f64) * (num_active.saturating_sub(1)) as f64).round() as usize;
    let idx = idx.min(num_active.saturating_sub(1));

    // Reuse backend top-k kernels when the requested quantile rank is small.
    // The kernels are intentionally k-pass, so for large ranks a single host
    // threshold read + CPU sort is less work than asking for most of the vector.
    if idx < 1024
        && matches!(
            grpo_kl_auxiliary_route,
            GrpoKlAuxiliaryRoute::CudaRocmDeviceFastPath
        )
    {
        let flat = policy_log_probs
            .to_f32_dtype()?
            .flatten_all()?
            .reshape(vec![num_active])?
            .contiguous()?;
        match try_topk_on_device(&flat, idx + 1) {
            Ok(pairs) => {
                let threshold = pairs.get(idx).map(|(_, value)| *value).ok_or_else(|| {
                    anyhow::anyhow!("entropy-aware KL top-k returned too few values")
                })?;
                return Ok(threshold);
            }
            Err(err) => {
                tracing::debug!(
                    error = %err,
                    "entropy-aware KL device top-k declined; falling back to host threshold sort"
                );
            }
        }
    }

    let plp_host: Vec<f32> = policy_log_probs
        .flatten_all()?
        .to_device(cpu_device())?
        .to_vec1::<f32>()?;
    anyhow::ensure!(
        plp_host.len() == num_active,
        "entropy-aware KL threshold plp len {} != num_active {num_active}",
        plp_host.len()
    );
    let mut neg = plp_host.iter().map(|p| -(*p as f64)).collect::<Vec<_>>();
    neg.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let thr = neg[idx];
    Ok((-thr) as f32)
}

pub(crate) fn entropy_aware_kl_mask_kt(
    grpo_kl_auxiliary_route: GrpoKlAuxiliaryRoute,
    policy_log_probs: &Tensor,
    params: GrpoLossParams,
    device: &Device,
) -> Result<Option<Tensor>> {
    let Some(q) = params.entropy_aware_kl_quantile else {
        return Ok(None);
    };
    if !(q.is_finite() && (0.0..1.0).contains(&q)) {
        return Ok(None);
    }

    let num_active = policy_log_probs.elem_count();
    if num_active == 0 {
        return Ok(Some(zeros_f32_on(policy_log_probs.shape(), device)?));
    }

    let threshold = entropy_aware_kl_threshold_from_policy_log_probs(
        grpo_kl_auxiliary_route,
        policy_log_probs,
        q,
        num_active,
    )?;
    let policy_f32 = policy_log_probs
        .to_f32_dtype()?
        .flatten_all()?
        .reshape(vec![num_active])?
        .contiguous()?;
    let threshold_tensor = policy_f32.affine(0.0, threshold as f64)?.contiguous()?;
    let keep_kl = kiln_tensor::ops::le(&policy_f32, &threshold_tensor)?;
    let ones = policy_f32.affine(0.0, 1.0)?.contiguous()?;
    let zeros = policy_f32.affine(0.0, 0.0)?.contiguous()?;
    let mask = keep_kl.where_cond(&ones, &zeros)?;
    mask.reshape(policy_log_probs.dims().to_vec())?
        .contiguous()
        .map(Some)
        .map_err(Into::into)
}

/// Compute the GRPO loss from policy, behavior-policy, and KL-reference
/// log-probs.
///
/// Returns a scalar loss tensor suitable for backward(). The scalar is
/// `params.loss_normalizer * sum_over_active_tokens(per_token_loss)`.
///
/// The structure of `per_token_loss` depends on `params.is_level`:
///   * `IsLevel::Token` — historical per-token PPO `min(r·A, clip(r)·A)`.
///   * `IsLevel::Sequence` — GSPO sequence-level scalar ratio
///     `s = exp(mean(log_ratio))`, then `min(s·A, clip(s)·A)` broadcast to
///     every active token before the configured loss aggregation.
///   * `IsLevel::Cispo` — CISPO weight clipping: the per-token gradient
///     factor `stop_grad(min(r, cispo_max_weight))·A` multiplies `log π_θ`,
///     so every token contributes a gradient without a lower weight floor.
// `pub(crate)` so the GRPO tape-authoritative loss-root shim
// (`crate::grpo_tape_shim`) can recompute the EXACT same scalar PG (+ KL)
// loss inside its candle-autograd backward composite (#1082 CP-4).
pub(crate) fn grpo_loss(
    policy_log_probs: &Tensor,
    behavior_log_probs: &Tensor,
    kl_reference_log_probs: &Tensor,
    params: GrpoLossParams,
    device: &Device,
) -> Result<Tensor> {
    grpo_loss_with_kl_auxiliary_route(
        GrpoKlAuxiliaryRoute::HostComposite,
        policy_log_probs,
        behavior_log_probs,
        kl_reference_log_probs,
        params,
        device,
    )
}

pub(crate) fn grpo_loss_with_kl_auxiliary_route(
    grpo_kl_auxiliary_route: GrpoKlAuxiliaryRoute,
    policy_log_probs: &Tensor,
    behavior_log_probs: &Tensor,
    kl_reference_log_probs: &Tensor,
    params: GrpoLossParams,
    device: &Device,
) -> Result<Tensor> {
    let num_active = policy_log_probs.elem_count();
    if num_active == 0 {
        // Scalar zero loss (empty active set). kt-native.
        return zeros_f32_on((), device).map_err(Into::into);
    }

    anyhow::ensure!(
        behavior_log_probs.elem_count() == num_active,
        "GRPO behavior log-probability count {} did not match policy count {num_active}",
        behavior_log_probs.elem_count()
    );
    anyhow::ensure!(
        kl_reference_log_probs.elem_count() == num_active,
        "GRPO KL-reference log-probability count {} did not match policy count {num_active}",
        kl_reference_log_probs.elem_count()
    );

    // `reinforce` is the explicit no-importance-correction mode. Its ratio is
    // one by value while retaining the policy gradient; the independently
    // configured KL term below still uses `kl_reference_log_probs`.
    let importance_log_ratio = if params.reinforce {
        (policy_log_probs - policy_log_probs.detach())?
    } else {
        (policy_log_probs - behavior_log_probs)?
    };
    let ratio = importance_log_ratio.exp()?;
    let ratio_shape = ratio.dims().to_vec();
    let kl_log_ratio = (policy_log_probs - kl_reference_log_probs)?;

    // Asymmetric PPO clip range: [1 - clip_low, 1 + clip_high]. CISPO
    // interprets clip_high separately below as its absolute upper weight cap.
    let lo_val = 1.0 - params.clip_low;
    let hi_val = 1.0 + params.clip_high;

    // Per-token KL term selected by KlEstimator (shared across IS levels).
    let kl_penalty_raw = match params.kl_estimator {
        KlEstimator::None => zeros_f32_on(ratio.shape(), device)?,
        KlEstimator::K1 => kl_log_ratio.affine(params.kl_coeff, 0.0)?,
        KlEstimator::K3 => {
            let neg_log_ratio = kl_log_ratio.neg()?;
            let term = (neg_log_ratio.exp()?.affine(1.0, -1.0)? + &kl_log_ratio)?;
            term.affine(params.kl_coeff, 0.0)?
        }
    };
    // Phase 3c — selective KL gating: zero KL on tokens below the proxy-entropy threshold.
    let kl_penalty = if let Some(q) = params.entropy_aware_kl_quantile {
        if q.is_finite() && (0.0..1.0).contains(&q) {
            let mask = entropy_aware_kl_mask_kt(
                grpo_kl_auxiliary_route,
                policy_log_probs,
                params,
                device,
            )?
            .ok_or_else(|| anyhow::anyhow!("entropy-aware KL mask unexpectedly absent"))?;
            (&kl_penalty_raw * &mask)?
        } else {
            kl_penalty_raw
        }
    } else {
        kl_penalty_raw
    };

    let neg_surrogate = if params.reinforce {
        ratio.affine(-params.advantage, 0.0)?
    } else {
        match params.is_level {
            IsLevel::Token => {
                // Per-token surrogate: -min(r·A, clip(r)·A).
                // (#1082) kt `clamp` takes scalar bounds directly; advantage folds
                // into `affine` (constant scalar, gradient flows through the ratio).
                let clipped_ratio = ratio.clamp(lo_val, hi_val)?;
                let surr1 = ratio.affine(params.advantage, 0.0)?;
                let surr2 = clipped_ratio.affine(params.advantage, 0.0)?;
                let surrogate = surr1.minimum(&surr2)?;
                surrogate.neg()?
            }
            IsLevel::Sequence => {
                // GSPO: s = exp(mean(log_ratio)), surrogate at sequence level,
                // gradient distributed back equally to every active token.
                //
                // The sequence surrogate is replicated over active tokens;
                // the outer loss normalizer performs the sequence mean.
                let u = importance_log_ratio.mean_keepdim(0)?;
                let s = u.exp()?;
                // (#1082) kt scalar clamp + scalar `affine` for the constant
                // advantage (gradient flows through `s`).
                let clipped = s.clamp(lo_val, hi_val)?;
                let surr1 = s.affine(params.advantage, 0.0)?;
                let surr2 = clipped.affine(params.advantage, 0.0)?;
                let surrogate = surr1.minimum(&surr2)?;
                // Repeat the sequence loss across its active token positions.
                // The outer per-sample normalizer divides the sum by
                // num_active, matching TRL/GSPO. The derivative of the shared
                // sequence ratio already contributes its own 1/num_active.
                let neg = surrogate.neg()?;
                neg.broadcast_as(&ratio_shape)?
            }
            IsLevel::Cispo => {
                // CISPO: gradient through `log π_θ` only; the IS weight is the
                // *clipped* ratio with stop-gradient. The total loss contribution
                // is `-stop_grad(clip(r)) · A · log π_θ` per token.
                // (#1082) kt scalar clamp; advantage folds into `affine`. `weight`
                // is detached either way, so the constant scalar mul is exact.
                let clipped_ratio = ratio.clamp(0.0, params.clip_high)?.detach();
                // log π_θ = policy_log_probs (already in tensor form).
                let weight = clipped_ratio.affine(params.advantage, 0.0)?.detach();
                (&weight * policy_log_probs)?.neg()?
            }
        }
    };

    let per_token_loss = (&neg_surrogate + &kl_penalty)?;
    let total = per_token_loss.sum_all()?;
    total
        .affine(params.loss_normalizer, 0.0)
        .map_err(Into::into)
}

/// Run one GRPO training step with exact reverse-mode checkpointing.
///
/// Reference log-probs are pre-computed and passed in. The policy path mirrors
/// `checkpointed_forward_backward`: compute detached segment boundaries, seed
/// the final boundary with an analytic GRPO tail gradient, then walk segments
/// backward with gradient injection. This preserves full-sequence context and
/// propagates downstream gradients into every LoRA segment.
// Optional `echo: Option<EchoTailParams>` last param. When `Some`, the
// analytic tail folds the env-CE term into the same vocab-chunk
// forward+backward loop so the checkpointed GRPO path applies ECHO too
// (Phase 1 follow-up of docs/plans/echo-integration-plan.md).

// (#1082 CP-4) `pub(crate)` so the OPD tape-authoritative test in `opd.rs`'s
// own `#[cfg(test)] mod tests` can reuse the BF16 tiny-model fixtures
// (`tiny_config_bf16` / `tiny_weights_bf16`) instead of duplicating them —
// single source of truth for the BF16 CUDA fixture. Still `#[cfg(test)]`, so
// it carries no cost in non-test builds.
#[cfg(test)]
pub(crate) mod tests;
