//! CUDA-native training entry points.
//!
//! (#1082 Wave F2) The hand-rolled candle-autograd CUDA training engine
//! that used to back this module — `kiln_model::cuda_train`'s
//! `CudaTrainTensor` / `CudaBackwardOp` graph, `cuda_backward`,
//! `cuda_adamw_step_from_store`, and the `Cuda*` LoRA model-forward
//! helpers (`cuda_*_lora_*`, the layerwise reverse-recompute step, the
//! CUDA-native LoRA adapter save) — has been **deleted**. It was a
//! candle-authoritative training path; the kt tape (`kiln_autograd::Tape`)
//! is now the sole gradient producer, so the candle engine and every
//! consumer of it are gone.
//!
//! What remains are the three public CUDA-native entry-point symbols the
//! server / bench call (`cuda_native_sft_train`, `cuda_native_grpo_train`,
//! `cuda_native_grpo_train_jsonl`). Since the #1063 fix, all three
//! **already delegate** to the `crate::trainer` path (`sft_train` /
//! `grpo_train` / `grpo_train_jsonl`), which dispatches each layer
//! through `BackendRuntime` and runs the production autograd path. The
//! legacy/recompute env-flag bypass branches (`KILN_CUDA_LEGACY_NATIVE_STEP`
//! / `KILN_CUDA_RECOMPUTE_SFT`) drove the now-deleted candle engine, so
//! they are removed; the entry points are pure delegators. The
//! `trainer::` path itself is being migrated to the kt tape in Wave E1
//! (Var→`kiln_param::Parameter`, optimizer→`kiln_optim::AdamW`), so these
//! wrappers pick up the kt path transparently once E1 lands.
//!
//! The symbols stay so the server's `KILN_CUDA_NATIVE_TRAINING=1`
//! routing (`kiln-server::training_queue` / `bench`) keeps a stable
//! function to call without needing to know that the CUDA-native step no
//! longer has a hand-rolled engine of its own.

use anyhow::Result;

use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::forward::GpuWeights;
use std::path::{Path, PathBuf};

use crate::trainer::ProgressCallback;
use crate::{GrpoConfig, GrpoGroup, SftConfig, SftExample};

/// CUDA-native SFT entry point. Delegates to [`crate::trainer::sft_train`]
/// (the BackendRuntime + autograd path; #1063), which is being migrated to
/// the kt tape in Wave E1.
#[allow(clippy::too_many_arguments)]
pub fn cuda_native_sft_train(
    examples: &[SftExample],
    config: &SftConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
    gpu_step_lock: Option<std::sync::Arc<tokio::sync::RwLock<()>>>,
) -> Result<PathBuf> {
    tracing::info!(
        num_examples = examples.len(),
        epochs = config.epochs,
        lr = config.effective_learning_rate(),
        rank = config.lora_rank,
        alpha = config.lora_alpha,
        adapter_name,
        path = "backend_runtime_via_sft_train",
        "cuda_native_sft_train: routing through the trainer BackendRuntime path"
    );
    crate::trainer::sft_train(
        examples,
        config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        adapter_name,
        progress_cb,
        None,
        gpu_step_lock,
    )
}

/// CUDA-native GRPO entry point. Delegates to [`crate::trainer::grpo_train`].
///
/// The `cuda_train` module never had a separate GRPO step kernel of its
/// own; the symmetric thing for GRPO is to route through `grpo_train` and
/// pick up the same autograd path as SFT.
#[allow(clippy::too_many_arguments)]
pub fn cuda_native_grpo_train(
    groups: &[GrpoGroup],
    config: &GrpoConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
) -> Result<PathBuf> {
    tracing::info!(
        num_groups = groups.len(),
        learning_rate = config.effective_learning_rate(),
        kl_coeff = config.kl_coeff,
        rank = config.lora_rank,
        adapter_name,
        path = "backend_runtime_via_grpo_train",
        "cuda_native_grpo_train: routing through the trainer BackendRuntime path"
    );
    crate::trainer::grpo_train(
        groups,
        config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        adapter_name,
        progress_cb,
        None,
    )
}

/// Streaming-dataset variant of [`cuda_native_grpo_train`]. Delegates to
/// [`crate::trainer::grpo_train_jsonl`] so the server's `dataset_path`
/// GRPO path also has a CUDA-native entry point.
#[allow(clippy::too_many_arguments)]
pub fn cuda_native_grpo_train_jsonl(
    dataset_path: &Path,
    config: &GrpoConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
) -> Result<PathBuf> {
    tracing::info!(
        dataset_path = %dataset_path.display(),
        learning_rate = config.effective_learning_rate(),
        kl_coeff = config.kl_coeff,
        rank = config.lora_rank,
        adapter_name,
        path = "backend_runtime_via_grpo_train_jsonl",
        "cuda_native_grpo_train_jsonl: routing through the trainer BackendRuntime path"
    );
    crate::trainer::grpo_train_jsonl(
        dataset_path,
        config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        adapter_name,
        progress_cb,
        None,
    )
}
