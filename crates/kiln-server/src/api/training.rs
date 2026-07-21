//! Training API endpoints — pure Rust, in-process LoRA training.
//!
//! Training requests are enqueued in a FIFO queue and executed sequentially
//! by a background worker. This prevents GPU memory conflicts between
//! concurrent training jobs.

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path as AxumPath, State, rejection::JsonRejection},
    routing::{delete, get, post},
};

use kiln_train::{
    DistillMergeRequest, DistillPumpRequest, DistillRefreshRequest, DistillSelfRequest, GrpoGroup,
    GrpoRequest, OpdRequest, SftRequest, TrainingResponse, TrainingState, TrainingStatus,
};
use serde::Serialize;

use std::{
    path::{Path, PathBuf},
    sync::atomic::Ordering,
};

use crate::error::ApiError;
use crate::metrics::{TrainingMetricStatus, TrainingMetricType};
use crate::state::TrainingWorkload;
use crate::state::{AppState, ModelBackend, TrainingJobInfo, TrainingJobType};
use crate::training_preflight::{
    self, EstimateOptions, LoraResidency, SftEstimateOptions, WeightResidency,
    auto_fit_checkpoint_segments, estimate_step_working_set_with_options,
    estimate_vk_native_recompute_working_set_with_residency, format_oom_message_with_source,
};
use crate::training_queue::{QueueEntry, QueuedJob};
use kiln_memory::vram::VramSource;
use kiln_train::TrainingDataProvenance;

#[derive(Debug)]
struct GrpoSubmissionStats {
    num_groups: Option<usize>,
    total_completions: Option<usize>,
    max_seq_len: usize,
    streaming_dataset: bool,
    source_receipt: Option<crate::training_queue::GrpoJsonlAdmissionReceipt>,
}

pub(crate) fn parse_training_json<T>(
    payload: Result<Json<T>, JsonRejection>,
    surface: &str,
) -> Result<T, ApiError> {
    payload.map(|Json(value)| value).map_err(|error| {
        if error.status() == axum::http::StatusCode::PAYLOAD_TOO_LARGE {
            ApiError::training_request_too_large(surface)
        } else {
            ApiError::training_invalid_request(format!(
                "invalid {surface} JSON: {}",
                error.body_text()
            ))
        }
    })
}

#[derive(Debug, Clone)]
struct SftSubmissionStats {
    rows_read: usize,
    num_examples: usize,
    rows_rejected: usize,
    max_seq_len: usize,
    max_supervised_tokens: usize,
    streaming_dataset: bool,
}

struct TrainingAdmissionResult {
    queue_position: usize,
    sft_summaries: std::collections::HashMap<String, SftSubmissionStats>,
    effective_seeds: std::collections::HashMap<String, u64>,
}

impl TrainingAdmissionResult {
    fn effective_seed(&self, job_id: &str) -> Result<u64, ApiError> {
        self.effective_seeds.get(job_id).copied().ok_or_else(|| {
            ApiError::internal(format!(
                "training admission completed without an effective seed for job {job_id}"
            ))
        })
    }
}

fn retained_correction_ids(
    ids: Vec<String>,
    ingestion: &kiln_train::SftIngestionReceipt,
) -> Result<Vec<String>, ApiError> {
    ingestion.validate().map_err(|error| {
        ApiError::internal(format!(
            "corrections SFT ingestion receipt failed validation: {error:#}"
        ))
    })?;
    if ids.len() != ingestion.rows_read {
        return Err(ApiError::internal(format!(
            "corrections row IDs ({}) differ from ingested rows ({})",
            ids.len(),
            ingestion.rows_read
        )));
    }

    let rejected = ingestion
        .rejected_rows
        .iter()
        .map(|row| row.row_index)
        .collect::<std::collections::HashSet<_>>();
    let retained = ids
        .into_iter()
        .enumerate()
        .filter_map(|(index, id)| (!rejected.contains(&(index + 1))).then_some(id))
        .collect::<Vec<_>>();
    if retained.len() != ingestion.rows_kept {
        return Err(ApiError::internal(format!(
            "retained correction IDs ({}) differ from kept SFT rows ({})",
            retained.len(),
            ingestion.rows_kept
        )));
    }
    Ok(retained)
}

struct PreflightAdmission {
    reserved_bytes: u64,
    checkpoint_segments: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct TrainingMemoryAvailability {
    bytes: u64,
    live_bytes: u64,
    allocator_bytes: Option<u64>,
    reclaimable_kv_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
struct TrainingActivationEstimate {
    bytes_per_elem: usize,
    streaming_gdn_tile_tokens: Option<usize>,
}

fn model_dtype_bytes(dtype: kiln_core::config::DType) -> usize {
    match dtype {
        kiln_core::config::DType::BF16 | kiln_core::config::DType::FP16 => 2,
        kiln_core::config::DType::FP32 => 4,
    }
}

const GDN_TAPE_EFFECTIVE_BYTES_PER_ELEM: usize = 10;

fn training_activation_bytes_per_elem(
    base: usize,
    uses_f32_activations: bool,
    has_linear_attention: bool,
) -> usize {
    if uses_f32_activations {
        return 4;
    }
    if has_linear_attention {
        base.max(GDN_TAPE_EFFECTIVE_BYTES_PER_ELEM)
    } else {
        base
    }
}

fn training_activation_estimate_for_streaming_policy(
    base: usize,
    uses_f32_activations: bool,
    has_linear_attention: bool,
    streaming_prefill: kiln_model::StreamingPrefillExecutionPolicy,
    max_seq_len: usize,
) -> TrainingActivationEstimate {
    // Mirror kiln-train's GDN tape sizing. The server stamps resolved
    // checkpoint segments at submission time, so an optimistic bf16-only
    // preflight would bypass the trainer's more conservative auto-tuner.
    let bytes_per_elem =
        training_activation_bytes_per_elem(base, uses_f32_activations, has_linear_attention);
    let streaming_gdn_tile_tokens = if has_linear_attention {
        streaming_prefill
            .enabled_for(max_seq_len)
            .then(|| streaming_prefill.tape_tile_tokens())
            .filter(|&tile| tile > 0 && tile < max_seq_len)
    } else {
        None
    };
    TrainingActivationEstimate {
        bytes_per_elem,
        streaming_gdn_tile_tokens,
    }
}

fn training_activation_estimate_for_state(
    state: &AppState,
    max_seq_len: usize,
) -> TrainingActivationEstimate {
    let base = model_dtype_bytes(state.model_config.dtype);
    let ModelBackend::Real { runner, .. } = state.backend.as_ref() else {
        return TrainingActivationEstimate {
            bytes_per_elem: base,
            streaming_gdn_tile_tokens: None,
        };
    };
    let Ok(runner) = runner.read() else {
        tracing::warn!(
            "model runner lock poisoned while sizing training preflight; using model dtype width"
        );
        return TrainingActivationEstimate {
            bytes_per_elem: base,
            streaming_gdn_tile_tokens: None,
        };
    };
    let capabilities = runner.backend_capabilities();
    let has_linear_attention = state.model_config.num_full_attention_layers
        < state.model_config.num_layers
        || runner
            .weights
            .linear_attention_layers_in_prefix(runner.config.num_layers)
            > 0;
    training_activation_estimate_for_streaming_policy(
        base,
        runner
            .training_precision_policy()
            .uses_f32_activations_for_mixed_base_weights(),
        has_linear_attention,
        state
            .training_runtime
            .resolved_streaming_prefill_policy(capabilities.device),
        max_seq_len,
    )
}

fn sft_loss_route_for_state(
    state: &AppState,
) -> Result<kiln_model::backend::SftFlceLossRoute, ApiError> {
    let ModelBackend::Real { runner, .. } = state.backend.as_ref() else {
        return Ok(kiln_model::backend::SftFlceLossRoute::FullLogits);
    };
    runner
        .read()
        .map_err(|_| {
            ApiError::internal("model runner lock poisoned while resolving SFT loss route")
        })
        .map(|runner| runner.sft_flce_loss_route())
}

fn training_optimizer_support_api_error(
    backend: &str,
    error: kiln_model::TrainingOptimizerSupportError,
) -> ApiError {
    match error {
        kiln_model::TrainingOptimizerSupportError::UnsupportedBaseWeightDType { .. } => {
            ApiError::training_backend_unsupported(format!(
                "backend `{backend}` cannot train its resident base-weight dtype: {error}"
            ))
        }
        kiln_model::TrainingOptimizerSupportError::UnsupportedRequest { .. } => {
            ApiError::training_invalid_request(format!(
                "optimizer request is unsupported by backend `{backend}`: {error}"
            ))
        }
    }
}

fn enforce_model_lora_rank_admission(state: &AppState, lora_rank: usize) -> Result<(), ApiError> {
    let model_rank_ceiling = training_preflight::model_lora_rank_ceiling(&state.model_config);
    if lora_rank == 0 || lora_rank > model_rank_ceiling {
        return Err(ApiError::training_invalid_request(format!(
            "LoRA rank {lora_rank} is invalid for this model; supported range is 1..={model_rank_ceiling}"
        )));
    }
    Ok(())
}

pub(crate) fn enforce_training_optimizer_admission(
    state: &AppState,
    optimizer: kiln_train::Optimizer,
    lora_rank: usize,
) -> Result<(), ApiError> {
    optimizer.validate_hyperparameters().map_err(|error| {
        ApiError::training_invalid_request(format!("invalid optimizer configuration: {error:#}"))
    })?;
    enforce_model_lora_rank_admission(state, lora_rank)?;
    let ModelBackend::Real { runner, .. } = state.backend.as_ref() else {
        return Ok(());
    };
    let runner = runner.read().map_err(|_| {
        ApiError::internal("model runner lock poisoned while validating training optimizer")
    })?;
    let capabilities = runner.backend_capabilities();
    let base_weight_device = runner.weights.embed_tokens.device();
    if capabilities.device != base_weight_device {
        return Err(ApiError::training_backend_unsupported(format!(
            "training backend `{}` reports device {} but resident weights are on {}; native optimizer admission requires an exact backend identity",
            capabilities.backend, capabilities.device, base_weight_device
        )));
    }
    let base_weight_dtype = runner.weights.embed_tokens.dtype();
    capabilities
        .training
        .resolve_optimizer_request(
            optimizer.kind(),
            base_weight_dtype,
            kiln_model::TrainingOptimizerRounding::RoundToNearest,
            lora_rank,
        )
        .map(|_| ())
        .map_err(|error| training_optimizer_support_api_error(capabilities.backend, error))
}

pub(crate) fn enforce_training_workload_admission(
    state: &AppState,
    workload: TrainingWorkload,
) -> Result<(), ApiError> {
    if matches!(state.backend.as_ref(), ModelBackend::Mock { .. }) {
        return Err(ApiError::mock_mode_no_training());
    }
    if let Some(reason) = state.training_workload_unavailable_reason(workload) {
        return Err(ApiError::training_backend_unsupported(reason));
    }
    Ok(())
}

fn effective_checkpoint_segments(config: kiln_train::CheckpointConfig) -> usize {
    if config.enabled {
        config.num_segments
    } else {
        1
    }
}

fn ensure_sft_checkpoint_plan_supported(
    sft: Option<SftEstimateOptions>,
    num_segments: usize,
) -> Result<(), ApiError> {
    if num_segments > 1
        && sft
            .is_some_and(|sft| sft.loss_route == kiln_model::backend::SftFlceLossRoute::FullLogits)
    {
        return Err(ApiError::training_invalid_request(
            "checkpointed SFT does not support backend loss route `full_logits`: \
             checkpoint tails run outside an active kt tape; disable gradient \
             checkpointing or use a backend with a checkpoint-compatible SFT loss route",
        ));
    }
    Ok(())
}

fn combine_training_available_bytes(
    live_bytes: u64,
    allocator_bytes: Option<u64>,
    reclaimable_kv_bytes: u64,
    total_bytes: u64,
    floor_bytes: u64,
    unified_memory: bool,
) -> u64 {
    let base = if unified_memory {
        live_bytes
    } else {
        allocator_bytes.map_or(live_bytes, |bytes| live_bytes.max(bytes))
    };
    let with_reclaimable = base.saturating_add(reclaimable_kv_bytes);
    if total_bytes == 0 {
        with_reclaimable
    } else {
        with_reclaimable.min(total_bytes.saturating_sub(floor_bytes))
    }
}

fn apply_configured_training_budget_cap(
    available_bytes: u64,
    configured_training_memory_gb: Option<f64>,
    effective_training_budget_bytes: u64,
) -> u64 {
    if configured_training_memory_gb.is_some() {
        available_bytes.min(effective_training_budget_bytes)
    } else {
        available_bytes
    }
}

fn dynamic_training_availability(
    state: &AppState,
    vram: &kiln_memory::vram::GpuVramInfo,
    live_policy_available: u64,
    soft_reserved: u64,
) -> TrainingMemoryAvailability {
    let governor = kiln_memory::MemoryGovernor::global();
    let floor_bytes = governor.config().floor_bytes;
    // The governor has already applied the effective capacity, safety floor,
    // and outstanding soft reservations to this value.
    let live_bytes = live_policy_available;
    let mut allocator_bytes = None;
    let mut reclaimable_kv_bytes = 0u64;

    if let ModelBackend::Real {
        runner,
        paged_cache,
        ..
    } = state.backend.as_ref()
        && let Ok(runner) = runner.read()
    {
        let capabilities = runner.backend_capabilities();
        let device = runner.weights.embed_tokens.device();
        allocator_bytes = crate::device_memory::allocator_safe_available_bytes_with_soft_reserved(
            capabilities.storage.gpu_allocator_memory_probe_policy,
            &device,
            floor_bytes,
            soft_reserved,
        );

        let cache_device_matches_model = paged_cache
            .device()
            .is_some_and(|cache_device| cache_device == device);
        if capabilities.storage.kv_cache_device_memory_pressure && cache_device_matches_model {
            let current_blocks = paged_cache.num_blocks();
            let bytes_per_block = paged_cache.bytes_per_block() as u64;
            reclaimable_kv_bytes = current_blocks.saturating_sub(1) as u64 * bytes_per_block;
        }
    }

    let dynamic_bytes = combine_training_available_bytes(
        live_bytes,
        allocator_bytes,
        reclaimable_kv_bytes,
        vram.total_bytes,
        floor_bytes,
        vram.unified,
    );
    let bytes = apply_configured_training_budget_cap(
        dynamic_bytes,
        state.memory_config.training_memory_gb,
        state.memory_budget.training_budget_bytes,
    );
    if bytes < dynamic_bytes {
        tracing::debug!(
            dynamic_available_bytes = dynamic_bytes,
            configured_training_budget_bytes = bytes,
            "training availability capped by memory.training_memory_gb"
        );
    }
    TrainingMemoryAvailability {
        bytes,
        live_bytes,
        allocator_bytes,
        reclaimable_kv_bytes,
    }
}

fn validate_grpo_jsonl_submission(
    dataset_path: &str,
    snapshot_root: &Path,
    prepared_data_permit: &mut crate::training_queue::PreparedTrainingDataPermit,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    config: &kiln_train::GrpoConfig,
    model_num_layers: usize,
    contamination: Option<(
        &kiln_eval::EvalContaminationIndex,
        &kiln_eval::PostEvalConfig,
    )>,
) -> Result<GrpoSubmissionStats, ApiError> {
    use std::fs::OpenOptions;
    use std::io::{BufRead, BufReader, Read, Seek, Write};

    use sha2::{Digest, Sha256};

    let canonical_path = std::fs::canonicalize(dataset_path).map_err(|e| {
        ApiError::training_invalid_request(format!(
            "failed to resolve GRPO dataset_path '{dataset_path}': {e}"
        ))
    })?;

    #[cfg(unix)]
    let file = {
        use std::os::unix::fs::OpenOptionsExt as _;
        let mut options = OpenOptions::new();
        options
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        options.open(&canonical_path)
    };
    #[cfg(not(unix))]
    let file = std::fs::File::open(&canonical_path);
    let file = file.map_err(|e| {
        ApiError::training_invalid_request(format!(
            "failed to open GRPO dataset_path '{dataset_path}': {e}"
        ))
    })?;
    let metadata = file.metadata().map_err(|e| {
        ApiError::training_invalid_request(format!(
            "failed to inspect GRPO dataset_path '{dataset_path}': {e}"
        ))
    })?;
    if !metadata.is_file() {
        return Err(ApiError::training_invalid_request(format!(
            "GRPO dataset_path '{dataset_path}' must resolve to a regular file"
        )));
    }
    if metadata.len() > kiln_train::HF_TRL_GRPO_MAX_DATASET_BYTES {
        return Err(ApiError::training_invalid_request(format!(
            "GRPO dataset_path '{dataset_path}' is {} bytes; maximum supported size is {} bytes",
            metadata.len(),
            kiln_train::HF_TRL_GRPO_MAX_DATASET_BYTES
        )));
    }
    if metadata.len() > crate::training_queue::MAX_LIVE_PREPARED_TRAINING_BYTES {
        return Err(ApiError::training_invalid_request(format!(
            "GRPO dataset_path '{dataset_path}' is {} bytes; the live server-owned training-data limit is {} bytes",
            metadata.len(),
            crate::training_queue::MAX_LIVE_PREPARED_TRAINING_BYTES
        )));
    }
    prepared_data_permit
        .grow_to(metadata.len())
        .map_err(|(current, requested)| {
            ApiError::training_prepared_data_full(current, requested)
        })?;

    let snapshot_root = crate::training_queue::prepare_grpo_snapshot_root(snapshot_root)
        .map_err(ApiError::internal)?;
    let snapshot_path = crate::training_queue::new_grpo_snapshot_path(&snapshot_root);
    struct RemoveIncompleteSnapshot {
        path: PathBuf,
        armed: bool,
    }
    impl Drop for RemoveIncompleteSnapshot {
        fn drop(&mut self) {
            if self.armed {
                let _ = crate::training_queue::remove_regular_grpo_snapshot(&self.path);
            }
        }
    }
    let mut cleanup_incomplete = RemoveIncompleteSnapshot {
        path: snapshot_path.clone(),
        armed: true,
    };
    // Declare the file after the cleanup guard so error unwinding closes the
    // descriptor before removal on platforms that forbid deleting open files.
    let mut snapshot = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&snapshot_path)
        .map_err(|error| {
            ApiError::internal(format!(
                "create private GRPO admission snapshot {}: {error}",
                snapshot_path.display()
            ))
        })?;
    // Pin a read-only descriptor while the create-new writer still owns the
    // name. The writer is closed before the snapshot becomes queued, so the
    // retained descriptor cannot mutate admitted bytes.
    #[cfg(unix)]
    let snapshot_reader = {
        use std::os::unix::fs::OpenOptionsExt as _;
        let mut options = OpenOptions::new();
        options
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        options.open(&snapshot_path)
    };
    #[cfg(not(unix))]
    let snapshot_reader = std::fs::File::open(&snapshot_path);
    let mut snapshot_reader = snapshot_reader.map_err(|error| {
        ApiError::internal(format!(
            "pin private GRPO admission snapshot {}: {error}",
            snapshot_path.display()
        ))
    })?;

    let expected_bytes = metadata.len();
    let mut reader = BufReader::with_capacity(256 * 1024, file);
    let mut row = Vec::new();
    let mut hasher = Sha256::new();
    let mut bytes_read = 0u64;
    let mut line_no = 0usize;
    let mut groups = 0usize;
    let mut completions = 0usize;
    let mut max_seq_len = 0usize;
    let mut max_row_bytes = 0u64;

    loop {
        row.clear();
        let read = (&mut reader)
            .take(kiln_train::trainer::MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES + 1)
            .read_until(b'\n', &mut row)
            .map_err(|e| {
                ApiError::training_invalid_request(format!(
                    "failed to read GRPO dataset_path '{dataset_path}' line {}: {e}",
                    line_no + 1
                ))
            })?;
        if read == 0 {
            break;
        }
        line_no = line_no.checked_add(1).ok_or_else(|| {
            ApiError::training_invalid_request("GRPO dataset line count overflow")
        })?;
        if row.len() as u64 > kiln_train::trainer::MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES {
            return Err(ApiError::training_invalid_request(format!(
                "GRPO JSONL line {line_no} in '{dataset_path}' exceeds the {} byte row limit",
                kiln_train::trainer::MAX_STREAMED_GRPO_PREFLIGHT_ROW_BYTES
            )));
        }
        max_row_bytes = max_row_bytes.max(row.len() as u64);
        let projected_host_bytes = kiln_train::trainer::streamed_grpo_preflight_host_bytes(
            groups,
            completions,
            max_row_bytes,
            model_num_layers,
            config.reward_filter_var_min.is_some() || config.reward_filter_var_max.is_some(),
        )
        .map_err(|error| {
            ApiError::training_invalid_request(format!(
                "GRPO dataset_path '{dataset_path}' exceeds the streamed preflight host-memory contract before parsing line {line_no}: {error:#}"
            ))
        })?;
        bytes_read = bytes_read.checked_add(read as u64).ok_or_else(|| {
            ApiError::training_invalid_request("GRPO dataset byte count overflow")
        })?;
        if bytes_read > crate::training_queue::MAX_LIVE_PREPARED_TRAINING_BYTES {
            return Err(ApiError::training_invalid_request(format!(
                "GRPO dataset_path '{dataset_path}' grew beyond the live server-owned training-data limit of {} bytes while it was being admitted",
                crate::training_queue::MAX_LIVE_PREPARED_TRAINING_BYTES
            )));
        }
        let projected_admission_bytes = expected_bytes
            .checked_add(projected_host_bytes)
            .ok_or_else(|| ApiError::training_invalid_request("GRPO admission weight overflow"))?;
        prepared_data_permit
            .grow_to(projected_admission_bytes)
            .map_err(|(current, requested)| {
                ApiError::training_prepared_data_full(current, requested)
            })?;
        hasher.update(&row);
        snapshot.write_all(&row).map_err(|error| {
            ApiError::internal(format!(
                "write GRPO admission snapshot {}: {error}",
                snapshot_path.display()
            ))
        })?;
        let line = std::str::from_utf8(&row).map_err(|e| {
            ApiError::training_invalid_request(format!(
                "GRPO JSONL line {line_no} in '{dataset_path}' is not UTF-8: {e}"
            ))
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let group: GrpoGroup = serde_json::from_str(trimmed).map_err(|e| {
            ApiError::training_invalid_request(format!(
                "invalid GRPO JSONL group at line {} in '{dataset_path}': {e}",
                line_no
            ))
        })?;
        if let Some((index, post_eval)) = contamination
            && let Some(overlap) = check_grpo_group_contamination(index, &group)
        {
            return Err(contamination_error(post_eval, overlap));
        }
        if group.completions.is_empty() {
            return Err(ApiError::training_invalid_request(format!(
                "GRPO JSONL group at line {line_no} in '{dataset_path}' has no completions"
            )));
        }
        if group.completions.len() > kiln_train::HF_TRL_GRPO_MAX_COMPLETIONS_PER_GROUP {
            return Err(ApiError::training_invalid_request(format!(
                "GRPO JSONL group at line {line_no} in '{dataset_path}' has {} completions; maximum is {}",
                group.completions.len(),
                kiln_train::HF_TRL_GRPO_MAX_COMPLETIONS_PER_GROUP
            )));
        }
        groups = groups.checked_add(1).ok_or_else(|| {
            ApiError::training_invalid_request("GRPO dataset group count overflow")
        })?;
        if groups as u64 > kiln_train::HF_TRL_GRPO_MAX_GROUPS {
            return Err(ApiError::training_invalid_request(format!(
                "GRPO dataset_path '{dataset_path}' exceeds the {} group limit",
                kiln_train::HF_TRL_GRPO_MAX_GROUPS
            )));
        }
        completions = completions
            .checked_add(group.completions.len())
            .ok_or_else(|| {
                ApiError::training_invalid_request("GRPO dataset completion count overflow")
            })?;
        let projected_host_bytes = kiln_train::trainer::streamed_grpo_preflight_host_bytes(
            groups,
            completions,
            max_row_bytes,
            model_num_layers,
            config.reward_filter_var_min.is_some() || config.reward_filter_var_max.is_some(),
        )
        .map_err(|error| {
            ApiError::training_invalid_request(format!(
                "GRPO dataset_path '{dataset_path}' exceeds the streamed preflight host-memory contract at line {line_no}: {error:#}"
            ))
        })?;
        let projected_admission_bytes = expected_bytes
            .checked_add(projected_host_bytes)
            .ok_or_else(|| ApiError::training_invalid_request("GRPO admission weight overflow"))?;
        prepared_data_permit
            .grow_to(projected_admission_bytes)
            .map_err(|(current, requested)| {
                ApiError::training_prepared_data_full(current, requested)
            })?;
        let row_max = kiln_train::trainer::validate_grpo_group_policy_data_and_max_seq_len(
            &group, config, tokenizer, line_no,
        )
        .map_err(|error| {
            ApiError::training_invalid_request(format!(
                "invalid GRPO JSONL group at line {line_no} in '{dataset_path}': {error:#}"
            ))
        })?;
        max_seq_len = max_seq_len.max(row_max);
    }

    if bytes_read != expected_bytes {
        return Err(ApiError::training_invalid_request(format!(
            "GRPO dataset_path '{dataset_path}' changed while it was being admitted: expected {expected_bytes} bytes, read {bytes_read}"
        )));
    }
    if groups == 0 {
        return Err(ApiError::training_invalid_request(format!(
            "GRPO dataset_path '{dataset_path}' contains no groups"
        )));
    }
    snapshot.flush().map_err(|error| {
        ApiError::internal(format!(
            "flush GRPO admission snapshot {}: {error}",
            snapshot_path.display()
        ))
    })?;
    snapshot.sync_data().map_err(|error| {
        ApiError::internal(format!(
            "sync GRPO admission snapshot {}: {error}",
            snapshot_path.display()
        ))
    })?;
    drop(snapshot);
    let mut snapshot_permissions = std::fs::metadata(&snapshot_path)
        .map_err(|error| {
            ApiError::internal(format!(
                "inspect GRPO admission snapshot {}: {error}",
                snapshot_path.display()
            ))
        })?
        .permissions();
    snapshot_permissions.set_readonly(true);
    std::fs::set_permissions(&snapshot_path, snapshot_permissions).map_err(|error| {
        ApiError::internal(format!(
            "make GRPO admission snapshot {} read-only: {error}",
            snapshot_path.display()
        ))
    })?;
    snapshot_reader.rewind().map_err(|error| {
        ApiError::internal(format!(
            "rewind GRPO admission snapshot {}: {error}",
            snapshot_path.display()
        ))
    })?;
    let preflight_host_bytes = kiln_train::trainer::streamed_grpo_preflight_host_bytes(
        groups,
        completions,
        max_row_bytes,
        model_num_layers,
        config.reward_filter_var_min.is_some() || config.reward_filter_var_max.is_some(),
    )
    .map_err(|error| {
        ApiError::training_invalid_request(format!(
            "GRPO dataset_path '{dataset_path}' exceeds the streamed preflight host-memory contract: {error:#}"
        ))
    })?;
    let admission_weight = bytes_read
        .checked_add(preflight_host_bytes)
        .ok_or_else(|| ApiError::training_invalid_request("GRPO admission weight overflow"))?;
    prepared_data_permit
        .grow_to(admission_weight)
        .map_err(|(current, requested)| {
            ApiError::training_prepared_data_full(current, requested)
        })?;
    let digest: [u8; 32] = hasher.finalize().into();
    let source_sha256 = kiln_train::train_receipt::format_sha256_digest(&digest);
    let receipt = crate::training_queue::GrpoJsonlAdmissionReceipt::new_server_owned(
        snapshot_path,
        snapshot_reader,
        source_sha256,
        bytes_read,
        groups,
        completions,
        max_seq_len,
        preflight_host_bytes,
    )
    .map_err(ApiError::internal)?;
    cleanup_incomplete.armed = false;
    Ok(GrpoSubmissionStats {
        num_groups: Some(groups),
        total_completions: Some(completions),
        max_seq_len,
        streaming_dataset: true,
        source_receipt: Some(receipt),
    })
}

/// Estimate the per-step working set against the corrected memory
/// budget, and reject the submission with HTTP 413 if it cannot fit.
///
/// `max_seq_len` is approximated upstream by the request-specific
/// helper (`approximate_max_seq_len_sft` / `_grpo`) so this helper
/// stays SFT/GRPO-agnostic.
/// Validate a training submission fits in VRAM AND return the estimated per-step
/// working-set bytes (so the caller can stash it on the queue entry and hold a
/// governor reservation across the job — #24), plus the resolved dynamic
/// checkpoint segment count. CPU-only execution has no accelerator reservation;
/// an accelerated backend without a trustworthy live memory signal is rejected.
fn enforce_training_preflight(
    state: &AppState,
    max_seq_len: usize,
    mut options: EstimateOptions,
    lora_rank: usize,
    vk_native_recompute: bool,
) -> Result<PreflightAdmission, ApiError> {
    enforce_model_lora_rank_admission(state, lora_rank)?;
    // Vulkan's registry owns mirror buffers even on a dGPU; ROCm aliases
    // storage even on an APU. Runtime backend identity is the only safe key.
    options.lora_residency = match state.backend.as_ref() {
        ModelBackend::Mock { .. } => LoraResidency::StorageOwned,
        ModelBackend::Real { runner, .. } => runner
            .read()
            .map(|runner| LoraResidency::for_backend_name(runner.backend_name()))
            .unwrap_or_else(|_| LoraResidency::for_backend_name("unknown")),
    };
    if max_seq_len == 0 {
        return Ok(PreflightAdmission {
            reserved_bytes: 0,
            checkpoint_segments: None,
        });
    }
    let activation_estimate = training_activation_estimate_for_state(state, max_seq_len);
    if options.activation_bytes_per_elem.is_none() {
        options.activation_bytes_per_elem = Some(activation_estimate.bytes_per_elem);
    }
    if options.streaming_gdn_tile_tokens.is_none() {
        options.streaming_gdn_tile_tokens = activation_estimate.streaming_gdn_tile_tokens;
    }
    let vram = state.vram_info;
    if state.vram_probe_selector == kiln_memory::vram::VramProbeSelector::None {
        // CPU training does not allocate accelerator memory. Its allocations
        // are governed by the host and do not belong in the VRAM governor.
        return Ok(PreflightAdmission {
            reserved_bytes: 0,
            checkpoint_segments: None,
        });
    }

    let governor = kiln_memory::MemoryGovernor::global();
    // HTTP admission consumes the sampler's last published observation. A
    // driver/sysfs refresh can stall and therefore must not run on an async
    // request thread.
    let live_observation = governor.cached_observation();
    let live_snapshot = live_observation.snapshot;
    if vram.total_bytes == 0
        || live_snapshot.total_bytes == 0
        || !live_observation.sample_status.healthy
    {
        return Err(ApiError::training_will_not_fit(format!(
            "training memory preflight could not establish a safe live capacity for the selected accelerator ({:?}); check /v1/config memory diagnostics",
            state.vram_probe_selector
        )));
    }
    let live_policy_available = live_observation.available_bytes;
    let available = dynamic_training_availability(
        state,
        &vram,
        live_policy_available,
        live_observation.soft_reserved_bytes,
    );
    if available.bytes > live_policy_available {
        tracing::info!(
            live_available_gb = available.live_bytes as f64 / 1e9,
            effective_available_gb = available.bytes as f64 / 1e9,
            allocator_available_gb = available.allocator_bytes.map(|bytes| bytes as f64 / 1e9),
            reclaimable_kv_gb = available.reclaimable_kv_bytes as f64 / 1e9,
            "training preflight using dynamic memory availability"
        );
    }
    let checkpoint_policy = state.training_runtime.gradient_checkpoint_policy();
    let checkpoint_policy_is_fixed = !checkpoint_policy.is_auto();
    let configured_segments = if vk_native_recompute || checkpoint_policy_is_fixed {
        effective_checkpoint_segments(kiln_train::CheckpointConfig::from_runtime(
            state.model_config.num_layers,
            &state.training_runtime,
        ))
    } else {
        1
    };
    // Until the resident registry (Phase 1.2-1.4) lands, weights on
    // Vulkan APUs live in BOTH candle CPU storage and VulkanBuffer
    // caches — same physical RAM on unified memory. The estimator
    // must reflect that or the preflight will accept payloads that
    // ultimately exhaust the host. Once Phase 1.2 is deployed,
    // switch this to WeightResidency::SingleCopy.
    let residency = WeightResidency::for_vram(&vram);
    // Admission uses a live driver/allocator snapshot from the already-running
    // server, after model upload and KV allocation. Base weights are therefore
    // already deducted on every accelerator topology; counting them again
    // would reject CUDA/ROCm jobs roughly one model footprint too early.
    let weights_already_resident = true;
    let (num_segments, estimate) = if vk_native_recompute {
        (
            configured_segments,
            estimate_vk_native_recompute_working_set_with_residency(
                &state.model_config,
                max_seq_len,
                lora_rank,
                residency,
                weights_already_resident,
                options.optimizer,
                options.lora_residency,
            ),
        )
    } else if checkpoint_policy_is_fixed {
        (
            configured_segments,
            estimate_step_working_set_with_options(
                &state.model_config,
                max_seq_len,
                lora_rank,
                configured_segments,
                residency,
                weights_already_resident,
                options,
            ),
        )
    } else {
        auto_fit_checkpoint_segments(
            &state.model_config,
            max_seq_len,
            lora_rank,
            state.model_config.num_layers,
            residency,
            weights_already_resident,
            options,
            available.bytes,
        )
    };
    ensure_sft_checkpoint_plan_supported(options.sft, num_segments)?;
    let rank_ceiling = training_preflight::lora_rank_ceiling_for_budget(
        &state.model_config,
        options.optimizer,
        options.lora_residency,
        available.bytes,
        &estimate,
    );
    if estimate.breakdown.fixed_bytes() <= available.bytes && lora_rank > rank_ceiling.effective {
        return Err(ApiError::training_will_not_fit(format!(
            "LoRA rank {lora_rank} exceeds the live memory ceiling {} for optimizer {:?} with {:?} residency (model ceiling {}, resource ceiling {}, {} bytes per rank); lower lora_rank or free accelerator memory",
            rank_ceiling.effective,
            options.optimizer,
            options.lora_residency,
            rank_ceiling.model,
            rank_ceiling.resource,
            rank_ceiling.bytes_per_rank
        )));
    }
    if estimate.total_bytes > available.bytes {
        let msg = format_oom_message_with_source(
            &estimate,
            available.bytes,
            lora_rank,
            num_segments,
            Some(vram.source),
        );
        return Err(ApiError::training_will_not_fit(msg));
    }
    // Carry the exact live-admitted plan into the queued job. Replanning from
    // static startup capacity can select fewer segments and exceed both the
    // live fit decision and its reservation.
    let checkpoint_segments = Some(num_segments);
    let reserved_bytes = estimate.total_bytes;
    Ok(PreflightAdmission {
        reserved_bytes,
        checkpoint_segments,
    })
}

fn validate_grpo_submission_source(
    req: &GrpoRequest,
    tokenizer: Option<&kiln_core::tokenizer::KilnTokenizer>,
) -> Result<(), ApiError> {
    let source_count = usize::from(!req.groups.is_empty())
        + usize::from(req.dataset_path.is_some())
        + usize::from(req.dataset.is_some());
    if source_count != 1 {
        return Err(ApiError::training_invalid_request(
            "GRPO request must use exactly one of groups, dataset_path, or dataset",
        ));
    }
    if req.dataset_split.is_some() && req.dataset.is_none() {
        return Err(ApiError::training_invalid_request(
            "GRPO dataset_split is valid only with a named dataset",
        ));
    }
    // Fail fast on loss compositions the kt-tape trainer cannot train
    // (ECHO env-CE with environment tokens, no_policy_loss, reserved OPD
    // slot) — the worker would otherwise dequeue a job guaranteed to die,
    // possibly hours later behind a long queue.
    let has_env_tokens = req.groups.iter().any(|g| {
        g.completions.iter().any(|c| {
            c.trajectory
                .iter()
                .any(|seg| seg.kind == kiln_train::trajectory::TurnKind::Observation)
        })
    });
    req.config
        .loss
        .validate_for_kt_tape(has_env_tokens)
        .map_err(ApiError::training_invalid_request)?;
    req.config
        .validate_policy_config()
        .map_err(ApiError::training_invalid_request)?;
    if req.config.behavior_policy == kiln_train::BehaviorPolicy::Recorded {
        for (group_idx, group) in req.groups.iter().enumerate() {
            for (completion_idx, completion) in group.completions.iter().enumerate() {
                if completion.provenance.is_none() {
                    return Err(ApiError::training_invalid_request(format!(
                        "GRPO group {group_idx} completion {completion_idx} is missing exact rollout provenance required by behavior_policy=recorded"
                    )));
                }
            }
            if let Some(tokenizer) = tokenizer {
                kiln_train::trainer::validate_grpo_group_policy_data(
                    group,
                    &req.config,
                    tokenizer,
                )
                .map_err(|error| {
                    ApiError::training_invalid_request(format!(
                        "GRPO group {group_idx} has invalid recorded behavior provenance: {error:#}"
                    ))
                })?;
            }
        }
    }
    Ok(())
}

/// Response for queue listing.
#[derive(serde::Serialize)]
struct QueueResponse {
    /// Currently running job (if any).
    running: Option<TrainingStatus>,
    /// Jobs waiting in the queue.
    queued: Vec<QueueStatusEntry>,
    /// Recently completed/failed jobs.
    completed: Vec<TrainingStatus>,
}

#[derive(serde::Serialize)]
struct QueueStatusEntry {
    job_id: String,
    job_type: TrainingJobType,
    adapter_name: String,
    position: usize,
}

/// Submit-time guard for the trainer's LoRA alpha/rank safety gate.
///
/// The trainer enforces this anyway — but only after the job has been
/// accepted, queued, and run, so the caller sees a 200 and a doomed job
/// instead of a 400. The dashboard's corrections train shipped a
/// rank-8/alpha-32 (ratio 4.0) config that failed every job this way
/// while the UI had already marked the basket trained. Same message as
/// the trainer's so the two surfaces never disagree.
pub(crate) fn validate_lora_scale_at_submit(
    lora_rank: usize,
    lora_alpha: f32,
    allow_high_lora_scale: bool,
) -> Result<(), ApiError> {
    kiln_train::lora_scaling::validate_lora_scaling(lora_rank, lora_alpha, allow_high_lora_scale)
        .map(|_| ())
        .map_err(|e| ApiError::training_invalid_request(format!("{e:#}")))
}

fn normalize_sft_config_at_submit(config: &mut kiln_train::SftConfig) -> Result<(), ApiError> {
    config
        .validate_native_contract()
        .map_err(|error| ApiError::training_invalid_request(format!("{error:#}")))?;
    crate::training_queue::normalize_server_sft_mtp_policy(config)
        .map_err(ApiError::training_invalid_request)
}

fn validate_opd_loss_at_submit(loss: kiln_train::OpdLossGranularity) -> Result<(), ApiError> {
    if matches!(loss, kiln_train::OpdLossGranularity::FullVocab) {
        return Err(ApiError::training_invalid_request(
            "OPD loss is unavailable: full_vocab has no concrete server-built teacher source; use teacher_top_k"
                .to_string(),
        ));
    }
    match loss.unsupported_reason() {
        Some(reason) => Err(ApiError::training_invalid_request(format!(
            "OPD loss is unavailable: {reason}"
        ))),
        None => Ok(()),
    }
}

fn validate_opd_config_at_submit(config: &kiln_train::OpdConfig) -> Result<(), ApiError> {
    validate_opd_loss_at_submit(config.loss)?;
    config
        .validate_runtime_contract()
        .map_err(|error| ApiError::training_invalid_request(format!("{error:#}")))
}

fn require_off_policy_fixture_mode(
    surface: &str,
    config: &kiln_train::OpdConfig,
) -> Result<(), ApiError> {
    if !matches!(
        config.training_mode,
        kiln_train::opd::OpdTrainingMode::OffPolicy
    ) {
        return Err(ApiError::training_invalid_request(format!(
            "{surface} materializes teacher logits for fixed action sequences and requires config.training_mode=\"off_policy\""
        )));
    }
    Ok(())
}

fn require_remote_teacher_off_policy(
    surface: &str,
    spec: &super::teachers::TeacherSpec,
    config: &kiln_train::OpdConfig,
) -> Result<(), ApiError> {
    if matches!(spec.kind, super::teachers::TeacherKind::Remote) {
        require_off_policy_fixture_mode(surface, config).map_err(|_| {
            ApiError::training_invalid_request(format!(
                "{surface} cannot use a remote teacher with training_mode=\"on_policy\": remote logits must be prefetched before GPU coordination; use training_mode=\"off_policy\" with fixed assistant actions"
            ))
        })?;
    }
    Ok(())
}

fn registered_teacher_top_k_limit(
    spec: &super::teachers::TeacherSpec,
    requested_top_k: usize,
) -> usize {
    let configured = spec.max_top_k.unwrap_or(0);
    let provider_limit =
        if matches!(spec.kind, super::teachers::TeacherKind::Remote) && configured == 0 {
            kiln_train::RemoteProvider::Vllm.default_max_top_k()
        } else if configured == 0 {
            requested_top_k
        } else {
            configured
        };
    spec.vocab_size
        .map_or(provider_limit, |vocab_size| provider_limit.min(vocab_size))
}

/// Resolve the user-facing OPD K to a value implemented by the active KT
/// kernels before any job or teacher request is admitted.
fn resolve_opd_top_k_at_submit(
    config: &mut kiln_train::OpdConfig,
    source_max_top_k: usize,
) -> Result<Option<(usize, usize)>, ApiError> {
    let requested = config.top_k;
    let effective = kiln_train::resolve_opd_top_k(requested, source_max_top_k).map_err(|error| {
        ApiError::training_invalid_request(format!(
            "OPD top_k {requested} is not executable with source cap {source_max_top_k}: {error:#}"
        ))
    })?;
    config.top_k = effective;
    Ok((effective != requested).then_some((requested, effective)))
}

fn top_k_adjustment_suffix(adjustment: Option<(usize, usize)>) -> String {
    adjustment.map_or_else(String::new, |(requested, effective)| {
        format!(
            " Requested top_k {requested} was resolved to effective top_k {effective} for the teacher and active OPD kernel."
        )
    })
}

fn validate_self_distill_context_at_submit(
    req: &kiln_train::DistillSelfRequest,
) -> Result<(), ApiError> {
    let require_prompts = || {
        req.prompts.as_deref().filter(|prompts| !prompts.is_empty()).ok_or_else(|| {
            ApiError::training_invalid_request(
                "distill/self: this privileged-information mode requires explicit non-empty `prompts`"
                    .to_string(),
            )
        })
    };
    match req.mode {
        kiln_train::SelfDistillMode::GroundTruthConditioning => {
            let prompts = require_prompts()?;
            let answers = req.ground_truth.as_deref().ok_or_else(|| {
                ApiError::training_invalid_request(
                    "distill/self: ground_truth_conditioning requires `ground_truth`".to_string(),
                )
            })?;
            if answers.len() != prompts.len()
                || answers.iter().any(|answer| answer.trim().is_empty())
            {
                return Err(ApiError::training_invalid_request(format!(
                    "distill/self: ground_truth must contain one non-empty answer per prompt ({} prompts, {} answers)",
                    prompts.len(),
                    answers.len()
                )));
            }
        }
        kiln_train::SelfDistillMode::DocumentAsPi => {
            let prompts = require_prompts()?;
            let documents = req.documents.as_deref().ok_or_else(|| {
                ApiError::training_invalid_request(
                    "distill/self: document_as_pi requires `documents`".to_string(),
                )
            })?;
            if documents.len() != prompts.len()
                || documents.iter().any(|document| document.trim().is_empty())
            {
                return Err(ApiError::training_invalid_request(format!(
                    "distill/self: documents must contain one non-empty context per prompt ({} prompts, {} documents)",
                    prompts.len(),
                    documents.len()
                )));
            }
        }
        kiln_train::SelfDistillMode::Conciseness | kiln_train::SelfDistillMode::ReverseTeacher => {}
    }
    Ok(())
}

fn opd_prompt_has_action(prompt: &kiln_train::opd::OpdPrompt) -> bool {
    prompt
        .messages
        .iter()
        .any(|message| message.role == "assistant" && !message.content.trim().is_empty())
        || prompt.trajectory.iter().any(|segment| {
            matches!(segment.kind, kiln_train::trajectory::TurnKind::Action)
                && !segment.content.trim().is_empty()
        })
}

fn validate_opd_prompts_at_submit(
    surface: &str,
    prompts: &[kiln_train::opd::OpdPrompt],
    require_action: bool,
) -> Result<(), ApiError> {
    if prompts.is_empty() {
        return Err(ApiError::training_invalid_request(format!(
            "{surface} requires at least one prompt"
        )));
    }
    for (prompt_idx, prompt) in prompts.iter().enumerate() {
        if prompt.messages.is_empty() {
            return Err(ApiError::training_invalid_request(format!(
                "{surface} prompt {prompt_idx} has no messages"
            )));
        }
        if require_action && !opd_prompt_has_action(prompt) {
            return Err(ApiError::training_invalid_request(format!(
                "{surface} prompt {prompt_idx} requires a non-empty assistant action for off-policy scoring"
            )));
        }
    }
    Ok(())
}

fn validate_opd_request_at_submit(req: &kiln_train::OpdRequest) -> Result<(), ApiError> {
    if req.prompts.is_empty() == req.dataset_path.is_none() {
        return Err(ApiError::training_invalid_request(
            "OPD request must use exactly one of non-empty prompts or dataset_path".to_string(),
        ));
    }
    if req.teacher.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "OPD request must specify a teacher alias".to_string(),
        ));
    }
    if let Some(path) = req.dataset_path.as_deref() {
        if path.trim().is_empty() {
            return Err(ApiError::training_invalid_request(
                "OPD dataset_path must be non-empty".to_string(),
            ));
        }
        if !crate::dataset_resolve::is_agent_traces_selector(path)
            && !matches!(
                req.config.training_mode,
                kiln_train::opd::OpdTrainingMode::OffPolicy
            )
        {
            return Err(ApiError::training_invalid_request(
                "OPD teacher-logprob JSONL requires config.training_mode=\"off_policy\"; for on-policy training on pi sessions use an `agent_traces:` selector"
                    .to_string(),
            ));
        }
    } else {
        let require_action = matches!(
            req.config.training_mode,
            kiln_train::opd::OpdTrainingMode::OffPolicy
        );
        validate_opd_prompts_at_submit("OPD", &req.prompts, require_action)?;
    }
    validate_opd_config_at_submit(&req.config)
}

fn validate_distill_merge_at_submit(
    state: &AppState,
    req: &kiln_train::DistillMergeRequest,
) -> Result<(), ApiError> {
    if req.name.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "distill_merge: name must be non-empty".to_string(),
        ));
    }
    super::adapters::validate_adapter_name(&req.name)?;
    if req.sources.is_empty() {
        return Err(ApiError::training_invalid_request(
            "distill_merge: sources must be non-empty".to_string(),
        ));
    }
    if req.rollout_budget == 0 {
        return Err(ApiError::training_invalid_request(
            "distill_merge: rollout_budget must be greater than zero".to_string(),
        ));
    }
    validate_opd_config_at_submit(&req.config)?;
    require_off_policy_fixture_mode("distill_merge", &req.config)?;
    validate_lora_scale_at_submit(
        req.config.lora_rank,
        req.config.lora_alpha,
        req.config.allow_high_lora_scale,
    )?;
    let mut source_names = std::collections::HashSet::with_capacity(req.sources.len());
    for source in &req.sources {
        super::adapters::validate_adapter_name(&source.adapter)?;
        if !source_names.insert(source.adapter.as_str()) {
            return Err(ApiError::training_invalid_request(format!(
                "distill_merge: duplicate source adapter {:?}",
                source.adapter
            )));
        }
        if !source.weight.is_finite() || source.weight <= 0.0 {
            return Err(ApiError::training_invalid_request(format!(
                "distill_merge: source adapter {:?} weight must be finite and greater than zero",
                source.adapter
            )));
        }
        let dir = state.adapter_dir.join(&source.adapter);
        if !dir.is_dir() {
            return Err(ApiError::training_invalid_request(format!(
                "distill_merge: source adapter {:?} not found at {}",
                source.adapter,
                dir.display()
            )));
        }
    }
    Ok(())
}

fn validate_distill_merge_sampling_contract(
    req: &kiln_train::DistillMergeRequest,
    materialized_counts: &[usize],
) -> Result<usize, ApiError> {
    if req.sources.len() != materialized_counts.len()
        || materialized_counts.is_empty()
        || materialized_counts.contains(&0)
    {
        return Err(ApiError::internal(
            "distill_merge sampling validation received mismatched materialized sources",
        ));
    }
    let prompt_count = materialized_counts
        .iter()
        .try_fold(0usize, |total, count| total.checked_add(*count))
        .ok_or_else(|| {
            ApiError::training_invalid_request("DistillMerge aggregate prompt count overflow")
        })?;
    let expected_rollout_budget = prompt_count.checked_mul(req.config.epochs).ok_or_else(|| {
        ApiError::training_invalid_request("DistillMerge prompt count times epochs overflows")
    })?;
    let weight_sum = req.sources.iter().map(|source| source.weight).sum::<f64>();
    let weights_match_materialized_mixture = weight_sum.is_finite()
        && weight_sum > 0.0
        && req
            .sources
            .iter()
            .zip(materialized_counts)
            .all(|(source, count)| {
                let declared = source.weight / weight_sum;
                let materialized = *count as f64 / prompt_count as f64;
                (declared - materialized).abs() <= 1e-12
            });
    if req.rollout_budget != expected_rollout_budget || !weights_match_materialized_mixture {
        return Err(ApiError::training_invalid_request(format!(
            "distill_merge cannot silently approximate weighted sampling: this materialized corpus executes {expected_rollout_budget} prompt-epochs and source shares proportional to replay row counts; set rollout_budget={expected_rollout_budget} and each source weight proportional to its replay row count, or pre-sample the replay logs accordingly"
        )));
    }
    Ok(expected_rollout_budget)
}

fn distill_merge_prompt_sequence_sha256(
    prompt: &kiln_train::opd::OpdPrompt,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
) -> Result<[u8; 32], ApiError> {
    use sha2::{Digest, Sha256};

    let example = kiln_train::SftExample {
        messages: prompt.messages.clone(),
    };
    let (tokens, _) =
        kiln_train::trainer::tokenize_for_training(&example, tokenizer).map_err(|error| {
            ApiError::training_invalid_request(format!(
                "distill_merge prompt failed canonical training tokenization: {error:#}"
            ))
        })?;
    let mut digest = Sha256::new();
    digest.update((tokens.len() as u64).to_be_bytes());
    for token in tokens {
        digest.update(token.to_be_bytes());
    }
    let output = digest.finalize();
    let mut key = [0u8; 32];
    key.copy_from_slice(&output);
    Ok(key)
}

fn validate_distill_self_at_submit(req: &kiln_train::DistillSelfRequest) -> Result<(), ApiError> {
    if req.name.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "distill/self: name must be non-empty".to_string(),
        ));
    }
    super::adapters::validate_adapter_name(&req.name)?;
    if matches!(req.mode, kiln_train::SelfDistillMode::ReverseTeacher) {
        return Err(ApiError::training_invalid_request(
            "distill/self: reverse_teacher requires a distinct reverse objective".to_string(),
        ));
    }
    validate_opd_config_at_submit(&req.config)?;
    require_off_policy_fixture_mode("distill/self", &req.config)?;
    validate_lora_scale_at_submit(
        req.config.lora_rank,
        req.config.lora_alpha,
        req.config.allow_high_lora_scale,
    )?;
    let prompts = req.prompts.as_deref().ok_or_else(|| {
        ApiError::training_invalid_request(
            "distill/self requires explicit off-policy prompts with assistant actions".to_string(),
        )
    })?;
    validate_opd_prompts_at_submit("distill/self", prompts, true)?;
    validate_self_distill_context_at_submit(req)
}

fn validate_distill_pump_at_submit(req: &kiln_train::DistillPumpRequest) -> Result<(), ApiError> {
    if req.name.trim().is_empty() || req.teacher.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "distill/pump requires non-empty name and teacher".to_string(),
        ));
    }
    super::adapters::validate_adapter_name(&req.name)?;
    validate_opd_config_at_submit(&req.config)?;
    let off_policy = matches!(
        req.config.training_mode,
        kiln_train::opd::OpdTrainingMode::OffPolicy
    );
    match &req.mode {
        kiln_train::DistillPumpMode::Examples { examples } => {
            validate_opd_prompts_at_submit("distill/pump", examples, off_policy)?;
        }
        _ if off_policy => {
            return Err(ApiError::training_invalid_request(
                    "distill/pump domain and wide seed modes require on_policy training; off_policy requires explicit examples with assistant actions"
                        .to_string(),
                ));
        }
        _ => {}
    }
    Ok(())
}

fn validate_distill_refresh_at_submit(
    req: &kiln_train::DistillRefreshRequest,
) -> Result<(), ApiError> {
    if req.name.trim().is_empty() || req.behavioural_teacher.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "distill/refresh requires non-empty name and behavioural_teacher".to_string(),
        ));
    }
    validate_opd_config_at_submit(&req.config)?;
    if let kiln_train::NewKnowledgeSource::Inline { examples } = &req.new_data {
        // The first phase is SFT, so every inline example needs an assistant
        // target even when the recovery phase samples on-policy.
        validate_opd_prompts_at_submit("distill/refresh", examples, true)?;
    }
    let automatic_gate_count = usize::from(
        req.post_eval
            .as_ref()
            .is_some_and(|post_eval| post_eval.min_accuracy.is_some()),
    ) + usize::from(req.if_eval_suite.is_some())
        + usize::from(req.new_knowledge_eval_suite.is_some());
    if req.config.auto_load && automatic_gate_count > 1 {
        return Err(ApiError::training_invalid_request(format!(
            "distill/refresh automatic promotion accepts one versioned held-out suite, but {automatic_gate_count} gated suites were configured; compose the required domains into one suite or set config.auto_load=false and review the independent diagnostics"
        )));
    }
    Ok(())
}

/// Normalize every OPD-class queue payload, including recipes, scheduled
/// self-improvement, and the intent-aware front door that bypass dedicated
/// endpoint handlers.
pub(crate) fn normalize_queued_opd_top_k(
    state: &AppState,
    job: &mut QueuedJob,
) -> Result<Option<(usize, usize)>, ApiError> {
    match job {
        QueuedJob::Opd(req) => {
            validate_opd_request_at_submit(req)?;
            let spec = super::teachers::require_registered_teacher(
                state,
                &req.teacher,
                format!("OPD teacher alias '{}' is not registered", req.teacher),
            )?;
            let prescored = req
                .dataset_path
                .as_deref()
                .is_some_and(|path| !crate::dataset_resolve::is_agent_traces_selector(path));
            require_remote_teacher_off_policy("OPD", &spec, &req.config)?;
            let source_limit = if prescored {
                req.config.top_k
            } else {
                registered_teacher_top_k_limit(&spec, req.config.top_k)
            };
            resolve_opd_top_k_at_submit(&mut req.config, source_limit)
        }
        QueuedJob::DistillRefresh(req) => {
            validate_distill_refresh_at_submit(req)?;
            let spec = super::teachers::require_registered_teacher(
                state,
                &req.behavioural_teacher,
                format!(
                    "DistillRefresh: behavioural_teacher alias '{}' is not registered",
                    req.behavioural_teacher
                ),
            )?;
            require_remote_teacher_off_policy("DistillRefresh", &spec, &req.config)?;
            let source_limit = registered_teacher_top_k_limit(&spec, req.config.top_k);
            resolve_opd_top_k_at_submit(&mut req.config, source_limit)
        }
        QueuedJob::DistillPump(req) => {
            validate_distill_pump_at_submit(req)?;
            let spec = super::teachers::require_registered_teacher(
                state,
                &req.teacher,
                format!(
                    "distill/pump: teacher alias '{}' is not registered",
                    req.teacher
                ),
            )?;
            require_remote_teacher_off_policy("distill/pump", &spec, &req.config)?;
            let source_limit = registered_teacher_top_k_limit(&spec, req.config.top_k);
            resolve_opd_top_k_at_submit(&mut req.config, source_limit)
        }
        QueuedJob::DistillMerge(req) => {
            validate_distill_merge_at_submit(state, req)?;
            let requested = req.config.top_k;
            resolve_opd_top_k_at_submit(&mut req.config, requested)
        }
        QueuedJob::DistillSelf(req) => {
            validate_distill_self_at_submit(req)?;
            let requested = req.config.top_k;
            resolve_opd_top_k_at_submit(&mut req.config, requested)
        }
        QueuedJob::Sft(_) | QueuedJob::Grpo(_) => Ok(None),
    }
}

async fn submit_sft(
    State(state): State<AppState>,
    payload: Result<Json<SftRequest>, JsonRejection>,
) -> Result<Json<TrainingResponse>, ApiError> {
    let req = parse_training_json(payload, "SFT request")?;
    Ok(Json(admit_sft_request(state, req).await?))
}

pub(crate) async fn admit_sft_request(
    state: AppState,
    mut req: SftRequest,
) -> Result<TrainingResponse, ApiError> {
    ensure_training_backend_admission(&state)?;
    // Reject new jobs during shutdown
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }

    validate_post_eval_suite(&state, req.post_eval.as_ref())?;

    // Reject when the queue is at its configured cap. This protects the
    // server from queue-exhaustion DoS where a client submits jobs faster
    // than the trainer can drain them. Audit reference: security-audit-v0.1
    // §4 part 1.
    let max_queued = state.max_queued_training_jobs;
    let queued_now = state.training_queue.lock().unwrap().len();
    if queued_now >= max_queued {
        return Err(ApiError::training_queue_full(max_queued));
    }

    // Reject when the tracking map is at its configured cap. The training
    // worker GC's terminal entries on a TTL, but a flood of `Completed` /
    // `Failed` entries could still pin the map and exhaust memory. Audit
    // reference: security-audit-v0.1 §4 part 2.
    let max_tracked = state.max_tracked_jobs;
    let tracked_now = state.training_jobs.read().unwrap().len();
    if tracked_now >= max_tracked {
        return Err(ApiError::training_tracked_full(max_tracked));
    }

    if let Some(path) = req.dataset_path.take() {
        let path = path.trim().to_string();
        if !path.is_empty() {
            req.dataset_path = Some(path);
        }
    }
    let source_count = usize::from(!req.examples.is_empty())
        + usize::from(req.dataset_path.is_some())
        + usize::from(req.dataset.is_some());
    if source_count != 1 {
        return Err(ApiError::training_invalid_request(
            "SFT request must use exactly one of examples, dataset_path, or dataset",
        ));
    }
    if req.dataset_split.is_some()
        && req
            .dataset
            .as_deref()
            .is_none_or(|dataset| dataset == "corrections:active")
    {
        return Err(ApiError::training_invalid_request(
            "SFT dataset_split is valid only with a registered named dataset",
        ));
    }
    if let Some(name) = req.config.output_name.as_deref() {
        super::adapters::validate_adapter_name(name)?;
    }
    normalize_sft_config_at_submit(&mut req.config)?;

    let job_id = uuid::Uuid::new_v4().to_string();
    let adapter_name = req
        .config
        .output_name
        .clone()
        .unwrap_or_else(|| format!("sft-{}", &job_id[..8]));
    let auto_load = req.config.auto_load;

    let invalid_row_policy = req.config.invalid_row_policy;
    let training_profile = req.config.training_profile;

    // Register the job in the tracking map
    let info = TrainingJobInfo {
        job_id: job_id.clone(),
        adapter_name: adapter_name.clone(),
        job_type: TrainingJobType::Sft,
        effective_seed: None,
        state: TrainingState::Queued,
        progress: 0.0,
        loss: None,
        epoch: None,
        adapter_path: None,
        submitted_at: std::time::Instant::now(),
        submitted_unix_ms: crate::recent_requests::now_unix_ms(),
        auto_load,
        consumed_correction_ids: Vec::new(),
        training_data: None,
        finished_at: None,
        finished_unix_ms: None,
        error: None,
        linked_eval_job_ids: Vec::new(),
        post_eval_verdict: None,
        gate_outcome: None,
        post_eval_gate_evidence: Vec::new(),
        loss_history: Vec::new(),
        cancel_requested: Default::default(),
    };
    // Enqueue and publish the tracking record under one admission lock pair.
    let mut admission = admit_training_jobs_with_summary(
        &state,
        vec![(
            info,
            QueueEntry {
                job_id: job_id.clone(),
                reserved_bytes: 0,
                teacher_bindings: Vec::new(),
                admitted_resume_checkpoint: None,
                prepared_data: Default::default(),
                prepared_data_permit: Default::default(),
                job: QueuedJob::Sft(req),
            },
        )],
    )?;
    let stats = admission.sft_summaries.remove(&job_id).ok_or_else(|| {
        ApiError::internal("SFT admission completed without an exact corpus summary")
    })?;
    let effective_seed = admission.effective_seed(&job_id)?;
    let queue_position = admission.queue_position;
    let num_examples = stats.num_examples;

    tracing::info!(
        num_examples,
        training_profile = %training_profile,
        rows_read = stats.rows_read,
        rows_rejected = stats.rows_rejected,
        invalid_row_policy = %invalid_row_policy,
        job_id = %job_id,
        adapter = %adapter_name,
        max_seq_len = stats.max_seq_len,
        max_supervised_tokens = stats.max_supervised_tokens,
        streaming_dataset = stats.streaming_dataset,
        "SFT training request queued after bounded corpus admission"
    );

    Ok(TrainingResponse {
        job_id,
        state: TrainingState::Queued,
        effective_seed: effective_seed.to_string(),
        message: format!(
            "Queued {} SFT with {num_examples} kept examples and {} rejected rows \
             under {} policy (position {queue_position} in queue)",
            training_profile, stats.rows_rejected, invalid_row_policy,
        ),
    })
}

async fn submit_grpo(
    State(state): State<AppState>,
    payload: Result<Json<GrpoRequest>, JsonRejection>,
) -> Result<Json<TrainingResponse>, ApiError> {
    let mut req = parse_training_json(payload, "GRPO request")?;
    ensure_training_backend_admission(&state)?;
    // Reject new jobs during shutdown
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }

    validate_post_eval_suite(&state, req.post_eval.as_ref())?;

    // Reject when the queue is at its configured cap. See submit_sft above
    // for the audit reference.
    let max_queued = state.max_queued_training_jobs;
    let queued_now = state.training_queue.lock().unwrap().len();
    if queued_now >= max_queued {
        return Err(ApiError::training_queue_full(max_queued));
    }

    // Reject when the tracking map is at its configured cap. See submit_sft
    // above for the audit reference (security-audit-v0.1 §4 part 2).
    let max_tracked = state.max_tracked_jobs;
    let tracked_now = state.training_jobs.read().unwrap().len();
    if tracked_now >= max_tracked {
        return Err(ApiError::training_tracked_full(max_tracked));
    }

    if let Some(path) = req.dataset_path.take() {
        let path = path.trim().to_string();
        if !path.is_empty() {
            req.dataset_path = Some(path);
        }
    }
    validate_grpo_submission_source(&req, Some(state.tokenizer.as_ref()))?;

    let stats = if req.dataset_path.is_some() || req.dataset.is_some() {
        // The authoritative queue admission performs the bounded full-corpus
        // scan while holding the process admission permit. Do not duplicate
        // that expensive work on the async handler before capacity is owned.
        GrpoSubmissionStats {
            num_groups: None,
            total_completions: None,
            max_seq_len: 0,
            streaming_dataset: true,
            source_receipt: None,
        }
    } else {
        GrpoSubmissionStats {
            num_groups: Some(req.groups.len()),
            total_completions: Some(req.groups.iter().map(|g| g.completions.len()).sum()),
            max_seq_len: training_preflight::approximate_max_seq_len_grpo(
                &req.groups,
                Some(state.tokenizer.as_ref()),
            ),
            streaming_dataset: false,
            source_receipt: None,
        }
    };
    let job_id = uuid::Uuid::new_v4().to_string();
    if let Some(name) = req.config.output_name.as_deref() {
        super::adapters::validate_adapter_name(name)?;
    }
    validate_lora_scale_at_submit(
        req.config.lora_rank,
        req.config.lora_alpha,
        req.config.allow_high_lora_scale,
    )?;
    if req.config.checkpoint_interval == Some(0) {
        return Err(ApiError::training_invalid_request(
            "GRPO checkpoint_interval must be greater than zero",
        ));
    }
    let adapter_name = req
        .config
        .output_name
        .clone()
        .unwrap_or_else(|| format!("grpo-{}", &job_id[..8]));
    let auto_load = req.config.auto_load;

    if stats.streaming_dataset {
        tracing::info!(
            dataset_path = req.dataset_path.as_deref().unwrap_or_default(),
            dataset = req.dataset.as_deref().unwrap_or_default(),
            dataset_split = req.dataset_split.unwrap_or_default().as_str(),
            job_id = %job_id,
            adapter = %adapter_name,
            "streamed GRPO training request entering bounded full-corpus admission"
        );
    } else {
        tracing::info!(
            num_groups = stats.num_groups.unwrap_or(0),
            total_completions = stats.total_completions.unwrap_or(0),
            job_id = %job_id,
            adapter = %adapter_name,
            "GRPO training request queued"
        );
    }

    // Register the job in the tracking map
    let info = TrainingJobInfo {
        job_id: job_id.clone(),
        adapter_name: adapter_name.clone(),
        job_type: TrainingJobType::Grpo,
        effective_seed: None,
        state: TrainingState::Queued,
        progress: 0.0,
        loss: None,
        epoch: None,
        adapter_path: None,
        submitted_at: std::time::Instant::now(),
        submitted_unix_ms: crate::recent_requests::now_unix_ms(),
        auto_load,
        consumed_correction_ids: Vec::new(),
        training_data: None,
        finished_at: None,
        finished_unix_ms: None,
        error: None,
        linked_eval_job_ids: Vec::new(),
        post_eval_verdict: None,
        gate_outcome: None,
        post_eval_gate_evidence: Vec::new(),
        loss_history: Vec::new(),
        cancel_requested: Default::default(),
    };
    // Enqueue and publish the tracking record under one admission lock pair.
    let admission = admit_training_jobs_with_summary(
        &state,
        vec![(
            info,
            QueueEntry {
                job_id: job_id.clone(),
                // The authoritative queue admission scans/materializes every
                // job source and overwrites this estimate before publication.
                reserved_bytes: 0,
                teacher_bindings: Vec::new(),
                admitted_resume_checkpoint: None,
                prepared_data: Default::default(),
                prepared_data_permit: Default::default(),
                job: QueuedJob::Grpo(req),
            },
        )],
    )?;
    let queue_position = admission.queue_position;
    let effective_seed = admission.effective_seed(&job_id)?;

    Ok(Json(TrainingResponse {
        job_id,
        state: TrainingState::Queued,
        effective_seed: effective_seed.to_string(),
        message: if stats.streaming_dataset {
            format!(
                "Queued streamed GRPO training from dataset_path (position {queue_position} in queue)"
            )
        } else {
            let num_groups = stats.num_groups.unwrap_or(0);
            let total_completions = stats.total_completions.unwrap_or(0);
            format!(
                "Queued GRPO training with {num_groups} groups ({total_completions} completions, position {queue_position} in queue)"
            )
        },
    }))
}

/// `POST /v1/train/opd` — submit an On-Policy Distillation training run.
///
/// Mirror of `submit_grpo` adapted to the §3.1 OPD recipe. The request
/// shape is `OpdRequest` (defined in `kiln-train::opd`): a list of
/// prompts, a teacher alias, and an `OpdConfig` whose §6 paper-cited
/// defaults select the bounded executable path (top_k=32, temperature=1.0,
/// top_p=0.9, max_tokens=7K, direct reverse-KL, Stable-OPD off).
///
/// Same queue / hot-swap / auto-load / post-eval semantics as SFT/GRPO.
/// Job tracking via `/v1/train/status`, `/v1/train/queue`, etc.
async fn submit_opd(
    State(state): State<AppState>,
    payload: Result<Json<OpdRequest>, JsonRejection>,
) -> Result<Json<TrainingResponse>, ApiError> {
    let mut req = parse_training_json(payload, "OPD request")?;
    ensure_training_backend_admission(&state)?;
    // Reject during shutdown.
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }

    validate_post_eval_suite(&state, req.post_eval.as_ref())?;

    // Queue / tracking caps — mirror SFT/GRPO.
    let max_queued = state.max_queued_training_jobs;
    let queued_now = state.training_queue.lock().unwrap().len();
    if queued_now >= max_queued {
        return Err(ApiError::training_queue_full(max_queued));
    }
    let max_tracked = state.max_tracked_jobs;
    let tracked_now = state.training_jobs.read().unwrap().len();
    if tracked_now >= max_tracked {
        return Err(ApiError::training_tracked_full(max_tracked));
    }

    // Trim a blank dataset_path the same way GRPO does.
    if let Some(path) = req.dataset_path.take() {
        let path = path.trim().to_string();
        if !path.is_empty() {
            req.dataset_path = Some(path);
        }
    }
    validate_opd_request_at_submit(&req)?;

    if req.prompts.is_empty() && req.dataset_path.is_none() {
        return Err(ApiError::training_invalid_request(
            "OPD request must include at least one prompt or a dataset_path".to_string(),
        ));
    }
    if req.dataset_path.is_some() && !req.prompts.is_empty() {
        return Err(ApiError::training_invalid_request(
            "OPD request must use either prompts or dataset_path, not both".to_string(),
        ));
    }
    if req.teacher.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "OPD request must specify a teacher alias (e.g. \"qwen3.6-27b@local\")".to_string(),
        ));
    }
    // A plain-file dataset_path is pre-scored off-policy teacher JSONL;
    // `agent_traces:` selectors are on-policy prompt sources. The worker
    // enforces this too — but at submission the caller can still fix it.
    if let Some(path) = req.dataset_path.as_deref() {
        if !crate::dataset_resolve::is_agent_traces_selector(path)
            && !matches!(
                req.config.training_mode,
                kiln_train::opd::OpdTrainingMode::OffPolicy
            )
        {
            return Err(ApiError::training_invalid_request(
                "OPD dataset_path (teacher-logprob JSONL) requires config.training_mode = \
                 \"off_policy\"; for on-policy training on pi sessions use an \
                 `agent_traces:` selector instead"
                    .to_string(),
            ));
        }
    }
    for (i, prompt) in req.prompts.iter().enumerate() {
        if prompt.messages.is_empty() {
            return Err(ApiError::training_invalid_request(format!(
                "OPD prompt {i} has no messages"
            )));
        }
    }
    if req.config.top_k == 0 {
        return Err(ApiError::training_invalid_request(
            "OPD top_k must be > 0".to_string(),
        ));
    }
    validate_opd_config_at_submit(&req.config)?;
    if req.config.samples_per_prompt == 0 {
        return Err(ApiError::training_invalid_request(
            "OPD samples_per_prompt must be > 0".to_string(),
        ));
    }

    let job_id = uuid::Uuid::new_v4().to_string();
    if let Some(name) = req.config.output_name.as_deref() {
        super::adapters::validate_adapter_name(name)?;
    }
    validate_lora_scale_at_submit(
        req.config.lora_rank,
        req.config.lora_alpha,
        req.config.allow_high_lora_scale,
    )?;
    // The worker resolves the teacher only at dequeue — a typo'd alias
    // used to enqueue a job guaranteed to fail later, possibly hours
    // later behind a long queue. Fail here with the remediation. (After
    // the pure-input checks above: a malformed request is the caller's
    // first problem, an unregistered teacher the second.)
    let teacher_spec = super::teachers::require_registered_teacher(
        &state,
        &req.teacher,
        format!("OPD teacher alias '{}' is not registered", req.teacher),
    )?;
    let uses_prescored_dataset = req
        .dataset_path
        .as_deref()
        .is_some_and(|path| !crate::dataset_resolve::is_agent_traces_selector(path));
    require_remote_teacher_off_policy("OPD", &teacher_spec, &req.config)?;
    let source_max_top_k = if uses_prescored_dataset {
        req.config.top_k
    } else {
        registered_teacher_top_k_limit(&teacher_spec, req.config.top_k)
    };
    let top_k_adjustment = resolve_opd_top_k_at_submit(&mut req.config, source_max_top_k)?;
    let adapter_name = req
        .config
        .output_name
        .clone()
        .unwrap_or_else(|| format!("opd-{}", &job_id[..8]));
    let auto_load = req.config.auto_load;

    tracing::info!(
        num_prompts = req.prompts.len(),
        teacher = %req.teacher,
        loss = ?req.config.loss,
        top_k = req.config.top_k,
        requested_top_k = top_k_adjustment.map(|(requested, _)| requested),
        samples_per_prompt = req.config.samples_per_prompt,
        job_id = %job_id,
        adapter = %adapter_name,
        "OPD training request queued"
    );

    let info = TrainingJobInfo {
        job_id: job_id.clone(),
        adapter_name: adapter_name.clone(),
        job_type: TrainingJobType::Opd,
        effective_seed: None,
        state: TrainingState::Queued,
        progress: 0.0,
        loss: None,
        epoch: None,
        adapter_path: None,
        submitted_at: std::time::Instant::now(),
        submitted_unix_ms: crate::recent_requests::now_unix_ms(),
        auto_load,
        consumed_correction_ids: Vec::new(),
        training_data: None,
        finished_at: None,
        finished_unix_ms: None,
        error: None,
        linked_eval_job_ids: Vec::new(),
        post_eval_verdict: None,
        gate_outcome: None,
        post_eval_gate_evidence: Vec::new(),
        loss_history: Vec::new(),
        cancel_requested: Default::default(),
    };
    let admission = admit_training_jobs_with_summary(
        &state,
        vec![(
            info,
            QueueEntry {
                job_id: job_id.clone(),
                reserved_bytes: 0,
                teacher_bindings: Vec::new(),
                admitted_resume_checkpoint: None,
                prepared_data: Default::default(),
                prepared_data_permit: Default::default(),
                job: QueuedJob::Opd(req),
            },
        )],
    )?;
    let queue_position = admission.queue_position;
    let effective_seed = admission.effective_seed(&job_id)?;

    Ok(Json(TrainingResponse {
        job_id,
        state: TrainingState::Queued,
        effective_seed: effective_seed.to_string(),
        message: format!(
            "Queued OPD training (position {queue_position} in queue).{}",
            top_k_adjustment_suffix(top_k_adjustment)
        ),
    }))
}

/// `POST /v1/distill/refresh` — §3.6 continual-learning recipe
/// (Lu 2025 instruction-following recovery experiment).
///
/// Body: [`DistillRefreshRequest`]. The runtime mid-trains on
/// `new_data` then OPD-recovers against the prior-self
/// `behavioural_teacher`, gated on dual eval (IF-eval recovery +
/// new-knowledge gain). Same queue / receipt / auto-load semantics
/// as `/v1/train/opd`. The route currently fails closed after cheap request
/// and teacher validation until exact two-phase admission is implemented.
async fn submit_distill_refresh(
    State(state): State<AppState>,
    payload: Result<Json<DistillRefreshRequest>, JsonRejection>,
) -> Result<Json<TrainingResponse>, ApiError> {
    let mut req = parse_training_json(payload, "distill/refresh request")?;
    ensure_training_backend_admission(&state)?;
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }

    validate_post_eval_suite(&state, req.post_eval.as_ref())?;
    validate_distill_refresh_at_submit(&req)?;
    let max_queued = state.max_queued_training_jobs;
    let queued_now = state.training_queue.lock().unwrap().len();
    if queued_now >= max_queued {
        return Err(ApiError::training_queue_full(max_queued));
    }
    let max_tracked = state.max_tracked_jobs;
    let tracked_now = state.training_jobs.read().unwrap().len();
    if tracked_now >= max_tracked {
        return Err(ApiError::training_tracked_full(max_tracked));
    }
    if req.name.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "DistillRefresh: `name` must be non-empty".to_string(),
        ));
    }
    super::adapters::validate_adapter_name(&req.name)?;
    validate_opd_config_at_submit(&req.config)?;
    validate_lora_scale_at_submit(
        req.config.lora_rank,
        req.config.lora_alpha,
        req.config.allow_high_lora_scale,
    )?;
    if req.behavioural_teacher.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "DistillRefresh: `behavioural_teacher` alias must be non-empty".to_string(),
        ));
    }
    let teacher_spec = super::teachers::require_registered_teacher(
        &state,
        &req.behavioural_teacher,
        format!(
            "DistillRefresh: behavioural_teacher alias '{}' is not registered",
            req.behavioural_teacher
        ),
    )?;
    require_remote_teacher_off_policy("DistillRefresh", &teacher_spec, &req.config)?;
    let source_max_top_k = registered_teacher_top_k_limit(&teacher_spec, req.config.top_k);
    let top_k_adjustment = resolve_opd_top_k_at_submit(&mut req.config, source_max_top_k)?;
    if !(0.0..=1.0).contains(&req.require_if_eval_recovery) {
        return Err(ApiError::training_invalid_request(
            "require_if_eval_recovery must be in [0.0, 1.0]".to_string(),
        ));
    }
    enforce_training_workload_admission(&state, TrainingWorkload::DistillRefresh)?;

    let job_id = uuid::Uuid::new_v4().to_string();
    let adapter_name = format!("{}@refresh-{}", req.name, &job_id[..8]);
    let auto_load = req.config.auto_load;

    tracing::info!(
        name = %req.name,
        behavioural_teacher = %req.behavioural_teacher,
        background_chat = %req.background_chat,
        require_if_eval_recovery = req.require_if_eval_recovery,
        require_internal_qa_gain = req.require_internal_qa_gain,
        job_id = %job_id,
        adapter = %adapter_name,
        "distill/refresh request queued"
    );

    let info = TrainingJobInfo {
        job_id: job_id.clone(),
        adapter_name: adapter_name.clone(),
        // Reuse the Opd job type — refresh is structurally an OPD run
        // with extra orchestration. Dashboards group both as OPD-class.
        job_type: TrainingJobType::Opd,
        effective_seed: None,
        state: TrainingState::Queued,
        progress: 0.0,
        loss: None,
        epoch: None,
        adapter_path: None,
        submitted_at: std::time::Instant::now(),
        submitted_unix_ms: crate::recent_requests::now_unix_ms(),
        auto_load,
        consumed_correction_ids: Vec::new(),
        training_data: None,
        finished_at: None,
        finished_unix_ms: None,
        error: None,
        linked_eval_job_ids: Vec::new(),
        post_eval_verdict: None,
        gate_outcome: None,
        post_eval_gate_evidence: Vec::new(),
        loss_history: Vec::new(),
        cancel_requested: Default::default(),
    };
    let admission = admit_training_jobs_with_summary(
        &state,
        vec![(
            info,
            QueueEntry {
                job_id: job_id.clone(),
                reserved_bytes: 0,
                teacher_bindings: Vec::new(),
                admitted_resume_checkpoint: None,
                prepared_data: Default::default(),
                prepared_data_permit: Default::default(),
                job: QueuedJob::DistillRefresh(req),
            },
        )],
    )?;
    let queue_position = admission.queue_position;
    let effective_seed = admission.effective_seed(&job_id)?;

    Ok(Json(TrainingResponse {
        job_id,
        state: TrainingState::Queued,
        effective_seed: effective_seed.to_string(),
        message: format!(
            "Queued distill/refresh (position {queue_position} in queue).{}",
            top_k_adjustment_suffix(top_k_adjustment)
        ),
    }))
}

/// `POST /v1/adapters/distill_merge` — §3.4 behaviour-space merge.
async fn submit_distill_merge(
    State(state): State<AppState>,
    payload: Result<Json<DistillMergeRequest>, JsonRejection>,
) -> Result<Json<TrainingResponse>, ApiError> {
    let mut req = parse_training_json(payload, "distill_merge request")?;
    ensure_training_backend_admission(&state)?;
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }

    validate_post_eval_suite(&state, req.post_eval.as_ref())?;
    validate_distill_merge_at_submit(&state, &req)?;
    if req.sources.is_empty() {
        return Err(ApiError::training_invalid_request(
            "distill_merge: `sources` must be non-empty".to_string(),
        ));
    }
    if req.name.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "distill_merge: `name` must be non-empty".to_string(),
        ));
    }
    super::adapters::validate_adapter_name(&req.name)?;
    validate_opd_config_at_submit(&req.config)?;
    require_off_policy_fixture_mode("distill_merge", &req.config)?;
    let requested_top_k = req.config.top_k;
    let top_k_adjustment = resolve_opd_top_k_at_submit(&mut req.config, requested_top_k)?;
    // A source adapter that doesn't exist on disk is a typo — fail now,
    // not after the job dequeues and silently falls back to the base
    // model for that source's prompts.
    for source in &req.sources {
        super::adapters::validate_adapter_name(&source.adapter)?;
        let dir = state.adapter_dir.join(&source.adapter);
        if !dir.is_dir() {
            return Err(ApiError::training_invalid_request(format!(
                "distill_merge: source adapter `{}` not found at {}",
                source.adapter,
                dir.display()
            )));
        }
    }
    enforce_queue_caps(&state)?;
    let job_id = uuid::Uuid::new_v4().to_string();
    let adapter_name = req.name.clone();
    let auto_load = req.config.auto_load;
    let effective_seed = register_and_enqueue_distill(
        &state,
        &job_id,
        &adapter_name,
        auto_load,
        QueuedJob::DistillMerge(req),
    )?;
    Ok(Json(TrainingResponse {
        job_id,
        state: TrainingState::Queued,
        effective_seed: effective_seed.to_string(),
        message: format!(
            "Queued distill_merge.{}",
            top_k_adjustment_suffix(top_k_adjustment)
        ),
    }))
}

/// `POST /v1/distill/pump` — §3.5 Knowledge Pump.
async fn submit_distill_pump(
    State(state): State<AppState>,
    payload: Result<Json<DistillPumpRequest>, JsonRejection>,
) -> Result<Json<TrainingResponse>, ApiError> {
    let mut req = parse_training_json(payload, "distill/pump request")?;
    ensure_training_backend_admission(&state)?;
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }

    validate_post_eval_suite(&state, req.post_eval.as_ref())?;
    validate_distill_pump_at_submit(&req)?;
    if req.teacher.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "distill/pump: `teacher` alias must be non-empty".to_string(),
        ));
    }
    let teacher_spec = super::teachers::require_registered_teacher(
        &state,
        &req.teacher,
        format!(
            "distill/pump: teacher alias '{}' is not registered",
            req.teacher
        ),
    )?;
    require_remote_teacher_off_policy("distill/pump", &teacher_spec, &req.config)?;
    super::adapters::validate_adapter_name(&req.name)?;
    validate_opd_config_at_submit(&req.config)?;
    // The worker overrides config.lora_rank with the request's top-level
    // `rank` when set (training_queue.rs pump arm) — validate the rank
    // that will actually train, not the config default it shadows.
    validate_lora_scale_at_submit(
        req.rank.unwrap_or(req.config.lora_rank),
        req.config.lora_alpha,
        req.config.allow_high_lora_scale,
    )?;
    let source_max_top_k = registered_teacher_top_k_limit(&teacher_spec, req.config.top_k);
    let top_k_adjustment = resolve_opd_top_k_at_submit(&mut req.config, source_max_top_k)?;
    enforce_queue_caps(&state)?;
    let job_id = uuid::Uuid::new_v4().to_string();
    let adapter_name = req.name.clone();
    let auto_load = req.config.auto_load;
    let effective_seed = register_and_enqueue_distill(
        &state,
        &job_id,
        &adapter_name,
        auto_load,
        QueuedJob::DistillPump(req),
    )?;
    Ok(Json(TrainingResponse {
        job_id,
        state: TrainingState::Queued,
        effective_seed: effective_seed.to_string(),
        message: format!(
            "Queued distill/pump.{}",
            top_k_adjustment_suffix(top_k_adjustment)
        ),
    }))
}

/// `POST /v1/distill/self` — §3.12 PI self-distillation.
async fn submit_distill_self(
    State(state): State<AppState>,
    payload: Result<Json<DistillSelfRequest>, JsonRejection>,
) -> Result<Json<TrainingResponse>, ApiError> {
    let mut req = parse_training_json(payload, "distill/self request")?;
    ensure_training_backend_admission(&state)?;
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }

    validate_post_eval_suite(&state, req.post_eval.as_ref())?;
    validate_distill_self_at_submit(&req)?;
    if req.name.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "distill/self: `name` must be non-empty".to_string(),
        ));
    }
    if matches!(req.mode, kiln_train::SelfDistillMode::ReverseTeacher) {
        return Err(ApiError::training_invalid_request(
            "distill/self: `reverse_teacher` is unsupported because it requires a distinct reverse objective; negated logprobs are invalid"
                .to_string(),
        ));
    }
    validate_self_distill_context_at_submit(&req)?;
    validate_opd_config_at_submit(&req.config)?;
    require_off_policy_fixture_mode("distill/self", &req.config)?;
    super::adapters::validate_adapter_name(&req.name)?;
    let requested_top_k = req.config.top_k;
    let top_k_adjustment = resolve_opd_top_k_at_submit(&mut req.config, requested_top_k)?;
    enforce_queue_caps(&state)?;
    let job_id = uuid::Uuid::new_v4().to_string();
    let adapter_name = req.name.clone();
    let auto_load = req.config.auto_load;
    let effective_seed = register_and_enqueue_distill(
        &state,
        &job_id,
        &adapter_name,
        auto_load,
        QueuedJob::DistillSelf(req),
    )?;
    Ok(Json(TrainingResponse {
        job_id,
        state: TrainingState::Queued,
        effective_seed: effective_seed.to_string(),
        message: format!(
            "Queued distill/self.{}",
            top_k_adjustment_suffix(top_k_adjustment)
        ),
    }))
}

/// Shared submission gate: queue cap, tracked-jobs cap, and mock-mode
/// rejection. Used by the distill_* endpoints and every other surface
/// that enqueues training jobs (front door, recipes, agent endpoints).
/// Submission-time §8.7 gate validation: a post_eval naming a suite that
/// isn't installed must reject NOW, not after training burns GPU-hours
/// and the eval worker discovers the name resolves to nothing (round-4
/// discovery: docs/tests pointed at "agentic-core" while the installed
/// builtin is "qwen3.5-agentic-core" — gated rounds trained forever and
/// never promoted).
pub(crate) fn validate_post_eval_suite(
    state: &AppState,
    post_eval: Option<&kiln_eval::PostEvalConfig>,
) -> Result<(), ApiError> {
    let Some(cfg) = post_eval else {
        return Ok(());
    };
    if let Some(min_accuracy) = cfg.min_accuracy
        && !(0.0..=1.0).contains(&min_accuracy)
    {
        return Err(ApiError::training_invalid_request(
            "post_eval.min_accuracy must be finite and in [0.0, 1.0]",
        ));
    }
    if cfg.data_scope == kiln_eval::PostEvalDataScope::TrainSetEval && cfg.min_accuracy.is_some() {
        return Err(ApiError::training_invalid_request(
            "post_eval.data_scope \"train-set-eval\" is diagnostic only and cannot set min_accuracy",
        ));
    }
    let Some(registry) = state.suite_registry.as_ref() else {
        // No registry on this state (mock/test shapes) — the eval worker
        // will surface the failure; don't block submission.
        return Ok(());
    };
    if registry.load(&cfg.suite).is_err() {
        let available: Vec<String> = registry.list().into_iter().map(|s| s.name).collect();
        return Err(ApiError::training_invalid_request(format!(
            "post_eval.suite '{}' is not an installed eval suite — available: [{}]",
            cfg.suite,
            available.join(", ")
        )));
    }
    Ok(())
}

fn post_eval_contamination_index(
    state: &AppState,
    post_eval: Option<&kiln_eval::PostEvalConfig>,
) -> Result<Option<kiln_eval::EvalContaminationIndex>, ApiError> {
    let Some(cfg) = post_eval else {
        return Ok(None);
    };
    if cfg.data_scope == kiln_eval::PostEvalDataScope::TrainSetEval {
        return Ok(None);
    }
    let Some(registry) = state.suite_registry.as_ref() else {
        // Minimal unit-test states may omit the registry. Production startup
        // always installs it, and validate_post_eval_suite owns that contract.
        return Ok(None);
    };
    let suite = registry.load(&cfg.suite).map_err(|error| {
        ApiError::training_invalid_request(format!(
            "load held-out post_eval suite {:?} during admission: {error}",
            cfg.suite
        ))
    })?;
    Ok(Some(kiln_eval::EvalContaminationIndex::from_suite(&suite)))
}

fn contamination_error(
    post_eval: &kiln_eval::PostEvalConfig,
    overlap: kiln_eval::ContaminationMatch,
) -> ApiError {
    ApiError::training_invalid_request(format!(
        "post_eval suite {:?} is declared held-out but overlaps admitted training data via {}; use a disjoint suite or explicitly set post_eval.data_scope to \"train-set-eval\" for a non-gating diagnostic",
        post_eval.suite,
        overlap.as_str()
    ))
}

fn check_prompt_variants(
    index: &kiln_eval::EvalContaminationIndex,
    messages: &[kiln_eval::EvalChatMessage],
    target: &str,
) -> Option<kiln_eval::ContaminationMatch> {
    let mut variants = Vec::with_capacity(4);
    variants.push(messages.to_vec());
    let without_system = messages
        .iter()
        .filter(|message| message.role != "system")
        .cloned()
        .collect::<Vec<_>>();
    variants.push(without_system);
    for prompt in variants.clone() {
        let mut trimmed = prompt;
        while trimmed.last().is_some_and(|message| message.role == "tool") {
            trimmed.pop();
        }
        variants.push(trimmed);
    }
    variants
        .iter()
        .find_map(|prompt| index.check_example(prompt, Some(target)))
}

fn check_sft_contamination(
    index: &kiln_eval::EvalContaminationIndex,
    examples: &[kiln_train::SftExample],
) -> Option<kiln_eval::ContaminationMatch> {
    for example in examples {
        for (assistant_index, assistant) in example.messages.iter().enumerate() {
            if assistant.role != "assistant" {
                continue;
            }
            let Some(target) = kiln_eval::assistant_target_text(assistant) else {
                continue;
            };
            if let Some(overlap) =
                check_prompt_variants(index, &example.messages[..assistant_index], &target)
            {
                return Some(overlap);
            }
        }
    }
    None
}

fn check_grpo_group_contamination(
    index: &kiln_eval::EvalContaminationIndex,
    group: &kiln_train::GrpoGroup,
) -> Option<kiln_eval::ContaminationMatch> {
    group
        .completions
        .iter()
        .find_map(|completion| check_prompt_variants(index, &group.messages, &completion.text))
}

fn check_split_contamination(
    index: &kiln_eval::EvalContaminationIndex,
    split: &kiln_eval::DatasetSplitManifest,
    selected: kiln_eval::DatasetSplit,
) -> Option<kiln_eval::ContaminationMatch> {
    split
        .rows
        .iter()
        .filter(|row| row.split == selected)
        .find_map(|row| index.check_source_row(row))
}

struct NamedDatasetAdmission {
    path: PathBuf,
    split: kiln_eval::DatasetSplit,
    manifest: crate::eval::DatasetManifest,
    split_manifest: kiln_eval::DatasetSplitManifest,
}

fn resolve_named_training_dataset(
    state: &AppState,
    name: &str,
    requested_split: Option<kiln_eval::DatasetSplit>,
    expected_format: crate::eval::DatasetFormat,
) -> Result<NamedDatasetAdmission, ApiError> {
    let registry = state
        .dataset_registry
        .as_ref()
        .ok_or_else(ApiError::dataset_registry_unavailable)?;
    let manifest = registry.load_manifest(name).map_err(|error| match error {
        crate::eval::DatasetError::NotFound(_) => ApiError::dataset_not_found(name),
        crate::eval::DatasetError::InvalidName(_) => ApiError::dataset_invalid(name),
        other => ApiError::dataset_invalid(format!("{other}")),
    })?;
    if manifest.format != expected_format {
        return Err(ApiError::training_invalid_request(format!(
            "dataset {name:?} has format {:?}; this training route requires {:?}",
            manifest.format, expected_format
        )));
    }
    let split = requested_split.unwrap_or_default();
    let split_manifest = registry
        .load_split(name)
        .map_err(|error| ApiError::dataset_invalid(format!("{error}")))?;
    let path = registry
        .split_path(name, split)
        .map_err(|error| ApiError::dataset_invalid(format!("{error}")))?;
    Ok(NamedDatasetAdmission {
        path,
        split,
        manifest,
        split_manifest,
    })
}

fn named_training_provenance(
    source: &NamedDatasetAdmission,
    admitted_corpus_sha256: String,
    rows: u64,
) -> TrainingDataProvenance {
    TrainingDataProvenance {
        source: "named_dataset".to_string(),
        dataset: Some(source.manifest.name.clone()),
        split: Some(source.split),
        dataset_corpus_sha256: Some(source.manifest.corpus_sha256.clone()),
        split_manifest_sha256: Some(source.manifest.split_manifest_sha256.clone()),
        admitted_corpus_sha256,
        rows,
    }
}

pub(crate) fn enforce_queue_caps(state: &AppState) -> Result<(), ApiError> {
    enforce_queue_capacity_for(state, 1)
}

pub(crate) fn enforce_queue_capacity_for(
    state: &AppState,
    additional_jobs: usize,
) -> Result<(), ApiError> {
    // Keep the same lock order as the atomic admission path below. This is
    // only an advisory fast-fail check: admission rechecks both caps while
    // holding write access to the tracking map and exclusive queue access.
    let tracked = state.training_jobs.read().unwrap();
    let queue = state.training_queue.lock().unwrap();
    validate_training_admission_capacity(
        queue.len(),
        state.max_queued_training_jobs,
        tracked.len(),
        state.max_tracked_jobs,
        additional_jobs,
        !matches!(state.backend.as_ref(), ModelBackend::Mock { .. }),
    )
}

fn validate_training_admission_capacity(
    queued: usize,
    max_queued: usize,
    tracked: usize,
    max_tracked: usize,
    additional_jobs: usize,
    training_supported: bool,
) -> Result<(), ApiError> {
    if queued.saturating_add(additional_jobs) > max_queued {
        return Err(ApiError::training_queue_full(max_queued));
    }
    if tracked.saturating_add(additional_jobs) > max_tracked {
        return Err(ApiError::training_tracked_full(max_tracked));
    }
    if !training_supported {
        return Err(ApiError::mock_mode_no_training());
    }
    Ok(())
}

#[cfg(test)]
fn validate_prepared_training_data_capacity(
    current_bytes: u64,
    requested_bytes: u64,
) -> Result<(), ApiError> {
    if current_bytes.saturating_add(requested_bytes)
        > crate::training_queue::MAX_LIVE_PREPARED_TRAINING_BYTES
    {
        return Err(ApiError::training_prepared_data_full(
            current_bytes,
            requested_bytes,
        ));
    }
    Ok(())
}

fn admit_training_jobs_into(
    training_jobs: &crate::state::TrainingJobs,
    training_queue: &crate::training_queue::SharedTrainingQueue,
    max_queued: usize,
    max_tracked: usize,
    training_supported: bool,
    mut pending: Vec<(TrainingJobInfo, QueueEntry)>,
) -> Result<usize, ApiError> {
    let additional_jobs = pending.len();

    // `list_queue` already uses this order. Holding both guards makes the
    // capacity decision and every insert one transaction with respect to all
    // other API submissions routed through this function.
    let mut tracked = training_jobs.write().unwrap();
    let mut queue = training_queue.lock().unwrap();
    validate_training_admission_capacity(
        queue.len(),
        max_queued,
        tracked.len(),
        max_tracked,
        additional_jobs,
        training_supported,
    )?;
    let mut pending_ids = std::collections::HashSet::with_capacity(additional_jobs);
    for (info, entry) in &pending {
        if info.job_id != entry.job_id {
            return Err(ApiError::internal(format!(
                "training admission job id mismatch: tracking={} queue={}",
                info.job_id, entry.job_id
            )));
        }
        if tracked.contains_key(&info.job_id) || !pending_ids.insert(info.job_id.clone()) {
            return Err(ApiError::internal(format!(
                "training admission duplicate job id: {}",
                info.job_id
            )));
        }
    }

    for (_, entry) in &mut pending {
        let bytes = entry.prepared_data.admission_weight_bytes();
        if entry.prepared_data_permit.bytes() == 0 {
            entry.prepared_data_permit =
                crate::training_queue::PreparedTrainingDataPermit::acquire(bytes).map_err(
                    |(current, requested)| {
                        ApiError::training_prepared_data_full(current, requested)
                    },
                )?;
        } else if entry.prepared_data_permit.bytes() < bytes {
            return Err(ApiError::internal(format!(
                "training entry {} prepared-data permit covers {} bytes but materialized data requires at least {bytes}",
                entry.job_id,
                entry.prepared_data_permit.bytes()
            )));
        }
    }

    for (info, entry) in pending {
        tracked.insert(info.job_id.clone(), info);
        queue.push(entry);
    }
    Ok(queue.len())
}

/// Attach authoritative submit-time teacher snapshots before queue
/// publication. A failure leaves all shared admission state untouched.
fn pin_registered_teachers(
    state: &AppState,
    pending: &mut [(TrainingJobInfo, QueueEntry)],
) -> Result<(), ApiError> {
    // Snapshot every registry-backed teacher immediately before publication.
    // All product entry points route through this function, so recipes and
    // scheduled/intent-driven submissions receive the same identity binding as
    // the dedicated endpoints. Reject caller-supplied bindings to keep this
    // boundary authoritative.
    for (_, entry) in pending.iter_mut() {
        if !entry.teacher_bindings.is_empty() {
            return Err(ApiError::internal(format!(
                "training admission received pre-populated teacher bindings for job {}",
                entry.job_id
            )));
        }
        if let Some(alias) = entry.job.registered_teacher_alias() {
            let spec = super::teachers::require_registered_teacher(
                state,
                alias,
                format!("teacher alias {alias:?} is not registered"),
            )?;
            let (surface, config) = match &entry.job {
                QueuedJob::Opd(req) => ("OPD", &req.config),
                QueuedJob::DistillRefresh(req) => ("DistillRefresh", &req.config),
                QueuedJob::DistillPump(req) => ("distill/pump", &req.config),
                QueuedJob::Sft(_)
                | QueuedJob::Grpo(_)
                | QueuedJob::DistillMerge(_)
                | QueuedJob::DistillSelf(_) => {
                    unreachable!("registered_teacher_alias returned Some for a teacher-free job")
                }
            };
            require_remote_teacher_off_policy(surface, &spec, config)?;
            entry.teacher_bindings.push(spec);
        }
    }
    Ok(())
}

pub(crate) fn ensure_training_backend_admission(state: &AppState) -> Result<(), ApiError> {
    state
        .ensure_backend_healthy()
        .map_err(ApiError::backend_quarantined)?;
    state.ensure_training_gpu_ownership_allowed().map_err(|_| {
        ApiError::serving_profile_conflict(
            state.serving_profile.profile(),
            "training GPU ownership",
        )
    })?;

    if matches!(state.backend.as_ref(), ModelBackend::Real { .. }) {
        state
            .training_runtime
            .resolve_device_for_weights(state.model_weight_device)
            .map_err(ApiError::training_backend_unsupported)?;
    }
    Ok(())
}

const MAX_MATERIALIZED_OPD_DATASET_BYTES: u64 = 64 * 1024 * 1024;
const MAX_MATERIALIZED_OPD_PROMPTS: usize = 100_000;
const MAX_MATERIALIZED_OPD_PROMPT_BYTES: u64 = 64 * 1024 * 1024;
fn sft_materialized_weight_bytes(
    examples: &[kiln_train::SftExample],
    ingestion: &kiln_train::SftIngestionReceipt,
) -> Result<u64, ApiError> {
    if examples.len() > crate::sft_dataset::MAX_SFT_JSONL_ROWS {
        return Err(ApiError::training_invalid_request(format!(
            "SFT corpus has {} rows; maximum is {}",
            examples.len(),
            crate::sft_dataset::MAX_SFT_JSONL_ROWS
        )));
    }
    let mut encoded_bytes = 0u64;
    for (index, example) in examples.iter().enumerate() {
        let bytes = serde_json::to_vec(example).map_err(|error| {
            ApiError::training_invalid_request(format!(
                "SFT row {index} could not be measured: {error}"
            ))
        })?;
        if bytes.len() > crate::sft_dataset::MAX_SFT_JSONL_ROW_BYTES {
            return Err(ApiError::training_invalid_request(format!(
                "SFT row {index} is {} bytes; maximum is {}",
                bytes.len(),
                crate::sft_dataset::MAX_SFT_JSONL_ROW_BYTES
            )));
        }
        encoded_bytes = encoded_bytes
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| ApiError::training_invalid_request("SFT corpus size overflow"))?;
        if encoded_bytes > crate::sft_dataset::MAX_SFT_JSONL_BYTES {
            return Err(ApiError::training_invalid_request(format!(
                "SFT corpus materializes to more than {} bytes; split it into smaller jobs",
                crate::sft_dataset::MAX_SFT_JSONL_BYTES
            )));
        }
    }
    let receipt_bytes = serde_json::to_vec(ingestion)
        .map_err(|error| ApiError::internal(format!("measure SFT ingestion receipt: {error}")))?
        .len() as u64;
    Ok(encoded_bytes
        .saturating_mul(4)
        .saturating_add(receipt_bytes.saturating_mul(2)))
}

fn validate_materialized_opd_prompts(
    surface: &str,
    prompts: &[kiln_train::opd::OpdPrompt],
) -> Result<(), ApiError> {
    measure_materialized_opd_prompts(surface, prompts).map(|_| ())
}

fn measure_materialized_opd_prompts(
    surface: &str,
    prompts: &[kiln_train::opd::OpdPrompt],
) -> Result<(usize, u64), ApiError> {
    if prompts.is_empty() {
        return Err(ApiError::training_invalid_request(format!(
            "{surface} resolved to zero prompts"
        )));
    }
    if prompts.len() > MAX_MATERIALIZED_OPD_PROMPTS {
        return Err(ApiError::training_invalid_request(format!(
            "{surface} resolved to {} prompts; maximum is {MAX_MATERIALIZED_OPD_PROMPTS}",
            prompts.len()
        )));
    }
    if let Some(index) = prompts.iter().position(|prompt| prompt.messages.is_empty()) {
        return Err(ApiError::training_invalid_request(format!(
            "{surface} prompt {index} has no messages"
        )));
    }
    let mut encoded_bytes = 0u64;
    for (index, prompt) in prompts.iter().enumerate() {
        let bytes = serde_json::to_vec(prompt).map_err(|error| {
            ApiError::training_invalid_request(format!(
                "{surface} prompt {index} could not be measured: {error}"
            ))
        })?;
        encoded_bytes = encoded_bytes
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| {
                ApiError::training_invalid_request(format!(
                    "{surface} materialized prompt size overflow"
                ))
            })?;
        if encoded_bytes > MAX_MATERIALIZED_OPD_PROMPT_BYTES {
            return Err(ApiError::training_invalid_request(format!(
                "{surface} materializes to more than {MAX_MATERIALIZED_OPD_PROMPT_BYTES} bytes; split it into smaller jobs"
            )));
        }
    }
    Ok((prompts.len(), encoded_bytes))
}

fn exact_opd_admission_max_seq_len(
    prompts: &[kiln_train::opd::OpdPrompt],
    config: &kiln_train::OpdConfig,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
) -> Result<usize, ApiError> {
    let mut longest = 0usize;
    for (index, prompt) in prompts.iter().enumerate() {
        let rendered = tokenizer
            .apply_chat_template(&prompt.messages)
            .map_err(|error| {
                ApiError::training_invalid_request(format!(
                    "OPD prompt {index} failed chat-template rendering during admission: {error}"
                ))
            })?;
        let tokens = tokenizer.encode(&rendered).map_err(|error| {
            ApiError::training_invalid_request(format!(
                "OPD prompt {index} failed tokenization during admission: {error}"
            ))
        })?;
        longest = longest.max(tokens.len());
    }
    longest.checked_add(config.max_tokens).ok_or_else(|| {
        ApiError::training_invalid_request("OPD prompt plus rollout token count overflow")
    })
}

fn conservative_opd_fixture_sequence_tokens(
    prompt: &kiln_train::opd::OpdPrompt,
    config: &kiln_train::OpdConfig,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
) -> Result<u64, ApiError> {
    // Runtime fixture keys use tokenized sequences, but charging at least one
    // token per compact source byte also covers owned strings, trajectory
    // masks, asymmetric teacher context, and chat-template bookkeeping.
    let source_bytes = serde_json::to_vec(prompt)
        .map_err(|error| {
            ApiError::training_invalid_request(format!(
                "measure OPD prompt for teacher-fixture admission: {error}"
            ))
        })?
        .len() as u64;
    let rendered_tokens = if prompt.trajectory.is_empty() {
        let rendered = tokenizer
            .apply_chat_template(&prompt.messages)
            .map_err(|error| {
                ApiError::training_invalid_request(format!(
                    "render OPD prompt for teacher-fixture admission: {error}"
                ))
            })?;
        tokenizer
            .encode(&rendered)
            .map_err(|error| {
                ApiError::training_invalid_request(format!(
                    "tokenize OPD prompt for teacher-fixture admission: {error}"
                ))
            })?
            .len() as u64
    } else {
        kiln_train::trajectory_mask::build_masks_from_trajectory(
            &prompt.trajectory,
            &prompt.messages,
            tokenizer,
            &kiln_train::trajectory_mask::MaskConfig::default(),
        )
        .map_err(|error| {
            ApiError::training_invalid_request(format!(
                "tokenize OPD trajectory for teacher-fixture admission: {error:#}"
            ))
        })?
        .input_ids
        .len() as u64
    };
    let rollout_tokens = if matches!(
        config.training_mode,
        kiln_train::opd::OpdTrainingMode::OnPolicy
    ) {
        config.max_tokens as u64
    } else {
        0
    };
    source_bytes
        .max(rendered_tokens)
        .checked_add(rollout_tokens)
        // Fixed slack covers generation/chat-template sentinel tokens whose
        // text is not present in the serialized prompt.
        .and_then(|tokens| tokens.checked_add(64))
        .ok_or_else(|| {
            ApiError::training_invalid_request("OPD teacher-fixture token bound overflow")
        })
}

fn conservative_opd_fixture_bytes(
    prompts: &[kiln_train::opd::OpdPrompt],
    config: &kiln_train::OpdConfig,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    extra_teacher_context_bytes: u64,
    fixture_copies: u64,
) -> Result<u64, ApiError> {
    let sequence_tokens = prompts.iter().try_fold(0u64, |total, prompt| {
        total
            .checked_add(conservative_opd_fixture_sequence_tokens(
                prompt, config, tokenizer,
            )?)
            .ok_or_else(|| {
                ApiError::training_invalid_request("OPD teacher-fixture token total overflow")
            })
    })?;
    let sequence_tokens = sequence_tokens
        .checked_add(extra_teacher_context_bytes)
        .ok_or_else(|| ApiError::training_invalid_request("OPD teacher context size overflow"))?;
    crate::training_queue::conservative_teacher_fixture_bytes(
        sequence_tokens,
        config.top_k as u64,
        fixture_copies,
    )
    .ok_or_else(|| {
        ApiError::training_invalid_request("OPD teacher-fixture memory estimate overflow")
    })
}

fn queued_request_weight_bytes<T: serde::Serialize>(request: &T) -> u64 {
    serde_json::to_vec(request)
        .ok()
        .and_then(|bytes| u64::try_from(bytes.len()).ok())
        .unwrap_or(u64::MAX)
        .saturating_mul(4)
}

fn retained_teacher_fixture_bytes_for_entry(
    state: &AppState,
    entry: &QueueEntry,
) -> Result<u64, ApiError> {
    use crate::training_queue::PreparedTrainingData;

    let tokenizer = state.tokenizer.as_ref();
    match (&entry.job, &entry.prepared_data) {
        (QueuedJob::Sft(_), _) | (QueuedJob::Grpo(_), _) => Ok(0),
        (QueuedJob::Opd(request), prepared) => {
            let prompts = match prepared {
                PreparedTrainingData::None => request.prompts.as_slice(),
                PreparedTrainingData::OpdPrompts(prompts) => prompts.as_slice(),
                PreparedTrainingData::OpdOffPolicy(data) => data.prepared.prompts.as_slice(),
                _ => {
                    return Err(ApiError::internal(
                        "OPD teacher-fixture admission received mismatched prepared data",
                    ));
                }
            };
            conservative_opd_fixture_bytes(prompts, &request.config, tokenizer, 0, 1)
        }
        (
            QueuedJob::DistillRefresh(request),
            PreparedTrainingData::DistillRefreshPrompts(prompts),
        ) => conservative_opd_fixture_bytes(prompts, &request.config, tokenizer, 0, 1),
        (
            QueuedJob::DistillMerge(request),
            PreparedTrainingData::DistillMergePrompts(per_source),
        ) => per_source.iter().try_fold(0u64, |total, source| {
            total
                .checked_add(conservative_opd_fixture_bytes(
                    &source.prompts,
                    &request.config,
                    tokenizer,
                    0,
                    // Merge transplants each source fixture into the unified
                    // fixture while both maps are live.
                    2,
                )?)
                .ok_or_else(|| {
                    ApiError::training_invalid_request(
                        "DistillMerge teacher-fixture memory estimate overflow",
                    )
                })
        }),
        (QueuedJob::DistillPump(request), PreparedTrainingData::DistillPumpPrompts(prompts)) => {
            conservative_opd_fixture_bytes(prompts, &request.config, tokenizer, 0, 1)
        }
        (QueuedJob::DistillSelf(request), PreparedTrainingData::None) => {
            let prompts = request.prompts.as_deref().unwrap_or(&[]);
            let extra_context_bytes = request
                .ground_truth
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .chain(request.documents.as_deref().unwrap_or(&[]))
                .try_fold(0u64, |total, context| {
                    total.checked_add(context.len() as u64).ok_or_else(|| {
                        ApiError::training_invalid_request(
                            "self-distillation teacher context size overflow",
                        )
                    })
                })?;
            // The self-distill builder temporarily retains teacher-keyed and
            // student-keyed fixtures at the same time.
            conservative_opd_fixture_bytes(
                prompts,
                &request.config,
                tokenizer,
                extra_context_bytes,
                2,
            )
        }
        _ => Err(ApiError::internal(
            "teacher-fixture admission received mismatched queued/prepared data",
        )),
    }
}

fn acquire_training_entry_prepared_data_permit(
    state: &AppState,
    entry: &mut QueueEntry,
) -> Result<(), ApiError> {
    let materialized_bytes = entry.prepared_data.admission_weight_bytes();
    let deferred_fixture_bytes = retained_teacher_fixture_bytes_for_entry(state, entry)?;
    let queued_request_bytes = match &entry.job {
        QueuedJob::Opd(request) => queued_request_weight_bytes(request),
        QueuedJob::DistillRefresh(request) => queued_request_weight_bytes(request),
        QueuedJob::DistillMerge(request) => queued_request_weight_bytes(request),
        QueuedJob::DistillPump(request) => queued_request_weight_bytes(request),
        QueuedJob::DistillSelf(request) => queued_request_weight_bytes(request),
        QueuedJob::Sft(_) | QueuedJob::Grpo(_) => 0,
    };
    let required_bytes = materialized_bytes
        .checked_add(deferred_fixture_bytes)
        .and_then(|bytes| bytes.checked_add(queued_request_bytes))
        .ok_or_else(|| {
            ApiError::training_invalid_request("prepared training-data memory estimate overflow")
        })?;
    entry
        .prepared_data_permit
        .grow_to(required_bytes)
        .map_err(|(current, requested)| {
            ApiError::training_prepared_data_full(current, requested)
        })?;
    Ok(())
}

fn opd_preflight_admission(
    state: &AppState,
    prompts: &[kiln_train::opd::OpdPrompt],
    config: &kiln_train::OpdConfig,
    lora_rank: usize,
) -> Result<PreflightAdmission, ApiError> {
    let max_seq_len = exact_opd_admission_max_seq_len(prompts, config, state.tokenizer.as_ref())?;
    enforce_training_preflight(
        state,
        max_seq_len,
        EstimateOptions {
            optimizer: config.optimizer,
            ..Default::default()
        },
        lora_rank,
        false,
    )
}

fn load_bounded_off_policy_dataset(
    path: &str,
    prepared_data_permit: &mut crate::training_queue::PreparedTrainingDataPermit,
) -> Result<(PathBuf, u64, kiln_train::LoadedOffPolicyDistillationDataset), ApiError> {
    use std::io::Read;

    let canonical = std::fs::canonicalize(path).map_err(|error| {
        ApiError::training_invalid_request(format!(
            "failed to resolve OPD dataset_path {path:?}: {error}"
        ))
    })?;
    let file = std::fs::File::open(&canonical).map_err(|error| {
        ApiError::training_invalid_request(format!(
            "failed to open OPD dataset_path {path:?}: {error}"
        ))
    })?;
    let metadata = file.metadata().map_err(|error| {
        ApiError::training_invalid_request(format!(
            "failed to inspect OPD dataset_path {path:?}: {error}"
        ))
    })?;
    if !metadata.is_file() || metadata.len() > MAX_MATERIALIZED_OPD_DATASET_BYTES {
        return Err(ApiError::training_invalid_request(format!(
            "OPD dataset_path {path:?} must be a regular file no larger than {MAX_MATERIALIZED_OPD_DATASET_BYTES} bytes (found {} bytes)",
            metadata.len()
        )));
    }
    prepared_data_permit
        .grow_to(metadata.len().saturating_mul(8))
        .map_err(|(current, requested)| {
            ApiError::training_prepared_data_full(current, requested)
        })?;
    let expected_bytes = metadata.len();
    let mut source = Vec::with_capacity(expected_bytes as usize);
    file.take(MAX_MATERIALIZED_OPD_DATASET_BYTES + 1)
        .read_to_end(&mut source)
        .map_err(|error| {
            ApiError::training_invalid_request(format!(
                "read off-policy OPD dataset_path {path:?}: {error}"
            ))
        })?;
    prepared_data_permit
        .grow_to((source.len() as u64).saturating_mul(8))
        .map_err(|(current, requested)| {
            ApiError::training_prepared_data_full(current, requested)
        })?;
    if source.len() as u64 > MAX_MATERIALIZED_OPD_DATASET_BYTES
        || source.len() as u64 != expected_bytes
    {
        return Err(ApiError::training_invalid_request(format!(
            "OPD dataset_path {path:?} changed while it was being admitted: expected {expected_bytes} bytes, read {}",
            source.len()
        )));
    }
    let source = std::str::from_utf8(&source).map_err(|error| {
        ApiError::training_invalid_request(format!(
            "off-policy OPD dataset_path {path:?} is not UTF-8: {error}"
        ))
    })?;
    let loaded =
        kiln_train::parse_off_policy_distillation_dataset_str(source).map_err(|error| {
            ApiError::training_invalid_request(format!(
                "load off-policy OPD dataset_path {path:?}: {error:#}"
            ))
        })?;
    if loaded.examples.is_empty() {
        return Err(ApiError::training_invalid_request(format!(
            "OPD dataset_path {path:?} contains no examples"
        )));
    }
    Ok((canonical, expected_bytes, loaded))
}

fn prepare_off_policy_opd_admission(
    state: &AppState,
    req: &mut OpdRequest,
    teacher_spec: &super::teachers::TeacherSpec,
    prepared_data_permit: &mut crate::training_queue::PreparedTrainingDataPermit,
) -> Result<crate::training_queue::PreparedOffPolicyAdmission, ApiError> {
    let path = req.dataset_path.as_deref().ok_or_else(|| {
        ApiError::internal("off-policy OPD materialization requires dataset_path")
    })?;
    let (canonical, source_size_bytes, loaded) =
        load_bounded_off_policy_dataset(path, prepared_data_permit)?;
    let contains_numeric_teacher_logits = loaded.examples.iter().any(|example| {
        example
            .teacher_tokens
            .iter()
            .any(|token| token.logprob.is_some() || !token.top_logprobs.is_empty())
    });
    let teacher_identity = loaded
        .manifest
        .as_ref()
        .map(|manifest| manifest.teacher_identity().clone());
    if contains_numeric_teacher_logits && teacher_identity.is_none() {
        return Err(ApiError::training_invalid_request(format!(
            "off-policy OPD dataset_path {path:?} contains numeric teacher logits but has no canonical {} first record",
            kiln_train::OFF_POLICY_DISTILLATION_MANIFEST_SCHEMA_V1
        )));
    }
    if let Some(identity) = teacher_identity.as_ref() {
        let expected = teacher_spec.identity.as_ref().ok_or_else(|| {
            ApiError::training_invalid_request(format!(
                "off-policy OPD dataset_path {path:?} declares teacher revision sha256:{}, but registered teacher {:?} has no authoritative identity",
                identity.content_revision(),
                teacher_spec.alias
            ))
        })?;
        if identity != expected {
            return Err(ApiError::training_invalid_request(format!(
                "off-policy OPD dataset_path {path:?} teacher revision sha256:{} does not match pinned registered teacher revision sha256:{}",
                identity.content_revision(),
                expected.content_revision()
            )));
        }
    }
    let source_sha256 = loaded.source_sha256.clone();
    // Reserve the worst-case retained fixture before building it. The parsed
    // source is bounded independently, while the top-K maps can be hundreds of
    // times wider than JSON. Building first and checking afterward would let a
    // small admitted source transiently allocate multiple GiB.
    let projected_prompts = loaded
        .examples
        .iter()
        .map(|example| {
            if example.trajectory.is_empty() {
                let mut messages = example.messages.clone();
                messages.push(kiln_train::ChatMessage::new(
                    "assistant",
                    example.teacher_response.clone(),
                ));
                kiln_train::opd::OpdPrompt {
                    messages,
                    teacher_extra_messages: Vec::new(),
                    trajectory: Vec::new(),
                }
            } else {
                kiln_train::opd::OpdPrompt {
                    messages: example.messages.clone(),
                    teacher_extra_messages: Vec::new(),
                    trajectory: example.trajectory.clone(),
                }
            }
        })
        .collect::<Vec<_>>();
    let projected_materialized_bytes = source_size_bytes
        .saturating_mul(8)
        .max(crate::training_queue::PreparedTrainingData::prompt_weight_bytes(&projected_prompts));
    let projected_fixture_bytes = conservative_opd_fixture_bytes(
        &projected_prompts,
        &req.config,
        state.tokenizer.as_ref(),
        0,
        1,
    )?;
    let projected_total = projected_materialized_bytes
        .checked_add(projected_fixture_bytes)
        .and_then(|bytes| bytes.checked_add(queued_request_weight_bytes(req)))
        .ok_or_else(|| {
            ApiError::training_invalid_request("off-policy OPD prepared-data estimate overflow")
        })?;
    prepared_data_permit
        .grow_to(projected_total)
        .map_err(|(current, requested)| {
            ApiError::training_prepared_data_full(current, requested)
        })?;
    drop(projected_prompts);
    let prepared = kiln_train::prepare_off_policy_distillation_dataset_with_identity(
        &loaded.examples,
        state.tokenizer.as_ref(),
        req.teacher.clone(),
        teacher_identity.clone(),
        state.model_config.vocab_size,
        req.config.top_k,
        req.config.objective,
        req.config.echo.as_ref(),
    )
    .map_err(|error| {
        ApiError::training_invalid_request(format!(
            "prepare off-policy OPD dataset_path {path:?}: {error:#}"
        ))
    })?;
    validate_materialized_opd_prompts("OPD dataset_path", &prepared.prompts)?;
    req.dataset_path = Some(canonical.to_string_lossy().into_owned());
    Ok(crate::training_queue::PreparedOffPolicyAdmission {
        prepared,
        source_sha256,
        source_size_bytes,
        teacher_identity,
    })
}

fn queued_training_optimizer_request(job: &QueuedJob) -> (kiln_train::Optimizer, usize) {
    let (optimizer, lora_rank) = match job {
        QueuedJob::Sft(req) => (req.config.optimizer, req.config.lora_rank),
        QueuedJob::Grpo(req) => (req.config.optimizer, req.config.lora_rank),
        QueuedJob::Opd(req) => (req.config.optimizer, req.config.lora_rank),
        QueuedJob::DistillRefresh(req) => (req.config.optimizer, req.config.lora_rank),
        QueuedJob::DistillMerge(req) => (req.config.optimizer, req.config.lora_rank),
        QueuedJob::DistillPump(req) => (
            req.config.optimizer,
            req.rank.unwrap_or(req.config.lora_rank),
        ),
        QueuedJob::DistillSelf(req) => (req.config.optimizer, req.config.lora_rank),
    };
    (optimizer, lora_rank)
}

fn queued_training_workload(job: &QueuedJob) -> TrainingWorkload {
    match job {
        QueuedJob::Sft(_) => TrainingWorkload::Sft,
        QueuedJob::Grpo(_) => TrainingWorkload::Grpo,
        QueuedJob::DistillRefresh(_) => TrainingWorkload::DistillRefresh,
        QueuedJob::Opd(_)
        | QueuedJob::DistillMerge(_)
        | QueuedJob::DistillPump(_)
        | QueuedJob::DistillSelf(_) => TrainingWorkload::Opd,
    }
}

pub(crate) fn enforce_queued_training_optimizer_admission(
    state: &AppState,
    job: &QueuedJob,
) -> Result<(), ApiError> {
    let (optimizer, lora_rank) = queued_training_optimizer_request(job);
    enforce_training_optimizer_admission(state, optimizer, lora_rank)?;
    Ok(())
}

pub(crate) fn enforce_queued_training_workload_admission(
    state: &AppState,
    job: &QueuedJob,
) -> Result<(), ApiError> {
    enforce_training_workload_admission(state, queued_training_workload(job))
}

fn prepare_training_entry_admission(
    state: &AppState,
    info: &mut TrainingJobInfo,
    entry: &mut QueueEntry,
) -> Result<(), ApiError> {
    use crate::training_queue::{
        PreparedDistillMergeSource, PreparedSftAdmission, PreparedTrainingData, QueuedJob,
    };

    enforce_queued_training_workload_admission(state, &entry.job)?;
    enforce_queued_training_optimizer_admission(state, &entry.job)?;

    match &mut entry.job {
        QueuedJob::Sft(req) => {
            if !matches!(entry.prepared_data, PreparedTrainingData::None) {
                return Err(ApiError::internal(
                    "SFT queue entry carried unexpected prepared training data",
                ));
            }
            let source_count = usize::from(!req.examples.is_empty())
                + usize::from(req.dataset_path.is_some())
                + usize::from(req.dataset.is_some());
            if source_count != 1 {
                return Err(ApiError::training_invalid_request(
                    "SFT request must use exactly one of examples, dataset_path, or dataset",
                ));
            }
            let mut named_source = None;
            let prepared = if let Some(dataset_name) = req.dataset.as_deref() {
                if dataset_name == "corrections:active" {
                    let store = super::corrections::CorrectionsStore::for_state(state);
                    let (ids, examples) = store.trainable_rows();
                    if examples.is_empty() {
                        return Err(ApiError::training_invalid_request(
                            "corrections:active has no trainable rows; write an ideal answer for at least one correction first",
                        ));
                    }
                    let prepared = kiln_train::prepare_sft_examples(
                        examples,
                        state.tokenizer.as_ref(),
                        req.config.invalid_row_policy,
                        "corrections",
                        Some(dataset_name.to_string()),
                    )
                    .map_err(|error| {
                        ApiError::training_invalid_request(format!(
                            "invalid corrections SFT rows: {error:#}"
                        ))
                    })?;
                    info.consumed_correction_ids =
                        retained_correction_ids(ids, &prepared.ingestion)?;
                    prepared
                } else {
                    let source = resolve_named_training_dataset(
                        state,
                        dataset_name,
                        req.dataset_split,
                        crate::eval::DatasetFormat::SftChat,
                    )?;
                    let prepared = crate::sft_dataset::prepare_sft_jsonl(
                        &source.path,
                        state.tokenizer.as_ref(),
                        req.config.invalid_row_policy,
                        "named_dataset",
                        Some(format!("{dataset_name}:{}", source.split.as_str())),
                    )
                    .map_err(|error| {
                        ApiError::training_invalid_request(format!(
                            "invalid SFT dataset {dataset_name:?} {} split: {error:#}",
                            source.split.as_str()
                        ))
                    })?;
                    named_source = Some(source);
                    prepared
                }
            } else if let Some(path) = req.dataset_path.as_deref() {
                let canonical = std::fs::canonicalize(path).map_err(|error| {
                    ApiError::training_invalid_request(format!(
                        "failed to resolve SFT dataset_path {path:?}: {error}"
                    ))
                })?;
                let prepared = crate::sft_dataset::prepare_sft_jsonl(
                    &canonical,
                    state.tokenizer.as_ref(),
                    req.config.invalid_row_policy,
                    "dataset_path",
                    Some(canonical.display().to_string()),
                )
                .map_err(|error| {
                    ApiError::training_invalid_request(format!(
                        "invalid SFT dataset_path {path:?}: {error:#}"
                    ))
                })?;
                req.dataset_path = Some(canonical.to_string_lossy().into_owned());
                prepared
            } else {
                kiln_train::prepare_sft_examples(
                    std::mem::take(&mut req.examples),
                    state.tokenizer.as_ref(),
                    req.config.invalid_row_policy,
                    "inline",
                    None,
                )
                .map_err(|error| {
                    ApiError::training_invalid_request(format!("invalid SFT rows: {error:#}"))
                })?
            };
            let contamination = post_eval_contamination_index(state, req.post_eval.as_ref())?;
            if let Some(index) = contamination.as_ref() {
                let overlap = named_source
                    .as_ref()
                    .and_then(|source| {
                        check_split_contamination(index, &source.split_manifest, source.split)
                    })
                    .or_else(|| check_sft_contamination(index, &prepared.examples));
                if let Some(overlap) = overlap {
                    return Err(contamination_error(
                        req.post_eval
                            .as_ref()
                            .expect("contamination index requires post_eval"),
                        overlap,
                    ));
                }
            }
            info.training_data = Some(if let Some(source) = named_source.as_ref() {
                named_training_provenance(
                    source,
                    prepared.ingestion.kept_corpus_sha256.clone(),
                    prepared.ingestion.rows_kept as u64,
                )
            } else {
                TrainingDataProvenance {
                    source: prepared.ingestion.source.clone(),
                    dataset: req.dataset.clone(),
                    split: None,
                    dataset_corpus_sha256: None,
                    split_manifest_sha256: None,
                    admitted_corpus_sha256: prepared.ingestion.kept_corpus_sha256.clone(),
                    rows: prepared.ingestion.rows_kept as u64,
                }
            });
            let admission_weight_bytes =
                sft_materialized_weight_bytes(&prepared.examples, &prepared.ingestion)?;
            let loss_route = sft_loss_route_for_state(state)?;
            let admission = enforce_training_preflight(
                state,
                prepared.max_seq_len,
                EstimateOptions {
                    sft: Some(SftEstimateOptions {
                        max_active_tokens: prepared.max_supervised_tokens,
                        loss_route,
                        checkpoint_boundary_policy: state
                            .training_runtime
                            .checkpoint_boundary_policy(),
                    }),
                    optimizer: req.config.optimizer,
                    ..Default::default()
                },
                req.config.lora_rank,
                false,
            )?;
            req.config.grad_checkpoint_segments = admission.checkpoint_segments;
            req.ingestion = Some(prepared.ingestion.clone());
            req.examples = prepared.examples;
            // The queue owns the exact admitted rows. Preserve source identity
            // in the ingestion receipt, while keeping the replay request a
            // valid single-source inline SFT payload.
            req.dataset_path = None;
            req.dataset = None;
            req.dataset_split = None;
            entry.reserved_bytes = admission.reserved_bytes;
            entry.prepared_data = PreparedTrainingData::Sft(PreparedSftAdmission {
                ingestion: prepared.ingestion,
                max_seq_len: prepared.max_seq_len,
                max_supervised_tokens: prepared.max_supervised_tokens,
                admission_weight_bytes,
                loss_route,
            });
        }
        QueuedJob::Grpo(req) => {
            let source_count = usize::from(!req.groups.is_empty())
                + usize::from(req.dataset_path.is_some())
                + usize::from(req.dataset.is_some());
            if source_count != 1 {
                return Err(ApiError::training_invalid_request(
                    "GRPO request must use exactly one of groups, dataset_path, or dataset",
                ));
            }
            let named_source = req
                .dataset
                .as_deref()
                .map(|dataset| {
                    resolve_named_training_dataset(
                        state,
                        dataset,
                        req.dataset_split,
                        crate::eval::DatasetFormat::GrpoGroups,
                    )
                })
                .transpose()?;
            let contamination = post_eval_contamination_index(state, req.post_eval.as_ref())?;
            if let (Some(index), Some(source)) = (contamination.as_ref(), named_source.as_ref())
                && let Some(overlap) =
                    check_split_contamination(index, &source.split_manifest, source.split)
            {
                return Err(contamination_error(
                    req.post_eval
                        .as_ref()
                        .expect("contamination index requires post_eval"),
                    overlap,
                ));
            }
            let source_path = named_source
                .as_ref()
                .map(|source| source.path.as_path())
                .or_else(|| req.dataset_path.as_deref().map(std::path::Path::new));
            let (max_seq_len, prepared, admitted_corpus_sha256, admitted_rows) = if let Some(path) =
                source_path
            {
                let receipt = match std::mem::take(&mut entry.prepared_data) {
                    PreparedTrainingData::None => validate_grpo_jsonl_submission(
                        &path.to_string_lossy(),
                        &state.adapter_dir.join(".training-inputs"),
                        &mut entry.prepared_data_permit,
                        state.tokenizer.as_ref(),
                        &req.config,
                        state.model_config.num_layers,
                        contamination.as_ref().map(|index| {
                            (
                                index,
                                req.post_eval
                                    .as_ref()
                                    .expect("contamination index requires post_eval"),
                            )
                        }),
                    )?
                    .source_receipt
                    .ok_or_else(|| ApiError::internal("GRPO scan omitted source receipt"))?,
                    PreparedTrainingData::GrpoJsonl(receipt) => receipt,
                    _ => {
                        return Err(ApiError::internal(
                            "GRPO queue entry carried mismatched prepared training data",
                        ));
                    }
                };
                let admitted_corpus_sha256 = receipt.source_sha256.clone();
                let admitted_rows = receipt.groups as u64;
                req.dataset_path = Some(receipt.path.to_string_lossy().into_owned());
                (
                    receipt.max_seq_len,
                    PreparedTrainingData::GrpoJsonl(receipt),
                    admitted_corpus_sha256,
                    admitted_rows,
                )
            } else {
                if !matches!(entry.prepared_data, PreparedTrainingData::None) {
                    return Err(ApiError::internal(
                        "inline GRPO queue entry carried external source data",
                    ));
                }
                let mut maximum = 0usize;
                for (index, group) in req.groups.iter().enumerate() {
                    if let Some(contamination) = contamination.as_ref()
                        && let Some(overlap) = check_grpo_group_contamination(contamination, group)
                    {
                        return Err(contamination_error(
                            req.post_eval
                                .as_ref()
                                .expect("contamination index requires post_eval"),
                            overlap,
                        ));
                    }
                    let row_max =
                        kiln_train::trainer::validate_grpo_group_policy_data_and_max_seq_len(
                            group,
                            &req.config,
                            state.tokenizer.as_ref(),
                            index + 1,
                        )
                        .map_err(|error| {
                            ApiError::training_invalid_request(format!(
                                "invalid inline GRPO group {index}: {error:#}"
                            ))
                        })?;
                    maximum = maximum.max(row_max);
                }
                let corpus = serde_json::to_value(&req.groups).map_err(|error| {
                    ApiError::internal(format!("serialize admitted inline GRPO groups: {error}"))
                })?;
                (
                    maximum,
                    PreparedTrainingData::None,
                    kiln_eval::sha256_json(&corpus),
                    req.groups.len() as u64,
                )
            };
            info.training_data = Some(if let Some(source) = named_source.as_ref() {
                named_training_provenance(source, admitted_corpus_sha256, admitted_rows)
            } else {
                TrainingDataProvenance {
                    source: if req.dataset_path.is_some() {
                        "dataset_path".to_string()
                    } else {
                        "inline".to_string()
                    },
                    dataset: None,
                    split: None,
                    dataset_corpus_sha256: None,
                    split_manifest_sha256: None,
                    admitted_corpus_sha256,
                    rows: admitted_rows,
                }
            });
            req.dataset = None;
            req.dataset_split = None;
            let admission = enforce_training_preflight(
                state,
                max_seq_len,
                EstimateOptions {
                    optimizer: req.config.optimizer,
                    ..Default::default()
                },
                req.config.lora_rank,
                false,
            )?;
            req.config.grad_checkpoint_segments = admission.checkpoint_segments;
            entry.reserved_bytes = admission.reserved_bytes;
            entry.prepared_data = prepared;
        }
        QueuedJob::Opd(req) => {
            if req.dataset_path.is_some() == !req.prompts.is_empty() {
                return Err(ApiError::training_invalid_request(
                    "OPD request must use exactly one of prompts or dataset_path",
                ));
            }
            validate_opd_config_at_submit(&req.config)?;
            let prepared = if let Some(path) = req.dataset_path.as_deref() {
                if !matches!(entry.prepared_data, PreparedTrainingData::None) {
                    return Err(ApiError::internal(
                        "OPD queue entry carried caller-supplied prepared training data",
                    ));
                }
                if crate::dataset_resolve::is_agent_traces_selector(path) {
                    let prompts = crate::dataset_resolve::resolve_agent_trace_prompts(
                        &state.adapter_dir,
                        path,
                        crate::recent_requests::now_unix_ms() as i64,
                    )
                    .map_err(ApiError::training_invalid_request)?;
                    validate_materialized_opd_prompts("OPD agent-trace selector", &prompts)?;
                    PreparedTrainingData::OpdPrompts(prompts)
                } else {
                    let teacher_spec = entry.teacher_bindings.first().ok_or_else(|| {
                        ApiError::internal("OPD dataset admission has no pinned teacher")
                    })?;
                    PreparedTrainingData::OpdOffPolicy(prepare_off_policy_opd_admission(
                        state,
                        req,
                        teacher_spec,
                        &mut entry.prepared_data_permit,
                    )?)
                }
            } else {
                validate_materialized_opd_prompts("OPD request", &req.prompts)?;
                PreparedTrainingData::None
            };
            let prompts = match &prepared {
                PreparedTrainingData::OpdPrompts(prompts) => prompts.as_slice(),
                PreparedTrainingData::OpdOffPolicy(data) => data.prepared.prompts.as_slice(),
                PreparedTrainingData::None => req.prompts.as_slice(),
                _ => unreachable!("OPD admission constructed an OPD variant"),
            };
            let admission =
                opd_preflight_admission(state, prompts, &req.config, req.config.lora_rank)?;
            req.config.grad_checkpoint_segments = admission.checkpoint_segments;
            entry.reserved_bytes = admission.reserved_bytes;
            entry.prepared_data = prepared;
        }
        QueuedJob::DistillRefresh(req) => {
            if !matches!(entry.prepared_data, PreparedTrainingData::None) {
                return Err(ApiError::internal(
                    "DistillRefresh queue entry carried caller-supplied prepared training data",
                ));
            }
            validate_opd_config_at_submit(&req.config)?;
            let prompts = match &req.new_data {
                kiln_train::NewKnowledgeSource::Inline { examples } => examples.clone(),
                kiln_train::NewKnowledgeSource::Dataset { dataset } => {
                    crate::dataset_resolve::resolve_opd_dataset_selector(
                        dataset,
                        &state.adapter_dir,
                        state.dataset_registry.as_deref(),
                        crate::recent_requests::now_unix_ms() as i64,
                    )
                    .map_err(ApiError::training_invalid_request)?
                }
            };
            validate_materialized_opd_prompts("DistillRefresh new_data", &prompts)?;
            let admission =
                opd_preflight_admission(state, &prompts, &req.config, req.config.lora_rank)?;
            req.config.grad_checkpoint_segments = admission.checkpoint_segments;
            entry.reserved_bytes = admission.reserved_bytes;
            entry.prepared_data = PreparedTrainingData::DistillRefreshPrompts(prompts);
        }
        QueuedJob::DistillMerge(req) => {
            if !matches!(entry.prepared_data, PreparedTrainingData::None) {
                return Err(ApiError::internal(
                    "DistillMerge queue entry carried caller-supplied prepared training data",
                ));
            }
            validate_opd_config_at_submit(&req.config)?;
            if req.sources.is_empty() {
                return Err(ApiError::training_invalid_request(
                    "distill_merge: sources must be non-empty",
                ));
            }
            if req.rollout_budget == 0 {
                return Err(ApiError::training_invalid_request(
                    "distill_merge: rollout_budget must be greater than zero",
                ));
            }
            let mut per_source = Vec::with_capacity(req.sources.len());
            let mut aggregate_prompt_count = 0usize;
            let mut aggregate_prompt_bytes = 0u64;
            let mut max_seq_len = 0usize;
            let mut source_names = std::collections::HashSet::with_capacity(req.sources.len());
            let mut prompt_owners = std::collections::HashMap::new();
            let _adapter_guard = crate::adapter_swap::adapter_mutation_guard_blocking(state)
                .map_err(ApiError::training_invalid_request)?;
            for source in &req.sources {
                super::adapters::validate_adapter_name(&source.adapter)?;
                if !source_names.insert(source.adapter.as_str()) {
                    return Err(ApiError::training_invalid_request(format!(
                        "distill_merge: duplicate source adapter {:?}",
                        source.adapter
                    )));
                }
                if !source.weight.is_finite() || source.weight <= 0.0 {
                    return Err(ApiError::training_invalid_request(format!(
                        "distill_merge: source adapter {:?} weight must be finite and greater than zero",
                        source.adapter
                    )));
                }
                let source_dir = std::fs::canonicalize(state.adapter_dir.join(&source.adapter))
                    .map_err(|error| {
                        ApiError::training_invalid_request(format!(
                            "distill_merge: resolve source adapter `{}`: {error}",
                            source.adapter
                        ))
                    })?;
                let prompts =
                    crate::training_queue::derive_source_prompts(&source_dir, &source.adapter)
                        .map_err(ApiError::training_invalid_request)?;
                let (source_prompt_count, source_prompt_bytes) = measure_materialized_opd_prompts(
                    &format!("DistillMerge source {:?}", source.adapter),
                    &prompts,
                )?;
                aggregate_prompt_count = aggregate_prompt_count
                    .checked_add(source_prompt_count)
                    .ok_or_else(|| {
                        ApiError::training_invalid_request(
                            "DistillMerge aggregate prompt count overflow",
                        )
                    })?;
                aggregate_prompt_bytes = aggregate_prompt_bytes
                    .checked_add(source_prompt_bytes)
                    .ok_or_else(|| {
                        ApiError::training_invalid_request(
                            "DistillMerge aggregate prompt size overflow",
                        )
                    })?;
                if aggregate_prompt_count > MAX_MATERIALIZED_OPD_PROMPTS
                    || aggregate_prompt_bytes > MAX_MATERIALIZED_OPD_PROMPT_BYTES
                {
                    return Err(ApiError::training_invalid_request(format!(
                        "DistillMerge materializes {aggregate_prompt_count} prompts / {aggregate_prompt_bytes} bytes across sources; maximum is {MAX_MATERIALIZED_OPD_PROMPTS} prompts / {MAX_MATERIALIZED_OPD_PROMPT_BYTES} bytes"
                    )));
                }
                for prompt in &prompts {
                    let prompt_key =
                        distill_merge_prompt_sequence_sha256(prompt, state.tokenizer.as_ref())?;
                    if let Some(owner) = prompt_owners.get(&prompt_key) {
                        if owner != &source.adapter {
                            return Err(ApiError::training_invalid_request(format!(
                                "distill_merge: source adapters {owner:?} and {:?} contain the same tokenized prompt; shared prompts require weighted multi-teacher aggregation, which is not implemented, so deduplicate or pre-sample the replay logs before submitting",
                                source.adapter
                            )));
                        }
                    } else {
                        prompt_owners.insert(prompt_key, source.adapter.clone());
                    }
                }
                max_seq_len = max_seq_len.max(exact_opd_admission_max_seq_len(
                    &prompts,
                    &req.config,
                    state.tokenizer.as_ref(),
                )?);
                let source_identity =
                    kiln_model::lora_loader::LoraSourceIdentity::from_adapter_dir(&source_dir)
                        .map_err(|error| {
                            ApiError::training_invalid_request(format!(
                                "distill_merge: fingerprint source adapter `{}` at {}: {error:#}",
                                source.adapter,
                                source_dir.display()
                            ))
                        })?;
                per_source.push(PreparedDistillMergeSource {
                    source: source.clone(),
                    prompts,
                    adapter_path: source_dir,
                    source_identity,
                });
            }
            let materialized_counts = per_source
                .iter()
                .map(|source| source.prompts.len())
                .collect::<Vec<_>>();
            validate_distill_merge_sampling_contract(req, &materialized_counts)?;
            let admission = enforce_training_preflight(
                state,
                max_seq_len,
                EstimateOptions {
                    optimizer: req.config.optimizer,
                    ..Default::default()
                },
                req.config.lora_rank,
                false,
            )?;
            req.config.grad_checkpoint_segments = admission.checkpoint_segments;
            entry.reserved_bytes = admission.reserved_bytes;
            entry.prepared_data = PreparedTrainingData::DistillMergePrompts(per_source);
        }
        QueuedJob::DistillPump(req) => {
            if !matches!(entry.prepared_data, PreparedTrainingData::None) {
                return Err(ApiError::internal(
                    "DistillPump queue entry carried caller-supplied prepared training data",
                ));
            }
            validate_opd_config_at_submit(&req.config)?;
            let prompts = match &req.mode {
                kiln_train::DistillPumpMode::Examples { examples } => examples.clone(),
                kiln_train::DistillPumpMode::Domain { domain } => {
                    crate::training_queue::canonical_domain_seed_prompts(domain)
                        .map_err(ApiError::training_invalid_request)?
                }
                kiln_train::DistillPumpMode::Wide { .. } => {
                    crate::training_queue::wide_seed_prompts()
                }
            };
            validate_materialized_opd_prompts("DistillPump mode", &prompts)?;
            let lora_rank = req.rank.unwrap_or(req.config.lora_rank);
            let admission = opd_preflight_admission(state, &prompts, &req.config, lora_rank)?;
            req.config.grad_checkpoint_segments = admission.checkpoint_segments;
            entry.reserved_bytes = admission.reserved_bytes;
            entry.prepared_data = PreparedTrainingData::DistillPumpPrompts(prompts);
        }
        QueuedJob::DistillSelf(req) => {
            if !matches!(entry.prepared_data, PreparedTrainingData::None) {
                return Err(ApiError::internal(
                    "DistillSelf queue entry carried unexpected prepared training data",
                ));
            }
            validate_opd_config_at_submit(&req.config)?;
            let prompts = req.prompts.as_deref().unwrap_or(&[]);
            validate_materialized_opd_prompts("DistillSelf request", prompts)?;
            let admission =
                opd_preflight_admission(state, prompts, &req.config, req.config.lora_rank)?;
            req.config.grad_checkpoint_segments = admission.checkpoint_segments;
            entry.reserved_bytes = admission.reserved_bytes;
            entry.prepared_data = PreparedTrainingData::None;
        }
    }
    Ok(())
}

/// Atomically reserve queue/tracking capacity and publish a complete batch.
/// A rejected batch leaves both the tracking map and FIFO unchanged.
fn admit_training_jobs_with_summary(
    state: &AppState,
    mut pending: Vec<(TrainingJobInfo, QueueEntry)>,
) -> Result<TrainingAdmissionResult, ApiError> {
    let _admission_guard = match state.training_data_admission_lock.try_lock() {
        Ok(guard) => guard,
        Err(std::sync::TryLockError::WouldBlock) => {
            return Err(ApiError::training_admission_busy());
        }
        Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
    };
    // Own the single process admission permit and normalize cheap immutable
    // metadata first. Capacity and optimizer support must reject before any
    // resume checkpoint is loaded or caller-controlled corpus is scanned.
    for (_, entry) in &mut pending {
        if let QueuedJob::Sft(req) = &mut entry.job {
            normalize_sft_config_at_submit(&mut req.config)?;
        }
    }
    pin_registered_teachers(state, &mut pending)?;
    enforce_queue_capacity_for(state, pending.len())?;
    ensure_training_backend_admission(state)?;
    for (info, entry) in &mut pending {
        enforce_queued_training_workload_admission(state, &entry.job)?;
        enforce_queued_training_optimizer_admission(state, &entry.job)?;
        if entry.admitted_resume_checkpoint.is_some() {
            return Err(ApiError::training_invalid_request(
                "training queue entry carried a caller-supplied resume checkpoint identity",
            ));
        }
        let resume_admission = crate::training_queue::materialize_queued_job_effective_seed(
            &mut entry.job,
            &state.adapter_dir,
            &info.adapter_name,
        )
        .map_err(ApiError::training_invalid_request)?;
        if let Some(recorded) = info.effective_seed
            && recorded != resume_admission.effective_seed
        {
            return Err(ApiError::training_invalid_request(format!(
                "training job seed {recorded} does not match materialized request seed {}",
                resume_admission.effective_seed
            )));
        }
        info.effective_seed = Some(resume_admission.effective_seed);
        entry.admitted_resume_checkpoint = resume_admission.checkpoint;
    }
    for (info, entry) in &mut pending {
        prepare_training_entry_admission(state, info, entry)?;
        acquire_training_entry_prepared_data_permit(state, entry)?;
    }
    let sft_summaries = pending
        .iter()
        .filter_map(|(info, entry)| {
            let crate::training_queue::PreparedTrainingData::Sft(prepared) = &entry.prepared_data
            else {
                return None;
            };
            Some((
                info.job_id.clone(),
                SftSubmissionStats {
                    rows_read: prepared.ingestion.rows_read,
                    num_examples: prepared.ingestion.rows_kept,
                    rows_rejected: prepared.ingestion.rows_rejected,
                    max_seq_len: prepared.max_seq_len,
                    max_supervised_tokens: prepared.max_supervised_tokens,
                    streaming_dataset: !matches!(
                        prepared.ingestion.source.as_str(),
                        "inline" | "corrections"
                    ),
                },
            ))
        })
        .collect();
    let effective_seeds = pending
        .iter()
        .map(|(info, _)| {
            info.effective_seed
                .map(|seed| (info.job_id.clone(), seed))
                .ok_or_else(|| {
                    ApiError::internal(format!(
                        "training job {} is missing its materialized effective seed",
                        info.job_id
                    ))
                })
        })
        .collect::<Result<std::collections::HashMap<_, _>, _>>()?;
    let queue_position = admit_training_jobs_into(
        &state.training_jobs,
        &state.training_queue,
        state.max_queued_training_jobs,
        state.max_tracked_jobs,
        !matches!(state.backend.as_ref(), ModelBackend::Mock { .. }),
        pending,
    )?;
    Ok(TrainingAdmissionResult {
        queue_position,
        sft_summaries,
        effective_seeds,
    })
}

pub(crate) fn admit_training_jobs(
    state: &AppState,
    pending: Vec<(TrainingJobInfo, QueueEntry)>,
) -> Result<usize, ApiError> {
    admit_training_jobs_with_summary(state, pending).map(|result| result.queue_position)
}

/// Return exact decimal seeds for a just-admitted group of jobs. Composite
/// entry points use this so the initial response carries the same immutable
/// provenance as the dedicated one-job endpoints.
pub(crate) fn admitted_training_seeds(
    state: &AppState,
    job_ids: &[String],
) -> Result<std::collections::BTreeMap<String, String>, ApiError> {
    let jobs = state.training_jobs.read().unwrap();
    job_ids
        .iter()
        .map(|job_id| {
            let seed = jobs
                .get(job_id)
                .and_then(|job| job.effective_seed)
                .ok_or_else(|| {
                    ApiError::internal(format!(
                        "admitted training job {job_id} is missing its effective seed"
                    ))
                })?;
            Ok((job_id.clone(), seed.to_string()))
        })
        .collect()
}

/// Shared registration+enqueue for distill_* endpoints. Same shape as
/// `submit_distill_refresh`/`submit_opd` but inlined for the simpler
/// distill variants.
fn register_and_enqueue_distill(
    state: &AppState,
    job_id: &str,
    adapter_name: &str,
    auto_load: bool,
    job: QueuedJob,
) -> Result<u64, ApiError> {
    let info = TrainingJobInfo {
        job_id: job_id.to_string(),
        adapter_name: adapter_name.to_string(),
        job_type: TrainingJobType::Opd,
        effective_seed: None,
        state: TrainingState::Queued,
        progress: 0.0,
        loss: None,
        epoch: None,
        adapter_path: None,
        submitted_at: std::time::Instant::now(),
        submitted_unix_ms: crate::recent_requests::now_unix_ms(),
        auto_load,
        consumed_correction_ids: Vec::new(),
        training_data: None,
        finished_at: None,
        finished_unix_ms: None,
        error: None,
        linked_eval_job_ids: Vec::new(),
        post_eval_verdict: None,
        gate_outcome: None,
        post_eval_gate_evidence: Vec::new(),
        loss_history: Vec::new(),
        cancel_requested: Default::default(),
    };
    let admission = admit_training_jobs_with_summary(
        state,
        vec![(
            info,
            QueueEntry {
                job_id: job_id.to_string(),
                reserved_bytes: 0,
                teacher_bindings: Vec::new(),
                admitted_resume_checkpoint: None,
                prepared_data: Default::default(),
                prepared_data_permit: Default::default(),
                job,
            },
        )],
    )?;
    admission.effective_seed(job_id)
}

fn training_status_from_info(j: &crate::state::TrainingJobInfo) -> TrainingStatus {
    TrainingStatus {
        job_id: j.job_id.clone(),
        state: j.state,
        progress: j.progress,
        current_loss: j.loss,
        adapter_name: Some(j.adapter_name.clone()),
        effective_seed: j.effective_seed.map(|seed| seed.to_string()),
        started_at: format!("{}s ago", j.submitted_at.elapsed().as_secs()),
        elapsed_secs: j.submitted_at.elapsed().as_secs_f64(),
        submitted_unix_ms: Some(j.submitted_unix_ms),
        finished_unix_ms: j.finished_unix_ms,
        job_type: Some(
            match j.job_type {
                TrainingJobType::Sft => "sft",
                TrainingJobType::Grpo => "grpo",
                TrainingJobType::Opd => "opd",
            }
            .into(),
        ),
        training_data: j.training_data.clone(),
        error: j.error.clone(),
        post_eval_verdict: j.post_eval_verdict.clone(),
        gate_outcome: j.gate_outcome.clone(),
        post_eval_gate_evidence: j.post_eval_gate_evidence.clone(),
    }
}

/// GET /v1/train/status — overall training status (list all tracked jobs).
async fn training_status(State(state): State<AppState>) -> Json<Vec<TrainingStatus>> {
    let jobs = state.training_jobs.read().unwrap();
    let statuses: Vec<TrainingStatus> = jobs.values().map(training_status_from_info).collect();
    Json(statuses)
}

/// GET /v1/train/status/:job_id — per-job status.
async fn job_status(
    State(state): State<AppState>,
    AxumPath(job_id): AxumPath<String>,
) -> Result<Json<TrainingStatus>, ApiError> {
    let jobs = state.training_jobs.read().unwrap();
    let job = jobs
        .get(&job_id)
        .ok_or_else(|| ApiError::training_job_not_found(&job_id))?;

    Ok(Json(training_status_from_info(job)))
}

/// GET /v1/train/queue — list queue contents organized by state.
async fn list_queue(State(state): State<AppState>) -> Json<QueueResponse> {
    let jobs = state.training_jobs.read().unwrap();
    let queue = state.training_queue.lock().unwrap();

    let mut running = None;
    let mut completed = Vec::new();

    for j in jobs.values() {
        let status = training_status_from_info(j);
        match j.state {
            TrainingState::Running => running = Some(status),
            TrainingState::Completed | TrainingState::Failed => completed.push(status),
            TrainingState::Queued => {} // handled from queue below
        }
    }

    // Build queued list from the actual queue (preserves FIFO order)
    let queued: Vec<QueueStatusEntry> = queue
        .queue
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let (job_type, adapter_name) = jobs
                .get(&entry.job_id)
                .map(|j| (j.job_type, j.adapter_name.clone()))
                .unwrap_or((TrainingJobType::Sft, "unknown".into()));
            QueueStatusEntry {
                job_id: entry.job_id.clone(),
                job_type,
                adapter_name,
                position: i + 1,
            }
        })
        .collect();

    // Sort completed by most-recently-finished first (falls back to submit
    // time when the terminal-transition timestamp is missing — e.g., an
    // archived entry that pre-dates the `finished_unix_ms` field).
    completed.sort_by(|a, b| {
        let a_t = a
            .finished_unix_ms
            .unwrap_or_else(|| a.submitted_unix_ms.unwrap_or(0));
        let b_t = b
            .finished_unix_ms
            .unwrap_or_else(|| b.submitted_unix_ms.unwrap_or(0));
        b_t.cmp(&a_t)
    });

    Json(QueueResponse {
        running,
        queued,
        completed,
    })
}

/// DELETE /v1/train/queue/:job_id — cancel a queued OR running job.
///
/// Queued: removed from the queue immediately. Running: the job's
/// cooperative cancel flag is set; the trainer aborts at the next step
/// boundary (typically one decode/optimizer step) and the job lands in
/// `Failed` with error "cancelled by user" and receipt failure_reason
/// "cancelled".
async fn cancel_queued_job(
    State(state): State<AppState>,
    AxumPath(job_id): AxumPath<String>,
) -> Result<Json<CancelTrainingJobResponse>, ApiError> {
    // Check job state; flag running jobs for cooperative cancellation.
    {
        let jobs = state.training_jobs.read().unwrap();
        let job = jobs
            .get(&job_id)
            .ok_or_else(|| ApiError::training_job_not_found(&job_id))?;
        if job.state == TrainingState::Running {
            job.cancel_requested
                .store(true, std::sync::atomic::Ordering::Relaxed);
            tracing::info!(job_id = %job_id, "cancellation requested for running training job");
            return Ok(Json(CancelTrainingJobResponse::Cancelling {
                job_id,
                message: "stop requested — the trainer aborts at the next step boundary",
            }));
        }
        if job.state != TrainingState::Queued {
            return Err(ApiError::training_job_not_cancellable(
                &job_id,
                format!("{:?}", job.state),
            ));
        }
    }

    // Remove from queue
    let removed = {
        let mut q = state.training_queue.lock().unwrap();
        q.remove(&job_id)
    };

    if removed {
        // Mark as failed (cancelled) in the tracking map
        let metric_type = {
            let mut jobs = state.training_jobs.write().unwrap();
            let jt = jobs.get(&job_id).map(|j| j.job_type);
            if let Some(job) = jobs.get_mut(&job_id) {
                job.state = TrainingState::Failed;
                job.error = Some("cancelled while queued".to_string());
                job.finished_at = Some(std::time::Instant::now());
                job.finished_unix_ms = Some(crate::recent_requests::now_unix_ms());
            }
            jt
        };
        if let Some(jt) = metric_type {
            let mt = match jt {
                TrainingJobType::Sft => TrainingMetricType::Sft,
                TrainingJobType::Grpo => TrainingMetricType::Grpo,
                TrainingJobType::Opd => TrainingMetricType::Opd,
            };
            state
                .metrics
                .inc_training(mt, TrainingMetricStatus::Cancelled);
        }
        Ok(Json(CancelTrainingJobResponse::Cancelled { job_id }))
    } else {
        Err(ApiError::training_job_already_started(&job_id))
    }
}

#[derive(Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum CancelTrainingJobResponse {
    Cancelling {
        job_id: String,
        message: &'static str,
    },
    Cancelled {
        job_id: String,
    },
}

/// DELETE /v1/train/jobs/:job_id — permanently delete a terminal training
/// job from both the in-memory tracking map and the on-disk archive. Refuses
/// to delete jobs that are still queued / running (use
/// `DELETE /v1/train/queue/:job_id` for those).
async fn delete_archived_job(
    State(state): State<AppState>,
    AxumPath(job_id): AxumPath<String>,
) -> Result<Json<DeleteTrainingJobResponse>, ApiError> {
    // Refuse if the job is still active. The in-memory map is the source
    // of truth for live state; an archived entry will only exist for jobs
    // already in a terminal state.
    {
        let jobs = state.training_jobs.read().unwrap();
        if let Some(job) = jobs.get(&job_id) {
            match job.state {
                TrainingState::Queued | TrainingState::Running => {
                    return Err(ApiError::training_job_not_cancellable(
                        &job_id,
                        format!("{:?}", job.state),
                    ));
                }
                _ => {}
            }
        }
        // Missing from in-memory but present on-disk is also valid — we'll
        // still try to delete the archive file below.
    }

    // Remove from in-memory map (idempotent — missing is fine).
    {
        let mut jobs = state.training_jobs.write().unwrap();
        jobs.remove(&job_id);
    }

    // Delete the on-disk archive file. Missing is fine (already gone).
    let archive_path =
        crate::training_history::archive_dir(&state.adapter_dir).join(format!("{job_id}.json"));
    let removed_file = match std::fs::remove_file(&archive_path) {
        Ok(_) => true,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
        Err(e) => {
            return Err(ApiError::internal(format!(
                "failed to delete archive file {}: {}",
                archive_path.display(),
                e
            )));
        }
    };

    Ok(Json(DeleteTrainingJobResponse {
        job_id,
        status: "deleted",
        removed_archive_file: removed_file,
    }))
}

#[derive(Serialize)]
struct DeleteTrainingJobResponse {
    job_id: String,
    status: &'static str,
    removed_archive_file: bool,
}

/// Rich detail payload exposed at `GET /v1/train/jobs/:job_id`. Flattens
/// `TrainingStatus` so the wire shape stays a superset (no field drift)
/// and adds the curve + back-references the UI's drill-in panel needs.
#[derive(Serialize)]
struct TrainingJobDetail {
    #[serde(flatten)]
    status: TrainingStatus,
    epoch: Option<u32>,
    adapter_path: Option<String>,
    auto_load: bool,
    /// Eval job IDs queued by `post_eval`. `None` when no post-eval was
    /// requested; otherwise newest-first.
    linked_eval_job_ids: Vec<String>,
    /// Time-series of progress samples. Empty until the trainer emits
    /// its first callback.
    loss_history: Vec<crate::state::TrainingLossSample>,
    /// Machine-readable training receipt from the adapter directory when
    /// present. This carries the resolved hyperparameters, data hashes,
    /// token counts, and backend audit trail.
    train_receipt: Option<serde_json::Value>,
    /// Replay request summary from `replay.jsonl`. Large inline datasets are
    /// reduced to counts so the drill-in remains usable.
    replay_request: Option<serde_json::Value>,
    /// Newest manifest-valid immutable checkpoint for this job's training
    /// type. The basename can be sent back as `config.resume_checkpoint`;
    /// submission performs full artifact/checksum validation before GPU work.
    latest_checkpoint: Option<TrainingCheckpointSummary>,
    /// Non-fatal discovery errors for checkpoint-shaped directories. A valid
    /// older checkpoint may still be returned in `latest_checkpoint`.
    checkpoint_error: Option<String>,
    /// Non-fatal metadata read/parse error. Missing metadata is represented
    /// by null fields rather than an error.
    metadata_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct TrainingCheckpointSummary {
    /// Stable basename accepted by the matching training config.
    resume_checkpoint: String,
    checkpoint_id: String,
    training_kind: kiln_train::checkpoint::TrainingKind,
    data_source_kind: String,
    global_step: u64,
    total_steps: u64,
    next_epoch_index: u64,
    next_cursor_in_epoch: u64,
    complete: bool,
    created_at: String,
    /// Exact resolved configuration stored in the validated manifest. The UI
    /// uses this rather than a best-effort reconstruction when preparing a
    /// resume form.
    effective_config: serde_json::Value,
    data_content_sha256: String,
    data_item_count: u64,
    /// OPD-only teacher provenance, extracted from the validated auxiliary
    /// state. Absent for SFT/GRPO and legacy OPD checkpoints. The identity
    /// revision is comparable with `GET /v1/teachers`; the content revision
    /// binds the exact numeric source and may instead hash materialized rows.
    #[serde(skip_serializing_if = "Option::is_none")]
    teacher_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    teacher_identity_revision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    teacher_content_revision: Option<String>,
}

fn checkpoint_teacher_identity_revision(
    auxiliary_state: &serde_json::Value,
) -> serde_json::Result<Option<String>> {
    let Some(value) = auxiliary_state.get("teacher_identity") else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let identity: kiln_train::TeacherIdentityV1 = serde_json::from_value(value.clone())?;
    Ok(Some(format!("sha256:{}", identity.content_revision())))
}

fn discover_latest_training_checkpoint(
    adapter_root: &Path,
    adapter_name: &str,
    expected_kind: kiln_train::checkpoint::TrainingKind,
) -> (Option<TrainingCheckpointSummary>, Option<String>) {
    const MAX_REPORTED_ERRORS: usize = 4;

    let entries = match std::fs::read_dir(adapter_root) {
        Ok(entries) => entries,
        Err(error) => {
            return (
                None,
                Some(format!(
                    "read checkpoint root {}: {error}",
                    adapter_root.display()
                )),
            );
        }
    };
    let prefix = format!("{adapter_name}-checkpoint-");
    let suffix = kiln_train::checkpoint::TRAINING_CHECKPOINT_DIRECTORY_SUFFIX;
    let mut candidates = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            (name.starts_with(&prefix) && name.ends_with(suffix)).then_some((name, entry.path()))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.0.cmp(&right.0));

    let mut latest: Option<TrainingCheckpointSummary> = None;
    let mut errors = Vec::new();
    let mut omitted_errors = 0_usize;
    for (name, path) in candidates {
        let manifest = match kiln_train::checkpoint::read_training_checkpoint_manifest(&path) {
            Ok(manifest) => manifest,
            Err(error) => {
                if errors.len() < MAX_REPORTED_ERRORS {
                    errors.push(format!("{name}: {error:#}"));
                } else {
                    omitted_errors += 1;
                }
                continue;
            }
        };
        if manifest.adapter_name != adapter_name {
            if errors.len() < MAX_REPORTED_ERRORS {
                errors.push(format!(
                    "{name}: manifest adapter {:?} does not match {:?}",
                    manifest.adapter_name, adapter_name
                ));
            } else {
                omitted_errors += 1;
            }
            continue;
        }
        if manifest.training_kind != expected_kind {
            continue;
        }
        if let Err(error) = kiln_train::checkpoint::validate_exact_checkpoint_artifact_provenance(
            &manifest.auxiliary_state,
        ) {
            if errors.len() < MAX_REPORTED_ERRORS {
                errors.push(format!(
                    "{name}: invalid exact checkpoint artifact provenance: {error:#}"
                ));
            } else {
                omitted_errors += 1;
            }
            continue;
        }
        let teacher_id = manifest
            .auxiliary_state
            .pointer("/teacher_capabilities/teacher_id")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
        let teacher_content_revision = manifest
            .auxiliary_state
            .get("teacher_content_revision")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
        let teacher_identity_revision =
            match checkpoint_teacher_identity_revision(&manifest.auxiliary_state) {
                Ok(revision) => revision,
                Err(error) => {
                    if errors.len() < MAX_REPORTED_ERRORS {
                        errors.push(format!(
                            "{name}: invalid checkpoint teacher_identity: {error}"
                        ));
                    } else {
                        omitted_errors += 1;
                    }
                    continue;
                }
            };
        let summary = TrainingCheckpointSummary {
            resume_checkpoint: name,
            checkpoint_id: manifest.checkpoint_id,
            training_kind: manifest.training_kind,
            data_source_kind: manifest.data.source_kind,
            global_step: manifest.progress.global_step,
            total_steps: manifest.progress.total_steps,
            next_epoch_index: manifest.progress.epoch_index,
            next_cursor_in_epoch: manifest.progress.cursor_in_epoch,
            complete: manifest.progress.global_step == manifest.progress.total_steps,
            created_at: manifest.created_at,
            effective_config: manifest.effective_config,
            data_content_sha256: manifest.data.content_sha256,
            data_item_count: manifest.data.item_count,
            teacher_id,
            teacher_identity_revision,
            teacher_content_revision,
        };
        let replace = latest.as_ref().is_none_or(|current| {
            (
                summary.global_step,
                &summary.created_at,
                &summary.resume_checkpoint,
            ) > (
                current.global_step,
                &current.created_at,
                &current.resume_checkpoint,
            )
        });
        if replace {
            latest = Some(summary);
        }
    }
    if omitted_errors > 0 {
        errors.push(format!(
            "{omitted_errors} additional checkpoint errors omitted"
        ));
    }
    let error = (!errors.is_empty()).then(|| errors.join("; "));
    (latest, error)
}

fn training_job_adapter_dir(
    adapter_root: &Path,
    adapter_name: &str,
    adapter_path: Option<&str>,
) -> PathBuf {
    if let Some(path) = adapter_path {
        let path = PathBuf::from(path);
        if path.is_absolute() || path.exists() {
            return path;
        }
    }
    adapter_root.join(adapter_name)
}

fn summarize_replay_request(mut value: serde_json::Value) -> serde_json::Value {
    if let Some(body) = value
        .as_object_mut()
        .and_then(|obj| obj.get_mut("request_body"))
        .and_then(|body| body.as_object_mut())
    {
        for key in ["examples", "groups", "prompts"] {
            if let Some(rows) = body.remove(key) {
                let count = rows.as_array().map(|rows| rows.len()).unwrap_or(0);
                body.insert(format!("{key}_count"), serde_json::json!(count));
            }
        }
    }
    value
}

fn read_replay_request(
    adapter_dir: &Path,
    job_id: &str,
) -> Result<Option<serde_json::Value>, String> {
    let replay_path = adapter_dir.join("replay.jsonl");
    let file = match std::fs::File::open(&replay_path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("read {}: {e}", replay_path.display())),
    };
    use std::io::BufRead;
    let mut first_request = None;
    for (idx, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line =
            line.map_err(|e| format!("read {} line {}: {e}", replay_path.display(), idx + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(trimmed)
            .map_err(|e| format!("parse {} line {}: {e}", replay_path.display(), idx + 1))?;
        if value.get("type").and_then(|v| v.as_str()) != Some("request") {
            continue;
        }
        let summary = summarize_replay_request(value);
        if summary.get("request_id").and_then(|v| v.as_str()) == Some(job_id) {
            return Ok(Some(summary));
        }
        if first_request.is_none() {
            first_request = Some(summary);
        }
    }
    Ok(first_request)
}

fn read_training_job_metadata(
    adapter_dir: &Path,
    job_id: &str,
) -> (
    Option<serde_json::Value>,
    Option<serde_json::Value>,
    Option<String>,
) {
    let mut errors = Vec::new();
    let train_receipt = match kiln_train::TrainReceipt::read_from_adapter_dir(adapter_dir) {
        Ok(Some(receipt)) => match serde_json::to_value(receipt) {
            Ok(value) => Some(value),
            Err(err) => {
                errors.push(format!("serialize validated train receipt: {err}"));
                None
            }
        },
        Ok(None) => None,
        Err(err) => {
            errors.push(format!("read validated train receipt: {err:#}"));
            None
        }
    };
    let replay_request = match read_replay_request(adapter_dir, job_id) {
        Ok(value) => value,
        Err(err) => {
            errors.push(err);
            None
        }
    };
    let metadata_error = if errors.is_empty() {
        None
    } else {
        Some(errors.join("; "))
    };
    (train_receipt, replay_request, metadata_error)
}

async fn job_detail(
    State(state): State<AppState>,
    AxumPath(job_id): AxumPath<String>,
) -> Result<Json<TrainingJobDetail>, ApiError> {
    // Build the response inside a tight read-lock scope and drop the
    // guard before serializing. The drill-modal poll runs at 1.5s and
    // clones up to 1024 loss samples here; the trainer's progress
    // callback contends for the WRITE lock on every step. Building
    // outside a `let _ = jobs;` would extend the borrow until end of
    // function.
    let (mut detail, metadata_dir, checkpoint_kind) = {
        let jobs = state.training_jobs.read().unwrap();
        let job = jobs
            .get(&job_id)
            .ok_or_else(|| ApiError::training_job_not_found(&job_id))?;
        let checkpoint_kind = match job.job_type {
            TrainingJobType::Sft => Some(kiln_train::checkpoint::TrainingKind::Sft),
            TrainingJobType::Grpo => Some(kiln_train::checkpoint::TrainingKind::Grpo),
            TrainingJobType::Opd => Some(kiln_train::checkpoint::TrainingKind::Opd),
        };
        (
            TrainingJobDetail {
                status: training_status_from_info(job),
                epoch: job.epoch,
                adapter_path: job.adapter_path.clone(),
                auto_load: job.auto_load,
                linked_eval_job_ids: job.linked_eval_job_ids.clone(),
                loss_history: job.loss_history.clone(),
                train_receipt: None,
                replay_request: None,
                latest_checkpoint: None,
                checkpoint_error: None,
                metadata_error: None,
            },
            training_job_adapter_dir(
                &state.adapter_dir,
                &job.adapter_name,
                job.adapter_path.as_deref(),
            ),
            checkpoint_kind,
        )
    };
    let (train_receipt, replay_request, metadata_error) =
        read_training_job_metadata(&metadata_dir, &job_id);
    detail.train_receipt = train_receipt;
    detail.replay_request = replay_request;
    detail.metadata_error = metadata_error;
    if let (Some(expected_kind), Some(adapter_name)) =
        (checkpoint_kind, detail.status.adapter_name.as_deref())
    {
        (detail.latest_checkpoint, detail.checkpoint_error) =
            discover_latest_training_checkpoint(&state.adapter_dir, adapter_name, expected_kind);
    }
    Ok(Json(detail))
}

const TRAINING_REQUEST_BODY_LIMIT_BYTES: usize = 64 * 1024 * 1024;

fn routes_with_body_limit(limit_bytes: usize) -> Router<AppState> {
    Router::new()
        .route(
            "/v1/train/sft",
            post(submit_sft).layer(DefaultBodyLimit::max(limit_bytes)),
        )
        .route(
            "/v1/train/grpo",
            post(submit_grpo).layer(DefaultBodyLimit::max(limit_bytes)),
        )
        .route(
            "/v1/training/grpo",
            post(submit_grpo).layer(DefaultBodyLimit::max(limit_bytes)),
        )
        // Canonical alias for /v1/train/grpo after the ECHO trajectory
        // schema landing. The "agentic" name reflects what the endpoint
        // actually trains: multi-turn rollouts with action/observation
        // segments. Both routes serve the same handler; legacy callers
        // keep working unchanged.
        .route(
            "/v1/train/agentic",
            post(submit_grpo).layer(DefaultBodyLimit::max(limit_bytes)),
        )
        .route(
            "/v1/train/opd",
            post(submit_opd).layer(DefaultBodyLimit::max(limit_bytes)),
        )
        .route(
            "/v1/distill/refresh",
            post(submit_distill_refresh).layer(DefaultBodyLimit::max(limit_bytes)),
        )
        .route(
            "/v1/adapters/distill_merge",
            post(submit_distill_merge).layer(DefaultBodyLimit::max(limit_bytes)),
        )
        .route(
            "/v1/distill/pump",
            post(submit_distill_pump).layer(DefaultBodyLimit::max(limit_bytes)),
        )
        .route(
            "/v1/distill/self",
            post(submit_distill_self).layer(DefaultBodyLimit::max(limit_bytes)),
        )
        .route("/v1/train/status", get(training_status))
        .route("/v1/train/status/{job_id}", get(job_status))
        .route(
            "/v1/train/jobs/{job_id}",
            get(job_detail).delete(delete_archived_job),
        )
        .route("/v1/train/queue", get(list_queue))
        .route("/v1/train/queue/{job_id}", delete(cancel_queued_job))
}

pub fn routes() -> Router<AppState> {
    routes_with_body_limit(TRAINING_REQUEST_BODY_LIMIT_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_train::checkpoint::{
        CheckpointArtifact, CheckpointFileRole, TrainingCheckpointData, TrainingCheckpointManifest,
        TrainingCheckpointOptimizer, TrainingCheckpointPrecision, TrainingCheckpointProgress,
        TrainingCheckpointScheduler, TrainingCheckpointStateFiles, TrainingKind,
        write_training_checkpoint_atomic,
    };
    use kiln_train::opd::StableOpdMode;
    use kiln_train::opd::{OpdConfig, OpdPrompt};
    use kiln_train::{
        ChatMessage, GrpoConfig, OpdLossGranularity, OpdObjective, ScoredCompletion, SftConfig,
        SftExample, SftInvalidRowPolicy,
    };
    use std::sync::{Arc, Barrier, RwLock};
    use tower::ServiceExt;

    use crate::TEST_ENV_LOCK as ENV_LOCK;

    #[test]
    fn optimizer_support_errors_preserve_structured_http_semantics() {
        let substrate = training_optimizer_support_api_error(
            "cuda",
            kiln_model::TrainingOptimizerSupportError::UnsupportedBaseWeightDType {
                actual: kiln_tensor::DType::F16,
                supported: &[kiln_tensor::DType::F32, kiln_tensor::DType::BF16],
            },
        );
        assert_eq!(substrate.code, "training_backend_unsupported");
        assert_eq!(substrate.status, axum::http::StatusCode::NOT_IMPLEMENTED);

        let request = training_optimizer_support_api_error(
            "metal",
            kiln_model::TrainingOptimizerSupportError::UnsupportedRequest {
                request: kiln_model::TrainingOptimizerRequest {
                    kind: kiln_model::TrainingOptimizerKind::Sgd,
                    parameter_dtype: kiln_tensor::DType::BF16,
                    rounding: kiln_model::TrainingOptimizerRounding::RoundToNearest,
                    lora_rank: 4,
                },
                supported_dtypes: &[],
                supported_rounding: &[kiln_model::TrainingOptimizerRounding::RoundToNearest],
                muon_min_lora_rank: Some(2),
                muon_max_lora_rank: Some(32),
            },
        );
        assert_eq!(request.code, "training_invalid_request");
        assert_eq!(request.status, axum::http::StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn training_routes_reject_oversized_json_before_handler_admission() {
        let app = routes_with_body_limit(128).with_state(teacher_binding_test_state());
        for route in [
            "/v1/train/sft",
            "/v1/train/grpo",
            "/v1/training/grpo",
            "/v1/train/agentic",
            "/v1/train/opd",
            "/v1/distill/refresh",
            "/v1/adapters/distill_merge",
            "/v1/distill/pump",
            "/v1/distill/self",
        ] {
            let response = app
                .clone()
                .oneshot(
                    axum::http::Request::post(route)
                        .header(axum::http::header::CONTENT_TYPE, "application/json")
                        .body(axum::body::Body::from(vec![b' '; 129]))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                axum::http::StatusCode::PAYLOAD_TOO_LARGE,
                "{route}"
            );
            let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
                .await
                .unwrap();
            let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(
                body["error"]["code"], "training_request_too_large",
                "{route}: {body}"
            );
        }
    }

    #[test]
    fn correction_ids_follow_the_server_owned_skip_manifest() {
        let tokenizer = crate::api::test_tokenizer().with_chat_template(
            "{% for message in messages %}{{ message.content }}{% endfor %}".to_string(),
        );
        let valid = |user: &str, assistant: &str| SftExample {
            messages: vec![
                ChatMessage::new("user", user),
                ChatMessage::new("assistant", assistant),
            ],
        };
        let prepared = kiln_train::prepare_sft_examples(
            vec![
                valid("a", "b"),
                SftExample { messages: vec![] },
                valid("b", "a"),
            ],
            &tokenizer,
            SftInvalidRowPolicy::Skip,
            "corrections",
            None,
        )
        .unwrap();

        let retained = retained_correction_ids(
            vec!["first".into(), "rejected".into(), "third".into()],
            &prepared.ingestion,
        )
        .unwrap();
        assert_eq!(retained, ["first", "third"]);

        let error =
            retained_correction_ids(vec!["first".into(), "rejected".into()], &prepared.ingestion)
                .unwrap_err();
        assert_eq!(error.code, "internal_error");
        assert!(
            error
                .message
                .contains("row IDs (2) differ from ingested rows (3)")
        );
    }

    #[test]
    fn distill_merge_rejects_unimplemented_weight_or_budget_semantics() {
        let mut request: DistillMergeRequest = serde_json::from_value(serde_json::json!({
            "name": "merged",
            "sources": [
                {"adapter": "one", "weight": 1.0},
                {"adapter": "two", "weight": 2.0}
            ],
            "rollout_budget": 6,
            "config": {"epochs": 2}
        }))
        .unwrap();
        assert_eq!(
            validate_distill_merge_sampling_contract(&request, &[1, 2]).unwrap(),
            6
        );

        request.sources[1].weight = 1.0;
        let error = validate_distill_merge_sampling_contract(&request, &[1, 2]).unwrap_err();
        assert!(error.message.contains("cannot silently approximate"));

        request.sources[1].weight = 2.0;
        request.rollout_budget = 5;
        let error = validate_distill_merge_sampling_contract(&request, &[1, 2]).unwrap_err();
        assert!(error.message.contains("rollout_budget=6"));
    }

    fn discovery_teacher_identity() -> kiln_train::TeacherIdentityV1 {
        kiln_train::TeacherIdentityV1::new(
            "teacher-v1-model",
            "aa".repeat(32),
            "bb".repeat(32),
            "cc".repeat(32),
            None,
            32,
            16,
            128,
            256,
            "kiln-checkpoint-discovery-test",
            "dd".repeat(32),
        )
        .unwrap()
    }

    fn write_discovery_checkpoint(
        root: &Path,
        directory_name: &str,
        adapter_name: &str,
        training_kind: TrainingKind,
        global_step: u64,
        total_steps: u64,
    ) {
        let data_order = (global_step < total_steps)
            .then_some(vec![0])
            .unwrap_or_default();
        let base_weight_manifest = kiln_core::model_provenance::BaseWeightShardManifest::new(vec![
            kiln_core::model_provenance::BaseWeightShardIdentity::from_digest(
                "model.safetensors",
                11,
                [0x42; 32],
            )
            .unwrap(),
        ])
        .unwrap();
        let mut auxiliary_state = serde_json::json!({
            "base_model_weights_sha256": base_weight_manifest.aggregate_sha256.clone(),
            "base_weight_shard_manifest": base_weight_manifest,
            "execution_provenance": crate::execution_provenance::test_execution_provenance(),
        });
        if training_kind == TrainingKind::Opd {
            let object = auxiliary_state.as_object_mut().unwrap();
            object.insert(
                "teacher_capabilities".to_string(),
                serde_json::json!({"teacher_id": "teacher-v1"}),
            );
            object.insert(
                "teacher_identity".to_string(),
                serde_json::to_value(discovery_teacher_identity()).unwrap(),
            );
            object.insert(
                "teacher_content_revision".to_string(),
                serde_json::json!(format!("sha256:{}", "22".repeat(32))),
            );
        }
        let manifest = TrainingCheckpointManifest::new(
            format!("checkpoint-{global_step}"),
            training_kind,
            adapter_name,
            serde_json::json!({"epochs": total_steps}),
            TrainingCheckpointPrecision {
                parameter_dtype: "f32".into(),
                optimizer_state_dtype: "none".into(),
                activation_dtype: "f32".into(),
                gradient_dtype: "f32".into(),
                stochastic_rounding: serde_json::json!({"mode": "round_to_nearest"}),
            },
            TrainingCheckpointProgress {
                global_step,
                total_steps,
                epoch_index: global_step,
                cursor_in_epoch: 0,
                data_order,
            },
            TrainingCheckpointData {
                source_kind: "test".into(),
                content_sha256: "11".repeat(32),
                item_count: 1,
            },
            Default::default(),
            TrainingCheckpointOptimizer {
                kind: "sgd".into(),
                step: global_step,
                hyperparameters: serde_json::json!({"learning_rate": 0.1}),
                state_file: None,
            },
            TrainingCheckpointScheduler {
                kind: "constant".into(),
                step: global_step,
                state: serde_json::json!({"learning_rate": 0.1}),
            },
            TrainingCheckpointStateFiles {
                adapter_parameters: "adapter.safetensors".into(),
                optimizer_state: None,
                reference_state: None,
                ema_state: None,
                reward_normalization_state: None,
                loss_history: None,
            },
            auxiliary_state,
        );
        let artifacts = [CheckpointArtifact {
            relative_path: "adapter.safetensors".into(),
            role: CheckpointFileRole::AdapterParameters,
        }];
        write_training_checkpoint_atomic(
            &root.join(directory_name),
            manifest,
            &artifacts,
            |staging| {
                std::fs::write(staging.join("adapter.safetensors"), b"test adapter")?;
                Ok(())
            },
        )
        .unwrap();
    }

    #[test]
    fn checkpoint_discovery_returns_latest_resume_basename_for_every_training_kind() {
        let temp = tempfile::tempdir().unwrap();
        write_discovery_checkpoint(
            temp.path(),
            "demo-checkpoint-step-00000002.kiln-checkpoint",
            "demo",
            TrainingKind::Sft,
            2,
            8,
        );
        write_discovery_checkpoint(
            temp.path(),
            "demo-checkpoint-step-00000004.kiln-checkpoint",
            "demo",
            TrainingKind::Sft,
            4,
            8,
        );
        write_discovery_checkpoint(
            temp.path(),
            "demo-checkpoint-step-00000006.kiln-checkpoint",
            "demo",
            TrainingKind::Grpo,
            6,
            8,
        );
        write_discovery_checkpoint(
            temp.path(),
            "demo-checkpoint-step-00000007.kiln-checkpoint",
            "demo",
            TrainingKind::Opd,
            7,
            8,
        );
        let corrupt = temp
            .path()
            .join("demo-checkpoint-step-00000009.kiln-checkpoint");
        std::fs::create_dir(&corrupt).unwrap();
        std::fs::write(corrupt.join("checkpoint_manifest.json"), b"{}").unwrap();

        let (latest, error) =
            discover_latest_training_checkpoint(temp.path(), "demo", TrainingKind::Sft);
        let latest = latest.expect("latest valid SFT checkpoint");
        assert_eq!(latest.training_kind, TrainingKind::Sft);
        assert_eq!(latest.data_source_kind, "test");
        assert_eq!(latest.global_step, 4);
        assert_eq!(latest.total_steps, 8);
        assert_eq!(
            latest.resume_checkpoint,
            "demo-checkpoint-step-00000004.kiln-checkpoint"
        );
        assert!(!latest.complete);
        assert!(
            error
                .as_deref()
                .is_some_and(|error| error.contains("step-00000009")),
            "corrupt checkpoint candidates must remain visible to operators"
        );

        let (latest, error) =
            discover_latest_training_checkpoint(temp.path(), "demo", TrainingKind::Grpo);
        let latest = latest.expect("latest valid GRPO checkpoint");
        assert_eq!(latest.training_kind, TrainingKind::Grpo);
        assert_eq!(latest.global_step, 6);
        assert_eq!(
            latest.resume_checkpoint,
            "demo-checkpoint-step-00000006.kiln-checkpoint"
        );
        assert!(
            error
                .as_deref()
                .is_some_and(|error| error.contains("step-00000009"))
        );

        let (latest, error) =
            discover_latest_training_checkpoint(temp.path(), "demo", TrainingKind::Opd);
        let latest = latest.expect("latest valid OPD checkpoint");
        assert_eq!(latest.training_kind, TrainingKind::Opd);
        assert_eq!(latest.global_step, 7);
        assert_eq!(latest.effective_config, serde_json::json!({"epochs": 8}));
        assert_eq!(latest.data_content_sha256, "11".repeat(32));
        assert_eq!(latest.data_item_count, 1);
        assert_eq!(latest.teacher_id.as_deref(), Some("teacher-v1"));
        let expected_teacher_identity_revision =
            format!("sha256:{}", discovery_teacher_identity().content_revision());
        assert_eq!(
            latest.teacher_identity_revision.as_deref(),
            Some(expected_teacher_identity_revision.as_str())
        );
        let expected_teacher_content_revision = format!("sha256:{}", "22".repeat(32));
        assert_eq!(
            latest.teacher_content_revision.as_deref(),
            Some(expected_teacher_content_revision.as_str())
        );
        assert_ne!(
            latest.teacher_identity_revision, latest.teacher_content_revision,
            "model identity and exact numeric-source content are distinct contracts"
        );
        assert_eq!(
            latest.resume_checkpoint,
            "demo-checkpoint-step-00000007.kiln-checkpoint"
        );
        assert!(
            error
                .as_deref()
                .is_some_and(|error| error.contains("step-00000009"))
        );
    }

    #[test]
    fn checkpoint_discovery_rejects_malformed_teacher_identity() {
        let temp = tempfile::tempdir().unwrap();
        let directory_name = "demo-checkpoint-step-00000002.kiln-checkpoint";
        write_discovery_checkpoint(temp.path(), directory_name, "demo", TrainingKind::Opd, 2, 8);
        let manifest_path = temp
            .path()
            .join(directory_name)
            .join(kiln_train::checkpoint::TRAINING_CHECKPOINT_MANIFEST_FILENAME);
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        manifest["auxiliary_state"]["teacher_identity"] =
            serde_json::json!({"schema": "not-a-teacher-identity"});
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let (latest, error) =
            discover_latest_training_checkpoint(temp.path(), "demo", TrainingKind::Opd);
        assert!(latest.is_none(), "invalid identity must not be resumable");
        assert!(
            error
                .as_deref()
                .is_some_and(|error| error.contains("invalid checkpoint teacher_identity")),
            "identity rejection must remain visible to the operator: {error:?}"
        );
    }

    #[test]
    fn checkpoint_discovery_rejects_tampered_execution_provenance() {
        let temp = tempfile::tempdir().unwrap();
        let directory_name = "demo-checkpoint-step-00000002.kiln-checkpoint";
        write_discovery_checkpoint(temp.path(), directory_name, "demo", TrainingKind::Sft, 2, 8);
        let manifest_path = temp
            .path()
            .join(directory_name)
            .join(kiln_train::checkpoint::TRAINING_CHECKPOINT_MANIFEST_FILENAME);
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        manifest["auxiliary_state"]["execution_provenance"]["backend"]["device"] =
            serde_json::json!("tampered:0");
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let (latest, error) =
            discover_latest_training_checkpoint(temp.path(), "demo", TrainingKind::Sft);
        assert!(latest.is_none(), "tampered execution must not be resumable");
        assert!(
            error.as_deref().is_some_and(|error| {
                error.contains("invalid exact checkpoint artifact provenance")
                    && error.contains("execution provenance digest mismatch")
            }),
            "execution-provenance rejection must remain visible to the operator: {error:?}"
        );
    }

    #[test]
    fn training_metadata_rejects_tampered_receipt_execution_provenance() {
        let temp = tempfile::tempdir().unwrap();
        let tokenizer = kiln_core::tokenizer::KilnTokenizer::from_bytes(
            br#"{
                "version":"1.0",
                "model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[]}
            }"#,
        )
        .unwrap();
        let mut receipt = kiln_train::TrainReceipt::new(
            "demo",
            "sft",
            &kiln_core::config::ModelConfig::qwen3_5_4b(),
            &tokenizer,
            kiln_train::train_receipt::HyperparameterReceipt {
                mode: "sft".to_string(),
                rank: 8,
                alpha: 16.0,
                alpha_over_rank: Some(2.0),
                learning_rate: 1e-4,
                epochs: 1,
                seed: Some(17),
                shuffle: false,
            },
            serde_json::json!({}),
        );
        receipt.runtime.execution_provenance =
            Some(crate::execution_provenance::test_execution_provenance());
        let path = temp.path().join(kiln_train::TRAIN_RECEIPT_FILENAME);
        std::fs::write(&path, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();

        let (loaded, _, error) = read_training_job_metadata(temp.path(), "job");
        assert!(loaded.is_some());
        assert_eq!(error, None);

        let mut value = serde_json::to_value(receipt).unwrap();
        value["runtime"]["execution_provenance"]["backend"]["device"] =
            serde_json::json!("tampered:0");
        std::fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        let (loaded, _, error) = read_training_job_metadata(temp.path(), "job");
        assert!(loaded.is_none());
        assert!(
            error
                .as_deref()
                .is_some_and(|error| error.contains("execution provenance digest mismatch")),
            "tampered receipt rejection must remain visible to the operator: {error:?}"
        );
    }

    fn pending_sft_job(job_id: impl Into<String>) -> (TrainingJobInfo, QueueEntry) {
        let job_id = job_id.into();
        let info = TrainingJobInfo {
            job_id: job_id.clone(),
            adapter_name: format!("adapter-{job_id}"),
            job_type: TrainingJobType::Sft,
            effective_seed: None,
            state: TrainingState::Queued,
            progress: 0.0,
            loss: None,
            epoch: None,
            adapter_path: None,
            submitted_at: std::time::Instant::now(),
            submitted_unix_ms: crate::recent_requests::now_unix_ms(),
            auto_load: false,
            consumed_correction_ids: Vec::new(),
            training_data: None,
            finished_at: None,
            finished_unix_ms: None,
            error: None,
            linked_eval_job_ids: Vec::new(),
            post_eval_verdict: None,
            gate_outcome: None,
            post_eval_gate_evidence: Vec::new(),
            loss_history: Vec::new(),
            cancel_requested: Default::default(),
        };
        let request = SftRequest {
            examples: Vec::new(),
            dataset_path: None,
            dataset: None,
            dataset_split: None,
            config: SftConfig::default(),
            ingestion: None,
            post_eval: None,
        };
        (
            info,
            QueueEntry {
                job_id,
                reserved_bytes: 0,
                teacher_bindings: Vec::new(),
                admitted_resume_checkpoint: None,
                prepared_data: Default::default(),
                prepared_data_permit: Default::default(),
                job: QueuedJob::Sft(request),
            },
        )
    }

    #[test]
    fn server_sft_normalizes_implicit_mtp_training_to_disabled() {
        let mut config = SftConfig::default();
        assert_eq!(config.train_mtp, None);

        normalize_sft_config_at_submit(&mut config).unwrap();

        assert_eq!(config.train_mtp, Some(false));
    }

    #[test]
    fn server_sft_rejects_explicit_mtp_training() {
        let mut config = SftConfig {
            train_mtp: Some(true),
            ..SftConfig::default()
        };

        let error = normalize_sft_config_at_submit(&mut config).unwrap_err();

        assert_eq!(error.code, "training_invalid_request");
        assert!(error.message.contains("train_mtp=true"), "{error:?}");
        assert!(error.message.contains("GPU-ownership"), "{error:?}");
    }

    #[test]
    fn central_admission_rejects_mtp_sft_without_publication() {
        let state = teacher_binding_test_state();
        let (info, mut entry) = pending_sft_job("mtp-sft");
        let QueuedJob::Sft(request) = &mut entry.job else {
            unreachable!("pending_sft_job must construct SFT")
        };
        request.config.train_mtp = Some(true);

        let error = admit_training_jobs(&state, vec![(info, entry)]).unwrap_err();

        assert_eq!(error.code, "training_invalid_request");
        assert!(error.message.contains("train_mtp=true"), "{error:?}");
        assert!(state.training_jobs.read().unwrap().is_empty());
        assert_eq!(state.training_queue.lock().unwrap().len(), 0);
    }

    #[test]
    fn central_admission_rejects_invalid_native_sft_without_publication() {
        let state = teacher_binding_test_state();
        let (info, mut entry) = pending_sft_job("invalid-sft-profile");
        let QueuedJob::Sft(request) = &mut entry.job else {
            unreachable!("pending_sft_job must construct SFT")
        };
        request.config.epochs = 0;

        let error = admit_training_jobs(&state, vec![(info, entry)]).unwrap_err();
        assert_eq!(error.code, "training_invalid_request");
        assert!(error.message.contains("epochs"), "{error:?}");
        assert!(state.training_jobs.read().unwrap().is_empty());
        assert_eq!(state.training_queue.lock().unwrap().len(), 0);
    }

    #[test]
    fn training_seed_status_and_composite_receipts_preserve_all_u64_bits() {
        let state = teacher_binding_test_state();
        let (mut info, _) = pending_sft_job("seeded");
        info.effective_seed = Some(u64::MAX);
        let status = training_status_from_info(&info);
        assert_eq!(
            serde_json::to_value(status).unwrap()["effective_seed"],
            u64::MAX.to_string()
        );
        state
            .training_jobs
            .write()
            .unwrap()
            .insert(info.job_id.clone(), info);
        assert_eq!(
            admitted_training_seeds(&state, &["seeded".into()])
                .unwrap()
                .get("seeded")
                .map(String::as_str),
            Some("18446744073709551615")
        );
    }

    fn teacher_binding_test_state() -> AppState {
        let mut config = kiln_core::config::ModelConfig::qwen3_5_4b();
        config.vocab_size = 2;
        let scheduler = kiln_scheduler::Scheduler::new(
            kiln_scheduler::SchedulerConfig {
                max_batch_tokens: 1024,
                max_batch_size: 1,
                block_size: 16,
                ..Default::default()
            },
            32,
        );
        let engine = kiln_model::engine::MockEngine::new(config.clone());
        let tokenizer = kiln_core::tokenizer::KilnTokenizer::from_bytes(
            br#"{
                "version":"1.0",
                "model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[]}
            }"#,
        )
        .unwrap();
        AppState::new_mock(
            config,
            scheduler,
            Arc::new(engine),
            tokenizer,
            60,
            "binding-test".into(),
        )
    }

    #[test]
    fn train_set_eval_cannot_be_a_promotion_gate() {
        let state = teacher_binding_test_state();
        let config = kiln_eval::PostEvalConfig {
            suite: "diagnostic".to_string(),
            data_scope: kiln_eval::PostEvalDataScope::TrainSetEval,
            generation: None,
            min_accuracy: Some(0.8),
            include_baseline: false,
        };
        let error = validate_post_eval_suite(&state, Some(&config)).unwrap_err();
        assert_eq!(error.code, "training_invalid_request");
        assert!(error.message.contains("diagnostic only"));
    }

    #[test]
    fn promotion_accuracy_floor_must_match_the_published_unit_interval() {
        let state = teacher_binding_test_state();
        for min_accuracy in [-0.01, 1.01, f32::NAN] {
            let config = kiln_eval::PostEvalConfig {
                suite: "held-out".to_string(),
                data_scope: kiln_eval::PostEvalDataScope::HeldOut,
                generation: None,
                min_accuracy: Some(min_accuracy),
                include_baseline: false,
            };

            let error = validate_post_eval_suite(&state, Some(&config)).unwrap_err();
            assert_eq!(error.code, "training_invalid_request");
            assert!(error.message.contains("finite and in [0.0, 1.0]"));
        }
    }

    #[test]
    fn named_training_defaults_to_persisted_train_partition() {
        let temp = tempfile::tempdir().unwrap();
        let registry = crate::eval::DatasetRegistry::new(temp.path().join("datasets"));
        let rows = (0..9)
            .map(|index| {
                serde_json::json!({
                    "group_id": format!("group-{index}"),
                    "messages": [
                        {"role": "user", "content": format!("prompt {index}")},
                        {"role": "assistant", "content": format!("answer {index}")}
                    ]
                })
                .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n");
        registry
            .create(
                "partitioned",
                crate::eval::DatasetFormat::SftChat,
                None,
                rows.as_bytes(),
            )
            .unwrap();
        let mut state = teacher_binding_test_state();
        state.dataset_registry = Some(Arc::new(registry));

        let source = resolve_named_training_dataset(
            &state,
            "partitioned",
            None,
            crate::eval::DatasetFormat::SftChat,
        )
        .unwrap();
        assert_eq!(source.split, kiln_eval::DatasetSplit::Train);
        assert_eq!(
            source.path.file_name().and_then(|name| name.to_str()),
            Some("train.jsonl")
        );
        assert_eq!(
            source
                .split_manifest
                .rows
                .iter()
                .filter(|row| row.split == kiln_eval::DatasetSplit::Train)
                .count() as u64,
            source.manifest.split_counts.train
        );
    }

    #[test]
    fn sft_contamination_check_covers_normalized_and_stripped_prompt_variants() {
        let suite = kiln_eval::EvalSuite {
            name: "held-out".to_string(),
            description: None,
            default_scorer: kiln_eval::Scorer::ExactMatch {
                case_sensitive: true,
                strip_whitespace: true,
            },
            generation: Default::default(),
            aggregation: Default::default(),
            system_prompt: None,
            examples: vec![kiln_eval::EvalExample {
                messages: vec![ChatMessage::new("user", "answer this")],
                target: Some("forty two".to_string()),
                ..Default::default()
            }],
            schema_version: kiln_eval::SUITE_SCHEMA_VERSION,
            tools: None,
        };
        let index = kiln_eval::EvalContaminationIndex::from_suite(&suite);
        let examples = vec![SftExample {
            messages: vec![
                ChatMessage::new("system", "private training frame"),
                ChatMessage::new("user", " Answer\nTHIS "),
                ChatMessage::new("assistant", "FORTY   TWO"),
            ],
        }];
        assert_eq!(
            check_sft_contamination(&index, &examples),
            Some(kiln_eval::ContaminationMatch::NormalizedExample)
        );
    }

    fn fixture_teacher_spec(alias: &str) -> super::super::teachers::TeacherSpec {
        super::super::teachers::TeacherSpec {
            alias: alias.into(),
            kind: super::super::teachers::TeacherKind::Fixture,
            provider: None,
            model_id: "fixture-model".into(),
            max_top_k: Some(32),
            vocab_size: Some(1024),
            supports_full_vocab: Some(false),
            tokenizer_hash: None,
            identity: None,
            url: None,
            credential_id: None,
            notes: None,
            adapter: None,
        }
    }

    fn remote_teacher_spec(
        alias: &str,
        tokenizer_vocab_sha256: &str,
    ) -> super::super::teachers::TeacherSpec {
        const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        const C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
        let mut spec = fixture_teacher_spec(alias);
        spec.kind = super::super::teachers::TeacherKind::Remote;
        spec.provider = Some(kiln_train::RemoteProvider::Vllm);
        spec.url = Some("http://127.0.0.1:8000".into());
        spec.max_top_k = Some(2);
        spec.vocab_size = Some(2);
        spec.tokenizer_hash = Some(tokenizer_vocab_sha256.into());
        spec.identity = Some(
            kiln_train::TeacherIdentityV1::new(
                "fixture-model",
                A,
                tokenizer_vocab_sha256,
                C,
                None,
                2,
                2,
                4096,
                65_536,
                "test-runtime",
                A,
            )
            .unwrap(),
        );
        spec
    }

    fn pending_job(job_id: &str, job: QueuedJob) -> (TrainingJobInfo, QueueEntry) {
        let (info, mut entry) = pending_sft_job(job_id);
        entry.job = job;
        (info, entry)
    }

    #[test]
    fn central_admission_pins_every_registered_teacher_job_shape_only() {
        let state = teacher_binding_test_state();
        let spec = fixture_teacher_spec("teacher");
        state.teacher_registry.insert(spec.clone());

        let opd =
            serde_json::from_str::<OpdRequest>(r#"{"prompts":[],"teacher":"teacher"}"#).unwrap();
        let refresh = serde_json::from_str::<DistillRefreshRequest>(
            r#"{"name":"refresh","new_data":{"dataset":"new"},"behavioural_teacher":"teacher"}"#,
        )
        .unwrap();
        let pump = serde_json::from_str::<DistillPumpRequest>(
            r#"{"name":"pump","teacher":"teacher","mode":{"wide":true}}"#,
        )
        .unwrap();
        let merge = serde_json::from_str::<DistillMergeRequest>(r#"{"name":"merge","sources":[]}"#)
            .unwrap();
        let distill_self =
            serde_json::from_str::<DistillSelfRequest>(r#"{"name":"self","mode":"conciseness"}"#)
                .unwrap();
        let grpo = GrpoRequest {
            groups: Vec::new(),
            dataset_path: None,
            dataset: None,
            dataset_split: None,
            config: GrpoConfig::default(),
            post_eval: None,
        };
        let sft = SftRequest {
            examples: Vec::new(),
            dataset_path: None,
            dataset: None,
            dataset_split: None,
            config: SftConfig::default(),
            ingestion: None,
            post_eval: None,
        };
        let mut pending = vec![
            pending_job("opd", QueuedJob::Opd(opd)),
            pending_job("refresh", QueuedJob::DistillRefresh(refresh)),
            pending_job("pump", QueuedJob::DistillPump(pump)),
            pending_job("sft", QueuedJob::Sft(sft)),
            pending_job("grpo", QueuedJob::Grpo(grpo)),
            pending_job("merge", QueuedJob::DistillMerge(merge)),
            pending_job("self", QueuedJob::DistillSelf(distill_self)),
        ];

        pin_registered_teachers(&state, &mut pending).unwrap();

        for (_, entry) in &pending[..3] {
            assert_eq!(entry.teacher_bindings, vec![spec.clone()]);
        }
        for (_, entry) in &pending[3..] {
            assert!(
                entry.teacher_bindings.is_empty(),
                "unrelated job {} must not acquire a teacher binding",
                entry.job_id
            );
        }
    }

    #[test]
    fn teacher_pinning_failure_does_not_publish_partial_batch() {
        let state = teacher_binding_test_state();
        state
            .teacher_registry
            .insert(fixture_teacher_spec("registered"));
        let registered: OpdRequest =
            serde_json::from_str(r#"{"prompts":[],"teacher":"registered"}"#).unwrap();
        let missing: OpdRequest =
            serde_json::from_str(r#"{"prompts":[],"teacher":"missing"}"#).unwrap();

        let error = admit_training_jobs(
            &state,
            vec![
                pending_job("first", QueuedJob::Opd(registered)),
                pending_job("second", QueuedJob::Opd(missing)),
            ],
        )
        .unwrap_err();

        assert_eq!(error.code, "teacher_not_registered");
        assert!(state.training_jobs.read().unwrap().is_empty());
        assert_eq!(state.training_queue.lock().unwrap().len(), 0);
    }

    #[test]
    fn central_admission_rejects_caller_supplied_teacher_bindings() {
        let state = teacher_binding_test_state();
        let spec = fixture_teacher_spec("teacher");
        state.teacher_registry.insert(spec.clone());
        let opd: OpdRequest =
            serde_json::from_str(r#"{"prompts":[],"teacher":"teacher"}"#).unwrap();
        let mut pending = vec![pending_job("opd", QueuedJob::Opd(opd))];
        pending[0].1.teacher_bindings.push(spec);

        let error = pin_registered_teachers(&state, &mut pending).unwrap_err();
        assert_eq!(error.code, "internal_error");
    }

    #[test]
    fn central_admission_rejects_every_remote_on_policy_job_before_publication() {
        let state = teacher_binding_test_state();
        let tokenizer_vocab_sha256 = state
            .tokenizer
            .vocab_identity_sha256()
            .strip_prefix("sha256:")
            .unwrap()
            .to_string();
        state
            .teacher_registry
            .insert(remote_teacher_spec("remote", &tokenizer_vocab_sha256));

        let opd = serde_json::from_str::<OpdRequest>(
            r#"{"prompts":[],"teacher":"remote","config":{"training_mode":"on_policy"}}"#,
        )
        .unwrap();
        let refresh = serde_json::from_str::<DistillRefreshRequest>(
            r#"{"name":"refresh","new_data":{"dataset":"new"},"behavioural_teacher":"remote","config":{"training_mode":"on_policy"}}"#,
        )
        .unwrap();
        let pump = serde_json::from_str::<DistillPumpRequest>(
            r#"{"name":"pump","teacher":"remote","mode":{"wide":true},"config":{"training_mode":"on_policy"}}"#,
        )
        .unwrap();

        for (job_id, job) in [
            ("opd", QueuedJob::Opd(opd)),
            ("refresh", QueuedJob::DistillRefresh(refresh)),
            ("pump", QueuedJob::DistillPump(pump)),
        ] {
            let error = admit_training_jobs(&state, vec![pending_job(job_id, job)]).unwrap_err();
            assert_eq!(error.code, "training_invalid_request");
            assert!(
                error
                    .message
                    .contains("use training_mode=\"off_policy\" with fixed assistant actions"),
                "{}",
                error.message
            );
            assert!(state.training_jobs.read().unwrap().is_empty());
            assert_eq!(state.training_queue.lock().unwrap().len(), 0);
        }
    }

    #[test]
    fn concurrent_batch_admission_is_capacity_atomic() {
        let tracked: crate::state::TrainingJobs = Arc::new(RwLock::new(Default::default()));
        let queue = crate::training_queue::new_shared_queue();
        let start = Arc::new(Barrier::new(2));

        let handles: Vec<_> = (0..2)
            .map(|batch| {
                let tracked = tracked.clone();
                let queue = queue.clone();
                let start = start.clone();
                std::thread::spawn(move || {
                    let pending = vec![
                        pending_sft_job(format!("batch-{batch}-a")),
                        pending_sft_job(format!("batch-{batch}-b")),
                    ];
                    start.wait();
                    admit_training_jobs_into(&tracked, &queue, 3, 3, true, pending)
                })
            })
            .collect();

        let results: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().expect("admission thread panicked"))
            .collect();
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(results.iter().filter(|result| result.is_err()).count(), 1);

        let tracked = tracked.read().unwrap();
        let queue = queue.lock().unwrap();
        assert_eq!(tracked.len(), 2);
        assert_eq!(queue.len(), 2);
        assert!(
            queue
                .queue
                .iter()
                .all(|entry| tracked.contains_key(&entry.job_id))
        );
    }

    #[test]
    fn concurrent_single_submission_cannot_bypass_batch_capacity() {
        let tracked: crate::state::TrainingJobs = Arc::new(RwLock::new(Default::default()));
        let queue = crate::training_queue::new_shared_queue();
        let start = Arc::new(Barrier::new(2));

        let batch_handle = {
            let tracked = tracked.clone();
            let queue = queue.clone();
            let start = start.clone();
            std::thread::spawn(move || {
                start.wait();
                admit_training_jobs_into(
                    &tracked,
                    &queue,
                    2,
                    2,
                    true,
                    vec![pending_sft_job("batch-a"), pending_sft_job("batch-b")],
                )
            })
        };
        let single_handle = {
            let tracked = tracked.clone();
            let queue = queue.clone();
            let start = start.clone();
            std::thread::spawn(move || {
                start.wait();
                admit_training_jobs_into(
                    &tracked,
                    &queue,
                    2,
                    2,
                    true,
                    vec![pending_sft_job("single")],
                )
            })
        };

        let batch_result = batch_handle.join().expect("batch thread panicked");
        let single_result = single_handle.join().expect("single thread panicked");
        assert_ne!(batch_result.is_ok(), single_result.is_ok());

        let tracked = tracked.read().unwrap();
        let queue = queue.lock().unwrap();
        let expected_len = if batch_result.is_ok() { 2 } else { 1 };
        assert_eq!(tracked.len(), expected_len);
        assert_eq!(queue.len(), expected_len);
        assert!(queue.len() <= 2);
    }

    #[test]
    fn rejected_batch_leaves_tracking_and_queue_unchanged() {
        let tracked: crate::state::TrainingJobs = Arc::new(RwLock::new(Default::default()));
        let queue = crate::training_queue::new_shared_queue();
        let error = admit_training_jobs_into(
            &tracked,
            &queue,
            10,
            1,
            true,
            vec![pending_sft_job("a"), pending_sft_job("b")],
        )
        .unwrap_err();

        assert_eq!(error.code, "training_tracked_full");
        assert!(tracked.read().unwrap().is_empty());
        assert_eq!(queue.lock().unwrap().len(), 0);
    }

    #[test]
    fn single_job_admission_preserves_queue_position_semantics() {
        let tracked: crate::state::TrainingJobs = Arc::new(RwLock::new(Default::default()));
        let queue = crate::training_queue::new_shared_queue();

        let position = admit_training_jobs_into(
            &tracked,
            &queue,
            2,
            2,
            true,
            vec![pending_sft_job("single")],
        )
        .unwrap();

        assert_eq!(position, 1);
        assert!(tracked.read().unwrap().contains_key("single"));
        let queue = queue.lock().unwrap();
        assert_eq!(queue.len(), 1);
        assert_eq!(queue.queue.front().unwrap().job_id, "single");
    }

    #[test]
    fn sampled_token_opd_is_rejected_at_submission() {
        let err = validate_opd_loss_at_submit(OpdLossGranularity::SampledToken).unwrap_err();
        assert!(err.to_string().contains("identically zero"), "{err}");
        validate_opd_loss_at_submit(OpdLossGranularity::TeacherTopK).unwrap();
        let err = validate_opd_loss_at_submit(OpdLossGranularity::FullVocab).unwrap_err();
        assert!(err.to_string().contains("no concrete"), "{err}");
    }

    #[test]
    fn unwired_opd_semantics_are_rejected_at_submission() {
        let mut config = OpdConfig::default();
        validate_opd_config_at_submit(&config).unwrap();

        config.objective = OpdObjective::CrossEntropy;
        assert!(
            validate_opd_config_at_submit(&config)
                .unwrap_err()
                .to_string()
                .contains("cross_entropy")
        );

        config = OpdConfig::default();
        config.stable_opd = StableOpdMode::Auto;
        assert!(
            validate_opd_config_at_submit(&config)
                .unwrap_err()
                .to_string()
                .contains("Stable-OPD")
        );
    }

    #[test]
    fn alternate_opd_admission_rejects_empty_and_unscored_off_policy_work() {
        let mut request = OpdRequest {
            prompts: Vec::new(),
            dataset_path: None,
            teacher: "teacher".into(),
            config: OpdConfig::default(),
            post_eval: None,
        };
        assert!(validate_opd_request_at_submit(&request).is_err());

        request.config.training_mode = kiln_train::opd::OpdTrainingMode::OffPolicy;
        request.prompts = vec![OpdPrompt {
            messages: vec![ChatMessage::new("user", "question")],
            teacher_extra_messages: Vec::new(),
            trajectory: Vec::new(),
        }];
        let error = validate_opd_request_at_submit(&request).unwrap_err();
        assert!(error.to_string().contains("assistant action"), "{error}");
    }

    #[test]
    fn fixed_fixture_request_defaults_are_off_policy_and_self_requires_actions() {
        let merge: kiln_train::DistillMergeRequest =
            serde_json::from_str(r#"{"name":"merged","sources":[{"adapter":"source"}]}"#).unwrap();
        assert!(matches!(
            merge.config.training_mode,
            kiln_train::opd::OpdTrainingMode::OffPolicy
        ));

        let mut self_request: kiln_train::DistillSelfRequest = serde_json::from_str(
            r#"{"name":"self","mode":"conciseness","prompts":[{"messages":[{"role":"user","content":"question"},{"role":"assistant","content":"answer"}]}]}"#,
        )
        .unwrap();
        assert!(matches!(
            self_request.config.training_mode,
            kiln_train::opd::OpdTrainingMode::OffPolicy
        ));
        validate_distill_self_at_submit(&self_request).unwrap();

        self_request.prompts = None;
        let error = validate_distill_self_at_submit(&self_request).unwrap_err();
        assert!(error.to_string().contains("explicit off-policy prompts"));
    }

    #[test]
    fn pump_examples_are_validated_before_enqueue_in_both_training_modes() {
        let mut request: kiln_train::DistillPumpRequest =
            serde_json::from_str(r#"{"name":"pump","teacher":"teacher","mode":{"examples":[]}}"#)
                .unwrap();
        let error = validate_distill_pump_at_submit(&request).unwrap_err();
        assert!(error.to_string().contains("at least one prompt"), "{error}");

        request.mode = kiln_train::DistillPumpMode::Examples {
            examples: vec![OpdPrompt {
                messages: Vec::new(),
                teacher_extra_messages: Vec::new(),
                trajectory: Vec::new(),
            }],
        };
        let error = validate_distill_pump_at_submit(&request).unwrap_err();
        assert!(error.to_string().contains("has no messages"), "{error}");

        request.mode = kiln_train::DistillPumpMode::Examples {
            examples: vec![OpdPrompt {
                messages: vec![ChatMessage::new("user", "question")],
                teacher_extra_messages: Vec::new(),
                trajectory: Vec::new(),
            }],
        };
        validate_distill_pump_at_submit(&request).unwrap();

        request.config.training_mode = kiln_train::opd::OpdTrainingMode::OffPolicy;
        let error = validate_distill_pump_at_submit(&request).unwrap_err();
        assert!(error.to_string().contains("assistant action"), "{error}");
    }

    #[test]
    fn opd_top_k_is_resolved_to_the_executable_kernel_envelope() {
        let mut config = OpdConfig::default();
        assert_eq!(
            resolve_opd_top_k_at_submit(&mut config, 20).unwrap(),
            Some((32, 16))
        );
        assert_eq!(config.top_k, 16);

        config.top_k = 32;
        assert_eq!(resolve_opd_top_k_at_submit(&mut config, 32).unwrap(), None);
        assert_eq!(config.top_k, 32);

        config.top_k = 15;
        let error = resolve_opd_top_k_at_submit(&mut config, 20).unwrap_err();
        assert!(error.to_string().contains("not executable"), "{error}");
    }

    #[test]
    fn stock_vllm_registration_limits_opd_to_its_default_twenty() {
        let mut spec = super::super::teachers::TeacherSpec {
            alias: "remote".into(),
            kind: super::super::teachers::TeacherKind::Remote,
            provider: Some(kiln_train::RemoteProvider::Vllm),
            model_id: "model".into(),
            max_top_k: None,
            vocab_size: Some(1024),
            supports_full_vocab: Some(false),
            tokenizer_hash: None,
            identity: None,
            url: Some("http://vllm.local".into()),
            credential_id: None,
            notes: None,
            adapter: None,
        };
        assert_eq!(registered_teacher_top_k_limit(&spec, 32), 20);
        spec.max_top_k = Some(32);
        assert_eq!(registered_teacher_top_k_limit(&spec, 32), 32);
    }

    #[test]
    fn self_distill_privileged_modes_require_one_nonempty_context_per_prompt() {
        let prompts = vec![OpdPrompt {
            messages: vec![ChatMessage::new("user", "question")],
            teacher_extra_messages: vec![],
            trajectory: vec![],
        }];
        let mut req = kiln_train::DistillSelfRequest {
            name: "self-test".into(),
            mode: kiln_train::SelfDistillMode::GroundTruthConditioning,
            prompts: Some(prompts.clone()),
            ground_truth: None,
            documents: None,
            config: OpdConfig::default(),
            post_eval: None,
        };
        assert!(validate_self_distill_context_at_submit(&req).is_err());
        req.ground_truth = Some(vec!["  ".into()]);
        assert!(validate_self_distill_context_at_submit(&req).is_err());
        req.ground_truth = Some(vec!["answer".into()]);
        validate_self_distill_context_at_submit(&req).unwrap();

        req.mode = kiln_train::SelfDistillMode::DocumentAsPi;
        req.ground_truth = None;
        assert!(validate_self_distill_context_at_submit(&req).is_err());
        req.documents = Some(vec!["context".into()]);
        validate_self_distill_context_at_submit(&req).unwrap();

        req.mode = kiln_train::SelfDistillMode::Conciseness;
        req.prompts = None;
        req.documents = None;
        validate_self_distill_context_at_submit(&req).unwrap();
    }

    fn grpo_group() -> GrpoGroup {
        GrpoGroup {
            messages: vec![ChatMessage::new("user", "prompt")],
            completions: vec![ScoredCompletion {
                text: "completion".to_string(),
                reward: 1.0,
                ..Default::default()
            }],
        }
    }

    fn grpo_req(dataset_path: Option<&str>, groups: Vec<GrpoGroup>) -> GrpoRequest {
        GrpoRequest {
            dataset: None,
            dataset_split: None,
            groups,
            dataset_path: dataset_path.map(str::to_string),
            config: GrpoConfig::default(),
            post_eval: None,
        }
    }

    #[test]
    fn server_preflight_activation_width_matches_gdn_trainer_tape() {
        assert_eq!(training_activation_bytes_per_elem(2, false, false), 2);
        assert_eq!(training_activation_bytes_per_elem(2, false, true), 10);
        assert_eq!(training_activation_bytes_per_elem(2, true, true), 4);
    }

    #[test]
    fn server_preflight_streaming_policy_uses_runtime_policy() {
        let max_seq_len = 104_412;
        let configured = kiln_model::StreamingPrefillExecutionPolicy::resolve(
            kiln_model::StreamingPrefillBackendPolicy::for_backend(
                "rocm",
                kiln_tensor::Device::Rocm(0),
            ),
            kiln_model::StreamingPrefillMode::Enabled,
            None,
            None,
            Some(2048),
            None,
            true,
        );
        let rocm = training_activation_estimate_for_streaming_policy(
            2,
            false,
            true,
            configured,
            max_seq_len,
        );
        let cpu_storage = training_activation_estimate_for_streaming_policy(
            2,
            false,
            true,
            kiln_model::StreamingPrefillExecutionPolicy::for_device(kiln_tensor::Device::Cpu),
            max_seq_len,
        );

        assert_eq!(rocm.streaming_gdn_tile_tokens, Some(2048));
        assert_eq!(cpu_storage.streaming_gdn_tile_tokens, None);
    }

    #[test]
    fn dynamic_training_available_counts_allocator_and_reclaimable_kv() {
        let gb = 1024 * 1024 * 1024;
        assert_eq!(
            combine_training_available_bytes(21 * gb, Some(80 * gb), 8 * gb, 120 * gb, gb, false,),
            88 * gb
        );
    }

    #[test]
    fn dynamic_training_available_does_not_trust_allocator_over_live_unified_memory() {
        let gb = 1024 * 1024 * 1024;
        assert_eq!(
            combine_training_available_bytes(21 * gb, Some(80 * gb), 8 * gb, 120 * gb, gb, true,),
            29 * gb
        );
    }

    #[test]
    fn dynamic_training_available_is_capped_by_total_minus_floor() {
        let gb = 1024 * 1024 * 1024;
        assert_eq!(
            combine_training_available_bytes(21 * gb, Some(118 * gb), 8 * gb, 120 * gb, gb, false,),
            119 * gb
        );
    }

    #[test]
    fn configured_training_memory_is_an_enforced_working_set_cap() {
        let gb = 1024 * 1024 * 1024;
        assert_eq!(
            apply_configured_training_budget_cap(20 * gb, Some(4.0), 4 * gb),
            4 * gb
        );
        assert_eq!(
            apply_configured_training_budget_cap(20 * gb, None, 4 * gb),
            20 * gb
        );
    }

    #[test]
    fn queued_materialized_training_data_has_an_aggregate_host_memory_cap() {
        validate_prepared_training_data_capacity(128 * 1024 * 1024, 64 * 1024 * 1024).unwrap();
        let error = validate_prepared_training_data_capacity(480 * 1024 * 1024, 64 * 1024 * 1024)
            .unwrap_err();
        assert_eq!(error.code, "training_prepared_data_full");
        assert_eq!(error.status, axum::http::StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn teacher_fixture_admission_charges_top_k_pairs_and_overlapping_batches() {
        let small = crate::training_queue::conservative_teacher_fixture_bytes(1_024, 4, 1)
            .expect("small fixture estimate");
        let wide = crate::training_queue::conservative_teacher_fixture_bytes(1_024, 32, 1)
            .expect("wide fixture estimate");
        let self_distill = crate::training_queue::conservative_teacher_fixture_bytes(1_024, 32, 2)
            .expect("self-distill fixture estimate");
        assert!(wide > small);
        assert_eq!(self_distill, wide * 2);

        let million_token_top_k =
            crate::training_queue::conservative_teacher_fixture_bytes(1_000_000, 32, 1)
                .expect("large fixture estimate");
        assert!(
            million_token_top_k > crate::training_queue::MAX_LIVE_PREPARED_TRAINING_BYTES,
            "a small textual corpus must not hide a multi-GiB-capable top-K fixture"
        );
    }

    #[test]
    fn auto_mode_reservation_uses_the_live_admitted_exact_plan() {
        let cfg = kiln_core::config::ModelConfig::qwen3_5_4b();
        let gb = 1024 * 1024 * 1024;
        let vram = kiln_memory::vram::GpuVramInfo {
            total_bytes: 120 * gb,
            source: VramSource::LinuxDrmSysfsUnified,
            unified: true,
        };
        let options = EstimateOptions {
            sft: Some(SftEstimateOptions {
                max_active_tokens: 512,
                loss_route: kiln_model::backend::SftFlceLossRoute::VulkanActiveRows,
                checkpoint_boundary_policy: kiln_train::CheckpointBoundaryPolicy::default(),
            }),
            activation_bytes_per_elem: Some(10),
            streaming_gdn_tile_tokens: Some(1024),
            optimizer: kiln_train::Optimizer::default(),
            lora_residency: LoraResidency::default(),
        };
        let max_seq_len = 104_412;
        let one_segment = training_preflight::estimate_step_working_set_with_options(
            &cfg,
            max_seq_len,
            8,
            1,
            WeightResidency::for_vram(&vram),
            true,
            options,
        );
        let eight_segments = training_preflight::estimate_step_working_set_with_options(
            &cfg,
            max_seq_len,
            8,
            8,
            WeightResidency::for_vram(&vram),
            true,
            options,
        );
        let (admitted_segments, admitted) = training_preflight::auto_fit_checkpoint_segments(
            &cfg,
            max_seq_len,
            8,
            cfg.num_layers,
            WeightResidency::for_vram(&vram),
            true,
            options,
            eight_segments.total_bytes,
        );
        assert!(
            admitted_segments > 1 && admitted.total_bytes <= eight_segments.total_bytes,
            "live-tight admission should resolve a checkpointed plan"
        );
        assert!(one_segment.total_bytes > admitted.total_bytes);
    }

    #[test]
    fn full_logits_checkpoint_plan_helper_rejects_multiple_segments() {
        let sft = SftEstimateOptions {
            max_active_tokens: 32,
            loss_route: kiln_model::backend::SftFlceLossRoute::FullLogits,
            checkpoint_boundary_policy: kiln_train::CheckpointBoundaryPolicy::default(),
        };
        ensure_sft_checkpoint_plan_supported(Some(sft), 1).unwrap();
        let error = ensure_sft_checkpoint_plan_supported(Some(sft), 2).unwrap_err();
        assert_eq!(error.code, "training_invalid_request");
        assert!(error.message.contains("full_logits"));
        assert!(error.message.contains("outside an active kt tape"));
    }

    #[test]
    fn grpo_dataset_path_submission_allows_generic_streaming_route() {
        let req = grpo_req(Some("/tmp/grpo.jsonl"), Vec::new());
        validate_grpo_submission_source(&req, None).unwrap();
    }

    #[test]
    fn grpo_submission_rejects_ambiguous_or_empty_sources() {
        let both = grpo_req(Some("/tmp/grpo.jsonl"), vec![grpo_group()]);
        let err = validate_grpo_submission_source(&both, None).unwrap_err();
        assert_eq!(
            err.message,
            "Invalid training request: GRPO request must use exactly one of groups, dataset_path, or dataset"
        );

        let empty = grpo_req(None, Vec::new());
        let err = validate_grpo_submission_source(&empty, None).unwrap_err();
        assert_eq!(
            err.message,
            "Invalid training request: GRPO request must use exactly one of groups, dataset_path, or dataset"
        );

        let inline = grpo_req(None, vec![grpo_group()]);
        validate_grpo_submission_source(&inline, None).unwrap();
    }

    #[test]
    fn grpo_submission_rejects_missing_recorded_behavior_provenance_and_kl_reference() {
        let mut recorded = grpo_req(None, vec![grpo_group()]);
        recorded.config.behavior_policy = kiln_train::BehaviorPolicy::Recorded;
        let error = validate_grpo_submission_source(&recorded, None).unwrap_err();
        assert!(error.message.contains("missing exact rollout provenance"));

        let mut missing_kl_reference = grpo_req(None, vec![grpo_group()]);
        missing_kl_reference.config.kl_reference_policy = kiln_train::KlReferencePolicy::None;
        let error = validate_grpo_submission_source(&missing_kl_reference, None).unwrap_err();
        assert!(error.message.contains("requires kl_estimator=none"));

        let mut invalid_cispo = grpo_req(None, vec![grpo_group()]);
        invalid_cispo.config.is_level = kiln_train::IsLevel::Cispo;
        invalid_cispo.config.cispo_max_weight = 0.0;
        let error = validate_grpo_submission_source(&invalid_cispo, None).unwrap_err();
        assert!(error.message.contains("cispo_max_weight"));
    }

    /// ECHO env-CE trains again (resurrection PR2), so echo-enabled
    /// submissions with Observation segments now pass validation — the
    /// flagship agentic shape. Still rejected at submission (not at worker
    /// dequeue hours later): no_policy_loss (not yet re-wired) and the
    /// reserved OPD slot.
    #[test]
    fn grpo_submission_validates_loss_configs() {
        // ECHO + a rollout carrying an Observation segment.
        let mut group = grpo_group();
        group.completions[0].trajectory = vec![
            kiln_train::trajectory::TurnSegment {
                role: "assistant".into(),
                content: "running".into(),
                kind: kiln_train::trajectory::TurnKind::Action,
                tool_call_id: None,
                warning_prefix_len: None,
            },
            kiln_train::trajectory::TurnSegment {
                role: "tool".into(),
                content: "exit 0".into(),
                kind: kiln_train::trajectory::TurnKind::Observation,
                tool_call_id: None,
                warning_prefix_len: None,
            },
        ];
        let mut req = grpo_req(None, vec![group.clone()]);
        req.config.loss.echo = Some(kiln_train::EchoConfig::default());
        validate_grpo_submission_source(&req, None)
            .expect("echo + observation segments is the flagship agentic shape");

        // Same data WITHOUT echo: also fine — the policy loss trains the
        // trajectory's action tokens.
        let req = grpo_req(None, vec![group]);
        validate_grpo_submission_source(&req, None).unwrap();

        // ECHO on legacy single-turn rollouts: zero env term, harmless.
        let mut req = grpo_req(None, vec![grpo_group()]);
        req.config.loss.echo = Some(kiln_train::EchoConfig::default());
        validate_grpo_submission_source(&req, None).unwrap();

        // no_policy_loss + default ECHO = §5.5 verifier-free mode: valid.
        let mut req = grpo_req(None, vec![grpo_group()]);
        req.config.loss.no_policy_loss = true;
        validate_grpo_submission_source(&req, None).expect("verifier-free mode validates");
        // Without ECHO there is nothing to train on — still rejected.
        req.config.loss.echo = None;
        let err = validate_grpo_submission_source(&req, None).unwrap_err();
        assert!(err.message.contains("no_policy_loss"), "{}", err.message);
    }

    #[test]
    fn grpo_dataset_path_submission_rejects_an_invalid_tail_row() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("grpo.jsonl");
        let mut first_group = grpo_group();
        first_group.messages = vec![ChatMessage::new("user", "a")];
        first_group.completions[0].text = "b".to_string();
        let first = serde_json::to_string(&first_group).unwrap();
        std::fs::write(&path, format!("{first}\nthis is not json\n")).unwrap();
        let tokenizer = crate::api::test_tokenizer().with_chat_template(
            "{% for message in messages %}{{ message.content }}{% endfor %}".to_string(),
        );

        let mut permit = crate::training_queue::PreparedTrainingDataPermit::default();
        let error = validate_grpo_jsonl_submission(
            path.to_str().unwrap(),
            dir.path(),
            &mut permit,
            &tokenizer,
            &GrpoConfig::default(),
            2,
            None,
        )
        .unwrap_err();
        assert!(error.message.contains("line 2"), "{}", error.message);
        assert!(
            std::fs::read_dir(dir.path())
                .unwrap()
                .filter_map(Result::ok)
                .all(|entry| !entry.file_name().to_string_lossy().starts_with("grpo-")),
            "invalid admission must remove its incomplete private snapshot"
        );
    }

    #[test]
    fn grpo_dataset_path_submission_scans_every_row_for_the_maximum_shape() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("grpo.jsonl");
        let mut short = grpo_group();
        short.messages = vec![ChatMessage::new("user", "a")];
        short.completions[0].text = "b".to_string();
        let mut long = grpo_group();
        long.messages = vec![ChatMessage::new("user", "a")];
        long.completions[0].text = "ab".repeat(64);
        std::fs::write(
            &path,
            format!(
                "{}\n{}\n",
                serde_json::to_string(&short).unwrap(),
                serde_json::to_string(&long).unwrap()
            ),
        )
        .unwrap();
        let tokenizer = crate::api::test_tokenizer().with_chat_template(
            "{% for message in messages %}{{ message.content }}{% endfor %}".to_string(),
        );

        let mut permit = crate::training_queue::PreparedTrainingDataPermit::default();
        let stats = validate_grpo_jsonl_submission(
            path.to_str().unwrap(),
            dir.path(),
            &mut permit,
            &tokenizer,
            &GrpoConfig::default(),
            2,
            None,
        )
        .unwrap();
        assert!(stats.streaming_dataset);
        assert_eq!(stats.num_groups, Some(2));
        assert_eq!(stats.total_completions, Some(2));
        assert!(stats.max_seq_len > 0);
        let receipt = stats.source_receipt.unwrap();
        assert_eq!(receipt.groups, 2);
        assert_eq!(receipt.completions, 2);
        assert_eq!(receipt.max_seq_len, stats.max_seq_len);
        assert!(receipt.source_sha256.starts_with("sha256:"));
        let original = std::fs::canonicalize(&path).unwrap();
        assert_ne!(receipt.path, original);
        assert!(receipt.server_owned);
        assert!(
            std::fs::metadata(&receipt.path)
                .unwrap()
                .permissions()
                .readonly()
        );
        let snapshot_path = receipt.path.clone();
        let snapshot_sha256 = kiln_train::train_receipt::sha256_file(&snapshot_path).unwrap();
        std::fs::write(&path, b"caller replaced the original after admission\n").unwrap();
        assert_eq!(
            kiln_train::train_receipt::sha256_file(&snapshot_path).unwrap(),
            snapshot_sha256,
            "the trainer source must be independent of the caller path"
        );
        assert_eq!(
            permit.bytes(),
            receipt.size_bytes + receipt.preflight_host_bytes
        );
        drop(receipt);
        assert!(!snapshot_path.exists());
    }

    #[test]
    fn opd_dataset_materialization_is_bounded_before_reading() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("too-large.jsonl");
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(MAX_MATERIALIZED_OPD_DATASET_BYTES + 1)
            .unwrap();

        let mut permit = crate::training_queue::PreparedTrainingDataPermit::default();
        let error =
            load_bounded_off_policy_dataset(path.to_str().unwrap(), &mut permit).unwrap_err();
        assert!(
            error.message.contains("no larger than"),
            "{}",
            error.message
        );
    }

    fn opd_request_payload() -> OpdRequest {
        OpdRequest {
            prompts: vec![OpdPrompt {
                messages: vec![ChatMessage::new("user", "Solve 5x + 7 = 22")],
                teacher_extra_messages: vec![],
                trajectory: vec![],
            }],
            dataset_path: None,
            teacher: "qwen3.6-27b@local".into(),
            config: OpdConfig::default(),
            post_eval: None,
        }
    }

    #[test]
    fn opd_request_serde_round_trip_carries_grand_plan_defaults() {
        let req = opd_request_payload();
        let json = serde_json::to_string(&req).expect("serialize");
        let parsed: OpdRequest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.teacher, "qwen3.6-27b@local");
        assert_eq!(parsed.config.top_k, 32);
        assert_eq!(parsed.config.samples_per_prompt, 4);
        assert!((parsed.config.top_p - 0.9).abs() < 1e-9);
        assert!(matches!(
            parsed.config.loss,
            OpdLossGranularity::TeacherTopK
        ));
        assert_eq!(parsed.config.max_tokens, 7168);
    }

    #[test]
    fn opd_checkpoint_plan_is_not_client_deserializable() {
        let req: OpdRequest = serde_json::from_value(serde_json::json!({
            "prompts": [{"messages": [{"role": "user", "content": "a"}]}],
            "teacher": "fixture",
            "config": {"grad_checkpoint_segments": 1}
        }))
        .unwrap();
        assert_eq!(req.config.grad_checkpoint_segments, None);
        let encoded = serde_json::to_value(&req.config).unwrap();
        assert!(encoded.get("grad_checkpoint_segments").is_none());

        let mut admitted = req.config;
        admitted.grad_checkpoint_segments = Some(7);
        assert_eq!(
            serde_json::to_value(&admitted).unwrap()["grad_checkpoint_segments"],
            7
        );
    }

    #[test]
    fn opd_request_accepts_dataset_path_in_place_of_prompts() {
        // A streaming-dataset payload — no inline prompts but a
        // `dataset_path` set. The `submit_opd` handler treats this as
        // valid; tested at the wire level.
        let json = r#"{"prompts":[],"dataset_path":"/tmp/opd.jsonl","teacher":"qwen3.6-27b@vllm"}"#;
        let req: OpdRequest = serde_json::from_str(json).unwrap();
        assert!(req.prompts.is_empty());
        assert_eq!(req.dataset_path.as_deref(), Some("/tmp/opd.jsonl"));
        assert_eq!(req.teacher, "qwen3.6-27b@vllm");
    }

    #[test]
    fn opd_request_rejects_unknown_loss_granularity() {
        let json = r#"{"prompts":[],"teacher":"x","config":{"loss":"sampled_lobotomy"}}"#;
        let parsed: Result<OpdRequest, _> = serde_json::from_str(json);
        assert!(
            parsed.is_err(),
            "unknown loss value should fail to deserialize"
        );
    }

    #[test]
    fn distill_refresh_request_minimal_json_parses() {
        let json = r#"{
            "name": "company-assistant",
            "new_data": {"dataset": "q4-2026"},
            "behavioural_teacher": "company-assistant@v17"
        }"#;
        let req: DistillRefreshRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.name, "company-assistant");
        assert_eq!(req.behavioural_teacher, "company-assistant@v17");
        // Defaults populate.
        assert_eq!(req.background_chat, "tulu3");
        assert!((req.require_if_eval_recovery - 0.95).abs() < 1e-9);
        assert!((req.require_internal_qa_gain - 0.05).abs() < 1e-9);
    }

    #[test]
    fn distill_refresh_rejects_multi_suite_automatic_promotion() {
        let mut req: DistillRefreshRequest = serde_json::from_str(
            r#"{
                "name": "company-assistant",
                "new_data": {"dataset": "q4-2026"},
                "behavioural_teacher": "company-assistant@v17",
                "if_eval_suite": "if-held-out-v3",
                "new_knowledge_eval_suite": "qa-held-out-v3"
            }"#,
        )
        .unwrap();

        let error = validate_distill_refresh_at_submit(&req).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("automatic promotion accepts one versioned held-out suite"),
            "{error}"
        );

        req.config.auto_load = false;
        validate_distill_refresh_at_submit(&req)
            .expect("independent multi-suite diagnostics are allowed without auto-load");
    }

    #[test]
    fn distill_refresh_accepts_one_automatic_promotion_suite() {
        let req: DistillRefreshRequest = serde_json::from_str(
            r#"{
                "name": "company-assistant",
                "new_data": {"dataset": "q4-2026"},
                "behavioural_teacher": "company-assistant@v17",
                "if_eval_suite": "if-held-out-v3"
            }"#,
        )
        .unwrap();

        validate_distill_refresh_at_submit(&req)
            .expect("one paired suite owns automatic promotion");
    }
}
