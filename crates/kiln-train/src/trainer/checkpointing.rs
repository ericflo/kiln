use super::*;

pub(super) const SFT_CHECKPOINT_LOOP_STATE_SCHEMA_VERSION: u32 = 1;
pub(super) const SFT_CHECKPOINT_LOOP_STATE_TYPE: &str = "kiln.sft-loop-state.v1";
pub(super) const SFT_CHECKPOINT_ADAPTER_FILE: &str = "adapter.safetensors";
pub(super) const SFT_CHECKPOINT_OPTIMIZER_FILE: &str = "optimizer.safetensors";
pub(super) const SFT_CHECKPOINT_LOOP_STATE_FILE: &str = "sft_loop_state.json";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct SftCheckpointLoopState {
    pub(super) schema_version: u32,
    pub(super) state_type: String,
    pub(super) global_step: u64,
    pub(super) epoch_index: u64,
    pub(super) cursor_in_epoch: u64,
    pub(super) loss_history: Vec<f64>,
    pub(super) last_loss: f64,
    pub(super) current_epoch_loss_sum: f64,
    pub(super) current_epoch_items: u64,
    pub(super) first_epoch_loss: Option<f64>,
    pub(super) best_epoch_loss: Option<f64>,
    pub(super) lora_grad_norms: crate::train_receipt::LoraGradNormAccumulator,
}

impl SftCheckpointLoopState {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn capture(
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

    pub(super) fn validate(
        &self,
        progress: &crate::checkpoint::TrainingCheckpointProgress,
    ) -> Result<()> {
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
pub(super) struct SftCheckpointDescriptor {
    pub(super) adapter_name: String,
    pub(super) effective_config: serde_json::Value,
    pub(super) precision_policy: crate::checkpoint::TrainingCheckpointPrecision,
    pub(super) data: crate::checkpoint::TrainingCheckpointData,
    pub(super) init_seed: u64,
    pub(super) shuffle_seed: u64,
    pub(super) optimizer: Optimizer,
    pub(super) learning_rate: f64,
    pub(super) total_steps: usize,
    pub(super) base_model_weights_sha256: Option<String>,
    pub(super) auxiliary_state: serde_json::Value,
}

#[derive(Debug)]
pub(super) struct SftCheckpointSnapshot {
    pub(super) target: PathBuf,
    pub(super) manifest: crate::checkpoint::TrainingCheckpointManifest,
    pub(super) artifacts: Vec<crate::checkpoint::CheckpointArtifact>,
    pub(super) adapter_parameters: CheckpointTensorSnapshot,
    pub(super) optimizer_state: Option<CheckpointTensorSnapshot>,
    pub(super) loop_state_bytes: Vec<u8>,
}

impl SftCheckpointSnapshot {
    pub(super) fn publish(self) -> Result<PathBuf> {
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
    pub(super) fn optimizer_state_file(&self) -> Option<String> {
        (!matches!(self.optimizer, Optimizer::Sgd))
            .then(|| SFT_CHECKPOINT_OPTIMIZER_FILE.to_string())
    }

    pub(super) fn optimizer_manifest(
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

    pub(super) fn scheduler_manifest(
        &self,
        step: u64,
    ) -> crate::checkpoint::TrainingCheckpointScheduler {
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

    pub(super) fn rng_states(
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

    pub(super) fn manifest(
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

    pub(super) fn validate_resume(
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
    pub(super) fn capture(
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
    pub(super) fn save(
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

pub(super) fn checkpoint_dtype_name(dtype: KtDType) -> String {
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

pub(super) fn sft_checkpoint_effective_config(
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

pub(super) fn sft_checkpoint_auxiliary_state(
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

pub(super) fn load_sft_checkpoint_loop_state(
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

pub(super) const GRPO_CHECKPOINT_LOOP_STATE_SCHEMA_VERSION: u32 = 1;
pub(super) const GRPO_CHECKPOINT_LOOP_STATE_TYPE: &str = "kiln.grpo-loop-state.v1";
pub(super) const GRPO_CHECKPOINT_ADAPTER_FILE: &str = "adapter.safetensors";
pub(super) const GRPO_CHECKPOINT_OPTIMIZER_FILE: &str = "optimizer.safetensors";
pub(super) const GRPO_CHECKPOINT_REFERENCE_FILE: &str = "reference.safetensors";
pub(super) const GRPO_CHECKPOINT_LOOP_STATE_FILE: &str = "grpo_loop_state.json";

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum GrpoCheckpointRoute {
    Inline,
    Jsonl,
}

impl GrpoCheckpointRoute {
    pub(super) fn source_kind(self) -> &'static str {
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
pub(super) struct GrpoCheckpointLoopState {
    pub(super) schema_version: u32,
    pub(super) state_type: String,
    pub(super) route: GrpoCheckpointRoute,
    pub(super) global_step: u64,
    /// Exact byte offset of the next unread JSONL line. Inline runs have no
    /// source-file cursor and therefore store `None`.
    pub(super) source_byte_offset: Option<u64>,
    /// Number of physical JSONL lines already consumed. This restores line
    /// attribution after seeking, including blank lines.
    pub(super) source_lines_consumed: Option<u64>,
    pub(super) processed_completions: u64,
    pub(super) loss_history: Vec<f64>,
    pub(super) last_loss: Option<f64>,
    pub(super) data_stats: crate::train_receipt::DataStatsReceipt,
    pub(super) token_counts: crate::train_receipt::TokenCountReceipt,
    pub(super) dynamic_groups_filtered: u64,
    pub(super) echo_metrics: crate::train_receipt::EchoActivityMetrics,
    pub(super) lora_grad_norms: crate::train_receipt::LoraGradNormAccumulator,
    pub(super) policy_audit: crate::train_receipt::GrpoPolicyAuditAccumulator,
    pub(super) phase_timings: GrpoBenchmarkTimings,
    pub(super) gpu_writer_timings: GrpoGpuWriterTimings,
    /// Present exactly when the KL reference is an EMA snapshot.
    pub(super) ema_groups_since_refresh: Option<u64>,
}

impl GrpoCheckpointLoopState {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn capture(
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

    pub(super) fn validate(
        &self,
        progress: &crate::checkpoint::TrainingCheckpointProgress,
    ) -> Result<()> {
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
pub(super) struct GrpoCheckpointDescriptor {
    pub(super) route: GrpoCheckpointRoute,
    pub(super) adapter_name: String,
    pub(super) effective_config: serde_json::Value,
    pub(super) precision_policy: crate::checkpoint::TrainingCheckpointPrecision,
    pub(super) data: crate::checkpoint::TrainingCheckpointData,
    pub(super) init_seed: u64,
    pub(super) optimizer: Optimizer,
    pub(super) learning_rate: f64,
    pub(super) total_steps: usize,
    pub(super) base_model_weights_sha256: Option<String>,
    pub(super) auxiliary_state: serde_json::Value,
    pub(super) ema_refresh_every: Option<usize>,
}

#[derive(Debug)]
pub(super) struct GrpoCheckpointSnapshot {
    pub(super) target: PathBuf,
    pub(super) manifest: crate::checkpoint::TrainingCheckpointManifest,
    pub(super) artifacts: Vec<crate::checkpoint::CheckpointArtifact>,
    pub(super) adapter_parameters: CheckpointTensorSnapshot,
    pub(super) optimizer_state: Option<CheckpointTensorSnapshot>,
    pub(super) reference_state: Option<CheckpointTensorSnapshot>,
    pub(super) loop_state_bytes: Vec<u8>,
}

impl GrpoCheckpointSnapshot {
    pub(super) fn replace_loop_state(
        &mut self,
        loop_state: &GrpoCheckpointLoopState,
    ) -> Result<()> {
        self.loop_state_bytes = serde_json::to_vec_pretty(loop_state)
            .context("serialize GRPO checkpoint loop state")?;
        Ok(())
    }

    pub(super) fn publish(self) -> Result<PathBuf> {
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
    pub(super) fn optimizer_state_file(&self) -> Option<String> {
        (!matches!(self.optimizer, Optimizer::Sgd))
            .then(|| GRPO_CHECKPOINT_OPTIMIZER_FILE.to_string())
    }

    pub(super) fn reference_state_file(&self) -> Option<String> {
        self.ema_refresh_every
            .map(|_| GRPO_CHECKPOINT_REFERENCE_FILE.to_string())
    }

    pub(super) fn optimizer_manifest(
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

    pub(super) fn scheduler_manifest(
        &self,
        step: u64,
    ) -> crate::checkpoint::TrainingCheckpointScheduler {
        crate::checkpoint::TrainingCheckpointScheduler {
            kind: "constant".to_string(),
            step,
            state: serde_json::json!({"learning_rate": self.learning_rate}),
        }
    }

    pub(super) fn rng_states(
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

    pub(super) fn data_order(&self) -> Vec<u64> {
        (0..self.total_steps as u64).collect()
    }

    pub(super) fn state_files(&self) -> crate::checkpoint::TrainingCheckpointStateFiles {
        crate::checkpoint::TrainingCheckpointStateFiles {
            adapter_parameters: GRPO_CHECKPOINT_ADAPTER_FILE.to_string(),
            optimizer_state: self.optimizer_state_file(),
            reference_state: self.reference_state_file(),
            ema_state: None,
            reward_normalization_state: None,
            loss_history: Some(GRPO_CHECKPOINT_LOOP_STATE_FILE.to_string()),
        }
    }

    pub(super) fn progress(
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

    pub(super) fn manifest(
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

    pub(super) fn validate_resume(
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

    pub(super) fn capture(
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
    pub(super) fn save(
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

pub(super) fn load_grpo_checkpoint_loop_state(
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

pub(super) fn grpo_checkpoint_effective_config(
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
pub(super) fn grpo_checkpoint_auxiliary_state(
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
