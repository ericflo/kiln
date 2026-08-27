use super::*;

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
    pub(super) fn add_tokenize(&mut self, elapsed: Duration) {
        self.tokenize_ms += elapsed.as_secs_f64() * 1000.0;
    }

    pub(super) fn add_mask_build(&mut self, elapsed: Duration) {
        self.mask_build_ms += elapsed.as_secs_f64() * 1000.0;
    }

    pub(super) fn add_reference_forward(&mut self, elapsed: Duration) {
        self.reference_forward_ms += elapsed.as_secs_f64() * 1000.0;
    }

    pub(super) fn add_backward(&mut self, elapsed: Duration) {
        self.backward_ms += elapsed.as_secs_f64() * 1000.0;
    }

    pub(super) fn add_optimizer(&mut self, elapsed: Duration) {
        self.optimizer_ms += elapsed.as_secs_f64() * 1000.0;
    }

    pub(super) fn to_receipt(&self) -> crate::train_receipt::TrainingPhaseTimingsReceipt {
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
pub(super) fn make_step_progress(
    total_steps: usize,
    label: &str,
) -> Option<indicatif::ProgressBar> {
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

pub(super) fn resolve_and_validate_base_adapter(
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
pub(super) fn epoch_order(seed: u64, epoch: usize, n: usize) -> Vec<usize> {
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
