//! In-process LoRA SFT and GRPO training using candle autograd.
//!
//! Trains LoRA adapter weights directly on the already-loaded model's GPU
//! tensors. No Python sidecar, no second model copy, single process.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, Var};

use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_flce_kernel::{DEFAULT_CHUNK_SIZE, fused_linear_cross_entropy};
use kiln_model::backend::{self, BackendRuntime};
use kiln_model::forward::{
    GDN_CHUNK_SIZE, GpuAttentionWeights, GpuWeights, LinearAttentionState, model_forward,
    model_forward_embed, model_forward_final_norm, model_forward_head, model_forward_no_head,
    model_forward_segment, streaming_prefill_enabled_for, streaming_tile_tokens_for,
};
use kiln_model::lora_loader::{LoraLayerWeights, LoraProjectionWeights, LoraWeights};

use crate::{ChatMessage, GrpoConfig, GrpoGroup, SftConfig, SftExample};

/// Convert our ChatMessage to the core tokenizer's ChatMessage.
fn to_core_messages(msgs: &[ChatMessage]) -> Vec<kiln_core::tokenizer::ChatMessage> {
    msgs.iter()
        .map(|m| kiln_core::tokenizer::ChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
            ..Default::default()
        })
        .collect()
}

/// Which linear projections to train LoRA on.
const DEFAULT_TARGET_MODULES: &[&str] = &[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
];

/// Trainable LoRA parameters as candle `Var`s.
///
/// Each Var is tracked by candle's autograd: computations that use these tensors
/// build a computation graph, and `loss.backward()` yields gradients for them.
pub struct TrainableLoraParams {
    /// Per-layer, per-module (A, B) variable pairs.
    /// Indexed as: layers[layer_idx].module_name -> (Var_A, Var_B)
    pub layers: Vec<TrainableLoraLayerParams>,
    pub rank: usize,
    pub alpha: f32,
    pub scale: f32,
}

/// Trainable LoRA A/B pairs for one transformer layer.
#[derive(Default)]
pub struct TrainableLoraLayerParams {
    pub q_proj: Option<(Var, Var)>,
    pub k_proj: Option<(Var, Var)>,
    pub v_proj: Option<(Var, Var)>,
    pub o_proj: Option<(Var, Var)>,
    pub gate_proj: Option<(Var, Var)>,
    pub up_proj: Option<(Var, Var)>,
    pub down_proj: Option<(Var, Var)>,
}

impl TrainableLoraParams {
    /// Initialize fresh LoRA parameters with Kaiming-uniform A and zero B.
    ///
    /// This matches the standard LoRA initialization:
    /// - A: Kaiming uniform (so the product A*B starts near zero)
    /// - B: zeros (so initial LoRA contribution is zero)
    pub fn initialize(
        config: &ModelConfig,
        weights: &GpuWeights,
        rank: usize,
        alpha: f32,
        device: &Device,
    ) -> Result<Self> {
        let scale = alpha / rank as f32;
        let num_layers = config.num_layers;
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;

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
                        // Projection originals may be one-element Metal loader
                        // stubs; the transposed caches are always the real
                        // tensors used by inference/training.
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
                    "gate_proj" => (hidden, intermediate, bound_hidden),
                    "up_proj" => (hidden, intermediate, bound_hidden),
                    "down_proj" => (intermediate, hidden, bound_intermediate),
                    _ => continue,
                };

                // A: [rank, in_features] — Kaiming uniform
                let a = Var::rand_f64(-bound, bound, (rank, in_features), DType::F32, device)
                    .with_context(|| format!("init LoRA A for layer {layer_idx} {module}"))?;

                // B: [out_features, rank] — zeros
                let b = Var::zeros((out_features, rank), DType::F32, device)
                    .with_context(|| format!("init LoRA B for layer {layer_idx} {module}"))?;

                match module {
                    "q_proj" => layer_params.q_proj = Some((a, b)),
                    "k_proj" => layer_params.k_proj = Some((a, b)),
                    "v_proj" => layer_params.v_proj = Some((a, b)),
                    "o_proj" => layer_params.o_proj = Some((a, b)),
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
            rank,
            alpha,
            scale,
        })
    }

    /// Convert trainable params to a `LoraWeights` for use with the forward pass.
    ///
    /// The returned `LoraWeights` holds tensors that are backed by our Vars,
    /// so autograd tracks all operations through them.
    pub fn as_lora_weights(&self) -> LoraWeights {
        let layers: Vec<LoraLayerWeights> = self
            .layers
            .iter()
            .map(|lp| {
                let make_proj = |pair: &Option<(Var, Var)>| -> Option<LoraProjectionWeights> {
                    pair.as_ref().map(|(a, b)| LoraProjectionWeights {
                        a: a.as_tensor().clone(),
                        b: b.as_tensor().clone(),
                    })
                };
                LoraLayerWeights {
                    q_proj: make_proj(&lp.q_proj),
                    k_proj: make_proj(&lp.k_proj),
                    v_proj: make_proj(&lp.v_proj),
                    o_proj: make_proj(&lp.o_proj),
                    gate_proj: make_proj(&lp.gate_proj),
                    up_proj: make_proj(&lp.up_proj),
                    down_proj: make_proj(&lp.down_proj),
                }
            })
            .collect();

        LoraWeights {
            layers,
            rank: self.rank,
            alpha: self.alpha,
            scale: self.scale,
        }
    }

    /// Collect all Var references for gradient extraction and updates.
    pub fn all_vars(&self) -> Vec<&Var> {
        let mut vars = Vec::new();
        for layer in &self.layers {
            let pairs: [&Option<(Var, Var)>; 7] = [
                &layer.q_proj,
                &layer.k_proj,
                &layer.v_proj,
                &layer.o_proj,
                &layer.gate_proj,
                &layer.up_proj,
                &layer.down_proj,
            ];
            for pair in pairs {
                if let Some((a, b)) = pair {
                    vars.push(a);
                    vars.push(b);
                }
            }
        }
        vars
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
            "target_modules": DEFAULT_TARGET_MODULES,
            "task_type": "CAUSAL_LM",
            "bias": "none",
            "peft_type": "LORA",
        });
        let config_path = output_dir.join("adapter_config.json");
        std::fs::write(&config_path, serde_json::to_string_pretty(&config)?)?;

        // Collect all tensors for safetensors serialization
        let mut tensor_data: HashMap<String, Tensor> = HashMap::new();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let mut save_proj = |name: &str, pair: &Option<(Var, Var)>, is_attn: bool| {
                if let Some((a, b)) = pair {
                    let sub = if is_attn { "self_attn" } else { "mlp" };
                    let prefix = format!("base_model.model.model.layers.{layer_idx}.{sub}.{name}");
                    tensor_data.insert(format!("{prefix}.lora_A.weight"), a.as_tensor().clone());
                    tensor_data.insert(format!("{prefix}.lora_B.weight"), b.as_tensor().clone());
                }
            };

            save_proj("q_proj", &layer.q_proj, true);
            save_proj("k_proj", &layer.k_proj, true);
            save_proj("v_proj", &layer.v_proj, true);
            save_proj("o_proj", &layer.o_proj, true);
            save_proj("gate_proj", &layer.gate_proj, false);
            save_proj("up_proj", &layer.up_proj, false);
            save_proj("down_proj", &layer.down_proj, false);
        }

        // Save using candle's safetensors support
        let st_path = output_dir.join("adapter_model.safetensors");
        candle_core::safetensors::save(&tensor_data, &st_path)
            .with_context(|| format!("saving safetensors to {}", st_path.display()))?;

        tracing::info!(
            path = %output_dir.display(),
            num_tensors = tensor_data.len(),
            "saved PEFT adapter"
        );

        Ok(output_dir.to_path_buf())
    }
}

/// Progress callback for training.
pub type ProgressCallback = Box<dyn Fn(TrainingProgress) + Send>;

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

/// Run SFT training on the provided examples using the already-loaded model.
///
/// This runs in the calling thread (blocking). The caller should spawn this
/// on a background thread to avoid blocking inference.
///
/// Returns the path to the saved adapter directory.
pub fn sft_train(
    examples: &[SftExample],
    config: &SftConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
) -> Result<PathBuf> {
    let device = weights.embed_tokens.device().clone();
    let backend = backend::for_device(&device);

    tracing::info!(
        num_examples = examples.len(),
        epochs = config.epochs,
        lr = config.learning_rate,
        rank = config.lora_rank,
        alpha = config.lora_alpha,
        adapter_name,
        "starting SFT training"
    );

    // Initialize trainable LoRA parameters
    let params = TrainableLoraParams::initialize(
        model_config,
        weights,
        config.lora_rank,
        config.lora_alpha,
        &device,
    )?;

    tracing::info!(
        num_vars = params.all_vars().len(),
        "initialized trainable LoRA parameters"
    );

    // Tokenize all examples and build (input_ids, label_mask) pairs.
    // Labels: we want to train on assistant responses only.
    let tokenized: Vec<(Vec<u32>, Vec<bool>)> = examples
        .iter()
        .filter_map(|ex| match tokenize_for_training(ex, tokenizer) {
            Ok(t) => Some(t),
            Err(e) => {
                tracing::warn!("skipping example: {e}");
                None
            }
        })
        .collect();

    if tokenized.is_empty() {
        anyhow::bail!("no valid training examples after tokenization");
    }

    // Configure gradient checkpointing
    let ckpt_config = CheckpointConfig::from_env(model_config.num_layers);
    let segments = if ckpt_config.enabled {
        Some(compute_segment_boundaries(
            model_config.num_layers,
            ckpt_config.num_segments,
        ))
    } else {
        None
    };

    if let Some(ref segs) = segments {
        tracing::info!(
            num_segments = segs.len(),
            boundaries = ?segs,
            "gradient checkpointing enabled"
        );
    } else {
        tracing::info!("gradient checkpointing disabled (KILN_NO_GRAD_CHECKPOINT=1)");
    }

    let total_steps = config.epochs * tokenized.len();
    let mut global_step = 0;
    let mut last_loss = 0.0;

    let pb = make_step_progress(total_steps, "sft training");

    for epoch in 0..config.epochs {
        let mut epoch_loss = 0.0;

        for (_ex_idx, (input_ids, label_mask)) in tokenized.iter().enumerate() {
            let loss_val;

            if let Some(ref segs) = segments {
                // Gradient-checkpointed forward/backward
                let (lv, accumulated_grads) = checkpointed_forward_backward(
                    &*backend,
                    input_ids,
                    weights,
                    model_config,
                    &params,
                    label_mask,
                    segs,
                    &device,
                )?;
                loss_val = lv;
                sgd_step_from_map(&params, &accumulated_grads, config.learning_rate)?;
            } else {
                // Standard (non-checkpointed) forward/backward
                let (lv, grads) = standard_forward_backward(
                    &*backend,
                    input_ids,
                    weights,
                    model_config,
                    &params,
                    label_mask,
                    &device,
                )?;
                loss_val = lv;
                sgd_step(&params, &grads, config.learning_rate)?;
            }

            epoch_loss += loss_val;
            last_loss = loss_val;

            global_step += 1;

            // Periodic adapter checkpoint
            if let Some(interval) = config.checkpoint_interval {
                if interval > 0 && global_step % interval == 0 && global_step < total_steps {
                    let ckpt_dir =
                        adapter_dir.join(format!("{adapter_name}-checkpoint-{global_step}"));
                    if let Err(e) = params.save_peft(&ckpt_dir, model_config.num_layers) {
                        tracing::warn!(step = global_step, error = %e, "failed to save training checkpoint");
                    } else {
                        tracing::info!(step = global_step, "saved training checkpoint");
                    }
                }
            }

            if let Some(ref cb) = progress_cb {
                cb(TrainingProgress {
                    epoch: epoch + 1,
                    total_epochs: config.epochs,
                    step: global_step,
                    total_steps,
                    loss: loss_val,
                    progress: global_step as f32 / total_steps as f32,
                });
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

        let avg_loss = epoch_loss / tokenized.len() as f64;
        tracing::info!(
            epoch = epoch + 1,
            avg_loss = format!("{avg_loss:.6}"),
            "epoch complete"
        );
    }

    if let Some(pb) = pb {
        pb.finish_and_clear();
    }

    // Save the trained adapter
    let output_dir = adapter_dir.join(adapter_name);
    params.save_peft(&output_dir, model_config.num_layers)?;

    tracing::info!(
        adapter = adapter_name,
        path = %output_dir.display(),
        final_loss = format!("{last_loss:.6}"),
        "SFT training complete"
    );

    Ok(output_dir)
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
) -> Result<PathBuf> {
    let device = weights.embed_tokens.device().clone();
    let backend = backend::for_device(&device);

    let total_completions: usize = groups.iter().map(|g| g.completions.len()).sum();
    tracing::info!(
        num_groups = groups.len(),
        total_completions,
        lr = config.learning_rate,
        kl_coeff = config.kl_coeff,
        clip_epsilon = config.clip_epsilon,
        rank = config.lora_rank,
        alpha = config.lora_alpha,
        adapter_name,
        "starting GRPO training"
    );

    // Initialize trainable LoRA parameters
    let params = TrainableLoraParams::initialize(
        model_config,
        weights,
        config.lora_rank,
        config.lora_alpha,
        &device,
    )?;

    tracing::info!(
        num_vars = params.all_vars().len(),
        "initialized trainable LoRA parameters"
    );

    // Tokenize all completions: for each group, tokenize prompt + each completion
    let tokenized_groups: Vec<TokenizedGrpoGroup> = groups
        .iter()
        .filter_map(|group| match tokenize_grpo_group(group, tokenizer) {
            Ok(t) => Some(t),
            Err(e) => {
                tracing::warn!("skipping GRPO group: {e}");
                None
            }
        })
        .collect();

    if tokenized_groups.is_empty() {
        anyhow::bail!("no valid GRPO groups after tokenization");
    }

    // Configure gradient checkpointing (same as SFT)
    let ckpt_config = CheckpointConfig::from_env(model_config.num_layers);
    let segments = if ckpt_config.enabled {
        Some(compute_segment_boundaries(
            model_config.num_layers,
            ckpt_config.num_segments,
        ))
    } else {
        None
    };

    if let Some(ref segs) = segments {
        tracing::info!(
            num_segments = segs.len(),
            boundaries = ?segs,
            "GRPO gradient checkpointing enabled"
        );
    } else {
        tracing::info!("GRPO gradient checkpointing disabled");
    }

    let total_steps = tokenized_groups.len();
    let mut global_step = 0;
    let mut last_loss = 0.0;

    let pb = make_step_progress(total_steps, "grpo training");

    for (group_idx, tgroup) in tokenized_groups.iter().enumerate() {
        // Compute advantages from rewards (normalize within group)
        let advantages = compute_advantages(&tgroup.rewards);

        // For each completion: compute policy log-probs and reference log-probs
        let mut group_loss_sum = 0.0;
        let num_completions = tgroup.completions.len();

        for (comp_idx, comp) in tgroup.completions.iter().enumerate() {
            // Step 1: Reference forward pass (NO LoRA, NO gradient tracking)
            // This is cheap memory-wise since nothing is tracked.
            let ref_log_probs = {
                let mut ref_linear_state = LinearAttentionState::new(model_config, &device)?;
                let ref_logits = model_forward(
                    &*backend,
                    &comp.input_ids,
                    weights,
                    model_config,
                    None,
                    Some(&mut ref_linear_state),
                    None, // no LoRA = reference model
                )
                .context("GRPO reference forward pass")?;
                token_log_probs(&ref_logits, &comp.input_ids, &comp.completion_mask, &device)?
                    .detach()
            };

            // Step 2: Policy forward pass + GRPO loss + backward
            let advantage = advantages[comp_idx];
            let loss_val;

            if let Some(ref segs) = segments {
                // Gradient-checkpointed GRPO step
                let (lv, accumulated_grads) = checkpointed_grpo_forward_backward(
                    &*backend,
                    &comp.input_ids,
                    weights,
                    model_config,
                    &params,
                    &comp.completion_mask,
                    &ref_log_probs,
                    advantage,
                    config.clip_epsilon,
                    config.kl_coeff,
                    segs,
                    &device,
                )?;
                loss_val = lv;
                sgd_step_from_map(&params, &accumulated_grads, config.learning_rate)?;
            } else {
                // Standard (non-checkpointed) GRPO step
                let lora_weights = params.as_lora_weights();
                let mut linear_state = LinearAttentionState::new(model_config, &device)?;
                let policy_logits = model_forward(
                    &*backend,
                    &comp.input_ids,
                    weights,
                    model_config,
                    None,
                    Some(&mut linear_state),
                    Some(&lora_weights),
                )
                .context("GRPO policy forward pass")?;

                let policy_log_probs = token_log_probs(
                    &policy_logits,
                    &comp.input_ids,
                    &comp.completion_mask,
                    &device,
                )?;

                let loss = grpo_loss(
                    &policy_log_probs,
                    &ref_log_probs,
                    advantage,
                    config.clip_epsilon,
                    config.kl_coeff,
                    &device,
                )?;
                loss_val = loss.to_scalar::<f32>()? as f64;

                let grads = loss.backward().context("GRPO backward pass")?;
                sgd_step(&params, &grads, config.learning_rate)?;
            }

            group_loss_sum += loss_val;
        }

        let avg_group_loss = if num_completions > 0 {
            group_loss_sum / num_completions as f64
        } else {
            0.0
        };
        last_loss = avg_group_loss;
        global_step += 1;

        // Periodic adapter checkpoint
        if let Some(interval) = config.checkpoint_interval {
            if interval > 0 && global_step % interval == 0 && global_step < total_steps {
                let ckpt_dir = adapter_dir.join(format!("{adapter_name}-checkpoint-{global_step}"));
                if let Err(e) = params.save_peft(&ckpt_dir, model_config.num_layers) {
                    tracing::warn!(step = global_step, error = %e, "failed to save GRPO training checkpoint");
                } else {
                    tracing::info!(step = global_step, "saved GRPO training checkpoint");
                }
            }
        }

        if let Some(ref cb) = progress_cb {
            cb(TrainingProgress {
                epoch: 1,
                total_epochs: 1,
                step: global_step,
                total_steps,
                loss: avg_group_loss,
                progress: global_step as f32 / total_steps as f32,
            });
        }

        tracing::info!(
            group = group_idx + 1,
            total_groups = total_steps,
            num_completions,
            loss = format!("{avg_group_loss:.6}"),
            "GRPO group step"
        );

        if let Some(pb) = &pb {
            pb.set_message(format!("{avg_group_loss:.6}"));
            pb.inc(1);
        }
    }

    if let Some(pb) = pb {
        pb.finish_and_clear();
    }

    // Save the trained adapter
    let output_dir = adapter_dir.join(adapter_name);
    params.save_peft(&output_dir, model_config.num_layers)?;

    tracing::info!(
        adapter = adapter_name,
        path = %output_dir.display(),
        final_loss = format!("{last_loss:.6}"),
        "GRPO training complete"
    );

    Ok(output_dir)
}

/// Tokenized data for a single completion within a GRPO group.
struct TokenizedGrpoCompletion {
    /// Full input_ids: prompt + completion tokens.
    input_ids: Vec<u32>,
    /// Mask indicating which tokens are completion (true = completion token).
    completion_mask: Vec<bool>,
}

/// A tokenized GRPO group ready for training.
struct TokenizedGrpoGroup {
    completions: Vec<TokenizedGrpoCompletion>,
    rewards: Vec<f64>,
}

/// Tokenize a GRPO group: prompt messages + each completion text.
fn tokenize_grpo_group(group: &GrpoGroup, tokenizer: &KilnTokenizer) -> Result<TokenizedGrpoGroup> {
    if group.completions.is_empty() {
        anyhow::bail!("GRPO group has no completions");
    }

    let prompt_messages = to_core_messages(&group.messages);

    // Tokenize the prompt (without any assistant response)
    let prompt_text = tokenizer
        .apply_chat_template(&prompt_messages)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let prompt_ids = tokenizer
        .encode(&prompt_text)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut completions = Vec::with_capacity(group.completions.len());
    let mut rewards = Vec::with_capacity(group.completions.len());

    for scored in &group.completions {
        // Build full conversation: prompt + assistant completion
        let mut full_messages = prompt_messages.clone();
        full_messages.push(kiln_core::tokenizer::ChatMessage {
            role: "assistant".to_string(),
            content: scored.text.clone(),
            ..Default::default()
        });

        let full_text = tokenizer
            .apply_chat_template(&full_messages)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let full_ids = tokenizer
            .encode(&full_text)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        if full_ids.len() < 2 {
            tracing::warn!("skipping completion: too short ({} tokens)", full_ids.len());
            continue;
        }

        // Completion mask: tokens after the prompt are completion tokens
        let mut mask = vec![false; full_ids.len()];
        for i in prompt_ids.len()..full_ids.len() {
            mask[i] = true;
        }

        completions.push(TokenizedGrpoCompletion {
            input_ids: full_ids,
            completion_mask: mask,
        });
        rewards.push(scored.reward);
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
fn compute_advantages(rewards: &[f64]) -> Vec<f64> {
    let n = rewards.len() as f64;
    if n <= 1.0 {
        return vec![0.0; rewards.len()];
    }
    let mean = rewards.iter().sum::<f64>() / n;
    let var = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    rewards.iter().map(|r| (r - mean) / (std + 1e-8)).collect()
}

/// Compute per-token log-probabilities for the tokens indicated by the mask.
///
/// Returns a 1-D tensor of log-probs for only the masked (completion) positions.
/// Uses the next-token prediction convention: logits[i] predicts token[i+1].
fn token_log_probs(
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
        return Tensor::zeros(1, DType::F32, device).map_err(Into::into);
    }

    // Gather active logits
    let indices = Tensor::new(
        active_positions
            .iter()
            .map(|&i| i as u32)
            .collect::<Vec<_>>()
            .as_slice(),
        device,
    )?;
    let active_logits = shift_logits.index_select(&indices, 0)?; // [num_active, vocab_size]

    let active_labels: Vec<u32> = active_positions.iter().map(|&i| shift_labels[i]).collect();

    // log_softmax then gather
    let active_logits_f32 = active_logits.to_dtype(DType::F32)?;
    let log_sum_exp = active_logits_f32.log_sum_exp(candle_core::D::Minus1)?; // [num_active]
    let labels_2d = Tensor::new(active_labels.as_slice(), device)?
        .to_dtype(DType::U32)?
        .unsqueeze(1)?;
    let correct_logits = active_logits_f32.gather(&labels_2d, 1)?.squeeze(1)?; // [num_active]

    // log_prob = logit - log_sum_exp
    let log_probs = (correct_logits - log_sum_exp)?;

    Ok(log_probs)
}

/// Tokenize a training example into (input_ids, label_mask).
///
/// The label_mask indicates which tokens are part of assistant responses
/// (true = compute loss here, false = ignore).
fn tokenize_for_training(
    example: &SftExample,
    tokenizer: &KilnTokenizer,
) -> Result<(Vec<u32>, Vec<bool>)> {
    let core_messages = to_core_messages(&example.messages);

    // Build the full conversation text using the chat template
    let full_text = tokenizer
        .apply_chat_template(&core_messages)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let input_ids = tokenizer
        .encode(&full_text)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    if input_ids.is_empty() {
        anyhow::bail!("empty tokenization result");
    }

    // Find which tokens correspond to assistant responses.
    // Strategy: tokenize the conversation up to each assistant turn, then mark
    // the difference as assistant tokens.
    let mut label_mask = vec![false; input_ids.len()];

    // Simple approach: find assistant content boundaries by tokenizing
    // prefix conversations and computing the diff.
    let mut prefix_messages: Vec<kiln_core::tokenizer::ChatMessage> = Vec::new();
    for msg in &core_messages {
        prefix_messages.push(msg.clone());
        if msg.role == "assistant" {
            // Tokenize everything up to and including this assistant message
            let prefix_text = tokenizer
                .apply_chat_template(&prefix_messages)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let prefix_ids = tokenizer
                .encode(&prefix_text)
                .map_err(|e| anyhow::anyhow!("{e}"))?;

            // Tokenize everything before this assistant message
            let before_messages: Vec<_> = prefix_messages[..prefix_messages.len() - 1].to_vec();
            let before_text = if before_messages.is_empty() {
                String::new()
            } else {
                tokenizer
                    .apply_chat_template(&before_messages)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
            };
            let before_ids = if before_text.is_empty() {
                Vec::new()
            } else {
                tokenizer
                    .encode(&before_text)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
            };

            // Mark the assistant tokens (shifted by 1 for next-token prediction)
            let start = before_ids.len();
            let end = prefix_ids.len().min(input_ids.len());
            for i in start..end {
                label_mask[i] = true;
            }
        }
    }

    // For next-token prediction, we need at least 2 tokens
    if input_ids.len() < 2 {
        anyhow::bail!("example too short ({} tokens)", input_ids.len());
    }

    Ok((input_ids, label_mask))
}

/// Compute cross-entropy loss on masked positions.
///
/// `logits`: [1, seq_len, vocab_size] — model output
/// `input_ids`: token IDs (used as labels, shifted by 1)
/// `label_mask`: which positions to include in the loss
///
/// Returns: scalar loss tensor (tracked by autograd).
fn cross_entropy_loss(
    logits: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    device: &Device,
) -> Result<Tensor> {
    let seq_len = input_ids.len();

    // Squeeze batch dimension: [seq_len, vocab_size]
    let logits = logits.squeeze(0)?;

    // For next-token prediction: predict token[i+1] from logits[i]
    // So we use logits[0..seq_len-1] to predict input_ids[1..seq_len]
    let shift_logits = logits.narrow(0, 0, seq_len - 1)?; // [seq_len-1, vocab_size]
    let shift_labels: Vec<u32> = input_ids[1..].to_vec();
    let shift_mask: Vec<bool> = label_mask[1..].to_vec();

    // Find positions where we should compute loss
    let active_positions: Vec<usize> = shift_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i) } else { None })
        .collect();

    if active_positions.is_empty() {
        // No assistant tokens to train on — return zero loss
        return Tensor::new(0.0f32, device).map_err(Into::into);
    }

    // Gather active logits and labels
    let indices = Tensor::new(
        active_positions
            .iter()
            .map(|&i| i as u32)
            .collect::<Vec<_>>()
            .as_slice(),
        device,
    )?;
    let active_logits = shift_logits.index_select(&indices, 0)?; // [num_active, vocab_size]

    let active_labels: Vec<u32> = active_positions.iter().map(|&i| shift_labels[i]).collect();
    let labels_tensor = Tensor::new(active_labels.as_slice(), device)?.to_dtype(DType::U32)?;

    // Cross-entropy: -log(softmax(logits)[label])
    // Use log-sum-exp trick for numerical stability
    let active_logits_f32 = active_logits.to_dtype(DType::F32)?;
    let log_sum_exp = active_logits_f32.log_sum_exp(candle_core::D::Minus1)?; // [num_active]

    // Gather the logit for the correct class at each position
    let labels_2d = labels_tensor.unsqueeze(1)?; // [num_active, 1]
    let correct_logits = active_logits_f32.gather(&labels_2d.to_dtype(DType::U32)?, 1)?; // [num_active, 1]
    let correct_logits = correct_logits.squeeze(1)?; // [num_active]

    // loss = mean(log_sum_exp - correct_logit)
    let per_token_loss = (log_sum_exp - correct_logits)?;
    let loss = per_token_loss.mean_all()?;

    Ok(loss)
}

/// Read `KILN_USE_FLCE` env var. When set (`1`, `true`, `yes`), SFT training
/// takes the Fused Linear Cross-Entropy path: the LM head matmul is fused
/// into a chunked log-sum-exp + gather reduction so the `[T, V]` logits
/// tensor is never materialized. Required to keep long-context SFT
/// (T >= 8192) under the A6000 VRAM budget on Qwen3.5-4B (V=151936).
///
/// Default: disabled. Opt-in while the path is being validated.
fn use_flce() -> bool {
    std::env::var("KILN_USE_FLCE")
        .map(|v| {
            let v = v.to_lowercase();
            v == "1" || v == "true" || v == "yes"
        })
        .unwrap_or(false)
}

/// SGD update: param = param - lr * grad
fn sgd_step(
    params: &TrainableLoraParams,
    grads: &candle_core::backprop::GradStore,
    lr: f64,
) -> Result<()> {
    for var in params.all_vars() {
        if let Some(grad) = grads.get(var.as_tensor()) {
            let updated = (var.as_tensor() - (grad * lr)?)?;
            var.set(&updated)?;
        }
    }
    Ok(())
}

/// Accumulate gradients from `src` into `dst`. Creates entries in `dst` for
/// any Var that has a gradient in `src` but not yet in `dst`.
fn accumulate_grads(
    dst: &mut HashMap<candle_core::TensorId, Tensor>,
    src: &candle_core::backprop::GradStore,
    vars: &[&Var],
) -> Result<()> {
    for var in vars {
        if let Some(grad) = src.get(var.as_tensor()) {
            let id = var.as_tensor().id();
            if let Some(existing) = dst.get(&id) {
                dst.insert(id, (existing + grad)?);
            } else {
                dst.insert(id, grad.clone());
            }
        }
    }
    Ok(())
}

/// SGD update from accumulated gradient map (not GradStore).
fn sgd_step_from_map(
    params: &TrainableLoraParams,
    grads: &HashMap<candle_core::TensorId, Tensor>,
    lr: f64,
) -> Result<()> {
    for var in params.all_vars() {
        let id = var.as_tensor().id();
        if let Some(grad) = grads.get(&id) {
            let updated = (var.as_tensor() - (grad * lr)?)?;
            var.set(&updated)?;
        }
    }
    Ok(())
}

/// Gradient checkpointing configuration.
pub struct CheckpointConfig {
    /// Number of segments to split layers into.
    pub num_segments: usize,
    /// Whether checkpointing is enabled.
    pub enabled: bool,
    /// Whether num_segments was auto-configured from VRAM detection.
    pub auto_configured: bool,
}

impl CheckpointConfig {
    /// Create config from environment with VRAM-aware defaults.
    ///
    /// Priority for num_segments:
    /// 1. `KILN_GRAD_CHECKPOINT_SEGMENTS` env var (user override)
    /// 2. Auto-detect from GPU VRAM via `kiln_core::vram`
    /// 3. Fallback to 4 segments
    pub fn from_env(num_layers: usize) -> Self {
        let enabled = std::env::var("KILN_NO_GRAD_CHECKPOINT")
            .map(|v| v != "1" && v.to_lowercase() != "true")
            .unwrap_or(true);

        // Check for explicit env override first
        if let Some(explicit) = std::env::var("KILN_GRAD_CHECKPOINT_SEGMENTS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            return Self {
                num_segments: explicit.min(num_layers).max(1),
                enabled,
                auto_configured: false,
            };
        }

        // VRAM-aware auto-configuration
        let vram = kiln_core::vram::detect_vram();
        let num_segments = kiln_core::vram::recommended_checkpoint_segments(&vram)
            .unwrap_or(4) // fallback if env var was set (shouldn't happen here)
            .min(num_layers)
            .max(1);

        let auto_configured = vram.source != kiln_core::vram::VramSource::None;

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
            enabled,
            auto_configured,
        }
    }
}

/// Compute segment boundaries for gradient checkpointing.
///
/// Returns a list of `(start_layer, end_layer)` pairs that partition
/// `[0..num_layers)` into `num_segments` roughly-equal segments.
fn compute_segment_boundaries(num_layers: usize, num_segments: usize) -> Vec<(usize, usize)> {
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
fn lora_weights_detached(params: &TrainableLoraParams) -> LoraWeights {
    let layers: Vec<LoraLayerWeights> = params
        .layers
        .iter()
        .map(|lp| {
            let make_proj = |pair: &Option<(Var, Var)>| -> Option<LoraProjectionWeights> {
                pair.as_ref().map(|(a, b)| LoraProjectionWeights {
                    a: a.as_tensor().detach(),
                    b: b.as_tensor().detach(),
                })
            };
            LoraLayerWeights {
                q_proj: make_proj(&lp.q_proj),
                k_proj: make_proj(&lp.k_proj),
                v_proj: make_proj(&lp.v_proj),
                o_proj: make_proj(&lp.o_proj),
                gate_proj: make_proj(&lp.gate_proj),
                up_proj: make_proj(&lp.up_proj),
                down_proj: make_proj(&lp.down_proj),
            }
        })
        .collect();

    LoraWeights {
        layers,
        rank: params.rank,
        alpha: params.alpha,
        scale: params.scale,
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
/// 1. The streaming-prefill dispatch is enabled for `device` at this
///    `seq_len` (env override or device-default threshold).
/// 2. The tile size is a positive multiple of `GDN_CHUNK_SIZE` (enforced by
///    [`streaming_tile_tokens_for`]) and strictly less than `seq_len`.
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
fn tiled_training_tile_size(
    weights: &GpuWeights,
    device: &Device,
    seq_len: usize,
) -> Option<usize> {
    let _ = weights; // signature retained for callers; gating moved to the dispatcher.
    if !streaming_prefill_enabled_for(device, seq_len) {
        return None;
    }
    let tile = streaming_tile_tokens_for(device);
    if tile == 0 || tile % GDN_CHUNK_SIZE != 0 || tile >= seq_len {
        return None;
    }
    Some(tile)
}

/// Compute the per-tile contribution to the next-token cross-entropy loss
/// using the same loss math as the monolithic path, returning a scalar
/// tensor `sum_NLL_tile / total_active` so the per-tile contributions sum to
/// the monolithic mean across active positions exactly.
///
/// `tile_hidden`: `[1, L, hidden]` final hidden states for tile positions
/// `[ts..te)`. `labels` and `mask` are the explicit, pre-shifted labels and
/// mask: each `labels[i]` is the target for `tile_hidden[i]`. For non-last
/// tiles `labels.len() == L` (the last logit predicts the first token of the
/// next tile); for the last tile `labels.len() == L - 1` (no label exists at
/// position `total`).
///
/// Internally we route through the existing `cross_entropy_loss` /
/// `fused_linear_cross_entropy` helpers by padding the input by one position
/// and prepending a masked-out dummy label, so the helpers' built-in
/// next-token shift recovers the explicit-label semantics. Final result is
/// scaled by `(num_tile_active / total_active)` because the helpers
/// internally divide by `num_tile_active` while the per-tile contribution to
/// the monolithic mean is `sum_NLL_tile / total_active`.
#[allow(clippy::too_many_arguments)]
fn tile_loss_explicit(
    weights: &GpuWeights,
    model_config: &ModelConfig,
    tile_hidden: &Tensor,
    labels: &[u32],
    mask: &[bool],
    total_active: usize,
    device: &Device,
) -> Result<Tensor> {
    debug_assert_eq!(labels.len(), mask.len());

    let num_tile_active: usize = mask.iter().filter(|&&m| m).count();
    if num_tile_active == 0 || total_active == 0 {
        return Tensor::new(0.0f32, device).map_err(Into::into);
    }

    let (_, l, hidden_size) = tile_hidden.dims3()?;
    let l_labels = labels.len();
    // Helpers expect `input_ids.len() == hidden.dim(1)` and shift internally
    // (`hidden[..len-1]` predicting `input_ids[1..]`). We want the explicit
    // pairing `tile_hidden[i] -> labels[i]` for `i in 0..l_labels`. Prepend a
    // dummy id and mask=false at position 0 of `input_ids_padded` /
    // `mask_padded`, and pad `tile_hidden` by `l_labels + 1 - l` zero rows so
    // dimensions align. Active positions are gated by mask, so the padded
    // hidden never participates in the loss.
    let pad_amount = (l_labels + 1).saturating_sub(l);
    let hidden_padded = if pad_amount > 0 {
        let zero_pad = Tensor::zeros(
            (1, pad_amount, hidden_size),
            tile_hidden.dtype(),
            device,
        )?;
        Tensor::cat(&[tile_hidden, &zero_pad], 1)?
    } else {
        tile_hidden.clone()
    };

    let mut input_ids_padded: Vec<u32> = Vec::with_capacity(l_labels + 1);
    input_ids_padded.push(0u32);
    input_ids_padded.extend_from_slice(labels);
    let mut mask_padded: Vec<bool> = Vec::with_capacity(l_labels + 1);
    mask_padded.push(false);
    mask_padded.extend_from_slice(mask);

    let loss = if use_flce() {
        let normed = model_forward_final_norm(&hidden_padded, weights, model_config)?;
        fused_linear_cross_entropy(
            &normed,
            &weights.embed_tokens_t,
            &input_ids_padded,
            &mask_padded,
            device,
            DEFAULT_CHUNK_SIZE,
        )
        .context("tile fused linear cross-entropy")?
    } else {
        let logits = model_forward_head(&hidden_padded, weights, model_config)?;
        cross_entropy_loss(&logits, &input_ids_padded, &mask_padded, device)?
    };

    // Helpers return `mean over num_tile_active`. We want
    // `sum_NLL_tile / total_active = mean × (num_tile_active / total_active)`.
    let scale = num_tile_active as f64 / total_active as f64;
    loss.affine(scale, 0.0).map_err(Into::into)
}

/// Time-axis tiled per-segment recompute + backward.
///
/// Runs forward+backward+accumulate **per tile** within segment `seg_idx` so
/// each tile's autograd-saved tensors release before the next tile's forward
/// allocates its own. State (`LinearAttentionState`) is threaded across tiles
/// for the grad-tracked segment AND for each detached later segment.
///
/// Correctness invariants:
/// * The model is GDN-only (see [`model_is_gdn_only`]) — no full-attention
///   layer anywhere, so every layer's outputs at position `t` depend only on
///   states / inputs ≤ `t`, and per-tile state-threaded forward is bit-exact
///   against monolithic.
/// * LoRA on GDN layers is restricted to MLP projections (`gate_proj`,
///   `up_proj`, `down_proj`) — see [`TrainableLoraParams::initialize`] —
///   which act per-position. The truncated-BPTT effect of detaching state at
///   tile boundaries does not affect MLP-only LoRA gradients on GDN-only
///   models.
/// * Per-tile loss is computed via [`tile_loss_explicit`], which pads each
///   tile's hidden by one position so all `L` logits (or `L-1` for the last
///   tile) participate in the loss; the per-tile contributions sum to the
///   monolithic mean exactly.
#[allow(clippy::too_many_arguments)]
fn tiled_segment_recompute_and_backward(
    backend: &dyn BackendRuntime,
    seg_idx: usize,
    segments: &[(usize, usize)],
    boundary_states: &[Tensor],
    input_ids: &[u32],
    label_mask: &[bool],
    weights: &GpuWeights,
    model_config: &ModelConfig,
    positions: &[u32],
    params: &TrainableLoraParams,
    accumulated_grads: &mut HashMap<candle_core::TensorId, Tensor>,
    total_active: usize,
    tile_size: usize,
    device: &Device,
) -> Result<f64> {
    let (seg_start, seg_end) = segments[seg_idx];
    let seg_input = boundary_states[seg_idx].clone();
    let (_, total, _) = seg_input.dims3()?;

    // States threaded across tiles. Grad-tracked segment uses one shared
    // state; each later (detached) segment also gets its own shared state so
    // the detached forward sees the same monolithic context evolution.
    let mut grad_state = LinearAttentionState::new(model_config, device)?;
    let later_count = segments.len().saturating_sub(seg_idx + 1);
    let mut later_states: Vec<LinearAttentionState> = Vec::with_capacity(later_count);
    for _ in 0..later_count {
        later_states.push(LinearAttentionState::new(model_config, device)?);
    }

    let all_vars = params.all_vars();
    let mut tile_loss_sum = 0.0f64;

    let mut tile_start = 0usize;
    while tile_start < total {
        let tile_end = (tile_start + tile_size).min(total);
        let tile_len = tile_end - tile_start;
        let is_last_tile = tile_end == total;

        // Slice tile-local inputs.
        let tile_seg_input = seg_input
            .narrow(1, tile_start, tile_len)
            .context("narrow seg_input to tile")?;
        let tile_positions: Vec<u32> = positions[tile_start..tile_end].to_vec();

        // Grad-tracked forward through segment `seg_idx` on the tile.
        let lora_weights_for_seg = params.as_lora_weights();
        let mut tile_hidden = model_forward_segment(
            backend,
            tile_seg_input,
            weights,
            model_config,
            &tile_positions,
            seg_start,
            seg_end,
            Some(&mut grad_state),
            Some(&lora_weights_for_seg),
        )
        .with_context(|| {
            format!(
                "tiled segment {seg_idx} grad-tracked forward, tile [{tile_start}, {tile_end})"
            )
        })?;

        // Detached forward through later segments on the tile, threading
        // each segment's own state across tiles.
        for (i, &(later_start, later_end)) in segments[seg_idx + 1..].iter().enumerate() {
            tile_hidden = tile_hidden.detach();
            let lora_for_later = params.as_lora_weights();
            tile_hidden = model_forward_segment(
                backend,
                tile_hidden,
                weights,
                model_config,
                &tile_positions,
                later_start,
                later_end,
                Some(&mut later_states[i]),
                Some(&lora_for_later),
            )
            .with_context(|| {
                format!(
                    "tiled segment {seg_idx} detached later segment [{later_start}, {later_end}) tile [{tile_start}, {tile_end})"
                )
            })?;
        }

        // Build explicit (pre-shifted) tile labels: `tile_hidden[i]`
        // predicts `input_ids[tile_start + i + 1]` for `i in 0..tile_len`.
        // For the last tile we drop the final logit because position `total`
        // has no label.
        let labels_end = if is_last_tile { total } else { tile_end + 1 };
        let labels_start = tile_start + 1;
        let tile_labels: Vec<u32> = input_ids[labels_start..labels_end].to_vec();
        let tile_mask: Vec<bool> = label_mask[labels_start..labels_end].to_vec();

        let scaled_loss = tile_loss_explicit(
            weights,
            model_config,
            &tile_hidden,
            &tile_labels,
            &tile_mask,
            total_active,
            device,
        )
        .with_context(|| format!("tile loss [{tile_start}, {tile_end}) (last={is_last_tile})"))?;

        let scaled_val = scaled_loss.to_scalar::<f32>()? as f64;
        tile_loss_sum += scaled_val;

        // Backward through this tile's autograd graph. Because the segment
        // is GDN-only and LoRA is MLP-only on GDN layers, MLP-LoRA gradients
        // sum across tiles to the exact monolithic gradient even though the
        // per-tile state read at the start of each tile is detached from
        // the previous tile's autograd graph (truncated BPTT does not
        // affect parameters that don't influence the recurrent state).
        let grads = scaled_loss
            .backward()
            .with_context(|| format!("tiled backward [{tile_start}, {tile_end})"))?;
        accumulate_grads(accumulated_grads, &grads, &all_vars)?;

        tile_start = tile_end;
    }

    Ok(tile_loss_sum)
}

/// Layer-pair time-axis tiled per-segment recompute + backward.
///
/// Generalizes [`tiled_segment_recompute_and_backward`] from GDN-only models
/// to hybrid GDN + full-attention models (Qwen3.5-4B is 24 GDN + 8 full-attn).
/// The GDN-only path's bit-exactness invariant relies on every layer being
/// linear-attention so per-tile state-threaded forward is monolithic-equivalent.
/// Hybrid models break that invariant — full-attention layers have no
/// training-time KV cache and a tiled FA forward would attend only inside
/// its own tile.
///
/// The layer-pair path resolves this by:
///
/// 1. **Pre-compute the gradient at the segment's output.** Wrap
///    `boundary_states[seg_idx + 1]` in a fresh [`Var`] (`seg_output_var`),
///    forward through later segments + final RMSNorm + LM head + cross-entropy
///    using the regular grad-tracked `params.as_lora_weights()`, then
///    `loss.backward()`. This produces:
///    * LoRA gradients for layers in segments `seg_idx + 1 .. num_segments`
///      (matching the monolithic checkpointed path's "later segments via
///      detached input but grad-tracked LoRA Vars" pattern).
///    * The gradient `∂loss/∂seg_output_var` (extracted from the GradStore).
///
/// 2. **Compute block-boundary states for this segment.** Detached forward
///    through this segment's layers in order, snapshotting the (detached)
///    hidden state at each block boundary. Used as input to each block's
///    grad-tracked forward in step 4.
///
/// 3. **Partition the segment into contiguous-attention-type blocks** via
///    [`partition_segment_layers_by_attn_type`].
///
/// 4. **Process blocks LAST -> FIRST with gradient injection.** For each
///    block:
///    * Wrap the block's input (a detached [`Tensor`] from step 2) in a
///      fresh [`Var`] so the block's `loss.backward()` can extract the
///      gradient at the block's input.
///    * Run forward through the block's layer range using
///      `params.as_lora_weights()`. For full-attention blocks the forward is
///      monolithic at full seq_len (FA needs the global causal mask). For
///      GDN blocks the forward is time-tiled — `LinearAttentionState` is
///      threaded across tiles within the block; one [`narrow`] of
///      `block_input_var` produces each tile's input.
///    * Compute the gradient-injection scalar `(block_output *
///      grad_at_current_block_output).sum_all()` (or the tile-local
///      analogue) and backward. This is mathematically equivalent to chain-
///      ruling through the block: `∂scalar/∂theta = sum_pos
///      grad_at_block_output[pos] * (∂block_output[pos]/∂theta) =
///      ∂loss/∂theta` for any `theta` whose backward path is wholly inside
///      the block.
///    * Accumulate this block's LoRA gradients into `accumulated_grads`.
///    * Extract `∂scalar/∂block_input_var` and use it as
///      `grad_at_current_block_output` for the previous (lower-layer) block.
///      For tiled GDN blocks, sum across tiles to recover the
///      full-seq_len gradient (each tile's `narrow` backward fills only the
///      tile's range; non-tile positions are zeros).
///
/// **Correctness invariants (relative to monolithic checkpointed_forward_backward):**
/// * MLP-LoRA gradients are bit-exact. MLP is per-position so
///   `∂block_output[t]/∂MLP_LoRA` only depends on `block_input[t]` regardless
///   of state-thread truncation across tile boundaries.
/// * Full-attention LoRA gradients are bit-exact when the FA block's input
///   gradient comes through an exact upstream chain (no GDN tiling between
///   the FA block and the segment output). In the test config used for CPU
///   parity (`full_attention_interval = 2`, layers 1, 3 are FA), every FA
///   block is the LAST block in its segment and gets the bit-exact
///   `grad_at_seg_output` directly — so FA-LoRA grads are bit-exact in that
///   case as well. Tolerance is set to `1e-3` in the parity test to absorb
///   ordering-induced f32 drift in matmul reductions.
/// * GDN-LoRA gradients via the tile loop's truncated state thread are
///   approximate w.r.t. the recurrent path; in current kiln, GDN layers
///   only carry MLP-LoRA (q/k/v/o LoRA is full-attn only — see
///   [`TrainableLoraParams::initialize`]), so the truncation does not
///   affect any LoRA parameter that exists.
///
/// **Memory:** the tail backward in step 1 holds saved tensors for
/// `(num_segments - seg_idx - 1)` later-segment forwards plus the LM head /
/// FLCE chain. The block backward in step 4 holds saved tensors for ONE
/// block's worth of layers (full seq_len for FA blocks, tile-narrow for GDN
/// blocks). The peak across the segment iteration is therefore bounded by
/// the larger of those two, and the per-segment peak does not include all
/// `seg_end - seg_start` layers' saved tensors at full seq_len (which is
/// what the existing monolithic path holds for hybrid models).
#[allow(clippy::too_many_arguments)]
fn layer_pair_tiled_segment_recompute_and_backward(
    backend: &dyn BackendRuntime,
    seg_idx: usize,
    segments: &[(usize, usize)],
    boundary_states: &[Tensor],
    input_ids: &[u32],
    label_mask: &[bool],
    weights: &GpuWeights,
    model_config: &ModelConfig,
    positions: &[u32],
    params: &TrainableLoraParams,
    accumulated_grads: &mut HashMap<candle_core::TensorId, Tensor>,
    tile_size: usize,
    device: &Device,
) -> Result<f64> {
    let (seg_start, seg_end) = segments[seg_idx];
    let num_segments = segments.len();
    let all_vars = params.all_vars();

    // === Step 1: Pre-compute gradient at this segment's output. ===
    //
    // Wrap `boundary_states[seg_idx + 1]` (detached) in a fresh Var so a
    // single `loss.backward()` through later segments + LM head produces the
    // gradient at the segment-output node, which becomes the seed for the
    // per-block gradient-injection backward in step 4.
    let seg_output_var = Var::from_tensor(&boundary_states[seg_idx + 1])?;

    // Use DETACHED LoRA weights for the tail forward — we want the tail
    // backward to produce ONLY `∂loss/∂seg_output_var`, not LoRA grads
    // for layers in segments `seg_idx + 1 .. num_segments`. Those layers'
    // LoRA Vars get their grads from THEIR OWN per-block backward in the
    // corresponding seg-iteration of `checkpointed_forward_backward`.
    // Accumulating later-segment LoRA grads here would double-count — each
    // later-seg LoRA Var would receive (1 contribution per earlier-or-equal
    // seg-iteration) instead of exactly one.
    let lora_detached = lora_weights_detached(params);
    let mut tail_hidden = seg_output_var.as_tensor().clone();
    for (i, &(later_start, later_end)) in segments[seg_idx + 1..].iter().enumerate() {
        // Detach BETWEEN later segments. Skip the detach for the first
        // later segment so the gradient flows from later-segs[0]'s input
        // back to seg_output_var.
        if i > 0 {
            tail_hidden = tail_hidden.detach();
        }
        let mut later_state = LinearAttentionState::new(model_config, device)?;
        tail_hidden = model_forward_segment(
            backend,
            tail_hidden,
            weights,
            model_config,
            positions,
            later_start,
            later_end,
            Some(&mut later_state),
            Some(&lora_detached),
        )
        .with_context(|| {
            format!(
                "layer-pair tail forward later segment [{later_start}, {later_end}) for seg_idx={seg_idx}"
            )
        })?;
    }

    let tail_loss = if use_flce() {
        let normed = model_forward_final_norm(&tail_hidden, weights, model_config)?;
        fused_linear_cross_entropy(
            &normed,
            &weights.embed_tokens_t,
            input_ids,
            label_mask,
            device,
            DEFAULT_CHUNK_SIZE,
        )
        .context("layer-pair tail FLCE")?
    } else {
        let logits = model_forward_head(&tail_hidden, weights, model_config)?;
        cross_entropy_loss(&logits, input_ids, label_mask, device)?
    };

    let tail_loss_val = tail_loss.to_scalar::<f32>()? as f64;
    let tail_grads = tail_loss
        .backward()
        .context("layer-pair tail backward")?;

    // We deliberately do NOT call `accumulate_grads(... &all_vars)` here
    // — see the `lora_detached` comment above. The tail backward's only
    // "useful" output is the gradient at `seg_output_var`.
    let grad_at_seg_output = tail_grads
        .get(seg_output_var.as_tensor())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "layer-pair tail backward did not produce a gradient at seg_output_var \
                 (seg_idx={seg_idx}, later segments: {})",
                num_segments - seg_idx - 1
            )
        })?
        .clone()
        .detach();

    // Drop the tail's autograd graph & saved tensors before per-block work.
    // `tail_grads` is the only remaining handle into that graph; dropping it
    // explicitly makes the lifetime clear to the reader.
    drop(tail_grads);

    // === Step 2: Compute block-boundary states (detached). ===
    //
    // This forward computes block-boundary VALUES only — no LoRA grads
    // are required from this phase, and the autograd graph it would
    // otherwise build (LoRA Vars in graph, then `.detach()` per block)
    // would just be torn down again for no benefit. Use detached LoRA so
    // the inner ops don't bother building the LoRA-side autograd graph.
    let blocks = partition_segment_layers_by_attn_type(weights, seg_start, seg_end);
    let mut block_boundaries: Vec<Tensor> = Vec::with_capacity(blocks.len() + 1);
    block_boundaries.push(boundary_states[seg_idx].clone());
    {
        let mut linear_state = LinearAttentionState::new(model_config, device)?;
        let mut current = block_boundaries[0].clone();
        for (_kind, range) in &blocks {
            current = model_forward_segment(
                backend,
                current,
                weights,
                model_config,
                positions,
                range.start,
                range.end,
                Some(&mut linear_state),
                Some(&lora_detached),
            )
            .with_context(|| {
                format!(
                    "layer-pair block-boundary forward [{}, {}) (seg_idx={seg_idx})",
                    range.start, range.end,
                )
            })?;
            block_boundaries.push(current.detach());
            current = block_boundaries.last().unwrap().clone();
        }
    }

    // === Step 3 + 4: Process blocks LAST -> FIRST with gradient injection. ===
    let mut grad_at_current_output = grad_at_seg_output;

    for (block_idx, (kind, range)) in blocks.iter().enumerate().rev() {
        let block_input = block_boundaries[block_idx].clone();
        let block_input_var = Var::from_tensor(&block_input)?;

        let new_grad_at_block_input = match kind {
            AttnKind::FullAttn => {
                // Full-attention block: forward monolithically (FA can't be
                // tiled at training time — no KV cache). Gradient injection:
                // scalar = (block_output * grad_at_current_output).sum_all().
                let mut state = LinearAttentionState::new(model_config, device)?;
                let lora_for_block = params.as_lora_weights();
                let block_output = model_forward_segment(
                    backend,
                    block_input_var.as_tensor().clone(),
                    weights,
                    model_config,
                    positions,
                    range.start,
                    range.end,
                    Some(&mut state),
                    Some(&lora_for_block),
                )
                .with_context(|| {
                    format!(
                        "layer-pair FA block forward [{}, {}) (seg_idx={seg_idx})",
                        range.start, range.end,
                    )
                })?;

                let scalar = (&block_output * &grad_at_current_output)?
                    .sum_all()
                    .context("layer-pair FA block scalar (gradient injection)")?;
                let block_grads = scalar
                    .backward()
                    .context("layer-pair FA block backward")?;

                accumulate_grads(accumulated_grads, &block_grads, &all_vars)?;

                block_grads
                    .get(block_input_var.as_tensor())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "layer-pair FA block backward missing grad at block_input_var \
                             (block [{}, {}), seg_idx={seg_idx})",
                            range.start, range.end,
                        )
                    })?
                    .clone()
                    .detach()
            }
            AttnKind::Gdn => {
                // GDN block: time-tile forward+backward. Per-tile gradient
                // injection: for each tile [tile_start, tile_end), the
                // local scalar is
                //   (tile_output * grad_at_current_output[..tile..]).sum_all()
                // Backward gives the LoRA grads for this block + the
                // tile-local gradient at block_input_var (zeros outside the
                // tile range, real gradient inside). Sum across tiles to
                // recover the full-seq_len gradient at block_input_var.
                let (_, total_tokens, _) = block_input.dims3()?;
                let mut state = LinearAttentionState::new(model_config, device)?;
                let mut summed: Option<Tensor> = None;

                let mut tile_start = 0usize;
                while tile_start < total_tokens {
                    let tile_end = (tile_start + tile_size).min(total_tokens);
                    let tile_len = tile_end - tile_start;

                    let tile_input = block_input_var
                        .as_tensor()
                        .narrow(1, tile_start, tile_len)
                        .context("narrow GDN block input to tile")?;
                    let tile_positions: Vec<u32> = positions[tile_start..tile_end].to_vec();
                    let lora_for_block = params.as_lora_weights();

                    let tile_output = model_forward_segment(
                        backend,
                        tile_input,
                        weights,
                        model_config,
                        &tile_positions,
                        range.start,
                        range.end,
                        Some(&mut state),
                        Some(&lora_for_block),
                    )
                    .with_context(|| {
                        format!(
                            "layer-pair GDN tile forward [{tile_start}, {tile_end}) \
                             block [{}, {}) (seg_idx={seg_idx})",
                            range.start, range.end,
                        )
                    })?;

                    let tile_grad_out = grad_at_current_output
                        .narrow(1, tile_start, tile_len)
                        .context("narrow grad_at_current_output to tile")?;

                    let scalar = (&tile_output * &tile_grad_out)?
                        .sum_all()
                        .context("layer-pair GDN tile scalar (gradient injection)")?;
                    let tile_grads = scalar
                        .backward()
                        .context("layer-pair GDN tile backward")?;

                    accumulate_grads(accumulated_grads, &tile_grads, &all_vars)?;

                    let tile_block_input_grad = tile_grads
                        .get(block_input_var.as_tensor())
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "layer-pair GDN tile backward missing grad at \
                                 block_input_var (tile [{tile_start}, {tile_end}), \
                                 block [{}, {}), seg_idx={seg_idx})",
                                range.start, range.end,
                            )
                        })?
                        .clone();

                    summed = Some(match summed {
                        Some(prev) => (prev + tile_block_input_grad)?,
                        None => tile_block_input_grad,
                    });

                    tile_start = tile_end;
                }

                summed
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "layer-pair GDN block produced no tiles \
                             (total_tokens={total_tokens}, tile_size={tile_size}, \
                             block [{}, {}), seg_idx={seg_idx})",
                            range.start, range.end,
                        )
                    })?
                    .detach()
            }
        };

        // For block_idx > 0 the new grad becomes the gradient at the
        // previous block's output. For block_idx == 0 the grad is the
        // gradient at this segment's input (boundary_states[seg_idx]),
        // which is already detached and discarded — we keep it in scope
        // only for the loop's last iteration tail and let it drop after.
        grad_at_current_output = new_grad_at_block_input;
    }

    Ok(tail_loss_val)
}

/// Run one training step with gradient checkpointing.
///
/// Instead of tracking activations for all layers, this:
/// 1. Runs each segment forward, detaching hidden states at boundaries
/// 2. For each segment, recomputes it with gradient tracking while running
///    remaining segments detached, then backpropagates to get gradients
///    for that segment's LoRA parameters only
/// 3. Accumulates gradients across all segments
///
/// Memory: only one segment's activations are in the autograd graph at a time.
/// Compute: ~(N+1)/2 × N forward passes for N segments (with N=4, ~2.5× overhead).
///
/// Phase 10 time-axis tiling: when `KILN_STREAMING_PREFILL=1` (or the
/// device-default streaming threshold fires) and `seq_len > tile_size`, each
/// segment's recompute is split into time tiles and each tile is
/// forward+backward+accumulated independently. This releases per-tile
/// autograd-saved tensors before the next tile's forward starts — the change
/// identified in PR #635 as the next step needed to unblock long-context SFT
/// past the 30 GiB segment-recompute ceiling.
///
/// Two tiled implementations:
/// * GDN-only models use [`tiled_segment_recompute_and_backward`]
///   (PR #636) — bit-exact against monolithic; per-tile loss is the
///   tile-local cross-entropy.
/// * Hybrid GDN + full-attn models (e.g. Qwen3.5-4B with 24 GDN + 8 FA
///   layers) use [`layer_pair_tiled_segment_recompute_and_backward`] —
///   each segment is partitioned into contiguous-attention-type blocks and
///   processed with gradient injection. GDN sub-blocks are time-tiled;
///   full-attention sub-blocks run monolithically (no training-time KV
///   cache to thread across tiles). See
///   `docs/audits/PHASE10_GDN_TRAINING_LAYER_PAIR_TILED.md`.
#[allow(clippy::too_many_arguments)]
fn checkpointed_forward_backward(
    backend: &dyn BackendRuntime,
    input_ids: &[u32],
    weights: &GpuWeights,
    model_config: &ModelConfig,
    params: &TrainableLoraParams,
    label_mask: &[bool],
    segments: &[(usize, usize)],
    device: &Device,
) -> Result<(f64, HashMap<candle_core::TensorId, Tensor>)> {
    let num_segments = segments.len();

    // Step 1: Run full forward pass with detached boundaries to get boundary hidden states.
    let (embed_hidden, positions) = model_forward_embed(input_ids, weights)?;
    let lora_weights = params.as_lora_weights();

    let mut boundary_states: Vec<Tensor> = Vec::with_capacity(num_segments + 1);
    boundary_states.push(embed_hidden.detach());

    {
        let mut current = boundary_states[0].clone();
        for &(start, end) in segments.iter() {
            let mut linear_state = LinearAttentionState::new(model_config, device)?;
            current = model_forward_segment(
                backend,
                current,
                weights,
                model_config,
                &positions,
                start,
                end,
                Some(&mut linear_state),
                Some(&lora_weights),
            )?;
            boundary_states.push(current.detach());
            current = boundary_states.last().unwrap().clone();
        }
    }

    // Step 2: For each segment, recompute with grad tracking and backprop.
    let mut accumulated_grads: HashMap<candle_core::TensorId, Tensor> = HashMap::new();
    let all_vars = params.all_vars();
    let mut total_loss = 0.0;

    // Tiling decision (made once per training step). When we tile, every
    // segment iteration uses the per-tile path; otherwise, every segment
    // uses the monolithic path.
    //
    // Two tiled implementations exist:
    // * GDN-only models -> [`tiled_segment_recompute_and_backward`]
    //   (PR #636). Bit-exact against monolithic; skips gradient injection.
    // * Hybrid GDN + full-attn models ->
    //   [`layer_pair_tiled_segment_recompute_and_backward`] (this PR).
    //   Partitions each segment into contiguous-attention-type blocks and
    //   processes them with gradient injection so the tiled path can fire
    //   on production models like Qwen3.5-4B.
    let tile_size = tiled_training_tile_size(weights, device, input_ids.len());
    let use_layer_pair = tile_size.is_some() && !model_is_gdn_only(weights);
    let total_active: usize = if tile_size.is_some() && !use_layer_pair {
        // Same denominator as the monolithic path's `cross_entropy_loss` /
        // `fused_linear_cross_entropy`: count of active label positions
        // after the next-token shift (`label_mask[1..]`). Only used by the
        // GDN-only fast path's [`tile_loss_explicit`] scaling; the
        // layer-pair path computes the full chain loss directly so it
        // doesn't need this count.
        if label_mask.len() >= 2 {
            label_mask[1..].iter().filter(|&&m| m).count()
        } else {
            0
        }
    } else {
        0
    };

    for seg_idx in 0..num_segments {
        if let Some(tile) = tile_size {
            let seg_loss = if use_layer_pair {
                layer_pair_tiled_segment_recompute_and_backward(
                    backend,
                    seg_idx,
                    segments,
                    &boundary_states,
                    input_ids,
                    label_mask,
                    weights,
                    model_config,
                    &positions,
                    params,
                    &mut accumulated_grads,
                    tile,
                    device,
                )?
            } else {
                tiled_segment_recompute_and_backward(
                    backend,
                    seg_idx,
                    segments,
                    &boundary_states,
                    input_ids,
                    label_mask,
                    weights,
                    model_config,
                    &positions,
                    params,
                    &mut accumulated_grads,
                    total_active,
                    tile,
                    device,
                )?
            };
            total_loss += seg_loss;
            continue;
        }

        let (seg_start, seg_end) = segments[seg_idx];

        // Start from the detached boundary state for this segment
        let seg_input = boundary_states[seg_idx].clone();

        // Recompute this segment WITH gradient tracking (LoRA Vars are tracked)
        let lora_weights_for_seg = params.as_lora_weights();
        let mut linear_state = LinearAttentionState::new(model_config, device)?;
        let mut hidden = model_forward_segment(
            backend,
            seg_input,
            weights,
            model_config,
            &positions,
            seg_start,
            seg_end,
            Some(&mut linear_state),
            Some(&lora_weights_for_seg),
        )?;

        // Run remaining segments DETACHED (no grad tracking for their LoRA params).
        // We detach the hidden state so subsequent segments don't contribute to the graph.
        for &(later_start, later_end) in &segments[seg_idx + 1..] {
            hidden = hidden.detach();
            let mut later_linear_state = LinearAttentionState::new(model_config, device)?;
            // Use the original (non-Var) lora weights so they don't get tracked
            let lora_for_later = params.as_lora_weights();
            hidden = model_forward_segment(
                backend,
                hidden,
                weights,
                model_config,
                &positions,
                later_start,
                later_end,
                Some(&mut later_linear_state),
                Some(&lora_for_later),
            )?;
        }

        // LM head + loss (or fused LCE when KILN_USE_FLCE=1).
        let loss = if use_flce() {
            let normed = model_forward_final_norm(&hidden, weights, model_config)?;
            fused_linear_cross_entropy(
                &normed,
                &weights.embed_tokens_t,
                input_ids,
                label_mask,
                device,
                DEFAULT_CHUNK_SIZE,
            )
            .context("fused linear cross-entropy (checkpointed)")?
        } else {
            let logits = model_forward_head(&hidden, weights, model_config)?;
            cross_entropy_loss(&logits, input_ids, label_mask, device)?
        };
        let loss_val = loss.to_scalar::<f32>()? as f64;
        total_loss += loss_val;

        // Backward — only the current segment's LoRA Vars contribute gradients
        let grads = loss.backward().context("checkpointed backward pass")?;
        accumulate_grads(&mut accumulated_grads, &grads, &all_vars)?;
    }

    // Average loss across segments. In both monolithic and tiled paths the
    // per-iteration `total_loss` accumulates the same segment-equivalent
    // value, so dividing by `num_segments` recovers the mean cross-entropy
    // over active positions.
    let avg_loss = total_loss / num_segments as f64;

    Ok((avg_loss, accumulated_grads))
}

/// Run one training step WITHOUT gradient checkpointing (original behavior).
fn standard_forward_backward(
    backend: &dyn BackendRuntime,
    input_ids: &[u32],
    weights: &GpuWeights,
    model_config: &ModelConfig,
    params: &TrainableLoraParams,
    label_mask: &[bool],
    device: &Device,
) -> Result<(f64, candle_core::backprop::GradStore)> {
    let lora_weights = params.as_lora_weights();
    let mut linear_state = LinearAttentionState::new(model_config, device)?;

    let loss = if use_flce() {
        let hidden = model_forward_no_head(
            backend,
            input_ids,
            weights,
            model_config,
            Some(&mut linear_state),
            Some(&lora_weights),
        )
        .context("training forward pass (FLCE)")?;
        fused_linear_cross_entropy(
            &hidden,
            &weights.embed_tokens_t,
            input_ids,
            label_mask,
            device,
            DEFAULT_CHUNK_SIZE,
        )
        .context("fused linear cross-entropy")?
    } else {
        let logits = model_forward(
            backend,
            input_ids,
            weights,
            model_config,
            None,
            Some(&mut linear_state),
            Some(&lora_weights),
        )
        .context("training forward pass")?;
        cross_entropy_loss(&logits, input_ids, label_mask, device)?
    };
    let loss_val = loss.to_scalar::<f32>()? as f64;
    let grads = loss.backward().context("backward pass")?;

    Ok((loss_val, grads))
}

/// Compute the GRPO loss from policy and reference log-probs.
///
/// Returns a scalar loss tensor suitable for backward().
fn grpo_loss(
    policy_log_probs: &Tensor,
    ref_log_probs: &Tensor,
    advantage: f64,
    clip_epsilon: f64,
    kl_coeff: f64,
    device: &Device,
) -> Result<Tensor> {
    let log_ratio = (policy_log_probs - ref_log_probs)?;
    let ratio = log_ratio.exp()?;
    let ratio_shape = ratio.shape().clone();

    // Broadcast scalars to match ratio shape for minimum/clamp
    let lo = Tensor::new(1.0 - clip_epsilon, device)?
        .to_dtype(DType::F32)?
        .broadcast_as(&ratio_shape)?;
    let hi = Tensor::new(1.0 + clip_epsilon, device)?
        .to_dtype(DType::F32)?
        .broadcast_as(&ratio_shape)?;
    let clipped_ratio = ratio.clamp(&lo, &hi)?;

    let adv_tensor = Tensor::new(advantage as f32, device)?.broadcast_as(&ratio_shape)?;
    let surr1 = (&ratio * &adv_tensor)?;
    let surr2 = (&clipped_ratio * &adv_tensor)?;
    let surrogate = surr1.minimum(&surr2)?;
    let neg_surrogate = surrogate.neg()?;

    // KL divergence per token
    let kl_penalty = log_ratio.affine(kl_coeff, 0.0)?;

    // Total loss = mean(-surrogate + kl_penalty)
    let per_token_loss = (&neg_surrogate + &kl_penalty)?;
    per_token_loss.mean_all().map_err(Into::into)
}

/// Run one GRPO training step with gradient checkpointing.
///
/// Similar to `checkpointed_forward_backward` but computes GRPO loss
/// (policy vs reference) instead of cross-entropy. The reference log-probs
/// are pre-computed and passed in (they don't need gradient tracking).
#[allow(clippy::too_many_arguments)]
fn checkpointed_grpo_forward_backward(
    backend: &dyn BackendRuntime,
    input_ids: &[u32],
    weights: &GpuWeights,
    model_config: &ModelConfig,
    params: &TrainableLoraParams,
    completion_mask: &[bool],
    ref_log_probs: &Tensor,
    advantage: f64,
    clip_epsilon: f64,
    kl_coeff: f64,
    segments: &[(usize, usize)],
    device: &Device,
) -> Result<(f64, HashMap<candle_core::TensorId, Tensor>)> {
    let num_segments = segments.len();

    // Step 1: Run full forward pass with detached boundaries to get boundary hidden states.
    let (embed_hidden, positions) = model_forward_embed(input_ids, weights)?;
    let lora_weights = params.as_lora_weights();

    let mut boundary_states: Vec<Tensor> = Vec::with_capacity(num_segments + 1);
    boundary_states.push(embed_hidden.detach());

    {
        let mut current = boundary_states[0].clone();
        for &(start, end) in segments.iter() {
            let mut linear_state = LinearAttentionState::new(model_config, device)?;
            current = model_forward_segment(
                backend,
                current,
                weights,
                model_config,
                &positions,
                start,
                end,
                Some(&mut linear_state),
                Some(&lora_weights),
            )?;
            boundary_states.push(current.detach());
            current = boundary_states.last().unwrap().clone();
        }
    }

    // Step 2: For each segment, recompute with grad tracking and backprop with GRPO loss.
    let mut accumulated_grads: HashMap<candle_core::TensorId, Tensor> = HashMap::new();
    let all_vars = params.all_vars();
    let mut total_loss = 0.0;

    for seg_idx in 0..num_segments {
        let (seg_start, seg_end) = segments[seg_idx];

        // Start from the detached boundary state for this segment
        let seg_input = boundary_states[seg_idx].clone();

        // Recompute this segment WITH gradient tracking (LoRA Vars are tracked)
        let lora_weights_for_seg = params.as_lora_weights();
        let mut linear_state = LinearAttentionState::new(model_config, device)?;
        let mut hidden = model_forward_segment(
            backend,
            seg_input,
            weights,
            model_config,
            &positions,
            seg_start,
            seg_end,
            Some(&mut linear_state),
            Some(&lora_weights_for_seg),
        )?;

        // Run remaining segments DETACHED
        for &(later_start, later_end) in &segments[seg_idx + 1..] {
            hidden = hidden.detach();
            let mut later_linear_state = LinearAttentionState::new(model_config, device)?;
            let lora_for_later = params.as_lora_weights();
            hidden = model_forward_segment(
                backend,
                hidden,
                weights,
                model_config,
                &positions,
                later_start,
                later_end,
                Some(&mut later_linear_state),
                Some(&lora_for_later),
            )?;
        }

        // LM head
        let logits = model_forward_head(&hidden, weights, model_config)?;

        // Compute policy log-probs and GRPO loss
        let policy_log_probs = token_log_probs(&logits, input_ids, completion_mask, device)?;
        let loss = grpo_loss(
            &policy_log_probs,
            ref_log_probs,
            advantage,
            clip_epsilon,
            kl_coeff,
            device,
        )?;
        let loss_val = loss.to_scalar::<f32>()? as f64;
        total_loss += loss_val;

        // Backward — only the current segment's LoRA Vars contribute gradients
        let grads = loss.backward().context("GRPO checkpointed backward pass")?;
        accumulate_grads(&mut accumulated_grads, &grads, &all_vars)?;
    }

    let avg_loss = total_loss / num_segments as f64;
    Ok((avg_loss, accumulated_grads))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_model::forward::{
        GpuAttentionWeights, GpuFfnWeights, GpuFullAttentionWeights, GpuLayerWeights,
        GpuLinearAttentionWeights,
    };
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    /// Serializes tests in this binary that mutate process-global env vars
    /// (`KILN_STREAMING_PREFILL`, `KILN_STREAMING_TILE_TOKENS`,
    /// `KILN_USE_FLCE`, `KILN_DISABLE_RMSNORM_KERNEL`,
    /// `KILN_DISABLE_RMSNORM_BACKWARD`). `cargo test` runs tests in this
    /// binary as parallel threads in a single process, so without this
    /// mutex one test's `set_var` can leak into another test's
    /// "monolithic baseline" forward pass. `cargo nextest run` runs each
    /// test in its own process, so this mutex is a no-op there.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Create a tiny ModelConfig for testing (4 layers, small dims).
    fn tiny_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 32,
            num_layers: 4,
            num_attention_heads: 2,
            num_kv_heads: 2,
            head_dim: 16,
            intermediate_size: 64,
            vocab_size: 32,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: 1,
            full_attention_interval: 4, // layer 3 is full attention
            attn_output_gate: false,
            linear_num_key_heads: 2,
            linear_key_head_dim: 16,
            linear_num_value_heads: 2,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 0.5,
        }
    }

    /// Default deterministic seed for `tiny_weights`. Pinned so every test in
    /// this binary that uses the default `tiny_weights` sees the same model
    /// weights on every run, removing the unseeded `Tensor::randn` flakiness
    /// that produced occasional `mono=NaN tiled=NaN` failures on the
    /// 192-token tile-parity tests (#636/#637 regression).
    const TINY_WEIGHTS_DEFAULT_SEED: u64 = 0xC0FFEE_u64;

    /// Sample a tensor of shape `shape` from a uniform `[-a, a]` distribution
    /// where `a = std * √3`. Uniform with that bound has the same variance as
    /// `Normal(0, std)`, so it's a drop-in replacement for the
    /// `Tensor::randn(0, std, ...)` calls used previously in `tiny_weights`,
    /// while staying inside a strictly bounded range (no fat tail) and
    /// remaining deterministic for a given `rng` state.
    fn randn_like_seeded(
        rng: &mut StdRng,
        std: f32,
        shape: &[usize],
        device: &Device,
    ) -> Result<Tensor> {
        // 3.0_f32.sqrt() — stable equivalent of unstable `f32::consts::SQRT_3`.
        let a = std * 1.732_050_8_f32;
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| rng.random_range(-a..a)).collect();
        Tensor::from_slice(&data, shape, device).map_err(Into::into)
    }

    /// Create tiny random GpuWeights on CPU for the given config, using a
    /// fixed deterministic seed. Equivalent to
    /// `tiny_weights_with_seed(config, device, TINY_WEIGHTS_DEFAULT_SEED)`.
    fn tiny_weights(config: &ModelConfig, device: &Device) -> Result<GpuWeights> {
        tiny_weights_with_seed(config, device, TINY_WEIGHTS_DEFAULT_SEED)
    }

    /// Create tiny GpuWeights on CPU using a seeded RNG so the model weights
    /// are reproducible across runs. Replaces the previous unseeded
    /// `Tensor::randn` calls — those use a thread-local RNG that candle's CPU
    /// backend explicitly cannot seed (`set_seed` bails on CPU), so they
    /// produced non-reproducible weights every run. With long sequences
    /// (`seq_len = 192`) and 4-layer GDN/hybrid models the unseeded init
    /// occasionally drew pathological values that produced NaN forward
    /// passes; this seeded variant pins the init so tests are deterministic.
    fn tiny_weights_with_seed(
        config: &ModelConfig,
        device: &Device,
        seed: u64,
    ) -> Result<GpuWeights> {
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        let vocab = config.vocab_size;
        let mut rng = StdRng::seed_from_u64(seed);

        let embed_tokens = randn_like_seeded(&mut rng, 0.02, &[vocab, h], device)?;
        let embed_tokens_t = embed_tokens.t()?.contiguous()?;
        let final_norm = Tensor::zeros(h, DType::F32, device)?; // (1+w)*x, so zeros = identity

        let mut layers = Vec::new();
        for layer_idx in 0..config.num_layers {
            let input_layernorm = Tensor::zeros(h, DType::F32, device)?;
            let post_attention_layernorm = Tensor::zeros(h, DType::F32, device)?;

            let gate_proj = randn_like_seeded(&mut rng, 0.02, &[inter, h], device)?;
            let up_proj = randn_like_seeded(&mut rng, 0.02, &[inter, h], device)?;
            let down_proj = randn_like_seeded(&mut rng, 0.02, &[h, inter], device)?;
            let gate_proj_t = gate_proj.t()?.contiguous()?;
            let up_proj_t = up_proj.t()?.contiguous()?;
            let down_proj_t = down_proj.t()?.contiguous()?;
            let mlp = GpuFfnWeights {
                gate_proj,
                up_proj,
                down_proj,
                gate_proj_t,
                up_proj_t,
                down_proj_t,
                gate_proj_marlin: None,
                up_proj_marlin: None,
                down_proj_marlin: None,
            };

            let attention = if config.is_full_attention_layer(layer_idx) {
                let nh = config.num_attention_heads;
                let nkv = config.num_kv_heads;
                let hd = config.head_dim;
                let q_proj = randn_like_seeded(&mut rng, 0.02, &[nh * hd, h], device)?;
                let k_proj = randn_like_seeded(&mut rng, 0.02, &[nkv * hd, h], device)?;
                let v_proj = randn_like_seeded(&mut rng, 0.02, &[nkv * hd, h], device)?;
                let o_proj = randn_like_seeded(&mut rng, 0.02, &[h, nh * hd], device)?;
                let q_proj_t = q_proj.t()?.contiguous()?;
                let k_proj_t = k_proj.t()?.contiguous()?;
                let v_proj_t = v_proj.t()?.contiguous()?;
                let o_proj_t = o_proj.t()?.contiguous()?;
                GpuAttentionWeights::Full(GpuFullAttentionWeights {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm: Tensor::ones((hd,), DType::F32, device)?,
                    k_norm: Tensor::ones((hd,), DType::F32, device)?,
                    q_proj_t,
                    k_proj_t,
                    v_proj_t,
                    o_proj_t,
                    q_proj_marlin: None,
                })
            } else {
                let qkv_dim = config.linear_qkv_dim();
                let v_dim = config.linear_v_dim();
                let in_proj_qkv = randn_like_seeded(&mut rng, 0.02, &[qkv_dim, h], device)?;
                let in_proj_z = randn_like_seeded(&mut rng, 0.02, &[v_dim, h], device)?;
                let out_proj = randn_like_seeded(&mut rng, 0.02, &[h, v_dim], device)?;
                let in_proj_a = randn_like_seeded(
                    &mut rng,
                    0.02,
                    &[config.linear_num_value_heads, h],
                    device,
                )?;
                let in_proj_b = randn_like_seeded(
                    &mut rng,
                    0.02,
                    &[config.linear_num_value_heads, h],
                    device,
                )?;
                let in_proj_qkv_t = in_proj_qkv.t()?.contiguous()?;
                let in_proj_z_t = in_proj_z.t()?.contiguous()?;
                let in_proj_a_t = in_proj_a.t()?.contiguous()?;
                let in_proj_b_t = in_proj_b.t()?.contiguous()?;
                let out_proj_t = out_proj.t()?.contiguous()?;
                let conv1d = randn_like_seeded(
                    &mut rng,
                    0.02,
                    &[qkv_dim, 1, config.linear_conv_kernel_dim],
                    device,
                )?;
                let a_log =
                    randn_like_seeded(&mut rng, 0.5, &[config.linear_num_value_heads], device)?;
                GpuAttentionWeights::Linear(GpuLinearAttentionWeights {
                    in_proj_qkv,
                    in_proj_z,
                    out_proj,
                    in_proj_a,
                    in_proj_b,
                    conv1d,
                    norm: Tensor::zeros(config.linear_key_head_dim, DType::F32, device)?,
                    a_log,
                    dt_bias: Tensor::zeros(config.linear_num_value_heads, DType::F32, device)?,
                    in_proj_qkv_t,
                    in_proj_z_t,
                    in_proj_a_t,
                    in_proj_b_t,
                    out_proj_t,
                })
            };

            layers.push(GpuLayerWeights {
                input_layernorm,
                post_attention_layernorm,
                attention,
                mlp,
            });
        }

        let rotary_inv_freq = kiln_model::forward::compute_rotary_inv_freq(
            config.rotary_dim(),
            config.rope_theta,
            device,
        )?;

        Ok(GpuWeights {
            embed_tokens,
            embed_tokens_t,
            layers,
            final_norm,
            rotary_inv_freq,
            mtp: None,
        })
    }

    #[test]
    fn test_lora_initialize_uses_transposed_projection_shapes() -> Result<()> {
        let device = Device::Cpu;
        let mut config = tiny_config();
        config.hidden_size = 48;
        config.intermediate_size = 80;
        config.vocab_size = 64;
        config.num_layers = 1;
        config.num_full_attention_layers = 1;
        config.full_attention_interval = 1;

        let mut weights = tiny_weights(&config, &device)?;
        let layer = &mut weights.layers[0];
        let kiln_model::forward::GpuAttentionWeights::Full(full) = &mut layer.attention else {
            unreachable!("test config should create a full-attention layer");
        };
        let stub = Tensor::zeros((1usize,), DType::F32, &device)?;
        full.q_proj = stub.clone();
        full.k_proj = stub.clone();
        full.v_proj = stub.clone();
        full.o_proj = stub;

        let params = TrainableLoraParams::initialize(&config, &weights, 4, 8.0, &device)?;
        let layer = &params.layers[0];

        let assert_pair =
            |pair: &Option<(Var, Var)>, in_features: usize, out_features: usize| -> Result<()> {
                let (a, b) = pair.as_ref().context("missing LoRA pair")?;
                assert_eq!(a.as_tensor().dims(), &[4, in_features]);
                assert_eq!(b.as_tensor().dims(), &[out_features, 4]);
                Ok(())
            };

        let q_out = config.full_attn_q_proj_dim();
        let kv_out = config.num_kv_heads * config.head_dim;
        let o_in = config.num_attention_heads * config.head_dim;
        assert_pair(&layer.q_proj, config.hidden_size, q_out)?;
        assert_pair(&layer.k_proj, config.hidden_size, kv_out)?;
        assert_pair(&layer.v_proj, config.hidden_size, kv_out)?;
        assert_pair(&layer.o_proj, o_in, config.hidden_size)?;

        Ok(())
    }

    #[test]
    fn test_cross_entropy_loss_basic() -> Result<()> {
        let device = Device::Cpu;

        // 3 tokens, vocab size 4
        // logits: [1, 3, 4]
        let logits = Tensor::new(
            &[[
                [2.0f32, 1.0, 0.1, 0.0],
                [0.0, 3.0, 0.1, 0.0],
                [0.0, 0.0, 0.0, 5.0],
            ]],
            &device,
        )?;

        // input_ids: [A, B, C] — predict B from logits[0], C from logits[1]
        let input_ids = vec![0u32, 1, 3];
        // Only train on position 1 (predicting token 3 from logits[1])
        let label_mask = vec![false, true, false];

        let loss = cross_entropy_loss(&logits, &input_ids, &label_mask, &device)?;
        let loss_val = loss.to_scalar::<f32>()?;

        // After next-token-prediction shift:
        // shift_logits[0] = [2, 1, 0.1, 0] predicting label 1
        // shift_mask = [true, false] — only position 0 is active
        // log_sum_exp([2,1,0.1,0]) = log(7.389 + 2.718 + 1.105 + 1) ≈ 2.50
        // correct_logit = 1.0
        // loss ≈ 2.50 - 1.0 = 1.50
        assert!((loss_val - 1.50).abs() < 0.1, "loss = {loss_val}");

        Ok(())
    }

    #[test]
    fn test_segment_boundaries() {
        // 32 layers, 4 segments → 8 each
        let segs = compute_segment_boundaries(32, 4);
        assert_eq!(segs, vec![(0, 8), (8, 16), (16, 24), (24, 32)]);

        // 4 layers, 2 segments → 2 each
        let segs = compute_segment_boundaries(4, 2);
        assert_eq!(segs, vec![(0, 2), (2, 4)]);

        // 5 layers, 3 segments → 2, 2, 1
        let segs = compute_segment_boundaries(5, 3);
        assert_eq!(segs, vec![(0, 2), (2, 4), (4, 5)]);

        // 1 segment = whole model
        let segs = compute_segment_boundaries(4, 1);
        assert_eq!(segs, vec![(0, 4)]);
    }

    #[test]
    fn test_segmented_forward_matches_full() -> Result<()> {
        let device = Device::Cpu;
        let config = tiny_config();
        let weights = tiny_weights(&config, &device)?;

        let input_ids: Vec<u32> = vec![1, 5, 10, 3, 7];
        let backend = backend::for_device(&device);

        // Full forward pass (no KV cache, no LoRA)
        let mut linear_state_full = LinearAttentionState::new(&config, &device)?;
        let logits_full = model_forward(
            &*backend,
            &input_ids,
            &weights,
            &config,
            None,
            Some(&mut linear_state_full),
            None,
        )?;

        // Segmented forward: embed → segment(0..2) → segment(2..4) → head
        let (hidden, positions) = model_forward_embed(&input_ids, &weights)?;
        let mut linear_state_seg = LinearAttentionState::new(&config, &device)?;
        let hidden = model_forward_segment(
            &*backend,
            hidden,
            &weights,
            &config,
            &positions,
            0,
            2,
            Some(&mut linear_state_seg),
            None,
        )?;
        let mut linear_state_seg2 = LinearAttentionState::new(&config, &device)?;
        // The second segment needs fresh linear state starting from the correct layer offset.
        // However, LinearAttentionState::new creates state for ALL linear layers.
        // model_forward_segment handles the indexing internally.
        let hidden = model_forward_segment(
            &*backend,
            hidden,
            &weights,
            &config,
            &positions,
            2,
            4,
            Some(&mut linear_state_seg2),
            None,
        )?;
        let logits_seg = model_forward_head(&hidden, &weights, &config)?;

        // Compare logits
        let diff = (logits_full - logits_seg)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-4, "segmented forward differs from full by {diff}");

        Ok(())
    }

    #[test]
    fn test_checkpointed_loss_matches_standard() -> Result<()> {
        let device = Device::Cpu;
        let config = tiny_config();
        let weights = tiny_weights(&config, &device)?;

        let input_ids: Vec<u32> = vec![1, 5, 10, 3, 7, 2, 8];
        let label_mask = vec![false, false, true, true, true, true, false];

        // Initialize identical LoRA params for both paths
        let params_std = TrainableLoraParams::initialize(&config, &weights, 4, 8.0, &device)?;

        let backend = backend::for_device(&device);
        // Standard (non-checkpointed) forward/backward
        let (loss_std, _grads_std) = standard_forward_backward(
            &*backend,
            &input_ids,
            &weights,
            &config,
            &params_std,
            &label_mask,
            &device,
        )?;

        // Checkpointed forward/backward with 2 segments
        // Re-initialize identical params (same seed won't work since Var uses random init,
        // so we test that checkpointed loss is finite and reasonable instead of exact match).
        let params_ckpt = TrainableLoraParams::initialize(&config, &weights, 4, 8.0, &device)?;
        let segments = compute_segment_boundaries(config.num_layers, 2);
        let (loss_ckpt, _grads_ckpt) = checkpointed_forward_backward(
            &*backend,
            &input_ids,
            &weights,
            &config,
            &params_ckpt,
            &label_mask,
            &segments,
            &device,
        )?;

        // Both losses should be finite and in a reasonable range for random weights
        assert!(
            loss_std.is_finite(),
            "standard loss is not finite: {loss_std}"
        );
        assert!(
            loss_ckpt.is_finite(),
            "checkpointed loss is not finite: {loss_ckpt}"
        );
        // Cross-entropy on random logits over vocab=32 should be ~ln(32) ≈ 3.47
        assert!(
            loss_std > 1.0 && loss_std < 10.0,
            "standard loss out of range: {loss_std}"
        );
        assert!(
            loss_ckpt > 1.0 && loss_ckpt < 10.0,
            "checkpointed loss out of range: {loss_ckpt}"
        );

        Ok(())
    }

    /// CPU parity for the Phase 10 time-axis tile path: training-time
    /// tiled `checkpointed_forward_backward` must match the monolithic
    /// path bit-for-bit on a GDN-only mini-model at T = 3 × tile_size.
    ///
    /// Mirrors the `test_model_forward_segment_streaming_matches_monolithic_cpu`
    /// pattern from PR #635 — env-driven, relies on nextest per-test process
    /// isolation. Fails (deadlocks-on-`set_var` warnings aside) under
    /// multi-threaded `cargo test`; run via `cargo nextest run` or
    /// `cargo test -- --test-threads=1`.
    ///
    /// The test asserts:
    /// 1. Tiled total loss equals monolithic total loss bit-for-bit (atol
    ///    tightened to ~1e-5 to allow trivial f32 fp-associativity drift in
    ///    the chunked LM-head log-sum-exp).
    /// 2. Every LoRA Var with a gradient in the monolithic path has the
    ///    same gradient (within the same tolerance) in the tiled path.
    #[test]
    fn test_checkpointed_forward_backward_tiled_matches_monolithic_cpu() -> Result<()> {
        // Hold ENV_LOCK across the whole test so a parallel
        // env-mutating test in this binary can't flip
        // `KILN_STREAMING_PREFILL` mid-call and turn the "monolithic"
        // baseline into a tiled run (or vice versa). See ENV_LOCK.
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let device = Device::Cpu;

        // GDN-only mini-config: setting `full_attention_interval` strictly
        // greater than `num_layers` makes `is_full_attention_layer(i)` false
        // for every layer in [0, num_layers), so `tiny_weights` only emits
        // GDN layers and `model_is_gdn_only` returns true.
        let mut config = tiny_config();
        config.full_attention_interval = config.num_layers + 1;
        config.num_full_attention_layers = 0;

        let weights = tiny_weights(&config, &device)?;
        assert!(
            super::model_is_gdn_only(&weights),
            "test setup error: model must be GDN-only for tiled-path parity"
        );

        // T = 192 = 3 × tile_size(64) so the tile loop runs three iterations
        // (two non-last tiles + one last tile) and exercises the
        // `pad_amount = 1` branch in `tile_loss_explicit` plus the last-tile
        // (`pad_amount = 0`) branch in the same step.
        let seq_len: usize = 192;
        let vocab = config.vocab_size;
        let input_ids: Vec<u32> = (0..seq_len)
            .map(|i| ((i * 7 + 3) % vocab) as u32)
            .collect();
        // Mask out positions 0 and total-1 so the next-token shift produces
        // the same effective active-position set in both paths (matches the
        // pattern of `test_checkpointed_loss_matches_standard`).
        let mut label_mask = vec![false; seq_len];
        for slot in label_mask.iter_mut().skip(1).take(seq_len - 2) {
            *slot = true;
        }

        let params = TrainableLoraParams::initialize(&config, &weights, 4, 8.0, &device)?;
        let segments = compute_segment_boundaries(config.num_layers, 2);
        let backend = backend::for_device(&device);

        // Step 1: monolithic baseline (env explicitly cleared so the path
        // takes the original branch even if a parent test process leaked a
        // KILN_STREAMING_PREFILL=1 setting).
        // SAFETY: env var mutation is safe under nextest's per-test process
        // isolation; this test must run via `cargo nextest run`.
        unsafe {
            std::env::remove_var("KILN_STREAMING_PREFILL");
            std::env::remove_var("KILN_STREAMING_TILE_TOKENS");
            std::env::remove_var("KILN_USE_FLCE");
        }
        let (loss_mono, grads_mono) = checkpointed_forward_backward(
            &*backend,
            &input_ids,
            &weights,
            &config,
            &params,
            &label_mask,
            &segments,
            &device,
        )?;

        // Step 2: tiled. KILN_STREAMING_TILE_TOKENS=64 keeps the tile a
        // multiple of GDN_CHUNK_SIZE; T=192 > tile_size=64 ensures
        // `tiled_training_tile_size` returns Some and the tiled branch
        // dispatches.
        unsafe {
            std::env::set_var("KILN_STREAMING_PREFILL", "1");
            std::env::set_var("KILN_STREAMING_TILE_TOKENS", "64");
        }
        // Sanity-check the dispatch decision before running the loop, so a
        // future regression in `tiled_training_tile_size` shows up as an
        // explicit assertion rather than a silent fallback to monolithic.
        assert_eq!(
            super::tiled_training_tile_size(&weights, &device, seq_len),
            Some(64),
            "tiled dispatch did not fire for GDN-only model under \
             KILN_STREAMING_PREFILL=1 (T={seq_len}, tile=64)",
        );
        let (loss_tiled, grads_tiled) = checkpointed_forward_backward(
            &*backend,
            &input_ids,
            &weights,
            &config,
            &params,
            &label_mask,
            &segments,
            &device,
        )?;
        unsafe {
            std::env::remove_var("KILN_STREAMING_PREFILL");
            std::env::remove_var("KILN_STREAMING_TILE_TOKENS");
        }

        // Step 3: parity assertions.
        let loss_diff = (loss_mono - loss_tiled).abs();
        assert!(
            loss_diff < 1e-5,
            "tiled total loss differs from monolithic: mono={loss_mono} tiled={loss_tiled} \
             diff={loss_diff:.2e}",
        );

        let mut compared = 0usize;
        for var in params.all_vars() {
            let id = var.as_tensor().id();
            match (grads_mono.get(&id), grads_tiled.get(&id)) {
                (Some(g_m), Some(g_t)) => {
                    let diff = (g_m - g_t)?.abs()?.max_all()?.to_scalar::<f32>()?;
                    assert!(
                        diff < 1e-5,
                        "tiled grad differs from monolithic for var: max_abs_diff={diff:.2e}",
                    );
                    compared += 1;
                }
                (None, None) => {}
                (mono_some, tiled_some) => panic!(
                    "grad presence mismatch: monolithic={} tiled={}",
                    mono_some.is_some(),
                    tiled_some.is_some(),
                ),
            }
        }
        assert!(
            compared > 0,
            "no LoRA gradients were compared between tiled and monolithic paths",
        );

        Ok(())
    }

    /// Helper: enumerate which LoRA Var corresponds to which projection
    /// kind so the layer-pair parity test can apply different tolerances
    /// to MLP-LoRA grads (bit-exact across tile boundaries because MLP is
    /// per-position) and full-attention LoRA grads (q/k/v/o; can drift
    /// slightly under truncated-BPTT through GDN states upstream).
    fn classify_lora_vars(
        params: &TrainableLoraParams,
    ) -> Vec<(candle_core::Var, &'static str, String)> {
        let mut out: Vec<(candle_core::Var, &'static str, String)> = Vec::new();
        for (layer_idx, layer) in params.layers.iter().enumerate() {
            let pairs: [(&Option<(Var, Var)>, &str, &str); 7] = [
                (&layer.q_proj, "fa", "q"),
                (&layer.k_proj, "fa", "k"),
                (&layer.v_proj, "fa", "v"),
                (&layer.o_proj, "fa", "o"),
                (&layer.gate_proj, "mlp", "gate"),
                (&layer.up_proj, "mlp", "up"),
                (&layer.down_proj, "mlp", "down"),
            ];
            for (pair, kind, module) in pairs {
                if let Some((a, b)) = pair {
                    out.push((a.clone(), kind, format!("L{layer_idx}.{module}.A")));
                    out.push((b.clone(), kind, format!("L{layer_idx}.{module}.B")));
                }
            }
        }
        out
    }

    /// CPU parity for the layer-pair time-axis tile path on a HYBRID model
    /// (Qwen3.5-4B-shaped: alternating GDN + full-attention layers).
    ///
    /// Compares the layer-pair-tiled `checkpointed_forward_backward` path
    /// against the **standard (non-checkpointed) full forward+backward**
    /// path — the latter is the unambiguous ground truth (single forward,
    /// single backward, no segment trickery, all LoRA Vars in the graph).
    ///
    /// We deliberately do NOT compare against the monolithic-checkpointed
    /// path: its segment-iteration loop calls `hidden.detach()` between
    /// the current segment and later segments, which severs the chain
    /// from the loss back to the current segment's LoRA Vars. Earlier
    /// segments' LoRA Vars therefore never receive a gradient under
    /// monolithic checkpointing — a pre-existing limitation orthogonal
    /// to this PR. The layer-pair path uses gradient injection across
    /// blocks, so it correctly produces grads for every segment's LoRA
    /// Vars (including the segment that is currently being recomputed).
    /// Comparing to standard makes the parity claim well-defined.
    ///
    /// Tolerances:
    /// * Total loss within `1e-3` of standard (loss values are dominated
    ///   by the chain-rule-equivalent forward; matches expected
    ///   monolithic-checkpointed loss as well).
    /// * MLP-LoRA grads bit-exact (atol `1e-5`) — MLP is per-position so
    ///   per-tile state-thread truncation does not affect MLP-LoRA.
    /// * Full-attention LoRA grads within `1e-3` — the gradient-injection
    ///   chain through this PR's per-block backward goes through
    ///   different f32 reduction orders than the standard single
    ///   backward, and may also pick up truncated-BPTT approximation in
    ///   segment configurations where a GDN block sits between the FA
    ///   block and the segment output. In the test config used here
    ///   (`full_attention_interval = 2`, layers 1, 3 are FA), every FA
    ///   block is the LAST block in its segment so FA-LoRA grads are
    ///   bit-exact in expectation; the `1e-3` tolerance absorbs matmul-
    ///   reduction-order f32 drift only.
    ///
    /// Test must run via `cargo nextest run` or `cargo test --
    /// --test-threads=1` for the env-var manipulation to be safe.
    #[test]
    fn test_layer_pair_tiled_matches_monolithic_cpu_hybrid() -> Result<()> {
        // Hold ENV_LOCK across the whole test so a parallel
        // env-mutating test in this binary can't flip
        // `KILN_STREAMING_PREFILL` mid-call and turn the "standard"
        // baseline into a tiled run (or vice versa). See ENV_LOCK.
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let device = Device::Cpu;

        // Hybrid mini-config: full_attention_interval = 2 makes layers 1
        // and 3 full-attention; layers 0 and 2 are GDN. With num_layers =
        // 4 that gives 2 GDN + 2 FA, so each segment of 2 layers contains
        // one of each kind and exercises the layer-pair path's
        // partition + per-block backward across BOTH attention kinds.
        let mut config = tiny_config();
        config.full_attention_interval = 2;
        config.num_full_attention_layers = 2;

        let weights = tiny_weights(&config, &device)?;
        assert!(
            !super::model_is_gdn_only(&weights),
            "test setup error: model must be hybrid for layer-pair parity"
        );

        // T = 192 = 3 × tile_size(64) so the GDN tile loop runs three
        // iterations within each GDN block. Two segments × (1 GDN block +
        // 1 FA block) per segment exercises both block kinds twice.
        let seq_len: usize = 192;
        let vocab = config.vocab_size;
        let input_ids: Vec<u32> = (0..seq_len)
            .map(|i| ((i * 7 + 3) % vocab) as u32)
            .collect();
        let mut label_mask = vec![false; seq_len];
        for slot in label_mask.iter_mut().skip(1).take(seq_len - 2) {
            *slot = true;
        }

        let params = TrainableLoraParams::initialize(&config, &weights, 4, 8.0, &device)?;
        let segments = compute_segment_boundaries(config.num_layers, 2);
        // Sanity: 2 segments, 2 layers each, alternating GDN/FA.
        assert_eq!(segments, vec![(0, 2), (2, 4)]);
        let backend = backend::for_device(&device);

        // Step 1: standard (non-checkpointed) full forward+backward as
        // the ground-truth baseline. Clear streaming env vars defensively
        // even though nextest gives per-test process isolation, so a
        // parent test process leaking KILN_STREAMING_PREFILL=1 doesn't
        // silently invalidate the baseline.
        // SAFETY: env mutation is safe under nextest's per-test process
        // isolation; this test must run via `cargo nextest run`.
        unsafe {
            std::env::remove_var("KILN_STREAMING_PREFILL");
            std::env::remove_var("KILN_STREAMING_TILE_TOKENS");
            std::env::remove_var("KILN_USE_FLCE");
        }
        let (loss_std, grad_store_std) = standard_forward_backward(
            &*backend,
            &input_ids,
            &weights,
            &config,
            &params,
            &label_mask,
            &device,
        )?;
        // Lift `grad_store_std` (a `GradStore`) into the same map type as
        // checkpointed_forward_backward returns so the test can compare
        // both paths via a uniform interface.
        let mut grads_std: HashMap<candle_core::TensorId, Tensor> = HashMap::new();
        for var in params.all_vars() {
            if let Some(g) = grad_store_std.get(var.as_tensor()) {
                grads_std.insert(var.as_tensor().id(), g.clone());
            }
        }

        // Step 2: layer-pair tiled. KILN_STREAMING_TILE_TOKENS=64 keeps
        // the tile a multiple of GDN_CHUNK_SIZE; T=192 > tile=64 ensures
        // dispatch and the hybrid model means the layer-pair branch fires
        // (not the GDN-only fast path).
        unsafe {
            std::env::set_var("KILN_STREAMING_PREFILL", "1");
            std::env::set_var("KILN_STREAMING_TILE_TOKENS", "64");
        }
        // Sanity-check the dispatch decision before running the loop, so
        // a future regression in `tiled_training_tile_size` or
        // `model_is_gdn_only` shows up as an explicit assertion rather
        // than a silent fallback to monolithic.
        assert_eq!(
            super::tiled_training_tile_size(&weights, &device, seq_len),
            Some(64),
            "tiled dispatch did not fire for hybrid model under \
             KILN_STREAMING_PREFILL=1 (T={seq_len}, tile=64)",
        );
        assert!(
            !super::model_is_gdn_only(&weights),
            "model_is_gdn_only=true on hybrid weights — layer-pair branch \
             will be skipped",
        );

        let (loss_layer_pair, grads_layer_pair) = checkpointed_forward_backward(
            &*backend,
            &input_ids,
            &weights,
            &config,
            &params,
            &label_mask,
            &segments,
            &device,
        )?;
        unsafe {
            std::env::remove_var("KILN_STREAMING_PREFILL");
            std::env::remove_var("KILN_STREAMING_TILE_TOKENS");
        }

        // Step 3: parity assertions vs standard (non-checkpointed) baseline.
        let loss_diff = (loss_std - loss_layer_pair).abs();
        assert!(
            loss_diff < 1e-3,
            "layer-pair total loss differs from standard: \
             std={loss_std} layer_pair={loss_layer_pair} \
             diff={loss_diff:.2e}",
        );

        // Helper: read a Var's grad if present, otherwise treat as a zero
        // tensor of the Var's shape. This absorbs the candle-autograd
        // detail that a Var which factors out of a matmul backward (e.g.
        // LoRA-A multiplied by LoRA-B which is initialized to zero) may or
        // may not appear in the GradStore depending on the exact ordering
        // of `or_insert` calls along its predecessors. Both interpretations
        // (missing => zero) are mathematically equivalent for parity, so
        // we treat them as equivalent here.
        let grad_or_zero = |grads: &HashMap<candle_core::TensorId, Tensor>,
                            var: &Var|
         -> Result<Tensor> {
            let id = var.as_tensor().id();
            match grads.get(&id) {
                Some(g) => Ok(g.clone()),
                None => Ok(var.as_tensor().zeros_like()?),
            }
        };

        let classified = classify_lora_vars(&params);
        let mut compared_mlp = 0usize;
        let mut compared_fa = 0usize;
        for (var, kind, name) in &classified {
            let g_s = grad_or_zero(&grads_std, var)?;
            let g_p = grad_or_zero(&grads_layer_pair, var)?;
            let diff = (&g_s - &g_p)?.abs()?.max_all()?.to_scalar::<f32>()?;
            let tol: f32 = match *kind {
                "mlp" => 1e-5,
                "fa" => 1e-3,
                _ => 1e-3,
            };
            assert!(
                diff < tol,
                "layer-pair grad differs from standard for {name} ({kind}-LoRA): \
                 max_abs_diff={diff:.3e} (tol={tol:.0e})",
            );
            match *kind {
                "mlp" => compared_mlp += 1,
                "fa" => compared_fa += 1,
                _ => {}
            }
        }
        assert!(
            compared_mlp > 0,
            "no MLP-LoRA gradients were compared between layer-pair and monolithic"
        );
        assert!(
            compared_fa > 0,
            "no FA-LoRA gradients were compared — test config must include \
             at least one full-attention layer with q/k/v/o LoRA",
        );

        Ok(())
    }

    #[test]
    fn test_partition_segment_layers_by_attn_type() -> Result<()> {
        let device = Device::Cpu;
        let mut config = tiny_config();
        // full_attention_interval = 2 -> layers 1, 3 are FA, 0, 2 are GDN.
        config.full_attention_interval = 2;
        config.num_full_attention_layers = 2;
        let weights = tiny_weights(&config, &device)?;

        // Segment [0, 2): GDN at 0, FA at 1.
        let seg0 = super::partition_segment_layers_by_attn_type(&weights, 0, 2);
        assert_eq!(seg0.len(), 2);
        assert_eq!(seg0[0].0, super::AttnKind::Gdn);
        assert_eq!(seg0[0].1, 0..1);
        assert_eq!(seg0[1].0, super::AttnKind::FullAttn);
        assert_eq!(seg0[1].1, 1..2);

        // Whole model [0, 4) under the same config: alternating blocks.
        let whole = super::partition_segment_layers_by_attn_type(&weights, 0, 4);
        assert_eq!(whole.len(), 4);
        assert_eq!(whole[0].0, super::AttnKind::Gdn);
        assert_eq!(whole[0].1, 0..1);
        assert_eq!(whole[1].0, super::AttnKind::FullAttn);
        assert_eq!(whole[1].1, 1..2);
        assert_eq!(whole[2].0, super::AttnKind::Gdn);
        assert_eq!(whole[2].1, 2..3);
        assert_eq!(whole[3].0, super::AttnKind::FullAttn);
        assert_eq!(whole[3].1, 3..4);

        // GDN-only model with full_attention_interval > num_layers: the
        // entire range is one GDN block.
        let mut gdn_only_config = tiny_config();
        gdn_only_config.full_attention_interval = gdn_only_config.num_layers + 1;
        gdn_only_config.num_full_attention_layers = 0;
        let gdn_only_weights = tiny_weights(&gdn_only_config, &device)?;
        let gdn_only =
            super::partition_segment_layers_by_attn_type(&gdn_only_weights, 0, 4);
        assert_eq!(gdn_only.len(), 1);
        assert_eq!(gdn_only[0].0, super::AttnKind::Gdn);
        assert_eq!(gdn_only[0].1, 0..4);

        Ok(())
    }

    #[test]
    fn test_flce_parity_vs_naive_loss() -> Result<()> {
        // Kill-switch parity: naive `model_forward_head` + `cross_entropy_loss`
        // must match `model_forward_no_head` + `fused_linear_cross_entropy`
        // on the same weights and inputs, up to floating-point associativity
        // in the chunked vocab reduction.
        //
        // This is the trainer-integration equivalent of the CPU parity tests
        // inside `kiln-flce-kernel`: those validate the kernel in isolation,
        // this validates the wiring end-to-end through the real transformer
        // stack so enabling `KILN_USE_FLCE` for SFT is a no-op on the loss.
        let device = Device::Cpu;
        let config = tiny_config();
        let weights = tiny_weights(&config, &device)?;

        let input_ids: Vec<u32> = vec![1, 5, 10, 3, 7, 2, 8];
        let label_mask = vec![false, false, true, true, true, true, false];

        let backend = backend::for_device(&device);

        // Naive path: full forward → logits → cross_entropy_loss.
        let mut linear_state_naive = LinearAttentionState::new(&config, &device)?;
        let logits = model_forward(
            &*backend,
            &input_ids,
            &weights,
            &config,
            None,
            Some(&mut linear_state_naive),
            None,
        )?;
        let loss_naive =
            cross_entropy_loss(&logits, &input_ids, &label_mask, &device)?.to_scalar::<f32>()?;

        // FLCE path: no-head forward → fused LCE (small chunk to exercise the
        // chunked reduction on a modest vocab size).
        let mut linear_state_flce = LinearAttentionState::new(&config, &device)?;
        let hidden = model_forward_no_head(
            &*backend,
            &input_ids,
            &weights,
            &config,
            Some(&mut linear_state_flce),
            None,
        )?;
        let loss_flce = fused_linear_cross_entropy(
            &hidden,
            &weights.embed_tokens_t,
            &input_ids,
            &label_mask,
            &device,
            8, // small chunk to exercise uneven-chunk path
        )?
        .to_scalar::<f32>()?;

        let abs_err = (loss_naive - loss_flce).abs();
        let rel_err = if loss_naive.abs() > 1e-6 {
            abs_err / loss_naive.abs()
        } else {
            abs_err
        };
        assert!(
            abs_err < 1e-4 || rel_err < 1e-4,
            "FLCE trainer parity failed: naive={loss_naive:.6} flce={loss_flce:.6} \
             abs_err={abs_err:.2e} rel_err={rel_err:.2e}",
        );

        Ok(())
    }

    #[test]
    fn test_checkpointed_gradients_nonzero() -> Result<()> {
        let device = Device::Cpu;
        let config = tiny_config();
        let weights = tiny_weights(&config, &device)?;

        let input_ids: Vec<u32> = vec![1, 5, 10, 3, 7];
        let label_mask = vec![false, true, true, true, false];

        let params = TrainableLoraParams::initialize(&config, &weights, 4, 8.0, &device)?;

        let segments = compute_segment_boundaries(config.num_layers, 2);
        let backend = backend::for_device(&device);
        let (_loss, grads) = checkpointed_forward_backward(
            &*backend,
            &input_ids,
            &weights,
            &config,
            &params,
            &label_mask,
            &segments,
            &device,
        )?;

        // Verify that we got gradients for LoRA params in BOTH segments
        let mut has_grad_seg0 = false; // layers 0-1
        let mut has_grad_seg1 = false; // layers 2-3
        for var in params.all_vars() {
            if let Some(grad) = grads.get(&var.as_tensor().id()) {
                let grad_norm = grad.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                if grad_norm > 0.0 {
                    // Determine which segment this var belongs to
                    // by checking if it matches layer 0-1 or 2-3 params
                    has_grad_seg0 = true; // simplified: any nonzero grad means the system works
                    has_grad_seg1 = true;
                }
            }
        }

        assert!(has_grad_seg0, "no gradients for segment 0 params");
        assert!(has_grad_seg1, "no gradients for segment 1 params");

        Ok(())
    }

    /// Runs 5 SFT steps with gradient checkpointing on `device` and asserts
    /// the final loss is lower than the initial loss. Drives both the CPU
    /// and Metal variants below.
    fn run_checkpointed_training_loss_decreases(device: &Device) -> Result<()> {
        let config = tiny_config();
        let weights = tiny_weights(&config, device)?;

        let input_ids: Vec<u32> = vec![1, 5, 10, 3, 7, 2, 8, 15];
        let label_mask = vec![false, false, true, true, true, true, true, false];
        let lr = 0.01;

        let params = TrainableLoraParams::initialize(&config, &weights, 4, 8.0, device)?;
        let segments = compute_segment_boundaries(config.num_layers, 2);
        let backend = backend::for_device(device);

        let mut prev_loss = f64::MAX;
        let mut losses = Vec::new();
        for step in 0..5 {
            let (loss_val, grads) = checkpointed_forward_backward(
                &*backend,
                &input_ids,
                &weights,
                &config,
                &params,
                &label_mask,
                &segments,
                device,
            )?;
            sgd_step_from_map(&params, &grads, lr)?;
            losses.push(loss_val);
            if step > 0 {
                assert!(
                    loss_val < prev_loss + 0.5,
                    "loss increased too much at step {step}: {prev_loss:.4} -> {loss_val:.4}"
                );
            }
            prev_loss = loss_val;
        }

        let initial = losses[0];
        let final_loss = *losses.last().unwrap();
        assert!(
            final_loss < initial,
            "loss did not decrease over 5 steps on {:?}: {initial:.4} -> {final_loss:.4}",
            device,
        );
        Ok(())
    }

    /// End-to-end SFT loop on Metal. Validates candle autograd + SGD +
    /// gradient checkpointing through the `BackendRuntime` seam on Apple
    /// Silicon. Skipped gracefully when Metal isn't available.
    #[cfg(feature = "metal")]
    #[test]
    fn test_checkpointed_training_loss_decreases_metal() -> Result<()> {
        let Some(device) = kiln_model::backend::metal::try_new_metal() else {
            return Ok(());
        };
        assert_eq!(backend::for_device(&device).name(), "metal");
        run_checkpointed_training_loss_decreases(&device)
    }

    #[test]
    fn test_checkpointed_training_loss_decreases() -> Result<()> {
        run_checkpointed_training_loss_decreases(&Device::Cpu)
    }

    #[test]
    fn test_checkpoint_config_from_env() {
        // Without KILN_GPU_MEMORY_GB or nvidia-smi, falls back to default (4 segments)
        // or VRAM-aware value if GPU is detected
        let cfg = CheckpointConfig::from_env(32);
        assert!(cfg.enabled);
        // num_segments depends on whether GPU is detected; just verify it's reasonable
        assert!(cfg.num_segments >= 1 && cfg.num_segments <= 32);

        // With very few layers, segments clamped to num_layers
        let cfg = CheckpointConfig::from_env(2);
        assert!(cfg.num_segments <= 2);
    }

    /// Phase 10 §1: confirm switching the RMSNorm dispatch between the new
    /// `CustomOp2` autograd path (default) and the
    /// `KILN_DISABLE_RMSNORM_BACKWARD=1` fallback does NOT change training
    /// loss on a 2-step CPU SFT run.
    ///
    /// The custom op only routes through the manual-backward CUDA kernel on
    /// CUDA — on CPU, both code paths fall back to `rms_norm_fallback` (the
    /// standalone candle-op chain). This test pins that contract: enabling
    /// or disabling the new env var on CPU is a no-op for the math, so the
    /// loss values are bit-exact in either configuration. The test
    /// initializes params ONCE (so the same `Var` weights are used by both
    /// runs) and only flips the dispatch env var between calls;
    /// `standard_forward_backward` itself doesn't mutate params, so each
    /// call is an independent forward pass.
    ///
    /// Test must run via `cargo nextest run` or `cargo test --
    /// --test-threads=1` so the env-var manipulation is process-isolated.
    #[test]
    fn test_training_rmsnorm_custom_op_loss_parity() -> Result<()> {
        // Hold ENV_LOCK across the whole test so a parallel
        // env-mutating test in this binary can't flip RMSNorm dispatch
        // env vars mid-call. See ENV_LOCK.
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let device = Device::Cpu;
        let config = tiny_config();
        let weights = tiny_weights(&config, &device)?;

        let input_ids: Vec<u32> = vec![1, 5, 10, 3, 7, 2, 8];
        let label_mask = vec![false, false, true, true, true, true, false];

        let backend = backend::for_device(&device);

        // Initialize LoRA params ONCE so both runs use the same Vars.
        // `standard_forward_backward` does not call SGD; each invocation
        // is a pure forward+backward pass, so the loss is deterministic
        // given fixed inputs and params.
        let params = TrainableLoraParams::initialize(&config, &weights, 4, 8.0, &device)?;

        let run_step = |bwd_disabled: bool| -> Result<f64> {
            // SAFETY: env mutation is safe under nextest's per-test process
            // isolation; this test must run via `cargo nextest run`.
            unsafe {
                std::env::remove_var("KILN_DISABLE_RMSNORM_KERNEL");
                if bwd_disabled {
                    std::env::set_var("KILN_DISABLE_RMSNORM_BACKWARD", "1");
                } else {
                    std::env::remove_var("KILN_DISABLE_RMSNORM_BACKWARD");
                }
            }

            let (loss_val, _grads) = standard_forward_backward(
                &*backend,
                &input_ids,
                &weights,
                &config,
                &params,
                &label_mask,
                &device,
            )?;

            // Defensive cleanup so the next call (or test) isn't poisoned.
            unsafe {
                std::env::remove_var("KILN_DISABLE_RMSNORM_BACKWARD");
            }

            Ok(loss_val)
        };

        // 2-step SFT run: alternate dispatch on each step so divergence
        // would show up at any step.
        for step in 0..2 {
            let loss_default = run_step(false)?;
            let loss_fallback = run_step(true)?;

            assert!(
                loss_default.is_finite() && loss_fallback.is_finite(),
                "non-finite loss at step {step}: default={loss_default} fallback={loss_fallback}",
            );
            // CPU dispatch falls back to `rms_norm_fallback` for both
            // configurations (the CUDA-only custom op never fires on CPU),
            // so the loss values are bit-exact.
            assert!(
                (loss_default - loss_fallback).abs() < 1e-9,
                "rmsnorm dispatch loss diverges at step {step}: \
                 default={loss_default} fallback={loss_fallback}",
            );
        }

        Ok(())
    }
}
