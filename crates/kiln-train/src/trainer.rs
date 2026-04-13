//! In-process LoRA SFT training using candle autograd.
//!
//! Trains LoRA adapter weights directly on the already-loaded model's GPU
//! tensors. No Python sidecar, no second model copy, single process.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, Var};

use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::forward::{
    model_forward, GpuWeights, LinearAttentionState,
};
use kiln_model::lora_loader::{
    LoraLayerWeights, LoraProjectionWeights, LoraWeights,
};

use crate::{ChatMessage, SftConfig, SftExample};

/// Convert our ChatMessage to the core tokenizer's ChatMessage.
fn to_core_messages(msgs: &[ChatMessage]) -> Vec<kiln_core::tokenizer::ChatMessage> {
    msgs.iter()
        .map(|m| kiln_core::tokenizer::ChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
        })
        .collect()
}

/// Which linear projections to train LoRA on.
const DEFAULT_TARGET_MODULES: &[&str] = &[
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
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
                        // Get actual dimensions from the weight tensor
                        let w = match &layer_weights.attention {
                            kiln_model::forward::GpuAttentionWeights::Full(full) => {
                                match module {
                                    "q_proj" => &full.q_proj,
                                    "k_proj" => &full.k_proj,
                                    "v_proj" => &full.v_proj,
                                    "o_proj" => &full.o_proj,
                                    _ => unreachable!(),
                                }
                            }
                            // Linear attention layers don't have q/k/v/o_proj
                            kiln_model::forward::GpuAttentionWeights::Linear(_) => {
                                continue;
                            }
                        };
                        let dims = w.dims();
                        // Weight is [out_features, in_features]
                        (dims[1], dims[0], bound_hidden)
                    }
                    "gate_proj" => (hidden, intermediate, bound_hidden),
                    "up_proj" => (hidden, intermediate, bound_hidden),
                    "down_proj" => (intermediate, hidden, bound_intermediate),
                    _ => continue,
                };

                // A: [rank, in_features] — Kaiming uniform
                let a = Var::rand_f64(
                    -bound, bound,
                    (rank, in_features),
                    DType::F32,
                    device,
                ).with_context(|| format!("init LoRA A for layer {layer_idx} {module}"))?;

                // B: [out_features, rank] — zeros
                let b = Var::zeros(
                    (out_features, rank),
                    DType::F32,
                    device,
                ).with_context(|| format!("init LoRA B for layer {layer_idx} {module}"))?;

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
        let layers: Vec<LoraLayerWeights> = self.layers.iter().map(|lp| {
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
        }).collect();

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
                &layer.q_proj, &layer.k_proj, &layer.v_proj, &layer.o_proj,
                &layer.gate_proj, &layer.up_proj, &layer.down_proj,
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
            let mut save_proj = |name: &str, pair: &Option<(Var, Var)>,
                             is_attn: bool| {
                if let Some((a, b)) = pair {
                    let sub = if is_attn { "self_attn" } else { "mlp" };
                    let prefix = format!(
                        "base_model.model.model.layers.{layer_idx}.{sub}.{name}"
                    );
                    tensor_data.insert(
                        format!("{prefix}.lora_A.weight"),
                        a.as_tensor().clone(),
                    );
                    tensor_data.insert(
                        format!("{prefix}.lora_B.weight"),
                        b.as_tensor().clone(),
                    );
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
        .filter_map(|ex| {
            match tokenize_for_training(ex, tokenizer) {
                Ok(t) => Some(t),
                Err(e) => {
                    tracing::warn!("skipping example: {e}");
                    None
                }
            }
        })
        .collect();

    if tokenized.is_empty() {
        anyhow::bail!("no valid training examples after tokenization");
    }

    let total_steps = config.epochs * tokenized.len();
    let mut global_step = 0;
    let mut last_loss = 0.0;

    for epoch in 0..config.epochs {
        let mut epoch_loss = 0.0;

        for (_ex_idx, (input_ids, label_mask)) in tokenized.iter().enumerate() {
            // Forward pass with trainable LoRA weights.
            let lora_weights = params.as_lora_weights();

            // Create fresh linear attention state for each example
            let mut linear_state = LinearAttentionState::new(model_config, &device)?;

            let logits = model_forward(
                input_ids,
                weights,
                model_config,
                None,                          // no KV cache needed for training
                Some(&mut linear_state),
                Some(&lora_weights),
            ).context("training forward pass")?;

            // Compute cross-entropy loss on assistant tokens only.
            let loss = cross_entropy_loss(&logits, input_ids, label_mask, &device)?;
            let loss_val = loss.to_scalar::<f32>()? as f64;
            epoch_loss += loss_val;
            last_loss = loss_val;

            // Backward pass: compute gradients for all trainable vars.
            let grads = loss.backward().context("backward pass")?;

            // SGD update step
            sgd_step(&params, &grads, config.learning_rate)?;

            global_step += 1;

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
        }

        let avg_loss = epoch_loss / tokenized.len() as f64;
        tracing::info!(
            epoch = epoch + 1,
            avg_loss = format!("{avg_loss:.6}"),
            "epoch complete"
        );
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
    let full_text = tokenizer.apply_chat_template(&core_messages)
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
            let prefix_text = tokenizer.apply_chat_template(&prefix_messages)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let prefix_ids = tokenizer
                .encode(&prefix_text)
                .map_err(|e| anyhow::anyhow!("{e}"))?;

            // Tokenize everything before this assistant message
            let before_messages: Vec<_> = prefix_messages[..prefix_messages.len() - 1].to_vec();
            let before_text = if before_messages.is_empty() {
                String::new()
            } else {
                tokenizer.apply_chat_template(&before_messages)
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
        active_positions.iter().map(|&i| i as u32).collect::<Vec<_>>().as_slice(),
        device,
    )?;
    let active_logits = shift_logits.index_select(&indices, 0)?; // [num_active, vocab_size]

    let active_labels: Vec<u32> = active_positions
        .iter()
        .map(|&i| shift_labels[i])
        .collect();
    let labels_tensor = Tensor::new(active_labels.as_slice(), device)?
        .to_dtype(DType::U32)?;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_entropy_loss_basic() -> Result<()> {
        let device = Device::Cpu;

        // 3 tokens, vocab size 4
        // logits: [1, 3, 4]
        let logits = Tensor::new(
            &[[[2.0f32, 1.0, 0.1, 0.0],
               [0.0, 3.0, 0.1, 0.0],
               [0.0, 0.0, 0.0, 5.0]]],
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
}
