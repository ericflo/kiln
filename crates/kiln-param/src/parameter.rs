//! `Parameter` — one logical parameter, one stable `TensorId`,
//! multiple physical storages.
//!
//! Per the Phase 2.5 issue bullet:
//!
//! > `Parameter { forward_storage, backward_storage, transposed_cache?,
//! > lora_delta?, tensor_id }` in a new `kiln-param` crate.
//!
//! # Anti-pattern 11 enforcement
//!
//! The `tensor_id` is **stable across storage-variant transitions**.
//! Adding a Marlin/FP8 forward variant on top of an existing BF16-
//! master, or hot-swapping a LoRA delta, does NOT change the id.
//! Otherwise AdamW moments (`HashMap<TensorId, AdamWMoments>` at
//! `crates/kiln-train/src/trainer.rs:555,592`) get orphaned on
//! weight-form transitions.

use std::sync::Arc;

use kiln_tensor::{Storage, Tensor, TensorId};

use crate::AmpPolicy;

/// Forward-storage variant. Drives the `forward()` dispatch.
///
/// `#[non_exhaustive]` — Phase 8.10 adds `Fp4Packed`; Phase 8.9 may add
/// 2:4-sparse Marlin.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ForwardStorage {
    /// Raw BF16 / F16 / F32 tensor. The default for non-quantized
    /// parameters.
    Plain(Tensor),
    /// Marlin W4A16 packed storage. Forward-only (anti-pattern: the
    /// supported quantized-training mode is LoRA-around-Marlin per
    /// Phase 2.5).
    ///
    /// Carries:
    /// - `packed`: the int4-packed weight tensor (DType::Int4Packed)
    /// - `scales`: per-channel BF16 scales
    Marlin {
        packed: Tensor,
        scales: Tensor,
    },
    /// FP8 E4M3 forward storage. Phase 8.4 training path.
    Fp8 {
        packed: Tensor,
        scales: Tensor,
    },
    /// FP4-packed forward storage. **Phase 8.10 stub** — no impl yet.
    Fp4Packed {
        packed: Tensor,
        scales: Tensor,
    },
}

impl ForwardStorage {
    /// Stable kind name. Used by the storage-coherence state machine
    /// + the autotune cache key.
    pub fn kind_name(&self) -> &'static str {
        match self {
            ForwardStorage::Plain(_) => "plain",
            ForwardStorage::Marlin { .. } => "marlin",
            ForwardStorage::Fp8 { .. } => "fp8",
            ForwardStorage::Fp4Packed { .. } => "fp4_packed",
        }
    }

    /// Borrow the primary tensor (the packed/plain weight). For Marlin
    /// / FP8 / FP4Packed this is the *packed* form, not the master.
    pub fn primary_tensor(&self) -> &Tensor {
        match self {
            ForwardStorage::Plain(t)
            | ForwardStorage::Marlin { packed: t, .. }
            | ForwardStorage::Fp8 { packed: t, .. }
            | ForwardStorage::Fp4Packed { packed: t, .. } => t,
        }
    }

    /// Borrow the primary storage (`Arc<dyn StorageBackend>`).
    pub fn primary_storage(&self) -> &Storage {
        self.primary_tensor().storage()
    }
}

/// Role tag for an output head. Composes with anti-pattern 17 (tied
/// weights — `lm_head` ← `embed_tokens`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OutputHeadRole {
    /// The language-model head producing per-token logits.
    LmHead,
    /// Multi-token-prediction head (Qwen3.5-4B's k=1 MTP).
    MtpHead,
    /// Value head for RL training (PPO, GRPO with critic).
    ValueHead,
    /// Reward-model head for preference training (DPO/IPO with explicit
    /// reward inference).
    RewardHead,
}

impl OutputHeadRole {
    pub const fn name(self) -> &'static str {
        match self {
            OutputHeadRole::LmHead => "lm_head",
            OutputHeadRole::MtpHead => "mtp_head",
            OutputHeadRole::ValueHead => "value_head",
            OutputHeadRole::RewardHead => "reward_head",
        }
    }
}

/// One output head sharing trunk storage. Lives on a `Parameter`
/// alongside the forward/backward storages.
#[derive(Debug, Clone)]
pub struct OutputHead {
    pub role: OutputHeadRole,
    pub head_storage: Tensor,
    pub requires_grad: bool,
}

/// Unified Parameter handle.
///
/// **One logical parameter, one stable `TensorId`, multiple physical
/// storages.** Replaces the bookkeeping spread across
/// `packed_weight_registry.rs`, `transposed_weight_cache.rs`,
/// `marlin_proj.rs`, `fp8.rs`, and `lora_loader.rs`.
#[derive(Debug, Clone)]
pub struct Parameter {
    /// Stable identity. Survives quantization, LoRA swap, transposed-
    /// cache add/remove. Optimizer keys on this.
    tensor_id: TensorId,
    /// Forward-pass storage. Hot-path read.
    forward_storage: ForwardStorage,
    /// Optional BF16/F32 master. Always populated for trainable
    /// parameters; `None` for pure-inference deployments (the
    /// 16 GiB-serve-only tier).
    backward_storage: Option<Tensor>,
    /// Optional transposed cache (the existing `gate_up_proj_t` /
    /// `qkv_proj_t` slots).
    transposed_cache: Option<Tensor>,
    /// Optional LoRA delta. Hot-swappable without changing the
    /// `tensor_id` (anti-pattern 11).
    lora_delta: Option<Tensor>,
    /// Output heads sharing this parameter as their trunk.
    heads: Vec<OutputHead>,
    /// Per-Parameter AMP policy.
    amp_policy: AmpPolicy,
    /// Human-readable name (e.g. "model.layers.0.mlp.gate_proj.weight").
    /// Used by checkpoint save and by Phase 9 diagnostics. Optional —
    /// not every parameter has a meaningful safetensors name.
    name: Option<Arc<str>>,
}

impl Parameter {
    /// Construct an inference-only Parameter (no `backward_storage`).
    pub fn inference_only(forward_storage: ForwardStorage) -> Self {
        let tensor_id = forward_storage.primary_tensor().id();
        Parameter {
            tensor_id,
            forward_storage,
            backward_storage: None,
            transposed_cache: None,
            lora_delta: None,
            heads: Vec::new(),
            amp_policy: AmpPolicy::default(),
            name: None,
        }
    }

    /// Construct a trainable Parameter with master storage.
    pub fn trainable(forward_storage: ForwardStorage, master: Tensor, policy: AmpPolicy) -> Self {
        let tensor_id = forward_storage.primary_tensor().id();
        Parameter {
            tensor_id,
            forward_storage,
            backward_storage: Some(master),
            transposed_cache: None,
            lora_delta: None,
            heads: Vec::new(),
            amp_policy: policy,
            name: None,
        }
    }

    /// Stable identity.
    pub fn tensor_id(&self) -> TensorId {
        self.tensor_id
    }

    /// Borrow forward storage.
    pub fn forward_storage(&self) -> &ForwardStorage {
        &self.forward_storage
    }

    /// Mutate the forward storage (e.g. swap BF16 → Marlin in place).
    /// **Preserves `tensor_id`** per anti-pattern 11.
    pub fn replace_forward_storage(&mut self, new: ForwardStorage) {
        self.forward_storage = new;
        // tensor_id intentionally unchanged.
    }

    /// Borrow backward storage (the master tensor).
    pub fn backward_storage(&self) -> Option<&Tensor> {
        self.backward_storage.as_ref()
    }

    /// Borrow the transposed cache if present.
    pub fn transposed_cache(&self) -> Option<&Tensor> {
        self.transposed_cache.as_ref()
    }

    /// Set / replace the transposed cache.
    pub fn set_transposed_cache(&mut self, cache: Tensor) {
        self.transposed_cache = Some(cache);
    }

    /// Borrow the LoRA delta if present.
    pub fn lora_delta(&self) -> Option<&Tensor> {
        self.lora_delta.as_ref()
    }

    /// Set / replace the LoRA delta. **Preserves `tensor_id`** per
    /// anti-pattern 11 — the delta is part of the parameter, not a
    /// new parameter.
    pub fn set_lora_delta(&mut self, delta: Option<Tensor>) {
        self.lora_delta = delta;
    }

    /// Output heads attached to this parameter.
    pub fn heads(&self) -> &[OutputHead] {
        &self.heads
    }

    /// Attach a new output head.
    pub fn add_head(&mut self, head: OutputHead) {
        self.heads.push(head);
    }

    /// AMP policy (Phase 6.5 `kiln-optim` reads this).
    pub fn amp_policy(&self) -> AmpPolicy {
        self.amp_policy
    }

    /// Override the AMP policy. Use sparingly — the policy is
    /// declared at construction.
    pub fn set_amp_policy(&mut self, policy: AmpPolicy) {
        self.amp_policy = policy;
    }

    /// Optional safetensors-name. Set during model load.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Set the parameter's name (e.g. during safetensors load).
    pub fn set_name(&mut self, name: impl Into<Arc<str>>) {
        self.name = Some(name.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_tensor::{DType, Tensor};

    fn plain_f32() -> ForwardStorage {
        ForwardStorage::Plain(Tensor::zeros_cpu(vec![4, 4], DType::F32))
    }

    #[test]
    fn inference_only_no_master() {
        let p = Parameter::inference_only(plain_f32());
        assert!(p.backward_storage().is_none());
        assert!(p.lora_delta().is_none());
        assert!(p.transposed_cache().is_none());
        assert!(p.heads().is_empty());
        assert_eq!(p.amp_policy(), AmpPolicy::default());
    }

    #[test]
    fn trainable_carries_master() {
        let fwd = plain_f32();
        let master = Tensor::zeros_cpu(vec![4, 4], DType::BF16);
        let p = Parameter::trainable(fwd, master, AmpPolicy::default());
        assert!(p.backward_storage().is_some());
    }

    #[test]
    fn replace_forward_preserves_tensor_id() {
        // Anti-pattern 11 contract.
        let original = plain_f32();
        let original_id = original.primary_tensor().id();
        let mut p = Parameter::inference_only(original);
        assert_eq!(p.tensor_id(), original_id);

        // Swap forward to a Marlin form. Even though the new packed
        // Tensor has a *different* id, the Parameter's tensor_id
        // stays the same.
        let new_fwd = ForwardStorage::Marlin {
            packed: Tensor::zeros_cpu(vec![4, 4], DType::Int4Packed),
            scales: Tensor::zeros_cpu(vec![4], DType::BF16),
        };
        p.replace_forward_storage(new_fwd);
        assert_eq!(p.tensor_id(), original_id);
        assert_eq!(p.forward_storage().kind_name(), "marlin");
    }

    #[test]
    fn lora_swap_preserves_tensor_id() {
        let mut p = Parameter::inference_only(plain_f32());
        let id = p.tensor_id();
        let delta = Tensor::zeros_cpu(vec![4, 4], DType::BF16);
        p.set_lora_delta(Some(delta));
        assert!(p.lora_delta().is_some());
        assert_eq!(p.tensor_id(), id);
        p.set_lora_delta(None);
        assert!(p.lora_delta().is_none());
        assert_eq!(p.tensor_id(), id);
    }

    #[test]
    fn transposed_cache_slot() {
        let mut p = Parameter::inference_only(plain_f32());
        assert!(p.transposed_cache().is_none());
        let cache = Tensor::zeros_cpu(vec![4, 4], DType::BF16);
        p.set_transposed_cache(cache);
        assert!(p.transposed_cache().is_some());
    }

    #[test]
    fn output_head_registry() {
        let mut p = Parameter::inference_only(plain_f32());
        let head_storage = Tensor::zeros_cpu(vec![4, 16], DType::F32);
        p.add_head(OutputHead {
            role: OutputHeadRole::LmHead,
            head_storage: head_storage.clone(),
            requires_grad: true,
        });
        p.add_head(OutputHead {
            role: OutputHeadRole::MtpHead,
            head_storage,
            requires_grad: true,
        });
        assert_eq!(p.heads().len(), 2);
        assert_eq!(p.heads()[0].role, OutputHeadRole::LmHead);
        assert_eq!(p.heads()[1].role, OutputHeadRole::MtpHead);
    }

    #[test]
    fn forward_storage_kind_names() {
        assert_eq!(plain_f32().kind_name(), "plain");
        assert_eq!(
            ForwardStorage::Marlin {
                packed: Tensor::zeros_cpu(vec![1], DType::Int4Packed),
                scales: Tensor::zeros_cpu(vec![1], DType::BF16),
            }
            .kind_name(),
            "marlin"
        );
        assert_eq!(
            ForwardStorage::Fp8 {
                packed: Tensor::zeros_cpu(vec![1], DType::F8E4M3),
                scales: Tensor::zeros_cpu(vec![1], DType::BF16),
            }
            .kind_name(),
            "fp8"
        );
    }

    #[test]
    fn output_head_role_names() {
        assert_eq!(OutputHeadRole::LmHead.name(), "lm_head");
        assert_eq!(OutputHeadRole::MtpHead.name(), "mtp_head");
        assert_eq!(OutputHeadRole::ValueHead.name(), "value_head");
        assert_eq!(OutputHeadRole::RewardHead.name(), "reward_head");
    }

    #[test]
    fn set_name_round_trips() {
        let mut p = Parameter::inference_only(plain_f32());
        assert!(p.name().is_none());
        p.set_name("model.layers.0.mlp.gate_proj.weight");
        assert_eq!(p.name(), Some("model.layers.0.mlp.gate_proj.weight"));
    }

    #[test]
    fn replace_amp_policy() {
        let mut p = Parameter::inference_only(plain_f32());
        assert_eq!(p.amp_policy(), AmpPolicy::default());
        p.set_amp_policy(AmpPolicy::fp32_reference());
        assert_eq!(p.amp_policy(), AmpPolicy::fp32_reference());
    }
}
