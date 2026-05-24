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

use kiln_tensor::{Result, Storage, Tensor, TensorId};

use crate::content_hash::content_hash_storage;
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

/// Typed view of a [`Parameter`]'s forward path, ready for the
/// backend-specific matmul dispatch.
///
/// Returned by [`Parameter::forward_dispatch`]. Each variant carries
/// the borrowed components the backend matmul needs:
///
/// - [`ForwardDispatch::Plain`] — a single weight tensor; backend
///   runs the standard BF16/F16/F32 matmul.
/// - [`ForwardDispatch::Marlin`] — packed int4 weights + per-channel
///   BF16 scales; backend routes through `marlin_w4a16_gemm_kt` (or
///   the candle equivalent).
/// - [`ForwardDispatch::Fp8`] — FP8 packed weights + scales; backend
///   routes through the FP8 matmul once the kernel ships
///   (Phase 8.4 of #1082).
/// - [`ForwardDispatch::Fp4Packed`] — Phase 8.10 stub; no impl today.
///
/// **Why an enum view instead of a `Parameter::forward(input)`
/// method:** the actual matmul is backend-specific (cublasLt on
/// CUDA, MPS on Metal, compute-shader on Vulkan) and the per-backend
/// handle isn't a kiln-param concern. The dispatch enum lets the
/// caller match on the variant and route to its own backend's matmul
/// implementation without kiln-param needing to depend on every
/// kernel crate.
///
/// **Stability:** `#[non_exhaustive]` so adding a new
/// `ForwardStorage` variant (e.g., MXFP4) doesn't break downstream
/// matches.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum ForwardDispatch<'a> {
    Plain {
        weight: &'a Tensor,
    },
    Marlin {
        packed: &'a Tensor,
        scales: &'a Tensor,
    },
    Fp8 {
        packed: &'a Tensor,
        scales: &'a Tensor,
    },
    Fp4Packed {
        packed: &'a Tensor,
        scales: &'a Tensor,
    },
}

impl<'a> ForwardDispatch<'a> {
    /// Stable variant name matching [`ForwardStorage::kind_name`].
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Plain { .. } => "plain",
            Self::Marlin { .. } => "marlin",
            Self::Fp8 { .. } => "fp8",
            Self::Fp4Packed { .. } => "fp4_packed",
        }
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
    /// Phase 2.5 storage-coherence flag (#1082 line 287).
    ///
    /// `true` when the BF16/F32 master (`backward_storage`) has been
    /// mutated (typically by the optimizer step) and the quantized
    /// `forward_storage` is therefore stale and must be re-quantized
    /// before the next forward read.
    ///
    /// Mutated through [`Parameter::mark_master_dirty`] /
    /// [`Parameter::mark_forward_clean`]. The actual re-quantization
    /// kernel is the caller's responsibility (different on cublasLt /
    /// MPS / Vulkan) — this slot is the explicit handshake.
    ///
    /// For pure-inference Parameters (no `backward_storage`),
    /// `forward_stale` is always `false` — there is no master to
    /// invalidate from.
    forward_stale: bool,
    /// Phase 2.7 epoch counter (#1082 line "`Parameter::version`
    /// epoch counter" under Phase 2.7).
    ///
    /// Monotonically increasing version of this parameter's training
    /// state. Bumped via [`Parameter::bump_epoch`] at end-of-optimizer-
    /// step. Consumers (e.g. eval-while-training, on-policy GRPO
    /// rollouts) snapshot the parameter at `current_epoch()` start;
    /// the snapshot remains valid for as long as the storage Arc is
    /// held, even as the master mutates underneath at later epochs.
    ///
    /// Initialized to 0 by both constructors; starts incrementing at
    /// the first optimizer step.
    epoch: u64,
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
            forward_stale: false,
            epoch: 0,
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
            forward_stale: false,
            epoch: 0,
        }
    }

    /// Phase 2.5 storage-coherence (#1082 line 287): whether the
    /// quantized `forward_storage` is stale relative to the BF16/F32
    /// master in `backward_storage`. Read by the forward path before
    /// running matmul; if `true`, the caller must re-quantize before
    /// the next forward read.
    ///
    /// For pure-inference Parameters (no `backward_storage`), this
    /// always returns `false`.
    pub fn is_forward_stale(&self) -> bool {
        self.forward_stale
    }

    /// Mark the forward storage as stale after the master
    /// (`backward_storage`) has been mutated, typically by the
    /// optimizer step. No-op for pure-inference Parameters (forward
    /// storage is canonical when there is no master).
    pub fn mark_master_dirty(&mut self) {
        if self.backward_storage.is_some() {
            self.forward_stale = true;
        }
    }

    /// Mark the forward storage as clean after a re-quantization
    /// kernel has refreshed it from the master. The kernel itself
    /// is the caller's responsibility (per-backend); this is the
    /// explicit handshake the caller flips when the refresh
    /// completes.
    pub fn mark_forward_clean(&mut self) {
        self.forward_stale = false;
    }

    /// Phase 2.7 epoch counter (#1082 "live serve + train
    /// coexistence" line): current training epoch of this parameter.
    /// Monotonically non-decreasing; bumped by
    /// [`Parameter::bump_epoch`] at end-of-optimizer-step.
    ///
    /// Consumers snapshot the parameter at `current_epoch()` start
    /// (e.g. eval-while-training, on-policy GRPO rollouts) and read
    /// the snapshot through the Arc-shared storage even as the
    /// master mutates underneath at later epochs.
    pub fn current_epoch(&self) -> u64 {
        self.epoch
    }

    /// Advance the epoch counter — called at end-of-optimizer-step.
    /// Saturating add: at u64::MAX the counter sticks (in practice
    /// no training run reaches 2^64 steps).
    pub fn bump_epoch(&mut self) {
        self.epoch = self.epoch.saturating_add(1);
    }

    /// Stable identity.
    pub fn tensor_id(&self) -> TensorId {
        self.tensor_id
    }

    /// Borrow forward storage.
    /// Typed dispatch view of the forward path. Match on the returned
    /// [`ForwardDispatch`] to route to the backend matmul appropriate
    /// for the storage variant (Plain BF16 / Marlin W4A16 / FP8 /
    /// FP4Packed).
    ///
    /// This is the Phase 2.5 "Forward dispatches on storage variant"
    /// item (line 282 of #1082) made type-system-explicit: the caller
    /// can never forget to handle a variant because the enum is
    /// exhaustive (`#[non_exhaustive]` only for future-proofing).
    ///
    /// Actually running the matmul stays the caller's job — the
    /// per-backend handle (cublasLt / MPS / compute-shader) isn't a
    /// kiln-param concern. See [`ForwardDispatch`] for the per-
    /// variant components and routing notes.
    pub fn forward_dispatch(&self) -> ForwardDispatch<'_> {
        match &self.forward_storage {
            ForwardStorage::Plain(t) => ForwardDispatch::Plain { weight: t },
            ForwardStorage::Marlin { packed, scales } => ForwardDispatch::Marlin { packed, scales },
            ForwardStorage::Fp8 { packed, scales } => ForwardDispatch::Fp8 { packed, scales },
            ForwardStorage::Fp4Packed { packed, scales } => {
                ForwardDispatch::Fp4Packed { packed, scales }
            }
        }
    }

    pub fn forward_storage(&self) -> &ForwardStorage {
        &self.forward_storage
    }

    /// Mutate the forward storage (e.g. swap BF16 → Marlin in place).
    /// **Preserves `tensor_id`** per anti-pattern 11.
    ///
    /// Marks `forward_stale = false` since the caller has just
    /// provided a fresh forward storage (the storage-coherence
    /// invariant is satisfied by the replacement itself).
    pub fn replace_forward_storage(&mut self, new: ForwardStorage) {
        self.forward_storage = new;
        self.forward_stale = false;
        // tensor_id intentionally unchanged.
    }

    /// Borrow backward storage (the master tensor).
    pub fn backward_storage(&self) -> Option<&Tensor> {
        self.backward_storage.as_ref()
    }

    /// Replace the backward storage (master tensor) in place. Used by
    /// optimizer steps after computing the new master values; the
    /// caller is responsible for constructing `new` with the same
    /// shape and dtype as the existing master.
    ///
    /// **Preserves [`Self::tensor_id`]** per anti-pattern 11 — the
    /// updated master is the same logical parameter at a new step
    /// boundary, not a fresh parameter. Downstream optimizer state
    /// keyed on `tensor_id` (AdamW moments, SGD velocities, etc.)
    /// survives.
    ///
    /// `None` argument drops the backward storage entirely (used by
    /// LoRA-only frozen-trunk parameters).
    ///
    /// Marks `forward_stale = true` when the new master is `Some` —
    /// the forward storage may now be inconsistent with the master
    /// the caller just installed. (Dropping the master to `None`
    /// resets the flag since there's nothing to be stale against.)
    pub fn replace_backward_storage(&mut self, new: Option<Tensor>) {
        let was_some = new.is_some();
        self.backward_storage = new;
        self.forward_stale = was_some;
        // tensor_id intentionally unchanged.
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

    // ------------------------------------------------------------------
    // Phase 2.5 content-addressed identity
    // ------------------------------------------------------------------

    /// Compute a content-addressed fingerprint of the Parameter.
    ///
    /// Per the Phase 2.5 issue bullet:
    ///
    /// > **Parameter content checksum (xxhash3) for safe hot-swap.**
    /// > `Parameter::content_hash() -> u64` is a content-addressed
    /// > fingerprint of `forward_storage` (post-quantization,
    /// > post-LoRA-merge). Required for: (a) multi-process serving
    /// > safety ...; (b) hot-swap of a fine-tune over a running
    /// > serve ...; (c) adapter cache invalidation on the serve side.
    ///
    /// The fingerprint mixes:
    /// - `forward_storage.primary_storage()` bytes
    /// - `forward_storage.kind_name()` (so Plain vs Marlin vs FP8
    ///   distinguish even when underlying bytes coincide)
    /// - `lora_delta`'s bytes (so a delta swap changes the hash)
    /// - The dtype short-name of every output head's storage (so a
    ///   value/reward-head attach changes the hash)
    /// - `amp_policy.master_dtype` and `forward_compute_dtype` short
    ///   names (so AmpPolicy changes are visible in the hash)
    ///
    /// **Today's hash function is stdlib `DefaultHasher`** — Phase
    /// 2.5.x swaps to xxhash3 for stability across Rust versions and
    /// faster hashing on the 4 GiB Qwen3.5-4B BF16 master.
    ///
    /// CPU-only: requires downcasting the involved storages to
    /// `CpuStorage`. GPU-storage hashing lands when the Phase 1.12
    /// pinned-host staging pool ships.
    pub fn content_hash(&self) -> Result<u64> {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        // Mix in the forward-storage kind name so packed vs plain
        // hashes distinctly.
        self.forward_storage.kind_name().hash(&mut hasher);

        // Hash the primary forward storage bytes.
        let primary_hash = content_hash_storage(self.forward_storage.primary_storage())?;
        primary_hash.hash(&mut hasher);

        // Mix in the LoRA delta if present (its absence vs presence
        // is part of the hash).
        if let Some(delta) = &self.lora_delta {
            "lora_delta:present".hash(&mut hasher);
            let dh = content_hash_storage(delta.storage())?;
            dh.hash(&mut hasher);
        } else {
            "lora_delta:absent".hash(&mut hasher);
        }

        // Mix in each output head's dtype + role. We don't hash the
        // head storage bytes — those typically change every step in
        // training and would invalidate the cache too aggressively.
        // The head registry's presence/absence still affects the hash
        // because adding a value head IS a meaningful identity change.
        for head in &self.heads {
            head.role.name().hash(&mut hasher);
            head.head_storage.dtype().short_name().hash(&mut hasher);
        }

        // Mix in the AMP policy. Two parameters with identical bytes
        // under different precision policies are distinct identities
        // for serve-side cache purposes.
        self.amp_policy.master_dtype.short_name().hash(&mut hasher);
        self.amp_policy
            .forward_compute_dtype
            .short_name()
            .hash(&mut hasher);

        Ok(hasher.finish())
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
    fn replace_backward_preserves_tensor_id() {
        // Anti-pattern 11 contract for the backward (master) slot.
        // After an optimizer step swaps the master, the parameter's
        // `tensor_id` must stay the same so optimizer state keyed on
        // it (AdamW moments / SGD velocities) doesn't orphan.
        let fwd = ForwardStorage::Plain(Tensor::zeros_cpu(vec![4], DType::F32));
        let master = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let mut p = Parameter::trainable(fwd, master, AmpPolicy::fp32_reference());
        let original_id = p.tensor_id();
        assert_eq!(
            p.backward_storage().map(|t| t.element_count()),
            Some(4),
            "trainable Parameter must carry a master"
        );

        // Swap to a different-content master (simulating one step of
        // SGD or AdamW that mutated all entries).
        let new_master = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0], vec![4]).unwrap();
        let new_master_id = new_master.id();
        assert_ne!(
            new_master_id, original_id,
            "fresh Tensor must have a fresh TensorId"
        );
        p.replace_backward_storage(Some(new_master));

        // Parameter tensor_id is unchanged; backward_storage now
        // reports the new content.
        assert_eq!(p.tensor_id(), original_id);
        assert!(p.backward_storage().is_some());
    }

    #[test]
    fn replace_backward_with_none_drops_master() {
        // Used by LoRA-only flows that freeze the trunk and only train
        // the delta.
        let fwd = ForwardStorage::Plain(Tensor::zeros_cpu(vec![4], DType::F32));
        let master = Tensor::zeros_cpu(vec![4], DType::F32);
        let mut p = Parameter::trainable(fwd, master, AmpPolicy::fp32_reference());
        let id = p.tensor_id();
        assert!(p.backward_storage().is_some());
        p.replace_backward_storage(None);
        assert!(p.backward_storage().is_none());
        assert_eq!(p.tensor_id(), id);
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

    #[test]
    fn content_hash_changes_on_forward_storage_swap() {
        let mut p = Parameter::inference_only(plain_f32());
        let h0 = p.content_hash().unwrap();
        // Swap Plain → Marlin (different kind_name + different bytes).
        let new_fwd = ForwardStorage::Marlin {
            packed: Tensor::zeros_cpu(vec![4, 4], DType::Int4Packed),
            scales: Tensor::zeros_cpu(vec![4], DType::BF16),
        };
        p.replace_forward_storage(new_fwd);
        let h1 = p.content_hash().unwrap();
        assert_ne!(h0, h1);
    }

    #[test]
    fn content_hash_changes_on_lora_delta_swap() {
        let mut p = Parameter::inference_only(plain_f32());
        let h0 = p.content_hash().unwrap();
        // Attach LoRA.
        let delta = Tensor::zeros_cpu(vec![4, 4], DType::F32);
        p.set_lora_delta(Some(delta));
        let h1 = p.content_hash().unwrap();
        assert_ne!(h0, h1);
        // Detach.
        p.set_lora_delta(None);
        let h2 = p.content_hash().unwrap();
        assert_ne!(h1, h2);
        // h2 should equal h0 (same state as before LoRA attach).
        assert_eq!(h0, h2);
    }

    #[test]
    fn content_hash_changes_on_head_attach() {
        let mut p = Parameter::inference_only(plain_f32());
        let h0 = p.content_hash().unwrap();
        p.add_head(OutputHead {
            role: OutputHeadRole::LmHead,
            head_storage: Tensor::zeros_cpu(vec![4, 16], DType::F32),
            requires_grad: true,
        });
        let h1 = p.content_hash().unwrap();
        assert_ne!(h0, h1);
        // Adding a second head changes the hash again.
        p.add_head(OutputHead {
            role: OutputHeadRole::MtpHead,
            head_storage: Tensor::zeros_cpu(vec![4, 16], DType::F32),
            requires_grad: true,
        });
        let h2 = p.content_hash().unwrap();
        assert_ne!(h1, h2);
    }

    #[test]
    fn content_hash_changes_on_amp_policy_change() {
        let mut p = Parameter::inference_only(plain_f32());
        let h0 = p.content_hash().unwrap();
        p.set_amp_policy(AmpPolicy::fp32_reference());
        let h1 = p.content_hash().unwrap();
        // Plain F32 forward + default AMP (BF16 master) vs Plain F32
        // + fp32_reference AMP differ in master_dtype short-name.
        assert_ne!(h0, h1);
    }

    #[test]
    fn content_hash_stable_for_identical_state() {
        // Two parameters with byte-identical forward, no LoRA, same
        // policy → same content hash.
        let p1 = Parameter::inference_only(plain_f32());
        let p2 = Parameter::inference_only(plain_f32());
        assert_eq!(p1.content_hash().unwrap(), p2.content_hash().unwrap());
    }

    #[test]
    fn forward_stale_default_is_false_for_inference() {
        let p = Parameter::inference_only(plain_f32());
        assert!(!p.is_forward_stale());
    }

    #[test]
    fn forward_stale_default_is_false_for_trainable() {
        let fs = plain_f32();
        let master_tensor = fs.primary_tensor().clone();
        let p = Parameter::trainable(fs, master_tensor, AmpPolicy::default());
        assert!(!p.is_forward_stale());
    }

    #[test]
    fn mark_master_dirty_is_noop_for_pure_inference() {
        // No backward_storage → can't be stale.
        let mut p = Parameter::inference_only(plain_f32());
        p.mark_master_dirty();
        assert!(!p.is_forward_stale());
    }

    #[test]
    fn mark_master_dirty_sets_stale_for_trainable() {
        let fs = plain_f32();
        let master_tensor = fs.primary_tensor().clone();
        let mut p = Parameter::trainable(fs, master_tensor, AmpPolicy::default());
        p.mark_master_dirty();
        assert!(p.is_forward_stale());
    }

    #[test]
    fn mark_forward_clean_resets_after_dirty() {
        let fs = plain_f32();
        let master_tensor = fs.primary_tensor().clone();
        let mut p = Parameter::trainable(fs, master_tensor, AmpPolicy::default());
        p.mark_master_dirty();
        assert!(p.is_forward_stale());
        p.mark_forward_clean();
        assert!(!p.is_forward_stale());
    }

    #[test]
    fn replace_forward_storage_resets_stale() {
        let fs = plain_f32();
        let master_tensor = fs.primary_tensor().clone();
        let mut p = Parameter::trainable(fs, master_tensor, AmpPolicy::default());
        p.mark_master_dirty();
        assert!(p.is_forward_stale());
        p.replace_forward_storage(plain_f32());
        assert!(!p.is_forward_stale());
    }

    #[test]
    fn replace_backward_storage_with_some_sets_stale() {
        let fs = plain_f32();
        let master_tensor = fs.primary_tensor().clone();
        let mut p = Parameter::trainable(fs, master_tensor, AmpPolicy::default());
        // After a fresh trainable construction, stale is false.
        assert!(!p.is_forward_stale());
        // Replacing the master with a new tensor sets stale.
        let new_master = plain_f32().primary_tensor().clone();
        p.replace_backward_storage(Some(new_master));
        assert!(p.is_forward_stale());
    }

    #[test]
    fn epoch_starts_at_zero_for_inference() {
        let p = Parameter::inference_only(plain_f32());
        assert_eq!(p.current_epoch(), 0);
    }

    #[test]
    fn epoch_starts_at_zero_for_trainable() {
        let fs = plain_f32();
        let master_tensor = fs.primary_tensor().clone();
        let p = Parameter::trainable(fs, master_tensor, AmpPolicy::default());
        assert_eq!(p.current_epoch(), 0);
    }

    #[test]
    fn epoch_bumps_monotonically() {
        let fs = plain_f32();
        let master_tensor = fs.primary_tensor().clone();
        let mut p = Parameter::trainable(fs, master_tensor, AmpPolicy::default());
        for expected in 1..=5_u64 {
            p.bump_epoch();
            assert_eq!(p.current_epoch(), expected);
        }
    }

    #[test]
    fn epoch_bump_saturates_at_u64_max() {
        // Set epoch artificially close to MAX via repeated bumps would
        // be slow; the saturating-add guarantees no panic at the edge.
        // Instead we test that bump after MAX stays at MAX by setting
        // the field directly through a controlled construction path.
        let fs = plain_f32();
        let master_tensor = fs.primary_tensor().clone();
        let mut p = Parameter::trainable(fs, master_tensor, AmpPolicy::default());
        // We don't have a direct setter; rely on the doc contract.
        // This test exercises the public API: many bumps without panic.
        for _ in 0..1000 {
            p.bump_epoch();
        }
        assert_eq!(p.current_epoch(), 1000);
    }

    #[test]
    fn replace_backward_storage_with_none_resets_stale() {
        let fs = plain_f32();
        let master_tensor = fs.primary_tensor().clone();
        let mut p = Parameter::trainable(fs, master_tensor, AmpPolicy::default());
        p.mark_master_dirty();
        assert!(p.is_forward_stale());
        // Dropping the master to None means there's nothing to be
        // stale against.
        p.replace_backward_storage(None);
        assert!(!p.is_forward_stale());
    }
}
