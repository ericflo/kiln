//! `Activation` — first-class type for tape-aware tensors that are
//! NOT weights.
//!
//! Per the Phase 1 bullet in #1082:
//!
//! > **`Activation` registry (sibling to `Parameter`).** `Activation` is
//! > the first-class type for tape-aware tensors that are *not*
//! > weights: forward activations preserved across the backward,
//! > checkpoint-segment boundaries, KV cache slots, sampler scratch,
//! > intermediate matmul outputs that the optimizer never sees.
//! > `Activation` carries `OffloadPolicy { Device, PinnedHost, NvmeCache }`
//! > independently of `Parameter`'s policy — because the 16 GiB SFT-at-
//! > 8K-context OOM driver is *activations* ..., not weights.
//!
//! # Phase 1 scope
//!
//! This PR ships:
//!
//! - `Activation` struct — a thin wrapper around `Tensor` plus an
//!   [`ActivationKind`] tag, an [`OffloadPolicy`], and a stable
//!   [`ActivationId`] keyed off the underlying [`TensorId`].
//! - `OffloadPolicy` — three modes: `Device`, `PinnedHost`, `NvmeCache`.
//!   Today only `Device` actually offloads (the others scaffold the
//!   API for Phase 6.5's selective-recompute).
//! - `ActivationKind` — five variants: `ForwardActivation`, `Checkpoint`,
//!   `KvCacheSlot`, `SamplerScratch`, `Intermediate`. Each carries
//!   default policy + recompute-cost classification (`is_cheap`).
//! - `selective_recompute_recommendation` — given an
//!   `ActivationKind`, returns whether the activation should be
//!   saved or recomputed during backward. The default policy follows
//!   Phase 6.5's "cheap activations save, expensive ones recompute"
//!   contract.
//!
//! # What this PR does NOT do
//!
//! - Actual host/NVMe offload (Phase 1.12 pinned-host staging pool).
//! - Wiring into the autograd tape (Phase 6a).
//! - Auto-sizing of the offload boundary (Phase 6.5).
//!
//! The type lives here so other Phase 1.x modules can refer to it
//! without circular dependency.

use std::sync::Arc;

use crate::{Tensor, TensorId};

/// Activation kind. Drives the default offload + recompute policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ActivationKind {
    /// A forward-pass output that the backward pass will need
    /// (matmul outputs, attention scores). Expensive to recompute on
    /// backward; preferred policy is "save".
    ForwardActivation,
    /// The boundary between two checkpoint segments. Always saved;
    /// the segment's interior activations are recomputed from this.
    Checkpoint,
    /// A row in the paged KV cache. Long-lived; never offloaded
    /// (latency-critical).
    KvCacheSlot,
    /// Per-iteration sampler scratch (top-k buffers, logit masks).
    /// Short-lived; never offloaded.
    SamplerScratch,
    /// Cheap intermediate the backward pass can recompute (norm
    /// outputs, residual results, dropout masks, attention masks).
    /// Default policy: recompute.
    Intermediate,
}

impl ActivationKind {
    /// Stable short name (matches Phase 0.4 / Phase 9 audit keys).
    pub const fn name(self) -> &'static str {
        match self {
            ActivationKind::ForwardActivation => "forward_activation",
            ActivationKind::Checkpoint => "checkpoint",
            ActivationKind::KvCacheSlot => "kv_cache_slot",
            ActivationKind::SamplerScratch => "sampler_scratch",
            ActivationKind::Intermediate => "intermediate",
        }
    }

    /// Is recomputing this activation on backward cheap (≤ a few µs)?
    ///
    /// `true` for `Intermediate`, `Checkpoint` (already saved), and
    /// `SamplerScratch` (not reused).
    pub const fn is_cheap_to_recompute(self) -> bool {
        matches!(
            self,
            ActivationKind::Intermediate
                | ActivationKind::Checkpoint
                | ActivationKind::SamplerScratch
        )
    }

    /// Default offload policy for this kind.
    pub const fn default_offload_policy(self) -> OffloadPolicy {
        match self {
            // Forward activations + checkpoints stay on device by
            // default; Phase 6.5's auto-sizer may demote to PinnedHost
            // on 16 GiB tier.
            ActivationKind::ForwardActivation | ActivationKind::Checkpoint => OffloadPolicy::Device,
            // KV cache + sampler scratch are latency-critical and
            // small; never offload.
            ActivationKind::KvCacheSlot | ActivationKind::SamplerScratch => OffloadPolicy::Device,
            // Cheap intermediates: device is fine (they're not big);
            // recomputation handles memory pressure.
            ActivationKind::Intermediate => OffloadPolicy::Device,
        }
    }
}

/// Where the activation's storage lives.
///
/// Per the issue:
///
/// > `kiln-tensor::OffloadPolicy` per Parameter: `Always` (default,
/// > full VRAM resident), `OnDemand` (page in for backward, page out
/// > after step), `Host` (mirror in pinned host RAM).
///
/// This mirrors that policy enum but for `Activation` rather than
/// `Parameter`. The Phase 1.x pinned-host pool + Phase 6.5
/// auto-sizer drive the actual offload — today every variant just
/// records the intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OffloadPolicy {
    /// Resident on the original device throughout the forward + backward.
    Device,
    /// Mirrored to a pinned-host buffer; pageable to device on demand.
    /// Scaffold for Phase 6.5's activation offload.
    PinnedHost,
    /// Mmapped to NVMe scratch. For 16 GiB tier with very-long
    /// context. Scaffold; not implemented in Phase 1.
    NvmeCache,
}

impl OffloadPolicy {
    pub const fn name(self) -> &'static str {
        match self {
            OffloadPolicy::Device => "device",
            OffloadPolicy::PinnedHost => "pinned_host",
            OffloadPolicy::NvmeCache => "nvme_cache",
        }
    }

    /// Returns `true` iff this policy keeps the activation resident
    /// on the original device through forward + backward.
    pub const fn is_resident(self) -> bool {
        matches!(self, OffloadPolicy::Device)
    }
}

/// Stable identity for an [`Activation`]. Derived from the underlying
/// [`TensorId`] so two `Activation` views of the same Tensor share id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ActivationId(TensorId);

impl ActivationId {
    pub const fn from_tensor_id(id: TensorId) -> Self {
        ActivationId(id)
    }
    pub const fn tensor_id(self) -> TensorId {
        self.0
    }
}

/// First-class type for tape-aware tensors that are NOT weights.
///
/// Sibling to `Parameter` (Phase 2.5). Carries an
/// [`ActivationKind`] tag, an [`OffloadPolicy`], and a stable
/// [`ActivationId`] derived from the underlying Tensor's TensorId.
#[derive(Debug, Clone)]
pub struct Activation {
    tensor: Tensor,
    kind: ActivationKind,
    policy: OffloadPolicy,
}

impl Activation {
    /// Wrap a [`Tensor`] with default policy for the given kind.
    pub fn new(tensor: Tensor, kind: ActivationKind) -> Self {
        Activation {
            policy: kind.default_offload_policy(),
            tensor,
            kind,
        }
    }

    /// Wrap with an explicit policy override (e.g. the Phase 6.5
    /// auto-sizer setting `PinnedHost` on 16 GiB tier).
    pub fn with_policy(tensor: Tensor, kind: ActivationKind, policy: OffloadPolicy) -> Self {
        Activation {
            tensor,
            kind,
            policy,
        }
    }

    /// Borrow the wrapped Tensor.
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    /// Consume and return the wrapped Tensor.
    pub fn into_tensor(self) -> Tensor {
        self.tensor
    }

    /// Activation kind tag.
    pub fn kind(&self) -> ActivationKind {
        self.kind
    }

    /// Current offload policy.
    pub fn policy(&self) -> OffloadPolicy {
        self.policy
    }

    /// Stable identity. Derived from the wrapped Tensor's `TensorId`,
    /// so cloning an `Activation` preserves its id.
    pub fn id(&self) -> ActivationId {
        ActivationId::from_tensor_id(self.tensor.id())
    }

    /// Set a new offload policy (e.g. Phase 6.5 auto-sizer migrating
    /// later-layer activations to PinnedHost on 16 GiB tier).
    pub fn set_policy(&mut self, policy: OffloadPolicy) {
        self.policy = policy;
    }
}

/// Re-export to make `Arc<Activation>` ergonomic for the registry path
/// where multiple consumers refer to the same activation.
pub type ActivationRef = Arc<Activation>;

/// Recommendation for whether the backward pass should recompute or
/// reuse an activation, given its kind.
///
/// Phase 6.5's selective-recompute policy reads this. Today the policy
/// is "cheap → recompute, expensive → save"; Phase 6.5 may layer a
/// memory-pressure auto-tuner on top.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecomputeRecommendation {
    /// Save this activation; backward reads it directly.
    Save,
    /// Drop this activation after forward; backward recomputes it
    /// from upstream activations.
    Recompute,
}

/// Per-kind default recompute recommendation. Phase 6.5's
/// auto-tuner can override on a per-Activation basis if memory
/// pressure on a 16 GiB tier requires a more aggressive policy.
pub const fn selective_recompute_recommendation(
    kind: ActivationKind,
) -> RecomputeRecommendation {
    match kind {
        // KV cache + sampler scratch must persist; they're not on the
        // tape's recompute path.
        ActivationKind::KvCacheSlot | ActivationKind::SamplerScratch => {
            RecomputeRecommendation::Save
        }
        // Checkpoint boundaries: always saved (that's their purpose).
        ActivationKind::Checkpoint => RecomputeRecommendation::Save,
        // Expensive forward activations (matmul outputs, attention
        // scores): save by default.
        ActivationKind::ForwardActivation => RecomputeRecommendation::Save,
        // Cheap intermediates: recompute.
        ActivationKind::Intermediate => RecomputeRecommendation::Recompute,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    fn cpu_tensor() -> Tensor {
        Tensor::zeros_cpu(vec![2, 3], DType::F32)
    }

    #[test]
    fn kind_name_strings() {
        assert_eq!(ActivationKind::ForwardActivation.name(), "forward_activation");
        assert_eq!(ActivationKind::Checkpoint.name(), "checkpoint");
        assert_eq!(ActivationKind::KvCacheSlot.name(), "kv_cache_slot");
        assert_eq!(ActivationKind::SamplerScratch.name(), "sampler_scratch");
        assert_eq!(ActivationKind::Intermediate.name(), "intermediate");
    }

    #[test]
    fn cheapness_classification() {
        assert!(!ActivationKind::ForwardActivation.is_cheap_to_recompute());
        assert!(ActivationKind::Intermediate.is_cheap_to_recompute());
        assert!(ActivationKind::Checkpoint.is_cheap_to_recompute());
        assert!(ActivationKind::SamplerScratch.is_cheap_to_recompute());
        assert!(!ActivationKind::KvCacheSlot.is_cheap_to_recompute());
    }

    #[test]
    fn default_policies() {
        for k in [
            ActivationKind::ForwardActivation,
            ActivationKind::Checkpoint,
            ActivationKind::KvCacheSlot,
            ActivationKind::SamplerScratch,
            ActivationKind::Intermediate,
        ] {
            // Phase 1 default: every kind lives on device. Phase 6.5
            // overrides this for the 16 GiB tier.
            assert_eq!(k.default_offload_policy(), OffloadPolicy::Device);
            assert!(k.default_offload_policy().is_resident());
        }
    }

    #[test]
    fn policy_names() {
        assert_eq!(OffloadPolicy::Device.name(), "device");
        assert_eq!(OffloadPolicy::PinnedHost.name(), "pinned_host");
        assert_eq!(OffloadPolicy::NvmeCache.name(), "nvme_cache");
        assert!(OffloadPolicy::Device.is_resident());
        assert!(!OffloadPolicy::PinnedHost.is_resident());
        assert!(!OffloadPolicy::NvmeCache.is_resident());
    }

    #[test]
    fn activation_carries_id_kind_policy() {
        let t = cpu_tensor();
        let original_id = t.id();
        let a = Activation::new(t, ActivationKind::ForwardActivation);
        assert_eq!(a.kind(), ActivationKind::ForwardActivation);
        assert_eq!(a.policy(), OffloadPolicy::Device);
        assert_eq!(a.id().tensor_id(), original_id);
    }

    #[test]
    fn activation_clone_preserves_id() {
        let t = cpu_tensor();
        let a = Activation::new(t, ActivationKind::Checkpoint);
        let b = a.clone();
        assert_eq!(a.id(), b.id());
    }

    #[test]
    fn with_policy_overrides_default() {
        let t = cpu_tensor();
        let a = Activation::with_policy(
            t,
            ActivationKind::ForwardActivation,
            OffloadPolicy::PinnedHost,
        );
        assert_eq!(a.policy(), OffloadPolicy::PinnedHost);
    }

    #[test]
    fn set_policy_mutates_in_place() {
        let t = cpu_tensor();
        let mut a = Activation::new(t, ActivationKind::ForwardActivation);
        assert_eq!(a.policy(), OffloadPolicy::Device);
        a.set_policy(OffloadPolicy::PinnedHost);
        assert_eq!(a.policy(), OffloadPolicy::PinnedHost);
    }

    #[test]
    fn into_tensor_unwraps() {
        let t = cpu_tensor();
        let original_id = t.id();
        let a = Activation::new(t, ActivationKind::Intermediate);
        let back = a.into_tensor();
        assert_eq!(back.id(), original_id);
    }

    #[test]
    fn recompute_recommendations() {
        assert_eq!(
            selective_recompute_recommendation(ActivationKind::ForwardActivation),
            RecomputeRecommendation::Save
        );
        assert_eq!(
            selective_recompute_recommendation(ActivationKind::Intermediate),
            RecomputeRecommendation::Recompute
        );
        assert_eq!(
            selective_recompute_recommendation(ActivationKind::Checkpoint),
            RecomputeRecommendation::Save
        );
        assert_eq!(
            selective_recompute_recommendation(ActivationKind::KvCacheSlot),
            RecomputeRecommendation::Save
        );
        assert_eq!(
            selective_recompute_recommendation(ActivationKind::SamplerScratch),
            RecomputeRecommendation::Save
        );
    }
}
