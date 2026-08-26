//! Authoritative replay-plan contract shared by backend graph runners.
//!
//! `CapturedGraph` is the low-level replayable object. `ReplayPlan` is the
//! engine-facing contract the model/training layer should target: stable keys,
//! stable resident inputs, invalidation reasons, and typed replay outputs.

use kiln_tensor::{Backend, DType, Tensor, TensorId};

use crate::{CaptureError, CapturedGraph};

/// Stable replay operation key used for bucketing and cache lookup.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ReplayKey {
    pub backend: Backend,
    pub operation: String,
    pub shape_key: Vec<usize>,
    pub dtype: Option<DType>,
    pub max_batch: usize,
    pub replay_safe: bool,
}

impl ReplayKey {
    pub fn new(
        backend: Backend,
        operation: impl Into<String>,
        shape_key: Vec<usize>,
        dtype: Option<DType>,
        max_batch: usize,
        replay_safe: bool,
    ) -> Self {
        Self {
            backend,
            operation: operation.into(),
            shape_key,
            dtype,
            max_batch,
            replay_safe,
        }
    }
}

/// Replay stability attached to one resident resource reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReplayResourceStability {
    NotReplayStable,
    StableWithinStep,
    StableAcrossReplay,
}

impl ReplayResourceStability {
    pub const fn is_replay_stable(self) -> bool {
        !matches!(self, Self::NotReplayStable)
    }
}

/// Graph-layer reference to a backend-resident resource.
///
/// This is intentionally pure metadata. Backend-specific handles remain inside
/// CUDA graph execs, HIP graph execs, Metal ICBs, or Vulkan command batches.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResidentResourceRef {
    pub tensor_id: Option<TensorId>,
    pub backend: Backend,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub start_offset: usize,
    pub contiguous: bool,
    pub byte_len: usize,
    pub replay_stability: ReplayResourceStability,
}

impl ResidentResourceRef {
    pub fn from_tensor(
        tensor: &Tensor,
        backend: Backend,
        replay_stability: ReplayResourceStability,
    ) -> Self {
        let element_count = tensor.element_count();
        Self {
            tensor_id: Some(tensor.id()),
            backend,
            dtype: tensor.dtype(),
            shape: tensor.shape().to_vec(),
            strides: tensor.strides().to_vec(),
            start_offset: tensor.layout().start_offset(),
            contiguous: tensor.layout().is_contiguous(),
            byte_len: tensor.dtype().packed_buffer_bytes(element_count),
            replay_stability,
        }
    }

    pub const fn is_replay_stable(&self) -> bool {
        self.replay_stability.is_replay_stable()
    }
}

/// Inputs supplied to one replay invocation.
#[derive(Debug, Clone, Copy)]
pub struct ReplayInputs<'a> {
    pub key: &'a ReplayKey,
    pub resources: &'a [ResidentResourceRef],
}

impl<'a> ReplayInputs<'a> {
    pub const fn new(key: &'a ReplayKey, resources: &'a [ResidentResourceRef]) -> Self {
        Self { key, resources }
    }
}

/// Outputs produced or updated by one replay invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayOutputs {
    pub resources: Vec<ResidentResourceRef>,
    pub replay_count: u64,
}

impl ReplayOutputs {
    pub fn new(resources: Vec<ResidentResourceRef>, replay_count: u64) -> Self {
        Self {
            resources,
            replay_count,
        }
    }
}

/// Stable reason a cached replay plan cannot be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidateReason {
    KeyChanged {
        expected: ReplayKey,
        actual: ReplayKey,
    },
    InputCountChanged {
        expected: usize,
        actual: usize,
    },
    InputResourceChanged {
        index: usize,
        expected: ResidentResourceRef,
        actual: ResidentResourceRef,
    },
    UnstableInput {
        index: usize,
        tensor_id: Option<TensorId>,
    },
    Backend(String),
}

impl InvalidateReason {
    pub fn message(&self) -> String {
        match self {
            Self::KeyChanged { expected, actual } => {
                format!("replay key changed: expected {expected:?}, got {actual:?}")
            }
            Self::InputCountChanged { expected, actual } => {
                format!("replay input count changed: expected {expected}, got {actual}")
            }
            Self::InputResourceChanged { index, .. } => {
                format!("replay input resource changed at index {index}")
            }
            Self::UnstableInput { index, tensor_id } => {
                format!("replay input at index {index} is not stable: {tensor_id:?}")
            }
            Self::Backend(reason) => reason.clone(),
        }
    }
}

/// Captured state used to validate a replay plan before execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayState {
    pub key: ReplayKey,
    pub inputs: Vec<ResidentResourceRef>,
}

impl ReplayState {
    pub fn new(key: ReplayKey, inputs: Vec<ResidentResourceRef>) -> Self {
        Self { key, inputs }
    }

    pub fn invalidate_reason(
        &self,
        key: &ReplayKey,
        inputs: &[ResidentResourceRef],
    ) -> Option<InvalidateReason> {
        if self.key != *key {
            return Some(InvalidateReason::KeyChanged {
                expected: self.key.clone(),
                actual: key.clone(),
            });
        }
        if self.inputs.len() != inputs.len() {
            return Some(InvalidateReason::InputCountChanged {
                expected: self.inputs.len(),
                actual: inputs.len(),
            });
        }
        for (index, input) in inputs.iter().enumerate() {
            if !input.is_replay_stable() {
                return Some(InvalidateReason::UnstableInput {
                    index,
                    tensor_id: input.tensor_id,
                });
            }
            if self.inputs[index] != *input {
                return Some(InvalidateReason::InputResourceChanged {
                    index,
                    expected: self.inputs[index].clone(),
                    actual: input.clone(),
                });
            }
        }
        None
    }

    pub fn validate(
        &self,
        key: &ReplayKey,
        inputs: &[ResidentResourceRef],
    ) -> Result<(), CaptureError> {
        match self.invalidate_reason(key, inputs) {
            Some(reason) => Err(CaptureError::ReplayInvalidated {
                reason: reason.message(),
            }),
            None => Ok(()),
        }
    }
}

/// Engine-facing replay contract.
///
/// Backends keep their native primitive (CUDA graph, HIP graph, Metal ICB,
/// Vulkan command batch), while the common layer owns keys, input stability,
/// invalidation, fallback policy, and diagnostics.
pub trait ReplayPlan: Send + Sync + std::fmt::Debug {
    fn backend(&self) -> Backend;

    fn key(&self) -> ReplayKey;

    fn validate_inputs(&self, inputs: ReplayInputs<'_>) -> Result<(), CaptureError>;

    fn replay(&mut self, inputs: ReplayInputs<'_>) -> Result<ReplayOutputs, CaptureError>;

    fn invalidate_reason(&self, state: &ReplayState) -> Option<InvalidateReason>;
}

/// Adapter that exposes an existing low-level [`CapturedGraph`] as a
/// [`ReplayPlan`].
#[derive(Debug)]
pub struct CapturedGraphReplayPlan<G: CapturedGraph> {
    graph: G,
    key: ReplayKey,
    state: ReplayState,
}

impl<G: CapturedGraph> CapturedGraphReplayPlan<G> {
    pub fn new(
        graph: G,
        key: ReplayKey,
        inputs: Vec<ResidentResourceRef>,
    ) -> Result<Self, CaptureError> {
        if graph.backend() != key.backend {
            return Err(CaptureError::InvalidReplayInput {
                reason: format!(
                    "graph backend {} does not match replay key backend {}",
                    graph.backend(),
                    key.backend
                ),
            });
        }
        Ok(Self {
            graph,
            state: ReplayState::new(key.clone(), inputs),
            key,
        })
    }

    pub const fn graph(&self) -> &G {
        &self.graph
    }
}

impl<G: CapturedGraph> ReplayPlan for CapturedGraphReplayPlan<G> {
    fn backend(&self) -> Backend {
        self.graph.backend()
    }

    fn key(&self) -> ReplayKey {
        self.key.clone()
    }

    fn validate_inputs(&self, inputs: ReplayInputs<'_>) -> Result<(), CaptureError> {
        self.state.validate(inputs.key, inputs.resources)
    }

    fn replay(&mut self, inputs: ReplayInputs<'_>) -> Result<ReplayOutputs, CaptureError> {
        self.state.validate(inputs.key, inputs.resources)?;
        self.graph.replay()?;
        Ok(ReplayOutputs::new(
            inputs.resources.to_vec(),
            self.graph.replay_count(),
        ))
    }

    fn invalidate_reason(&self, state: &ReplayState) -> Option<InvalidateReason> {
        self.state.invalidate_reason(&state.key, &state.inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CapturedGraph;
    use kiln_tensor::{Backend, DType, Tensor};
    use std::sync::atomic::{AtomicU64, Ordering};

    #[derive(Debug)]
    struct DummyReplayPlan {
        key: ReplayKey,
        state: ReplayState,
        replay_count: u64,
    }

    impl DummyReplayPlan {
        fn new(key: ReplayKey, inputs: Vec<ResidentResourceRef>) -> Self {
            Self {
                state: ReplayState::new(key.clone(), inputs),
                key,
                replay_count: 0,
            }
        }
    }

    impl ReplayPlan for DummyReplayPlan {
        fn backend(&self) -> Backend {
            self.key.backend
        }

        fn key(&self) -> ReplayKey {
            self.key.clone()
        }

        fn validate_inputs(&self, inputs: ReplayInputs<'_>) -> Result<(), CaptureError> {
            self.state.validate(inputs.key, inputs.resources)
        }

        fn replay(&mut self, inputs: ReplayInputs<'_>) -> Result<ReplayOutputs, CaptureError> {
            self.state.validate(inputs.key, inputs.resources)?;
            self.replay_count += 1;
            Ok(ReplayOutputs::new(
                inputs.resources.to_vec(),
                self.replay_count,
            ))
        }

        fn invalidate_reason(&self, state: &ReplayState) -> Option<InvalidateReason> {
            self.state.invalidate_reason(&state.key, &state.inputs)
        }
    }

    fn stable_resource() -> ResidentResourceRef {
        let tensor = Tensor::zeros_cpu(vec![2, 3], DType::F32);
        ResidentResourceRef::from_tensor(
            &tensor,
            Backend::Cuda,
            ReplayResourceStability::StableAcrossReplay,
        )
    }

    #[test]
    fn replay_state_accepts_matching_stable_inputs() {
        let key = ReplayKey::new(
            Backend::Cuda,
            "decode",
            vec![1, 128],
            Some(DType::F32),
            1,
            true,
        );
        let input = stable_resource();
        let state = ReplayState::new(key.clone(), vec![input.clone()]);
        assert_eq!(state.invalidate_reason(&key, &[input]), None);
    }

    #[test]
    fn replay_state_rejects_key_changes() {
        let key = ReplayKey::new(
            Backend::Cuda,
            "decode",
            vec![1, 128],
            Some(DType::F32),
            1,
            true,
        );
        let changed = ReplayKey::new(
            Backend::Cuda,
            "decode",
            vec![2, 128],
            Some(DType::F32),
            2,
            true,
        );
        let input = stable_resource();
        let state = ReplayState::new(key, vec![input.clone()]);
        assert!(matches!(
            state.invalidate_reason(&changed, &[input]),
            Some(InvalidateReason::KeyChanged { .. })
        ));
    }

    #[test]
    fn replay_state_rejects_unstable_inputs() {
        let key = ReplayKey::new(
            Backend::Cuda,
            "decode",
            vec![1, 128],
            Some(DType::F32),
            1,
            true,
        );
        let mut input = stable_resource();
        let state = ReplayState::new(key.clone(), vec![input.clone()]);
        input.replay_stability = ReplayResourceStability::NotReplayStable;
        assert!(matches!(
            state.invalidate_reason(&key, &[input]),
            Some(InvalidateReason::UnstableInput { .. })
        ));
    }

    #[test]
    fn replay_state_accepts_stable_within_step_inputs() {
        let key = ReplayKey::new(
            Backend::Cuda,
            "decode",
            vec![1, 128],
            Some(DType::F32),
            1,
            true,
        );
        let mut input = stable_resource();
        input.replay_stability = ReplayResourceStability::StableWithinStep;
        let state = ReplayState::new(key.clone(), vec![input.clone()]);
        assert_eq!(state.invalidate_reason(&key, &[input]), None);
    }

    #[test]
    fn replay_resource_ref_tracks_packed_byte_len_and_layout() {
        let packed = Tensor::zeros_cpu(vec![9], DType::Int4Packed);
        let packed_ref = ResidentResourceRef::from_tensor(
            &packed,
            Backend::Cuda,
            ReplayResourceStability::StableAcrossReplay,
        );
        assert_eq!(packed_ref.byte_len, 5);
        assert_eq!(packed_ref.strides, vec![1]);
        assert_eq!(packed_ref.start_offset, 0);
        assert!(packed_ref.contiguous);

        let matrix = Tensor::zeros_cpu(vec![2, 3], DType::F32);
        let transposed = matrix.t().expect("transpose should succeed");
        let transposed_ref = ResidentResourceRef::from_tensor(
            &transposed,
            Backend::Cuda,
            ReplayResourceStability::StableAcrossReplay,
        );
        assert_eq!(transposed_ref.shape, vec![3, 2]);
        assert_eq!(transposed_ref.start_offset, 0);
        assert!(!transposed_ref.contiguous);
    }

    #[test]
    fn replay_state_rejects_layout_changes() {
        let key = ReplayKey::new(
            Backend::Cuda,
            "decode",
            vec![1, 128],
            Some(DType::F32),
            1,
            true,
        );
        let input = stable_resource();
        let mut changed = input.clone();
        changed.contiguous = false;
        let state = ReplayState::new(key.clone(), vec![input]);
        assert!(matches!(
            state.invalidate_reason(&key, &[changed]),
            Some(InvalidateReason::InputResourceChanged { .. })
        ));
    }

    #[test]
    fn replay_plan_validates_inputs_and_counts_replays() {
        let key = ReplayKey::new(
            Backend::Cuda,
            "decode",
            vec![1, 128],
            Some(DType::F32),
            1,
            true,
        );
        let input = stable_resource();
        let mut plan = DummyReplayPlan::new(key.clone(), vec![input.clone()]);

        plan.validate_inputs(ReplayInputs::new(&key, std::slice::from_ref(&input)))
            .unwrap();
        let outputs = plan
            .replay(ReplayInputs::new(&key, std::slice::from_ref(&input)))
            .unwrap();
        assert_eq!(outputs.replay_count, 1);
        assert_eq!(outputs.resources, vec![input]);
    }

    #[test]
    fn replay_plan_validate_inputs_rejects_key_changes() {
        let key = ReplayKey::new(
            Backend::Cuda,
            "decode",
            vec![1, 128],
            Some(DType::F32),
            1,
            true,
        );
        let changed = ReplayKey::new(
            Backend::Cuda,
            "decode",
            vec![2, 128],
            Some(DType::F32),
            2,
            true,
        );
        let input = stable_resource();
        let plan = DummyReplayPlan::new(key, vec![input.clone()]);

        assert!(matches!(
            plan.validate_inputs(ReplayInputs::new(&changed, &[input])),
            Err(CaptureError::ReplayInvalidated { reason }) if reason.contains("replay key changed")
        ));
    }

    #[test]
    fn backend_invalidation_reason_surfaces_backend_message() {
        let reason = InvalidateReason::Backend("driver reset".to_string());
        assert_eq!(reason.message(), "driver reset");
    }

    #[derive(Debug)]
    struct DummyCapturedGraph {
        backend: Backend,
        replay_count: AtomicU64,
    }

    impl DummyCapturedGraph {
        fn new(backend: Backend) -> Self {
            Self {
                backend,
                replay_count: AtomicU64::new(0),
            }
        }
    }

    impl CapturedGraph for DummyCapturedGraph {
        fn backend(&self) -> Backend {
            self.backend
        }

        fn replay(&self) -> Result<(), CaptureError> {
            self.replay_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn replay_count(&self) -> u64 {
            self.replay_count.load(Ordering::SeqCst)
        }

        fn scratch_bytes(&self) -> usize {
            0
        }
    }

    #[test]
    fn captured_graph_adapter_replays_with_shared_validation() {
        let key = ReplayKey::new(
            Backend::Cuda,
            "decode",
            vec![1, 128],
            Some(DType::F32),
            1,
            true,
        );
        let input = stable_resource();
        let graph = DummyCapturedGraph::new(Backend::Cuda);
        let mut plan = CapturedGraphReplayPlan::new(graph, key.clone(), vec![input.clone()])
            .expect("backend should match key");

        let outputs = plan
            .replay(ReplayInputs::new(&key, std::slice::from_ref(&input)))
            .unwrap();
        assert_eq!(outputs.replay_count, 1);
        assert_eq!(plan.graph().replay_count(), 1);

        let changed = ReplayKey::new(
            Backend::Cuda,
            "decode",
            vec![2, 128],
            Some(DType::F32),
            2,
            true,
        );
        assert!(matches!(
            plan.replay(ReplayInputs::new(&changed, &[input])),
            Err(CaptureError::ReplayInvalidated { .. })
        ));
    }

    #[test]
    fn captured_graph_adapter_rejects_backend_mismatch() {
        let key = ReplayKey::new(
            Backend::Cuda,
            "decode",
            vec![1, 128],
            Some(DType::F32),
            1,
            true,
        );
        let input = stable_resource();
        let graph = DummyCapturedGraph::new(Backend::Metal);
        assert!(matches!(
            CapturedGraphReplayPlan::new(graph, key, vec![input]),
            Err(CaptureError::InvalidReplayInput { .. })
        ));
    }
}
