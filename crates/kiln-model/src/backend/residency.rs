//! Shared resident-resource contracts for backend registries.
//!
//! Phase 3 uses these metadata types to describe the residency lifecycle that
//! CUDA/ROCm/Metal storage-owned registries and Vulkan upload-owned registries
//! already implement through `BackendRuntime` hooks. They are intentionally
//! backend-neutral and carry no allocation or synchronization behavior.

use kiln_tensor::{DType, Device, Tensor, TensorId};

/// High-level operation family that owns a resident resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidentResourceFamily {
    Activation,
    GdnRecurrentState,
    OptimizerParam,
    OptimizerMoment,
    PagedKv,
    ReplayInput,
    Other,
}

/// Where the resident bytes are owned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidentOwnership {
    /// The kt storage itself owns the resident device bytes.
    StorageOwned,
    /// A backend registry owns an uploaded/cached resident copy.
    RegistryOwned,
}

/// Default ownership model for activation resources reported by a backend.
pub fn resident_ownership_for_backend(backend_name: &str) -> ResidentOwnership {
    match backend_name {
        "cuda" | "cuda-portable" | "metal" | "metal-portable" | "rocm" => {
            ResidentOwnership::StorageOwned
        }
        "vulkan" | "vulkan-portable" | "cpu" | "portable" => ResidentOwnership::RegistryOwned,
        _ => ResidentOwnership::RegistryOwned,
    }
}

/// Backend-neutral resident lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidentResourceState {
    Unregistered,
    RegisteredClean,
    DirtyHost,
    DirtyDevice,
}

/// Replay-safety metadata attached to a resident resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReplayStability {
    NotReplayStable,
    StableWithinStep,
    StableAcrossReplay,
}

/// Metadata for a tensor-like resource known to a backend residency registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentResource {
    pub tensor_id: TensorId,
    pub device: Device,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub element_count: usize,
    pub byte_len: usize,
    pub family: ResidentResourceFamily,
    pub ownership: ResidentOwnership,
    pub state: ResidentResourceState,
    pub replay_stability: ReplayStability,
}

impl ResidentResource {
    pub fn from_tensor(
        tensor: &Tensor,
        family: ResidentResourceFamily,
        ownership: ResidentOwnership,
    ) -> Self {
        let dtype = tensor.dtype();
        let element_count = tensor.element_count();
        Self {
            tensor_id: tensor.id(),
            device: tensor.device(),
            dtype,
            shape: tensor.shape().to_vec(),
            element_count,
            byte_len: dtype.packed_buffer_bytes(element_count),
            family,
            ownership,
            state: ResidentResourceState::RegisteredClean,
            replay_stability: ReplayStability::NotReplayStable,
        }
    }

    pub fn with_state(mut self, state: ResidentResourceState) -> Self {
        self.state = state;
        self
    }

    pub fn with_replay_stability(mut self, replay_stability: ReplayStability) -> Self {
        self.replay_stability = replay_stability;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_resource_describes_tensor_metadata() -> anyhow::Result<()> {
        let tensor = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], vec![2, 2])?;
        let resource = ResidentResource::from_tensor(
            &tensor,
            ResidentResourceFamily::Activation,
            ResidentOwnership::StorageOwned,
        );

        assert_eq!(resource.tensor_id, tensor.id());
        assert_eq!(resource.device, Device::Cpu);
        assert_eq!(resource.dtype, DType::F32);
        assert_eq!(resource.shape, vec![2, 2]);
        assert_eq!(resource.element_count, 4);
        assert_eq!(resource.byte_len, 16);
        assert_eq!(resource.family, ResidentResourceFamily::Activation);
        assert_eq!(resource.ownership, ResidentOwnership::StorageOwned);
        assert_eq!(resource.state, ResidentResourceState::RegisteredClean);
        assert_eq!(resource.replay_stability, ReplayStability::NotReplayStable);
        Ok(())
    }

    #[test]
    fn resident_resource_tracks_lifecycle_and_replay_metadata() -> anyhow::Result<()> {
        let tensor = Tensor::from_slice(&[1_u32, 2, 3], vec![3])?;
        let resource = ResidentResource::from_tensor(
            &tensor,
            ResidentResourceFamily::ReplayInput,
            ResidentOwnership::RegistryOwned,
        )
        .with_state(ResidentResourceState::DirtyDevice)
        .with_replay_stability(ReplayStability::StableAcrossReplay);

        assert_eq!(resource.byte_len, 12);
        assert_eq!(resource.state, ResidentResourceState::DirtyDevice);
        assert_eq!(resource.replay_stability, ReplayStability::StableAcrossReplay);
        Ok(())
    }

    #[test]
    fn resident_ownership_for_backend_maps_current_backends() {
        assert_eq!(
            resident_ownership_for_backend("cuda"),
            ResidentOwnership::StorageOwned
        );
        assert_eq!(
            resident_ownership_for_backend("metal"),
            ResidentOwnership::StorageOwned
        );
        assert_eq!(
            resident_ownership_for_backend("rocm"),
            ResidentOwnership::StorageOwned
        );
        assert_eq!(
            resident_ownership_for_backend("vulkan"),
            ResidentOwnership::RegistryOwned
        );
        assert_eq!(
            resident_ownership_for_backend("cpu"),
            ResidentOwnership::RegistryOwned
        );
        assert_eq!(
            resident_ownership_for_backend("future-backend"),
            ResidentOwnership::RegistryOwned
        );
    }
}
