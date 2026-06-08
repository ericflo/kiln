//! Shared resident-resource contracts for backend registries.
//!
//! Phase 3 uses these metadata types to describe the residency lifecycle that
//! CUDA/ROCm/Metal storage-owned registries and Vulkan upload-owned registries
//! already implement through focused residency hooks. They are intentionally
//! backend-neutral and carry no allocation or synchronization behavior.

use anyhow::Result;
use kiln_graph::{ReplayResourceStability, ResidentResourceRef};
use kiln_tensor::{Backend, DType, Device, Tensor, TensorId};

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

/// Runtime backend identity, preserving backend names that use CPU sentinel tensors.
pub fn resident_backend_for_runtime(backend_name: &str, device: Device) -> Backend {
    match backend_name {
        "cpu" | "portable" => Backend::Cpu,
        "cuda" | "cuda-portable" => Backend::Cuda,
        "metal" | "metal-portable" => Backend::Metal,
        "vulkan" | "vulkan-portable" => Backend::Vulkan,
        "rocm" => Backend::Rocm,
        _ => device.backend(),
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

/// Tensor layout metadata attached to a resident resource descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentResourceLayout {
    pub strides: Vec<usize>,
    pub start_offset: usize,
    pub contiguous: bool,
}

impl ResidentResourceLayout {
    pub fn from_tensor(tensor: &Tensor) -> Self {
        Self {
            strides: tensor.strides().to_vec(),
            start_offset: tensor.layout().start_offset(),
            contiguous: tensor.layout().is_contiguous(),
        }
    }
}

/// Metadata for a tensor-like resource known to a backend residency registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentResource {
    pub tensor_id: TensorId,
    pub backend: Backend,
    pub device: Device,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub layout: ResidentResourceLayout,
    pub element_count: usize,
    pub byte_len: usize,
    pub addressable_byte_len: usize,
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
        Self::from_tensor_for_backend(tensor, tensor.device().backend(), family, ownership)
    }

    pub fn from_tensor_for_backend(
        tensor: &Tensor,
        backend: Backend,
        family: ResidentResourceFamily,
        ownership: ResidentOwnership,
    ) -> Self {
        let dtype = tensor.dtype();
        let element_count = tensor.element_count();
        let byte_len = dtype.packed_buffer_bytes(element_count);
        let addressable_elements = tensor.layout().addressable_byte_size(1);
        Self {
            tensor_id: tensor.id(),
            backend,
            device: tensor.device(),
            dtype,
            shape: tensor.shape().to_vec(),
            layout: ResidentResourceLayout::from_tensor(tensor),
            element_count,
            byte_len,
            addressable_byte_len: dtype.packed_buffer_bytes(addressable_elements),
            family,
            ownership,
            state: ResidentResourceState::RegisteredClean,
            replay_stability: ReplayStability::NotReplayStable,
        }
    }

    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    pub fn with_state(mut self, state: ResidentResourceState) -> Self {
        self.state = state;
        self
    }

    pub fn with_replay_stability(mut self, replay_stability: ReplayStability) -> Self {
        self.replay_stability = replay_stability;
        self
    }

    pub fn to_replay_resource_ref(&self) -> ResidentResourceRef {
        self.to_replay_resource_ref_for_backend(self.backend)
    }

    pub fn to_replay_resource_ref_for_backend(&self, backend: Backend) -> ResidentResourceRef {
        ResidentResourceRef {
            tensor_id: Some(self.tensor_id),
            backend,
            dtype: self.dtype,
            shape: self.shape.clone(),
            byte_len: self.byte_len,
            replay_stability: self.replay_stability.into(),
        }
    }
}

impl From<ReplayStability> for ReplayResourceStability {
    fn from(stability: ReplayStability) -> Self {
        match stability {
            ReplayStability::NotReplayStable => ReplayResourceStability::NotReplayStable,
            ReplayStability::StableWithinStep => ReplayResourceStability::StableWithinStep,
            ReplayStability::StableAcrossReplay => ReplayResourceStability::StableAcrossReplay,
        }
    }
}

impl From<&ResidentResource> for ResidentResourceRef {
    fn from(resource: &ResidentResource) -> Self {
        resource.to_replay_resource_ref()
    }
}

/// Backend-neutral lifecycle surface for resident tensor-like resources.
///
/// Backends can implement this over storage-owned membership (CUDA/ROCm/Metal)
/// or upload-owned registries (Vulkan). Returning `Ok(None)`/`None` means the
/// backend declined or does not track that resource family.
pub trait ResidentRegistry: Send + Sync {
    fn register_resource(
        &self,
        _tensor: &Tensor,
        _family: ResidentResourceFamily,
    ) -> Result<Option<ResidentResource>> {
        Ok(None)
    }

    fn update_resource(
        &self,
        _tensor: &Tensor,
        _family: ResidentResourceFamily,
    ) -> Result<Option<ResidentResource>> {
        Ok(None)
    }

    fn evict_resource(&self, _tensor: &Tensor, _family: ResidentResourceFamily) {}

    fn resident_resource(
        &self,
        _tensor: &Tensor,
        _family: ResidentResourceFamily,
    ) -> Option<ResidentResource> {
        None
    }

    fn has_resident_resource(&self, tensor: &Tensor, family: ResidentResourceFamily) -> bool {
        self.resident_resource(tensor, family).is_some()
    }

    fn resolve_resource(
        &self,
        _tensor: &Tensor,
        _family: ResidentResourceFamily,
        _shape: &[usize],
        _dtype: DType,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct LifecycleProbeRegistry {
        states: Mutex<HashMap<(TensorId, ResidentResourceFamily), ResidentResourceState>>,
    }

    impl LifecycleProbeRegistry {
        fn state_for(
            &self,
            tensor: &Tensor,
            family: ResidentResourceFamily,
        ) -> Option<ResidentResourceState> {
            self.states
                .lock()
                .expect("lifecycle probe registry poisoned")
                .get(&(tensor.id(), family))
                .copied()
        }

        fn resource_with_state(
            tensor: &Tensor,
            family: ResidentResourceFamily,
            state: ResidentResourceState,
        ) -> ResidentResource {
            ResidentResource::from_tensor(tensor, family, ResidentOwnership::RegistryOwned)
                .with_state(state)
        }
    }

    impl ResidentRegistry for LifecycleProbeRegistry {
        fn register_resource(
            &self,
            tensor: &Tensor,
            family: ResidentResourceFamily,
        ) -> Result<Option<ResidentResource>> {
            self.states
                .lock()
                .expect("lifecycle probe registry poisoned")
                .insert(
                    (tensor.id(), family),
                    ResidentResourceState::RegisteredClean,
                );
            Ok(Some(Self::resource_with_state(
                tensor,
                family,
                ResidentResourceState::RegisteredClean,
            )))
        }

        fn update_resource(
            &self,
            tensor: &Tensor,
            family: ResidentResourceFamily,
        ) -> Result<Option<ResidentResource>> {
            self.states
                .lock()
                .expect("lifecycle probe registry poisoned")
                .insert((tensor.id(), family), ResidentResourceState::DirtyDevice);
            Ok(Some(Self::resource_with_state(
                tensor,
                family,
                ResidentResourceState::DirtyDevice,
            )))
        }

        fn evict_resource(&self, tensor: &Tensor, family: ResidentResourceFamily) {
            self.states
                .lock()
                .expect("lifecycle probe registry poisoned")
                .insert((tensor.id(), family), ResidentResourceState::Unregistered);
        }

        fn resident_resource(
            &self,
            tensor: &Tensor,
            family: ResidentResourceFamily,
        ) -> Option<ResidentResource> {
            let state = self.state_for(tensor, family)?;
            if state == ResidentResourceState::Unregistered {
                return None;
            }
            Some(Self::resource_with_state(tensor, family, state))
        }

        fn resolve_resource(
            &self,
            tensor: &Tensor,
            family: ResidentResourceFamily,
            _shape: &[usize],
            _dtype: DType,
        ) -> Result<Option<Tensor>> {
            if self.resident_resource(tensor, family).is_some() {
                Ok(Some(tensor.clone()))
            } else {
                Ok(None)
            }
        }
    }

    #[derive(Default)]
    struct DropDrainingProbeRegistry {
        ids: Arc<Mutex<HashSet<TensorId>>>,
    }

    impl DropDrainingProbeRegistry {
        fn shared_ids(&self) -> Arc<Mutex<HashSet<TensorId>>> {
            Arc::clone(&self.ids)
        }
    }

    impl Drop for DropDrainingProbeRegistry {
        fn drop(&mut self) {
            self.ids
                .lock()
                .expect("drop-draining probe registry poisoned")
                .clear();
        }
    }

    impl ResidentRegistry for DropDrainingProbeRegistry {
        fn register_resource(
            &self,
            tensor: &Tensor,
            family: ResidentResourceFamily,
        ) -> Result<Option<ResidentResource>> {
            if family != ResidentResourceFamily::Activation {
                return Ok(None);
            }
            self.ids
                .lock()
                .expect("drop-draining probe registry poisoned")
                .insert(tensor.id());
            Ok(Some(ResidentResource::from_tensor(
                tensor,
                family,
                ResidentOwnership::RegistryOwned,
            )))
        }

        fn resident_resource(
            &self,
            tensor: &Tensor,
            family: ResidentResourceFamily,
        ) -> Option<ResidentResource> {
            if family != ResidentResourceFamily::Activation {
                return None;
            }
            if self
                .ids
                .lock()
                .expect("drop-draining probe registry poisoned")
                .contains(&tensor.id())
            {
                Some(ResidentResource::from_tensor(
                    tensor,
                    family,
                    ResidentOwnership::RegistryOwned,
                ))
            } else {
                None
            }
        }
    }

    #[test]
    fn resident_resource_describes_tensor_metadata() -> anyhow::Result<()> {
        let tensor = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], vec![2, 2])?;
        let resource = ResidentResource::from_tensor(
            &tensor,
            ResidentResourceFamily::Activation,
            ResidentOwnership::StorageOwned,
        );

        assert_eq!(resource.tensor_id, tensor.id());
        assert_eq!(resource.backend, Backend::Cpu);
        assert_eq!(resource.device, Device::Cpu);
        assert_eq!(resource.dtype, DType::F32);
        assert_eq!(resource.shape, vec![2, 2]);
        assert_eq!(resource.layout.strides, vec![2, 1]);
        assert_eq!(resource.layout.start_offset, 0);
        assert!(resource.layout.contiguous);
        assert_eq!(resource.element_count, 4);
        assert_eq!(resource.byte_len, 16);
        assert_eq!(resource.addressable_byte_len, 16);
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
        assert_eq!(
            resource.replay_stability,
            ReplayStability::StableAcrossReplay
        );
        let replay_ref = resource.to_replay_resource_ref();
        assert_eq!(replay_ref.tensor_id, Some(resource.tensor_id));
        assert_eq!(replay_ref.backend, Backend::Cpu);
        assert_eq!(
            replay_ref.replay_stability,
            ReplayResourceStability::StableAcrossReplay
        );
        Ok(())
    }

    #[test]
    fn resident_registry_lifecycle_contract_tracks_register_update_evict() -> anyhow::Result<()> {
        let registry = LifecycleProbeRegistry::default();
        let tensor = Tensor::from_slice(&[1.0_f32, 2.0], vec![2])?;
        let family = ResidentResourceFamily::Activation;

        assert!(
            ResidentRegistry::resolve_resource(&registry, &tensor, family, &[2], DType::F32)?
                .is_none()
        );

        let registered = ResidentRegistry::register_resource(&registry, &tensor, family)?.unwrap();
        assert_eq!(registered.state, ResidentResourceState::RegisteredClean);
        assert_eq!(
            registry.state_for(&tensor, family),
            Some(ResidentResourceState::RegisteredClean)
        );
        assert!(
            ResidentRegistry::resolve_resource(&registry, &tensor, family, &[2], DType::F32)?
                .is_some()
        );

        let updated = ResidentRegistry::update_resource(&registry, &tensor, family)?.unwrap();
        assert_eq!(updated.state, ResidentResourceState::DirtyDevice);
        assert_eq!(
            registry.state_for(&tensor, family),
            Some(ResidentResourceState::DirtyDevice)
        );

        ResidentRegistry::evict_resource(&registry, &tensor, family);
        assert_eq!(
            registry.state_for(&tensor, family),
            Some(ResidentResourceState::Unregistered)
        );
        assert!(ResidentRegistry::resident_resource(&registry, &tensor, family).is_none());
        assert!(
            ResidentRegistry::resolve_resource(&registry, &tensor, family, &[2], DType::F32)?
                .is_none()
        );

        Ok(())
    }

    #[test]
    fn resident_registry_drop_drains_registry_on_backend_drop() -> anyhow::Result<()> {
        let tensor = Tensor::from_slice(&[1.0_f32, 2.0], vec![2])?;
        let shared_ids = {
            let registry = DropDrainingProbeRegistry::default();
            let shared_ids = registry.shared_ids();
            ResidentRegistry::register_resource(
                &registry,
                &tensor,
                ResidentResourceFamily::Activation,
            )?;
            assert!(ResidentRegistry::has_resident_resource(
                &registry,
                &tensor,
                ResidentResourceFamily::Activation
            ));
            assert_eq!(
                shared_ids
                    .lock()
                    .expect("drop-draining probe registry poisoned")
                    .len(),
                1
            );
            shared_ids
        };

        assert!(
            shared_ids
                .lock()
                .expect("drop-draining probe registry poisoned")
                .is_empty(),
            "dropping the backend-owned registry should drain resident membership"
        );
        Ok(())
    }

    #[test]
    fn resident_resource_can_target_explicit_replay_backend() -> anyhow::Result<()> {
        let tensor = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], vec![2, 2])?;
        let resource = ResidentResource::from_tensor(
            &tensor,
            ResidentResourceFamily::ReplayInput,
            ResidentOwnership::RegistryOwned,
        )
        .with_replay_stability(ReplayStability::StableWithinStep);

        let vk_resource = resource.clone().with_backend(Backend::Vulkan);
        let replay_ref = vk_resource.to_replay_resource_ref();
        assert_eq!(replay_ref.backend, Backend::Vulkan);
        assert_eq!(replay_ref.tensor_id, Some(resource.tensor_id));

        let replay_ref = resource.to_replay_resource_ref_for_backend(Backend::Vulkan);
        assert_eq!(replay_ref.backend, Backend::Vulkan);
        assert_eq!(replay_ref.tensor_id, Some(resource.tensor_id));
        assert_eq!(replay_ref.dtype, DType::F32);
        assert_eq!(replay_ref.shape, vec![2, 2]);
        assert_eq!(replay_ref.byte_len, resource.byte_len);
        assert_eq!(
            replay_ref.replay_stability,
            ReplayResourceStability::StableWithinStep
        );
        assert!(!replay_ref.is_replay_stable());
        Ok(())
    }

    #[test]
    fn resident_resource_describes_strided_layout_metadata() -> anyhow::Result<()> {
        let tensor = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let narrowed = tensor.narrow(1, 1, 2)?;
        let resource = ResidentResource::from_tensor(
            &narrowed,
            ResidentResourceFamily::Activation,
            ResidentOwnership::RegistryOwned,
        );

        assert_eq!(resource.shape, vec![2, 2]);
        assert_eq!(resource.layout.strides, vec![3, 1]);
        assert_eq!(resource.layout.start_offset, 1);
        assert!(!resource.layout.contiguous);
        assert_eq!(resource.byte_len, 16);
        assert_eq!(resource.addressable_byte_len, 24);
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

    #[test]
    fn resident_backend_for_runtime_preserves_cpu_sentinel_backends() {
        assert_eq!(
            resident_backend_for_runtime("vulkan", Device::Cpu),
            Backend::Vulkan
        );
        assert_eq!(
            resident_backend_for_runtime("metal-portable", Device::Cpu),
            Backend::Metal
        );
        assert_eq!(
            resident_backend_for_runtime("future-backend", Device::Cpu),
            Backend::Cpu
        );
    }
}
