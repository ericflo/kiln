//! Metal resident activation registry contract.
//!
//! Metal tensors already own their UMA GPU buffers, so residency tracks
//! membership only. The optimizer reads each operand's live Metal storage from
//! the kt tensor passed into the dispatch call.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Mutex;

use kiln_tensor::{DType, MetalStorage, Tensor, TensorId};

pub(super) type ResidentActivationRegistry =
    Mutex<HashMap<TensorId, super::residency::ResidentResource>>;

pub(super) fn new_resident_activation_registry() -> ResidentActivationRegistry {
    Mutex::new(HashMap::new())
}

/// Short, self-recovering accessor over the registry mutex (poison recovery
/// returns the inner data so a panicking caller can't wedge the registry).
/// Mirrors Vulkan's `with_resident_registry`.
fn with_metal_resident_registry<F, R>(registry: &ResidentActivationRegistry, f: F) -> R
where
    F: FnOnce(&mut HashMap<TensorId, super::residency::ResidentResource>) -> R,
{
    let mut guard = registry.lock().unwrap_or_else(|e| e.into_inner());
    f(&mut guard)
}

fn is_metal_backed(tensor: &Tensor) -> bool {
    tensor
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .is_some()
}

fn metal_resident_activation_resource(
    tensor: &Tensor,
    state: super::residency::ResidentResourceState,
) -> super::residency::ResidentResource {
    super::residency::ResidentResource::from_tensor_for_backend(
        tensor,
        super::residency::resident_backend_for_runtime("metal", tensor.device()),
        super::residency::ResidentResourceFamily::Activation,
        super::residency::ResidentOwnership::StorageOwned,
    )
    .with_state(state)
    .with_replay_stability(super::residency::ReplayStability::StableWithinStep)
}

pub(super) fn register_resident_activation(
    registry: &ResidentActivationRegistry,
    tensor: &Tensor,
) -> Result<Option<super::residency::ResidentResource>> {
    // Metal tensors already carry their GPU buffer; registration only records
    // membership. Decline any non-Metal tensor so callers fall through to the
    // host path via `has_resident_activation == false`.
    if !is_metal_backed(tensor) || tensor.element_count() == 0 {
        return Ok(None);
    }
    let resource = metal_resident_activation_resource(
        tensor,
        super::residency::ResidentResourceState::RegisteredClean,
    );
    with_metal_resident_registry(registry, |resources| {
        resources.insert(tensor.id(), resource.clone());
    });
    Ok(Some(resource))
}

pub(super) fn has_resident_activation(
    registry: &ResidentActivationRegistry,
    tensor: &Tensor,
) -> bool {
    with_metal_resident_registry(registry, |resources| resources.contains_key(&tensor.id()))
}

pub(super) fn update_resident_activation(
    registry: &ResidentActivationRegistry,
    tensor: &Tensor,
) -> Result<Option<super::residency::ResidentResource>> {
    // The registry holds no separate copy, so the tensor's own UMA buffer is
    // already the source of truth whether or not it is registered.
    let resource = with_metal_resident_registry(registry, |resources| {
        if !resources.contains_key(&tensor.id()) {
            return None;
        }
        let resource = metal_resident_activation_resource(
            tensor,
            super::residency::ResidentResourceState::DirtyDevice,
        );
        resources.insert(tensor.id(), resource.clone());
        Some(resource)
    });
    Ok(resource)
}

pub(super) fn evict_resident_activation(registry: &ResidentActivationRegistry, tensor: &Tensor) {
    with_metal_resident_registry(registry, |resources| {
        resources.remove(&tensor.id());
    });
}

pub(super) fn resident_activation_resource(
    registry: &ResidentActivationRegistry,
    tensor: &Tensor,
) -> Option<super::residency::ResidentResource> {
    with_metal_resident_registry(registry, |resources| resources.get(&tensor.id()).cloned())
}

pub(super) fn resolve_resident_activation(
    registry: &ResidentActivationRegistry,
    tensor: &Tensor,
    shape: &[usize],
    dtype: DType,
) -> Result<Option<Tensor>> {
    if !has_resident_activation(registry, tensor) {
        return Ok(None);
    }

    let resolved = kiln_tensor::metal_deep_copy(tensor)
        .context("resolve_resident_activation: metal_deep_copy")?;
    if resolved.dims() != shape || resolved.dtype() != dtype {
        anyhow::bail!(
            "resolve_resident_activation: registry tensor shape/dtype ({:?},{:?}) \
             != requested ({:?},{:?})",
            resolved.dims(),
            resolved.dtype(),
            shape,
            dtype,
        );
    }
    Ok(Some(resolved))
}

pub(super) fn all_registered(registry: &ResidentActivationRegistry, tensors: &[&Tensor]) -> bool {
    with_metal_resident_registry(registry, |resources| {
        tensors
            .iter()
            .all(|tensor| resources.contains_key(&tensor.id()))
    })
}
