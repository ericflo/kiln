//! Metal resident activation registry contract.
//!
//! Metal tensors already own their UMA GPU buffers, so residency tracks
//! membership only. The optimizer reads each operand's live Metal storage from
//! the kt tensor passed into the dispatch call.

use anyhow::{Context, Result};
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use kiln_tensor::{DType, MetalStorage, Tensor, TensorId};

static METAL_RESIDENT_ACTIVATION_REGISTRY: OnceLock<Mutex<HashSet<TensorId>>> = OnceLock::new();

fn metal_resident_registry() -> &'static Mutex<HashSet<TensorId>> {
    METAL_RESIDENT_ACTIVATION_REGISTRY.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Short, self-recovering accessor over the registry mutex (poison recovery
/// returns the inner data so a panicking caller can't wedge the registry).
/// Mirrors Vulkan's `with_resident_registry`.
fn with_metal_resident_registry<F, R>(f: F) -> R
where
    F: FnOnce(&mut HashSet<TensorId>) -> R,
{
    let mut guard = metal_resident_registry()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    f(&mut guard)
}

fn is_metal_backed(tensor: &Tensor) -> bool {
    tensor
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .is_some()
}

pub(super) fn register_resident_activation(tensor: &Tensor) -> Result<()> {
    // Metal tensors already carry their GPU buffer; registration only records
    // membership. Decline any non-Metal tensor so callers fall through to the
    // host path via `has_resident_activation == false`.
    if !is_metal_backed(tensor) || tensor.element_count() == 0 {
        return Ok(());
    }
    with_metal_resident_registry(|set| {
        set.insert(tensor.id());
    });
    Ok(())
}

pub(super) fn has_resident_activation(tensor: &Tensor) -> bool {
    with_metal_resident_registry(|set| set.contains(&tensor.id()))
}

pub(super) fn update_resident_activation(_tensor: &Tensor) -> Result<()> {
    // The registry holds no separate copy, so the tensor's own UMA buffer is
    // already the source of truth whether or not it is registered.
    Ok(())
}

pub(super) fn evict_resident_activation(tensor: &Tensor) {
    with_metal_resident_registry(|set| {
        set.remove(&tensor.id());
    });
}

pub(super) fn resolve_resident_activation(
    tensor: &Tensor,
    shape: &[usize],
    dtype: DType,
) -> Result<Option<Tensor>> {
    if !has_resident_activation(tensor) {
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

pub(super) fn all_registered(tensors: &[&Tensor]) -> bool {
    with_metal_resident_registry(|set| tensors.iter().all(|tensor| set.contains(&tensor.id())))
}
