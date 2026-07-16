//! Vulkan resident activation registry storage.
//!
//! `ResidencyBackend` methods on `VulkanBackend` use this module's
//! backend-owned TensorId -> VulkanBuffer maps for optimizer dispatch and
//! recurrent-state residency.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub(super) struct ResidentActivationEntry {
    pub(super) buffer: Arc<kiln_vulkan_kernel::VulkanBuffer>,
    pub(super) resource: super::residency::ResidentResource,
}

impl ResidentActivationEntry {
    pub(super) fn new(
        buffer: Arc<kiln_vulkan_kernel::VulkanBuffer>,
        resource: super::residency::ResidentResource,
    ) -> Self {
        Self { buffer, resource }
    }
}

pub(super) type ResidentActivationRegistry =
    Mutex<HashMap<kiln_tensor::TensorId, ResidentActivationEntry>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct PrefillRecurrentStateKey {
    pub(super) owner_id: u64,
    pub(super) layer_idx: usize,
}

#[derive(Debug)]
struct RecurrentStateResidentEntry {
    buffer: Arc<kiln_vulkan_kernel::VulkanBuffer>,
    prefill_key: Option<PrefillRecurrentStateKey>,
}

#[derive(Debug, Default)]
pub(super) struct RecurrentStateResidentEntries {
    by_tensor: HashMap<kiln_tensor::TensorId, RecurrentStateResidentEntry>,
    by_prefill: HashMap<PrefillRecurrentStateKey, kiln_tensor::TensorId>,
}

pub(super) type RecurrentStateResidentRegistry = Mutex<RecurrentStateResidentEntries>;

pub(super) fn new_resident_activation_registry() -> ResidentActivationRegistry {
    Mutex::new(HashMap::new())
}

pub(super) fn new_recurrent_state_resident_registry() -> RecurrentStateResidentRegistry {
    Mutex::new(RecurrentStateResidentEntries::default())
}

thread_local! {
    static RECURRENT_STATE_RESIDENT_SCOPE_DEPTH: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static PREFILL_RECURRENT_STATE_OWNER_STACK: std::cell::RefCell<Vec<u64>> =
        const { std::cell::RefCell::new(Vec::new()) };
    static PREFILL_RECURRENT_STATE_LAYER_STACK: std::cell::RefCell<Vec<usize>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Helper: short, self-recovering accessor that wraps the registry's mutex.
/// Poison recovery returns the inner data so the registry never stays
/// inaccessible just because some panicking code touched it.
pub(super) fn with_resident_registry<F, R>(registry: &ResidentActivationRegistry, f: F) -> R
where
    F: FnOnce(&mut HashMap<kiln_tensor::TensorId, ResidentActivationEntry>) -> R,
{
    let mut guard = registry.lock().unwrap_or_else(|e| e.into_inner());
    f(&mut guard)
}

pub(super) fn recurrent_state_resident_scope_active() -> bool {
    RECURRENT_STATE_RESIDENT_SCOPE_DEPTH.with(|depth| depth.get() > 0)
}

pub(super) fn enter_recurrent_state_resident_scope() {
    RECURRENT_STATE_RESIDENT_SCOPE_DEPTH.with(|depth| {
        depth.set(depth.get() + 1);
    });
}

pub(super) fn exit_recurrent_state_resident_scope() {
    RECURRENT_STATE_RESIDENT_SCOPE_DEPTH.with(|depth| {
        let previous = depth.get();
        if previous == 0 {
            return;
        }
        depth.set(previous - 1);
    });
}

pub(super) fn enter_prefill_recurrent_state_resident_scope(owner_id: u64) {
    enter_recurrent_state_resident_scope();
    PREFILL_RECURRENT_STATE_OWNER_STACK.with(|owners| owners.borrow_mut().push(owner_id));
}

pub(super) fn exit_prefill_recurrent_state_resident_scope() {
    PREFILL_RECURRENT_STATE_OWNER_STACK.with(|owners| {
        owners.borrow_mut().pop();
    });
    exit_recurrent_state_resident_scope();
}

pub(super) fn enter_prefill_recurrent_state_layer_scope(layer_idx: usize) -> bool {
    let has_owner = PREFILL_RECURRENT_STATE_OWNER_STACK.with(|owners| !owners.borrow().is_empty());
    if has_owner {
        PREFILL_RECURRENT_STATE_LAYER_STACK.with(|layers| layers.borrow_mut().push(layer_idx));
    }
    has_owner
}

pub(super) fn exit_prefill_recurrent_state_layer_scope() {
    PREFILL_RECURRENT_STATE_LAYER_STACK.with(|layers| {
        layers.borrow_mut().pop();
    });
}

fn active_prefill_recurrent_state_key() -> Option<PrefillRecurrentStateKey> {
    let owner_id =
        PREFILL_RECURRENT_STATE_OWNER_STACK.with(|owners| owners.borrow().last().copied())?;
    let layer_idx =
        PREFILL_RECURRENT_STATE_LAYER_STACK.with(|layers| layers.borrow().last().copied())?;
    Some(PrefillRecurrentStateKey {
        owner_id,
        layer_idx,
    })
}

fn remove_recurrent_state_entry(
    entries: &mut RecurrentStateResidentEntries,
    id: kiln_tensor::TensorId,
) -> Option<RecurrentStateResidentEntry> {
    let entry = entries.by_tensor.remove(&id)?;
    if let Some(key) = entry.prefill_key
        && entries.by_prefill.get(&key) == Some(&id)
    {
        entries.by_prefill.remove(&key);
    }
    Some(entry)
}

pub(super) fn get_recurrent_state_resident_buffer(
    registry: &RecurrentStateResidentRegistry,
    id: kiln_tensor::TensorId,
) -> Option<Arc<kiln_vulkan_kernel::VulkanBuffer>> {
    let entries = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    active_prefill_recurrent_state_key()
        .and_then(|key| entries.by_prefill.get(&key))
        .and_then(|resident_id| entries.by_tensor.get(resident_id))
        .or_else(|| entries.by_tensor.get(&id))
        .map(|entry| Arc::clone(&entry.buffer))
}

pub(super) fn take_recurrent_state_resident_buffer(
    registry: &RecurrentStateResidentRegistry,
    id: kiln_tensor::TensorId,
) -> Option<Arc<kiln_vulkan_kernel::VulkanBuffer>> {
    let mut entries = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    remove_recurrent_state_entry(&mut entries, id).map(|entry| entry.buffer)
}

pub(super) fn get_prefill_recurrent_state_resident_buffer(
    registry: &RecurrentStateResidentRegistry,
    owner_id: u64,
    layer_idx: usize,
    fallback_id: kiln_tensor::TensorId,
) -> Option<Arc<kiln_vulkan_kernel::VulkanBuffer>> {
    let entries = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let key = PrefillRecurrentStateKey {
        owner_id,
        layer_idx,
    };
    entries
        .by_prefill
        .get(&key)
        .and_then(|id| entries.by_tensor.get(id))
        .or_else(|| entries.by_tensor.get(&fallback_id))
        .map(|entry| Arc::clone(&entry.buffer))
}

pub(super) fn take_prefill_recurrent_state_resident_buffer(
    registry: &RecurrentStateResidentRegistry,
    owner_id: u64,
    layer_idx: usize,
    fallback_id: kiln_tensor::TensorId,
) -> Option<Arc<kiln_vulkan_kernel::VulkanBuffer>> {
    let mut entries = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let key = PrefillRecurrentStateKey {
        owner_id,
        layer_idx,
    };
    let id = entries.by_prefill.remove(&key).unwrap_or(fallback_id);
    remove_recurrent_state_entry(&mut entries, id)
        .or_else(|| remove_recurrent_state_entry(&mut entries, fallback_id))
        .map(|entry| entry.buffer)
}

pub(super) fn replace_prefill_recurrent_state_resident_buffer(
    registry: &RecurrentStateResidentRegistry,
    owner_id: u64,
    layer_idx: usize,
    fallback_id: kiln_tensor::TensorId,
    buffer: Arc<kiln_vulkan_kernel::VulkanBuffer>,
) -> bool {
    let mut entries = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let key = PrefillRecurrentStateKey {
        owner_id,
        layer_idx,
    };
    let id = entries
        .by_prefill
        .get(&key)
        .copied()
        .filter(|id| entries.by_tensor.contains_key(id))
        .or_else(|| {
            entries
                .by_tensor
                .contains_key(&fallback_id)
                .then_some(fallback_id)
        });
    let Some(id) = id else {
        return false;
    };
    entries
        .by_tensor
        .get_mut(&id)
        .expect("resident GDN state disappeared while registry lock was held")
        .buffer = buffer;
    true
}

pub(super) fn insert_recurrent_state_resident_buffer(
    registry: &RecurrentStateResidentRegistry,
    id: kiln_tensor::TensorId,
    buffer: Arc<kiln_vulkan_kernel::VulkanBuffer>,
) {
    let mut entries = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    remove_recurrent_state_entry(&mut entries, id);
    let prefill_key = active_prefill_recurrent_state_key();
    if let Some(key) = prefill_key {
        if let Some(previous_id) = entries.by_prefill.insert(key, id)
            && previous_id != id
        {
            remove_recurrent_state_entry(&mut entries, previous_id);
            entries.by_prefill.insert(key, id);
        }
    }
    entries.by_tensor.insert(
        id,
        RecurrentStateResidentEntry {
            buffer,
            prefill_key,
        },
    );
}

/// Atomically transfer a recurrent buffer between tensor identities.
///
/// A resident host tensor is a metadata handle, not the authoritative state
/// value. Functional dtype/layout operations mint a new `TensorId`; moving the
/// registry entry with that identity change keeps later materialization and
/// cancellation keyed to the handle retained by `LinearAttentionState`.
pub(super) fn rekey_recurrent_state_resident_buffer(
    registry: &RecurrentStateResidentRegistry,
    old_id: kiln_tensor::TensorId,
    new_id: kiln_tensor::TensorId,
) -> anyhow::Result<bool> {
    let mut entries = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if old_id == new_id {
        return Ok(entries.by_tensor.contains_key(&old_id));
    }
    if !entries.by_tensor.contains_key(&old_id) {
        return Ok(false);
    }
    anyhow::ensure!(
        !entries.by_tensor.contains_key(&new_id),
        "cannot rekey resident GDN state: destination TensorId already owns a buffer"
    );
    let entry = entries
        .by_tensor
        .remove(&old_id)
        .expect("resident GDN state disappeared while registry lock was held");
    if let Some(key) = entry.prefill_key {
        entries.by_prefill.insert(key, new_id);
    }
    entries.by_tensor.insert(new_id, entry);
    Ok(true)
}

pub(super) fn remove_recurrent_state_resident_buffer(
    registry: &RecurrentStateResidentRegistry,
    id: kiln_tensor::TensorId,
) {
    let mut entries = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    remove_recurrent_state_entry(&mut entries, id);
}

pub(super) fn remove_prefill_recurrent_state_resident_buffers(
    registry: &RecurrentStateResidentRegistry,
    owner_id: u64,
) -> usize {
    let mut entries = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let ids: Vec<_> = entries
        .by_tensor
        .iter()
        .filter_map(|(id, entry)| {
            entry
                .prefill_key
                .is_some_and(|key| key.owner_id == owner_id)
                .then_some(*id)
        })
        .collect();
    let mut removed = 0;
    for id in ids {
        removed += usize::from(remove_recurrent_state_entry(&mut entries, id).is_some());
    }
    entries.by_prefill.retain(|key, _| key.owner_id != owner_id);
    removed
}

pub(super) fn contains_recurrent_state_resident_buffer(
    registry: &RecurrentStateResidentRegistry,
    id: kiln_tensor::TensorId,
) -> bool {
    let entries = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    entries.by_tensor.contains_key(&id)
}

pub(super) fn recurrent_state_residency_stats(
    registry: &RecurrentStateResidentRegistry,
) -> super::GdnRecurrentStateResidencyStats {
    let entries = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    entries.by_tensor.values().fold(
        super::GdnRecurrentStateResidencyStats {
            entry_count: entries.by_tensor.len(),
            ..Default::default()
        },
        |mut stats, entry| {
            stats.buffer_bytes = stats.buffer_bytes.saturating_add(entry.buffer.size());
            stats.allocation_bytes = stats
                .allocation_bytes
                .saturating_add(entry.buffer.allocation_size());
            stats
        },
    )
}

pub(super) fn recurrent_state_resident_buffers_for<I>(
    registry: &RecurrentStateResidentRegistry,
    ids: I,
) -> Option<Vec<Arc<kiln_vulkan_kernel::VulkanBuffer>>>
where
    I: IntoIterator<Item = kiln_tensor::TensorId>,
{
    let entries = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    ids.into_iter()
        .map(|id| {
            entries
                .by_tensor
                .get(&id)
                .map(|entry| Arc::clone(&entry.buffer))
        })
        .collect::<Option<Vec<_>>>()
}

pub(super) fn replace_recurrent_state_resident_buffer(
    registry: &RecurrentStateResidentRegistry,
    old_id: kiln_tensor::TensorId,
    new_id: kiln_tensor::TensorId,
    buffer: Arc<kiln_vulkan_kernel::VulkanBuffer>,
) {
    let mut entries = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let prefill_key =
        remove_recurrent_state_entry(&mut entries, old_id).and_then(|entry| entry.prefill_key);
    remove_recurrent_state_entry(&mut entries, new_id);
    if let Some(key) = prefill_key {
        entries.by_prefill.insert(key, new_id);
    }
    entries.by_tensor.insert(
        new_id,
        RecurrentStateResidentEntry {
            buffer,
            prefill_key,
        },
    );
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::backend::ResidencyBackend;
    use crate::backend::vulkan::VulkanBackend;

    fn test_backend() -> VulkanBackend {
        VulkanBackend::new(kiln_tensor::Device::Cpu)
    }

    fn skip_without_vulkan(backend: &VulkanBackend, message: &str) -> bool {
        if backend.has_vulkan() {
            false
        } else {
            eprintln!("{message}");
            true
        }
    }

    /// `update_resident_activation` must overwrite the registry buffer with
    /// the tensor's current bytes. The SGD path relies on this to keep resident
    /// LoRA parameters coherent after host-side storage changes.
    #[test]
    fn update_resident_activation_overwrites_buffer() -> Result<()> {
        let backend = test_backend();
        if skip_without_vulkan(&backend, "Vulkan device unavailable, skipping") {
            return Ok(());
        }

        let initial = kiln_tensor::Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2))?
            .to_dtype(kiln_tensor::DType::BF16)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &initial)?;
        let resolved = ResidencyBackend::runtime_resolve_resident_activation(
            &backend,
            &initial,
            &[2, 2],
            kiln_tensor::DType::BF16,
        )?
        .expect("must resolve right after register");
        let init_v: Vec<f32> = resolved
            .to_dtype(kiln_tensor::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(init_v, vec![1.0, 2.0, 3.0, 4.0]);

        let v = kiln_tensor::Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0], (2, 2))?
            .to_dtype(kiln_tensor::DType::BF16)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &v)?;
        let resolved_v = ResidencyBackend::runtime_resolve_resident_activation(
            &backend,
            &v,
            &[2, 2],
            kiln_tensor::DType::BF16,
        )?
        .expect("v must resolve after register");
        let v_init_v: Vec<f32> = resolved_v
            .to_dtype(kiln_tensor::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(v_init_v, vec![10.0, 20.0, 30.0, 40.0]);

        let newer_data =
            kiln_tensor::Tensor::from_vec(vec![100.0f32, 200.0, 300.0, 400.0], (2, 2))?
                .to_dtype(kiln_tensor::DType::BF16)?;
        v.slice_set(&newer_data, 0, 0)?;
        ResidencyBackend::runtime_update_resident_activation(&backend, &v)?;
        let resolved_after = ResidencyBackend::runtime_resolve_resident_activation(
            &backend,
            &v,
            &[2, 2],
            kiln_tensor::DType::BF16,
        )?
        .expect("v must resolve after update");
        let after_v: Vec<f32> = resolved_after
            .to_dtype(kiln_tensor::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(after_v, vec![100.0, 200.0, 300.0, 400.0]);

        ResidencyBackend::runtime_evict_resident_activation(&backend, &initial);
        ResidencyBackend::runtime_evict_resident_activation(&backend, &v);
        Ok(())
    }

    /// Updating an unregistered tensor is a no-op so dtype-agnostic callers can
    /// safely call it for both resident and legacy host tensors.
    #[test]
    fn update_resident_activation_noop_when_not_registered() -> Result<()> {
        let backend = test_backend();
        if skip_without_vulkan(&backend, "Vulkan device unavailable, skipping") {
            return Ok(());
        }
        let t = kiln_tensor::Tensor::from_vec(vec![1.0f32; 4], (4,))?;

        ResidencyBackend::runtime_update_resident_activation(&backend, &t)?;
        assert!(!ResidencyBackend::runtime_has_resident_activation(
            &backend, &t
        ));
        Ok(())
    }

    /// Re-registration after eviction must work for the trainer's per-step
    /// lifecycle: evict old boundaries, then register fresh values.
    #[test]
    fn resident_activation_re_register_after_evict() -> Result<()> {
        let backend = test_backend();
        if skip_without_vulkan(&backend, "Vulkan device unavailable, skipping") {
            return Ok(());
        }
        let t = kiln_tensor::Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2))?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &t)?;
        assert!(ResidencyBackend::runtime_has_resident_activation(
            &backend, &t
        ));
        ResidencyBackend::runtime_evict_resident_activation(&backend, &t);
        assert!(!ResidencyBackend::runtime_has_resident_activation(
            &backend, &t
        ));

        ResidencyBackend::runtime_register_resident_activation(&backend, &t)?;
        assert!(
            ResidencyBackend::runtime_has_resident_activation(&backend, &t),
            "tensor must be registered again after eviction"
        );
        let resolved = ResidencyBackend::runtime_resolve_resident_activation(
            &backend,
            &t,
            &[2, 2],
            kiln_tensor::DType::F32,
        )?
        .expect("must resolve after re-register");
        let data: Vec<f32> = resolved.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
        ResidencyBackend::runtime_evict_resident_activation(&backend, &t);
        Ok(())
    }

    /// Empty-tensor input must not panic the Vulkan allocator. The hook bails
    /// silently and leaves the tensor unregistered.
    #[test]
    fn register_resident_activation_handles_empty_tensor() -> Result<()> {
        let backend = test_backend();
        if skip_without_vulkan(&backend, "Vulkan device unavailable, skipping") {
            return Ok(());
        }
        let empty: kiln_tensor::Tensor = kiln_tensor::Tensor::from_vec(Vec::<f32>::new(), (0,))?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &empty)?;
        assert!(
            !ResidencyBackend::runtime_has_resident_activation(&backend, &empty),
            "empty tensor must not be registered (zero-size driver issue)"
        );
        Ok(())
    }

    /// `resolve_resident_activation` must reconstruct a kt tensor whose data
    /// matches the originally registered tensor's bytes.
    #[test]
    fn resolve_resident_activation_round_trip() -> Result<()> {
        let backend = test_backend();
        if skip_without_vulkan(&backend, "Vulkan device unavailable, skipping") {
            return Ok(());
        }
        let original_data = vec![1.5f32, -2.5, 3.25, -4.75];
        let t = kiln_tensor::Tensor::from_vec(original_data.clone(), (2, 2))?;

        let unresolved = ResidencyBackend::runtime_resolve_resident_activation(
            &backend,
            &t,
            &[2, 2],
            kiln_tensor::DType::F32,
        )?;
        assert!(unresolved.is_none(), "unregistered tensor must not resolve");

        ResidencyBackend::runtime_register_resident_activation(&backend, &t)?;
        let resolved = ResidencyBackend::runtime_resolve_resident_activation(
            &backend,
            &t,
            &[2, 2],
            kiln_tensor::DType::F32,
        )?
        .expect("must resolve once registered");
        assert_eq!(resolved.dims(), &[2, 2]);
        let resolved_data: Vec<f32> = resolved.flatten_all()?.to_vec1::<f32>()?;
        for (i, (got, want)) in resolved_data.iter().zip(original_data.iter()).enumerate() {
            assert!((got - want).abs() < 1e-9, "idx {i}: got {got} want {want}");
        }

        ResidencyBackend::runtime_evict_resident_activation(&backend, &t);
        let unresolved = ResidencyBackend::runtime_resolve_resident_activation(
            &backend,
            &t,
            &[2, 2],
            kiln_tensor::DType::F32,
        )?;
        assert!(unresolved.is_none());
        Ok(())
    }

    #[test]
    fn resident_activation_register_evict_round_trip() -> Result<()> {
        let backend = test_backend();
        assert!(
            ResidencyBackend::runtime_supports_resident_activation(&backend),
            "VulkanBackend must advertise resident-activation support"
        );
        if skip_without_vulkan(
            &backend,
            "Vulkan device unavailable, skipping live registry test",
        ) {
            return Ok(());
        }

        let t = kiln_tensor::Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2))?;
        assert!(
            !ResidencyBackend::runtime_has_resident_activation(&backend, &t),
            "fresh tensor must not be registered"
        );
        ResidencyBackend::runtime_register_resident_activation(&backend, &t)?;
        assert!(
            ResidencyBackend::runtime_has_resident_activation(&backend, &t),
            "tensor must be registered after register_resident_activation"
        );
        ResidencyBackend::runtime_register_resident_activation(&backend, &t)?;
        assert!(ResidencyBackend::runtime_has_resident_activation(
            &backend, &t
        ));
        ResidencyBackend::runtime_evict_resident_activation(&backend, &t);
        assert!(
            !ResidencyBackend::runtime_has_resident_activation(&backend, &t),
            "tensor must be unregistered after evict_resident_activation"
        );
        ResidencyBackend::runtime_evict_resident_activation(&backend, &t);
        Ok(())
    }
}
