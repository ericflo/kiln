//! Vulkan resident activation registry storage.
//!
//! The runtime methods still live on `VulkanBackend`; this module owns the
//! process-global TensorId -> VulkanBuffer map they share with optimizer
//! dispatch.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

type ResidentActivationRegistry =
    Mutex<HashMap<kiln_tensor::TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>>;

/// General-purpose resident-activation registry keyed by the kt
/// `kiln_tensor::TensorId`. Process-global (not thread-local) so worker threads
/// spawned by rayon, etc. see the same registry as the thread that registered.
///
/// Separate from Vulkan's recurrent-state cache so the GDN-specific hot path
/// can keep its own thread-local scope-limited lifecycle without growing
/// accidental coupling to non-recurrent activations.
static RESIDENT_ACTIVATION_REGISTRY: OnceLock<ResidentActivationRegistry> = OnceLock::new();

thread_local! {
    static RECURRENT_STATE_RESIDENT_SCOPE_DEPTH: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static RECURRENT_STATE_RESIDENT_CACHE:
        std::cell::RefCell<HashMap<kiln_tensor::TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>> =
        std::cell::RefCell::new(HashMap::new());
}

fn resident_registry() -> &'static ResidentActivationRegistry {
    RESIDENT_ACTIVATION_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Helper: short, self-recovering accessor that wraps the registry's mutex.
/// Poison recovery returns the inner data so the registry never stays
/// inaccessible just because some panicking code touched it.
pub(super) fn with_resident_registry<F, R>(f: F) -> R
where
    F: FnOnce(&mut HashMap<kiln_tensor::TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>) -> R,
{
    let mut guard = resident_registry()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
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

pub(super) fn get_recurrent_state_resident_buffer(
    id: kiln_tensor::TensorId,
) -> Option<Arc<kiln_vulkan_kernel::VulkanBuffer>> {
    RECURRENT_STATE_RESIDENT_CACHE.with(|cache| cache.borrow().get(&id).cloned())
}

pub(super) fn take_recurrent_state_resident_buffer(
    id: kiln_tensor::TensorId,
) -> Option<Arc<kiln_vulkan_kernel::VulkanBuffer>> {
    RECURRENT_STATE_RESIDENT_CACHE.with(|cache| cache.borrow_mut().remove(&id))
}

pub(super) fn insert_recurrent_state_resident_buffer(
    id: kiln_tensor::TensorId,
    buffer: Arc<kiln_vulkan_kernel::VulkanBuffer>,
) {
    RECURRENT_STATE_RESIDENT_CACHE.with(|cache| {
        cache.borrow_mut().insert(id, buffer);
    });
}

pub(super) fn remove_recurrent_state_resident_buffer(id: kiln_tensor::TensorId) {
    RECURRENT_STATE_RESIDENT_CACHE.with(|cache| {
        cache.borrow_mut().remove(&id);
    });
}

pub(super) fn contains_recurrent_state_resident_buffer(id: kiln_tensor::TensorId) -> bool {
    RECURRENT_STATE_RESIDENT_CACHE.with(|cache| cache.borrow().contains_key(&id))
}

pub(super) fn recurrent_state_resident_buffers_for<I>(
    ids: I,
) -> Option<Vec<Arc<kiln_vulkan_kernel::VulkanBuffer>>>
where
    I: IntoIterator<Item = kiln_tensor::TensorId>,
{
    RECURRENT_STATE_RESIDENT_CACHE.with(|cache| {
        let cache = cache.borrow();
        ids.into_iter()
            .map(|id| cache.get(&id).cloned())
            .collect::<Option<Vec<_>>>()
    })
}

pub(super) fn replace_recurrent_state_resident_buffer(
    old_id: kiln_tensor::TensorId,
    new_id: kiln_tensor::TensorId,
    buffer: Arc<kiln_vulkan_kernel::VulkanBuffer>,
) {
    RECURRENT_STATE_RESIDENT_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        cache.remove(&old_id);
        cache.insert(new_id, buffer);
    });
}
