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
