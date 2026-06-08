//! Vulkan resident activation registry storage.
//!
//! `ResidencyBackend` methods on `VulkanBackend` use this module's
//! backend-owned TensorId -> VulkanBuffer maps for optimizer dispatch and
//! recurrent-state residency.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub(super) type ResidentActivationRegistry =
    Mutex<HashMap<kiln_tensor::TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>>;

pub(super) fn new_resident_activation_registry() -> ResidentActivationRegistry {
    Mutex::new(HashMap::new())
}

thread_local! {
    static RECURRENT_STATE_RESIDENT_SCOPE_DEPTH: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static RECURRENT_STATE_RESIDENT_CACHE:
        std::cell::RefCell<HashMap<kiln_tensor::TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>> =
        std::cell::RefCell::new(HashMap::new());
}

/// Helper: short, self-recovering accessor that wraps the registry's mutex.
/// Poison recovery returns the inner data so the registry never stays
/// inaccessible just because some panicking code touched it.
pub(super) fn with_resident_registry<F, R>(registry: &ResidentActivationRegistry, f: F) -> R
where
    F: FnOnce(&mut HashMap<kiln_tensor::TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>) -> R,
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
