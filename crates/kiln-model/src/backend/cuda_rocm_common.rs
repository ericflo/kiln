//! Shared CUDA/ROCm backend helpers.
//!
//! Keep this module limited to contracts that are genuinely backend-neutral.
//! Device ownership, kernel dispatch, graph capture, and dispatch logging stay
//! in the concrete CUDA/ROCm modules.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use kiln_tensor_id::TensorId;

pub(crate) type ResidentTensorIdRegistry = OnceLock<Mutex<HashSet<TensorId>>>;

/// kt-native `kiln_tensor_id::TensorId` for CUDA/ROCm resident-activation
/// membership registries. The BackendRuntime trait surface is kt-typed, so the
/// shared CUDA/ROCm membership contract keys directly off the kt tensor id.
#[inline]
fn kt_tensor_id(tensor: &kiln_tensor::Tensor) -> TensorId {
    tensor.id()
}

fn with_resident_tensor_ids<R>(
    registry: &'static ResidentTensorIdRegistry,
    poison_message: &'static str,
    f: impl FnOnce(&mut HashSet<TensorId>) -> R,
) -> R {
    let registry = registry.get_or_init(|| Mutex::new(HashSet::new()));
    let mut guard = registry.lock().expect(poison_message);
    f(&mut guard)
}

pub(crate) fn mark_resident_activation(
    registry: &'static ResidentTensorIdRegistry,
    tensor: &kiln_tensor::Tensor,
    poison_message: &'static str,
) {
    with_resident_tensor_ids(registry, poison_message, |ids| {
        ids.insert(kt_tensor_id(tensor));
    });
}

pub(crate) fn evict_resident_activation(
    registry: &'static ResidentTensorIdRegistry,
    tensor: &kiln_tensor::Tensor,
    poison_message: &'static str,
) {
    with_resident_tensor_ids(registry, poison_message, |ids| {
        ids.remove(&kt_tensor_id(tensor));
    });
}

pub(crate) fn has_resident_activation(
    registry: &'static ResidentTensorIdRegistry,
    tensor: &kiln_tensor::Tensor,
    poison_message: &'static str,
) -> bool {
    with_resident_tensor_ids(registry, poison_message, |ids| {
        ids.contains(&kt_tensor_id(tensor))
    })
}

pub(crate) fn optimizer_tensors_supported_for_kt(
    tensors: &[&kiln_tensor::Tensor],
    device_matches: impl Fn(kiln_tensor::Device) -> bool,
) -> bool {
    let Some(first) = tensors.first() else {
        return false;
    };
    let dtype = first.dtype();
    let element_count = first.element_count();
    device_matches(first.device())
        && matches!(dtype, kiln_tensor::DType::F32 | kiln_tensor::DType::BF16)
        && first.is_contiguous()
        && tensors.iter().all(|tensor| {
            device_matches(tensor.device())
                && tensor.dtype() == dtype
                && tensor.element_count() == element_count
                && tensor.is_contiguous()
        })
}
