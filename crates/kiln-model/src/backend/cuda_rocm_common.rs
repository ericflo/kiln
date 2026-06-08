//! Shared CUDA/ROCm backend helpers.
//!
//! Keep this module limited to contracts that are genuinely backend-neutral.
//! Device ownership, kernel dispatch, graph capture, and dispatch logging stay
//! in the concrete CUDA/ROCm modules.

use std::collections::HashMap;
use std::sync::Mutex;

use kiln_tensor_id::TensorId;

pub(crate) type ResidentTensorIdRegistry =
    Mutex<HashMap<TensorId, super::residency::ResidentResource>>;

pub(crate) fn new_resident_tensor_id_registry() -> ResidentTensorIdRegistry {
    Mutex::new(HashMap::new())
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CudaRocmSupportPredicates {
    pub(crate) gdn_enabled: bool,
    pub(crate) gdn_gates_enabled: bool,
    pub(crate) gdn_gated_rms_norm_enabled: bool,
    pub(crate) gdn_decode_unexpanded_qk_enabled: bool,
    pub(crate) gdn_decode_qk_norm_recurrent_enabled: bool,
    pub(crate) fused_conv1d_enabled: bool,
}

impl CudaRocmSupportPredicates {
    pub(crate) fn supports_flash_attn_prefill(self) -> bool {
        true
    }

    pub(crate) fn supports_flash_attn_paged_decode(self) -> bool {
        true
    }

    pub(crate) fn supports_strict_paged_decode_contiguous_batch(self) -> bool {
        false
    }

    pub(crate) fn supports_gdn_forward_substitution(self) -> bool {
        self.gdn_enabled
    }

    pub(crate) fn supports_gdn_recurrent_step(self) -> bool {
        self.gdn_enabled
    }

    pub(crate) fn supports_gdn_chunk_prep(self) -> bool {
        self.gdn_enabled
    }

    pub(crate) fn supports_gdn_chunk_scan(self) -> bool {
        self.gdn_enabled
    }

    pub(crate) fn supports_gdn_full_chunk_forward(self) -> bool {
        self.gdn_enabled
    }

    pub(crate) fn supports_gdn_decode_gates_recurrent_unexpanded_qk(self) -> bool {
        self.gdn_decode_unexpanded_qk_enabled
    }

    pub(crate) fn supports_gdn_decode_qk_norm_gates_recurrent(self) -> bool {
        self.gdn_decode_qk_norm_recurrent_enabled
    }

    pub(crate) fn supports_gdn_gates(self) -> bool {
        self.gdn_gates_enabled
    }

    pub(crate) fn supports_gdn_gated_rms_norm(self) -> bool {
        self.gdn_gated_rms_norm_enabled
    }

    pub(crate) fn supports_causal_conv1d_update(self) -> bool {
        self.fused_conv1d_enabled
    }

    pub(crate) fn supports_causal_conv1d_prefill(self) -> bool {
        self.fused_conv1d_enabled
    }
}

/// kt-native `kiln_tensor_id::TensorId` for CUDA/ROCm resident-activation
/// membership registries. The BackendRuntime trait surface is kt-typed, so the
/// shared CUDA/ROCm membership contract keys directly off the kt tensor id.
#[inline]
fn kt_tensor_id(tensor: &kiln_tensor::Tensor) -> TensorId {
    tensor.id()
}

fn with_resident_tensor_registry<R>(
    registry: &ResidentTensorIdRegistry,
    poison_message: &'static str,
    f: impl FnOnce(&mut HashMap<TensorId, super::residency::ResidentResource>) -> R,
) -> R {
    let mut guard = registry.lock().expect(poison_message);
    f(&mut guard)
}

pub(crate) fn mark_resident_activation(
    registry: &ResidentTensorIdRegistry,
    tensor: &kiln_tensor::Tensor,
    resource: super::residency::ResidentResource,
    poison_message: &'static str,
) -> super::residency::ResidentResource {
    with_resident_tensor_registry(registry, poison_message, |resources| {
        resources.insert(kt_tensor_id(tensor), resource.clone());
    });
    resource
}

pub(crate) fn evict_resident_activation(
    registry: &ResidentTensorIdRegistry,
    tensor: &kiln_tensor::Tensor,
    poison_message: &'static str,
) {
    with_resident_tensor_registry(registry, poison_message, |resources| {
        resources.remove(&kt_tensor_id(tensor));
    });
}

pub(crate) fn has_resident_activation(
    registry: &ResidentTensorIdRegistry,
    tensor: &kiln_tensor::Tensor,
    poison_message: &'static str,
) -> bool {
    with_resident_tensor_registry(registry, poison_message, |resources| {
        resources.contains_key(&kt_tensor_id(tensor))
    })
}

pub(crate) fn resident_activation_resource(
    registry: &ResidentTensorIdRegistry,
    tensor: &kiln_tensor::Tensor,
    poison_message: &'static str,
) -> Option<super::residency::ResidentResource> {
    with_resident_tensor_registry(registry, poison_message, |resources| {
        resources.get(&kt_tensor_id(tensor)).cloned()
    })
}

pub(crate) fn optimizer_tensors_supported_for_kt(
    tensors: &[&kiln_tensor::Tensor],
    device_matches: impl Fn(kiln_tensor::Device) -> bool + Copy,
) -> bool {
    let Some(first) = tensors.first() else {
        return false;
    };
    let dtype = first.dtype();
    let element_count = first.element_count();
    tensors_on_backend_device(tensors, device_matches)
        && matches!(dtype, kiln_tensor::DType::F32 | kiln_tensor::DType::BF16)
        && first.is_contiguous()
        && tensors.iter().all(|tensor| {
            tensor.dtype() == dtype
                && tensor.element_count() == element_count
                && tensor.is_contiguous()
        })
}

pub(crate) fn tensors_on_backend_device(
    tensors: &[&kiln_tensor::Tensor],
    device_matches: impl Fn(kiln_tensor::Device) -> bool,
) -> bool {
    tensors.iter().all(|tensor| device_matches(tensor.device()))
}

pub(crate) fn optimizer_args_ready_for_kt(
    registry: &ResidentTensorIdRegistry,
    tensors: &[&kiln_tensor::Tensor],
    poison_message: &'static str,
    device_matches: impl Fn(kiln_tensor::Device) -> bool + Copy,
) -> bool {
    tensors
        .iter()
        .all(|tensor| has_resident_activation(registry, tensor, poison_message))
        && optimizer_tensors_supported_for_kt(tensors, device_matches)
        && kiln_rmsnorm_kernel::supports_optimizer_step_kt(tensors)
}
