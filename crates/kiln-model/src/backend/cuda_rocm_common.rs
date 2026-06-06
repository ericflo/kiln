//! Shared CUDA/ROCm backend helpers.
//!
//! Keep this module limited to contracts that are genuinely backend-neutral.
//! Device ownership, kernel dispatch, graph capture, and logging stay in the
//! concrete CUDA/ROCm modules.

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
