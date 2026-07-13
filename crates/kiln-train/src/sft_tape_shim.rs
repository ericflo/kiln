//! kt-tape SFT loss adapters that live on trainer-side backend glue.
//!
//! CUDA/ROCm use `kiln-flce-kernel`'s kt FLCE API directly. Vulkan has its
//! own fused FLCE shaders over active rows and canonical `[vocab, hidden]`
//! tied weights, so this module bridges kt tensors to `VkTensor`, records a
//! kt-tape loss root, and scatters the fused active-row backward result back to
//! `[1, seq_len, hidden]`.

#[cfg(feature = "vulkan")]
use anyhow::{Context, Result};

#[cfg(feature = "vulkan")]
fn tensor_err(msg: impl Into<String>) -> kiln_tensor::Error {
    kiln_tensor::Error::Msg(msg.into())
}

#[cfg(feature = "vulkan")]
fn vulkan_device_index(t: &kiln_tensor::Tensor, context: &str) -> kiln_tensor::Result<usize> {
    match t.device() {
        kiln_tensor::Device::Vulkan(i) => Ok(i),
        other => Err(tensor_err(format!(
            "{context}: expected Vulkan tensor, got {other}"
        ))),
    }
}

#[cfg(feature = "vulkan")]
fn shifted_active_positions_and_labels(
    input_ids: &[u32],
    label_mask: &[bool],
    vocab: usize,
) -> kiln_tensor::Result<(Vec<u32>, Vec<u32>)> {
    if label_mask.len() != input_ids.len() {
        return Err(tensor_err(format!(
            "vulkan SFT FLCE: label_mask len {} != input_ids len {}",
            label_mask.len(),
            input_ids.len()
        )));
    }
    if input_ids.len() < 2 {
        return Ok((Vec::new(), Vec::new()));
    }

    let mut positions = Vec::new();
    let mut labels = Vec::new();
    for (shift_idx, &is_active) in label_mask[1..].iter().enumerate() {
        if !is_active {
            continue;
        }
        let label = input_ids[shift_idx + 1];
        if label as usize >= vocab {
            return Err(tensor_err(format!(
                "vulkan SFT FLCE: label {label} >= vocab {vocab}"
            )));
        }
        positions.push(u32::try_from(shift_idx).map_err(|_| {
            tensor_err(format!(
                "vulkan SFT FLCE: active position {shift_idx} exceeds u32"
            ))
        })?);
        labels.push(label);
    }
    Ok((positions, labels))
}

#[cfg(feature = "vulkan")]
fn prepare_vulkan_sft_active_hidden(
    hidden: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
) -> kiln_tensor::Result<(
    kiln_vulkan_kernel::vk_tensor::VkTensor,
    Vec<u32>,
    Vec<u32>,
    usize,
    usize,
    usize,
)> {
    let dims = hidden.dims();
    if dims.len() != 3 || dims[0] != 1 {
        return Err(tensor_err(format!(
            "vulkan SFT FLCE: hidden must be [1, seq_len, hidden], got {dims:?}"
        )));
    }
    let seq_len = dims[1];
    let hidden_size = dims[2];
    if hidden.dtype() != kiln_tensor::DType::F32 {
        return Err(tensor_err(format!(
            "vulkan SFT FLCE: hidden must be F32, got {}",
            hidden.dtype()
        )));
    }
    if weight.dims().len() != 2 || weight.dims()[1] != hidden_size {
        return Err(tensor_err(format!(
            "vulkan SFT FLCE: weight must be [vocab, hidden={hidden_size}], got {:?}",
            weight.dims()
        )));
    }
    if !matches!(
        weight.dtype(),
        kiln_tensor::DType::F32 | kiln_tensor::DType::BF16
    ) {
        return Err(tensor_err(format!(
            "vulkan SFT FLCE: weight must be F32/BF16, got {}",
            weight.dtype()
        )));
    }
    if input_ids.len() != seq_len {
        return Err(tensor_err(format!(
            "vulkan SFT FLCE: input_ids len {} != seq_len {seq_len}",
            input_ids.len()
        )));
    }

    let (active_positions, active_labels) =
        shifted_active_positions_and_labels(input_ids, label_mask, weight.dims()[0])?;
    if active_positions.is_empty() {
        return Err(tensor_err("vulkan SFT FLCE: no active shifted labels"));
    }

    let hidden_2d = hidden.squeeze(0).and_then(|t| {
        if t.is_contiguous() {
            Ok(t)
        } else {
            t.contiguous()
        }
    })?;
    let hidden_vk = kiln_tensor::vk_tensor_from_kt(&hidden_2d)
        .map_err(|e| tensor_err(format!("vulkan SFT FLCE: bridge hidden: {e}")))?;
    let active_hidden_vk = kiln_vulkan_kernel::vk_ops::index_select::vk_index_select_rows(
        &hidden_vk,
        &active_positions,
    )
    .map_err(|e| tensor_err(format!("vulkan SFT FLCE: gather active rows: {e}")))?;
    Ok((
        active_hidden_vk,
        active_positions,
        active_labels,
        seq_len,
        hidden_size,
        vulkan_device_index(hidden, "vulkan SFT FLCE hidden")?,
    ))
}

#[cfg(feature = "vulkan")]
pub(crate) fn vulkan_sft_flce_loss_kt(
    hidden: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
) -> kiln_tensor::Result<(
    kiln_tensor::Tensor,
    kiln_vulkan_kernel::vk_ops::flce::FlceSavedState,
    Vec<u32>,
    usize,
    usize,
)> {
    let (active_hidden_vk, active_positions, active_labels, seq_len, _hidden_size, device_index) =
        prepare_vulkan_sft_active_hidden(hidden, weight, input_ids, label_mask)?;
    let weight_kt = if weight.is_contiguous() {
        weight.clone()
    } else {
        weight.contiguous()?
    };
    let weight_vk = kiln_tensor::vk_tensor_from_kt(&weight_kt)
        .map_err(|e| tensor_err(format!("vulkan SFT FLCE: bridge weight: {e}")))?;
    let (loss_vk, saved) = kiln_vulkan_kernel::vk_ops::flce::vk_flce_loss_with_saved_state(
        &active_hidden_vk,
        &weight_vk,
        &active_labels,
        0,
    )
    .map_err(|e| tensor_err(format!("vulkan SFT FLCE fused loss: {e}")))?;
    let loss = kiln_tensor::kt_tensor_from_vk(&loss_vk, device_index)
        .map_err(|e| tensor_err(format!("vulkan SFT FLCE: bridge loss: {e}")))?;
    Ok((loss, saved, active_positions, seq_len, device_index))
}

#[cfg(feature = "vulkan")]
fn vulkan_sft_flce_grad_from_saved_kt(
    hidden: &kiln_tensor::Tensor,
    saved: &kiln_vulkan_kernel::vk_ops::flce::FlceSavedState,
    active_positions: &[u32],
    seq_len: usize,
    device_index: usize,
    grad_loss: &kiln_tensor::Tensor,
) -> kiln_tensor::Result<kiln_tensor::Tensor> {
    let hidden_2d = hidden.squeeze(0).and_then(|t| {
        if t.is_contiguous() {
            Ok(t)
        } else {
            t.contiguous()
        }
    })?;
    let hidden_vk = kiln_tensor::vk_tensor_from_kt(&hidden_2d)
        .map_err(|e| tensor_err(format!("vulkan SFT FLCE grad: bridge hidden: {e}")))?;
    let active_hidden_vk = kiln_vulkan_kernel::vk_ops::index_select::vk_index_select_rows(
        &hidden_vk,
        active_positions,
    )
    .map_err(|e| tensor_err(format!("vulkan SFT FLCE grad: gather active rows: {e}")))?;
    let grad_loss_kt = if grad_loss.is_contiguous() {
        grad_loss.clone()
    } else {
        grad_loss.contiguous()?
    };
    let grad_loss_vk = kiln_tensor::vk_tensor_from_kt(&grad_loss_kt)
        .map_err(|e| tensor_err(format!("vulkan SFT FLCE grad: bridge grad_loss: {e}")))?;
    let grad_active_vk = kiln_vulkan_kernel::vk_ops::flce::vk_flce_backward_with_saved_state(
        &active_hidden_vk,
        saved,
        &grad_loss_vk,
    )
    .map_err(|e| tensor_err(format!("vulkan SFT FLCE fused backward: {e}")))?;
    let grad_full_vk = kiln_vulkan_kernel::vk_ops::index_select::vk_scatter_rows_to_full(
        &grad_active_vk,
        active_positions,
        seq_len,
    )
    .map_err(|e| tensor_err(format!("vulkan SFT FLCE grad: scatter active rows: {e}")))?;
    let grad_full = kiln_tensor::kt_tensor_from_vk(&grad_full_vk, device_index)
        .map_err(|e| tensor_err(format!("vulkan SFT FLCE grad: bridge full grad: {e}")))?;
    grad_full
        .unsqueeze(0)
        .map_err(|e| tensor_err(format!("vulkan SFT FLCE grad: unsqueeze: {e}")))
}

#[cfg(feature = "vulkan")]
pub(crate) fn vulkan_sft_flce_loss_and_grad_kt(
    hidden: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
) -> kiln_tensor::Result<(kiln_tensor::Tensor, kiln_tensor::Tensor)> {
    let (loss, saved, active_positions, seq_len, device_index) =
        vulkan_sft_flce_loss_kt(hidden, weight, input_ids, label_mask)?;
    let grad_seed = kiln_tensor::Tensor::from_vec_on(hidden.device(), vec![1.0f32], vec![1])
        .map_err(|e| tensor_err(format!("vulkan SFT FLCE: grad seed: {e}")))?;
    let grad = vulkan_sft_flce_grad_from_saved_kt(
        hidden,
        &saved,
        &active_positions,
        seq_len,
        device_index,
        &grad_seed,
    )?;
    Ok((loss, grad))
}

#[cfg(feature = "vulkan")]
#[derive(Debug)]
struct VulkanSftFlceBackward {
    hidden: kiln_tensor::Tensor,
    saved: kiln_vulkan_kernel::vk_ops::flce::FlceSavedState,
    active_positions: Vec<u32>,
    seq_len: usize,
    device_index: usize,
}

#[cfg(feature = "vulkan")]
impl kiln_autograd::BackwardOp for VulkanSftFlceBackward {
    fn name(&self) -> &'static str {
        "vulkan_sft_flce_backward"
    }

    fn input_count(&self) -> usize {
        1
    }

    fn requires_input(&self, _idx: usize) -> bool {
        false
    }

    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        let grad_hidden = vulkan_sft_flce_grad_from_saved_kt(
            &self.hidden,
            &self.saved,
            &self.active_positions,
            self.seq_len,
            self.device_index,
            grad_output,
        )?;
        Ok(vec![Some(grad_hidden)])
    }
}

#[cfg(feature = "vulkan")]
pub fn try_tape_sft_flce_vulkan_kt(
    hidden: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
) -> Result<Option<kiln_tensor::Tensor>> {
    use kiln_autograd::{Tape, tape_scope_active, with_active_tape};

    if !tape_scope_active() {
        return Ok(None);
    }
    if !matches!(hidden.device(), kiln_tensor::Device::Vulkan(_))
        || !matches!(weight.device(), kiln_tensor::Device::Vulkan(_))
    {
        return Ok(None);
    }
    if input_ids.len() < 2 || !label_mask.get(1..).is_some_and(|m| m.iter().any(|&v| v)) {
        return Ok(None);
    }

    let (loss_kt, saved, active_positions, seq_len, device_index) =
        vulkan_sft_flce_loss_kt(hidden, weight, input_ids, label_mask)
            .context("vulkan SFT FLCE scalar kt loss")?;
    let loss_kt = match with_active_tape(|tape: &mut Tape| -> Result<_> {
        tape.record(
            &loss_kt,
            &[hidden],
            Box::new(VulkanSftFlceBackward {
                hidden: hidden.clone(),
                saved,
                active_positions,
                seq_len,
                device_index,
            }),
        );
        Ok(loss_kt)
    }) {
        Some(result) => result?,
        None => return Ok(None),
    };

    Ok(Some(loss_kt))
}

#[cfg(all(test, feature = "vulkan"))]
mod tests {
    use super::shifted_active_positions_and_labels;

    #[test]
    fn shifted_active_positions_follow_next_token_labels() {
        let input_ids = [10, 20, 30, 40];
        let label_mask = [false, true, false, true];
        let (positions, labels) =
            shifted_active_positions_and_labels(&input_ids, &label_mask, 64).unwrap();
        assert_eq!(positions, vec![0, 2]);
        assert_eq!(labels, vec![20, 40]);
    }
}
