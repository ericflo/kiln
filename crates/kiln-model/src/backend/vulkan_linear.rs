//! Vulkan linear/lm-head operation helpers.
//!
//! `backend/vulkan.rs` owns the `BackendRuntime` facade. This module keeps the
//! per-submit safety policy and cached-weight decode dispatches next to the
//! linear concern without hiding the explicit VkTensor/buffer dispatch boundary.

use anyhow::{Context, Result};
use std::sync::OnceLock;

use super::vulkan::VulkanBackend;
use super::vulkan_tensor_bridge::{kt_tensor_from_f32_bytes, kt_tensor_to_f32_bytes_with_shape};
use super::{BackendMatmulLayout, requested_matmul_layout};

/// FLOP estimate for `[batch, hidden] @ [hidden, out_dim]` (one mul + one
/// add per inner term).
fn matmul_flop(batch: usize, hidden: usize, out_dim: usize) -> u64 {
    (batch as u64)
        .saturating_mul(hidden as u64)
        .saturating_mul(out_dim as u64)
        .saturating_mul(2)
}

/// (#1082) Per-dispatch FLOP ceiling for the Vulkan-routed matmul.
///
/// Migrated inline from the deleted `backend::vulkan_linear_op` module
/// (its `candle_core::CustomOp1` training wrapper was removed when the kt
/// autograd tape became the sole grad producer). The forward-only FLCE
/// offset path in `linear_prefill_apply_offset` still needs the ceiling to
/// sub-chunk oversized dispatches. Multi-million-workgroup submissions can
/// hang Vulkan implementations, so the immutable policy owns the exact limit
/// and malformed or late process state cannot disable it.
fn max_flop_per_dispatch() -> u64 {
    kiln_vulkan_kernel::kernels::vulkan_kernel_policy().linear_max_flop_per_dispatch
}

/// True when the requested matmul shape would exceed the per-dispatch FLOP
/// ceiling; the caller sub-chunks via [`max_chunk_dim_for_flop`].
pub(super) fn dispatch_exceeds_safety_ceiling(batch: usize, hidden: usize, out_dim: usize) -> bool {
    matmul_flop(batch, hidden, out_dim) > max_flop_per_dispatch()
}

/// Largest `chunk_dim` such that `2 x other_dim_product x chunk_dim <=
/// max_flop_per_dispatch()`. Always >= 1; returns `usize::MAX` when the
/// guard is disabled.
pub(super) fn max_chunk_dim_for_flop(other_dim_product: usize) -> usize {
    let max_flop = max_flop_per_dispatch();
    if max_flop == u64::MAX {
        return usize::MAX;
    }
    let denom = (other_dim_product as u64).saturating_mul(2).max(1);
    let chunk = (max_flop / denom) as usize;
    chunk.max(1)
}

/// Request-only contract for the resident mixed projection route below.
///
/// `NativeWithConstraints` for this shape means the runtime operands must both
/// be resident on the same Vulkan companion device. The backend-private
/// CPU-staging and packed-weight-cache paths are intentionally excluded: they
/// may make additional requests work at runtime, but they are not a resident
/// native-matmul capability.
pub(super) fn resident_mixed_rank2_request_supported(
    req: &super::capability::MatmulRequest,
) -> bool {
    req.rank() == Some(2)
        && req.to_blas_request(1).is_ok()
        && matches!(req.logical_mnk(), Some((m, n, k)) if m > 0 && n > 0 && k > 0)
        && req.lhs_layout == super::capability::MatmulOperandLayout::RowMajor
        && req.rhs_layout == super::capability::MatmulOperandLayout::RowMajor
        && req.out_layout == super::capability::MatmulOperandLayout::RowMajor
        && req.epilogue == super::capability::MatmulEpilogue::Identity
        && req.lhs_dtype == kiln_tensor::DType::F32
        && req.rhs_dtype == kiln_tensor::DType::BF16
        && req.out_dtype == kiln_tensor::DType::F32
}

pub(super) fn matmul_request_support(
    req: &super::capability::MatmulRequest,
) -> super::capability::Support {
    if resident_mixed_rank2_request_supported(req) {
        return super::capability::Support::NativeWithConstraints;
    }
    let Some(rank) = super::matmul_request_support_rank(req) else {
        return super::capability::Support::Unsupported;
    };
    super::matmul_support_from_native(
        matches!(req.epilogue, super::capability::MatmulEpilogue::Identity)
            && (req.lhs_dtype == kiln_tensor::DType::F32
                || req.lhs_dtype == kiln_tensor::DType::BF16 && rank > 2),
    )
}

pub(super) fn matmul(
    backend: &VulkanBackend,
    req: &super::capability::MatmulRequest,
    lhs: &kiln_tensor::Tensor,
    rhs: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    let Some(layout) = requested_matmul_layout(req, lhs, rhs) else {
        return Ok(None);
    };

    if matches!(lhs.device(), kiln_tensor::Device::Vulkan(_))
        && matches!(rhs.device(), kiln_tensor::Device::Vulkan(_))
    {
        if resident_mixed_rank2_request_supported(req) {
            debug_assert_eq!(layout, BackendMatmulLayout::Plain);
            let (rows, hidden) = lhs.dims2()?;
            let (_, out_dim) = rhs.dims2()?;
            let lhs_rank3 = lhs.reshape((1usize, rows, hidden))?;
            let Some(out_rank3) = resident_linear_decode(&lhs_rank3, rhs)? else {
                return Ok(None);
            };
            return Ok(Some(out_rank3.reshape((rows, out_dim))?));
        }
        return resident_matmul(req, lhs, rhs, layout);
    }

    if layout == BackendMatmulLayout::Plain
        && matches!(lhs.device(), kiln_tensor::Device::Cpu)
        && matches!(rhs.device(), kiln_tensor::Device::Cpu)
        && lhs.is_contiguous()
        && rhs.is_contiguous()
        && lhs.dtype() == kiln_tensor::DType::F32
        && matches!(
            rhs.dtype(),
            kiln_tensor::DType::F32 | kiln_tensor::DType::BF16
        )
        && req.out_dtype == kiln_tensor::DType::F32
    {
        return cached_linear_matmul(backend, lhs, rhs);
    }

    Ok(None)
}

fn resident_matmul(
    req: &super::capability::MatmulRequest,
    lhs: &kiln_tensor::Tensor,
    rhs: &kiln_tensor::Tensor,
    layout: BackendMatmulLayout,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !lhs.is_contiguous()
        || !rhs.is_contiguous()
        || req.out_dtype != lhs.dtype()
        || req.lhs_dtype != req.rhs_dtype
        || !matches!(
            lhs.dtype(),
            kiln_tensor::DType::F32 | kiln_tensor::DType::BF16
        )
    {
        return Ok(None);
    }

    let out = match layout {
        BackendMatmulLayout::Plain => {
            if lhs.rank() == 2 {
                if lhs.dtype() != kiln_tensor::DType::F32 {
                    return Ok(None);
                }
                kiln_tensor::vulkan_matmul(lhs, rhs)?
            } else {
                kiln_tensor::vulkan_matmul_batched(lhs, rhs)?
            }
        }
        BackendMatmulLayout::LhsTransposed => kiln_tensor::vulkan_matmul_lhs_transposed(lhs, rhs)?,
        BackendMatmulLayout::RhsTransposed => kiln_tensor::vulkan_matmul_rhs_transposed(lhs, rhs)?,
        BackendMatmulLayout::BothTransposed => {
            let rank = lhs.rank();
            let lhs_t = lhs.transpose(rank - 2, rank - 1)?.contiguous()?;
            let rhs_t = rhs.transpose(rank - 2, rank - 1)?.contiguous()?;
            if lhs_t.rank() == 2 {
                if lhs_t.dtype() != kiln_tensor::DType::F32 {
                    return Ok(None);
                }
                kiln_tensor::vulkan_matmul(&lhs_t, &rhs_t)?
            } else {
                kiln_tensor::vulkan_matmul_batched(&lhs_t, &rhs_t)?
            }
        }
    };
    Ok(Some(out))
}

fn cached_linear_matmul(
    backend: &VulkanBackend,
    lhs: &kiln_tensor::Tensor,
    rhs: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    if lhs.rank() < 2 || rhs.dims().len() != 2 {
        return Ok(None);
    }
    let l_dims = lhs.dims().to_vec();
    let hidden = *l_dims.last().unwrap();
    let Ok((weight_hidden, out_dim)) = rhs.dims2() else {
        return Ok(None);
    };
    if weight_hidden != hidden {
        return Ok(None);
    }

    let lead = l_dims[..l_dims.len() - 1].iter().product::<usize>();
    let dispatch_x = if l_dims.len() == 3 {
        lhs.clone()
    } else {
        lhs.reshape((lead, 1usize, hidden))?
    };
    let Some(out) = linear_decode(backend, &dispatch_x, rhs)? else {
        return Ok(None);
    };

    let mut out_shape = l_dims[..l_dims.len() - 1].to_vec();
    out_shape.push(out_dim);
    Ok(Some(out.reshape(out_shape.as_slice())?))
}

pub(super) fn linear_decode(
    backend: &VulkanBackend,
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    // kt guards read directly off the kt args before the bridge.
    if !backend.has_vulkan()
        || !backend.linear_decode_enabled
        || x.dtype() != kiln_tensor::DType::F32
    {
        return Ok(None);
    }
    // Layer-bounded prefill selects the final hidden row before the LM-head
    // projection, so inference can arrive here as `[rows, hidden]` even though
    // the resident kernel's canonical interface is
    // `[batch, sequence, hidden]`. Normalize every supported leading shape
    // through that same kernel and restore it afterward. The helper recurses
    // only with its canonical rank-3 dispatch tensor.
    if x.rank() != 3 {
        return cached_linear_matmul(backend, x, weight_t);
    }
    if matches!(x.device(), kiln_tensor::Device::Vulkan(_))
        && matches!(weight_t.device(), kiln_tensor::Device::Vulkan(_))
    {
        return resident_linear_decode(x, weight_t);
    }
    if matches!(x.device(), kiln_tensor::Device::Cpu)
        && matches!(weight_t.device(), kiln_tensor::Device::Vulkan(_))
    {
        let resident_x = x
            .to_device(weight_t.device())
            .context("upload linear activation to resident Vulkan weight device")?;
        return resident_linear_decode(&resident_x, weight_t);
    }
    if matches!(x.device(), kiln_tensor::Device::Vulkan(_))
        && matches!(weight_t.device(), kiln_tensor::Device::Cpu)
    {
        // Model loading may keep the immutable kt weight on CPU while Vulkan's
        // private decode cache owns its packed device copy. kt Vulkan tensors
        // and that private cache do not necessarily share one logical device,
        // so never bind their raw buffers in the same dispatch. Read back only
        // the small activation, then reuse the already-prewarmed cached weight.
        let host_x = x
            .to_device(kiln_tensor::Device::Cpu)
            .context("read back resident linear activation for cached Vulkan weight")?;
        return linear_decode(backend, &host_x, weight_t);
    }
    if !matches!(x.device(), kiln_tensor::Device::Cpu)
        || !matches!(weight_t.device(), kiln_tensor::Device::Cpu)
    {
        return Ok(None);
    }
    // (#1082) Fully kt-native: read shapes off the kt tensors, extract
    // f32 bytes straight from kt storage, and key the weight buffer cache
    // on the **stable** kt `TensorId`. The old path bridged BOTH x and the
    // (large) weight through `kt_logits_to_candle` every call -- minting a
    // fresh candle id per token so the weight cache missed every step and
    // re-uploaded ~1 GB/token. Now the weight uploads exactly once.
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return Ok(None);
    };
    let Ok((weight_hidden, out_dim)) = weight_t.dims2() else {
        return Ok(None);
    };
    if weight_hidden != hidden {
        return Ok(None);
    }

    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
    let row_count = batch * seq_len;
    // x is [batch, seq_len, hidden] contiguous F32; the kernel consumes a
    // flat [row_count, hidden] f32 buffer, so the [.,1,.] reshape the candle
    // path did is a no-op on the bytes -- extract them straight from kt.
    let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
    let packed = backend.use_bf16_packed_linear_weight_kt(weight_t);
    let weight_buf = if packed {
        backend.cached_bf16_packed_weight_buffer_kt(weight_t)?
    } else {
        backend.cached_f32_weight_buffer_kt(weight_t)?
    };
    let out_data = kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached_bytes(
        vk_device,
        &x_data,
        &weight_buf,
        row_count,
        hidden,
        out_dim,
        packed,
    )
    .context("linear_decode kernel failed")?;
    Ok(Some(kt_tensor_from_f32_bytes(
        &out_data,
        &[batch, seq_len, out_dim],
        kiln_tensor::DType::F32,
    )?))
}

fn resident_linear_shape_supported(
    batch: usize,
    seq_len: usize,
    hidden: usize,
    out_dim: usize,
) -> bool {
    batch > 0 && seq_len > 0 && hidden > 0 && out_dim > 0
}

fn resident_linear_decode(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    use kiln_tensor::{DType, Device, kt_tensor_from_vk, vk_tensor_from_kt};
    use kiln_vulkan_kernel::vk_tensor::{VkDType, VkTensor};

    if x.dtype() != DType::F32
        || !matches!(weight_t.dtype(), DType::F32 | DType::BF16)
        || x.device() != weight_t.device()
    {
        return Ok(None);
    }
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return Ok(None);
    };
    let Ok((weight_hidden, out_dim)) = weight_t.dims2() else {
        return Ok(None);
    };
    if weight_hidden != hidden || !resident_linear_shape_supported(batch, seq_len, hidden, out_dim)
    {
        return Ok(None);
    }
    let Device::Vulkan(device_index) = x.device() else {
        return Ok(None);
    };
    let rows = batch
        .checked_mul(seq_len)
        .context("resident linear row-count overflow")?;
    let whole_contiguous = |tensor: &kiln_tensor::Tensor, name: &str| -> Result<_> {
        if tensor.is_contiguous() && tensor.layout().start_offset() == 0 {
            Ok(tensor.clone())
        } else {
            kiln_tensor::vulkan_contiguous(tensor)
                .with_context(|| format!("resident linear {name} whole-buffer contiguous"))
        }
    };
    let x_2d = whole_contiguous(x, "activation")?
        .reshape((rows, hidden))
        .context("resident linear activation flatten")?;
    let weight_t = whole_contiguous(weight_t, "weight")?;
    let vk_weight = vk_tensor_from_kt(&weight_t).context("resident linear weight bridge")?;
    let dispatch_rows = |x_rows: &kiln_tensor::Tensor| -> Result<kiln_tensor::Tensor> {
        let row_count = x_rows.dims2()?.0;
        let x_rows = whole_contiguous(x_rows, "row chunk")?;
        let vk_x = vk_tensor_from_kt(&x_rows).context("resident linear activation bridge")?;
        anyhow::ensure!(
            std::sync::Arc::ptr_eq(vk_x.device(), vk_weight.device()),
            "resident linear activation and weight use different Vulkan logical devices"
        );
        let out_buffer = kiln_vulkan_kernel::buffer_pool::pool_alloc_f32(
            vk_x.device(),
            row_count
                .checked_mul(out_dim)
                .context("resident linear output-size overflow")?,
        )
        .context("resident linear output allocation")?;
        if weight_t.dtype() == DType::BF16 {
            kiln_vulkan_kernel::resident::dispatch_linear_decode_cached_bf16_weights_resident(
                vk_x.device(),
                vk_x.buffer(),
                vk_weight.buffer(),
                &out_buffer,
                row_count,
                hidden,
                out_dim,
            )
            .context("resident BF16-weight linear dispatch")?;
        } else {
            kiln_vulkan_kernel::resident::dispatch_linear_decode_cached_resident(
                vk_x.device(),
                vk_x.buffer(),
                vk_weight.buffer(),
                &out_buffer,
                row_count,
                hidden,
                out_dim,
            )
            .context("resident F32-weight linear dispatch")?;
        }
        let vk_out = VkTensor::from_buffer(
            out_buffer,
            vec![row_count, out_dim],
            VkDType::F32,
            std::sync::Arc::clone(vk_x.device()),
        );
        kt_tensor_from_vk(&vk_out, device_index).context("resident linear output bridge")
    };

    let max_rows = max_chunk_dim_for_flop(hidden.saturating_mul(out_dim)).min(rows);
    let out_2d = if rows <= max_rows {
        dispatch_rows(&x_2d)?
    } else {
        let mut chunks = Vec::with_capacity(rows.div_ceil(max_rows));
        for row_start in (0..rows).step_by(max_rows) {
            let row_count = (rows - row_start).min(max_rows);
            let x_rows = x_2d
                .narrow(0, row_start, row_count)
                .context("resident linear row chunk")?;
            chunks.push(dispatch_rows(&x_rows)?);
        }
        let refs: Vec<&kiln_tensor::Tensor> = chunks.iter().collect();
        kiln_tensor::Tensor::cat(&refs, 0).context("resident linear output chunks concatenate")?
    };
    let out = out_2d
        .reshape((batch, seq_len, out_dim))
        .context("resident linear output reshape")?;
    Ok(Some(out))
}

pub(super) fn linear_prefill_apply(
    _backend: &VulkanBackend,
    _x: &kiln_tensor::Tensor,
    _weight_t: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    // (#1082) Decline. This hook previously routed the training-time
    // projection matmul through `VulkanLinearOp` (a
    // `candle_core::CustomOp1`) so candle's `loss.backward()` could
    // produce the input gradient. With the kt autograd tape
    // (`kiln_autograd`) as the sole grad producer that candle autograd
    // island is gone -- the projection matmul is recorded onto the tape
    // by the portable kt matmul path in forward.rs, and
    // `Tape::backward()` produces the gradient. Returning `Ok(None)`
    // routes the caller to that kt-recorded path.
    //
    // NOTE: the forward-only inference linear kernel still lives in
    // `linear_decode` (declines tracked tensors); only the
    // autograd-wrapping prefill path is removed here.
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{
        LinearBackend,
        capability::{MatmulEpilogue, MatmulOperandLayout, MatmulRequest, Support},
    };
    use kiln_tensor::{DType, Device, Tensor};

    #[test]
    fn resident_mixed_linear_declines_zero_dimensions_before_dispatch() {
        assert!(resident_linear_shape_supported(1, 2, 4, 3));
        assert!(!resident_linear_shape_supported(0, 2, 4, 3));
        assert!(!resident_linear_shape_supported(1, 0, 4, 3));
        assert!(!resident_linear_shape_supported(1, 2, 0, 3));
        assert!(!resident_linear_shape_supported(1, 2, 4, 0));
    }

    fn mixed_rank2_request() -> MatmulRequest {
        MatmulRequest::plain(vec![2, 4], vec![4, 3], DType::F32, false).with_dtypes(
            DType::F32,
            DType::BF16,
            DType::F32,
        )
    }

    #[test]
    fn mixed_resident_rank2_matmul_capability_is_exact_and_preserves_homogeneous_routes() {
        let supported = mixed_rank2_request();
        assert!(resident_mixed_rank2_request_supported(&supported));
        assert_eq!(
            matmul_request_support(&supported),
            Support::NativeWithConstraints
        );

        let mixed_batched = MatmulRequest::plain(vec![2, 2, 4], vec![2, 4, 3], DType::F32, false)
            .with_dtypes(DType::F32, DType::BF16, DType::F32);
        let mixed_transposed = MatmulRequest::plain(vec![2, 4], vec![3, 4], DType::F32, false)
            .with_dtypes(DType::F32, DType::BF16, DType::F32)
            .with_layouts(
                MatmulOperandLayout::RowMajor,
                MatmulOperandLayout::ColMajor,
                MatmulOperandLayout::RowMajor,
            );
        let mixed_bias = mixed_rank2_request().with_epilogue(MatmulEpilogue::Bias);
        let mixed_wrong_output =
            mixed_rank2_request().with_dtypes(DType::F32, DType::BF16, DType::BF16);
        let mixed_wrong_weight =
            mixed_rank2_request().with_dtypes(DType::F32, DType::F16, DType::F32);
        let mixed_zero_rows = MatmulRequest::plain(vec![0, 4], vec![4, 3], DType::F32, false)
            .with_dtypes(DType::F32, DType::BF16, DType::F32);
        let mixed_incompatible = MatmulRequest::plain(vec![2, 4], vec![5, 3], DType::F32, false)
            .with_dtypes(DType::F32, DType::BF16, DType::F32);

        for unsupported in [
            mixed_batched,
            mixed_transposed,
            mixed_bias,
            mixed_wrong_output,
            mixed_wrong_weight,
            mixed_zero_rows,
            mixed_incompatible,
        ] {
            assert!(!resident_mixed_rank2_request_supported(&unsupported));
            assert_eq!(matmul_request_support(&unsupported), Support::Unsupported);
        }

        let homogeneous_f32 = MatmulRequest::plain(vec![2, 4], vec![4, 3], DType::F32, false);
        let homogeneous_bf16_rank2 =
            MatmulRequest::plain(vec![2, 4], vec![4, 3], DType::BF16, false);
        let homogeneous_bf16_batched =
            MatmulRequest::plain(vec![2, 2, 4], vec![2, 4, 3], DType::BF16, false);
        assert_eq!(
            matmul_request_support(&homogeneous_f32),
            Support::NativeWithConstraints
        );
        assert_eq!(
            matmul_request_support(&homogeneous_bf16_rank2),
            Support::HostFallbackAllowed
        );
        assert_eq!(
            matmul_request_support(&homogeneous_bf16_batched),
            Support::NativeWithConstraints
        );
    }

    fn mixed_linear_fixture() -> (Tensor, Tensor) {
        let x = Tensor::from_vec(
            vec![0.25_f32, -0.5, 0.75, 1.0, -0.25, 0.5, -0.75, -1.0],
            (1, 2, 4),
        )
        .expect("F32 activation fixture");
        let weight = Tensor::from_vec(
            vec![
                0.5_f32, -0.25, 0.125, -0.5, 0.75, 0.25, 1.0, -0.125, 0.375, -0.75, 0.625, 0.5,
            ],
            (4, 3),
        )
        .expect("F32 weight fixture")
        .to_dtype(DType::BF16)
        .expect("BF16 weight fixture");
        (x, weight)
    }

    fn assert_fixture_values(output: &Tensor, reversed: bool) {
        let positive = [0.375_f32, 0.09375, 0.6875];
        let negative = [-0.375_f32, -0.09375, -0.6875];
        let expected = if reversed {
            [negative, positive].concat()
        } else {
            [positive, negative].concat()
        };
        let actual = output
            .to_device(Device::Cpu)
            .expect("fixture output readback")
            .flatten_all()
            .expect("fixture output flatten")
            .to_vec1::<f32>()
            .expect("fixture output values");
        assert_eq!(actual, expected);
    }

    #[test]
    fn mixed_f32_bf16_linear_uses_runner_and_resident_backends() {
        if std::env::var("KILN_TENSOR_VULKAN_TEST").ok().as_deref() != Some("1") {
            return;
        }
        assert!(
            crate::backend::vulkan::vulkan_is_available(),
            "KILN_TENSOR_VULKAN_TEST=1 requires a working Vulkan device"
        );
        let backend = VulkanBackend::new(Device::Cpu);

        // Hybrid CPU-storage route used by legacy Vulkan loading.
        let (x, weight) = mixed_linear_fixture();
        let output = LinearBackend::runtime_linear_decode(&backend, &x, &weight)
            .expect("mixed Vulkan linear dispatch")
            .expect("Vulkan backend must own F32 activation x BF16 weight");
        assert_eq!(output.dims(), &[1, 2, 3]);
        assert_eq!(output.dtype(), DType::F32);
        assert_fixture_values(&output, false);

        // The flattened matmul facade must preserve the same mixed contract.
        let (x, weight) = mixed_linear_fixture();
        let x = x.reshape((2, 4)).expect("flatten activation rows");
        let request =
            MatmulRequest::plain(x.dims().to_vec(), weight.dims().to_vec(), DType::F32, false)
                .with_dtypes(DType::F32, DType::BF16, DType::F32);
        let output = LinearBackend::runtime_matmul(&backend, &request, &x, &weight)
            .expect("mixed Vulkan matmul dispatch")
            .expect("Vulkan backend must own flattened F32 x BF16 matmul");
        assert_eq!(output.dims(), &[2, 3]);
        assert_eq!(output.dtype(), DType::F32);
        assert_fixture_values(&output, false);

        let x = x
            .to_device(Device::Vulkan(0))
            .expect("resident flattened activation");
        let cached_weight = weight.clone();
        let weight = weight
            .to_device(Device::Vulkan(0))
            .expect("resident flattened BF16 weight");
        let request =
            MatmulRequest::plain(x.dims().to_vec(), weight.dims().to_vec(), DType::F32, false)
                .with_dtypes(DType::F32, DType::BF16, DType::F32);
        let output = LinearBackend::runtime_matmul(&backend, &request, &x, &weight)
            .expect("resident mixed Vulkan matmul dispatch")
            .expect("Vulkan backend must own resident flattened F32 x BF16 matmul");
        assert_eq!(output.dims(), &[2, 3]);
        assert_eq!(output.dtype(), DType::F32);
        assert_eq!(output.device(), Device::Vulkan(0));
        assert_fixture_values(&output, false);

        // Layer-bounded prefill feeds the LM head this exact flattened shape
        // through runtime_linear_decode rather than the generic matmul facade.
        let output = LinearBackend::runtime_linear_decode(&backend, &x, &weight)
            .expect("resident rank-2 Vulkan linear dispatch")
            .expect("Vulkan decode hook must normalize flattened LM-head rows");
        assert_eq!(output.dims(), &[2, 3]);
        assert_eq!(output.dtype(), DType::F32);
        assert_eq!(output.device(), Device::Vulkan(0));
        assert_fixture_values(&output, false);

        // Production model loading may keep the immutable kt weight on CPU
        // while the backend's private packed-weight cache owns its Vulkan
        // copy. A resident final hidden row must still reach that cache.
        let output = LinearBackend::runtime_linear_decode(&backend, &x, &cached_weight)
            .expect("resident activation with cached Vulkan weight dispatch")
            .expect("Vulkan decode hook must bridge the small activation");
        assert_eq!(output.dims(), &[2, 3]);
        assert_eq!(output.dtype(), DType::F32);
        assert_eq!(output.device(), Device::Cpu);
        assert_fixture_values(&output, false);

        // GDN recurrence currently returns a CPU activation while the output
        // projection stays resident. Only the small activation crosses the
        // boundary; the BF16 base weight must remain on its companion device.
        let (x, weight) = mixed_linear_fixture();
        let weight = weight
            .to_device(Device::Vulkan(0))
            .expect("cross-resident BF16 weight");
        let output = LinearBackend::runtime_linear_decode(&backend, &x, &weight)
            .expect("cross-residency mixed Vulkan linear dispatch")
            .expect("Vulkan backend must upload CPU activation to resident weight");
        assert_eq!(output.dims(), &[1, 2, 3]);
        assert_eq!(output.dtype(), DType::F32);
        assert_eq!(output.device(), Device::Vulkan(0));
        assert_fixture_values(&output, false);

        // Current loading keeps weights and activations genuinely resident.
        // Include a non-zero-offset sequence view: chunked LM-head scoring
        // narrows normalized hidden rows before calling this same hook.
        let (x, weight) = mixed_linear_fixture();
        let x = Tensor::cat(&[&x, &x], 1)
            .expect("extended activation")
            .to_device(Device::Vulkan(0))
            .expect("resident activation");
        let x = x.narrow(1, 1, 2).expect("offset activation chunk");
        assert_ne!(x.layout().start_offset(), 0);
        let weight = weight
            .to_device(Device::Vulkan(0))
            .expect("resident BF16 weight");
        let output = LinearBackend::runtime_linear_decode(&backend, &x, &weight)
            .expect("resident mixed Vulkan linear dispatch")
            .expect("Vulkan backend must own resident F32 x BF16 projection");
        assert_eq!(output.dims(), &[1, 2, 3]);
        assert_eq!(output.dtype(), DType::F32);
        assert_eq!(output.device(), Device::Vulkan(0));
        assert_fixture_values(&output, true);
    }
}

pub(super) fn linear_prefill_apply_offset(
    backend: &VulkanBackend,
    x: &kiln_tensor::Tensor,
    full_weight_t: &kiln_tensor::Tensor,
    chunk_start: usize,
    chunk_len: usize,
) -> Result<Option<kiln_tensor::Tensor>> {
    // kt guards read directly off the kt args before the bridge.
    if !backend.has_vulkan() || !backend.linear_decode_enabled {
        return Ok(None);
    }
    if !matches!(x.device(), kiln_tensor::Device::Cpu)
        || !matches!(full_weight_t.device(), kiln_tensor::Device::Cpu)
    {
        return Ok(None);
    }
    // Only the bf16-packed kernel has an offset variant today; require
    // bf16 weights so the cached buffer matches the dispatch shader.
    if full_weight_t.dtype() != kiln_tensor::DType::BF16 {
        return Ok(None);
    }
    // (#1082) kt-native: the cached-weight offset kernel + FLOP-ceiling
    // sub-chunking run directly on the kt args (the FLCE caller owns its
    // own analytic backward, so this is forward-only).
    let Ok((_batch, _seq_len, hidden_x)) = x.dims3() else {
        return Ok(None);
    };
    let Ok((hidden_w, full_out_dim)) = full_weight_t.dims2() else {
        return Ok(None);
    };
    if hidden_x != hidden_w {
        return Ok(None);
    }
    if chunk_start + chunk_len > full_out_dim {
        return Ok(None);
    }
    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?
        .clone();
    let weight_buffer = backend.cached_bf16_packed_weight_buffer_kt(full_weight_t)?;
    // Promote x to f32 for the kernel (kernel expects f32 input).
    let x_f32 = if x.dtype() == kiln_tensor::DType::F32 {
        x.clone()
    } else {
        x.to_dtype(kiln_tensor::DType::F32)?
    };
    let dims = x_f32.dims().to_vec();
    let row_count: usize = dims[..dims.len() - 1].iter().product();
    let dispatch_x = if dims.len() == 3 && dims[1] == 1 {
        x_f32
    } else {
        x_f32.reshape((row_count, 1usize, hidden_x))?
    };
    // Per-dispatch FLOP guard. FLCE chunks at chunk_size=4096 sit
    // right at the 20 GFLOP ceiling for T=918; longer T or larger
    // chunk_len passed by future callers would put a single submit
    // over the safety limit. Sub-chunk along the chunk_len dim so
    // each submit fits -- that's strictly better than bailing to
    // FLCE's CPU fallback because each sub-chunk still uses the
    // same offset kernel with no re-upload of the weight buffer.
    let sub_chunk_len = if dispatch_exceeds_safety_ceiling(row_count, hidden_x, chunk_len) {
        max_chunk_dim_for_flop(row_count.saturating_mul(hidden_x)).min(chunk_len)
    } else {
        chunk_len
    };
    let out = if sub_chunk_len == chunk_len {
        let x_data = kt_tensor_to_f32_bytes_with_shape(&dispatch_x)?.0;
        let out_bytes =
            kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached_bf16_weights_offset_bytes(
                vk_device.as_ref(),
                &x_data,
                weight_buffer.as_ref(),
                row_count,
                hidden_x,
                chunk_len,
                chunk_start,
                full_out_dim,
            )
            .context("VulkanBackend: linear_prefill_apply_offset dispatch failed")?;
        kt_tensor_from_f32_bytes(
            &out_bytes,
            &[row_count, 1, chunk_len],
            kiln_tensor::DType::F32,
        )?
    } else {
        // One-shot trace so the operator can see when FLCE chunks
        // are themselves being sub-chunked. Combined with the
        // VulkanLinearOp chunking traces, gives a complete picture
        // of which paths are exceeding the safety ceiling.
        static FIRST_OFFSET_SUBCHUNK_LOGGED: OnceLock<()> = OnceLock::new();
        FIRST_OFFSET_SUBCHUNK_LOGGED.get_or_init(|| {
            let total_gflop = (2u64
                .saturating_mul(row_count as u64)
                .saturating_mul(hidden_x as u64)
                .saturating_mul(chunk_len as u64)) as f64
                / 1.0e9;
            let sub_count = chunk_len.div_ceil(sub_chunk_len);
            tracing::info!(
                row_count,
                hidden_x,
                chunk_len,
                full_out_dim,
                total_gflop,
                sub_chunk_len,
                sub_count,
                "linear_prefill_apply_offset first sub-chunked dispatch"
            );
        });
        // Walk chunk_len in sub_chunk_len-sized strides; concat
        // outputs along the last axis. Same kernel/buffer per
        // sub-dispatch, just different `chunk_start` offsets and
        // smaller `chunk_len` per submit.
        let mut sub_outputs: Vec<kiln_tensor::Tensor> = Vec::new();
        let mut sub_offset = 0usize;
        let x_data = kt_tensor_to_f32_bytes_with_shape(&dispatch_x)?.0;
        while sub_offset < chunk_len {
            let cur_len = (chunk_len - sub_offset).min(sub_chunk_len);
            let sub_bytes =
                kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached_bf16_weights_offset_bytes(
                    vk_device.as_ref(),
                    &x_data,
                    weight_buffer.as_ref(),
                    row_count,
                    hidden_x,
                    cur_len,
                    chunk_start + sub_offset,
                    full_out_dim,
                )
                .with_context(|| {
                    format!(
                        "VulkanBackend: linear_prefill_apply_offset sub-chunk \
                         (sub_offset={sub_offset}, cur_len={cur_len}, \
                          chunk_start={chunk_start}, chunk_len={chunk_len}) failed"
                    )
                })?;
            let sub = kt_tensor_from_f32_bytes(
                &sub_bytes,
                &[row_count, 1, cur_len],
                kiln_tensor::DType::F32,
            )?;
            sub_outputs.push(sub);
            sub_offset += cur_len;
        }
        let sub_refs: Vec<&kiln_tensor::Tensor> = sub_outputs.iter().collect();
        kiln_tensor::ops::concat(&sub_refs, 2).context("offset sub-chunk concat")?
    };
    // Output from kernel is `[row_count, 1, chunk_len]`. Restore the
    // caller's leading dims with chunk_len in the last position.
    let mut out_dims = dims;
    *out_dims.last_mut().unwrap() = chunk_len;
    let reshaped = out.reshape(out_dims.as_slice())?;
    Ok(Some(reshaped))
}

pub(super) fn supports_linear_decode_argmax(backend: &VulkanBackend) -> bool {
    backend.has_vulkan() && backend.linear_decode_enabled
}

pub(super) fn linear_decode_argmax(
    backend: &VulkanBackend,
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> Result<Option<u32>> {
    // kt guards read directly off the kt args before the bridge.
    if !backend.has_vulkan()
        || !backend.linear_decode_enabled
        || x.dtype() != kiln_tensor::DType::F32
    {
        return Ok(None);
    }
    if !matches!(x.device(), kiln_tensor::Device::Cpu)
        || !matches!(weight_t.device(), kiln_tensor::Device::Cpu)
    {
        return Ok(None);
    }
    // (#1082) Fully kt-native: the lm_head weight (the 778 MB table) was
    // re-bridged + re-uploaded per token under the candle-id cache; key on
    // the stable kt id so it uploads once.
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return Ok(None);
    };
    if batch != 1 || seq_len != 1 {
        return Ok(None);
    }
    let Ok((weight_hidden, out_dim)) = weight_t.dims2() else {
        return Ok(None);
    };
    if weight_hidden != hidden {
        return Ok(None);
    }

    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
    let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
    let token = if backend.use_bf16_packed_linear_weight_kt(weight_t) {
        let weight_buf = backend.cached_bf16_packed_weight_buffer_kt(weight_t)?;
        kiln_vulkan_kernel::kernels::dispatch_linear_decode_argmax_cached_bf16_weights_bytes(
            vk_device,
            &x_data,
            &weight_buf,
            hidden,
            out_dim,
        )
    } else {
        let weight_buf = backend.cached_f32_weight_buffer_kt(weight_t)?;
        kiln_vulkan_kernel::kernels::dispatch_linear_decode_argmax_cached_bytes(
            vk_device,
            &x_data,
            &weight_buf,
            hidden,
            out_dim,
        )
    }
    .context("linear_decode_argmax kernel failed")?;
    Ok(Some(token))
}

pub(super) fn supports_linear_decode_argmax_batch(backend: &VulkanBackend) -> bool {
    backend.has_vulkan() && backend.linear_decode_enabled && backend.linear_argmax_batch_enabled
}

pub(super) fn supports_linear_decode_sample(backend: &VulkanBackend, top_k: u32) -> bool {
    // The fused sample kernel only handles top_k in `1..=TOPK_SAMPLE_KERNEL_K_MAX`.
    // Larger requests fall back to the host sampler.
    backend.has_vulkan()
        && backend.linear_decode_enabled
        && top_k > 0
        && top_k <= kiln_vulkan_kernel::kernels::TOPK_SAMPLE_KERNEL_K_MAX
}

pub(super) fn linear_decode_sample(
    backend: &VulkanBackend,
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
    history_indices: &[u32],
    history_counts: &[u32],
    repetition_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
    seed: u64,
) -> Result<Option<u32>> {
    // kt guards read directly off the kt args before the bridge.
    if !supports_linear_decode_sample(backend, top_k) || x.dtype() != kiln_tensor::DType::F32 {
        return Ok(None);
    }
    if !matches!(x.device(), kiln_tensor::Device::Cpu)
        || !matches!(weight_t.device(), kiln_tensor::Device::Cpu)
    {
        return Ok(None);
    }
    // (#1082) Fully kt-native: lm_head weight keyed on the stable kt id.
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return Ok(None);
    };
    if batch != 1 || seq_len != 1 {
        return Ok(None);
    }
    let Ok((weight_hidden, out_dim)) = weight_t.dims2() else {
        return Ok(None);
    };
    if weight_hidden != hidden {
        return Ok(None);
    }

    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
    let packed_bf16 = backend.use_bf16_packed_linear_weight_kt(weight_t);
    let weight_buf = if packed_bf16 {
        backend.cached_bf16_packed_weight_buffer_kt(weight_t)?
    } else {
        backend.cached_f32_weight_buffer_kt(weight_t)?
    };
    let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
    let token = kiln_vulkan_kernel::kernels::dispatch_linear_decode_sample_bytes(
        vk_device,
        &x_data,
        &weight_buf,
        packed_bf16,
        hidden,
        out_dim,
        history_indices,
        history_counts,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        temperature,
        top_k,
        top_p,
        min_p,
        seed,
    )
    .context("fused linear_decode_sample dispatch failed")?;
    Ok(Some(token))
}

pub(super) fn supports_linear_decode_sample_batch(
    backend: &VulkanBackend,
    top_k: &[u32],
    temperatures: &[f32],
) -> bool {
    backend.has_vulkan()
        && backend.linear_decode_enabled
        && top_k.len() == temperatures.len()
        && !top_k.is_empty()
        && top_k.iter().zip(temperatures.iter()).all(|(&k, &temp)| {
            let greedy = temp == 0.0 || (k == 1 && temp.is_finite() && temp > 0.0);
            greedy
                || (temp.is_finite()
                    && temp > 0.0
                    && k > 0
                    && k <= kiln_vulkan_kernel::kernels::TOPK_SAMPLE_KERNEL_K_MAX)
        })
}

pub(super) fn linear_decode_sample_batch(
    backend: &VulkanBackend,
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
    history_rows: &[u32],
    history_indices: &[u32],
    history_counts: &[u32],
    repetition_penalties: &[f32],
    presence_penalties: &[f32],
    frequency_penalties: &[f32],
    temperatures: &[f32],
    top_k: &[u32],
    top_p: &[f32],
    min_p: &[f32],
    seeds: &[u64],
) -> Result<Option<Vec<u32>>> {
    if !supports_linear_decode_sample_batch(backend, top_k, temperatures)
        || x.dtype() != kiln_tensor::DType::F32
    {
        return Ok(None);
    }
    if !matches!(x.device(), kiln_tensor::Device::Cpu)
        || !matches!(weight_t.device(), kiln_tensor::Device::Cpu)
    {
        return Ok(None);
    }
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return Ok(None);
    };
    if batch == 0 || seq_len != 1 {
        return Ok(None);
    }
    let Ok((weight_hidden, out_dim)) = weight_t.dims2() else {
        return Ok(None);
    };
    if weight_hidden != hidden {
        return Ok(None);
    }

    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
    let packed_bf16 = backend.use_bf16_packed_linear_weight_kt(weight_t);
    let weight_buf = if packed_bf16 {
        backend.cached_bf16_packed_weight_buffer_kt(weight_t)?
    } else {
        backend.cached_f32_weight_buffer_kt(weight_t)?
    };
    let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
    let tokens = kiln_vulkan_kernel::kernels::dispatch_linear_decode_sample_batch_bytes(
        vk_device,
        &x_data,
        &weight_buf,
        packed_bf16,
        batch,
        hidden,
        out_dim,
        history_rows,
        history_indices,
        history_counts,
        repetition_penalties,
        presence_penalties,
        frequency_penalties,
        temperatures,
        top_k,
        top_p,
        min_p,
        seeds,
    )
    .context("fused linear_decode_sample_batch dispatch failed")?;
    Ok(Some(tokens))
}

pub(super) fn linear_decode_argmax_batch(
    backend: &VulkanBackend,
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> Result<Option<Vec<u32>>> {
    // kt guards read directly off the kt args before the bridge.
    if !backend.has_vulkan()
        || !backend.linear_decode_enabled
        || !backend.linear_argmax_batch_enabled
        || x.dtype() != kiln_tensor::DType::F32
    {
        return Ok(None);
    }
    if !matches!(x.device(), kiln_tensor::Device::Cpu)
        || !matches!(weight_t.device(), kiln_tensor::Device::Cpu)
    {
        return Ok(None);
    }
    // (#1082) Fully kt-native: lm_head weight keyed on the stable kt id.
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return Ok(None);
    };
    if batch == 0 || seq_len != 1 {
        return Ok(None);
    }
    let Ok((weight_hidden, out_dim)) = weight_t.dims2() else {
        return Ok(None);
    };
    if weight_hidden != hidden {
        return Ok(None);
    }

    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
    let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
    let tokens = if backend.use_bf16_packed_linear_weight_kt(weight_t) {
        let weight_buf = backend.cached_bf16_packed_weight_buffer_kt(weight_t)?;
        kiln_vulkan_kernel::kernels::dispatch_linear_decode_argmax_batched_cached_bf16_weights_bytes(
            vk_device,
            &x_data,
            &weight_buf,
            batch,
            hidden,
            out_dim,
        )
    } else {
        let weight_buf = backend.cached_f32_weight_buffer_kt(weight_t)?;
        kiln_vulkan_kernel::kernels::dispatch_linear_decode_argmax_batched_cached_bytes(
            vk_device,
            &x_data,
            &weight_buf,
            batch,
            hidden,
            out_dim,
        )
    }
    .context("linear_decode_argmax_batch kernel failed")?;
    Ok(Some(tokens))
}
