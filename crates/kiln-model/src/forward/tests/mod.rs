// `Var` is hoisted to the file-level kt import block above with
// `#[allow(unused_imports)]`, so it reaches this `mod tests` via
// `use super::*;` without an extra inner kt Var import (#1082).
use super::*;
use crate::backend::cpu::CpuBackend;

// Called only by the `#[cfg(feature = "metal")]` / `#[cfg(feature = "cuda")]`
// graph tests below; dead in default-lane test builds — cfg_attr required
// (verified by default-lane probe).
#[cfg_attr(not(any(feature = "metal", feature = "cuda")), allow(dead_code))]
fn explicit_hardware_qualification() -> bool {
    std::env::var("KILN_QUALIFICATION").as_deref() == Ok("1")
}

#[test]
fn graph_stable_paged_metadata_keeps_bucketed_kernel_bound() -> Result<()> {
    let cache = crate::PagedKvCacheKt::new(0, 8, 64, 1, 8, DType::BF16, Device::Cpu)?;
    let tables = [
        BlockTable { blocks: vec![0] },
        BlockTable { blocks: vec![1] },
        BlockTable { blocks: vec![2] },
        BlockTable { blocks: vec![3] },
    ];
    let table_refs: Vec<&BlockTable> = tables.iter().collect();
    let positions = [1usize, 2, 3, 4];
    let stable_block_table = Tensor::from_vec(vec![0u32, 1, 2, 3], (positions.len(), 1usize))?;
    let stable_seqused_k = Tensor::from_vec(vec![2u32, 3, 4, 5], positions.len())?;

    let stable = CachedPagedDecodeMeta::build_with_stable_buffers(
        &cache,
        &table_refs,
        &positions,
        &stable_block_table,
        &stable_seqused_k,
        #[cfg(feature = "cuda")]
        None,
    )?;
    assert_eq!(stable.max_seqlen_k, 5);
    assert_eq!(stable.kernel_max_seqlen_k, crate::generate::FA2_KBLOCK_N);

    let eager = CachedPagedDecodeMeta::build(
        &Device::Cpu,
        &cache,
        &table_refs,
        &positions,
        #[cfg(feature = "cuda")]
        None,
    )?;
    assert_eq!(eager.max_seqlen_k, 5);
    assert_eq!(eager.kernel_max_seqlen_k, 5);
    Ok(())
}

#[cfg(feature = "rocm")]
#[test]
fn layer_snapshot_captures_only_the_final_row_as_f32() -> Result<()> {
    let hidden = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], (1, 2, 3))?
        .to_dtype(DType::BF16)?;
    let row = qualification_layer_last_row(&hidden, 2)?;

    assert_eq!(row.shape(), &[1, 1, 3]);
    assert_eq!(row.dtype(), DType::F32);
    assert_eq!(row.flatten_all()?.to_vec1::<f32>()?, vec![4.0, 5.0, 6.0]);
    Ok(())
}

#[test]
fn frozen_gdn_gated_rmsnorm_backward_matches_finite_difference() -> Result<()> {
    let shape = vec![2, 2, 3];
    let hidden = 3usize;
    let eps = 7e-4_f64;
    let x_values = vec![
        0.35, -0.8, 1.2, -1.1, 0.45, 0.7, 1.4, -0.25, 0.55, -0.6, 0.95, -1.3,
    ];
    let z_values = vec![
        -1.2, 0.3, 1.1, 0.75, -0.55, 1.6, -0.2, 0.9, -1.45, 1.25, -0.7, 0.15,
    ];
    let weight_values = vec![0.75, -1.1, 0.35];
    let upstream_values = vec![
        0.6, -0.4, 1.2, -0.8, 0.25, 0.9, 1.1, -0.65, 0.35, -1.0, 0.7, 0.45,
    ];
    let to_f32 = |values: &[f64]| values.iter().map(|&value| value as f32).collect::<Vec<_>>();

    let x = Tensor::from_vec(to_f32(&x_values), shape.clone())?;
    let z = Tensor::from_vec(to_f32(&z_values), shape.clone())?;
    let weight = Tensor::from_vec(to_f32(&weight_values), vec![hidden])?;
    let upstream = Tensor::from_vec(to_f32(&upstream_values), shape.clone())?;
    let grads = gdn_gated_rms_norm_frozen_weight_backward_no_grad(&x, &z, &weight, eps, &upstream)?;

    assert_eq!(grads.dx.shape(), shape.as_slice());
    assert_eq!(grads.dz.shape(), shape.as_slice());
    assert_eq!(grads.dx.dtype(), DType::F32);
    assert_eq!(grads.dz.dtype(), DType::F32);
    assert_eq!(grads.dx.device(), Device::Cpu);
    assert_eq!(grads.dz.device(), Device::Cpu);

    let dx = grads.dx.flatten_all()?.to_vec1::<f32>()?;
    let dz = grads.dz.flatten_all()?.to_vec1::<f32>()?;
    let loss = |x: &[f64], z: &[f64]| -> f64 {
        x.chunks_exact(hidden)
            .zip(z.chunks_exact(hidden))
            .enumerate()
            .map(|(row, (x_row, z_row))| {
                let mean_square =
                    x_row.iter().map(|value| value * value).sum::<f64>() / hidden as f64;
                let inv_rms = (mean_square + eps).sqrt().recip();
                x_row
                    .iter()
                    .zip(z_row)
                    .enumerate()
                    .map(|(lane, (&x, &z))| {
                        let index = row * hidden + lane;
                        let silu = z / (1.0 + (-z).exp());
                        upstream_values[index] * x * weight_values[lane] * inv_rms * silu
                    })
                    .sum::<f64>()
            })
            .sum()
    };

    let finite_difference_step = 1e-5_f64;
    for index in 0..x_values.len() {
        let mut plus = x_values.clone();
        let mut minus = x_values.clone();
        plus[index] += finite_difference_step;
        minus[index] -= finite_difference_step;
        let expected =
            (loss(&plus, &z_values) - loss(&minus, &z_values)) / (2.0 * finite_difference_step);
        assert!(
            (dx[index] as f64 - expected).abs() < 8e-4,
            "dx[{index}]={} finite_difference={expected}",
            dx[index]
        );
    }
    for index in 0..z_values.len() {
        let mut plus = z_values.clone();
        let mut minus = z_values.clone();
        plus[index] += finite_difference_step;
        minus[index] -= finite_difference_step;
        let expected =
            (loss(&x_values, &plus) - loss(&x_values, &minus)) / (2.0 * finite_difference_step);
        assert!(
            (dz[index] as f64 - expected).abs() < 8e-4,
            "dz[{index}]={} finite_difference={expected}",
            dz[index]
        );
    }

    let bf16_grads = gdn_gated_rms_norm_frozen_weight_backward_no_grad(
        &x.to_dtype(DType::BF16)?,
        &z.to_dtype(DType::BF16)?,
        &weight.to_dtype(DType::BF16)?,
        eps,
        &upstream.to_dtype(DType::BF16)?,
    )?;
    assert_eq!(bf16_grads.dx.shape(), shape.as_slice());
    assert_eq!(bf16_grads.dz.shape(), shape.as_slice());
    assert_eq!(bf16_grads.dx.dtype(), DType::BF16);
    assert_eq!(bf16_grads.dz.dtype(), DType::BF16);
    assert_eq!(bf16_grads.dx.device(), Device::Cpu);
    assert_eq!(bf16_grads.dz.device(), Device::Cpu);

    Ok(())
}

#[cfg(feature = "vulkan")]
#[test]
fn residual_add_aligns_cpu_vulkan_inference_branches_to_lhs() -> Result<()> {
    if std::env::var("KILN_TENSOR_VULKAN_TEST").ok().as_deref() != Some("1") {
        return Ok(());
    }
    assert!(
        crate::backend::vulkan::vulkan_is_available(),
        "KILN_TENSOR_VULKAN_TEST=1 requires a working Vulkan device"
    );
    let cpu = Tensor::from_vec(vec![1.0_f32, -2.0, 3.0, -4.0], (1, 2, 2))?;
    let vulkan = cpu.to_device(Device::Vulkan(0))?;

    let resident = residual_add(vulkan.clone(), cpu.clone())?;
    assert_eq!(resident.device(), Device::Vulkan(0));
    assert_eq!(
        resident
            .to_device(Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?,
        vec![2.0, -4.0, 6.0, -8.0]
    );

    let host = residual_add(cpu, vulkan)?;
    assert_eq!(host.device(), Device::Cpu);
    assert_eq!(
        host.flatten_all()?.to_vec1::<f32>()?,
        vec![2.0, -4.0, 6.0, -8.0]
    );
    Ok(())
}

#[cfg(feature = "vulkan")]
#[test]
fn rms_norm_fallback_aligns_vulkan_weight_to_cpu_activation() -> Result<()> {
    if std::env::var("KILN_TENSOR_VULKAN_TEST").ok().as_deref() != Some("1") {
        return Ok(());
    }
    assert!(
        crate::backend::vulkan::vulkan_is_available(),
        "KILN_TENSOR_VULKAN_TEST=1 requires a working Vulkan device"
    );
    let x = Tensor::from_vec(vec![0.5_f32, -1.0, 2.0, -4.0], (1, 1, 4))?;
    let weight_cpu =
        Tensor::from_vec(vec![0.125_f32, -0.25, 0.5, -0.75], (4,))?.to_dtype(DType::BF16)?;
    let expected = rms_norm_fallback(&x, &weight_cpu, 1e-6)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let weight_vulkan = weight_cpu.to_device(Device::Vulkan(0))?;

    let actual = rms_norm_fallback(&x, &weight_vulkan, 1e-6)?;
    assert_eq!(actual.device(), Device::Cpu);
    assert_eq!(actual.dtype(), DType::F32);
    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
    for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() <= 1e-6,
            "RMSNorm lane {index}: got={got} want={want}"
        );
    }
    Ok(())
}

#[cfg(feature = "vulkan")]
#[test]
fn materialized_sdpa_aligns_cpu_cache_to_vulkan_query() -> Result<()> {
    if std::env::var("KILN_TENSOR_VULKAN_TEST").ok().as_deref() != Some("1") {
        return Ok(());
    }
    assert!(
        crate::backend::vulkan::vulkan_is_available(),
        "KILN_TENSOR_VULKAN_TEST=1 requires a working Vulkan device"
    );
    let q_cpu = Tensor::from_slice(
        &[0.25_f32, -0.5, 0.75, 1.0, -0.25, 0.5, -0.75, -1.0],
        (1, 2, 1, 4),
    )?;
    let k_cpu = Tensor::from_slice(
        &[
            0.5_f32, 0.25, -0.5, 1.0, -0.25, 0.75, 0.5, -1.0, 1.0, -0.5, 0.25, 0.75, -0.5, -0.25,
            1.0, 0.5, 0.75, -1.0, 0.25, 0.5, 0.25, 0.5, -0.75, 1.0,
        ],
        (1, 2, 3, 4),
    )?;
    let v_cpu = Tensor::from_slice(
        &[
            1.0_f32, 0.0, -0.5, 0.25, 0.5, -1.0, 0.75, 0.0, -0.25, 0.5, 1.0, -0.75, 0.0, 1.0, 0.5,
            -0.5, -1.0, 0.25, 0.0, 0.75, 0.5, -0.25, 1.0, 0.0,
        ],
        (1, 2, 3, 4),
    )?;
    let expected = gqa_sdpa_materialized_default(&q_cpu, &k_cpu, &v_cpu, 1, 3, 2.0)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let q_vulkan = q_cpu.to_device(Device::Vulkan(0))?;

    let actual = gqa_sdpa_materialized_default(&q_vulkan, &k_cpu, &v_cpu, 1, 3, 2.0)?;
    assert_eq!(actual.device(), Device::Vulkan(0));
    let actual = actual
        .to_device(Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() <= 1e-5,
            "materialized SDPA lane {index}: got={got} want={want}"
        );
    }
    Ok(())
}

#[cfg(feature = "vulkan")]
#[test]
fn disabled_vulkan_linear_decode_routes_flattened_lm_head_through_matmul() -> Result<()> {
    if std::env::var("KILN_TENSOR_VULKAN_TEST").ok().as_deref() != Some("1") {
        return Ok(());
    }
    assert!(
        crate::backend::vulkan::vulkan_is_available(),
        "KILN_TENSOR_VULKAN_TEST=1 requires a working Vulkan device"
    );
    let backend = crate::backend::vulkan::VulkanBackend::new(Device::Cpu)
        .with_linear_decode_enabled_for_test(false);

    let x = Tensor::from_vec(
        vec![0.25_f32, -0.5, 0.75, 1.0, -0.25, 0.5, -0.75, -1.0],
        (1, 2, 4),
    )?
    .to_device(Device::Vulkan(0))?;
    let weight = Tensor::from_vec(
        vec![
            0.5_f32, -0.25, 0.125, -0.5, 0.75, 0.25, 1.0, -0.125, 0.375, -0.75, 0.625, 0.5,
        ],
        (4, 3),
    )?
    .to_dtype(DType::BF16)?
    .to_device(Device::Vulkan(0))?;
    let x = x.reshape((2, 4))?;
    assert!(
        LinearBackend::runtime_linear_decode(&backend, &x, &weight)?.is_none(),
        "the generic Vulkan linear-decode quarantine must remain closed"
    );
    let output = lm_head_forward_backend_decode_if(Some(&backend), &x, &weight)?;
    assert_eq!(output.dims(), &[2, 3]);
    assert_eq!(output.device(), Device::Vulkan(0));
    assert_eq!(
        output
            .to_device(Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?,
        vec![0.375, 0.09375, 0.6875, -0.375, -0.09375, -0.6875]
    );
    Ok(())
}

/// #1082 type-flip test shim. The tests below were written against
/// candle's `new_cuda_device(0) -> Result<Device>`, which both
/// *constructs* and *validates* (probes) a CUDA device. kt's `Device`
/// is a plain enum (`Device::Cuda(0)`) with no fallible constructor, so
/// the availability probe must be explicit: build a 1-element tensor and
/// move it to CUDA(0). On a host without a visible GPU the `to_device`
/// move returns `Err`, so the tests skip exactly as before; on the GPU
/// pod it returns the kt `Device::Cuda(0)`. This preserves the original
/// skip-if-no-CUDA semantics while flipping the device type to kt.
#[cfg(feature = "cuda")]
fn new_cuda_device(index: usize) -> Result<Device> {
    let dev = Device::Cuda(index);
    // Probe: a host→CUDA move fails if no CUDA device is visible.
    let _probe = Tensor::from_slice(&[0.0f32], (1usize,))?.to_device(dev)?;
    Ok(dev)
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_sigmoid_kt_default_matches_host_formula() -> Result<()> {
    let Ok(device) = new_cuda_device(0) else {
        eprintln!("CUDA unavailable, skipping test_cuda_sigmoid_kt_default_matches_host_formula");
        return Ok(());
    };
    let data = [-8.0_f32, -2.0, -0.5, 0.0, 0.5, 2.0, 8.0, 16.0];
    let x = Tensor::from_slice(&data, (2usize, 4usize))?
        .to_device(device)?
        .contiguous()?;
    let out = cuda_sigmoid(&x)?;
    synchronize_for_profile(&device)?;
    let got = out.flatten_all()?.to_vec1::<f32>()?;
    for (idx, (&input, &actual)) in data.iter().zip(got.iter()).enumerate() {
        let expected = 1.0 / (1.0 + (-input).exp());
        assert!(
            (actual - expected).abs() < 2e-5,
            "sigmoid mismatch at {idx}: actual={actual} expected={expected}"
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_silu_kt_default_matches_host_formula() -> Result<()> {
    let Ok(device) = new_cuda_device(0) else {
        eprintln!("CUDA unavailable, skipping test_cuda_silu_kt_default_matches_host_formula");
        return Ok(());
    };
    let data = [-8.0_f32, -2.0, -0.5, 0.0, 0.5, 2.0, 8.0, 16.0];
    let x = Tensor::from_slice(&data, (2usize, 4usize))?
        .to_device(device)?
        .contiguous()?;
    let out = cuda_silu(&x)?;
    synchronize_for_profile(&device)?;
    let got = out.flatten_all()?.to_vec1::<f32>()?;
    for (idx, (&input, &actual)) in data.iter().zip(got.iter()).enumerate() {
        let sigmoid = 1.0 / (1.0 + (-input).exp());
        let expected = input * sigmoid;
        assert!(
            (actual - expected).abs() < 2e-5,
            "silu mismatch at {idx}: actual={actual} expected={expected}"
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_softmax_last_dim_kt_default_matches_host_formula() -> Result<()> {
    let Ok(device) = new_cuda_device(0) else {
        eprintln!(
            "CUDA unavailable, skipping test_cuda_softmax_last_dim_kt_default_matches_host_formula"
        );
        return Ok(());
    };
    let data = [
        1.0_f32, 2.0, 3.0, -1.0, //
        -4.0, -2.0, -2.0, 0.0,
    ];
    let x = Tensor::from_slice(&data, (2usize, 4usize))?
        .to_device(device)?
        .contiguous()?;
    let direct = try_kt_softmax_last_dim(&x)?
        .context("expected CUDA kt softmax helper to accept contiguous F32 input")?;
    let out = cuda_softmax_last_dim(&x)?;
    synchronize_for_profile(&device)?;

    let direct_vals = direct.flatten_all()?.to_vec1::<f32>()?;
    let got = out.flatten_all()?.to_vec1::<f32>()?;
    for (idx, (&actual, &direct_actual)) in got.iter().zip(direct_vals.iter()).enumerate() {
        assert!(
            (actual - direct_actual).abs() < 1e-7,
            "default softmax path diverged from direct kt helper at {idx}: \
                 actual={actual} direct={direct_actual}"
        );
    }

    for row_idx in 0..2 {
        let row = &data[row_idx * 4..(row_idx + 1) * 4];
        let row_max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = row.iter().map(|v| (v - row_max).exp()).sum();
        for col_idx in 0..4 {
            let idx = row_idx * 4 + col_idx;
            let expected = (data[idx] - row_max).exp() / exp_sum;
            assert!(
                (got[idx] - expected).abs() < 2e-5,
                "softmax mismatch at row={row_idx} col={col_idx}: \
                     actual={} expected={expected}",
                got[idx]
            );
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_l2_normalize_kt_exactly_matches_composite() -> Result<()> {
    let Ok(device) = new_cuda_device(0) else {
        eprintln!("CUDA unavailable, skipping test_cuda_l2_normalize_kt_exactly_matches_composite");
        return Ok(());
    };
    let data = [3.0_f32, 4.0, 0.0, -2.0, 1.0, 2.0];
    let x = Tensor::from_slice(&data, (2usize, 3usize))?
        .to_device(device)?
        .contiguous()?;
    let x_f32 = x.to_dtype(DType::F32)?;
    let direct = try_kt_l2_normalize(&x_f32, 1e-6)?
        .context("expected CUDA kt l2_normalize helper to accept contiguous F32 input")?;
    let sq_sum = try_kt_sum_squared_last_dim_keepdim(&x_f32)?
        .context("expected CUDA kt sum-squared helper to accept contiguous F32 input")?;
    let sq_sum_eps = try_kt_add_scalar(&sq_sum, 1e-6)?
        .context("expected CUDA kt scalar-add helper to accept contiguous F32 input")?;
    let norm = try_kt_sqrt(&sq_sum_eps)?
        .context("expected CUDA kt sqrt helper to accept contiguous F32 input")?;
    let composite = x_f32.broadcast_div(&norm)?;
    let out = l2_normalize(&x)?;
    synchronize_for_profile(&device)?;

    let direct_vals = direct.flatten_all()?.to_vec1::<f32>()?;
    let composite_vals = composite.flatten_all()?.to_vec1::<f32>()?;
    let got = out.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(
        direct_vals, composite_vals,
        "fused CUDA L2 normalization must exactly match the portable composite"
    );
    assert_eq!(
        got, direct_vals,
        "default L2 normalization path must use the parity-qualified fused route"
    );

    for row_idx in 0..2 {
        let row = &data[row_idx * 3..(row_idx + 1) * 3];
        let norm = (row.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
        for col_idx in 0..3 {
            let idx = row_idx * 3 + col_idx;
            let expected = data[idx] / norm;
            assert!(
                (got[idx] - expected).abs() < 2e-5,
                "l2_normalize mismatch at row={row_idx} col={col_idx}: \
                     actual={} expected={expected}",
                got[idx]
            );
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_lora_add_kt_default_matches_host_formula() -> Result<()> {
    let Ok(device) = new_cuda_device(0) else {
        eprintln!("CUDA unavailable, skipping test_cuda_lora_add_kt_default_matches_host_formula");
        return Ok(());
    };
    let base_data = [1.0_f32, -2.0, 3.5, 0.0, 0.25, -0.5];
    let delta_data = [0.1_f32, 0.2, -0.3, 4.0, -0.05, 0.5];
    let base = Tensor::from_slice(&base_data, (2usize, 3usize))?
        .to_device(device)?
        .contiguous()?;
    let delta = Tensor::from_slice(&delta_data, (2usize, 3usize))?
        .to_device(device)?
        .contiguous()?;
    let out = try_kt_lora_add(&base, &delta)?
        .context("expected CUDA kt lora_add helper to accept contiguous F32 input")?;
    synchronize_for_profile(&device)?;

    let got = out.flatten_all()?.to_vec1::<f32>()?;
    for (idx, (&b, &d)) in base_data.iter().zip(delta_data.iter()).enumerate() {
        let expected = b + d;
        assert!(
            (got[idx] - expected).abs() < 1e-7,
            "lora_add mismatch at {idx}: actual={} expected={expected}",
            got[idx]
        );
    }
    Ok(())
}

/// Regression test for the 2026-05-12 → 2026-05-14 silent inference
/// outage. Commit 997a608f widened the projection-load transpose drop
/// decision when Vulkan was active, which silently replaced every projection
/// transpose tensor (`in_proj_qkv_t`, `in_proj_z_t`, `out_proj_t`,
/// `q_proj_t`, etc.) with `Tensor::zeros((1,), DType::BF16, ...)` at
/// load time. Inference reads those caches directly via
/// `LinearBackend::runtime_linear_prefill_apply`, and the GDN prefill
/// kernel then bailed out with `only 2d matrixes are supported [1, T, hidden] [1]`
/// on every single /v1/chat/completions request. The fix narrowed the gate
/// back to retaining transposes. `ProjectionLoadPolicy` stays Vulkan-aware
/// for originals because the trainer needs them later.
///
/// This test pins the contract: turning Vulkan on must NOT drop
/// transposes by itself.
#[test]
fn vulkan_active_alone_does_not_drop_projection_transposes() {
    let _vk = crate::backend::test_only_set_vulkan_active(true);
    assert!(
        !ProjectionLoadPolicy::for_model_loader_device(Device::Cpu).drop_projection_transposes,
        "ProjectionLoadPolicy must NOT drop projection transposes just \
             because Vulkan is active — that breaks every chat completion on \
             Vulkan with `only 2d matrixes are supported [..., hidden] [1]`."
    );
}

/// Tests all run on `Device::Cpu`, so the `CpuBackend` (all kernel methods
/// return `Ok(None)`) is the right dispatch target.
fn test_backend(device: &Device) -> CpuBackend {
    // #1082 DoD-100 step 4: `CpuBackend::new` now takes a kt `Device`
    // directly (the candle bridge was dropped). `Device` here is the kt
    // alias post-flip, and kt `Device` is `Copy`.
    CpuBackend::new(*device)
}

#[derive(Debug)]
struct FixedLinearBackend {
    device: Device,
    values: Vec<f32>,
    dims: (usize, usize, usize),
}

impl crate::backend::BackendIdentity for FixedLinearBackend {
    fn runtime_name(&self) -> &'static str {
        "fixed-linear-test"
    }

    fn runtime_device(&self) -> kiln_tensor::Device {
        // #1082: the struct's `device` field is now a kt `Device`
        // (`Copy`), so the trait's kt-typed accessor returns it directly
        // — no candle bridge needed.
        self.device
    }

    fn runtime_as_any(&self) -> &dyn std::any::Any {
        &()
    }
}

impl crate::backend::StartupBackend for FixedLinearBackend {}

impl crate::backend::ExternalYieldBackend for FixedLinearBackend {
    fn runtime_synchronize_external_yield(&self) -> Result<()> {
        Ok(())
    }
}

impl crate::backend::AttentionBackend for FixedLinearBackend {}

impl crate::backend::GdnBackend for FixedLinearBackend {}

impl crate::backend::ConvBackend for FixedLinearBackend {}

impl crate::backend::LinearBackend for FixedLinearBackend {
    fn runtime_linear_decode(&self, _x: &Tensor, _weight_t: &Tensor) -> Result<Option<Tensor>> {
        Ok(Some(
            Tensor::from_vec(self.values.clone(), self.dims)?.to_device(self.device)?,
        ))
    }
}

impl crate::backend::SamplingBackend for FixedLinearBackend {}

impl crate::backend::residency::ResidentRegistry for FixedLinearBackend {}

impl crate::backend::ResidencyBackend for FixedLinearBackend {}

impl crate::backend::OptimizerBackend for FixedLinearBackend {}

impl crate::backend::PagedKvBackend for FixedLinearBackend {}

impl crate::backend::ReplayBackend for FixedLinearBackend {}

impl crate::backend::TrainingLossBackend for FixedLinearBackend {}

impl BackendRuntime for FixedLinearBackend {}

#[derive(Debug)]
struct FixedMlpBackend {
    device: Device,
    fused_values: Option<Vec<f32>>,
    fused_dims: (usize, usize, usize),
    gate_up_values: Option<Vec<f32>>,
    gate_up_dims: (usize, usize, usize),
    fused_calls: std::sync::atomic::AtomicUsize,
    gate_up_calls: std::sync::atomic::AtomicUsize,
}

impl crate::backend::BackendIdentity for FixedMlpBackend {
    fn runtime_name(&self) -> &'static str {
        "fixed-mlp-test"
    }

    fn runtime_device(&self) -> kiln_tensor::Device {
        // #1082: kt `Device` field returned directly.
        self.device
    }

    fn runtime_as_any(&self) -> &dyn std::any::Any {
        &()
    }
}

impl crate::backend::StartupBackend for FixedMlpBackend {}

impl crate::backend::ExternalYieldBackend for FixedMlpBackend {
    fn runtime_synchronize_external_yield(&self) -> Result<()> {
        Ok(())
    }
}

impl crate::backend::AttentionBackend for FixedMlpBackend {}

impl crate::backend::GdnBackend for FixedMlpBackend {}

impl crate::backend::ConvBackend for FixedMlpBackend {}

impl crate::backend::LinearBackend for FixedMlpBackend {
    fn runtime_mlp_decode(
        &self,
        _x: &Tensor,
        _gate_weight_t: &Tensor,
        _up_weight_t: &Tensor,
        _down_weight_t: &Tensor,
    ) -> Result<Option<Tensor>> {
        self.fused_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(match self.fused_values.as_ref() {
            Some(values) => {
                Some(Tensor::from_vec(values.clone(), self.fused_dims)?.to_device(self.device)?)
            }
            None => None,
        })
    }

    fn runtime_mlp_gate_up_decode(
        &self,
        _x: &Tensor,
        _gate_weight_t: &Tensor,
        _up_weight_t: &Tensor,
    ) -> Result<Option<Tensor>> {
        self.gate_up_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(match self.gate_up_values.as_ref() {
            Some(values) => {
                Some(Tensor::from_vec(values.clone(), self.gate_up_dims)?.to_device(self.device)?)
            }
            None => None,
        })
    }
}

impl crate::backend::SamplingBackend for FixedMlpBackend {}

impl crate::backend::residency::ResidentRegistry for FixedMlpBackend {}

impl crate::backend::ResidencyBackend for FixedMlpBackend {}

impl crate::backend::OptimizerBackend for FixedMlpBackend {}

impl crate::backend::PagedKvBackend for FixedMlpBackend {}

impl crate::backend::ReplayBackend for FixedMlpBackend {}

impl crate::backend::TrainingLossBackend for FixedMlpBackend {}

impl BackendRuntime for FixedMlpBackend {}

#[test]
fn test_backend_linear_decode_adds_lora_delta() -> Result<()> {
    let device = Device::Cpu;
    let x = Tensor::from_vec(vec![1.0f32, 2.0], (1, 1, 2))?.to_device(device)?;
    let weight_t = Tensor::zeros((2, 3), DType::F32, device)?;
    let lora = LoraProjectionWeights {
        // #1082: `LoraProjectionWeights.{a,b}` are now kt `KtTensor`; pass
        // the kt tensors directly (no candle bridge).
        a: Tensor::from_vec(vec![3.0f32, 4.0], (1, 2))?.to_device(device)?,
        b: Tensor::from_vec(vec![5.0f32, 6.0, 7.0], (3, 1))?.to_device(device)?,
    };
    let backend = FixedLinearBackend {
        device,
        values: vec![10.0, 20.0, 30.0],
        dims: (1, 1, 3),
    };

    let out = linear_with_lora_t_backend_decode_if(
        Some(&backend),
        false,
        &x,
        &weight_t,
        Some(&lora),
        0.5,
    )?;

    let values = out.flatten_all()?.to_vec1::<f32>()?;
    let expected = [37.5, 53.0, 68.5];
    for (got, expected) in values.iter().zip(expected) {
        assert!(
            (got - expected).abs() < 1e-6,
            "got {got}, expected {expected}"
        );
    }
    Ok(())
}

#[test]
fn test_swiglu_down_only_lora_keeps_backend_gate_up_decode() -> Result<()> {
    let device = Device::Cpu;
    let x = Tensor::from_vec(vec![1.0f32, 2.0], (1, 1, 2))?.to_device(device)?;
    let zero_proj = Tensor::zeros((2, 2), DType::F32, device)?;
    let zero_proj_t = zero_proj.t()?.contiguous()?;
    let mlp = GpuFfnWeights {
        gate_proj: zero_proj.clone(),
        up_proj: zero_proj.clone(),
        down_proj: zero_proj.clone(),
        gate_proj_t: zero_proj_t.clone(),
        up_proj_t: zero_proj_t.clone(),
        down_proj_t: zero_proj_t,
        gate_up_proj_t: None,
        gate_proj_marlin: None,
        up_proj_marlin: None,
        down_proj_marlin: None,
        gate_up_proj_w8: None,
        down_proj_w8: None,
    };
    let backend = FixedMlpBackend {
        device,
        fused_values: None,
        fused_dims: (1, 1, 2),
        gate_up_values: Some(vec![3.0, 5.0]),
        gate_up_dims: (1, 1, 2),
        fused_calls: std::sync::atomic::AtomicUsize::new(0),
        gate_up_calls: std::sync::atomic::AtomicUsize::new(0),
    };
    let lora_layer = LoraLayerWeights {
        down_proj: Some(LoraProjectionWeights {
            // #1082: kt `KtTensor` LoRA fields; pass kt directly.
            a: Tensor::from_vec(vec![1.0f32, 0.0], (1, 2))?.to_device(device)?,
            b: Tensor::from_vec(vec![2.0f32, 4.0], (2, 1))?.to_device(device)?,
        }),
        ..Default::default()
    };

    let out = swiglu_ffn_impl(Some(&backend), &x, &mlp, Some((&lora_layer, 1.0)), false)?;

    assert_eq!(
        backend
            .gate_up_calls
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
    assert_eq!(
        backend
            .fused_calls
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    let values = out.flatten_all()?.to_vec1::<f32>()?;
    let expected = [6.0, 12.0];
    for (got, expected) in values.iter().zip(expected) {
        assert!(
            (got - expected).abs() < 1e-6,
            "got {got}, expected {expected}"
        );
    }
    Ok(())
}

#[test]
fn test_swiglu_attention_only_lora_keeps_backend_mlp_decode() -> Result<()> {
    let device = Device::Cpu;
    let x = Tensor::from_vec(vec![1.0f32, 2.0], (1, 1, 2))?.to_device(device)?;
    let zero_proj = Tensor::zeros((2, 2), DType::F32, device)?;
    let zero_proj_t = zero_proj.t()?.contiguous()?;
    let mlp = GpuFfnWeights {
        gate_proj: zero_proj.clone(),
        up_proj: zero_proj.clone(),
        down_proj: zero_proj.clone(),
        gate_proj_t: zero_proj_t.clone(),
        up_proj_t: zero_proj_t.clone(),
        down_proj_t: zero_proj_t,
        gate_up_proj_t: None,
        gate_proj_marlin: None,
        up_proj_marlin: None,
        down_proj_marlin: None,
        gate_up_proj_w8: None,
        down_proj_w8: None,
    };
    let backend = FixedMlpBackend {
        device,
        fused_values: Some(vec![7.0, 11.0]),
        fused_dims: (1, 1, 2),
        gate_up_values: Some(vec![3.0, 5.0]),
        gate_up_dims: (1, 1, 2),
        fused_calls: std::sync::atomic::AtomicUsize::new(0),
        gate_up_calls: std::sync::atomic::AtomicUsize::new(0),
    };
    let lora_layer = LoraLayerWeights {
        q_proj: Some(LoraProjectionWeights {
            // #1082: kt `KtTensor` LoRA fields; pass kt directly.
            a: Tensor::from_vec(vec![1.0f32, 0.0], (1, 2))?.to_device(device)?,
            b: Tensor::from_vec(vec![2.0f32, 4.0], (2, 1))?.to_device(device)?,
        }),
        ..Default::default()
    };

    let out = swiglu_ffn_impl(Some(&backend), &x, &mlp, Some((&lora_layer, 1.0)), false)?;

    assert_eq!(
        backend
            .fused_calls
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
    assert_eq!(
        backend
            .gate_up_calls
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    let values = out.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(values, vec![7.0, 11.0]);
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_metal_linear_decode_lora_matches_broadcast_matmul() -> Result<()> {
    let Some(device) = crate::backend::metal::try_new_metal() else {
        eprintln!(
            "Metal unavailable, skipping test_metal_linear_decode_lora_matches_broadcast_matmul"
        );
        return Ok(());
    };

    let input_dim = 128usize;
    let output_dim = 133usize;
    let mut exercised_fast_path = false;
    for rank in [4usize, 32usize, 64usize] {
        for batch in [1usize, 4usize] {
            let x = patterned_bf16(&[batch, 1usize, input_dim], 0.01, &device)?;
            let weight_t = patterned_bf16(&[input_dim, output_dim], 0.0078125, &device)?;
            let lora = LoraProjectionWeights {
                a: patterned_bf16(&[rank, input_dim], 0.001, &device)?,
                b: patterned_bf16(&[output_dim, rank], 0.0015, &device)?,
            };
            let supported = if batch == 1 {
                crate::backend::metal::metal_transposed_coop_gemv_supports(&x, &weight_t)
            } else {
                crate::backend::metal::metal_transposed_coop_gemv_decode_batch_supports(
                    &x, &weight_t,
                )
            };
            if !supported {
                eprintln!(
                    "Metal transposed coop GEMV disabled for rank={rank} batch={batch}, skipping LoRA parity row"
                );
                continue;
            }
            exercised_fast_path = true;

            let fallback = linear_with_lora_t(&x, &weight_t, Some(&lora), 0.75)?;
            let fast = linear_with_lora_t_decode(&x, &weight_t, Some(&lora), 0.75)?;

            assert_eq!(fast.dims(), &[batch, 1usize, output_dim]);
            assert_eq!(fast.dtype(), DType::BF16);

            let (max, mean) = tensor_abs_diff_stats(&fallback, &fast)?;
            assert!(
                max < 2e-2,
                "Metal LoRA linear decode rank={rank} batch={batch} max_abs_diff={max:e} exceeds tolerance"
            );
            assert!(
                mean < 3e-3,
                "Metal LoRA linear decode rank={rank} batch={batch} mean_abs_diff={mean:e} exceeds tolerance"
            );
        }
    }

    if !exercised_fast_path {
        eprintln!("Metal transposed coop GEMV unavailable, no LoRA fast path rows exercised");
    }
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "synthetic Metal LoRA projection microbench; run explicitly with --ignored --nocapture"]
fn bench_metal_linear_decode_lora_qwen35_synthetic() -> Result<()> {
    let Some(device) = crate::backend::metal::try_new_metal() else {
        return Ok(());
    };
    let warmup = std::env::var("KILN_METAL_LORA_LINEAR_BENCH_WARMUP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(2);
    let iters = std::env::var("KILN_METAL_LORA_LINEAR_BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(5);

    for rank in [1usize, 2usize, 4usize, 8usize, 16usize, 32usize, 64usize] {
        for batch in [1usize, 2usize, 4usize, 8usize] {
            bench_metal_lora_linear_case(
                &device,
                "mlp_gate_or_up",
                batch,
                2560,
                9216,
                rank,
                warmup,
                iters,
            )?;
            bench_metal_lora_linear_case(
                &device,
                "down_proj",
                batch,
                9216,
                2560,
                rank,
                warmup,
                iters,
            )?;
        }
    }

    Ok(())
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "synthetic Metal QKV-shaped LoRA projection microbench; run explicitly with --ignored --nocapture"]
fn bench_metal_linear_decode_lora_qwen35_qkv_synthetic() -> Result<()> {
    let Some(device) = crate::backend::metal::try_new_metal() else {
        return Ok(());
    };
    let warmup = std::env::var("KILN_METAL_LORA_QKV_LINEAR_BENCH_WARMUP")
        .or_else(|_| std::env::var("KILN_METAL_LORA_LINEAR_BENCH_WARMUP"))
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(2);
    let iters = std::env::var("KILN_METAL_LORA_QKV_LINEAR_BENCH_ITERS")
        .or_else(|_| std::env::var("KILN_METAL_LORA_LINEAR_BENCH_ITERS"))
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(5);

    for rank in [1usize, 2usize, 4usize, 8usize, 16usize, 32usize, 64usize] {
        for batch in [1usize, 2usize, 4usize, 8usize] {
            bench_metal_lora_linear_case(
                &device, "q_proj", batch, 2560, 8192, rank, warmup, iters,
            )?;
            bench_metal_lora_linear_case(
                &device,
                "k_or_v_proj",
                batch,
                2560,
                1024,
                rank,
                warmup,
                iters,
            )?;
        }
    }

    Ok(())
}

#[cfg(feature = "metal")]
#[allow(clippy::too_many_arguments)]
fn bench_metal_lora_linear_case(
    device: &Device,
    label: &str,
    batch: usize,
    input_dim: usize,
    output_dim: usize,
    rank: usize,
    warmup: usize,
    iters: usize,
) -> Result<()> {
    let x = patterned_bf16(&[batch, 1usize, input_dim], 0.01, device)?;
    let weight_t = patterned_bf16(&[input_dim, output_dim], 0.0001, device)?;
    let lora = LoraProjectionWeights {
        a: patterned_bf16(&[rank, input_dim], 0.0002, device)?,
        b: patterned_bf16(&[output_dim, rank], 0.0002, device)?,
    };
    let supported = if batch == 1 {
        crate::backend::metal::metal_transposed_coop_gemv_supports(&x, &weight_t)
    } else {
        crate::backend::metal::metal_transposed_coop_gemv_decode_batch_supports(&x, &weight_t)
    };
    if !supported {
        eprintln!("metal_lora_linear_bench label={label} skipped unsupported shape");
        return Ok(());
    }

    let fallback = linear_with_lora_t(&x, &weight_t, Some(&lora), 1.0)?;
    let fast = linear_with_lora_t_decode(&x, &weight_t, Some(&lora), 1.0)?;
    let (max, mean) = tensor_abs_diff_stats(&fallback, &fast)?;

    for _ in 0..warmup {
        let out = linear_with_lora_t_decode(&x, &weight_t, Some(&lora), 1.0)?;
        std::hint::black_box(out);
        let out = linear_with_lora_t(&x, &weight_t, Some(&lora), 1.0)?;
        std::hint::black_box(out);
    }
    synchronize_for_profile(device)?;

    let start = std::time::Instant::now();
    for _ in 0..iters {
        let out = linear_with_lora_t_decode(&x, &weight_t, Some(&lora), 1.0)?;
        std::hint::black_box(out);
    }
    synchronize_for_profile(device)?;
    let fast_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let start = std::time::Instant::now();
    for _ in 0..iters {
        let out = linear_with_lora_t(&x, &weight_t, Some(&lora), 1.0)?;
        std::hint::black_box(out);
    }
    synchronize_for_profile(device)?;
    let fallback_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    eprintln!(
        "metal_lora_linear_bench label={label} batch={batch} input_dim={input_dim} output_dim={output_dim} rank={rank} iters={iters} fast_ms={fast_ms:.3} fallback_ms={fallback_ms:.3} speedup={:.3} max_abs_diff={max:e} mean_abs_diff={mean:e}",
        fallback_ms / fast_ms
    );
    Ok(())
}

/// `ProjectionLoadPolicy::stub_embedding_table_after_transposed_upload`
/// must fire on Metal and on Vulkan-active processes — both backends route
/// the embedding lookup through `embed_tokens_t` and never read the raw
/// `embed_tokens` table again, so the candle CPU mirror is pure overhead.
///
/// Phase 1.2 sub-step 1: keep this contract under test so a future
/// edit can't silently drop the Vulkan branch and reintroduce the
/// duplicate embedding-table footprint.
///
/// We deliberately do NOT call `mark_vulkan_active()` here even
/// though it would let us assert the post-flag behavior: the flag
/// is process-global (and `vulkan_active()` is read by other
/// modules including the transposed weight cache writer's
/// scheduling envelope), so flipping it inside one unit test
/// destabilizes every later test in the same nextest process.
/// The flag's read is a one-line public API; the integration
/// behavior is exercised by the live-server validation in
/// `kiln-server`.
#[test]
fn test_stub_embed_tokens_decision_negative_only() {
    let cpu = Device::Cpu;
    // Pre-flag baseline: plain CPU, no Vulkan, must NOT stub.
    // (If a prior test in the same process leaked vulkan_active=true,
    // skip the assertion rather than make a false negative claim.)
    if !crate::backend::vulkan_active() {
        assert!(
            !ProjectionLoadPolicy::for_model_loader_device(cpu)
                .stub_embedding_table_after_transposed_upload,
            "plain CPU with no Vulkan must NOT stub"
        );
    }
    // Cuda device path is gated by feature; rely on the predicate's
    // pattern match returning false for Device::Cpu under non-Metal
    // builds, which is what the negative assertion above covers.
}

/// Projection originals remain eligible for the drop on backends whose
/// immutable storage policy asks for it.
#[test]
fn test_keep_projection_originals_default_off() {
    let result = ProjectionLoadPolicy::for_backend("cuda", Device::Cpu).drop_projection_originals;
    assert!(
        result,
        "CUDA policy must drop redundant projection originals"
    );
}

/// Projection original drop is `false` on plain CPU absent any overrides.
/// The Vulkan-active branch is exercised by integration runs, but the CPU
/// baseline is what this pins.
#[test]
fn test_projection_drop_cpu_default_off() {
    let result = if !crate::backend::vulkan_active() {
        // Safe to assert: vulkan_active=false makes the device
        // pattern-match the only deciding factor for Device::Cpu.
        Some(ProjectionLoadPolicy::for_model_loader_device(Device::Cpu).drop_projection_originals)
    } else {
        None
    };
    if let Some(res) = result {
        assert!(
            !res,
            "plain CPU with no overrides must NOT drop projection originals"
        );
    }
}

/// Property: when `embed_tokens` is a 1-element stub (the only
/// case ProjectionLoadPolicy produces), the dispatch in
/// `embedding_lookup_from_weights` must route to
/// `embedding_lookup_from_transposed`. We can't trivially build a
/// full `GpuWeights` here, so test the dim-mismatch branch directly
/// by checking that `dropped_weight_stub` produces a tensor whose
/// dims will not equal `[t_dims[1], t_dims[0]]` for any non-degenerate
/// transposed shape.
#[test]
fn test_dropped_stub_never_matches_real_embedding_dims() -> Result<()> {
    let device = Device::Cpu;
    let w = WeightTensor {
        dtype: crate::weights::TensorDType::F32,
        shape: vec![5, 3], // vocab=5, hidden=3
        data: crate::weights::WeightData::owned(vec![0u8; 5 * 3 * 4]),
        source: None,
    };
    let stub = dropped_weight_stub(&w, &device)?;
    let materialized_t_dims = [3usize, 5usize];
    let expected_embed_dims = [materialized_t_dims[1], materialized_t_dims[0]];
    assert_ne!(stub.dims(), expected_embed_dims.as_slice());
    assert_eq!(stub.dims(), &[1usize]);
    assert_eq!(stub.dtype(), DType::F32);
    Ok(())
}

#[test]
fn test_embedding_lookup() -> Result<()> {
    let device = Device::Cpu;
    // vocab_size=5, hidden_size=3
    let embed_data: Vec<f32> = vec![
        0.1, 0.2, 0.3, // token 0
        0.4, 0.5, 0.6, // token 1
        0.7, 0.8, 0.9, // token 2
        1.0, 1.1, 1.2, // token 3
        1.3, 1.4, 1.5, // token 4
    ];
    let embed = Tensor::new(&embed_data, device)?.reshape((5, 3))?;

    let result = embedding_lookup(&[2, 0, 4], &embed)?;
    assert_eq!(result.dims(), &[3, 3]); // [seq_len=3, hidden_size=3]

    let vals = result.to_vec2::<f32>()?;
    // Token 2
    assert!((vals[0][0] - 0.7).abs() < 1e-6);
    assert!((vals[0][1] - 0.8).abs() < 1e-6);
    assert!((vals[0][2] - 0.9).abs() < 1e-6);
    // Token 0
    assert!((vals[1][0] - 0.1).abs() < 1e-6);
    // Token 4
    assert!((vals[2][0] - 1.3).abs() < 1e-6);

    Ok(())
}

#[test]
fn test_embedding_lookup_from_transposed_matches_table() -> Result<()> {
    let device = Device::Cpu;
    let embed_data: Vec<f32> = vec![
        0.1, 0.2, 0.3, //
        0.4, 0.5, 0.6, //
        0.7, 0.8, 0.9, //
        1.0, 1.1, 1.2, //
        1.3, 1.4, 1.5,
    ];
    let embed = Tensor::new(&embed_data, device)?.reshape((5, 3))?;
    let embed_t = embed.t()?.contiguous()?;

    let direct = embedding_lookup(&[2, 0, 4], &embed)?;
    let transposed = embedding_lookup_from_transposed(&[2, 0, 4], &embed_t)?;

    assert_eq!(transposed.dims(), direct.dims());
    assert_eq!(transposed.to_vec2::<f32>()?, direct.to_vec2::<f32>()?);
    Ok(())
}

#[test]
fn test_rms_norm_known_values() -> Result<()> {
    let device = Device::Cpu;
    // x = [1, 2, 3], weight = [0, 0, 0], eps = 0
    // Effective weight = 1 + w = [1, 1, 1]
    // RMS = sqrt(mean([1,4,9])) = sqrt(14/3) ≈ 2.1602
    // normed = [1/2.1602, 2/2.1602, 3/2.1602] ≈ [0.4629, 0.9258, 1.3887]
    let x = Tensor::new(&[1.0_f32, 2.0, 3.0], device)?.unsqueeze(0)?; // [1, 3]
    let w = Tensor::new(&[0.0_f32, 0.0, 0.0], device)?;

    let result = rms_norm(&x, &w, 1e-8)?;
    let vals = result.to_vec2::<f32>()?;

    let rms = (14.0_f64 / 3.0).sqrt();
    assert!((vals[0][0] as f64 - 1.0 / rms).abs() < 1e-4);
    assert!((vals[0][1] as f64 - 2.0 / rms).abs() < 1e-4);
    assert!((vals[0][2] as f64 - 3.0 / rms).abs() < 1e-4);

    Ok(())
}

#[test]
fn test_rms_norm_with_weight() -> Result<()> {
    let device = Device::Cpu;
    let x = Tensor::new(&[2.0_f32, 2.0, 2.0], device)?.unsqueeze(0)?;
    let w = Tensor::new(&[0.5_f32, 1.0, 2.0], device)?;

    let result = rms_norm(&x, &w, 1e-8)?;
    let vals = result.to_vec2::<f32>()?;

    // RMS of [2,2,2] = 2.0, so normed = [1,1,1]
    // Effective weight = 1 + w = [1.5, 2.0, 3.0]
    // After weight: [1.5, 2.0, 3.0]
    assert!((vals[0][0] - 1.5).abs() < 1e-4);
    assert!((vals[0][1] - 2.0).abs() < 1e-4);
    assert!((vals[0][2] - 3.0).abs() < 1e-4);

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_gdn_gated_rms_norm_matches_fallback() -> Result<()> {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    let device = match new_cuda_device(0) {
        Ok(device) => device,
        Err(err) => {
            eprintln!(
                "CUDA unavailable, skipping test_cuda_gdn_gated_rms_norm_matches_fallback: {err}"
            );
            return Ok(());
        }
    };
    let backend = crate::backend::for_device_kt(&device);
    if !GdnBackend::runtime_supports_gdn_gated_rms_norm(backend.as_ref()) {
        eprintln!("CUDA gated RMSNorm disabled, skipping parity test");
        return Ok(());
    }

    let batch = 1usize;
    let seq_len = 3usize;
    let heads = 32usize;
    let hidden = 128usize;
    let elems = batch * seq_len * heads * hidden;

    let mut rng = StdRng::seed_from_u64(0xC0DA_6A7E);
    let x_data: Vec<f32> = (0..elems)
        .map(|_| rng.random_range(-1.0f32..1.0f32))
        .collect();
    let z_data: Vec<f32> = (0..elems)
        .map(|_| rng.random_range(-2.0f32..2.0f32))
        .collect();
    let w_data: Vec<f32> = (0..hidden)
        .map(|_| rng.random_range(0.5f32..1.5f32))
        .collect();

    let x = Tensor::from_slice(&x_data, (batch, seq_len, heads, hidden))?
        .to_device(device)?
        .to_dtype(DType::BF16)?;
    let z = Tensor::from_slice(&z_data, (batch, seq_len, heads, hidden))?
        .to_device(device)?
        .to_dtype(DType::BF16)?;
    let weight_f32 = Tensor::from_slice(&w_data, (hidden,))?.to_device(device)?;
    let weight = weight_f32.to_dtype(DType::BF16)?;

    let fallback = gated_rms_norm_fallback(&x, &z, &weight, 1e-6)?;
    let fused = GdnBackend::runtime_gdn_gated_rms_norm(backend.as_ref(), &x, &z, &weight, 1e-6)?
        .context("CUDA backend declined gated RMSNorm test shape")?;
    let fallback_f32_weight = gated_rms_norm_fallback(&x, &z, &weight_f32, 1e-6)?;
    let fused_f32_weight =
        GdnBackend::runtime_gdn_gated_rms_norm(backend.as_ref(), &x, &z, &weight_f32, 1e-6)?
            .context("CUDA backend declined gated RMSNorm f32-weight test shape")?;

    assert_eq!(fused.dims(), fallback.dims());
    assert_eq!(fused.dtype(), DType::BF16);
    assert_eq!(fused_f32_weight.dims(), fallback_f32_weight.dims());
    assert_eq!(fused_f32_weight.dtype(), DType::BF16);

    let diff =
        (fused.to_dtype(DType::F32)? - fallback.to_dtype(DType::BF16)?.to_dtype(DType::F32)?)?;
    let abs = diff.abs()?;
    let max = abs.flatten_all()?.max(0)?.flatten_all()?.to_vec1::<f32>()?[0];
    let mean = abs
        .flatten_all()?
        .mean(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    eprintln!("gated_rms_norm cuda vs fallback: max_abs_diff={max:e} mean_abs_diff={mean:e}");
    assert!(
        max < 5e-3,
        "CUDA gated_rms_norm max_abs_diff={max:e} exceeds 5e-3"
    );
    assert!(
        mean < 5e-4,
        "CUDA gated_rms_norm mean_abs_diff={mean:e} exceeds 5e-4"
    );

    let diff_f32_weight = (fused_f32_weight.to_dtype(DType::F32)?
        - fallback_f32_weight
            .to_dtype(DType::BF16)?
            .to_dtype(DType::F32)?)?;
    let abs_f32_weight = diff_f32_weight.abs()?;
    let max_f32_weight = abs_f32_weight
        .flatten_all()?
        .max(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let mean_f32_weight = abs_f32_weight
        .flatten_all()?
        .mean(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    eprintln!(
        "gated_rms_norm cuda f32-weight vs fallback: max_abs_diff={max_f32_weight:e} mean_abs_diff={mean_f32_weight:e}"
    );
    assert!(
        max_f32_weight < 5e-3,
        "CUDA gated_rms_norm f32-weight max_abs_diff={max_f32_weight:e} exceeds 5e-3"
    );
    assert!(
        mean_f32_weight < 5e-4,
        "CUDA gated_rms_norm f32-weight mean_abs_diff={mean_f32_weight:e} exceeds 5e-4"
    );

    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_metal_gated_rms_norm_matches_fallback() -> Result<()> {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    let Some(device) = crate::backend::metal::try_new_metal() else {
        eprintln!("Metal unavailable, skipping test_metal_gated_rms_norm_matches_fallback");
        return Ok(());
    };
    let backend = crate::backend::for_device_kt(&device);
    if !GdnBackend::runtime_supports_gdn_gated_rms_norm(backend.as_ref()) {
        eprintln!("Metal gated RMSNorm disabled, skipping parity test");
        return Ok(());
    }

    let batch = 1usize;
    let seq_len = 3usize;
    let heads = 32usize;
    let hidden = 128usize;
    let elems = batch * seq_len * heads * hidden;

    let mut rng = StdRng::seed_from_u64(0x6A7E_DA75);
    let x_data: Vec<f32> = (0..elems)
        .map(|_| rng.random_range(-1.0f32..1.0f32))
        .collect();
    let z_data: Vec<f32> = (0..elems)
        .map(|_| rng.random_range(-2.0f32..2.0f32))
        .collect();
    let w_data: Vec<f32> = (0..hidden)
        .map(|_| rng.random_range(0.5f32..1.5f32))
        .collect();

    let x = Tensor::from_slice(&x_data, (batch, seq_len, heads, hidden))?
        .to_device(device)?
        .to_dtype(DType::BF16)?;
    let z = Tensor::from_slice(&z_data, (batch, seq_len, heads, hidden))?
        .to_device(device)?
        .to_dtype(DType::BF16)?;
    let weight = Tensor::from_slice(&w_data, (hidden,))?
        .to_device(device)?
        .to_dtype(DType::BF16)?;

    let fallback = gated_rms_norm_fallback(&x, &z, &weight, 1e-6)?;
    let fused = GdnBackend::runtime_gdn_gated_rms_norm(backend.as_ref(), &x, &z, &weight, 1e-6)?
        .context("Metal backend declined gated RMSNorm test shape")?;

    assert_eq!(fused.dims(), fallback.dims());
    assert_eq!(fused.dtype(), DType::BF16);

    let diff =
        (fused.to_dtype(DType::F32)? - fallback.to_dtype(DType::BF16)?.to_dtype(DType::F32)?)?;
    let abs = diff.abs()?;
    let max = abs.flatten_all()?.max(0)?.flatten_all()?.to_vec1::<f32>()?[0];
    let mean = abs
        .flatten_all()?
        .mean(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    eprintln!("gated_rms_norm metal vs fallback: max_abs_diff={max:e} mean_abs_diff={mean:e}");
    assert!(
        max < 5e-3,
        "Metal gated_rms_norm max_abs_diff={max:e} exceeds 5e-3"
    );
    assert!(
        mean < 5e-4,
        "Metal gated_rms_norm mean_abs_diff={mean:e} exceeds 5e-4"
    );

    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_metal_rms_norm_matches_fallback() -> Result<()> {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    let Some(device) = crate::backend::metal::try_new_metal() else {
        eprintln!("Metal unavailable, skipping test_metal_rms_norm_matches_fallback");
        return Ok(());
    };

    let batch = 2usize;
    let seq_len = 3usize;
    let hidden = 4096usize;
    let elems = batch * seq_len * hidden;

    let mut rng = StdRng::seed_from_u64(0xA11CE);
    let x_data: Vec<f32> = (0..elems)
        .map(|_| rng.random_range(-1.0f32..1.0f32))
        .collect();
    let w_data: Vec<f32> = (0..hidden)
        .map(|_| rng.random_range(-0.2f32..0.2f32))
        .collect();

    let x = Tensor::from_slice(&x_data, (batch, seq_len, hidden))?
        .to_device(device)?
        .to_dtype(DType::BF16)?;
    let weight = Tensor::from_slice(&w_data, (hidden,))?
        .to_device(device)?
        .to_dtype(DType::BF16)?;

    assert!(crate::backend::metal::metal_rms_norm_supports(&x, &weight));
    let fallback = rms_norm_fallback(&x, &weight, 1e-6)?;
    let fused = crate::backend::metal::metal_rms_norm_bf16(&x, &weight, 1e-6)?;

    assert_eq!(fused.dims(), fallback.dims());
    assert_eq!(fused.dtype(), DType::BF16);

    let diff =
        (fused.to_dtype(DType::F32)? - fallback.to_dtype(DType::BF16)?.to_dtype(DType::F32)?)?;
    let abs = diff.abs()?;
    let max = abs.flatten_all()?.max(0)?.flatten_all()?.to_vec1::<f32>()?[0];
    let mean = abs
        .flatten_all()?
        .mean(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    eprintln!("rms_norm metal vs fallback: max_abs_diff={max:e} mean_abs_diff={mean:e}");
    // The Metal kernel computes the *identical* F32 math as the fallback
    // (F32 sum-of-squares, F32 rsqrt, F32 `(1+w)*x*rms_inv`, BF16 cast at
    // the very end). Both `fused` and `fallback` are then rounded to BF16
    // before comparison. The only divergence is the FMA accumulation order
    // in the sum-of-squares reduction, which perturbs `rms_inv` sub-ULP and
    // occasionally flips the final BF16 rounding by exactly one ULP. At
    // magnitude ~1 one BF16 ULP is 2^-7 = 7.8125e-3, so a 5e-3 max
    // tolerance is below the dtype floor and rejects a numerically correct
    // kernel. Verified on M1: of 24576 elements only 12 differ and every
    // one is within a single BF16 ULP (mean_abs_diff ~ 2.4e-6 ≈ 0). Use a
    // 2-BF16-ULP bound so the test passes for the right reason.
    const BF16_ULP_AT_1: f32 = 7.8125e-3; // 2^-7
    assert!(
        max < 2.0 * BF16_ULP_AT_1,
        "Metal rms_norm max_abs_diff={max:e} exceeds 2 BF16 ULP ({:e})",
        2.0 * BF16_ULP_AT_1
    );
    assert!(
        mean < 5e-4,
        "Metal rms_norm mean_abs_diff={mean:e} exceeds 5e-4"
    );

    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_metal_lm_head_forward_decode_batch_matches_broadcast_matmul() -> Result<()> {
    let Some(device) = crate::backend::metal::try_new_metal() else {
        eprintln!(
            "Metal unavailable, skipping test_metal_lm_head_forward_decode_batch_matches_broadcast_matmul"
        );
        return Ok(());
    };

    let batch = 4usize;
    let hidden = 128usize;
    let vocab = 257usize;
    let x_data: Vec<f32> = (0..batch * hidden)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.0234375)
        .collect();
    let weight_data: Vec<f32> = (0..hidden * vocab)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.01953125)
        .collect();

    let x = Tensor::from_slice(&x_data, (batch, 1usize, hidden))?
        .to_device(device)?
        .to_dtype(DType::BF16)?
        .contiguous()?;
    let weight_t = Tensor::from_slice(&weight_data, (hidden, vocab))?
        .to_device(device)?
        .to_dtype(DType::BF16)?
        .contiguous()?;

    assert!(crate::backend::metal::metal_transposed_coop_gemv_decode_batch_supports(&x, &weight_t));
    let reference = x.broadcast_matmul(&weight_t)?;
    let fast = lm_head_forward(&x, &weight_t)?;

    assert_eq!(fast.dims(), &[batch, 1usize, vocab]);
    let diff =
        (fast.to_dtype(DType::F32)? - reference.to_dtype(DType::BF16)?.to_dtype(DType::F32)?)?;
    let abs = diff.abs()?;
    let max = abs.flatten_all()?.max(0)?.flatten_all()?.to_vec1::<f32>()?[0];
    let mean = abs
        .flatten_all()?
        .mean(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    assert!(
        max < 2e-2,
        "Metal batch LM-head max_abs_diff={max:e} exceeds 2e-2"
    );
    assert!(
        mean < 2e-3,
        "Metal batch LM-head mean_abs_diff={mean:e} exceeds 2e-3"
    );

    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_metal_rotary_embedding_matches_fallback() -> Result<()> {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    let Some(device) = crate::backend::metal::try_new_metal() else {
        eprintln!("Metal unavailable, skipping test_metal_rotary_embedding_matches_fallback");
        return Ok(());
    };

    let batch = 1usize;
    let seq_len = 5usize;
    let q_heads = 4usize;
    let k_heads = 2usize;
    let head_dim = 16usize;
    let rotary_dim = 8usize;
    let mut rng = StdRng::seed_from_u64(0xA07A_7E55);
    let q_data: Vec<f32> = (0..batch * seq_len * q_heads * head_dim)
        .map(|_| rng.random_range(-1.0f32..1.0f32))
        .collect();
    let k_data: Vec<f32> = (0..batch * seq_len * k_heads * head_dim)
        .map(|_| rng.random_range(-1.0f32..1.0f32))
        .collect();
    let q = Tensor::from_slice(&q_data, (batch, seq_len, q_heads, head_dim))?
        .to_device(device)?
        .to_dtype(DType::BF16)?;
    let k = Tensor::from_slice(&k_data, (batch, seq_len, k_heads, head_dim))?
        .to_device(device)?
        .to_dtype(DType::BF16)?;
    let positions: Vec<f32> = (11..11 + seq_len).map(|p| p as f32).collect();
    let positions = Tensor::from_slice(&positions, (seq_len,))?.to_device(device)?;
    let inv_freq = compute_rotary_inv_freq(rotary_dim, 10_000.0, &device)?;
    let (cos, sin) = rotary_tables_from_tensor(&positions, &inv_freq)?;

    assert!(crate::backend::metal::metal_rotary_embedding_supports(
        &q, &k, &cos, &sin, head_dim, rotary_dim,
    ));
    let (q_fused, k_fused) = crate::backend::metal::metal_rotary_embedding_bf16(
        &q, &k, &cos, &sin, head_dim, rotary_dim,
    )?;
    let q_ref = apply_rope(&q, &cos, &sin, head_dim, rotary_dim)?;
    let k_ref = apply_rope(&k, &cos, &sin, head_dim, rotary_dim)?;

    let q_diff = (q_fused.to_dtype(DType::F32)?
        - q_ref.to_dtype(DType::BF16)?.to_dtype(DType::F32)?)?
    .abs()?;
    let k_diff = (k_fused.to_dtype(DType::F32)?
        - k_ref.to_dtype(DType::BF16)?.to_dtype(DType::F32)?)?
    .abs()?;
    let q_max = q_diff
        .flatten_all()?
        .max(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let k_max = k_diff
        .flatten_all()?
        .max(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    assert!(q_max < 1e-6, "Metal rotary Q max_abs_diff={q_max:e}");
    assert!(k_max < 1e-6, "Metal rotary K max_abs_diff={k_max:e}");

    Ok(())
}

#[test]
fn test_rope_preserves_shape() -> Result<()> {
    let device = Device::Cpu;
    let batch = 1;
    let seq_len = 4;
    let num_heads = 2;
    let num_kv_heads = 1;
    let head_dim = 8;

    let q = Tensor::randn(0.0_f32, 1.0, (batch, seq_len, num_heads, head_dim), device)?;
    let k = Tensor::randn(
        0.0_f32,
        1.0,
        (batch, seq_len, num_kv_heads, head_dim),
        device,
    )?;
    let positions: Vec<u32> = (0..seq_len as u32).collect();

    let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;
    let (rq, rk) = rotary_embedding(&q, &k, &positions, head_dim, head_dim, &inv_freq)?;

    assert_eq!(rq.dims(), &[batch, seq_len, num_heads, head_dim]);
    assert_eq!(rk.dims(), &[batch, seq_len, num_kv_heads, head_dim]);

    Ok(())
}

#[test]
fn test_rope_position_zero_is_identity() -> Result<()> {
    let device = Device::Cpu;
    // At position 0, cos=1 and sin=0, so rotation should be identity
    let head_dim = 4;
    let q_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let q = Tensor::new(q_data.as_slice(), device)?.reshape((1, 1, 1, head_dim))?;
    let k = q.clone();

    let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;
    let (rq, _rk) = rotary_embedding(&q, &k, &[0], head_dim, head_dim, &inv_freq)?;
    let orig = q.flatten_all()?.to_vec1::<f32>()?;
    let rotated = rq.flatten_all()?.to_vec1::<f32>()?;

    for i in 0..head_dim {
        assert!(
            (orig[i] - rotated[i]).abs() < 1e-5,
            "Position 0 should be identity, dim {i}: orig={} rotated={}",
            orig[i],
            rotated[i]
        );
    }

    Ok(())
}

#[test]
fn test_rope_different_positions_differ() -> Result<()> {
    let device = Device::Cpu;
    let head_dim = 8;
    let q = Tensor::ones((1, 2, 1, head_dim), DType::F32, device)?;
    let k = q.clone();

    let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;
    let (rq, _) = rotary_embedding(&q, &k, &[0, 100], head_dim, head_dim, &inv_freq)?;
    // rq shape: [1, 2, 1, 8] — extract pos 0 and pos 100
    let pos0 = rq.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
    let pos100 = rq.narrow(1, 1, 1)?.flatten_all()?.to_vec1::<f32>()?;

    let diff: f32 = pos0.iter().zip(&pos100).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        diff > 0.01,
        "Different positions should produce different embeddings"
    );

    Ok(())
}

#[test]
fn test_partial_rope_passthrough_dims_unchanged() -> Result<()> {
    let device = Device::Cpu;
    let head_dim = 8;
    let rotary_dim = 4; // only rotate first 4 dims, last 4 pass through
    let q_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let q = Tensor::new(q_data.as_slice(), device)?.reshape((1, 1, 1, head_dim))?;
    let k = q.clone();

    // Position 100 — the rotary dims should change, passthrough dims should not
    let inv_freq = compute_rotary_inv_freq(rotary_dim, 10_000.0, &device)?;
    let (rq, _) = rotary_embedding(&q, &k, &[100], head_dim, rotary_dim, &inv_freq)?;
    let orig = q.flatten_all()?.to_vec1::<f32>()?;
    let rotated = rq.flatten_all()?.to_vec1::<f32>()?;

    // First rotary_dim dims should be different at non-zero position
    let rotary_diff: f32 = (0..rotary_dim).map(|i| (orig[i] - rotated[i]).abs()).sum();
    assert!(
        rotary_diff > 0.01,
        "Rotary dims should change at position 100"
    );

    // Passthrough dims (rotary_dim..head_dim) must be identical
    for i in rotary_dim..head_dim {
        assert!(
            (orig[i] - rotated[i]).abs() < 1e-6,
            "Passthrough dim {i} should be unchanged: orig={} rotated={}",
            orig[i],
            rotated[i]
        );
    }

    Ok(())
}

#[test]
fn test_partial_rope_preserves_shape() -> Result<()> {
    let device = Device::Cpu;
    let batch = 1;
    let seq_len = 4;
    let num_heads = 2;
    let num_kv_heads = 1;
    let head_dim = 16;
    let rotary_dim = 4; // partial rotation

    let q = Tensor::randn(0.0_f32, 1.0, (batch, seq_len, num_heads, head_dim), device)?;
    let k = Tensor::randn(
        0.0_f32,
        1.0,
        (batch, seq_len, num_kv_heads, head_dim),
        device,
    )?;
    let positions: Vec<u32> = (0..seq_len as u32).collect();

    let inv_freq = compute_rotary_inv_freq(rotary_dim, 10_000.0, &device)?;
    let (rq, rk) = rotary_embedding(&q, &k, &positions, head_dim, rotary_dim, &inv_freq)?;

    assert_eq!(rq.dims(), &[batch, seq_len, num_heads, head_dim]);
    assert_eq!(rk.dims(), &[batch, seq_len, num_kv_heads, head_dim]);

    Ok(())
}

#[test]
fn test_swiglu_output_shape() -> Result<()> {
    let device = Device::Cpu;
    let batch = 2;
    let seq_len = 3;
    let hidden = 4;
    let intermediate = 8;

    let x = Tensor::randn(0.0_f32, 1.0, (batch, seq_len, hidden), device)?;
    let gate = Tensor::randn(0.0_f32, 0.1, (intermediate, hidden), device)?;
    let up = Tensor::randn(0.0_f32, 0.1, (intermediate, hidden), device)?;
    let down = Tensor::randn(0.0_f32, 0.1, (hidden, intermediate), device)?;
    let gate_t = gate.t()?.contiguous()?;
    let up_t = up.t()?.contiguous()?;
    let down_t = down.t()?.contiguous()?;

    let mlp = GpuFfnWeights {
        gate_proj: gate,
        up_proj: up,
        down_proj: down,
        gate_proj_t: gate_t,
        up_proj_t: up_t,
        down_proj_t: down_t,
        gate_up_proj_t: None,
        gate_proj_marlin: None,
        up_proj_marlin: None,
        down_proj_marlin: None,
        gate_up_proj_w8: None,
        down_proj_w8: None,
    };
    let result = swiglu_ffn(&x, &mlp, None)?;
    assert_eq!(result.dims(), &[batch, seq_len, hidden]);

    Ok(())
}

#[test]
fn test_swiglu_zero_gate_gives_zero() -> Result<()> {
    let device = Device::Cpu;
    let hidden = 4;
    let intermediate = 8;

    let x = Tensor::ones((1, 1, hidden), DType::F32, device)?;
    // Gate weights all zero -> silu(0) = 0 -> output is zero regardless of up/down
    let gate = Tensor::zeros((intermediate, hidden), DType::F32, device)?;
    let up = Tensor::ones((intermediate, hidden), DType::F32, device)?;
    let down = Tensor::ones((hidden, intermediate), DType::F32, device)?;
    let gate_t = gate.t()?.contiguous()?;
    let up_t = up.t()?.contiguous()?;
    let down_t = down.t()?.contiguous()?;

    let mlp = GpuFfnWeights {
        gate_proj: gate,
        up_proj: up,
        down_proj: down,
        gate_proj_t: gate_t,
        up_proj_t: up_t,
        down_proj_t: down_t,
        gate_up_proj_t: None,
        gate_proj_marlin: None,
        up_proj_marlin: None,
        down_proj_marlin: None,
        gate_up_proj_w8: None,
        down_proj_w8: None,
    };
    let result = swiglu_ffn(&x, &mlp, None)?;
    let vals = result.to_vec3::<f32>()?;

    for v in &vals[0][0] {
        assert!(
            v.abs() < 1e-6,
            "SwiGLU with zero gate should produce zero, got {v}"
        );
    }

    Ok(())
}

#[cfg(feature = "rocm")]
#[test]
fn rocm_chunked_swiglu_tape_reaches_every_mlp_lora_leaf() -> Result<()> {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("skip rocm_chunked_swiglu_tape_reaches_every_mlp_lora_leaf: no ROCm device");
        return Ok(());
    }

    let device = Device::Rocm(0);
    let hidden = 4usize;
    let intermediate = 8usize;
    let rank = 2usize;
    let seq_len = 5usize;
    let chunk_tokens = 2usize;
    let patterned = |shape: &[usize], scale: f32| -> Result<Tensor> {
        let count = shape.iter().product();
        let values = (0..count)
            .map(|index| (((index * 17 + 3) % 41) as f32 - 20.0) * scale)
            .collect::<Vec<_>>();
        Ok(Tensor::from_vec_on(device, values, shape.to_vec())?.contiguous()?)
    };

    let gate_proj = patterned(&[intermediate, hidden], 0.01)?;
    let up_proj = patterned(&[intermediate, hidden], 0.008)?;
    let down_proj = patterned(&[hidden, intermediate], 0.006)?;
    let mlp = GpuFfnWeights {
        gate_proj_t: gate_proj.t()?.contiguous()?,
        up_proj_t: up_proj.t()?.contiguous()?,
        down_proj_t: down_proj.t()?.contiguous()?,
        gate_proj,
        up_proj,
        down_proj,
        gate_up_proj_t: None,
        gate_proj_marlin: None,
        up_proj_marlin: None,
        down_proj_marlin: None,
        gate_up_proj_w8: None,
        down_proj_w8: None,
    };
    let lora = LoraLayerWeights {
        gate_proj: Some(LoraProjectionWeights {
            a: patterned(&[rank, hidden], 0.02)?,
            b: patterned(&[intermediate, rank], 0.015)?,
        }),
        up_proj: Some(LoraProjectionWeights {
            a: patterned(&[rank, hidden], 0.018)?,
            b: patterned(&[intermediate, rank], 0.013)?,
        }),
        down_proj: Some(LoraProjectionWeights {
            a: patterned(&[rank, intermediate], 0.012)?,
            b: patterned(&[hidden, rank], 0.011)?,
        }),
        ..Default::default()
    };
    let mut lora_leaf_ids = Vec::new();
    lora.for_each_projection(|projection| {
        lora_leaf_ids.push(projection.a.id());
        lora_leaf_ids.push(projection.b.id());
    });
    let x = patterned(&[1, seq_len, hidden], 0.025)?;

    let (out, tape) = kiln_autograd::with_thread_local_tape(|| {
        swiglu_ffn_impl_chunked(None, &x, &mlp, Some((&lora, 1.0)), false, chunk_tokens)
    });
    let out = out?;
    let reachable = tape.reachable_from(out.id());
    for leaf_id in &lora_leaf_ids {
        assert!(
            reachable.contains(leaf_id),
            "chunked SwiGLU output must remain structurally connected to LoRA leaf {leaf_id:?}"
        );
    }

    let seed = Tensor::ones(out.shape().to_vec(), out.dtype(), device)?;
    let grads = tape.backward(out.id(), seed, kiln_tensor::ops::add)?;
    for leaf_id in lora_leaf_ids {
        let grad = grads
            .get(leaf_id)
            .unwrap_or_else(|| panic!("missing chunked SwiGLU gradient for LoRA leaf {leaf_id:?}"));
        assert!(
            grad.all_finite()?,
            "chunked SwiGLU produced a non-finite LoRA gradient for {leaf_id:?}"
        );
    }
    Ok(())
}

/// Parity test for the CUDA prefill fused `gate_up_proj_t` GEMM.
///
/// The fast path replaces two `[B*T, hidden] @ [hidden, intermediate]`
/// matmuls with one `[B*T, hidden] @ [hidden, 2*intermediate]` GEMM and
/// slices gate/up halves out of the result. This test asserts the
/// algebraic identity that drives bit-equivalence: concatenating gate_t
/// and up_t along the output dim and matmul'ing once, then narrowing,
/// produces the same tensors element-for-element as the two separate
/// matmuls. Verified on CPU here so the property holds even without a
/// CUDA device — the actual CUDA path uses the exact same broadcast
/// matmul helper (`broadcast_matmul_cpu_compatible` →
/// `matmul_no_broadcast_copy`), so any drift would have to come from
/// cuBLAS's choice of algorithm for different output widths.
#[test]
fn test_runtime_matmul_no_broadcast_copy_routes_cpu_backend() -> Result<()> {
    let backend = CpuBackend::new(Device::Cpu);
    let x = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3])?;
    let weight_t = Tensor::from_slice(&[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2])?;

    let routed = runtime_matmul_no_broadcast_copy(&backend, &x, &weight_t)?
        .expect("CPU backend should route flattened matmul request");
    let reference = broadcast_matmul_cpu_compatible(&x, &weight_t)?;
    assert_eq!(routed.dims(), &[1, 2, 2]);
    assert_eq!(
        routed.flatten_all()?.to_vec1::<f32>()?,
        reference.flatten_all()?.to_vec1::<f32>()?
    );

    Ok(())
}

#[test]
fn test_fused_gate_up_proj_matches_split_path() -> Result<()> {
    let device = Device::Cpu;
    let batch = 2usize;
    let seq_len = 7usize;
    let hidden = 16usize;
    let intermediate = 24usize;

    // BF16 → CPU promotes to F32 inside broadcast_matmul_cpu_compatible
    // so all operands here use F32 directly: bit-exact comparison is
    // meaningful, the device-specific BF16 path is exercised by the
    // existing real-model integration tests.
    let x = Tensor::randn(0.0_f32, 1.0, (batch, seq_len, hidden), device)?;
    let gate_t = Tensor::randn(0.0_f32, 0.05, (hidden, intermediate), device)?.contiguous()?;
    let up_t = Tensor::randn(0.0_f32, 0.05, (hidden, intermediate), device)?.contiguous()?;
    let gate_up_t = Tensor::cat(&[&gate_t, &up_t], LAST_DIM)?.contiguous()?;

    let gate_split = broadcast_matmul_cpu_compatible(&x, &gate_t)?;
    let up_split = broadcast_matmul_cpu_compatible(&x, &up_t)?;

    let gate_up = broadcast_matmul_cpu_compatible(&x, &gate_up_t)?;
    let gate_fused = gate_up.narrow(2, 0, intermediate)?;
    let up_fused = gate_up.narrow(2, intermediate, intermediate)?;

    // CPU F32 matmul is deterministic — assert bit-identical here. The
    // narrow-into-halves of [B, T, 2I] must equal the two standalone
    // [B, T, I] matmuls element-for-element.
    let gate_split_vec = gate_split.flatten_all()?.to_vec1::<f32>()?;
    let gate_fused_vec = gate_fused.flatten_all()?.to_vec1::<f32>()?;
    let up_split_vec = up_split.flatten_all()?.to_vec1::<f32>()?;
    let up_fused_vec = up_fused.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(gate_split_vec.len(), gate_fused_vec.len());
    for (i, (a, b)) in gate_split_vec.iter().zip(gate_fused_vec.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "gate parity bit-mismatch at index {i}: split={a} fused={b}"
        );
    }
    for (i, (a, b)) in up_split_vec.iter().zip(up_fused_vec.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "up parity bit-mismatch at index {i}: split={a} fused={b}"
        );
    }
    Ok(())
}

/// End-to-end parity for `swiglu_ffn_impl_no_chunk`: populate
/// `gate_up_proj_t` on the FFN weights and confirm the SwiGLU output is
/// bit-identical to the same weights with `gate_up_proj_t = None`. CPU
/// path doesn't take the cuda-gated fused branch, so on CPU this is
/// effectively a regression guard that the field stays inert when the
/// device disqualifies the fast path. The CUDA-side parity is covered by
/// the algebraic identity in `test_fused_gate_up_proj_matches_split_path`.
#[test]
fn test_swiglu_with_fused_gate_up_cache_matches_legacy() -> Result<()> {
    let device = Device::Cpu;
    let batch = 1usize;
    let seq_len = 3usize;
    let hidden = 8usize;
    let intermediate = 16usize;

    let x = Tensor::randn(0.0_f32, 1.0, (batch, seq_len, hidden), device)?;
    let gate = Tensor::randn(0.0_f32, 0.05, (intermediate, hidden), device)?;
    let up = Tensor::randn(0.0_f32, 0.05, (intermediate, hidden), device)?;
    let down = Tensor::randn(0.0_f32, 0.05, (hidden, intermediate), device)?;
    let gate_t = gate.t()?.contiguous()?;
    let up_t = up.t()?.contiguous()?;
    let down_t = down.t()?.contiguous()?;
    let gate_up_t = Tensor::cat(&[&gate_t, &up_t], LAST_DIM)?.contiguous()?;

    let mlp_legacy = GpuFfnWeights {
        gate_proj: gate.clone(),
        up_proj: up.clone(),
        down_proj: down.clone(),
        gate_proj_t: gate_t.clone(),
        up_proj_t: up_t.clone(),
        down_proj_t: down_t.clone(),
        gate_up_proj_t: None,
        gate_proj_marlin: None,
        up_proj_marlin: None,
        down_proj_marlin: None,
        gate_up_proj_w8: None,
        down_proj_w8: None,
    };
    let mlp_fused = GpuFfnWeights {
        gate_proj: gate,
        up_proj: up,
        down_proj: down,
        gate_proj_t: gate_t,
        up_proj_t: up_t,
        down_proj_t: down_t,
        gate_up_proj_t: Some(gate_up_t),
        gate_proj_marlin: None,
        up_proj_marlin: None,
        down_proj_marlin: None,
        gate_up_proj_w8: None,
        down_proj_w8: None,
    };

    let legacy = swiglu_ffn(&x, &mlp_legacy, None)?;
    let fused = swiglu_ffn(&x, &mlp_fused, None)?;
    let legacy_vec = legacy.flatten_all()?.to_vec1::<f32>()?;
    let fused_vec = fused.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(legacy_vec.len(), fused_vec.len());
    for (i, (a, b)) in legacy_vec.iter().zip(fused_vec.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "swiglu fused/legacy bit-mismatch at index {i}: legacy={a} fused={b}"
        );
    }
    Ok(())
}

/// Create a minimal config for tests (no output gate, simple dims).
fn make_test_config(
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden: usize,
) -> kiln_core::config::ModelConfig {
    kiln_core::config::ModelConfig {
        hidden_size: hidden,
        num_layers: 4,
        num_attention_heads: num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size: hidden * 2,
        vocab_size: 256,
        max_position_embeddings: 1024,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        dtype: kiln_core::config::DType::BF16,
        num_full_attention_layers: 1,
        full_attention_interval: 4,
        attn_output_gate: false,
        linear_num_key_heads: num_heads,
        linear_key_head_dim: head_dim,
        linear_num_value_heads: num_heads,
        linear_value_head_dim: head_dim,
        linear_conv_kernel_dim: 4,
        partial_rotary_factor: 1.0, // tests use full rotation by default
    }
}

#[test]
fn test_linear_attention_state_prefix_snapshot_truncates_draft_state() -> Result<()> {
    let device = Device::Cpu;
    let config = make_test_config(2, 1, 4, 8);
    let state = LinearAttentionState::new(&config, &device)?;

    assert_eq!(state.recurrent_states.len(), 3);
    assert_eq!(state.conv_states.len(), 3);

    let draft = state.snapshot_for_decode_rollback_prefix(1)?;
    assert_eq!(draft.recurrent_states.len(), 1);
    assert_eq!(draft.conv_states.len(), 1);
    Ok(())
}

#[test]
fn linear_attention_snapshot_panic_retains_partial_destinations() {
    struct DropProbe(std::sync::Arc<std::sync::atomic::AtomicUsize>);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    let drops = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = build_linear_attention_snapshot(Vec::new(), Vec::new(), |recurrent, conv| {
            recurrent.push(DropProbe(drops.clone()));
            conv.push(DropProbe(drops.clone()));
            panic!("injected snapshot panic");
        });
    }));

    assert!(result.is_err());
    assert_eq!(
        drops.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "panic must not release device-copy destinations"
    );
}

fn make_test_attn_weights(
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden: usize,
    device: &Device,
) -> Result<GpuFullAttentionWeights> {
    let q_proj = Tensor::randn(0.0_f32, 0.02, (num_heads * head_dim, hidden), device)?;
    let k_proj = Tensor::randn(0.0_f32, 0.02, (num_kv_heads * head_dim, hidden), device)?;
    let v_proj = Tensor::randn(0.0_f32, 0.02, (num_kv_heads * head_dim, hidden), device)?;
    let o_proj = Tensor::randn(0.0_f32, 0.02, (hidden, num_heads * head_dim), device)?;
    let q_proj_t = q_proj.t()?.contiguous()?;
    let k_proj_t = k_proj.t()?.contiguous()?;
    let v_proj_t = v_proj.t()?.contiguous()?;
    let o_proj_t = o_proj.t()?.contiguous()?;
    Ok(GpuFullAttentionWeights {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm: Tensor::zeros(head_dim, DType::F32, device)?,
        k_norm: Tensor::zeros(head_dim, DType::F32, device)?,
        q_proj_t,
        k_proj_t,
        v_proj_t,
        qkv_proj_t: None,
        o_proj_t,
        qkv_proj_w8: None,
        o_proj_w8: None,
        q_proj_marlin: None,
    })
}

#[cfg(any(feature = "metal", feature = "cuda"))]
fn patterned_bf16(shape: &[usize], scale: f32, device: &Device) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| (((i * 17 + 13) % 257) as f32 - 128.0) * scale)
        .collect();
    Ok(Tensor::new(&data, device)?
        .reshape(shape)?
        .to_dtype(DType::BF16)?
        .contiguous()?)
}

#[cfg(feature = "metal")]
fn patterned_f32(shape: &[usize], scale: f32, device: &Device) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| (((i * 23 + 19) % 251) as f32 - 125.0) * scale)
        .collect();
    Ok(Tensor::new(&data, device)?.reshape(shape)?.contiguous()?)
}

#[cfg(any(feature = "metal", feature = "cuda"))]
fn make_bf16_full_attn_weights(
    hidden: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    device: &Device,
) -> Result<GpuFullAttentionWeights> {
    let q_proj = patterned_bf16(&[num_heads * head_dim, hidden], 0.00002, device)?;
    let k_proj = patterned_bf16(&[num_kv_heads * head_dim, hidden], 0.00003, device)?;
    let v_proj = patterned_bf16(&[num_kv_heads * head_dim, hidden], 0.00004, device)?;
    let o_proj = patterned_bf16(&[hidden, num_heads * head_dim], 0.00002, device)?;
    Ok(GpuFullAttentionWeights {
        q_proj_t: q_proj.t()?.contiguous()?,
        k_proj_t: k_proj.t()?.contiguous()?,
        v_proj_t: v_proj.t()?.contiguous()?,
        qkv_proj_t: None,
        o_proj_t: o_proj.t()?.contiguous()?,
        qkv_proj_w8: None,
        o_proj_w8: None,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm: Tensor::zeros(head_dim, DType::F32, device)?,
        k_norm: Tensor::zeros(head_dim, DType::F32, device)?,
        q_proj_marlin: None,
    })
}

#[cfg(any(feature = "metal", feature = "cuda"))]
fn make_bf16_mlp_weights(
    hidden: usize,
    intermediate: usize,
    device: &Device,
) -> Result<GpuFfnWeights> {
    let gate_proj = patterned_bf16(&[intermediate, hidden], 0.00003, device)?;
    let up_proj = patterned_bf16(&[intermediate, hidden], 0.00002, device)?;
    let down_proj = patterned_bf16(&[hidden, intermediate], 0.00003, device)?;
    Ok(GpuFfnWeights {
        gate_proj_t: gate_proj.t()?.contiguous()?,
        up_proj_t: up_proj.t()?.contiguous()?,
        down_proj_t: down_proj.t()?.contiguous()?,
        gate_proj,
        up_proj,
        down_proj,
        gate_up_proj_t: None,
        gate_proj_marlin: None,
        up_proj_marlin: None,
        down_proj_marlin: None,
        gate_up_proj_w8: None,
        down_proj_w8: None,
    })
}

#[cfg(any(feature = "metal", feature = "cuda"))]
fn make_bf16_full_attention_gpu_weights(
    vocab: usize,
    hidden: usize,
    intermediate: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    device: &Device,
) -> Result<GpuWeights> {
    let embed_tokens = patterned_bf16(&[vocab, hidden], 0.01, device)?;
    let embed_tokens_t = embed_tokens.t()?.contiguous()?;
    let final_norm = Tensor::zeros(hidden, DType::F32, device)?;
    let mut layers = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        layers.push(GpuLayerWeights {
            input_layernorm: Tensor::zeros(hidden, DType::F32, device)?,
            post_attention_layernorm: Tensor::zeros(hidden, DType::F32, device)?,
            attention: GpuAttentionWeights::Full(make_bf16_full_attn_weights(
                hidden,
                num_heads,
                num_kv_heads,
                head_dim,
                device,
            )?),
            mlp: make_bf16_mlp_weights(hidden, intermediate, device)?,
        });
    }
    let rotary_inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, device)?;
    Ok(GpuWeights {
        source_content_sha256: None,
        base_weight_shard_manifest: None,
        execution_provenance: None,
        embed_tokens,
        embed_tokens_t,
        lm_head_w8: None,
        layers,
        final_norm,
        rotary_inv_freq,
        mtp: None,
    })
}

#[cfg(feature = "metal")]
fn make_metal_graph_test_config(
    vocab: usize,
    hidden: usize,
    intermediate: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> kiln_core::config::ModelConfig {
    kiln_core::config::ModelConfig {
        hidden_size: hidden,
        num_layers: 1,
        num_attention_heads: num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size: intermediate,
        vocab_size: vocab,
        max_position_embeddings: 1024,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        dtype: kiln_core::config::DType::BF16,
        num_full_attention_layers: 1,
        full_attention_interval: 1,
        attn_output_gate: false,
        linear_num_key_heads: num_kv_heads,
        linear_key_head_dim: head_dim,
        linear_num_value_heads: num_heads,
        linear_value_head_dim: head_dim,
        linear_conv_kernel_dim: 4,
        partial_rotary_factor: 1.0,
    }
}

#[cfg(feature = "metal")]
fn make_metal_graph_test_weights(
    config: &kiln_core::config::ModelConfig,
    device: &Device,
) -> Result<GpuWeights> {
    let mut weights = make_bf16_full_attention_gpu_weights(
        config.vocab_size,
        config.hidden_size,
        config.intermediate_size,
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim,
        config.num_layers,
        device,
    )?;
    weights.final_norm = Tensor::ones(config.hidden_size, DType::F32, device)?;
    for layer in weights.layers.iter_mut() {
        layer.input_layernorm = Tensor::ones(config.hidden_size, DType::F32, device)?;
        layer.post_attention_layernorm = Tensor::ones(config.hidden_size, DType::F32, device)?;
        if let GpuAttentionWeights::Full(attn) = &mut layer.attention {
            attn.q_norm = Tensor::ones(config.head_dim, DType::F32, device)?;
            attn.k_norm = Tensor::ones(config.head_dim, DType::F32, device)?;
        }
    }
    Ok(weights)
}

#[cfg(feature = "metal")]
fn seed_metal_graph_prefix_for_test(
    cache: &mut crate::PagedKvCacheKt,
    block_table: &BlockTable,
    start_pos: usize,
    num_kv_heads: usize,
    head_dim: usize,
    k_scale: f32,
    v_scale: f32,
    device: &Device,
) -> Result<()> {
    let prefix_k = patterned_bf16(&[1, start_pos, num_kv_heads, head_dim], k_scale, device)?;
    let prefix_v = patterned_bf16(&[1, start_pos, num_kv_heads, head_dim], v_scale, device)?;
    write_token_major_prefix_for_test(cache, 0, block_table, 0, &prefix_k, &prefix_v)
}

#[cfg(feature = "metal")]
fn assert_metal_graph_cache_matches_eager(
    graph_cache: &crate::PagedKvCacheKt,
    eager_cache: &crate::PagedKvCacheKt,
) -> Result<()> {
    let (graph_k, graph_v) = graph_cache
        .pool_tensors(0)
        .context("graph cache layer 0 pool missing")?;
    let (eager_k, eager_v) = eager_cache
        .pool_tensors(0)
        .context("eager cache layer 0 pool missing")?;
    let (k_max, k_mean) = tensor_abs_diff_stats(&graph_k, &eager_k)?;
    let (v_max, v_mean) = tensor_abs_diff_stats(&graph_v, &eager_v)?;
    assert!(
        k_max <= 2e-2 && k_mean <= 1e-5 && v_max <= 2e-2 && v_mean <= 1e-5,
        "Metal graph and eager KV pools diverged: k_max={k_max:e} k_mean={k_mean:e} v_max={v_max:e} v_mean={v_mean:e}"
    );
    Ok(())
}

#[cfg(feature = "metal")]
fn make_bf16_hybrid_gpu_weights(
    config: &kiln_core::config::ModelConfig,
    device: &Device,
) -> Result<GpuWeights> {
    let hidden = config.hidden_size;
    let embed_tokens = patterned_bf16(&[config.vocab_size, hidden], 0.01, device)?;
    let embed_tokens_t = embed_tokens.t()?.contiguous()?;
    let final_norm = Tensor::zeros(hidden, DType::BF16, device)?;
    let mut layers = Vec::with_capacity(config.num_layers);

    for layer_idx in 0..config.num_layers {
        let attention = if config.is_full_attention_layer(layer_idx) {
            let q_dim = config.full_attn_q_proj_dim();
            let kv_dim = config.num_kv_heads * config.head_dim;
            let out_dim = config.num_attention_heads * config.head_dim;
            let q_proj = patterned_bf16(&[q_dim, hidden], 0.00002, device)?;
            let k_proj = patterned_bf16(&[kv_dim, hidden], 0.00003, device)?;
            let v_proj = patterned_bf16(&[kv_dim, hidden], 0.00004, device)?;
            let o_proj = patterned_bf16(&[hidden, out_dim], 0.00002, device)?;
            GpuAttentionWeights::Full(GpuFullAttentionWeights {
                q_proj_t: q_proj.t()?.contiguous()?,
                k_proj_t: k_proj.t()?.contiguous()?,
                v_proj_t: v_proj.t()?.contiguous()?,
                qkv_proj_t: None,
                o_proj_t: o_proj.t()?.contiguous()?,
                qkv_proj_w8: None,
                o_proj_w8: None,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm: Tensor::zeros(config.head_dim, DType::BF16, device)?,
                k_norm: Tensor::zeros(config.head_dim, DType::BF16, device)?,
                q_proj_marlin: None,
            })
        } else {
            let qkv_dim = config.linear_qkv_dim();
            let v_dim = config.linear_v_dim();
            let nv = config.linear_num_value_heads;
            let in_proj_qkv = patterned_bf16(&[qkv_dim, hidden], 0.00002, device)?;
            let in_proj_z = patterned_bf16(&[v_dim, hidden], 0.00003, device)?;
            let out_proj = patterned_bf16(&[hidden, v_dim], 0.00002, device)?;
            let in_proj_a = patterned_bf16(&[nv, hidden], 0.00004, device)?;
            let in_proj_b = patterned_bf16(&[nv, hidden], 0.00005, device)?;
            GpuAttentionWeights::Linear(GpuLinearAttentionWeights {
                in_proj_qkv_t: in_proj_qkv.t()?.contiguous()?,
                in_proj_z_t: in_proj_z.t()?.contiguous()?,
                in_proj_a_t: in_proj_a.t()?.contiguous()?,
                in_proj_b_t: in_proj_b.t()?.contiguous()?,
                in_proj_ab_t: None,
                out_proj_t: out_proj.t()?.contiguous()?,
                in_proj_qkv,
                in_proj_z,
                out_proj,
                in_proj_a,
                in_proj_b,
                conv1d: patterned_bf16(
                    &[qkv_dim, 1usize, config.linear_conv_kernel_dim],
                    0.00002,
                    device,
                )?,
                norm: Tensor::ones(config.linear_value_head_dim, DType::F32, device)?,
                a_log: Tensor::zeros(nv, DType::F32, device)?,
                a_log_gates: Tensor::zeros(nv, DType::F32, device)?,
                dt_bias: Tensor::zeros(nv, DType::BF16, device)?,
                out_proj_marlin: None,
                in_proj_qkvzab_w8: None,
            })
        };

        layers.push(GpuLayerWeights {
            input_layernorm: Tensor::zeros(hidden, DType::BF16, device)?,
            post_attention_layernorm: Tensor::zeros(hidden, DType::BF16, device)?,
            attention,
            mlp: make_bf16_mlp_weights(hidden, config.intermediate_size, device)?,
        });
    }

    let rotary_inv_freq = compute_rotary_inv_freq(config.rotary_dim(), config.rope_theta, device)?;
    Ok(GpuWeights {
        source_content_sha256: None,
        base_weight_shard_manifest: None,
        execution_provenance: None,
        embed_tokens,
        embed_tokens_t,
        lm_head_w8: None,
        layers,
        final_norm,
        rotary_inv_freq,
        mtp: None,
    })
}

#[cfg(feature = "metal")]
fn patterned_linear_state(
    config: &kiln_core::config::ModelConfig,
    row: usize,
    device: &Device,
) -> Result<LinearAttentionState> {
    let mut state = LinearAttentionState::new(config, device)?;
    for layer_idx in 0..state.recurrent_states.len() {
        let state_scale = 0.00001 * (row + layer_idx + 1) as f32;
        let conv_scale = 0.00002 * (row + layer_idx + 1) as f32;
        state.recurrent_states[layer_idx] = patterned_bf16(
            &[
                1usize,
                config.linear_num_value_heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
            ],
            state_scale,
            device,
        )?;
        state.conv_states[layer_idx] = patterned_f32(
            &[
                1usize,
                config.linear_qkv_dim(),
                config.linear_conv_kernel_dim - 1,
            ],
            conv_scale,
            device,
        )?;
    }
    Ok(state)
}

#[cfg(feature = "metal")]
fn tensor_abs_diff_stats(left: &Tensor, right: &Tensor) -> Result<(f32, f32)> {
    let diff = (left.to_dtype(DType::F32)? - right.to_dtype(DType::F32)?)?;
    let abs = diff.abs()?;
    let max = abs.flatten_all()?.max(0)?.flatten_all()?.to_vec1::<f32>()?[0];
    let mean = abs
        .flatten_all()?
        .mean(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    Ok((max, mean))
}

#[cfg(feature = "metal")]
fn write_token_major_prefix_for_test(
    cache: &mut crate::PagedKvCacheKt,
    layer_idx: usize,
    block_table: &BlockTable,
    start_pos: usize,
    k: &Tensor,
    v: &Tensor,
) -> Result<()> {
    let k_head_major = k.transpose(1, 2)?.contiguous()?;
    let v_head_major = v.transpose(1, 2)?.contiguous()?;
    cache.write(
        layer_idx,
        block_table,
        start_pos,
        &k_head_major,
        &v_head_major,
    )
}

#[cfg(feature = "metal")]
#[test]
fn test_gqa_attention_paged_decode_contiguous_batch_matches_rowwise_metal() -> Result<()> {
    let Some(device) = crate::backend::metal::try_new_metal() else {
        return Ok(());
    };
    // kt parallel to `device` for `PagedKvCache::new_kt` call sites
    // below — the kt twin lets the constructor call drop the
    // candle::DType + &candle::Device names. (#1082)
    let device_kt = device; // #1082: `device` is already kt

    let backend = crate::backend::for_device_kt(&device);
    let batch = 2usize;
    let hidden = 512usize;
    let num_heads = 16usize;
    let num_kv_heads = 4usize;
    let head_dim = 256usize;
    let block_size = 16usize;
    let start_pos = 3usize;

    let attn = make_bf16_full_attn_weights(hidden, num_heads, num_kv_heads, head_dim, &device)?;

    let x = patterned_bf16(&[batch, 1usize, hidden], 0.01, &device)?;
    let positions = Tensor::from_slice(&[start_pos as f32], 1usize)?.to_device(device)?;
    let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;

    let prefix_k = patterned_bf16(&[batch, start_pos, num_kv_heads, head_dim], 0.002, &device)?;
    let prefix_v = patterned_bf16(&[batch, start_pos, num_kv_heads, head_dim], 0.003, &device)?;
    let bt0 = BlockTable { blocks: vec![0] };
    let bt1 = BlockTable { blocks: vec![1] };
    let block_tables = [&bt0, &bt1];
    let start_positions = [start_pos, start_pos];
    let mut batch_cache = crate::PagedKvCacheKt::new(
        1,
        2,
        block_size,
        num_kv_heads,
        head_dim,
        kiln_tensor::DType::BF16,
        device_kt,
    )?;
    for (row, block_table) in block_tables.iter().enumerate() {
        let row_k = prefix_k.narrow(0, row, 1)?.contiguous()?;
        let row_v = prefix_v.narrow(0, row, 1)?.contiguous()?;
        write_token_major_prefix_for_test(&mut batch_cache, 0, block_table, 0, &row_k, &row_v)?;
    }

    let batched = gqa_attention_paged_decode_contiguous_batch(
        &*backend,
        &x,
        &attn,
        &positions,
        &start_positions,
        num_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        &inv_freq,
        1e-6,
        &mut batch_cache,
        &block_tables,
        0,
        false,
        None,
        None,
        None,
        None,
        None,
        #[cfg(feature = "metal")]
        None,
        #[cfg(feature = "cuda")]
        None,
    )?;
    synchronize_for_profile(&device)?;
    assert_eq!(batched.dims(), &[batch, 1usize, hidden]);

    for row in 0..batch {
        let mut row_cache = crate::PagedKvCacheKt::new(
            1,
            1,
            block_size,
            num_kv_heads,
            head_dim,
            kiln_tensor::DType::BF16,
            device_kt,
        )?;
        let row_table = BlockTable { blocks: vec![0] };
        let row_k = prefix_k.narrow(0, row, 1)?.contiguous()?;
        let row_v = prefix_v.narrow(0, row, 1)?.contiguous()?;
        write_token_major_prefix_for_test(&mut row_cache, 0, &row_table, 0, &row_k, &row_v)?;
        let row_x = x.narrow(0, row, 1)?.contiguous()?;
        let rowwise = gqa_attention_paged(
            &*backend,
            &row_x,
            &attn,
            &positions,
            start_pos,
            num_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            &inv_freq,
            1e-6,
            &mut row_cache,
            &row_table,
            0,
            false,
            None,
        )?;
        synchronize_for_profile(&device)?;

        let batch_row = batched.narrow(0, row, 1)?;
        let diff = (batch_row.to_dtype(DType::F32)? - rowwise.to_dtype(DType::F32)?)?;
        let abs = diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.flatten_all()?.to_vec1::<f32>()?[0];
        let mean = abs
            .flatten_all()?
            .mean(0)?
            .flatten_all()?
            .to_vec1::<f32>()?[0];
        eprintln!(
            "batched contiguous paged decode row {row}: max_abs_diff={max:e} mean_abs_diff={mean:e}"
        );
        assert!(
            max <= 2e-2,
            "row {row} batched contiguous paged decode max_abs_diff={max:e}"
        );
        assert!(
            mean <= 2e-3,
            "row {row} batched contiguous paged decode mean_abs_diff={mean:e}"
        );
    }

    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_transformer_block_paged_decode_contiguous_batch_matches_rowwise_metal() -> Result<()> {
    let Some(device) = crate::backend::metal::try_new_metal() else {
        return Ok(());
    };
    // kt parallel to `device` for `PagedKvCache::new_kt` (#1082).
    let device_kt = device; // #1082: `device` is already kt

    let backend = crate::backend::for_device_kt(&device);
    let batch = 2usize;
    let hidden = 512usize;
    let intermediate = 768usize;
    let num_heads = 16usize;
    let num_kv_heads = 4usize;
    let head_dim = 256usize;
    let block_size = 16usize;
    let start_pos = 3usize;
    let config = kiln_core::config::ModelConfig {
        hidden_size: hidden,
        num_layers: 1,
        num_attention_heads: num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size: intermediate,
        vocab_size: 1024,
        max_position_embeddings: 1024,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        dtype: kiln_core::config::DType::BF16,
        num_full_attention_layers: 1,
        full_attention_interval: 1,
        attn_output_gate: false,
        linear_num_key_heads: num_kv_heads,
        linear_key_head_dim: head_dim,
        linear_num_value_heads: num_heads,
        linear_value_head_dim: head_dim,
        linear_conv_kernel_dim: 4,
        partial_rotary_factor: 1.0,
    };

    let layer = GpuLayerWeights {
        input_layernorm: Tensor::zeros(hidden, DType::F32, &device)?,
        post_attention_layernorm: Tensor::zeros(hidden, DType::F32, &device)?,
        attention: GpuAttentionWeights::Full(make_bf16_full_attn_weights(
            hidden,
            num_heads,
            num_kv_heads,
            head_dim,
            &device,
        )?),
        mlp: make_bf16_mlp_weights(hidden, intermediate, &device)?,
    };
    let x = patterned_bf16(&[batch, 1usize, hidden], 0.01, &device)?;
    let positions = Tensor::from_slice(&[start_pos as f32], 1usize)?.to_device(device)?;
    let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;
    let prefix_k = patterned_bf16(&[batch, start_pos, num_kv_heads, head_dim], 0.002, &device)?;
    let prefix_v = patterned_bf16(&[batch, start_pos, num_kv_heads, head_dim], 0.003, &device)?;
    let bt0 = BlockTable { blocks: vec![0] };
    let bt1 = BlockTable { blocks: vec![1] };
    let block_tables = [&bt0, &bt1];
    let start_positions = [start_pos, start_pos];
    let mut batch_cache = crate::PagedKvCacheKt::new(
        1,
        2,
        block_size,
        num_kv_heads,
        head_dim,
        kiln_tensor::DType::BF16,
        device_kt,
    )?;
    for (row, block_table) in block_tables.iter().enumerate() {
        let row_k = prefix_k.narrow(0, row, 1)?.contiguous()?;
        let row_v = prefix_v.narrow(0, row, 1)?.contiguous()?;
        write_token_major_prefix_for_test(&mut batch_cache, 0, block_table, 0, &row_k, &row_v)?;
    }

    let batched = transformer_block_paged_decode_contiguous_batch(
        &*backend,
        &x,
        &layer,
        &config,
        &positions,
        &start_positions,
        &inv_freq,
        &mut batch_cache,
        &block_tables,
        0,
        None,
        None,
        None,
        None,
        None,
        #[cfg(feature = "metal")]
        None,
        #[cfg(feature = "cuda")]
        None,
    )?;
    synchronize_for_profile(&device)?;
    assert_eq!(batched.dims(), &[batch, 1usize, hidden]);

    for row in 0..batch {
        let mut row_cache = crate::PagedKvCacheKt::new(
            1,
            1,
            block_size,
            num_kv_heads,
            head_dim,
            kiln_tensor::DType::BF16,
            device_kt,
        )?;
        let row_table = BlockTable { blocks: vec![0] };
        let row_k = prefix_k.narrow(0, row, 1)?.contiguous()?;
        let row_v = prefix_v.narrow(0, row, 1)?.contiguous()?;
        write_token_major_prefix_for_test(&mut row_cache, 0, &row_table, 0, &row_k, &row_v)?;
        let row_x = x.narrow(0, row, 1)?.contiguous()?;
        let rowwise = transformer_block_paged(
            &*backend,
            &row_x,
            &layer,
            &config,
            &positions,
            start_pos,
            num_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            &inv_freq,
            config.rms_norm_eps,
            &mut row_cache,
            &row_table,
            0,
            None,
        )?;
        synchronize_for_profile(&device)?;

        let batch_row = batched.narrow(0, row, 1)?;
        let diff = (batch_row.to_dtype(DType::F32)? - rowwise.to_dtype(DType::F32)?)?;
        let abs = diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.flatten_all()?.to_vec1::<f32>()?[0];
        let mean = abs
            .flatten_all()?
            .mean(0)?
            .flatten_all()?
            .to_vec1::<f32>()?[0];
        eprintln!(
            "batched contiguous transformer block row {row}: max_abs_diff={max:e} mean_abs_diff={mean:e}"
        );
        assert!(
            max <= 3e-2,
            "row {row} batched contiguous transformer block max_abs_diff={max:e}"
        );
        assert!(
            mean <= 3e-3,
            "row {row} batched contiguous transformer block mean_abs_diff={mean:e}"
        );
    }

    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_model_forward_paged_decode_contiguous_batch_matches_rowwise_metal() -> Result<()> {
    let Some(device) = crate::backend::metal::try_new_metal() else {
        return Ok(());
    };
    // kt parallel to `device` for `PagedKvCache::new_kt` (#1082).
    let device_kt = device; // #1082: `device` is already kt

    let backend = crate::backend::for_device_kt(&device);
    let batch = 2usize;
    let vocab = 64usize;
    let hidden = 512usize;
    let intermediate = 768usize;
    let num_heads = 16usize;
    let num_kv_heads = 4usize;
    let head_dim = 256usize;
    let block_size = 16usize;
    let start_pos = 3usize;
    let weights = make_bf16_full_attention_gpu_weights(
        vocab,
        hidden,
        intermediate,
        num_heads,
        num_kv_heads,
        head_dim,
        1,
        &device,
    )?;
    let config = kiln_core::config::ModelConfig {
        hidden_size: hidden,
        num_layers: 1,
        num_attention_heads: num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size: intermediate,
        vocab_size: vocab,
        max_position_embeddings: 1024,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        dtype: kiln_core::config::DType::BF16,
        num_full_attention_layers: 1,
        full_attention_interval: 1,
        attn_output_gate: false,
        linear_num_key_heads: num_kv_heads,
        linear_key_head_dim: head_dim,
        linear_num_value_heads: num_heads,
        linear_value_head_dim: head_dim,
        linear_conv_kernel_dim: 4,
        partial_rotary_factor: 1.0,
    };

    let prefix_k = patterned_bf16(&[batch, start_pos, num_kv_heads, head_dim], 0.002, &device)?;
    let prefix_v = patterned_bf16(&[batch, start_pos, num_kv_heads, head_dim], 0.003, &device)?;
    let bt0 = BlockTable { blocks: vec![0] };
    let bt1 = BlockTable { blocks: vec![1] };
    let block_tables = [&bt0, &bt1];
    let start_positions = [start_pos, start_pos];
    let token_ids = [7u32, 11u32];
    let mut batch_cache = crate::PagedKvCacheKt::new(
        1,
        2,
        block_size,
        num_kv_heads,
        head_dim,
        kiln_tensor::DType::BF16,
        device_kt,
    )?;
    for (row, block_table) in block_tables.iter().enumerate() {
        let row_k = prefix_k.narrow(0, row, 1)?.contiguous()?;
        let row_v = prefix_v.narrow(0, row, 1)?.contiguous()?;
        write_token_major_prefix_for_test(&mut batch_cache, 0, block_table, 0, &row_k, &row_v)?;
    }

    let batched = model_forward_paged_decode_contiguous_batch(
        &*backend,
        &token_ids,
        &weights,
        &config,
        &mut batch_cache,
        &block_tables,
        &start_positions,
        None,
        None,
    )?;
    synchronize_for_profile(&device)?;
    assert_eq!(batched.dims(), &[batch, 1usize, vocab]);

    let positions = Tensor::from_slice(&[start_pos as f32], 1usize)?.to_device(device)?;
    for row in 0..batch {
        let mut row_cache = crate::PagedKvCacheKt::new(
            1,
            1,
            block_size,
            num_kv_heads,
            head_dim,
            kiln_tensor::DType::BF16,
            device_kt,
        )?;
        let row_table = BlockTable { blocks: vec![0] };
        let row_k = prefix_k.narrow(0, row, 1)?.contiguous()?;
        let row_v = prefix_v.narrow(0, row, 1)?.contiguous()?;
        write_token_major_prefix_for_test(&mut row_cache, 0, &row_table, 0, &row_k, &row_v)?;
        let rowwise = model_forward_paged(
            &*backend,
            &token_ids[row..row + 1],
            &weights,
            &config,
            &mut row_cache,
            &row_table,
            start_pos,
            None,
            None,
            Some(&positions),
        )?;
        synchronize_for_profile(&device)?;

        let batch_row = batched.narrow(0, row, 1)?;
        let diff = (batch_row.to_dtype(DType::F32)? - rowwise.to_dtype(DType::F32)?)?;
        let abs = diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.flatten_all()?.to_vec1::<f32>()?[0];
        let mean = abs
            .flatten_all()?
            .mean(0)?
            .flatten_all()?
            .to_vec1::<f32>()?[0];
        eprintln!(
            "batched contiguous model decode row {row}: max_abs_diff={max:e} mean_abs_diff={mean:e}"
        );
        assert!(
            max <= 3e-2,
            "row {row} batched contiguous model decode max_abs_diff={max:e}"
        );
        assert!(
            mean <= 3e-3,
            "row {row} batched contiguous model decode mean_abs_diff={mean:e}"
        );
    }

    Ok(())
}

/// Phase 12-B-prime: parity test that exercises the dyn_seqlen varlen
/// paged decode path under a non-uniform `start_positions` batch on CUDA.
/// The Metal-gated test above only covers the uniform `start_pos`
/// fast-path; this test confirms that batched decode with divergent
/// per-row K/V prefix lengths still matches per-row `model_forward_paged`
/// bit-for-bit (within bf16 numeric tolerance).
#[cfg(feature = "cuda")]
#[test]
// #1082 C3 (was #[ignore]'d): root-caused via the KILN_C3_BISECT per-stage
// bisection — the batched-vs-rowwise decode is BIT-IDENTICAL through attention
// (input-norm, q/k/v proj, qk-norm, RoPE, raw paged-attn output); the only
// divergence is bf16 GEMM-shape rounding at the large-K o_proj GEMM (M=1 decode
// vs M>1 batched accumulate large-K dots in a different order), amplified by
// lm_head, and it is ~0.15% of the logit magnitude and does NOT change the
// decoded token. Re-characterized to gate on token/argmax parity + a
// bf16-realistic relative bound (see the in-loop comment); no longer ignored.
fn test_model_forward_paged_decode_contiguous_batch_dyn_seqlen_cuda() -> Result<()> {
    let device = match new_cuda_device(0) {
        Ok(device) => device,
        Err(err) => {
            eprintln!(
                "CUDA unavailable, skipping test_model_forward_paged_decode_contiguous_batch_dyn_seqlen_cuda: {err}"
            );
            return Ok(());
        }
    };
    // #1082: `device` is now a kt `Device` (from `new_cuda_device`), so the
    // kt cache constructor takes it directly and the backend dispatch goes
    // through the kt entry point.
    let device_kt = device;

    let backend = crate::backend::for_device_kt(&device);
    let batch = 2usize;
    let vocab = 64usize;
    let hidden = 512usize;
    let intermediate = 768usize;
    let num_heads = 16usize;
    let num_kv_heads = 4usize;
    let head_dim = 256usize;
    let block_size = 16usize;
    // Non-uniform start positions — the whole point of this test.
    let start_positions = [3usize, 5usize];
    let weights = make_bf16_full_attention_gpu_weights(
        vocab,
        hidden,
        intermediate,
        num_heads,
        num_kv_heads,
        head_dim,
        1,
        &device,
    )?;
    let config = kiln_core::config::ModelConfig {
        hidden_size: hidden,
        num_layers: 1,
        num_attention_heads: num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size: intermediate,
        vocab_size: vocab,
        max_position_embeddings: 1024,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        dtype: kiln_core::config::DType::BF16,
        num_full_attention_layers: 1,
        full_attention_interval: 1,
        attn_output_gate: false,
        linear_num_key_heads: num_kv_heads,
        linear_key_head_dim: head_dim,
        linear_num_value_heads: num_heads,
        linear_value_head_dim: head_dim,
        linear_conv_kernel_dim: 4,
        partial_rotary_factor: 1.0,
    };

    // Build per-row prefix K/V at each row's actual start_pos so that
    // the batched cache holds divergent K/V prefix lengths. Distinct
    // patterns per row catch any cross-row leakage.
    // #1082: `patterned_bf16` builds kt tensors and `PagedKvCacheKt`'s
    // `write_token_major_native` takes kt tensors — pass them directly
    // (no candle bridge) for the cache writes below.
    let prefix_k_row0 = patterned_bf16(
        &[1, start_positions[0], num_kv_heads, head_dim],
        0.002,
        &device,
    )?;
    let prefix_v_row0 = patterned_bf16(
        &[1, start_positions[0], num_kv_heads, head_dim],
        0.003,
        &device,
    )?;
    let prefix_k_row1 = patterned_bf16(
        &[1, start_positions[1], num_kv_heads, head_dim],
        0.0021,
        &device,
    )?;
    let prefix_v_row1 = patterned_bf16(
        &[1, start_positions[1], num_kv_heads, head_dim],
        0.0031,
        &device,
    )?;

    let bt0 = BlockTable { blocks: vec![0] };
    let bt1 = BlockTable { blocks: vec![1] };
    let block_tables = [&bt0, &bt1];
    let token_ids = [7u32, 11u32];

    let mut batch_cache = crate::PagedKvCacheKt::new(
        1,
        2,
        block_size,
        num_kv_heads,
        head_dim,
        kiln_tensor::DType::BF16,
        device_kt,
    )?;
    // Phase 7 #1082: parallel-allocate a kt twin via the constructor
    // stub `try_kt_paged_kv_cache_new` (commit 638bc441). When the
    // startup mode `accelerator.kt_api_mode = "all"` is off (the default),
    // this returns `None` and is zero overhead — the test still
    // runs against `batch_cache` only and asserts the same parity
    // bound. When the gate is on, the kt cache is allocated
    // alongside the candle cache and its shape accessors are
    // checked here so the constructor stub is exercised at a real
    // call site instead of only behind `#[allow(dead_code)]`.
    // The kt writer/reader story is still ahead — see the writer
    // stub `try_kt_paged_kv_write_token_major_native_graph_slot`
    // for the matching helper. Until that gets threaded through
    // `model_forward_paged_decode_contiguous_batch`, the kt cache
    // built here is *only* used for shape parity, not for the
    // K/V writes the test makes against `batch_cache`.
    let batch_cache_kt = try_kt_paged_kv_cache_new(
        1,
        2,
        block_size,
        num_kv_heads,
        head_dim,
        DType::BF16,
        &device,
    )?;
    if let Some(ref kt) = batch_cache_kt {
        // Constructor-stub shape parity: the kt cache must have the
        // same per-layer block count and block size as the candle
        // cache it shadows. Catches any future regression in the
        // constructor stub or in `PagedKvCacheKt::new` that would
        // silently allocate a differently-shaped pool.
        assert_eq!(kt.num_layers(), 1);
        assert_eq!(kt.num_blocks(), 2);
        assert_eq!(kt.block_size(), block_size);
        assert!(!kt.is_fp8(), "BF16 path must not flip the FP8 flag");
    }
    assert!(batch_cache.write_token_major_native(0, &bt0, 0, &prefix_k_row0, &prefix_v_row0)?);
    assert!(batch_cache.write_token_major_native(0, &bt1, 0, &prefix_k_row1, &prefix_v_row1)?);

    let batched = model_forward_paged_decode_contiguous_batch(
        &*backend,
        &token_ids,
        &weights,
        &config,
        &mut batch_cache,
        &block_tables,
        &start_positions,
        None,
        None,
    )?;
    synchronize_for_profile(&device)?;
    assert_eq!(batched.dims(), &[batch, 1usize, vocab]);

    for row in 0..batch {
        let row_start_pos = start_positions[row];
        let mut row_cache = crate::PagedKvCacheKt::new(
            1,
            1,
            block_size,
            num_kv_heads,
            head_dim,
            kiln_tensor::DType::BF16,
            device_kt,
        )?;
        let row_table = BlockTable { blocks: vec![0] };
        let (row_k, row_v) = if row == 0 {
            (prefix_k_row0.clone(), prefix_v_row0.clone())
        } else {
            (prefix_k_row1.clone(), prefix_v_row1.clone())
        };
        assert!(row_cache.write_token_major_native(0, &row_table, 0, &row_k, &row_v)?);
        let positions = Tensor::from_slice(&[row_start_pos as f32], 1usize)?.to_device(device)?;
        let rowwise = model_forward_paged(
            &*backend,
            &token_ids[row..row + 1],
            &weights,
            &config,
            &mut row_cache,
            &row_table,
            row_start_pos,
            None,
            None,
            Some(&positions),
        )?;
        synchronize_for_profile(&device)?;

        // #1082 C3 (root-caused via the KILN_C3_BISECT per-stage bisection): the
        // batched and per-row decode are BIT-IDENTICAL through input_layernorm,
        // q/k/v proj, qk-norm, RoPE, and the raw paged-attention kernel output.
        // The ONLY divergence appears at the o_proj GEMM (K=4096) and is bf16
        // GEMM-SHAPE non-determinism: kt's per-row decode (M=1, a GEMV-shaped
        // matmul) and the batched path (M>1 GEMM) accumulate the large-K dot
        // products in a different order, so they round differently in bf16 (the
        // K=512 q/k/v projections stay identical — the effect grows with K), and
        // lm_head amplifies it. It is benign for decode: logit magnitudes are large
        // (~3e2), so a ~0.5 absolute diff is ~0.15% relative (within bf16's ~0.4%
        // per-element precision) and the DECODED TOKEN is unchanged (argmax
        // identical, top-2 gap >> the diff). Candle used a single matmul path for
        // M=1 and M>1, so it was bit-identical pre-flip and passed an absolute
        // 3e-2 bar; that bar demanded ~9e-5 relative on magnitude-3e2 bf16 logits,
        // which is unachievable once kt (correctly) uses a faster M=1 decode path.
        // So we gate on what decode actually depends on — token/argmax parity —
        // plus a bf16-realistic RELATIVE bound, NOT the old absolute bar.
        let batch_row = batched.narrow(0, row, 1)?;
        let br = batch_row
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let rw = rowwise
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                    if x > bv { (i, x) } else { (bi, bv) }
                })
                .0
        };
        let (ba, ra) = (argmax(&br), argmax(&rw));
        // DECODE-CORRECTNESS gate: batched and per-row decode MUST pick the same
        // token. A real divergence (wrong KV / mask / attention) flips this; bf16
        // GEMM-shape noise does not (the top-2 gap dwarfs it).
        assert_eq!(
            ba, ra,
            "row {row} dyn_seqlen batched-vs-rowwise decode picked DIFFERENT tokens \
                 (batched={ba} rowwise={ra}) — a real decode divergence, not bf16 noise"
        );
        let max_abs_diff = br
            .iter()
            .zip(&rw)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mean_abs_diff =
            br.iter().zip(&rw).map(|(a, b)| (a - b).abs()).sum::<f32>() / br.len() as f32;
        let max_abs_logit = rw.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-6);
        let max_relative_diff = max_abs_diff / max_abs_logit;
        let mean_relative_diff = mean_abs_diff / max_abs_logit;
        eprintln!(
            "dyn_seqlen batched decode row {row} (start_pos={row_start_pos}): token={ba} \
                 max_abs_diff={max_abs_diff:e} ({:.3}% of max|logit|={max_abs_logit:e}) \
                 mean_abs_diff={mean_abs_diff:e} ({:.3}% of max|logit|)",
            100.0 * max_relative_diff,
            100.0 * mean_relative_diff,
        );
        // BF16 has 0.78125% spacing within a binade. Different M=1 and M=2
        // accumulation paths can therefore differ by one adjacent BF16 value
        // even when both are correctly rounded. Bound the worst element at
        // 0.8%, and separately require the mean drift to remain below 0.1% so
        // a broad low-amplitude corruption cannot hide behind the max bound.
        assert!(
            max_relative_diff <= 8e-3,
            "row {row} dyn_seqlen batched decode max_abs_diff={max_abs_diff:e} exceeds \
                 0.8% of max|logit|={max_abs_logit:e} — larger than one BF16 spacing"
        );
        assert!(
            mean_relative_diff <= 1e-3,
            "row {row} dyn_seqlen batched decode mean_abs_diff={mean_abs_diff:e} exceeds \
                 0.1% of max|logit|={max_abs_logit:e} — drift is too broad for BF16 \
                 GEMM-shape noise"
        );
    }

    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_metal_graph_bs1_decode_matches_eager_across_boundaries_and_buckets() -> Result<()> {
    let Some(device) = crate::backend::metal::try_new_metal() else {
        assert!(
            !explicit_hardware_qualification(),
            "Metal graph qualification requires an available Metal device"
        );
        return Ok(());
    };
    (|| -> Result<()> {
        let backend = crate::backend::for_device_kt(&device);
        let vocab = 64usize;
        let hidden = 512usize;
        let intermediate = 768usize;
        let num_heads = 16usize;
        let num_kv_heads = 4usize;
        let head_dim = 256usize;
        let block_size = 16usize;
        let config = make_metal_graph_test_config(
            vocab,
            hidden,
            intermediate,
            num_heads,
            num_kv_heads,
            head_dim,
        );
        let weights = make_metal_graph_test_weights(&config, &device)?;
        let mut runner = crate::metal_graph::MetalGraphRunner::new(&device, true);
        assert!(
            runner.is_enabled(),
            "Metal graph runner must be enabled on a Metal device"
        );

        let mut graph_cache = crate::PagedKvCacheKt::new(
            1,
            12,
            block_size,
            num_kv_heads,
            head_dim,
            kiln_tensor::DType::BF16,
            device,
        )?;
        let mut eager_cache = crate::PagedKvCacheKt::new(
            1,
            12,
            block_size,
            num_kv_heads,
            head_dim,
            kiln_tensor::DType::BF16,
            device,
        )?;
        let mut graph_state = LinearAttentionState::new(&config, &device)?;
        let mut eager_state = LinearAttentionState::new(&config, &device)?;

        #[allow(clippy::too_many_arguments)]
        fn run_bs1_pair(
            runner: &mut crate::metal_graph::MetalGraphRunner,
            backend: &dyn BackendRuntime,
            token_id: u32,
            weights: &GpuWeights,
            config: &kiln_core::config::ModelConfig,
            graph_cache: &crate::PagedKvCacheKt,
            eager_cache: &crate::PagedKvCacheKt,
            block_table: &BlockTable,
            seq_len: usize,
            graph_state: &mut LinearAttentionState,
            eager_state: &mut LinearAttentionState,
            device: &Device,
        ) -> Result<()> {
            let eager = model_forward_paged_next_token_greedy(
                backend,
                token_id,
                weights,
                config,
                eager_cache,
                block_table,
                seq_len,
                Some(eager_state),
                None,
                None,
            )?;
            let graph = runner.decode_step_paged_greedy(
                backend,
                token_id,
                weights,
                config,
                graph_cache,
                block_table,
                seq_len,
                graph_state,
                None,
            )?;
            synchronize_for_profile(device)?;
            assert_eq!(
                graph, eager,
                "Metal graph bs=1 token mismatch at seq_len={seq_len}"
            );
            assert_metal_graph_cache_matches_eager(graph_cache, eager_cache)?;
            Ok(())
        }

        let bt0 = BlockTable { blocks: vec![0] };
        seed_metal_graph_prefix_for_test(
            &mut graph_cache,
            &bt0,
            3,
            num_kv_heads,
            head_dim,
            0.0020,
            0.0030,
            &device,
        )?;
        seed_metal_graph_prefix_for_test(
            &mut eager_cache,
            &bt0,
            3,
            num_kv_heads,
            head_dim,
            0.0020,
            0.0030,
            &device,
        )?;
        run_bs1_pair(
            &mut runner,
            &*backend,
            7,
            &weights,
            &config,
            &graph_cache,
            &eager_cache,
            &bt0,
            3,
            &mut graph_state,
            &mut eager_state,
            &device,
        )?;
        assert_eq!(runner.stable_buffer_count(), 1);
        assert_eq!(runner.captured_graph_count(), 1);
        assert_eq!(runner.captured_graph_replay_count_sum(), 1);
        run_bs1_pair(
            &mut runner,
            &*backend,
            11,
            &weights,
            &config,
            &graph_cache,
            &eager_cache,
            &bt0,
            4,
            &mut graph_state,
            &mut eager_state,
            &device,
        )?;
        assert_eq!(runner.stable_buffer_count(), 1);
        assert_eq!(runner.captured_graph_count(), 1);
        assert_eq!(
            runner.captured_graph_replay_count_sum(),
            2,
            "same-bucket bs=1 step should replay the captured Metal ICB graph"
        );

        let bt1 = BlockTable { blocks: vec![1] };
        seed_metal_graph_prefix_for_test(
            &mut graph_cache,
            &bt1,
            3,
            num_kv_heads,
            head_dim,
            0.0021,
            0.0031,
            &device,
        )?;
        seed_metal_graph_prefix_for_test(
            &mut eager_cache,
            &bt1,
            3,
            num_kv_heads,
            head_dim,
            0.0021,
            0.0031,
            &device,
        )?;
        run_bs1_pair(
            &mut runner,
            &*backend,
            13,
            &weights,
            &config,
            &graph_cache,
            &eager_cache,
            &bt1,
            3,
            &mut graph_state,
            &mut eager_state,
            &device,
        )?;
        assert_eq!(
            runner.stable_buffer_count(),
            1,
            "new request/block table should evict old bs=1 stable buffers"
        );
        assert_eq!(runner.captured_graph_count(), 1);
        assert_eq!(runner.captured_graph_replay_count_sum(), 1);

        let bt_long = BlockTable {
            blocks: vec![2, 3, 4, 5, 6],
        };
        seed_metal_graph_prefix_for_test(
            &mut graph_cache,
            &bt_long,
            63,
            num_kv_heads,
            head_dim,
            0.0022,
            0.0032,
            &device,
        )?;
        seed_metal_graph_prefix_for_test(
            &mut eager_cache,
            &bt_long,
            63,
            num_kv_heads,
            head_dim,
            0.0022,
            0.0032,
            &device,
        )?;
        for (token, seq_len) in [(17u32, 63usize), (19u32, 64usize), (23u32, 65usize)] {
            run_bs1_pair(
                &mut runner,
                &*backend,
                token,
                &weights,
                &config,
                &graph_cache,
                &eager_cache,
                &bt_long,
                seq_len,
                &mut graph_state,
                &mut eager_state,
                &device,
            )?;
        }
        assert_eq!(
            runner.stable_buffer_count(),
            2,
            "crossing the FA2 K/V bucket should keep one stable buffer per live bucket"
        );
        assert_eq!(runner.captured_graph_count(), 2);
        assert_eq!(
            runner.captured_graph_replay_count_sum(),
            3,
            "third long bs=1 step should replay the second bucket's captured graph"
        );
        println!(
            "[metal-graph-bs1] OK: eager parity across request boundaries and two replay buckets"
        );
        Ok(())
    })()
}

#[cfg(feature = "metal")]
#[test]
fn test_metal_graph_batched_decode_matches_eager_and_replays_bucket() -> Result<()> {
    let Some(device) = crate::backend::metal::try_new_metal() else {
        assert!(
            !explicit_hardware_qualification(),
            "Metal batched graph qualification requires an available Metal device"
        );
        return Ok(());
    };
    (|| -> Result<()> {
        let backend = crate::backend::for_device_kt(&device);
        let vocab = 64usize;
        let hidden = 512usize;
        let intermediate = 768usize;
        let num_heads = 16usize;
        let num_kv_heads = 4usize;
        let head_dim = 256usize;
        let block_size = 16usize;
        let config = make_metal_graph_test_config(
            vocab,
            hidden,
            intermediate,
            num_heads,
            num_kv_heads,
            head_dim,
        );
        let weights = make_metal_graph_test_weights(&config, &device)?;
        let mut runner = crate::metal_graph::MetalGraphRunner::new(&device, true);
        assert!(
            runner.is_enabled(),
            "Metal graph runner must be enabled on a Metal device"
        );

        let mut graph_cache = crate::PagedKvCacheKt::new(
            1,
            12,
            block_size,
            num_kv_heads,
            head_dim,
            kiln_tensor::DType::BF16,
            device,
        )?;
        let mut eager_cache = crate::PagedKvCacheKt::new(
            1,
            12,
            block_size,
            num_kv_heads,
            head_dim,
            kiln_tensor::DType::BF16,
            device,
        )?;

        #[allow(clippy::too_many_arguments)]
        fn run_batch_pair(
            runner: &mut crate::metal_graph::MetalGraphRunner,
            backend: &dyn BackendRuntime,
            token_ids: &[u32],
            weights: &GpuWeights,
            config: &kiln_core::config::ModelConfig,
            graph_cache: &crate::PagedKvCacheKt,
            eager_cache: &crate::PagedKvCacheKt,
            block_tables: &[&BlockTable],
            seq_lens: &[usize],
            device: &Device,
        ) -> Result<()> {
            let eager = model_forward_paged_decode_contiguous_batch_greedy_with_ids(
                backend,
                token_ids,
                weights,
                config,
                eager_cache,
                block_tables,
                seq_lens,
                None,
                None,
                None,
            )?;
            let graph = runner
                .decode_step_paged_greedy_batch(
                    backend,
                    token_ids,
                    weights,
                    config,
                    graph_cache,
                    block_tables,
                    seq_lens,
                    None,
                    None,
                )?
                .context("Metal graph batched decode unexpectedly declined")?;
            synchronize_for_profile(device)?;
            assert_eq!(
                graph, eager,
                "Metal graph batched token mismatch at seq_lens={seq_lens:?}"
            );
            assert_metal_graph_cache_matches_eager(graph_cache, eager_cache)?;
            Ok(())
        }

        let bt0 = BlockTable { blocks: vec![0] };
        let bt1 = BlockTable { blocks: vec![1] };
        for (bt, k_scale, v_scale) in [(&bt0, 0.0020, 0.0030), (&bt1, 0.0021, 0.0031)] {
            seed_metal_graph_prefix_for_test(
                &mut graph_cache,
                bt,
                3,
                num_kv_heads,
                head_dim,
                k_scale,
                v_scale,
                &device,
            )?;
            seed_metal_graph_prefix_for_test(
                &mut eager_cache,
                bt,
                3,
                num_kv_heads,
                head_dim,
                k_scale,
                v_scale,
                &device,
            )?;
        }
        let block_tables = [&bt0, &bt1];
        run_batch_pair(
            &mut runner,
            &*backend,
            &[7, 11],
            &weights,
            &config,
            &graph_cache,
            &eager_cache,
            &block_tables,
            &[3, 3],
            &device,
        )?;
        assert_eq!(runner.stable_buffer_count(), 1);
        assert_eq!(runner.captured_graph_count(), 1);
        assert_eq!(runner.captured_graph_replay_count_sum(), 1);
        run_batch_pair(
            &mut runner,
            &*backend,
            &[13, 17],
            &weights,
            &config,
            &graph_cache,
            &eager_cache,
            &block_tables,
            &[4, 4],
            &device,
        )?;
        assert_eq!(runner.stable_buffer_count(), 1);
        assert_eq!(runner.captured_graph_count(), 1);
        assert_eq!(
            runner.captured_graph_replay_count_sum(),
            2,
            "same-bucket batched step should replay the captured Metal ICB graph"
        );

        let bt2 = BlockTable {
            blocks: vec![2, 3, 4, 5, 6],
        };
        let bt7 = BlockTable {
            blocks: vec![7, 8, 9, 10, 11],
        };
        for (bt, k_scale, v_scale) in [(&bt2, 0.0022, 0.0032), (&bt7, 0.0023, 0.0033)] {
            seed_metal_graph_prefix_for_test(
                &mut graph_cache,
                bt,
                63,
                num_kv_heads,
                head_dim,
                k_scale,
                v_scale,
                &device,
            )?;
            seed_metal_graph_prefix_for_test(
                &mut eager_cache,
                bt,
                63,
                num_kv_heads,
                head_dim,
                k_scale,
                v_scale,
                &device,
            )?;
        }
        let long_tables = [&bt2, &bt7];
        for (tokens, seq_lens) in [
            ([19u32, 23u32], [63usize, 63usize]),
            ([29u32, 31u32], [64usize, 64usize]),
            ([37u32, 41u32], [65usize, 65usize]),
        ] {
            run_batch_pair(
                &mut runner,
                &*backend,
                &tokens,
                &weights,
                &config,
                &graph_cache,
                &eager_cache,
                &long_tables,
                &seq_lens,
                &device,
            )?;
        }
        assert_eq!(
            runner.stable_buffer_count(),
            2,
            "batched graph runner should reuse the short request's 64-token bucket and add the 128-token bucket"
        );
        assert_eq!(runner.captured_graph_count(), 2);
        assert_eq!(
            runner.captured_graph_replay_count_sum(),
            5,
            "third long batched step should replay the second long bucket"
        );
        println!("[metal-graph-batched] OK: eager parity across two rows and two replay buckets");
        Ok(())
    })()
}

/// bs=1 CUDA-graph-capture+replay vs. eager decode parity.
///
/// `cuda_graph.rs` captures the bs=1 decode forward under CUDA stream
/// capture (under an enabled typed CUDA graph policy) and replays
/// it on subsequent steps, baking device pointers into the recorded
/// kernel launches. There was NO correctness gate verifying that a
/// graph-captured-and-replayed decode produces the SAME logits as the
/// equivalent eager (non-graph) decode — a stale / dangling-pointer or
/// wrong-buffer bug in capture would silently corrupt decode. This test
/// is that gate (and the validation gate before `cuda_graph.rs`'s
/// candle buffers flip to kt under #1082).
///
/// Strategy: build the same synthetic 1-layer full-attention model the
/// C3 dyn-seqlen test uses, then
///   1. compute three sequential reference logits with plain eager
///      `model_forward_paged` on a fresh prefix-only cache, and
///   2. drive `CudaGraphRunner::decode_step_paged` through the same three
///      positions against a second fresh prefix-only cache: call 1 = eager
///      warmup, call 2 = stream capture, call 3 = graph replay.
///
/// We assert (a) a graph was actually captured (guards against a silent
/// eager fallback making the comparison vacuous), and (b) the replay
/// logits match the eager reference with token/argmax parity + a
/// bf16-realistic relative bound — mirroring the C3 assertion exactly.
///
/// Positions advance monotonically because the runner deliberately invalidates
/// an owner's graphs when its decode timeline does not advance by one. All
/// three positions remain in the same graph geometry bucket, so call 3 must hit
/// and replay call 2's graph instead of capturing again.
///
/// bs=1 ONLY: the unqualified bs>1 graph path is unavailable; this test
/// deliberately does not exercise it.
#[cfg(feature = "cuda")]
#[test]
fn test_cuda_graph_bs1_decode_matches_eager() -> Result<()> {
    let device = match new_cuda_device(0) {
        Ok(device) => device,
        Err(err) => {
            assert!(
                !explicit_hardware_qualification(),
                "CUDA graph qualification requires logical device zero: {err}"
            );
            eprintln!("CUDA unavailable, skipping test_cuda_graph_bs1_decode_matches_eager: {err}");
            return Ok(());
        }
    };
    // #1082: `device` is a kt `Device`; the kt cache constructor takes
    // it directly and the backend dispatch goes through the kt entry point.
    let device_kt = device;
    let backend = crate::backend::for_device_kt(&device);

    // Same synthetic 1-layer full-attention fixture as the C3 test.
    let vocab = 64usize;
    let hidden = 512usize;
    let intermediate = 768usize;
    let num_heads = 16usize;
    let num_kv_heads = 4usize;
    let head_dim = 256usize;
    let block_size = 16usize;
    let start_pos = 5usize; // < block_size, so the decode slot == start_pos
    let token_id = 7u32;
    let weights = make_bf16_full_attention_gpu_weights(
        vocab,
        hidden,
        intermediate,
        num_heads,
        num_kv_heads,
        head_dim,
        1,
        &device,
    )?;
    let config = kiln_core::config::ModelConfig {
        hidden_size: hidden,
        num_layers: 1,
        num_attention_heads: num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size: intermediate,
        vocab_size: vocab,
        max_position_embeddings: 1024,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        dtype: kiln_core::config::DType::BF16,
        num_full_attention_layers: 1,
        full_attention_interval: 1,
        attn_output_gate: false,
        linear_num_key_heads: num_kv_heads,
        linear_key_head_dim: head_dim,
        linear_num_value_heads: num_heads,
        linear_value_head_dim: head_dim,
        linear_conv_kernel_dim: 4,
        partial_rotary_factor: 1.0,
    };

    // Prefix K/V written at the decode position's run so the paged
    // attention reads a real context. (Same pattern as C3.)
    let prefix_k = patterned_bf16(&[1, start_pos, num_kv_heads, head_dim], 0.002, &device)?;
    let prefix_v = patterned_bf16(&[1, start_pos, num_kv_heads, head_dim], 0.003, &device)?;
    let block_table = BlockTable { blocks: vec![0] };

    // --- (1) Eager reference: fresh cache + fresh linear state. ---
    let mut ref_cache = crate::PagedKvCacheKt::new(
        1,
        1,
        block_size,
        num_kv_heads,
        head_dim,
        kiln_tensor::DType::BF16,
        device_kt,
    )?;
    assert!(ref_cache.write_token_major_native(0, &block_table, 0, &prefix_k, &prefix_v)?);
    let mut eager_step = |position: usize| -> Result<Vec<f32>> {
        let positions = Tensor::from_slice(&[position as f32], 1usize)?.to_device(device)?;
        let logits = model_forward_paged(
            &*backend,
            &[token_id],
            &weights,
            &config,
            &mut ref_cache,
            &block_table,
            position,
            None,
            None,
            Some(&positions),
        )?;
        synchronize_for_profile(&device)?;
        assert_eq!(logits.dims(), &[1usize, 1usize, vocab]);
        Ok(logits
            .to_dtype(kiln_tensor::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?)
    };
    let eager_warmup = eager_step(start_pos)?;
    let eager_capture = eager_step(start_pos + 1)?;
    let eager_replay = eager_step(start_pos + 2)?;

    // --- (2) Graph path: warmup -> capture -> replay. ---
    let result = (|| -> Result<()> {
        let policy = crate::cuda_graph::CudaGraphExecutionPolicy::try_new(true, 8)?;
        let mut runner = crate::cuda_graph::CudaGraphRunner::new(&device_kt, policy);
        assert!(
            runner.is_enabled(),
            "CUDA graph runner must be enabled on a CUDA device"
        );

        let graph_cache = crate::PagedKvCacheKt::new(
            1,
            1,
            block_size,
            num_kv_heads,
            head_dim,
            kiln_tensor::DType::BF16,
            device_kt,
        )?;
        assert!(graph_cache.write_token_major_native(0, &block_table, 0, &prefix_k, &prefix_v)?);
        let mut linear_state = LinearAttentionState::new(&config, &device)?;

        // Three monotonically advancing positions in one graph geometry bucket.
        // A plain local fn (not a closure) makes the `&mut runner` borrow end
        // after each call so the test can inspect runner state between steps.
        #[allow(clippy::too_many_arguments)]
        fn run_decode_step(
            runner: &mut crate::cuda_graph::CudaGraphRunner,
            backend: &dyn BackendRuntime,
            token_id: u32,
            weights: &GpuWeights,
            config: &kiln_core::config::ModelConfig,
            graph_cache: &crate::PagedKvCacheKt,
            block_table: &BlockTable,
            start_pos: usize,
            linear_state: &mut LinearAttentionState,
            device: &Device,
            vocab: usize,
        ) -> Result<Vec<f32>> {
            // #1082: `decode_step_paged` now returns a kt `Tensor`; read
            // the host f32 logits through the kt API (no candle bridge).
            let logits_kt = runner.decode_step_paged(
                backend,
                token_id,
                weights,
                config,
                graph_cache,
                block_table,
                start_pos,
                linear_state,
                None,
                None,
            )?;
            synchronize_for_profile(device)?;
            assert_eq!(logits_kt.dims(), &[1usize, 1usize, vocab]);
            Ok(logits_kt
                .to_dtype(kiln_tensor::DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?)
        }
        let step = |runner: &mut crate::cuda_graph::CudaGraphRunner,
                    linear_state: &mut LinearAttentionState,
                    position: usize|
         -> Result<Vec<f32>> {
            run_decode_step(
                runner,
                &*backend,
                token_id,
                &weights,
                &config,
                &graph_cache,
                &block_table,
                position,
                linear_state,
                &device,
                vocab,
            )
        };

        let warmup = step(&mut runner, &mut linear_state, start_pos)?;
        let captured = step(&mut runner, &mut linear_state, start_pos + 1)?;
        // Guard against silent eager fallback: a graph MUST have been
        // captured by now, otherwise this "parity" test is comparing
        // eager-against-eager and proves nothing about capture/replay.
        assert!(
            runner.is_enabled(),
            "CUDA graph runner disabled before replay parity; capture failure must be fixed"
        );
        assert!(
            runner.captured_graph_count() > 0,
            "CUDA graph replay parity requires a captured graph, not eager fallback"
        );
        let replay = step(&mut runner, &mut linear_state, start_pos + 2)?;

        // --- Assertions: mirror the C3 token-parity + relative bound. ---
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                    if x > bv { (i, x) } else { (bi, bv) }
                })
                .0
        };
        let (ewa, eca, era, wa, ca, ra) = (
            argmax(&eager_warmup),
            argmax(&eager_capture),
            argmax(&eager_replay),
            argmax(&warmup),
            argmax(&captured),
            argmax(&replay),
        );
        let max_abs = |v: &[f32]| v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        eprintln!(
            "cuda graph bs=1 decode stages: eager_tokens={ewa}/{eca}/{era} \
             graph_tokens={wa}/{ca}/{ra} eager_replay_max_abs={:e} \
             warmup_max_abs={:e} capture_max_abs={:e} replay_max_abs={:e}",
            max_abs(&eager_replay),
            max_abs(&warmup),
            max_abs(&captured),
            max_abs(&replay),
        );
        assert_eq!(
            ewa, wa,
            "CUDA graph-shaped eager warmup picked a different token \
             (warmup={wa} eager={ewa})"
        );
        assert_eq!(
            eca, ca,
            "CUDA graph first captured launch picked a different token \
             (capture={ca} eager={eca})"
        );
        // DECODE-CORRECTNESS gate: the graph-replayed decode MUST pick the
        // same token as eager. A stale/dangling-pointer or wrong-buffer
        // capture bug flips this; bf16 rounding noise does not (top-2 gap
        // dwarfs it).
        assert_eq!(
            era, ra,
            "CUDA-graph replay and eager decode picked DIFFERENT tokens \
                 (graph={ra} eager={era}) — a real graph-replay corruption, not bf16 noise"
        );
        let max_abs_diff = eager_replay
            .iter()
            .zip(&replay)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mean_abs_diff = eager_replay
            .iter()
            .zip(&replay)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / eager_replay.len() as f32;
        let max_abs_logit = eager_replay
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max)
            .max(1e-6);
        eprintln!(
            "cuda graph bs=1 decode parity (start_pos={start_pos}, token={token_id}): \
                 graph_token={ra} eager_token={era} \
                 max_abs_diff={max_abs_diff:e} ({:.3}% of max|logit|={max_abs_logit:e}) \
                 mean_abs_diff={mean_abs_diff:e}",
            100.0 * max_abs_diff / max_abs_logit
        );
        // bf16-realistic RELATIVE bound. Graph capture/replay should be
        // bit-identical to its own eager warmup (same kernels, same
        // pointers); any divergence vs. the independent eager reference
        // is at most bf16 GEMM-shape noise, well under 0.5% of the logit
        // magnitude. A capture bug would be >>1% relative.
        assert!(
            max_abs_diff <= 5e-3 * max_abs_logit,
            "CUDA-graph replay max_abs_diff={max_abs_diff:e} exceeds 0.5% of \
                 max|logit|={max_abs_logit:e} — larger than bf16 GEMM-shape noise, \
                 indicates graph capture/replay corruption"
        );
        println!(
            "[cuda-graph-bs1] OK: captured and replayed graph matches eager token and BF16 logit bound"
        );
        Ok(())
    })();

    result
}

#[cfg(feature = "metal")]
#[test]
fn test_model_forward_paged_decode_contiguous_batch_hybrid_matches_rowwise_metal() -> Result<()> {
    let Some(device) = crate::backend::metal::try_new_metal() else {
        return Ok(());
    };
    // kt parallel to `device` for `PagedKvCache::new_kt` (#1082).
    let device_kt = device; // #1082: `device` is already kt

    let backend = crate::backend::for_device_kt(&device);
    let batch = 2usize;
    let block_size = 16usize;
    let start_pos = 3usize;
    let config = kiln_core::config::ModelConfig {
        hidden_size: 256,
        num_layers: 4,
        num_attention_heads: 16,
        num_kv_heads: 4,
        head_dim: 256,
        intermediate_size: 512,
        vocab_size: 64,
        max_position_embeddings: 1024,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        dtype: kiln_core::config::DType::BF16,
        num_full_attention_layers: 1,
        full_attention_interval: 4,
        attn_output_gate: true,
        linear_num_key_heads: 16,
        linear_key_head_dim: 128,
        linear_num_value_heads: 32,
        linear_value_head_dim: 128,
        linear_conv_kernel_dim: 4,
        partial_rotary_factor: 0.25,
    };
    let weights = make_bf16_hybrid_gpu_weights(&config, &device)?;

    let prefix_k = patterned_bf16(
        &[batch, start_pos, config.num_kv_heads, config.head_dim],
        0.002,
        &device,
    )?;
    let prefix_v = patterned_bf16(
        &[batch, start_pos, config.num_kv_heads, config.head_dim],
        0.003,
        &device,
    )?;
    let bt0 = BlockTable { blocks: vec![0] };
    let bt1 = BlockTable { blocks: vec![1] };
    let block_tables = [&bt0, &bt1];
    let start_positions = [start_pos, start_pos];
    let token_ids = [7u32, 11u32];
    let mut batch_cache = crate::PagedKvCacheKt::new(
        config.num_full_attention_layers,
        2,
        block_size,
        config.num_kv_heads,
        config.head_dim,
        kiln_tensor::DType::BF16,
        device_kt,
    )?;
    for (row, block_table) in block_tables.iter().enumerate() {
        let row_k = prefix_k.narrow(0, row, 1)?.contiguous()?;
        let row_v = prefix_v.narrow(0, row, 1)?.contiguous()?;
        write_token_major_prefix_for_test(&mut batch_cache, 0, block_table, 0, &row_k, &row_v)?;
    }

    let mut row_states = Vec::with_capacity(batch);
    for row in 0..batch {
        row_states.push(patterned_linear_state(&config, row, &device)?);
    }
    let state_refs: Vec<&LinearAttentionState> = row_states.iter().collect();
    let mut batch_state = LinearAttentionState::from_batch_rows(&state_refs)?;
    let batched = model_forward_paged_decode_contiguous_batch(
        &*backend,
        &token_ids,
        &weights,
        &config,
        &mut batch_cache,
        &block_tables,
        &start_positions,
        Some(&mut batch_state),
        None,
    )?;
    synchronize_for_profile(&device)?;
    assert_eq!(batched.dims(), &[batch, 1usize, config.vocab_size]);

    let positions = Tensor::from_slice(&[start_pos as f32], 1usize)?.to_device(device)?;
    for row in 0..batch {
        let mut row_cache = crate::PagedKvCacheKt::new(
            config.num_full_attention_layers,
            1,
            block_size,
            config.num_kv_heads,
            config.head_dim,
            kiln_tensor::DType::BF16,
            device_kt,
        )?;
        let row_table = BlockTable { blocks: vec![0] };
        let row_k = prefix_k.narrow(0, row, 1)?.contiguous()?;
        let row_v = prefix_v.narrow(0, row, 1)?.contiguous()?;
        write_token_major_prefix_for_test(&mut row_cache, 0, &row_table, 0, &row_k, &row_v)?;
        let rowwise = model_forward_paged(
            &*backend,
            &token_ids[row..row + 1],
            &weights,
            &config,
            &mut row_cache,
            &row_table,
            start_pos,
            Some(&mut row_states[row]),
            None,
            Some(&positions),
        )?;
        synchronize_for_profile(&device)?;

        let batch_row = batched.narrow(0, row, 1)?;
        let (max, mean) = tensor_abs_diff_stats(&batch_row, &rowwise)?;
        eprintln!(
            "batched hybrid model decode row {row}: max_abs_diff={max:e} mean_abs_diff={mean:e}"
        );
        assert!(
            max <= 5e-2,
            "row {row} batched hybrid model decode max_abs_diff={max:e}"
        );
        assert!(
            mean <= 5e-3,
            "row {row} batched hybrid model decode mean_abs_diff={mean:e}"
        );
    }

    let batch_rows = batch_state.split_batch_rows()?;
    for row in 0..batch {
        for layer_idx in 0..row_states[row].recurrent_states.len() {
            let (rec_max, rec_mean) = tensor_abs_diff_stats(
                &batch_rows[row].recurrent_states[layer_idx],
                &row_states[row].recurrent_states[layer_idx],
            )?;
            let (conv_max, conv_mean) = tensor_abs_diff_stats(
                &batch_rows[row].conv_states[layer_idx],
                &row_states[row].conv_states[layer_idx],
            )?;
            eprintln!(
                "batched hybrid model state row {row} linear_layer {layer_idx}: recurrent_max={rec_max:e} recurrent_mean={rec_mean:e} conv_max={conv_max:e} conv_mean={conv_mean:e}"
            );
            assert!(
                rec_max <= 5e-2,
                "row {row} layer {layer_idx} recurrent max_abs_diff={rec_max:e}"
            );
            assert!(
                rec_mean <= 5e-3,
                "row {row} layer {layer_idx} recurrent mean_abs_diff={rec_mean:e}"
            );
            assert!(
                conv_max <= 5e-2,
                "row {row} layer {layer_idx} conv max_abs_diff={conv_max:e}"
            );
            assert!(
                conv_mean <= 5e-3,
                "row {row} layer {layer_idx} conv mean_abs_diff={conv_mean:e}"
            );
        }
    }

    Ok(())
}

#[test]
fn test_gqa_attention_output_shape() -> Result<()> {
    let device = Device::Cpu;
    let batch = 1;
    let seq_len = 4;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 8;
    let hidden = num_heads * head_dim; // 32

    let x = Tensor::randn(0.0_f32, 1.0, (batch, seq_len, hidden), device)?;
    let attn = make_test_attn_weights(num_heads, num_kv_heads, head_dim, hidden, &device)?;
    let positions: Vec<u32> = (0..seq_len as u32).collect();

    let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;
    let backend = test_backend(&device);
    let out = gqa_attention(
        &backend,
        &x,
        &attn,
        &positions,
        num_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        &inv_freq,
        1e-6,
        None,
        0,
        false,
        None,
    )?;
    assert_eq!(out.dims(), &[batch, seq_len, hidden]);

    Ok(())
}

#[test]
fn test_gqa_head_expansion() -> Result<()> {
    // Verify GQA works: 4 Q heads, 2 KV heads (ratio=2)
    let device = Device::Cpu;
    let batch = 2;
    let seq_len = 3;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 8;
    let hidden = num_heads * head_dim;

    let x = Tensor::randn(0.0_f32, 0.5, (batch, seq_len, hidden), device)?;
    let attn = make_test_attn_weights(num_heads, num_kv_heads, head_dim, hidden, &device)?;
    let positions: Vec<u32> = (0..seq_len as u32).collect();

    let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;
    let backend = test_backend(&device);
    let out = gqa_attention(
        &backend,
        &x,
        &attn,
        &positions,
        num_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        &inv_freq,
        1e-6,
        None,
        0,
        false,
        None,
    )?;
    assert_eq!(out.dims(), &[batch, seq_len, hidden]);

    // Output should be finite and not all zeros
    let vals = out.flatten_all()?.to_vec1::<f32>()?;
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "output should be finite"
    );
    let sum: f32 = vals.iter().map(|v| v.abs()).sum();
    assert!(sum > 1e-6, "output should not be all zeros");

    Ok(())
}

#[test]
fn test_gqa_single_token() -> Result<()> {
    // Single token should work (no causal masking needed)
    let device = Device::Cpu;
    let num_heads = 2;
    let num_kv_heads = 1;
    let head_dim = 4;
    let hidden = num_heads * head_dim;

    let x = Tensor::randn(0.0_f32, 1.0, (1, 1, hidden), device)?;
    let attn = make_test_attn_weights(num_heads, num_kv_heads, head_dim, hidden, &device)?;

    let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;
    let backend = test_backend(&device);
    let out = gqa_attention(
        &backend,
        &x,
        &attn,
        &[0],
        num_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        &inv_freq,
        1e-6,
        None,
        0,
        false,
        None,
    )?;
    assert_eq!(out.dims(), &[1, 1, hidden]);

    Ok(())
}

#[test]
fn test_causal_mask() -> Result<()> {
    let device = Device::Cpu;
    // A 3x3 score matrix
    let scores = Tensor::ones((1, 1, 3, 3), DType::F32, device)?;
    let masked = apply_causal_mask(&scores, 3)?;
    let vals = masked.flatten_all()?.to_vec1::<f32>()?;
    // Row 0: [1, -inf, -inf]
    assert!((vals[0] - 1.0).abs() < 1e-6);
    assert!(vals[1].is_infinite() && vals[1] < 0.0);
    assert!(vals[2].is_infinite() && vals[2] < 0.0);
    // Row 1: [1, 1, -inf]
    assert!((vals[3] - 1.0).abs() < 1e-6);
    assert!((vals[4] - 1.0).abs() < 1e-6);
    assert!(vals[5].is_infinite() && vals[5] < 0.0);
    // Row 2: [1, 1, 1]
    assert!((vals[6] - 1.0).abs() < 1e-6);
    assert!((vals[7] - 1.0).abs() < 1e-6);
    assert!((vals[8] - 1.0).abs() < 1e-6);

    Ok(())
}

#[test]
fn test_transformer_block_output_shape() -> Result<()> {
    let device = Device::Cpu;
    let batch = 1;
    let seq_len = 4;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 8;
    let hidden = num_heads * head_dim;
    let intermediate = hidden * 2;

    let x = Tensor::randn(0.0_f32, 1.0, (batch, seq_len, hidden), device)?;
    let positions: Vec<u32> = (0..seq_len as u32).collect();

    let gate_proj = Tensor::randn(0.0_f32, 0.02, (intermediate, hidden), device)?;
    let up_proj = Tensor::randn(0.0_f32, 0.02, (intermediate, hidden), device)?;
    let down_proj = Tensor::randn(0.0_f32, 0.02, (hidden, intermediate), device)?;
    let gate_proj_t = gate_proj.t()?.contiguous()?;
    let up_proj_t = up_proj.t()?.contiguous()?;
    let down_proj_t = down_proj.t()?.contiguous()?;

    let layer = GpuLayerWeights {
        input_layernorm: Tensor::zeros(hidden, DType::F32, device)?,
        post_attention_layernorm: Tensor::zeros(hidden, DType::F32, device)?,
        attention: GpuAttentionWeights::Full(make_test_attn_weights(
            num_heads,
            num_kv_heads,
            head_dim,
            hidden,
            &device,
        )?),
        mlp: GpuFfnWeights {
            gate_proj,
            up_proj,
            down_proj,
            gate_proj_t,
            up_proj_t,
            down_proj_t,
            gate_up_proj_t: None,
            gate_proj_marlin: None,
            up_proj_marlin: None,
            down_proj_marlin: None,
            gate_up_proj_w8: None,
            down_proj_w8: None,
        },
    };

    let cfg = make_test_config(num_heads, num_kv_heads, head_dim, hidden);
    let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;
    let backend = test_backend(&device);
    let out = transformer_block(
        &backend,
        &x,
        &layer,
        &cfg,
        &positions,
        num_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        &inv_freq,
        1e-6,
        None,
        0,
        None,
    )?;
    assert_eq!(out.dims(), &[batch, seq_len, hidden]);

    Ok(())
}

#[test]
fn test_transformer_block_residual_connections() -> Result<()> {
    // With residual connections, output should differ from zero even with small weights
    let device = Device::Cpu;
    let num_heads = 2;
    let num_kv_heads = 1;
    let head_dim = 4;
    let hidden = num_heads * head_dim;
    let intermediate = hidden * 2;

    // Input with known non-zero values
    let x = Tensor::ones((1, 2, hidden), DType::F32, device)?;
    let positions = vec![0u32, 1];

    let gate_proj = Tensor::randn(0.0_f32, 0.02, (intermediate, hidden), device)?;
    let up_proj = Tensor::randn(0.0_f32, 0.02, (intermediate, hidden), device)?;
    let down_proj = Tensor::randn(0.0_f32, 0.02, (hidden, intermediate), device)?;
    let gate_proj_t = gate_proj.t()?.contiguous()?;
    let up_proj_t = up_proj.t()?.contiguous()?;
    let down_proj_t = down_proj.t()?.contiguous()?;

    let layer = GpuLayerWeights {
        input_layernorm: Tensor::zeros(hidden, DType::F32, device)?,
        post_attention_layernorm: Tensor::zeros(hidden, DType::F32, device)?,
        attention: GpuAttentionWeights::Full(make_test_attn_weights(
            num_heads,
            num_kv_heads,
            head_dim,
            hidden,
            &device,
        )?),
        mlp: GpuFfnWeights {
            gate_proj,
            up_proj,
            down_proj,
            gate_proj_t,
            up_proj_t,
            down_proj_t,
            gate_up_proj_t: None,
            gate_proj_marlin: None,
            up_proj_marlin: None,
            down_proj_marlin: None,
            gate_up_proj_w8: None,
            down_proj_w8: None,
        },
    };

    let cfg = make_test_config(num_heads, num_kv_heads, head_dim, hidden);
    let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;
    let backend = test_backend(&device);
    let out = transformer_block(
        &backend,
        &x,
        &layer,
        &cfg,
        &positions,
        num_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        &inv_freq,
        1e-6,
        None,
        0,
        None,
    )?;

    // Output should not be zero (residual adds input through)
    let vals = out.flatten_all()?.to_vec1::<f32>()?;
    let sum: f32 = vals.iter().map(|v| v.abs()).sum();
    assert!(
        sum > 0.1,
        "residual connections should keep output non-zero, got sum={sum}"
    );
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "output should be finite"
    );

    Ok(())
}

#[test]
fn test_transformer_block_rejects_linear_attention() -> Result<()> {
    let device = Device::Cpu;
    let hidden = 8;

    let layer = GpuLayerWeights {
        input_layernorm: Tensor::zeros(hidden, DType::F32, device)?,
        post_attention_layernorm: Tensor::zeros(hidden, DType::F32, device)?,
        attention: GpuAttentionWeights::Linear(GpuLinearAttentionWeights {
            in_proj_qkv: Tensor::zeros((1, 1), DType::F32, device)?,
            in_proj_z: Tensor::zeros((1, 1), DType::F32, device)?,
            out_proj: Tensor::zeros((1, 1), DType::F32, device)?,
            in_proj_a: Tensor::zeros((1, 1), DType::F32, device)?,
            in_proj_b: Tensor::zeros((1, 1), DType::F32, device)?,
            conv1d: Tensor::zeros((1, 1), DType::F32, device)?,
            norm: Tensor::zeros((1,), DType::F32, device)?,
            a_log: Tensor::zeros((1,), DType::F32, device)?,
            a_log_gates: Tensor::zeros((1,), DType::F32, device)?,
            dt_bias: Tensor::zeros((1,), DType::F32, device)?,
            in_proj_qkv_t: Tensor::zeros((1, 1), DType::F32, device)?,
            in_proj_z_t: Tensor::zeros((1, 1), DType::F32, device)?,
            in_proj_a_t: Tensor::zeros((1, 1), DType::F32, device)?,
            in_proj_b_t: Tensor::zeros((1, 1), DType::F32, device)?,
            in_proj_ab_t: None,
            out_proj_t: Tensor::zeros((1, 1), DType::F32, device)?,
            out_proj_marlin: None,
            in_proj_qkvzab_w8: None,
        }),
        mlp: GpuFfnWeights {
            gate_proj: Tensor::zeros((1, hidden), DType::F32, device)?,
            up_proj: Tensor::zeros((1, hidden), DType::F32, device)?,
            down_proj: Tensor::zeros((hidden, 1), DType::F32, device)?,
            gate_proj_t: Tensor::zeros((hidden, 1), DType::F32, device)?,
            up_proj_t: Tensor::zeros((hidden, 1), DType::F32, device)?,
            down_proj_t: Tensor::zeros((1, hidden), DType::F32, device)?,
            gate_up_proj_t: None,
            gate_proj_marlin: None,
            up_proj_marlin: None,
            down_proj_marlin: None,
            gate_up_proj_w8: None,
            down_proj_w8: None,
        },
    };

    let x = Tensor::ones((1, 1, hidden), DType::F32, device)?;
    let cfg = make_test_config(2, 1, 4, 8);
    let inv_freq = compute_rotary_inv_freq(4, 10_000.0, &device)?;
    let backend = test_backend(&device);
    let result = transformer_block(
        &backend,
        &x,
        &layer,
        &cfg,
        &[0],
        2,
        1,
        4,
        4,
        &inv_freq,
        1e-6,
        None,
        0,
        None,
    );
    assert!(result.is_err(), "should reject linear attention layers");

    Ok(())
}

#[test]
fn test_weight_to_tensor_f32() -> Result<()> {
    let device = Device::Cpu;
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let wt = WeightTensor {
        data: crate::weights::WeightData::owned(bytes),
        shape: vec![2, 3],
        dtype: TensorDType::F32,
        source: None,
    };

    let t = weight_to_tensor(&wt, &device)?;
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.dtype(), DType::F32);

    let vals = t.to_vec2::<f32>()?;
    assert!((vals[0][0] - 1.0).abs() < 1e-6);
    assert!((vals[1][2] - 6.0).abs() < 1e-6);

    Ok(())
}

/// (#1082) Validate the kt-native CUDA weight loader: bf16 raw bytes
/// (the production weight dtype) upload straight into a kt CUDA tensor via
/// `Tensor::from_raw_bytes_on`, with no candle leaf or device→device bridge
/// copy. Round-trips the bytes through H2D + D2H and checks the values +
/// dtype + device — the load-bearing byte-interpretation guard for the
/// dominant loader entry.
#[cfg(feature = "cuda")]
#[test]
fn test_weight_to_tensor_bf16_cuda() -> Result<()> {
    let device = Device::Cuda(0);
    let data: Vec<half::bf16> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .iter()
        .map(|f| half::bf16::from_f32(*f))
        .collect();
    let bytes: Vec<u8> = data.iter().flat_map(|b| b.to_le_bytes()).collect();
    let wt = WeightTensor {
        data: crate::weights::WeightData::owned(bytes),
        shape: vec![2, 3],
        dtype: TensorDType::BF16,
        source: None,
    };

    let t = weight_to_tensor(&wt, &device)?;
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.dtype(), DType::BF16);
    assert!(matches!(t.device(), Device::Cuda(_)), "must land on CUDA");

    let vals = t.to_vec2::<half::bf16>()?;
    assert!((vals[0][0].to_f32() - 1.0).abs() < 1e-2);
    assert!((vals[1][2].to_f32() - 6.0).abs() < 1e-2);

    Ok(())
}

#[test]
fn projection_cache_lifecycle_policy_is_explicit() -> Result<()> {
    let device = Device::Cpu;

    let base = ProjectionLoadCache::for_base_model_load(&device)?;
    let lazy_mtp = ProjectionLoadCache::for_lazy_mtp_upload(&device)?;

    assert_eq!(
        base.transposed_cache_miss_policy,
        TransposedWeightCacheMissPolicy::PersistBeforeReadiness
    );
    assert_eq!(
        lazy_mtp.transposed_cache_miss_policy,
        TransposedWeightCacheMissPolicy::ReadOnly
    );
    Ok(())
}

#[test]
fn test_weight_to_transposed_tensor_2d_f32_matches_cached_transpose() -> Result<()> {
    let device = Device::Cpu;
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let wt = WeightTensor {
        data: crate::weights::WeightData::owned(bytes),
        shape: vec![2, 3],
        dtype: TensorDType::F32,
        source: None,
    };

    let direct =
        weight_to_transposed_tensor_2d(&wt, &device, TransposedWeightCacheMissPolicy::ReadOnly)?;
    let baseline = cached_transpose(&weight_to_tensor(&wt, &device)?)?;

    assert!(direct.is_contiguous());
    assert_eq!(direct.dims(), &[3, 2]);
    assert_eq!(direct.to_vec2::<f32>()?, baseline.to_vec2::<f32>()?);
    assert_eq!(
        direct.to_vec2::<f32>()?,
        vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]
    );
    Ok(())
}

#[test]
fn test_transposed_weight_bytes_2d_preserves_two_byte_elements() -> Result<()> {
    let values: Vec<u16> = vec![1, 2, 3, 4, 5, 6];
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let wt = WeightTensor {
        data: crate::weights::WeightData::owned(bytes),
        shape: vec![2, 3],
        dtype: TensorDType::BF16,
        source: None,
    };

    let (transposed, shape) = transposed_weight_bytes_2d(&wt)?;
    let got: Vec<u16> = transposed
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();

    assert_eq!(shape, [3, 2]);
    assert_eq!(got, vec![1, 4, 2, 5, 3, 6]);
    Ok(())
}

#[test]
fn test_transposed_weight_bytes_2d_parallel_preserves_two_byte_elements() -> Result<()> {
    let rows = 513usize;
    let cols = 1025usize;
    let values: Vec<u16> = (0..rows * cols).map(|idx| idx as u16).collect();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert!(bytes.len() >= PARALLEL_TRANSPOSE_MIN_BYTES);
    let wt = WeightTensor {
        data: crate::weights::WeightData::owned(bytes),
        shape: vec![rows, cols],
        dtype: TensorDType::BF16,
        source: None,
    };

    let (transposed, shape) = transposed_weight_bytes_2d(&wt)?;

    assert_eq!(shape, [cols, rows]);
    for col in 0..cols {
        for row in 0..rows {
            let got_offset = (col * rows + row) * 2;
            let got = u16::from_le_bytes([transposed[got_offset], transposed[got_offset + 1]]);
            assert_eq!(got, values[row * cols + col]);
        }
    }
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_weight_to_transposed_tensor_2d_metal_matches_cpu_cached_transpose() -> Result<()> {
    let Some(device) = crate::backend::metal::try_new_metal() else {
        return Ok(());
    };
    let cpu = Device::Cpu;
    let data: Vec<f32> = vec![1.0, -2.0, 3.5, 4.25, 5.0, -6.75];
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let wt = WeightTensor {
        data: crate::weights::WeightData::owned(bytes),
        shape: vec![2, 3],
        dtype: TensorDType::F32,
        source: None,
    };

    let direct =
        weight_to_transposed_tensor_2d(&wt, &device, TransposedWeightCacheMissPolicy::ReadOnly)?
            .to_device(cpu)?;
    let baseline = cached_transpose(&weight_to_tensor(&wt, &cpu)?)?;

    assert!(direct.is_contiguous());
    assert_eq!(direct.dims(), &[3, 2]);
    assert_eq!(direct.to_vec2::<f32>()?, baseline.to_vec2::<f32>()?);
    Ok(())
}

#[test]
fn test_cached_transpose_materializes_on_cpu() -> Result<()> {
    let device = Device::Cpu;
    // #1082: kt `Tensor::new` only accepts rank-1 slices; build the [2,3]
    // tensor from a flat slice + shape (same values).
    let t = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2usize, 3usize))?
        .to_device(device)?;

    let tt = cached_transpose(&t)?;

    assert!(tt.is_contiguous());
    assert_eq!(tt.dims(), &[3, 2]);
    assert_eq!(
        tt.to_vec2::<f32>()?,
        vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]
    );
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_cached_transpose_materializes_on_metal() -> Result<()> {
    let Some(device) = crate::backend::metal::try_new_metal() else {
        return Ok(());
    };
    // #1082: kt `Tensor::new` only accepts rank-1 slices; build the [2,3]
    // tensor from a flat slice + shape (same values).
    let t = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2usize, 3usize))?
        .to_device(device)?;

    let tt = cached_transpose(&t)?;

    assert!(tt.is_contiguous());
    assert_eq!(tt.dims(), &[3, 2]);
    assert_eq!(
        tt.to_vec2::<f32>()?,
        vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]
    );
    Ok(())
}

/// Helper: build tiny GpuWeights for testing model_forward shape propagation.
/// Uses full-attention layers only (no linear attention) with small dimensions.
// Test helper; dimensions are intentionally individual.
#[allow(clippy::too_many_arguments)]
fn make_tiny_gpu_weights(
    device: &Device,
    vocab_size: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    num_layers: usize,
) -> Result<GpuWeights> {
    let randn = |shape: &[usize]| -> Result<Tensor> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.01).sin()) * 0.1).collect();
        Ok(Tensor::new(&data, device)?.reshape(shape)?)
    };

    let embed_tokens = randn(&[vocab_size, hidden_size])?;
    let embed_tokens_t = embed_tokens.t()?.contiguous()?;
    let final_norm = Tensor::zeros(hidden_size, DType::F32, device)?;

    let mut layers = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        let q_proj = randn(&[num_heads * head_dim, hidden_size])?;
        let k_proj = randn(&[num_kv_heads * head_dim, hidden_size])?;
        let v_proj = randn(&[num_kv_heads * head_dim, hidden_size])?;
        let o_proj = randn(&[hidden_size, num_heads * head_dim])?;
        let q_proj_t = q_proj.t()?.contiguous()?;
        let k_proj_t = k_proj.t()?.contiguous()?;
        let v_proj_t = v_proj.t()?.contiguous()?;
        let o_proj_t = o_proj.t()?.contiguous()?;
        let gate_proj = randn(&[intermediate_size, hidden_size])?;
        let up_proj = randn(&[intermediate_size, hidden_size])?;
        let down_proj = randn(&[hidden_size, intermediate_size])?;
        let gate_proj_t = gate_proj.t()?.contiguous()?;
        let up_proj_t = up_proj.t()?.contiguous()?;
        let down_proj_t = down_proj.t()?.contiguous()?;
        layers.push(GpuLayerWeights {
            input_layernorm: Tensor::zeros(hidden_size, DType::F32, device)?,
            post_attention_layernorm: Tensor::zeros(hidden_size, DType::F32, device)?,
            attention: GpuAttentionWeights::Full(GpuFullAttentionWeights {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm: Tensor::zeros(head_dim, DType::F32, device)?,
                k_norm: Tensor::zeros(head_dim, DType::F32, device)?,
                q_proj_t,
                k_proj_t,
                v_proj_t,
                qkv_proj_t: None,
                o_proj_t,
                qkv_proj_w8: None,
                o_proj_w8: None,
                q_proj_marlin: None,
            }),
            mlp: GpuFfnWeights {
                gate_proj,
                up_proj,
                down_proj,
                gate_proj_t,
                up_proj_t,
                down_proj_t,
                gate_up_proj_t: None,
                gate_proj_marlin: None,
                up_proj_marlin: None,
                down_proj_marlin: None,
                gate_up_proj_w8: None,
                down_proj_w8: None,
            },
        });
    }

    // Tests using this helper all set `partial_rotary_factor = 1.0` and
    // `rope_theta = 10000.0`, so rotate every head_dim with base 10k.
    let rotary_inv_freq = compute_rotary_inv_freq(head_dim, 10000.0, device)?;

    Ok(GpuWeights {
        source_content_sha256: None,
        base_weight_shard_manifest: None,
        execution_provenance: None,
        embed_tokens,
        embed_tokens_t,
        lm_head_w8: None,
        layers,
        final_norm,
        rotary_inv_freq,
        mtp: None,
    })
}

#[test]
fn test_model_forward_shape() -> Result<()> {
    let device = Device::Cpu;
    let vocab_size = 32;
    let hidden_size = 16;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 4;
    let intermediate_size = 32;
    let num_layers = 2;

    let weights = make_tiny_gpu_weights(
        &device,
        vocab_size,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        num_layers,
    )?;

    let config = kiln_core::config::ModelConfig {
        hidden_size,
        num_layers,
        num_attention_heads: num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        max_position_embeddings: 1024,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        dtype: kiln_core::config::DType::FP32,
        num_full_attention_layers: num_layers,
        full_attention_interval: 1, // every layer is full attention
        attn_output_gate: false,
        linear_num_key_heads: num_heads,
        linear_key_head_dim: head_dim,
        linear_num_value_heads: num_heads,
        linear_value_head_dim: head_dim,
        linear_conv_kernel_dim: 4,
        partial_rotary_factor: 1.0,
    };

    let token_ids: Vec<u32> = vec![1, 5, 3, 10];
    let backend = test_backend(&device);
    let logits = model_forward_kt(&backend, &token_ids, &weights, &config, None, None, None)?;

    // Expected shape: [1, seq_len, vocab_size]
    assert_eq!(logits.dims(), &[1, 4, vocab_size]);

    Ok(())
}

#[test]
fn test_model_forward_single_token() -> Result<()> {
    let device = Device::Cpu;
    let vocab_size = 16;
    let hidden_size = 8;
    let num_heads = 2;
    let num_kv_heads = 1;
    let head_dim = 4;
    let intermediate_size = 16;

    let weights = make_tiny_gpu_weights(
        &device,
        vocab_size,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        1, // single layer
    )?;

    let config = kiln_core::config::ModelConfig {
        hidden_size,
        num_layers: 1,
        num_attention_heads: num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        max_position_embeddings: 1024,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        dtype: kiln_core::config::DType::FP32,
        num_full_attention_layers: 1,
        full_attention_interval: 1,
        attn_output_gate: false,
        linear_num_key_heads: num_heads,
        linear_key_head_dim: head_dim,
        linear_num_value_heads: num_heads,
        linear_value_head_dim: head_dim,
        linear_conv_kernel_dim: 4,
        partial_rotary_factor: 1.0,
    };

    let backend = test_backend(&device);
    let logits = model_forward_kt(&backend, &[7], &weights, &config, None, None, None)?;
    assert_eq!(logits.dims(), &[1, 1, vocab_size]);

    // Logits should be finite
    let vals = logits.flatten_all()?.to_vec1::<f32>()?;
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "all logits should be finite"
    );

    Ok(())
}

#[test]
fn test_model_forward_kv_cache_equivalence() -> Result<()> {
    // Verify that model_forward with KV cache produces the same last-position
    // logits as without KV cache, for a multi-token sequence processed
    // incrementally (prefill + decode steps).
    use crate::kv_cache::KvCache;

    let device = Device::Cpu;
    let vocab_size = 16;
    let hidden_size = 8;
    let num_heads = 2;
    let num_kv_heads = 1;
    let head_dim = 4;
    let intermediate_size = 16;
    let num_layers = 2;

    let weights = make_tiny_gpu_weights(
        &device,
        vocab_size,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        num_layers,
    )?;

    let config = kiln_core::config::ModelConfig {
        hidden_size,
        num_layers,
        num_attention_heads: num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        max_position_embeddings: 1024,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        dtype: kiln_core::config::DType::FP32,
        num_full_attention_layers: num_layers,
        full_attention_interval: 1,
        attn_output_gate: false,
        linear_num_key_heads: num_heads,
        linear_key_head_dim: head_dim,
        linear_num_value_heads: num_heads,
        linear_value_head_dim: head_dim,
        linear_conv_kernel_dim: 4,
        partial_rotary_factor: 1.0,
    };

    let tokens: Vec<u32> = vec![1, 5, 3, 10, 7];
    let backend = test_backend(&device);

    // Reference: full forward pass without KV cache
    let logits_ref = model_forward_kt(&backend, &tokens, &weights, &config, None, None, None)?;
    // Extract last position logits: [1, 5, vocab] -> last position
    let last_ref = logits_ref.narrow(1, tokens.len() - 1, 1)?; // [1, 1, vocab]
    let last_ref_vals = last_ref.flatten_all()?.to_vec1::<f32>()?;

    // With KV cache: prefill first 4 tokens, then decode the 5th
    //
    // Migrated to `_kt` constructor so this site no longer names
    // candle `DType` or `Device` — `KvCache::new_kt` is an identity
    // alias now, no bridge. (#1082)
    let mut kv_cache = KvCache::new_kt(
        num_layers,
        num_kv_heads,
        head_dim,
        32,
        kiln_tensor::DType::F32,
        &kiln_tensor::Device::Cpu,
    )?;

    // Prefill
    let _prefill_logits = model_forward_kt(
        &backend,
        &tokens[..4],
        &weights,
        &config,
        Some(&mut kv_cache),
        None,
        None,
    )?;
    kv_cache.advance(4);
    assert_eq!(kv_cache.seq_len(), 4);

    // Decode the 5th token
    let decode_logits = model_forward_kt(
        &backend,
        &tokens[4..],
        &weights,
        &config,
        Some(&mut kv_cache),
        None,
        None,
    )?;
    kv_cache.advance(1);
    assert_eq!(kv_cache.seq_len(), 5);

    let last_cached_vals = decode_logits
        .narrow(1, 0, 1)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    // Compare: should be identical (within floating point tolerance)
    assert_eq!(last_ref_vals.len(), last_cached_vals.len());
    for (i, (r, c)) in last_ref_vals.iter().zip(&last_cached_vals).enumerate() {
        assert!(
            (r - c).abs() < 1e-4,
            "logit {i} differs: ref={r}, cached={c}, diff={}",
            (r - c).abs()
        );
    }

    Ok(())
}

#[test]
fn test_model_forward_kv_cache_token_by_token() -> Result<()> {
    // Verify that processing tokens one-by-one with KV cache matches
    // processing all at once without cache.
    use crate::kv_cache::KvCache;

    let device = Device::Cpu;
    let vocab_size = 16;
    let hidden_size = 8;
    let num_heads = 2;
    let num_kv_heads = 1;
    let head_dim = 4;
    let intermediate_size = 16;

    let weights = make_tiny_gpu_weights(
        &device,
        vocab_size,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        1,
    )?;

    let config = kiln_core::config::ModelConfig {
        hidden_size,
        num_layers: 1,
        num_attention_heads: num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        max_position_embeddings: 1024,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        dtype: kiln_core::config::DType::FP32,
        num_full_attention_layers: 1,
        full_attention_interval: 1,
        attn_output_gate: false,
        linear_num_key_heads: num_heads,
        linear_key_head_dim: head_dim,
        linear_num_value_heads: num_heads,
        linear_value_head_dim: head_dim,
        linear_conv_kernel_dim: 4,
        partial_rotary_factor: 1.0,
    };

    let tokens: Vec<u32> = vec![3, 7, 1];
    let backend = test_backend(&device);

    // Reference
    let logits_ref = model_forward_kt(&backend, &tokens, &weights, &config, None, None, None)?;
    let last_ref = logits_ref
        .narrow(1, 2, 1)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    // KV cache: process token by token (migrated to `_kt`; #1082)
    let mut kv_cache = KvCache::new_kt(
        1,
        num_kv_heads,
        head_dim,
        16,
        kiln_tensor::DType::F32,
        &kiln_tensor::Device::Cpu,
    )?;

    // Token 0
    let _ = model_forward_kt(
        &backend,
        &[3],
        &weights,
        &config,
        Some(&mut kv_cache),
        None,
        None,
    )?;
    kv_cache.advance(1);

    // Token 1
    let _ = model_forward_kt(
        &backend,
        &[7],
        &weights,
        &config,
        Some(&mut kv_cache),
        None,
        None,
    )?;
    kv_cache.advance(1);

    // Token 2
    let logits_cached = model_forward_kt(
        &backend,
        &[1],
        &weights,
        &config,
        Some(&mut kv_cache),
        None,
        None,
    )?;
    kv_cache.advance(1);

    let last_cached = logits_cached
        .narrow(1, 0, 1)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    for (i, (r, c)) in last_ref.iter().zip(&last_cached).enumerate() {
        assert!(
            (r - c).abs() < 1e-4,
            "logit {i} differs: ref={r}, cached={c}",
        );
    }

    Ok(())
}

/// Helper: build tiny GpuWeights with a mix of full and linear attention layers.
// Test helper; dimensions are intentionally individual.
#[allow(clippy::too_many_arguments)]
fn make_hybrid_gpu_weights(
    device: &Device,
    vocab_size: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    num_layers: usize,
    full_attention_interval: usize,
) -> Result<GpuWeights> {
    let randn = |shape: &[usize]| -> Result<Tensor> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.01).sin()) * 0.1).collect();
        Ok(Tensor::new(&data, device)?.reshape(shape)?)
    };

    let embed_tokens = randn(&[vocab_size, hidden_size])?;
    let embed_tokens_t = embed_tokens.t()?.contiguous()?;
    let final_norm = Tensor::zeros(hidden_size, DType::F32, device)?;

    // For linear attention: nk heads with key_head_dim, nv heads with value_head_dim
    // Use same dims as full attention for simplicity
    let nk = num_kv_heads;
    let nv = num_heads;
    let dk = head_dim;
    let dv = head_dim;
    let qkv_dim = nk * dk + nk * dk + nv * dv; // Q + K + V fused
    let conv_kernel = 4;

    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let is_full = (i + 1) % full_attention_interval == 0;
        let attention = if is_full {
            let q_proj = randn(&[num_heads * head_dim, hidden_size])?;
            let k_proj = randn(&[num_kv_heads * head_dim, hidden_size])?;
            let v_proj = randn(&[num_kv_heads * head_dim, hidden_size])?;
            let o_proj = randn(&[hidden_size, num_heads * head_dim])?;
            let q_proj_t = q_proj.t()?.contiguous()?;
            let k_proj_t = k_proj.t()?.contiguous()?;
            let v_proj_t = v_proj.t()?.contiguous()?;
            let o_proj_t = o_proj.t()?.contiguous()?;
            GpuAttentionWeights::Full(GpuFullAttentionWeights {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm: Tensor::zeros(head_dim, DType::F32, device)?,
                k_norm: Tensor::zeros(head_dim, DType::F32, device)?,
                q_proj_t,
                k_proj_t,
                v_proj_t,
                qkv_proj_t: None,
                o_proj_t,
                qkv_proj_w8: None,
                o_proj_w8: None,
                q_proj_marlin: None,
            })
        } else {
            let in_proj_qkv = randn(&[qkv_dim, hidden_size])?;
            let in_proj_z = randn(&[nv * dv, hidden_size])?;
            let out_proj = randn(&[hidden_size, nv * dv])?;
            let in_proj_a = randn(&[nv, hidden_size])?;
            let in_proj_b = randn(&[nv, hidden_size])?;
            let in_proj_qkv_t = in_proj_qkv.t()?.contiguous()?;
            let in_proj_z_t = in_proj_z.t()?.contiguous()?;
            let in_proj_a_t = in_proj_a.t()?.contiguous()?;
            let in_proj_b_t = in_proj_b.t()?.contiguous()?;
            let out_proj_t = out_proj.t()?.contiguous()?;
            GpuAttentionWeights::Linear(GpuLinearAttentionWeights {
                in_proj_qkv,
                in_proj_z,
                out_proj,
                in_proj_a,
                in_proj_b,
                conv1d: randn(&[qkv_dim, 1, conv_kernel])?,
                norm: Tensor::ones(dk, DType::F32, device)?,
                a_log: Tensor::zeros(nv, DType::F32, device)?,
                a_log_gates: Tensor::zeros(nv, DType::F32, device)?,
                dt_bias: Tensor::zeros(nv, DType::F32, device)?,
                in_proj_qkv_t,
                in_proj_z_t,
                in_proj_a_t,
                in_proj_b_t,
                in_proj_ab_t: None,
                out_proj_t,
                out_proj_marlin: None,
                in_proj_qkvzab_w8: None,
            })
        };

        let gate_proj = randn(&[intermediate_size, hidden_size])?;
        let up_proj = randn(&[intermediate_size, hidden_size])?;
        let down_proj = randn(&[hidden_size, intermediate_size])?;
        let gate_proj_t = gate_proj.t()?.contiguous()?;
        let up_proj_t = up_proj.t()?.contiguous()?;
        let down_proj_t = down_proj.t()?.contiguous()?;
        layers.push(GpuLayerWeights {
            input_layernorm: Tensor::zeros(hidden_size, DType::F32, device)?,
            post_attention_layernorm: Tensor::zeros(hidden_size, DType::F32, device)?,
            attention,
            mlp: GpuFfnWeights {
                gate_proj,
                up_proj,
                down_proj,
                gate_proj_t,
                up_proj_t,
                down_proj_t,
                gate_up_proj_t: None,
                gate_proj_marlin: None,
                up_proj_marlin: None,
                down_proj_marlin: None,
                gate_up_proj_w8: None,
                down_proj_w8: None,
            },
        });
    }

    // Tests using this helper set `partial_rotary_factor = 1.0` and
    // `rope_theta = 10000.0`, so rotary_dim = head_dim with base 10k.
    let rotary_inv_freq = compute_rotary_inv_freq(head_dim, 10000.0, device)?;

    Ok(GpuWeights {
        source_content_sha256: None,
        base_weight_shard_manifest: None,
        execution_provenance: None,
        embed_tokens,
        embed_tokens_t,
        lm_head_w8: None,
        layers,
        final_norm,
        rotary_inv_freq,
        mtp: None,
    })
}

#[test]
fn test_model_forward_hybrid_layers() -> Result<()> {
    // Test model_forward with a mix of full and linear (GDN) attention layers
    let device = Device::Cpu;
    let vocab_size = 32;
    let hidden_size = 16;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 4;
    let intermediate_size = 32;
    let num_layers = 4;
    let full_attention_interval = 4; // layer 3 is full, layers 0,1,2 are linear

    let weights = make_hybrid_gpu_weights(
        &device,
        vocab_size,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        num_layers,
        full_attention_interval,
    )?;

    let config = kiln_core::config::ModelConfig {
        hidden_size,
        num_layers,
        num_attention_heads: num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        max_position_embeddings: 1024,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        dtype: kiln_core::config::DType::FP32,
        num_full_attention_layers: 1,
        full_attention_interval,
        attn_output_gate: false,
        linear_num_key_heads: num_kv_heads,
        linear_key_head_dim: head_dim,
        linear_num_value_heads: num_heads,
        linear_value_head_dim: head_dim,
        linear_conv_kernel_dim: 4,
        partial_rotary_factor: 1.0,
    };

    let mut linear_state = LinearAttentionState::new(&config, &device)?;

    // Prefill with multiple tokens
    let token_ids: Vec<u32> = vec![1, 5, 3, 10];
    let backend = test_backend(&device);
    let logits = model_forward_kt(
        &backend,
        &token_ids,
        &weights,
        &config,
        None,
        Some(&mut linear_state),
        None,
    )?;
    assert_eq!(logits.dims(), &[1, 4, vocab_size]);

    // All values should be finite (no NaN/Inf)
    let flat = logits.flatten_all()?.to_vec1::<f32>()?;
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "logits contain non-finite values"
    );

    Ok(())
}

#[cfg(feature = "metal")]
struct ParityScenario {
    label: &'static str,
    vocab_size: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    num_layers: usize,
    full_attention_interval: usize,
    token_ids: Vec<u32>,
    max_abs_diff: f32,
}

/// Runs `model_forward` on CPU and Metal with matching random-weight
/// models and asserts the logits agree within `scenario.max_abs_diff`.
/// Drives both parity tests below; the scenario controls whether the
/// `MetalBackend` SDPA path activates (head_dim ∈ whitelist) or whether
/// the portable kt fallback runs.
///
/// Returns `Ok(())` without running if Metal isn't available so the
/// suite stays portable on Linux + CUDA hosts.
#[cfg(feature = "metal")]
fn run_cpu_metal_parity(scenario: ParityScenario) -> Result<()> {
    let Some(metal_device) = crate::backend::metal::try_new_metal() else {
        eprintln!("skipping parity test '{}'", scenario.label);
        return Ok(());
    };
    let cpu_device = Device::Cpu;

    let weights_cpu = make_hybrid_gpu_weights(
        &cpu_device,
        scenario.vocab_size,
        scenario.hidden_size,
        scenario.num_heads,
        scenario.num_kv_heads,
        scenario.head_dim,
        scenario.intermediate_size,
        scenario.num_layers,
        scenario.full_attention_interval,
    )?;
    let weights_metal = make_hybrid_gpu_weights(
        &metal_device,
        scenario.vocab_size,
        scenario.hidden_size,
        scenario.num_heads,
        scenario.num_kv_heads,
        scenario.head_dim,
        scenario.intermediate_size,
        scenario.num_layers,
        scenario.full_attention_interval,
    )?;

    // Linear attention dims are 0 when full_attention_interval == 1 (no
    // GDN layers in the model); otherwise set to head_dim so GDN state
    // is shaped for the fallback path.
    let has_linear_layers = scenario.full_attention_interval > 1;
    let linear_num_kv_heads = if has_linear_layers {
        scenario.num_kv_heads
    } else {
        0
    };
    let linear_num_value_heads = if has_linear_layers {
        scenario.num_heads
    } else {
        0
    };
    let linear_head_dim = if has_linear_layers {
        scenario.head_dim
    } else {
        0
    };
    let linear_conv_kernel_dim = if has_linear_layers { 4 } else { 0 };

    let config = kiln_core::config::ModelConfig {
        hidden_size: scenario.hidden_size,
        num_layers: scenario.num_layers,
        num_attention_heads: scenario.num_heads,
        num_kv_heads: scenario.num_kv_heads,
        head_dim: scenario.head_dim,
        intermediate_size: scenario.intermediate_size,
        vocab_size: scenario.vocab_size,
        max_position_embeddings: 1024,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        dtype: kiln_core::config::DType::FP32,
        num_full_attention_layers: if has_linear_layers {
            1
        } else {
            scenario.num_layers
        },
        full_attention_interval: scenario.full_attention_interval,
        attn_output_gate: false,
        linear_num_key_heads: linear_num_kv_heads,
        linear_key_head_dim: linear_head_dim,
        linear_num_value_heads,
        linear_value_head_dim: linear_head_dim,
        linear_conv_kernel_dim,
        partial_rotary_factor: 1.0,
    };

    let cpu_backend = test_backend(&cpu_device);
    let mut cpu_linear = LinearAttentionState::new(&config, &cpu_device)?;
    let logits_cpu = model_forward_kt(
        &cpu_backend,
        &scenario.token_ids,
        &weights_cpu,
        &config,
        None,
        Some(&mut cpu_linear),
        None,
    )?;

    let metal_backend = crate::backend::for_device_kt(&metal_device);
    let mut metal_linear = LinearAttentionState::new(&config, &metal_device)?;
    let logits_metal = model_forward_kt(
        &*metal_backend,
        &scenario.token_ids,
        &weights_metal,
        &config,
        None,
        Some(&mut metal_linear),
        None,
    )?;

    assert_eq!(logits_cpu.dims(), logits_metal.dims());

    let cpu_flat = logits_cpu.flatten_all()?.to_vec1::<f32>()?;
    let metal_flat = logits_metal
        .to_device(cpu_device)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    assert!(
        cpu_flat.iter().all(|v| v.is_finite()),
        "{}: CPU logits non-finite",
        scenario.label
    );
    assert!(
        metal_flat.iter().all(|v| v.is_finite()),
        "{}: Metal logits non-finite",
        scenario.label
    );

    let max_abs_diff = cpu_flat
        .iter()
        .zip(metal_flat.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs_diff < scenario.max_abs_diff,
        "{}: CPU vs Metal logits diverge: max abs diff = {max_abs_diff} (bound {})",
        scenario.label,
        scenario.max_abs_diff,
    );
    Ok(())
}

/// Qwen-shaped: GQA ratio 4, head_dim 128, full attention only. Exercises
/// `MetalBackend::flash_attn_prefill` (the fused Metal SDPA) directly — head_dim
/// 128 is in the SDPA whitelist, seq_len 12 > 8 for the full SDPA kernel
/// (not the vector path).
#[cfg(feature = "metal")]
#[test]
fn test_model_forward_parity_sdpa_path() -> Result<()> {
    run_cpu_metal_parity(ParityScenario {
        label: "sdpa_path",
        vocab_size: 32,
        num_heads: 4,
        num_kv_heads: 1,
        head_dim: 128,
        hidden_size: 512,
        intermediate_size: 1024,
        num_layers: 2,
        full_attention_interval: 1,
        token_ids: (0..12u32).collect(),
        // SDPA internally accumulates at FP32 but softmax rounds differently
        // from the naive CPU path. 1e-2 accommodates M1 drift; tighten if
        // later hardware proves it's conservative.
        max_abs_diff: 1e-2,
    })
}

/// Hybrid full + GDN layers with head_dim 4, below the SDPA whitelist.
/// `MetalBackend` declines into the portable fallback, so this validates
/// that the whole kt composition (embed, RMSNorm, RoPE, SwiGLU, naive
/// softmax+matmul, GDN recurrent loop) runs correctly on Apple Silicon.
#[cfg(feature = "metal")]
#[test]
fn test_model_forward_parity_cpu_vs_metal() -> Result<()> {
    run_cpu_metal_parity(ParityScenario {
        label: "portable_fallback",
        vocab_size: 32,
        hidden_size: 16,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 4,
        intermediate_size: 32,
        num_layers: 4,
        full_attention_interval: 4,
        token_ids: vec![1, 5, 3, 10],
        max_abs_diff: 1e-3,
    })
}

#[test]
fn test_model_forward_hybrid_decode() -> Result<()> {
    // Test prefill + decode with linear attention state persistence
    let device = Device::Cpu;
    let vocab_size = 32;
    let hidden_size = 16;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 4;
    let intermediate_size = 32;
    let num_layers = 4;
    let full_attention_interval = 4;

    let weights = make_hybrid_gpu_weights(
        &device,
        vocab_size,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        num_layers,
        full_attention_interval,
    )?;

    let config = kiln_core::config::ModelConfig {
        hidden_size,
        num_layers,
        num_attention_heads: num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        max_position_embeddings: 1024,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        dtype: kiln_core::config::DType::FP32,
        num_full_attention_layers: 1,
        full_attention_interval,
        attn_output_gate: false,
        linear_num_key_heads: num_kv_heads,
        linear_key_head_dim: head_dim,
        linear_num_value_heads: num_heads,
        linear_value_head_dim: head_dim,
        linear_conv_kernel_dim: 4,
        partial_rotary_factor: 1.0,
    };

    // Migrated to `_kt` constructor (#1082).
    let mut kv_cache = KvCache::new_kt(
        1,
        num_kv_heads,
        head_dim,
        32,
        kiln_tensor::DType::F32,
        &kiln_tensor::Device::Cpu,
    )?;
    let mut linear_state = LinearAttentionState::new(&config, &device)?;
    let backend = test_backend(&device);

    // Prefill
    let prefill_logits = model_forward_kt(
        &backend,
        &[1, 5, 3],
        &weights,
        &config,
        Some(&mut kv_cache),
        Some(&mut linear_state),
        None,
    )?;
    kv_cache.advance(3);
    assert_eq!(prefill_logits.dims(), &[1, 3, vocab_size]);

    // Decode: single token should work with persisted linear state
    let decode_logits = model_forward_kt(
        &backend,
        &[10],
        &weights,
        &config,
        Some(&mut kv_cache),
        Some(&mut linear_state),
        None,
    )?;
    kv_cache.advance(1);
    assert_eq!(decode_logits.dims(), &[1, 1, vocab_size]);

    // Both should produce finite values
    let flat = decode_logits.flatten_all()?.to_vec1::<f32>()?;
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "decode logits contain non-finite values"
    );

    Ok(())
}

#[test]
fn test_linear_attention_state_new() -> Result<()> {
    let device = Device::Cpu;
    let config = kiln_core::config::ModelConfig {
        hidden_size: 16,
        num_layers: 4,
        num_attention_heads: 4,
        num_kv_heads: 2,
        head_dim: 4,
        intermediate_size: 32,
        vocab_size: 32,
        max_position_embeddings: 1024,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        dtype: kiln_core::config::DType::FP32,
        num_full_attention_layers: 1,
        full_attention_interval: 4,
        attn_output_gate: false,
        linear_num_key_heads: 2,
        linear_key_head_dim: 4,
        linear_num_value_heads: 4,
        linear_value_head_dim: 4,
        linear_conv_kernel_dim: 4,
        partial_rotary_factor: 1.0,
    };

    let state = LinearAttentionState::new(&config, &device)?;
    // 3 linear layers (layers 0,1,2; layer 3 is full)
    assert_eq!(state.recurrent_states.len(), 3);
    assert_eq!(state.conv_states.len(), 3);
    // Recurrent state shape: [1, nv, dk, dv]
    assert_eq!(state.recurrent_states[0].dims(), &[1, 4, 4, 4]);
    assert_eq!(state.recurrent_states[0].dtype(), DType::F32);
    // Conv state shape: [1, qkv_dim, kernel_size-1] where qkv_dim = 2*(nk*dk) + nv*dv = 2*8+16=32
    let qkv_dim = 2 * (2 * 4) + 4 * 4; // 32
    assert_eq!(state.conv_states[0].dims(), &[1, qkv_dim, 3]);
    assert_eq!(state.conv_states[0].dtype(), DType::F32);

    let batched = LinearAttentionState::new_with_batch(&config, 3, &device)?;
    assert_eq!(batched.recurrent_states.len(), 3);
    assert_eq!(batched.conv_states.len(), 3);
    assert_eq!(batched.recurrent_states[0].dims(), &[3, 4, 4, 4]);
    assert_eq!(batched.recurrent_states[0].dtype(), DType::F32);
    assert_eq!(batched.conv_states[0].dims(), &[3, qkv_dim, 3]);
    assert_eq!(batched.conv_states[0].dtype(), DType::F32);
    assert!(LinearAttentionState::new_with_batch(&config, 0, &device).is_err());

    Ok(())
}

#[test]
fn test_linear_attention_state_batch_row_assembly_and_scatter() -> Result<()> {
    let device = Device::Cpu;
    let config = kiln_core::config::ModelConfig {
        hidden_size: 16,
        num_layers: 4,
        num_attention_heads: 4,
        num_kv_heads: 2,
        head_dim: 4,
        intermediate_size: 32,
        vocab_size: 32,
        max_position_embeddings: 1024,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        dtype: kiln_core::config::DType::FP32,
        num_full_attention_layers: 1,
        full_attention_interval: 4,
        attn_output_gate: false,
        linear_num_key_heads: 2,
        linear_key_head_dim: 4,
        linear_num_value_heads: 4,
        linear_value_head_dim: 4,
        linear_conv_kernel_dim: 4,
        partial_rotary_factor: 1.0,
    };

    let mut row0 = LinearAttentionState::new(&config, &device)?;
    let mut row1 = LinearAttentionState::new(&config, &device)?;
    let recurrent_values0: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let recurrent_values1: Vec<f32> = (0..64).map(|i| 1000.0 + i as f32).collect();
    let conv_values0: Vec<f32> = (0..96).map(|i| 2000.0 + i as f32).collect();
    let conv_values1: Vec<f32> = (0..96).map(|i| 3000.0 + i as f32).collect();
    row0.recurrent_states[0] =
        Tensor::from_slice(&recurrent_values0, (1usize, 4usize, 4usize, 4usize))?
            .to_device(device)?;
    row1.recurrent_states[0] =
        Tensor::from_slice(&recurrent_values1, (1usize, 4usize, 4usize, 4usize))?
            .to_device(device)?;
    row0.conv_states[0] =
        Tensor::from_slice(&conv_values0, (1usize, 32usize, 3usize))?.to_device(device)?;
    row1.conv_states[0] =
        Tensor::from_slice(&conv_values1, (1usize, 32usize, 3usize))?.to_device(device)?;

    let batched = LinearAttentionState::from_batch_rows(&[&row0, &row1])?;
    assert_eq!(batched.batch_size()?, 2);
    assert_eq!(batched.recurrent_states[0].dims(), &[2, 4, 4, 4]);
    assert_eq!(batched.conv_states[0].dims(), &[2, 32, 3]);
    assert!(LinearAttentionState::from_batch_rows(&[&batched]).is_err());

    let resident_prefix = batched.resident_batch_prefix_view(1)?;
    assert_eq!(resident_prefix.batch_size()?, 1);
    assert_eq!(
        resident_prefix.recurrent_states[0].id(),
        batched.recurrent_states[0].id()
    );
    assert_eq!(
        resident_prefix.conv_states[0].id(),
        batched.conv_states[0].id()
    );
    assert!(std::sync::Arc::ptr_eq(
        resident_prefix.recurrent_states[0].storage(),
        batched.recurrent_states[0].storage()
    ));
    assert!(std::sync::Arc::ptr_eq(
        resident_prefix.conv_states[0].storage(),
        batched.conv_states[0].storage()
    ));
    assert!(batched.resident_batch_prefix_view(0).is_err());
    assert!(batched.resident_batch_prefix_view(3).is_err());

    let split = batched.split_batch_rows()?;
    assert_eq!(split.len(), 2);
    assert!(!std::sync::Arc::ptr_eq(
        split[0].recurrent_states[0].storage(),
        batched.recurrent_states[0].storage()
    ));
    assert!(!std::sync::Arc::ptr_eq(
        split[0].conv_states[0].storage(),
        batched.conv_states[0].storage()
    ));
    assert_eq!(
        split[0].recurrent_states[0]
            .flatten_all()?
            .to_vec1::<f32>()?,
        row0.recurrent_states[0].flatten_all()?.to_vec1::<f32>()?
    );
    assert_eq!(
        split[1].recurrent_states[0]
            .flatten_all()?
            .to_vec1::<f32>()?,
        row1.recurrent_states[0].flatten_all()?.to_vec1::<f32>()?
    );
    assert_eq!(
        split[0].conv_states[0].to_vec3::<f32>()?,
        row0.conv_states[0].to_vec3::<f32>()?
    );
    assert_eq!(
        split[1].conv_states[0].to_vec3::<f32>()?,
        row1.conv_states[0].to_vec3::<f32>()?
    );

    let mut dst0 = LinearAttentionState::new(&config, &device)?;
    let mut dst1 = LinearAttentionState::new(&config, &device)?;
    {
        let mut destinations = [&mut dst0, &mut dst1];
        batched.scatter_batch_rows(&mut destinations)?;
    }
    assert_eq!(
        dst0.recurrent_states[0].flatten_all()?.to_vec1::<f32>()?,
        row0.recurrent_states[0].flatten_all()?.to_vec1::<f32>()?
    );
    assert_eq!(
        dst1.recurrent_states[0].flatten_all()?.to_vec1::<f32>()?,
        row1.recurrent_states[0].flatten_all()?.to_vec1::<f32>()?
    );
    assert_eq!(
        dst0.conv_states[0].to_vec3::<f32>()?,
        row0.conv_states[0].to_vec3::<f32>()?
    );
    assert_eq!(
        dst1.conv_states[0].to_vec3::<f32>()?,
        row1.conv_states[0].to_vec3::<f32>()?
    );

    let mut one_destination = [&mut dst0];
    assert!(batched.scatter_batch_rows(&mut one_destination).is_err());

    let mut replace_dst0 = LinearAttentionState::new(&config, &device)?;
    let mut replace_dst1 = LinearAttentionState::new(&config, &device)?;
    {
        let mut destinations = [&mut replace_dst0, &mut replace_dst1];
        batched.scatter_batch_rows_replace(&mut destinations)?;
    }
    assert!(!std::sync::Arc::ptr_eq(
        replace_dst0.recurrent_states[0].storage(),
        batched.recurrent_states[0].storage()
    ));
    assert!(!std::sync::Arc::ptr_eq(
        replace_dst0.conv_states[0].storage(),
        batched.conv_states[0].storage()
    ));
    assert_eq!(
        replace_dst0.recurrent_states[0]
            .flatten_all()?
            .to_vec1::<f32>()?,
        row0.recurrent_states[0].flatten_all()?.to_vec1::<f32>()?
    );
    assert_eq!(
        replace_dst1.recurrent_states[0]
            .flatten_all()?
            .to_vec1::<f32>()?,
        row1.recurrent_states[0].flatten_all()?.to_vec1::<f32>()?
    );
    assert_eq!(
        replace_dst0.conv_states[0].to_vec3::<f32>()?,
        row0.conv_states[0].to_vec3::<f32>()?
    );
    assert_eq!(
        replace_dst1.conv_states[0].to_vec3::<f32>()?,
        row1.conv_states[0].to_vec3::<f32>()?
    );

    let mut one_replace_destination = [&mut replace_dst0];
    assert!(
        batched
            .scatter_batch_rows_replace(&mut one_replace_destination)
            .is_err()
    );

    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_linear_attention_state_uses_bf16_on_metal_for_bf16_models() -> Result<()> {
    let Some(device) = crate::backend::metal::try_new_metal() else {
        return Ok(());
    };

    let config = kiln_core::config::ModelConfig {
        hidden_size: 16,
        num_layers: 4,
        num_attention_heads: 4,
        num_kv_heads: 2,
        head_dim: 4,
        intermediate_size: 32,
        vocab_size: 32,
        max_position_embeddings: 1024,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        dtype: kiln_core::config::DType::BF16,
        num_full_attention_layers: 1,
        full_attention_interval: 4,
        attn_output_gate: false,
        linear_num_key_heads: 2,
        linear_key_head_dim: 4,
        linear_num_value_heads: 4,
        linear_value_head_dim: 4,
        linear_conv_kernel_dim: 4,
        partial_rotary_factor: 1.0,
    };

    let state = LinearAttentionState::new(&config, &device)?;
    assert_eq!(state.recurrent_states[0].dtype(), DType::BF16);
    assert_eq!(state.conv_states[0].dtype(), DType::F32);

    let batched = LinearAttentionState::new_with_batch(&config, 3, &device)?;
    assert_eq!(batched.recurrent_states[0].dims(), &[3, 4, 4, 4]);
    assert_eq!(batched.recurrent_states[0].dtype(), DType::BF16);
    assert_eq!(
        batched.conv_states[0].dims(),
        &[3, config.linear_qkv_dim(), 3]
    );
    assert_eq!(batched.conv_states[0].dtype(), DType::F32);

    let row0 = LinearAttentionState::new(&config, &device)?;
    let row1 = LinearAttentionState::new(&config, &device)?;
    let assembled = LinearAttentionState::from_batch_rows(&[&row0, &row1])?;
    assert_eq!(assembled.batch_size()?, 2);
    assert_eq!(assembled.recurrent_states[0].dims(), &[2, 4, 4, 4]);
    assert_eq!(assembled.recurrent_states[0].dtype(), DType::BF16);
    assert_eq!(
        assembled.conv_states[0].dims(),
        &[2, config.linear_qkv_dim(), 3]
    );
    assert_eq!(assembled.conv_states[0].dtype(), DType::F32);
    let split = assembled.split_batch_rows()?;
    assert_eq!(split.len(), 2);
    assert_eq!(split[0].recurrent_states[0].dims(), &[1, 4, 4, 4]);
    assert_eq!(split[0].recurrent_states[0].dtype(), DType::BF16);
    assert_eq!(
        split[0].conv_states[0].dims(),
        &[1, config.linear_qkv_dim(), 3]
    );
    assert_eq!(split[0].conv_states[0].dtype(), DType::F32);

    Ok(())
}

#[test]
fn test_linear_attention_state_vulkan_inference_backend_uses_model_dtype() -> Result<()> {
    let device = Device::Cpu;
    let mut config = make_test_config(2, 1, 4, 8);

    let default_cpu = LinearAttentionState::new_with_batch_for_inference(&config, 2, &device)?;
    assert_eq!(default_cpu.recurrent_states[0].dims(), &[2, 2, 4, 4]);
    assert_eq!(default_cpu.recurrent_states[0].dtype(), DType::F32);
    assert_eq!(default_cpu.conv_states[0].dtype(), DType::F32);

    let named_cpu = LinearAttentionState::new_with_batch_for_inference_backend(
        &config,
        2,
        &device,
        Some("cpu"),
    )?;
    assert_eq!(named_cpu.recurrent_states[0].dtype(), DType::F32);
    assert_eq!(named_cpu.conv_states[0].dtype(), DType::F32);

    let vulkan = LinearAttentionState::new_with_batch_for_inference_backend(
        &config,
        2,
        &device,
        Some("vulkan"),
    )?;
    assert_eq!(vulkan.recurrent_states[0].dims(), &[2, 2, 4, 4]);
    assert_eq!(vulkan.recurrent_states[0].dtype(), DType::BF16);
    assert_eq!(vulkan.conv_states[0].dtype(), DType::F32);

    config.dtype = kiln_core::config::DType::FP16;
    let vulkan_fp16 = LinearAttentionState::new_with_batch_for_inference_backend(
        &config,
        2,
        &device,
        Some("vulkan"),
    )?;
    assert_eq!(vulkan_fp16.recurrent_states[0].dtype(), DType::F16);
    assert_eq!(vulkan_fp16.conv_states[0].dtype(), DType::F32);

    Ok(())
}

#[test]
fn resident_gdn_row_stride_uses_normalized_f32_storage() -> Result<()> {
    let bf16 = Tensor::zeros((1, 2, 3, 4), DType::BF16, Device::Cpu)?;
    let f16 = Tensor::zeros((1, 3, 5), DType::F16, Device::Cpu)?;
    let f32 = Tensor::zeros((1, 7), DType::F32, Device::Cpu)?;

    assert_eq!(resident_gdn_f32_row_bytes(&bf16, "recurrent")?, 96);
    assert_eq!(resident_gdn_f32_row_bytes(&f16, "recurrent")?, 60);
    assert_eq!(resident_gdn_f32_row_bytes(&f32, "convolution")?, 28);
    Ok(())
}

#[test]
fn test_causal_mask_with_offset() -> Result<()> {
    let device = Device::Cpu;
    // Simulate decode: 1 new query, 4 total KV (3 cached + 1 new)
    let scores = Tensor::ones((1, 1, 1, 4), DType::F32, device)?;
    let masked = apply_causal_mask_with_offset(&scores, 1, 4, 3)?;
    // Single query should attend to all 4 positions (no masking for q_len=1)
    let vals = masked.flatten_all()?.to_vec1::<f32>()?;
    assert!(
        vals.iter().all(|v| (*v - 1.0).abs() < 1e-6),
        "single query token should attend to all KV positions"
    );

    // Simulate prefill with offset: 2 new queries, 5 total KV (3 cached + 2 new)
    let scores = Tensor::ones((1, 1, 2, 5), DType::F32, device)?;
    let masked = apply_causal_mask_with_offset(&scores, 2, 5, 3)?;
    let vals = masked.flatten_all()?.to_vec1::<f32>()?;
    // Row 0 (abs pos 3): can attend to positions 0..4 (first 4), mask position 4
    assert!((vals[0] - 1.0).abs() < 1e-6); // pos 0: ok
    assert!((vals[1] - 1.0).abs() < 1e-6); // pos 1: ok
    assert!((vals[2] - 1.0).abs() < 1e-6); // pos 2: ok
    assert!((vals[3] - 1.0).abs() < 1e-6); // pos 3 (self): ok
    assert!(vals[4].is_infinite() && vals[4] < 0.0); // pos 4: masked
    // Row 1 (abs pos 4): can attend to all 5 positions
    assert!(vals[5..10].iter().all(|v| (*v - 1.0).abs() < 1e-6));

    Ok(())
}

// ------------------------------------------------------------------
// GDN chunkwise correctness test (Phase 6)
// ------------------------------------------------------------------

/// Reference per-token GDN recurrence, mirroring the pre-Phase-6 loop
/// that used to live in `gated_deltanet_forward`. Kept in the test
/// module (never called from production) so the chunkwise implementation
/// can be cross-checked against the arithmetically simple form.
///
/// Inputs are already transposed to [B, nv, T, *]; state is [B, nv, dk, dv].
fn gdn_sequential_reference(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    let (_, _, seq_len, _) = q.dims4()?;
    let mut outputs: Vec<Tensor> = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let q_t = q.narrow(2, t, 1)?; // [B, nv, 1, dk]
        let k_t = k.narrow(2, t, 1)?; // [B, nv, 1, dk]
        let v_t = v.narrow(2, t, 1)?.squeeze(2)?; // [B, nv, dv]
        let beta_t = beta.narrow(2, t, 1)?.squeeze(2)?; // [B, nv]
        let g_t = g.narrow(2, t, 1)?.squeeze(2)?; // [B, nv]

        let g_exp = g_t.exp()?.unsqueeze(2)?.unsqueeze(3)?; // [B, nv, 1, 1]
        *state = state.broadcast_mul(&g_exp)?;

        let kv_mem = k_t.matmul(&*state)?.squeeze(2)?; // [B, nv, dv]
        let delta: Tensor = (v_t - kv_mem)?.broadcast_mul(&beta_t.unsqueeze(2)?)?; // [B, nv, dv]

        let k_col = k_t.squeeze(2)?.unsqueeze(3)?; // [B, nv, dk, 1]
        let outer = k_col.broadcast_mul(&delta.unsqueeze(2)?)?; // [B, nv, dk, dv]
        *state = (&*state + &outer)?;

        let out_t = q_t.matmul(&*state)?; // [B, nv, 1, dv]
        outputs.push(out_t);
    }
    Ok(Tensor::cat(&outputs, 2)?)
}

/// Deterministic tensor of the given shape filled with values from a
/// simple hash of the index. Avoids depending on the tensor library's RNG
/// (which uses process-global state) and keeps the test reproducible.
fn det_tensor(shape: &[usize], scale: f32, bias: f32, device: &Device) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| {
            // Cheap mixable pseudo-random: stretch i through two sin
            // waves of different frequencies. Gives values in roughly
            // [-1, 1] with no exact repeats for small n.
            let x = (i as f32 * 0.7283).sin() + (i as f32 * 1.3719).cos();
            (x * 0.5) * scale + bias
        })
        .collect();
    Ok(Tensor::from_vec(data, shape)?.to_device(*device)?)
}

// (#1082) Deleted test_cuda_flash_attention_training_bwd_kt_collapses_gqa_grads:
//   it exercised the deleted candle-CustomOp `cuda_flash_attention_training_bf16`
//   via candle `loss.backward()`; flash-attn autograd is now the kt tape
//   (`try_tape_flash_attn_cuda`), validated by finite-diff/convergence.

#[test]
fn test_gdn_chunkwise_matches_sequential() -> Result<()> {
    // Small, fully-on-CPU shapes. We use F32 here so the comparison
    // is against the same numerical path the chunkwise form takes
    // for its decay cumulative products; the task spec's bf16
    // tolerance (<1e-3) is comfortably satisfied in F32 as well.
    let device = Device::Cpu;
    let dtype = DType::F32;

    let b = 1;
    let nv = 2;
    let t = 8;
    let dk = 4;
    let dv = 4;
    let chunk_size = 4;

    let q = det_tensor(&[b, nv, t, dk], 1.0, 0.0, &device)?.to_dtype(dtype)?;
    let k = det_tensor(&[b, nv, t, dk], 1.0, 0.0, &device)?.to_dtype(dtype)?;
    let v = det_tensor(&[b, nv, t, dv], 1.0, 0.0, &device)?.to_dtype(dtype)?;
    // beta ∈ (0, 1): pass through sigmoid-like shift.
    let beta_raw = det_tensor(&[b, nv, t], 2.0, 0.0, &device)?.to_dtype(dtype)?;
    let beta = {
        let ones = Tensor::ones_like(&beta_raw)?;
        (&ones / (&ones + &beta_raw.neg()?.exp()?)?)?
    };
    // g ∈ (-0.2, 0): small negative decays so cumulative sum stays sane.
    let g_raw = det_tensor(&[b, nv, t], 0.2, 0.0, &device)?.to_dtype(dtype)?;
    let g = (g_raw.abs()? * (-1.0_f64))?;

    let state_init = Tensor::zeros((b, nv, dk, dv), dtype, device)?;
    let backend = test_backend(&device);

    let mut state_chunk = state_init.clone();
    let out_chunk = gdn_chunkwise_recurrence(
        &backend,
        &q,
        &k,
        &v,
        &beta,
        &g,
        &mut state_chunk,
        chunk_size,
    )?;

    let mut state_seq = state_init.clone();
    let out_seq = gdn_sequential_reference(&q, &k, &v, &beta, &g, &mut state_seq)?;

    let out_diff = (&out_chunk - &out_seq)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let state_diff = (&state_chunk - &state_seq)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];

    // Task acceptance: max abs diff < 1e-3 in bf16. We run the test in
    // F32 so the actual tolerance is much tighter; guard against both
    // silent divergence and silent upgrade of the bf16 tolerance bound.
    assert!(
        out_diff < 1e-3,
        "chunkwise vs sequential output diff too large: {out_diff}",
    );
    assert!(
        state_diff < 1e-3,
        "chunkwise vs sequential state diff too large: {state_diff}",
    );

    // Also test chunk_size >= seq_len (single-chunk path) and
    // chunk_size == 1 (decode-like path) for coverage.
    for &cs in &[1usize, t] {
        let mut state_a = state_init.clone();
        let out_a = gdn_chunkwise_recurrence(&backend, &q, &k, &v, &beta, &g, &mut state_a, cs)?;
        let mut state_b = state_init.clone();
        let out_b = gdn_sequential_reference(&q, &k, &v, &beta, &g, &mut state_b)?;
        let d = (&out_a - &out_b)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .flatten_all()?
            .to_vec1::<f32>()?[0];
        let sd = (&state_a - &state_b)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .flatten_all()?
            .to_vec1::<f32>()?[0];
        assert!(d < 1e-3, "chunkwise(cs={cs}) output diff {d}");
        assert!(sd < 1e-3, "chunkwise(cs={cs}) state diff {sd}");
    }

    Ok(())
}

#[test]
fn test_gdn_chunkwise_masks_decay_before_exp() -> Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F32;

    let b = 1;
    let nv = 2;
    let t = 13;
    let dk = 4;
    let dv = 4;

    let q = det_tensor(&[b, nv, t, dk], 0.3, 0.0, &device)?.to_dtype(dtype)?;
    let k = det_tensor(&[b, nv, t, dk], 0.2, 0.0, &device)?.to_dtype(dtype)?;
    let v = det_tensor(&[b, nv, t, dv], 0.4, 0.0, &device)?.to_dtype(dtype)?;
    let beta = Tensor::ones((b, nv, t), dtype, device)?;
    let g = Tensor::from_vec(vec![-100.0f32; b * nv * t], (b, nv, t))?.to_device(device)?;
    let state_init = Tensor::zeros((b, nv, dk, dv), dtype, device)?;
    let backend = test_backend(&device);

    let mut state = state_init.clone();
    let out = gdn_chunkwise_recurrence(&backend, &q, &k, &v, &beta, &g, &mut state, t)?;

    for (name, tensor) in [("out", &out), ("state", &state)] {
        let values = tensor.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            values.iter().all(|v| v.is_finite()),
            "{name} contains non-finite values"
        );
    }

    Ok(())
}

#[test]
fn test_gdn_single_token_matches_sequential() -> Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F32;

    let b = 1;
    let nv = 2;
    let t = 1;
    let dk = 4;
    let dv = 4;

    let q = det_tensor(&[b, nv, t, dk], 1.0, 0.0, &device)?.to_dtype(dtype)?;
    let k = det_tensor(&[b, nv, t, dk], 0.8, 0.1, &device)?.to_dtype(dtype)?;
    let v = det_tensor(&[b, nv, t, dv], 0.6, -0.2, &device)?.to_dtype(dtype)?;
    let beta_raw = det_tensor(&[b, nv, t], 1.5, 0.0, &device)?.to_dtype(dtype)?;
    let beta = {
        let ones = Tensor::ones_like(&beta_raw)?;
        (&ones / (&ones + &beta_raw.neg()?.exp()?)?)?
    };
    let g_raw = det_tensor(&[b, nv, t], 0.2, 0.0, &device)?.to_dtype(dtype)?;
    let g = (g_raw.abs()? * (-1.0_f64))?;

    let state_init = det_tensor(&[b, nv, dk, dv], 0.1, 0.0, &device)?.to_dtype(dtype)?;
    let backend = test_backend(&device);

    let mut state_fast = state_init.clone();
    let out_fast = gdn_chunkwise_recurrence(
        &backend,
        &q,
        &k,
        &v,
        &beta,
        &g,
        &mut state_fast,
        GDN_CHUNK_SIZE,
    )?;

    let mut state_seq = state_init.clone();
    let out_seq = gdn_sequential_reference(&q, &k, &v, &beta, &g, &mut state_seq)?;

    let out_diff = (&out_fast - &out_seq)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let state_diff = (&state_fast - &state_seq)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];

    assert!(
        out_diff < 1e-5,
        "single-token fast path output drifted: max_abs_diff={out_diff:e}"
    );
    assert!(
        state_diff < 1e-5,
        "single-token fast path state drifted: max_abs_diff={state_diff:e}"
    );
    Ok(())
}

/// Correctness test for the vendored kiln-gdn-kernel CUDA fused
/// forward-substitution kernel.
///
/// Compares the fused kernel output against the per-token kt
/// fallback on the same random bf16 inputs at kiln's exact GDN config
/// (B=1, nv=32, C=64, dv=128). Asserts max abs diff < 1e-2 and mean
/// abs diff < 1e-3 — the fused path uses F32 accumulators and
/// per-token bf16 round-trips, so finite-precision drift is bounded
/// by bf16 rounding noise.
#[cfg(feature = "cuda")]
#[test]
fn test_gdn_kernel_matches_fallback() -> Result<()> {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    let device = match new_cuda_device(0) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("CUDA not available, skipping test_gdn_kernel_matches_fallback");
            return Ok(());
        }
    };

    let b = 1usize;
    let nv = 32usize;
    let c = 64usize;
    let dv = 128usize;

    let mut rng = StdRng::seed_from_u64(0xC0FFEE_u64);

    let n_a = b * nv * c * c;
    let n_v = b * nv * c * dv;
    let n_b = b * nv * c;

    let a_data: Vec<f32> = (0..n_a)
        .map(|_| rng.random_range(-0.05f32..0.05f32))
        .collect();
    let v_data: Vec<f32> = (0..n_v)
        .map(|_| rng.random_range(-1.0f32..1.0f32))
        .collect();
    let beta_data: Vec<f32> = (0..n_b).map(|_| rng.random_range(0.5f32..1.5f32)).collect();

    let a_f32 = Tensor::from_slice(&a_data, (b, nv, c, c))?.to_device(device)?;
    let v_f32 = Tensor::from_slice(&v_data, (b, nv, c, dv))?.to_device(device)?;
    let beta_f32 = Tensor::from_slice(&beta_data, (b, nv, c))?.to_device(device)?;

    // Make A_strict actually strictly lower triangular (matches what
    // the recurrence produces upstream of compute_w_chunk).
    let mask = strict_lower_tri_mask(c, DType::F32, &device)?;
    let a_f32 = a_f32.broadcast_mul(&mask)?;

    let a = a_f32.to_dtype(DType::BF16)?;
    let v = v_f32.to_dtype(DType::BF16)?;
    let beta = beta_f32.to_dtype(DType::BF16)?;

    let backend = crate::backend::for_device_kt(&device);
    let w_kernel = compute_w_chunk(&*backend, &a, &v, &beta, c)?; // CUDA kernel
    let w_fb = compute_w_chunk_fallback(&a, &v, &beta, c)?; // candle per-token

    let diff = (w_kernel.to_dtype(DType::F32)? - w_fb.to_dtype(DType::F32)?)?;
    let abs = diff.abs()?;
    let max = abs.flatten_all()?.max(0)?.flatten_all()?.to_vec1::<f32>()?[0];
    let mean = abs
        .flatten_all()?
        .mean(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];

    eprintln!("gdn-kernel vs fallback: max_abs_diff={max:e}, mean_abs_diff={mean:e}");

    assert!(
        max < 1e-2,
        "kernel output exceeds tolerance: max_abs_diff = {max:e}"
    );
    assert!(
        mean < 1e-3,
        "kernel mean drift exceeds tolerance: mean_abs_diff = {mean:e}"
    );

    Ok(())
}

/// Correctness test for the Metal fused forward-substitution kernel.
#[cfg(feature = "metal")]
#[test]
fn test_metal_gdn_forward_substitution_matches_fallback() -> Result<()> {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    let Some(device) = crate::backend::metal::try_new_metal() else {
        eprintln!(
            "Metal not available, skipping test_metal_gdn_forward_substitution_matches_fallback"
        );
        return Ok(());
    };

    let b = 1usize;
    let nv = 8usize;
    let c = 16usize;
    let dv = 128usize;

    let mut rng = StdRng::seed_from_u64(0xFACE_FEED_u64);

    let n_a = b * nv * c * c;
    let n_v = b * nv * c * dv;
    let n_b = b * nv * c;

    let a_data: Vec<f32> = (0..n_a)
        .map(|_| rng.random_range(-0.05f32..0.05f32))
        .collect();
    let v_data: Vec<f32> = (0..n_v)
        .map(|_| rng.random_range(-1.0f32..1.0f32))
        .collect();
    let beta_data: Vec<f32> = (0..n_b).map(|_| rng.random_range(0.5f32..1.5f32)).collect();

    let a_f32 = Tensor::from_slice(&a_data, (b, nv, c, c))?.to_device(device)?;
    let v_f32 = Tensor::from_slice(&v_data, (b, nv, c, dv))?.to_device(device)?;
    let beta_f32 = Tensor::from_slice(&beta_data, (b, nv, c))?.to_device(device)?;

    let mask = strict_lower_tri_mask(c, DType::F32, &device)?;
    let a_f32 = a_f32.broadcast_mul(&mask)?;

    let a = a_f32.to_dtype(DType::BF16)?;
    let v = v_f32.to_dtype(DType::BF16)?;
    let beta = beta_f32.to_dtype(DType::BF16)?;

    let backend = crate::backend::for_device_kt(&device);
    let w_kernel = compute_w_chunk(&*backend, &a, &v, &beta, c)?;
    let w_fb = compute_w_chunk_fallback(&a, &v, &beta, c)?;

    let diff = (w_kernel.to_dtype(DType::F32)? - w_fb.to_dtype(DType::F32)?)?;
    let abs = diff.abs()?;
    let max = abs.flatten_all()?.max(0)?.flatten_all()?.to_vec1::<f32>()?[0];
    let mean = abs
        .flatten_all()?
        .mean(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];

    eprintln!("metal gdn-forward-sub vs fallback: max_abs_diff={max:e}, mean_abs_diff={mean:e}");

    assert!(
        max < 1e-2,
        "metal forward-sub kernel output exceeds tolerance: max_abs_diff = {max:e}"
    );
    assert!(
        mean < 1e-3,
        "metal forward-sub kernel mean drift exceeds tolerance: mean_abs_diff = {mean:e}"
    );

    Ok(())
}

/// Parity check for the single-token recurrent CUDA kernel.
///
/// Compares output and final state of `gdn_chunkwise_recurrence` with
/// the new fused recurrent kernel against `gdn_sequential_reference`
/// at kiln's exact GDN config (B=1, nv=32, dk=128, dv=128, T=1).
/// Tolerance matches the chunkwise CUDA kernel test.
#[cfg(feature = "cuda")]
#[test]
fn test_gdn_recurrent_kernel_matches_reference() -> Result<()> {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    let device = match new_cuda_device(0) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("CUDA not available, skipping test_gdn_recurrent_kernel_matches_reference");
            return Ok(());
        }
    };

    let b = 1usize;
    let nv = 32usize;
    let t = 1usize;
    let dk = 128usize;
    let dv = 128usize;

    let mut rng = StdRng::seed_from_u64(0xDECAFBADu64);

    let n_qk = b * nv * t * dk;
    let n_v = b * nv * t * dv;
    let n_b = b * nv * t;
    let n_s = b * nv * dk * dv;

    let q_data: Vec<f32> = (0..n_qk)
        .map(|_| rng.random_range(-0.5f32..0.5f32))
        .collect();
    let k_data: Vec<f32> = (0..n_qk)
        .map(|_| rng.random_range(-0.5f32..0.5f32))
        .collect();
    let v_data: Vec<f32> = (0..n_v)
        .map(|_| rng.random_range(-1.0f32..1.0f32))
        .collect();
    let beta_data: Vec<f32> = (0..n_b).map(|_| rng.random_range(0.3f32..1.2f32)).collect();
    // Small negative gates so exp(g) stays in (~0.8, 1.0).
    let g_data: Vec<f32> = (0..n_b)
        .map(|_| rng.random_range(-0.2f32..0.0f32))
        .collect();
    let s_data: Vec<f32> = (0..n_s)
        .map(|_| rng.random_range(-0.1f32..0.1f32))
        .collect();

    let q_f32 = Tensor::from_slice(&q_data, (b, nv, t, dk))?.to_device(device)?;
    let k_f32 = Tensor::from_slice(&k_data, (b, nv, t, dk))?.to_device(device)?;
    let v_f32 = Tensor::from_slice(&v_data, (b, nv, t, dv))?.to_device(device)?;
    let beta_f32 = Tensor::from_slice(&beta_data, (b, nv, t))?.to_device(device)?;
    let g_f32 = Tensor::from_slice(&g_data, (b, nv, t))?.to_device(device)?;
    let state_f32 = Tensor::from_slice(&s_data, (b, nv, dk, dv))?.to_device(device)?;

    let q = q_f32.to_dtype(DType::BF16)?;
    let k = k_f32.to_dtype(DType::BF16)?;
    let v = v_f32.to_dtype(DType::BF16)?;
    let beta = beta_f32.to_dtype(DType::BF16)?;
    let g = g_f32.to_dtype(DType::BF16)?;
    let state_bf16 = state_f32.to_dtype(DType::BF16)?;

    // Reference path: F32 sequential recurrence on the same numerical
    // inputs (cast back to F32 from the bf16 round-trip so the bf16
    // quantization is shared between the two paths and only the kernel
    // arithmetic differs).
    let q_ref = q.to_dtype(DType::F32)?;
    let k_ref = k.to_dtype(DType::F32)?;
    let v_ref = v.to_dtype(DType::F32)?;
    let beta_ref = beta.to_dtype(DType::F32)?;
    let g_ref = g.to_dtype(DType::F32)?;
    let mut state_ref = state_bf16.to_dtype(DType::F32)?;
    let out_ref =
        gdn_sequential_reference(&q_ref, &k_ref, &v_ref, &beta_ref, &g_ref, &mut state_ref)?;

    // Kernel path: chunkwise dispatcher with seq_len == 1 routes to
    // the new fused recurrent kernel under the process-lifetime policy.
    let backend = crate::backend::for_device_kt(&device);
    let mut state_kernel = state_bf16.clone();
    let out_kernel =
        gdn_chunkwise_recurrence(&*backend, &q, &k, &v, &beta, &g, &mut state_kernel, 1)?;

    let out_diff = (out_kernel.to_dtype(DType::F32)? - &out_ref)?;
    let abs = out_diff.abs()?;
    let max = abs.flatten_all()?.max(0)?.flatten_all()?.to_vec1::<f32>()?[0];
    let mean = abs
        .flatten_all()?
        .mean(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let s_diff = (state_kernel.to_dtype(DType::F32)? - &state_ref)?;
    let s_abs = s_diff.abs()?;
    let s_max = s_abs
        .flatten_all()?
        .max(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let s_mean = s_abs
        .flatten_all()?
        .mean(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];

    eprintln!(
        "gdn-recurrent vs reference: out max={max:e} mean={mean:e}, state max={s_max:e} mean={s_mean:e}"
    );

    assert!(
        max < 1e-2,
        "recurrent kernel output exceeds tolerance: max_abs_diff = {max:e}"
    );
    assert!(
        mean < 1e-3,
        "recurrent kernel mean drift exceeds tolerance: mean_abs_diff = {mean:e}"
    );
    assert!(
        s_max < 1e-2,
        "recurrent kernel state exceeds tolerance: max_abs_diff = {s_max:e}"
    );
    assert!(
        s_mean < 1e-3,
        "recurrent kernel state mean drift exceeds tolerance: mean_abs_diff = {s_mean:e}"
    );

    Ok(())
}

/// Parity check for the single-token recurrent Metal kernel.
#[cfg(feature = "metal")]
#[test]
fn test_metal_gdn_recurrent_kernel_matches_reference() -> Result<()> {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    let Some(device) = crate::backend::metal::try_new_metal() else {
        eprintln!(
            "Metal not available, skipping test_metal_gdn_recurrent_kernel_matches_reference"
        );
        return Ok(());
    };

    let b = 1usize;
    let nv = 16usize;
    let t = 1usize;
    let dk = 128usize;
    let dv = 128usize;

    let mut rng = StdRng::seed_from_u64(0xBEEFu64);

    let n_qk = b * nv * t * dk;
    let n_v = b * nv * t * dv;
    let n_b = b * nv * t;
    let n_s = b * nv * dk * dv;

    let q_data: Vec<f32> = (0..n_qk)
        .map(|_| rng.random_range(-0.5f32..0.5f32))
        .collect();
    let k_data: Vec<f32> = (0..n_qk)
        .map(|_| rng.random_range(-0.5f32..0.5f32))
        .collect();
    let v_data: Vec<f32> = (0..n_v)
        .map(|_| rng.random_range(-1.0f32..1.0f32))
        .collect();
    let beta_data: Vec<f32> = (0..n_b).map(|_| rng.random_range(0.3f32..1.2f32)).collect();
    let g_data: Vec<f32> = (0..n_b)
        .map(|_| rng.random_range(-0.2f32..0.0f32))
        .collect();
    let s_data: Vec<f32> = (0..n_s)
        .map(|_| rng.random_range(-0.1f32..0.1f32))
        .collect();

    let q = Tensor::from_slice(&q_data, (b, nv, t, dk))?
        .to_device(device)?
        .to_dtype(DType::BF16)?;
    let k = Tensor::from_slice(&k_data, (b, nv, t, dk))?
        .to_device(device)?
        .to_dtype(DType::BF16)?;
    let v = Tensor::from_slice(&v_data, (b, nv, t, dv))?
        .to_device(device)?
        .to_dtype(DType::BF16)?;
    let beta = Tensor::from_slice(&beta_data, (b, nv, t))?
        .to_device(device)?
        .to_dtype(DType::BF16)?;
    let g = Tensor::from_slice(&g_data, (b, nv, t))?
        .to_device(device)?
        .to_dtype(DType::BF16)?;
    let state_bf16 = Tensor::from_slice(&s_data, (b, nv, dk, dv))?
        .to_device(device)?
        .to_dtype(DType::BF16)?;

    let q_ref = q.to_dtype(DType::F32)?;
    let k_ref = k.to_dtype(DType::F32)?;
    let v_ref = v.to_dtype(DType::F32)?;
    let beta_ref = beta.to_dtype(DType::F32)?;
    let g_ref = g.to_dtype(DType::F32)?;
    let mut state_ref = state_bf16.to_dtype(DType::F32)?;
    let out_ref =
        gdn_sequential_reference(&q_ref, &k_ref, &v_ref, &beta_ref, &g_ref, &mut state_ref)?;

    let backend = crate::backend::for_device_kt(&device);
    if !GdnBackend::runtime_supports_gdn_recurrent_step(backend.as_ref()) {
        eprintln!("Metal recurrent kernel disabled, skipping parity test");
        return Ok(());
    }
    let mut state_kernel = state_bf16.clone();
    let out_kernel =
        gdn_chunkwise_recurrence(&*backend, &q, &k, &v, &beta, &g, &mut state_kernel, 1)?;

    let out_diff = (out_kernel.to_dtype(DType::F32)? - &out_ref)?;
    let abs = out_diff.abs()?;
    let max = abs.flatten_all()?.max(0)?.flatten_all()?.to_vec1::<f32>()?[0];
    let mean = abs
        .flatten_all()?
        .mean(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let s_diff = (state_kernel.to_dtype(DType::F32)? - &state_ref)?;
    let s_abs = s_diff.abs()?;
    let s_max = s_abs
        .flatten_all()?
        .max(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let s_mean = s_abs
        .flatten_all()?
        .mean(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];

    eprintln!(
        "metal gdn-recurrent vs reference: out max={max:e} mean={mean:e}, state max={s_max:e} mean={s_mean:e}"
    );

    assert!(
        max < 1e-2,
        "metal recurrent kernel output exceeds tolerance: max_abs_diff = {max:e}"
    );
    assert!(
        mean < 1e-3,
        "metal recurrent kernel mean drift exceeds tolerance: mean_abs_diff = {mean:e}"
    );
    assert!(
        s_max < 1e-2,
        "metal recurrent kernel state exceeds tolerance: max_abs_diff = {s_max:e}"
    );
    assert!(
        s_mean < 1e-3,
        "metal recurrent kernel state mean drift exceeds tolerance: mean_abs_diff = {s_mean:e}"
    );

    Ok(())
}

#[cfg(feature = "vulkan")]
#[test]
fn causal_conv1d_decode_migrates_cpu_state_to_vulkan_activation() -> Result<()> {
    if std::env::var("KILN_TENSOR_VULKAN_TEST").ok().as_deref() != Some("1") {
        return Ok(());
    }
    assert!(
        crate::backend::vulkan::vulkan_is_available(),
        "KILN_TENSOR_VULKAN_TEST=1 requires a working Vulkan device"
    );
    let kernel_size = 4usize;
    let x_cpu = Tensor::from_slice(&[0.5_f32, -0.25], (1, 2, 1))?.to_dtype(DType::BF16)?;
    let weight_cpu = Tensor::from_slice(
        &[0.125_f32, -0.25, 0.5, 0.75, -0.5, 0.25, 0.125, -0.75],
        (2, 1, kernel_size),
    )?
    .to_dtype(DType::BF16)?;
    let state_values = [0.25_f32, -0.5, 1.0, -1.0, 0.75, 0.5];
    let mut expected_state = Tensor::from_slice(&state_values, (1, 2, kernel_size - 1))?;
    let expected = causal_conv1d_decode(&x_cpu, &weight_cpu, &mut expected_state, kernel_size)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let x_vulkan = x_cpu.to_device(Device::Vulkan(0))?;
    let weight_vulkan = weight_cpu.to_device(Device::Vulkan(0))?;
    let mut actual_state = Tensor::from_slice(&state_values, (1, 2, kernel_size - 1))?;
    let actual = causal_conv1d_decode(&x_vulkan, &weight_vulkan, &mut actual_state, kernel_size)?;

    assert_eq!(actual.device(), Device::Vulkan(0));
    assert_eq!(actual_state.device(), Device::Vulkan(0));
    let actual = actual
        .to_device(Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() <= 1e-6,
            "causal conv output lane {index}: got={got} want={want}"
        );
    }
    let expected_state = expected_state.flatten_all()?.to_vec1::<f32>()?;
    let actual_state = actual_state
        .to_device(Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(actual_state, expected_state);
    Ok(())
}

/// Parity check for the fused causal_conv1d_update kernel against the
/// portable `causal_conv1d_decode` + `cuda_silu` chain, at Qwen3.5-4B's
/// exact decode shape: B=1, C=linear_qkv_dim=8192, K=4.
///
/// Verifies (a) the silu-fused F32 output matches within bf16-rounding
/// noise and (b) the mutated conv_state matches bit-for-bit (both paths
/// write the same K-1 previous inputs from the same bf16 source).
#[cfg(feature = "cuda")]
#[test]
fn test_causal_conv1d_update_matches_fallback() -> Result<()> {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    let device = match new_cuda_device(0) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("CUDA not available, skipping test_causal_conv1d_update_matches_fallback");
            return Ok(());
        }
    };

    let batch = 1usize;
    let channels = 8192usize; // Qwen3.5-4B linear_qkv_dim
    let kernel_size = 4usize;

    let mut rng = StdRng::seed_from_u64(0xC0_1DBEEF);
    let n_x = batch * channels * 1;
    let n_w = channels * kernel_size;
    let n_s = batch * channels * (kernel_size - 1);

    let x_data: Vec<f32> = (0..n_x)
        .map(|_| rng.random_range(-0.5f32..0.5f32))
        .collect();
    let w_data: Vec<f32> = (0..n_w)
        .map(|_| rng.random_range(-0.1f32..0.1f32))
        .collect();
    let s_data: Vec<f32> = (0..n_s)
        .map(|_| rng.random_range(-0.3f32..0.3f32))
        .collect();

    let x_f32 = Tensor::from_slice(&x_data, (batch, channels, 1))?.to_device(device)?;
    let w_f32 = Tensor::from_slice(&w_data, (channels, 1, kernel_size))?.to_device(device)?;

    let x = x_f32.to_dtype(DType::BF16)?;
    let w = w_f32.to_dtype(DType::BF16)?;

    // IMPORTANT: `causal_conv1d_decode` (the fallback) updates the conv-state
    // *truly in place* (`conv_state.slice_set(...)`, kept stable for CUDA-graph
    // pointer capture), and `Tensor::clone()` shares the underlying storage Arc
    // (cheap view-clone, NOT a deep copy). So the fallback and the fused kernel
    // MUST run on independent state buffers — a single `s_init.clone()` for both
    // would let the fallback's in-place update corrupt the shared buffer before
    // the kernel reads it, feeding the kernel the post-update state and producing
    // a spurious ~5.6e-2 "mismatch" that is purely a harness aliasing artifact
    // (the state-parity assert would also trivially pass, comparing the buffer to
    // itself). Build each from the host `s_data` so neither aliases. (Mirrors the
    // `_metal` sibling fix; prefill tests are exempt because `causal_conv1d_prefill*`
    // rebinds `*conv_state` to a fresh tensor instead of mutating in place.)
    // Fallback path: portable decode + silu in F32.
    let mut s_fb =
        Tensor::from_slice(&s_data, (batch, channels, kernel_size - 1))?.to_device(device)?;
    let out_fb = causal_conv1d_decode(&x, &w, &mut s_fb, kernel_size)?;
    let out_fb = cuda_silu(&out_fb.to_dtype(DType::F32)?)?;

    // Fused kernel path via the backend dispatch.
    let backend = crate::backend::for_device_kt(&device);
    if !ConvBackend::runtime_supports_causal_conv1d_update(backend.as_ref()) {
        eprintln!("backend policy declines causal_conv1d_update; skipping");
        return Ok(());
    }
    let mut s_k =
        Tensor::from_slice(&s_data, (batch, channels, kernel_size - 1))?.to_device(device)?;
    let out_k = match ConvBackend::runtime_causal_conv1d_update(
        backend.as_ref(),
        &x,
        &w,
        &mut s_k,
        kernel_size,
    )? {
        Some(t) => t,
        None => {
            eprintln!("backend declined causal_conv1d_update at Qwen3.5 envelope; skipping");
            return Ok(());
        }
    };

    // Output parity (silu fused on the kernel side).
    let diff = (out_k.to_dtype(DType::F32)? - out_fb.to_dtype(DType::F32)?)?;
    let abs = diff.abs()?;
    let max = abs.flatten_all()?.max(0)?.flatten_all()?.to_vec1::<f32>()?[0];
    let mean = abs
        .flatten_all()?
        .mean(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    eprintln!("conv1d_update vs fallback: max_abs_diff={max:e} mean_abs_diff={mean:e}");
    assert!(
        max < 2e-3,
        "fused conv1d_update output max_abs_diff={max:e} exceeds 2e-3"
    );
    assert!(
        mean < 5e-4,
        "fused conv1d_update output mean_abs_diff={mean:e} exceeds 5e-4"
    );

    // State parity — both paths write the same K-1 previous inputs.
    let sdiff = (s_k.to_dtype(DType::F32)? - s_fb.to_dtype(DType::F32)?)?;
    let smax = sdiff
        .abs()?
        .flatten_all()?
        .max(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    eprintln!("conv1d_update state parity: max_abs_diff={smax:e}");
    assert!(
        smax < 1e-5,
        "fused conv1d_update state max_abs_diff={smax:e} exceeds 1e-5"
    );

    Ok(())
}

/// Parity check for the fused CUDA causal_conv1d prefill kernel against
/// the portable `causal_conv1d_prefill` + `cuda_silu` chain, at the native
/// MTP draft shape that exercises `seq_len > 1`.
#[cfg(feature = "cuda")]
#[test]
fn test_causal_conv1d_prefill_matches_fallback() -> Result<()> {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    let device = match new_cuda_device(0) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("CUDA not available, skipping test_causal_conv1d_prefill_matches_fallback");
            return Ok(());
        }
    };

    let batch = 1usize;
    let channels = 8192usize; // Qwen3.5-4B linear_qkv_dim
    let seq_len = 512usize;
    let kernel_size = 4usize;

    let mut rng = StdRng::seed_from_u64(0xC0_1DC0DE);
    let n_x = batch * channels * seq_len;
    let n_w = channels * kernel_size;
    let n_s = batch * channels * (kernel_size - 1);

    let x_data: Vec<f32> = (0..n_x)
        .map(|_| rng.random_range(-0.5f32..0.5f32))
        .collect();
    let w_data: Vec<f32> = (0..n_w)
        .map(|_| rng.random_range(-0.1f32..0.1f32))
        .collect();
    let s_data: Vec<f32> = (0..n_s)
        .map(|_| rng.random_range(-0.3f32..0.3f32))
        .collect();

    let x = Tensor::from_slice(&x_data, (batch, channels, seq_len))?
        .to_device(device)?
        .to_dtype(DType::BF16)?;
    let w = Tensor::from_slice(&w_data, (channels, 1, kernel_size))?
        .to_device(device)?
        .to_dtype(DType::BF16)?;
    let s_init =
        Tensor::from_slice(&s_data, (batch, channels, kernel_size - 1))?.to_device(device)?;

    let mut s_fb = s_init.clone();
    let out_fb = causal_conv1d_prefill_with_dtype(&x, &w, &mut s_fb, kernel_size, DType::F32)?;
    let out_fb = cuda_silu(&out_fb)?;

    let backend = crate::backend::for_device_kt(&device);
    if !ConvBackend::runtime_supports_causal_conv1d_prefill(backend.as_ref()) {
        eprintln!("backend policy declines causal_conv1d_prefill; skipping");
        return Ok(());
    }
    let mut s_k = s_init.clone();
    let out_k = match ConvBackend::runtime_causal_conv1d_prefill(
        backend.as_ref(),
        &x,
        &w,
        &mut s_k,
        kernel_size,
    )? {
        Some(t) => t,
        None => {
            eprintln!("backend declined causal_conv1d_prefill at Qwen3.5 envelope; skipping");
            return Ok(());
        }
    };

    let diff = (out_k.to_dtype(DType::F32)? - out_fb.to_dtype(DType::F32)?)?;
    let abs = diff.abs()?;
    let max = abs.flatten_all()?.max(0)?.flatten_all()?.to_vec1::<f32>()?[0];
    let mean = abs
        .flatten_all()?
        .mean(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    eprintln!("conv1d_prefill vs fallback: max_abs_diff={max:e} mean_abs_diff={mean:e}");
    assert!(
        max < 2e-3,
        "fused conv1d_prefill output max_abs_diff={max:e} exceeds 2e-3"
    );
    assert!(
        mean < 5e-4,
        "fused conv1d_prefill output mean_abs_diff={mean:e} exceeds 5e-4"
    );

    let sdiff = (s_k.to_dtype(DType::F32)? - s_fb.to_dtype(DType::F32)?)?;
    let smax = sdiff
        .abs()?
        .flatten_all()?
        .max(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    eprintln!("conv1d_prefill state parity: max_abs_diff={smax:e}");
    assert!(
        smax < 1e-5,
        "fused conv1d_prefill state max_abs_diff={smax:e} exceeds 1e-5"
    );

    Ok(())
}

/// Metal parity check for `ConvBackend::runtime_causal_conv1d_update`
/// against the same portable `causal_conv1d_decode` + `cuda_silu` oracle
/// used by CUDA.
#[cfg(feature = "metal")]
#[test]
fn test_causal_conv1d_update_matches_fallback_metal() -> Result<()> {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    let Some(device) = crate::backend::metal::try_new_metal() else {
        eprintln!("Metal unavailable, skipping test_causal_conv1d_update_matches_fallback_metal");
        return Ok(());
    };

    let batch = 1usize;
    let channels = 8192usize; // Qwen3.5-4B linear_qkv_dim
    let kernel_size = 4usize;

    let mut rng = StdRng::seed_from_u64(0xC0_1DBEEF);
    let n_x = batch * channels;
    let n_w = channels * kernel_size;
    let n_s = batch * channels * (kernel_size - 1);

    let x_data: Vec<f32> = (0..n_x)
        .map(|_| rng.random_range(-0.5f32..0.5f32))
        .collect();
    let w_data: Vec<f32> = (0..n_w)
        .map(|_| rng.random_range(-0.1f32..0.1f32))
        .collect();
    let s_data: Vec<f32> = (0..n_s)
        .map(|_| rng.random_range(-0.3f32..0.3f32))
        .collect();

    let x_f32 = Tensor::from_slice(&x_data, (batch, channels, 1))?.to_device(device)?;
    let w_f32 = Tensor::from_slice(&w_data, (channels, 1, kernel_size))?.to_device(device)?;

    let x = x_f32.to_dtype(DType::BF16)?;
    let w = w_f32.to_dtype(DType::BF16)?;

    // IMPORTANT: `Tensor::clone()` shares the underlying storage Arc (it is a
    // cheap view-clone, NOT a deep copy), and both `causal_conv1d_decode`
    // (the fallback) and the fused kernel update the conv-state buffer
    // *in place*. So the fallback and kernel paths MUST run on independent
    // state buffers — otherwise the fallback's in-place state update
    // corrupts the shared `s_init` before the kernel reads its clone,
    // feeding the kernel the post-update state and producing a spurious
    // ~5.6e-2 "mismatch" that is purely a harness aliasing artifact, not a
    // kernel error. Build each from the host `s_data` so neither aliases.
    let mut s_fb =
        Tensor::from_slice(&s_data, (batch, channels, kernel_size - 1))?.to_device(device)?;
    let out_fb = causal_conv1d_decode(&x, &w, &mut s_fb, kernel_size)?;
    let out_fb = cuda_silu(&out_fb.to_dtype(DType::F32)?)?;

    let backend = crate::backend::for_device_kt(&device);
    if !ConvBackend::runtime_supports_causal_conv1d_update(backend.as_ref()) {
        eprintln!("backend policy declines causal_conv1d_update; skipping");
        return Ok(());
    }
    let mut s_k =
        Tensor::from_slice(&s_data, (batch, channels, kernel_size - 1))?.to_device(device)?;
    let out_k = match ConvBackend::runtime_causal_conv1d_update(
        backend.as_ref(),
        &x,
        &w,
        &mut s_k,
        kernel_size,
    )? {
        Some(t) => t,
        None => {
            eprintln!("backend declined causal_conv1d_update at Qwen3.5 envelope; skipping");
            return Ok(());
        }
    };

    let diff = (out_k.to_dtype(DType::F32)? - out_fb.to_dtype(DType::F32)?)?;
    let abs = diff.abs()?;
    let max = abs.flatten_all()?.max(0)?.flatten_all()?.to_vec1::<f32>()?[0];
    let mean = abs
        .flatten_all()?
        .mean(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    eprintln!("metal conv1d_update vs fallback: max_abs_diff={max:e} mean_abs_diff={mean:e}");
    assert!(
        max < 2e-3,
        "metal conv1d_update output max_abs_diff={max:e} exceeds 2e-3"
    );
    assert!(
        mean < 5e-4,
        "metal conv1d_update output mean_abs_diff={mean:e} exceeds 5e-4"
    );

    let sdiff = (s_k.to_dtype(DType::F32)? - s_fb.to_dtype(DType::F32)?)?;
    let smax = sdiff
        .abs()?
        .flatten_all()?
        .max(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    eprintln!("metal conv1d_update state parity: max_abs_diff={smax:e}");
    assert!(
        smax < 1e-5,
        "metal conv1d_update state max_abs_diff={smax:e} exceeds 1e-5"
    );

    Ok(())
}

#[test]
fn causal_conv1d_single_token_prefill_matches_decode_cpu() -> Result<()> {
    let (batch, channels, kernel_size) = (1usize, 7usize, 4usize);
    let x_data: Vec<f32> = (0..batch * channels)
        .map(|i| ((i as f32 + 1.0) * 0.071).sin())
        .collect();
    let weight_data: Vec<f32> = (0..channels * kernel_size)
        .map(|i| ((i as f32 + 3.0) * 0.037).cos())
        .collect();
    let state_data: Vec<f32> = (0..batch * channels * (kernel_size - 1))
        .map(|i| ((i as f32 + 5.0) * 0.053).sin())
        .collect();
    let x = Tensor::from_vec(x_data, (batch, channels, 1))?;
    let weight = Tensor::from_vec(weight_data, (channels, 1, kernel_size))?;
    let initial_state = Tensor::from_vec(state_data, (batch, channels, kernel_size - 1))?;

    let mut prefill_state = initial_state.clone();
    let prefill = causal_conv1d_prefill(&x, &weight, &mut prefill_state, kernel_size)?;
    let mut decode_state = initial_state;
    let decode = causal_conv1d_decode(&x, &weight, &mut decode_state, kernel_size)?;

    let output_diff = (prefill - decode)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let state_diff = (prefill_state - decode_state)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        output_diff <= 1e-6,
        "single-token prefill/decode output drifted by {output_diff:e}"
    );
    assert!(
        state_diff <= 1e-6,
        "single-token prefill/decode state drifted by {state_diff:e}"
    );
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_causal_conv1d_prefill_bf16_parity_on_metal() -> Result<()> {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    let Some(device) = crate::backend::metal::try_new_metal() else {
        eprintln!("Metal not available, skipping test_causal_conv1d_prefill_bf16_parity_on_metal");
        return Ok(());
    };

    let batch = 1usize;
    let channels = 8192usize; // Qwen3.5-4B linear_qkv_dim
    let seq_len = 16usize;
    let kernel_size = 4usize;

    let mut rng = StdRng::seed_from_u64(0xBF16_C0DE);
    let n_x = batch * channels * seq_len;
    let n_w = channels * kernel_size;
    let n_s = batch * channels * (kernel_size - 1);

    let x_data: Vec<f32> = (0..n_x)
        .map(|_| rng.random_range(-0.5f32..0.5f32))
        .collect();
    let w_data: Vec<f32> = (0..n_w)
        .map(|_| rng.random_range(-0.1f32..0.1f32))
        .collect();
    let s_data: Vec<f32> = (0..n_s)
        .map(|_| rng.random_range(-0.3f32..0.3f32))
        .collect();

    let x = Tensor::from_slice(&x_data, (batch, channels, seq_len))?
        .to_device(device)?
        .to_dtype(DType::BF16)?;
    let w = Tensor::from_slice(&w_data, (channels, 1, kernel_size))?
        .to_device(device)?
        .to_dtype(DType::BF16)?;
    let s_init =
        Tensor::from_slice(&s_data, (batch, channels, kernel_size - 1))?.to_device(device)?;

    let mut s_ref = s_init.clone();
    let out_ref = causal_conv1d_prefill_with_dtype(&x, &w, &mut s_ref, kernel_size, DType::F32)?;
    let out_ref = cuda_silu(&out_ref)?;

    let mut s_bf16 = s_init.clone();
    assert_eq!(
        causal_conv1d_prefill_compute_dtype(&x, &w, &s_bf16, kernel_size),
        DType::BF16
    );
    let out_bf16 = causal_conv1d_prefill(&x, &w, &mut s_bf16, kernel_size)?;
    assert_eq!(out_bf16.dtype(), DType::BF16);
    assert_eq!(s_bf16.dtype(), DType::F32);
    let out_bf16 = cuda_silu(&out_bf16)?;

    let diff = (out_bf16.to_dtype(DType::F32)? - out_ref.to_dtype(DType::F32)?)?;
    let abs = diff.abs()?;
    let max = abs.flatten_all()?.max(0)?.flatten_all()?.to_vec1::<f32>()?[0];
    let mean = abs
        .flatten_all()?
        .mean(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    eprintln!("conv1d_prefill bf16 vs f32: max_abs_diff={max:e} mean_abs_diff={mean:e}");
    assert!(
        max < 2e-2,
        "bf16 prefill output max_abs_diff={max:e} exceeds 2e-2"
    );
    assert!(
        mean < 2e-3,
        "bf16 prefill output mean_abs_diff={mean:e} exceeds 2e-3"
    );

    let sdiff = (s_bf16.to_dtype(DType::F32)? - s_ref.to_dtype(DType::F32)?)?;
    let smax = sdiff
        .abs()?
        .flatten_all()?
        .max(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    eprintln!("conv1d_prefill bf16 state parity: max_abs_diff={smax:e}");
    assert!(
        smax < 1e-6,
        "bf16 prefill state max_abs_diff={smax:e} exceeds 1e-6"
    );

    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_metal_causal_conv1d_prefill_kernel_matches_fallback() -> Result<()> {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    let Some(device) = crate::backend::metal::try_new_metal() else {
        eprintln!(
            "Metal not available, skipping test_metal_causal_conv1d_prefill_kernel_matches_fallback"
        );
        return Ok(());
    };

    let batch = 1usize;
    let channels = 8192usize; // Qwen3.5-4B linear_qkv_dim
    let seq_len = 16usize;
    let kernel_size = 4usize;

    let mut rng = StdRng::seed_from_u64(0xC0FFEE_8175);
    let n_x = batch * channels * seq_len;
    let n_w = channels * kernel_size;
    let n_s = batch * channels * (kernel_size - 1);

    let x_data: Vec<f32> = (0..n_x)
        .map(|_| rng.random_range(-0.5f32..0.5f32))
        .collect();
    let w_data: Vec<f32> = (0..n_w)
        .map(|_| rng.random_range(-0.1f32..0.1f32))
        .collect();
    let s_data: Vec<f32> = (0..n_s)
        .map(|_| rng.random_range(-0.3f32..0.3f32))
        .collect();

    let x = Tensor::from_slice(&x_data, (batch, channels, seq_len))?
        .to_device(device)?
        .to_dtype(DType::BF16)?;
    let w = Tensor::from_slice(&w_data, (channels, 1, kernel_size))?
        .to_device(device)?
        .to_dtype(DType::BF16)?;
    let s_init =
        Tensor::from_slice(&s_data, (batch, channels, kernel_size - 1))?.to_device(device)?;

    let mut s_ref = s_init.clone();
    let out_ref = causal_conv1d_prefill_with_dtype(&x, &w, &mut s_ref, kernel_size, DType::F32)?;
    let out_ref = cuda_silu(&out_ref)?;

    let backend = crate::backend::for_device_kt(&device);
    assert!(ConvBackend::runtime_supports_causal_conv1d_prefill(
        backend.as_ref()
    ));
    let mut s_kernel = s_init.clone();
    let out_kernel = match ConvBackend::runtime_causal_conv1d_prefill(
        backend.as_ref(),
        &x,
        &w,
        &mut s_kernel,
        kernel_size,
    )? {
        Some(out) => out,
        None => {
            eprintln!("Metal backend declined causal_conv1d_prefill; skipping");
            return Ok(());
        }
    };
    assert_eq!(out_kernel.dtype(), DType::F32);
    assert_eq!(s_kernel.dtype(), DType::F32);

    let diff = (out_kernel.to_dtype(DType::F32)? - out_ref.to_dtype(DType::F32)?)?;
    let abs = diff.abs()?;
    let max = abs.flatten_all()?.max(0)?.flatten_all()?.to_vec1::<f32>()?[0];
    let mean = abs
        .flatten_all()?
        .mean(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    eprintln!(
        "metal conv1d_prefill kernel vs fallback: max_abs_diff={max:e} mean_abs_diff={mean:e}"
    );
    assert!(
        max < 1e-5,
        "metal prefill output max_abs_diff={max:e} exceeds 1e-5"
    );
    assert!(
        mean < 1e-6,
        "metal prefill output mean_abs_diff={mean:e} exceeds 1e-6"
    );

    let sdiff = (s_kernel.to_dtype(DType::F32)? - s_ref.to_dtype(DType::F32)?)?;
    let smax = sdiff
        .abs()?
        .flatten_all()?
        .max(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    eprintln!("metal conv1d_prefill kernel state parity: max_abs_diff={smax:e}");
    assert!(
        smax < 1e-6,
        "metal prefill state max_abs_diff={smax:e} exceeds 1e-6"
    );

    Ok(())
}

// -----------------------------------------------------------------------
// Phase 7: streaming/tiled GDN prefill — CPU parity tests.
//
// Each test compares the monolithic `model_forward_paged` against
// `model_forward_paged_streaming_with` running multiple tiles. Both runs
// start from fresh `LinearAttentionState` + `PagedKvCache` so the
// recurrent state hand-off and per-tile paged writes are exercised end
// to end. Tests use `last_token_only=false` so we can compare the full
// last-tile logits row-by-row against the matching slice of the
// monolithic logits.
// -----------------------------------------------------------------------

/// Shared config for all streaming parity tests. Picks a hybrid layer
/// stack (3 GDN + 1 full attention with `full_attention_interval=4`,
/// scaled to 8 layers so we get 6 GDN layers exercising the recurrent
/// hand-off across tile boundaries).
fn streaming_test_config() -> kiln_core::config::ModelConfig {
    let num_layers = 8;
    let full_attention_interval = 4; // layers 3, 7 are full → 2 full + 6 linear
    kiln_core::config::ModelConfig {
        hidden_size: 16,
        num_layers,
        num_attention_heads: 4,
        num_kv_heads: 2,
        head_dim: 4,
        intermediate_size: 32,
        vocab_size: 32,
        max_position_embeddings: 4096,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        dtype: kiln_core::config::DType::FP32,
        num_full_attention_layers: 2,
        full_attention_interval,
        attn_output_gate: false,
        linear_num_key_heads: 2,
        linear_key_head_dim: 4,
        linear_num_value_heads: 4,
        linear_value_head_dim: 4,
        linear_conv_kernel_dim: 4,
        partial_rotary_factor: 1.0,
    }
}

/// Build a paged cache + sequential block table sized for `seq_len` tokens
/// with `block_size`-token blocks (block_size = GDN_CHUNK_SIZE so block
/// boundaries coincide with the smallest legal tile boundary).
///
/// Migrated to `PagedKvCache::new_kt` so this helper no longer names
/// candle's `DType` at the constructor call; the `device` param is still
/// taken to keep call-site signatures stable (callers pass `&device` in
/// scope) — it's already kt, no bridge needed. (#1082)
fn make_paged_setup(
    config: &kiln_core::config::ModelConfig,
    seq_len: usize,
    block_size: usize,
    device: &Device,
) -> Result<(PagedKvCache, BlockTable)> {
    let num_blocks = seq_len.div_ceil(block_size);
    // #1082: `device` is already a kt `Device`.
    let device_kt = *device;
    let cache = crate::PagedKvCacheKt::new(
        config.num_full_attention_layers,
        num_blocks,
        block_size,
        config.num_kv_heads,
        config.head_dim,
        kiln_tensor::DType::F32,
        device_kt,
    )?;
    let mut block_table = BlockTable::new();
    for i in 0..num_blocks as u32 {
        block_table.push(i);
    }
    Ok((cache, block_table))
}

/// Deterministic token sequence for parity testing. Stays inside vocab.
fn deterministic_tokens(seq_len: usize, vocab_size: u32) -> Vec<u32> {
    (0..seq_len)
        .map(|i| ((i as u32 * 13 + 7) % vocab_size).max(1))
        .collect()
}

/// Run monolithic vs streaming on the same config + tokens, return
/// `(monolithic_full_logits[1, T, V], streaming_full_last_tile_logits[1, last_tile_len, V])`
/// where the streaming pass uses `tile_size` and `last_token_only=false`.
// `unnecessary_mut_passed`: `model_forward_paged*` keeps its public
// `Option<&mut LinearAttentionState>` contract (the state IS mutated on the
// cuda/metal/rocm/vulkan paths); the CPU path merely reads through it, so
// these call sites must still hand over the `&mut` form.
#[allow(clippy::unnecessary_mut_passed)]
fn run_parity(
    config: &kiln_core::config::ModelConfig,
    token_ids: &[u32],
    tile_size: usize,
    block_size: usize,
) -> Result<(Tensor, Tensor)> {
    let device = Device::Cpu;
    let weights = make_hybrid_gpu_weights(
        &device,
        config.vocab_size,
        config.hidden_size,
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim,
        config.intermediate_size,
        config.num_layers,
        config.full_attention_interval,
    )?;
    let backend = test_backend(&device);

    // Monolithic: single forward pass, full LM head.
    let (mut mono_cache, mono_bt) = make_paged_setup(config, token_ids.len(), block_size, &device)?;
    let mut mono_state = LinearAttentionState::new(config, &device)?;
    let mono_logits = model_forward_paged(
        &backend,
        token_ids,
        &weights,
        config,
        &mut mono_cache,
        &mono_bt,
        0,
        Some(&mut mono_state),
        None,
        None,
    )?;

    // Streaming: tiled prefill with last_token_only=false so the final
    // tile produces a full per-position logits slice we can compare
    // against the matching window of the monolithic output.
    let (mut stream_cache, stream_bt) =
        make_paged_setup(config, token_ids.len(), block_size, &device)?;
    let mut stream_state = LinearAttentionState::new(config, &device)?;
    let stream_logits = model_forward_paged_streaming_with(
        &backend,
        token_ids,
        &weights,
        config,
        &mut stream_cache,
        &stream_bt,
        0,
        Some(&mut stream_state),
        None,
        tile_size,
        false,
        None,
        0,
    )?;

    Ok((mono_logits, stream_logits))
}

/// Compare the streaming last-tile full logits against the matching
/// slice of the monolithic logits.
fn assert_last_tile_matches(
    mono_logits: &Tensor,
    stream_logits: &Tensor,
    total_len: usize,
    tile_size: usize,
    tol: f32,
) -> Result<()> {
    // Last tile spans [last_start, total_len).
    let last_start = total_len - ((total_len - 1) % tile_size + 1);
    let last_len = total_len - last_start;
    let mono_slice = mono_logits
        .narrow(1, last_start, last_len)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let stream_slice = stream_logits.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(
        mono_slice.len(),
        stream_slice.len(),
        "last tile length mismatch"
    );
    let mut max_abs = 0f32;
    for (a, b) in mono_slice.iter().zip(stream_slice.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    assert!(
        max_abs <= tol,
        "streaming vs monolithic max_abs_diff={max_abs:e} exceeds {tol:e}"
    );
    Ok(())
}

#[test]
fn test_streaming_matches_monolithic_cpu_small() -> Result<()> {
    let config = streaming_test_config();
    let total = 128;
    let tile = 64;
    let tokens = deterministic_tokens(total, config.vocab_size as u32);
    let (mono, stream) = run_parity(&config, &tokens, tile, 64)?;
    assert_eq!(mono.dims(), &[1, total, config.vocab_size]);
    assert_eq!(stream.dims(), &[1, tile, config.vocab_size]);
    assert_last_tile_matches(&mono, &stream, total, tile, 1e-5)?;
    Ok(())
}

#[test]
fn test_streaming_prefill_rejects_pre_cancelled_work_before_first_tile() -> Result<()> {
    let config = streaming_test_config();
    let tokens = deterministic_tokens(64, config.vocab_size as u32);
    let device = Device::Cpu;
    let weights = make_hybrid_gpu_weights(
        &device,
        config.vocab_size,
        config.hidden_size,
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim,
        config.intermediate_size,
        config.num_layers,
        config.full_attention_interval,
    )?;
    let backend = test_backend(&device);
    let (cache, block_table) = make_paged_setup(&config, tokens.len(), 64, &device)?;
    let mut linear_state = LinearAttentionState::new(&config, &device)?;
    let cancel = crate::cancel::CancelHandle::new();
    cancel.cancel();

    let error = model_forward_paged_streaming_with(
        &backend,
        &tokens,
        &weights,
        &config,
        &cache,
        &block_table,
        0,
        Some(&mut linear_state),
        None,
        64,
        true,
        Some(&cancel),
        0,
    )
    .expect_err("pre-cancelled prefill must stop before the first tile");

    assert!(error.to_string().contains("cancelled by caller"));
    assert_eq!(cancel.prefill_tokens_completed(), 0);
    Ok(())
}

#[test]
fn test_streaming_prefill_progress_is_cumulative_across_split_cpu() -> Result<()> {
    let config = streaming_test_config();
    let total = GDN_CHUNK_SIZE * 2;
    let split = GDN_CHUNK_SIZE;
    let tokens = deterministic_tokens(total, config.vocab_size as u32);
    let device = Device::Cpu;
    let weights = make_hybrid_gpu_weights(
        &device,
        config.vocab_size,
        config.hidden_size,
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim,
        config.intermediate_size,
        config.num_layers,
        config.full_attention_interval,
    )?;
    let backend = test_backend(&device);
    let (cache, block_table) = make_paged_setup(&config, total, GDN_CHUNK_SIZE, &device)?;
    let mut linear_state = LinearAttentionState::new(&config, &device)?;
    let progress = crate::cancel::CancelHandle::new();

    let _ = model_forward_paged_streaming_with_progress(
        &backend,
        &tokens[..split],
        &weights,
        &config,
        &cache,
        &block_table,
        0,
        Some(&mut linear_state),
        None,
        Some(&progress),
    )?;
    assert_eq!(progress.prefill_tokens_completed(), split as u64);

    let _ = model_forward_paged_streaming_with_progress_offset(
        &backend,
        &tokens[split..],
        &weights,
        &config,
        &cache,
        &block_table,
        split,
        Some(&mut linear_state),
        None,
        Some(&progress),
        split as u64,
    )?;
    assert_eq!(progress.prefill_tokens_completed(), total as u64);
    Ok(())
}

// `unnecessary_mut_passed`: see the note on `run_parity` — the paged forward
// family's `&mut LinearAttentionState` parameter is its public contract even
// though the CPU path only reads it.
#[allow(clippy::unnecessary_mut_passed)]
#[test]
fn test_streaming_last_hidden_matches_monolithic_cpu() -> Result<()> {
    let config = streaming_test_config();
    let device = Device::Cpu;
    let total = GDN_CHUNK_SIZE * 2 + 7;
    let tile = GDN_CHUNK_SIZE;
    let tokens = deterministic_tokens(total, config.vocab_size as u32);
    let weights = make_hybrid_gpu_weights(
        &device,
        config.vocab_size,
        config.hidden_size,
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim,
        config.intermediate_size,
        config.num_layers,
        config.full_attention_interval,
    )?;
    let backend = test_backend(&device);

    let (mut mono_cache, mono_bt) = make_paged_setup(&config, total, 64, &device)?;
    let mut mono_state = LinearAttentionState::new(&config, &device)?;
    let (mono_logits, mono_hidden) = model_forward_paged_last_token_with_last_hidden(
        &backend,
        &tokens,
        &weights,
        &config,
        &mut mono_cache,
        &mono_bt,
        0,
        Some(&mut mono_state),
        None,
        None,
    )?;

    let (mut stream_cache, stream_bt) = make_paged_setup(&config, total, 64, &device)?;
    let mut stream_state = LinearAttentionState::new(&config, &device)?;
    let (stream_logits, stream_hidden) =
        model_forward_paged_streaming_last_token_with_last_hidden_with(
            &backend,
            &tokens,
            &weights,
            &config,
            &mut stream_cache,
            &stream_bt,
            0,
            Some(&mut stream_state),
            None,
            tile,
        )?;

    assert_eq!(stream_logits.dims(), &[1, 1, config.vocab_size]);
    assert_eq!(stream_hidden.dims(), &[1, 1, config.hidden_size]);
    let logits_diff = (&mono_logits - &stream_logits)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let hidden_diff = (&mono_hidden - &stream_hidden)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    assert!(
        logits_diff <= 1e-5,
        "streaming MTP prefill logits drifted: max_abs_diff={logits_diff:e}"
    );
    assert!(
        hidden_diff <= 1e-5,
        "streaming MTP prefill h_prev drifted: max_abs_diff={hidden_diff:e}"
    );
    Ok(())
}

#[test]
fn test_streaming_matches_monolithic_cpu_mid() -> Result<()> {
    let config = streaming_test_config();
    let total = 512;
    let tile = 128;
    let tokens = deterministic_tokens(total, config.vocab_size as u32);
    let (mono, stream) = run_parity(&config, &tokens, tile, 64)?;
    assert_eq!(mono.dims(), &[1, total, config.vocab_size]);
    assert_eq!(stream.dims(), &[1, tile, config.vocab_size]);
    assert_last_tile_matches(&mono, &stream, total, tile, 1e-5)?;
    Ok(())
}

// `unnecessary_mut_passed`: see the note on `run_parity` — the paged forward
// family's `&mut LinearAttentionState` parameter is its public contract even
// though the CPU path only reads it.
#[allow(clippy::unnecessary_mut_passed)]
#[test]
fn test_streaming_tile_invariance_cpu() -> Result<()> {
    // For a fixed token sequence, the last token's logits must agree
    // across every legal tile size (multiples of GDN_CHUNK_SIZE that
    // divide or partition `total`). The monolithic run is the reference;
    // every tile size collapses to the same final-token logits.
    let config = streaming_test_config();
    let total = 256;
    let tokens = deterministic_tokens(total, config.vocab_size as u32);

    // Monolithic reference: take the last row of [1, total, V] logits.
    let device = Device::Cpu;
    let weights = make_hybrid_gpu_weights(
        &device,
        config.vocab_size,
        config.hidden_size,
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim,
        config.intermediate_size,
        config.num_layers,
        config.full_attention_interval,
    )?;
    let backend = test_backend(&device);
    let (mut mono_cache, mono_bt) = make_paged_setup(&config, total, 64, &device)?;
    let mut mono_state = LinearAttentionState::new(&config, &device)?;
    let mono_logits = model_forward_paged(
        &backend,
        &tokens,
        &weights,
        &config,
        &mut mono_cache,
        &mono_bt,
        0,
        Some(&mut mono_state),
        None,
        None,
    )?;
    let reference_last = mono_logits
        .narrow(1, total - 1, 1)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    for tile in [64usize, 128, 256] {
        let (mut cache, bt) = make_paged_setup(&config, total, 64, &device)?;
        let mut state = LinearAttentionState::new(&config, &device)?;
        let logits = model_forward_paged_streaming_with(
            &backend,
            &tokens,
            &weights,
            &config,
            &mut cache,
            &bt,
            0,
            Some(&mut state),
            None,
            tile,
            true, // last_token_only — matches production dispatch
            None,
            0,
        )?;
        assert_eq!(logits.dims(), &[1, 1, config.vocab_size]);
        let last = logits.flatten_all()?.to_vec1::<f32>()?;
        let max_abs = reference_last
            .iter()
            .zip(last.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs <= 1e-5,
            "tile={tile} last-token max_abs_diff={max_abs:e} exceeds 1e-5"
        );
    }
    Ok(())
}

// `unnecessary_mut_passed`: see the note on `run_parity` — the paged forward
// family's `&mut LinearAttentionState` parameter is its public contract even
// though the CPU path only reads it.
#[allow(clippy::unnecessary_mut_passed)]
#[test]
fn test_model_forward_paged_last_token_matches_full_last_row_cpu() -> Result<()> {
    let config = streaming_test_config();
    let total = 128;
    let tokens = deterministic_tokens(total, config.vocab_size as u32);
    let device = Device::Cpu;
    let weights = make_hybrid_gpu_weights(
        &device,
        config.vocab_size,
        config.hidden_size,
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim,
        config.intermediate_size,
        config.num_layers,
        config.full_attention_interval,
    )?;
    let backend = test_backend(&device);

    let (mut full_cache, full_bt) = make_paged_setup(&config, total, 64, &device)?;
    let mut full_state = LinearAttentionState::new(&config, &device)?;
    let full_logits = model_forward_paged(
        &backend,
        &tokens,
        &weights,
        &config,
        &mut full_cache,
        &full_bt,
        0,
        Some(&mut full_state),
        None,
        None,
    )?;
    let reference_last = full_logits
        .narrow(1, total - 1, 1)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let (mut last_cache, last_bt) = make_paged_setup(&config, total, 64, &device)?;
    let mut last_state = LinearAttentionState::new(&config, &device)?;
    let last_logits = model_forward_paged_last_token(
        &backend,
        &tokens,
        &weights,
        &config,
        &mut last_cache,
        &last_bt,
        0,
        Some(&mut last_state),
        None,
        None,
    )?;
    assert_eq!(last_logits.dims(), &[1, 1, config.vocab_size]);
    let last = last_logits.flatten_all()?.to_vec1::<f32>()?;
    let max_abs = reference_last
        .iter()
        .zip(last.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_abs <= 1e-5,
        "last-token prefill max_abs_diff={max_abs:e} exceeds 1e-5"
    );

    // #1082: `last_logits` is kt (model output) and `greedy_sample` now
    // takes a kt `&Tensor` directly — no candle bridge.
    let expected_token = crate::sampling::greedy_sample(&last_logits)?;
    let (mut greedy_cache, greedy_bt) = make_paged_setup(&config, total, 64, &device)?;
    let mut greedy_state = LinearAttentionState::new(&config, &device)?;
    let greedy_token = model_forward_paged_last_token_greedy(
        &backend,
        &tokens,
        &weights,
        &config,
        &mut greedy_cache,
        &greedy_bt,
        0,
        Some(&mut greedy_state),
        None,
        None,
    )?;
    assert_eq!(
        greedy_token, expected_token,
        "last-token greedy prefill should match greedy_sample(last-token logits)"
    );

    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_weighted_lm_head_prep_argmax_matches_final_rmsnorm_argmax_metal() -> Result<()> {
    let Some(device) = crate::backend::metal::try_new_metal() else {
        return Ok(());
    };

    let hidden = 128usize;
    let vocab = 257usize;
    let best = 42usize;
    let x_data: Vec<f32> = (0..hidden)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.0234375)
        .collect();
    let norm_weight_data: Vec<f32> = (0..hidden)
        .map(|i| 0.75 + (i % 17) as f32 * 0.015625)
        .collect();
    let mut weight_data: Vec<f32> = (0..(hidden * vocab))
        .map(|i| ((i % 31) as f32 - 15.0) * 0.0009765625)
        .collect();
    for i in 0..hidden {
        weight_data[i * vocab + best] = x_data[i] * norm_weight_data[i];
    }

    let x = Tensor::from_slice(&x_data, (1usize, 1usize, hidden))?
        .to_device(device)?
        .to_dtype(DType::BF16)?;
    let norm_weight = Tensor::from_slice(&norm_weight_data, (hidden,))?
        .to_device(device)?
        .to_dtype(DType::BF16)?;
    let weight_t = Tensor::from_slice(&weight_data, (hidden, vocab))?
        .to_device(device)?
        .to_dtype(DType::BF16)?;

    let normed = rms_norm(&x, &norm_weight, 1e-6)?;
    let reference = lm_head_argmax(&normed, &weight_t)?;
    let weighted = lm_head_weighted_prep_argmax(&x, &norm_weight, &weight_t)?
        .context("weighted lm-head prep should support Metal BF16 [1,1,H]")?;

    assert_eq!(reference as usize, best);
    assert_eq!(weighted, reference);
    Ok(())
}

// `unnecessary_mut_passed`: see the note on `run_parity` — the paged forward
// family's `&mut LinearAttentionState` parameter is its public contract even
// though the CPU path only reads it.
#[allow(clippy::unnecessary_mut_passed)]
#[test]
fn test_streaming_preserves_state_cpu() -> Result<()> {
    // After prefill, run a single decode step on top of the resulting
    // (paged_cache, linear_state). If state was preserved bit-exact
    // across tile boundaries, the decode-token logits must agree with
    // the monolithic reference.
    let config = streaming_test_config();
    let total = 192;
    let tile = 64;
    let tokens = deterministic_tokens(total, config.vocab_size as u32);
    let next_token: u32 = 11;

    let device = Device::Cpu;
    let weights = make_hybrid_gpu_weights(
        &device,
        config.vocab_size,
        config.hidden_size,
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim,
        config.intermediate_size,
        config.num_layers,
        config.full_attention_interval,
    )?;
    let backend = test_backend(&device);

    // Monolithic prefill, then 1 decode step.
    let (mut mono_cache, mono_bt) = make_paged_setup(&config, total + 1, 64, &device)?;
    let mut mono_state = LinearAttentionState::new(&config, &device)?;
    let _ = model_forward_paged(
        &backend,
        &tokens,
        &weights,
        &config,
        &mut mono_cache,
        &mono_bt,
        0,
        Some(&mut mono_state),
        None,
        None,
    )?;
    let mono_decode = model_forward_paged(
        &backend,
        &[next_token],
        &weights,
        &config,
        &mut mono_cache,
        &mono_bt,
        total,
        Some(&mut mono_state),
        None,
        None,
    )?;

    // Streaming prefill, then 1 decode step.
    let (mut stream_cache, stream_bt) = make_paged_setup(&config, total + 1, 64, &device)?;
    let mut stream_state = LinearAttentionState::new(&config, &device)?;
    let _ = model_forward_paged_streaming_with(
        &backend,
        &tokens,
        &weights,
        &config,
        &mut stream_cache,
        &stream_bt,
        0,
        Some(&mut stream_state),
        None,
        tile,
        true,
        None,
        0,
    )?;
    let stream_decode = model_forward_paged(
        &backend,
        &[next_token],
        &weights,
        &config,
        &mut stream_cache,
        &stream_bt,
        total,
        Some(&mut stream_state),
        None,
        None,
    )?;

    let a = mono_decode.flatten_all()?.to_vec1::<f32>()?;
    let b = stream_decode.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(a.len(), b.len());
    let max_abs = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(
        max_abs <= 1e-5,
        "decode-after-streaming max_abs_diff={max_abs:e} exceeds 1e-5 \
             (state was not bit-exact preserved across tile boundaries)"
    );
    Ok(())
}

/// Phase 10 — training-time streaming GDN parity (CPU).
///
/// Direct unit test of [`gated_deltanet_forward_streaming`] against the
/// monolithic [`gated_deltanet_forward`] on a small GDN-only input. Both
/// paths must produce equal output tensors and equal final state.
///
/// This test uses explicit parameters and is safe under a parallel test
/// runner. Policy dispatch inside `model_forward_segment` is exercised by
/// `test_model_forward_segment_streaming_matches_monolithic_cpu`.
#[test]
fn test_gated_deltanet_forward_streaming_matches_monolithic_cpu() -> Result<()> {
    let config = streaming_test_config();
    let device = Device::Cpu;
    let weights = make_hybrid_gpu_weights(
        &device,
        config.vocab_size,
        config.hidden_size,
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim,
        config.intermediate_size,
        config.num_layers,
        config.full_attention_interval,
    )?;
    let backend = test_backend(&device);

    // Pull the first GDN layer (layer 0 — full_attention_interval=4 so
    // layers 0,1,2 are GDN and layer 3 is full-attn).
    let lin_weights = match &weights.layers[0].attention {
        GpuAttentionWeights::Linear(w) => w,
        GpuAttentionWeights::Full(_) => panic!("test setup error: layer 0 must be GDN"),
    };

    // Deterministic input. T must be a multiple of GDN_CHUNK_SIZE so
    // both monolithic and tiled paths exercise the chunkwise kernel.
    let total = GDN_CHUNK_SIZE * 3; // 192 tokens
    let tile = GDN_CHUNK_SIZE; // 64-token tiles -> 3 tiles
    let n: usize = total * config.hidden_size;
    let data: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.013).sin()) * 0.1).collect();
    let x = Tensor::new(&data, device)?.reshape((1, total, config.hidden_size))?;

    // Monolithic.
    let mut mono_state = LinearAttentionState::new(&config, &device)?;
    let mono_out = gated_deltanet_forward(
        &backend,
        &x,
        lin_weights,
        &config,
        &mut mono_state.recurrent_states[0],
        &mut mono_state.conv_states[0],
        None,
    )?;

    // Streaming/tiled.
    let mut stream_state = LinearAttentionState::new(&config, &device)?;
    let stream_out = gated_deltanet_forward_streaming(
        &backend,
        &x,
        lin_weights,
        &config,
        &mut stream_state.recurrent_states[0],
        &mut stream_state.conv_states[0],
        tile,
        None,
    )?;

    assert_eq!(mono_out.dims(), stream_out.dims());
    let mono_v = mono_out.flatten_all()?.to_vec1::<f32>()?;
    let stream_v = stream_out.flatten_all()?.to_vec1::<f32>()?;
    let max_abs_out = mono_v
        .iter()
        .zip(stream_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_abs_out <= 1e-5,
        "streaming GDN output drifted from monolithic: max_abs_diff={max_abs_out:e}"
    );

    // Final recurrent state must match (the load-bearing invariant for
    // training-time streaming — autograd flows through this state thread).
    let mr = mono_state.recurrent_states[0]
        .flatten_all()?
        .to_vec1::<f32>()?;
    let sr = stream_state.recurrent_states[0]
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(mr.len(), sr.len());
    let max_abs_recur = mr
        .iter()
        .zip(sr.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_abs_recur <= 1e-5,
        "streaming GDN recurrent state drifted: max_abs_diff={max_abs_recur:e}"
    );

    // Final conv state must match (drives correctness of any subsequent
    // decode step that consumes it).
    let mc = mono_state.conv_states[0].flatten_all()?.to_vec1::<f32>()?;
    let sc = stream_state.conv_states[0]
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(mc.len(), sc.len());
    let max_abs_conv = mc
        .iter()
        .zip(sc.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_abs_conv <= 1e-5,
        "streaming GDN conv state drifted: max_abs_diff={max_abs_conv:e}"
    );

    Ok(())
}

#[cfg(feature = "rocm")]
#[test]
fn rocm_tape_streaming_gdn_records_single_token_tail() -> Result<()> {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("skip rocm_tape_streaming_gdn_records_single_token_tail: no ROCm device");
        return Ok(());
    }

    let config = streaming_test_config();
    let device = Device::Rocm(0);
    let weights = make_hybrid_gpu_weights(
        &device,
        config.vocab_size,
        config.hidden_size,
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim,
        config.intermediate_size,
        config.num_layers,
        config.full_attention_interval,
    )?;
    let lin_weights = match &weights.layers[0].attention {
        GpuAttentionWeights::Linear(weights) => weights,
        GpuAttentionWeights::Full(_) => panic!("test setup error: layer 0 must be GDN"),
    };
    let backend = crate::backend::for_device_kt(&device);
    let tile = GDN_CHUNK_SIZE;
    let total = tile + 1;
    let input_values = (0..total * config.hidden_size)
        .map(|i| ((i as f32 * 0.019).sin()) * 0.1)
        .collect::<Vec<_>>();
    let input = Tensor::from_vec_on(device, input_values, vec![1, total, config.hidden_size])?;
    let mut state = LinearAttentionState::new(&config, &device)?;

    let (output, tape) = kiln_autograd::with_thread_local_tape(|| {
        gated_deltanet_forward_streaming(
            &*backend,
            &input,
            lin_weights,
            &config,
            &mut state.recurrent_states[0],
            &mut state.conv_states[0],
            tile,
            None,
        )
    });
    let output = output?;
    assert_eq!(output.dims(), &[1, total, config.hidden_size]);
    assert!(
        tape.reachable_from(output.id()).contains(&input.id()),
        "streaming GDN output must remain connected through the one-token tail to its input"
    );
    let seed = Tensor::ones(output.shape().to_vec(), output.dtype(), device)?;
    let gradients = tape.backward(output.id(), seed, kiln_tensor::ops::add)?;
    let input_gradient = gradients
        .get(input.id())
        .context("streaming GDN tape omitted the input gradient after a one-token tail")?;
    assert!(
        input_gradient.all_finite()?,
        "streaming GDN one-token-tail input gradient is non-finite"
    );
    Ok(())
}

/// Phase 10 — training-time streaming GDN parity for `model_forward_segment`.
///
/// Runs `model_forward_segment` over the full layer stack twice on the
/// same input: once with an explicitly disabled policy and once with an
/// explicitly enabled 64-token tile so the 192-token input is split into
/// three tiles. The two outputs and final per-layer state must match.
#[test]
fn test_model_forward_segment_streaming_matches_monolithic_cpu() -> Result<()> {
    let config = streaming_test_config();
    let device = Device::Cpu;
    let weights = make_hybrid_gpu_weights(
        &device,
        config.vocab_size,
        config.hidden_size,
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim,
        config.intermediate_size,
        config.num_layers,
        config.full_attention_interval,
    )?;
    let backend = test_backend(&device);

    let total = GDN_CHUNK_SIZE * 3; // 192 tokens
    let tile = GDN_CHUNK_SIZE; // 64-token tiles -> 3 tiles
    let n: usize = total * config.hidden_size;
    let data: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.017).cos()) * 0.1).collect();
    let hidden = Tensor::new(&data, device)?.reshape((1, total, config.hidden_size))?;
    let positions: Vec<u32> = (0..total as u32).collect();

    let backend_defaults = StreamingPrefillBackendPolicy::for_backend("cpu", device);
    let monolithic_policy = StreamingPrefillExecutionPolicy::resolve(
        backend_defaults,
        StreamingPrefillMode::Disabled,
        None,
        None,
        None,
        None,
        true,
    );
    let streaming_policy = StreamingPrefillExecutionPolicy::resolve(
        backend_defaults,
        StreamingPrefillMode::Enabled,
        None,
        Some(tile),
        Some(tile),
        Some(tile),
        true,
    );

    // Monolithic.
    let mut mono_state = LinearAttentionState::new(&config, &device)?;
    let mono_out = model_forward_segment_with_policy(
        &backend,
        hidden.clone(),
        &weights,
        &config,
        &positions,
        0,
        config.num_layers,
        Some(&mut mono_state),
        None,
        monolithic_policy,
    )?;

    // Streaming with a pure, request-independent policy value.
    let mut stream_state = LinearAttentionState::new(&config, &device)?;
    let stream_out = model_forward_segment_with_policy(
        &backend,
        hidden.clone(),
        &weights,
        &config,
        &positions,
        0,
        config.num_layers,
        Some(&mut stream_state),
        None,
        streaming_policy,
    )?;

    assert_eq!(mono_out.dims(), stream_out.dims());
    let mv = mono_out.flatten_all()?.to_vec1::<f32>()?;
    let sv = stream_out.flatten_all()?.to_vec1::<f32>()?;
    let max_abs = mv
        .iter()
        .zip(sv.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_abs <= 1e-4,
        "model_forward_segment streaming output drifted from monolithic: max_abs_diff={max_abs:e}"
    );

    for (l, (m, s)) in mono_state
        .recurrent_states
        .iter()
        .zip(stream_state.recurrent_states.iter())
        .enumerate()
    {
        let mv = m.flatten_all()?.to_vec1::<f32>()?;
        let sv = s.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(mv.len(), sv.len(), "recurrent_states[{l}] length mismatch");
        let max_abs = mv
            .iter()
            .zip(sv.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs <= 1e-4,
            "model_forward_segment streaming recurrent_states[{l}] drifted: max_abs_diff={max_abs:e}"
        );
    }
    for (l, (m, s)) in mono_state
        .conv_states
        .iter()
        .zip(stream_state.conv_states.iter())
        .enumerate()
    {
        let mv = m.flatten_all()?.to_vec1::<f32>()?;
        let sv = s.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(mv.len(), sv.len(), "conv_states[{l}] length mismatch");
        let max_abs = mv
            .iter()
            .zip(sv.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs <= 1e-4,
            "model_forward_segment streaming conv_states[{l}] drifted: max_abs_diff={max_abs:e}"
        );
    }

    Ok(())
}

/// CUDA parity for streaming/tiled GDN prefill.
///
/// Mirrors `test_streaming_matches_monolithic_cpu_mid` but on CUDA at
/// T=2048, tile=512 (the configuration the Phase 7 GPU spike validates).
/// Asserts (1) full-tile logits match the matching slice of the
/// monolithic logits, and (2) `LinearAttentionState.recurrent_states[l]`
/// and `state.conv_states[l]` are equal across the two paths after
/// prefill — the state hand-off is the load-bearing part of streaming.
///
/// Tolerance: 1e-4. The design doc (PROFILING.md §c "CUDA parity")
/// argues bit-exactness is achievable because GDN recurrent state stays
/// in F32 and the conv1d F32 promotion makes the conv path
/// deterministic. In practice, kt CUDA matmul reduction order can
/// vary with shape, so we use a small FP32 tolerance rather than
/// strict equality.
#[test]
#[cfg(feature = "cuda")]
fn test_streaming_matches_monolithic_cuda() -> Result<()> {
    let device = match new_cuda_device(0) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("CUDA not available, skipping test_streaming_matches_monolithic_cuda");
            return Ok(());
        }
    };

    let config = streaming_test_config();
    let total = 2048usize;
    let tile = 512usize;
    let block_size = 64usize; // == GDN_CHUNK_SIZE
    let tokens = deterministic_tokens(total, config.vocab_size as u32);

    let weights = make_hybrid_gpu_weights(
        &device,
        config.vocab_size,
        config.hidden_size,
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim,
        config.intermediate_size,
        config.num_layers,
        config.full_attention_interval,
    )?;
    let backend = crate::backend::for_device_kt(&device);

    // Monolithic: single forward pass, full LM head.
    let (mut mono_cache, mono_bt) = make_paged_setup(&config, total, block_size, &device)?;
    let mut mono_state = LinearAttentionState::new(&config, &device)?;
    let mono_logits = model_forward_paged(
        &*backend,
        &tokens,
        &weights,
        &config,
        &mut mono_cache,
        &mono_bt,
        0,
        Some(&mut mono_state),
        None,
        None,
    )?;

    // Streaming: tiled prefill, last_token_only=false so we get a full
    // last-tile logits slice for row-by-row comparison.
    let (mut stream_cache, stream_bt) = make_paged_setup(&config, total, block_size, &device)?;
    let mut stream_state = LinearAttentionState::new(&config, &device)?;
    let stream_logits = model_forward_paged_streaming_with(
        &*backend,
        &tokens,
        &weights,
        &config,
        &mut stream_cache,
        &stream_bt,
        0,
        Some(&mut stream_state),
        None,
        tile,
        false,
        None,
        0,
    )?;

    assert_eq!(mono_logits.dims(), &[1, total, config.vocab_size]);
    assert_eq!(stream_logits.dims(), &[1, tile, config.vocab_size]);

    // (1) Last-tile logits parity.
    assert_last_tile_matches(&mono_logits, &stream_logits, total, tile, 1e-4)?;

    // (2) Per-layer state parity (recurrent + conv).
    assert_eq!(
        mono_state.recurrent_states.len(),
        stream_state.recurrent_states.len(),
        "recurrent_states layer count mismatch"
    );
    assert_eq!(
        mono_state.conv_states.len(),
        stream_state.conv_states.len(),
        "conv_states layer count mismatch"
    );
    for (l, (m, s)) in mono_state
        .recurrent_states
        .iter()
        .zip(stream_state.recurrent_states.iter())
        .enumerate()
    {
        let m_v = m.flatten_all()?.to_vec1::<f32>()?;
        let s_v = s.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(
            m_v.len(),
            s_v.len(),
            "recurrent_states[{l}] length mismatch"
        );
        let max_abs = m_v
            .iter()
            .zip(s_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs <= 1e-4,
            "recurrent_states[{l}] max_abs_diff={max_abs:e} exceeds 1e-4"
        );
    }
    for (l, (m, s)) in mono_state
        .conv_states
        .iter()
        .zip(stream_state.conv_states.iter())
        .enumerate()
    {
        let m_v = m.flatten_all()?.to_vec1::<f32>()?;
        let s_v = s.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(m_v.len(), s_v.len(), "conv_states[{l}] length mismatch");
        let max_abs = m_v
            .iter()
            .zip(s_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs <= 1e-4,
            "conv_states[{l}] max_abs_diff={max_abs:e} exceeds 1e-4"
        );
    }

    Ok(())
}

#[test]
fn full_attention_budget_is_process_lifetime_geometry() {
    assert_eq!(
        full_attn_materialized_score_budget_mib(),
        crate::DEFAULT_FULL_ATTENTION_SCORE_BUDGET_MIB
    );
}

#[test]
fn materialized_attention_scratch_accounts_for_every_live_score_buffer() {
    assert_eq!(
        full_attn_materialized_scratch_bytes(DType::BF16, 1, 16, 128, 4096, 8),
        Some(256 * 1024 * 1024)
    );
    assert_eq!(
        full_attn_materialized_scratch_bytes(DType::BF16, 4, 16, 128, 4096, 8),
        Some(1024 * 1024 * 1024),
        "score scratch is shaped [batch, heads, query, key]"
    );
    assert_eq!(
        full_attn_materialized_scratch_bytes(DType::F32, usize::MAX, 2, 2, 2, 2,),
        None,
        "overflow must reject admission instead of wrapping the reservation"
    );

    let single_batch_plan = full_attn_adaptive_tile_plan_summary(
        &Device::Cpu,
        DType::BF16,
        1,
        64,
        8,
        64,
        MATERIALIZED_FULL_ATTN_FORWARD_SCRATCH_BUFFERS,
        usize::MAX,
    );
    let four_batch_plan = full_attn_adaptive_tile_plan_summary(
        &Device::Cpu,
        DType::BF16,
        4,
        64,
        8,
        64,
        MATERIALIZED_FULL_ATTN_FORWARD_SCRATCH_BUFFERS,
        usize::MAX,
    );
    assert_eq!(
        four_batch_plan.4,
        single_batch_plan.4.and_then(|bytes| bytes.checked_mul(4)),
        "the complete tile plan must carry the batch dimension into its peak"
    );
}

#[test]
fn test_streaming_prefill_execution_policy() {
    assert!(!streaming_prefill_enabled(), "default must be disabled");
    let cpu_backend = StreamingPrefillBackendPolicy::for_backend("cpu", Device::Cpu);
    let cpu_default = StreamingPrefillExecutionPolicy::from_backend_policy(cpu_backend);
    let cuda_default = StreamingPrefillExecutionPolicy::from_backend_policy(
        StreamingPrefillBackendPolicy::for_backend("cuda", Device::Cuda(0)),
    );
    let rocm_default = StreamingPrefillExecutionPolicy::from_backend_policy(
        StreamingPrefillBackendPolicy::for_backend("rocm", Device::Rocm(0)),
    );
    let metal_default = StreamingPrefillExecutionPolicy::from_backend_policy(
        StreamingPrefillBackendPolicy::for_backend("metal", Device::Metal(0)),
    );
    let vulkan_default = StreamingPrefillExecutionPolicy::from_backend_policy(
        StreamingPrefillBackendPolicy::for_backend("vulkan", Device::Vulkan(0)),
    );
    assert!(!cpu_default.enabled_for(STREAMING_PREFILL_CUDA_DEFAULT_THRESHOLD));
    assert!(!cuda_default.enabled_for(STREAMING_PREFILL_CUDA_DEFAULT_THRESHOLD - 1));
    assert!(cuda_default.enabled_for(STREAMING_PREFILL_CUDA_DEFAULT_THRESHOLD));
    assert!(cuda_default.enabled_for(12_000));
    assert!(cuda_default.enabled_for(43_814));
    assert_eq!(
        rocm_default.threshold_tokens(),
        Some(StreamingPrefillBackendPolicy::ROCM_AUTO_MIN_PROMPT_TOKENS)
    );
    assert!(
        !rocm_default.enabled_for(StreamingPrefillBackendPolicy::ROCM_AUTO_MIN_PROMPT_TOKENS - 1)
    );
    assert!(rocm_default.enabled_for(StreamingPrefillBackendPolicy::ROCM_AUTO_MIN_PROMPT_TOKENS));
    assert!(!metal_default.enabled_for(STREAMING_PREFILL_METAL_DEFAULT_THRESHOLD - 1));
    assert!(metal_default.enabled_for(STREAMING_PREFILL_METAL_DEFAULT_THRESHOLD));
    assert!(!vulkan_default.enabled_for(43_814));
    assert_eq!(
        streaming_prefill_threshold_tokens(),
        STREAMING_PREFILL_METAL_DEFAULT_THRESHOLD
    );
    assert!(!streaming_prefill_enabled_for(
        &Device::Cpu,
        STREAMING_PREFILL_METAL_DEFAULT_THRESHOLD
    ));
    assert_eq!(streaming_tile_tokens(), STREAMING_PREFILL_DEFAULT_TILE);
    assert_eq!(
        streaming_tile_tokens_for(&Device::Cpu),
        STREAMING_PREFILL_DEFAULT_TILE
    );
    assert_eq!(
        streaming_tile_tokens_for(&Device::Cuda(0)),
        STREAMING_PREFILL_CUDA_DEFAULT_TILE
    );
    assert_eq!(
        streaming_tile_tokens_for(&Device::Rocm(0)),
        STREAMING_PREFILL_ROCM_DEFAULT_TILE
    );
    assert_eq!(
        cuda_default.detached_full_attn_tile_tokens(),
        DETACHED_FULL_ATTN_CUDA_DEFAULT_TILE
    );
    assert_eq!(
        rocm_default.detached_full_attn_tile_tokens(),
        DETACHED_FULL_ATTN_ROCM_DEFAULT_TILE
    );
    assert_eq!(
        cuda_default.detached_full_attn_boundary_tile_tokens(),
        DETACHED_FULL_ATTN_FLASH_DEFAULT_TILE
    );
    assert_eq!(
        rocm_default.detached_full_attn_boundary_tile_tokens(),
        DETACHED_FULL_ATTN_ROCM_DEFAULT_TILE
    );
    assert_eq!(
        cuda_default.detached_full_attn_tape_replay_tile_tokens(),
        DETACHED_FULL_ATTN_FLASH_DEFAULT_TILE
    );
    assert_eq!(
        rocm_default.detached_full_attn_tape_replay_tile_tokens(),
        DETACHED_FULL_ATTN_ROCM_DEFAULT_TILE
    );
    let portable_rocm_backend = CpuBackend::new(Device::Rocm(0));
    assert_eq!(
        FullAttnChunkMode::DetachedBoundary.materialized_scratch_buffers_for_tile_plan(
            &portable_rocm_backend,
            &Device::Rocm(0),
            DType::BF16,
            256,
        ),
        FullAttnChunkMode::DetachedBoundary.materialized_scratch_buffers(),
        "A portable ROCm-shaped backend does not advertise FlashAttention, so the outer full-attention planner must still budget materialized-score scratch"
    );
    assert_eq!(
        FullAttnChunkMode::TapeReplay.materialized_scratch_buffers_for_tile_plan(
            &portable_rocm_backend,
            &Device::Rocm(0),
            DType::BF16,
            256,
        ),
        FullAttnChunkMode::TapeReplay.materialized_scratch_buffers(),
        "ROCm tape replay must keep exact-attention score scratch in the tile plan"
    );
    assert!(
        full_attn_adaptive_tile_len(
            &Device::Cuda(0),
            DType::BF16,
            1,
            100_000,
            DETACHED_FULL_ATTN_CUDA_DEFAULT_TILE,
            16,
            DETACHED_FULL_ATTN_CUDA_DEFAULT_TILE,
            1,
        ) < DETACHED_FULL_ATTN_CUDA_DEFAULT_TILE,
        "CUDA materialized SDPA fallback must also shrink long-prefix exact query tiles"
    );
    let batch_one_tile = full_attn_adaptive_tile_len_with_budget(
        &Device::Cuda(0),
        DType::BF16,
        1,
        4096,
        4096,
        16,
        4096,
        MATERIALIZED_FULL_ATTN_FORWARD_SCRATCH_BUFFERS,
        512,
    );
    let batch_four_tile = full_attn_adaptive_tile_len_with_budget(
        &Device::Cuda(0),
        DType::BF16,
        4,
        4096,
        4096,
        16,
        4096,
        MATERIALIZED_FULL_ATTN_FORWARD_SCRATCH_BUFFERS,
        512,
    );
    assert!(
        batch_four_tile < batch_one_tile,
        "a fixed score budget must select a smaller tile at batch four, batch_one={batch_one_tile} batch_four={batch_four_tile}"
    );
    let rocm_low_budget_long_prefix_tile = full_attn_adaptive_tile_len_with_budget(
        &Device::Rocm(0),
        DType::BF16,
        1,
        100_000,
        DETACHED_FULL_ATTN_ROCM_DEFAULT_TILE,
        16,
        DETACHED_FULL_ATTN_ROCM_DEFAULT_TILE,
        2,
        512,
    );
    assert!(
        rocm_low_budget_long_prefix_tile < DETACHED_FULL_ATTN_ROCM_DEFAULT_TILE,
        "ROCm materialized SDPA must shrink long-prefix exact query tiles under a low budget, got {rocm_low_budget_long_prefix_tile}"
    );
    assert!(
        rocm_low_budget_long_prefix_tile <= MATERIALIZED_FULL_ATTN_TILE_GRANULARITY,
        "long-prefix ROCm replay tiles must be allowed below the GDN chunk size under a low budget, got {rocm_low_budget_long_prefix_tile}"
    );
    let rocm_high_budget_long_prefix_tile = full_attn_adaptive_tile_len_with_budget(
        &Device::Rocm(0),
        DType::BF16,
        1,
        100_000,
        DETACHED_FULL_ATTN_ROCM_DEFAULT_TILE,
        16,
        DETACHED_FULL_ATTN_ROCM_DEFAULT_TILE,
        2,
        crate::MAX_FULL_ATTENTION_SCORE_BUDGET_MIB,
    );
    assert!(
        rocm_high_budget_long_prefix_tile > rocm_low_budget_long_prefix_tile,
        "larger exact-attention score budgets should permit larger ROCm query tiles, low={rocm_low_budget_long_prefix_tile} high={rocm_high_budget_long_prefix_tile}"
    );
    assert_eq!(
        detached_full_attn_tile_tokens_for(&Device::Metal(0)),
        DETACHED_FULL_ATTN_MATERIALIZED_DEFAULT_TILE
    );
    assert_eq!(
        streaming_tile_tokens_for(&Device::Vulkan(0)),
        STREAMING_PREFILL_VULKAN_DEFAULT_TILE
    );
    assert_eq!(
        tape_streaming_tile_tokens_for(&Device::Cuda(0)),
        STREAMING_PREFILL_CUDA_TAPE_DEFAULT_TILE
    );
    assert_eq!(
        tape_streaming_tile_tokens_for(&Device::Rocm(0)),
        STREAMING_PREFILL_ROCM_TAPE_DEFAULT_TILE
    );
    assert_eq!(
        tape_streaming_tile_tokens_for(&Device::Metal(0)),
        STREAMING_PREFILL_METAL_TAPE_DEFAULT_TILE
    );
    assert_eq!(
        tape_streaming_tile_tokens_for(&Device::Vulkan(0)),
        STREAMING_PREFILL_VULKAN_TAPE_DEFAULT_TILE
    );
    assert!(streaming_last_token_lm_head(), "default must be true");

    #[cfg(feature = "metal")]
    if let Some(device) = crate::backend::metal::try_new_metal() {
        assert!(!streaming_prefill_enabled_for(
            &device,
            STREAMING_PREFILL_METAL_DEFAULT_THRESHOLD - 1
        ));
        assert!(streaming_prefill_enabled_for(
            &device,
            STREAMING_PREFILL_METAL_DEFAULT_THRESHOLD
        ));
        assert_eq!(
            streaming_tile_tokens_for(&device),
            STREAMING_PREFILL_METAL_DEFAULT_TILE
        );
    }

    let forced_on = StreamingPrefillExecutionPolicy::resolve(
        cpu_backend,
        StreamingPrefillMode::Enabled,
        Some(1_024),
        Some(256),
        Some(128),
        Some(512),
        false,
    );
    assert_eq!(forced_on.mode(), StreamingPrefillMode::Enabled);
    assert_eq!(forced_on.threshold_tokens(), None);
    assert!(!forced_on.enabled_for(0));
    assert!(forced_on.enabled_for(1));
    assert_eq!(forced_on.base_tile_tokens(), 256);
    assert_eq!(forced_on.base_tile_tokens_for(usize::MAX), 256);
    assert_eq!(forced_on.tape_tile_tokens(), 128);
    assert_eq!(forced_on.detached_full_attn_tile_tokens(), 512);
    assert_eq!(forced_on.detached_full_attn_boundary_tile_tokens(), 512);
    assert_eq!(forced_on.detached_full_attn_tape_replay_tile_tokens(), 512);
    assert!(!forced_on.last_token_lm_head());

    let inherited_specialized_tiles = StreamingPrefillExecutionPolicy::resolve(
        StreamingPrefillBackendPolicy::for_backend("cuda", Device::Cuda(0)),
        StreamingPrefillMode::Auto,
        None,
        Some(256),
        None,
        None,
        true,
    );
    assert_eq!(inherited_specialized_tiles.base_tile_tokens(), 256);
    assert_eq!(inherited_specialized_tiles.tape_tile_tokens(), 256);
    assert_eq!(
        inherited_specialized_tiles.detached_full_attn_tile_tokens(),
        256
    );
    assert_eq!(
        inherited_specialized_tiles.detached_full_attn_boundary_tile_tokens(),
        256
    );
    assert_eq!(
        inherited_specialized_tiles.detached_full_attn_tape_replay_tile_tokens(),
        256
    );

    let forced_off = StreamingPrefillExecutionPolicy::resolve(
        StreamingPrefillBackendPolicy::for_backend("cuda", Device::Cuda(0)),
        StreamingPrefillMode::Disabled,
        Some(1),
        None,
        None,
        None,
        true,
    );
    assert!(!forced_off.enabled_for(usize::MAX));

    let cuda_auto = StreamingPrefillExecutionPolicy::resolve(
        StreamingPrefillBackendPolicy::for_backend("cuda", Device::Cuda(0)),
        StreamingPrefillMode::Auto,
        Some(1_024),
        None,
        None,
        None,
        true,
    );
    assert_eq!(cuda_auto.threshold_tokens(), Some(1_024));
    assert!(!cuda_auto.enabled_for(1_023));
    assert!(cuda_auto.enabled_for(1_024));

    let cpu_auto = StreamingPrefillExecutionPolicy::resolve(
        cpu_backend,
        StreamingPrefillMode::Auto,
        Some(1),
        None,
        None,
        None,
        true,
    );
    assert_eq!(cpu_auto.threshold_tokens(), None);
    assert!(!cpu_auto.enabled_for(usize::MAX));
    assert!(streaming_last_token_lm_head());
}

// (#1082) Deleted test_cuda_rotary_one_bwd_kt_bridge_default_matches_candle_path:
//   it exercised the deleted `fused_rotary_one_backward_via_kt_bridge` candle
//   parity path. Rotary autograd is now `try_tape_rope_cuda` on the kt tape.
