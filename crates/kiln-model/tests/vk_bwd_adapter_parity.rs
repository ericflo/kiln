//! PR4b — gradient validation of [`kiln_model::vk_bwd_adapter::VkBwdAdapter`]
//! on the real Vulkan GPU (#1082).
//!
//! Two complementary checks per Wave-1 family (matmul / rms_norm / rope /
//! softmax_lastdim — the families whose forward exists in `vk_ops` and that
//! PR4a's [`family_ported`] lists):
//!
//!   1. **EXACT-vs-direct** (strongest, cheapest). Run the real `vk_ops`
//!      forward to get a `VkTensor` whose `.grad_fn()` IS the leaf
//!      `VkBackwardOp`. Drive that backward TWO ways and assert the produced
//!      input grads are *byte-identical*:
//!
//!      - (a) directly: `grad_fn.backward(&grad_out_vk)`;
//!      - (b) through the adapter: wrap `grad_fn` in `VkBwdAdapter`, bridge
//!        `grad_out` to a kt `Tensor(VulkanStorage)`, call
//!        `BackwardOp::apply`, bridge each result back.
//!
//!      The adapter is a pure wrapper over the same kernel, so (a) and (b)
//!      MUST match to `max_abs_err == 0.0` per input grad. This isolates the
//!      PR3b kt<->vk bridge round-trip from kernel correctness, and proves the
//!      `None`-slot / input-order contract.
//!
//!   2. **FD oracle** (matmul / rms_norm / rope / softmax_lastdim — all four
//!      ported families). Central finite-difference of the forward vs the
//!      adapter-produced analytic grad on a TINY F32 tensor; `max_abs_err <
//!      1e-2` (FD is coarse). Confirms the backward KERNEL is actually
//!      *correct*, not merely consistent between the two call paths — an
//!      exact-vs-direct identity cannot catch a wrong-but-self-consistent
//!      kernel. The analytic side always comes through `VkBwdAdapter::apply`;
//!      the perturbed forward evals use a CPU oracle that mirrors the shader.
//!
//! Plus bounded contract tests: zero-copy buffer sharing (Arc strong_count > 1)
//!
//! - zero-copy buffer sharing (Arc strong_count > 1 + pointer identity)
//! - non-float-dtype rejection (I64 grad)
//! - `input_count` vs `backward()` Vec length
//! - non-Vulkan-grad rejection
//!
//! HOST-SAFETY: every test is a SINGLE bounded GPU dispatch over tiny shapes
//! (<= [4,5]@[5,3]). NO training loop, NO multi-step iteration. Each test
//! self-skips unless `KILN_TENSOR_VULKAN_TEST=1` AND a Vulkan device is
//! present (mirrors the PR2/PR3 `KILN_TENSOR_VULKAN_TEST` gating). Run ONE
//! named test at a time:
//!     KILN_TENSOR_VULKAN_TEST=1 CARGO_TARGET_DIR=/path/to/kiln/target \
//!       cargo test -p kiln-model --features vulkan \
//!       vk_bwd_adapter_matmul_exact_vs_direct -- --exact --nocapture

#![cfg(feature = "vulkan")]

use std::sync::Arc;

use anyhow::Result;

use kiln_autograd::BackwardOp;
use kiln_model::vk_bwd_adapter::VkBwdAdapter;
use kiln_tensor::{Device, Tensor};
use kiln_vulkan_kernel::VulkanDevice;
use kiln_vulkan_kernel::vk_tensor::{VkBackwardOp, VkDType, VkTensor};

// ----------------------------------------------------------------------------
// Gate + helpers.
// ----------------------------------------------------------------------------

/// Bounded GPU run is opt-in: `KILN_TENSOR_VULKAN_TEST=1` AND a device present.
/// Returns the device Arc, or `None` to self-skip cleanly.
fn vk_dev(test_name: &str) -> Option<Arc<VulkanDevice>> {
    if std::env::var("KILN_TENSOR_VULKAN_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip {test_name}: KILN_TENSOR_VULKAN_TEST unset");
        return None;
    }
    if !VulkanDevice::probe() {
        eprintln!("skip {test_name}: no Vulkan device");
        return None;
    }
    match VulkanDevice::new() {
        Ok(d) => Some(Arc::new(d)),
        Err(e) => {
            eprintln!("skip {test_name}: VulkanDevice::new failed: {e}");
            None
        }
    }
}

const DEV_IDX: usize = 0;

/// F32 `VkTensor` *parameter* leaf (requires_grad=true so the real forward
/// attaches a `grad_fn`). Uploads `data` to a fresh device-local buffer.
fn vk_param_f32(data: &[f32], shape: &[usize], dev: &Arc<VulkanDevice>) -> Result<VkTensor> {
    let leaf = VkTensor::from_f32_slice(data, shape.to_vec(), Arc::clone(dev))?;
    Ok(VkTensor::parameter(
        Arc::clone(leaf.buffer()),
        shape.to_vec(),
        VkDType::F32,
        Arc::clone(dev),
        kiln_vulkan_kernel::vk_tensor::TensorId::next(),
    ))
}

/// F32 `VkTensor` plain leaf (requires_grad=false) — for ancillary inputs
/// (cos/sin tables) that the backward closes over but does not differentiate.
fn vk_leaf_f32(data: &[f32], shape: &[usize], dev: &Arc<VulkanDevice>) -> Result<VkTensor> {
    VkTensor::from_f32_slice(data, shape.to_vec(), Arc::clone(dev))
}

/// Build a Vulkan-resident kt `Tensor` (grad_output the adapter consumes).
fn kt_vk_f32(data: &[f32], shape: &[usize]) -> Result<Tensor> {
    Ok(Tensor::from_vec_on(
        Device::Vulkan(DEV_IDX),
        data.to_vec(),
        shape.to_vec(),
    )?)
}

/// Read a kt Vulkan `Tensor` back to host f32.
fn kt_to_host(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_device(Device::Cpu)?.to_vec::<f32>()?)
}

fn max_abs_diff(got: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(
        got.len(),
        expected.len(),
        "len mismatch {} vs {}",
        got.len(),
        expected.len()
    );
    got.iter()
        .zip(expected.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0_f32, f32::max)
}

fn naive_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0_f32;
            for ki in 0..k {
                acc += a[mi * k + ki] * b[ki * n + ni];
            }
            c[mi * n + ni] = acc;
        }
    }
    c
}

/// Central finite-difference gradient of `loss(param)` w.r.t. each entry.
fn fd_grad(param: &mut [f32], eps: f32, mut loss: impl FnMut(&[f32]) -> f32) -> Vec<f32> {
    let mut grad = vec![0.0_f32; param.len()];
    for i in 0..param.len() {
        let orig = param[i];
        param[i] = orig + eps;
        let lp = loss(param);
        param[i] = orig - eps;
        let lm = loss(param);
        param[i] = orig;
        grad[i] = (lp - lm) / (2.0 * eps);
    }
    grad
}

// ----------------------------------------------------------------------------
// Core check: run a leaf VkBackwardOp two ways (direct + through the adapter)
// over the SAME grad_output, and assert each input-grad slot matches exactly.
//
// `grad_fn` is the actual `.grad_fn()` of a real forward output (so this is the
// production record path, not a hand-built struct). `grad_out_data`/`shape` are
// the upstream grad; the adapter consumes a kt Vulkan Tensor, the direct path
// consumes the bridged VkTensor — same bytes, same device buffer.
//
// Returns the direct-path host grads (Some slots) so the FD test can reuse them.
// ----------------------------------------------------------------------------
fn exact_vs_direct(
    grad_fn: &Arc<dyn VkBackwardOp>,
    grad_out_data: &[f32],
    grad_out_shape: &[usize],
) -> Result<Vec<Option<Vec<f32>>>> {
    // (a) DIRECT: call the leaf backward straight on a bridged VkTensor.
    let grad_out_kt = kt_vk_f32(grad_out_data, grad_out_shape)?;
    let grad_out_vk = kiln_tensor::vk_tensor_from_kt(&grad_out_kt)?;
    let direct = grad_fn.backward(&grad_out_vk)?;
    let direct_host: Vec<Option<Vec<f32>>> = direct
        .iter()
        .map(|o| o.as_ref().map(|v| v.to_vec_f32()).transpose())
        .collect::<Result<_, _>>()?;

    // (b) ADAPTER: wrap the SAME grad_fn, drive it through BackwardOp::apply
    // with a kt Vulkan grad_output. The adapter bridges in, runs the kernel,
    // bridges each result back to a kt Tensor(VulkanStorage).
    let adapter = VkBwdAdapter(Arc::clone(grad_fn));
    assert_eq!(
        adapter.input_count(),
        grad_fn.input_refs().len(),
        "adapter input_count must equal inner.input_refs().len()"
    );
    assert_eq!(
        adapter.name(),
        grad_fn.op_name(),
        "adapter name must equal op_name"
    );
    let via_adapter = adapter.apply(&grad_out_kt)?;
    let adapter_host: Vec<Option<Vec<f32>>> = via_adapter
        .iter()
        .map(|o| o.as_ref().map(kt_to_host).transpose())
        .collect::<Result<_, _>>()?;

    // Slot-for-slot exactness: same length, same None positions, byte-identical.
    assert_eq!(
        direct_host.len(),
        adapter_host.len(),
        "grad slot count differs: direct {} vs adapter {}",
        direct_host.len(),
        adapter_host.len()
    );
    for (i, (d, a)) in direct_host.iter().zip(adapter_host.iter()).enumerate() {
        match (d, a) {
            (None, None) => {}
            (Some(dv), Some(av)) => {
                let err = max_abs_diff(av, dv);
                eprintln!(
                    "[{}] slot {i}: exact-vs-direct max_abs_err = {err:e}",
                    grad_fn.op_name()
                );
                assert_eq!(
                    err,
                    0.0,
                    "[{}] slot {i}: adapter grad must be byte-identical to direct (got {err:e})",
                    grad_fn.op_name()
                );
            }
            (d, a) => panic!(
                "[{}] slot {i}: None-position mismatch direct={} adapter={}",
                grad_fn.op_name(),
                d.is_some(),
                a.is_some()
            ),
        }
    }
    Ok(direct_host)
}

// ============================================================================
// matmul — exact-vs-direct + FD.
// ============================================================================

#[test]
fn vk_bwd_adapter_matmul_exact_vs_direct() -> Result<()> {
    let Some(dev) = vk_dev("vk_bwd_adapter_matmul_exact_vs_direct") else {
        return Ok(());
    };
    let (m, k, n) = (4usize, 5usize, 3usize);
    let a_data: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.2 - 0.5).collect();

    let a = vk_param_f32(&a_data, &[m, k], &dev)?;
    let b = vk_param_f32(&b_data, &[k, n], &dev)?;
    // Real forward; the output's grad_fn IS MatmulBackward over [a, b].
    let out = kiln_vulkan_kernel::vk_ops::matmul::vk_matmul(&a, &b)?;
    let grad_fn = out
        .grad_fn()
        .expect("matmul forward must attach grad_fn")
        .clone();

    // grad_output dC = ones / (M*N)  (= d mean(A@B))
    let dc_data = vec![1.0_f32 / (m * n) as f32; m * n];
    let grads = exact_vs_direct(&grad_fn, &dc_data, &[m, n])?;
    assert_eq!(grads.len(), 2, "matmul backward returns 2 slots");
    assert!(
        grads[0].is_some() && grads[1].is_some(),
        "matmul: both grads present"
    );
    Ok(())
}

#[test]
fn vk_bwd_adapter_matmul_fd() -> Result<()> {
    let Some(dev) = vk_dev("vk_bwd_adapter_matmul_fd") else {
        return Ok(());
    };
    let (m, k, n) = (4usize, 5usize, 3usize);
    let a_data: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.2 - 0.5).collect();

    let a = vk_param_f32(&a_data, &[m, k], &dev)?;
    let b = vk_param_f32(&b_data, &[k, n], &dev)?;
    let out = kiln_vulkan_kernel::vk_ops::matmul::vk_matmul(&a, &b)?;
    let grad_fn = out.grad_fn().expect("matmul grad_fn").clone();

    // Analytic dA via the adapter, loss = mean(A@B), dC = ones/(M*N).
    let dc_data = vec![1.0_f32 / (m * n) as f32; m * n];
    let adapter = VkBwdAdapter(Arc::clone(&grad_fn));
    let grads = adapter.apply(&kt_vk_f32(&dc_data, &[m, n])?)?;
    let da = kt_to_host(grads[0].as_ref().expect("dA"))?;
    let db = kt_to_host(grads[1].as_ref().expect("dB"))?;

    // Central FD of the same scalar loss on the host (naive_matmul oracle).
    let loss_a = |ap: &[f32]| -> f32 {
        let c = naive_matmul(ap, &b_data, m, n, k);
        c.iter().sum::<f32>() / (m * n) as f32
    };
    let loss_b = |bp: &[f32]| -> f32 {
        let c = naive_matmul(&a_data, bp, m, n, k);
        c.iter().sum::<f32>() / (m * n) as f32
    };
    let mut a_mut = a_data.clone();
    let mut b_mut = b_data.clone();
    let fd_da = fd_grad(&mut a_mut, 1e-3, loss_a);
    let fd_db = fd_grad(&mut b_mut, 1e-3, loss_b);

    let err_a = max_abs_diff(&da, &fd_da);
    let err_b = max_abs_diff(&db, &fd_db);
    eprintln!("matmul FD: dA max_abs_err = {err_a:e}, dB max_abs_err = {err_b:e}");
    assert!(err_a < 1e-2, "dA FD mismatch {err_a:e} >= 1e-2");
    assert!(err_b < 1e-2, "dB FD mismatch {err_b:e} >= 1e-2");
    Ok(())
}

// ============================================================================
// rms_norm — exact-vs-direct ([Some, None] slot contract) + FD on dx.
// ============================================================================

#[test]
fn vk_bwd_adapter_rmsnorm_exact_vs_direct() -> Result<()> {
    let Some(dev) = vk_dev("vk_bwd_adapter_rmsnorm_exact_vs_direct") else {
        return Ok(());
    };
    let (rows, hidden) = (2usize, 4usize);
    let eps = 1e-6_f32;
    let x_data: Vec<f32> = (0..(rows * hidden))
        .map(|i| (i as f32) * 0.3 - 0.5)
        .collect();
    let w_data: Vec<f32> = vec![0.25, -0.5, 0.75, 1.0];

    let x = vk_param_f32(&x_data, &[rows, hidden], &dev)?;
    let w = vk_param_f32(&w_data, &[hidden], &dev)?;
    let out = kiln_vulkan_kernel::vk_ops::rmsnorm::vk_rmsnorm(&x, &w, eps)?;
    let grad_fn = out.grad_fn().expect("rmsnorm grad_fn").clone();

    let dy_data = vec![1.0_f32 / (rows * hidden) as f32; rows * hidden];
    let grads = exact_vs_direct(&grad_fn, &dy_data, &[rows, hidden])?;
    assert_eq!(
        grads.len(),
        2,
        "rmsnorm backward returns 2 slots [Some, None]"
    );
    assert!(grads[0].is_some(), "rmsnorm dx present (slot 0)");
    assert!(
        grads[1].is_none(),
        "rmsnorm frozen-weight grad MUST be None (slot 1)"
    );
    Ok(())
}

#[test]
fn vk_bwd_adapter_rmsnorm_fd() -> Result<()> {
    let Some(dev) = vk_dev("vk_bwd_adapter_rmsnorm_fd") else {
        return Ok(());
    };
    let (rows, hidden) = (2usize, 4usize);
    let eps = 1e-6_f32;
    let x_data: Vec<f32> = (0..(rows * hidden))
        .map(|i| (i as f32) * 0.3 - 0.5)
        .collect();
    let w_data: Vec<f32> = vec![0.25, -0.5, 0.75, 1.0];

    let x = vk_param_f32(&x_data, &[rows, hidden], &dev)?;
    let w = vk_param_f32(&w_data, &[hidden], &dev)?;
    let out = kiln_vulkan_kernel::vk_ops::rmsnorm::vk_rmsnorm(&x, &w, eps)?;
    let grad_fn = out.grad_fn().expect("rmsnorm grad_fn").clone();

    // dx via adapter, loss = mean(rmsnorm(x, w)), dy = ones/N.
    let dy_data = vec![1.0_f32 / (rows * hidden) as f32; rows * hidden];
    let adapter = VkBwdAdapter(Arc::clone(&grad_fn));
    let grads = adapter.apply(&kt_vk_f32(&dy_data, &[rows, hidden])?)?;
    let dx = kt_to_host(grads[0].as_ref().expect("dx"))?;

    // CPU rmsnorm oracle matching the shader: y = (1 + w) * x / sqrt(mean(x^2)+eps).
    let cpu_rmsnorm_mean = |xp: &[f32]| -> f32 {
        let mut acc = 0.0_f32;
        for r in 0..rows {
            let row = &xp[r * hidden..(r + 1) * hidden];
            let ms = row.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
            let inv = 1.0_f32 / (ms + eps).sqrt();
            for h in 0..hidden {
                acc += (1.0 + w_data[h]) * row[h] * inv;
            }
        }
        acc / (rows * hidden) as f32
    };
    let mut x_mut = x_data.clone();
    let fd_dx = fd_grad(&mut x_mut, 1e-3, cpu_rmsnorm_mean);

    let err = max_abs_diff(&dx, &fd_dx);
    eprintln!("rmsnorm FD: dx max_abs_err = {err:e}");
    assert!(err < 1e-2, "rmsnorm dx FD mismatch {err:e} >= 1e-2");
    Ok(())
}

/// CPU oracle of the `vk_rope_f32` forward (must match the shader exactly —
/// see csrc/shaders/vk_rope_f32.comp). Input `[rows, heads, head_dim]`; cos/sin
/// `[rows, half]`. Per rotary pair (i, i+half):
///   x0' = x0*cos - x1*sin ;  x1' = x0*sin + x1*cos ; tail (d >= rotary_dim) copies.
fn cpu_rope(
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
) -> Vec<f32> {
    let half = rotary_dim / 2;
    let mut out = vec![0.0_f32; rows * heads * head_dim];
    for r in 0..rows {
        for h in 0..heads {
            let base = (r * heads + h) * head_dim;
            for d in 0..head_dim {
                let lin = base + d;
                if d >= rotary_dim {
                    out[lin] = x[lin];
                    continue;
                }
                let (pair, is_low) = if d < half {
                    (d, true)
                } else {
                    (d - half, false)
                };
                let low = x[base + pair];
                let high = x[base + pair + half];
                let c = cos[r * half + pair];
                let s = sin[r * half + pair];
                out[lin] = if is_low {
                    low * c - high * s
                } else {
                    low * s + high * c
                };
            }
        }
    }
    out
}

/// CPU oracle of `vk_softmax_lastdim_f32` (numerically stable: max → exp → sum →
/// divide), matching csrc/shaders/vk_softmax_lastdim_f32.comp.
fn cpu_softmax_lastdim(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for (j, &v) in row.iter().enumerate() {
            let e = (v - m).exp();
            out[r * cols + j] = e;
            sum += e;
        }
        for j in 0..cols {
            out[r * cols + j] /= sum;
        }
    }
    out
}

// ============================================================================
// rope — exact-vs-direct (single input) + central FD on dx.
// ============================================================================

#[test]
fn vk_bwd_adapter_rope_exact_vs_direct() -> Result<()> {
    let Some(dev) = vk_dev("vk_bwd_adapter_rope_exact_vs_direct") else {
        return Ok(());
    };
    let (rows, heads, head_dim) = (2usize, 1usize, 4usize);
    let rotary_dim = 4usize;
    let half = rotary_dim / 2;
    let total = rows * heads * head_dim;
    let x_data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.2 - 0.4).collect();
    // Arbitrary but valid cos/sin tables, shape [rows, half].
    let cos_data: Vec<f32> = (0..(rows * half)).map(|i| (0.3 * i as f32).cos()).collect();
    let sin_data: Vec<f32> = (0..(rows * half)).map(|i| (0.3 * i as f32).sin()).collect();

    let x = vk_param_f32(&x_data, &[rows, heads, head_dim], &dev)?;
    let cos = vk_leaf_f32(&cos_data, &[rows, half], &dev)?;
    let sin = vk_leaf_f32(&sin_data, &[rows, half], &dev)?;
    let out = kiln_vulkan_kernel::vk_ops::rope::vk_rope(&x, &cos, &sin, rotary_dim)?;
    let grad_fn = out.grad_fn().expect("rope grad_fn").clone();

    let dy_data = vec![1.0_f32 / total as f32; total];
    let grads = exact_vs_direct(&grad_fn, &dy_data, &[rows, heads, head_dim])?;
    assert_eq!(grads.len(), 1, "rope backward returns 1 slot");
    assert!(grads[0].is_some(), "rope dx present");
    Ok(())
}

#[test]
fn vk_bwd_adapter_rope_fd() -> Result<()> {
    let Some(dev) = vk_dev("vk_bwd_adapter_rope_fd") else {
        return Ok(());
    };
    let (rows, heads, head_dim) = (2usize, 1usize, 4usize);
    let rotary_dim = 4usize;
    let half = rotary_dim / 2;
    let total = rows * heads * head_dim;
    let x_data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.2 - 0.4).collect();
    let cos_data: Vec<f32> = (0..(rows * half)).map(|i| (0.3 * i as f32).cos()).collect();
    let sin_data: Vec<f32> = (0..(rows * half)).map(|i| (0.3 * i as f32).sin()).collect();

    let x = vk_param_f32(&x_data, &[rows, heads, head_dim], &dev)?;
    let cos = vk_leaf_f32(&cos_data, &[rows, half], &dev)?;
    let sin = vk_leaf_f32(&sin_data, &[rows, half], &dev)?;
    let out = kiln_vulkan_kernel::vk_ops::rope::vk_rope(&x, &cos, &sin, rotary_dim)?;
    let grad_fn = out.grad_fn().expect("rope grad_fn").clone();

    // Analytic dx via the adapter. Loss = sum(w_i * y_i) with a non-uniform
    // linear combo w (= dy), so the gradient is non-trivial in every slot.
    let dy_data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.05 - 0.1).collect();
    let adapter = VkBwdAdapter(Arc::clone(&grad_fn));
    let grads = adapter.apply(&kt_vk_f32(&dy_data, &[rows, heads, head_dim])?)?;
    let dx = kt_to_host(grads[0].as_ref().expect("rope dx"))?;

    // Central FD of the SAME scalar loss on the CPU rope oracle.
    let loss = |xp: &[f32]| -> f32 {
        let y = cpu_rope(xp, &cos_data, &sin_data, rows, heads, head_dim, rotary_dim);
        y.iter()
            .zip(dy_data.iter())
            .map(|(yi, wi)| yi * wi)
            .sum::<f32>()
    };
    let mut x_mut = x_data.clone();
    let fd_dx = fd_grad(&mut x_mut, 1e-3, loss);

    let err = max_abs_diff(&dx, &fd_dx);
    eprintln!("rope FD: dx max_abs_err = {err:e}");
    assert!(err < 1e-2, "rope dx FD mismatch {err:e} >= 1e-2");
    Ok(())
}

// ============================================================================
// softmax_lastdim — exact-vs-direct (single input) + central FD on dx.
// ============================================================================

#[test]
fn vk_bwd_adapter_softmax_exact_vs_direct() -> Result<()> {
    let Some(dev) = vk_dev("vk_bwd_adapter_softmax_exact_vs_direct") else {
        return Ok(());
    };
    let (rows, cols) = (3usize, 4usize);
    let x_data: Vec<f32> = (0..(rows * cols))
        .map(|i| (i as f32) * 0.17 - 0.6)
        .collect();

    let x = vk_param_f32(&x_data, &[rows, cols], &dev)?;
    let out = kiln_vulkan_kernel::vk_ops::softmax::vk_softmax_lastdim(&x)?;
    let grad_fn = out.grad_fn().expect("softmax grad_fn").clone();

    let dy_data: Vec<f32> = (0..(rows * cols))
        .map(|i| (i as f32) * 0.05 - 0.1)
        .collect();
    let grads = exact_vs_direct(&grad_fn, &dy_data, &[rows, cols])?;
    assert_eq!(grads.len(), 1, "softmax backward returns 1 slot");
    assert!(grads[0].is_some(), "softmax dx present");
    Ok(())
}

#[test]
fn vk_bwd_adapter_softmax_fd() -> Result<()> {
    let Some(dev) = vk_dev("vk_bwd_adapter_softmax_fd") else {
        return Ok(());
    };
    let (rows, cols) = (3usize, 4usize);
    let x_data: Vec<f32> = (0..(rows * cols))
        .map(|i| (i as f32) * 0.17 - 0.6)
        .collect();

    let x = vk_param_f32(&x_data, &[rows, cols], &dev)?;
    let out = kiln_vulkan_kernel::vk_ops::softmax::vk_softmax_lastdim(&x)?;
    let grad_fn = out.grad_fn().expect("softmax grad_fn").clone();

    // Analytic dx via the adapter. Loss = sum(w_i * y_i) with a NON-uniform w
    // (= dy): a uniform dy would give a zero gradient (softmax rows sum to 1),
    // so the linear combo must vary across columns to exercise the kernel.
    let dy_data: Vec<f32> = (0..(rows * cols))
        .map(|i| (i as f32) * 0.05 - 0.1)
        .collect();
    let adapter = VkBwdAdapter(Arc::clone(&grad_fn));
    let grads = adapter.apply(&kt_vk_f32(&dy_data, &[rows, cols])?)?;
    let dx = kt_to_host(grads[0].as_ref().expect("softmax dx"))?;

    // Central FD of the SAME scalar loss on the CPU softmax oracle.
    let loss = |xp: &[f32]| -> f32 {
        let y = cpu_softmax_lastdim(xp, rows, cols);
        y.iter()
            .zip(dy_data.iter())
            .map(|(yi, wi)| yi * wi)
            .sum::<f32>()
    };
    let mut x_mut = x_data.clone();
    let fd_dx = fd_grad(&mut x_mut, 1e-3, loss);

    let err = max_abs_diff(&dx, &fd_dx);
    eprintln!("softmax FD: dx max_abs_err = {err:e}");
    assert!(err < 1e-2, "softmax dx FD mismatch {err:e} >= 1e-2");
    Ok(())
}

// ============================================================================
// Zero-copy: bridging a kt Vulkan grad to a VkTensor SHARES the underlying
// VulkanBuffer (Arc refcount bump, no host copy). Proven by strong_count > 1
// while the source kt Tensor is still alive, and by pointer identity.
// ============================================================================

#[test]
fn vk_bwd_adapter_zero_copy_shares_buffer() -> Result<()> {
    let Some(_dev) = vk_dev("vk_bwd_adapter_zero_copy_shares_buffer") else {
        return Ok(());
    };
    // A kt Vulkan-resident grad tensor (the kind apply() consumes).
    let grad = kt_vk_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;

    // The kt storage's own Arc<VulkanBuffer> (one strong ref lives in `grad`).
    let kt_storage = grad
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::VulkanStorage>()
        .expect("grad must be Vulkan-backed");
    let kt_buf_ptr = std::sync::Arc::as_ptr(&kt_storage.buffer_arc());

    // Bridge to a VkTensor — this must Arc::clone the SAME buffer, not copy.
    let vk_grad = kiln_tensor::vk_tensor_from_kt(&grad)?;
    let vk_buf = vk_grad.buffer();

    // Same allocation (pointer identity) → genuinely zero-copy.
    assert_eq!(
        std::sync::Arc::as_ptr(vk_buf),
        kt_buf_ptr,
        "vk_tensor_from_kt must share the kt storage's VulkanBuffer (zero-copy)"
    );
    // strong_count > 1: the kt storage holds one ref, the bridged VkTensor
    // holds another. (>=2; other transient clones may push it higher.)
    let count = std::sync::Arc::strong_count(vk_buf);
    eprintln!("zero_copy: Arc::strong_count(buffer) = {count}");
    assert!(
        count > 1,
        "bridged VkTensor must share (not copy) the buffer: strong_count={count} (expected > 1)"
    );
    drop(grad);
    Ok(())
}

// ============================================================================
// dtype rejection: a non-F32/BF16 (I64 index-style) Vulkan grad must be
// rejected by the bridge, never silently mis-bridged as F32.
// ============================================================================

#[test]
fn vk_bwd_adapter_rejects_non_float_dtype() -> Result<()> {
    let Some(_dev) = vk_dev("vk_bwd_adapter_rejects_non_float_dtype") else {
        return Ok(());
    };
    // An I64 (index/id) tensor placed on the Vulkan device: kt_dtype_to_vk
    // returns None for I64, so vk_tensor_from_kt must error rather than
    // reinterpret the 8-byte ints as F32.
    let idx_grad = Tensor::from_vec_on(Device::Vulkan(DEV_IDX), vec![1_i64, 2, 3, 4], vec![2, 2])?;
    let res = kiln_tensor::vk_tensor_from_kt(&idx_grad);
    assert!(
        res.is_err(),
        "I64 Vulkan grad must be rejected by vk_tensor_from_kt (F32/BF16 only)"
    );
    Ok(())
}

// ============================================================================
// input_count() contract: the adapter's input_count matches the Vec length the
// underlying family's backward() returns (so the tape walker binds every slot).
// ============================================================================

#[test]
fn vk_bwd_adapter_input_count_matches_backward_len() -> Result<()> {
    let Some(dev) = vk_dev("vk_bwd_adapter_input_count_matches_backward_len") else {
        return Ok(());
    };
    // matmul: 2 inputs (a, b) → backward returns 2 grad slots.
    let (m, k, n) = (2usize, 2usize, 2usize);
    let a = vk_param_f32(&[1.0, 2.0, 3.0, 4.0], &[m, k], &dev)?;
    let b = vk_param_f32(&[1.0, 0.0, 0.0, 1.0], &[k, n], &dev)?;
    let out = kiln_vulkan_kernel::vk_ops::matmul::vk_matmul(&a, &b)?;
    let grad_fn = out.grad_fn().expect("matmul grad_fn").clone();
    let adapter = VkBwdAdapter(Arc::clone(&grad_fn));

    let dc = kt_vk_f32(&vec![1.0_f32 / (m * n) as f32; m * n], &[m, n])?;
    let grads = adapter.apply(&dc)?;
    assert_eq!(
        adapter.input_count(),
        grads.len(),
        "input_count {} must equal backward() Vec length {}",
        adapter.input_count(),
        grads.len()
    );
    assert_eq!(adapter.input_count(), 2, "matmul has 2 inputs");
    Ok(())
}

// ============================================================================
// Adapter rejects a non-Vulkan grad (loud failure, no silent host bounce).
// ============================================================================

#[test]
fn vk_bwd_adapter_rejects_non_vulkan_grad() -> Result<()> {
    let Some(dev) = vk_dev("vk_bwd_adapter_rejects_non_vulkan_grad") else {
        return Ok(());
    };
    let (m, k, n) = (2usize, 2usize, 2usize);
    let a = vk_param_f32(&[1.0, 2.0, 3.0, 4.0], &[m, k], &dev)?;
    let b = vk_param_f32(&[1.0, 0.0, 0.0, 1.0], &[k, n], &dev)?;
    let out = kiln_vulkan_kernel::vk_ops::matmul::vk_matmul(&a, &b)?;
    let grad_fn = out.grad_fn().expect("matmul grad_fn").clone();
    let adapter = VkBwdAdapter(Arc::clone(&grad_fn));

    // CPU grad — the adapter's vk_tensor_from_kt downcast to VulkanStorage fails.
    let cpu_grad = Tensor::from_vec(vec![1.0_f32; m * n], vec![m, n])?;
    let res = adapter.apply(&cpu_grad);
    assert!(
        res.is_err(),
        "CPU grad must be rejected (not Vulkan-backed)"
    );
    Ok(())
}
