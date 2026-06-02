//! PR4 — `VkBwdAdapter` finite-difference / parity / reachability test scaffold.
//!
//! ============================ WIP — DO NOT COMPILE YET ============================
//!
//! This file is a DROP-IN SCAFFOLD for the implementer of PR4 (#1082). It is
//! intentionally NOT wired into any crate's `tests/` yet, because it references
//! the not-yet-written `kiln_model::vk_bwd_adapter::VkBwdAdapter` and the PR1.1
//! Cargo activations (`kiln-model` `vulkan` feature must add `dep:kiln-autograd`
//! + `kiln-tensor/vulkan` — see PR4-spec.md §1.1).
//!
//! When PR4 lands, move this to:
//!     crates/kiln-model/tests/vk_bwd_adapter_parity.rs
//! and DELETE the file-level `#![cfg(any())]` guard below (it exists purely so a
//! stray `cargo test` cannot try to build this scaffold against a tree where the
//! adapter does not exist yet). Every test additionally guards on
//! `VulkanDevice::probe()` and returns early when no GPU is present, so it skips
//! cleanly on CI / dev hosts without Vulkan.
//!
//! HOST-SAFETY: every test here is a SINGLE bounded GPU dispatch over tiny
//! (<= [17,33]) shapes. There is NO training loop, NO multi-step iteration, NO
//! long-running binary. Run ONE named test at a time:
//!     CARGO_TARGET_DIR=/path/to/kiln/target \
//!       cargo test -p kiln-model --features vulkan \
//!       vk_bwd_adapter_matmul_fd_parity -- --exact --nocapture
//! Do NOT run the whole suite on the crash-prone dev host.
//!
//! The finite-difference harness (`fd_grad`, `naive_matmul`, `max_abs_diff`) is
//! copied verbatim from the PROVEN, shipping
//! `crates/kiln-vulkan-kernel/tests/vk_matmul_parity.rs` so the numerics are
//! already trusted. Acceptance thresholds mirror the Metal OPD / §9.2 gate:
//! analytic-comparable F32 ops at `max_abs_err < 1e-5`; FD-perturbed smooth ops
//! at `< 2e-3` to absorb central-difference truncation.

// ----------------------------------------------------------------------------
// File-level WIP guard. `any()` is always false → the whole module is cfg'd out
// so an accidental build is a no-op. DELETE THIS LINE when wiring into the crate.
// ----------------------------------------------------------------------------
#![cfg(any())]
#![cfg(feature = "vulkan")]

use std::sync::Arc;

use anyhow::Result;

// --- shared substrate ---
use kiln_autograd::BackwardOp;
use kiln_tensor::{DType, Device, Tensor};

// --- vk leaf kernels + the PR4 adapter under test ---
use kiln_model::vk_bwd_adapter::VkBwdAdapter; // <-- NEW in PR4 (does not exist yet)
use kiln_vulkan_kernel::vk_tensor::{VkBackwardOp, VkTensor};
use kiln_vulkan_kernel::VulkanDevice;

// ----------------------------------------------------------------------------
// Device probe + tiny-tensor helpers (mirror vk_matmul_parity.rs:26).
// ----------------------------------------------------------------------------

fn vk_dev() -> Option<Arc<VulkanDevice>> {
    if !VulkanDevice::probe() {
        return None;
    }
    VulkanDevice::new().ok().map(Arc::new)
}

/// Build a contiguous F32 `Tensor` resident on `Device::Vulkan(0)`.
/// `Tensor::from_vec(values, shape)` infers F32 from the element type
/// (tensor.rs:138 — NOTE: 2 args, dtype is from `E`, not a 3rd arg), then
/// `to_device(Device::Vulkan(0))` uses PR2's un-NYI'd Vulkan arm
/// (tensor.rs:921 `host_to_vulkan_copy`).
fn vk_tensor_f32(data: &[f32], shape: &[usize]) -> Result<Tensor> {
    let host = Tensor::from_vec(data.to_vec(), shape.to_vec())?;
    Ok(host.to_device(Device::Vulkan(0))?)
}

/// Read a Vulkan `Tensor` back to a flat host `Vec<f32>`.
/// IMPLEMENTER: kt `Tensor` has no `to_vec_f32` (that is a `VkTensor` method).
/// Use the project's standard host-readback after `to_device(Cpu)` — e.g. the
/// `CpuStorage` downcast + `as_bytes()`/`bytemuck::cast_slice` pattern used at
/// `crates/kiln-tensor/src/tensor.rs:659` (grep `downcast_ref::<CpuStorage>`),
/// or whatever flat-f32 accessor kt exposes at PR4 time. Left as a TODO so the
/// scaffold does not cite a non-existent method.
fn to_host_f32(_t: &Tensor) -> Result<Vec<f32>> {
    todo!("readback via CpuStorage downcast after to_device(Cpu) — see tensor.rs:659")
}

// ----------------------------------------------------------------------------
// Finite-difference harness — copied verbatim from vk_matmul_parity.rs.
// ----------------------------------------------------------------------------

fn max_abs_diff(got: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(got.len(), expected.len(), "len mismatch");
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
/// eps ~ 1e-3 for f32. (vk_matmul_parity.rs:109)
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
// Helper: build a `VkBwdAdapter` over a hand-constructed `VkBackwardOp`.
//
// This is the PR4-ONLY checkable unit — it does NOT need PR5's recorders.
// We construct the leaf VkBackwardOp directly (e.g. `MatmulBackward { inputs }`),
// wrap it, and drive `apply()` with a Vulkan-resident grad tensor.
// ----------------------------------------------------------------------------

fn adapter_for(inner: Arc<dyn VkBackwardOp>, device_index: usize) -> VkBwdAdapter {
    VkBwdAdapter::new(inner, device_index)
}

/// Wrap a Vulkan `Tensor`'s storage as a `VkTensor` (mirrors the adapter's
/// internal `tensor_to_vk`; duplicated here so the test can hand the leaf
/// VkBackwardOp its saved inputs without depending on adapter internals).
fn tensor_to_vk_test(t: &Tensor) -> Result<VkTensor> {
    use kiln_tensor::VulkanStorage;
    use kiln_vulkan_kernel::vk_tensor::VkDType;
    let t = t.contiguous()?;
    let vs = t
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .expect("vulkan-backed");
    let dt = match vs.dtype() {
        DType::F32 => VkDType::F32,
        DType::BF16 => VkDType::Bf16,
        other => panic!("unsupported dtype {other:?}"),
    };
    Ok(VkTensor::from_buffer(
        vs.buffer_arc(),
        t.shape().to_vec(),
        dt,
        Arc::clone(vs.vulkan_device()),
    ))
}

// ============================================================================
// Wave 1 — matmul (analytic + FD, tightest tolerance).
// ============================================================================

/// `apply(grad_output)` of the matmul backward, wrapped by `VkBwdAdapter`,
/// must reproduce the analytic `dA = dC @ B.T`, `dB = A.T @ dC` to 1e-5 (F32).
#[test]
fn vk_bwd_adapter_matmul_fd_parity() -> Result<()> {
    let Some(_dev) = vk_dev() else {
        eprintln!("no Vulkan device; skipping vk_bwd_adapter_matmul_fd_parity");
        return Ok(());
    };
    let (m, k, n) = (4usize, 5usize, 3usize);
    let a_data: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.2 - 0.5).collect();

    let a = vk_tensor_f32(&a_data, &[m, k])?;
    let b = vk_tensor_f32(&b_data, &[k, n])?;

    // Hand-build the leaf MatmulBackward over the two saved inputs.
    use kiln_vulkan_kernel::vk_ops::matmul::MatmulBackward;
    let inner = Arc::new(MatmulBackward {
        inputs: [tensor_to_vk_test(&a)?, tensor_to_vk_test(&b)?],
    }) as Arc<dyn VkBackwardOp>;
    let adapter = adapter_for(inner, 0);

    // grad_output dC = ones / (M*N), shape [M, N] — i.e. d mean(A@B).
    let n_total = (m * n) as f32;
    let dc_data = vec![1.0_f32 / n_total; m * n];
    let dc = vk_tensor_f32(&dc_data, &[m, n])?;

    // adapter contract: returns Vec<Option<Tensor>> of len input_count()==2.
    let grads = adapter.apply(&dc)?;
    assert_eq!(grads.len(), 2, "matmul backward must return 2 grad slots");
    let da = to_host_f32(grads[0].as_ref().expect("dA present"))?;
    let db = to_host_f32(grads[1].as_ref().expect("dB present"))?;

    // analytic reference
    let mut b_t = vec![0.0; n * k];
    for ki in 0..k {
        for ni in 0..n {
            b_t[ni * k + ki] = b_data[ki * n + ni];
        }
    }
    let exp_da = naive_matmul(&dc_data, &b_t, m, k, n);
    let mut a_t = vec![0.0; k * m];
    for mi in 0..m {
        for ki in 0..k {
            a_t[ki * m + mi] = a_data[mi * k + ki];
        }
    }
    let exp_db = naive_matmul(&a_t, &dc_data, k, n, m);

    assert!(max_abs_diff(&da, &exp_da) < 1e-5, "dA mad");
    assert!(max_abs_diff(&db, &exp_db) < 1e-5, "dB mad");
    Ok(())
}

// ============================================================================
// Wave 1 — rmsnorm: assert the frozen-weight slot is `None` (slot-order contract).
// ============================================================================

#[test]
fn vk_bwd_adapter_preserves_none_slots() -> Result<()> {
    let Some(_dev) = vk_dev() else {
        eprintln!("no Vulkan device; skipping vk_bwd_adapter_preserves_none_slots");
        return Ok(());
    };
    let (rows, hidden) = (2usize, 4usize);
    let x_data: Vec<f32> = (0..(rows * hidden)).map(|i| (i as f32) * 0.3 - 0.5).collect();
    let w_data: Vec<f32> = vec![1.0; hidden];
    let x = vk_tensor_f32(&x_data, &[rows, hidden])?;
    let w = vk_tensor_f32(&w_data, &[hidden])?;

    use kiln_vulkan_kernel::vk_ops::rmsnorm::RmsNormBackward;
    let inner = Arc::new(RmsNormBackward {
        eps: 1e-6,
        inputs: [tensor_to_vk_test(&x)?, tensor_to_vk_test(&w)?],
    }) as Arc<dyn VkBackwardOp>;
    let adapter = adapter_for(inner, 0);

    let dy_data = vec![1.0_f32 / (rows * hidden) as f32; rows * hidden];
    let dy = vk_tensor_f32(&dy_data, &[rows, hidden])?;

    let grads = adapter.apply(&dy)?;
    assert_eq!(grads.len(), 2, "rmsnorm backward returns 2 slots [Some, None]");
    assert!(grads[0].is_some(), "dx present");
    assert!(grads[1].is_none(), "frozen weight grad MUST be None (slot 1)");
    Ok(())
}

// ============================================================================
// Adapter contract: input_count() == inner.input_refs().len().
// ============================================================================

#[test]
fn vk_bwd_adapter_input_count_matches() -> Result<()> {
    let Some(_dev) = vk_dev() else {
        eprintln!("no Vulkan device; skipping vk_bwd_adapter_input_count_matches");
        return Ok(());
    };
    let a = vk_tensor_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let b = vk_tensor_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2])?;
    use kiln_vulkan_kernel::vk_ops::matmul::MatmulBackward;
    let inner = Arc::new(MatmulBackward {
        inputs: [tensor_to_vk_test(&a)?, tensor_to_vk_test(&b)?],
    }) as Arc<dyn VkBackwardOp>;
    let adapter = adapter_for(Arc::clone(&inner) as Arc<dyn VkBackwardOp>, 0);
    assert_eq!(adapter.input_count(), inner.input_refs().len());
    assert_eq!(adapter.name(), inner.op_name());
    Ok(())
}

// ============================================================================
// Adapter rejects non-Vulkan grad storage (loud failure, no silent host bounce).
// ============================================================================

#[test]
fn vk_bwd_adapter_rejects_non_vulkan_grad() -> Result<()> {
    let Some(_dev) = vk_dev() else {
        eprintln!("no Vulkan device; skipping vk_bwd_adapter_rejects_non_vulkan_grad");
        return Ok(());
    };
    let a = vk_tensor_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let b = vk_tensor_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2])?;
    use kiln_vulkan_kernel::vk_ops::matmul::MatmulBackward;
    let inner = Arc::new(MatmulBackward {
        inputs: [tensor_to_vk_test(&a)?, tensor_to_vk_test(&b)?],
    }) as Arc<dyn VkBackwardOp>;
    let adapter = adapter_for(inner, 0);

    // CPU grad — adapter's downcast to VulkanStorage must fail.
    let cpu_grad = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2], DType::F32)?;
    let err = adapter.apply(&cpu_grad);
    assert!(err.is_err(), "CPU grad must be rejected (not Vulkan-backed)");
    Ok(())
}

// ============================================================================
// Zero-copy guard: the rewrapped grad must SHARE the kernel result's buffer.
//
// A host bounce (read_back + upload) would mint a fresh Arc<VulkanBuffer>.
// We can't reach into the adapter's internals, so this test instead asserts
// the *forward* of a matmul (which the adapter shares) keeps strong_count > 1
// when both a VkTensor and a kt Tensor wrap the same Arc. Implementer: tighten
// this to inspect the adapter's actual output Arc once a test hook exists.
// ============================================================================

#[test]
fn vk_bwd_adapter_zero_copy_shares_buffer() -> Result<()> {
    let Some(_dev) = vk_dev() else {
        eprintln!("no Vulkan device; skipping vk_bwd_adapter_zero_copy_shares_buffer");
        return Ok(());
    };
    // Build a VkTensor leaf, wrap its buffer into a kt VulkanStorage Tensor via
    // the same from_arc_buffer path the adapter uses, and assert the Arc is
    // shared (strong_count >= 2), proving no device copy happened.
    use kiln_tensor::VulkanStorage;
    let vt = VkTensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2], _dev_clone()?)?;
    let buf = Arc::clone(vt.buffer());
    let before = Arc::strong_count(&buf);
    let storage = VulkanStorage::from_arc_buffer(
        Arc::clone(vt.device()),
        0,
        DType::F32,
        Arc::clone(vt.buffer()),
        vt.byte_size() as u64,
    )?;
    let _t = Tensor::from_parts(
        Arc::new(storage),
        kiln_tensor::Layout::contiguous(vec![2, 2]),
        kiln_tensor::TensorId::next(),
    )?;
    let after = Arc::strong_count(&buf);
    assert!(after > before, "from_arc_buffer must share (no copy): {before} -> {after}");
    Ok(())
}

/// Local device handle for the zero-copy test (the others probe in-test).
fn _dev_clone() -> Result<Arc<VulkanDevice>> {
    Ok(VulkanDevice::new().map(Arc::new).expect("device"))
}

// ============================================================================
// Wave 1 — rmsnorm / rope / softmax FD parity (templates; fill loss closures).
//
// Each: build x on Vulkan, run the leaf forward+backward via the adapter,
// compare dx against a CENTRAL FINITE-DIFFERENCE gradient of the same scalar
// loss. Tolerance 2e-3 (absorbs FD truncation; the ops are smooth).
//
// These are stubbed with `#[ignore]` because they additionally require the
// PR5 forward recorder (or a hand-rolled CPU reference for the forward) to
// produce the scalar loss the FD perturbs. The matmul test above is the
// PR4-only-checkable one (analytic oracle, no forward recorder needed).
// ============================================================================

#[test]
#[ignore = "PR4 WIP: needs the rmsnorm forward CPU reference / PR5 recorder for the FD loss"]
fn vk_bwd_adapter_rmsnorm_fd_parity() -> Result<()> {
    // 1. x = vk_tensor_f32(...); w = ones.
    // 2. loss(x) = mean( rmsnorm(x, w, eps) ) computed on CPU as the FD oracle.
    // 3. dx_adapter = VkBwdAdapter(RmsNormBackward).apply(ones/N)[0].
    // 4. dx_fd = fd_grad(&mut x_data, 1e-3, |xp| cpu_rmsnorm_mean(xp, ...)).
    // 5. assert max_abs_diff(dx_adapter, dx_fd) < 2e-3.
    Ok(())
}

#[test]
#[ignore = "PR4 WIP: needs the rope forward CPU reference for the FD loss"]
fn vk_bwd_adapter_rope_fd_parity() -> Result<()> {
    Ok(())
}

#[test]
#[ignore = "PR4 WIP: needs the softmax forward CPU reference for the FD loss"]
fn vk_bwd_adapter_softmax_fd_parity() -> Result<()> {
    Ok(())
}

// ============================================================================
// Wave 2 — opd / flce / gdn (ported later; mirror the OPD reverse-KL composite
// gate at 1e-5/1e-4 from vk_cuda_opd_parity.rs and the GDN FD harness at
// vk_gdn_backward_parity.rs). Stubbed ignored until those families are ported.
// ============================================================================

#[test]
#[ignore = "PR4 Wave 2 WIP: port OpdLossBackward adapter; mirror vk_cuda_opd_parity 1e-5/1e-4 gate"]
fn vk_bwd_adapter_opd_fd_parity() -> Result<()> {
    Ok(())
}

#[test]
#[ignore = "PR4 Wave 2 WIP: port FlceBackward adapter (host metadata already in struct)"]
fn vk_bwd_adapter_flce_fd_parity() -> Result<()> {
    Ok(())
}

#[test]
#[ignore = "PR4 Wave 2 WIP: port GdnChunkwiseBackward (5 inputs); reuse vk_gdn_backward_parity FD harness"]
fn vk_bwd_adapter_gdn_chunkwise_fd_parity() -> Result<()> {
    Ok(())
}
