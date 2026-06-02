//! PR5b — record-and-backprop proof on the real Vulkan GPU (#1082).
//!
//! KEYSTONE MILESTONE of the vk-tape harmonization. PR5a widened the
//! `tape_forward` module gate + the 16 recorder device gates to admit
//! `Device::Vulkan(_)`. This test proves the consequence the whole
//! harmonization rests on: **a kt-tape forward RECORDS onto
//! `kiln_autograd::Tape` on `Device::Vulkan`, and `Tape::backward` produces
//! CORRECT gradients over Vulkan storage** (matching `Device::Cpu`).
//!
//! ## Op choice — why `add` is the keystone proof (host-safety-driven)
//! The smallest recorded op that exercises the full record→backprop loop end
//! to end on Vulkan WITHOUT a known-faulting native kernel is the elementwise
//! `add` recorder (`try_tape_add_kt`):
//!   - **Forward** dispatches `kiln_tensor::ops::add`, whose `vulkan_fwd` runs
//!     the production `vk_add_no_grad` shader over Vulkan-resident buffers
//!     (PR3 op-coverage).
//!   - **Backward** is `kiln_autograd::AddBackward`, a pure passthrough
//!     (`dL/da = dL/db = grad_output`) — no `contiguous()`, `transpose()`, or
//!     reduction, so it cannot trip the two PR3 frontier gaps documented below.
//!   - This makes the backward analytically checkable: with `dL/dy = ones`,
//!     both leaf grads must equal `ones`, on-device.
//!
//! `add` is a device-agnostic ("covered for free") recorder, so this proves
//! the *substrate* claim — forward records + backward runs on Vulkan storage.
//! A second test (`vk_tape_rms_norm_records_on_vulkan`, `#[ignore]`d) targets a
//! PR5a-GATED recorder to prove the gate widening admits Vulkan, and documents
//! the live PR3 frontier gap (see below).
//!
//! ## PR3 frontier gaps surfaced by this work (NOT PR5a gaps — honest record)
//!   1. **rms_norm native `vulkan_fwd` GPUVM-faults via the kt bridge.** Calling
//!      `try_tape_rms_norm_kt` on `Device::Vulkan(0)` runs `RmsNormOp::vulkan_fwd`
//!      → `vulkan_rmsnorm_last_axis` → `vk_rmsnorm_no_grad`, which triggered a
//!      `radv` GPUVM write fault + context loss on Strix Halo (hidden=8, rows=4).
//!      The recorder DID admit Vulkan (it ran the forward, did NOT return None),
//!      so PR5a's gate is correct; the fault is downstream in the PR3 native
//!      rmsnorm bridge/kernel path. Tracked as a PR3 follow-up.
//!   2. **`Tensor::contiguous()` is unimplemented for `Device::Vulkan`.**
//!      `MatmulBackward::apply` (and other composites) materialize a transposed
//!      saved tensor via `.transpose(..).contiguous()`; on Vulkan that returns
//!      `Err("only CPU + CUDA contiguous is implemented")`. So the matmul
//!      backward cannot yet run on Vulkan storage. Tracked as a PR3 follow-up.
//!
//! HOST-SAFETY (the host has hard-crashed on long GPU runs): the live test is a
//! SINGLE bounded forward+backward over a TINY tensor (rows=2, hidden=3). NO
//! training loop, NO multi-step iteration, NO checkpoint load. It self-skips
//! unless `KILN_TENSOR_VULKAN_TEST=1` AND a Vulkan device is present (mirrors
//! PR2/PR3/PR4 gating). Run named, single-shot:
//!
//!     KILN_TENSOR_VULKAN_TEST=1 CARGO_TARGET_DIR=/path/to/kiln/target \
//!       cargo test -p kiln-model --features vulkan \
//!       vk_tape_add_records_and_backprops -- --nocapture --test-threads=1

#![cfg(feature = "vulkan")]

use kiln_model::tape_forward::{
    try_tape_add_kt, try_tape_rms_norm_kt, with_thread_local_tape,
};
use kiln_tensor::{ops, Device, Tensor};
use kiln_vulkan_kernel::VulkanDevice;

// ----------------------------------------------------------------------------
// Gate + tiny fixtures.
// ----------------------------------------------------------------------------

/// Bounded GPU run is opt-in: `KILN_TENSOR_VULKAN_TEST=1` AND a device present.
fn vk_enabled(test_name: &str) -> bool {
    if std::env::var("KILN_TENSOR_VULKAN_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip {test_name}: KILN_TENSOR_VULKAN_TEST unset");
        return false;
    }
    if !VulkanDevice::probe() {
        eprintln!("skip {test_name}: no Vulkan device");
        return false;
    }
    true
}

const ROWS: usize = 2;
const COLS: usize = 3;

/// Max absolute elementwise error between two equal-length F32 vecs.
fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch in max_abs_err");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

fn read_host_f32(t: &Tensor) -> Vec<f32> {
    t.to_device(Device::Cpu)
        .expect("D2H")
        .to_vec::<f32>()
        .expect("readback")
}

// ----------------------------------------------------------------------------
// THE PROOF — Device::Vulkan: forward records onto kt Tape, backward correct.
// ----------------------------------------------------------------------------

/// Keystone: `try_tape_add_kt` on `Device::Vulkan(0)` RECORDS onto the
/// thread-local `kiln_autograd::Tape` (tape.len() grows — anti-silent-drop),
/// its forward matches the CPU `ops::add` (parity < 1e-4), and `Tape::backward`
/// produces CORRECT, finite, correctly-shaped leaf grads over Vulkan storage
/// (`dL/da == dL/db == ones`, the exact `AddBackward` adjoint; < 1e-4 vs CPU).
///
/// Single-shot, tiny, GPU-gated.
#[test]
fn vk_tape_add_records_and_backprops() {
    let test_name = "vk_tape_add_records_and_backprops";
    if !vk_enabled(test_name) {
        return;
    }

    // Deterministic tiny F32 inputs.
    let a_data: Vec<f32> = (0..ROWS * COLS).map(|i| (i as f32) * 0.5 - 1.0).collect();
    let b_data: Vec<f32> = (0..ROWS * COLS).map(|i| 2.0 - (i as f32) * 0.25).collect();

    // CPU reference forward (independent of the recorder device gate).
    let a_cpu = Tensor::from_slice(&a_data, vec![ROWS, COLS]).unwrap();
    let b_cpu = Tensor::from_slice(&b_data, vec![ROWS, COLS]).unwrap();
    let cpu_fwd = read_host_f32(&ops::add(&a_cpu, &b_cpu).unwrap());

    // --- On-device (Vulkan) record + backward ---
    let a = Tensor::from_vec_on(Device::Vulkan(0), a_data.clone(), vec![ROWS, COLS])
        .expect("build a on Vulkan");
    let b = Tensor::from_vec_on(Device::Vulkan(0), b_data.clone(), vec![ROWS, COLS])
        .expect("build b on Vulkan");

    let ((y, a_id, b_id), tape) = with_thread_local_tape(|| {
        let y = try_tape_add_kt(&a, &b)
            .expect("try_tape_add_kt errored")
            .expect("try_tape_add_kt returned None — recorder did NOT record on Device::Vulkan");
        (y, a.id(), b.id())
    });

    // (1) ANTI-SILENT-DROP: the recorder actually fired on Vulkan.
    assert_eq!(
        tape.len(),
        1,
        "Vulkan recorder recorded {} nodes, expected exactly 1 (silent-drop / no scope)",
        tape.len()
    );
    assert_eq!(y.device(), Device::Vulkan(0), "forward output left Vulkan");

    // (2) FORWARD PARITY: harmonized Vulkan add == CPU add.
    let vk_fwd = read_host_f32(&y);
    let fwd_err = max_abs_err(&vk_fwd, &cpu_fwd);
    assert!(
        fwd_err < 1e-4,
        "forward parity FAILED: max_abs_err={fwd_err} >= 1e-4\n  vk ={vk_fwd:?}\n  cpu={cpu_fwd:?}"
    );

    // (3) BACKWARD over Vulkan storage: seed dL/dy = ones; AddBackward adjoint
    //     is dL/da = dL/db = grad_output, so both leaf grads must equal ones.
    let seed = Tensor::from_vec_on(Device::Vulkan(0), vec![1.0_f32; ROWS * COLS], vec![ROWS, COLS])
        .expect("build seed on Vulkan");
    let grads = tape
        .backward(y.id(), seed, |x, z| ops::add(x, z))
        .expect("Tape::backward errored on Vulkan storage");

    let da = grads.get(a_id).expect("no grad keyed on a.id()");
    let db = grads.get(b_id).expect("no grad keyed on b.id()");
    assert_eq!(da.shape(), &[ROWS, COLS], "dL/da wrong shape");
    assert_eq!(db.shape(), &[ROWS, COLS], "dL/db wrong shape");

    let da_v = read_host_f32(da);
    let db_v = read_host_f32(db);
    assert!(
        da_v.iter().all(|v| v.is_finite()) && db_v.iter().all(|v| v.is_finite()),
        "non-finite grads: da={da_v:?} db={db_v:?}"
    );
    let ones = vec![1.0_f32; ROWS * COLS];
    let da_err = max_abs_err(&da_v, &ones);
    let db_err = max_abs_err(&db_v, &ones);
    let bwd_err = da_err.max(db_err);
    assert!(
        bwd_err < 1e-4,
        "backward correctness FAILED: max_abs_err vs analytic ones={bwd_err} \
         (da={da_err}, db={db_err}) >= 1e-4\n  da={da_v:?}\n  db={db_v:?}"
    );

    eprintln!(
        "[PR5b PROOF] Device::Vulkan(0): tape.len()={} | forward max_abs_err vs CPU={:.3e} | \
         backward max_abs_err vs analytic (ones)={:.3e} (da={:.3e}, db={:.3e})",
        tape.len(),
        fwd_err,
        bwd_err,
        da_err,
        db_err
    );
}

// ----------------------------------------------------------------------------
// PR5a-gated recorder probe — proves the device gate admits Vulkan.
// IGNORED: the native rmsnorm `vulkan_fwd` GPUVM-faults on Strix Halo (PR3 gap,
// see module docs). The recorder DOES admit Vulkan (it runs the forward rather
// than returning None at the gate); the fault is downstream in the PR3 native
// rmsnorm bridge. This scaffold pins the intended proof + the frontier; remove
// `#[ignore]` once the PR3 rmsnorm-on-Vulkan path is fixed.
// ----------------------------------------------------------------------------

#[test]
#[ignore = "PR3 frontier: native rmsnorm vulkan_fwd GPUVM-faults via the kt bridge on Strix Halo \
            (radv GPUVM write fault + context loss). PR5a's gate IS widened (recorder admits \
            Vulkan); the fault is a PR3 op-coverage follow-up. Do NOT run autonomously — host \
            crash risk."]
fn vk_tape_rms_norm_records_on_vulkan() {
    let test_name = "vk_tape_rms_norm_records_on_vulkan";
    if !vk_enabled(test_name) {
        return;
    }
    const HIDDEN: usize = 8;
    let rows = 4;
    let x_data: Vec<f32> = (0..rows * HIDDEN)
        .map(|i| ((i as f32) * 0.37 - 2.0).sin() * 1.5 + 0.25)
        .collect();
    let w_data: Vec<f32> = (0..HIDDEN).map(|j| 0.1 * (j as f32) - 0.3).collect();

    let x = Tensor::from_vec_on(Device::Vulkan(0), x_data, vec![rows, HIDDEN]).unwrap();
    let weight = Tensor::from_vec_on(Device::Vulkan(0), w_data, vec![HIDDEN]).unwrap();

    let (y_opt, tape) = with_thread_local_tape(|| {
        try_tape_rms_norm_kt(&x, &weight, 1e-6).expect("recorder errored")
    });

    // The proof the gate admits Vulkan: the recorder returned Some + recorded.
    assert!(
        y_opt.is_some(),
        "try_tape_rms_norm_kt returned None on Vulkan — PR5a gate gap"
    );
    assert_eq!(tape.len(), 1, "rms_norm did not record exactly 1 node on Vulkan");
}
