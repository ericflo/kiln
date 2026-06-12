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
    try_tape_add_kt, try_tape_matmul_bf16w_kt, try_tape_rms_norm_kt, with_thread_local_tape,
};
use kiln_tensor::{DType, Device, Tensor, ops};
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
    let seed = Tensor::from_vec_on(
        Device::Vulkan(0),
        vec![1.0_f32; ROWS * COLS],
        vec![ROWS, COLS],
    )
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
// #1443 step1 — F32-act × BF16-weight matmul recorder + recorded dx backward.
// ----------------------------------------------------------------------------

/// #1443 step1 proof: `try_tape_matmul_bf16w_kt` on `Device::Vulkan(0)` RECORDS
/// the mixed-precision base projection (`out = x @ W.T`, F32 activation × frozen
/// BF16 weight) onto the kt `Tape` as exactly ONE node, its forward matches a
/// BF16-cast F32 reference (`vulkan_matmul`) to BF16 tolerance, and
/// `Tape::backward` produces the correct `dL/dx` over Vulkan storage via the
/// recorded `MatmulBf16wBackward`. Critically: the WEIGHT IS FROZEN — it is not
/// recorded as a tape input, so `grads` carries a key for `x` but NOT for the
/// weight (`input_count() == 1`, no `dW`).
///
/// Single-shot, tiny (rows=4, K=8, N=6), GPU-gated.
#[test]
fn vk_tape_matmul_bf16w_records_and_backprops() {
    let test_name = "vk_tape_matmul_bf16w_records_and_backprops";
    if !vk_enabled(test_name) {
        return;
    }
    let (rows, k, n) = (4usize, 8usize, 6usize);
    let x_data: Vec<f32> = (0..rows * k).map(|i| (i as f32) * 0.07 - 0.9).collect();
    let w_data: Vec<f32> = (0..n * k).map(|i| ((i % 5) as f32) * 0.2 - 0.3).collect();

    let x = Tensor::from_vec_on(Device::Vulkan(0), x_data.clone(), vec![rows, k])
        .expect("build x on Vulkan");
    // Frozen BF16 base weight [N, K] (transposed-weight layout).
    let w_f32 = Tensor::from_vec_on(Device::Vulkan(0), w_data.clone(), vec![n, k])
        .expect("build weight on Vulkan");
    let w_bf16 = kiln_tensor::vulkan_cast(&w_f32, DType::BF16).expect("cast weight -> BF16");

    let ((y, x_id, w_id), tape) = with_thread_local_tape(|| {
        let y = try_tape_matmul_bf16w_kt(&x, &w_bf16)
            .expect("try_tape_matmul_bf16w_kt errored")
            .expect("try_tape_matmul_bf16w_kt returned None — recorder did NOT record on Vulkan");
        (y, x.id(), w_bf16.id())
    });

    // (1) ANTI-SILENT-DROP: exactly one node recorded.
    assert_eq!(
        tape.len(),
        1,
        "bf16w recorder recorded {} nodes, expected 1",
        tape.len()
    );
    assert_eq!(y.device(), Device::Vulkan(0), "forward output left Vulkan");
    assert_eq!(y.shape(), &[rows, n]);
    assert_eq!(y.dtype(), DType::F32);

    // (2) FORWARD PARITY: vs the same BF16 weight cast back to F32 then F32 matmul.
    let w_f32_ref = kiln_tensor::vulkan_cast(&w_bf16, DType::F32).expect("bf16->f32 ref");
    let w_t_ref = w_f32_ref.transpose(0, 1).unwrap().contiguous().unwrap();
    let ref_fwd = read_host_f32(&kiln_tensor::vulkan_matmul(&x, &w_t_ref).expect("ref matmul"));
    let vk_fwd = read_host_f32(&y);
    let fwd_err = max_abs_err(&vk_fwd, &ref_fwd);
    assert!(
        fwd_err < 2e-2,
        "forward parity FAILED: max_abs_err={fwd_err}"
    );

    // (3) BACKWARD: seed dL/dy = ones. dx = grad_out @ W (frozen weight).
    let seed = Tensor::from_vec_on(Device::Vulkan(0), vec![1.0_f32; rows * n], vec![rows, n])
        .expect("seed on Vulkan");
    let grads = tape
        .backward(y.id(), seed, |g, z| ops::add(g, z))
        .expect("Tape::backward errored on bf16w Vulkan graph");

    // dx is keyed on x.id(); the FROZEN weight has NO grad key (not a tape input).
    let dx = grads.get(x_id).expect("no grad keyed on x.id()");
    assert_eq!(dx.shape(), &[rows, k], "dL/dx wrong shape");
    assert!(
        grads.get(w_id).is_none(),
        "FROZEN weight must have NO gradient, but a dW was recorded"
    );

    // Analytic dx straight from the kernel bridge: must match the recorded one.
    let dx_v = read_host_f32(dx);
    let analytic = read_host_f32(
        &kiln_tensor::vulkan_matmul_bf16w_bwd(
            &Tensor::from_vec_on(Device::Vulkan(0), vec![1.0_f32; rows * n], vec![rows, n])
                .unwrap(),
            &w_bf16,
        )
        .expect("analytic dx"),
    );
    assert!(
        dx_v.iter().all(|v| v.is_finite()),
        "non-finite dx: {dx_v:?}"
    );
    let dx_err = max_abs_err(&dx_v, &analytic);
    assert!(
        dx_err < 2e-2,
        "recorded dx != analytic dx: max_abs_err={dx_err}"
    );

    eprintln!(
        "[#1443 step1 PROOF] Device::Vulkan(0): tape.len()={} | fwd max_abs_err vs F32-cast ref={:.3e} | \
         dx max_abs_err vs analytic={:.3e} | weight FROZEN (no dW): {}",
        tape.len(),
        fwd_err,
        dx_err,
        grads.get(w_id).is_none()
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
// UPDATE (2026-06-01, post-review): the FORWARD-RECORD path below was re-run
// standalone on Strix Halo (RADV) and PASSED cleanly — `try_tape_rms_norm_kt`
// records exactly 1 node on Device::Vulkan(0), tape.len()==1, no GPUVM fault,
// no context loss (2.12s). So PR5a's gate genuinely works for a real GATED
// recorder (not just the device-agnostic `add` recorder), which closes the
// PR5b "add-only proof" review finding for the FORWARD direction.
//
// Still `#[ignore]`d (runs only when explicitly named) because: (a) the
// implementing agent observed a RADV GPUVM write fault + context loss on the
// NATIVE rmsnorm path under back-to-back GPU load, and (b) this test does not
// yet drive Tape::backward on rmsnorm — the backward composite + the native
// rmsnorm fault are the human-GPU-soak frontier (recovering from a context
// loss needs a human at the console). Extend to backward+CPU-parity during the
// PR6 soak, not autonomously. See docs/vk-harmonization/STATUS-AND-SOAK-HANDOFF.md.
#[ignore = "human-soak only: forward-record verified PASS standalone; backward on native \
            rmsnorm is the GPUVM-fault soak frontier — do not run autonomously (host crash risk)"]
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
    assert_eq!(
        tape.len(),
        1,
        "rms_norm did not record exactly 1 node on Vulkan"
    );
}

/// BISECT PROBE: isolate `ops::rms_norm` on Vulkan (no recorder, no cast, no
/// add_scalar) + a post-op queue-health H2D probe. If THIS faults, the rms_norm
/// kt path (vulkan_rmsnorm_last_axis bounce / kernel) wedges the queue; if it
/// passes, the culprit is upstream (cast / add_scalar host-fallback). Single-shot.
#[test]
fn vk_isolated_rms_norm_forward() {
    if !vk_enabled("vk_isolated_rms_norm_forward") {
        return;
    }
    let x = Tensor::from_vec_on(
        Device::Vulkan(0),
        (0..32).map(|i| (i as f32) * 0.1 - 1.0).collect(),
        vec![4, 8],
    )
    .unwrap();
    let w = Tensor::from_vec_on(Device::Vulkan(0), vec![0.5_f32; 8], vec![8]).unwrap();
    let y = ops::rms_norm(&x, &w, 1e-6).expect("ops::rms_norm errored");
    let yv = read_host_f32(&y); // forces a D2H submit — surfaces any async fault
    eprintln!("[BISECT] ops::rms_norm completed, y[0..4]={:?}", &yv[..4]);
    // Queue-health probe: a fresh H2D after rms_norm. If the queue is wedged by
    // an async GPUVM fault from rms_norm, THIS submit fails (as the seed did).
    let _probe = Tensor::from_vec_on(Device::Vulkan(0), vec![1.0_f32; 8], vec![8])
        .expect("post-rms_norm H2D probe failed — rms_norm wedged the queue");
    eprintln!("[BISECT] post-rms_norm H2D probe OK — queue healthy after rms_norm");
}

/// FRONTIER PROBE (2026-06-01, host-safety constraint relaxed with the user at
/// the console): drive `Tape::backward` on the rms_norm graph on
/// `Device::Vulkan(0)` and confirm it COMPLETES — no RADV GPUVM write-fault, no
/// context loss — with finite, correctly-shaped leaf grads. This is the smallest
/// single-shot payload (4x8) that resolves the open rmsnorm-backward GPUVM-fault
/// question (the PR5b soak frontier). Numerical parity is a deliberate follow-up:
/// the goal of THIS run is the binary "does the backward composite fault on
/// Vulkan storage?". GPU-gated; self-skips without KILN_TENSOR_VULKAN_TEST=1.
#[test]
fn vk_tape_rms_norm_backprops_on_vulkan() {
    let test_name = "vk_tape_rms_norm_backprops_on_vulkan";
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

    let ((y, x_id, w_id), tape) = with_thread_local_tape(|| {
        let y = try_tape_rms_norm_kt(&x, &weight, 1e-6)
            .expect("recorder errored")
            .expect("try_tape_rms_norm_kt returned None on Vulkan");
        (y, x.id(), weight.id())
    });
    assert_eq!(
        tape.len(),
        1,
        "rms_norm did not record exactly 1 node on Vulkan"
    );
    assert_eq!(y.device(), Device::Vulkan(0), "forward output left Vulkan");

    // The frontier: walk the tape (seed dL/dy = ones). If the native rmsnorm
    // backward composite GPUVM-faults on Vulkan storage, this is where it hangs.
    let seed = Tensor::from_vec_on(
        Device::Vulkan(0),
        vec![1.0_f32; rows * HIDDEN],
        vec![rows, HIDDEN],
    )
    .expect("seed on Vulkan");
    let grads = tape
        .backward(y.id(), seed, |g, z| ops::add(g, z))
        .expect("Tape::backward errored on the rms_norm Vulkan graph");

    let dx = grads.get(x_id).expect("no grad keyed on x.id()");
    let dw = grads.get(w_id).expect("no grad keyed on weight.id()");
    assert_eq!(dx.shape(), &[rows, HIDDEN], "dL/dx wrong shape");
    assert_eq!(dw.shape(), &[HIDDEN], "dL/dweight wrong shape");

    let dx_v = read_host_f32(dx);
    let dw_v = read_host_f32(dw);
    assert!(
        dx_v.iter().chain(dw_v.iter()).all(|v| v.is_finite()),
        "non-finite rms_norm grads: dx={dx_v:?} dw={dw_v:?}"
    );
    eprintln!(
        "[FRONTIER PASS] rms_norm Tape::backward COMPLETED on Device::Vulkan(0) \
         (no GPUVM fault): dx[0..3]={:?} dw={:?}",
        &dx_v[..3.min(dx_v.len())],
        dw_v
    );
}
