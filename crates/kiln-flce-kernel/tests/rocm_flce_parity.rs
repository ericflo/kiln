//! Phase R.7 — ROCm parity for the Fused Linear Cross-Entropy (FLCE) kernel.
//!
//! kiln-flce-kernel is a PURE-KT COMPOSITE: its forward + backward are
//! expressed entirely in `kiln_tensor::Tensor` ops (matmul, exp, max_axis,
//! sum_axis, index_select, scatter_add, ...), which carry their own backend
//! dispatch. There is no `.cu` / FFI / build.rs of its own — the `rocm`
//! feature only forwards to `kiln-tensor/rocm`. So this test exercises the
//! integration: the SAME `_kt` entry points run on `Device::Rocm(0)` inputs
//! must match the SAME entry points run on `Device::Cpu` inputs.
//!
//! The chunked log-sum-exp reduction sweeps the vocab (last) axis in chunks;
//! a wave64 reduction bug in any underlying kt-tensor reduction (`max_axis`,
//! `sum_axis`, the CE/softmax family) compiles cleanly and only manifests at
//! widths straddling the 32-/64-lane wavefront boundary. So the forward sweep
//! covers vocab widths {31,32,33,63,64,65,127,128,129,256,1024}, with a
//! chunk_size deliberately smaller than the width so the running-max /
//! running-sumexp accumulation is genuinely cross-chunk.
//!
//! Skips cleanly when no ROCm device is present.
//!
//! Run (wave32): `cargo test -p kiln-flce-kernel --features rocm --test rocm_flce_parity`
//! Run (wave64): `KILN_ROCM_WAVE64=1 cargo test -p kiln-flce-kernel --features rocm --test rocm_flce_parity`
#![cfg(feature = "rocm")]

use kiln_flce_kernel::kt_api::{
    fused_linear_cross_entropy_phase_b_backward_kt, fused_linear_cross_entropy_phase_b_kt,
    fused_linear_cross_entropy_phase_b_with_metadata_kt,
};
use kiln_tensor::{Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.7 FLCE parity test");
        true
    } else {
        false
    }
}

/// Deterministic pseudo-random f32 in ~[-1, 1) from a linear index.
fn val(i: usize, seed: usize) -> f32 {
    let x = ((i
        .wrapping_mul(2654435761)
        .wrapping_add(seed.wrapping_mul(40503)))
        % 2000) as f32;
    x / 1000.0 - 1.0
}

/// Read a (possibly-ROCm) tensor back to a host `Vec<f32>`.
///
/// Under `--features rocm` (without `cuda`), `Tensor::to_device(Cpu)` does NOT
/// handle the Rocm→Cpu transition, so `to_vec` would error on a ROCm tensor.
/// Mirror the established kiln-tensor ROCm parity tests: stage through the
/// explicit `rocm_to_host_copy` helper first, then `to_vec` on the CPU view.
/// A CPU tensor passes straight through `to_vec`.
fn read_host_f32(t: &Tensor) -> Vec<f32> {
    match t.device() {
        Device::Rocm(_) => {
            let host = kiln_tensor::rocm_to_host_copy(t).expect("rocm_to_host_copy");
            host.to_vec::<f32>().expect("to_vec (host view)")
        }
        _ => t.to_vec::<f32>().expect("to_vec"),
    }
}

/// Build the `(hidden, head_t)` pair on the requested device.
///
/// `hidden`: `[1, seq_len, hidden_size]` F32. `head_t`: `[hidden_size,
/// vocab_size]` F32. Same bytes regardless of device so CPU and ROCm runs are
/// comparing identical inputs.
fn build_inputs(
    device: Device,
    seq_len: usize,
    hidden_size: usize,
    vocab_size: usize,
) -> (Tensor, Tensor) {
    let hidden_n = seq_len * hidden_size;
    let hidden_data: Vec<f32> = (0..hidden_n).map(|i| val(i, 1) * 0.5).collect();
    let head_n = hidden_size * vocab_size;
    let head_data: Vec<f32> = (0..head_n).map(|i| val(i, 2) * 0.3).collect();

    let hidden = Tensor::from_vec_on(device, hidden_data, vec![1, seq_len, hidden_size])
        .unwrap_or_else(|e| panic!("from_vec_on hidden ({device}): {e}"));
    let head_t = Tensor::from_vec_on(device, head_data, vec![hidden_size, vocab_size])
        .unwrap_or_else(|e| panic!("from_vec_on head_t ({device}): {e}"));
    (hidden, head_t)
}

/// Forward parity: ROCm-device FLCE loss must match CPU FLCE loss across a
/// wavefront-boundary sweep of vocab widths. chunk_size < vocab so the
/// running-max / running-sumexp accumulation crosses chunk boundaries (this is
/// the path that exercises the underlying last-axis reductions).
#[test]
fn flce_forward_parity_wavefront_boundary_sweep() {
    if no_rocm() {
        return;
    }
    let hidden_size = 8usize;
    let seq_len = 6usize;
    // Last-axis (vocab) widths that straddle the 32-/64-lane boundaries.
    let vocab_widths = [31usize, 32, 33, 63, 64, 65, 127, 128, 129, 256, 1024];

    for &vocab in &vocab_widths {
        // chunk_size deliberately smaller than vocab → genuine cross-chunk
        // accumulation. 23 is coprime-ish to the widths so chunk seams fall at
        // varied offsets across the sweep.
        let chunk_size = 23usize.min(vocab);

        // input_ids: targets land inside the vocab range; all shifted
        // positions active so every row contributes to the mean.
        let ids: Vec<u32> = (0..seq_len as u32)
            .map(|i| (i.wrapping_mul(37).wrapping_add(5)) % vocab as u32)
            .collect();
        let mask = vec![true; seq_len];

        let (h_cpu, w_cpu) = build_inputs(Device::Cpu, seq_len, hidden_size, vocab);
        let (h_roc, w_roc) = build_inputs(Device::Rocm(0), seq_len, hidden_size, vocab);

        let loss_cpu =
            fused_linear_cross_entropy_phase_b_kt(&h_cpu, &w_cpu, &ids, &mask, chunk_size)
                .unwrap_or_else(|e| panic!("cpu forward (vocab={vocab}): {e}"));
        let loss_roc =
            fused_linear_cross_entropy_phase_b_kt(&h_roc, &w_roc, &ids, &mask, chunk_size)
                .unwrap_or_else(|e| panic!("rocm forward (vocab={vocab}): {e}"));

        assert!(loss_cpu.shape().is_empty(), "cpu loss must be rank-0");
        assert!(loss_roc.shape().is_empty(), "rocm loss must be rank-0");

        let c = read_host_f32(&loss_cpu);
        let r = read_host_f32(&loss_roc);
        assert_eq!(c.len(), 1, "cpu loss is a scalar");
        assert_eq!(r.len(), 1, "rocm loss is a scalar");

        let cv = c[0];
        let rv = r[0];
        let diff = (cv - rv).abs();
        let tol = 2e-4 + 2e-4 * cv.abs();
        assert!(
            diff <= tol,
            "FLCE forward mismatch at vocab={vocab} chunk={chunk_size}: cpu {cv} rocm {rv} \
             diff {diff} (a wave64 last-axis reduction bug shows up exactly here)"
        );
    }
    eprintln!(
        "FLCE forward CPU-vs-ROCm parity passed across wavefront-boundary vocab widths {vocab_widths:?}"
    );
}

#[test]
fn flce_forward_long_sft_shape_is_finite() {
    if no_rocm() {
        return;
    }

    let seq_len = 40_314usize;
    let hidden_size = 2_560usize;
    let vocab_size = 4_096usize;
    let active_rows = 1_014usize;
    let chunk_size = 4_096usize;

    let hidden_data: Vec<f32> = (0..seq_len * hidden_size)
        .map(|i| val(i, 41) * 0.02)
        .collect();
    let head_data: Vec<f32> = (0..hidden_size * vocab_size)
        .map(|i| val(i, 73) * 0.02)
        .collect();

    let hidden = Tensor::from_vec_on(Device::Rocm(0), hidden_data, vec![1, seq_len, hidden_size])
        .unwrap_or_else(|e| panic!("from_vec_on long hidden: {e}"));
    let head_t = Tensor::from_vec_on(Device::Rocm(0), head_data, vec![hidden_size, vocab_size])
        .unwrap_or_else(|e| panic!("from_vec_on long head_t: {e}"));

    let mut input_ids = vec![0u32; seq_len];
    let mut label_mask = vec![false; seq_len];
    for active_idx in 0..active_rows {
        let shift_idx = active_idx * 39 + 17;
        let label_pos = shift_idx + 1;
        label_mask[label_pos] = true;
        input_ids[label_pos] = ((active_idx * 97 + 11) % vocab_size) as u32;
    }

    let (loss, metadata) = fused_linear_cross_entropy_phase_b_with_metadata_kt(
        &hidden,
        &head_t,
        &input_ids,
        &label_mask,
        chunk_size,
    )
    .unwrap_or_else(|e| panic!("long SFT-shaped ROCm FLCE forward: {e}"));
    let host = kiln_tensor::rocm_to_host_copy(&loss).expect("rocm_to_host_copy loss");
    let values = host.to_vec::<f32>().expect("loss to_vec");
    assert_eq!(values.len(), 1);
    assert!(
        values[0].is_finite(),
        "long SFT-shaped FLCE loss must be finite, got {}",
        values[0]
    );
    assert!(
        metadata.is_some(),
        "long SFT-shaped FLCE should have active metadata for {active_rows} active rows"
    );
}

#[test]
fn flce_forward_backward_row_tiled_many_active_rows_matches_cpu() {
    if no_rocm() {
        return;
    }

    let active_rows = 5_000usize;
    let seq_len = active_rows + 1;
    let hidden_size = 32usize;
    let vocab_size = 257usize;
    let chunk_size = 64usize;

    let mut input_ids = vec![0u32; seq_len];
    let mut label_mask = vec![false; seq_len];
    for row in 0..active_rows {
        let label_pos = row + 1;
        label_mask[label_pos] = true;
        input_ids[label_pos] = ((row * 53 + 7) % vocab_size) as u32;
    }

    let (h_cpu, w_cpu) = build_inputs(Device::Cpu, seq_len, hidden_size, vocab_size);
    let (h_roc, w_roc) = build_inputs(Device::Rocm(0), seq_len, hidden_size, vocab_size);

    let loss_cpu =
        fused_linear_cross_entropy_phase_b_kt(&h_cpu, &w_cpu, &input_ids, &label_mask, chunk_size)
            .unwrap_or_else(|e| panic!("cpu row-tiled forward reference: {e}"));
    let loss_roc =
        fused_linear_cross_entropy_phase_b_kt(&h_roc, &w_roc, &input_ids, &label_mask, chunk_size)
            .unwrap_or_else(|e| panic!("rocm row-tiled forward: {e}"));

    let cpu_loss = read_host_f32(&loss_cpu)[0];
    let rocm_loss = read_host_f32(&loss_roc)[0];
    let loss_diff = (cpu_loss - rocm_loss).abs();
    let loss_tol = 2e-4 + 2e-4 * cpu_loss.abs();
    assert!(
        loss_diff <= loss_tol,
        "row-tiled FLCE forward mismatch: cpu {cpu_loss} rocm {rocm_loss} diff {loss_diff}"
    );

    let grad_cpu = Tensor::from_vec_on(Device::Cpu, vec![1.0f32], vec![]).expect("cpu scalar grad");
    let grad_roc =
        Tensor::from_vec_on(Device::Rocm(0), vec![1.0f32], vec![]).expect("rocm scalar grad");
    let g_cpu = fused_linear_cross_entropy_phase_b_backward_kt(
        &h_cpu,
        &w_cpu,
        &input_ids,
        &label_mask,
        chunk_size,
        &grad_cpu,
    )
    .unwrap_or_else(|e| panic!("cpu row-tiled backward reference: {e}"));
    let g_roc = fused_linear_cross_entropy_phase_b_backward_kt(
        &h_roc,
        &w_roc,
        &input_ids,
        &label_mask,
        chunk_size,
        &grad_roc,
    )
    .unwrap_or_else(|e| panic!("rocm row-tiled backward: {e}"));

    let cpu_grad = read_host_f32(&g_cpu);
    let rocm_grad = read_host_f32(&g_roc);
    assert_eq!(cpu_grad.len(), rocm_grad.len());
    let mut max_abs = 0.0f32;
    let mut max_mag = 0.0f32;
    for (a, b) in cpu_grad.iter().zip(rocm_grad.iter()) {
        max_abs = max_abs.max((a - b).abs());
        max_mag = max_mag.max(a.abs()).max(b.abs());
    }
    let grad_tol = 5e-4 + 5e-4 * max_mag;
    assert!(
        max_abs <= grad_tol,
        "row-tiled FLCE backward mismatch: max_abs {max_abs} tol {grad_tol} max_mag {max_mag}"
    );
}

/// Backward parity: ROCm-device FLCE `dhidden` must match CPU FLCE `dhidden`
/// element-for-element across the same wavefront-boundary vocab sweep. The
/// backward reruns the same chunked reductions (pass 1) plus a per-chunk
/// softmax / matmul (pass 2), so it stresses the same wave-size-sensitive
/// reduction surface as the forward.
#[test]
fn flce_backward_parity_wavefront_boundary_sweep() {
    if no_rocm() {
        return;
    }
    let hidden_size = 8usize;
    let seq_len = 6usize;
    let vocab_widths = [31usize, 32, 33, 63, 64, 65, 127, 128, 129, 256, 1024];

    for &vocab in &vocab_widths {
        let chunk_size = 23usize.min(vocab);
        let ids: Vec<u32> = (0..seq_len as u32)
            .map(|i| (i.wrapping_mul(37).wrapping_add(5)) % vocab as u32)
            .collect();
        let mask = vec![true; seq_len];

        let (h_cpu, w_cpu) = build_inputs(Device::Cpu, seq_len, hidden_size, vocab);
        let (h_roc, w_roc) = build_inputs(Device::Rocm(0), seq_len, hidden_size, vocab);

        // Seed grad must live on the same device as `hidden` so the backward's
        // broadcast-mul stays single-device (mixed cpu+rocm operands error).
        let grad_cpu = Tensor::from_vec_on(Device::Cpu, vec![1.0f32], vec![]).unwrap();
        let grad_roc = Tensor::from_vec_on(Device::Rocm(0), vec![1.0f32], vec![]).unwrap();

        let g_cpu = fused_linear_cross_entropy_phase_b_backward_kt(
            &h_cpu, &w_cpu, &ids, &mask, chunk_size, &grad_cpu,
        )
        .unwrap_or_else(|e| panic!("cpu backward (vocab={vocab}): {e}"));
        let g_roc = fused_linear_cross_entropy_phase_b_backward_kt(
            &h_roc, &w_roc, &ids, &mask, chunk_size, &grad_roc,
        )
        .unwrap_or_else(|e| panic!("rocm backward (vocab={vocab}): {e}"));

        assert_eq!(
            g_cpu.shape(),
            &[1, seq_len, hidden_size],
            "cpu dhidden shape (vocab={vocab})"
        );
        assert_eq!(
            g_roc.shape(),
            &[1, seq_len, hidden_size],
            "rocm dhidden shape (vocab={vocab})"
        );

        let c = read_host_f32(&g_cpu);
        let r = read_host_f32(&g_roc);
        assert_eq!(c.len(), r.len(), "dhidden element count (vocab={vocab})");

        let mut max_abs = 0.0f32;
        let mut max_mag = 0.0f32;
        for (a, b) in c.iter().zip(r.iter()) {
            max_abs = max_abs.max((a - b).abs());
            max_mag = max_mag.max(a.abs());
        }
        let rel = if max_mag > 1e-6 {
            max_abs / max_mag
        } else {
            max_abs
        };
        assert!(
            max_abs < 2e-4 || rel < 2e-4,
            "FLCE backward mismatch at vocab={vocab} chunk={chunk_size}: \
             max_abs={max_abs:.3e} max_mag={max_mag:.6} rel={rel:.3e} \
             (a wave64 reduction bug shows up exactly here)"
        );
    }
    eprintln!(
        "FLCE backward CPU-vs-ROCm parity passed across wavefront-boundary vocab widths {vocab_widths:?}"
    );
}
