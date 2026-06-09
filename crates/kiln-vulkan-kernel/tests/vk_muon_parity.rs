//! Numerical parity for the fused Vulkan Muon optimizer kernel
//! (`dispatch_muon_step_f32`) against a CPU reference that replicates the
//! exact algorithm (heavy-ball momentum -> Nesterov look-ahead ->
//! gram-space P-accumulator Newton-Schulz -> RMS-matching scale ->
//! decoupled-weight-decay descent). The CPU reference here mirrors
//! `kiln_optim::lion_muon::{Muon::step, newton_schulz}`; the *algorithm*
//! is separately validated by kiln-optim's `newton_schulz_matches_direct_iteration`
//! test, so this test's job is to confirm the GPU kernel *executes* that
//! algorithm correctly on real hardware (barriers, indexing, push-constant
//! ABI, BF16/F32 lanes). Skips cleanly when no Vulkan device is present.
//! (candle-free; #1082)

use anyhow::Result;
use kiln_vulkan_kernel::VulkanDevice;
use kiln_vulkan_kernel::kernels::{dispatch_muon_step_bf16, dispatch_muon_step_f32};
use kiln_vulkan_kernel::vk_tensor::VkTensor;
use std::sync::Arc;

/// Round an f32 to its nearest bf16 grid point and back (RNE), matching
/// the shader's f32->bf16 lane write, so input rounding isn't charged to
/// the GPU-vs-oracle error budget.
fn bf16_round(v: f32) -> f32 {
    let bits = v.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return v; // NaN
    }
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    f32::from_bits(rounded & 0xffff_0000)
}

fn vk_dev() -> Option<Arc<VulkanDevice>> {
    if !VulkanDevice::probe() {
        return None;
    }
    VulkanDevice::new().ok().map(Arc::new)
}

// ---- CPU reference (mirrors kiln_optim::lion_muon) ----

fn matmul_kk(a: &[f32], b: &[f32], out: &mut [f32], k: usize) {
    for i in 0..k {
        for j in 0..k {
            let mut s = 0.0f32;
            for t in 0..k {
                s += a[i * k + t] * b[t * k + j];
            }
            out[i * k + j] = s;
        }
    }
}

fn newton_schulz(w: &[f32], rows: usize, cols: usize, iters: u32) -> Vec<f32> {
    let n = rows * cols;
    let frob = w.iter().map(|&v| v * v).sum::<f32>().sqrt();
    if frob == 0.0 {
        return vec![0.0; n];
    }
    let inv_frob = 1.0 / frob;
    let inv_frob2 = inv_frob * inv_frob;
    let transpose = rows > cols;
    let k = rows.min(cols);
    let mut a = vec![0.0f32; k * k];
    if !transpose {
        for i in 0..k {
            for j in 0..k {
                let mut s = 0.0f32;
                for c in 0..cols {
                    s += w[i * cols + c] * w[j * cols + c];
                }
                a[i * k + j] = s * inv_frob2;
            }
        }
    } else {
        for i in 0..k {
            for j in 0..k {
                let mut s = 0.0f32;
                for r in 0..rows {
                    s += w[r * cols + i] * w[r * cols + j];
                }
                a[i * k + j] = s * inv_frob2;
            }
        }
    }
    let mut p = vec![0.0f32; k * k];
    for i in 0..k {
        p[i * k + i] = 1.0;
    }
    let (ca, cb, cc) = (3.4445f32, -4.7750f32, 2.0315f32);
    let mut a2 = vec![0.0f32; k * k];
    let mut m = vec![0.0f32; k * k];
    let mut tmp = vec![0.0f32; k * k];
    for _ in 0..iters {
        matmul_kk(&a, &a, &mut a2, k);
        for i in 0..k {
            for j in 0..k {
                let id = if i == j { 1.0 } else { 0.0 };
                m[i * k + j] = ca * id + cb * a[i * k + j] + cc * a2[i * k + j];
            }
        }
        if !transpose {
            matmul_kk(&m, &p, &mut tmp, k);
        } else {
            matmul_kk(&p, &m, &mut tmp, k);
        }
        p.copy_from_slice(&tmp);
        matmul_kk(&m, &a, &mut tmp, k);
        matmul_kk(&tmp, &m, &mut a, k);
    }
    let mut o = vec![0.0f32; n];
    if !transpose {
        for i in 0..rows {
            for c in 0..cols {
                let mut s = 0.0f32;
                for j in 0..k {
                    s += p[i * k + j] * w[j * cols + c];
                }
                o[i * cols + c] = s * inv_frob;
            }
        }
    } else {
        for r in 0..rows {
            for c in 0..cols {
                let mut s = 0.0f32;
                for j in 0..k {
                    s += w[r * cols + j] * p[j * k + c];
                }
                o[r * cols + c] = s * inv_frob;
            }
        }
    }
    o
}

#[allow(clippy::too_many_arguments)]
fn cpu_muon(
    param: &[f32],
    grad: &[f32],
    mom_in: &[f32],
    rows: usize,
    cols: usize,
    lr: f32,
    mom: f32,
    nesterov: bool,
    ns_iters: u32,
    wd: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = rows * cols;
    let mut momentum = mom_in.to_vec();
    for i in 0..n {
        momentum[i] = mom * momentum[i] + grad[i];
    }
    let b: Vec<f32> = (0..n)
        .map(|i| {
            if nesterov {
                grad[i] + mom * momentum[i]
            } else {
                momentum[i]
            }
        })
        .collect();
    let k = rows.min(cols);
    let do_ortho = rows >= 2 && cols >= 2 && k <= 32;
    let update = if do_ortho {
        let mut o = newton_schulz(&b, rows, cols, ns_iters);
        let scale = (rows.max(cols) as f32).sqrt();
        for v in o.iter_mut() {
            *v *= scale;
        }
        o
    } else {
        b
    };
    let mut p = param.to_vec();
    for i in 0..n {
        p[i] = p[i] * (1.0 - lr * wd) - lr * update[i];
    }
    (p, momentum)
}

fn det_fill(seed: u64, len: usize, scale: f32) -> Vec<f32> {
    // splitmix64 -> uniform-ish in [-scale, scale]
    let mut z = seed;
    (0..len)
        .map(|_| {
            z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            x ^= x >> 31;
            let u = (x >> 40) as f32 / (1u32 << 24) as f32; // [0,1)
            (u * 2.0 - 1.0) * scale
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    dev: &Arc<VulkanDevice>,
    rows: usize,
    cols: usize,
    nesterov: bool,
    wd: f32,
    seed: u64,
) -> Result<()> {
    let n = rows * cols;
    let (lr, mom, ns_iters) = (0.02f32, 0.95f32, 5u32);
    let param_data = det_fill(seed, n, 1.0);
    let grad_data = det_fill(seed ^ 0xa5a5, n, 0.5);
    let mom_data = det_fill(seed ^ 0x1234, n, 0.3);

    let (exp_param, exp_mom) = cpu_muon(
        &param_data,
        &grad_data,
        &mom_data,
        rows,
        cols,
        lr,
        mom,
        nesterov,
        ns_iters,
        wd,
    );

    let param = VkTensor::from_f32_slice(&param_data, vec![rows, cols], Arc::clone(dev))?;
    let grad = VkTensor::from_f32_slice(&grad_data, vec![rows, cols], Arc::clone(dev))?;
    let momentum = VkTensor::from_f32_slice(&mom_data, vec![rows, cols], Arc::clone(dev))?;

    dispatch_muon_step_f32(
        dev,
        param.buffer(),
        grad.buffer(),
        momentum.buffer(),
        n,
        rows,
        cols,
        lr,
        mom,
        nesterov,
        ns_iters,
        wd,
    )?;

    let got_param = param.to_vec_f32()?;
    let got_mom = momentum.to_vec_f32()?;

    let tol = 2e-3f32;
    for i in 0..n {
        assert!(
            (got_param[i] - exp_param[i]).abs() < tol,
            "param[{i}] shape {rows}x{cols} nesterov={nesterov} wd={wd}: gpu={} cpu={}",
            got_param[i],
            exp_param[i]
        );
        assert!(
            (got_mom[i] - exp_mom[i]).abs() < tol,
            "momentum[{i}] shape {rows}x{cols}: gpu={} cpu={}",
            got_mom[i],
            exp_mom[i]
        );
    }
    Ok(())
}

#[test]
fn vk_muon_parity_orthogonalized_shapes() -> Result<()> {
    let Some(dev) = vk_dev() else {
        eprintln!("Vulkan device unavailable, skipping vk_muon_parity_orthogonalized_shapes");
        return Ok(());
    };
    // short-fat (k=rows), tall-skinny (k=cols), square — all rank-2 with k<=32.
    let cases: &[(usize, usize)] = &[(8, 32), (32, 8), (16, 16), (4, 96), (96, 4), (24, 24)];
    for (idx, &(rows, cols)) in cases.iter().enumerate() {
        run_case(&dev, rows, cols, true, 0.0, 0x1000 + idx as u64)?;
        run_case(&dev, rows, cols, false, 0.0, 0x2000 + idx as u64)?;
        run_case(&dev, rows, cols, true, 0.01, 0x3000 + idx as u64)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_case_bf16(
    dev: &Arc<VulkanDevice>,
    rows: usize,
    cols: usize,
    nesterov: bool,
    wd: f32,
    seed: u64,
) -> Result<()> {
    let n = rows * cols;
    let (lr, mom, ns_iters) = (0.02f32, 0.95f32, 5u32);
    // Round inputs to the bf16 grid up front so the GPU and the f32 oracle
    // start from identical values; the remaining gap is the GPU's
    // intermediate bf16 rounding (momentum + param stored bf16).
    let param_data: Vec<f32> = det_fill(seed, n, 1.0)
        .iter()
        .map(|&v| bf16_round(v))
        .collect();
    let grad_data: Vec<f32> = det_fill(seed ^ 0xa5a5, n, 0.5)
        .iter()
        .map(|&v| bf16_round(v))
        .collect();
    let mom_data: Vec<f32> = det_fill(seed ^ 0x1234, n, 0.3)
        .iter()
        .map(|&v| bf16_round(v))
        .collect();

    let (exp_param, exp_mom) = cpu_muon(
        &param_data,
        &grad_data,
        &mom_data,
        rows,
        cols,
        lr,
        mom,
        nesterov,
        ns_iters,
        wd,
    );

    let param = VkTensor::from_f32_slice_as_bf16(&param_data, vec![rows, cols], Arc::clone(dev))?;
    let grad = VkTensor::from_f32_slice_as_bf16(&grad_data, vec![rows, cols], Arc::clone(dev))?;
    let momentum = VkTensor::from_f32_slice_as_bf16(&mom_data, vec![rows, cols], Arc::clone(dev))?;

    dispatch_muon_step_bf16(
        dev,
        param.buffer(),
        grad.buffer(),
        momentum.buffer(),
        n,
        rows,
        cols,
        lr,
        mom,
        nesterov,
        ns_iters,
        wd,
    )?;

    let got_param = param.to_vec_f32()?;
    let got_mom = momentum.to_vec_f32()?;

    // bf16 carries ~8 mantissa bits; the gram sum over `cols` terms + the NS
    // iterations compound that, so use a relative+absolute bf16 budget. A
    // lane-addressing bug would corrupt values FAR beyond this, so the test
    // still discriminates correct vs wrong lane access.
    let close = |g: f32, e: f32| (g - e).abs() <= 6e-2 * (1.0 + e.abs());
    for i in 0..n {
        assert!(
            close(got_param[i], exp_param[i]),
            "bf16 param[{i}] {rows}x{cols} nesterov={nesterov} wd={wd}: gpu={} cpu={}",
            got_param[i],
            exp_param[i]
        );
        assert!(
            close(got_mom[i], exp_mom[i]),
            "bf16 momentum[{i}] {rows}x{cols}: gpu={} cpu={}",
            got_mom[i],
            exp_mom[i]
        );
    }
    Ok(())
}

#[test]
fn vk_muon_parity_bf16_shapes() -> Result<()> {
    let Some(dev) = vk_dev() else {
        eprintln!("Vulkan device unavailable, skipping vk_muon_parity_bf16_shapes");
        return Ok(());
    };
    // Even and odd cols exercise the bf16 lane pack/unpack (2 lanes/word)
    // and the odd-length last-word path.
    let cases: &[(usize, usize)] = &[(8, 32), (32, 8), (16, 16), (4, 96), (5, 31), (31, 5)];
    for (idx, &(rows, cols)) in cases.iter().enumerate() {
        run_case_bf16(&dev, rows, cols, true, 0.0, 0x7000 + idx as u64)?;
        run_case_bf16(&dev, rows, cols, false, 0.01, 0x8000 + idx as u64)?;
    }
    Ok(())
}

#[test]
fn vk_muon_parity_non_matrix_falls_back_to_momentum_sgd() -> Result<()> {
    let Some(dev) = vk_dev() else {
        eprintln!("Vulkan device unavailable, skipping vk_muon_parity_non_matrix");
        return Ok(());
    };
    // A [1, N] "matrix" has min dim 1 -> do_ortho is false -> plain
    // (Nesterov) momentum SGD on both GPU and the CPU reference.
    run_case(&dev, 1, 64, true, 0.0, 0xbeef)?;
    run_case(&dev, 64, 1, false, 0.0, 0xcafe)?;
    Ok(())
}
