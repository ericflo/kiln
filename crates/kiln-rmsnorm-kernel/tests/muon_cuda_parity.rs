//! Numerical parity for the fused CUDA/ROCm Muon optimizer kernel
//! (`muon_step_f32_kt` / `muon_step_bf16_kt`) against a CPU reference that
//! replicates the exact algorithm (heavy-ball momentum -> Nesterov
//! look-ahead -> gram-space P-accumulator Newton-Schulz -> RMS-matching
//! scale -> decoupled-weight-decay descent). The CPU reference mirrors
//! `kiln_optim::lion_muon::{Muon::step, newton_schulz}`; the *algorithm*
//! is separately validated by kiln-optim's
//! `newton_schulz_matches_direct_iteration` test (and on real GPU silicon
//! by `kiln-vulkan-kernel`'s `vk_muon_parity`), so this test's job is to
//! confirm the GPU kernel *executes* that algorithm correctly. The same
//! `optimizer_step.cu` source is compiled by nvcc for CUDA and hipcc for
//! ROCm, so this test is backend-neutral.
//!
//! Constructs device tensors directly and reads results back through the
//! backend-specific D2H helper. Skips cleanly when no compiled backend device
//! is present.
#![cfg(any(feature = "cuda", feature = "rocm"))]

use half::bf16;

use kiln_rmsnorm_kernel::{muon_step_bf16_kt, muon_step_f32_kt};
use kiln_tensor::{DType, Device, Element, Tensor};

#[derive(Clone, Copy, Debug)]
enum TestGpu {
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "rocm")]
    Rocm,
}

fn available_gpu(test: &str) -> Option<TestGpu> {
    #[cfg(feature = "cuda")]
    {
        if kiln_tensor::primary_cuda_context(0).is_ok() {
            return Some(TestGpu::Cuda);
        }
    }
    #[cfg(feature = "rocm")]
    {
        if kiln_tensor::rocm_is_available() {
            return Some(TestGpu::Rocm);
        }
    }
    eprintln!("no CUDA/ROCm device available; skipping {test}");
    None
}

fn device_tensor_from_slice<E: Element>(gpu: TestGpu, data: &[E], shape: Vec<usize>) -> Tensor {
    match gpu {
        #[cfg(feature = "cuda")]
        TestGpu::Cuda => Tensor::cuda_from_slice(data, shape, 0).expect("cuda tensor"),
        #[cfg(feature = "rocm")]
        TestGpu::Rocm => {
            Tensor::from_vec_on(Device::Rocm(0), data.to_vec(), shape).expect("rocm tensor")
        }
    }
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
    for i in 0..k {
        for j in 0..k {
            let mut s = 0.0f32;
            if !transpose {
                for c in 0..cols {
                    s += w[i * cols + c] * w[j * cols + c];
                }
            } else {
                for r in 0..rows {
                    s += w[r * cols + i] * w[r * cols + j];
                }
            }
            a[i * k + j] = s * inv_frob2;
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
    for r in 0..rows {
        for c in 0..cols {
            let mut s = 0.0f32;
            if !transpose {
                for j in 0..k {
                    s += p[r * k + j] * w[j * cols + c];
                }
            } else {
                for j in 0..k {
                    s += w[r * cols + j] * p[j * k + c];
                }
            }
            o[r * cols + c] = s * inv_frob;
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
    // K_MAX for the CUDA kernel is 48.
    let do_ortho = rows >= 2 && cols >= 2 && k <= 48;
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
    let mut z = seed;
    (0..len)
        .map(|_| {
            z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            x ^= x >> 31;
            let u = (x >> 40) as f32 / (1u32 << 24) as f32;
            (u * 2.0 - 1.0) * scale
        })
        .collect()
}

/// Copy a (possibly BF16) CUDA tensor to host and read it as F32. The
/// `to_dtype(F32)` is a no-op for F32 tensors and the BF16->F32 promotion
/// for BF16 ones (so `to_vec1::<f32>` always sees an F32 host tensor).
fn host_f32(gpu: TestGpu, t: &Tensor) -> Vec<f32> {
    let host = match gpu {
        #[cfg(feature = "cuda")]
        TestGpu::Cuda => kiln_tensor::cuda_to_host_copy(t).expect("cuda_to_host_copy"),
        #[cfg(feature = "rocm")]
        TestGpu::Rocm => kiln_tensor::rocm_to_host_copy(t).expect("rocm_to_host_copy"),
    };
    host.to_dtype(DType::F32)
        .expect("to_dtype f32")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("to_vec1")
}

#[allow(clippy::too_many_arguments)]
fn run_case_f32(gpu: TestGpu, rows: usize, cols: usize, nesterov: bool, wd: f32, seed: u64) {
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

    let param = device_tensor_from_slice(gpu, &param_data, vec![rows, cols]);
    let grad = device_tensor_from_slice(gpu, &grad_data, vec![rows, cols]);
    let momentum = device_tensor_from_slice(gpu, &mom_data, vec![rows, cols]);

    muon_step_f32_kt(&param, &grad, &momentum, lr, mom, nesterov, ns_iters, wd)
        .expect("muon_step_f32_kt");

    let got_param = host_f32(gpu, &param);
    let got_mom = host_f32(gpu, &momentum);

    let tol = if n >= 4096 { 8e-3f32 } else { 2e-3f32 };
    for i in 0..n {
        assert!(
            (got_param[i] - exp_param[i]).abs() < tol,
            "param[{i}] {rows}x{cols} nesterov={nesterov} wd={wd}: gpu={} cpu={}",
            got_param[i],
            exp_param[i]
        );
        assert!(
            (got_mom[i] - exp_mom[i]).abs() < tol,
            "momentum[{i}] {rows}x{cols}: gpu={} cpu={}",
            got_mom[i],
            exp_mom[i]
        );
    }
}

#[test]
fn muon_cuda_parity_f32_shapes() {
    let Some(gpu) = available_gpu("muon_cuda_parity_f32_shapes") else {
        return;
    };
    // short-fat (k=rows), tall-skinny (k=cols), square — all k<=48.
    let cases: &[(usize, usize)] = &[(8, 32), (32, 8), (16, 16), (4, 96), (96, 4), (48, 48)];
    for (idx, &(rows, cols)) in cases.iter().enumerate() {
        run_case_f32(gpu, rows, cols, true, 0.0, 0x1000 + idx as u64);
        run_case_f32(gpu, rows, cols, false, 0.0, 0x2000 + idx as u64);
        run_case_f32(gpu, rows, cols, true, 0.01, 0x3000 + idx as u64);
    }
}

#[test]
fn muon_cuda_parity_large_lora_shapes_hit_parallel_path() {
    let Some(gpu) = available_gpu("muon_cuda_parity_large_lora_shapes_hit_parallel_path") else {
        return;
    };
    // These rank-8 LoRA shapes exceed KILN_MUON_PARALLEL_THRESHOLD and force
    // the multi-block Gram/Frobenius/apply path used by production Qwen LoRA.
    let cases: &[(usize, usize)] = &[(8, 4096), (4096, 8)];
    for (idx, &(rows, cols)) in cases.iter().enumerate() {
        run_case_f32(gpu, rows, cols, true, 0.0, 0x4000 + idx as u64);
        run_case_f32(gpu, rows, cols, false, 0.01, 0x5000 + idx as u64);
    }
}

#[test]
fn muon_cuda_parity_non_matrix_falls_back_to_momentum_sgd() {
    let Some(gpu) = available_gpu("muon_cuda_parity_non_matrix_falls_back_to_momentum_sgd") else {
        return;
    };
    // [1, N] has min dim 1 -> do_ortho false -> plain (Nesterov) momentum SGD.
    run_case_f32(gpu, 1, 64, true, 0.0, 0xbeef);
    run_case_f32(gpu, 64, 1, false, 0.0, 0xcafe);
}

#[test]
fn muon_cuda_parity_bf16_runs_and_is_close() {
    let Some(gpu) = available_gpu("muon_cuda_parity_bf16_runs_and_is_close") else {
        return;
    };
    // BF16 path: compare against the f32 oracle within a bf16 budget (a
    // lane-addressing bug would corrupt values far beyond bf16 rounding).
    let (rows, cols) = (16usize, 16usize);
    let n = rows * cols;
    let (lr, mom, ns_iters) = (0.02f32, 0.95f32, 5u32);
    let round = |v: f32| bf16::from_f32(v).to_f32();
    let param_data: Vec<f32> = det_fill(0x55, n, 1.0).iter().map(|&v| round(v)).collect();
    let grad_data: Vec<f32> = det_fill(0x66, n, 0.5).iter().map(|&v| round(v)).collect();
    let mom_data: Vec<f32> = det_fill(0x77, n, 0.3).iter().map(|&v| round(v)).collect();

    let (exp_param, _exp_mom) = cpu_muon(
        &param_data,
        &grad_data,
        &mom_data,
        rows,
        cols,
        lr,
        mom,
        true,
        ns_iters,
        0.0,
    );

    let to_bf16 = |d: &[f32]| -> Vec<bf16> { d.iter().map(|&v| bf16::from_f32(v)).collect() };
    let param = device_tensor_from_slice(gpu, &to_bf16(&param_data), vec![rows, cols]);
    let grad = device_tensor_from_slice(gpu, &to_bf16(&grad_data), vec![rows, cols]);
    let momentum = device_tensor_from_slice(gpu, &to_bf16(&mom_data), vec![rows, cols]);

    muon_step_bf16_kt(&param, &grad, &momentum, lr, mom, true, ns_iters, 0.0)
        .expect("muon_step_bf16_kt");

    let got_param = host_f32(gpu, &param);
    for i in 0..n {
        assert!(
            (got_param[i] - exp_param[i]).abs() <= 6e-2 * (1.0 + exp_param[i].abs()),
            "bf16 param[{i}]: gpu={} cpu={}",
            got_param[i],
            exp_param[i]
        );
    }
}
