//! Parity test for the fused GDN gated RMSNorm CUDA kernel.
//!
//! The kernel in `kiln_gdn_kernel::gdn_gated_rms_norm` replaces the
//! candle chain `to_f32 -> rms_norm -> silu -> mul -> bf16` used by
//! `kiln-model` in the `kiln/gdn/gated_norm` NVTX range.

use candle_core::{DType, Device, Tensor};
use kiln_gdn_kernel::{gdn_gated_rms_norm, gdn_gated_rms_norm_supports};

fn lcg_seed(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = ((*state >> 33) as u32) & 0x7fff_ffff;
    (bits as f32 / (i32::MAX as f32)) - 0.5
}

fn fill(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..n).map(|_| lcg_seed(&mut state) * scale).collect()
}

fn sigmoid(x: &Tensor) -> candle_core::Result<Tensor> {
    let neg_x = x.neg()?;
    let exp_neg_x = neg_x.exp()?;
    let one_plus = (exp_neg_x + 1.0)?;
    one_plus.recip()
}

fn silu(x: &Tensor) -> candle_core::Result<Tensor> {
    Ok((x * sigmoid(x)?)?)
}

fn reference(x: &Tensor, z: &Tensor, weight: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let z_f32 = z.to_dtype(DType::F32)?;
    let w_f32 = weight.to_dtype(DType::F32)?;
    let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let rms_inv = (variance + eps)?.sqrt()?.recip()?;
    let normed = x_f32.broadcast_mul(&rms_inv)?.broadcast_mul(&w_f32)?;
    Ok((normed * silu(&z_f32)?)?)
}

fn run_case(device: &Device, batch: usize, seq_len: usize, heads: usize, hidden: usize, seed: u64, label: &str) {
    let elems = batch * seq_len * heads * hidden;
    let x_data = fill(seed ^ 0xA11C_E5, elems, 2.0);
    let z_data = fill(seed ^ 0x6A7E, elems, 4.0);
    let w_data: Vec<f32> = fill(seed ^ 0xBEEF, hidden, 0.5)
        .into_iter()
        .map(|v| v + 1.0)
        .collect();

    let x = Tensor::from_vec(x_data, (batch, seq_len, heads, hidden), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(device)
        .unwrap();
    let z = Tensor::from_vec(z_data, (batch, seq_len, heads, hidden), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(device)
        .unwrap();
    let weight = Tensor::from_vec(w_data, (hidden,), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(device)
        .unwrap();

    assert!(
        gdn_gated_rms_norm_supports(&x, &z, &weight),
        "{label}: envelope check failed"
    );

    let fused = gdn_gated_rms_norm(&x, &z, &weight, 1e-6).expect("fused gated RMSNorm");
    let fallback = reference(&x, &z, &weight, 1e-6).expect("fallback gated RMSNorm");

    assert_eq!(fused.dims(), fallback.dims());
    assert_eq!(fused.dtype(), DType::BF16);

    let diff = (fused.to_dtype(DType::F32).unwrap()
        - fallback.to_dtype(DType::BF16).unwrap().to_dtype(DType::F32).unwrap())
    .unwrap();
    let abs = diff.abs().unwrap();
    let max = abs.flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
    let mean = abs.flatten_all().unwrap().mean(0).unwrap().to_scalar::<f32>().unwrap();
    println!("[{label}] shape=[{batch},{seq_len},{heads},{hidden}] max_abs={max:.3e} mean_abs={mean:.3e}");
    assert!(max < 5e-3, "{label}: max_abs_diff {max} exceeds tolerance");
    assert!(mean < 5e-4, "{label}: mean_abs_diff {mean} exceeds tolerance");
}

#[test]
fn gdn_gated_rms_norm_parity_vs_candle_reference() {
    let device = match Device::new_cuda(0) {
        Ok(device) => device,
        Err(err) => {
            eprintln!("Skipping gdn_gated_rms_norm parity test: no CUDA device ({err})");
            return;
        }
    };

    run_case(&device, 1, 1, 32, 128, 0xCAFE_F00D, "decode");
    run_case(&device, 1, 64, 32, 128, 0xDEAD_BEEF, "prefill/T=64");
}
