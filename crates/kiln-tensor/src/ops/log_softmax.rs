//! Log-softmax along the trailing axis.
//!
//! Numerically stable formulation: `log_softmax = x - log_sum_exp(x)`
//! after the max-subtraction step. Used together with `nll_loss` as
//! the two-step alternative to `cross_entropy`.

use std::sync::Arc;

use crate::{CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId, bail};

/// Numerically stable log-softmax that preserves the input dtype.
///
/// Use [`log_softmax_last_dim_f32`] when downstream consumers need full F32
/// log-probability precision from BF16 or F16 logits.
pub fn log_softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    log_softmax_last_dim_impl(x, false)
}

/// Numerically stable log-softmax with an F32 result.
///
/// F32, BF16, and F16 inputs are read directly in their storage dtype. The
/// reduction and final subtraction are performed in F32, and the result stays
/// in F32 on the input device. In particular, reduced-precision inputs are not
/// first expanded into a same-shape F32 tensor and the result is never rounded
/// through the input dtype.
pub fn log_softmax_last_dim_f32(x: &Tensor) -> Result<Tensor> {
    log_softmax_last_dim_impl(x, true)
}

fn log_softmax_last_dim_impl(x: &Tensor, output_f32: bool) -> Result<Tensor> {
    let label = if output_f32 {
        "log_softmax_last_dim_f32"
    } else {
        "log_softmax_last_dim"
    };
    if x.rank() == 0 {
        bail!("{label}: input must have rank >= 1");
    }
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("{label}: dtype must be F32/BF16/F16, got {}", x.dtype());
    }
    if !x.is_contiguous() {
        bail!("{label}: input must be contiguous");
    }
    if x.shape().last() == Some(&0) {
        bail!("{label}: trailing axis must be non-empty");
    }
    // A softmax-then-log composition is not a valid log-softmax
    // implementation: tail probabilities can round to zero even in F32 and
    // then become -Inf. The fused kernels form the stable max-subtracted
    // identity directly and allocate only the output tensor.
    #[cfg(feature = "cuda")]
    if matches!(x.device(), crate::Device::Cuda(_)) {
        return if output_f32 {
            crate::cuda_log_softmax_last_axis_f32(x)
        } else {
            crate::cuda_log_softmax_last_axis(x)
        };
    }

    #[cfg(feature = "rocm")]
    if matches!(x.device(), crate::Device::Rocm(_)) {
        return if output_f32 {
            crate::rocm_log_softmax_last_axis_f32(x)
        } else {
            crate::rocm_log_softmax_last_axis(x)
        };
    }

    // Metal fast path: kiln-owned MSL log-softmax kernel (numerically
    // stable y_i = x_i - logsumexp(row)). Mirrors the CUDA fast path and
    // the CPU reference below; required by the OPD forward composite
    // (`per_position_forward_kt`) on Metal storage (#1082 OPD lane).
    #[cfg(feature = "metal")]
    if matches!(x.device(), crate::Device::Metal(_)) {
        return if output_f32 {
            crate::metal_log_softmax_last_axis_f32(x)
        } else {
            crate::metal_log_softmax_last_axis(x)
        };
    }

    // Vulkan host-fallback (#1082 PR3c): there is no Vulkan log-softmax
    // (or standalone `log` activation) kernel in `vk_ops` today — only a
    // forward softmax (`vk_softmax_lastdim_no_grad`), which can't be
    // composed into a numerically-stable log-softmax without a `log`
    // shader. Until such a kernel lands, stage to host, run the CPU
    // reference below, and move the result back to the Vulkan device.
    //
    // Unlike `sum_axis` / `*_scalar` (DeviceOp1s whose `dispatch1`
    // host-fallback in `device_op.rs` owns the bounce), `log_softmax` is
    // a free function, so the round-trip is wired explicitly here. This
    // is correctness-first / perf-wrong; the residual TODO is to author a
    // `vk_ops::softmax::vk_log_softmax_lastdim` shader and route through
    // it.
    #[cfg(feature = "vulkan")]
    if matches!(x.device(), crate::Device::Vulkan(_)) {
        let dev = x.device();
        let cpu_in = x.to_device(crate::Device::Cpu)?;
        let cpu_out = if output_f32 {
            log_softmax_last_dim_f32(&cpu_in)?
        } else {
            log_softmax_last_dim(&cpu_in)?
        };
        return cpu_out.to_device(dev);
    }

    log_softmax_last_dim_cpu(x, output_f32, label)
}

fn cpu_value(bytes: &[u8], dtype: DType, index: usize) -> f32 {
    match dtype {
        DType::F32 => f32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap()),
        DType::BF16 => {
            half::bf16::from_le_bytes(bytes[index * 2..index * 2 + 2].try_into().unwrap()).to_f32()
        }
        DType::F16 => {
            half::f16::from_le_bytes(bytes[index * 2..index * 2 + 2].try_into().unwrap()).to_f32()
        }
        _ => unreachable!("input dtype validated by log_softmax_last_dim_impl"),
    }
}

fn log_softmax_last_dim_cpu(x: &Tensor, output_f32: bool, label: &str) -> Result<Tensor> {
    let input_dtype = x.dtype();
    let output_dtype = if output_f32 { DType::F32 } else { input_dtype };
    let shape = x.shape().to_vec();
    let last = *shape.last().unwrap();
    let outer = x.element_count() / last;
    let cpu = x
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::Msg(format!("{label}: storage must be CpuStorage")))?;
    let bytes = cpu.as_bytes();
    let mut out = vec![0u8; x.element_count() * output_dtype.size_in_bytes()];

    for r in 0..outer {
        let row_start = r * last;
        let mut m = f32::NEG_INFINITY;
        for i in 0..last {
            m = m.max(cpu_value(bytes, input_dtype, row_start + i));
        }

        let mut sum_e = 0.0_f32;
        for i in 0..last {
            sum_e += (cpu_value(bytes, input_dtype, row_start + i) - m).exp();
        }
        let log_sum_e = sum_e.ln();

        for i in 0..last {
            // Keep the max subtraction separate. Reconstructing
            // `m + log(sum_e)` first can round the normalization term away
            // when the row has a large common offset.
            let y = (cpu_value(bytes, input_dtype, row_start + i) - m) - log_sum_e;
            let idx = row_start + i;
            match output_dtype {
                DType::F32 => out[idx * 4..idx * 4 + 4].copy_from_slice(&y.to_le_bytes()),
                DType::BF16 => out[idx * 2..idx * 2 + 2]
                    .copy_from_slice(&half::bf16::from_f32(y).to_le_bytes()),
                DType::F16 => {
                    out[idx * 2..idx * 2 + 2].copy_from_slice(&half::f16::from_f32(y).to_le_bytes())
                }
                _ => unreachable!(),
            }
        }
    }
    let cpu_out = CpuStorage::from_bytes(output_dtype, out)?;
    let storage: Storage = Arc::new(cpu_out);
    Tensor::from_parts(storage, Layout::contiguous(shape), TensorId::next())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn approx(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < tol, "idx {i}: got {x}, want {y}");
        }
    }

    #[test]
    fn log_softmax_uniform() {
        // x = zeros[3]; softmax = [1/3]*3; log_softmax = log(1/3) each.
        let x = Tensor::from_slice(&[0.0f32, 0.0, 0.0], vec![1, 3]).unwrap();
        let y = read_f32(&log_softmax_last_dim(&x).unwrap());
        approx(&y, &[(1.0_f32 / 3.0).ln(); 3], 1e-5);
    }

    #[test]
    fn log_softmax_sums_to_log_probs() {
        // The result exponentiated should sum to 1 per row.
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let y = read_f32(&log_softmax_last_dim(&x).unwrap());
        let probs_sum: f32 = y.iter().map(|v| v.exp()).sum();
        assert!((probs_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn log_softmax_compose_with_nll_matches_cross_entropy() {
        use crate::ops::{cross_entropy, nll_loss};
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 0.5, 1.5], vec![2, 3]).unwrap();
        let targets = Tensor::from_slice(&[0i64, 2], vec![2]).unwrap();
        let ce = cross_entropy(&logits, &targets).unwrap();
        let log_p = log_softmax_last_dim(&logits).unwrap();
        let nll = nll_loss(&log_p, &targets).unwrap();
        // ce and nll(log_softmax) should match within float tolerance.
        let cpu_ce = ce.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let cpu_nll = nll.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let ce_v = f32::from_le_bytes(cpu_ce.as_bytes()[..4].try_into().unwrap());
        let nll_v = f32::from_le_bytes(cpu_nll.as_bytes()[..4].try_into().unwrap());
        assert!((ce_v - nll_v).abs() < 1e-5, "ce={ce_v}, nll={nll_v}");
    }

    #[test]
    fn log_softmax_numerically_stable_with_extreme_spread() {
        // The -1000 tail underflows in a softmax-then-log implementation.
        let x = Tensor::from_slice(&[1000.0f32, 0.0, -1000.0], vec![1, 3]).unwrap();
        let y = read_f32(&log_softmax_last_dim(&x).unwrap());
        for v in &y {
            assert!(v.is_finite(), "log_softmax produced non-finite: {v}");
        }
        approx(&y, &[0.0, -1000.0, -2000.0], 1e-4);
    }

    #[test]
    fn log_softmax_is_stable_under_large_common_offset() {
        let expected = -(3.0_f32).ln();

        let f32_input = Tensor::from_slice(&[100_000_000.0_f32; 3], vec![1, 3]).unwrap();
        let same_dtype = log_softmax_last_dim(&f32_input).unwrap();
        approx(&read_f32(&same_dtype), &[expected; 3], 1e-6);

        let f32_output = log_softmax_last_dim_f32(&f32_input).unwrap();
        approx(&read_f32(&f32_output), &[expected; 3], 1e-6);

        let bf16_values = vec![half::bf16::from_f32(100_000_000.0); 3];
        let bf16_input = Tensor::from_slice(&bf16_values, vec![1, 3]).unwrap();
        let mixed_output = log_softmax_last_dim_f32(&bf16_input).unwrap();
        approx(&read_f32(&mixed_output), &[expected; 3], 1e-6);

        let preserved_output = log_softmax_last_dim(&bf16_input).unwrap();
        let preserved_values = preserved_output.to_vec::<half::bf16>().unwrap();
        let expected_bf16 = half::bf16::from_f32(expected);
        assert_eq!(preserved_values, vec![expected_bf16; 3]);

        let f16_values = vec![half::f16::from_f32(65_504.0); 3];
        let f16_input = Tensor::from_slice(&f16_values, vec![1, 3]).unwrap();
        let mixed_output = log_softmax_last_dim_f32(&f16_input).unwrap();
        approx(&read_f32(&mixed_output), &[expected; 3], 1e-6);

        let preserved_output = log_softmax_last_dim(&f16_input).unwrap();
        let preserved_values = preserved_output.to_vec::<half::f16>().unwrap();
        let expected_f16 = half::f16::from_f32(expected);
        assert_eq!(preserved_values, vec![expected_f16; 3]);
    }

    #[test]
    fn log_softmax_bf16_round_trip() {
        let bf: Vec<half::bf16> = [1.0f32, 2.0, 3.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let x = Tensor::from_slice(&bf, vec![1, 3]).unwrap();
        let y = log_softmax_last_dim(&x).unwrap();
        assert_eq!(y.dtype(), DType::BF16);
    }

    #[test]
    fn log_softmax_f32_accepts_all_float_input_dtypes() {
        let f32_input = Tensor::from_slice(&[1000.0f32, 0.0, -1000.0], vec![1, 3]).unwrap();
        let bf16_values = [100.0f32, 0.0, -100.0]
            .into_iter()
            .map(half::bf16::from_f32)
            .collect::<Vec<_>>();
        let bf16_input = Tensor::from_slice(&bf16_values, vec![1, 3]).unwrap();
        let f16_values = [20.0f32, 0.0, -20.0]
            .into_iter()
            .map(half::f16::from_f32)
            .collect::<Vec<_>>();
        let f16_input = Tensor::from_slice(&f16_values, vec![1, 3]).unwrap();

        for (input, expected) in [
            (f32_input, [0.0, -1000.0, -2000.0]),
            (bf16_input, [0.0, -100.0, -200.0]),
            (f16_input, [0.0, -20.0, -40.0]),
        ] {
            let output = log_softmax_last_dim_f32(&input).unwrap();
            assert_eq!(output.dtype(), DType::F32);
            assert_eq!(output.device(), input.device());
            assert_eq!(output.shape(), input.shape());
            let values = read_f32(&output);
            assert!(values.iter().all(|value| value.is_finite()));
            approx(&values, &expected, 1e-4);
        }
    }

    #[test]
    fn log_softmax_rejects_empty_trailing_axis_for_both_output_contracts() {
        let input = Tensor::from_slice::<f32>(&[], vec![2, 0]).unwrap();
        for result in [
            log_softmax_last_dim(&input),
            log_softmax_last_dim_f32(&input),
        ] {
            let error = result.unwrap_err().to_string();
            assert!(error.contains("trailing axis must be non-empty"), "{error}");
        }
    }

    #[test]
    fn log_softmax_f32_bf16_result_is_not_post_quantized() {
        let input_values = vec![half::bf16::from_f32(0.0); 3];
        let input = Tensor::from_slice(&input_values, vec![1, 3]).unwrap();
        let output = log_softmax_last_dim_f32(&input).unwrap();
        let values = read_f32(&output);
        let expected = -(3.0f32).ln();
        let bf16_round_trip = half::bf16::from_f32(expected).to_f32();

        approx(&values, &[expected; 3], 1e-6);
        assert!(
            values
                .iter()
                .all(|value| (value - bf16_round_trip).abs() > 1e-4),
            "result was rounded through BF16: {values:?}"
        );
    }

    // ------------------------------------------------------------------
    // PR3c (#1082): Vulkan host-fallback parity vs CPU reference.
    //
    // No Vulkan log-softmax kernel exists; log_softmax_last_dim stages to
    // host, runs the CPU reference, and moves the result back. This
    // validates that round-trip is numerically identical (bytes are not
    // touched on the GPU) and stays on the Vulkan device. Bounded:
    // tiny F32 input, single-shot, gated on KILN_TENSOR_VULKAN_TEST or strict
    // KILN_QUALIFICATION mode.
    // ------------------------------------------------------------------

    #[cfg(feature = "vulkan")]
    fn vulkan_test_enabled() -> bool {
        std::env::var("KILN_TENSOR_VULKAN_TEST").ok().as_deref() == Some("1")
            || std::env::var("KILN_QUALIFICATION").ok().as_deref() == Some("1")
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn log_softmax_vulkan_host_fallback_parity() {
        if !vulkan_test_enabled() {
            eprintln!("skip: KILN_TENSOR_VULKAN_TEST unset");
            return;
        }
        if crate::primary_vulkan_device(0).is_err() {
            if std::env::var("KILN_QUALIFICATION").ok().as_deref() == Some("1") {
                panic!("Vulkan device unavailable while KILN_QUALIFICATION=1");
            }
            eprintln!("skip: no Vulkan device");
            return;
        }
        let dev = crate::Device::Vulkan(0);
        let data: Vec<f32> = vec![1000.0, 0.0, -1000.0, 4.0, 0.5, 1.5];

        // CPU reference.
        let x_cpu = Tensor::from_slice(&data, vec![2, 3]).unwrap();
        let ref_vals = read_f32(&log_softmax_last_dim(&x_cpu).unwrap());

        // Vulkan input -> host-fallback round-trip.
        let x_vk = Tensor::from_vec_on(dev, data.clone(), vec![2, 3]).unwrap();
        let y_vk = log_softmax_last_dim(&x_vk).unwrap();
        assert_eq!(y_vk.device(), dev, "result must stay on Vulkan");

        let got = y_vk
            .to_device(crate::Device::Cpu)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let mut max_abs_err = 0.0f32;
        for (g, r) in got.iter().zip(ref_vals.iter()) {
            max_abs_err = max_abs_err.max((g - r).abs());
        }
        eprintln!("log_softmax vulkan host-fallback max_abs_err = {max_abs_err:e}");
        assert!(max_abs_err < 1e-4, "max_abs_err {max_abs_err} >= 1e-4");
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn log_softmax_f32_vulkan_bf16_host_fallback_preserves_precision() {
        if !vulkan_test_enabled() {
            eprintln!("skip: KILN_TENSOR_VULKAN_TEST unset");
            return;
        }
        if crate::primary_vulkan_device(0).is_err() {
            eprintln!("skip: no Vulkan device");
            return;
        }
        let dev = crate::Device::Vulkan(0);
        let data = vec![half::bf16::from_f32(0.0); 3];
        let input = Tensor::from_vec_on(dev, data, vec![1, 3]).unwrap();
        let output = log_softmax_last_dim_f32(&input).unwrap();
        assert_eq!(output.device(), dev);
        assert_eq!(output.dtype(), DType::F32);

        let values = output
            .to_device(crate::Device::Cpu)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let expected = -(3.0f32).ln();
        let bf16_round_trip = half::bf16::from_f32(expected).to_f32();
        approx(&values, &[expected; 3], 1e-6);
        assert!(
            values
                .iter()
                .all(|value| (value - bf16_round_trip).abs() > 1e-4)
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn log_softmax_vulkan_host_fallback_preserves_large_offset_normalizer() {
        if !vulkan_test_enabled() {
            eprintln!("skip: KILN_TENSOR_VULKAN_TEST unset");
            return;
        }
        if crate::primary_vulkan_device(0).is_err() {
            if std::env::var("KILN_QUALIFICATION").ok().as_deref() == Some("1") {
                panic!("Vulkan device unavailable while KILN_QUALIFICATION=1");
            }
            eprintln!("skip: no Vulkan device");
            return;
        }

        let dev = crate::Device::Vulkan(0);
        let expected = -(3.0_f32).ln();
        for (dtype, offset, same_dtype_tolerance) in [
            (DType::F32, 100_000_000.0_f32, 1e-6),
            (DType::BF16, 100_000_000.0_f32, 5e-3),
            (DType::F16, 65_504.0_f32, 5e-4),
        ] {
            let cpu_input = Tensor::from_slice(&[offset; 3], vec![1, 3])
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let input = cpu_input.to_device(dev).unwrap();

            let f32_output = log_softmax_last_dim_f32(&input).unwrap();
            assert_eq!(f32_output.device(), dev);
            let f32_values = f32_output
                .to_device(crate::Device::Cpu)
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            approx(&f32_values, &[expected; 3], 1e-6);

            let same_dtype_output = log_softmax_last_dim(&input).unwrap();
            assert_eq!(same_dtype_output.device(), dev);
            let same_dtype_values = same_dtype_output
                .to_device(crate::Device::Cpu)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            approx(&same_dtype_values, &[expected; 3], same_dtype_tolerance);
        }
    }
}
