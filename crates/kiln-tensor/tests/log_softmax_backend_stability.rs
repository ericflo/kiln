//! Extreme-range log-softmax coverage for CUDA and ROCm.
//!
//! Run on the target machine with one of:
//! `KILN_QUALIFICATION=1 cargo test -p kiln-tensor --features rocm --test log_softmax_backend_stability`
//! `KILN_QUALIFICATION=1 cargo test -p kiln-tensor --features cuda --test log_softmax_backend_stability`
#![cfg(any(feature = "cuda", feature = "rocm"))]

use kiln_tensor::{DType, Device, Tensor, ops};

fn qualification_required(value: Option<&str>) -> bool {
    value == Some("1")
}

fn device_available_or_skip(available: bool, backend: &str) -> bool {
    if available {
        return true;
    }
    if qualification_required(std::env::var("KILN_QUALIFICATION").ok().as_deref()) {
        panic!("{backend} device unavailable while KILN_QUALIFICATION=1");
    }
    eprintln!("no {backend} device available; skipping log-softmax stability test");
    false
}

#[test]
fn qualification_mode_is_exact_opt_in() {
    assert!(qualification_required(Some("1")));
    assert!(!qualification_required(None));
    assert!(!qualification_required(Some("")));
    assert!(!qualification_required(Some("0")));
    assert!(!qualification_required(Some("true")));
}

fn values_f32(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_dtype(DType::F32)
        .expect("promote output")
        .to_device(Device::Cpu)
        .expect("copy output to host")
        .to_vec2::<f32>()
        .expect("read output")
        .into_iter()
        .flatten()
        .collect()
}

fn assert_extreme_tail_stays_finite(device: Device) {
    let cases = [
        (DType::F32, vec![1000.0f32, 0.0, -1000.0], 1e-4),
        (DType::BF16, vec![100.0f32, 0.0, -100.0], 1.0),
        (DType::F16, vec![20.0f32, 0.0, -20.0], 0.1),
    ];

    for (dtype, data, tolerance) in cases {
        let cpu_logits = Tensor::from_slice(&data, vec![1, data.len()])
            .expect("allocate CPU logits")
            .to_dtype(dtype)
            .expect("cast CPU logits");
        let expected = values_f32(&ops::log_softmax_last_dim(&cpu_logits).expect("CPU reference"));

        let logits = Tensor::from_vec_on(device, data.clone(), vec![1, data.len()])
            .expect("allocate logits")
            .to_dtype(dtype)
            .expect("cast logits");
        let output = ops::log_softmax_last_dim(&logits).expect("stable log-softmax");
        assert_eq!(output.dtype(), dtype);
        assert_eq!(output.device(), device);
        assert_eq!(output.shape(), logits.shape());

        let values = values_f32(&output);
        assert!(
            values.iter().all(|value| value.is_finite()),
            "{device} {dtype} produced non-finite log-probabilities: {values:?}"
        );
        for (got, expected) in values.iter().zip(&expected) {
            assert!(
                (got - expected).abs() <= tolerance,
                "{device} {dtype}: got {got}, expected {expected}; row={values:?}"
            );
        }
    }
}

fn assert_f32_output_contract(device: Device) {
    let cases = [
        (DType::F32, vec![1000.0f32, 0.0, -1000.0], 1e-4),
        (DType::BF16, vec![100.0f32, 0.0, -100.0], 1e-4),
        (DType::F16, vec![20.0f32, 0.0, -20.0], 1e-4),
    ];

    for (dtype, data, tolerance) in cases {
        let cpu_logits = Tensor::from_slice(&data, vec![1, data.len()])
            .expect("allocate CPU logits")
            .to_dtype(dtype)
            .expect("cast CPU logits");
        let expected = values_f32(
            &ops::log_softmax_last_dim_f32(&cpu_logits).expect("CPU F32-output reference"),
        );

        let logits = Tensor::from_vec_on(device, data.clone(), vec![1, data.len()])
            .expect("allocate device logits")
            .to_dtype(dtype)
            .expect("cast device logits");
        let output = ops::log_softmax_last_dim_f32(&logits).expect("device F32-output log-softmax");
        assert_eq!(output.dtype(), DType::F32);
        assert_eq!(output.device(), device);
        assert_eq!(output.shape(), logits.shape());

        let values = values_f32(&output);
        assert!(
            values.iter().all(|value| value.is_finite()),
            "{device} {dtype} produced non-finite F32 log-probabilities: {values:?}"
        );
        for (got, expected) in values.iter().zip(&expected) {
            assert!(
                (got - expected).abs() <= tolerance,
                "{device} {dtype}: got {got}, expected {expected}; row={values:?}"
            );
        }
    }

    // Uniform BF16 input has an F32 log-softmax value that is not exactly
    // representable in BF16. This distinguishes a direct F32 kernel output
    // from a same-dtype result followed by a cast to F32.
    let input = Tensor::from_vec_on(device, vec![half::bf16::from_f32(0.0); 3], vec![1, 3])
        .expect("allocate BF16 precision probe");
    let output = ops::log_softmax_last_dim_f32(&input).expect("F32 precision probe");
    let values = values_f32(&output);
    let expected = -(3.0f32).ln();
    let bf16_round_trip = half::bf16::from_f32(expected).to_f32();
    for value in values {
        assert!(
            (value - expected).abs() <= 1e-5,
            "{device}: got {value}, expected {expected}"
        );
        assert!(
            (value - bf16_round_trip).abs() > 1e-4,
            "{device}: F32 output was rounded through BF16 ({value})"
        );
    }
}

fn assert_special_value_semantics(device: Device) {
    let cases = [
        (
            vec![0.0f32, f32::NEG_INFINITY],
            vec![0.0, f32::NEG_INFINITY],
        ),
        (
            vec![f32::NEG_INFINITY, f32::NEG_INFINITY],
            vec![f32::NAN, f32::NAN],
        ),
        (vec![f32::INFINITY, 0.0], vec![f32::NAN, f32::NAN]),
    ];

    for (data, expected) in cases {
        let logits =
            Tensor::from_vec_on(device, data.clone(), vec![1, data.len()]).expect("device logits");
        for (label, values) in [
            (
                "same-dtype",
                values_f32(&ops::log_softmax_last_dim(&logits).expect("log-softmax")),
            ),
            (
                "F32-output",
                values_f32(
                    &ops::log_softmax_last_dim_f32(&logits).expect("F32-output log-softmax"),
                ),
            ),
        ] {
            for (got, expected) in values.iter().zip(&expected) {
                if expected.is_nan() {
                    assert!(
                        got.is_nan(),
                        "{device} {label}: expected NaN, got {got}; row={values:?}"
                    );
                } else {
                    assert_eq!(*got, *expected, "{device} {label}: row={values:?}");
                }
            }
        }
    }
}

fn assert_width_and_row_sweep(device: Device) {
    let widths = [
        1usize, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1023, 1024, 1025, 4097, 248_320,
    ];

    for width in widths {
        let rows = 2;
        let data = (0..rows * width)
            .map(|index| {
                let row = index / width;
                let column = index % width;
                ((column * 17 + row * 29) % 257) as f32 - 128.0
            })
            .collect::<Vec<_>>();
        let shape = vec![rows, width];
        let cpu_logits = Tensor::from_slice(&data, shape.clone()).expect("CPU sweep logits");
        let expected =
            values_f32(&ops::log_softmax_last_dim(&cpu_logits).expect("CPU sweep log-softmax"));

        let logits = Tensor::from_vec_on(device, data, shape).expect("device sweep logits");
        let values =
            values_f32(&ops::log_softmax_last_dim(&logits).expect("device sweep log-softmax"));
        assert_eq!(values.len(), expected.len(), "{device} width {width}");
        let max_abs_error = values
            .iter()
            .zip(&expected)
            .map(|(got, expected)| (got - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_error <= 1e-4,
            "{device} width {width}: max_abs_error={max_abs_error:e}"
        );

        let f32_output =
            ops::log_softmax_last_dim_f32(&logits).expect("device F32-output sweep log-softmax");
        assert_eq!(f32_output.dtype(), DType::F32);
        let f32_values = values_f32(&f32_output);
        let f32_max_abs_error = f32_values
            .iter()
            .zip(&expected)
            .map(|(got, expected)| (got - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(
            f32_max_abs_error <= 1e-4,
            "{device} F32-output width {width}: max_abs_error={f32_max_abs_error:e}"
        );
    }
}

#[cfg(feature = "rocm")]
#[test]
fn rocm_log_softmax_extreme_tail_stays_finite() {
    if !device_available_or_skip(kiln_tensor::rocm_is_available(), "ROCm") {
        return;
    }
    assert_extreme_tail_stays_finite(Device::Rocm(0));
    assert_f32_output_contract(Device::Rocm(0));
    assert_special_value_semantics(Device::Rocm(0));
    assert_width_and_row_sweep(Device::Rocm(0));
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_log_softmax_extreme_tail_stays_finite() {
    if !device_available_or_skip(kiln_tensor::cuda_is_available(), "CUDA") {
        return;
    }
    assert_extreme_tail_stays_finite(Device::Cuda(0));
    assert_f32_output_contract(Device::Cuda(0));
    assert_special_value_semantics(Device::Cuda(0));
    assert_width_and_row_sweep(Device::Cuda(0));
}
