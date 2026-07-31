use kiln_tensor::{DType, Device, Result, Tensor, ops};

#[test]
fn documented_cpu_matmul_contract() -> Result<()> {
    let lhs = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], (2, 2))?;
    let rhs = Tensor::from_slice(&[5.0_f32, 6.0, 7.0, 8.0], (2, 2))?;

    let output = ops::matmul(&lhs, &rhs)?;

    assert_eq!(output.device(), Device::Cpu);
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.shape(), &[2, 2]);
    assert_eq!(output.to_vec::<f32>()?, vec![19.0, 22.0, 43.0, 50.0]);
    Ok(())
}

#[test]
fn documented_shape_and_dtype_failures_are_explicit() {
    let shape_error = Tensor::from_slice(&[1.0_f32, 2.0, 3.0], (2, 2))
        .expect_err("three values cannot fill a 2x2 tensor");
    assert!(
        shape_error
            .to_string()
            .contains("has 4 elements but slice has 3")
    );

    let tensor = Tensor::from_slice(&[1.0_f32, 2.0], (2,)).expect("valid tensor");
    let dtype_error = tensor
        .to_vec::<u32>()
        .expect_err("readback cannot reinterpret f32 as u32");
    assert!(dtype_error.to_string().contains("no implicit cast"));
}
