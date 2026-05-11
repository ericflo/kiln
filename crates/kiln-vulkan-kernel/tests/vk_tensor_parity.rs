//! Parity tests for vk-native training Phase A: VkTensor + autograd
//! tape + element-wise + reductions.
//!
//! Each test computes the same operation via candle on CPU and via
//! the vk-native path on GPU, then asserts max-abs-diff under
//! tolerance. Tests skip cleanly if no Vulkan device is available.

use anyhow::Result;
use candle_core::{Device, Tensor};
use kiln_vulkan_kernel::vk_autograd::vk_backward;
use kiln_vulkan_kernel::vk_ops::cast::{
    vk_cast, vk_cast_bf16_to_f32_no_grad, vk_cast_f32_to_bf16_no_grad,
};
use kiln_vulkan_kernel::vk_ops::elementwise::{vk_add, vk_div, vk_mul, vk_sub};
use kiln_vulkan_kernel::vk_ops::reduce::{vk_mean_all, vk_sum_all};
use kiln_vulkan_kernel::vk_ops::shape::{vk_reshape, vk_transpose_2d, vk_transpose_2d_no_grad};
use kiln_vulkan_kernel::vk_tensor::{VkDType, VkTensor};
use kiln_vulkan_kernel::VulkanDevice;
use std::sync::Arc;

fn vk_dev() -> Option<Arc<VulkanDevice>> {
    if !VulkanDevice::probe() {
        return None;
    }
    VulkanDevice::new().ok().map(Arc::new)
}

fn upload_f32(dev: &Arc<VulkanDevice>, data: &[f32], shape: &[usize]) -> Result<VkTensor> {
    let t = Tensor::from_vec(data.to_vec(), shape.to_vec(), &Device::Cpu)?;
    VkTensor::from_candle(&t, Arc::clone(dev))
}

fn upload_param_f32(
    dev: &Arc<VulkanDevice>,
    data: &[f32],
    shape: &[usize],
) -> Result<(candle_core::Var, VkTensor)> {
    let t = Tensor::from_vec(data.to_vec(), shape.to_vec(), &Device::Cpu)?;
    let var = candle_core::Var::from_tensor(&t)?;
    let vk = VkTensor::from_candle(&t, Arc::clone(dev))?;
    let pid = var.id();
    let param = VkTensor::parameter(
        Arc::clone(vk.buffer()),
        vk.shape().to_vec(),
        vk.dtype(),
        Arc::clone(vk.device()),
        pid,
    );
    Ok((var, param))
}

fn max_abs_diff(got: &[f32], expected: &[f32]) -> f32 {
    got.iter()
        .zip(expected.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0_f32, f32::max)
}

#[test]
fn vk_add_forward_parity() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let a_data: Vec<f32> = (0..16).map(|i| i as f32 * 0.5 - 3.0).collect();
    let b_data: Vec<f32> = (0..16).map(|i| (i as f32 * 0.25).sin()).collect();
    let a = upload_f32(&dev, &a_data, &[4, 4])?;
    let b = upload_f32(&dev, &b_data, &[4, 4])?;
    let y = vk_add(&a, &b)?;
    let got = y.to_vec_f32()?;
    let expected: Vec<f32> = a_data.iter().zip(&b_data).map(|(x, y)| x + y).collect();
    assert!(max_abs_diff(&got, &expected) < 1e-6, "got {got:?} vs {expected:?}");
    Ok(())
}

#[test]
fn vk_sub_mul_div_forward_parity() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let a_data: Vec<f32> = vec![2.0, -3.0, 4.5, -1.25, 0.5, 7.0];
    let b_data: Vec<f32> = vec![1.0, 2.0, 0.5, 0.25, 4.0, -2.0];
    let a = upload_f32(&dev, &a_data, &[6])?;
    let b = upload_f32(&dev, &b_data, &[6])?;

    let sub = vk_sub(&a, &b)?.to_vec_f32()?;
    let exp_sub: Vec<f32> = a_data.iter().zip(&b_data).map(|(x, y)| x - y).collect();
    assert!(max_abs_diff(&sub, &exp_sub) < 1e-6);

    let mul = vk_mul(&a, &b)?.to_vec_f32()?;
    let exp_mul: Vec<f32> = a_data.iter().zip(&b_data).map(|(x, y)| x * y).collect();
    assert!(max_abs_diff(&mul, &exp_mul) < 1e-6);

    let div = vk_div(&a, &b)?.to_vec_f32()?;
    let exp_div: Vec<f32> = a_data.iter().zip(&b_data).map(|(x, y)| x / y).collect();
    assert!(max_abs_diff(&div, &exp_div) < 1e-6);

    Ok(())
}

#[test]
fn vk_sum_all_forward_parity() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let data: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.0001 - 0.05).collect();
    let t = upload_f32(&dev, &data, &[1024])?;
    let summed = vk_sum_all(&t)?;
    assert_eq!(summed.shape(), &[1]);
    let got = summed.to_vec_f32()?;
    let expected: f32 = data.iter().sum();
    assert!((got[0] - expected).abs() < 1e-3, "got {} vs {}", got[0], expected);
    Ok(())
}

#[test]
fn vk_mean_all_forward_parity() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let t = upload_f32(&dev, &data, &[2, 4])?;
    let m = vk_mean_all(&t)?.to_vec_f32()?;
    let expected = data.iter().sum::<f32>() / data.len() as f32;
    assert!((m[0] - expected).abs() < 1e-5, "got {} vs {}", m[0], expected);
    Ok(())
}

/// Backward parity for the chain: loss = mean((x + a) * b)
/// Expected gradients (n = num_elements):
///   d loss / dx = b / n     (since d/dx (x+a)*b = b; mean spreads 1/n)
///   d loss / da = b / n
///   d loss / db = (x + a) / n
#[test]
fn vk_chain_backward_parity() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let a_data: Vec<f32> = vec![0.5, -0.5, 1.0, -1.0];
    let b_data: Vec<f32> = vec![2.0, 3.0, -1.0, 0.5];
    let n = x_data.len() as f32;

    let (_x_var, x) = upload_param_f32(&dev, &x_data, &[4])?;
    let (_a_var, a) = upload_param_f32(&dev, &a_data, &[4])?;
    let (_b_var, b) = upload_param_f32(&dev, &b_data, &[4])?;

    let x_plus_a = vk_add(&x, &a)?;
    let inner = vk_mul(&x_plus_a, &b)?;
    let loss = vk_mean_all(&inner)?;
    let loss_val = loss.to_vec_f32()?;
    let expected_loss: f32 = x_data
        .iter()
        .zip(&a_data)
        .zip(&b_data)
        .map(|((x, a), b)| (x + a) * b)
        .sum::<f32>()
        / n;
    assert!(
        (loss_val[0] - expected_loss).abs() < 1e-5,
        "loss got {} vs {}",
        loss_val[0],
        expected_loss
    );

    let grads = vk_backward(&loss)?;
    assert_eq!(grads.len(), 3, "expected 3 param grads, got {}", grads.len());

    let grad_x = grads.get(x.param_id().unwrap()).expect("dx").to_vec_f32()?;
    let grad_a = grads.get(a.param_id().unwrap()).expect("da").to_vec_f32()?;
    let grad_b = grads.get(b.param_id().unwrap()).expect("db").to_vec_f32()?;

    let exp_dx: Vec<f32> = b_data.iter().map(|b| b / n).collect();
    let exp_da: Vec<f32> = b_data.iter().map(|b| b / n).collect();
    let exp_db: Vec<f32> = x_data
        .iter()
        .zip(&a_data)
        .map(|(x, a)| (x + a) / n)
        .collect();

    assert!(
        max_abs_diff(&grad_x, &exp_dx) < 1e-5,
        "dx {:?} vs {:?}",
        grad_x,
        exp_dx
    );
    assert!(
        max_abs_diff(&grad_a, &exp_da) < 1e-5,
        "da {:?} vs {:?}",
        grad_a,
        exp_da
    );
    assert!(
        max_abs_diff(&grad_b, &exp_db) < 1e-5,
        "db {:?} vs {:?}",
        grad_b,
        exp_db
    );

    Ok(())
}

/// Backward through a parameter used twice in one expression:
/// loss = mean(x * x)  →  d loss / dx = 2x / n
/// Exercises the grad accumulation path (vk_add_no_grad inside vk_backward).
#[test]
fn vk_reused_parameter_grad_accumulates() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let x_data: Vec<f32> = vec![1.0, 2.0, -1.5, 3.5];
    let n = x_data.len() as f32;
    let (_x_var, x) = upload_param_f32(&dev, &x_data, &[4])?;

    // mul(x, x) — same VkTensor on both sides; the MulBackward will
    // return two grads keyed to the same op_id, exercising accumulation.
    let sq = vk_mul(&x, &x)?;
    let loss = vk_mean_all(&sq)?;
    let grads = vk_backward(&loss)?;

    let grad_x = grads.get(x.param_id().unwrap()).expect("dx").to_vec_f32()?;
    let expected: Vec<f32> = x_data.iter().map(|x| 2.0 * x / n).collect();
    assert!(
        max_abs_diff(&grad_x, &expected) < 1e-5,
        "dx {:?} vs {:?}",
        grad_x,
        expected
    );
    Ok(())
}

#[test]
fn vk_detach_drops_grad_link() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let (_x_var, x) = upload_param_f32(&dev, &[1.0, 2.0], &[2])?;
    let y = vk_add(&x, &x)?; // requires_grad
    assert!(y.requires_grad());
    let detached = y.detach();
    assert!(!detached.requires_grad());
    assert!(detached.grad_fn().is_none());
    Ok(())
}

#[test]
fn vk_dtype_byte_size() {
    assert_eq!(VkDType::F32.byte_size(), 4);
    assert_eq!(VkDType::Bf16.byte_size(), 2);
}

// ---- cast tests ----

#[test]
fn vk_cast_f32_bf16_roundtrip() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let data: Vec<f32> = vec![1.0, -2.5, 0.125, 3.75, 100.0, -0.0078125];
    let t = upload_f32(&dev, &data, &[6])?;
    let bf = vk_cast_f32_to_bf16_no_grad(&t)?;
    assert_eq!(bf.dtype(), VkDType::Bf16);
    let back = vk_cast_bf16_to_f32_no_grad(&bf)?.to_vec_f32()?;
    // BF16 has ~7-bit mantissa; values chosen to be exactly representable
    for (i, &expected) in data.iter().enumerate() {
        assert!(
            (back[i] - expected).abs() < 1e-3,
            "idx {i}: {} vs {}",
            back[i],
            expected
        );
    }
    Ok(())
}

#[test]
fn vk_cast_odd_count_packs_correctly() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    // Odd count exercises the high-lane preservation in the packed shader.
    let data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let t = upload_f32(&dev, &data, &[3])?;
    let bf = vk_cast_f32_to_bf16_no_grad(&t)?;
    let back = vk_cast_bf16_to_f32_no_grad(&bf)?.to_vec_f32()?;
    assert_eq!(back.len(), 3);
    for (i, &expected) in data.iter().enumerate() {
        assert!((back[i] - expected).abs() < 1e-3);
    }
    Ok(())
}

#[test]
fn vk_cast_autograd_passthrough() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let n = x_data.len() as f32;
    let (_x_var, x) = upload_param_f32(&dev, &x_data, &[4])?;
    // y = cast(x, bf16); cast(y, f32); loss = mean(y * y)
    // After cast f32->bf16->f32 with values exactly representable in bf16,
    // gradient should be 2*x/n.
    let yb = vk_cast(&x, VkDType::Bf16)?;
    let yf = vk_cast(&yb, VkDType::F32)?;
    let sq = vk_mul(&yf, &yf)?;
    let loss = vk_mean_all(&sq)?;
    let grads = vk_backward(&loss)?;
    let grad_x = grads.get(x.param_id().unwrap()).expect("dx").to_vec_f32()?;
    let expected: Vec<f32> = x_data.iter().map(|x| 2.0 * x / n).collect();
    assert!(
        max_abs_diff(&grad_x, &expected) < 1e-3,
        "dx {:?} vs {:?}",
        grad_x,
        expected
    );
    Ok(())
}

// ---- shape tests ----

#[test]
fn vk_reshape_preserves_data() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = upload_f32(&dev, &data, &[2, 3, 4])?;
    let r = vk_reshape(&t, &[6, 4])?;
    assert_eq!(r.shape(), &[6, 4]);
    let back = r.to_vec_f32()?;
    assert_eq!(back, data);
    Ok(())
}

#[test]
fn vk_transpose_2d_forward_parity() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    // 3x4 matrix
    let data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, // row 0
        5.0, 6.0, 7.0, 8.0, // row 1
        9.0, 10.0, 11.0, 12.0, // row 2
    ];
    let t = upload_f32(&dev, &data, &[3, 4])?;
    let tt = vk_transpose_2d_no_grad(&t)?;
    assert_eq!(tt.shape(), &[4, 3]);
    let back = tt.to_vec_f32()?;
    // Expected: column-major view, 4 rows x 3 cols
    let expected: Vec<f32> = vec![
        1.0, 5.0, 9.0, // col 0 of original
        2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
    ];
    assert_eq!(back, expected);
    Ok(())
}

#[test]
fn vk_transpose_2d_autograd() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let n = x_data.len() as f32;
    let (_x_var, x) = upload_param_f32(&dev, &x_data, &[2, 3])?;
    // y = x.T; loss = mean(y * y) = mean(x * x) (same elements)
    let yt = vk_transpose_2d(&x)?;
    let sq = vk_mul(&yt, &yt)?;
    let loss = vk_mean_all(&sq)?;
    let grads = vk_backward(&loss)?;
    let grad_x = grads.get(x.param_id().unwrap()).expect("dx").to_vec_f32()?;
    let expected: Vec<f32> = x_data.iter().map(|x| 2.0 * x / n).collect();
    assert!(
        max_abs_diff(&grad_x, &expected) < 1e-5,
        "dx {:?} vs {:?}",
        grad_x,
        expected
    );
    Ok(())
}

#[test]
fn vk_reshape_autograd_passthrough() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let x_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.5).collect();
    let n = x_data.len() as f32;
    let (_x_var, x) = upload_param_f32(&dev, &x_data, &[3, 4])?;
    // y = x.reshape([12]); loss = mean(y * y)
    let r = vk_reshape(&x, &[12])?;
    let sq = vk_mul(&r, &r)?;
    let loss = vk_mean_all(&sq)?;
    let grads = vk_backward(&loss)?;
    let grad_x = grads.get(x.param_id().unwrap()).expect("dx").to_vec_f32()?;
    let expected: Vec<f32> = x_data.iter().map(|x| 2.0 * x / n).collect();
    assert!(max_abs_diff(&grad_x, &expected) < 1e-5);
    Ok(())
}
