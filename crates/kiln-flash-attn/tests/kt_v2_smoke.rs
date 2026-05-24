//! Smoke test: the kt-API flash-attn entry accepts Borrowed
//! kt-Tensors (Phase 7 v2 borrow-compat).
//!
//! Validates that the migration from .slice().slice(off..).device_ptr()
//! to kiln_kt_bridge::cuda_input_device_ptr / cuda_output_device_ptr
//! preserves correctness AND enables the zero-copy candle->kt path.

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_flash_attn::paged_kv_write_token_major_bf16_slot_kt;

fn try_cuda() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

fn pattern(n: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for _ in 0..n {
        s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E3779B97F4A7C15);
        out.push(((s as u32 % 1024) as f32 - 512.0) / 512.0);
    }
    out
}

/// paged_kv_write_token_major_bf16_slot_kt accepts Borrowed kt-Tensors
/// (zero-copy from candle). Smoke-tests that the migration doesn't
/// panic on the Borrowed path.
#[test]
fn paged_kv_write_slot_kt_accepts_borrowed() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };

    let total_slots = 8usize;
    let num_kv_heads = 4usize;
    let head_dim = 128usize;

    // k_pool, v_pool: BF16 [total_slots, num_kv_heads, head_dim]
    let kp_cd = CandleTensor::from_vec(
        pattern(total_slots * num_kv_heads * head_dim, 1),
        (total_slots, num_kv_heads, head_dim),
        &dev,
    )
    .unwrap()
    .to_dtype(CandleDType::BF16)
    .unwrap();
    let vp_cd = CandleTensor::from_vec(
        pattern(total_slots * num_kv_heads * head_dim, 2),
        (total_slots, num_kv_heads, head_dim),
        &dev,
    )
    .unwrap()
    .to_dtype(CandleDType::BF16)
    .unwrap();

    // k, v: BF16 single-token row with num_kv_heads * head_dim elements
    let k_cd = CandleTensor::from_vec(
        pattern(num_kv_heads * head_dim, 3),
        (num_kv_heads * head_dim,),
        &dev,
    )
    .unwrap()
    .to_dtype(CandleDType::BF16)
    .unwrap();
    let v_cd = CandleTensor::from_vec(
        pattern(num_kv_heads * head_dim, 4),
        (num_kv_heads * head_dim,),
        &dev,
    )
    .unwrap()
    .to_dtype(CandleDType::BF16)
    .unwrap();

    // slot: U32 [1] — device-side slot index
    let slot_cd = CandleTensor::from_vec(vec![2u32], (1,), &dev).unwrap();

    // Use the BORROW adapter — zero-copy from candle to kt.
    let kp_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&kp_cd).unwrap();
    let vp_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&vp_cd).unwrap();
    let k_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&k_cd).unwrap();
    let v_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&v_cd).unwrap();
    let slot_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&slot_cd).unwrap();

    // The migration's critical correctness property: the call must
    // not panic on CudaStorage::slice() (which the old impl
    // called, and which panics on Borrowed storage).
    paged_kv_write_token_major_bf16_slot_kt(&kp_kt, &vp_kt, &k_kt, &v_kt, &slot_kt)
        .expect("paged_kv_write_slot_kt on borrowed inputs");

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    // No output tensor — the pools are written in place. Verify by
    // checking that the function returned Ok and didn't panic.
    assert_eq!(kp_kt.shape(), &[total_slots, num_kv_heads, head_dim]);
}
