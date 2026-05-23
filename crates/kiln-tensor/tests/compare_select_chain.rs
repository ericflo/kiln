//! Integration test for the compare → bool-reduce → where_select chain.
//!
//! Proves that the new comparison and boolean-reduction primitives
//! compose with where_select / masked_fill for typical "threshold +
//! mask + select" workloads.

use kiln_tensor::ops::{
    all_axis, all_reduce, any_axis, any_reduce, eq, ge, gt, le, lt, masked_fill, maximum,
    minimum, ne, where_select,
};
use kiln_tensor::{CpuStorage, Tensor};

fn read_f32(t: &Tensor) -> Vec<f32> {
    let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
    cpu.as_bytes()
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn read_u8(t: &Tensor) -> Vec<u8> {
    let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
    cpu.as_bytes().to_vec()
}

#[test]
fn lt_mask_drives_where_select() {
    // x = [-2, 0, 3, -5, 1]; replace negatives with zeros.
    let x = Tensor::from_slice(&[-2.0f32, 0.0, 3.0, -5.0, 1.0], vec![5]).unwrap();
    let zero = Tensor::from_slice(&[0.0f32; 5], vec![5]).unwrap();
    let mask = lt(&x, &zero).unwrap();
    let out = where_select(&mask, &zero, &x).unwrap();
    assert_eq!(read_f32(&out), vec![0.0, 0.0, 3.0, 0.0, 1.0]);
}

#[test]
fn gt_mask_with_masked_fill() {
    // Threshold above-2 values to a sentinel via masked_fill.
    let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
    let two = Tensor::from_slice(&[2.0f32; 4], vec![4]).unwrap();
    let mask = gt(&x, &two).unwrap();
    let out = masked_fill(&x, &mask, -999.0).unwrap();
    assert_eq!(read_f32(&out), vec![1.0, 2.0, -999.0, -999.0]);
}

#[test]
fn eq_and_ne_complement() {
    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
    let b = Tensor::from_slice(&[1.0f32, 5.0, 3.0], vec![3]).unwrap();
    let e = eq(&a, &b).unwrap();
    let n = ne(&a, &b).unwrap();
    // eq XOR ne should be all-1.
    let e_v = read_u8(&e);
    let n_v = read_u8(&n);
    for i in 0..3 {
        assert_eq!(e_v[i] ^ n_v[i], 1);
    }
}

#[test]
fn le_then_all_axis() {
    // Every row entry ≤ 5? Row 0: [1, 2, 3] all ≤ 5 → 1. Row 1: [4, 6, 2] one is > 5 → 0.
    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 6.0, 2.0], vec![2, 3]).unwrap();
    let five = Tensor::from_slice(&[5.0f32; 6], vec![2, 3]).unwrap();
    let mask = le(&a, &five).unwrap();
    let allm = all_axis(&mask, 1).unwrap();
    assert_eq!(read_u8(&allm), vec![1, 0]);
}

#[test]
fn ge_then_any_axis() {
    // Any row entry ≥ 5? Row 0: [1, 2, 3] none → 0. Row 1: [4, 6, 2] one → 1.
    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 6.0, 2.0], vec![2, 3]).unwrap();
    let five = Tensor::from_slice(&[5.0f32; 6], vec![2, 3]).unwrap();
    let mask = ge(&a, &five).unwrap();
    let anym = any_axis(&mask, 1).unwrap();
    assert_eq!(read_u8(&anym), vec![0, 1]);
}

#[test]
fn minimum_maximum_compose() {
    // Clamp manually via minimum + maximum.
    let x = Tensor::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], vec![5]).unwrap();
    let lo = Tensor::from_slice(&[-1.0f32; 5], vec![5]).unwrap();
    let hi = Tensor::from_slice(&[1.0f32; 5], vec![5]).unwrap();
    let clamped = minimum(&maximum(&x, &lo).unwrap(), &hi).unwrap();
    assert_eq!(read_f32(&clamped), vec![-1.0, -1.0, 0.0, 1.0, 1.0]);
}

#[test]
fn all_reduce_global_predicate() {
    // Check that every value in a vec is positive.
    let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
    let zero = Tensor::from_slice(&[0.0f32; 3], vec![3]).unwrap();
    let mask = gt(&x, &zero).unwrap();
    let global = all_reduce(&mask).unwrap();
    assert_eq!(global.shape(), &[] as &[usize]);
    assert_eq!(read_u8(&global), vec![1]);
}

#[test]
fn any_reduce_global_predicate() {
    // Check that any value is exactly 5.
    let x = Tensor::from_slice(&[1.0f32, 2.0, 5.0], vec![3]).unwrap();
    let five = Tensor::from_slice(&[5.0f32; 3], vec![3]).unwrap();
    let mask = eq(&x, &five).unwrap();
    let global = any_reduce(&mask).unwrap();
    assert_eq!(read_u8(&global), vec![1]);
}
