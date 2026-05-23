//! Integration parity test covering the new ops shipped in PRs
//! #1263-#1302: chunk/split, unbind, tile, roll, flip, gather,
//! masked_select, nonzero, bincount, unique, sort, searchsorted,
//! cumprod, meshgrid, pad, einsum, clip_grad_value, lerp, addmm,
//! tensor_norm, interpolate_1d. One end-to-end "you can compose
//! many of these together and get sensible numbers out" test that
//! Phase 9's parity gate will hook into.

use kiln_tensor::ops;
use kiln_tensor::{CpuStorage, DType, Tensor};

fn read_f32(t: &Tensor) -> Vec<f32> {
    let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
    cpu.as_bytes()
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
fn read_i64(t: &Tensor) -> Vec<i64> {
    let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
    cpu.as_bytes()
        .chunks(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn split_then_concat_is_identity() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let x = Tensor::from_slice(&data, vec![3, 4]).unwrap();
    let chunks = ops::chunk(&x, 2, 1).unwrap();
    assert_eq!(chunks.len(), 2);
    // narrow views aren't contiguous along axis 1; concat requires
    // contiguous inputs. Materialize each chunk before concatenating.
    let chunks_contig: Vec<Tensor> =
        chunks.iter().map(|c| c.contiguous().unwrap()).collect();
    let chunk_refs: Vec<&Tensor> = chunks_contig.iter().collect();
    let back = ops::concat(&chunk_refs, 1).unwrap();
    assert_eq!(back.shape(), x.shape());
    assert_eq!(read_f32(&back), data);
}

#[test]
fn unbind_then_stack_is_identity() {
    // PR #1306 ensures each unbind output is contiguous via the op's
    // internal contiguous() call. stack expects contiguous inputs.
    let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
    let parts = ops::unbind(&x, 0).unwrap();
    let part_refs: Vec<&Tensor> = parts.iter().collect();
    let back = ops::stack(&part_refs, 0).unwrap();
    assert_eq!(back.shape(), x.shape());
    assert_eq!(read_f32(&back), read_f32(&x));
}

#[test]
fn flip_twice_is_identity() {
    let x = Tensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    )
    .unwrap();
    let once = ops::flip(&x, &[1]).unwrap();
    let twice = ops::flip(&once, &[1]).unwrap();
    assert_eq!(read_f32(&twice), read_f32(&x));
}

#[test]
fn roll_full_period_is_identity() {
    let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
    let y = ops::roll(&x, 4, 0).unwrap();
    assert_eq!(read_f32(&y), read_f32(&x));
}

#[test]
fn tile_then_chunk_recovers_original() {
    // tile(x, [2]) then chunk(_, 2, 0) → two copies of x.
    // chunks are narrow views; materialize via contiguous() so
    // read_f32 sees only the slice's bytes (not the underlying tiled
    // buffer).
    let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
    let tiled = ops::tile(&x, &[2]).unwrap();
    let chunks = ops::chunk(&tiled, 2, 0).unwrap();
    assert_eq!(chunks.len(), 2);
    for c in &chunks {
        let c_contig = c.contiguous().unwrap();
        assert_eq!(read_f32(&c_contig), vec![1.0, 2.0, 3.0]);
    }
}

#[test]
fn gather_then_index_recovers_original_at_identity_indices() {
    // gather with index = arange should be identity.
    let x = Tensor::from_slice(&[10.0f32, 20.0, 30.0], vec![3]).unwrap();
    let idx = Tensor::from_slice(&[0i64, 1, 2], vec![3]).unwrap();
    let y = ops::gather(&x, 0, &idx).unwrap();
    assert_eq!(read_f32(&y), vec![10.0, 20.0, 30.0]);
}

#[test]
fn masked_select_round_trip_matches_nonzero_count() {
    // Pick all-true mask → masked_select returns flatten; nonzero
    // returns each coordinate exactly once.
    let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let mask = Tensor::from_slice(&[1u8, 1, 1, 1], vec![2, 2]).unwrap();
    let sel = ops::masked_select(&x, &mask).unwrap();
    let nz = ops::nonzero(&mask).unwrap();
    assert_eq!(sel.element_count(), nz.shape()[0]);
}

#[test]
fn bincount_then_unique_consistent_max() {
    let x = Tensor::from_slice(&[2i64, 5, 2, 0, 5, 7], vec![6]).unwrap();
    let counts = ops::bincount(&x, 0).unwrap();
    let (values, _vcounts) = ops::unique(&x).unwrap();
    // The max unique value should be one less than counts.len().
    let max_unique = *read_i64(&values).iter().max().unwrap();
    assert_eq!(counts.shape()[0] as i64, max_unique + 1);
}

#[test]
fn sort_then_argsort_indices_match() {
    let x = Tensor::from_slice(&[3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0], vec![6]).unwrap();
    let (_v, i_from_sort) = ops::sort(&x, false).unwrap();
    let i_from_argsort = ops::argsort(&x, false).unwrap();
    assert_eq!(read_i64(&i_from_sort), read_i64(&i_from_argsort));
}

#[test]
fn searchsorted_at_each_unique_value_returns_correct_index() {
    let sorted = Tensor::from_slice(&[1.0f32, 3.0, 5.0, 7.0], vec![4]).unwrap();
    // Values: 0 → 0; 1 → 0 (left side); 3 → 1; 5 → 2; 7 → 3; 8 → 4.
    let v = Tensor::from_slice(&[0.0f32, 1.0, 3.0, 5.0, 7.0, 8.0], vec![6]).unwrap();
    let pos = ops::searchsorted(&sorted, &v, false).unwrap();
    assert_eq!(read_i64(&pos), vec![0, 0, 1, 2, 3, 4]);
}

#[test]
fn cumprod_of_ones_is_ones() {
    let x = Tensor::from_slice(&[1.0f32; 5], vec![5]).unwrap();
    let y = ops::cumprod(&x, 0).unwrap();
    assert_eq!(read_f32(&y), vec![1.0; 5]);
}

#[test]
fn meshgrid_then_index_recovers_axes() {
    let a = Tensor::from_slice(&[10.0f32, 20.0, 30.0], vec![3]).unwrap();
    let b = Tensor::from_slice(&[100.0f32, 200.0], vec![2]).unwrap();
    let outs = ops::meshgrid(&[&a, &b]).unwrap();
    assert_eq!(outs.len(), 2);
    // outs[0] is [3, 2] = [[10,10],[20,20],[30,30]]; narrow on axis 1
    // gives a [3, 1] view whose underlying bytes still span the full
    // [3, 2] buffer. Materialize before reading so read_f32 sees only
    // the column.
    let col0_a = outs[0].narrow(1, 0, 1).unwrap().contiguous().unwrap();
    assert_eq!(read_f32(&col0_a), vec![10.0, 20.0, 30.0]);
}

#[test]
fn pad_unpad_via_narrow_recovers_original() {
    let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
    let padded = ops::pad(&x, &[(1, 2)], 0.0).unwrap();
    // Strip the pad via narrow.
    let recovered = padded.narrow(0, 1, 3).unwrap();
    assert_eq!(read_f32(&recovered.contiguous().unwrap()), vec![1.0, 2.0, 3.0]);
}

#[test]
fn einsum_matmul_matches_matmul_op() {
    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
    let via_einsum = ops::einsum("ij,jk->ik", &[&a, &b]).unwrap();
    let via_matmul = ops::matmul(&a, &b).unwrap();
    let v_ein = read_f32(&via_einsum);
    let v_mm = read_f32(&via_matmul);
    for (e, m) in v_ein.iter().zip(v_mm.iter()) {
        assert!((e - m).abs() < 1e-5);
    }
}

#[test]
fn clip_grad_value_idempotent_below_threshold() {
    let g = Tensor::from_slice(&[0.1f32, -0.2, 0.3], vec![3]).unwrap();
    let out = ops::clip_grad_value(&[&g], 10.0).unwrap();
    assert_eq!(read_f32(&out[0]), read_f32(&g));
}

#[test]
fn lerp_zero_and_one_recover_endpoints() {
    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
    let b = Tensor::from_slice(&[10.0f32, 20.0, 30.0], vec![3]).unwrap();
    assert_eq!(read_f32(&ops::lerp(&a, &b, 0.0).unwrap()), read_f32(&a));
    assert_eq!(read_f32(&ops::lerp(&a, &b, 1.0).unwrap()), read_f32(&b));
}

#[test]
fn addmm_beta_zero_equals_matmul_times_alpha() {
    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
    let c = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
    let alpha = 2.0;
    let via_addmm = ops::addmm(&c, &a, &b, alpha, 0.0).unwrap();
    let raw = ops::matmul(&a, &b).unwrap();
    let scaled = ops::mul_scalar(&raw, alpha).unwrap();
    let v_addmm = read_f32(&via_addmm);
    let v_scaled = read_f32(&scaled);
    for (x, y) in v_addmm.iter().zip(v_scaled.iter()) {
        assert!((x - y).abs() < 1e-5);
    }
}

#[test]
fn tensor_norms_obey_pythagorean_identity_at_scale() {
    // For x = [3, 4]: ‖x‖_2 = 5; ‖x‖_∞ = 4; ‖x‖_1 = 7.
    let x = Tensor::from_slice(&[3.0f32, -4.0], vec![2]).unwrap();
    let l1 = ops::l1_norm(&x).unwrap();
    let l2 = ops::l2_norm_scalar(&x).unwrap();
    let linf = ops::linf_norm(&x).unwrap();
    assert!((read_f32(&l1)[0] - 7.0).abs() < 1e-5);
    assert!((read_f32(&l2)[0] - 5.0).abs() < 1e-5);
    assert!((read_f32(&linf)[0] - 4.0).abs() < 1e-5);
}

#[test]
fn interpolate_1d_then_back_is_close_to_identity_for_smooth_signals() {
    // Linear ramp upsampled then downsampled is exact under
    // align_corners=Yes (endpoints preserved, intermediate values
    // are linear blends of linearly-distributed inputs).
    let x_data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let x = Tensor::from_slice(&x_data, vec![8]).unwrap();
    let up = ops::interpolate_1d(&x, 16, ops::AlignCorners::Yes).unwrap();
    let back =
        ops::interpolate_1d(&up, 8, ops::AlignCorners::Yes).unwrap();
    let v = read_f32(&back);
    for (a, b) in v.iter().zip(x_data.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "linear ramp interp round trip should be exact: {a} vs {b}"
        );
    }
}
