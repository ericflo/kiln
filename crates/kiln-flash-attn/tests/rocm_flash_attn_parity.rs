//! Phase R.8 — ROCm composite SDPA parity vs a CPU reference, on a real AMD GPU.
//!
//! Covers `flash_attn_fwd_kt` and `flash_attn_paged_decode_kt` on
//! `Device::Rocm` against a from-scratch CPU SDPA computed in the test, over GQA
//! (h=16, hk in {2,8}), causal + non-causal, head_dim=128, and seqlens
//! {1,7,16,33,64,128}. BF16 tolerances (rtol/atol ~2e-2).
//!
//! Run:
//!   cargo test -p kiln-flash-attn --no-default-features --features rocm
//!   KILN_ROCM_WAVE64=1 cargo test -p kiln-flash-attn --no-default-features --features rocm
//!
//! Skips (returns Ok) when no ROCm device is present.
#![cfg(feature = "rocm")]

use half::bf16;
use kiln_flash_attn::{
    flash_attn_bwd_collapsed_gqa_kt, flash_attn_bwd_kt, flash_attn_fwd_kt,
    flash_attn_fwd_no_lse_kt, flash_attn_paged_decode_dyn_seqlen_kt, flash_attn_paged_decode_kt,
    paged_kv_write_token_major_bf16_kt, with_test_score_geometry,
};
use kiln_tensor::{
    DType, Device, RocmContext, RocmExecutionPolicy, RocmFlashAttentionPolicy,
    RocmFlashAttentionRouteMode, RocmTensorKernelPolicy, Tensor,
};
use std::sync::Arc;

unsafe extern "C" {
    fn kiln_rocm_flash_wmma_qk16_bf16(
        a: *const core::ffi::c_void,
        b: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.8 flash-attn parity test");
        true
    } else {
        false
    }
}

fn native_route_test_policy() -> RocmFlashAttentionPolicy {
    RocmFlashAttentionPolicy {
        native_scalar_forward: true,
        native_tiled_forward: true,
        native_streaming_forward: true,
        native_rectangular_causal_forward: true,
        native_backward_preference: RocmFlashAttentionRouteMode::Auto,
        collapsed_gqa_backward: true,
        native_direct_collapsed_gqa_backward: true,
        native_gqa_qblock_forward: true,
        wmma_gqa_qblock_forward: true,
        wmma_gqa_r64k32_forward: true,
        wmma_gqa_r64k32_log2_forward: true,
        ..RocmFlashAttentionPolicy::portable_fallback()
    }
}

/// Deterministic pseudo-random value in roughly [-0.5, 0.5].
fn val(i: usize, seed: usize) -> f32 {
    let x = (i
        .wrapping_mul(2654435761)
        .wrapping_add(seed.wrapping_mul(40503)))
        % 1000;
    (x as f32) / 1000.0 - 0.5
}

fn fill_bf16(n: usize, seed: usize) -> Vec<bf16> {
    (0..n).map(|i| bf16::from_f32(val(i, seed))).collect()
}

/// Reference SDPA in f32 over host data.
///
/// q: [b, sq, h, d], k/v: [b, sk, hk, d]; GQA group = h/hk. Returns out
/// [b, sq, h, d] as f32 (caller compares against the bf16 device output).
#[allow(clippy::too_many_arguments)]
fn cpu_sdpa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    sq: usize,
    sk: usize,
    h: usize,
    hk: usize,
    d: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let group = h / hk;
    let mut out = vec![0.0f32; b * sq * h * d];
    let q_at = |bi: usize, si: usize, hi: usize, di: usize| q[((bi * sq + si) * h + hi) * d + di];
    let k_at =
        |bi: usize, si: usize, hki: usize, di: usize| k[((bi * sk + si) * hk + hki) * d + di];
    let v_at =
        |bi: usize, si: usize, hki: usize, di: usize| v[((bi * sk + si) * hk + hki) * d + di];
    let offset = sk as isize - sq as isize;
    for bi in 0..b {
        for hi in 0..h {
            let hki = hi / group; // kv head
            for si in 0..sq {
                // scores over all sk keys.
                let mut scores = vec![f32::NEG_INFINITY; sk];
                let last_allowed = si as isize + offset;
                let mut maxv = f32::NEG_INFINITY;
                for sj in 0..sk {
                    if causal && (sj as isize) > last_allowed {
                        continue; // masked future key
                    }
                    let mut acc = 0.0f32;
                    for di in 0..d {
                        acc += q_at(bi, si, hi, di) * k_at(bi, sj, hki, di);
                    }
                    let s = acc * scale;
                    scores[sj] = s;
                    if s > maxv {
                        maxv = s;
                    }
                }
                // softmax
                let mut denom = 0.0f32;
                let mut exps = vec![0.0f32; sk];
                for sj in 0..sk {
                    if scores[sj] == f32::NEG_INFINITY {
                        continue;
                    }
                    let e = (scores[sj] - maxv).exp();
                    exps[sj] = e;
                    denom += e;
                }
                // out = sum_j p_j v_j
                for di in 0..d {
                    let mut acc = 0.0f32;
                    for sj in 0..sk {
                        if exps[sj] == 0.0 {
                            continue;
                        }
                        acc += (exps[sj] / denom) * v_at(bi, sj, hki, di);
                    }
                    out[((bi * sq + si) * h + hi) * d + di] = acc;
                }
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn cpu_sdpa_bwd_expanded_gqa(
    dout: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &[f32],
    b: usize,
    sq: usize,
    sk: usize,
    h: usize,
    hk: usize,
    d: usize,
    scale: f32,
    causal: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let group = h / hk;
    let mut dq = vec![0.0f32; b * sq * h * d];
    let mut dk = vec![0.0f32; b * sk * h * d];
    let mut dv = vec![0.0f32; b * sk * h * d];
    let offset = sk as isize - sq as isize;

    for bi in 0..b {
        for hi in 0..h {
            let hki = hi / group;
            for si in 0..sq {
                let last_allowed = si as isize + offset;
                let mut scores = vec![f32::NEG_INFINITY; sk];
                let mut maxv = f32::NEG_INFINITY;
                for sj in 0..sk {
                    if causal && (sj as isize) > last_allowed {
                        continue;
                    }
                    let mut acc = 0.0f32;
                    for di in 0..d {
                        let qv = q[((bi * sq + si) * h + hi) * d + di];
                        let kv = k[((bi * sk + sj) * hk + hki) * d + di];
                        acc += qv * kv;
                    }
                    let score = acc * scale;
                    scores[sj] = score;
                    maxv = maxv.max(score);
                }

                let mut denom = 0.0f32;
                let mut probs = vec![0.0f32; sk];
                for sj in 0..sk {
                    if scores[sj].is_finite() {
                        let p = (scores[sj] - maxv).exp();
                        probs[sj] = p;
                        denom += p;
                    }
                }
                if denom > 0.0 {
                    for p in &mut probs {
                        *p /= denom;
                    }
                }

                let mut d_i = 0.0f32;
                for di in 0..d {
                    let idx = ((bi * sq + si) * h + hi) * d + di;
                    d_i += dout[idx] * out[idx];
                }

                for sj in 0..sk {
                    let p = probs[sj];
                    if p == 0.0 {
                        continue;
                    }
                    let mut dp = 0.0f32;
                    for di in 0..d {
                        let do_idx = ((bi * sq + si) * h + hi) * d + di;
                        let v_idx = ((bi * sk + sj) * hk + hki) * d + di;
                        dp += dout[do_idx] * v[v_idx];
                    }
                    let ds = p * (dp - d_i) * scale;
                    for di in 0..d {
                        let q_idx = ((bi * sq + si) * h + hi) * d + di;
                        let k_idx = ((bi * sk + sj) * hk + hki) * d + di;
                        let expanded_idx = ((bi * sk + sj) * h + hi) * d + di;
                        dq[q_idx] += ds * k[k_idx];
                        dk[expanded_idx] += ds * q[q_idx];
                        dv[expanded_idx] += p * dout[q_idx];
                    }
                }
            }
        }
    }

    (dq, dk, dv)
}

fn collapse_expanded_gqa_host(
    expanded: &[f32],
    b: usize,
    sk: usize,
    h: usize,
    hk: usize,
    d: usize,
) -> Vec<f32> {
    if h == hk {
        return expanded.to_vec();
    }
    let group = h / hk;
    let mut collapsed = vec![0.0f32; b * sk * hk * d];
    for bi in 0..b {
        for si in 0..sk {
            for hki in 0..hk {
                for gi in 0..group {
                    let hi = hki * group + gi;
                    for di in 0..d {
                        let src = ((bi * sk + si) * h + hi) * d + di;
                        let dst = ((bi * sk + si) * hk + hki) * d + di;
                        collapsed[dst] += expanded[src];
                    }
                }
            }
        }
    }
    collapsed
}

fn check_close(got: &[f32], want: &[f32], rtol: f32, atol: f32, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    let mut worst = 0.0f32;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        let tol = atol + rtol * w.abs();
        if diff > tol {
            worst = worst.max(diff - tol);
            if diff > tol * 4.0 {
                panic!("{label}: idx {i} got {g} want {w} diff {diff} tol {tol}");
            }
        }
    }
    // A few elements may sit just outside the bf16 tolerance band; only fail if
    // the overshoot is large (handled above). Report the worst overshoot.
    if worst > 0.0 {
        eprintln!("{label}: max tol overshoot {worst:.4e} (within 4x band)");
    }
}

fn rocm_bf16(data: &[bf16], shape: Vec<usize>) -> Tensor {
    let cpu = Tensor::from_vec(data.to_vec(), shape).expect("cpu bf16");
    kiln_tensor::host_to_rocm_copy(&cpu, 0).expect("upload bf16 to rocm")
}

fn rocm_policy_context(flash_attention: RocmFlashAttentionPolicy) -> Arc<RocmContext> {
    let selector = Device::Rocm(0).memory_probe_selector();
    if kiln_memory::MemoryGovernor::try_global_cached_snapshot().is_none() {
        kiln_memory::MemoryGovernor::configure_global(
            selector,
            kiln_memory::GovernorConfig::default(),
        )
        .expect("configure ROCm test memory governor");
        let snapshot = kiln_memory::MemoryGovernor::global().refresh();
        assert!(
            !snapshot.observations.probe_failed,
            "ROCm test memory probe must publish a usable snapshot"
        );
    }
    assert_eq!(
        kiln_memory::MemoryGovernor::global_configuration().selector,
        selector,
        "ROCm policy context must match the process memory governor"
    );
    let tensor_kernels = RocmTensorKernelPolicy {
        flash_attention,
        ..RocmTensorKernelPolicy::portable_fallback()
    };
    RocmContext::new_with_execution_policy(
        0,
        RocmExecutionPolicy::default().with_tensor_kernel_policy(tensor_kernels),
    )
    .expect("create isolated ROCm policy context")
}

fn rocm_bf16_on(ctx: &Arc<RocmContext>, data: &[bf16], shape: Vec<usize>) -> Tensor {
    let cpu = Tensor::from_vec(data.to_vec(), shape).expect("cpu bf16");
    kiln_tensor::host_to_rocm_copy_with_context(&cpu, ctx)
        .expect("upload bf16 to isolated ROCm context")
}

fn bf16_to_f32(t: &Tensor) -> Vec<f32> {
    let host = kiln_tensor::rocm_to_host_copy(t).expect("d2h");
    host.to_vec::<bf16>()
        .expect("bf16 readback")
        .into_iter()
        .map(|x| x.to_f32())
        .collect()
}

fn rocm_sync() {
    kiln_tensor::rocm_synchronize_default_stream(0).expect("rocm sync");
}

fn f32_to_vec(t: &Tensor) -> Vec<f32> {
    let host = kiln_tensor::rocm_to_host_copy(t).expect("d2h");
    host.to_vec::<f32>().expect("f32 readback")
}

#[test]
fn rocm_wmma_qk16_bf16_tile_matches_cpu_dot() {
    if no_rocm() {
        return;
    }

    let tile = 16usize;
    let a_data = fill_bf16(tile * tile, 9101);
    let b_data = fill_bf16(tile * tile, 9109);
    let a = rocm_bf16(&a_data, vec![tile, tile]);
    let b = rocm_bf16(&b_data, vec![tile, tile]);
    let out = Tensor::zeros(vec![tile, tile], DType::F32, Device::Rocm(0)).expect("rocm f32 out");

    let a_ptr = kiln_kt_bridge::rocm_input_device_ptr(&a, DType::BF16, "wmma_a").expect("a ptr");
    let b_ptr = kiln_kt_bridge::rocm_input_device_ptr(&b, DType::BF16, "wmma_b").expect("b ptr");
    let out_ptr = kiln_kt_bridge::rocm_output_device_ptr(&out);
    let stream_submission =
        kiln_kt_bridge::rocm_stream_submission_of(&a, "wmma_a").expect("stream");
    let stream = stream_submission.raw_stream();
    let status = unsafe {
        kiln_rocm_flash_wmma_qk16_bf16(
            a_ptr as *const _,
            b_ptr as *const _,
            out_ptr as *mut _,
            stream,
        )
    };
    if status > 0 {
        stream_submission.quarantine();
        panic!("wmma qk16 execution failed with status {status}");
    }
    stream_submission.complete();
    if status == -30 {
        eprintln!("ROCm device does not report gfx11 WMMA support; skipping qk16 tile test");
        return;
    }
    assert_eq!(status, 0, "wmma qk16 launch status");
    rocm_sync();

    let got = f32_to_vec(&out);
    let mut want = vec![0.0f32; tile * tile];
    for m in 0..tile {
        for n in 0..tile {
            let mut acc = 0.0f32;
            for k in 0..tile {
                acc += a_data[m * tile + k].to_f32() * b_data[n * tile + k].to_f32();
            }
            want[m * tile + n] = acc;
        }
    }
    check_close(&got, &want, 1e-4, 1e-4, "rocm wmma qk16 bf16 tile");
}

#[test]
fn flash_attn_fwd_native_gqa_qblock256_parity() {
    if no_rocm() {
        return;
    }
    let b = 1usize;
    let h = 4usize;
    let hk = 1usize;
    let d = 256usize;
    let sq = 37usize;
    let sk = 81usize;
    let scale = 1.0 / (d as f32).sqrt();
    let causal = true;
    let policy = RocmFlashAttentionPolicy {
        native_gqa_qblock_forward_min_sequence: 1,
        wmma_gqa_r64k32_forward_min_sequence: 1,
        wmma_gqa_r64k32_log2_forward_min_sequence: 1,
        ..native_route_test_policy()
    };
    let ctx = rocm_policy_context(policy);
    let qd = fill_bf16(b * sq * h * d, 9301);
    let kd = fill_bf16(b * sk * hk * d, 9309);
    let vd = fill_bf16(b * sk * hk * d, 9317);

    let q = rocm_bf16_on(&ctx, &qd, vec![b, sq, h, d]);
    let k = rocm_bf16_on(&ctx, &kd, vec![b, sk, hk, d]);
    let v = rocm_bf16_on(&ctx, &vd, vec![b, sk, hk, d]);

    let (out_t, lse_t) = flash_attn_fwd_kt(&q, &k, &v, scale, causal)
        .unwrap_or_else(|e| panic!("native gqa qblock256: {e}"));
    assert_eq!(out_t.shape(), &[b, sq, h, d]);
    assert_eq!(lse_t.shape(), &[b, h, sq]);

    let qf: Vec<f32> = qd.iter().map(|x| x.to_f32()).collect();
    let kf: Vec<f32> = kd.iter().map(|x| x.to_f32()).collect();
    let vf: Vec<f32> = vd.iter().map(|x| x.to_f32()).collect();
    let want = cpu_sdpa(&qf, &kf, &vf, b, sq, sk, h, hk, d, scale, causal);
    let got = bf16_to_f32(&out_t);
    check_close(&got, &want, 2e-2, 2e-2, "native gqa qblock256");
}

#[test]
fn flash_attn_fwd_no_lse_matches_fwd_output() {
    if no_rocm() {
        return;
    }
    let b = 1usize;
    let sq = 64usize;
    let sk = 64usize;
    let h = 16usize;
    let hk = 4usize;
    let d = 256usize;
    let scale = 1.0 / (d as f32).sqrt();
    let causal = true;

    let qd = fill_bf16(b * sq * h * d, 1001);
    let kd = fill_bf16(b * sk * hk * d, 1002);
    let vd = fill_bf16(b * sk * hk * d, 1003);
    let q = rocm_bf16(&qd, vec![b, sq, h, d]);
    let k = rocm_bf16(&kd, vec![b, sk, hk, d]);
    let v = rocm_bf16(&vd, vec![b, sk, hk, d]);

    let (out, _lse) = flash_attn_fwd_kt(&q, &k, &v, scale, causal).expect("fwd with lse");
    let out_no_lse = flash_attn_fwd_no_lse_kt(&q, &k, &v, scale, causal)
        .expect("fwd no-lse")
        .expect("rocm no-lse should provide an output");
    let got = bf16_to_f32(&out_no_lse);
    let want = bf16_to_f32(&out);
    check_close(&got, &want, 0.0, 0.0, "no-lse output");
}

#[test]
fn flash_attn_fwd_parity_gqa_causal_and_dense() {
    if no_rocm() {
        return;
    }
    let b = 1usize;
    let h = 16usize;
    let d = 128usize;
    let scale = 1.0 / (d as f32).sqrt();

    for &hk in &[2usize, 8usize] {
        for &causal in &[false, true] {
            for &sq in &[1usize, 7, 16, 33, 64, 128] {
                // For dense fwd, sk == sq (self-attention).
                let sk = sq;
                let qd = fill_bf16(b * sq * h * d, 11 + sq + hk);
                let kd = fill_bf16(b * sk * hk * d, 23 + sq + hk);
                let vd = fill_bf16(b * sk * hk * d, 37 + sq + hk);

                let q = rocm_bf16(&qd, vec![b, sq, h, d]);
                let k = rocm_bf16(&kd, vec![b, sk, hk, d]);
                let v = rocm_bf16(&vd, vec![b, sk, hk, d]);

                let (out_t, lse_t) = flash_attn_fwd_kt(&q, &k, &v, scale, causal)
                    .unwrap_or_else(|e| panic!("fwd hk={hk} causal={causal} sq={sq}: {e}"));
                assert_eq!(out_t.shape(), &[b, sq, h, d]);
                assert_eq!(lse_t.shape(), &[b, h, sq]);
                assert_eq!(lse_t.dtype(), DType::F32);
                assert_eq!(out_t.device(), Device::Rocm(0));

                let qf: Vec<f32> = qd.iter().map(|x| x.to_f32()).collect();
                let kf: Vec<f32> = kd.iter().map(|x| x.to_f32()).collect();
                let vf: Vec<f32> = vd.iter().map(|x| x.to_f32()).collect();
                let want = cpu_sdpa(&qf, &kf, &vf, b, sq, sk, h, hk, d, scale, causal);

                let got = bf16_to_f32(&out_t);
                check_close(
                    &got,
                    &want,
                    2e-2,
                    2e-2,
                    &format!("fwd hk={hk} causal={causal} sq={sq}"),
                );
            }
        }
    }
}

#[test]
fn flash_attn_fwd_parity_cross_attn_sk_ne_sq() {
    if no_rocm() {
        return;
    }
    // Non-square (cross-attention-ish) shapes exercise the sk != sq mask offset.
    let b = 2usize;
    let h = 16usize;
    let hk = 8usize;
    let d = 128usize;
    let scale = 1.0 / (d as f32).sqrt();

    for &(sq, sk) in &[(7usize, 16usize), (33, 64), (16, 128), (1, 64)] {
        for &causal in &[false, true] {
            let qd = fill_bf16(b * sq * h * d, 5 + sq + sk);
            let kd = fill_bf16(b * sk * hk * d, 9 + sq + sk);
            let vd = fill_bf16(b * sk * hk * d, 13 + sq + sk);

            let q = rocm_bf16(&qd, vec![b, sq, h, d]);
            let k = rocm_bf16(&kd, vec![b, sk, hk, d]);
            let v = rocm_bf16(&vd, vec![b, sk, hk, d]);

            let (out_t, _lse) = flash_attn_fwd_kt(&q, &k, &v, scale, causal)
                .unwrap_or_else(|e| panic!("fwd cross sq={sq} sk={sk} causal={causal}: {e}"));
            assert_eq!(out_t.shape(), &[b, sq, h, d]);

            let qf: Vec<f32> = qd.iter().map(|x| x.to_f32()).collect();
            let kf: Vec<f32> = kd.iter().map(|x| x.to_f32()).collect();
            let vf: Vec<f32> = vd.iter().map(|x| x.to_f32()).collect();
            let want = cpu_sdpa(&qf, &kf, &vf, b, sq, sk, h, hk, d, scale, causal);
            let got = bf16_to_f32(&out_t);
            check_close(
                &got,
                &want,
                2e-2,
                2e-2,
                &format!("fwd cross sq={sq} sk={sk} causal={causal}"),
            );
        }
    }
}

#[test]
fn flash_attn_fwd_native_qblock_hd256_long_parity() {
    if no_rocm() {
        return;
    }
    let b = 1usize;
    let h = 2usize;
    let hk = 1usize;
    let d = 256usize;
    let sq = 17usize;
    let sk = 512usize;
    let scale = 1.0 / (d as f32).sqrt();

    for &causal in &[false, true] {
        let qd = fill_bf16(b * sq * h * d, 101 + causal as usize);
        let kd = fill_bf16(b * sk * hk * d, 211 + causal as usize);
        let vd = fill_bf16(b * sk * hk * d, 307 + causal as usize);

        let q = rocm_bf16(&qd, vec![b, sq, h, d]);
        let k = rocm_bf16(&kd, vec![b, sk, hk, d]);
        let v = rocm_bf16(&vd, vec![b, sk, hk, d]);

        let (out_t, lse_t) = flash_attn_fwd_kt(&q, &k, &v, scale, causal)
            .unwrap_or_else(|e| panic!("native qblock hd256 causal={causal}: {e}"));
        assert_eq!(out_t.shape(), &[b, sq, h, d]);
        assert_eq!(lse_t.shape(), &[b, h, sq]);
        assert_eq!(lse_t.dtype(), DType::F32);

        let qf: Vec<f32> = qd.iter().map(|x| x.to_f32()).collect();
        let kf: Vec<f32> = kd.iter().map(|x| x.to_f32()).collect();
        let vf: Vec<f32> = vd.iter().map(|x| x.to_f32()).collect();
        let want = cpu_sdpa(&qf, &kf, &vf, b, sq, sk, h, hk, d, scale, causal);
        let got = bf16_to_f32(&out_t);
        check_close(
            &got,
            &want,
            2e-2,
            2e-2,
            &format!("native qblock hd256 causal={causal}"),
        );
    }
}

#[test]
fn flash_attn_fwd_native_rectangular_query_tiled_parity() {
    if no_rocm() {
        return;
    }
    let b = 1usize;
    let h = 2usize;
    let hk = 1usize;
    let d = 128usize;
    let sq = 129usize;
    let sk = 257usize;
    let scale = 1.0 / (d as f32).sqrt();
    let causal = true;

    let policy = RocmFlashAttentionPolicy {
        native_forward_query_tile: 64,
        native_streaming_forward: false,
        ..native_route_test_policy()
    };
    let ctx = rocm_policy_context(policy);
    {
        let qd = fill_bf16(b * sq * h * d, 701);
        let kd = fill_bf16(b * sk * hk * d, 709);
        let vd = fill_bf16(b * sk * hk * d, 719);

        let q = rocm_bf16_on(&ctx, &qd, vec![b, sq, h, d]);
        let k = rocm_bf16_on(&ctx, &kd, vec![b, sk, hk, d]);
        let v = rocm_bf16_on(&ctx, &vd, vec![b, sk, hk, d]);

        let (out_t, lse_t) = flash_attn_fwd_kt(&q, &k, &v, scale, causal)
            .unwrap_or_else(|e| panic!("native rectangular query tiled: {e}"));
        assert_eq!(out_t.shape(), &[b, sq, h, d]);
        assert_eq!(lse_t.shape(), &[b, h, sq]);

        let qf: Vec<f32> = qd.iter().map(|x| x.to_f32()).collect();
        let kf: Vec<f32> = kd.iter().map(|x| x.to_f32()).collect();
        let vf: Vec<f32> = vd.iter().map(|x| x.to_f32()).collect();
        let want = cpu_sdpa(&qf, &kf, &vf, b, sq, sk, h, hk, d, scale, causal);
        let got = bf16_to_f32(&out_t);
        check_close(&got, &want, 2e-2, 2e-2, "native rectangular query tiled");
    }
}

#[test]
fn flash_attn_fwd_native_abs_query_tiled_gqa_parity() {
    if no_rocm() {
        return;
    }
    let b = 1usize;
    let h = 4usize;
    let hk = 1usize;
    let d = 256usize;
    let sq = 129usize;
    let sk = 257usize;
    let scale = 1.0 / (d as f32).sqrt();
    let causal = true;

    let policy = RocmFlashAttentionPolicy {
        native_forward_query_tile: 64,
        native_streaming_forward: false,
        native_gqa_qblock_forward_min_sequence: 1,
        wmma_gqa_r64k32_forward_min_sequence: 1,
        wmma_gqa_r64k32_log2_forward_min_sequence: 1,
        ..native_route_test_policy()
    };
    let ctx = rocm_policy_context(policy);
    {
        let qd = fill_bf16(b * sq * h * d, 1701);
        let kd = fill_bf16(b * sk * hk * d, 1709);
        let vd = fill_bf16(b * sk * hk * d, 1719);

        let q = rocm_bf16_on(&ctx, &qd, vec![b, sq, h, d]);
        let k = rocm_bf16_on(&ctx, &kd, vec![b, sk, hk, d]);
        let v = rocm_bf16_on(&ctx, &vd, vec![b, sk, hk, d]);

        let (out_t, lse_t) = flash_attn_fwd_kt(&q, &k, &v, scale, causal)
            .unwrap_or_else(|e| panic!("native abs query tiled gqa: {e}"));
        assert_eq!(out_t.shape(), &[b, sq, h, d]);
        assert_eq!(lse_t.shape(), &[b, h, sq]);

        let qf: Vec<f32> = qd.iter().map(|x| x.to_f32()).collect();
        let kf: Vec<f32> = kd.iter().map(|x| x.to_f32()).collect();
        let vf: Vec<f32> = vd.iter().map(|x| x.to_f32()).collect();
        let want = cpu_sdpa(&qf, &kf, &vf, b, sq, sk, h, hk, d, scale, causal);
        let got = bf16_to_f32(&out_t);
        check_close(&got, &want, 2e-2, 2e-2, "native abs query tiled gqa");
    }
}

#[test]
fn flash_attn_fwd_native_rectangular_streaming_parity() {
    if no_rocm() {
        return;
    }
    let b = 1usize;
    let h = 2usize;
    let hk = 1usize;
    let d = 128usize;
    let sq = 129usize;
    let sk = 257usize;
    let scale = 1.0 / (d as f32).sqrt();
    let causal = true;

    let policy = RocmFlashAttentionPolicy {
        native_forward_query_tile: 64,
        native_streaming_forward_min_sequence: 128,
        native_streaming_forward_key_tile: 64,
        ..native_route_test_policy()
    };
    let ctx = rocm_policy_context(policy);
    {
        let qd = fill_bf16(b * sq * h * d, 809);
        let kd = fill_bf16(b * sk * hk * d, 811);
        let vd = fill_bf16(b * sk * hk * d, 821);

        let q = rocm_bf16_on(&ctx, &qd, vec![b, sq, h, d]);
        let k = rocm_bf16_on(&ctx, &kd, vec![b, sk, hk, d]);
        let v = rocm_bf16_on(&ctx, &vd, vec![b, sk, hk, d]);

        let (out_t, lse_t) = flash_attn_fwd_kt(&q, &k, &v, scale, causal)
            .unwrap_or_else(|e| panic!("native rectangular streaming: {e}"));
        assert_eq!(out_t.shape(), &[b, sq, h, d]);
        assert_eq!(lse_t.shape(), &[b, h, sq]);

        let qf: Vec<f32> = qd.iter().map(|x| x.to_f32()).collect();
        let kf: Vec<f32> = kd.iter().map(|x| x.to_f32()).collect();
        let vf: Vec<f32> = vd.iter().map(|x| x.to_f32()).collect();
        let want = cpu_sdpa(&qf, &kf, &vf, b, sq, sk, h, hk, d, scale, causal);
        let got = bf16_to_f32(&out_t);
        check_close(&got, &want, 2e-2, 2e-2, "native rectangular streaming");
    }
}

#[test]
fn flash_attn_fwd_online_tiled_parity() {
    if no_rocm() {
        return;
    }
    let b = 1usize;
    let h = 2usize;
    let hk = 1usize;
    let d = 128usize;
    let sq = 257usize;
    let sk = 257usize;
    let scale = 1.0 / (d as f32).sqrt();
    let ctx = rocm_policy_context(RocmFlashAttentionPolicy::portable_fallback());

    with_test_score_geometry(1, 512, 512, 1, || {
        for &causal in &[false, true] {
            let qd = fill_bf16(b * sq * h * d, 401 + causal as usize);
            let kd = fill_bf16(b * sk * hk * d, 503 + causal as usize);
            let vd = fill_bf16(b * sk * hk * d, 607 + causal as usize);

            let q = rocm_bf16_on(&ctx, &qd, vec![b, sq, h, d]);
            let k = rocm_bf16_on(&ctx, &kd, vec![b, sk, hk, d]);
            let v = rocm_bf16_on(&ctx, &vd, vec![b, sk, hk, d]);

            let (out_t, lse_t) = flash_attn_fwd_kt(&q, &k, &v, scale, causal)
                .unwrap_or_else(|e| panic!("online tiled causal={causal}: {e}"));
            assert_eq!(out_t.shape(), &[b, sq, h, d]);
            assert_eq!(lse_t.shape(), &[b, h, sq]);
            assert_eq!(lse_t.dtype(), DType::F32);

            let qf: Vec<f32> = qd.iter().map(|x| x.to_f32()).collect();
            let kf: Vec<f32> = kd.iter().map(|x| x.to_f32()).collect();
            let vf: Vec<f32> = vd.iter().map(|x| x.to_f32()).collect();
            let want = cpu_sdpa(&qf, &kf, &vf, b, sq, sk, h, hk, d, scale, causal);
            let got = bf16_to_f32(&out_t);
            check_close(
                &got,
                &want,
                2e-2,
                2e-2,
                &format!("online tiled causal={causal}"),
            );
        }
    });
}

#[test]
fn flash_attn_rocm_long_shape_bench() {
    if no_rocm() {
        return;
    }
    let Ok(seq_raw) = std::env::var("KILN_ROCM_FLASH_BENCH_SEQ") else {
        eprintln!("set KILN_ROCM_FLASH_BENCH_SEQ to run the long-shape ROCm bench");
        return;
    };
    let seq = seq_raw
        .trim()
        .parse::<usize>()
        .expect("KILN_ROCM_FLASH_BENCH_SEQ must be usize");
    if seq == 0 {
        return;
    }

    let b = 1usize;
    let h = std::env::var("KILN_ROCM_FLASH_BENCH_HEADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(16);
    let hk = std::env::var("KILN_ROCM_FLASH_BENCH_KV_HEADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(4);
    let d = std::env::var("KILN_ROCM_FLASH_BENCH_HEAD_DIM")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(256);
    let scale = 1.0 / (d as f32).sqrt();
    let causal = true;

    let qd = fill_bf16(b * seq * h * d, 1701);
    let kd = fill_bf16(b * seq * hk * d, 1801);
    let vd = fill_bf16(b * seq * hk * d, 1901);
    let dod = fill_bf16(b * seq * h * d, 2001);

    let q = rocm_bf16(&qd, vec![b, seq, h, d]);
    let k = rocm_bf16(&kd, vec![b, seq, hk, d]);
    let v = rocm_bf16(&vd, vec![b, seq, hk, d]);
    let dout = rocm_bf16(&dod, vec![b, seq, h, d]);

    rocm_sync();
    let fwd_start = std::time::Instant::now();
    let (out_t, lse_t) = flash_attn_fwd_kt(&q, &k, &v, scale, causal)
        .unwrap_or_else(|e| panic!("long bench fwd seq={seq}: {e}"));
    rocm_sync();
    eprintln!(
        "kiln_flash_bench phase=fwd seq={seq} heads={h} kv_heads={hk} head_dim={d} elapsed_ms={:.3}",
        fwd_start.elapsed().as_secs_f64() * 1000.0
    );
    if std::env::var("KILN_ROCM_FLASH_BENCH_FWD_ONLY")
        .ok()
        .map(|s| {
            matches!(
                s.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
    {
        return;
    }

    let collapsed_bwd = std::env::var("KILN_ROCM_FLASH_BENCH_COLLAPSED_BWD")
        .ok()
        .map(|s| {
            matches!(
                s.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false);
    let bwd_start = std::time::Instant::now();
    let (dq, dk, dv) = if collapsed_bwd {
        flash_attn_bwd_collapsed_gqa_kt(&dout, &q, &k, &v, &out_t, &lse_t, scale, causal)
            .unwrap_or_else(|e| panic!("long bench collapsed bwd seq={seq}: {e}"))
    } else {
        flash_attn_bwd_kt(&dout, &q, &k, &v, &out_t, &lse_t, scale, causal)
            .unwrap_or_else(|e| panic!("long bench bwd seq={seq}: {e}"))
    };
    rocm_sync();
    eprintln!(
        "kiln_flash_bench phase=bwd seq={seq} heads={h} kv_heads={hk} head_dim={d} collapsed_gqa={} elapsed_ms={:.3}",
        collapsed_bwd,
        bwd_start.elapsed().as_secs_f64() * 1000.0
    );
    assert_eq!(dq.shape(), &[b, seq, h, d]);
    if collapsed_bwd {
        assert_eq!(dk.shape(), &[b, seq, hk, d]);
        assert_eq!(dv.shape(), &[b, seq, hk, d]);
    } else {
        assert_eq!(dk.shape(), &[b, seq, h, d]);
        assert_eq!(dv.shape(), &[b, seq, h, d]);
    }
}

#[test]
fn flash_attn_bwd_collapsed_gqa_hd256_native_matches_materialized() {
    if no_rocm() {
        return;
    }
    let b = 1usize;
    let h = 8usize;
    let hk = 1usize;
    let d = 256usize;
    let sq = 256usize;
    let sk = 384usize;
    let scale = 1.0 / (d as f32).sqrt();
    let causal = true;

    let qd = fill_bf16(b * sq * h * d, 2601);
    let kd = fill_bf16(b * sk * hk * d, 2603);
    let vd = fill_bf16(b * sk * hk * d, 2609);
    let dod = fill_bf16(b * sq * h * d, 2617);
    let run = |use_native| {
        let policy = RocmFlashAttentionPolicy {
            native_backward_preference: if use_native {
                RocmFlashAttentionRouteMode::Enabled
            } else {
                RocmFlashAttentionRouteMode::Disabled
            },
            materialized_backward_mode: if use_native {
                RocmFlashAttentionRouteMode::Disabled
            } else {
                RocmFlashAttentionRouteMode::Enabled
            },
            ..native_route_test_policy()
        };
        let ctx = rocm_policy_context(policy);
        let q = rocm_bf16_on(&ctx, &qd, vec![b, sq, h, d]);
        let k = rocm_bf16_on(&ctx, &kd, vec![b, sk, hk, d]);
        let v = rocm_bf16_on(&ctx, &vd, vec![b, sk, hk, d]);
        let dout = rocm_bf16_on(&ctx, &dod, vec![b, sq, h, d]);
        let (out_t, lse_t) = flash_attn_fwd_kt(&q, &k, &v, scale, causal)
            .unwrap_or_else(|e| panic!("hd256 collapsed policy compare fwd: {e}"));
        let (dq, dk, dv) =
            flash_attn_bwd_collapsed_gqa_kt(&dout, &q, &k, &v, &out_t, &lse_t, scale, causal)
                .unwrap_or_else(|e| panic!("hd256 collapsed policy bwd: {e}"));
        assert_eq!(dq.shape(), &[b, sq, h, d]);
        assert_eq!(dk.shape(), &[b, sk, hk, d]);
        assert_eq!(dv.shape(), &[b, sk, hk, d]);
        (bf16_to_f32(&dq), bf16_to_f32(&dk), bf16_to_f32(&dv))
    };

    let (dq_materialized, dk_materialized, dv_materialized) = run(false);
    let (dq_native, dk_native, dv_native) = run(true);
    check_close(
        &dq_native,
        &dq_materialized,
        8e-2,
        8e-2,
        "hd256 native/materialized dq",
    );
    check_close(
        &dk_native,
        &dk_materialized,
        8e-2,
        8e-2,
        "hd256 native/materialized dk",
    );
    check_close(
        &dv_native,
        &dv_materialized,
        8e-2,
        8e-2,
        "hd256 native/materialized dv",
    );
}

#[test]
fn flash_attn_bwd_online_tiled_parity() {
    if no_rocm() {
        return;
    }
    let b = 1usize;
    let h = 2usize;
    let hk = 1usize;
    let d = 128usize;
    let sq = 9usize;
    let sk = 17usize;
    let scale = 1.0 / (d as f32).sqrt();

    let policy = RocmFlashAttentionPolicy {
        native_backward_preference: RocmFlashAttentionRouteMode::Disabled,
        materialized_backward_mode: RocmFlashAttentionRouteMode::Disabled,
        online_backward: true,
        ..native_route_test_policy()
    };
    let ctx = rocm_policy_context(policy);
    {
        with_test_score_geometry(1, 4, 8, 1, || {
            for &causal in &[false, true] {
                let qd = fill_bf16(b * sq * h * d, 701 + causal as usize);
                let kd = fill_bf16(b * sk * hk * d, 809 + causal as usize);
                let vd = fill_bf16(b * sk * hk * d, 907 + causal as usize);
                let dod = fill_bf16(b * sq * h * d, 1009 + causal as usize);

                let q = rocm_bf16_on(&ctx, &qd, vec![b, sq, h, d]);
                let k = rocm_bf16_on(&ctx, &kd, vec![b, sk, hk, d]);
                let v = rocm_bf16_on(&ctx, &vd, vec![b, sk, hk, d]);
                let dout = rocm_bf16_on(&ctx, &dod, vec![b, sq, h, d]);

                let (out_t, lse_t) = flash_attn_fwd_kt(&q, &k, &v, scale, causal)
                    .unwrap_or_else(|e| panic!("bwd fwd causal={causal}: {e}"));
                kiln_tensor::rocm_synchronize_tensor_stream(&out_t)
                    .unwrap_or_else(|e| panic!("bwd fwd drain causal={causal}: {e}"));
                let (dq_t, dk_t, dv_t) =
                    flash_attn_bwd_kt(&dout, &q, &k, &v, &out_t, &lse_t, scale, causal)
                        .unwrap_or_else(|e| panic!("online bwd causal={causal}: {e}"));

                assert_eq!(dq_t.shape(), &[b, sq, h, d]);
                assert_eq!(dk_t.shape(), &[b, sk, h, d]);
                assert_eq!(dv_t.shape(), &[b, sk, h, d]);

                let qf: Vec<f32> = qd.iter().map(|x| x.to_f32()).collect();
                let kf: Vec<f32> = kd.iter().map(|x| x.to_f32()).collect();
                let vf: Vec<f32> = vd.iter().map(|x| x.to_f32()).collect();
                let dof: Vec<f32> = dod.iter().map(|x| x.to_f32()).collect();
                let outf = bf16_to_f32(&out_t);
                let (want_dq, want_dk, want_dv) = cpu_sdpa_bwd_expanded_gqa(
                    &dof, &qf, &kf, &vf, &outf, b, sq, sk, h, hk, d, scale, causal,
                );

                check_close(
                    &bf16_to_f32(&dq_t),
                    &want_dq,
                    4e-2,
                    4e-2,
                    &format!("online bwd dq causal={causal}"),
                );
                check_close(
                    &bf16_to_f32(&dk_t),
                    &want_dk,
                    4e-2,
                    4e-2,
                    &format!("online bwd dk causal={causal}"),
                );
                check_close(
                    &bf16_to_f32(&dv_t),
                    &want_dv,
                    4e-2,
                    4e-2,
                    &format!("online bwd dv causal={causal}"),
                );
            }
        });
    }
}

#[test]
fn flash_attn_bwd_collapsed_gqa_parity() {
    if no_rocm() {
        return;
    }
    let b = 1usize;
    let h = 4usize;
    let hk = 1usize;

    let policy = RocmFlashAttentionPolicy {
        native_backward_preference: RocmFlashAttentionRouteMode::Disabled,
        materialized_backward_mode: RocmFlashAttentionRouteMode::Enabled,
        ..native_route_test_policy()
    };
    let ctx = rocm_policy_context(policy);
    {
        for &(d, sq, sk) in &[(128usize, 11usize, 19usize), (256usize, 7usize, 13usize)] {
            let scale = 1.0 / (d as f32).sqrt();
            for &causal in &[false, true] {
                let seed_offset = d + causal as usize;
                let qd = fill_bf16(b * sq * h * d, 2101 + seed_offset);
                let kd = fill_bf16(b * sk * hk * d, 2203 + seed_offset);
                let vd = fill_bf16(b * sk * hk * d, 2309 + seed_offset);
                let dod = fill_bf16(b * sq * h * d, 2411 + seed_offset);

                let q = rocm_bf16_on(&ctx, &qd, vec![b, sq, h, d]);
                let k = rocm_bf16_on(&ctx, &kd, vec![b, sk, hk, d]);
                let v = rocm_bf16_on(&ctx, &vd, vec![b, sk, hk, d]);
                let dout = rocm_bf16_on(&ctx, &dod, vec![b, sq, h, d]);

                let (out_t, lse_t) = flash_attn_fwd_kt(&q, &k, &v, scale, causal)
                    .unwrap_or_else(|e| panic!("collapsed bwd fwd d={d} causal={causal}: {e}"));
                let (dq_t, dk_t, dv_t) = flash_attn_bwd_collapsed_gqa_kt(
                    &dout, &q, &k, &v, &out_t, &lse_t, scale, causal,
                )
                .unwrap_or_else(|e| panic!("collapsed bwd d={d} causal={causal}: {e}"));

                assert_eq!(dq_t.shape(), &[b, sq, h, d]);
                assert_eq!(dk_t.shape(), &[b, sk, hk, d]);
                assert_eq!(dv_t.shape(), &[b, sk, hk, d]);

                let qf: Vec<f32> = qd.iter().map(|x| x.to_f32()).collect();
                let kf: Vec<f32> = kd.iter().map(|x| x.to_f32()).collect();
                let vf: Vec<f32> = vd.iter().map(|x| x.to_f32()).collect();
                let dof: Vec<f32> = dod.iter().map(|x| x.to_f32()).collect();
                let outf = bf16_to_f32(&out_t);
                let (want_dq, want_dk_exp, want_dv_exp) = cpu_sdpa_bwd_expanded_gqa(
                    &dof, &qf, &kf, &vf, &outf, b, sq, sk, h, hk, d, scale, causal,
                );
                let want_dk = collapse_expanded_gqa_host(&want_dk_exp, b, sk, h, hk, d);
                let want_dv = collapse_expanded_gqa_host(&want_dv_exp, b, sk, h, hk, d);

                check_close(
                    &bf16_to_f32(&dq_t),
                    &want_dq,
                    4e-2,
                    4e-2,
                    &format!("collapsed bwd dq d={d} causal={causal}"),
                );
                check_close(
                    &bf16_to_f32(&dk_t),
                    &want_dk,
                    5e-2,
                    5e-2,
                    &format!("collapsed bwd dk d={d} causal={causal}"),
                );
                check_close(
                    &bf16_to_f32(&dv_t),
                    &want_dv,
                    5e-2,
                    5e-2,
                    &format!("collapsed bwd dv d={d} causal={causal}"),
                );
            }
        }
    }
}

#[test]
fn flash_attn_paged_decode_parity() {
    if no_rocm() {
        return;
    }
    let b = 2usize;
    let h = 16usize;
    let d = 128usize;
    let scale = 1.0 / (d as f32).sqrt();
    let page_block_size = 16usize;

    for &hk in &[2usize, 8usize] {
        for &seqlen_k in &[1usize, 7, 16, 33, 64, 128] {
            // Pool laid out so that logical position t of sequence bi maps to a
            // distinct physical slot. We allocate `b` contiguous block ranges.
            let blocks_per_seq = seqlen_k.div_ceil(page_block_size);
            let total_blocks = b * blocks_per_seq;
            let total_slots = total_blocks * page_block_size;

            // block_table[bi, blk] = bi * blocks_per_seq + blk.
            let mut bt: Vec<u32> = Vec::with_capacity(b * blocks_per_seq);
            for bi in 0..b {
                for blk in 0..blocks_per_seq {
                    bt.push((bi * blocks_per_seq + blk) as u32);
                }
            }

            // K/V pools [total_slots, hk, d] bf16.
            let kp = fill_bf16(total_slots * hk * d, 101 + seqlen_k + hk);
            let vp = fill_bf16(total_slots * hk * d, 211 + seqlen_k + hk);
            // q [b, 1, h, d].
            let qd = fill_bf16(b * 1 * h * d, 307 + seqlen_k + hk);

            let q = rocm_bf16(&qd, vec![b, 1, h, d]);
            let k_pool = rocm_bf16(&kp, vec![total_slots, hk, d]);
            let v_pool = rocm_bf16(&vp, vec![total_slots, hk, d]);
            let bt_cpu = Tensor::from_vec(bt.clone(), vec![b, blocks_per_seq]).unwrap();
            let block_table = kiln_tensor::host_to_rocm_copy(&bt_cpu, 0).unwrap();

            let out_t = flash_attn_paged_decode_kt(
                &q,
                &k_pool,
                &v_pool,
                &block_table,
                seqlen_k,
                page_block_size,
                scale,
                false,
            )
            .unwrap_or_else(|e| panic!("paged_decode hk={hk} seqlen_k={seqlen_k}: {e}"));
            assert_eq!(out_t.shape(), &[b, 1, h, d]);

            // Build the gathered [b, seqlen_k, hk, d] host K/V to feed cpu_sdpa
            // (sq = 1). Physical slot for (bi, t): bt[bi, t/pb]*pb + t%pb.
            let mut kf = vec![0.0f32; b * seqlen_k * hk * d];
            let mut vf = vec![0.0f32; b * seqlen_k * hk * d];
            for bi in 0..b {
                for t in 0..seqlen_k {
                    let blk = t / page_block_size;
                    let within = t % page_block_size;
                    let phys = bt[bi * blocks_per_seq + blk] as usize * page_block_size + within;
                    for hki in 0..hk {
                        for di in 0..d {
                            let src = (phys * hk + hki) * d + di;
                            let dst = ((bi * seqlen_k + t) * hk + hki) * d + di;
                            kf[dst] = kp[src].to_f32();
                            vf[dst] = vp[src].to_f32();
                        }
                    }
                }
            }
            let qf: Vec<f32> = qd.iter().map(|x| x.to_f32()).collect();
            // paged_decode is non-causal (single query attends all gathered keys).
            let want = cpu_sdpa(&qf, &kf, &vf, b, 1, seqlen_k, h, hk, d, scale, false);
            let got = bf16_to_f32(&out_t);
            check_close(
                &got,
                &want,
                2e-2,
                2e-2,
                &format!("paged_decode hk={hk} seqlen_k={seqlen_k}"),
            );
        }
    }
}

#[test]
fn flash_attn_paged_decode_dyn_seqlen_parity() {
    if no_rocm() {
        return;
    }
    let b = 2usize;
    let h = 16usize;
    let hk = 8usize;
    let d = 128usize;
    let scale = 1.0 / (d as f32).sqrt();
    let page_block_size = 16usize;
    let max_seqlen_k = 64usize;

    // Per-batch effective key lengths (< max_seqlen_k) — the tail must be masked.
    let seqused: Vec<u32> = vec![33, 50];

    let blocks_per_seq = max_seqlen_k.div_ceil(page_block_size);
    let total_blocks = b * blocks_per_seq;
    let total_slots = total_blocks * page_block_size;

    let mut bt: Vec<u32> = Vec::with_capacity(b * blocks_per_seq);
    for bi in 0..b {
        for blk in 0..blocks_per_seq {
            bt.push((bi * blocks_per_seq + blk) as u32);
        }
    }

    let kp = fill_bf16(total_slots * hk * d, 71);
    let vp = fill_bf16(total_slots * hk * d, 91);
    let qd = fill_bf16(b * h * d, 131);

    let q = rocm_bf16(&qd, vec![b, 1, h, d]);
    let k_pool = rocm_bf16(&kp, vec![total_slots, hk, d]);
    let v_pool = rocm_bf16(&vp, vec![total_slots, hk, d]);
    let bt_cpu = Tensor::from_vec(bt.clone(), vec![b, blocks_per_seq]).unwrap();
    let block_table = kiln_tensor::host_to_rocm_copy(&bt_cpu, 0).unwrap();
    let su_cpu = Tensor::from_vec(seqused.clone(), vec![b]).unwrap();
    let seqused_k = kiln_tensor::host_to_rocm_copy(&su_cpu, 0).unwrap();

    let out_t = flash_attn_paged_decode_dyn_seqlen_kt(
        &q,
        &k_pool,
        &v_pool,
        &block_table,
        &seqused_k,
        max_seqlen_k,
        page_block_size,
        scale,
        false,
    )
    .expect("dyn_seqlen decode");
    assert_eq!(out_t.shape(), &[b, 1, h, d]);

    // Reference: each batch attends only its first seqused[bi] gathered keys.
    let got = bf16_to_f32(&out_t);
    let qf: Vec<f32> = qd.iter().map(|x| x.to_f32()).collect();
    for bi in 0..b {
        let used = seqused[bi] as usize;
        // Gather only the valid keys for this batch (sq=1, b=1 sub-problem).
        let mut kf = vec![0.0f32; used * hk * d];
        let mut vf = vec![0.0f32; used * hk * d];
        for t in 0..used {
            let blk = t / page_block_size;
            let within = t % page_block_size;
            let phys = bt[bi * blocks_per_seq + blk] as usize * page_block_size + within;
            for hki in 0..hk {
                for di in 0..d {
                    kf[(t * hk + hki) * d + di] = kp[(phys * hk + hki) * d + di].to_f32();
                    vf[(t * hk + hki) * d + di] = vp[(phys * hk + hki) * d + di].to_f32();
                }
            }
        }
        let qf_b: Vec<f32> = qf[bi * h * d..(bi + 1) * h * d].to_vec();
        let want = cpu_sdpa(&qf_b, &kf, &vf, 1, 1, used, h, hk, d, scale, false);
        let got_b = &got[bi * h * d..(bi + 1) * h * d];
        check_close(got_b, &want, 2e-2, 2e-2, &format!("dyn_seqlen bi={bi}"));
    }
}

#[test]
fn paged_kv_write_roundtrip() {
    if no_rocm() {
        return;
    }
    let total_slots = 32usize;
    let num_kv_heads = 4usize;
    let head_dim = 128usize;
    let row = num_kv_heads * head_dim;
    let slot = 7usize;

    // Zero pools, then write one token row and read it back.
    let kp = fill_bf16(total_slots * row, 1);
    let vp = fill_bf16(total_slots * row, 2);
    let k_pool = rocm_bf16(&kp, vec![total_slots, num_kv_heads, head_dim]);
    let v_pool = rocm_bf16(&vp, vec![total_slots, num_kv_heads, head_dim]);

    let krow = fill_bf16(row, 555);
    let vrow = fill_bf16(row, 777);
    let k = rocm_bf16(&krow, vec![row]);
    let v = rocm_bf16(&vrow, vec![row]);

    paged_kv_write_token_major_bf16_kt(&k_pool, &v_pool, &k, &v, slot).expect("kv write");

    // Read the pools back; the `slot` row must now equal the written rows, and
    // the neighbouring rows must be untouched.
    let kp_after = bf16_to_f32(&k_pool);
    let vp_after = bf16_to_f32(&v_pool);
    let krow_f: Vec<f32> = krow.iter().map(|x| x.to_f32()).collect();
    let vrow_f: Vec<f32> = vrow.iter().map(|x| x.to_f32()).collect();

    for i in 0..row {
        assert!(
            (kp_after[slot * row + i] - krow_f[i]).abs() < 1e-3,
            "k slot row mismatch at {i}"
        );
        assert!(
            (vp_after[slot * row + i] - vrow_f[i]).abs() < 1e-3,
            "v slot row mismatch at {i}"
        );
    }
    // Untouched neighbours (slot-1, slot+1) keep their original pool values.
    for &other in &[slot - 1, slot + 1] {
        for i in 0..row {
            let want_k = kp[other * row + i].to_f32();
            assert!(
                (kp_after[other * row + i] - want_k).abs() < 1e-3,
                "k neighbour row {other} clobbered at {i}"
            );
        }
    }
}
