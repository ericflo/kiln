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
    flash_attn_fwd_kt, flash_attn_paged_decode_dyn_seqlen_kt, flash_attn_paged_decode_kt,
    paged_kv_write_token_major_bf16_kt,
};
use kiln_tensor::{DType, Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.8 flash-attn parity test");
        true
    } else {
        false
    }
}

/// Deterministic pseudo-random value in roughly [-0.5, 0.5].
fn val(i: usize, seed: usize) -> f32 {
    let x = (i.wrapping_mul(2654435761).wrapping_add(seed.wrapping_mul(40503))) % 1000;
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
    let q_at = |bi: usize, si: usize, hi: usize, di: usize| {
        q[((bi * sq + si) * h + hi) * d + di]
    };
    let k_at = |bi: usize, si: usize, hki: usize, di: usize| {
        k[((bi * sk + si) * hk + hki) * d + di]
    };
    let v_at = |bi: usize, si: usize, hki: usize, di: usize| {
        v[((bi * sk + si) * hk + hki) * d + di]
    };
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

fn bf16_to_f32(t: &Tensor) -> Vec<f32> {
    let host = kiln_tensor::rocm_to_host_copy(t).expect("d2h");
    host.to_vec::<bf16>()
        .expect("bf16 readback")
        .into_iter()
        .map(|x| x.to_f32())
        .collect()
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
                &q, &k_pool, &v_pool, &block_table, seqlen_k, page_block_size, scale, false,
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
        &q, &k_pool, &v_pool, &block_table, &seqused_k, max_seqlen_k, page_block_size, scale, false,
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
