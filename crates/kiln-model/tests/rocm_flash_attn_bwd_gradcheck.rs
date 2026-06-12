//! #37 — ACCURATE gradcheck of the ROCm flash-attn tape backward.
//!
//! `flash_attn_bwd_rocm` (the Phase R.8 composite) was DEAD CODE until #37 wired
//! `try_tape_flash_attn_kt` to record `FlashAttnBackward` on ROCm. Before trusting
//! it as the ROCm training attention backward we verify dq/dk/dv against an
//! INDEPENDENT, host-scalar, F32 analytic attention backward that shares ZERO code
//! with the rocm kernel — so a wrong transpose, causal mask, softmax-scale, or GQA
//! collapse cannot hide. (A BF16 finite-difference gradcheck is inconclusive here:
//! the dq/dk grads are small and sit below the BF16 forward's quantization noise.)
//!
//! The kernel casts its BF16 q/k/v to F32 internally before the backward, so we
//! feed the host reference the SAME BF16-quantized values (as F32) — making the
//! comparison tight (only rocm-matmul/softmax vs host-scalar F32 numerics differ).
//!
//! Covers head_dim ∈ {128, 256} (256 is Qwen3.5-4B's actual head_dim) with GQA.
//!
//! Run: `cargo test -p kiln-model --no-default-features --features rocm \
//!        --test rocm_flash_attn_bwd_gradcheck -- --nocapture --test-threads=1`
#![cfg(feature = "rocm")]

use kiln_tensor::{DType, Device, Tensor};

fn bf16_round(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}
fn host_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec()
        .unwrap()
}

/// Independent host-scalar F32 causal-GQA attention backward (b=1).
/// q: [sq,hq,hd], k/v: [sk,hkv,hd], seed = dL/dout [sq,hq,hd] — all BF16-quantized
/// values as F32. Returns (dq[sq,hq,hd], dk[sk,hkv,hd], dv[sk,hkv,hd]).
#[allow(clippy::too_many_arguments)]
fn ref_backward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seed: &[f32],
    sq: usize,
    sk: usize,
    hq: usize,
    hkv: usize,
    hd: usize,
    scale: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let groups = hq / hkv;
    let qi = |i: usize, h: usize, d: usize| q[(i * hq + h) * hd + d];
    let ki = |j: usize, hk: usize, d: usize| k[(j * hkv + hk) * hd + d];
    let vi = |j: usize, hk: usize, d: usize| v[(j * hkv + hk) * hd + d];
    let si = |i: usize, h: usize, d: usize| seed[(i * hq + h) * hd + d];

    let mut dq = vec![0.0f32; sq * hq * hd];
    let mut dk = vec![0.0f32; sk * hkv * hd];
    let mut dv = vec![0.0f32; sk * hkv * hd];

    for h in 0..hq {
        let hk = h / groups;
        for i in 0..sq {
            let n = i + 1; // causal: keys 0..=i
            let mut scores = vec![0.0f32; n];
            for j in 0..n {
                let mut s = 0.0f32;
                for d in 0..hd {
                    s += qi(i, h, d) * ki(j, hk, d);
                }
                scores[j] = s * scale;
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            let p: Vec<f32> = {
                let ex: Vec<f32> = scores.iter().map(|&s| (s - m).exp()).collect();
                for &e in &ex {
                    sum += e;
                }
                ex.iter().map(|&e| e / sum).collect()
            };
            let mut dp = vec![0.0f32; n];
            for j in 0..n {
                let mut acc = 0.0f32;
                for d in 0..hd {
                    acc += si(i, h, d) * vi(j, hk, d);
                }
                dp[j] = acc;
            }
            let pdp: f32 = (0..n).map(|j| p[j] * dp[j]).sum();
            let ds: Vec<f32> = (0..n).map(|j| p[j] * (dp[j] - pdp)).collect();
            for d in 0..hd {
                let mut dq_acc = 0.0f32;
                for j in 0..n {
                    dv[(j * hkv + hk) * hd + d] += p[j] * si(i, h, d);
                    dq_acc += ds[j] * ki(j, hk, d);
                    dk[(j * hkv + hk) * hd + d] += scale * ds[j] * qi(i, h, d);
                }
                dq[(i * hq + h) * hd + d] = scale * dq_acc;
            }
        }
    }
    (dq, dk, dv)
}

fn compare(name: &str, reference: &[f32], got: &[f32]) -> (f32, usize) {
    assert_eq!(reference.len(), got.len(), "{name}: length mismatch");
    assert!(
        got.iter().all(|x| x.is_finite()),
        "{name}: non-finite grads from kernel"
    );
    let scale = reference.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let thresh = scale * 0.02;
    let mut max_rel = 0.0f32;
    let mut n = 0usize;
    let mut worst = (0usize, 0.0f32, 0.0f32);
    for (idx, (&r, &g)) in reference.iter().zip(got).enumerate() {
        if r.abs() < thresh {
            continue;
        }
        n += 1;
        let denom = r.abs().max(g.abs()).max(1e-4);
        let re = (r - g).abs() / denom;
        if re > max_rel {
            max_rel = re;
            worst = (idx, r, g);
        }
    }
    eprintln!(
        "[gradcheck {name}] significant={n} max_rel={max_rel:.4} (worst idx={} ref={:+.5} got={:+.5})",
        worst.0, worst.1, worst.2
    );
    (max_rel, n)
}

fn run_gradcheck(sq: usize, sk: usize, hq: usize, hkv: usize, hd: usize) {
    unsafe {
        std::env::set_var("KILN_USE_TAPE_FORWARD", "1");
        std::env::set_var("KILN_USE_TAPE_FLASH_ATTN", "1");
    }
    let scale = 1.0 / (hd as f32).sqrt();
    let q_f: Vec<f32> = (0..sq * hq * hd)
        .map(|i| bf16_round((((i * 17) % 23) as f32 - 11.0) * 0.05))
        .collect();
    let k_f: Vec<f32> = (0..sk * hkv * hd)
        .map(|i| bf16_round((((i * 13) % 29) as f32 - 14.0) * 0.04))
        .collect();
    let v_f: Vec<f32> = (0..sk * hkv * hd)
        .map(|i| bf16_round((((i * 7) % 19) as f32 - 9.0) * 0.06))
        .collect();
    let seed_f: Vec<f32> = (0..sq * hq * hd)
        .map(|i| bf16_round((((i * 5) % 11) as f32 - 5.0) * 0.1))
        .collect();

    let (ref_dq, ref_dk, ref_dv) =
        ref_backward(&q_f, &k_f, &v_f, &seed_f, sq, sk, hq, hkv, hd, scale);

    let mk = |data: &[f32], shape: Vec<usize>| {
        Tensor::from_vec(data.to_vec(), shape)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(Device::Rocm(0))
            .unwrap()
            .contiguous()
            .unwrap()
    };
    let q = mk(&q_f, vec![1, sq, hq, hd]);
    let k = mk(&k_f, vec![1, sk, hkv, hd]);
    let v = mk(&v_f, vec![1, sk, hkv, hd]);
    let seed = mk(&seed_f, vec![1, sq, hq, hd]);

    let (out, tape) = kiln_model::tape_forward::with_thread_local_tape(|| {
        kiln_model::tape_forward::try_tape_flash_attn_kt(&q, &k, &v, hq, hkv, hd)
    });
    let out = out
        .expect("try_tape_flash_attn_kt errored")
        .expect("returned None — flash-attn did NOT record on ROCm (gate rejected)");
    assert_eq!(
        tape.len(),
        1,
        "flash-attn must record exactly one tape node"
    );
    let grads = tape
        .backward(out.id(), seed, |a, b| kiln_tensor::ops::add(a, b))
        .expect("tape backward on ROCm flash-attn graph");
    let got_dq = host_f32(grads.get(q.id()).expect("dq"));
    let got_dk = host_f32(grads.get(k.id()).expect("dk"));
    let got_dv = host_f32(grads.get(v.id()).expect("dv"));

    // 6% accommodates hipBLASLt's reduced-precision matmul (~3.5% worst on dv);
    // a structural bug (transpose/mask/scale/GQA) blows past it by 10-100x.
    const TOL: f32 = 0.06;
    let (rq, nq) = compare("dq", &ref_dq, &got_dq);
    let (rk, nk) = compare("dk", &ref_dk, &got_dk);
    let (rv, nv) = compare("dv", &ref_dv, &got_dv);
    assert!(
        nq > 5 && nk > 5 && nv > 5,
        "too few significant grads to be a real check"
    );
    assert!(
        rq < TOL,
        "dq disagrees with F32 reference (max_rel {rq:.4} > {TOL}) — ROCm flash-attn dq WRONG"
    );
    assert!(
        rk < TOL,
        "dk disagrees with F32 reference (max_rel {rk:.4} > {TOL}) — ROCm flash-attn dk WRONG"
    );
    assert!(
        rv < TOL,
        "dv disagrees with F32 reference (max_rel {rv:.4} > {TOL}) — ROCm flash-attn dv WRONG"
    );
    eprintln!("[rocm-flash-bwd hd={hd}] OK: dq/dk/dv match the independent F32 analytic reference");
}

#[test]
fn rocm_flash_attn_bwd_gradcheck_hd128() {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("skip: no ROCm device");
        return;
    }
    run_gradcheck(6, 6, 4, 2, 128);
}

#[test]
fn rocm_flash_attn_bwd_gradcheck_hd256() {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("skip: no ROCm device");
        return;
    }
    // head_dim=256 + 16 q-heads / 4 kv-heads = Qwen3.5-4B's actual attention shape.
    run_gradcheck(6, 6, 16, 4, 256);
}
