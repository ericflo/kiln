//! ROCm paged decode attention parity for Qwen3.5-4B's split GQA4 D=256 path.
//!
//! Run: `cargo test -p kiln-tensor --features rocm --test rocm_paged_attn_decode_parity -- --nocapture --test-threads=1`
#![cfg(feature = "rocm")]

use half::bf16;
use kiln_tensor::{DType, Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping paged decode attention parity test");
        true
    } else {
        false
    }
}

fn val(i: usize, mul: usize, add: usize) -> f32 {
    (((i * mul + add) % 2048) as f32 - 1024.0) / 1024.0
}

fn bf16_on_rocm(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
    Tensor::from_vec_on(Device::Rocm(0), data, shape)
        .expect("from_vec_on f32")
        .to_dtype(DType::BF16)
        .expect("cast to bf16")
}

fn read_bf16_to_f32(t: &Tensor) -> Vec<f32> {
    let host = kiln_tensor::rocm_to_host_copy(t).expect("rocm_to_host_copy");
    host.to_vec::<bf16>()
        .expect("to_vec bf16")
        .into_iter()
        .map(|v| v.to_f32())
        .collect()
}

fn bf16_rounded(data: &[f32]) -> Vec<f32> {
    data.iter()
        .map(|&value| bf16::from_f32(value).to_f32())
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn cpu_paged_gqa_reference(
    q: &[f32],
    k_pool: &[f32],
    v_pool: &[f32],
    block_table: &[u32],
    b: usize,
    h: usize,
    hk: usize,
    d: usize,
    max_seqlen_k: usize,
    page_block_size: usize,
    scale: f32,
) -> Vec<f32> {
    let max_blocks = max_seqlen_k.div_ceil(page_block_size);
    let group = h / hk;
    let mut output = vec![0.0f32; b * h * d];
    let mut scores = vec![0.0f32; max_seqlen_k];

    for batch in 0..b {
        for head in 0..h {
            let kv_head = head / group;
            let q_row = &q[(batch * h + head) * d..(batch * h + head + 1) * d];
            for (token, score) in scores.iter_mut().enumerate() {
                let block = block_table[batch * max_blocks + token / page_block_size] as usize;
                let physical_row = block * page_block_size + token % page_block_size;
                let k_start = (physical_row * hk + kv_head) * d;
                *score = q_row
                    .iter()
                    .zip(&k_pool[k_start..k_start + d])
                    .map(|(&qv, &kv)| qv * kv)
                    .sum::<f32>()
                    * scale;
            }
            let score_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator = scores
                .iter_mut()
                .map(|score| {
                    *score = (*score - score_max).exp();
                    *score
                })
                .sum::<f32>();
            for (token, &unnormalized) in scores.iter().enumerate() {
                let block = block_table[batch * max_blocks + token / page_block_size] as usize;
                let physical_row = block * page_block_size + token % page_block_size;
                let v_start = (physical_row * hk + kv_head) * d;
                let probability = unnormalized / denominator;
                let out_start = (batch * h + head) * d;
                for dim in 0..d {
                    output[out_start + dim] += probability * v_pool[v_start + dim];
                }
            }
        }
    }
    output
}

fn assert_close(got: &[f32], reference: &[f32]) {
    assert_eq!(got.len(), reference.len(), "length mismatch");
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut max_idx = 0usize;
    for (i, (&g, &r)) in got.iter().zip(reference.iter()).enumerate() {
        assert!(g.is_finite(), "got non-finite value at {i}: {g}");
        assert!(r.is_finite(), "reference non-finite value at {i}: {r}");
        let abs = (g - r).abs();
        let rel = abs / r.abs().max(1.0e-3);
        if abs > max_abs {
            max_abs = abs;
            max_rel = rel;
            max_idx = i;
        }
    }
    assert!(
        max_abs <= 2.0e-2 || max_rel <= 2.0e-2,
        "paged decode attention mismatch: max_abs={max_abs} max_rel={max_rel} at idx={max_idx}, got={} reference={}",
        got[max_idx],
        reference[max_idx]
    );
}

#[test]
fn rocm_paged_attn_gqa4_d256_split_matches_cpu_reference() {
    if no_rocm() {
        return;
    }

    let b = 1usize;
    let h = 16usize;
    let hk = 4usize;
    let d = 256usize;
    let page_block_size = 64usize;
    let max_blocks = 32usize;
    let max_seqlen_k = page_block_size * max_blocks;
    let pool_rows = max_seqlen_k;

    let q_host: Vec<f32> = (0..b * h * d).map(|i| val(i, 17, 3)).collect();
    let k_host: Vec<f32> = (0..pool_rows * hk * d).map(|i| val(i, 37, 11)).collect();
    let v_host: Vec<f32> = (0..pool_rows * hk * d).map(|i| val(i, 53, 19)).collect();
    let q = bf16_on_rocm(q_host.clone(), vec![b, 1, h, d]);
    let k_pool = bf16_on_rocm(k_host.clone(), vec![pool_rows, hk, d]);
    let v_pool = bf16_on_rocm(v_host.clone(), vec![pool_rows, hk, d]);
    let block_table: Vec<u32> = (0..max_blocks)
        .map(|block| ((block * 17 + 5) % max_blocks) as u32)
        .collect();
    let block_table_device =
        Tensor::from_vec_on(Device::Rocm(0), block_table.clone(), vec![b, max_blocks])
            .expect("block table");
    let seqused_k = Tensor::from_vec_on(Device::Rocm(0), vec![max_seqlen_k as u32], vec![b])
        .expect("seqused_k");
    let scale = 1.0f32 / (d as f32).sqrt();

    let fused = kiln_tensor::rocm_paged_attn_decode_bf16(
        &q,
        &k_pool,
        &v_pool,
        &block_table_device,
        Some(&seqused_k),
        max_seqlen_k,
        page_block_size,
        scale,
    )
    .expect("fused paged attention");
    kiln_tensor::rocm_synchronize_default_stream(0).expect("sync fused");

    let reference = cpu_paged_gqa_reference(
        &bf16_rounded(&q_host),
        &bf16_rounded(&k_host),
        &bf16_rounded(&v_host),
        &block_table,
        b,
        h,
        hk,
        d,
        max_seqlen_k,
        page_block_size,
        scale,
    );
    assert_close(&read_bf16_to_f32(&fused), &reference);
}
