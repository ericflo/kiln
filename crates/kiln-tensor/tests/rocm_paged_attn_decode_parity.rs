//! ROCm paged decode attention parity for Qwen3.5-4B's split GQA4 D=256 path.
//!
//! Run: `cargo test -p kiln-tensor --features rocm --test rocm_paged_attn_decode_parity -- --nocapture --test-threads=1`
#![cfg(feature = "rocm")]

use half::bf16;
use kiln_tensor::{DType, Device, Tensor};
use std::ffi::OsString;
use std::sync::Mutex;

const DISABLE_D256: &str = "KILN_DISABLE_ROCM_GQA_D256_PARALLEL";
static ENV_LOCK: Mutex<()> = Mutex::new(());

struct EnvRestore {
    key: &'static str,
    value: Option<OsString>,
}

impl EnvRestore {
    fn capture(key: &'static str) -> Self {
        Self {
            key,
            value: std::env::var_os(key),
        }
    }
}

impl Drop for EnvRestore {
    fn drop(&mut self) {
        // SAFETY: this integration test holds ENV_LOCK while mutating the
        // process environment, and restores the original value before exit.
        unsafe {
            match &self.value {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

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
fn rocm_paged_attn_gqa4_d256_split_matches_fallback() {
    if no_rocm() {
        return;
    }
    let _env_lock = ENV_LOCK.lock().expect("env lock poisoned");
    let _restore = EnvRestore::capture(DISABLE_D256);

    let b = 1usize;
    let h = 16usize;
    let hk = 4usize;
    let d = 256usize;
    let page_block_size = 64usize;
    let max_blocks = 32usize;
    let max_seqlen_k = page_block_size * max_blocks;
    let pool_rows = max_seqlen_k;

    let q = bf16_on_rocm(
        (0..b * h * d).map(|i| val(i, 17, 3)).collect(),
        vec![b, 1, h, d],
    );
    let k_pool = bf16_on_rocm(
        (0..pool_rows * hk * d).map(|i| val(i, 37, 11)).collect(),
        vec![pool_rows, hk, d],
    );
    let v_pool = bf16_on_rocm(
        (0..pool_rows * hk * d).map(|i| val(i, 53, 19)).collect(),
        vec![pool_rows, hk, d],
    );
    let block_table: Vec<u32> = (0..max_blocks)
        .map(|block| ((block * 17 + 5) % max_blocks) as u32)
        .collect();
    let block_table = Tensor::from_vec_on(Device::Rocm(0), block_table, vec![b, max_blocks])
        .expect("block table");
    let seqused_k =
        Tensor::from_vec_on(Device::Rocm(0), vec![max_seqlen_k as u32], vec![b])
            .expect("seqused_k");
    let scale = 1.0f32 / (d as f32).sqrt();

    // SAFETY: guarded by ENV_LOCK; the original value is restored by _restore.
    unsafe {
        std::env::set_var(DISABLE_D256, "1");
    }
    let fallback = kiln_tensor::rocm_paged_attn_decode_bf16(
        &q,
        &k_pool,
        &v_pool,
        &block_table,
        Some(&seqused_k),
        max_seqlen_k,
        page_block_size,
        scale,
    )
    .expect("fallback paged attention");
    kiln_tensor::rocm_synchronize_default_stream(0).expect("sync fallback");

    // SAFETY: guarded by ENV_LOCK; force the optimized D=256 path for compare.
    unsafe {
        std::env::remove_var(DISABLE_D256);
    }
    let fused = kiln_tensor::rocm_paged_attn_decode_bf16(
        &q,
        &k_pool,
        &v_pool,
        &block_table,
        Some(&seqused_k),
        max_seqlen_k,
        page_block_size,
        scale,
    )
    .expect("fused paged attention");
    kiln_tensor::rocm_synchronize_default_stream(0).expect("sync fused");

    assert_close(&read_bf16_to_f32(&fused), &read_bf16_to_f32(&fallback));
}
