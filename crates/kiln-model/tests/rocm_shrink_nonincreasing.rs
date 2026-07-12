//! On-box proof that a committed physical KV shrink reduces live ROCm-pool use.
//! The resize transaction retains every old layer until all replacement layers
//! and copies are complete, so allocator `reserved` high-water may increase by
//! the staged replacement. That temporary headroom is the cost of making every
//! failure leave the old cache usable; the post-commit `used` value must still
//! fall by the removed KV bytes.
//!
//! Runs in its own process so unrelated allocations do not obscure the live-use
//! delta.
//!
//! Run: `cargo test -p kiln-model --no-default-features --features rocm \
//!        --test rocm_shrink_nonincreasing -- --nocapture --test-threads=1`
#![cfg(feature = "rocm")]

use kiln_model::PagedKvCacheKt;
use kiln_tensor::{DType, Device};

const LAYERS: usize = 4;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const BLOCK_SIZE: usize = 16;
const START_BLOCKS: usize = 4000; // ~1 GB
const SHRUNK_BLOCKS: usize = 3000; // small shrink → big D2D overshoot if present

fn used_mb() -> f64 {
    kiln_tensor::rocm_pool_stats(0).unwrap().1 as f64 / (1024.0 * 1024.0)
}

#[test]
fn rocm_shrink_reduces_live_pool_use() {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("skip rocm_shrink_nonincreasing: no ROCm device");
        return;
    }
    let dev = Device::Rocm(0);
    let cache = PagedKvCacheKt::new(
        LAYERS,
        START_BLOCKS,
        BLOCK_SIZE,
        KV_HEADS,
        HEAD_DIM,
        DType::BF16,
        dev,
    )
    .expect("alloc cache");
    kiln_tensor::rocm_synchronize_default_stream(0).ok();
    let used_before = used_mb();
    // Per-layer pool bytes (k or v): blocks*block_size*kv_heads*head_dim*2(bf16).
    let layer_pool_mb =
        (START_BLOCKS * BLOCK_SIZE * KV_HEADS * HEAD_DIM * 2) as f64 / (1024.0 * 1024.0);
    eprintln!(
        "[shrink-live-use] used before shrink: {used_before:.0} MB (one k-pool ~= {layer_pool_mb:.0} MB)"
    );

    cache
        .physical_resize_to(SHRUNK_BLOCKS, dev)
        .expect("shrink");
    kiln_tensor::rocm_synchronize_default_stream(0).ok();
    let used_after = used_mb();
    let reduction = used_before - used_after;
    eprintln!(
        "[shrink-live-use] used after shrink: {used_after:.0} MB (reduction {reduction:.0} MB)"
    );

    let expected_reduction =
        ((START_BLOCKS - SHRUNK_BLOCKS) * BLOCK_SIZE * KV_HEADS * HEAD_DIM * 2 * LAYERS * 2) as f64
            / (1024.0 * 1024.0);
    assert!(
        reduction >= expected_reduction * 0.9,
        "shrink did not release the expected live KV allocation: reduced {reduction:.0} MB, expected about {expected_reduction:.0} MB"
    );
    assert_eq!(cache.num_blocks(), SHRUNK_BLOCKS);
    assert_eq!(cache.pool_identity().generation, 1);
    eprintln!("[shrink-live-use] OK: committed shrink reduced live pool use");
}
