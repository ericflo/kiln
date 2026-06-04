//! #38 — On-box proof that a physical KV SHRINK is VRAM STRICTLY NON-INCREASING.
//! The old D2D path allocated each new (smaller) layer pool BEFORE freeing the
//! old one, so the HIP pool's reserved high-water bumped by ~one layer mid-shrink
//! (an overshoot — bad precisely when shrinking under pressure). The host-staged
//! path (D2H prefix → drop old → H2D back) never exceeds the pre-shrink size, so
//! the pool's `reserved` does not grow across the shrink.
//!
//! Runs in its OWN process so the HIP pool's reserved high-water starts clean
//! (it's a process-global hoard). A SMALL shrink (3/4) makes a D2D overshoot
//! ~one near-full layer — clearly above the noise — while host-staging stays flat.
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

fn reserved_mb() -> f64 {
    kiln_tensor::rocm_pool_stats(0).unwrap().0 as f64 / (1024.0 * 1024.0)
}

#[test]
fn rocm_shrink_is_vram_nonincreasing() {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("skip rocm_shrink_nonincreasing: no ROCm device");
        return;
    }
    let dev = Device::Rocm(0);
    let cache = PagedKvCacheKt::new(
        LAYERS, START_BLOCKS, BLOCK_SIZE, KV_HEADS, HEAD_DIM, DType::BF16, dev,
    )
    .expect("alloc cache");
    kiln_tensor::rocm_synchronize_default_stream(0).ok();
    let reserved_before = reserved_mb();
    // Per-layer pool bytes (k or v): blocks*block_size*kv_heads*head_dim*2(bf16).
    let layer_pool_mb =
        (START_BLOCKS * BLOCK_SIZE * KV_HEADS * HEAD_DIM * 2) as f64 / (1024.0 * 1024.0);
    eprintln!(
        "[shrink-noninc] reserved before shrink: {reserved_before:.0} MB (one k-pool ≈ {layer_pool_mb:.0} MB)"
    );

    cache.physical_resize_to(SHRUNK_BLOCKS, dev).expect("shrink");
    kiln_tensor::rocm_synchronize_default_stream(0).ok();
    let reserved_after = reserved_mb();
    let growth = reserved_after - reserved_before;
    eprintln!(
        "[shrink-noninc] reserved after shrink:  {reserved_after:.0} MB (growth {growth:+.0} MB)"
    );

    // Host-staging must NOT grow the pool's reserved high-water. A D2D
    // alloc-then-drop would bump it by ~one new layer pool (3/4 of {layer_pool_mb}).
    // Allow a small slack for allocator rounding, well below a layer.
    let slack = (layer_pool_mb * 0.25).max(64.0);
    assert!(
        growth <= slack,
        "SHRINK overshoot: reserved grew {growth:.0} MB (> {slack:.0} MB slack) — \
         host-staging should keep VRAM non-increasing"
    );
    assert_eq!(cache.num_blocks(), SHRUNK_BLOCKS);
    eprintln!("[shrink-noninc] OK: shrink did not grow reserved VRAM (growth {growth:+.0} MB)");
}
