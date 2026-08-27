//! #26 — On-box proof that `PagedKvCacheKt::physical_resize_to` is a REAL
//! elastic VRAM actuator on `Device::Rocm`:
//!   1. Surviving KV is preserved byte-for-byte across the realloc (shrink+grow).
//!      The first newly grown slot is correct after its mandatory first write.
//!   2. A shrink RECLAIMS: it frees the KV bytes back to the device pool (pool
//!      `used` drops), and a subsequent ("training") allocation REUSES them
//!      WITHOUT growing the pool's reservation. That same-process, in-pool reuse
//!      — not returning memory to the OS — is what makes training/inference VRAM
//!      arbitration work and "never OOM".
//!
//! Memory is measured via `rocm_pool_stats` (the HIP pool's own reserved/used
//! counters): PROCESS-ISOLATED, immune to the coexisting llama-server that makes
//! the all-process DRM counters unusable for this measurement.
//!
//! Run: `cargo test -p kiln-model --no-default-features --features rocm \
//!        --test rocm_kv_physical_resize -- --nocapture --test-threads=1`
#![cfg(feature = "rocm")]

use kiln_model::PagedKvCacheKt;
use kiln_tensor::{DType, Device, Tensor};

// ~2 GB cache: per-slot bytes = KV_HEADS*HEAD_DIM*2(bf16); 2 pools/layer.
const LAYERS: usize = 4;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const BLOCK_SIZE: usize = 16;
const NUM_BLOCKS: usize = 8000;
const SHRUNK_BLOCKS: usize = 500;

fn pool_mb() -> (f64, f64) {
    let (r, u) = kiln_tensor::rocm_pool_stats(0).unwrap();
    (r as f64 / (1024.0 * 1024.0), u as f64 / (1024.0 * 1024.0))
}

#[test]
fn rocm_physical_resize_preserves_kv_and_reclaims_for_reuse() {
    if !kiln_tensor::rocm_is_available() {
        assert_ne!(
            std::env::var("KILN_QUALIFICATION").ok().as_deref(),
            Some("1"),
            "ROCm qualification requires a real device"
        );
        eprintln!("skip rocm_physical_resize: no ROCm device");
        return;
    }
    let dev = Device::Rocm(0);

    let cache = PagedKvCacheKt::new(
        LAYERS,
        NUM_BLOCKS,
        BLOCK_SIZE,
        KV_HEADS,
        HEAD_DIM,
        DType::BF16,
        dev,
    )
    .expect("alloc rocm kv cache");
    {
        let (k0, _) = cache.pool_tensors(0).expect("layer 0");
        assert!(
            matches!(k0.device(), Device::Rocm(_)),
            "pools must be device-resident (got {:?})",
            k0.device()
        );
    }

    // Marker into slot 7 of layer 0's K pool; reference read = bytes as stored
    // (BF16-rounded). A physical resize copies bytes verbatim, so later reads
    // must equal this EXACTLY.
    let marker: Vec<f32> = (0..(KV_HEADS * HEAD_DIM))
        .map(|i| (i as f32) * 0.5 + 1.0)
        .collect();
    let marker_t = Tensor::from_vec(marker.clone(), vec![1, KV_HEADS, HEAD_DIM])
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(dev)
        .unwrap();
    cache
        .pool_tensors(0)
        .unwrap()
        .0
        .slice_set(&marker_t, 0, 7)
        .expect("write marker");
    let read_slot = |c: &PagedKvCacheKt, slot: usize| -> Vec<f32> {
        c.pool_tensors(0)
            .unwrap()
            .0
            .narrow(0, slot, 1)
            .unwrap()
            .to_device(Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec()
            .unwrap()
    };
    let reference = read_slot(&cache, 7);
    let assert_marker = |got: &[f32], when: &str| {
        assert_eq!(
            got,
            &reference[..],
            "marker slot 7 not preserved byte-for-byte {when}"
        );
    };

    kiln_tensor::rocm_synchronize_default_stream(0).ok();
    let (res_full, used_full) = pool_mb();
    eprintln!("[resize] full cache: reserved={res_full:.0} used={used_full:.0} MB");
    assert!(
        used_full > 1500.0,
        "full cache should use ~2GB (got {used_full:.0})"
    );

    // SHRINK: realloc each layer's pool down to SHRUNK_BLOCKS → frees ~94% of KV.
    cache
        .physical_resize_to(SHRUNK_BLOCKS, dev)
        .expect("shrink");
    assert_eq!(cache.num_blocks(), SHRUNK_BLOCKS);
    kiln_tensor::rocm_synchronize_default_stream(0).ok();
    let (_res_shrunk, used_shrunk) = pool_mb();
    eprintln!("[resize] after shrink: used={used_shrunk:.0} MB (was {used_full:.0})");
    assert_marker(&read_slot(&cache, 7), "across shrink");
    assert!(
        used_full - used_shrunk > 1500.0,
        "shrink must free the KV bytes back to the pool: used {used_full:.0} -> {used_shrunk:.0} MB"
    );

    // RECLAIM PROOF: a ~1.8GB training-like workload (chunked) after the shrink
    // must REUSE the freed KV bytes — pool RESERVED must not grow past the full
    // cache's high-water. A leak would push reserved up by ~1.8GB.
    let chunk_elems = (200usize * 1024 * 1024) / 2;
    let mut training = Vec::new();
    for index in 0..9 {
        training
            .push(kiln_tensor::rocm_zeros_ctx(0, DType::BF16, chunk_elems).expect("train chunk"));
        kiln_tensor::rocm_synchronize_default_stream(0).ok();
        assert_marker(
            &read_slot(&cache, 7),
            &format!("after training allocation {index}"),
        );
    }
    kiln_tensor::rocm_synchronize_default_stream(0).ok();
    let (res_train, used_train) = pool_mb();
    let reserved_growth = res_train - res_full;
    eprintln!(
        "[resize] after +1.8GB training: reserved={res_train:.0} used={used_train:.0} MB \
         (reserved growth vs full-cache hwm: {reserved_growth:.0})"
    );
    assert!(
        reserved_growth < 400.0,
        "training after shrink must REUSE freed KV (reserved grew {reserved_growth:.0} MB; \
         0=perfect reuse, ~1800=leak)"
    );
    // Localize any corruption: is the marker still intact in the LIVE shrunk
    // pool after the training workload (before grow)?
    assert_marker(&read_slot(&cache, 7), "after training, before grow");
    drop(training);
    kiln_tensor::rocm_synchronize_default_stream(0).ok();
    assert_marker(&read_slot(&cache, 7), "after training drop, before grow");

    // GROW back to full: prefix (marker) intact, num_blocks restored.
    cache.physical_resize_to(NUM_BLOCKS, dev).expect("grow");
    assert_eq!(cache.num_blocks(), NUM_BLOCKS);
    assert_marker(&read_slot(&cache, 7), "across grow");

    // ROCm replacement pools deliberately leave the free grown tail
    // uninitialized: block tables cannot address it until assignment, and the
    // KV writer initializes a slot before attention can read it. Prove the
    // first newly addressable slot round-trips exactly after that first write.
    let first_grown_slot = SHRUNK_BLOCKS * BLOCK_SIZE;
    cache
        .pool_tensors(0)
        .unwrap()
        .0
        .slice_set(&marker_t, 0, first_grown_slot)
        .expect("write first grown slot");
    assert_marker(
        &read_slot(&cache, first_grown_slot),
        "after first write to grown tail",
    );
    eprintln!(
        "[resize] grow OK; marker intact; num_blocks={}",
        cache.num_blocks()
    );
}
