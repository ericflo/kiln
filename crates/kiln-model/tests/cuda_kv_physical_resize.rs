//! #26 — CUDA twin of `rocm_kv_physical_resize`: proves
//! `PagedKvCacheKt::physical_resize_to` preserves live KV byte-for-byte across a
//! shrink AND a grow on a real `Device::Cuda` device, exercising the CUDA arm of
//! `alloc_pool_pair` (`cuda_zeros_ctx`) + the device-agnostic narrow/slice_set
//! copy + the entry/zero-fill device syncs. Normal runs skip without CUDA;
//! qualification runs fail on missing hardware or insufficient free VRAM.
//!
//! Run: `cargo test -p kiln-model --no-default-features --features cuda \
//!        --test cuda_kv_physical_resize -- --nocapture --test-threads=1`
#![cfg(feature = "cuda")]

use kiln_model::PagedKvCacheKt;
use kiln_tensor::{DType, Device, Tensor};

const LAYERS: usize = 4;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const BLOCK_SIZE: usize = 16;
const NUM_BLOCKS: usize = 4000;
const SHRUNK_BLOCKS: usize = 500;
const MINIMUM_FREE_BYTES: usize = 3 * 1024 * 1024 * 1024;

fn qualification_required(value: Option<&str>) -> bool {
    value == Some("1")
}

fn cuda_available_or_skip() -> bool {
    if kiln_tensor::cuda_is_available() {
        return true;
    }
    if qualification_required(std::env::var("KILN_QUALIFICATION").ok().as_deref()) {
        panic!("CUDA device unavailable while KILN_QUALIFICATION=1");
    }
    eprintln!("skip cuda_physical_resize: no CUDA device");
    false
}

#[test]
fn cuda_physical_resize_preserves_kv() {
    if !cuda_available_or_skip() {
        return;
    }
    let (free, total) = kiln_tensor::cuda_mem_get_info(0).expect("read CUDA memory capacity");
    assert!(
        free >= MINIMUM_FREE_BYTES,
        "cuda_physical_resize requires at least {} MiB free, observed {} MiB of {} MiB",
        MINIMUM_FREE_BYTES / (1024 * 1024),
        free / (1024 * 1024),
        total / (1024 * 1024)
    );
    let dev = Device::Cuda(0);

    let cache = PagedKvCacheKt::new(
        LAYERS,
        NUM_BLOCKS,
        BLOCK_SIZE,
        KV_HEADS,
        HEAD_DIM,
        DType::BF16,
        dev,
    )
    .expect("alloc cuda kv cache");
    {
        let (k0, _) = cache.pool_tensors(0).expect("layer 0");
        assert!(
            matches!(k0.device(), Device::Cuda(_)),
            "pools must be device-resident (got {:?})",
            k0.device()
        );
    }

    // Marker into slot 7 of layer 0's K pool; reference = bytes as stored.
    let marker: Vec<f32> = (0..(KV_HEADS * HEAD_DIM))
        .map(|i| (i % 17) as f32 + 1.0)
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

    let read_marker = |c: &PagedKvCacheKt| -> Vec<f32> {
        c.pool_tensors(0)
            .unwrap()
            .0
            .narrow(0, 7, 1)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec()
            .unwrap()
    };
    let reference = read_marker(&cache);
    let assert_marker = |got: &[f32], when: &str| {
        assert_eq!(
            got,
            &reference[..],
            "marker slot 7 not preserved byte-for-byte {when}"
        );
    };

    // SHRINK (data preserved) then GROW (prefix preserved).
    cache
        .physical_resize_to(SHRUNK_BLOCKS, dev)
        .expect("shrink");
    assert_eq!(cache.num_blocks(), SHRUNK_BLOCKS);
    assert_marker(&read_marker(&cache), "across cuda shrink");

    cache.physical_resize_to(NUM_BLOCKS, dev).expect("grow");
    assert_eq!(cache.num_blocks(), NUM_BLOCKS);
    assert_marker(&read_marker(&cache), "across cuda grow");
    eprintln!(
        "[cuda-resize] OK: marker preserved across shrink {NUM_BLOCKS}->{SHRUNK_BLOCKS}->{NUM_BLOCKS}"
    );
}

#[test]
fn qualification_mode_is_exact_opt_in() {
    assert!(qualification_required(Some("1")));
    assert!(!qualification_required(None));
    assert!(!qualification_required(Some("")));
    assert!(!qualification_required(Some("0")));
    assert!(!qualification_required(Some("true")));
}
