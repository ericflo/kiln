//! #26 — CUDA verification of the exact device primitives the paged-KV physical
//! resize relies on: `cuda_zeros_ctx` (pool alloc), `slice_set` (prefix copy),
//! `narrow` (prefix view), and the device sync — WITHOUT pulling in kiln-model /
//! the heavy flash-attn CUTLASS build. The `physical_resize_to` ORCHESTRATION is
//! device-agnostic and already proven on CPU + ROCm (and its CPU path passes on
//! this same A6000); this confirms its CUDA building blocks are sound, so the
//! whole CUDA resize path is covered by composition.
//!
//! Run: `cargo test -p kiln-tensor --no-default-features --features cuda \
//!        --test cuda_resize_copy_primitives -- --nocapture --test-threads=1`
#![cfg(feature = "cuda")]

use kiln_tensor::{DType, Device, Layout, Tensor, TensorId, cuda_zeros_ctx};

// A KV-pool-shaped buffer: [total_slots, num_kv_heads, head_dim].
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;

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
    eprintln!("skip cuda_resize_copy: no CUDA device");
    false
}

fn cuda_pool(total_slots: usize) -> Tensor {
    let n = total_slots * KV_HEADS * HEAD_DIM;
    let storage = cuda_zeros_ctx(0, DType::BF16, n).expect("cuda_zeros_ctx");
    Tensor::from_parts(
        storage,
        Layout::contiguous(vec![total_slots, KV_HEADS, HEAD_DIM]),
        TensorId::next(),
    )
    .expect("wrap cuda pool")
}

#[test]
fn cuda_pool_prefix_copy_preserves_bytes() {
    if !cuda_available_or_skip() {
        return;
    }
    let dev = Device::Cuda(0);

    // OLD pool of 256 slots; write a marker into slot 7 (as the KV writer does).
    let old = cuda_pool(256);
    assert!(matches!(old.device(), Device::Cuda(_)));
    let marker: Vec<f32> = (0..(KV_HEADS * HEAD_DIM))
        .map(|i| (i % 19) as f32 + 1.0)
        .collect();
    let marker_t = Tensor::from_vec(marker.clone(), vec![1, KV_HEADS, HEAD_DIM])
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(dev)
        .unwrap();
    old.slice_set(&marker_t, 0, 7).expect("write marker slot 7");

    let read_slot7 = |t: &Tensor| -> Vec<f32> {
        t.narrow(0, 7, 1)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec()
            .unwrap()
    };
    let reference = read_slot7(&old);

    // SHRINK-style copy: new smaller pool (128 slots), copy prefix [0,128) via the
    // same narrow + slice_set physical_resize_to uses. Marker (slot 7 < 128) must
    // survive byte-for-byte.
    let new_small = cuda_pool(128);
    kiln_tensor::cuda_synchronize_default_stream(0).ok();
    let src = old.narrow(0, 0, 128).unwrap();
    new_small
        .slice_set(&src, 0, 0)
        .expect("copy prefix into smaller pool");
    kiln_tensor::cuda_synchronize_default_stream(0).ok();
    assert_eq!(
        read_slot7(&new_small),
        reference,
        "marker not preserved on CUDA shrink-copy"
    );

    // GROW-style copy: new larger pool (512 slots), copy all old slots into the
    // prefix; tail stays zero.
    let new_big = cuda_pool(512);
    kiln_tensor::cuda_synchronize_default_stream(0).ok();
    let src_all = old.narrow(0, 0, 256).unwrap();
    new_big
        .slice_set(&src_all, 0, 0)
        .expect("copy prefix into larger pool");
    kiln_tensor::cuda_synchronize_default_stream(0).ok();
    assert_eq!(
        read_slot7(&new_big),
        reference,
        "marker not preserved on CUDA grow-copy"
    );

    eprintln!(
        "[cuda-primitives] OK: cuda_zeros_ctx + narrow + slice_set preserve KV bytes across shrink+grow copies"
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
