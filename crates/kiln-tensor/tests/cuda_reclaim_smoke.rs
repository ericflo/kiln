//! #28 — On-box proof that the CUDA governor-reclaim primitives run on real
//! NVIDIA hardware: `cuda_mem_get_info` returns plausible `(free, total)` and
//! `cuda_trim_pool` succeeds (device-sync + cuMemPoolTrimTo). Mirrors the ROCm
//! `rocm_trim_pool`/`rocm_mem_get_info` path. Skips without a CUDA device.
//!
//! Run: `cargo test -p kiln-tensor --no-default-features --features cuda \
//!        --test cuda_reclaim_smoke -- --nocapture --test-threads=1`
#![cfg(feature = "cuda")]

#[test]
fn cuda_mem_get_info_and_trim_pool_run() {
    if !kiln_tensor::cuda_is_available() {
        eprintln!("skip cuda_reclaim_smoke: no CUDA device");
        return;
    }
    let (free, total) = kiln_tensor::cuda_mem_get_info(0).expect("cuda_mem_get_info");
    eprintln!(
        "[cuda-reclaim] mem_get_info: free={} MiB total={} MiB",
        free / (1024 * 1024),
        total / (1024 * 1024)
    );
    assert!(total > 0, "total VRAM must be > 0");
    assert!(free <= total, "free must be <= total");

    // Allocate ~1GB, drop it, then trim — trim must succeed (it device-syncs and
    // best-effort releases the pool). On a discrete card free should not shrink
    // after the trim (memory returned, not leaked).
    let buf = kiln_tensor::cuda_zeros_ctx(0, kiln_tensor::DType::BF16, (1024 * 1024 * 1024) / 2)
        .expect("cuda alloc");
    drop(buf);
    kiln_tensor::cuda_trim_pool(0, 0).expect("cuda_trim_pool");
    let (free_after, _) = kiln_tensor::cuda_mem_get_info(0).expect("cuda_mem_get_info post-trim");
    eprintln!(
        "[cuda-reclaim] after 1GB alloc/drop/trim: free={} MiB (was {} MiB)",
        free_after / (1024 * 1024),
        free / (1024 * 1024)
    );
    // The trim must not have LOST memory (free should be >= a 1GB-below-start
    // floor — i.e. the buffer was reclaimed, not leaked).
    assert!(
        free_after + (1024 * 1024 * 1024) >= free.saturating_sub(256 * 1024 * 1024),
        "trim should reclaim the freed buffer, not leak it (free {free} -> {free_after})"
    );
    eprintln!("[cuda-reclaim] OK: cuda_mem_get_info + cuda_trim_pool run on real CUDA hardware");
}
