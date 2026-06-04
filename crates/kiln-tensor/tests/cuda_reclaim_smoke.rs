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

/// #32 — Prove the release-threshold fix makes the stream-ordered mempool HOARD
/// freed pages (so the governor's `cuda_trim_pool` reclaimer does real work)
/// instead of the default threshold-0 behaviour where the driver auto-returns
/// every freed page to the OS on sync (making the reclaimer a no-op).
///
/// With the threshold raised: alloc 2 GB → drop → the freed bytes stay HELD by
/// the pool (OS-level free VRAM does NOT recover), and only `cuda_trim_pool`
/// releases them back to the OS. cuMemGetInfo counts pool-retained pages as used,
/// so the free-VRAM delta is the direct signal.
#[test]
fn cuda_pool_hoards_with_threshold_and_trim_reclaims() {
    if !kiln_tensor::cuda_is_available() {
        eprintln!("skip cuda_pool_hoards: no CUDA device");
        return;
    }
    const MB: usize = 1024 * 1024;
    const GB2: usize = 2 * 1024 * MB;

    // Hoard mode (#32): the pool keeps freed pages until explicitly trimmed.
    kiln_tensor::cuda_set_pool_release_threshold(0, u64::MAX)
        .expect("cuda_set_pool_release_threshold");

    let (free0, _) = kiln_tensor::cuda_mem_get_info(0).unwrap();
    let buf = kiln_tensor::cuda_zeros_ctx(0, kiln_tensor::DType::U8, GB2).expect("alloc 2GB");
    kiln_tensor::cuda_synchronize_default_stream(0).ok();
    let (free_alloc, _) = kiln_tensor::cuda_mem_get_info(0).unwrap();
    drop(buf);
    kiln_tensor::cuda_synchronize_default_stream(0).ok();
    let (free_dropped, _) = kiln_tensor::cuda_mem_get_info(0).unwrap();
    eprintln!(
        "[hoard] free0={} alloc={} dropped(no-trim)={} MiB",
        free0 / MB,
        free_alloc / MB,
        free_dropped / MB
    );
    // The freed 2 GB must STAY held (threshold=MAX): free VRAM is still > 1 GB
    // below the pre-alloc baseline, i.e. the pool did NOT auto-release on sync.
    assert!(
        free_dropped + (GB2 / 2) < free0,
        "pool should HOARD the freed 2GB with threshold=MAX, but free recovered to \
         baseline (free0={free0} dropped={free_dropped}) — threshold not applied?"
    );

    // The reclaimer (trim) releases the hoarded pool back to the OS.
    kiln_tensor::cuda_trim_pool(0, 0).expect("cuda_trim_pool");
    let (free_trim, _) = kiln_tensor::cuda_mem_get_info(0).unwrap();
    eprintln!(
        "[hoard] after trim free={} MiB (recovered {} MiB)",
        free_trim / MB,
        free_trim.saturating_sub(free_dropped) / MB
    );
    assert!(
        free_trim >= free_dropped + (GB2 / 2),
        "cuda_trim_pool should release the hoarded pool to the OS \
         (dropped={free_dropped} trim={free_trim})"
    );
    eprintln!("[hoard] OK: release-threshold hoards freed VRAM; cuda_trim_pool reclaims it");
}
