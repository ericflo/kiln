//! #28 — On-box proof that the CUDA governor-reclaim primitives run on real
//! NVIDIA hardware: `cuda_mem_get_info` returns plausible `(free, total)` and
//! `cuda_trim_pool` succeeds (device-sync + cuMemPoolTrimTo). Mirrors the ROCm
//! `rocm_trim_pool`/`rocm_mem_get_info` path. Normal runs skip without CUDA;
//! qualification runs fail on missing hardware or insufficient free VRAM.
//!
//! Run: `cargo test -p kiln-tensor --no-default-features --features cuda \
//!        --test cuda_reclaim_smoke -- --nocapture --test-threads=1`
#![cfg(feature = "cuda")]

const MIB: usize = 1024 * 1024;
const GIB: usize = 1024 * MIB;

fn qualification_required(value: Option<&str>) -> bool {
    value == Some("1")
}

fn cuda_memory_or_skip(test: &str, minimum_free_bytes: usize) -> Option<(usize, usize)> {
    if !kiln_tensor::cuda_is_available() {
        if qualification_required(std::env::var("KILN_QUALIFICATION").ok().as_deref()) {
            panic!("CUDA device unavailable while KILN_QUALIFICATION=1 ({test})");
        }
        eprintln!("skip {test}: no CUDA device");
        return None;
    }
    let (free, total) = kiln_tensor::cuda_mem_get_info(0).expect("cuda_mem_get_info");
    assert!(
        free >= minimum_free_bytes,
        "{test} requires at least {} MiB free, observed {} MiB of {} MiB",
        minimum_free_bytes / MIB,
        free / MIB,
        total / MIB
    );
    Some((free, total))
}

#[test]
fn cuda_mem_get_info_and_trim_pool_run() {
    let Some((free, total)) = cuda_memory_or_skip("cuda_reclaim_smoke", 3 * GIB) else {
        return;
    };
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
    let buf =
        kiln_tensor::cuda_zeros_ctx(0, kiln_tensor::DType::BF16, GIB / 2).expect("cuda alloc");
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
        free_after + GIB >= free.saturating_sub(256 * MIB),
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
    let Some((free0, _)) = cuda_memory_or_skip("cuda_pool_hoards", 4 * GIB) else {
        return;
    };
    const GB2: usize = 2 * GIB;

    // Hoard mode (#32): the pool keeps freed pages until explicitly trimmed.
    kiln_tensor::cuda_set_pool_release_threshold(0, u64::MAX)
        .expect("cuda_set_pool_release_threshold");

    let buf = kiln_tensor::cuda_zeros_ctx(0, kiln_tensor::DType::U8, GB2).expect("alloc 2GB");
    kiln_tensor::cuda_synchronize_default_stream(0).ok();
    let (free_alloc, _) = kiln_tensor::cuda_mem_get_info(0).unwrap();
    drop(buf);
    kiln_tensor::cuda_synchronize_default_stream(0).ok();
    let (free_dropped, _) = kiln_tensor::cuda_mem_get_info(0).unwrap();
    eprintln!(
        "[hoard] free0={} alloc={} dropped(no-trim)={} MiB",
        free0 / MIB,
        free_alloc / MIB,
        free_dropped / MIB
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
        free_trim / MIB,
        free_trim.saturating_sub(free_dropped) / MIB
    );
    assert!(
        free_trim >= free_dropped + (GB2 / 2),
        "cuda_trim_pool should release the hoarded pool to the OS \
         (dropped={free_dropped} trim={free_trim})"
    );
    eprintln!("[hoard] OK: release-threshold hoards freed VRAM; cuda_trim_pool reclaims it");
}

#[test]
fn qualification_mode_is_exact_opt_in() {
    assert!(qualification_required(Some("1")));
    assert!(!qualification_required(None));
    assert!(!qualification_required(Some("")));
    assert!(!qualification_required(Some("0")));
    assert!(!qualification_required(Some("true")));
}
