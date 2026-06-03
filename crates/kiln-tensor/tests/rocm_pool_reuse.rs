//! Decisive, PROCESS-ISOLATED measurement for the dynamic training/inference
//! goal: kiln does BOTH in ONE process / ONE HIP pool, so "free KV → training
//! reuses it" requires the pool to reuse freed blocks rather than reserve new
//! VRAM from the OS. Measured via `rocm_pool_stats` (the pool's own
//! reserved/used counters) — immune to the coexisting llama-server, unlike the
//! all-process DRM `current_memory_snapshot`.
#![cfg(feature = "rocm")]

use kiln_tensor::{rocm_pool_stats as stats, rocm_synchronize_default_stream as sync,
                  rocm_zeros_ctx as alloc, DType};

fn mb(bytes: u64) -> f64 { bytes as f64 / (1024.0 * 1024.0) }
fn mb_elems(m: usize) -> usize { (m * 1024 * 1024) / 2 } // bf16

#[test]
fn pool_reuses_freed_blocks_without_growing_reserved() {
    if !kiln_tensor::rocm_is_available() { eprintln!("skip"); return; }

    // Allocate 8x256MB (mimics KV pools), then free them.
    let (r0, u0) = stats(0).unwrap();
    let mut a = Vec::new();
    for _ in 0..8 { a.push(alloc(0, DType::BF16, mb_elems(256)).unwrap()); }
    sync(0).ok();
    let (r1, u1) = stats(0).unwrap();
    drop(a);
    sync(0).ok();
    let (r2, u2) = stats(0).unwrap();
    eprintln!("[pool] reserved: start={:.0} afterAlloc={:.0} afterFree={:.0} MB", mb(r0), mb(r1), mb(r2));
    eprintln!("[pool] used:     start={:.0} afterAlloc={:.0} afterFree={:.0} MB", mb(u0), mb(u1), mb(u2));

    // Alloc rose used by ~2GB; free returned used to ~baseline (pool keeps
    // reserved, by the release-threshold pin).
    assert!(u1 - u0 > mb_elems(8 * 256) as u64 / 2 || mb(u1) - mb(u0) > 1500.0,
        "used must rise ~2GB on alloc (start {:.0} -> {:.0})", mb(u0), mb(u1));
    assert!(mb(u2) - mb(u0) < 300.0, "used must drop back ~baseline after free (got {:.0})", mb(u2) - mb(u0));

    // Now allocate 2GB again in DIFFERENT-SIZED chunks (10x205MB ≈ 2GB), like a
    // training workload. If the pool reuses the freed blocks, RESERVED does not
    // grow beyond the earlier high-water (r1); a leak would push reserved up.
    let mut b = Vec::new();
    for _ in 0..10 { b.push(alloc(0, DType::BF16, mb_elems(205)).unwrap()); }
    sync(0).ok();
    let (r3, u3) = stats(0).unwrap();
    eprintln!("[pool] after 10x205MB realloc: reserved={:.0} used={:.0} MB (reserved growth vs hwm {:.0})",
        mb(r3), mb(u3), mb(r3) - mb(r1.max(r2)));
    let reserved_growth = mb(r3) - mb(r1.max(r2));
    assert!(
        reserved_growth < 400.0,
        "pool must REUSE freed blocks for a differently-sized workload: reserved grew {reserved_growth:.0} MB \
         (0 = perfect reuse; ~2048 = pool reserves fresh, no reuse)"
    );
    drop(b);
    sync(0).ok();
}
