#![cfg(feature = "metal")]

use kiln_tensor::{DType, MetalStorage, primary_metal_companion};

const MIB: usize = 1024 * 1024;
const ALLOCATION_BYTES: usize = 256 * MIB;

#[test]
#[ignore = "requires an explicit Metal qualification device"]
fn metal_ownership_release_reclaims_unified_memory() {
    assert_eq!(
        std::env::var("KILN_QUALIFICATION").as_deref(),
        Ok("1"),
        "Metal qualification test requires KILN_QUALIFICATION=1"
    );
    let companion = primary_metal_companion(0).expect("Metal qualification device zero");
    let baseline = companion.device().current_allocated_size();
    let storage = MetalStorage::zeros_kt(companion.device(), 0, DType::U8, ALLOCATION_BYTES)
        .expect("allocate controlled Metal storage");
    std::hint::black_box(&storage);
    let peak = companion.device().current_allocated_size();
    assert!(
        peak >= baseline.saturating_add(ALLOCATION_BYTES),
        "currentAllocatedSize did not observe the controlled allocation: \
         baseline={baseline}, peak={peak}"
    );

    drop(storage);
    let after_drop = companion.device().current_allocated_size();
    let reclaimed = peak.saturating_sub(after_drop);
    assert!(
        reclaimed >= ALLOCATION_BYTES,
        "dropping Metal ownership did not reclaim the controlled allocation: \
         peak={peak}, after_drop={after_drop}, reclaimed={reclaimed}"
    );
    assert!(
        after_drop <= baseline.saturating_add(MIB),
        "Metal ownership release did not return near baseline: \
         baseline={baseline}, after_drop={after_drop}"
    );

    let probe = MetalStorage::zeros_kt(companion.device(), 0, DType::U8, MIB)
        .expect("post-reclaim Metal allocation");
    drop(probe);
    println!(
        "[metal-reclaim] OK: ownership release reclaimed {reclaimed} bytes from \
         currentAllocatedSize after a {ALLOCATION_BYTES}-byte UMA allocation"
    );
}
