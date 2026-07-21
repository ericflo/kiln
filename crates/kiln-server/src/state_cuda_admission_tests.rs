#[test]
fn cuda_live_allocator_rejects_oversized_kv_before_allocation() {
    if std::env::var("KILN_QUALIFICATION").as_deref() != Ok("1") {
        return;
    }

    const MIB: u64 = 1024 * 1024;
    const SAFETY_FLOOR: u64 = 1024 * MIB;
    let device = kiln_tensor::Device::Cuda(0);
    let policy = GpuAllocatorMemoryProbePolicy::CUDA_MEM_GET_INFO;
    let allocator = crate::device_memory::allocator_memory_snapshot(policy, &device)
        .unwrap_or_else(|| {
            panic!("CUDA qualification requires cuMemGetInfo on logical device zero")
        });
    let live_budget = crate::device_memory::allocator_safe_available_bytes_with_soft_reserved(
        policy,
        &device,
        SAFETY_FLOOR,
        0,
    )
    .unwrap_or_else(|| {
        panic!("CUDA qualification requires a live allocator budget on logical device zero")
    });
    let max_blocks = live_budget / MIB;
    assert!(
        max_blocks > 0,
        "CUDA qualification requires more than one GiB free before the non-allocating admission probe (free={} MiB)",
        allocator.free_bytes / MIB
    );
    let requested_blocks = usize::try_from(max_blocks.saturating_add(1))
        .expect("CUDA live allocator block ceiling must fit usize");

    let error = validate_kv_allocation_against_live_budget(
        requested_blocks,
        MIB,
        live_budget,
        true,
        false,
        KvCacheAutoBlockPolicy::MEMORY_BUDGET_ONLY,
    )
    .expect_err("one block beyond the CUDA live allocator ceiling must be rejected")
    .to_string();
    assert!(error.contains("exceeds live accelerator/host residency memory"));
    assert!(error.contains("stricter governor/allocator budget"));
    assert!(error.contains(&format!("at most num_blocks={max_blocks}")));
    assert!(error.contains("Lower memory.num_blocks"));

    println!(
        "[cuda-admission] OK: rejected num_blocks={requested_blocks} above live ceiling={max_blocks} before allocation (free={} MiB, safety_floor={} MiB)",
        allocator.free_bytes / MIB,
        SAFETY_FLOOR / MIB,
    );
}
