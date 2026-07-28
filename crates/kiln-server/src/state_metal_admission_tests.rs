#[test]
#[ignore = "requires an explicit Metal qualification device"]
fn metal_unified_memory_admission_rejects_before_allocation() {
    assert_eq!(
        std::env::var("KILN_QUALIFICATION").as_deref(),
        Ok("1"),
        "Metal qualification test requires KILN_QUALIFICATION=1"
    );

    const MIB: u64 = 1024 * 1024;
    const CONTROLLED_AVAILABLE: u64 = 64 * MIB;
    let device = kiln_tensor::Device::Metal(0);
    let companion =
        kiln_tensor::primary_metal_companion(0).expect("Metal qualification device zero");
    let initial =
        kiln_memory::current_memory_snapshot_for(kiln_memory::VramProbeSelector::AppleUnified);
    assert!(
        initial.unified,
        "Metal qualification requires unified memory"
    );
    assert_eq!(initial.source, kiln_memory::vram::VramSource::AppleSilicon);
    assert!(
        initial.free_bytes > CONTROLLED_AVAILABLE,
        "qualification host needs more than 64 MiB of safe unified memory"
    );

    let governor = kiln_memory::MemoryGovernor::with_selector(
        kiln_memory::VramProbeSelector::AppleUnified,
        kiln_memory::GovernorConfig {
            floor_bytes: initial.free_bytes.saturating_sub(CONTROLLED_AVAILABLE),
            ..kiln_memory::GovernorConfig::default()
        },
    );
    let requested_blocks = usize::try_from(initial.total_bytes / MIB + 1)
        .expect("Apple unified-memory block count must fit usize");
    let allocated_before = companion.device().current_allocated_size();
    let error = validate_kv_allocation_against_live_allocator(
        &device,
        requested_blocks,
        MIB,
        GpuMemoryBudgetPolicy::DEVICE_MEMORY_AWARE,
        GpuAllocatorMemoryProbePolicy::for_backend("metal", device),
        KvCacheAutoBlockPolicy::for_backend("metal", device),
        &governor,
        0,
    )
    .expect_err("request above the controlled Apple unified-memory budget must fail")
    .to_string();
    let allocated_after = companion.device().current_allocated_size();
    assert_eq!(
        allocated_after, allocated_before,
        "admission rejection must occur before a Metal allocation"
    );
    assert!(error.contains("exceeds live accelerator/host residency memory"));
    assert!(error.contains("the memory governor budget"));
    assert!(error.contains("Lower memory.num_blocks"));
    let observation = governor.cached_observation();
    assert!(observation.snapshot.unified);

    println!(
        "[metal-admission] OK: controlled UMA budget rejected num_blocks={requested_blocks} \
         before allocation (live_available={} MiB, configured_floor={} MiB, \
         current_allocated={} MiB)",
        observation.available_bytes / MIB,
        governor.config().floor_bytes / MIB,
        allocated_after as u64 / MIB,
    );
}

#[test]
#[ignore = "requires an explicit Metal qualification device"]
fn metal_allocator_failure_retries_on_real_device_and_cleans_up() {
    assert_eq!(
        std::env::var("KILN_QUALIFICATION").as_deref(),
        Ok("1"),
        "Metal qualification test requires KILN_QUALIFICATION=1"
    );
    let companion =
        kiln_tensor::primary_metal_companion(0).expect("Metal qualification device zero");
    let baseline = companion.device().current_allocated_size();

    let allocation_attempt = std::cell::Cell::new(0usize);
    let compute_blocks = |fraction: f64| if fraction > 0.80 { 16 } else { 8 };
    let success = auto_size_with_retry(
        0.85,
        &[0.75],
        &compute_blocks,
        |num_blocks| -> Result<PagedKvCacheKt, String> {
            let attempt = allocation_attempt.get() + 1;
            allocation_attempt.set(attempt);
            if attempt == 1 {
                return Err("injected Metal allocation failure".to_string());
            }
            PagedKvCacheKt::new_with_fp8(
                1,
                num_blocks,
                16,
                1,
                4,
                DType::F32,
                kiln_tensor::Device::Metal(0),
                false,
            )
            .map_err(|error| format!("real Metal fallback allocation failed: {error}"))
        },
    )
    .unwrap_or_else(|failure| {
        panic!(
            "Metal auto-sizer did not recover from injected allocation failure: {:?}",
            failure.attempts
        )
    });

    assert_eq!(allocation_attempt.get(), 2);
    assert_eq!(success.fraction, 0.75);
    assert_eq!(success.num_blocks, 8);
    assert_eq!(success.attempted_failures.len(), 1);
    assert_eq!(
        success.attempted_failures[0].2,
        "injected Metal allocation failure"
    );
    let (key_pool, _) = success
        .cache
        .pool_tensors(0)
        .expect("real Metal fallback cache layer");
    assert_eq!(key_pool.device(), kiln_tensor::Device::Metal(0));
    drop(success);

    let probe = kiln_tensor::MetalStorage::zeros_kt(
        companion.device(),
        0,
        kiln_tensor::DType::U8,
        1024 * 1024,
    )
    .expect("post-recovery Metal allocation");
    drop(probe);
    let final_allocated = companion.device().current_allocated_size();
    assert!(
        final_allocated <= baseline.saturating_add(4 * 1024 * 1024),
        "Metal retry cleanup retained too much storage: baseline={baseline}, final={final_allocated}"
    );

    println!(
        "[metal-allocation-failure] OK: injected first allocator failure; fallback \
         allocated real Metal cache and cleanup returned near baseline"
    );
}
