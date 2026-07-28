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

#[test]
#[ignore = "requires an explicit CUDA qualification device"]
fn cuda_allocator_failure_retries_on_real_device_and_cleans_up() {
    assert!(
        kiln_tensor::cuda_is_available(),
        "CUDA qualification requires logical device zero"
    );

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
                return Err("injected CUDA allocation failure".to_string());
            }
            PagedKvCacheKt::new_with_fp8(
                1,
                num_blocks,
                16,
                1,
                4,
                DType::F32,
                kiln_tensor::Device::Cuda(0),
                false,
            )
            .map_err(|error| format!("real CUDA fallback allocation failed: {error}"))
        },
    )
    .unwrap_or_else(|failure| {
        panic!(
            "CUDA auto-sizer did not recover from injected allocation failure: {:?}",
            failure.attempts
        )
    });

    assert_eq!(allocation_attempt.get(), 2);
    assert_eq!(success.fraction, 0.75);
    assert_eq!(success.num_blocks, 8);
    assert_eq!(success.attempted_failures.len(), 1);
    assert_eq!(success.attempted_failures[0].0, 0.85);
    assert_eq!(success.attempted_failures[0].1, 16);
    assert_eq!(
        success.attempted_failures[0].2,
        "injected CUDA allocation failure"
    );
    let (key_pool, _) = success
        .cache
        .pool_tensors(0)
        .expect("real CUDA fallback cache layer");
    assert!(
        matches!(key_pool.device(), kiln_tensor::Device::Cuda(0)),
        "fallback cache must be allocated on logical CUDA device zero"
    );
    drop(success);

    kiln_tensor::cuda_synchronize_default_stream(0)
        .expect("synchronize real CUDA fallback allocation");
    kiln_tensor::cuda_trim_pool(0, 0).expect("trim after real CUDA fallback allocation");
    let probe = kiln_tensor::cuda_zeros_ctx(0, kiln_tensor::DType::U8, 1024 * 1024)
        .expect("post-recovery CUDA allocation");
    kiln_tensor::cuda_synchronize_default_stream(0)
        .expect("synchronize post-recovery CUDA allocation");
    drop(probe);
    kiln_tensor::cuda_trim_pool(0, 0).expect("final CUDA trim after allocation recovery");

    println!(
        "[cuda-allocation-failure] OK: injected first allocator failure; fallback allocated real CUDA cache and cleanup succeeded"
    );
}
