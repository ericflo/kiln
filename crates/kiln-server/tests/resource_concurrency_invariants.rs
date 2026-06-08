use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn read(path: &str) -> String {
    std::fs::read_to_string(workspace_root().join(path)).unwrap()
}

#[test]
fn rocm_disk_cache_persistence_has_no_server_call_sites() {
    let main = read("crates/kiln-server/src/main.rs");
    for forbidden in [
        "rocm_load_algo_cache_from_disk",
        "rocm_flush_algo_cache_to_disk",
        "hipblaslt autotune cache restored from disk",
        "hipblaslt autotune cache flushed to disk",
        "hipblaslt autotune cache flushed on shutdown",
    ] {
        assert!(
            !main.contains(forbidden),
            "ROCm hipBLASLt disk persistence must stay absent from server startup/shutdown: {forbidden}"
        );
    }
}

#[test]
fn rocm_tensor_has_no_public_disk_cache_api() {
    let lib = read("crates/kiln-tensor/src/lib.rs");
    let rocm_matmul = read("crates/kiln-tensor/src/rocm_matmul.rs");
    for forbidden in [
        "hipblaslt_cache_path",
        "rocm_load_algo_cache_from_disk",
        "rocm_flush_algo_cache_to_disk",
        "rocm_restore_into_shared_cache",
        "rocm_snapshot_algo_cache",
    ] {
        assert!(
            !lib.contains(forbidden),
            "ROCm disk/cache internals must not be public tensor API: {forbidden}"
        );
        assert!(
            !rocm_matmul.contains(&format!("pub fn {forbidden}")),
            "ROCm disk/cache helper must not be reintroduced as a public function: {forbidden}"
        );
    }
}

#[test]
fn blaslt_workspace_is_stream_owned_not_handle_global() {
    for path in [
        "crates/kiln-blas/src/cublaslt_handle.rs",
        "crates/kiln-rocblas/src/hipblaslt_handle.rs",
    ] {
        let src = read(path);
        assert!(
            src.contains("workspace_by_stream"),
            "{path}: BLASLt workspace must be keyed by the typed stream owner"
        );
        assert!(
            !src.contains("workspace_buf: Mutex<Option"),
            "{path}: a single handle-global workspace buffer is not concurrency-safe"
        );
        assert!(
            !src.contains(".default_stream()\n                .alloc"),
            "{path}: workspace allocation must use the caller's active stream"
        );
        assert!(
            src.contains("stream: &Arc<"),
            "{path}: matmul must accept a typed stream owner, not only a raw stream pointer"
        );
    }
}

#[test]
fn process_shared_persistence_uses_resource_layer() {
    let resource = read("crates/kiln-resource/src/lib.rs");
    for required in [
        "pub fn locked_update",
        "create_new(true)",
        "file.sync_all()",
        "std::fs::rename",
        "pub fn lock_path_for",
        "lock_owner_is_dead",
    ] {
        assert!(
            resource.contains(required),
            "kiln-resource must own the cross-process locked atomic write contract: {required}"
        );
    }

    for path in [
        "crates/kiln-blas/src/algo_cache.rs",
        "crates/kiln-rocblas/src/algo_cache.rs",
        "crates/kiln-server/src/api/teachers.rs",
        "crates/kiln-server/src/api/agent_traces.rs",
    ] {
        let src = read(path);
        assert!(
            src.contains("kiln_resource::"),
            "{path}: process-shared persistence must go through kiln-resource"
        );
        assert!(
            !src.contains("std::fs::write(path, bytes"),
            "{path}: direct overwrite writes are not concurrency-safe"
        );
    }
}

#[test]
fn rocm_graph_state_is_decode_row_owned() {
    let rocm_graph = read("crates/kiln-model/src/rocm_graph.rs");
    let generate = read("crates/kiln-model/src/generate.rs");

    for required in [
        "enum RocmGraphOwner",
        "DecodeRow(u64)",
        "struct RocmGraphCacheKey",
        "owner: RocmGraphOwner",
        "captured: HashMap<RocmGraphCacheKey, CapturedDecodeGraphRocm>",
        "decode_timelines: HashMap<RocmGraphOwner, RocmGraphOwnerTimeline>",
        "fn prepare_owner_decode",
        "self.captured.retain(|key, _| key.owner != owner)",
        "RocmGraphCacheKey::new(owner, requested_key.clone())",
        "RocmGraphCacheKey::new(owner, key)",
    ] {
        assert!(
            rocm_graph.contains(required),
            "ROCm HIP graph decode state must be keyed by decode-row owner: {required}"
        );
    }

    for forbidden in [
        "captured: HashMap<RocmGraphKey, CapturedDecodeGraphRocm>",
        "last_decode_seq_len: None",
        "last_decode_block0: None",
    ] {
        assert!(
            !rocm_graph.contains(forbidden),
            "ROCm HIP graph runner must not keep a runner-wide decode timeline: {forbidden}"
        );
    }

    assert!(
        generate.contains("Some(row_ids[0])"),
        "batched serving decode must pass the stable row id into the ROCm graph owner key"
    );
}

#[test]
fn cuda_graph_state_is_decode_row_owned() {
    let cuda_graph = read("crates/kiln-model/src/cuda_graph.rs");
    let generate = read("crates/kiln-model/src/generate.rs");

    for required in [
        "enum CudaGraphOwner",
        "DecodeRow(u64)",
        "struct CudaGraphCacheKey",
        "owner: CudaGraphOwner",
        "captured: HashMap<CudaGraphCacheKey, CapturedDecodeGraph>",
        "decode_timelines: HashMap<CudaGraphOwner, CudaGraphOwnerTimeline>",
        "fn prepare_owner_decode",
        "self.captured.retain(|key, _| key.owner != owner)",
        "CudaGraphCacheKey::new(owner, requested_key.clone())",
        "CudaGraphCacheKey::new(owner, key)",
    ] {
        assert!(
            cuda_graph.contains(required),
            "CUDA graph decode state must be keyed by decode-row owner: {required}"
        );
    }

    for forbidden in [
        "captured: HashMap<CudaGraphKey, CapturedDecodeGraph>",
        "last_decode_seq_len: None",
        "last_decode_block0: None",
    ] {
        assert!(
            !cuda_graph.contains(forbidden),
            "CUDA graph runner must not keep a runner-wide bs=1 decode timeline: {forbidden}"
        );
    }

    assert!(
        generate.contains("Some(row_ids[0])"),
        "batched serving decode must pass the stable row id into graph owner keys"
    );
}

#[test]
fn rocm_sampled_batches_have_native_hidden_decode_path() {
    let generate = read("crates/kiln-model/src/generate.rs");
    for required in [
        "ROCm sampled serving batches need a native decode path",
        "kiln_tensor::Device::Rocm(_)",
        "decode_hidden_paged_contiguous_batch_with_ids",
        "sample ROCm hidden batch",
        "row_count > 1",
    ] {
        assert!(
            generate.contains(required),
            "ROCm sampled continuous batches must not fall through to generic fallback: {required}"
        );
    }
}

#[test]
fn rocm_graph_replay_failure_is_a_circuit_breaker() {
    let rocm_graph = read("crates/kiln-model/src/rocm_graph.rs");
    assert!(
        rocm_graph.contains("disabling ROCm HIP graphs for this runner"),
        "ROCm graph replay failures must disable the runner instead of recapturing forever"
    );
    assert!(
        rocm_graph.matches("self.enabled = false;").count() >= 3,
        "ROCm graph replay/capture failure paths must set a runner-local circuit breaker"
    );
    assert!(
        rocm_graph.matches("self.captured.clear();").count() >= 4,
        "ROCm graph replay failures must clear captured graph state before eager fallback"
    );
}

#[test]
fn hip_runtime_errors_are_cleared_at_wrapper_boundary() {
    let hip = read("crates/kiln-hip/src/lib.rs");
    let sys = read("crates/kiln-hip/src/sys.rs");
    let rocm_graph = read("crates/kiln-model/src/rocm_graph.rs");

    assert!(
        sys.contains("pub fn hipGetLastError()"),
        "kiln-hip must bind hipGetLastError so failed runtime calls can clear sticky error state"
    );
    assert!(
        hip.contains("let _ = unsafe { sys::hipGetLastError() };"),
        "kiln-hip::check must clear sticky HIP runtime errors after surfacing direct API failures"
    );
    assert!(
        hip.contains("sticky per host thread"),
        "the sticky-error boundary is a concurrency/resource invariant and should stay documented"
    );

    for required in [
        "*linear_state = gdn_snapshot;",
        "*linear_state = capture_snapshot;",
        "freeze-pointers warm (Record) pass failed",
        "forward pass failed during graph capture",
        "end_capture failed",
        "execute captured decode graph (first run)",
    ] {
        assert!(
            rocm_graph.contains(required),
            "ROCm graph capture failures must restore decode state before eager fallback: {required}"
        );
    }
}
