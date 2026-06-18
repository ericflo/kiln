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

fn source_between<'a>(source: &'a str, start_marker: &str, end_marker: &str) -> &'a str {
    let start = source
        .find(start_marker)
        .unwrap_or_else(|| panic!("missing start marker: {start_marker}"));
    let rest = &source[start..];
    let end = rest
        .find(end_marker)
        .unwrap_or_else(|| panic!("missing end marker: {end_marker}"));
    &rest[..end]
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
fn paged_kv_row_scatter_materializes_rows_before_slice_set() {
    let src = read("crates/kiln-model/src/paged_kv_cache_kt.rs");
    assert!(
        src.contains("fn row_for_slice_set("),
        "paged-KV row-scatter must keep a single helper for slice_set-ready row materialization"
    );
    assert!(
        src.contains("contiguous row {row_idx}"),
        "paged-KV row-scatter helper must materialize zero-offset contiguous rows before slice_set"
    );

    for required in [
        "Self::row_for_slice_set(&k_flat, i, \"token_major k\")",
        "Self::row_for_slice_set(&v_flat, i, \"token_major v\")",
        "Self::row_for_slice_set(&k_flat, i, \"write_native k\")",
        "Self::row_for_slice_set(&v_flat, i, \"write_native v\")",
        "Self::row_for_slice_set(&k_q, i, \"write_fp8 k\")",
        "Self::row_for_slice_set(&v_q, i, \"write_fp8 v\")",
    ] {
        assert!(
            src.contains(required),
            "paged-KV row-scatter fallback must route through contiguous row materialization: {required}"
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
fn rocm_graph_warm_pass_prewarms_capture_stream() {
    let rocm_graph = read("crates/kiln-model/src/rocm_graph.rs");
    let capture = source_between(
        &rocm_graph,
        "fn try_capture_hidden(",
        "fn new_token_buffer(",
    );
    let warm_pass = source_between(
        capture,
        "let htod_before = kiln_tensor::rocm_htod_count();",
        "if let Err(err) = warm_result",
    );

    assert!(
        capture.contains("ROCm graph capture: sync kt default stream before warm pass"),
        "ROCm graph capture must make default-stream input fills visible before warming the capture stream"
    );
    assert!(
        warm_pass.contains("kiln_tensor::with_active_rocm_stream(stream.clone(), ||"),
        "ROCm graph warm pass must run on the same stream that will be captured, so hipBLASLt per-stream workspace is preallocated before begin_capture"
    );
    assert!(
        capture.contains("sync capture stream after ROCm graph warm pass"),
        "ROCm graph capture must wait for the warm pass before restoring state or falling back"
    );

    let warm_stream = capture
        .find("kiln_tensor::with_active_rocm_stream(stream.clone(), ||")
        .expect("warm pass should install active ROCm stream");
    let begin_capture = capture
        .find(".begin_capture()")
        .expect("capture path should call begin_capture");
    assert!(
        warm_stream < begin_capture,
        "ROCm capture stream must be warmed before hipStreamBeginCapture"
    );
}

#[test]
fn rocm_capture_arena_suppresses_active_stream_synchronizes() {
    let rocm_storage = read("crates/kiln-tensor/src/rocm_storage.rs");
    let compute_sync = source_between(
        &rocm_storage,
        "pub fn rocm_synchronize_compute_stream(",
        "/// Block until the active stream for a ROCm tensor",
    );
    let tensor_sync = source_between(
        &rocm_storage,
        "pub fn rocm_synchronize_tensor_stream(",
        "/// Refresh `dst`'s contents in place",
    );
    let contiguous = source_between(
        &rocm_storage,
        "pub fn rocm_contiguous(",
        "fn rocm_view_is_physically_compact(",
    );
    let concat = read("crates/kiln-tensor/src/rocm_ops/concat.rs");
    let bf16_matmul = read("crates/kiln-tensor/src/rocm_ops/bf16_matmul.rs");

    assert!(
        compute_sync.contains("if crate::rocm_capture_arena_active()"),
        "ROCm active compute-stream sync must no-op under HIP graph capture"
    );
    assert!(
        tensor_sync.contains("if crate::rocm_capture_arena_active()"),
        "ROCm tensor-stream sync must no-op under HIP graph capture"
    );
    assert!(
        contiguous
            .matches("if !crate::rocm_capture_arena_active()")
            .count()
            >= 3,
        "ROCm contiguous must not call hipStreamSynchronize inside capture"
    );
    assert!(
        concat
            .matches("if !crate::rocm_capture_arena_active()")
            .count()
            >= 3,
        "ROCm concat must not call hipStreamSynchronize inside capture"
    );
    assert!(
        bf16_matmul.contains("if !crate::rocm_capture_arena_active()"),
        "ROCm BF16 matmul fallback must not call hipStreamSynchronize inside capture"
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
