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
