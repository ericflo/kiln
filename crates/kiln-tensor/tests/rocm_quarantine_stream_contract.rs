//! CPU-only source contract for fail-closed ROCm raw-stream acquisition.

use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates directory")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn read(path: &str) -> String {
    std::fs::read_to_string(workspace_root().join(path))
        .unwrap_or_else(|error| panic!("read {path}: {error}"))
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

fn visit_rs_files(dir: &Path, visit: &mut impl FnMut(&Path, &str)) {
    for entry in std::fs::read_dir(dir).expect("read source directory") {
        let entry = entry.expect("read source entry");
        let path = entry.path();
        if path.is_dir() {
            visit_rs_files(&path, visit);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
            visit(&path, &source);
        }
    }
}

#[test]
fn storage_and_bridge_acquire_rocm_streams_fallibly() {
    let hip = read("crates/kiln-hip/src/lib.rs");
    let storage = read("crates/kiln-tensor/src/rocm_storage.rs");
    let bridge = read("crates/kiln-kt-bridge/src/lib.rs");

    assert!(hip.contains("pub fn hip_stream_for_execution(&self) -> Result<sys::hipStream_t>"));
    assert!(hip.contains("self.bind()?;\n        Ok(self.handle)"));
    assert!(storage.contains("pub fn rocm_stream_raw(&self) -> Result<*mut core::ffi::c_void>"));
    assert!(storage.contains("ensure_execution_available(\"RocmStorage::rocm_stream_raw\")"));
    assert!(storage.contains(".hip_stream_for_execution()"));
    assert!(bridge.contains("st.rocm_stream_raw().map_err(|error|"));
    assert!(!bridge.contains("Ok(st.rocm_stream_raw())"));
}

#[test]
fn tensor_rocm_ffi_launches_propagate_stream_acquisition_failures() {
    let src = workspace_root().join("crates/kiln-tensor/src");
    let mut acquisitions = 0usize;
    visit_rs_files(&src, &mut |path, source| {
        for (offset, _) in source.match_indices(".rocm_stream_raw()") {
            acquisitions += 1;
            let suffix = &source[offset + ".rocm_stream_raw()".len()..];
            assert!(
                suffix.starts_with('?') || suffix.starts_with(".map_err"),
                "{} has an unpropagated ROCm raw-stream acquisition",
                path.display()
            );
        }
    });
    assert!(
        acquisitions >= 50,
        "expected the shared guard to cover the full kt ROCm kernel surface"
    );
}

#[test]
fn external_rocm_launchers_do_not_use_the_infallible_raw_accessor() {
    let rocblas = read("crates/kiln-rocblas/src/hipblaslt_handle.rs");
    let rmsnorm = read("crates/kiln-rmsnorm-kernel/src/kt_api.rs");

    assert!(rocblas.contains(".hip_stream_for_execution()"));
    assert!(rocblas.contains("FfiError::StreamUnavailable"));
    assert!(!rocblas.contains("stream.hip_stream(),"));

    assert!(rmsnorm.contains(".hip_stream_for_execution()"));
    assert!(!rmsnorm.contains("default_stream().hip_stream()"));
}

#[test]
fn hipblaslt_submitters_hold_workspace_ownership_across_ffi() {
    let rocblas = read("crates/kiln-rocblas/src/hipblaslt_handle.rs");

    assert!(rocblas.contains("buffer: Option<Arc<RocmSlice>>"));
    assert!(rocblas.contains(") -> Result<(Arc<RocmSlice>, u64), FfiError>"));
    assert!(rocblas.contains("let buffer = Arc::clone(entry.buffer.as_ref()"));
    assert!(rocblas.contains("let (workspace, workspace_bytes) = self.ensure_workspace("));
    assert!(rocblas.contains("let workspace_ptr_raw = workspace.device_ptr() as *mut c_void;"));
    let submit = source_between(
        &rocblas,
        "let code = unsafe {\n            kiln_blas_hipblaslt_matmul(",
        "if let Some(e) = FfiError::from_code(code)",
    );
    assert!(submit.contains("drop(workspace);"));
}

#[test]
fn quarantine_is_device_wide_sticky_and_teardown_is_fail_closed() {
    let hip = read("crates/kiln-hip/src/lib.rs");
    let rocblas = read("crates/kiln-rocblas/src/hipblaslt_handle.rs");
    let model = read("crates/kiln-model/src/generate.rs");

    for required in [
        "fn device_cleanup_quarantine(",
        "static QUARANTINES: OnceLock<Mutex<HashMap<c_int, Arc<AtomicBool>>>>",
        "pub fn quarantine_execution(&self)",
        "fn bind_device_for_cleanup(",
        "cleanup_quarantined.store(true, Ordering::Release)",
        "restart the process",
    ] {
        assert!(
            hip.contains(required),
            "missing sticky quarantine contract: {required}"
        );
    }
    assert!(
        !hip.contains("cleanup_quarantined.store(false"),
        "recovery must never clear the process-lifetime device quarantine"
    );
    for resource in [
        "RocmStream::drop",
        "RocmEvent::drop",
        "RocmSlice::drop",
        "RocmGraph::drop",
        "RocmGraphExec::drop",
    ] {
        assert!(
            hip.contains(resource),
            "missing fail-closed destructor for {resource}"
        );
    }
    assert!(rocblas.contains("pub fn new_ctx("));
    assert!(rocblas.contains("if rocm_ctx.ordinal() != device_index"));
    assert!(rocblas.contains("fn workspace_map_for_cleanup("));
    assert!(rocblas.contains("self.rocm_ctx.quarantine_execution();"));
    assert!(rocblas.contains("retaining hipBLASLt context and workspaces until process exit"));

    assert!(model.contains("impl Drop for ModelRunner"));
    assert!(model.contains("self.backend_health.snapshot().quarantined"));
    assert!(model.contains("storage.context().quarantine_execution();"));
}

#[test]
fn graph_failures_cannot_retry_partially_advanced_state() {
    let graph = read("crates/kiln-model/src/rocm_graph.rs");

    assert!(graph.contains("fn fail_closed_after_rocm_warmup<T>("));
    assert_eq!(
        graph
            .matches("return fail_closed_after_rocm_warmup(weights, e);")
            .count(),
        3,
        "all graph-shaped warmup result forms must fail closed"
    );
    assert!(graph.contains("fn quarantine_rocm_tensor_context("));
    assert!(
        graph
            .matches("quarantine_rocm_tensor_context(&hidden);")
            .count()
            >= 2,
        "first-capture LM-head failures must quarantine before outer fallback logic"
    );
    assert!(graph.contains("fn complete_rollback(&mut self, rollback: Result<()>)"));
    assert!(graph.contains("Ok(()) if !context.cleanup_quarantined()"));
}
