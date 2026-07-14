//! Small source-level guard for the public ROCm submission boundary.
//!
//! Gate behavior and status classification are tested directly in `kiln-hip`.
//! These checks only protect the cross-crate API shape and prevent the removed
//! copyable raw-stream escape hatches from returning.

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
fn public_rocm_stream_access_is_submission_scoped() {
    let hip = read("crates/kiln-hip/src/lib.rs");
    let storage = read("crates/kiln-tensor/src/rocm_storage.rs");
    let bridge = read("crates/kiln-kt-bridge/src/lib.rs");

    assert!(hip.contains("pub struct RocmStreamSubmission"));
    assert!(hip.contains("impl Drop for RocmStreamSubmission"));
    assert!(
        storage.contains("pub fn rocm_stream_submission(&self) -> Result<RocmStreamSubmission>")
    );
    assert!(bridge.contains("pub enum DeviceStreamSubmission"));
}

#[test]
fn removed_rocm_raw_stream_escape_hatches_stay_removed() {
    let crates = workspace_root().join("crates");
    for entry in std::fs::read_dir(crates).expect("read crates directory") {
        let src = entry.expect("read crate entry").path().join("src");
        if !src.is_dir() {
            continue;
        }
        visit_rs_files(&src, &mut |path, source| {
            for removed in [
                "hip_stream_for_execution",
                "fn hip_stream(&self)",
                ".rocm_stream_raw()",
                "device_stream_raw_of",
                "rocm_stream_raw_of",
            ] {
                assert!(
                    !source.contains(removed),
                    "{} retains removed raw-stream API {removed}",
                    path.display()
                );
            }
        });
    }
}

#[test]
fn synchronized_host_copies_hold_one_admission_through_settlement() {
    let hip = read("crates/kiln-hip/src/lib.rs");

    for (method, next_method) in [
        (
            "pub fn memcpy_htod(",
            "pub unsafe fn memcpy_htod_raw_async(",
        ),
        ("pub fn memcpy_dtoh(", "pub unsafe fn memcpy_dtoh_raw("),
        ("pub unsafe fn memcpy_dtoh_raw(", "pub fn memcpy_dtod("),
    ] {
        let start = hip
            .find(method)
            .unwrap_or_else(|| panic!("missing {method}"));
        let end = hip[start..]
            .find(next_method)
            .map(|offset| start + offset)
            .unwrap_or_else(|| panic!("missing boundary {next_method} after {method}"));
        let implementation = &hip[start..end];

        assert!(
            implementation.contains("RocmAdmittedHostTransfer::new"),
            "{method} must own host memory with its admission permit"
        );
        assert!(
            implementation.contains("synchronize_admitted_for(transfer.permit()"),
            "{method} must settle under the enqueue admission"
        );
        assert!(
            !implementation.contains("drop(submission)"),
            "{method} must not expose an enqueue-to-wait admission gap"
        );
        assert!(
            !implementation.contains("self.synchronize_for("),
            "{method} must not reacquire admission for settlement"
        );
    }

    assert!(hip.contains("impl<T> Drop for RocmAdmittedHostTransfer<T>"));
    assert!(hip.contains("std::mem::forget(host)"));
}

#[test]
fn raw_pageable_htod_uses_the_explicit_synchronous_source_contract() {
    let hip = read("crates/kiln-hip/src/lib.rs");
    let sys = read("crates/kiln-hip/src/sys.rs");
    let start = hip
        .find("pub unsafe fn memcpy_htod_raw_async(")
        .expect("raw H2D helper");
    let end = hip[start..]
        .find("pub unsafe fn memset_zero_async(")
        .map(|offset| start + offset)
        .expect("next raw stream helper");
    let implementation = &hip[start..end];

    assert!(implementation.contains("sys::hipMemcpyAsync("));
    assert!(implementation.contains("sys::HIP_MEMCPY_HOST_TO_DEVICE"));
    assert!(!implementation.contains("sys::hipMemcpyHtoDAsync("));
    assert!(sys.contains("pub type hipMemcpyKind = c_uint;"));
    assert!(sys.contains("pub const HIP_MEMCPY_HOST_TO_DEVICE: hipMemcpyKind = 1;"));
    assert!(sys.contains("pub fn hipMemcpyAsync("));
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

    let settle_start = graph
        .find("fn settle_before_rollback(&mut self)")
        .expect("capture failure settlement helper");
    let settle_end = graph[settle_start..]
        .find("fn complete_rollback(&mut self")
        .map(|offset| settle_start + offset)
        .expect("capture rollback completion helper");
    let settle = &graph[settle_start..settle_end];
    assert!(
        settle.contains("RocmSyncReason::CaptureRollback")
            && settle.contains("record_settlement(result.is_ok())")
            && !settle.contains("RocmSyncReason::ErrorRecovery"),
        "acknowledged capture rollback must drain with the admission gate open"
    );

    let drop_start = graph
        .find("impl Drop for RocmCaptureFailureGuard")
        .expect("capture failure guard Drop");
    let drop_end = graph[drop_start..]
        .find("struct RocmAllocationKey")
        .map(|offset| drop_start + offset)
        .expect("next graph type");
    let unclassified_drop = &graph[drop_start..drop_end];
    assert!(
        unclassified_drop
            .find("quarantine_execution()")
            .expect("STOP publication")
            < unclassified_drop
                .find("RocmSyncReason::ErrorRecovery")
                .expect("fatal recovery drain"),
        "unclassified capture exit must publish STOP before ErrorRecovery"
    );

    let recoverable_start = graph
        .find("fn synchronize_after_rocm_graph_capture_failure")
        .expect("recoverable capture drain helper");
    let recoverable_end = graph[recoverable_start..]
        .find("struct RocmGraphKey")
        .map(|offset| recoverable_start + offset)
        .expect("next graph helper");
    let recoverable = &graph[recoverable_start..recoverable_end];
    assert!(recoverable.contains("RocmSyncReason::CaptureRollback"));
    assert!(!recoverable.contains("RocmSyncReason::ErrorRecovery"));
}
