//! Integration test: POST /v1/adapters/upload accepts a multipart tar.gz
//! archive and installs it as a new adapter directory under adapter_dir.

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde_json::json;
use tower::ServiceExt;

use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::engine::MockEngine;
use kiln_scheduler::{Scheduler, SchedulerConfig};
use kiln_server::api;
use kiln_server::state::AppState;

/// Minimal tokenizer for tests.
fn test_tokenizer() -> KilnTokenizer {
    let mut vocab: HashMap<String, u32> = HashMap::new();
    vocab.insert("a".to_string(), 0);
    vocab.insert("b".to_string(), 1);
    let json = json!({
        "version": "1.0",
        "model": { "type": "BPE", "vocab": vocab, "merges": [] }
    });
    KilnTokenizer::from_bytes(&serde_json::to_vec(&json).unwrap()).unwrap()
}

fn make_state(adapter_dir: std::path::PathBuf) -> AppState {
    let config = ModelConfig::qwen3_5_4b();
    let scheduler = Scheduler::new(
        SchedulerConfig {
            max_batch_tokens: 8192,
            max_batch_size: 64,
            block_size: 16,
            prefix_cache_enabled: false,
            ..Default::default()
        },
        256,
    );
    let engine = MockEngine::new(config.clone());
    let mut state = AppState::new_mock(
        config,
        scheduler,
        Arc::new(engine),
        test_tokenizer(),
        300,
        "qwen3.5-4b-kiln".to_string(),
    );
    state.adapter_dir = adapter_dir;
    state
}

fn write_adapter(adapter_dir: &std::path::Path, name: &str) -> std::path::PathBuf {
    let path = adapter_dir.join(name);
    std::fs::create_dir_all(&path).unwrap();
    std::fs::write(
        path.join("adapter_config.json"),
        br#"{"r": 8, "lora_alpha": 16}"#,
    )
    .unwrap();
    std::fs::write(
        path.join("adapter_model.safetensors"),
        b"\x00\x01\x02\x03fake-safetensors-bytes",
    )
    .unwrap();
    path
}

/// Build a multipart/form-data request body with a fixed boundary.
fn build_multipart_body(name: Option<&str>, archive: Option<&[u8]>) -> (String, Vec<u8>) {
    let boundary = "----test-boundary-XXX";
    let content_type = format!("multipart/form-data; boundary={boundary}");
    let mut body: Vec<u8> = Vec::new();

    if let Some(name) = name {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            b"Content-Disposition: form-data; name=\"name\"\r\nContent-Type: text/plain\r\n\r\n",
        );
        body.extend_from_slice(name.as_bytes());
        body.extend_from_slice(b"\r\n");
    }

    if let Some(archive) = archive {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            b"Content-Disposition: form-data; name=\"archive\"; filename=\"a.tar.gz\"\r\n\
              Content-Type: application/gzip\r\n\r\n",
        );
        body.extend_from_slice(archive);
        body.extend_from_slice(b"\r\n");
    }

    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (content_type, body)
}

/// Build a tar.gz archive containing entries at the given paths with the
/// given byte contents. Entries are written as regular files.
fn build_tar_gz(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let buf: Vec<u8> = Vec::new();
    let gz = GzEncoder::new(buf, Compression::default());
    let mut tar = tar::Builder::new(gz);
    for (path, data) in entries {
        let mut header = tar::Header::new_gnu();
        header.set_size(data.len() as u64);
        header.set_mode(0o644);
        header.set_entry_type(tar::EntryType::Regular);
        header.set_cksum();
        tar.append_data(&mut header, path, *data).unwrap();
    }
    let gz = tar.into_inner().unwrap();
    gz.finish().unwrap()
}

#[tokio::test]
async fn test_roundtrip_download_then_upload() {
    let tmp = tempfile::tempdir().unwrap();
    write_adapter(tmp.path(), "src-adapter");
    let state = make_state(tmp.path().to_path_buf());
    let app = api::router(state);

    // 1. Download the adapter as tar.gz.
    let download_req = Request::builder()
        .method("GET")
        .uri("/v1/adapters/src-adapter/download")
        .body(Body::empty())
        .unwrap();
    let download_resp = app.clone().oneshot(download_req).await.unwrap();
    assert_eq!(download_resp.status(), StatusCode::OK);
    let archive_bytes = axum::body::to_bytes(download_resp.into_body(), usize::MAX)
        .await
        .unwrap()
        .to_vec();

    // 2. Upload it back under a new name.
    let (content_type, body) = build_multipart_body(Some("dest-adapter"), Some(&archive_bytes));
    let upload_req = Request::builder()
        .method("POST")
        .uri("/v1/adapters/upload")
        .header("content-type", content_type)
        .body(Body::from(body))
        .unwrap();
    let upload_resp = app.clone().oneshot(upload_req).await.unwrap();
    let status = upload_resp.status();
    let body_bytes = axum::body::to_bytes(upload_resp.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "upload failed: {}",
        String::from_utf8_lossy(&body_bytes)
    );
    let upload_json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(upload_json["name"], "dest-adapter");
    assert_eq!(upload_json["files"], 2);

    // 3. Verify both files are present on disk with original bytes.
    let dest = tmp.path().join("dest-adapter");
    assert!(dest.exists() && dest.is_dir(), "dest-adapter dir missing");
    let cfg = std::fs::read(dest.join("adapter_config.json")).unwrap();
    assert_eq!(cfg.as_slice(), br#"{"r": 8, "lora_alpha": 16}"#);
    let weights = std::fs::read(dest.join("adapter_model.safetensors")).unwrap();
    assert_eq!(weights.as_slice(), b"\x00\x01\x02\x03fake-safetensors-bytes");
}

#[tokio::test]
async fn test_upload_rejects_invalid_name() {
    let tmp = tempfile::tempdir().unwrap();
    let state = make_state(tmp.path().to_path_buf());
    let app = api::router(state);

    let archive = build_tar_gz(&[("foo/a.txt", b"hello")]);

    // Various bad names that should all be rejected by validate_adapter_name.
    let bad_names = ["", "..", "/abs", "foo/bar", "foo\\bar", "../escape"];

    for bad in bad_names {
        let (content_type, body) = build_multipart_body(Some(bad), Some(&archive));
        let req = Request::builder()
            .method("POST")
            .uri("/v1/adapters/upload")
            .header("content-type", content_type)
            .body(Body::from(body))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "name {bad:?} should have been rejected"
        );
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let code = json["error"]["code"].as_str().unwrap_or("");
        assert!(
            code == "invalid_adapter_name" || code == "adapter_import_invalid",
            "name {bad:?} returned unexpected error code {code}"
        );
    }

    // No regular adapter directory should have been created.
    let real_entries: Vec<_> = std::fs::read_dir(tmp.path())
        .unwrap()
        .flatten()
        .filter(|e| {
            !e.file_name()
                .to_string_lossy()
                .starts_with(".upload-tmp-")
        })
        .collect();
    assert!(
        real_entries.is_empty(),
        "no adapter directory should exist after failed uploads"
    );
}

#[tokio::test]
async fn test_upload_rejects_existing_adapter() {
    let tmp = tempfile::tempdir().unwrap();
    write_adapter(tmp.path(), "already-here");
    let state = make_state(tmp.path().to_path_buf());
    let app = api::router(state);

    let archive = build_tar_gz(&[
        ("already-here/adapter_config.json", b"{}"),
        ("already-here/adapter_model.safetensors", b"x"),
    ]);
    let (content_type, body) = build_multipart_body(Some("already-here"), Some(&archive));
    let req = Request::builder()
        .method("POST")
        .uri("/v1/adapters/upload")
        .header("content-type", content_type)
        .body(Body::from(body))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CONFLICT);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["code"], "adapter_already_exists");

    // Original adapter contents must be untouched.
    let original = std::fs::read(tmp.path().join("already-here/adapter_config.json")).unwrap();
    assert_eq!(original.as_slice(), br#"{"r": 8, "lora_alpha": 16}"#);
}

#[tokio::test]
async fn test_upload_rejects_path_escape_in_archive() {
    let tmp = tempfile::tempdir().unwrap();
    let state = make_state(tmp.path().to_path_buf());
    let app = api::router(state);

    // Build a tar.gz with one normal entry and one whose path contains `..`
    // pointing outside the prefix. `tar::Builder::append_data` validates
    // paths via `prepare_header_path` and refuses `..`, so we hand-write the
    // malicious name into the GNU header's `name` field and use
    // `Builder::append` (which copies header bytes verbatim with no path
    // check). This mirrors what a hostile tar producer could ship and is the
    // payload the server's extraction code must defeat.
    let buf: Vec<u8> = Vec::new();
    let gz = GzEncoder::new(buf, Compression::default());
    let mut tar = tar::Builder::new(gz);

    let mut h1 = tar::Header::new_gnu();
    h1.set_size(5);
    h1.set_mode(0o644);
    h1.set_entry_type(tar::EntryType::Regular);
    h1.set_cksum();
    tar.append_data(&mut h1, "evil/ok.txt", &b"hello"[..])
        .unwrap();

    let mut h2 = tar::Header::new_gnu();
    h2.set_size(11);
    h2.set_mode(0o644);
    h2.set_entry_type(tar::EntryType::Regular);
    {
        let evil_name = b"evil/../../etc/passwd";
        let gnu = h2.as_gnu_mut().expect("GNU header");
        for b in gnu.name.iter_mut() {
            *b = 0;
        }
        gnu.name[..evil_name.len()].copy_from_slice(evil_name);
    }
    // set_cksum must run AFTER the name bytes are in place so the checksum
    // covers the final header.
    h2.set_cksum();
    tar.append(&h2, &b"PWNED-DATA!"[..]).unwrap();

    let gz = tar.into_inner().unwrap();
    let archive = gz.finish().unwrap();

    let (content_type, body) = build_multipart_body(Some("evil"), Some(&archive));
    let req = Request::builder()
        .method("POST")
        .uri("/v1/adapters/upload")
        .header("content-type", content_type)
        .body(Body::from(body))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();

    assert!(
        status.is_client_error(),
        "expected 4xx, got {status}: {}",
        String::from_utf8_lossy(&body_bytes)
    );
    let code = json["error"]["code"].as_str().unwrap_or("");
    assert!(
        code == "adapter_import_invalid" || code == "adapter_import_failed",
        "unexpected error code: {code}"
    );

    // No adapter directory should exist with the escaped data, and the
    // adapter dir should be otherwise empty (modulo cleaned-up temp dir).
    let leaked = tmp.path().parent().unwrap().join("etc/passwd");
    assert!(!leaked.exists(), "path traversal leaked outside adapter_dir");
    let real_entries: Vec<_> = std::fs::read_dir(tmp.path())
        .unwrap()
        .flatten()
        .filter(|e| {
            !e.file_name()
                .to_string_lossy()
                .starts_with(".upload-tmp-")
        })
        .collect();
    assert!(
        real_entries.is_empty(),
        "expected no adapter dir after failed unsafe upload, got {:?}",
        real_entries
            .iter()
            .map(|e| e.file_name())
            .collect::<Vec<_>>()
    );
}

