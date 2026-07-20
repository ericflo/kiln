use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use tower::ServiceExt;

use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::engine::MockEngine;
use kiln_scheduler::{Scheduler, SchedulerConfig};
use kiln_server::api;
use kiln_server::state::AppState;
use kiln_train::TeacherIdentityV1;

const HASH_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const HASH_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const MOCK_IO_TIMEOUT: Duration = Duration::from_secs(5);

static REMOTE_REGISTRATION_TEST_LOCK: Mutex<()> = Mutex::new(());

fn remote_registration_test_guard() -> MutexGuard<'static, ()> {
    REMOTE_REGISTRATION_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn tokenizer() -> KilnTokenizer {
    let mut vocab = HashMap::new();
    vocab.insert("a", 0u32);
    vocab.insert("b", 1u32);
    vocab.insert("c", 2u32);
    KilnTokenizer::from_bytes(
        &serde_json::to_vec(&json!({
            "version": "1.0",
            "model": {"type": "BPE", "vocab": vocab, "merges": []}
        }))
        .unwrap(),
    )
    .unwrap()
}

fn raw_sha256(value: &str) -> String {
    value.strip_prefix("sha256:").unwrap().to_owned()
}

fn identity(
    tokenizer: &KilnTokenizer,
    tokenizer_vocab_sha256: Option<String>,
) -> TeacherIdentityV1 {
    TeacherIdentityV1::new(
        "teacher-model",
        HASH_A,
        tokenizer_vocab_sha256.unwrap_or_else(|| raw_sha256(&tokenizer.vocab_identity_sha256())),
        raw_sha256(&tokenizer.tokenizer_config_sha256().unwrap()),
        None,
        3,
        2,
        64,
        3,
        "vllm:0.25.0",
        HASH_B,
    )
    .unwrap()
}

fn state(tokenizer: KilnTokenizer) -> (AppState, tempfile::TempDir) {
    let mut config = ModelConfig::qwen3_5_4b();
    config.vocab_size = 3;
    config.max_position_embeddings = 64;
    let scheduler = Scheduler::new(
        SchedulerConfig {
            max_batch_tokens: 128,
            max_batch_size: 4,
            block_size: 16,
            prefix_cache_enabled: false,
            ..Default::default()
        },
        32,
    );
    let engine = MockEngine::new(config.clone());
    let mut state = AppState::new_mock(
        config,
        scheduler,
        Arc::new(engine),
        tokenizer,
        30,
        "student-model".to_owned(),
    );
    let dir = tempfile::tempdir().unwrap();
    state.adapter_dir = dir.path().to_path_buf();
    (state, dir)
}

fn read_request(stream: &TcpStream) -> (Vec<String>, Value) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut request_line = String::new();
    reader.read_line(&mut request_line).unwrap();
    assert!(request_line.starts_with("POST /v1/completions "));
    let mut content_length = None;
    let mut headers = Vec::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        if line == "\r\n" {
            break;
        }
        if let Some(value) = line
            .strip_prefix("content-length:")
            .or_else(|| line.strip_prefix("Content-Length:"))
        {
            content_length = Some(value.trim().parse::<usize>().unwrap());
        }
        headers.push(line.trim_end().to_owned());
    }
    let mut body = vec![0u8; content_length.expect("request has Content-Length")];
    reader.read_exact(&mut body).unwrap();
    (headers, serde_json::from_slice(&body).unwrap())
}

fn completion_response(fingerprint: &str, top_k: usize) -> Value {
    let mut row = serde_json::Map::new();
    row.insert(
        "0".to_owned(),
        json!({"logprob": -1.0, "rank": 1, "decoded_token": "a"}),
    );
    if top_k == 2 {
        row.insert(
            "1".to_owned(),
            json!({"logprob": -2.0, "rank": 2, "decoded_token": "b"}),
        );
    }
    json!({
        "object": "text_completion",
        "model": "teacher-model",
        "system_fingerprint": fingerprint,
        "choices": [{
            "index": 0,
            "prompt_logprobs": [null, Value::Object(row)]
        }],
        "usage": {"prompt_tokens": 2, "completion_tokens": 0, "total_tokens": 2}
    })
}

fn spawn_teacher(fingerprint: String, requests: usize) -> (String, std::thread::JoinHandle<()>) {
    spawn_teacher_with_auth(fingerprint, requests, None)
}

fn spawn_teacher_with_auth(
    fingerprint: String,
    requests: usize,
    expected_authorization: Option<String>,
) -> (String, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = listener.local_addr().unwrap();
    let handle = std::thread::spawn(move || {
        for _ in 0..requests {
            let deadline = Instant::now() + MOCK_IO_TIMEOUT;
            let (mut stream, _) = loop {
                match listener.accept() {
                    Ok(connection) => break connection,
                    Err(error)
                        if error.kind() == std::io::ErrorKind::WouldBlock
                            && Instant::now() < deadline =>
                    {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("mock teacher accept failed: {error}"),
                }
            };
            stream.set_read_timeout(Some(MOCK_IO_TIMEOUT)).unwrap();
            stream.set_write_timeout(Some(MOCK_IO_TIMEOUT)).unwrap();
            let (headers, request) = read_request(&stream);
            let authorization = headers.iter().find_map(|header| {
                let (name, value) = header.split_once(':')?;
                name.eq_ignore_ascii_case("authorization")
                    .then(|| value.trim().to_owned())
            });
            assert_eq!(authorization, expected_authorization);
            assert_eq!(request["model"], "teacher-model");
            assert_eq!(request["prompt"], json!([0, 0]));
            let top_k = request["prompt_logprobs"].as_u64().unwrap() as usize;
            let body = serde_json::to_vec(&completion_response(&fingerprint, top_k)).unwrap();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .unwrap();
            stream.write_all(&body).unwrap();
        }
    });
    (format!("http://{address}"), handle)
}

struct ScopedTestEnvironment {
    name: &'static str,
    previous: Option<std::ffi::OsString>,
}

unsafe fn write_test_environment(name: &str, value: Option<&std::ffi::OsStr>) {
    if let Some(value) = value {
        unsafe { std::env::set_var(name, value) };
    } else {
        unsafe { std::env::remove_var(name) };
    }
}

impl ScopedTestEnvironment {
    fn set(name: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(name);
        unsafe {
            write_test_environment(name, Some(std::ffi::OsStr::new(value)));
        }
        Self { name, previous }
    }
}

impl Drop for ScopedTestEnvironment {
    fn drop(&mut self) {
        unsafe {
            write_test_environment(self.name, self.previous.as_deref());
        }
    }
}

async fn post_teacher(app: &axum::Router, url: &str) -> (StatusCode, Value) {
    post_teacher_body(
        app,
        json!({
            "alias": "remote@test",
            "kind": "remote",
            "provider": "vllm",
            "model_id": "teacher-model",
            "url": url
        }),
    )
    .await
}

async fn post_teacher_body(app: &axum::Router, body: Value) -> (StatusCode, Value) {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/teachers")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, serde_json::from_slice(&body).unwrap())
}

#[tokio::test]
async fn registration_rejects_caller_controlled_secret_environment_names() {
    let (state, _dir) = state(tokenizer());
    let app = api::router(state.clone());
    let attacker_chosen_env = "AWS_SECRET_ACCESS_KEY_DO_NOT_DISCLOSE";
    let (status, body) = post_teacher_body(
        &app,
        json!({
            "alias": "remote@forged-secret",
            "kind": "remote",
            "provider": "vllm",
            "model_id": "teacher-model",
            "url": "http://127.0.0.1:9",
            "api_key_env": attacker_chosen_env
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let rendered = body.to_string();
    assert!(rendered.contains("api_key_env"), "{body}");
    assert!(!rendered.contains(attacker_chosen_env), "{body}");
    assert!(state.teacher_registry.list().is_empty());
}

#[tokio::test]
async fn registration_rejects_credential_handle_for_a_different_origin() {
    let (mut state, _dir) = state(tokenizer());
    let trusted_env = "KILN_TEST_ORIGIN_SCOPED_SECRET_MUST_NOT_LEAK";
    let mut credentials = kiln_server::config::TeachersConfig::default();
    credentials.credentials.insert(
        "trusted-vllm".into(),
        kiln_server::config::TeacherCredentialConfig {
            origin: "http://127.0.0.1:8000".into(),
            api_key_env: trusted_env.into(),
        },
    );
    state.teacher_credentials = Arc::new(credentials);
    let app = api::router(state.clone());
    let (status, body) = post_teacher_body(
        &app,
        json!({
            "alias": "remote@wrong-origin",
            "kind": "remote",
            "provider": "vllm",
            "model_id": "teacher-model",
            "url": "http://127.0.0.1:8001",
            "credential_id": "trusted-vllm"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let rendered = body.to_string();
    assert!(rendered.contains("not authorized"), "{body}");
    assert!(!rendered.contains(trusted_env), "{body}");
    assert!(state.teacher_registry.list().is_empty());
}

#[tokio::test]
async fn registration_fails_closed_when_configured_secret_is_missing() {
    let (mut state, _dir) = state(tokenizer());
    let trusted_env = "KILN_TEST_REMOTE_SECRET_INTENTIONALLY_MISSING_8CC28B";
    let mut credentials = kiln_server::config::TeachersConfig::default();
    credentials.credentials.insert(
        "missing-secret".into(),
        kiln_server::config::TeacherCredentialConfig {
            origin: "http://127.0.0.1:8000".into(),
            api_key_env: trusted_env.into(),
        },
    );
    state.teacher_credentials = Arc::new(credentials);
    let app = api::router(state.clone());
    let (status, body) = post_teacher_body(
        &app,
        json!({
            "alias": "remote@missing-secret",
            "kind": "remote",
            "provider": "vllm",
            "model_id": "teacher-model",
            "url": "http://127.0.0.1:8000",
            "credential_id": "missing-secret"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let rendered = body.to_string();
    assert!(rendered.contains("missing or empty"), "{body}");
    assert!(!rendered.contains(trusted_env), "{body}");
    assert!(state.teacher_registry.list().is_empty());
}

#[tokio::test]
async fn registration_uses_scoped_credential_without_persisting_secret_metadata() {
    let _registration_guard = remote_registration_test_guard();
    const ENV: &str = "KILN_TEST_REMOTE_TEACHER_HANDLE_SECRET_716D3C";
    const SECRET: &str = "credential-value-that-must-not-be-persisted";
    let _environment = ScopedTestEnvironment::set(ENV, SECRET);

    let tokenizer = tokenizer();
    let identity = identity(&tokenizer, None);
    let (url, server) =
        spawn_teacher_with_auth(identity.fingerprint(), 2, Some(format!("Bearer {SECRET}")));
    let (mut state, _dir) = state(tokenizer);
    let mut credentials = kiln_server::config::TeachersConfig::default();
    credentials.credentials.insert(
        "authenticated-loopback".into(),
        kiln_server::config::TeacherCredentialConfig {
            origin: kiln_server::config::canonical_teacher_origin(&url).unwrap(),
            api_key_env: ENV.into(),
        },
    );
    state.teacher_credentials = Arc::new(credentials);
    let app = api::router(state.clone());

    let (status, body) = post_teacher_body(
        &app,
        json!({
            "alias": "remote@authenticated",
            "kind": "remote",
            "provider": "vllm",
            "model_id": "teacher-model",
            "url": url,
            "credential_id": "authenticated-loopback"
        }),
    )
    .await;
    server.join().unwrap();
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["spec"]["credential_id"], "authenticated-loopback",
        "{body}"
    );
    let response = body.to_string();
    assert!(!response.contains(ENV), "{body}");
    assert!(!response.contains(SECRET), "{body}");

    let persisted = std::fs::read_to_string(state.adapter_dir.join("teachers.json")).unwrap();
    assert!(persisted.contains("authenticated-loopback"));
    assert!(!persisted.contains(ENV), "{persisted}");
    assert!(!persisted.contains(SECRET), "{persisted}");
}

#[tokio::test]
async fn registration_probes_normalizes_and_persists_authoritative_identity() {
    let _registration_guard = remote_registration_test_guard();
    let tokenizer = tokenizer();
    let identity = identity(&tokenizer, None);
    let (url, server) = spawn_teacher(identity.fingerprint(), 2);
    let (state, _dir) = state(tokenizer);
    let app = api::router(state.clone());

    let (status, body) = post_teacher(&app, &url).await;
    server.join().unwrap();
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["spec"]["identity"]["base_model_sha256"], HASH_A);
    assert_eq!(body["spec"]["max_top_k"], 2);
    assert_eq!(body["spec"]["vocab_size"], 3);
    assert_eq!(
        body["spec"]["tokenizer_hash"],
        identity.tokenizer_vocab_sha256()
    );
    assert_eq!(body["capabilities"]["max_top_k"], 2);
    assert_eq!(body["status"], "verified");
    assert_eq!(body["usable"], true);
    assert_eq!(
        body["identity_revision"],
        format!("sha256:{}", identity.content_revision())
    );
    let manifest = body["off_policy_manifest"].as_str().unwrap();
    let manifest: kiln_train::OffPolicyDistillationManifestV1 =
        serde_json::from_str(manifest).unwrap();
    assert_eq!(manifest.teacher_identity(), &identity);
    assert_eq!(
        state.teacher_registry.get("remote@test").unwrap().identity,
        Some(identity)
    );
    assert!(state.adapter_dir.join("teachers.json").is_file());

    let (duplicate_status, duplicate_body) = post_teacher(&app, &url).await;
    assert_eq!(duplicate_status, StatusCode::CONFLICT, "{duplicate_body}");
    assert_eq!(duplicate_body["error"]["code"], "teacher_alias_exists");
    assert!(
        duplicate_body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("already registered")
    );
}

#[tokio::test]
async fn registration_rejects_stock_fingerprint_without_publishing() {
    let _registration_guard = remote_registration_test_guard();
    let tokenizer = tokenizer();
    let (url, server) = spawn_teacher("vllm-0.25.0-abcdef12".to_owned(), 1);
    let (state, _dir) = state(tokenizer);
    let app = api::router(state.clone());

    let (status, body) = post_teacher(&app, &url).await;
    server.join().unwrap();
    assert_eq!(status, StatusCode::BAD_GATEWAY, "{body}");
    assert_eq!(
        body["error"]["code"], "teacher_identity_probe_failed",
        "{body}"
    );
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("system_fingerprint")
    );
    assert!(state.teacher_registry.list().is_empty());
    assert!(!state.adapter_dir.join("teachers.json").exists());
}

#[tokio::test]
async fn registration_rejects_tokenizer_drift_after_operational_probe() {
    let _registration_guard = remote_registration_test_guard();
    let tokenizer = tokenizer();
    let drifted = identity(&tokenizer, Some(HASH_B.to_owned()));
    let (url, server) = spawn_teacher(drifted.fingerprint(), 2);
    let (state, _dir) = state(tokenizer);
    let app = api::router(state.clone());

    let (status, body) = post_teacher(&app, &url).await;
    server.join().unwrap();
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["code"], "teacher_identity_mismatch");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("different semantics")
    );
    assert!(state.teacher_registry.list().is_empty());
}
