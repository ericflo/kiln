//! Teacher registry endpoints (grand plan §3.2 + §4).
//!
//! - `GET /v1/teachers` — list configured `LogitSource` aliases with
//!   capabilities (vocab_size, max_top_k, supports_full_vocab,
//!   tokenizer_hash, etc.).
//! - `POST /v1/teachers` — register a new alias. Persists to the
//!   server's teacher store; subsequent `OpdRequest`s referencing the
//!   alias resolve to this source.
//! - `DELETE /v1/teachers/{alias}` — drop a registered teacher.
//!
//! The registry maps `alias -> TeacherSpec` (a serializable description
//! of how to build a `LogitSource`). At resolution time (inside
//! `run_opd` / the distill runtimes) the spec is materialised into a
//! concrete `LogitSource`:
//!
//! - **Fixture** — in-memory deterministic source for unit tests +
//!   `kiln-canonical` corpora that ship with pre-baked logits.
//! - **Local** — the model loaded in this process, optionally wearing a
//!   LoRA named by `adapter` (this is what makes "distil toward my
//!   prior self" and "judge as teacher" mean what they say). On-policy
//!   runs get a `LiveLocalTeacher`; off-policy runs pre-compute a
//!   fixture.
//! - **Remote** — HTTP numeric-ID `prompt_logprobs`. Only vLLM is wired
//!   today; registration rejects URLs that resolve to unsupported
//!   providers instead of letting the job fail at dequeue.

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use axum::Json;
use axum::Router;
use axum::extract::{Path as AxumPath, State};
use axum::routing::{delete, get};
use serde::{Deserialize, Serialize};

use kiln_train::{LogitSourceCaps, LogitSourceError};

use crate::error::ApiError;
use crate::state::AppState;

/// One entry in the teacher registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeacherSpec {
    /// Registry alias (e.g. `qwen3.6-27b@local`).
    pub alias: String,
    /// Concrete kind. Determines which builder runs at resolve time.
    pub kind: TeacherKind,
    /// Wire protocol for `Remote` teachers. Required explicitly because a URL
    /// cannot identify the server implementation reliably.
    #[serde(default)]
    pub provider: Option<kiln_train::RemoteProvider>,
    /// Provider's model id (e.g. `qwen/qwen-3.6-27b`).
    pub model_id: String,
    /// Maximum top-K this teacher can return. For vLLM, omitted/zero uses the
    /// upstream default of 20; a larger value asserts a matching
    /// `--max-logprobs` server configuration until capability probing lands.
    #[serde(default)]
    pub max_top_k: Option<usize>,
    /// Vocab size, when known at registration time.
    #[serde(default)]
    pub vocab_size: Option<usize>,
    /// Reserved capability field. Registrations setting this to true are
    /// rejected until a concrete server-built full-vocabulary source exists.
    #[serde(default)]
    pub supports_full_vocab: Option<bool>,
    /// Tokenizer hash for drift detection (§3.9 `TokenizerDrift`).
    #[serde(default)]
    pub tokenizer_hash: Option<String>,
    /// For `Remote` kinds: the base URL of the provider.
    #[serde(default)]
    pub url: Option<String>,
    /// For `Remote` kinds: which env var the runtime should read the
    /// API key from (never the key itself — §8.6 cost-lock policy).
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Free-form notes the user can attach (e.g. "shared cache key").
    #[serde(default)]
    pub notes: Option<String>,
    /// For `Local` kinds: a LoRA adapter (a directory name under the
    /// server's adapter dir) the teacher wears. Without this, a Local
    /// teacher always means the BARE BASE MODEL — which silently broke
    /// the two flagship continual-learning shapes: `distill_refresh`
    /// toward a prior self, and `self_improve`'s judge-as-teacher.
    #[serde(default)]
    pub adapter: Option<String>,
}

/// Concrete teacher kind. The §3.2 abstraction stays a closed enum
/// here so we can dispatch resolution at the server boundary;
/// arbitrary user-supplied sources go through a "register the binary
/// path" form that isn't shipped yet.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TeacherKind {
    /// Deterministic in-memory fixture (test harness; canonical
    /// pre-baked corpora).
    Fixture,
    /// Local model loaded in this kiln process (or accessible via a
    /// sibling process). Resolution at trainer time produces the
    /// `LocalTeacher` LogitSource impl.
    Local,
    /// HTTP logprobs. Registration currently accepts only vLLM; other
    /// providers need dedicated protocol adapters.
    Remote,
}

/// In-memory teacher registry. Thread-safe via `RwLock`. Persisted
/// to `state.adapter_dir/teachers.json` on every change so the
/// registry survives restart.
#[derive(Debug, Default)]
pub struct TeacherRegistry {
    inner: RwLock<BTreeMap<String, TeacherSpec>>,
}

impl TeacherRegistry {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(BTreeMap::new()),
        }
    }

    /// Load from disk if a `teachers.json` file exists at the given
    /// path. Errors are logged at WARN and the registry stays empty;
    /// missing file is not an error.
    pub fn load_from_path(path: &std::path::Path) -> Self {
        let reg = Self::new();
        if path.exists() {
            match std::fs::read(path) {
                Ok(bytes) => {
                    match serde_json::from_slice::<BTreeMap<String, TeacherSpec>>(&bytes) {
                        Ok(map) => {
                            *reg.inner.write().unwrap() = map;
                            tracing::info!(
                                path = %path.display(),
                                count = reg.inner.read().unwrap().len(),
                                "loaded teacher registry"
                            );
                        }
                        Err(e) => tracing::warn!(error = %e, "failed to parse teachers.json"),
                    }
                }
                Err(e) => tracing::warn!(error = %e, "failed to read teachers.json"),
            }
        }
        reg
    }

    /// Write the registry to `teachers.json`.
    pub fn save_to_path(&self, path: &std::path::Path) -> std::io::Result<()> {
        let map = self.inner.read().unwrap();
        let bytes = serde_json::to_vec_pretty(&*map)?;
        kiln_resource::locked_atomic_write(path, &bytes)
    }

    pub fn insert(&self, spec: TeacherSpec) {
        let mut m = self.inner.write().unwrap();
        m.insert(spec.alias.clone(), spec);
    }

    pub fn get(&self, alias: &str) -> Option<TeacherSpec> {
        self.inner.read().unwrap().get(alias).cloned()
    }

    pub fn list(&self) -> Vec<TeacherSpec> {
        self.inner.read().unwrap().values().cloned().collect()
    }

    pub fn remove(&self, alias: &str) -> bool {
        self.inner.write().unwrap().remove(alias).is_some()
    }
}

/// Capabilities surfaced by `GET /v1/teachers` — bundles the spec
/// with the resolved capability view (best-effort: for `Remote`
/// kinds where the registry doesn't know vocab_size, we report
/// what the user told us at registration time).
#[derive(Debug, Clone, Serialize)]
pub struct TeacherEntry {
    pub spec: TeacherSpec,
    /// Resolved capabilities. `None` when the resolver can't
    /// materialise the source (e.g. a Local spec whose model id
    /// doesn't match the loaded backend, which is the milestone-9
    /// state).
    pub capabilities: Option<LogitSourceCaps>,
}

#[derive(Debug, Serialize)]
struct TeachersListResponse {
    teachers: Vec<TeacherEntry>,
}

async fn list_teachers(State(state): State<AppState>) -> Json<TeachersListResponse> {
    let specs = state.teacher_registry.list();
    let teachers = specs
        .into_iter()
        .map(|spec| {
            let capabilities = resolve_caps_for(&spec);
            TeacherEntry { spec, capabilities }
        })
        .collect();
    Json(TeachersListResponse { teachers })
}

async fn register_teacher(
    State(state): State<AppState>,
    Json(spec): Json<TeacherSpec>,
) -> Result<Json<TeacherEntry>, ApiError> {
    validate_teacher_spec_for_use(&state, &spec)?;
    state.teacher_registry.insert(spec.clone());
    // Persist immediately so a crash doesn't lose the registration.
    let teachers_path = state.adapter_dir.join("teachers.json");
    if let Err(e) = state.teacher_registry.save_to_path(&teachers_path) {
        tracing::warn!(error = %e, path = %teachers_path.display(),
            "failed to persist teacher registry");
    }
    let capabilities = resolve_caps_for(&spec);
    Ok(Json(TeacherEntry { spec, capabilities }))
}

fn validate_remote_teacher_url(url: &str) -> Result<(), String> {
    kiln_train::normalize_vllm_completions_url(url).map(|_| ())
}

fn validate_teacher_spec_static(spec: &TeacherSpec) -> Result<(), String> {
    if spec.alias.trim().is_empty() {
        return Err("teacher alias must be non-empty".to_string());
    }
    if spec.model_id.trim().is_empty() {
        return Err("teacher model_id must be non-empty".to_string());
    }
    if spec.supports_full_vocab == Some(true) {
        return Err(
            "supports_full_vocab=true is not available: no concrete server-built teacher returns full-vocabulary logprobs"
                .to_string(),
        );
    }
    if spec.vocab_size == Some(0) {
        return Err("teacher vocab_size must be greater than zero when specified".to_string());
    }
    if spec.max_top_k == Some(0) && !matches!(spec.kind, TeacherKind::Remote) {
        return Err(
            "teacher max_top_k must be greater than zero when specified for a local or fixture teacher"
                .to_string(),
        );
    }
    if spec.adapter.is_some() && !matches!(spec.kind, TeacherKind::Local) {
        return Err(format!(
            "`adapter` is only valid on kind=local teachers (got kind={:?})",
            spec.kind
        ));
    }
    if matches!(spec.kind, TeacherKind::Remote) {
        match spec.provider {
            Some(kiln_train::RemoteProvider::Vllm) => {}
            Some(provider) => {
                return Err(format!(
                    "remote provider {provider:?} is not wired yet; only provider=\"vllm\" (vLLM numeric-ID prompt_logprobs) is supported today"
                ));
            }
            None => {
                return Err(
                    "Remote teacher requires explicit provider=\"vllm\" (vLLM); re-register legacy entries because provider inference from URLs is unsafe"
                        .to_string(),
                );
            }
        }
        let url = spec
            .url
            .as_deref()
            .ok_or_else(|| "Remote teacher requires a `url`".to_string())?;
        validate_remote_teacher_url(url)?;
        if spec
            .api_key_env
            .as_deref()
            .is_some_and(|name| name.trim().is_empty() || name.trim() != name)
        {
            return Err(
                "remote teacher api_key_env must be a non-empty environment-variable name without surrounding whitespace"
                    .to_string(),
            );
        }
    } else if spec.provider.is_some() || spec.url.is_some() || spec.api_key_env.is_some() {
        return Err(
            "`provider`, `url`, and `api_key_env` are only valid on kind=remote teachers"
                .to_string(),
        );
    }
    Ok(())
}

/// Apply the same admission contract to new and persisted registry entries.
/// Older `teachers.json` files can contain providers or capability claims that
/// current code no longer supports, so alias presence alone is insufficient.
fn validate_teacher_spec_for_use(state: &AppState, spec: &TeacherSpec) -> Result<(), ApiError> {
    validate_teacher_spec_static(spec).map_err(ApiError::training_invalid_request)?;
    if let Some(env_name) = spec.api_key_env.as_deref() {
        let value = std::env::var(env_name).map_err(|_| {
            ApiError::training_invalid_request(format!(
                "teacher api_key_env names {env_name:?}, but that environment variable is not set"
            ))
        })?;
        if value.trim().is_empty() {
            return Err(ApiError::training_invalid_request(format!(
                "teacher api_key_env names {env_name:?}, but that environment variable is empty"
            )));
        }
    }
    if let Some(adapter) = spec.adapter.as_deref() {
        super::adapters::validate_adapter_name(adapter)?;
        let dir = state.adapter_dir.join(adapter);
        if !dir.is_dir() {
            return Err(ApiError::training_invalid_request(format!(
                "teacher adapter `{adapter}` not found at {} — train or upload it first",
                dir.display()
            )));
        }
    }
    Ok(())
}

async fn delete_teacher(
    State(state): State<AppState>,
    AxumPath(alias): AxumPath<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if state.teacher_registry.remove(&alias) {
        // Persist after removal.
        let teachers_path = state.adapter_dir.join("teachers.json");
        if let Err(e) = state.teacher_registry.save_to_path(&teachers_path) {
            tracing::warn!(error = %e, "failed to persist teacher registry");
        }
        Ok(Json(serde_json::json!({
            "status": "deleted",
            "alias": alias
        })))
    } else {
        Err(ApiError::training_invalid_request(format!(
            "no teacher registered with alias {alias:?}"
        )))
    }
}

/// Build a best-effort `LogitSourceCaps` from the registration data
/// alone. Real resolution (actually constructing the `LogitSource`)
/// happens at trainer time when the model handles are available;
/// here we report what we know at registration so the dashboard can
/// show capabilities ahead of the first run.
fn resolve_caps_for(spec: &TeacherSpec) -> Option<LogitSourceCaps> {
    if matches!(spec.kind, TeacherKind::Remote)
        && spec.provider != Some(kiln_train::RemoteProvider::Vllm)
    {
        return None;
    }
    if !matches!(spec.kind, TeacherKind::Remote) && spec.max_top_k.is_none() {
        // Local and fixture sources are constructed with the admitted request
        // K when no bound was registered, so zero would be a false concrete
        // capability. Report unknown until construction instead.
        return None;
    }
    let configured_max_top_k = spec.max_top_k.unwrap_or(0);
    let configured_or_default_max_top_k =
        if matches!(spec.kind, TeacherKind::Remote) && configured_max_top_k == 0 {
            kiln_train::RemoteProvider::Vllm.default_max_top_k()
        } else {
            configured_max_top_k
        };
    let max_top_k = spec
        .vocab_size
        .map_or(configured_or_default_max_top_k, |vocab_size| {
            configured_or_default_max_top_k.min(vocab_size)
        });
    Some(LogitSourceCaps {
        teacher_id: spec.alias.clone(),
        vocab_size: spec.vocab_size.unwrap_or(0),
        max_top_k,
        supports_full_vocab: false,
        supports_batched: true,
        tokenizer_hash: spec.tokenizer_hash.clone(),
    })
}

/// Resolve a teacher alias to a `TeacherSpec`. Used by trainer code
/// inside `run_opd` (and future `/v1/distill/*` handlers) to look up
/// the registry entry. Returns a `LogitSourceError::Invalid` when
/// the alias is unknown.
#[allow(dead_code)]
pub fn resolve_teacher(
    registry: &TeacherRegistry,
    alias: &str,
) -> Result<TeacherSpec, LogitSourceError> {
    registry
        .get(alias)
        .ok_or_else(|| LogitSourceError::Invalid {
            teacher_id: alias.to_string(),
            message: format!("teacher alias {alias:?} not registered (POST /v1/teachers first)"),
        })
}

/// Resolve a teacher alias against the registry, failing with the
/// remediation-bearing 400 (`teacher_not_registered`) when missing.
/// Shared by every submission endpoint that names a teacher — a typo'd
/// alias must fail at submission, not at worker dequeue hours later
/// behind a queue.
pub(crate) fn require_registered_teacher(
    state: &AppState,
    alias: &str,
    detail: String,
) -> Result<TeacherSpec, ApiError> {
    if let Some(spec) = state.teacher_registry.get(alias) {
        validate_teacher_spec_for_use(state, &spec)?;
        return Ok(spec);
    }
    let registered: Vec<String> = state
        .teacher_registry
        .list()
        .into_iter()
        .map(|spec| spec.alias)
        .collect();
    Err(ApiError::teacher_not_registered(detail, &registered))
}

/// Shared registry handle stored on AppState.
pub type SharedTeacherRegistry = Arc<TeacherRegistry>;

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/teachers", get(list_teachers).post(register_teacher))
        .route("/v1/teachers/{alias}", delete(delete_teacher))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_insert_get_list_remove() {
        let reg = TeacherRegistry::new();
        let spec = TeacherSpec {
            alias: "qwen3.6-27b@local".into(),
            kind: TeacherKind::Local,
            provider: None,
            model_id: "qwen/qwen-3.6-27b".into(),
            max_top_k: Some(0),
            vocab_size: Some(152_064),
            supports_full_vocab: Some(true),
            tokenizer_hash: Some("sha256:abc".into()),
            url: None,
            api_key_env: None,
            notes: None,
            adapter: None,
        };
        reg.insert(spec.clone());
        let got = reg.get("qwen3.6-27b@local").unwrap();
        assert_eq!(got.model_id, spec.model_id);
        assert_eq!(reg.list().len(), 1);
        assert!(reg.remove("qwen3.6-27b@local"));
        assert!(reg.list().is_empty());
        assert!(!reg.remove("qwen3.6-27b@local"));
    }

    #[test]
    fn remote_registration_requires_explicit_supported_provider_and_valid_url() {
        validate_remote_teacher_url("http://127.0.0.1:8000").unwrap();
        validate_remote_teacher_url("http://127.0.0.1:8080").unwrap();
        assert!(validate_remote_teacher_url("").is_err());
        assert!(validate_remote_teacher_url("not a URL").is_err());
        assert!(validate_remote_teacher_url("http://vllm.local?mode=test").is_err());

        let mut spec = TeacherSpec {
            alias: "remote".into(),
            kind: TeacherKind::Remote,
            provider: Some(kiln_train::RemoteProvider::Vllm),
            model_id: "model".into(),
            max_top_k: None,
            vocab_size: Some(1024),
            supports_full_vocab: Some(false),
            tokenizer_hash: None,
            url: Some("http://127.0.0.1:8080".into()),
            api_key_env: None,
            notes: None,
            adapter: None,
        };
        validate_teacher_spec_static(&spec).unwrap();
        spec.provider = Some(kiln_train::RemoteProvider::Sglang);
        let error = validate_teacher_spec_static(&spec).unwrap_err();
        assert!(error.contains("Sglang"), "{error}");

        spec.provider = Some(kiln_train::RemoteProvider::Vllm);
        spec.api_key_env = Some(String::new());
        let error = validate_teacher_spec_static(&spec).unwrap_err();
        assert!(error.contains("api_key_env"), "{error}");

        spec.kind = TeacherKind::Local;
        spec.provider = None;
        spec.api_key_env = None;
        let error = validate_teacher_spec_static(&spec).unwrap_err();
        assert!(error.contains("only valid on kind=remote"), "{error}");
    }

    #[test]
    fn persisted_unsupported_remote_spec_is_rejected_when_reused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("teachers.json");
        std::fs::write(
            &path,
            r#"{
              "legacy": {
                "alias": "legacy",
                "kind": "remote",
                "model_id": "old-model",
                "url": "http://sglang.internal:30000",
                "max_top_k": 20,
                "vocab_size": 1024
              }
            }"#,
        )
        .unwrap();

        let registry = TeacherRegistry::load_from_path(&path);
        let spec = registry.get("legacy").expect("legacy entry loaded");
        let error = validate_teacher_spec_static(&spec).unwrap_err();
        assert!(error.contains("explicit provider=\"vllm\""), "{error}");
    }

    #[test]
    fn registry_round_trip_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("teachers.json");
        let reg = TeacherRegistry::new();
        reg.insert(TeacherSpec {
            alias: "fixture@test".into(),
            kind: TeacherKind::Fixture,
            provider: None,
            model_id: "test".into(),
            max_top_k: Some(32),
            vocab_size: Some(1024),
            supports_full_vocab: Some(false),
            tokenizer_hash: None,
            url: None,
            api_key_env: None,
            notes: Some("unit test".into()),
            adapter: None,
        });
        reg.save_to_path(&path).unwrap();

        let reg2 = TeacherRegistry::load_from_path(&path);
        let got = reg2.get("fixture@test").unwrap();
        assert_eq!(got.notes.as_deref(), Some("unit test"));
        assert!(matches!(got.kind, TeacherKind::Fixture));
    }

    #[test]
    fn resolve_caps_uses_registration_data() {
        let spec = TeacherSpec {
            alias: "x".into(),
            kind: TeacherKind::Remote,
            provider: Some(kiln_train::RemoteProvider::Vllm),
            model_id: "y".into(),
            max_top_k: None,
            vocab_size: Some(50_257),
            // Capability reporting must not echo an unimplemented claim.
            supports_full_vocab: Some(true),
            tokenizer_hash: None,
            url: Some("https://api".into()),
            api_key_env: None,
            notes: None,
            adapter: None,
        };
        let caps = resolve_caps_for(&spec).unwrap();
        assert_eq!(caps.teacher_id, "x");
        assert_eq!(caps.max_top_k, 20);
        assert_eq!(caps.vocab_size, 50_257);
        assert!(!caps.supports_full_vocab);
        assert!(caps.supports_batched);
    }

    #[test]
    fn resolve_caps_never_advertises_more_tokens_than_the_vocabulary() {
        let spec = TeacherSpec {
            alias: "tiny".into(),
            kind: TeacherKind::Remote,
            provider: Some(kiln_train::RemoteProvider::Vllm),
            model_id: "tiny-model".into(),
            max_top_k: Some(32),
            vocab_size: Some(8),
            supports_full_vocab: Some(false),
            tokenizer_hash: None,
            url: Some("https://api".into()),
            api_key_env: None,
            notes: None,
            adapter: None,
        };

        let caps = resolve_caps_for(&spec).unwrap();
        assert_eq!(caps.max_top_k, 8);
    }

    #[test]
    fn resolve_caps_reports_fixture_batching_and_unknown_dynamic_bounds_honestly() {
        let mut spec = TeacherSpec {
            alias: "fixture".into(),
            kind: TeacherKind::Fixture,
            provider: None,
            model_id: "fixture".into(),
            max_top_k: Some(16),
            vocab_size: Some(32),
            supports_full_vocab: Some(false),
            tokenizer_hash: None,
            url: None,
            api_key_env: None,
            notes: None,
            adapter: None,
        };

        let caps = resolve_caps_for(&spec).unwrap();
        assert_eq!(caps.max_top_k, 16);
        assert!(caps.supports_batched);

        spec.max_top_k = None;
        assert!(resolve_caps_for(&spec).is_none());
        spec.kind = TeacherKind::Local;
        assert!(resolve_caps_for(&spec).is_none());
    }

    #[test]
    fn resolve_alias_returns_invalid_on_unknown() {
        let reg = TeacherRegistry::new();
        let err = resolve_teacher(&reg, "nope").unwrap_err();
        match err {
            LogitSourceError::Invalid {
                teacher_id,
                message,
            } => {
                assert_eq!(teacher_id, "nope");
                assert!(message.contains("not registered"));
            }
            other => panic!("expected Invalid, got {other:?}"),
        }
    }

    #[test]
    fn spec_round_trips_through_serde() {
        let spec = TeacherSpec {
            alias: "vllm@qwen".into(),
            kind: TeacherKind::Remote,
            provider: Some(kiln_train::RemoteProvider::Vllm),
            model_id: "qwen/qwen-3.6-27b".into(),
            max_top_k: Some(20),
            vocab_size: None,
            supports_full_vocab: Some(false),
            tokenizer_hash: None,
            url: Some("http://vllm.internal:8000".into()),
            api_key_env: None,
            notes: None,
            adapter: None,
        };
        let s = serde_json::to_string(&spec).unwrap();
        let parsed: TeacherSpec = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed.alias, spec.alias);
        assert!(matches!(parsed.kind, TeacherKind::Remote));
        assert_eq!(parsed.provider, Some(kiln_train::RemoteProvider::Vllm));
    }
}
