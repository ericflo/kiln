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
//! of how to build a `LogitSource`). At resolution time (e.g. inside
//! `run_opd`) the spec is materialised into a concrete `LogitSource`
//! impl. For milestone-9 we ship two kinds:
//!
//! - **Fixture** — in-memory deterministic source for unit tests +
//!   `kiln-canonical` corpora that ship with pre-baked logits.
//! - **Local** — placeholder pointing at "the model loaded in this
//!   process at `model_id`." A second model handle is wired in when
//!   the §3.1 trainer body lands; for now the resolver returns an
//!   "OPD runtime not yet implemented" error.
//!
//! Future kinds (RemoteTeacher × 8 providers, CachedTeacher) plug in
//! the same way.

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use axum::Json;
use axum::Router;
use axum::extract::{Path as AxumPath, State};
use axum::routing::{delete, get, post};
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
    /// Provider's model id (e.g. `qwen/qwen-3.6-27b`).
    pub model_id: String,
    /// Maximum top-K this teacher can return.
    #[serde(default)]
    pub max_top_k: Option<usize>,
    /// Vocab size, when known at registration time.
    #[serde(default)]
    pub vocab_size: Option<usize>,
    /// Whether the source supports full-vocab logprobs (DeepSeek-V4-
    /// style multi-teacher consolidation needs this).
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
    /// HTTP `top_logprobs` (OpenAI-compatible). vLLM, sglang,
    /// llama.cpp, OpenRouter, Together, Fireworks, etc.
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
                Ok(bytes) => match serde_json::from_slice::<BTreeMap<String, TeacherSpec>>(&bytes)
                {
                    Ok(map) => {
                        *reg.inner.write().unwrap() = map;
                        tracing::info!(
                            path = %path.display(),
                            count = reg.inner.read().unwrap().len(),
                            "loaded teacher registry"
                        );
                    }
                    Err(e) => tracing::warn!(error = %e, "failed to parse teachers.json"),
                },
                Err(e) => tracing::warn!(error = %e, "failed to read teachers.json"),
            }
        }
        reg
    }

    /// Write the registry to `teachers.json`.
    pub fn save_to_path(&self, path: &std::path::Path) -> std::io::Result<()> {
        let map = self.inner.read().unwrap();
        let bytes = serde_json::to_vec_pretty(&*map)?;
        std::fs::write(path, bytes)
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
            TeacherEntry {
                spec,
                capabilities,
            }
        })
        .collect();
    Json(TeachersListResponse { teachers })
}

async fn register_teacher(
    State(state): State<AppState>,
    Json(spec): Json<TeacherSpec>,
) -> Result<Json<TeacherEntry>, ApiError> {
    if spec.alias.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "teacher alias must be non-empty".to_string(),
        ));
    }
    if spec.model_id.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "teacher model_id must be non-empty".to_string(),
        ));
    }
    if matches!(spec.kind, TeacherKind::Remote) && spec.url.is_none() {
        return Err(ApiError::training_invalid_request(
            "Remote teacher requires a `url`".to_string(),
        ));
    }
    state.teacher_registry.insert(spec.clone());
    // Persist immediately so a crash doesn't lose the registration.
    let teachers_path = state.adapter_dir.join("teachers.json");
    if let Err(e) = state.teacher_registry.save_to_path(&teachers_path) {
        tracing::warn!(error = %e, path = %teachers_path.display(),
            "failed to persist teacher registry");
    }
    let capabilities = resolve_caps_for(&spec);
    Ok(Json(TeacherEntry {
        spec,
        capabilities,
    }))
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
    Some(LogitSourceCaps {
        teacher_id: spec.alias.clone(),
        vocab_size: spec.vocab_size.unwrap_or(0),
        max_top_k: spec.max_top_k.unwrap_or(0),
        supports_full_vocab: spec.supports_full_vocab.unwrap_or(false),
        supports_batched: matches!(spec.kind, TeacherKind::Local | TeacherKind::Remote),
        tokenizer_hash: spec.tokenizer_hash.clone(),
    })
}

/// Resolve a teacher alias to a `TeacherSpec`. Used by trainer code
/// inside `run_opd` (and future `/v1/distill/*` handlers) to look up
/// the registry entry. Returns a `LogitSourceError::Invalid` when
/// the alias is unknown.
pub fn resolve_teacher(registry: &TeacherRegistry, alias: &str) -> Result<TeacherSpec, LogitSourceError> {
    registry.get(alias).ok_or_else(|| LogitSourceError::Invalid {
        teacher_id: alias.to_string(),
        message: format!("teacher alias {alias:?} not registered (POST /v1/teachers first)"),
    })
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
            model_id: "qwen/qwen-3.6-27b".into(),
            max_top_k: Some(0),
            vocab_size: Some(152_064),
            supports_full_vocab: Some(true),
            tokenizer_hash: Some("sha256:abc".into()),
            url: None,
            api_key_env: None,
            notes: None,
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
    fn registry_round_trip_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("teachers.json");
        let reg = TeacherRegistry::new();
        reg.insert(TeacherSpec {
            alias: "fixture@test".into(),
            kind: TeacherKind::Fixture,
            model_id: "test".into(),
            max_top_k: Some(32),
            vocab_size: Some(1024),
            supports_full_vocab: Some(false),
            tokenizer_hash: None,
            url: None,
            api_key_env: None,
            notes: Some("unit test".into()),
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
            model_id: "y".into(),
            max_top_k: Some(20),
            vocab_size: Some(50_257),
            supports_full_vocab: Some(false),
            tokenizer_hash: None,
            url: Some("https://api".into()),
            api_key_env: None,
            notes: None,
        };
        let caps = resolve_caps_for(&spec).unwrap();
        assert_eq!(caps.teacher_id, "x");
        assert_eq!(caps.max_top_k, 20);
        assert_eq!(caps.vocab_size, 50_257);
        assert!(!caps.supports_full_vocab);
        assert!(caps.supports_batched);
    }

    #[test]
    fn resolve_alias_returns_invalid_on_unknown() {
        let reg = TeacherRegistry::new();
        let err = resolve_teacher(&reg, "nope").unwrap_err();
        match err {
            LogitSourceError::Invalid { teacher_id, message } => {
                assert_eq!(teacher_id, "nope");
                assert!(message.contains("not registered"));
            }
            other => panic!("expected Invalid, got {other:?}"),
        }
    }

    #[test]
    fn spec_round_trips_through_serde() {
        let spec = TeacherSpec {
            alias: "openrouter@qwen".into(),
            kind: TeacherKind::Remote,
            model_id: "qwen/qwen-3.6-27b".into(),
            max_top_k: Some(20),
            vocab_size: None,
            supports_full_vocab: Some(false),
            tokenizer_hash: None,
            url: Some("https://openrouter.ai/api/v1".into()),
            api_key_env: Some("OPENROUTER_API_KEY".into()),
            notes: None,
        };
        let s = serde_json::to_string(&spec).unwrap();
        let parsed: TeacherSpec = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed.alias, spec.alias);
        assert!(matches!(parsed.kind, TeacherKind::Remote));
        assert_eq!(parsed.api_key_env.as_deref(), Some("OPENROUTER_API_KEY"));
    }
}
