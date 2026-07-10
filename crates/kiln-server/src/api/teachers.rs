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
use std::sync::{Arc, OnceLock, RwLock};

use axum::Json;
use axum::Router;
use axum::extract::{Path as AxumPath, State};
use axum::routing::{delete, get};
use serde::{Deserialize, Serialize};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use kiln_train::{LogitSourceCaps, LogitSourceError};

use crate::error::ApiError;
use crate::state::AppState;

const MAX_CONCURRENT_TEACHER_IDENTITY_PROBES: usize = 2;
const REGISTRATION_REMOTE_PROBE_TIMEOUT_MS: u64 = 10_000;
static TEACHER_IDENTITY_PROBE_SEMAPHORE: OnceLock<Arc<Semaphore>> = OnceLock::new();

fn try_teacher_identity_probe_permit() -> Result<OwnedSemaphorePermit, ApiError> {
    let semaphore = TEACHER_IDENTITY_PROBE_SEMAPHORE
        .get_or_init(|| Arc::new(Semaphore::new(MAX_CONCURRENT_TEACHER_IDENTITY_PROBES)))
        .clone();
    try_teacher_identity_probe_permit_from(semaphore)
}

fn try_teacher_identity_probe_permit_from(
    semaphore: Arc<Semaphore>,
) -> Result<OwnedSemaphorePermit, ApiError> {
    semaphore
        .try_acquire_owned()
        .map_err(|_| ApiError::teacher_identity_probe_busy())
}

fn registration_probe_config(
    mut config: kiln_train::RemoteTeacherConfig,
) -> kiln_train::RemoteTeacherConfig {
    config.timeout_ms = REGISTRATION_REMOTE_PROBE_TIMEOUT_MS;
    config
}

/// One entry in the teacher registry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
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
    /// Canonical content and protocol identity verified by an operational
    /// remote scoring probe. Remote entries without this field are legacy and
    /// cannot be used until they are re-registered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identity: Option<kiln_train::TeacherIdentityV1>,
    /// For `Remote` kinds: the base URL of the provider.
    #[serde(default)]
    pub url: Option<String>,
    /// For `Remote` kinds: opaque handle into the immutable server-owned
    /// `[teachers.credentials]` configuration. The backing environment-variable
    /// name is never accepted from or exposed to API clients.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credential_id: Option<String>,
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

/// Caller-controlled registration fields. Authoritative identity and
/// capability fields exist only on the persisted [`TeacherSpec`] and are
/// populated by the server after an operational probe.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegisterTeacherRequest {
    alias: String,
    kind: TeacherKind,
    #[serde(default)]
    provider: Option<kiln_train::RemoteProvider>,
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    credential_id: Option<String>,
    #[serde(default)]
    notes: Option<String>,
    #[serde(default)]
    adapter: Option<String>,
}

impl RegisterTeacherRequest {
    fn into_unverified_spec(self) -> TeacherSpec {
        TeacherSpec {
            alias: self.alias,
            kind: self.kind,
            provider: self.provider,
            model_id: self.model_id.unwrap_or_default(),
            max_top_k: None,
            vocab_size: None,
            supports_full_vocab: None,
            tokenizer_hash: None,
            identity: None,
            url: self.url,
            credential_id: self.credential_id,
            notes: self.notes,
            adapter: self.adapter,
        }
    }
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
                        Err(strict_error) => match migrate_legacy_secret_fields(&bytes) {
                            Ok(Some(map)) => {
                                if let Err(error) = persist_teacher_map(path, &map) {
                                    tracing::warn!(
                                        error = %error,
                                        path = %path.display(),
                                        "loaded sanitized legacy teacher registry but could not persist the migration"
                                    );
                                }
                                let count = map.len();
                                *reg.inner.write().unwrap() = map;
                                tracing::warn!(
                                    path = %path.display(),
                                    count,
                                    "removed legacy teacher secret-environment fields; remote entries must be re-registered with credential_id and an authoritative identity"
                                );
                            }
                            Ok(None) => tracing::warn!(
                                error = %strict_error,
                                "failed to parse teachers.json"
                            ),
                            Err(migration_error) => tracing::warn!(
                                error = %strict_error,
                                migration_error = %migration_error,
                                "failed to parse teachers.json"
                            ),
                        },
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

    /// Persist a new alias before publishing it in memory. Aliases are
    /// immutable: callers must delete and re-register explicitly rather than
    /// silently changing a deployment underneath queued jobs.
    pub fn insert_new_and_save(
        &self,
        spec: TeacherSpec,
        path: &std::path::Path,
    ) -> Result<(), TeacherRegistryMutationError> {
        let mut current = self.inner.write().unwrap();
        if current.contains_key(&spec.alias) {
            return Err(TeacherRegistryMutationError::Duplicate(spec.alias));
        }
        let mut next = current.clone();
        next.insert(spec.alias.clone(), spec);
        persist_teacher_map(path, &next)?;
        *current = next;
        Ok(())
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

    /// Persist a deletion before publishing it in memory.
    pub fn remove_and_save(
        &self,
        alias: &str,
        path: &std::path::Path,
    ) -> Result<bool, TeacherRegistryMutationError> {
        let mut current = self.inner.write().unwrap();
        if !current.contains_key(alias) {
            return Ok(false);
        }
        let mut next = current.clone();
        next.remove(alias);
        persist_teacher_map(path, &next)?;
        *current = next;
        Ok(true)
    }
}

/// One-way migration for registries written before credentials became
/// server-owned handles. The old environment-variable name is discarded, not
/// trusted or copied into the new spec. Every migrated remote entry remains
/// identity-less and therefore unusable until an explicit re-registration.
fn migrate_legacy_secret_fields(
    bytes: &[u8],
) -> Result<Option<BTreeMap<String, TeacherSpec>>, String> {
    let mut value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
    let entries = value
        .as_object_mut()
        .ok_or_else(|| "teacher registry must be a JSON object".to_string())?;
    let mut migrated = false;
    for spec in entries.values_mut() {
        let fields = spec
            .as_object_mut()
            .ok_or_else(|| "teacher registry entries must be JSON objects".to_string())?;
        migrated |= fields.remove("api_key_env").is_some();
    }
    if !migrated {
        return Ok(None);
    }
    serde_json::from_value(value)
        .map(Some)
        .map_err(|error| format!("legacy registry remains invalid after secret removal: {error}"))
}

#[derive(Debug, thiserror::Error)]
pub enum TeacherRegistryMutationError {
    #[error(
        "teacher alias {0:?} is already registered; delete it before registering a different deployment"
    )]
    Duplicate(String),
    #[error("persist teacher registry: {0}")]
    Persist(#[from] std::io::Error),
}

fn persist_teacher_map(
    path: &std::path::Path,
    map: &BTreeMap<String, TeacherSpec>,
) -> std::io::Result<()> {
    let bytes = serde_json::to_vec_pretty(map)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    kiln_resource::locked_atomic_write(path, &bytes)
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
    /// Stable product state. Remote teachers are usable only after an
    /// operational probe has pinned their canonical identity.
    pub status: TeacherStatus,
    /// Whether new jobs may name this alias.
    pub usable: bool,
    /// SHA-256 revision of the complete canonical identity, when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identity_revision: Option<String>,
    /// Exact canonical first line for pre-scored off-policy OPD JSONL. Clients
    /// can write this string verbatim instead of reconstructing identity JSON.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub off_policy_manifest: Option<String>,
    /// Bounded remediation for entries loaded from an older registry or made
    /// unavailable by a local configuration change.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status_message: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TeacherStatus {
    Verified,
    Configured,
    LegacyUnverified,
    Unavailable,
}

#[derive(Debug, Serialize)]
struct TeachersListResponse {
    teachers: Vec<TeacherEntry>,
}

async fn list_teachers(State(state): State<AppState>) -> Json<TeachersListResponse> {
    let specs = state.teacher_registry.list();
    let teachers = specs
        .into_iter()
        .map(|spec| teacher_entry(&state, spec))
        .collect();
    Json(TeachersListResponse { teachers })
}

async fn register_teacher(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<TeacherEntry>, ApiError> {
    let request: RegisterTeacherRequest = serde_json::from_value(body).map_err(|error| {
        ApiError::teacher_registration_invalid(format!(
            "{error}; identity, capability, and secret environment fields are server-controlled"
        ))
    })?;
    let mut spec = request.into_unverified_spec();
    validate_teacher_spec_for_registration(&state, &spec)?;
    if state.teacher_registry.get(&spec.alias).is_some() {
        return Err(ApiError::teacher_alias_exists(&spec.alias));
    }

    if matches!(spec.kind, TeacherKind::Remote) {
        let permit = try_teacher_identity_probe_permit()?;
        let probe_config =
            registration_probe_config(remote_teacher_config(&spec, &state.teacher_credentials)?);
        let identity = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            kiln_train::discover_vllm_identity(&probe_config)
        })
        .await
        .map_err(|error| ApiError::internal(format!("remote teacher probe panicked: {error}")))?
        .map_err(ApiError::teacher_identity_probe_failed)?;
        validate_remote_identity_against_student(&state, &identity)?;
        spec.model_id = identity.served_model_id().to_owned();
        spec.max_top_k = Some(identity.max_top_k() as usize);
        spec.vocab_size = Some(identity.vocab_size() as usize);
        spec.supports_full_vocab = Some(false);
        spec.tokenizer_hash = Some(identity.tokenizer_vocab_sha256().to_owned());
        spec.identity = Some(identity);
    } else if matches!(spec.kind, TeacherKind::Local) {
        let base_identity = state.base_teacher_identity.as_ref().ok_or_else(|| {
            ApiError::teacher_registration_invalid(
                "local teachers require a real loaded model with authoritative base identity",
            )
        })?;
        let identity = if let Some(adapter_name) = spec.adapter.clone() {
            let permit = try_teacher_identity_probe_permit()?;
            let adapter_dir = state.adapter_dir.join(&adapter_name);
            let source = tokio::task::spawn_blocking(move || {
                let _permit = permit;
                kiln_model::lora_loader::LoraSourceIdentity::from_adapter_dir(&adapter_dir)
            })
            .await
            .map_err(|error| {
                ApiError::internal(format!(
                    "local teacher fingerprint worker panicked: {error}"
                ))
            })?
            .map_err(|error| {
                ApiError::teacher_registration_invalid(format!(
                    "could not fingerprint local teacher adapter {adapter_name:?}: {error:#}"
                ))
            })?;
            crate::teacher_identity::build_local_adapter_teacher_identity_from_source(
                base_identity,
                &adapter_name,
                &source,
            )
            .map_err(ApiError::teacher_registration_invalid)?
        } else {
            base_identity.as_ref().clone()
        };
        spec.model_id = identity.served_model_id().to_owned();
        spec.max_top_k = Some(identity.max_top_k() as usize);
        spec.vocab_size = Some(identity.vocab_size() as usize);
        spec.supports_full_vocab = Some(false);
        spec.tokenizer_hash = Some(identity.tokenizer_vocab_sha256().to_owned());
        spec.identity = Some(identity);
    }

    validate_teacher_spec_for_use(&state, &spec)?;
    let teachers_path = state.adapter_dir.join("teachers.json");
    state
        .teacher_registry
        .insert_new_and_save(spec.clone(), &teachers_path)
        .map_err(|error| match error {
            TeacherRegistryMutationError::Duplicate(_) => {
                ApiError::teacher_alias_exists(&spec.alias)
            }
            TeacherRegistryMutationError::Persist(_) => ApiError::internal(error.to_string()),
        })?;
    Ok(Json(teacher_entry(&state, spec)))
}

fn teacher_entry(state: &AppState, spec: TeacherSpec) -> TeacherEntry {
    let identity_revision = spec
        .identity
        .as_ref()
        .map(|identity| format!("sha256:{}", identity.content_revision()));
    let off_policy_manifest = spec.identity.as_ref().map(|identity| {
        kiln_train::OffPolicyDistillationManifestV1::new(identity.clone()).canonical_json()
    });
    let capabilities = resolve_caps_for(&spec);
    let validation = validate_teacher_spec_for_use(state, &spec);
    let usable = validation.is_ok();
    let status = if matches!(spec.kind, TeacherKind::Remote | TeacherKind::Local)
        && spec.identity.is_none()
    {
        TeacherStatus::LegacyUnverified
    } else if !usable {
        TeacherStatus::Unavailable
    } else if matches!(spec.kind, TeacherKind::Remote | TeacherKind::Local) {
        TeacherStatus::Verified
    } else {
        TeacherStatus::Configured
    };
    let status_message = validation.err().map(|error| error.message);
    TeacherEntry {
        spec,
        capabilities,
        status,
        usable,
        identity_revision,
        off_policy_manifest,
        status_message,
    }
}

pub(crate) fn remote_teacher_config(
    spec: &TeacherSpec,
    credentials: &crate::config::TeachersConfig,
) -> Result<kiln_train::RemoteTeacherConfig, ApiError> {
    let url = spec
        .url
        .clone()
        .ok_or_else(|| ApiError::teacher_registration_invalid("remote teacher requires a url"))?;
    let api_key_env = credentials
        .resolve_api_key_env(spec.credential_id.as_deref(), &url)
        .map_err(ApiError::teacher_registration_invalid)?;
    Ok(kiln_train::RemoteTeacherConfig {
        provider: spec.provider.ok_or_else(|| {
            ApiError::teacher_registration_invalid("remote teacher requires provider=\"vllm\"")
        })?,
        model: spec.model_id.clone(),
        url,
        api_key_env,
        teacher_id: spec.alias.clone(),
        expected_identity: spec.identity.clone(),
        tokenizer_hash: spec.tokenizer_hash.clone(),
        max_top_k: spec.max_top_k.unwrap_or(0),
        vocab_size: spec.vocab_size.unwrap_or(0),
        max_cost_usd: None,
        timeout_ms: 60_000,
    })
}

fn raw_sha256(value: &str, field: &str) -> Result<String, ApiError> {
    value
        .strip_prefix("sha256:")
        .filter(|digest| {
            digest.len() == 64
                && digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
        .map(str::to_owned)
        .ok_or_else(|| {
            ApiError::internal(format!(
                "local {field} is not a canonical sha256:<64 lowercase hex> digest"
            ))
        })
}

fn validate_remote_identity_against_student(
    state: &AppState,
    identity: &kiln_train::TeacherIdentityV1,
) -> Result<(), ApiError> {
    let local_vocab_hash = raw_sha256(
        &state.tokenizer.vocab_identity_sha256(),
        "tokenizer vocabulary identity",
    )?;
    if identity.tokenizer_vocab_sha256() != local_vocab_hash {
        return Err(ApiError::teacher_identity_mismatch(format!(
            "remote teacher tokenizer vocabulary identity {} does not match the loaded student's {}; raw numeric token IDs would have different semantics",
            identity.tokenizer_vocab_sha256(),
            local_vocab_hash
        )));
    }
    let tokenizer_vocab_size = state.tokenizer.vocab_size();
    if identity.vocab_size() as usize != tokenizer_vocab_size
        || identity.vocab_size() as usize != state.model_config.vocab_size
    {
        return Err(ApiError::teacher_identity_mismatch(format!(
            "remote teacher vocab_size {} must match both the loaded tokenizer ({tokenizer_vocab_size}) and model ({})",
            identity.vocab_size(),
            state.model_config.vocab_size
        )));
    }
    Ok(())
}

fn validate_remote_teacher_url(url: &str) -> Result<(), String> {
    kiln_train::normalize_vllm_completions_url(url).map(|_| ())
}

fn validate_teacher_spec_static(spec: &TeacherSpec) -> Result<(), String> {
    if spec.alias.trim().is_empty() {
        return Err("teacher alias must be non-empty".to_string());
    }
    if spec.model_id.trim().is_empty() && !matches!(spec.kind, TeacherKind::Local) {
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
    if spec.identity.is_some() && !matches!(spec.kind, TeacherKind::Remote | TeacherKind::Local) {
        return Err("`identity` is only valid on remote or local teachers".to_string());
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
        if let Some(credential_id) = spec.credential_id.as_deref() {
            crate::config::validate_teacher_credential_id(credential_id)?;
        }
        validate_remote_registration_exposure(spec)?;
    } else if spec.provider.is_some() || spec.url.is_some() || spec.credential_id.is_some() {
        return Err(
            "`provider`, `url`, and `credential_id` are only valid on kind=remote teachers"
                .to_string(),
        );
    }
    Ok(())
}

fn validate_remote_registration_exposure(spec: &TeacherSpec) -> Result<(), String> {
    if remote_registration_exposure_denied(spec, crate::api::terminal::bind_host_is_loopback()) {
        return Err(
            "credential-free loopback teachers are disabled while Kiln is network-bound; configure an exact-origin credential_id or bind Kiln to loopback"
                .to_string(),
        );
    }
    Ok(())
}

fn remote_registration_exposure_denied(spec: &TeacherSpec, kiln_is_loopback: bool) -> bool {
    matches!(spec.kind, TeacherKind::Remote) && spec.credential_id.is_none() && !kiln_is_loopback
}

/// Apply the same admission contract to new and persisted registry entries.
/// Older `teachers.json` files can contain providers or capability claims that
/// current code no longer supports, so alias presence alone is insufficient.
fn validate_teacher_spec_for_registration(
    state: &AppState,
    spec: &TeacherSpec,
) -> Result<(), ApiError> {
    validate_teacher_spec_static(spec).map_err(ApiError::teacher_registration_invalid)?;
    if matches!(spec.kind, TeacherKind::Remote) {
        let url = spec.url.as_deref().ok_or_else(|| {
            ApiError::teacher_registration_invalid("remote teacher requires a url")
        })?;
        state
            .teacher_credentials
            .resolve_api_key_env(spec.credential_id.as_deref(), url)
            .map_err(ApiError::teacher_registration_invalid)?;
    }
    if let Some(adapter) = spec.adapter.as_deref() {
        super::adapters::validate_adapter_name(adapter)?;
        let dir = state.adapter_dir.join(adapter);
        if !dir.is_dir() {
            return Err(ApiError::teacher_registration_invalid(format!(
                "teacher adapter `{adapter}` not found at {} — train or upload it first",
                dir.display()
            )));
        }
    }
    Ok(())
}

fn validate_teacher_spec_for_use(state: &AppState, spec: &TeacherSpec) -> Result<(), ApiError> {
    validate_teacher_spec_for_registration(state, spec)?;
    if matches!(spec.kind, TeacherKind::Remote) {
        let identity = spec
            .identity
            .as_ref()
            .ok_or_else(|| ApiError::teacher_identity_required(&spec.alias))?;
        let config = remote_teacher_config(spec, &state.teacher_credentials)?;
        kiln_train::RemoteTeacher::new(config).map_err(|error| {
            ApiError::teacher_registration_invalid(format!(
                "remote teacher {:?} has an invalid pinned identity: {error}",
                spec.alias
            ))
        })?;
        validate_remote_identity_against_student(state, identity)?;
    } else if matches!(spec.kind, TeacherKind::Local) {
        validate_local_teacher_identity(state, spec)?;
    }
    Ok(())
}

fn validate_local_teacher_identity(state: &AppState, spec: &TeacherSpec) -> Result<(), ApiError> {
    let identity = spec
        .identity
        .as_ref()
        .ok_or_else(|| ApiError::teacher_identity_required(&spec.alias))?;
    let base = state
        .base_teacher_identity
        .as_ref()
        .ok_or_else(|| {
            ApiError::teacher_registration_invalid(
                "local teacher cannot be used because this server has no authoritative loaded base-model identity",
            )
        })?;
    let expected = match (spec.adapter.as_deref(), identity.adapter()) {
        (None, None) => base.as_ref().clone(),
        (Some(adapter_name), Some(adapter)) if adapter.name() == adapter_name => base
            .with_static_adapter(adapter.clone())
            .map_err(ApiError::teacher_registration_invalid)?,
        (Some(adapter_name), Some(adapter)) => {
            return Err(ApiError::teacher_identity_mismatch(format!(
                "local teacher adapter name {adapter_name:?} does not match pinned identity adapter {:?}",
                adapter.name()
            )));
        }
        (Some(adapter_name), None) => {
            return Err(ApiError::teacher_identity_mismatch(format!(
                "local teacher names adapter {adapter_name:?} but its pinned identity is the bare base model"
            )));
        }
        (None, Some(adapter)) => {
            return Err(ApiError::teacher_identity_mismatch(format!(
                "local bare-base teacher unexpectedly pins adapter {:?}",
                adapter.name()
            )));
        }
    };
    if &expected != identity {
        return Err(ApiError::teacher_identity_mismatch(format!(
            "local teacher identity revision sha256:{} is not derived from this server's loaded base revision sha256:{}",
            identity.content_revision(),
            base.content_revision()
        )));
    }
    if spec.model_id != identity.served_model_id() {
        return Err(ApiError::teacher_identity_mismatch(format!(
            "local teacher model_id {:?} does not match authoritative served model {:?}",
            spec.model_id,
            identity.served_model_id()
        )));
    }
    Ok(())
}

async fn delete_teacher(
    State(state): State<AppState>,
    AxumPath(alias): AxumPath<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let teachers_path = state.adapter_dir.join("teachers.json");
    if state
        .teacher_registry
        .remove_and_save(&alias, &teachers_path)
        .map_err(|error| ApiError::internal(error.to_string()))?
    {
        Ok(Json(serde_json::json!({
            "status": "deleted",
            "alias": alias
        })))
    } else {
        Err(ApiError::teacher_not_found(alias))
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
    if matches!(spec.kind, TeacherKind::Remote | TeacherKind::Local) {
        let identity = spec.identity.as_ref()?;
        return Some(LogitSourceCaps {
            teacher_id: spec.alias.clone(),
            vocab_size: identity.vocab_size() as usize,
            max_top_k: identity.max_top_k() as usize,
            supports_full_vocab: false,
            supports_batched: true,
            tokenizer_hash: Some(identity.tokenizer_vocab_sha256().to_owned()),
        });
    }
    if spec.max_top_k.is_none() {
        // Local and fixture sources are constructed with the admitted request
        // K when no bound was registered, so zero would be a false concrete
        // capability. Report unknown until construction instead.
        return None;
    }
    let configured_max_top_k = spec.max_top_k.unwrap_or(0);
    let configured_or_default_max_top_k = configured_max_top_k;
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

    const HASH_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const HASH_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const HASH_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn remote_identity(
        model: &str,
        vocab_size: u32,
        max_top_k: u32,
    ) -> kiln_train::TeacherIdentityV1 {
        kiln_train::TeacherIdentityV1::new(
            model,
            HASH_A,
            HASH_B,
            HASH_C,
            None,
            vocab_size,
            max_top_k,
            4096,
            65_536,
            "vllm:0.25.0",
            HASH_A,
        )
        .unwrap()
    }

    fn mock_teacher_state() -> AppState {
        let mut model_config = kiln_core::config::ModelConfig::qwen3_5_4b();
        model_config.vocab_size = 3;
        let scheduler = kiln_scheduler::Scheduler::new(
            kiln_scheduler::SchedulerConfig {
                max_batch_tokens: 128,
                max_batch_size: 4,
                block_size: 16,
                prefix_cache_enabled: false,
                ..Default::default()
            },
            32,
        );
        let engine = kiln_model::engine::MockEngine::new(model_config.clone());
        let mut vocab = std::collections::HashMap::new();
        vocab.insert("a", 0u32);
        vocab.insert("b", 1u32);
        vocab.insert("c", 2u32);
        let tokenizer = kiln_core::tokenizer::KilnTokenizer::from_bytes(
            &serde_json::to_vec(&serde_json::json!({
                "version": "1.0",
                "model": {"type": "BPE", "vocab": vocab, "merges": []}
            }))
            .unwrap(),
        )
        .unwrap();
        AppState::new_mock(
            model_config,
            scheduler,
            std::sync::Arc::new(engine),
            tokenizer,
            30,
            "student-model".into(),
        )
    }

    fn mock_teacher_state_with_base_identity() -> AppState {
        let mut state = mock_teacher_state();
        let identity = crate::teacher_identity::build_base_teacher_identity(
            &state.served_model_id,
            &format!("sha256:{HASH_A}"),
            &state.tokenizer,
            &state.model_config,
            "cpu",
            HASH_B,
            HASH_C,
        )
        .unwrap();
        state.base_teacher_identity = Some(std::sync::Arc::new(identity));
        state
    }

    fn write_minimal_local_adapter(root: &std::path::Path, name: &str) {
        let adapter = root.join(name);
        std::fs::create_dir_all(&adapter).unwrap();
        std::fs::write(
            adapter.join("adapter_config.json"),
            br#"{"r":1,"lora_alpha":1.0,"target_modules":[]}"#,
        )
        .unwrap();
        let tensor_bytes = 1.0f32.to_le_bytes();
        let tensor =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![1], &tensor_bytes)
                .unwrap();
        let bytes =
            safetensors::tensor::serialize([("ignored.weight", tensor)].into_iter(), None).unwrap();
        std::fs::write(adapter.join("adapter_model.safetensors"), bytes).unwrap();
    }

    #[test]
    fn legacy_remote_entry_is_unusable_with_remediation() {
        let spec = TeacherSpec {
            alias: "legacy@remote".into(),
            kind: TeacherKind::Remote,
            provider: Some(kiln_train::RemoteProvider::Vllm),
            model_id: "teacher-model".into(),
            max_top_k: Some(2),
            vocab_size: Some(3),
            supports_full_vocab: Some(false),
            tokenizer_hash: None,
            identity: None,
            url: Some("http://127.0.0.1:8000".into()),
            credential_id: None,
            notes: None,
            adapter: None,
        };
        let entry = teacher_entry(&mock_teacher_state(), spec);
        assert_eq!(entry.status, TeacherStatus::LegacyUnverified);
        assert!(!entry.usable);
        assert!(entry.capabilities.is_none());
        assert!(
            entry
                .status_message
                .as_deref()
                .unwrap()
                .contains("no authoritative identity")
        );
    }

    #[test]
    fn legacy_local_entry_is_unusable_instead_of_resolving_a_mutable_name() {
        let spec = TeacherSpec {
            alias: "legacy@local".into(),
            kind: TeacherKind::Local,
            provider: None,
            model_id: "caller-claim".into(),
            max_top_k: None,
            vocab_size: None,
            supports_full_vocab: None,
            tokenizer_hash: None,
            identity: None,
            url: None,
            credential_id: None,
            notes: None,
            adapter: None,
        };
        let entry = teacher_entry(&mock_teacher_state_with_base_identity(), spec);
        assert_eq!(entry.status, TeacherStatus::LegacyUnverified);
        assert!(!entry.usable);
        assert!(
            entry
                .status_message
                .as_deref()
                .unwrap()
                .contains("no authoritative identity")
        );
    }

    #[tokio::test]
    async fn local_registration_derives_base_and_adapter_identity_server_side() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_local_adapter(dir.path(), "prior-self");
        let mut state = mock_teacher_state_with_base_identity();
        state.adapter_dir = dir.path().to_path_buf();
        let base = state.base_teacher_identity.as_ref().unwrap().clone();

        let Json(entry) = register_teacher(
            State(state),
            Json(serde_json::json!({
                "alias": "prior@local",
                "kind": "local",
                "adapter": "prior-self"
            })),
        )
        .await
        .unwrap();

        assert_eq!(entry.status, TeacherStatus::Verified);
        assert!(entry.usable);
        assert_eq!(entry.spec.model_id, base.served_model_id());
        let identity = entry.spec.identity.unwrap();
        assert_eq!(identity.base_model_sha256(), base.base_model_sha256());
        assert_eq!(identity.adapter().unwrap().name(), "prior-self");
        assert_eq!(
            entry.identity_revision.unwrap(),
            format!("sha256:{}", identity.content_revision())
        );
    }

    #[tokio::test]
    async fn bare_local_registration_does_not_accept_a_caller_model_claim() {
        let dir = tempfile::tempdir().unwrap();
        let mut state = mock_teacher_state_with_base_identity();
        state.adapter_dir = dir.path().to_path_buf();
        let base = state.base_teacher_identity.as_ref().unwrap().clone();

        let Json(entry) = register_teacher(
            State(state),
            Json(serde_json::json!({
                "alias": "base@local",
                "kind": "local",
                "model_id": "untrusted-caller-claim"
            })),
        )
        .await
        .unwrap();
        assert_eq!(entry.spec.model_id, base.served_model_id());
        assert_eq!(entry.spec.identity.as_ref(), Some(base.as_ref()));
    }

    #[test]
    fn remote_probe_admission_is_fail_fast_and_timeout_is_bounded() {
        let semaphore = std::sync::Arc::new(Semaphore::new(2));
        let first = try_teacher_identity_probe_permit_from(semaphore.clone()).unwrap();
        let second = try_teacher_identity_probe_permit_from(semaphore.clone()).unwrap();
        let error = try_teacher_identity_probe_permit_from(semaphore.clone()).unwrap_err();
        assert_eq!(error.code, "teacher_identity_probe_busy");
        assert_eq!(error.status, axum::http::StatusCode::SERVICE_UNAVAILABLE);
        drop(first);
        assert!(try_teacher_identity_probe_permit_from(semaphore).is_ok());
        drop(second);

        let spec = TeacherSpec {
            alias: "remote".into(),
            kind: TeacherKind::Remote,
            provider: Some(kiln_train::RemoteProvider::Vllm),
            model_id: "model".into(),
            max_top_k: None,
            vocab_size: None,
            supports_full_vocab: None,
            tokenizer_hash: None,
            identity: None,
            url: Some("http://127.0.0.1:8000".into()),
            credential_id: None,
            notes: None,
            adapter: None,
        };
        let config =
            remote_teacher_config(&spec, &crate::config::TeachersConfig::default()).unwrap();
        assert_eq!(config.timeout_ms, 60_000);
        assert_eq!(
            registration_probe_config(config).timeout_ms,
            REGISTRATION_REMOTE_PROBE_TIMEOUT_MS
        );
        assert!(REGISTRATION_REMOTE_PROBE_TIMEOUT_MS < 60_000);
    }

    #[test]
    fn credentialless_loopback_teacher_is_rejected_when_kiln_is_network_bound() {
        let spec = TeacherSpec {
            alias: "loopback".into(),
            kind: TeacherKind::Remote,
            provider: Some(kiln_train::RemoteProvider::Vllm),
            model_id: "model".into(),
            max_top_k: None,
            vocab_size: None,
            supports_full_vocab: None,
            tokenizer_hash: None,
            identity: None,
            url: Some("http://127.0.0.1:8000".into()),
            credential_id: None,
            notes: None,
            adapter: None,
        };
        assert!(!remote_registration_exposure_denied(&spec, true));
        assert!(remote_registration_exposure_denied(&spec, false));

        let mut credentialed = spec;
        credentialed.credential_id = Some("trusted-loopback".into());
        assert!(!remote_registration_exposure_denied(&credentialed, false));
    }

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
            identity: None,
            url: None,
            credential_id: None,
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
            identity: None,
            url: Some("http://127.0.0.1:8080".into()),
            credential_id: None,
            notes: None,
            adapter: None,
        };
        validate_teacher_spec_static(&spec).unwrap();
        spec.provider = Some(kiln_train::RemoteProvider::Sglang);
        let error = validate_teacher_spec_static(&spec).unwrap_err();
        assert!(error.contains("Sglang"), "{error}");

        spec.provider = Some(kiln_train::RemoteProvider::Vllm);
        spec.credential_id = Some(String::new());
        let error = validate_teacher_spec_static(&spec).unwrap_err();
        assert!(error.contains("credential_id"), "{error}");

        spec.kind = TeacherKind::Local;
        spec.provider = None;
        spec.credential_id = None;
        let error = validate_teacher_spec_static(&spec).unwrap_err();
        assert!(error.contains("only valid on kind=remote"), "{error}");
    }

    #[test]
    fn registration_dto_rejects_server_controlled_fields() {
        let base = serde_json::json!({
            "alias": "remote",
            "kind": "remote",
            "provider": "vllm",
            "model_id": "model",
            "url": "http://127.0.0.1:8000"
        });
        for (field, value) in [
            ("api_key_env", serde_json::json!("AWS_SECRET_ACCESS_KEY")),
            ("identity", serde_json::Value::Null),
            ("max_top_k", serde_json::json!(32)),
            ("vocab_size", serde_json::json!(1024)),
            ("supports_full_vocab", serde_json::json!(false)),
            ("tokenizer_hash", serde_json::json!(HASH_A)),
        ] {
            let mut body = base.clone();
            body.as_object_mut()
                .unwrap()
                .insert(field.to_string(), value);
            let error = serde_json::from_value::<RegisterTeacherRequest>(body).unwrap_err();
            assert!(error.to_string().contains(field), "{field}: {error}");
        }
    }

    #[test]
    fn persisted_specs_reject_legacy_caller_controlled_environment_names() {
        let serialized = serde_json::json!({
            "alias": "legacy",
            "kind": "remote",
            "provider": "vllm",
            "model_id": "model",
            "url": "http://127.0.0.1:8000",
            "api_key_env": "AWS_SECRET_ACCESS_KEY"
        });
        let error = serde_json::from_value::<TeacherSpec>(serialized).unwrap_err();
        assert!(error.to_string().contains("api_key_env"), "{error}");
    }

    #[test]
    fn registry_migrates_legacy_secret_names_without_trusting_or_reexposing_them() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("teachers.json");
        let secret_env_name = "AWS_SECRET_ACCESS_KEY_DO_NOT_USE";
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "legacy": {
                    "alias": "legacy",
                    "kind": "remote",
                    "provider": "vllm",
                    "model_id": "teacher-model",
                    "max_top_k": 20,
                    "vocab_size": 1024,
                    "supports_full_vocab": false,
                    "tokenizer_hash": null,
                    "url": "https://teacher.example.com",
                    "api_key_env": secret_env_name,
                    "notes": null,
                    "adapter": null
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let registry = TeacherRegistry::load_from_path(&path);
        let spec = registry.get("legacy").expect("legacy entry is retained");
        assert!(spec.credential_id.is_none());
        assert!(spec.identity.is_none());
        let persisted = std::fs::read_to_string(path).unwrap();
        assert!(!persisted.contains("api_key_env"), "{persisted}");
        assert!(!persisted.contains(secret_env_name), "{persisted}");
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
            identity: None,
            url: None,
            credential_id: None,
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
    fn registry_mutations_publish_only_after_atomic_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("teachers.json");
        let reg = TeacherRegistry::new();
        let spec = TeacherSpec {
            alias: "fixture@test".into(),
            kind: TeacherKind::Fixture,
            provider: None,
            model_id: "fixture".into(),
            max_top_k: Some(4),
            vocab_size: Some(8),
            supports_full_vocab: Some(false),
            tokenizer_hash: None,
            identity: None,
            url: None,
            credential_id: None,
            notes: None,
            adapter: None,
        };
        reg.insert_new_and_save(spec.clone(), &path).unwrap();
        assert_eq!(reg.get(&spec.alias), Some(spec.clone()));
        let persisted = std::fs::read(&path).unwrap();

        let error = reg.insert_new_and_save(spec.clone(), &path).unwrap_err();
        assert!(matches!(error, TeacherRegistryMutationError::Duplicate(_)));
        assert_eq!(std::fs::read(&path).unwrap(), persisted);

        let blocked_parent = dir.path().join("not-a-directory");
        std::fs::write(&blocked_parent, b"block").unwrap();
        let impossible_path = blocked_parent.join("teachers.json");
        assert!(reg.remove_and_save(&spec.alias, &impossible_path).is_err());
        assert_eq!(reg.get(&spec.alias), Some(spec.clone()));
        assert!(reg.remove_and_save(&spec.alias, &path).unwrap());
        assert!(reg.get(&spec.alias).is_none());
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
            identity: Some(remote_identity("y", 50_257, 20)),
            url: Some("https://api".into()),
            credential_id: None,
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
    fn remote_caps_do_not_echo_unverified_registration_claims() {
        let spec = TeacherSpec {
            alias: "legacy".into(),
            kind: TeacherKind::Remote,
            provider: Some(kiln_train::RemoteProvider::Vllm),
            model_id: "claimed".into(),
            max_top_k: Some(65_536),
            vocab_size: Some(16_777_216),
            supports_full_vocab: Some(false),
            tokenizer_hash: Some(HASH_A.into()),
            identity: None,
            url: Some("https://example.invalid".into()),
            credential_id: None,
            notes: None,
            adapter: None,
        };
        assert!(resolve_caps_for(&spec).is_none());
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
            identity: Some(remote_identity("tiny-model", 8, 8)),
            url: Some("https://api".into()),
            credential_id: None,
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
            identity: None,
            url: None,
            credential_id: None,
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
            identity: Some(remote_identity("qwen/qwen-3.6-27b", 1024, 20)),
            url: Some("http://vllm.internal:8000".into()),
            credential_id: Some("primary-vllm".into()),
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
