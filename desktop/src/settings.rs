use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tauri::{AppHandle, Manager};

use crate::runtime_defaults::{DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT};
use crate::supervisor::SupervisorConfig;

const DESKTOP_RUNTIME_CONFIG_NAME: &str = "kiln-desktop-runtime.toml";
pub const SETTINGS_SCHEMA_VERSION: u32 = 1;
static SETTINGS_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static SETTINGS_WRITE_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SettingsLoadKind {
    Ok,
    Migrated,
    Partial,
    Recovered,
    Error,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SettingsSource {
    Defaults,
    Primary,
    Backup,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SettingsIssue {
    pub field: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SettingsLoadStatus {
    pub kind: SettingsLoadKind,
    pub source: SettingsSource,
    pub loaded_schema_version: Option<u32>,
    pub current_schema_version: u32,
    pub issues: Vec<SettingsIssue>,
    pub can_save: bool,
    pub auto_start_suppressed: bool,
    pub backup_available: bool,
}

impl SettingsLoadStatus {
    fn fresh() -> Self {
        Self {
            kind: SettingsLoadKind::Ok,
            source: SettingsSource::Defaults,
            loaded_schema_version: None,
            current_schema_version: SETTINGS_SCHEMA_VERSION,
            issues: Vec::new(),
            can_save: true,
            auto_start_suppressed: false,
            backup_available: false,
        }
    }

    pub fn saved() -> Self {
        Self {
            kind: SettingsLoadKind::Ok,
            source: SettingsSource::Primary,
            loaded_schema_version: Some(SETTINGS_SCHEMA_VERSION),
            current_schema_version: SETTINGS_SCHEMA_VERSION,
            issues: Vec::new(),
            can_save: true,
            auto_start_suppressed: false,
            // The next load probes the filesystem for the exact value. A
            // first-ever save has no prior document to retain.
            backup_available: false,
        }
    }

    pub fn has_issues(&self) -> bool {
        !self.issues.is_empty()
    }

    pub fn summary(&self) -> String {
        let prefix = match self.kind {
            SettingsLoadKind::Ok => "Settings loaded.",
            SettingsLoadKind::Migrated => "Legacy settings were migrated in memory.",
            SettingsLoadKind::Partial => {
                "Some settings were invalid; valid fields were preserved and defaults filled the rest."
            }
            SettingsLoadKind::Recovered => {
                "The primary settings file could not be used, so Kiln loaded its backup."
            }
            SettingsLoadKind::Error => {
                "Kiln could not load the settings file or its backup; safe defaults are active."
            }
            SettingsLoadKind::Unsupported => {
                "The settings file was written by a newer Kiln Desktop version."
            }
        };
        match self.issues.first() {
            Some(issue) => format!("{prefix} {}", issue.message),
            None => prefix.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SettingsLoadOutcome {
    pub settings: Settings,
    pub status: SettingsLoadStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub schema_version: u32,
    pub kiln_binary: Option<PathBuf>,
    pub model_path: Option<PathBuf>,
    pub host: String,
    pub port: u16,
    pub inference_fraction: f32,
    pub fp8_kv_cache: bool,
    pub cuda_graphs: bool,
    pub prefix_cache: bool,
    pub speculative_decoding: bool,
    pub adapter_dir: Option<PathBuf>,
    pub served_model_id: Option<String>,
    /// Default maximum number of reasoning tokens before the server forces the
    /// current `<think>` block closed. `None` leaves reasoning unlimited.
    pub default_thinking_budget_tokens: Option<usize>,
    /// Default reasoning wall-clock budget in milliseconds. `None` leaves
    /// reasoning unlimited. Request-level API fields can still override it.
    pub default_thinking_budget_ms: Option<u64>,
    pub auto_start: bool,
    pub auto_restart: bool,
    pub launch_at_login: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            schema_version: SETTINGS_SCHEMA_VERSION,
            kiln_binary: None,
            model_path: None,
            host: DEFAULT_SERVER_HOST.to_string(),
            port: DEFAULT_SERVER_PORT,
            inference_fraction: if cfg!(target_os = "macos") { 0.7 } else { 0.9 },
            fp8_kv_cache: false,
            cuda_graphs: !cfg!(target_os = "macos"),
            prefix_cache: true,
            speculative_decoding: false,
            adapter_dir: None,
            served_model_id: None,
            default_thinking_budget_tokens: None,
            default_thinking_budget_ms: None,
            auto_start: true,
            auto_restart: true,
            launch_at_login: false,
        }
    }
}

impl Settings {
    pub fn path(app: &AppHandle) -> Result<PathBuf, String> {
        let dir = app
            .path()
            .app_config_dir()
            .map_err(|e| format!("app_config_dir unavailable: {}", e))?;
        Ok(dir.join("settings.json"))
    }

    pub fn load(app: &AppHandle) -> SettingsLoadOutcome {
        let path = match Self::path(app) {
            Ok(p) => p,
            Err(error) => {
                return SettingsLoadOutcome {
                    settings: Self::default(),
                    status: SettingsLoadStatus {
                        kind: SettingsLoadKind::Error,
                        source: SettingsSource::Defaults,
                        loaded_schema_version: None,
                        current_schema_version: SETTINGS_SCHEMA_VERSION,
                        issues: vec![SettingsIssue {
                            field: None,
                            message: error,
                        }],
                        can_save: false,
                        auto_start_suppressed: true,
                        backup_available: false,
                    },
                };
            }
        };
        load_settings_from_path(&path)
    }

    pub fn save(&self, app: &AppHandle) -> Result<(), String> {
        let path = Self::path(app)?;
        save_settings_to_path(self, &path)
    }
}

#[derive(Debug)]
struct DecodedSettings {
    settings: Settings,
    source_version: u32,
    migrated: bool,
    issues: Vec<SettingsIssue>,
}

#[derive(Debug)]
enum SettingsCandidate {
    Missing,
    Loaded(DecodedSettings),
    Invalid(String),
    Unsupported(u64),
}

fn load_settings_from_path(path: &Path) -> SettingsLoadOutcome {
    let backup = settings_backup_path(path);
    let backup_available = backup.is_file();
    match read_settings_candidate(path) {
        SettingsCandidate::Loaded(decoded) => {
            outcome_from_primary(decoded, backup_available)
        }
        SettingsCandidate::Missing => match read_settings_candidate(&backup) {
            SettingsCandidate::Missing => SettingsLoadOutcome {
                settings: Settings::default(),
                status: SettingsLoadStatus::fresh(),
            },
            backup_candidate => outcome_from_backup(
                backup_candidate,
                SettingsIssue {
                    field: None,
                    message: "The primary settings file is missing; Kiln tried its backup."
                        .to_string(),
                },
                false,
                backup_available,
            ),
        },
        SettingsCandidate::Invalid(message) => outcome_from_backup(
            read_settings_candidate(&backup),
            SettingsIssue {
                field: None,
                message,
            },
            false,
            backup_available,
        ),
        SettingsCandidate::Unsupported(version) => outcome_from_backup(
            read_settings_candidate(&backup),
            SettingsIssue {
                field: Some("schema_version".to_string()),
                message: format!(
                    "settings.json uses schema version {version}, but this app supports version {SETTINGS_SCHEMA_VERSION}."
                ),
            },
            true,
            backup_available,
        ),
    }
}

fn outcome_from_primary(decoded: DecodedSettings, backup_available: bool) -> SettingsLoadOutcome {
    let damaged = !decoded.issues.is_empty();
    let mut issues = decoded.issues;
    if decoded.migrated {
        issues.push(SettingsIssue {
            field: Some("schema_version".to_string()),
            message: format!(
                "Legacy settings schema {} was migrated in memory; Save will persist schema {}.",
                decoded.source_version, SETTINGS_SCHEMA_VERSION
            ),
        });
    }
    let kind = if damaged {
        SettingsLoadKind::Partial
    } else if decoded.migrated {
        SettingsLoadKind::Migrated
    } else {
        SettingsLoadKind::Ok
    };
    SettingsLoadOutcome {
        settings: normalize_for_platform(decoded.settings),
        status: SettingsLoadStatus {
            kind,
            source: SettingsSource::Primary,
            loaded_schema_version: Some(decoded.source_version),
            current_schema_version: SETTINGS_SCHEMA_VERSION,
            issues,
            can_save: true,
            auto_start_suppressed: damaged,
            backup_available,
        },
    }
}

fn outcome_from_backup(
    backup_candidate: SettingsCandidate,
    primary_issue: SettingsIssue,
    primary_unsupported: bool,
    backup_available: bool,
) -> SettingsLoadOutcome {
    match backup_candidate {
        SettingsCandidate::Loaded(decoded) => {
            let mut issues = vec![primary_issue];
            issues.extend(decoded.issues);
            if decoded.migrated {
                issues.push(SettingsIssue {
                    field: Some("schema_version".to_string()),
                    message: format!(
                        "The backup used legacy schema {}; Save will persist schema {}.",
                        decoded.source_version, SETTINGS_SCHEMA_VERSION
                    ),
                });
            }
            SettingsLoadOutcome {
                settings: normalize_for_platform(decoded.settings),
                status: SettingsLoadStatus {
                    kind: SettingsLoadKind::Recovered,
                    source: SettingsSource::Backup,
                    loaded_schema_version: Some(decoded.source_version),
                    current_schema_version: SETTINGS_SCHEMA_VERSION,
                    issues,
                    can_save: !primary_unsupported,
                    auto_start_suppressed: true,
                    backup_available,
                },
            }
        }
        SettingsCandidate::Missing => defaults_after_load_failure(
            vec![
                primary_issue,
                SettingsIssue {
                    field: None,
                    message: "No settings backup is available.".to_string(),
                },
            ],
            primary_unsupported,
            backup_available,
        ),
        SettingsCandidate::Invalid(message) => defaults_after_load_failure(
            vec![
                primary_issue,
                SettingsIssue {
                    field: None,
                    message: format!("The settings backup is also unusable: {message}"),
                },
            ],
            primary_unsupported,
            backup_available,
        ),
        SettingsCandidate::Unsupported(version) => defaults_after_load_failure(
            vec![
                primary_issue,
                SettingsIssue {
                    field: Some("schema_version".to_string()),
                    message: format!(
                        "The settings backup uses unsupported schema version {version}."
                    ),
                },
            ],
            true,
            backup_available,
        ),
    }
}

fn defaults_after_load_failure(
    issues: Vec<SettingsIssue>,
    unsupported: bool,
    backup_available: bool,
) -> SettingsLoadOutcome {
    SettingsLoadOutcome {
        settings: Settings::default(),
        status: SettingsLoadStatus {
            kind: if unsupported {
                SettingsLoadKind::Unsupported
            } else {
                SettingsLoadKind::Error
            },
            source: SettingsSource::Defaults,
            loaded_schema_version: None,
            current_schema_version: SETTINGS_SCHEMA_VERSION,
            issues,
            can_save: !unsupported,
            auto_start_suppressed: true,
            backup_available,
        },
    }
}

fn read_settings_candidate(path: &Path) -> SettingsCandidate {
    let data = match std::fs::read_to_string(path) {
        Ok(data) => data,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return SettingsCandidate::Missing;
        }
        Err(error) => {
            return SettingsCandidate::Invalid(format!(
                "Could not read {}: {error}",
                display_file_name(path)
            ));
        }
    };
    match decode_settings_document(&data) {
        Ok(decoded) => SettingsCandidate::Loaded(decoded),
        Err(SettingsDocumentError::Invalid(message)) => SettingsCandidate::Invalid(format!(
            "Could not parse {}: {message}",
            display_file_name(path)
        )),
        Err(SettingsDocumentError::Unsupported(version)) => SettingsCandidate::Unsupported(version),
    }
}

#[derive(Debug)]
enum SettingsDocumentError {
    Invalid(String),
    Unsupported(u64),
}

fn decode_settings_document(data: &str) -> Result<DecodedSettings, SettingsDocumentError> {
    let value: Value = serde_json::from_str(data)
        .map_err(|error| SettingsDocumentError::Invalid(error.to_string()))?;
    let object = value.as_object().ok_or_else(|| {
        SettingsDocumentError::Invalid("the document root must be a JSON object".to_string())
    })?;

    let mut issues = Vec::new();
    let source_version = match object.get("schema_version") {
        None => 0,
        Some(value) => match value.as_u64() {
            Some(version) if version > u32::MAX as u64 => {
                return Err(SettingsDocumentError::Unsupported(version));
            }
            Some(version) => version as u32,
            None => {
                return Err(SettingsDocumentError::Invalid(
                    "`schema_version` must be a nonnegative whole number".to_string(),
                ));
            }
        },
    };
    if source_version > SETTINGS_SCHEMA_VERSION {
        return Err(SettingsDocumentError::Unsupported(source_version as u64));
    }

    let mut settings = Settings::default();
    if let Some(value) =
        decode_field::<Option<PathBuf>>(object, "kiln_binary", "a path string or null", &mut issues)
    {
        settings.kiln_binary = value;
    }
    if let Some(value) =
        decode_field::<Option<PathBuf>>(object, "model_path", "a path string or null", &mut issues)
    {
        settings.model_path = value;
    }
    if let Some(value) = decode_field::<String>(object, "host", "a nonempty string", &mut issues) {
        if value.trim().is_empty() {
            issues.push(invalid_field_issue("host", "a nonempty string"));
        } else {
            settings.host = value.trim().to_string();
        }
    }
    if let Some(value) = decode_field::<u16>(
        object,
        "port",
        "an integer from 1 through 65535",
        &mut issues,
    ) {
        if value == 0 {
            issues.push(invalid_field_issue(
                "port",
                "an integer from 1 through 65535",
            ));
        } else {
            settings.port = value;
        }
    }
    if let Some(value) = decode_field::<f32>(
        object,
        "inference_fraction",
        "a finite number from 0 through 1",
        &mut issues,
    ) {
        if value.is_finite() && (0.0..=1.0).contains(&value) {
            settings.inference_fraction = value;
        } else {
            issues.push(invalid_field_issue(
                "inference_fraction",
                "a finite number from 0 through 1",
            ));
        }
    }

    macro_rules! decode_setting {
        ($field:ident, $type:ty, $expected:literal) => {
            if let Some(value) =
                decode_field::<$type>(object, stringify!($field), $expected, &mut issues)
            {
                settings.$field = value;
            }
        };
    }

    decode_setting!(fp8_kv_cache, bool, "true or false");
    decode_setting!(cuda_graphs, bool, "true or false");
    decode_setting!(prefix_cache, bool, "true or false");
    decode_setting!(speculative_decoding, bool, "true or false");
    decode_setting!(adapter_dir, Option<PathBuf>, "a path string or null");
    decode_setting!(served_model_id, Option<String>, "a string or null");
    decode_setting!(
        default_thinking_budget_tokens,
        Option<usize>,
        "a nonnegative whole number or null"
    );
    decode_setting!(
        default_thinking_budget_ms,
        Option<u64>,
        "a nonnegative whole number or null"
    );
    decode_setting!(auto_start, bool, "true or false");
    decode_setting!(auto_restart, bool, "true or false");
    decode_setting!(launch_at_login, bool, "true or false");

    let known_fields = BTreeSet::from([
        "schema_version",
        "kiln_binary",
        "model_path",
        "host",
        "port",
        "inference_fraction",
        "fp8_kv_cache",
        "cuda_graphs",
        "prefix_cache",
        "speculative_decoding",
        "adapter_dir",
        "served_model_id",
        "default_thinking_budget_tokens",
        "default_thinking_budget_ms",
        "auto_start",
        "auto_restart",
        "launch_at_login",
    ]);
    let unknown_fields = object
        .keys()
        .filter(|key| !known_fields.contains(key.as_str()))
        .cloned()
        .collect::<BTreeSet<_>>();
    if !unknown_fields.is_empty() {
        issues.push(SettingsIssue {
            field: None,
            message: format!(
                "Unknown settings fields were ignored: {}.",
                unknown_fields.into_iter().collect::<Vec<_>>().join(", ")
            ),
        });
    }

    settings.schema_version = SETTINGS_SCHEMA_VERSION;
    Ok(DecodedSettings {
        settings,
        source_version,
        migrated: source_version < SETTINGS_SCHEMA_VERSION,
        issues,
    })
}

fn decode_field<T: DeserializeOwned>(
    object: &Map<String, Value>,
    field: &str,
    expected: &str,
    issues: &mut Vec<SettingsIssue>,
) -> Option<T> {
    let value = object.get(field)?;
    match serde_json::from_value::<T>(value.clone()) {
        Ok(value) => Some(value),
        Err(_) => {
            issues.push(invalid_field_issue(field, expected));
            None
        }
    }
}

fn invalid_field_issue(field: &str, expected: &str) -> SettingsIssue {
    SettingsIssue {
        field: Some(field.to_string()),
        message: format!(
            "Invalid `{field}` value; expected {expected}. The default is active for this field."
        ),
    }
}

fn validate_settings(settings: &Settings) -> Result<(), String> {
    let mut errors = Vec::new();
    if settings.schema_version != SETTINGS_SCHEMA_VERSION {
        errors.push(format!("schema_version must be {SETTINGS_SCHEMA_VERSION}"));
    }
    if settings.host.trim().is_empty() {
        errors.push("host must not be empty".to_string());
    }
    if settings.port == 0 {
        errors.push("port must be between 1 and 65535".to_string());
    }
    if !settings.inference_fraction.is_finite()
        || !(0.0..=1.0).contains(&settings.inference_fraction)
    {
        errors.push("inference_fraction must be between 0 and 1".to_string());
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!("invalid settings: {}", errors.join("; ")))
    }
}

fn save_settings_to_path(settings: &Settings, path: &Path) -> Result<(), String> {
    let _write_guard = SETTINGS_WRITE_LOCK
        .lock()
        .map_err(|_| "settings write lock is poisoned".to_string())?;
    let mut persisted = normalize_for_platform(settings.clone());
    persisted.schema_version = SETTINGS_SCHEMA_VERSION;
    persisted.host = persisted.host.trim().to_string();
    validate_settings(&persisted)?;
    let mut body =
        serde_json::to_vec_pretty(&persisted).map_err(|error| format!("serialize: {error}"))?;
    body.push(b'\n');
    atomic_replace_settings(path, &body)
}

fn atomic_replace_settings(path: &Path, body: &[u8]) -> Result<(), String> {
    atomic_replace_settings_with(path, body, |source, destination| {
        std::fs::rename(source, destination)
    })
}

fn atomic_replace_settings_with<F>(path: &Path, body: &[u8], promote: F) -> Result<(), String>
where
    F: FnOnce(&Path, &Path) -> std::io::Result<()>,
{
    let parent = path
        .parent()
        .ok_or_else(|| "settings path has no parent directory".to_string())?;
    std::fs::create_dir_all(parent)
        .map_err(|error| format!("create settings directory: {error}"))?;
    let temp = write_settings_temp_file(path, body)?;

    let mut displaced = None;
    if path_entry_exists(path) {
        let destination = match read_settings_candidate(path) {
            SettingsCandidate::Loaded(_) => settings_backup_path(path),
            _ => settings_invalid_path(path),
        };
        if path_entry_exists(&destination) {
            if let Err(error) = std::fs::remove_file(&destination) {
                let _ = std::fs::remove_file(&temp);
                return Err(format!(
                    "remove stale {}: {error}",
                    display_file_name(&destination)
                ));
            }
        }
        if let Err(error) = std::fs::rename(path, &destination) {
            let _ = std::fs::remove_file(&temp);
            return Err(format!(
                "preserve prior settings as {}: {error}",
                display_file_name(&destination)
            ));
        }
        displaced = Some(destination);
        sync_settings_directory(parent);
    }

    if let Err(error) = promote(&temp, path) {
        let rollback = displaced
            .as_ref()
            .map(|prior| std::fs::rename(prior, path))
            .transpose();
        let _ = std::fs::remove_file(&temp);
        return match rollback {
            Ok(_) => Err(format!(
                "promote staged settings: {error}; restored the prior settings"
            )),
            Err(rollback_error) => Err(format!(
                "promote staged settings: {error}; restoring the prior settings also failed: {rollback_error}"
            )),
        };
    }
    sync_settings_directory(parent);
    Ok(())
}

fn write_settings_temp_file(path: &Path, body: &[u8]) -> Result<PathBuf, String> {
    for _ in 0..32 {
        let sequence = SETTINGS_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let thread = format!("{:?}", std::thread::current().id())
            .chars()
            .filter(|character| character.is_ascii_alphanumeric())
            .collect::<String>();
        let temp = sibling_path(
            path,
            &format!(".{}.{}.{}.tmp", std::process::id(), thread, sequence),
        );
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = match options.open(&temp) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("create staged settings: {error}")),
        };
        if let Err(error) = file.write_all(body).and_then(|_| file.sync_all()) {
            drop(file);
            let _ = std::fs::remove_file(&temp);
            return Err(format!("write staged settings: {error}"));
        }
        drop(file);
        return Ok(temp);
    }
    Err("could not allocate a unique staged settings file".to_string())
}

fn settings_backup_path(path: &Path) -> PathBuf {
    sibling_path(path, ".bak")
}

fn settings_invalid_path(path: &Path) -> PathBuf {
    sibling_path(path, ".invalid")
}

fn sibling_path(path: &Path, suffix: &str) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("settings.json");
    path.with_file_name(format!("{file_name}{suffix}"))
}

fn display_file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("settings file")
        .to_string()
}

fn path_entry_exists(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok()
}

fn sync_settings_directory(path: &Path) {
    if let Ok(directory) = std::fs::File::open(path) {
        let _ = directory.sync_all();
    }
}

pub fn normalize_for_platform(s: Settings) -> Settings {
    #[cfg(target_os = "macos")]
    {
        let mut s = s;
        // These desktop toggles are CUDA-only today. Keep persisted settings
        // aligned with the actual macOS launch contract instead of storing
        // values the child process will ignore or internally override.
        s.fp8_kv_cache = false;
        s.cuda_graphs = false;
        s
    }
    #[cfg(not(target_os = "macos"))]
    {
        s
    }
}

/// Ensure the desktop always launches kiln with a config file under the app's
/// config directory so ambient `KILN_CONFIG` / `./kiln.toml` never alter the
/// server's behavior.
pub fn apply_desktop_launch_contract(
    app: &AppHandle,
    cfg: &mut SupervisorConfig,
) -> Result<(), String> {
    let path = desktop_runtime_config_path(app)?;
    ensure_desktop_runtime_config(&path)?;
    upsert_env(&mut cfg.envs, "KILN_CONFIG", path.display().to_string());
    Ok(())
}

fn desktop_runtime_config_path(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_config_dir()
        .map_err(|e| format!("app_config_dir unavailable: {}", e))?;
    Ok(dir.join(DESKTOP_RUNTIME_CONFIG_NAME))
}

fn ensure_desktop_runtime_config(path: &PathBuf) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create runtime config dir: {}", e))?;
    }
    if path.exists() {
        return Ok(());
    }
    let body = concat!(
        "# Managed by Kiln Desktop.\n",
        "# The desktop app injects runtime overrides via KILN_* env vars.\n",
        "# Keeping this file explicit prevents cwd-local kiln.toml files from\n",
        "# silently affecting the desktop child process.\n",
    );
    std::fs::write(path, body).map_err(|e| format!("write runtime config: {}", e))
}

fn upsert_env(envs: &mut Vec<(String, String)>, key: &str, value: String) {
    if let Some((_, existing)) = envs.iter_mut().find(|(name, _)| name == key) {
        *existing = value;
    } else {
        envs.push((key.to_string(), value));
    }
}

/// Translate desktop Settings into the env-var overrides the kiln server
/// recognizes (see `crates/kiln-server/src/config.rs::apply_env_overrides`).
/// kiln's CLI surface is `--config <toml>` plus subcommands; per-setting
/// overrides go through `KILN_*` env vars, not flags. Pre-0.1.5 the desktop
/// passed CLI flags that the server didn't accept, which made the supervisor
/// crashloop with "unexpected argument '--host'".
pub fn apply_to_supervisor_config(s: &Settings, cfg: &mut SupervisorConfig) {
    let fp8_kv_cache = if cfg!(target_os = "macos") {
        false
    } else {
        s.fp8_kv_cache
    };
    let cuda_graphs = if cfg!(target_os = "macos") {
        false
    } else {
        s.cuda_graphs
    };

    let mut envs: Vec<(String, String)> = Vec::new();
    envs.push(("KILN_HOST".to_string(), s.host.clone()));
    envs.push(("KILN_PORT".to_string(), s.port.to_string()));
    envs.push((
        "KILN_INFERENCE_MEMORY_FRACTION".to_string(),
        format!("{}", s.inference_fraction),
    ));
    // Booleans: always emit so the env value is the source of truth (a
    // setting toggled off must be able to override a true default).
    envs.push(("KILN_KV_CACHE_FP8".to_string(), bool_env(fp8_kv_cache)));
    envs.push(("KILN_CUDA_GRAPHS".to_string(), bool_env(cuda_graphs)));
    envs.push((
        "KILN_PREFIX_CACHE_ENABLED".to_string(),
        bool_env(s.prefix_cache),
    ));
    envs.push((
        "KILN_SPEC_ENABLED".to_string(),
        bool_env(s.speculative_decoding),
    ));
    envs.push((
        "KILN_SPEC_METHOD".to_string(),
        if s.speculative_decoding { "mtp" } else { "off" }.to_string(),
    ));
    if let Some(dir) = &s.adapter_dir {
        envs.push(("KILN_ADAPTER_DIR".to_string(), dir.display().to_string()));
    }
    if let Some(model) = &s.model_path {
        envs.push(("KILN_MODEL_PATH".to_string(), model.display().to_string()));
    }
    if let Some(id) = &s.served_model_id {
        let trimmed = id.trim();
        if !trimmed.is_empty() {
            envs.push(("KILN_SERVED_MODEL_ID".to_string(), trimmed.to_string()));
        }
    }
    // Always emit both dimensions. The child inherits the desktop process's
    // environment, so omission would let an ambient KILN_* value override the
    // Unlimited mode selected in the UI. The server accepts `unlimited` as the
    // explicit representation of `None`.
    envs.push((
        "KILN_DEFAULT_THINKING_BUDGET_TOKENS".to_string(),
        optional_limit_env(s.default_thinking_budget_tokens),
    ));
    envs.push((
        "KILN_DEFAULT_THINKING_BUDGET_MS".to_string(),
        optional_limit_env(s.default_thinking_budget_ms),
    ));

    cfg.args = Vec::new();
    cfg.envs = envs;
    cfg.auto_restart = s.auto_restart;
    cfg.host = s.host.clone();
    cfg.port = s.port;
    cfg.binary_path = s
        .kiln_binary
        .clone()
        .unwrap_or_else(|| PathBuf::from("kiln"));
}

fn bool_env(b: bool) -> String {
    if b {
        "1".into()
    } else {
        "0".into()
    }
}

fn optional_limit_env<T: ToString>(value: Option<T>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unlimited".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(label: &str) -> Self {
            let sequence = SETTINGS_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "kiln-desktop-settings-{label}-{}-{sequence}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn settings_path(&self) -> PathBuf {
            self.0.join("settings.json")
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write_settings_document(path: &Path, settings: &Settings) {
        std::fs::write(path, serde_json::to_vec_pretty(settings).unwrap()).unwrap();
    }

    fn read_settings_document(path: &Path) -> Settings {
        decode_settings_document(&std::fs::read_to_string(path).unwrap())
            .unwrap()
            .settings
    }

    #[test]
    fn runtime_defaults_contract_matches_desktop_defaults() {
        let contract: serde_json::Value =
            serde_json::from_str(include_str!("../../contracts/runtime-defaults-v1.json")).unwrap();

        assert_eq!(contract["contract_version"], 1);
        assert_eq!(contract["server"]["bind_host"], DEFAULT_SERVER_HOST);
        assert_eq!(contract["server"]["port"], DEFAULT_SERVER_PORT);
    }

    #[test]
    fn default_values_are_sane() {
        let s = Settings::default();
        assert_eq!(s.schema_version, SETTINGS_SCHEMA_VERSION);
        assert_eq!(s.host, DEFAULT_SERVER_HOST);
        assert_eq!(s.port, DEFAULT_SERVER_PORT);
        let expected_fraction = if cfg!(target_os = "macos") { 0.7 } else { 0.9 };
        assert!((s.inference_fraction - expected_fraction).abs() < f32::EPSILON);
        assert!(!s.fp8_kv_cache);
        assert_eq!(s.cuda_graphs, !cfg!(target_os = "macos"));
        assert!(s.prefix_cache);
        assert!(!s.speculative_decoding);
        assert_eq!(s.default_thinking_budget_tokens, None);
        assert_eq!(s.default_thinking_budget_ms, None);
        assert!(s.auto_start);
        assert!(s.auto_restart);
        assert!(!s.launch_at_login);

        let supervisor = SupervisorConfig::default();
        assert_eq!(supervisor.host, DEFAULT_SERVER_HOST);
        assert_eq!(supervisor.port, DEFAULT_SERVER_PORT);
    }

    #[test]
    fn apply_populates_env_vars_for_kiln_server() {
        let mut s = Settings::default();
        s.port = 9000;
        s.host = "0.0.0.0".to_string();
        s.fp8_kv_cache = true;
        s.model_path = Some(PathBuf::from("/models/foo"));
        s.adapter_dir = Some(PathBuf::from("/adapters"));
        s.served_model_id = Some("custom-id".to_string());
        s.default_thinking_budget_tokens = Some(0);
        s.default_thinking_budget_ms = Some(1_500);
        s.auto_restart = false;

        let mut cfg = SupervisorConfig::default();
        apply_to_supervisor_config(&s, &mut cfg);

        // No CLI flags — kiln's CLI doesn't accept per-setting overrides.
        assert!(cfg.args.is_empty(), "args should be empty: {:?}", cfg.args);

        // Settings flow through KILN_* env vars (see kiln-server config.rs).
        let env_get = |k: &str| -> Option<&str> {
            cfg.envs
                .iter()
                .find(|(name, _)| name == k)
                .map(|(_, v)| v.as_str())
        };
        assert_eq!(env_get("KILN_HOST"), Some("0.0.0.0"));
        assert_eq!(env_get("KILN_PORT"), Some("9000"));
        assert_eq!(
            env_get("KILN_KV_CACHE_FP8"),
            Some(if cfg!(target_os = "macos") { "0" } else { "1" })
        );
        assert_eq!(
            env_get("KILN_CUDA_GRAPHS"),
            Some(if cfg!(target_os = "macos") { "0" } else { "1" })
        );
        assert_eq!(env_get("KILN_PREFIX_CACHE_ENABLED"), Some("1")); // default true
        assert_eq!(
            env_get("KILN_SPEC_ENABLED"),
            Some(if cfg!(target_os = "macos") { "1" } else { "0" })
        );
        assert_eq!(
            env_get("KILN_SPEC_METHOD"),
            Some(if cfg!(target_os = "macos") {
                "mtp"
            } else {
                "off"
            })
        );
        assert_eq!(env_get("KILN_MODEL_PATH"), Some("/models/foo"));
        assert_eq!(env_get("KILN_ADAPTER_DIR"), Some("/adapters"));
        assert_eq!(env_get("KILN_SERVED_MODEL_ID"), Some("custom-id"));
        assert_eq!(env_get("KILN_DEFAULT_THINKING_BUDGET_TOKENS"), Some("0"));
        assert_eq!(env_get("KILN_DEFAULT_THINKING_BUDGET_MS"), Some("1500"));

        // Host/port also propagated as structured fields for the poller.
        assert_eq!(cfg.host, "0.0.0.0");
        assert_eq!(cfg.port, 9000);
        assert!(!cfg.auto_restart);
    }

    #[test]
    fn kiln_binary_defaults_to_path_lookup() {
        let s = Settings::default();
        let mut cfg = SupervisorConfig::default();
        apply_to_supervisor_config(&s, &mut cfg);
        assert_eq!(cfg.binary_path, PathBuf::from("kiln"));
    }

    #[test]
    fn kiln_binary_override_propagates() {
        let mut s = Settings::default();
        s.kiln_binary = Some(PathBuf::from("/opt/kiln/bin/kiln"));
        let mut cfg = SupervisorConfig::default();
        apply_to_supervisor_config(&s, &mut cfg);
        assert_eq!(cfg.binary_path, PathBuf::from("/opt/kiln/bin/kiln"));
    }

    #[test]
    fn roundtrip_json() {
        let s = Settings {
            default_thinking_budget_tokens: Some(0),
            default_thinking_budget_ms: Some(1_250),
            ..Settings::default()
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(back.schema_version, SETTINGS_SCHEMA_VERSION);
        assert_eq!(back.port, s.port);
        assert_eq!(back.host, s.host);
        assert_eq!(back.default_thinking_budget_tokens, Some(0));
        assert_eq!(back.default_thinking_budget_ms, Some(1_250));
    }

    #[test]
    fn missing_fields_fall_back_to_defaults() {
        let partial = r#"{ "port": 9001 }"#;
        let s: Settings = serde_json::from_str(partial).unwrap();
        assert_eq!(s.port, 9001);
        assert_eq!(s.host, "127.0.0.1");
        assert_eq!(s.cuda_graphs, !cfg!(target_os = "macos"));
        assert_eq!(s.default_thinking_budget_tokens, None);
        assert_eq!(s.default_thinking_budget_ms, None);
    }

    #[test]
    fn unlimited_thinking_budgets_override_ambient_env() {
        let s = Settings::default();
        let mut cfg = SupervisorConfig::default();
        apply_to_supervisor_config(&s, &mut cfg);

        let env_get = |key: &str| {
            cfg.envs
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str())
        };
        assert_eq!(
            env_get("KILN_DEFAULT_THINKING_BUDGET_TOKENS"),
            Some("unlimited")
        );
        assert_eq!(
            env_get("KILN_DEFAULT_THINKING_BUDGET_MS"),
            Some("unlimited")
        );
    }

    #[test]
    fn legacy_settings_migrate_to_unlimited_thinking_budgets() {
        let legacy = r#"{
            "host": "0.0.0.0",
            "port": 8420,
            "auto_start": false
        }"#;
        let decoded = decode_settings_document(legacy).unwrap();
        let s = decoded.settings;

        assert_eq!(decoded.source_version, 0);
        assert!(decoded.migrated);
        assert_eq!(s.schema_version, SETTINGS_SCHEMA_VERSION);
        assert_eq!(s.host, "0.0.0.0");
        assert_eq!(s.port, 8420);
        assert!(!s.auto_start);
        assert_eq!(s.default_thinking_budget_tokens, None);
        assert_eq!(s.default_thinking_budget_ms, None);
    }

    #[test]
    fn thinking_budget_fields_reject_negative_and_fractional_json_numbers() {
        assert!(
            serde_json::from_str::<Settings>(r#"{"default_thinking_budget_tokens":-1}"#).is_err()
        );
        assert!(
            serde_json::from_str::<Settings>(r#"{"default_thinking_budget_tokens":1.5}"#).is_err()
        );
        assert!(serde_json::from_str::<Settings>(r#"{"default_thinking_budget_ms":-1}"#).is_err());
        assert!(serde_json::from_str::<Settings>(r#"{"default_thinking_budget_ms":0.5}"#).is_err());
    }

    #[test]
    fn settings_ui_exposes_budget_mode_and_exact_millisecond_conversion() {
        let html = include_str!("../ui/settings.html");
        assert!(html.contains("data-budget-mode=\"unlimited\""));
        assert!(html.contains("data-budget-mode=\"custom\""));
        assert!(html.contains("id=\"default_thinking_budget_tokens\""));
        assert!(html.contains("id=\"default_thinking_budget_seconds\""));
        assert!(html.contains("default_thinking_budget_ms: milliseconds"));
        assert!(html.contains("function strictThinkingBudgetInteger(raw)"));
        assert!(html.contains("function strictThinkingBudgetMilliseconds(raw)"));
        assert!(html.contains("input.validity.badInput"));
        assert!(!html.contains("Math.round(milliseconds)"));
    }

    #[test]
    fn normalize_for_platform_forces_cuda_only_toggles_off_on_macos() {
        let mut s = Settings::default();
        s.fp8_kv_cache = true;
        s.cuda_graphs = true;
        let normalized = normalize_for_platform(s);
        assert_eq!(normalized.fp8_kv_cache, !cfg!(target_os = "macos"));
        assert_eq!(normalized.cuda_graphs, !cfg!(target_os = "macos"));
    }

    #[test]
    fn malformed_field_preserves_every_other_valid_field() {
        let dir = TestDirectory::new("partial");
        let path = dir.settings_path();
        std::fs::write(
            &path,
            r#"{
                "schema_version": 1,
                "host": "0.0.0.0",
                "port": "not-a-port",
                "auto_start": false,
                "prefix_cache": false
            }"#,
        )
        .unwrap();

        let loaded = load_settings_from_path(&path);
        assert_eq!(loaded.settings.host, "0.0.0.0");
        assert_eq!(loaded.settings.port, DEFAULT_SERVER_PORT);
        assert!(!loaded.settings.auto_start);
        assert!(!loaded.settings.prefix_cache);
        assert_eq!(loaded.status.kind, SettingsLoadKind::Partial);
        assert_eq!(loaded.status.source, SettingsSource::Primary);
        assert!(loaded.status.auto_start_suppressed);
        assert!(loaded.status.can_save);
        assert!(loaded.status.issues.iter().any(|issue| {
            issue.field.as_deref() == Some("port") && issue.message.contains("default is active")
        }));
    }

    #[test]
    fn corrupt_primary_recovers_backup_without_auto_starting() {
        let dir = TestDirectory::new("recover");
        let path = dir.settings_path();
        let backup = settings_backup_path(&path);
        std::fs::write(&path, b"{ definitely not json").unwrap();
        let settings = Settings {
            port: 9001,
            auto_start: false,
            ..Settings::default()
        };
        write_settings_document(&backup, &settings);

        let loaded = load_settings_from_path(&path);
        assert_eq!(loaded.settings.port, 9001);
        assert!(!loaded.settings.auto_start);
        assert_eq!(loaded.status.kind, SettingsLoadKind::Recovered);
        assert_eq!(loaded.status.source, SettingsSource::Backup);
        assert!(loaded.status.backup_available);
        assert!(loaded.status.auto_start_suppressed);
        assert!(loaded.status.can_save);
    }

    #[test]
    fn corrupt_settings_without_backup_are_visible_and_repairable() {
        let dir = TestDirectory::new("corrupt");
        let path = dir.settings_path();
        std::fs::write(&path, b"[").unwrap();

        let loaded = load_settings_from_path(&path);
        assert_eq!(loaded.settings, Settings::default());
        assert_eq!(loaded.status.kind, SettingsLoadKind::Error);
        assert_eq!(loaded.status.source, SettingsSource::Defaults);
        assert!(loaded.status.auto_start_suppressed);
        assert!(loaded.status.can_save);
        assert!(loaded.status.summary().contains("safe defaults"));
    }

    #[test]
    fn future_primary_schema_is_read_only_even_with_usable_backup() {
        let dir = TestDirectory::new("future");
        let path = dir.settings_path();
        std::fs::write(&path, r#"{"schema_version": 2, "port": 9002}"#).unwrap();
        let backup_settings = Settings {
            port: 9001,
            ..Settings::default()
        };
        write_settings_document(&settings_backup_path(&path), &backup_settings);

        let loaded = load_settings_from_path(&path);
        assert_eq!(loaded.settings.port, 9001);
        assert_eq!(loaded.status.kind, SettingsLoadKind::Recovered);
        assert_eq!(loaded.status.source, SettingsSource::Backup);
        assert!(!loaded.status.can_save);
        assert!(loaded.status.auto_start_suppressed);
        assert!(loaded.status.issues.iter().any(|issue| {
            issue.field.as_deref() == Some("schema_version")
                && issue.message.contains("schema version 2")
        }));
    }

    #[test]
    fn malformed_present_schema_version_is_not_treated_as_legacy() {
        let dir = TestDirectory::new("malformed-version");
        let path = dir.settings_path();
        std::fs::write(&path, r#"{"schema_version": "future", "port": 9002}"#).unwrap();

        let loaded = load_settings_from_path(&path);
        assert_eq!(loaded.settings, Settings::default());
        assert_eq!(loaded.status.kind, SettingsLoadKind::Error);
        assert!(loaded.status.auto_start_suppressed);
        assert!(loaded.status.issues.iter().any(|issue| {
            issue.message.contains("schema_version")
                && issue.message.contains("nonnegative whole number")
        }));
    }

    #[test]
    fn atomic_save_versions_document_and_keeps_previous_backup() {
        let dir = TestDirectory::new("atomic");
        let path = dir.settings_path();
        let first = Settings {
            port: 9001,
            ..Settings::default()
        };
        save_settings_to_path(&first, &path).unwrap();
        assert!(!settings_backup_path(&path).exists());

        let second = Settings {
            port: 9002,
            ..Settings::default()
        };
        save_settings_to_path(&second, &path).unwrap();

        assert_eq!(read_settings_document(&path).port, 9002);
        assert_eq!(
            read_settings_document(&settings_backup_path(&path)).port,
            9001
        );
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.contains(r#""schema_version": 1"#));
        assert!(raw.ends_with('\n'));
        assert!(!std::fs::read_dir(&dir.0)
            .unwrap()
            .filter_map(Result::ok)
            .any(|entry| entry.file_name().to_string_lossy().ends_with(".tmp")));
    }

    #[cfg(unix)]
    #[test]
    fn atomic_save_uses_private_unix_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let dir = TestDirectory::new("permissions");
        let path = dir.settings_path();
        save_settings_to_path(&Settings::default(), &path).unwrap();

        let mode = std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }

    #[test]
    fn repairing_corrupt_primary_preserves_it_without_clobbering_backup() {
        let dir = TestDirectory::new("preserve-corrupt");
        let path = dir.settings_path();
        let backup = settings_backup_path(&path);
        std::fs::write(&path, b"{broken").unwrap();
        let prior = Settings {
            port: 9001,
            ..Settings::default()
        };
        write_settings_document(&backup, &prior);
        let repaired = Settings {
            port: 9002,
            ..Settings::default()
        };

        save_settings_to_path(&repaired, &path).unwrap();

        assert_eq!(read_settings_document(&path).port, 9002);
        assert_eq!(read_settings_document(&backup).port, 9001);
        assert_eq!(
            std::fs::read_to_string(settings_invalid_path(&path)).unwrap(),
            "{broken"
        );
    }

    #[test]
    fn failed_atomic_promotion_restores_primary_and_removes_temp() {
        let dir = TestDirectory::new("rollback");
        let path = dir.settings_path();
        let original = Settings {
            port: 9001,
            ..Settings::default()
        };
        write_settings_document(&path, &original);
        let replacement = serde_json::to_vec_pretty(&Settings {
            port: 9002,
            ..Settings::default()
        })
        .unwrap();

        let error = atomic_replace_settings_with(&path, &replacement, |_, _| {
            Err(std::io::Error::other("injected promotion failure"))
        })
        .unwrap_err();

        assert!(error.contains("restored the prior settings"));
        assert_eq!(read_settings_document(&path).port, 9001);
        assert!(!settings_backup_path(&path).exists());
        assert!(!std::fs::read_dir(&dir.0)
            .unwrap()
            .filter_map(Result::ok)
            .any(|entry| entry.file_name().to_string_lossy().ends_with(".tmp")));
    }

    #[test]
    fn save_rejects_semantically_invalid_values_before_touching_disk() {
        let dir = TestDirectory::new("validation");
        let path = dir.settings_path();
        let invalid = Settings {
            port: 0,
            inference_fraction: 1.5,
            ..Settings::default()
        };

        let error = save_settings_to_path(&invalid, &path).unwrap_err();
        assert!(error.contains("port must be between 1 and 65535"));
        assert!(error.contains("inference_fraction must be between 0 and 1"));
        assert!(!path.exists());
    }
}
