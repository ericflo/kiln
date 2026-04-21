use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};

use crate::supervisor::SupervisorConfig;

const DESKTOP_RUNTIME_CONFIG_NAME: &str = "kiln-desktop-runtime.toml";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
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
    pub auto_start: bool,
    pub auto_restart: bool,
    pub launch_at_login: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            kiln_binary: None,
            model_path: None,
            host: "127.0.0.1".to_string(),
            port: 8000,
            inference_fraction: if cfg!(target_os = "macos") { 0.7 } else { 0.9 },
            fp8_kv_cache: false,
            cuda_graphs: !cfg!(target_os = "macos"),
            prefix_cache: true,
            speculative_decoding: false,
            adapter_dir: None,
            served_model_id: None,
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

    pub fn load(app: &AppHandle) -> Self {
        let path = match Self::path(app) {
            Ok(p) => p,
            Err(_) => return Self::default(),
        };
        let Ok(data) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        normalize_for_platform(serde_json::from_str(&data).unwrap_or_default())
    }

    pub fn save(&self, app: &AppHandle) -> Result<(), String> {
        let path = Self::path(app)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("create settings dir: {}", e))?;
        }
        let body = serde_json::to_string_pretty(self).map_err(|e| format!("serialize: {}", e))?;
        std::fs::write(&path, body).map_err(|e| format!("write settings.json: {}", e))
    }
}

pub fn normalize_for_platform(mut s: Settings) -> Settings {
    #[cfg(target_os = "macos")]
    {
        // These desktop toggles are CUDA-only today. Keep persisted settings
        // aligned with the actual macOS launch contract instead of storing
        // values the child process will ignore or internally override.
        s.fp8_kv_cache = false;
        s.cuda_graphs = false;
    }
    s
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
    if b { "1".into() } else { "0".into() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values_are_sane() {
        let s = Settings::default();
        assert_eq!(s.host, "127.0.0.1");
        assert_eq!(s.port, 8000);
        assert!((s.inference_fraction - 0.9).abs() < f32::EPSILON);
        assert!(!s.fp8_kv_cache);
        assert_eq!(s.cuda_graphs, !cfg!(target_os = "macos"));
        assert!(s.prefix_cache);
        assert!(!s.speculative_decoding);
        assert!(s.auto_start);
        assert!(s.auto_restart);
        assert!(!s.launch_at_login);
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
        assert_eq!(env_get("KILN_SPEC_ENABLED"), Some("0")); // default false
        assert_eq!(env_get("KILN_MODEL_PATH"), Some("/models/foo"));
        assert_eq!(env_get("KILN_ADAPTER_DIR"), Some("/adapters"));
        assert_eq!(env_get("KILN_SERVED_MODEL_ID"), Some("custom-id"));

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
        let s = Settings::default();
        let json = serde_json::to_string(&s).unwrap();
        let back: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(back.port, s.port);
        assert_eq!(back.host, s.host);
    }

    #[test]
    fn missing_fields_fall_back_to_defaults() {
        let partial = r#"{ "port": 9001 }"#;
        let s: Settings = serde_json::from_str(partial).unwrap();
        assert_eq!(s.port, 9001);
        assert_eq!(s.host, "127.0.0.1");
        assert_eq!(s.cuda_graphs, !cfg!(target_os = "macos"));
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
}
