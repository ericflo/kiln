use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};

use crate::supervisor::SupervisorConfig;

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
            inference_fraction: 0.9,
            fp8_kv_cache: false,
            cuda_graphs: true,
            prefix_cache: true,
            speculative_decoding: false,
            adapter_dir: None,
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
        serde_json::from_str(&data).unwrap_or_default()
    }

    pub fn save(&self, app: &AppHandle) -> Result<(), String> {
        let path = Self::path(app)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create settings dir: {}", e))?;
        }
        let body =
            serde_json::to_string_pretty(self).map_err(|e| format!("serialize: {}", e))?;
        std::fs::write(&path, body).map_err(|e| format!("write settings.json: {}", e))
    }
}

pub fn apply_to_supervisor_config(s: &Settings, cfg: &mut SupervisorConfig) {
    let mut args: Vec<String> = Vec::new();
    args.push("--host".to_string());
    args.push(s.host.clone());
    args.push("--port".to_string());
    args.push(s.port.to_string());
    args.push("--inference-fraction".to_string());
    args.push(format!("{}", s.inference_fraction));

    if s.fp8_kv_cache {
        args.push("--fp8-kv-cache".to_string());
    }
    if s.cuda_graphs {
        args.push("--cuda-graphs".to_string());
    }
    if s.prefix_cache {
        args.push("--prefix-cache".to_string());
    }
    if s.speculative_decoding {
        args.push("--speculative-decoding".to_string());
    }
    if let Some(dir) = &s.adapter_dir {
        args.push("--adapter-dir".to_string());
        args.push(dir.display().to_string());
    }
    if let Some(model) = &s.model_path {
        args.push("--model".to_string());
        args.push(model.display().to_string());
    }

    cfg.args = args;
    cfg.auto_restart = s.auto_restart;
    cfg.host = s.host.clone();
    cfg.port = s.port;
    cfg.binary_path = s
        .kiln_binary
        .clone()
        .unwrap_or_else(|| PathBuf::from("kiln"));
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
        assert!(s.cuda_graphs);
        assert!(s.prefix_cache);
        assert!(!s.speculative_decoding);
        assert!(s.auto_start);
        assert!(s.auto_restart);
        assert!(!s.launch_at_login);
    }

    #[test]
    fn apply_builds_args_with_expected_flags() {
        let mut s = Settings::default();
        s.port = 9000;
        s.host = "0.0.0.0".to_string();
        s.fp8_kv_cache = true;
        s.model_path = Some(PathBuf::from("/models/foo"));
        s.adapter_dir = Some(PathBuf::from("/adapters"));
        s.auto_restart = false;

        let mut cfg = SupervisorConfig::default();
        apply_to_supervisor_config(&s, &mut cfg);

        // Flags should contain the structured args in order.
        assert!(cfg.args.windows(2).any(|w| w == ["--host", "0.0.0.0"]));
        assert!(cfg.args.windows(2).any(|w| w == ["--port", "9000"]));
        // Host/port are also propagated as structured fields for the poller.
        assert_eq!(cfg.host, "0.0.0.0");
        assert_eq!(cfg.port, 9000);
        assert!(cfg.args.iter().any(|a| a == "--fp8-kv-cache"));
        assert!(cfg.args.iter().any(|a| a == "--cuda-graphs"));
        assert!(cfg.args.iter().any(|a| a == "--prefix-cache"));
        assert!(!cfg.args.iter().any(|a| a == "--speculative-decoding"));
        assert!(cfg
            .args
            .windows(2)
            .any(|w| w == ["--model", "/models/foo"]));
        assert!(cfg
            .args
            .windows(2)
            .any(|w| w == ["--adapter-dir", "/adapters"]));
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
        assert!(s.cuda_graphs);
    }
}
