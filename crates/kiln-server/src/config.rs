//! TOML configuration file support with typed validation and env var overrides.
//!
//! Configuration is loaded in this priority order (highest wins):
//! 1. Environment variables (`KILN_*`)
//! 2. TOML config file
//! 3. Built-in defaults
//!
//! The config file path is resolved as:
//! 1. Explicit path passed to `KilnConfig::load()`
//! 2. `KILN_CONFIG` environment variable
//! 3. `./kiln.toml` in the current working directory (if it exists)
//! 4. No file — use defaults only

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// Top-level configuration for kiln.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct KilnConfig {
    pub server: ServerConfig,
    pub model: ModelConfig,
    pub memory: MemoryConfig,
    pub training: TrainingConfig,
    pub logging: LoggingConfig,
    pub prefix_cache: PrefixCacheConfig,
    pub speculative: SpeculativeDecodingConfig,
}

/// HTTP server settings.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub request_timeout_secs: u64,
    pub shutdown_timeout_secs: u64,
}

/// Model and tokenizer paths.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    pub path: Option<String>,
    pub model_id: String,
    pub tokenizer_path: Option<String>,
    pub adapter_dir: Option<String>,
}

/// GPU memory allocation settings.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    pub num_blocks: Option<usize>,
    pub gpu_memory_gb: Option<f64>,
    pub inference_memory_fraction: f64,
    pub training_memory_gb: Option<f64>,
    /// Enable FP8 (E4M3FN) quantization for KV cache, halving memory usage.
    /// When enabled, K/V values are stored as 8-bit floats with per-tensor scaling.
    /// Default: false
    pub kv_cache_fp8: bool,
    /// Enable CUDA graph capture/replay for decode steps.
    /// Eliminates per-step kernel launch overhead for ~10-15% decode speedup.
    /// Automatically disabled on non-CUDA devices.
    /// Default: true
    pub cuda_graphs: bool,
}

/// Training-specific settings.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct TrainingConfig {
    pub grad_checkpoint_segments: Option<usize>,
    pub no_grad_checkpoint: bool,
    /// Save adapter weights every N training steps during a job.
    /// Per-job config overrides this. None = only save at the end.
    pub checkpoint_interval: Option<usize>,
}

/// Logging settings.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
}

/// Prefix caching settings.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct PrefixCacheConfig {
    /// Enable prefix caching for shared prompt prefixes (default: true).
    /// When enabled, KV cache blocks for shared prefixes are reused across requests.
    pub enabled: bool,
    /// Maximum number of KV cache blocks the prefix cache may retain.
    /// Default: 25% of total blocks. Set to 0 to use the default.
    pub max_blocks: Option<usize>,
}

/// Speculative decoding settings.
/// Uses self-speculative (skip-layer) approach: the first N layers of the main
/// model act as a lightweight draft, avoiding the need to load a second model.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct SpeculativeDecodingConfig {
    /// Enable speculative decoding (default: false).
    pub enabled: bool,
    /// Number of tokens the draft proposes per step (default: 4).
    pub num_speculative_tokens: usize,
    /// Number of layers to use for the draft model (default: 8).
    /// Uses the first N layers of the main model as the draft.
    pub draft_layers: usize,
}

// --- Defaults ---

impl Default for KilnConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            model: ModelConfig::default(),
            memory: MemoryConfig::default(),
            training: TrainingConfig::default(),
            logging: LoggingConfig::default(),
            prefix_cache: PrefixCacheConfig::default(),
            speculative: SpeculativeDecodingConfig::default(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".into(),
            port: 8420,
            request_timeout_secs: 300,
            shutdown_timeout_secs: 30,
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            path: None,
            model_id: "Qwen/Qwen3.5-4B".into(),
            tokenizer_path: None,
            adapter_dir: None,
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            num_blocks: None,
            gpu_memory_gb: None,
            inference_memory_fraction: 0.7,
            training_memory_gb: None,
            kv_cache_fp8: false,
            cuda_graphs: true,
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            grad_checkpoint_segments: None,
            no_grad_checkpoint: false,
            checkpoint_interval: None,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".into(),
            format: "json".into(),
        }
    }
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_blocks: None,
        }
    }
}

impl Default for SpeculativeDecodingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            num_speculative_tokens: 4,
            draft_layers: 8,
        }
    }
}

// --- Loading and validation ---

impl KilnConfig {
    /// Load configuration from an optional file path, then apply env var overrides.
    ///
    /// Resolution order for the file path:
    /// 1. `path` argument (if `Some`)
    /// 2. `KILN_CONFIG` env var
    /// 3. `./kiln.toml` (only if it exists)
    /// 4. No file — defaults only
    pub fn load(path: Option<&str>) -> Result<Self> {
        let config_path = path
            .map(String::from)
            .or_else(|| std::env::var("KILN_CONFIG").ok());

        let mut config = if let Some(ref p) = config_path {
            let contents = std::fs::read_to_string(p)
                .with_context(|| format!("failed to read config file: {p}"))?;
            toml::from_str(&contents)
                .with_context(|| format!("failed to parse config file: {p}"))?
        } else if Path::new("kiln.toml").exists() {
            let contents = std::fs::read_to_string("kiln.toml")
                .context("failed to read kiln.toml")?;
            toml::from_str(&contents).context("failed to parse kiln.toml")?
        } else {
            Self::default()
        };

        config.apply_env_overrides();
        config.validate()?;
        Ok(config)
    }

    /// Override config values with KILN_* environment variables (if set).
    fn apply_env_overrides(&mut self) {
        // Server
        if let Ok(v) = std::env::var("KILN_HOST") {
            self.server.host = v;
        }
        if let Ok(v) = std::env::var("KILN_PORT") {
            if let Ok(p) = v.parse() {
                self.server.port = p;
            }
        }
        if let Ok(v) = std::env::var("KILN_REQUEST_TIMEOUT_SECS") {
            if let Ok(s) = v.parse() {
                self.server.request_timeout_secs = s;
            }
        }
        if let Ok(v) = std::env::var("KILN_SHUTDOWN_TIMEOUT_SECS") {
            if let Ok(s) = v.parse() {
                self.server.shutdown_timeout_secs = s;
            }
        }

        // Model
        if let Ok(v) = std::env::var("KILN_MODEL_PATH") {
            self.model.path = Some(v);
        }
        if let Ok(v) = std::env::var("KILN_MODEL_ID") {
            self.model.model_id = v;
        }
        if let Ok(v) = std::env::var("KILN_TOKENIZER_PATH") {
            self.model.tokenizer_path = Some(v);
        }
        if let Ok(v) = std::env::var("KILN_ADAPTER_DIR") {
            self.model.adapter_dir = Some(v);
        }

        // Memory
        if let Ok(v) = std::env::var("KILN_NUM_BLOCKS") {
            if let Ok(n) = v.parse() {
                self.memory.num_blocks = Some(n);
            }
        }
        if let Ok(v) = std::env::var("KILN_GPU_MEMORY_GB") {
            if let Ok(g) = v.parse() {
                self.memory.gpu_memory_gb = Some(g);
            }
        }
        if let Ok(v) = std::env::var("KILN_INFERENCE_MEMORY_FRACTION") {
            if let Ok(f) = v.parse::<f64>() {
                self.memory.inference_memory_fraction = f;
            }
        }
        if let Ok(v) = std::env::var("KILN_TRAINING_MEMORY_GB") {
            if let Ok(g) = v.parse() {
                self.memory.training_memory_gb = Some(g);
            }
        }
        if let Ok(v) = std::env::var("KILN_KV_CACHE_FP8") {
            self.memory.kv_cache_fp8 = v == "1" || v.eq_ignore_ascii_case("true");
        }
        if let Ok(v) = std::env::var("KILN_CUDA_GRAPHS") {
            self.memory.cuda_graphs = v == "1" || v.eq_ignore_ascii_case("true");
        }

        // Training
        if let Ok(v) = std::env::var("KILN_GRAD_CHECKPOINT_SEGMENTS") {
            if let Ok(s) = v.parse() {
                self.training.grad_checkpoint_segments = Some(s);
            }
        }
        if let Ok(v) = std::env::var("KILN_NO_GRAD_CHECKPOINT") {
            self.training.no_grad_checkpoint = v == "1" || v.eq_ignore_ascii_case("true");
        }
        if let Ok(v) = std::env::var("KILN_CHECKPOINT_INTERVAL") {
            if let Ok(n) = v.parse() {
                self.training.checkpoint_interval = Some(n);
            }
        }

        // Logging
        if let Ok(v) = std::env::var("KILN_LOG_LEVEL") {
            self.logging.level = v;
        }
        if let Ok(v) = std::env::var("KILN_LOG_FORMAT") {
            self.logging.format = v;
        }

        // Prefix cache
        if let Ok(v) = std::env::var("KILN_PREFIX_CACHE_ENABLED") {
            self.prefix_cache.enabled = v == "1" || v.eq_ignore_ascii_case("true");
        }
        if let Ok(v) = std::env::var("KILN_PREFIX_CACHE_MAX_BLOCKS") {
            if let Ok(n) = v.parse() {
                self.prefix_cache.max_blocks = Some(n);
            }
        }

        // Speculative decoding
        if let Ok(v) = std::env::var("KILN_SPEC_ENABLED") {
            self.speculative.enabled = v == "1" || v.eq_ignore_ascii_case("true");
        }
        if let Ok(v) = std::env::var("KILN_SPEC_NUM_TOKENS") {
            if let Ok(n) = v.parse() {
                self.speculative.num_speculative_tokens = n;
            }
        }
        if let Ok(v) = std::env::var("KILN_SPEC_DRAFT_LAYERS") {
            if let Ok(n) = v.parse() {
                self.speculative.draft_layers = n;
            }
        }
    }

    /// Validate configuration values. Returns an error describing the first invalid value.
    fn validate(&self) -> Result<()> {
        if self.server.port == 0 {
            anyhow::bail!("server.port must be > 0");
        }
        if self.server.request_timeout_secs == 0 {
            anyhow::bail!("server.request_timeout_secs must be > 0");
        }
        if self.server.shutdown_timeout_secs == 0 {
            anyhow::bail!("server.shutdown_timeout_secs must be > 0");
        }

        let f = self.memory.inference_memory_fraction;
        if !(0.0..=1.0).contains(&f) {
            anyhow::bail!(
                "memory.inference_memory_fraction must be between 0.0 and 1.0, got {f}"
            );
        }

        let valid_levels = ["trace", "debug", "info", "warn", "error"];
        let level = self.logging.level.to_lowercase();
        // Allow both simple levels and tracing filter directives (contain '=')
        if !valid_levels.contains(&level.as_str()) && !level.contains('=') {
            anyhow::bail!(
                "logging.level must be one of {valid_levels:?} or a tracing filter directive, got '{}'",
                self.logging.level
            );
        }

        if self.speculative.enabled {
            if self.speculative.num_speculative_tokens == 0 {
                anyhow::bail!("speculative.num_speculative_tokens must be > 0");
            }
            if self.speculative.draft_layers == 0 {
                anyhow::bail!("speculative.draft_layers must be > 0");
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defaults() {
        let config = KilnConfig::default();
        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port, 8420);
        assert_eq!(config.server.request_timeout_secs, 300);
        assert_eq!(config.server.shutdown_timeout_secs, 30);
        assert_eq!(config.model.model_id, "Qwen/Qwen3.5-4B");
        assert!(config.model.path.is_none());
        assert!(config.model.tokenizer_path.is_none());
        assert!(config.model.adapter_dir.is_none());
        assert!(config.memory.num_blocks.is_none());
        assert_eq!(config.memory.inference_memory_fraction, 0.7);
        assert!(!config.memory.kv_cache_fp8);
        assert!(config.memory.cuda_graphs);
        assert!(!config.training.no_grad_checkpoint);
        assert!(config.training.checkpoint_interval.is_none());
        assert_eq!(config.logging.level, "info");
        assert_eq!(config.logging.format, "json");
        assert!(config.prefix_cache.enabled);
        assert!(config.prefix_cache.max_blocks.is_none());
        assert!(!config.speculative.enabled);
        assert_eq!(config.speculative.num_speculative_tokens, 4);
        assert_eq!(config.speculative.draft_layers, 8);
    }

    #[test]
    fn test_parse_full_toml() {
        let toml_str = r#"
[server]
host = "127.0.0.1"
port = 9000
request_timeout_secs = 60
shutdown_timeout_secs = 10

[model]
path = "/models/qwen"
model_id = "custom/model"
tokenizer_path = "/models/tokenizer.json"
adapter_dir = "/models/adapters"

[memory]
num_blocks = 128
gpu_memory_gb = 24.0
inference_memory_fraction = 0.5
training_memory_gb = 6.0
kv_cache_fp8 = true
cuda_graphs = false

[training]
grad_checkpoint_segments = 8
no_grad_checkpoint = false
checkpoint_interval = 50

[logging]
level = "debug"
format = "pretty"

[prefix_cache]
enabled = false
max_blocks = 32

[speculative]
enabled = true
num_speculative_tokens = 6
draft_layers = 10
"#;
        let config: KilnConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 9000);
        assert_eq!(config.server.request_timeout_secs, 60);
        assert_eq!(config.model.path.as_deref(), Some("/models/qwen"));
        assert_eq!(config.model.model_id, "custom/model");
        assert_eq!(config.memory.num_blocks, Some(128));
        assert_eq!(config.memory.gpu_memory_gb, Some(24.0));
        assert_eq!(config.memory.inference_memory_fraction, 0.5);
        assert_eq!(config.memory.training_memory_gb, Some(6.0));
        assert!(config.memory.kv_cache_fp8);
        assert!(!config.memory.cuda_graphs);
        assert_eq!(config.training.grad_checkpoint_segments, Some(8));
        assert_eq!(config.training.checkpoint_interval, Some(50));
        assert_eq!(config.logging.level, "debug");
        assert_eq!(config.logging.format, "pretty");
        assert!(!config.prefix_cache.enabled);
        assert_eq!(config.prefix_cache.max_blocks, Some(32));
        assert!(config.speculative.enabled);
        assert_eq!(config.speculative.num_speculative_tokens, 6);
        assert_eq!(config.speculative.draft_layers, 10);
    }

    #[test]
    fn test_partial_toml_uses_defaults() {
        let toml_str = r#"
[server]
port = 3000
"#;
        let config: KilnConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.server.port, 3000);
        assert_eq!(config.server.host, "0.0.0.0"); // default
        assert_eq!(config.server.request_timeout_secs, 300); // default
        assert_eq!(config.model.model_id, "Qwen/Qwen3.5-4B"); // default
        assert_eq!(config.memory.inference_memory_fraction, 0.7); // default
        assert_eq!(config.logging.level, "info"); // default
    }

    #[test]
    fn test_validation_rejects_port_zero() {
        let mut config = KilnConfig::default();
        config.server.port = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_fraction_above_one() {
        let mut config = KilnConfig::default();
        config.memory.inference_memory_fraction = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_negative_fraction() {
        let mut config = KilnConfig::default();
        config.memory.inference_memory_fraction = -0.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_bad_log_level() {
        let mut config = KilnConfig::default();
        config.logging.level = "banana".into();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_accepts_filter_directive() {
        let mut config = KilnConfig::default();
        config.logging.level = "kiln=trace,tower_http=warn".into();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validation_rejects_zero_timeout() {
        let mut config = KilnConfig::default();
        config.server.request_timeout_secs = 0;
        assert!(config.validate().is_err());

        let mut config2 = KilnConfig::default();
        config2.server.shutdown_timeout_secs = 0;
        assert!(config2.validate().is_err());
    }

    #[test]
    fn test_env_var_overrides() {
        // Safety: these tests manipulate env vars which is unsafe in Rust 1.78+.
        // They are safe because cargo test runs lib tests single-threaded by default.
        unsafe {
            std::env::set_var("KILN_HOST", "10.0.0.1");
            std::env::set_var("KILN_PORT", "7777");
            std::env::set_var("KILN_MODEL_PATH", "/tmp/model");
            std::env::set_var("KILN_INFERENCE_MEMORY_FRACTION", "0.9");
            std::env::set_var("KILN_LOG_LEVEL", "debug");
            std::env::set_var("KILN_NO_GRAD_CHECKPOINT", "1");
            std::env::set_var("KILN_CHECKPOINT_INTERVAL", "25");
            std::env::set_var("KILN_KV_CACHE_FP8", "1");
            std::env::set_var("KILN_CUDA_GRAPHS", "false");
            std::env::set_var("KILN_PREFIX_CACHE_ENABLED", "false");
            std::env::set_var("KILN_PREFIX_CACHE_MAX_BLOCKS", "128");
            std::env::set_var("KILN_SPEC_ENABLED", "1");
            std::env::set_var("KILN_SPEC_NUM_TOKENS", "6");
            std::env::set_var("KILN_SPEC_DRAFT_LAYERS", "10");
        }

        let mut config = KilnConfig::default();
        config.apply_env_overrides();

        assert_eq!(config.server.host, "10.0.0.1");
        assert_eq!(config.server.port, 7777);
        assert_eq!(config.model.path.as_deref(), Some("/tmp/model"));
        assert_eq!(config.memory.inference_memory_fraction, 0.9);
        assert_eq!(config.logging.level, "debug");
        assert!(config.training.no_grad_checkpoint);
        assert_eq!(config.training.checkpoint_interval, Some(25));
        assert!(config.memory.kv_cache_fp8);
        assert!(!config.memory.cuda_graphs);
        assert!(!config.prefix_cache.enabled);
        assert_eq!(config.prefix_cache.max_blocks, Some(128));
        assert!(config.speculative.enabled);
        assert_eq!(config.speculative.num_speculative_tokens, 6);
        assert_eq!(config.speculative.draft_layers, 10);

        // Clean up
        unsafe {
            std::env::remove_var("KILN_HOST");
            std::env::remove_var("KILN_PORT");
            std::env::remove_var("KILN_MODEL_PATH");
            std::env::remove_var("KILN_INFERENCE_MEMORY_FRACTION");
            std::env::remove_var("KILN_LOG_LEVEL");
            std::env::remove_var("KILN_NO_GRAD_CHECKPOINT");
            std::env::remove_var("KILN_CHECKPOINT_INTERVAL");
            std::env::remove_var("KILN_KV_CACHE_FP8");
            std::env::remove_var("KILN_CUDA_GRAPHS");
            std::env::remove_var("KILN_PREFIX_CACHE_ENABLED");
            std::env::remove_var("KILN_PREFIX_CACHE_MAX_BLOCKS");
            std::env::remove_var("KILN_SPEC_ENABLED");
            std::env::remove_var("KILN_SPEC_NUM_TOKENS");
            std::env::remove_var("KILN_SPEC_DRAFT_LAYERS");
        }
    }

    #[test]
    fn test_load_missing_file_returns_defaults() {
        // With no file and no KILN_CONFIG env var, should return defaults
        unsafe {
            std::env::remove_var("KILN_CONFIG");
            // Clear env vars that would override defaults
            std::env::remove_var("KILN_HOST");
            std::env::remove_var("KILN_PORT");
            std::env::remove_var("KILN_MODEL_PATH");
            std::env::remove_var("KILN_LOG_LEVEL");
            std::env::remove_var("KILN_LOG_FORMAT");
            std::env::remove_var("KILN_NO_GRAD_CHECKPOINT");
        }
        unsafe {
            std::env::remove_var("KILN_SPEC_ENABLED");
            std::env::remove_var("KILN_SPEC_NUM_TOKENS");
            std::env::remove_var("KILN_SPEC_DRAFT_LAYERS");
        }
        // Load from a path that doesn't exist via the CWD fallback (kiln.toml won't exist in test dir)
        let config = KilnConfig::load(None).unwrap();
        assert_eq!(config.server.port, 8420);
        assert_eq!(config.model.model_id, "Qwen/Qwen3.5-4B");
    }

    #[test]
    fn test_load_explicit_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.toml");
        std::fs::write(
            &path,
            r#"
[server]
port = 5555

[logging]
level = "warn"
"#,
        )
        .unwrap();

        unsafe {
            // Clear env vars so they don't interfere
            std::env::remove_var("KILN_PORT");
            std::env::remove_var("KILN_LOG_LEVEL");
            std::env::remove_var("KILN_LOG_FORMAT");
            std::env::remove_var("KILN_HOST");
            std::env::remove_var("KILN_MODEL_PATH");
            std::env::remove_var("KILN_NO_GRAD_CHECKPOINT");
            std::env::remove_var("KILN_SPEC_ENABLED");
            std::env::remove_var("KILN_SPEC_NUM_TOKENS");
            std::env::remove_var("KILN_SPEC_DRAFT_LAYERS");
        }

        let config = KilnConfig::load(Some(path.to_str().unwrap())).unwrap();
        assert_eq!(config.server.port, 5555);
        assert_eq!(config.logging.level, "warn");
        assert_eq!(config.server.host, "0.0.0.0"); // default
    }

    #[test]
    fn test_load_nonexistent_explicit_path_errors() {
        let result = KilnConfig::load(Some("/no/such/file.toml"));
        assert!(result.is_err());
    }
}
