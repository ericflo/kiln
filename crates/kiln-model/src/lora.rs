use serde::{Deserialize, Serialize};

// Re-export weight types from lora_loader for convenience.
pub use crate::lora_loader::{LoraLayerWeights, LoraWeights};

/// Metadata for a LoRA adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterInfo {
    /// Unique name for this adapter.
    pub name: String,
    /// LoRA rank.
    pub rank: usize,
    /// LoRA alpha (scaling factor).
    pub alpha: f32,
    /// Which modules the adapter targets (e.g., "all-linear").
    pub target_modules: Vec<String>,
    /// Path to the adapter weights on disk.
    pub path: String,
    /// Version counter — incremented on each training update.
    pub version: u64,
    /// Size in bytes.
    pub size_bytes: u64,
}

/// Status of the active adapter in the engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterStatus {
    /// Currently active adapter (None = base model only).
    pub active: Option<AdapterInfo>,
    /// Adapter being trained (if any).
    pub training: Option<String>,
    /// All available adapters.
    pub available: Vec<AdapterInfo>,
}
