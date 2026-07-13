//! Kiln HTTP server — library interface for integration testing.

/// Process environment is global mutable state. Every unit test in this crate
/// that reads around a temporary environment mutation must hold this lock.
#[cfg(test)]
pub(crate) static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub mod adapter_swap;
pub mod adapter_verify;
pub mod agent_runs;
pub mod api;
pub mod batching_engine;
pub mod cli;
pub mod config;
pub mod dataset_resolve;
pub mod decode_stats;
pub mod device;
pub(crate) mod device_memory;
pub mod error;
pub mod eval;
pub mod eval_adapter_cli;
pub mod eval_history;
pub mod execution_provenance;
pub mod hf_train_cli;
pub mod kv_autoscaler;
pub mod logging;
pub mod metrics;
pub mod pi_rpc;
pub mod recent_requests;
pub mod request_log;
pub(crate) mod response_delivery;
pub mod rollout_generate_cli;
pub(crate) mod sft_dataset;
pub mod state;
pub mod teacher_identity;
pub mod training_history;
pub mod training_preflight;
pub mod training_queue;
