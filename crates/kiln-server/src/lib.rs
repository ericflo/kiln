//! Kiln HTTP server — library interface for integration testing.

pub mod api;
pub mod adapter_verify;
pub mod batching_engine;
pub mod cli;
pub mod config;
pub mod decode_stats;
pub mod device;
pub mod error;
pub mod eval;
pub mod eval_history;
pub mod logging;
pub mod metrics;
pub mod recent_requests;
pub mod state;
pub mod training_history;
pub mod training_preflight;
pub mod training_queue;
