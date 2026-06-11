//! Single entry point for activating LoRA weights on the live runner.
//!
//! Correctness contract: **adapter weights must never change under an
//! in-flight request.** KV computed under the old weights followed by
//! decode steps under the new weights produces silently wrong text. The
//! historical swap sites (`ensure_runtime_adapter`, the training queue's
//! auto-load, the eval generator) each mutated the shared `ModelRunner`
//! behind a `write()` lock — which only waits for the *current decode
//! step* to finish, not the current request, so a streaming pi request
//! would continue mid-generation on different weights whenever training
//! auto-loaded or an eval swapped adapters.
//!
//! With the batching engine running (the default), the swap closure now
//! executes at the engine's between-requests barrier (the `ResizeKv`
//! pattern): admission pauses, active requests finish, then the weights
//! flip and queued requests resume. The engine-less fallback path keeps
//! the historical between-steps behavior — it serves one request at a
//! time, so the window is far smaller there.
//!
//! Cache coherence rides along: when the target adapter's directory
//! content changed (retrain auto-load, upload), its name-keyed cache
//! entries are purged inside the same barrier, after every request that
//! could have been computed under the old weights has finished.

use std::path::PathBuf;
use std::sync::Arc;

use kiln_model::ModelRunner;
use kiln_model::lora_loader::LoraWeights;

use crate::state::{AppState, ModelBackend};

/// What to activate.
#[derive(Debug, Clone)]
pub enum SwapTarget {
    /// Revert to the base model (no LoRA).
    Base,
    /// Adapter directory `<adapter_dir>/<name>`.
    Named(String),
    /// Pre-resolved directory with an explicit cache identity — composed
    /// adapters (`__composed:<hash>`) live outside `adapter_dir`.
    Resolved { active_name: String, dir: PathBuf },
}

impl SwapTarget {
    fn cache_name(&self) -> Option<String> {
        match self {
            SwapTarget::Base => None,
            SwapTarget::Named(name) => Some(name.clone()),
            SwapTarget::Resolved { active_name, .. } => Some(active_name.clone()),
        }
    }
}

/// One adapter activation request.
pub struct SwapRequest {
    pub target: SwapTarget,
    /// The target's on-disk content changed (retrain/import): force the
    /// reload even when the name is already active, and purge the name's
    /// cache entries at the barrier.
    pub content_changed: bool,
    /// For the transition log line.
    pub reason: &'static str,
}

/// Activate `req.target` on the live runner. Serialized via
/// `state.adapter_swap_lock`; no-ops when the target is already loaded
/// (unless `content_changed`). Returns the previously-loaded adapter name.
pub async fn swap_runtime_adapter(
    state: &AppState,
    req: SwapRequest,
) -> Result<Option<String>, String> {
    let _serial = state.adapter_swap_lock.lock().await;

    let target_name = req.target.cache_name();
    let current = state.loaded_adapter_name.read().unwrap().clone();
    if current == target_name && !req.content_changed {
        return Ok(current);
    }

    let (runner, engine) = real_backend_handles(state)?;
    let lora = match resolve_dir(state, &req.target)? {
        Some(dir) => {
            let (device, num_layers) = {
                let guard = runner.read().unwrap();
                (
                    guard.weights.embed_tokens.device().clone(),
                    guard.config.num_layers,
                )
            };
            // Load weights OUTSIDE the barrier (disk + H2D copy take real
            // time; the engine keeps serving meanwhile). Only the cheap
            // pointer flip happens at the barrier.
            let dir_clone = dir.clone();
            Some(
                tokio::task::spawn_blocking(move || {
                    LoraWeights::load(&dir_clone, num_layers, device).map_err(|e| format!("{e}"))
                })
                .await
                .map_err(|e| format!("join error: {e}"))??,
            )
        }
        None => None,
    };

    let closure = swap_closure(state, runner, lora, target_name.clone(), req.content_changed);
    match engine {
        Some(engine) => engine
            .swap_adapter(closure)
            .await
            .map_err(|e| format!("{e:#}"))?,
        None => {
            tokio::task::spawn_blocking(closure)
                .await
                .map_err(|e| format!("join error: {e}"))??;
        }
    }

    log_transition(&current, &target_name, req.reason);
    Ok(current)
}

/// Blocking variant for the training worker's job thread.
pub fn swap_runtime_adapter_blocking(
    state: &AppState,
    req: SwapRequest,
) -> Result<Option<String>, String> {
    let _serial = state.adapter_swap_lock.blocking_lock();

    let target_name = req.target.cache_name();
    let current = state.loaded_adapter_name.read().unwrap().clone();
    if current == target_name && !req.content_changed {
        return Ok(current);
    }

    let (runner, engine) = real_backend_handles(state)?;
    let lora = match resolve_dir(state, &req.target)? {
        Some(dir) => {
            let (device, num_layers) = {
                let guard = runner.read().unwrap();
                (
                    guard.weights.embed_tokens.device().clone(),
                    guard.config.num_layers,
                )
            };
            Some(LoraWeights::load(&dir, num_layers, device).map_err(|e| format!("{e}"))?)
        }
        None => None,
    };

    let closure = swap_closure(state, runner, lora, target_name.clone(), req.content_changed);
    match engine {
        Some(engine) => engine
            .swap_adapter_blocking(closure)
            .map_err(|e| format!("{e:#}"))?,
        None => closure()?,
    }

    log_transition(&current, &target_name, req.reason);
    Ok(current)
}

fn real_backend_handles(
    state: &AppState,
) -> Result<
    (
        Arc<std::sync::RwLock<ModelRunner>>,
        Option<crate::batching_engine::BatchingEngineHandle>,
    ),
    String,
> {
    match state.backend.as_ref() {
        ModelBackend::Real {
            runner,
            batching_engine,
            ..
        } => Ok((runner.clone(), batching_engine.clone())),
        ModelBackend::Mock { .. } => {
            Err("adapter swap requires the real model backend".to_string())
        }
    }
}

fn resolve_dir(state: &AppState, target: &SwapTarget) -> Result<Option<PathBuf>, String> {
    match target {
        SwapTarget::Base => Ok(None),
        SwapTarget::Named(name) => {
            let dir = state.adapter_dir.join(name);
            if !dir.exists() {
                return Err(format!("adapter `{name}` not found at {}", dir.display()));
            }
            Ok(Some(dir))
        }
        SwapTarget::Resolved { dir, .. } => Ok(Some(dir.clone())),
    }
}

/// The work that runs at the engine barrier (or inline on the fallback
/// path): flip the weights, update the physical-truth name, and purge the
/// target's stale cache entries when its content changed — after every
/// request that could have used the old weights has finished, so none can
/// re-register stale entries behind the purge.
fn swap_closure(
    state: &AppState,
    runner: Arc<std::sync::RwLock<ModelRunner>>,
    lora: Option<LoraWeights>,
    target_name: Option<String>,
    content_changed: bool,
) -> crate::batching_engine::AdapterSwapClosure {
    let state = state.clone();
    Box::new(move || {
        {
            let mut guard = runner.write().unwrap();
            guard.swap_lora(lora);
        }
        *state.loaded_adapter_name.write().unwrap() = target_name.clone();
        if content_changed {
            state.purge_adapter_caches(&target_name);
        }
        Ok(())
    })
}

fn log_transition(old: &Option<String>, new: &Option<String>, reason: &str) {
    tracing::info!(
        old_adapter = ?old,
        new_adapter = ?new,
        reason = reason,
        "adapter transition (barrier swap)"
    );
}
