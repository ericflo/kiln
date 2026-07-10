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
//! With the batching engine running (the default), the swap closure executes
//! at the engine's between-requests barrier (the `ResizeKv` pattern): admission
//! pauses, active requests finish, then the weights flip and queued requests
//! resume. The engine-less fallback takes exclusive GPU coordination ownership,
//! paired with the request-lifetime read owner held by direct inference.
//!
//! Cache coherence and the server-default selection ride along: when the
//! target adapter's directory content changed (retrain/import), its name-keyed
//! cache entries are purged inside the same barrier, after every request that
//! could have been computed under the old weights has finished.

use std::path::PathBuf;
use std::sync::Arc;

use kiln_model::ModelRunner;
use kiln_model::lora_loader::LoraWeights;

use crate::state::{
    AppState, LoadedAdapterIdentity, ModelBackend, gpu_coordination_write_guard_while_healthy,
};

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
    /// How this physical transition changes the server-default adapter.
    /// Per-request swaps preserve it; explicit eval pinning, promotion, and
    /// unload update it atomically with the loaded identity and runner weights.
    pub default_adapter: DefaultAdapterUpdate,
    /// For the transition log line.
    pub reason: &'static str,
}

/// Server-default update applied at the same publication point as a weight
/// flip. `ClearIf` lets a gate demote one adapter without clearing a different
/// default that won an earlier serialized transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefaultAdapterUpdate {
    Preserve,
    Replace(Option<String>),
    ClearIf(String),
}

pub(crate) type AdapterMutationGuard<'a> = tokio::sync::MutexGuard<'a, ()>;

/// Activate `req.target` on the live runner. Serialized via
/// `state.adapter_mutation_lock`; no-ops when the target is already loaded
/// (unless `content_changed`). Returns the previously-loaded adapter name.
pub async fn swap_runtime_adapter(
    state: &AppState,
    req: SwapRequest,
) -> Result<Option<String>, String> {
    let serial = adapter_mutation_guard(state).await?;
    swap_runtime_adapter_locked(state, req, &serial).await
}

/// Guarded form used by a larger filesystem/revision transaction. The guard
/// argument proves that the caller owns the shared mutation barrier.
pub(crate) async fn swap_runtime_adapter_locked(
    state: &AppState,
    req: SwapRequest,
    _serial: &AdapterMutationGuard<'_>,
) -> Result<Option<String>, String> {
    validate_default_update(&req)?;

    let target_name = req.target.cache_name();
    let current = state.loaded_adapter_name();
    if current == target_name && !req.content_changed {
        apply_default_update(state, &req.default_adapter);
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
    state
        .ensure_backend_healthy()
        .map_err(|error| format!("{error:#}"))?;
    let target_identity = loaded_identity(&target_name, lora.as_ref())?;

    let backend_health = state
        .backend_health_handle()
        .ok_or_else(|| "adapter swap requires the real model backend".to_string())?;

    let closure = swap_closure(
        state,
        runner,
        lora,
        target_identity,
        req.content_changed,
        req.default_adapter,
    );
    match engine {
        Some(engine) => engine
            .swap_adapter_while_healthy(closure, &backend_health)
            .await
            .map_err(|e| format!("{e:#}"))?,
        None => {
            let gpu_lock = state.gpu_lock.clone();
            let backend_health = backend_health.clone();
            tokio::task::spawn_blocking(move || {
                let _gpu_guard =
                    gpu_coordination_write_guard_while_healthy(&gpu_lock, &backend_health)
                        .map_err(|error| format!("{error:#}"))?;
                closure()
            })
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
    let serial = adapter_mutation_guard_blocking(state)?;
    swap_runtime_adapter_blocking_locked(state, req, &serial)
}

/// Blocking guarded form for training publication transactions.
pub(crate) fn swap_runtime_adapter_blocking_locked(
    state: &AppState,
    req: SwapRequest,
    _serial: &AdapterMutationGuard<'_>,
) -> Result<Option<String>, String> {
    validate_default_update(&req)?;

    let target_name = req.target.cache_name();
    let current = state.loaded_adapter_name();
    if current == target_name && !req.content_changed {
        apply_default_update(state, &req.default_adapter);
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
    state
        .ensure_backend_healthy()
        .map_err(|error| format!("{error:#}"))?;
    let target_identity = loaded_identity(&target_name, lora.as_ref())?;

    let backend_health = state
        .backend_health_handle()
        .ok_or_else(|| "adapter swap requires the real model backend".to_string())?;

    let closure = swap_closure(
        state,
        runner,
        lora,
        target_identity,
        req.content_changed,
        req.default_adapter,
    );
    match engine {
        Some(engine) => engine
            .swap_adapter_blocking_while_healthy(closure, &backend_health)
            .map_err(|e| format!("{e:#}"))?,
        None => {
            let _gpu_guard =
                gpu_coordination_write_guard_while_healthy(&state.gpu_lock, &backend_health)
                    .map_err(|error| format!("{error:#}"))?;
            closure()?;
        }
    }

    log_transition(&current, &target_name, req.reason);
    Ok(current)
}

pub(crate) async fn adapter_mutation_guard(
    state: &AppState,
) -> Result<AdapterMutationGuard<'_>, String> {
    loop {
        state
            .ensure_backend_healthy()
            .map_err(|error| format!("{error:#}"))?;
        if let Ok(guard) = state.adapter_mutation_lock.try_lock() {
            state
                .ensure_backend_healthy()
                .map_err(|error| format!("{error:#}"))?;
            return Ok(guard);
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
}

pub(crate) fn adapter_mutation_guard_blocking(
    state: &AppState,
) -> Result<AdapterMutationGuard<'_>, String> {
    loop {
        state
            .ensure_backend_healthy()
            .map_err(|error| format!("{error:#}"))?;
        if let Ok(guard) = state.adapter_mutation_lock.try_lock() {
            state
                .ensure_backend_healthy()
                .map_err(|error| format!("{error:#}"))?;
            return Ok(guard);
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
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

fn loaded_identity(
    target_name: &Option<String>,
    lora: Option<&LoraWeights>,
) -> Result<Option<LoadedAdapterIdentity>, String> {
    match (target_name, lora) {
        (None, None) => Ok(None),
        (Some(name), Some(lora)) => lora
            .source_identity()
            .map(|source| Some(LoadedAdapterIdentity::from_source(name.clone(), source)))
            .ok_or_else(|| format!("adapter `{name}` was loaded without an exact source identity")),
        (name, weights) => Err(format!(
            "adapter target/weight mismatch: target={name:?}, has_weights={}",
            weights.is_some()
        )),
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
    target_identity: Option<LoadedAdapterIdentity>,
    content_changed: bool,
    default_adapter: DefaultAdapterUpdate,
) -> crate::batching_engine::AdapterSwapClosure {
    let state = state.clone();
    Box::new(move || {
        state
            .ensure_backend_healthy()
            .map_err(|error| format!("{error:#}"))?;
        {
            let mut published = state.loaded_adapter.write().unwrap();
            let mut guard = runner.write().unwrap();
            guard
                .swap_lora(lora)
                .map_err(|error| format!("{error:#}"))?;
            *published = target_identity.clone();
            apply_default_update(&state, &default_adapter);
        }
        state
            .ensure_backend_healthy()
            .map_err(|error| format!("{error:#}"))?;
        if content_changed {
            state.purge_adapter_caches(
                &target_identity
                    .as_ref()
                    .map(|identity| identity.name.clone()),
            );
        }
        Ok(())
    })
}

fn validate_default_update(req: &SwapRequest) -> Result<(), String> {
    let target = req.target.cache_name();
    match &req.default_adapter {
        DefaultAdapterUpdate::Preserve | DefaultAdapterUpdate::ClearIf(_) => Ok(()),
        DefaultAdapterUpdate::Replace(default) if *default == target => Ok(()),
        DefaultAdapterUpdate::Replace(default) => Err(format!(
            "adapter transition cannot publish default {default:?} while loading {target:?}"
        )),
    }
}

pub(crate) fn apply_default_update(state: &AppState, update: &DefaultAdapterUpdate) {
    match update {
        DefaultAdapterUpdate::Preserve => {}
        DefaultAdapterUpdate::Replace(next) => {
            *state.active_adapter_name.write().unwrap() = next.clone();
        }
        DefaultAdapterUpdate::ClearIf(name) => {
            let mut active = state.active_adapter_name.write().unwrap();
            if active.as_deref() == Some(name.as_str()) {
                *active = None;
            }
        }
    }
}

fn log_transition(old: &Option<String>, new: &Option<String>, reason: &str) {
    tracing::info!(
        old_adapter = ?old,
        new_adapter = ?new,
        reason = reason,
        "adapter transition (barrier swap)"
    );
}
