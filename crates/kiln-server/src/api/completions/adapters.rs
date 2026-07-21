use super::*;

pub(super) async fn ensure_adapter(
    state: &AppState,
    runner: &std::sync::Arc<std::sync::RwLock<ModelRunner>>,
    req_adapter: &ChatAdapterSelection,
    request_id: &str,
) -> Result<(), ApiError> {
    let target = req_adapter.target_adapter_name(state.active_adapter_name.read().unwrap().clone());
    ensure_runtime_adapter(state, runner, target, request_id, req_adapter.reason()).await
}

pub(super) async fn ensure_batch_adapter(
    state: &AppState,
    runner: &std::sync::Arc<std::sync::RwLock<ModelRunner>>,
    req_adapter: &Option<String>,
    request_id: &str,
) -> Result<(), ApiError> {
    let target = req_adapter
        .clone()
        .or_else(|| state.active_adapter_name.read().unwrap().clone());
    ensure_runtime_adapter(
        state,
        runner,
        target,
        request_id,
        if req_adapter.is_some() {
            "batch_adapter_explicit_name"
        } else {
            "batch_adapter_missing_use_default"
        },
    )
    .await
}

pub(super) async fn ensure_runtime_adapter(
    state: &AppState,
    _runner: &std::sync::Arc<std::sync::RwLock<ModelRunner>>,
    target_adapter: Option<String>,
    request_id: &str,
    reason: &str,
) -> Result<(), ApiError> {
    let current = state.loaded_adapter_name();
    if target_adapter == current {
        return Ok(());
    }

    let target = match target_adapter.clone() {
        Some(name) => {
            validate_compose_name(&name)?;
            if !state.adapter_dir.join(&name).exists() {
                return Err(ApiError::adapter_not_found(&name));
            }
            crate::adapter_swap::SwapTarget::Named(name)
        }
        None => crate::adapter_swap::SwapTarget::Base,
    };

    // The actual weight flip happens at the batching engine's
    // between-requests barrier (see `adapter_swap`), so a streaming
    // request can never continue mid-generation on different weights.
    crate::adapter_swap::swap_runtime_adapter(
        state,
        crate::adapter_swap::SwapRequest {
            target,
            content_changed: false,
            default_adapter: crate::adapter_swap::DefaultAdapterUpdate::Preserve,
            reason: "per_request_adapter",
        },
    )
    .await
    .map_err(ApiError::adapter_load_failed)?;

    tracing::info!(
        request_id = %request_id,
        old_adapter = ?current,
        new_adapter = ?target_adapter,
        reason = reason,
        "adapter transition"
    );
    if state.eval_mode {
        tracing::warn!(
            request_id = %request_id,
            old_adapter = ?current,
            new_adapter = ?target_adapter,
            reason = reason,
            "adapter transition during eval mode"
        );
    }

    Ok(())
}

/// Disk handle for a composed adapter ready to be loaded.
#[derive(Debug, Clone)]
pub(super) struct ComposedTarget {
    /// Stable cache name embedded in the loaded-adapter identity once swapped in,
    /// e.g. `"__composed:abc123..."`. Used for cache-hit comparison and as
    /// the prefix-cache adapter key.
    active_name: String,
    /// On-disk directory holding the synthesized PEFT adapter.
    cache_dir: PathBuf,
}

/// Validate a single source-adapter name from an `adapters: [...]` request.
///
/// Names must be a single path segment with no separators or traversal — same
/// rules as `validate_adapter_name` in `api/adapters.rs`. Centralized here so
/// `chat_completions` can return a 404-shaped error consistent with the
/// existing single-adapter path (`adapter_not_found`).
pub(super) fn validate_compose_name(name: &str) -> Result<(), ApiError> {
    if name.is_empty()
        || name == "."
        || name == ".."
        || name.contains('/')
        || name.contains('\\')
        || name.contains("..")
        || Path::new(name).is_absolute()
    {
        return Err(ApiError::invalid_adapter_name(name));
    }
    Ok(())
}

pub(super) type ResolvedCompositionSource = (String, f32, PathBuf, String);

/// Resolve every source and bind the cache key to its exact PEFT revision.
/// The caller owns the adapter mutation guard, so config and weight identities
/// cannot change between hashing and the merge loader opening them.
pub(super) fn resolve_composition_sources(
    adapter_dir: &Path,
    adapters: &[(String, f32)],
) -> Result<(String, Vec<ResolvedCompositionSource>), ApiError> {
    use sha2::{Digest, Sha256};

    let mut sources = Vec::with_capacity(adapters.len());
    for (name, scale) in adapters {
        let path = adapter_dir.join(name);
        if !path.is_dir() {
            return Err(ApiError::adapter_not_found(name));
        }
        let content_revision = kiln_model::lora_loader::LoraSourceIdentity::from_adapter_dir(&path)
            .map_err(|error| {
                ApiError::adapter_merge_failed(format!(
                    "resolve exact source revision for '{}' at {}: {error:#}",
                    name,
                    path.display()
                ))
            })?
            .content_revision();
        sources.push((name.clone(), *scale, path, content_revision));
    }

    let mut canonical: Vec<_> = sources
        .iter()
        .map(|(name, scale, _, revision)| (name.as_str(), scale.to_bits(), revision.as_str()))
        .collect();
    canonical.sort_unstable();
    let mut hasher = Sha256::new();
    hasher.update(b"kiln-composed-adapter-v2\0");
    for (name, scale_bits, revision) in canonical {
        hasher.update((name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update(scale_bits.to_le_bytes());
        hasher.update((revision.len() as u64).to_le_bytes());
        hasher.update(revision.as_bytes());
    }
    let hash = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    Ok((hash, sources))
}

/// Synthesize (or reuse the on-disk cache for) a composed adapter spec.
///
/// On first call for a given source-revision hash, loads each source adapter, runs
/// `merge_concat`, and writes the result under `<adapter_dir>/.composed/<hash>/`.
/// Source-adapter lookup uses the same single-segment path resolution as
/// `ensure_adapter`; missing sources surface as 404. Publication uses a hidden
/// staging directory and one rename, so a failed merge never leaves a cache
/// hit that looks complete.
pub(super) async fn synthesize_composed_adapter_locked(
    adapter_dir: &Path,
    adapters: &[AdapterRef],
    _serial: &crate::adapter_swap::AdapterMutationGuard<'_>,
) -> Result<ComposedTarget, ApiError> {
    let adapter_dir = adapter_dir.to_path_buf();
    let adapters: Vec<_> = adapters
        .iter()
        .map(|source| (source.name.clone(), source.scale))
        .collect();
    tokio::task::spawn_blocking(move || {
        synthesize_composed_adapter_blocking(&adapter_dir, &adapters)
    })
    .await
    .map_err(|error| ApiError::internal(format!("join composed-adapter publisher: {error}")))?
}

/// Blocking half of [`synthesize_composed_adapter_locked`]. Keeping every
/// filesystem read and CPU merge in one blocking task prevents a large LoRA
/// fingerprint from stalling an async request worker.
pub(super) fn synthesize_composed_adapter_blocking(
    adapter_dir: &Path,
    adapters: &[(String, f32)],
) -> Result<ComposedTarget, ApiError> {
    let (hash, source_paths) = resolve_composition_sources(adapter_dir, adapters)?;
    let active_name = format!("__composed:{hash}");
    let composed_root = adapter_dir.join(".composed");
    let cache_dir = composed_root.join(&hash);

    if cache_dir.exists() {
        if let Err(error) =
            kiln_model::lora_loader::LoraSourceIdentity::from_adapter_dir(&cache_dir)
        {
            tracing::warn!(
                cache_dir = %cache_dir.display(),
                error = %format!("{error:#}"),
                "discarding incomplete composed-adapter cache entry"
            );
            std::fs::remove_dir_all(&cache_dir).map_err(|remove_error| {
                ApiError::adapter_merge_failed(format!(
                    "remove incomplete composed cache {}: {remove_error}",
                    cache_dir.display()
                ))
            })?;
        } else {
            // Cache hit: refresh the directory's mtime so LRU eviction treats this
            // entry as recently used. Best-effort — a failure does not block the
            // request, and stale mtimes only mean slightly less-accurate LRU
            // ordering.
            let now = filetime::FileTime::from_system_time(std::time::SystemTime::now());
            if let Err(e) = filetime::set_file_mtime(&cache_dir, now) {
                tracing::warn!(
                    cache_dir = %cache_dir.display(),
                    error = %e,
                    "failed to refresh composed-cache mtime on hit (LRU may be slightly off)"
                );
            }
            return Ok(ComposedTarget {
                active_name,
                cache_dir,
            });
        }
    }

    std::fs::create_dir_all(&composed_root).map_err(|error| {
        ApiError::adapter_merge_failed(format!("creating composed-cache dir: {error}"))
    })?;
    let staging = tempfile::Builder::new()
        .prefix(".compose-tmp-")
        .tempdir_in(&composed_root)
        .map_err(|error| {
            ApiError::adapter_merge_failed(format!("creating composed-cache staging dir: {error}"))
        })?;
    let staging_output = staging.path().join("adapter");

    let mut loaded: Vec<(PeftLora, f32)> = Vec::with_capacity(source_paths.len());
    for (name, scale, path, _revision) in source_paths {
        let adapter = PeftLora::load(&path).map_err(|error| {
            ApiError::adapter_merge_failed(format!(
                "loading source '{name}' from {}: {error}",
                path.display()
            ))
        })?;
        loaded.push((adapter, scale));
    }

    let refs: Vec<(&PeftLora, f32)> = loaded
        .iter()
        .map(|(adapter, scale)| (adapter, *scale))
        .collect();
    let merged = merge_concat(&refs)
        .map_err(|error| ApiError::adapter_merge_failed(format!("merge_concat: {error}")))?;
    merged.save(&staging_output).map_err(|error| {
        ApiError::adapter_merge_failed(format!("saving composed adapter: {error}"))
    })?;
    std::fs::rename(&staging_output, &cache_dir).map_err(|error| {
        ApiError::adapter_merge_failed(format!(
            "publishing composed adapter {}: {error}",
            cache_dir.display()
        ))
    })?;

    Ok(ComposedTarget {
        active_name,
        cache_dir,
    })
}

/// LRU-evict entries from `composed_root` until total entries `<= max_entries`
/// and total bytes `<= max_bytes`. Either bound being `None` disables that
/// dimension; if both are `None` the function is a no-op.
///
/// Eviction is best-effort: individual `remove_dir_all` failures are logged
/// and the loop continues. Hidden / non-directory entries (anything whose
/// name starts with `.`) are skipped — kiln only writes hash-named
/// subdirectories under `.composed/`, but a stray file should not be picked
/// for eviction.
///
/// Closes audit LOW §8 / roadmap item 8 (PR #620 capped uploaded adapters
/// but explicitly excluded this cache pending this LRU pass).
pub(super) fn evict_composed_cache_lru(
    composed_root: &Path,
    max_bytes: Option<u64>,
    max_entries: Option<u64>,
    protected: &Path,
    _serial: &crate::adapter_swap::AdapterMutationGuard<'_>,
) {
    if max_bytes.is_none() && max_entries.is_none() {
        return;
    }
    let read_dir = match std::fs::read_dir(composed_root) {
        Ok(rd) => rd,
        Err(_) => return, // Parent gone or unreadable — nothing to evict.
    };

    // Gather (path, mtime, size) for each cache entry. `mtime` is read via
    // `std::fs::Metadata::modified()`; if unavailable we fall back to
    // `UNIX_EPOCH` so the entry sorts as oldest and gets evicted first.
    let mut entries: Vec<(PathBuf, std::time::SystemTime, u64)> = Vec::new();
    let mut total_bytes: u64 = 0;
    for entry in read_dir.flatten() {
        let name = entry.file_name();
        let name_lossy = name.to_string_lossy();
        // Skip hidden / sentinel files (names starting with `.`). All real
        // entries are 64-hex-digit revision hashes.
        if name_lossy.starts_with('.') {
            continue;
        }
        let path = entry.path();
        let meta = match std::fs::symlink_metadata(&path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        if !meta.file_type().is_dir() {
            continue;
        }
        let mtime = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
        let size = composed_entry_size_bytes(&path);
        total_bytes = total_bytes.saturating_add(size);
        entries.push((path, mtime, size));
    }

    // Oldest first.
    entries.sort_by(|a, b| a.1.cmp(&b.1));

    let mut total_entries = entries.len() as u64;
    let mut iter = entries.into_iter();
    while (max_entries.is_some_and(|cap| total_entries > cap))
        || (max_bytes.is_some_and(|cap| total_bytes > cap))
    {
        let (path, _mtime, size) = match iter.next() {
            Some(e) => e,
            None => break, // Caps still exceeded but nothing left to evict.
        };
        if path == protected {
            continue;
        }
        match std::fs::remove_dir_all(&path) {
            Ok(()) => {
                total_entries = total_entries.saturating_sub(1);
                total_bytes = total_bytes.saturating_sub(size);
                tracing::info!(
                    evicted = %path.display(),
                    freed_bytes = size,
                    "composed-adapter cache LRU eviction"
                );
            }
            Err(e) => {
                tracing::warn!(
                    cache_dir = %path.display(),
                    error = %e,
                    "failed to evict composed-cache entry (will retry next eviction)"
                );
                // Don't decrement — couldn't free this one.
            }
        }
    }
}

/// Recursively sum regular-file byte sizes under a composed-cache entry.
/// Mirrors the conservative best-effort spirit of
/// `dir_size_recursive` in `api/adapters.rs` — symlinks and stat errors
/// count as zero.
pub(super) fn composed_entry_size_bytes(root: &Path) -> u64 {
    let meta = match std::fs::symlink_metadata(root) {
        Ok(m) => m,
        Err(_) => return 0,
    };
    if meta.file_type().is_file() {
        return meta.len();
    }
    if !meta.file_type().is_dir() {
        return 0;
    }
    let read_dir = match std::fs::read_dir(root) {
        Ok(rd) => rd,
        Err(_) => return 0,
    };
    let mut total: u64 = 0;
    for entry in read_dir.flatten() {
        total = total.saturating_add(composed_entry_size_bytes(&entry.path()));
    }
    total
}

/// Hot-swap the runner onto a synthesized composed adapter.
///
/// Same barrier semantics as `ensure_runtime_adapter` — composed names are
/// content-hashed (`__composed:<hash>`), so they never alias stale cache
/// entries and `content_changed` stays false. No-op if already active.
pub(super) async fn ensure_composed_adapter_swap_locked(
    state: &AppState,
    target: &ComposedTarget,
    serial: &crate::adapter_swap::AdapterMutationGuard<'_>,
) -> Result<(), ApiError> {
    {
        let current = state.loaded_adapter_identity();
        if current.as_ref().map(|identity| identity.name.as_str())
            == Some(target.active_name.as_str())
        {
            return Ok(());
        }
    }

    crate::adapter_swap::swap_runtime_adapter_locked(
        state,
        crate::adapter_swap::SwapRequest {
            target: crate::adapter_swap::SwapTarget::Resolved {
                active_name: target.active_name.clone(),
                dir: target.cache_dir.clone(),
            },
            content_changed: false,
            default_adapter: crate::adapter_swap::DefaultAdapterUpdate::Preserve,
            reason: "composed_adapter",
        },
        serial,
    )
    .await
    .map_err(ApiError::adapter_load_failed)?;
    if state.eval_mode {
        tracing::warn!(
            adapter = %target.active_name,
            "composed adapter transition during eval mode"
        );
    }

    Ok(())
}

/// Resolve, synthesize, publish, load, and evict a composed adapter while one
/// mutation guard covers every disk and loaded-weight transition.
pub(super) async fn ensure_composed_adapter_for_request(
    state: &AppState,
    adapters: &[AdapterRef],
) -> Result<(), ApiError> {
    let serial = crate::adapter_swap::adapter_mutation_guard(state)
        .await
        .map_err(ApiError::adapter_load_failed)?;
    let target = synthesize_composed_adapter_locked(&state.adapter_dir, adapters, &serial).await?;
    if matches!(state.backend.as_ref(), ModelBackend::Real { .. }) {
        ensure_composed_adapter_swap_locked(state, &target, &serial).await?;
    }
    evict_composed_cache_lru(
        &state.adapter_dir.join(".composed"),
        state.composed_cache_max_bytes,
        state.composed_cache_max_entries,
        &target.cache_dir,
        &serial,
    );
    Ok(())
}
