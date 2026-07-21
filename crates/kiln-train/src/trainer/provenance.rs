use super::*;

/// Per-job context the HTTP layer hands the trainer so the accepted request
/// and its parent lineage can be audited from the on-disk artifacts.
///
/// `request_id` is the same UUID the queue uses for the job; `request_body`
/// is the verbatim deserialized request the HTTP handler accepted. The
/// trainer is responsible for resolving the effective seed (using
/// `config.seed.unwrap_or_else(|| rand::random())` so every run records a
/// concrete number), opening the parent lineage if `config.base_adapter` is
/// set, appending the replay record before stepping the optimizer, writing
/// `lineage.json`, and finally appending the outcome record.
#[derive(Debug, Clone)]
pub struct ReplayContext {
    pub request_id: String,
    pub kind: ReplayKind,
    pub request_body: serde_json::Value,
    pub base_model: BaseModel,
}

/// Build a default `BaseModel` description for the only model kiln supports.
///
/// `id` is fixed to `Qwen/Qwen3.5-4B`; `revision` is left unset; the config
/// digest is a SHA-256 of the JSON-serialized `ModelConfig` so lineage
/// verification can detect mismatched architectures even when `id` matches.
pub fn default_base_model(config: &ModelConfig) -> BaseModel {
    let digest = serde_json::to_string(config).ok().map(|s| {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(s.as_bytes());
        let hex: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        format!("sha256:{hex}")
    });
    BaseModel {
        id: "Qwen/Qwen3.5-4B".to_string(),
        revision: None,
        config_digest: digest,
    }
}

/// State threaded through a training run so the request can be appended
/// before the optimizer step and an outcome can be appended afterward.
pub struct ReplayState {
    pub(super) log: ReplayLog,
    pub(super) lineage: Lineage,
    pub(super) request_id: String,
    pub(super) started_at: std::time::Instant,
}

/// Reserved directory used by staged server training to pin the adapter being
/// rewritten. It is intentionally hidden from adapter registry scans.
pub const STARTING_ADAPTER_SNAPSHOT_DIR: &str = ".starting-adapter";

/// Open the replay log + lineage for a training run *before* the optimizer
/// step runs. Returns the effective seed so the trainer can apply it
/// consistently to RNG sources used during init.
///
/// Writes the request record (durable, fsynced) and `lineage.json` before
/// returning so a crash mid-step still leaves a recoverable trail.
pub fn open_replay_state(
    ctx: &ReplayContext,
    config_seed: Option<u64>,
    parent_adapter: Option<&str>,
    adapter_dir: &Path,
    adapter_name: &str,
) -> Result<(ReplayState, u64)> {
    open_replay_state_to(
        ctx,
        config_seed,
        parent_adapter,
        adapter_dir,
        adapter_dir,
        adapter_name,
    )
}

/// Staged-output variant of [`open_replay_state`]. Parent lineage resolves
/// from the prepared starting snapshot when rewriting that same adapter, or
/// from the durable registry otherwise. New replay state remains beneath
/// `output_adapter_dir` until the caller publishes it.
pub fn open_replay_state_to(
    ctx: &ReplayContext,
    config_seed: Option<u64>,
    parent_adapter: Option<&str>,
    adapter_dir: &Path,
    output_adapter_dir: &Path,
    adapter_name: &str,
) -> Result<(ReplayState, u64)> {
    let seed = config_seed.unwrap_or_else(|| rand::random());
    let kiln_commit = replay::kiln_commit();
    let submitted_at = chrono::Utc::now().to_rfc3339();
    let request = RequestRecord {
        request_id: ctx.request_id.clone(),
        kind: ctx.kind,
        request_body: ctx.request_body.clone(),
        seed,
        kiln_commit: kiln_commit.clone(),
        submitted_at: submitted_at.clone(),
    };

    let parent_lora = match parent_adapter {
        Some(name) => {
            let parent_dir = resolve_base_adapter_dir_from_roots(
                name,
                adapter_dir,
                output_adapter_dir,
                adapter_name,
            );
            let parent_lineage = replay::read_lineage(&parent_dir)
                .with_context(|| format!("reading parent lineage at {}", parent_dir.display()))?;
            Some(ParentLora {
                name: name.to_string(),
                replay_hash: parent_lineage.replay_hash,
            })
        }
        None => None,
    };

    let output_dir = output_adapter_dir.join(adapter_name);
    let log = ReplayLog::new(&output_dir)?;
    log.append_request(&request)?;

    let parent_hash = parent_lora.as_ref().map(|p| p.replay_hash.as_str());
    let replay_hash = replay::compute_replay_hash(parent_hash, &ctx.base_model, &[&request])?;

    let lineage = Lineage {
        schema_version: replay::LINEAGE_SCHEMA_VERSION,
        adapter_name: adapter_name.to_string(),
        base_model: ctx.base_model.clone(),
        parent_lora,
        kiln_commit,
        created_at: submitted_at,
        replay_hash,
    };
    replay::write_lineage(&output_dir, &lineage)?;

    Ok((
        ReplayState {
            log,
            lineage,
            request_id: ctx.request_id.clone(),
            started_at: std::time::Instant::now(),
        },
        seed,
    ))
}

/// Append an outcome record after the optimizer step finishes (or fails).
///
/// `result` is `Ok(final_loss)` on success, `Err(message)` on failure.
pub fn close_replay_state(state: ReplayState, result: Result<f64, String>) -> Result<()> {
    let elapsed = state.started_at.elapsed().as_secs_f64();
    let outcome = match result {
        Ok(loss) => OutcomeRecord {
            request_id: state.request_id,
            status: OutcomeStatus::Completed,
            final_loss: Some(loss),
            elapsed_secs: Some(elapsed),
            error: None,
        },
        Err(msg) => OutcomeRecord {
            request_id: state.request_id,
            status: OutcomeStatus::Failed,
            final_loss: None,
            elapsed_secs: Some(elapsed),
            error: Some(msg),
        },
    };
    state.log.append_outcome(&outcome)?;
    let _ = state.lineage; // lineage already written; kept for diagnostics if extended
    Ok(())
}
