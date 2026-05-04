//! One-shot HuggingFace model download for the Settings window.
//!
//! The desktop app otherwise requires users to shell out to
//! `huggingface-cli download ...` before the model path setting has
//! anything useful to point at. This module lets the frontend pass a
//! `repo_id` (and optional revision / token for gated models), list the
//! relevant files via the public HF tree API, and stream each file into
//! `app_data_dir()/models/<sanitized_repo>/` while emitting progress
//! events the settings modal renders as a progress bar.

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};
use tokio::fs;
use tokio::io::AsyncWriteExt;

pub const HF_DOWNLOAD_PROGRESS_EVENT: &str = "hf-download-progress";
pub const HF_DOWNLOAD_DONE_EVENT: &str = "hf-download-done";
pub const HF_DOWNLOAD_ERROR_EVENT: &str = "hf-download-error";

const HF_API_BASE: &str = "https://huggingface.co";
const USER_AGENT: &str = concat!("kiln-desktop/", env!("CARGO_PKG_VERSION"));

/// File suffixes / exact names we download. Anything else in the repo
/// (READMEs, .gitattributes, *.bin when safetensors exist, onnx, etc.)
/// is skipped — kiln only needs the weights, tokenizer, and config.
const WEIGHT_SUFFIXES: &[&str] = &[".safetensors"];
const TOKENIZER_EXACT: &[&str] = &[
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
];

/// Per-file IPC throttle: report no more than once every 250ms or once
/// per 64KB of bytes (whichever fires first). Matches the cadence the
/// installer module uses so both progress bars feel similar.
const PROGRESS_BYTES_INTERVAL: u64 = 64 * 1024;
const PROGRESS_TIME_INTERVAL_MS: u64 = 250;

#[derive(Debug, Clone, Deserialize)]
pub struct HfDownloadRequest {
    pub repo_id: String,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub token: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum HfDownloadProgress {
    Listing,
    Downloading {
        file: String,
        received: u64,
        total: u64,
        overall_received: u64,
        overall_total: u64,
        file_index: usize,
        file_count: usize,
    },
    Verifying,
    Done {
        path: String,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct HfDownloadDone {
    pub path: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct HfDownloadError {
    pub error: String,
}

#[derive(Debug, Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    entry_type: String,
    path: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    lfs: Option<LfsMeta>,
}

#[derive(Debug, Deserialize)]
struct LfsMeta {
    #[serde(default)]
    size: u64,
}

/// Replace `/` with `__` so the repo id is safe as a single directory
/// name on every supported OS. Everything else is kept verbatim; HF repo
/// ids are ASCII-safe already (letters, digits, `-`, `_`, `.`).
fn sanitize_repo_id(repo_id: &str) -> String {
    repo_id.replace('/', "__")
}

/// Target directory for a given repo id, rooted at `app_data_dir()`.
fn target_dir(app: &AppHandle, repo_id: &str) -> Result<PathBuf, String> {
    let base = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("app_data_dir unavailable: {}", e))?;
    Ok(base.join("models").join(sanitize_repo_id(repo_id)))
}

/// Quick "is this repo already here?" check: the directory exists and
/// contains the tokenizer plus at least one safetensors shard. We
/// intentionally do NOT auto-resume for MVP — the error message tells
/// the user to delete the folder to re-download.
async fn looks_already_downloaded(dir: &Path) -> bool {
    if !fs::try_exists(dir).await.unwrap_or(false) {
        return false;
    }
    let tokenizer = dir.join("tokenizer.json");
    if !fs::try_exists(&tokenizer).await.unwrap_or(false) {
        return false;
    }
    let mut rd = match fs::read_dir(dir).await {
        Ok(r) => r,
        Err(_) => return false,
    };
    while let Ok(Some(entry)) = rd.next_entry().await {
        if entry
            .file_name()
            .to_str()
            .map(|s| s.ends_with(".safetensors"))
            .unwrap_or(false)
        {
            return true;
        }
    }
    false
}

/// Decide whether a file path returned by the HF tree API is something
/// kiln needs. Keeps only weights + tokenizer/config. The subdir logic
/// (basename match) lets us still pick up files inside vendor subdirs
/// if the repo layout ever ships that way.
fn should_keep(path: &str) -> bool {
    let basename = Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    if TOKENIZER_EXACT.iter().any(|t| *t == basename) {
        return true;
    }
    if WEIGHT_SUFFIXES
        .iter()
        .any(|suffix| basename.ends_with(suffix))
    {
        return true;
    }
    false
}

fn build_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .connect_timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| format!("build reqwest client: {}", e))
}

fn bearer(token: &Option<String>) -> Option<String> {
    token
        .as_ref()
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .map(|t| format!("Bearer {}", t))
}

/// List the files in `{repo_id}` at `{revision}`. Filter to the ones
/// kiln actually needs and drop `*.bin` if any `*.safetensors` are
/// present (repos often ship both). Returns `(path, size)` pairs.
async fn list_files(
    client: &reqwest::Client,
    repo_id: &str,
    revision: &str,
    token: &Option<String>,
) -> Result<Vec<(String, u64)>, String> {
    let url = format!(
        "{}/api/models/{}/tree/{}?recursive=true",
        HF_API_BASE, repo_id, revision
    );
    let mut req = client.get(&url).header("Accept", "application/json");
    if let Some(auth) = bearer(token) {
        req = req.header("Authorization", auth);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| format!("list request failed: {}", e))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if status.as_u16() == 401 || status.as_u16() == 403 {
            return Err(format!(
                "HF API returned {} — the repo may be gated or the token is invalid.",
                status
            ));
        }
        if status.as_u16() == 404 {
            return Err(format!(
                "HF API returned 404 — repo '{}' at revision '{}' not found.",
                repo_id, revision
            ));
        }
        return Err(format!(
            "HF API returned HTTP {}: {}",
            status,
            body.chars().take(200).collect::<String>()
        ));
    }
    let entries: Vec<TreeEntry> = resp
        .json()
        .await
        .map_err(|e| format!("parse tree listing: {}", e))?;

    let mut kept: Vec<(String, u64)> = entries
        .into_iter()
        .filter(|e| e.entry_type == "file")
        .filter(|e| should_keep(&e.path))
        .map(|e| {
            let size = e.lfs.as_ref().map(|l| l.size).unwrap_or(0);
            let size = if size > 0 { size } else { e.size };
            (e.path, size)
        })
        .collect();

    // If both *.bin and *.safetensors exist, drop the *.bin shards.
    // (We currently only list safetensors so this is a no-op, but it's
    // a cheap guard for forward compatibility if we ever widen
    // WEIGHT_SUFFIXES.)
    let has_safetensors = kept.iter().any(|(p, _)| p.ends_with(".safetensors"));
    if has_safetensors {
        kept.retain(|(p, _)| !p.ends_with(".bin"));
    }

    if !kept.iter().any(|(p, _)| p.ends_with(".safetensors")) {
        return Err(format!(
            "No *.safetensors files found in '{}' at revision '{}'. kiln requires safetensors weights.",
            repo_id, revision
        ));
    }

    Ok(kept)
}

fn emit_progress(app: &AppHandle, p: HfDownloadProgress) {
    let _ = app.emit(HF_DOWNLOAD_PROGRESS_EVENT, p);
}

#[allow(clippy::too_many_arguments)]
async fn download_file(
    app: &AppHandle,
    client: &reqwest::Client,
    repo_id: &str,
    revision: &str,
    token: &Option<String>,
    rel_path: &str,
    file_size: u64,
    dest: &Path,
    file_index: usize,
    file_count: usize,
    overall_received_before: u64,
    overall_total: u64,
) -> Result<u64, String> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)
            .await
            .map_err(|e| format!("mkdir {}: {}", parent.display(), e))?;
    }

    let url = format!(
        "{}/{}/resolve/{}/{}",
        HF_API_BASE, repo_id, revision, rel_path
    );
    let mut req = client.get(&url);
    if let Some(auth) = bearer(token) {
        req = req.header("Authorization", auth);
    }
    let mut resp = req
        .send()
        .await
        .map_err(|e| format!("download {} failed: {}", rel_path, e))?;
    if !resp.status().is_success() {
        return Err(format!(
            "download {} returned HTTP {}",
            rel_path,
            resp.status()
        ));
    }

    let total_from_header = resp.content_length();
    // Prefer the listing-reported size when the HTTP header is missing
    // (the HF CDN sometimes omits Content-Length on redirects).
    let file_total = total_from_header.unwrap_or(file_size);

    let mut file = fs::File::create(dest)
        .await
        .map_err(|e| format!("create {}: {}", dest.display(), e))?;

    let mut received: u64 = 0;
    let mut last_reported_bytes: u64 = 0;
    let mut last_emit = Instant::now();

    // Emit an initial 0/total sample so the progress bar paints before
    // the first chunk arrives on slow links.
    emit_progress(
        app,
        HfDownloadProgress::Downloading {
            file: rel_path.to_string(),
            received: 0,
            total: file_total,
            overall_received: overall_received_before,
            overall_total,
            file_index,
            file_count,
        },
    );

    loop {
        let chunk = resp
            .chunk()
            .await
            .map_err(|e| format!("chunk read failed on {}: {}", rel_path, e))?;
        let Some(bytes) = chunk else { break };
        received = received.saturating_add(bytes.len() as u64);
        file.write_all(&bytes)
            .await
            .map_err(|e| format!("write {}: {}", dest.display(), e))?;

        let time_ok = last_emit.elapsed().as_millis() as u64 >= PROGRESS_TIME_INTERVAL_MS;
        let bytes_ok = received.saturating_sub(last_reported_bytes) >= PROGRESS_BYTES_INTERVAL;
        if time_ok || bytes_ok {
            last_reported_bytes = received;
            last_emit = Instant::now();
            emit_progress(
                app,
                HfDownloadProgress::Downloading {
                    file: rel_path.to_string(),
                    received,
                    total: file_total.max(received),
                    overall_received: overall_received_before.saturating_add(received),
                    overall_total: overall_total
                        .max(overall_received_before.saturating_add(received)),
                    file_index,
                    file_count,
                },
            );
        }
    }

    file.flush()
        .await
        .map_err(|e| format!("flush {}: {}", dest.display(), e))?;

    emit_progress(
        app,
        HfDownloadProgress::Downloading {
            file: rel_path.to_string(),
            received,
            total: file_total.max(received),
            overall_received: overall_received_before.saturating_add(received),
            overall_total: overall_total.max(overall_received_before.saturating_add(received)),
            file_index,
            file_count,
        },
    );

    Ok(received)
}

/// Full download pipeline for one HF repo. Emits progress events and
/// returns the final target directory. Callers wire the result into
/// `settings.model_path` and surface the `Done` event to the modal.
pub async fn download_hf_model(app: AppHandle, req: HfDownloadRequest) -> Result<PathBuf, String> {
    let repo_id = req.repo_id.trim().to_string();
    if repo_id.is_empty() {
        return Err("Repo id is required (e.g. Qwen/Qwen3.5-4B).".into());
    }
    let revision = req
        .revision
        .as_ref()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "main".to_string());

    let dir = target_dir(&app, &repo_id)?;
    if looks_already_downloaded(&dir).await {
        return Err(format!(
            "Model already downloaded at {}. Delete the folder to re-download.",
            dir.display()
        ));
    }
    fs::create_dir_all(&dir)
        .await
        .map_err(|e| format!("mkdir {}: {}", dir.display(), e))?;

    emit_progress(&app, HfDownloadProgress::Listing);

    let client = build_client()?;
    let files = list_files(&client, &repo_id, &revision, &req.token).await?;

    let overall_total: u64 = files.iter().map(|(_, s)| *s).sum();
    let file_count = files.len();
    let mut overall_received: u64 = 0;

    for (idx, (rel, size)) in files.iter().enumerate() {
        let dest = dir.join(rel);
        let got = download_file(
            &app,
            &client,
            &repo_id,
            &revision,
            &req.token,
            rel,
            *size,
            &dest,
            idx + 1,
            file_count,
            overall_received,
            overall_total,
        )
        .await?;
        overall_received = overall_received.saturating_add(got);
    }

    emit_progress(&app, HfDownloadProgress::Verifying);

    // Basic post-check: the usual kiln-required trio landed.
    let tokenizer_ok = fs::try_exists(dir.join("tokenizer.json"))
        .await
        .unwrap_or(false);
    if !tokenizer_ok {
        return Err("Download completed but tokenizer.json is missing.".into());
    }

    emit_progress(
        &app,
        HfDownloadProgress::Done {
            path: dir.display().to_string(),
        },
    );

    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_repo_id_replaces_slash() {
        assert_eq!(sanitize_repo_id("Qwen/Qwen3.5-4B"), "Qwen__Qwen3.5-4B");
        assert_eq!(sanitize_repo_id("no-slash"), "no-slash");
        assert_eq!(sanitize_repo_id("org/sub/name"), "org__sub__name");
    }

    #[test]
    fn should_keep_filters_tokenizer_and_weights() {
        assert!(should_keep("model-00001-of-00002.safetensors"));
        assert!(should_keep("tokenizer.json"));
        assert!(should_keep("config.json"));
        assert!(should_keep("generation_config.json"));
        assert!(should_keep("chat_template.jinja"));
        assert!(should_keep("model.safetensors.index.json"));
        assert!(should_keep("special_tokens_map.json"));
        assert!(should_keep("subdir/tokenizer.json"));

        assert!(!should_keep("README.md"));
        assert!(!should_keep(".gitattributes"));
        assert!(!should_keep("model.onnx"));
        assert!(!should_keep("pytorch_model.bin"));
    }

    #[test]
    fn bearer_normalizes_empty_tokens() {
        assert_eq!(bearer(&None), None);
        assert_eq!(bearer(&Some("".to_string())), None);
        assert_eq!(bearer(&Some("   ".to_string())), None);
        assert_eq!(
            bearer(&Some("hf_abc".to_string())),
            Some("Bearer hf_abc".to_string())
        );
        assert_eq!(
            bearer(&Some("  hf_abc ".to_string())),
            Some("Bearer hf_abc".to_string())
        );
    }
}
