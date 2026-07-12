//! Server-owned, immutable HF/TRL handoff exports.

use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Path as AxumPath, State, rejection::JsonRejection};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use flate2::Compression;
use flate2::write::GzEncoder;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use kiln_train::{
    HF_TRL_BUNDLE_SUFFIX, HF_TRL_SFT_ENVIRONMENT_LOCK, HF_TRL_SFT_REFERENCE_SCRIPT,
    HfTrlExportManifestV1, HfTrlInputAdapterSource, HfTrlSftBundleInput, SftExample,
    SftInvalidRowPolicy, SftPreparedDataset, read_hf_trl_export_manifest,
    verify_hf_trl_export_bundle, write_hf_trl_sft_bundle,
};

use crate::error::ApiError;
use crate::state::AppState;

const EXPORT_REGISTRY_DIR: &str = ".hf_trl_exports";
const MAX_EXPORT_NAME_BYTES: usize = 128;
const MAX_SERVER_EXPORTS: usize = 256;
const MAX_INLINE_EXPORT_BODY_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SftExportRequest {
    name: String,
    #[serde(default)]
    examples: Vec<SftExample>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dataset_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dataset: Option<String>,
    #[serde(default)]
    invalid_row_policy: SftInvalidRowPolicy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    input_adapter: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    split_manifest: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct ExportSummary {
    name: String,
    task: kiln_train::HfTrlTask,
    export_sha256: String,
    source_name: String,
    row_count: u64,
    ordered_corpus_sha256: String,
    input_adapter: Option<String>,
    bundle_filename: String,
    download_url: String,
}

impl ExportSummary {
    fn from_manifest(name: &str, manifest: &HfTrlExportManifestV1) -> Self {
        Self {
            name: name.to_string(),
            task: manifest.task,
            export_sha256: manifest.export_sha256.clone(),
            source_name: manifest.data.source_name.clone(),
            row_count: manifest.data.row_count,
            ordered_corpus_sha256: manifest.data.ordered_corpus_sha256.clone(),
            input_adapter: manifest
                .input_adapter
                .as_ref()
                .map(|adapter| adapter.name.clone()),
            bundle_filename: bundle_filename(name),
            download_url: format!("/v1/train/hf/exports/{name}/download"),
        }
    }
}

#[derive(Debug, Serialize)]
struct ExportList {
    data: Vec<ExportSummary>,
}

#[derive(Debug, Serialize)]
struct ExportDetail {
    summary: ExportSummary,
    manifest: HfTrlExportManifestV1,
}

#[derive(Debug, Serialize)]
struct DeleteExportResponse {
    status: &'static str,
    name: String,
}

fn validate_export_name(name: &str) -> Result<(), ApiError> {
    if name.is_empty()
        || name.len() > MAX_EXPORT_NAME_BYTES
        || !name.bytes().enumerate().all(|(index, byte)| {
            byte.is_ascii_alphanumeric() || (index > 0 && matches!(byte, b'-' | b'_'))
        })
    {
        return Err(ApiError::hf_trl_invalid_request(format!(
            "export name {name:?} must be 1..={MAX_EXPORT_NAME_BYTES} ASCII bytes, start with an alphanumeric character, and contain only alphanumerics, '-' or '_'"
        )));
    }
    Ok(())
}

fn bundle_filename(name: &str) -> String {
    format!("{name}{HF_TRL_BUNDLE_SUFFIX}")
}

fn export_etag(export_sha256: &str) -> Result<HeaderValue, ApiError> {
    HeaderValue::from_str(&format!("\"{export_sha256}\""))
        .map_err(|error| ApiError::hf_trl_export_failed(format!("export ETag: {error}")))
}

fn parse_delete_if_match(headers: &HeaderMap) -> Result<Option<String>, ApiError> {
    let mut values = headers.get_all(header::IF_MATCH).iter();
    let Some(value) = values.next() else {
        return Ok(None);
    };
    if values.next().is_some() {
        return Err(ApiError::hf_trl_invalid_request(
            "DELETE accepts at most one If-Match header",
        ));
    }
    let value = value
        .to_str()
        .map_err(|_| ApiError::hf_trl_invalid_request("If-Match must be visible ASCII"))?;
    let digest = value
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .ok_or_else(|| {
            ApiError::hf_trl_invalid_request(
                "If-Match must be one strong quoted export SHA-256 entity tag",
            )
        })?;
    if digest.len() != "sha256:".len() + 64
        || !digest.starts_with("sha256:")
        || !digest["sha256:".len()..]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(ApiError::hf_trl_invalid_request(
            "If-Match must contain a lowercase sha256:<64-hex> export identity",
        ));
    }
    Ok(Some(digest.to_string()))
}

fn registry_path(adapter_dir: &Path) -> PathBuf {
    adapter_dir.join(EXPORT_REGISTRY_DIR)
}

fn bundle_path(adapter_dir: &Path, name: &str) -> PathBuf {
    registry_path(adapter_dir).join(bundle_filename(name))
}

fn ensure_private_registry(adapter_dir: &Path) -> io::Result<PathBuf> {
    let root = registry_path(adapter_dir);
    match fs::symlink_metadata(&root) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "HF/TRL export registry is not a real directory: {}",
                        root.display()
                    ),
                ));
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            let mut builder = fs::DirBuilder::new();
            #[cfg(unix)]
            builder.mode(0o700);
            builder.create(&root)?;
        }
        Err(error) => return Err(error),
    }
    #[cfg(unix)]
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700))?;
    let root = root.canonicalize()?;
    cleanup_crash_residue(&root)?;
    Ok(root)
}

fn cleanup_crash_residue(root: &Path) -> io::Result<()> {
    let mut removed = false;
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if !(name.starts_with(".deleting-")
            || (name.starts_with('.') && name.contains(".incomplete-")))
        {
            continue;
        }
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "HF/TRL crash residue is not a real directory: {}",
                    path.display()
                ),
            ));
        }
        fs::remove_dir_all(path)?;
        removed = true;
    }
    if removed {
        sync_directory(root)?;
    }
    Ok(())
}

fn existing_registry(adapter_dir: &Path) -> io::Result<Option<PathBuf>> {
    let root = registry_path(adapter_dir);
    match fs::symlink_metadata(&root) {
        Ok(metadata) if metadata.file_type().is_dir() && !metadata.file_type().is_symlink() => {
            root.canonicalize().map(Some)
        }
        Ok(_) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "HF/TRL export registry is not a real directory: {}",
                root.display()
            ),
        )),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

fn published_export_count(adapter_dir: &Path) -> io::Result<usize> {
    let Some(root) = existing_registry(adapter_dir)? else {
        return Ok(0);
    };
    let mut count = 0usize;
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let is_export = entry.file_type()?.is_dir()
            && entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.ends_with(HF_TRL_BUNDLE_SUFFIX));
        if is_export {
            count = count.saturating_add(1);
        }
    }
    Ok(count)
}

fn prepare_sft_source(
    state: &AppState,
    mut request: SftExportRequest,
) -> Result<PreparedExport, ApiError> {
    validate_export_name(&request.name)?;
    request.dataset_path = request
        .dataset_path
        .take()
        .map(|path| path.trim().to_string())
        .filter(|path| !path.is_empty());
    request.dataset = request.dataset.take().map(|name| name.trim().to_string());
    if request.dataset.as_deref() == Some("") {
        return Err(ApiError::hf_trl_invalid_request(
            "dataset must not be blank",
        ));
    }
    if request.input_adapter.as_deref().is_some_and(str::is_empty) {
        return Err(ApiError::hf_trl_invalid_request(
            "input_adapter must not be blank",
        ));
    }

    let source_count = usize::from(!request.examples.is_empty())
        + usize::from(request.dataset_path.is_some())
        + usize::from(request.dataset.is_some());
    if source_count != 1 {
        return Err(ApiError::hf_trl_invalid_request(
            "SFT export must use exactly one of examples, dataset_path, or dataset",
        ));
    }
    if let Some(adapter) = request.input_adapter.as_deref() {
        if adapter.trim() != adapter {
            return Err(ApiError::hf_trl_invalid_request(format!(
                "input_adapter {adapter:?} must not contain leading or trailing whitespace"
            )));
        }
        super::adapters::validate_adapter_name(adapter).map_err(|_| {
            ApiError::hf_trl_invalid_request(format!(
                "input_adapter {adapter:?} must be a path-safe adapter name"
            ))
        })?;
    }
    let split_manifest = request
        .split_manifest
        .map(|value| {
            if !value.is_object() {
                return Err(ApiError::hf_trl_invalid_request(
                    "split_manifest must be a JSON object",
                ));
            }
            serde_json::to_vec_pretty(&value).map_err(|error| {
                ApiError::hf_trl_invalid_request(format!(
                    "split_manifest cannot be serialized: {error}"
                ))
            })
        })
        .transpose()?;

    let prepared = if let Some(dataset_name) = request.dataset {
        if dataset_name == "corrections:active" {
            let (_, examples) =
                super::corrections::CorrectionsStore::for_state(state).trainable_rows();
            if examples.is_empty() {
                return Err(ApiError::hf_trl_invalid_request(
                    "corrections:active has no trainable rows",
                ));
            }
            kiln_train::prepare_sft_examples(
                examples,
                state.tokenizer.as_ref(),
                request.invalid_row_policy,
                "corrections",
                None,
            )
            .map_err(|error| {
                ApiError::hf_trl_invalid_request(format!("invalid corrections SFT rows: {error:#}"))
            })?
        } else {
            let registry = state
                .dataset_registry
                .as_ref()
                .ok_or_else(ApiError::dataset_registry_unavailable)?;
            let dataset_dir = registry
                .dataset_dir(&dataset_name)
                .map_err(|error| match error {
                    crate::eval::DatasetError::NotFound(_) => {
                        ApiError::dataset_not_found(&dataset_name)
                    }
                    crate::eval::DatasetError::InvalidName(_) => {
                        ApiError::dataset_invalid(&dataset_name)
                    }
                    other => ApiError::dataset_invalid(format!("{other}")),
                })?;
            let path = dataset_dir.join("data.jsonl");
            if !path.is_file() {
                return Err(ApiError::dataset_not_found(&dataset_name));
            }
            crate::sft_dataset::prepare_sft_jsonl(
                &path,
                state.tokenizer.as_ref(),
                request.invalid_row_policy,
                "named_dataset",
                Some(dataset_name.clone()),
            )
            .map_err(|error| {
                ApiError::hf_trl_invalid_request(format!(
                    "invalid SFT dataset {dataset_name:?}: {error:#}"
                ))
            })?
        }
    } else if let Some(path) = request.dataset_path {
        let path = PathBuf::from(path);
        crate::sft_dataset::prepare_sft_jsonl(
            &path,
            state.tokenizer.as_ref(),
            request.invalid_row_policy,
            "dataset_path",
            Some(path.display().to_string()),
        )
        .map_err(|error| {
            ApiError::hf_trl_invalid_request(format!(
                "invalid SFT dataset_path '{}': {error:#}",
                path.display()
            ))
        })?
    } else {
        kiln_train::prepare_sft_examples(
            request.examples,
            state.tokenizer.as_ref(),
            request.invalid_row_policy,
            "inline",
            None,
        )
        .map_err(|error| ApiError::hf_trl_invalid_request(format!("invalid SFT rows: {error:#}")))?
    };

    Ok(PreparedExport {
        name: request.name,
        input_adapter: request.input_adapter,
        split_manifest,
        prepared,
    })
}

struct PreparedExport {
    name: String,
    input_adapter: Option<String>,
    split_manifest: Option<Vec<u8>>,
    prepared: SftPreparedDataset,
}

async fn create_sft_export(
    State(state): State<AppState>,
    payload: Result<Json<SftExportRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let request = payload.map(|Json(request)| request).map_err(|error| {
        ApiError::hf_trl_invalid_request(format!("invalid SFT export JSON: {}", error.body_text()))
    })?;
    validate_export_name(&request.name)?;
    if state.shutdown.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }
    if fs::symlink_metadata(bundle_path(&state.adapter_dir, &request.name)).is_ok() {
        return Err(ApiError::hf_trl_export_exists(&request.name));
    }
    let base_weights = state
        .base_weight_shard_manifest
        .as_ref()
        .cloned()
        .ok_or_else(|| ApiError::hf_trl_unavailable("resident base-weight identity is absent"))?;
    let provenance = state
        .execution_provenance
        .as_ref()
        .cloned()
        .ok_or_else(|| ApiError::hf_trl_unavailable("startup execution provenance is absent"))?;
    base_weights.validate().map_err(|error| {
        ApiError::hf_trl_unavailable(format!("invalid base-weight identity: {error}"))
    })?;
    provenance.validate().map_err(|error| {
        ApiError::hf_trl_unavailable(format!("invalid execution provenance: {error}"))
    })?;
    let preparation_state = state.clone();
    let prepared =
        tokio::task::spawn_blocking(move || prepare_sft_source(&preparation_state, request))
            .await
            .map_err(|error| {
                ApiError::hf_trl_export_failed(format!("admission worker panicked: {error}"))
            })??;

    let export_guard = state.hf_trl_export_lock.clone().lock_owned().await;
    let export_count =
        published_export_count(&state.adapter_dir).map_err(ApiError::hf_trl_export_failed)?;
    if export_count >= MAX_SERVER_EXPORTS {
        return Err(ApiError::hf_trl_export_capacity(MAX_SERVER_EXPORTS));
    }
    let adapter_guard = if prepared.input_adapter.is_some() {
        Some(state.adapter_mutation_lock.clone().lock_owned().await)
    } else {
        None
    };
    let adapter_dir = state.adapter_dir.clone();
    let target = bundle_path(&adapter_dir, &prepared.name);
    if fs::symlink_metadata(&target).is_ok() {
        return Err(ApiError::hf_trl_export_exists(&prepared.name));
    }
    if let Some(adapter) = prepared.input_adapter.as_deref() {
        let path = adapter_dir.join(adapter);
        if !path.is_dir() {
            return Err(ApiError::adapter_not_found(adapter));
        }
    }

    let name = prepared.name.clone();
    let served_model_id = state.served_model_id.clone();
    let model_config = state.model_config.clone();
    let tokenizer = state.tokenizer.clone();
    let input_adapter = prepared.input_adapter.clone();
    let manifest = tokio::task::spawn_blocking(move || {
        let _export_guard = export_guard;
        let _adapter_guard = adapter_guard;
        ensure_private_registry(&adapter_dir)?;
        // Keep the owned adapter path alive for the complete synchronous copy.
        let adapter_path = input_adapter
            .as_ref()
            .map(|adapter| adapter_dir.join(adapter));
        let adapter_source =
            input_adapter
                .as_ref()
                .zip(adapter_path.as_ref())
                .map(|(adapter, path)| HfTrlInputAdapterSource {
                    name: adapter.as_str(),
                    directory: path.as_path(),
                });
        write_hf_trl_sft_bundle(
            &target,
            HfTrlSftBundleInput {
                served_model_id: &served_model_id,
                model_config: &model_config,
                tokenizer: tokenizer.as_ref(),
                base_weight_shard_manifest: base_weights.as_ref(),
                source_execution_provenance: provenance.as_ref(),
                prepared: &prepared.prepared,
                reference_script: HF_TRL_SFT_REFERENCE_SCRIPT,
                environment_lock: HF_TRL_SFT_ENVIRONMENT_LOCK,
                split_manifest: prepared.split_manifest.as_deref(),
                input_adapter: adapter_source,
            },
        )
    })
    .await
    .map_err(|error| ApiError::hf_trl_export_failed(format!("export worker panicked: {error}")))?
    .map_err(|error| ApiError::hf_trl_export_failed(format!("{error:#}")))?;

    tracing::info!(export = %name, export_sha256 = %manifest.export_sha256, rows = manifest.data.row_count, "published immutable HF/TRL SFT export");
    let etag = export_etag(&manifest.export_sha256)?;
    Ok((
        StatusCode::CREATED,
        [(header::ETAG, etag)],
        Json(ExportSummary::from_manifest(&name, &manifest)),
    )
        .into_response())
}

fn list_exports_sync(adapter_dir: &Path) -> anyhow::Result<Vec<ExportSummary>> {
    let Some(root) = existing_registry(adapter_dir)? else {
        return Ok(Vec::new());
    };
    let mut entries = fs::read_dir(&root)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    let mut exports = Vec::new();
    for entry in entries {
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)?;
        if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
            continue;
        }
        let Some(filename) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        let Some(name) = filename.strip_suffix(HF_TRL_BUNDLE_SUFFIX) else {
            continue;
        };
        validate_export_name(name).map_err(|error| anyhow::anyhow!(error.message))?;
        let manifest = read_hf_trl_export_manifest(&path)?;
        exports.push(ExportSummary::from_manifest(name, &manifest));
    }
    Ok(exports)
}

async fn list_exports(State(state): State<AppState>) -> Result<Json<ExportList>, ApiError> {
    let guard = state.hf_trl_export_lock.clone().lock_owned().await;
    let adapter_dir = state.adapter_dir.clone();
    let data = tokio::task::spawn_blocking(move || {
        let _guard = guard;
        list_exports_sync(&adapter_dir)
    })
    .await
    .map_err(|error| ApiError::hf_trl_export_failed(format!("list worker panicked: {error}")))?
    .map_err(|error| ApiError::hf_trl_export_failed(format!("{error:#}")))?;
    Ok(Json(ExportList { data }))
}

async fn get_export(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
) -> Result<Response, ApiError> {
    validate_export_name(&name)?;
    let guard = state.hf_trl_export_lock.clone().lock_owned().await;
    let path = bundle_path(&state.adapter_dir, &name);
    if !path.exists() {
        return Err(ApiError::hf_trl_export_not_found(&name));
    }
    let manifest = tokio::task::spawn_blocking(move || {
        let _guard = guard;
        verify_hf_trl_export_bundle(&path)
    })
    .await
    .map_err(|error| ApiError::hf_trl_export_failed(format!("verify worker panicked: {error}")))?
    .map_err(|error| ApiError::hf_trl_export_failed(format!("{error:#}")))?;
    let etag = export_etag(&manifest.export_sha256)?;
    Ok((
        [(header::ETAG, etag)],
        Json(ExportDetail {
            summary: ExportSummary::from_manifest(&name, &manifest),
            manifest,
        }),
    )
        .into_response())
}

fn append_bundle_to_tar<W: io::Write>(
    archive: &mut tar::Builder<W>,
    root: &Path,
    directory: &Path,
    prefix: &Path,
) -> io::Result<()> {
    let mut entries = fs::read_dir(directory)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("HF/TRL bundle contains a symlink: {}", path.display()),
            ));
        }
        let relative = path
            .strip_prefix(root)
            .expect("walk remains inside export root");
        let archive_path = prefix.join(relative);
        if metadata.file_type().is_dir() {
            append_bundle_to_tar(archive, root, &path, prefix)?;
        } else if metadata.file_type().is_file() {
            let mut file = File::open(&path)?;
            archive.append_file(archive_path, &mut file)?;
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("HF/TRL bundle contains a special file: {}", path.display()),
            ));
        }
    }
    Ok(())
}

async fn download_export(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
) -> Result<Response, ApiError> {
    validate_export_name(&name)?;
    let guard = state.hf_trl_export_lock.clone().lock_owned().await;
    let path = bundle_path(&state.adapter_dir, &name);
    if !path.exists() {
        return Err(ApiError::hf_trl_export_not_found(&name));
    }
    let (manifest, guard) = tokio::task::spawn_blocking(move || {
        let manifest = verify_hf_trl_export_bundle(&path)?;
        Ok::<_, anyhow::Error>((manifest, guard))
    })
    .await
    .map_err(|error| ApiError::hf_trl_export_failed(format!("verify worker panicked: {error}")))?
    .map_err(|error| ApiError::hf_trl_export_failed(format!("{error:#}")))?;

    let path = bundle_path(&state.adapter_dir, &name);
    let (tx, rx) = mpsc::channel::<Result<Vec<u8>, io::Error>>(8);
    let archive_name = bundle_filename(&name);
    let archive_prefix = archive_name.clone();
    tokio::task::spawn_blocking(move || {
        let _guard = guard;
        struct ChannelWriter {
            sender: mpsc::Sender<Result<Vec<u8>, io::Error>>,
        }
        impl io::Write for ChannelWriter {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                let length = bytes.len();
                self.sender.blocking_send(Ok(bytes.to_vec())).map_err(|_| {
                    io::Error::new(io::ErrorKind::BrokenPipe, "download client disconnected")
                })?;
                Ok(length)
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let writer = ChannelWriter { sender: tx.clone() };
        let encoder = GzEncoder::new(writer, Compression::default());
        let mut archive = tar::Builder::new(encoder);
        let result = append_bundle_to_tar(&mut archive, &path, &path, Path::new(&archive_prefix))
            .and_then(|()| archive.into_inner())
            .and_then(GzEncoder::finish)
            .map(|_| ());
        if let Err(error) = result {
            let _ = tx.blocking_send(Err(error));
        }
    });

    let disposition =
        HeaderValue::from_str(&format!("attachment; filename=\"{archive_name}.tar.gz\""))
            .map_err(|error| ApiError::hf_trl_export_failed(format!("download header: {error}")))?;
    let etag = export_etag(&manifest.export_sha256)?;
    tracing::info!(export = %name, export_sha256 = %manifest.export_sha256, "streaming verified HF/TRL export");
    Ok((
        StatusCode::OK,
        [
            (
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/gzip"),
            ),
            (header::CONTENT_DISPOSITION, disposition),
            (
                header::CACHE_CONTROL,
                HeaderValue::from_static("private, no-store"),
            ),
            (header::ETAG, etag),
        ],
        Body::from_stream(ReceiverStream::new(rx)),
    )
        .into_response())
}

fn sync_directory(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

async fn delete_export(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
    headers: HeaderMap,
) -> Result<Json<DeleteExportResponse>, ApiError> {
    validate_export_name(&name)?;
    let expected_export_sha256 = parse_delete_if_match(&headers)?;
    let guard = state.hf_trl_export_lock.clone().lock_owned().await;
    let adapter_dir = state.adapter_dir.clone();
    let name_for_task = name.clone();
    tokio::task::spawn_blocking(move || -> Result<(), ApiError> {
        let _guard = guard;
        let root = existing_registry(&adapter_dir)
            .map_err(ApiError::hf_trl_export_failed)?
            .ok_or_else(|| ApiError::hf_trl_export_not_found(&name_for_task))?;
        let target = root.join(bundle_filename(&name_for_task));
        let metadata = fs::symlink_metadata(&target).map_err(|error| {
            if error.kind() == io::ErrorKind::NotFound {
                ApiError::hf_trl_export_not_found(&name_for_task)
            } else {
                ApiError::hf_trl_export_failed(error)
            }
        })?;
        if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
            return Err(ApiError::hf_trl_export_failed(
                "HF/TRL export target is not a real directory",
            ));
        }
        if let Some(expected) = expected_export_sha256.as_deref() {
            let manifest = read_hf_trl_export_manifest(&target)
                .map_err(|error| ApiError::hf_trl_export_failed(format!("{error:#}")))?;
            if manifest.export_sha256 != expected {
                return Err(ApiError::hf_trl_export_precondition_failed(
                    &name_for_task,
                    expected,
                    manifest.export_sha256,
                ));
            }
        }
        let tombstone = root.join(format!(".deleting-{}-{}", name_for_task, Uuid::new_v4()));
        fs::rename(&target, &tombstone).map_err(ApiError::hf_trl_export_failed)?;
        sync_directory(&root).map_err(ApiError::hf_trl_export_failed)?;
        fs::remove_dir_all(&tombstone).map_err(ApiError::hf_trl_export_failed)?;
        sync_directory(&root).map_err(ApiError::hf_trl_export_failed)
    })
    .await
    .map_err(|error| {
        ApiError::hf_trl_export_failed(format!("delete worker panicked: {error}"))
    })??;
    Ok(Json(DeleteExportResponse {
        status: "deleted",
        name,
    }))
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route(
            "/v1/train/hf/sft/exports",
            post(create_sft_export).layer(DefaultBodyLimit::max(MAX_INLINE_EXPORT_BODY_BYTES)),
        )
        .route("/v1/train/hf/exports", get(list_exports))
        .route(
            "/v1/train/hf/exports/{name}",
            get(get_export).delete(delete_export),
        )
        .route("/v1/train/hf/exports/{name}/download", get(download_export))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use flate2::read::GzDecoder;
    use kiln_core::config::ModelConfig;
    use kiln_core::execution_provenance::{
        ExecutionBackendIdentity, ExecutionBuildIdentity, ExecutionConfigurationIdentity,
        ExecutionKernelIdentity, ExecutionModelIdentity, ExecutionPrecisionIdentity,
        ExecutionProvenanceV1,
    };
    use kiln_core::model_provenance::{BaseWeightShardIdentity, BaseWeightShardManifest};
    use kiln_core::tokenizer::KilnTokenizer;
    use kiln_model::engine::MockEngine;
    use kiln_scheduler::{Scheduler, SchedulerConfig};
    use tower::ServiceExt;

    use super::*;

    struct TestState {
        state: AppState,
        _directory: tempfile::TempDir,
    }

    fn test_state(with_identity: bool) -> TestState {
        let directory = tempfile::tempdir().unwrap();
        let adapter_dir = directory.path().join("adapters");
        fs::create_dir(&adapter_dir).unwrap();
        let model_config = ModelConfig::qwen3_5_4b();
        let tokenizer = KilnTokenizer::from_bytes(
            br#"{
                "version": "1.0",
                "model": {
                    "type": "WordLevel",
                    "vocab": {"[UNK]": 0, "a": 1, "b": 2},
                    "unk_token": "[UNK]"
                },
                "pre_tokenizer": {"type": "Whitespace"}
            }"#,
        )
        .unwrap()
        .with_chat_template(
            include_str!("../../../kiln-core/test_fixtures/qwen35_4b_chat_template.jinja")
                .to_string(),
        );
        let scheduler = Scheduler::new(
            SchedulerConfig {
                max_batch_tokens: 1024,
                max_batch_size: 1,
                block_size: 16,
                ..Default::default()
            },
            32,
        );
        let engine = MockEngine::new(model_config.clone());
        let mut state = AppState::new_mock(
            model_config.clone(),
            scheduler,
            Arc::new(engine),
            tokenizer,
            60,
            "test/qwen".to_string(),
        );
        state.adapter_dir = adapter_dir;

        if with_identity {
            let base_weights = BaseWeightShardManifest::new(vec![
                BaseWeightShardIdentity::new(
                    "model.safetensors",
                    4,
                    kiln_core::config_hashes::sha256_bytes(b"base"),
                )
                .unwrap(),
            ])
            .unwrap();
            let hash = |bytes: &[u8]| kiln_core::config_hashes::sha256_bytes(bytes);
            let model_identity = ExecutionModelIdentity {
                model_config_sha256: kiln_core::config_hashes::sha256_json_serializable(
                    &model_config,
                )
                .unwrap(),
                tokenizer_vocab_sha256: state.tokenizer.vocab_identity_sha256(),
                tokenizer_config_sha256: state.tokenizer.tokenizer_config_sha256().unwrap(),
                chat_template_sha256: state.tokenizer.chat_template_sha256(),
                training_chat_template_sha256: state.tokenizer.training_chat_template_sha256(),
            };
            let provenance = ExecutionProvenanceV1::new(
                ExecutionBackendIdentity {
                    name: "test".to_string(),
                    device: "cpu".to_string(),
                    numerical_runtime_sha256: hash(b"runtime"),
                },
                ExecutionBuildIdentity {
                    package_version: "test".to_string(),
                    target: "test".to_string(),
                    executable_sha256: hash(b"executable"),
                    git_commit: None,
                    source_tree_sha256: None,
                    source_dirty: None,
                },
                model_identity,
                ExecutionPrecisionIdentity {
                    inference_dtype: "f32".to_string(),
                    training_policy: "f32".to_string(),
                },
                ExecutionKernelIdentity::new(
                    BTreeMap::from([("test".to_string(), "v1".to_string())]),
                    Vec::new(),
                )
                .unwrap(),
                ExecutionConfigurationIdentity {
                    effective_server_config_sha256: hash(b"config"),
                    effective_environment_sha256: hash(b"environment"),
                },
            )
            .unwrap();
            state.base_weight_shard_manifest = Some(Arc::new(base_weights));
            state.execution_provenance = Some(Arc::new(provenance));
        }

        TestState {
            state,
            _directory: directory,
        }
    }

    fn json_request(method: &str, uri: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap()
    }

    async fn response_json(response: Response) -> serde_json::Value {
        serde_json::from_slice(
            &to_bytes(response.into_body(), 8 * 1024 * 1024)
                .await
                .unwrap(),
        )
        .unwrap()
    }

    fn export_request(name: &str) -> serde_json::Value {
        serde_json::json!({
            "name": name,
            "examples": [{
                "messages": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"}
                ]
            }],
            "invalid_row_policy": "fail",
            "input_adapter": "source.adapter",
            "split_manifest": {"schema": "test.split.v1", "train": [0]}
        })
    }

    #[test]
    fn delete_if_match_requires_one_strong_lowercase_export_digest() {
        let digest = format!("sha256:{}", "a".repeat(64));
        let mut headers = HeaderMap::new();
        headers.insert(
            header::IF_MATCH,
            HeaderValue::from_str(&format!("\"{digest}\"")).unwrap(),
        );
        assert_eq!(parse_delete_if_match(&headers).unwrap(), Some(digest));

        for invalid in [
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "W/\"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"",
            "\"sha256:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\"",
            "\"sha256:short\"",
        ] {
            headers.insert(header::IF_MATCH, HeaderValue::from_static(invalid));
            assert!(
                parse_delete_if_match(&headers).is_err(),
                "accepted {invalid}"
            );
        }
        headers.append(
            header::IF_MATCH,
            HeaderValue::from_static(
                "\"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\"",
            ),
        );
        assert!(parse_delete_if_match(&headers).is_err());
    }

    #[tokio::test]
    async fn create_list_download_and_delete_round_trip() {
        let fixture = test_state(true);
        let source_adapter = fixture.state.adapter_dir.join("source.adapter");
        fs::create_dir(&source_adapter).unwrap();
        fs::write(
            source_adapter.join("adapter_config.json"),
            b"{\"peft_type\":\"LORA\"}",
        )
        .unwrap();
        fs::write(source_adapter.join("adapter_model.safetensors"), b"adapter").unwrap();
        let app = routes().with_state(fixture.state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server_app = app.clone();
        let server = tokio::spawn(async move {
            axum::serve(listener, server_app).await.unwrap();
        });

        let cli_output_dir = tempfile::tempdir().unwrap();
        let cli_dataset = cli_output_dir.path().join("rows.jsonl");
        fs::write(
            &cli_dataset,
            r#"{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]}
"#,
        )
        .unwrap();
        let cli_output = cli_output_dir.path().join("cli_portable.tar.gz");
        crate::hf_train_cli::run_export_sft(crate::hf_train_cli::ExportSftOptions {
            url: format!("http://{address}"),
            file: Some(cli_dataset.to_string_lossy().into_owned()),
            dataset: None,
            name: "cli_portable".to_string(),
            output: Some(cli_output.clone()),
            invalid_row_policy: "fail".to_string(),
            input_adapter: None,
            split_manifest: None,
            keep_server_copy: false,
        })
        .await
        .unwrap();
        assert!(cli_output.is_file());
        assert!(!bundle_path(&fixture.state.adapter_dir, "cli_portable").exists());
        let collision =
            crate::hf_train_cli::run_export_sft(crate::hf_train_cli::ExportSftOptions {
                url: format!("http://{address}"),
                file: Some(cli_dataset.to_string_lossy().into_owned()),
                dataset: None,
                name: "cli_portable".to_string(),
                output: Some(cli_output.clone()),
                invalid_row_policy: "fail".to_string(),
                input_adapter: None,
                split_manifest: None,
                keep_server_copy: false,
            })
            .await
            .unwrap_err();
        assert!(collision.to_string().contains("refusing to overwrite"));

        let created = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/v1/train/hf/sft/exports",
                export_request("portable_run"),
            ))
            .await
            .unwrap();
        assert_eq!(created.status(), StatusCode::CREATED);
        let created_etag = created.headers()[header::ETAG].clone();
        let created = response_json(created).await;
        assert_eq!(created["name"], "portable_run");
        assert_eq!(created["row_count"], 1);
        assert_eq!(created["input_adapter"], "source.adapter");
        assert_eq!(
            created["download_url"],
            "/v1/train/hf/exports/portable_run/download"
        );
        assert_eq!(
            created_etag,
            format!("\"{}\"", created["export_sha256"].as_str().unwrap())
        );

        let bundle = bundle_path(&fixture.state.adapter_dir, "portable_run");
        let manifest = verify_hf_trl_export_bundle(&bundle).unwrap();
        assert_eq!(
            manifest.input_adapter.as_ref().unwrap().name,
            "source.adapter"
        );
        assert!(manifest.data.split_manifest.is_some());
        #[cfg(unix)]
        assert_eq!(
            fs::metadata(registry_path(&fixture.state.adapter_dir))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o700
        );

        let duplicate = app
            .clone()
            .oneshot(json_request(
                "POST",
                "/v1/train/hf/sft/exports",
                export_request("portable_run"),
            ))
            .await
            .unwrap();
        assert_eq!(duplicate.status(), StatusCode::CONFLICT);
        assert_eq!(
            response_json(duplicate).await["error"]["code"],
            "hf_trl_export_exists"
        );

        let listed = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/train/hf/exports")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(listed.status(), StatusCode::OK);
        let listed = response_json(listed).await;
        assert_eq!(listed["data"].as_array().unwrap().len(), 1);
        assert_eq!(listed["data"][0]["export_sha256"], manifest.export_sha256);

        let detail = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/train/hf/exports/portable_run")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(detail.status(), StatusCode::OK);
        assert_eq!(
            response_json(detail).await["manifest"]["export_sha256"],
            manifest.export_sha256
        );

        let download = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/train/hf/exports/portable_run/download")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(download.status(), StatusCode::OK);
        assert_eq!(
            download.headers()[header::CACHE_CONTROL],
            "private, no-store"
        );
        assert_eq!(
            download.headers()[header::ETAG],
            format!("\"{}\"", manifest.export_sha256)
        );
        let bytes = to_bytes(download.into_body(), 16 * 1024 * 1024)
            .await
            .unwrap();
        let archive_file = tempfile::NamedTempFile::new().unwrap();
        fs::write(archive_file.path(), &bytes).unwrap();
        let cli_verified = crate::hf_train_cli::verify_downloaded_archive(
            archive_file.path(),
            archive_file.path().parent().unwrap(),
            "portable_run",
            &manifest.export_sha256,
        )
        .unwrap();
        assert_eq!(cli_verified.export_sha256, manifest.export_sha256);
        let extracted = tempfile::tempdir().unwrap();
        tar::Archive::new(GzDecoder::new(bytes.as_ref()))
            .unpack(extracted.path())
            .unwrap();
        verify_hf_trl_export_bundle(&extracted.path().join("portable_run.kiln-hf")).unwrap();

        let refused_delete = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/v1/train/hf/exports/portable_run")
                    .header(header::IF_MATCH, format!("\"sha256:{}\"", "0".repeat(64)))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(refused_delete.status(), StatusCode::PRECONDITION_FAILED);
        assert_eq!(
            response_json(refused_delete).await["error"]["code"],
            "hf_trl_export_precondition_failed"
        );
        assert!(bundle.exists());

        let deleted = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/v1/train/hf/exports/portable_run")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(deleted.status(), StatusCode::OK);
        assert_eq!(response_json(deleted).await["status"], "deleted");
        assert!(!bundle.exists());

        let missing = app
            .oneshot(
                Request::builder()
                    .uri("/v1/train/hf/exports/portable_run")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing.status(), StatusCode::NOT_FOUND);
        server.abort();
    }

    #[tokio::test]
    async fn create_requires_closed_json_and_resident_identity() {
        let fixture = test_state(false);
        let app = routes().with_state(fixture.state);
        let mut request = export_request("missing_identity");
        request.as_object_mut().unwrap().remove("input_adapter");
        request["unknown"] = serde_json::Value::Bool(true);
        let invalid = app
            .clone()
            .oneshot(json_request("POST", "/v1/train/hf/sft/exports", request))
            .await
            .unwrap();
        assert_eq!(invalid.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            response_json(invalid).await["error"]["code"],
            "hf_trl_invalid_request"
        );

        let mut request = export_request("missing_identity");
        request.as_object_mut().unwrap().remove("input_adapter");
        let unavailable = app
            .oneshot(json_request("POST", "/v1/train/hf/sft/exports", request))
            .await
            .unwrap();
        assert_eq!(unavailable.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response_json(unavailable).await["error"]["code"],
            "hf_trl_unavailable"
        );
    }

    #[tokio::test]
    async fn create_refuses_an_exhausted_immutable_registry() {
        let fixture = test_state(true);
        let root = ensure_private_registry(&fixture.state.adapter_dir).unwrap();
        for index in 0..MAX_SERVER_EXPORTS {
            fs::create_dir(root.join(format!("occupied_{index}.kiln-hf"))).unwrap();
        }
        assert_eq!(
            published_export_count(&fixture.state.adapter_dir).unwrap(),
            MAX_SERVER_EXPORTS
        );
        let app = routes().with_state(fixture.state);
        let mut request = export_request("one_too_many");
        request.as_object_mut().unwrap().remove("input_adapter");
        let response = app
            .oneshot(json_request("POST", "/v1/train/hf/sft/exports", request))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response_json(response).await["error"]["code"],
            "hf_trl_export_capacity"
        );
    }

    #[test]
    fn private_registry_recovers_only_owned_crash_residue() {
        let directory = tempfile::tempdir().unwrap();
        let adapter_dir = directory.path().join("adapters");
        fs::create_dir(&adapter_dir).unwrap();
        let root = registry_path(&adapter_dir);
        fs::create_dir(&root).unwrap();
        fs::create_dir(root.join(".run.kiln-hf.incomplete-deadbeef")).unwrap();
        fs::create_dir(root.join(".deleting-run-deadbeef")).unwrap();
        fs::create_dir(root.join("keep-me")).unwrap();
        ensure_private_registry(&adapter_dir).unwrap();
        assert!(!root.join(".run.kiln-hf.incomplete-deadbeef").exists());
        assert!(!root.join(".deleting-run-deadbeef").exists());
        assert!(root.join("keep-me").exists());

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(
                root.join("keep-me"),
                root.join(".deleting-linked-deadbeef"),
            )
            .unwrap();
            let error = ensure_private_registry(&adapter_dir).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        }
    }
}
