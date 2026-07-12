//! Atomic, resident-identity-validated PEFT import from HF/TRL results.

use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Path as AxumPath, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use flate2::bufread::GzDecoder;
use futures::StreamExt;
use serde::Serialize;

use kiln_model::lora_loader::LoraSourceIdentity;
use kiln_train::{
    HF_TRL_ADAPTER_CONFIG_FILENAME, HF_TRL_ADAPTER_MODEL_FILENAME, HF_TRL_EXECUTED_SCRIPT_FILENAME,
    HF_TRL_EXPORT_MANIFEST_FILENAME, HF_TRL_IMPORT_ENVELOPE_SUFFIX,
    HF_TRL_IMPORT_MAX_ADAPTER_CONFIG_BYTES as MAX_IMPORT_ADAPTER_CONFIG_BYTES,
    HF_TRL_IMPORT_MAX_ARCHIVE_BYTES as MAX_IMPORT_ARCHIVE_BYTES,
    HF_TRL_IMPORT_MAX_ARCHIVE_ENTRIES as MAX_IMPORT_ARCHIVE_ENTRIES,
    HF_TRL_IMPORT_MAX_AUXILIARY_BYTES as MAX_IMPORT_AUXILIARY_BYTES,
    HF_TRL_IMPORT_MAX_EXPANDED_BYTES as MAX_IMPORT_EXPANDED_BYTES,
    HF_TRL_IMPORT_MAX_MANIFEST_BYTES as MAX_IMPORT_MANIFEST_BYTES,
    HF_TRL_IMPORT_MAX_SAFETENSORS_HEADER_BYTES as MAX_IMPORT_SAFETENSORS_HEADER_BYTES,
    HF_TRL_IMPORT_MAX_SCRIPT_BYTES as MAX_IMPORT_SCRIPT_BYTES,
    HF_TRL_IMPORT_MAX_TAR_ZERO_PADDING_BYTES as MAX_IMPORT_TAR_ZERO_PADDING_BYTES,
    HF_TRL_IMPORT_RECEIPT_FILENAME, HF_TRL_IMPORTED_ADAPTER_FILES as IMPORTED_ADAPTER_FILES,
    HF_TRL_RESULT_MANIFEST_FILENAME, HfTrlExportManifestV1, HfTrlImportReceiptV1,
    HfTrlResidentModelIdentity, HfTrlTrainingResultV1, read_hf_trl_export_manifest,
    read_hf_trl_import_receipt, read_hf_trl_training_result, validate_hf_trl_import_name,
    verify_hf_trl_import_envelope,
};

use crate::error::ApiError;
use crate::state::AppState;

const IMPORT_BODY_IDLE_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug, Serialize)]
struct ImportPeftResponse {
    status: &'static str,
    name: String,
    task: kiln_train::HfTrlTask,
    export_sha256: String,
    result_sha256: String,
    import_sha256: String,
    content_revision: String,
    used_exported_reference_script: bool,
    size_bytes: u64,
    files: usize,
}

struct PreparedImport {
    staging: PathBuf,
    export: HfTrlExportManifestV1,
    result: HfTrlTrainingResultV1,
    receipt: HfTrlImportReceiptV1,
    content_revision: String,
    size_bytes: u64,
}

enum PrepareImportError {
    Invalid(String),
    Identity(String),
    Failed(String),
}

pub(super) fn routes() -> Router<AppState> {
    Router::new().route("/v1/train/hf/peft/imports/{name}", post(import_peft))
}

async fn import_peft(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
    headers: HeaderMap,
    body: Body,
) -> Result<Response, ApiError> {
    super::adapters::validate_adapter_name(&name)?;
    validate_import_name(&name)?;
    super::adapters::ensure_adapter_mutation_admission(&state)?;
    if state.shutdown.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }
    validate_import_headers(&headers)?;
    let resident_model = resident_model_identity(&state)?;
    let model_config = state.model_config.clone();
    let target = state.adapter_dir.join(&name);
    ensure_target_absent(&target, &name)?;

    let temporary = tempfile::Builder::new()
        .prefix(".hf-peft-import-")
        .tempdir_in(&state.adapter_dir)
        .map_err(|error| ApiError::hf_trl_import_failed(format!("create staging: {error}")))?;
    let archive_path = temporary.path().join("envelope.tar.gz");
    stream_archive(body, &archive_path).await?;

    let archive_for_worker = archive_path.clone();
    let staging_root = temporary.path().to_path_buf();
    let name_for_worker = name.clone();
    let prepared = tokio::task::spawn_blocking(move || {
        prepare_import(
            &archive_for_worker,
            &staging_root,
            &name_for_worker,
            resident_model,
            model_config,
        )
    })
    .await
    .map_err(|error| ApiError::hf_trl_import_failed(format!("worker panicked: {error}")))?
    .map_err(|error| match error {
        PrepareImportError::Invalid(detail) => ApiError::hf_trl_import_invalid(detail),
        PrepareImportError::Identity(detail) => ApiError::hf_trl_import_identity_mismatch(detail),
        PrepareImportError::Failed(detail) => ApiError::hf_trl_import_failed(detail),
    })?;
    let etag = HeaderValue::from_str(&format!("\"{}\"", prepared.receipt.import_sha256))
        .map_err(|error| ApiError::hf_trl_import_failed(format!("import ETag: {error}")))?;

    let serial = crate::adapter_swap::adapter_mutation_guard(&state)
        .await
        .map_err(|error| match state.ensure_backend_healthy() {
            Ok(()) => ApiError::hf_trl_import_failed(error),
            Err(health_error) => ApiError::backend_quarantined(health_error),
        })?;
    ensure_target_absent(&target, &name)?;
    if state.loaded_adapter_name().as_deref() == Some(name.as_str()) {
        return Err(ApiError::adapter_loaded(&name));
    }
    if let Some(cap) = state.adapter_max_disk_bytes {
        let current = measure_finalized_adapter_dir_bytes_strict(&state.adapter_dir)
            .map_err(ApiError::hf_trl_import_failed)?;
        if current.saturating_add(prepared.size_bytes) > cap {
            return Err(ApiError::adapter_disk_quota_exceeded(format!(
                "{} bytes used + {} byte import > {} byte cap",
                current, prepared.size_bytes, cap
            )));
        }
    }
    match publish_staging(&prepared.staging, &target, &state.adapter_dir) {
        Ok(()) => {}
        Err(PublishError::AlreadyExists) => {
            return Err(ApiError::adapter_already_exists(&name));
        }
        Err(PublishError::Failed(error)) => {
            return Err(ApiError::hf_trl_import_failed(error));
        }
    }
    state.purge_adapter_caches(&Some(name.clone()));
    drop(serial);

    tracing::info!(
        adapter = %name,
        export_sha256 = %prepared.export.export_sha256,
        result_sha256 = %prepared.result.result_sha256,
        import_sha256 = %prepared.receipt.import_sha256,
        content_revision = %prepared.content_revision,
        size_bytes = prepared.size_bytes,
        "imported validated HF/TRL PEFT adapter"
    );
    Ok((
        StatusCode::CREATED,
        [(header::ETAG, etag)],
        Json(ImportPeftResponse {
            status: "imported",
            name,
            task: prepared.result.task,
            export_sha256: prepared.export.export_sha256,
            result_sha256: prepared.result.result_sha256,
            import_sha256: prepared.receipt.import_sha256,
            content_revision: prepared.content_revision,
            used_exported_reference_script: prepared.receipt.used_exported_reference_script,
            size_bytes: prepared.size_bytes,
            files: IMPORTED_ADAPTER_FILES.len(),
        }),
    )
        .into_response())
}

fn validate_import_name(name: &str) -> Result<(), ApiError> {
    validate_hf_trl_import_name(name)
        .map_err(|error| ApiError::hf_trl_import_invalid(error.to_string()))
}

fn validate_import_headers(headers: &HeaderMap) -> Result<(), ApiError> {
    let mut content_types = headers.get_all(header::CONTENT_TYPE).iter();
    let content_type = content_types
        .next()
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    if content_types.next().is_some() {
        return Err(ApiError::hf_trl_import_invalid(
            "Content-Type must appear exactly once",
        ));
    }
    if !content_type
        .split(';')
        .next()
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("application/gzip"))
    {
        return Err(ApiError::hf_trl_import_invalid(
            "Content-Type must be application/gzip",
        ));
    }
    let mut encodings = headers.get_all(header::CONTENT_ENCODING).iter();
    if let Some(encoding) = encodings.next() {
        if encodings.next().is_some() {
            return Err(ApiError::hf_trl_import_invalid(
                "Content-Encoding must not be repeated",
            ));
        }
        let encoding = encoding
            .to_str()
            .map_err(|_| ApiError::hf_trl_import_invalid("invalid Content-Encoding"))?;
        if !encoding.eq_ignore_ascii_case("identity") {
            return Err(ApiError::hf_trl_import_invalid(
                "Content-Encoding must be absent or identity",
            ));
        }
    }
    let mut lengths = headers.get_all(header::CONTENT_LENGTH).iter();
    if let Some(length) = lengths.next() {
        if lengths.next().is_some() {
            return Err(ApiError::hf_trl_import_invalid(
                "Content-Length must not be repeated",
            ));
        }
        let length = length
            .to_str()
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .ok_or_else(|| ApiError::hf_trl_import_invalid("invalid Content-Length"))?;
        if length > MAX_IMPORT_ARCHIVE_BYTES {
            return Err(ApiError::hf_trl_import_invalid(format!(
                "archive declares {length} bytes; limit is {MAX_IMPORT_ARCHIVE_BYTES}"
            )));
        }
    }
    Ok(())
}

fn ensure_target_absent(target: &Path, name: &str) -> Result<(), ApiError> {
    match fs::symlink_metadata(target) {
        Ok(_) => Err(ApiError::adapter_already_exists(name)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(ApiError::hf_trl_import_failed(format!(
            "inspect import target {}: {error}",
            target.display()
        ))),
    }
}

async fn stream_archive(body: Body, target: &Path) -> Result<(), ApiError> {
    stream_archive_with_idle_timeout(body, target, IMPORT_BODY_IDLE_TIMEOUT).await
}

async fn stream_archive_with_idle_timeout(
    body: Body,
    target: &Path,
    idle_timeout: Duration,
) -> Result<(), ApiError> {
    let mut file = tokio::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(target)
        .await
        .map_err(|error| ApiError::hf_trl_import_failed(format!("create archive: {error}")))?;
    let mut stream = body.into_data_stream();
    let mut total = 0u64;
    loop {
        let next = tokio::time::timeout(idle_timeout, stream.next())
            .await
            .map_err(|_| ApiError::hf_trl_import_timeout(idle_timeout.as_secs()))?;
        let Some(chunk) = next else {
            break;
        };
        let chunk = chunk.map_err(|error| {
            ApiError::hf_trl_import_invalid(format!("read archive body: {error}"))
        })?;
        total =
            total
                .checked_add(u64::try_from(chunk.len()).map_err(|_| {
                    ApiError::hf_trl_import_invalid("archive chunk length exceeds u64")
                })?)
                .ok_or_else(|| ApiError::hf_trl_import_invalid("archive byte count overflow"))?;
        if total > MAX_IMPORT_ARCHIVE_BYTES {
            return Err(ApiError::hf_trl_import_invalid(format!(
                "archive exceeds {MAX_IMPORT_ARCHIVE_BYTES} bytes"
            )));
        }
        tokio::io::AsyncWriteExt::write_all(&mut file, &chunk)
            .await
            .map_err(|error| ApiError::hf_trl_import_failed(format!("write archive: {error}")))?;
    }
    if total == 0 {
        return Err(ApiError::hf_trl_import_invalid("archive body is empty"));
    }
    tokio::io::AsyncWriteExt::flush(&mut file)
        .await
        .map_err(|error| ApiError::hf_trl_import_failed(format!("flush archive: {error}")))?;
    file.sync_all()
        .await
        .map_err(|error| ApiError::hf_trl_import_failed(format!("sync archive: {error}")))?;
    Ok(())
}

fn resident_model_identity(state: &AppState) -> Result<HfTrlResidentModelIdentity, ApiError> {
    let base_weight_shard_manifest = state
        .base_weight_shard_manifest
        .as_ref()
        .map(|manifest| manifest.as_ref().clone())
        .ok_or_else(|| ApiError::hf_trl_unavailable("resident base-weight identity is absent"))?;
    let identity = HfTrlResidentModelIdentity {
        served_model_id: state.served_model_id.clone(),
        base_weight_shard_manifest,
        model_config_sha256: kiln_core::config_hashes::sha256_json_serializable(
            &state.model_config,
        )
        .ok_or_else(|| ApiError::hf_trl_unavailable("model identity is not serializable"))?,
        tokenizer_vocab_sha256: state.tokenizer.vocab_identity_sha256(),
        tokenizer_config_sha256: state.tokenizer.tokenizer_config_sha256().map_err(|error| {
            ApiError::hf_trl_unavailable(format!("tokenizer identity: {error}"))
        })?,
        chat_template_sha256: state
            .tokenizer
            .chat_template_sha256()
            .ok_or_else(|| ApiError::hf_trl_unavailable("inference chat template is absent"))?,
        native_training_chat_template_sha256: state
            .tokenizer
            .training_chat_template_sha256()
            .ok_or_else(|| {
                ApiError::hf_trl_unavailable("native training chat template is absent")
            })?,
        trl_training_chat_template_sha256: state
            .tokenizer
            .trl_training_chat_template_sha256()
            .ok_or_else(|| ApiError::hf_trl_unavailable("TRL training chat template is absent"))?,
    };
    identity
        .validate()
        .map_err(|error| ApiError::hf_trl_unavailable(format!("resident identity: {error:#}")))?;
    Ok(identity)
}

fn prepare_import(
    archive: &Path,
    temporary_root: &Path,
    name: &str,
    resident_model: HfTrlResidentModelIdentity,
    model_config: kiln_core::config::ModelConfig,
) -> Result<PreparedImport, PrepareImportError> {
    let extraction = temporary_root.join("extracted");
    create_private_directory(&extraction)
        .map_err(|error| PrepareImportError::Failed(format!("create extraction dir: {error}")))?;
    let envelope = extract_archive(archive, &extraction, name)?;
    let (export, result) = verify_hf_trl_import_envelope(&envelope)
        .map_err(|error| PrepareImportError::Invalid(format!("{error:#}")))?;
    resident_model
        .validate_against_export(&export)
        .map_err(|error| PrepareImportError::Identity(format!("{error:#}")))?;

    let staging = temporary_root.join("adapter");
    create_private_directory(&staging)
        .map_err(|error| PrepareImportError::Failed(format!("create adapter staging: {error}")))?;
    for relative in [
        HF_TRL_ADAPTER_CONFIG_FILENAME,
        HF_TRL_ADAPTER_MODEL_FILENAME,
        HF_TRL_EXECUTED_SCRIPT_FILENAME,
        HF_TRL_EXPORT_MANIFEST_FILENAME,
        HF_TRL_RESULT_MANIFEST_FILENAME,
    ] {
        copy_new_synced(&envelope.join(relative), &staging.join(relative))?;
    }
    let receipt = HfTrlImportReceiptV1::new(name.to_string(), &export, &result, resident_model)
        .map_err(|error| PrepareImportError::Identity(format!("{error:#}")))?;
    write_json_new_synced(&staging.join(HF_TRL_IMPORT_RECEIPT_FILENAME), &receipt)?;
    let (verified_export, verified_result, verified_receipt, source_identity, size_bytes) =
        verify_imported_adapter(&staging, name, &model_config)?;
    if verified_export != export || verified_result != result || verified_receipt != receipt {
        return Err(PrepareImportError::Failed(
            "import identities changed while staging".to_string(),
        ));
    }
    sync_directory(&staging)
        .map_err(|error| PrepareImportError::Failed(format!("sync adapter staging: {error}")))?;
    Ok(PreparedImport {
        staging,
        export,
        result,
        receipt,
        content_revision: source_identity.content_revision(),
        size_bytes,
    })
}

fn extract_archive(
    archive_path: &Path,
    extraction: &Path,
    name: &str,
) -> Result<PathBuf, PrepareImportError> {
    let file = File::open(archive_path)
        .map_err(|error| PrepareImportError::Failed(format!("open archive: {error}")))?;
    let decoder = GzDecoder::new(BufReader::new(file));
    let mut archive = tar::Archive::new(decoder);
    let expected_root = format!("{name}{HF_TRL_IMPORT_ENVELOPE_SUFFIX}");
    let mut seen = BTreeSet::new();
    let mut portable_seen = BTreeSet::new();
    let mut entries = 0usize;
    let mut expanded = 0u64;
    let mut regular_files = 0usize;
    for entry in archive
        .entries()
        .map_err(|error| PrepareImportError::Invalid(format!("read tar entries: {error}")))?
    {
        let mut entry = entry
            .map_err(|error| PrepareImportError::Invalid(format!("read tar entry: {error}")))?;
        entries = entries.saturating_add(1);
        if entries > MAX_IMPORT_ARCHIVE_ENTRIES {
            return Err(PrepareImportError::Invalid(format!(
                "archive exceeds {MAX_IMPORT_ARCHIVE_ENTRIES} entries"
            )));
        }
        let portable = validate_path_bytes(&entry.path_bytes(), &expected_root)?;
        let path = entry
            .path()
            .map_err(|error| PrepareImportError::Invalid(format!("decode tar path: {error}")))?
            .into_owned();
        validate_path(&path, OsStr::new(&expected_root))?;
        if !seen.insert(path.clone()) || !portable_seen.insert(portable) {
            return Err(PrepareImportError::Invalid(format!(
                "duplicate or platform-aliased archive path {}",
                path.display()
            )));
        }
        let entry_type = entry.header().entry_type();
        if entry_type.is_file() {
            regular_files = regular_files.saturating_add(1);
            let size = entry.header().size().map_err(|error| {
                PrepareImportError::Invalid(format!("read tar entry size: {error}"))
            })?;
            expanded = expanded
                .checked_add(size)
                .ok_or_else(|| PrepareImportError::Invalid("expanded-size overflow".to_string()))?;
            if expanded > MAX_IMPORT_EXPANDED_BYTES {
                return Err(PrepareImportError::Invalid(format!(
                    "archive expands beyond {MAX_IMPORT_EXPANDED_BYTES} bytes"
                )));
            }
            validate_entry_size(&path, size)?;
        } else if !entry_type.is_dir() {
            return Err(PrepareImportError::Invalid(format!(
                "archive contains a link or special entry at {}",
                path.display()
            )));
        } else {
            let size = entry.header().size().map_err(|error| {
                PrepareImportError::Invalid(format!("read tar directory size: {error}"))
            })?;
            if size != 0 {
                return Err(PrepareImportError::Invalid(format!(
                    "archive directory entry {} declares non-zero size {size}",
                    path.display()
                )));
            }
        }
        let unpacked = entry.unpack_in(extraction).map_err(|error| {
            PrepareImportError::Invalid(format!("extract {}: {error}", path.display()))
        })?;
        if !unpacked {
            return Err(PrepareImportError::Invalid(format!(
                "archive entry escaped extraction root: {}",
                path.display()
            )));
        }
    }
    if regular_files == 0 {
        return Err(PrepareImportError::Invalid(
            "archive contains no regular files".to_string(),
        ));
    }
    let mut decoder = archive.into_inner();
    validate_decoded_tar_tail(&mut decoder)?;
    let mut compressed = decoder.into_inner();
    if !compressed
        .fill_buf()
        .map_err(|error| PrepareImportError::Invalid(format!("read gzip tail: {error}")))?
        .is_empty()
    {
        return Err(PrepareImportError::Invalid(
            "archive contains trailing bytes or another gzip member".to_string(),
        ));
    }
    let root = extraction.join(&expected_root);
    let metadata = fs::symlink_metadata(&root)
        .map_err(|error| PrepareImportError::Invalid(format!("missing archive root: {error}")))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
        return Err(PrepareImportError::Invalid(
            "archive root is not a real directory".to_string(),
        ));
    }
    Ok(root)
}

fn validate_decoded_tar_tail(reader: &mut impl Read) -> Result<(), PrepareImportError> {
    let mut total = 0u64;
    let mut buffer = [0u8; 4096];
    loop {
        let read = reader.read(&mut buffer).map_err(|error| {
            PrepareImportError::Invalid(format!("validate gzip trailer: {error}"))
        })?;
        if read == 0 {
            return Ok(());
        }
        if buffer[..read].iter().any(|byte| *byte != 0) {
            return Err(PrepareImportError::Invalid(
                "archive contains non-zero decoded bytes after the tar terminator".to_string(),
            ));
        }
        total = total
            .checked_add(u64::try_from(read).map_err(|_| {
                PrepareImportError::Invalid("tar padding length exceeds u64".to_string())
            })?)
            .ok_or_else(|| PrepareImportError::Invalid("tar padding overflow".to_string()))?;
        if total > MAX_IMPORT_TAR_ZERO_PADDING_BYTES {
            return Err(PrepareImportError::Invalid(format!(
                "archive contains more than {MAX_IMPORT_TAR_ZERO_PADDING_BYTES} zero-padding bytes after the tar terminator"
            )));
        }
    }
}

fn validate_path(path: &Path, expected_root: &OsStr) -> Result<(), PrepareImportError> {
    let mut components = path.components();
    if !matches!(components.next(), Some(Component::Normal(root)) if root == expected_root)
        || !components.all(|component| matches!(component, Component::Normal(_)))
    {
        return Err(PrepareImportError::Invalid(format!(
            "unsafe archive path {}",
            path.display()
        )));
    }
    Ok(())
}

fn validate_path_bytes(bytes: &[u8], expected_root: &str) -> Result<String, PrepareImportError> {
    if !bytes.is_ascii() {
        return Err(PrepareImportError::Invalid(
            "archive paths must be ASCII".to_string(),
        ));
    }
    let bytes = bytes.strip_suffix(b"/").unwrap_or(bytes);
    let mut components = bytes.split(|byte| *byte == b'/');
    if components.next() != Some(expected_root.as_bytes()) {
        return Err(PrepareImportError::Invalid(format!(
            "archive must use exact root {expected_root}"
        )));
    }
    for component in components {
        if component.is_empty()
            || component == b"."
            || component == b".."
            || component.contains(&b'\\')
        {
            return Err(PrepareImportError::Invalid(
                "archive path contains an unsafe component".to_string(),
            ));
        }
    }
    String::from_utf8(bytes.to_ascii_lowercase())
        .map_err(|error| PrepareImportError::Invalid(format!("archive path encoding: {error}")))
}

fn validate_entry_size(path: &Path, size: u64) -> Result<(), PrepareImportError> {
    let filename = path.file_name().and_then(OsStr::to_str).ok_or_else(|| {
        PrepareImportError::Invalid(format!("archive filename is not UTF-8: {}", path.display()))
    })?;
    let limit = match filename {
        HF_TRL_EXPORT_MANIFEST_FILENAME | HF_TRL_RESULT_MANIFEST_FILENAME => {
            MAX_IMPORT_MANIFEST_BYTES
        }
        HF_TRL_EXECUTED_SCRIPT_FILENAME => MAX_IMPORT_SCRIPT_BYTES,
        HF_TRL_ADAPTER_CONFIG_FILENAME => MAX_IMPORT_ADAPTER_CONFIG_BYTES,
        HF_TRL_ADAPTER_MODEL_FILENAME => MAX_IMPORT_EXPANDED_BYTES,
        _ => MAX_IMPORT_AUXILIARY_BYTES,
    };
    if size > limit {
        return Err(PrepareImportError::Invalid(format!(
            "archive entry {} declares {size} bytes; per-file limit is {limit}",
            path.display()
        )));
    }
    Ok(())
}

fn copy_new_synced(source: &Path, target: &Path) -> Result<(), PrepareImportError> {
    let metadata = fs::symlink_metadata(source).map_err(|error| {
        PrepareImportError::Invalid(format!("stat {}: {error}", source.display()))
    })?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(PrepareImportError::Invalid(format!(
            "import artifact is not a regular file: {}",
            source.display()
        )));
    }
    let mut reader = File::open(source).map_err(|error| {
        PrepareImportError::Failed(format!("open {}: {error}", source.display()))
    })?;
    let mut writer = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(target)
        .map_err(|error| {
            PrepareImportError::Failed(format!("create {}: {error}", target.display()))
        })?;
    io::copy(&mut reader, &mut writer).map_err(|error| {
        PrepareImportError::Failed(format!("copy {}: {error}", source.display()))
    })?;
    writer
        .sync_all()
        .map_err(|error| PrepareImportError::Failed(format!("sync {}: {error}", target.display())))
}

fn create_private_directory(path: &Path) -> io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(path)
}

fn write_json_new_synced<T: Serialize>(target: &Path, value: &T) -> Result<(), PrepareImportError> {
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| PrepareImportError::Failed(format!("serialize receipt: {error}")))?;
    bytes.push(b'\n');
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(target)
        .map_err(|error| {
            PrepareImportError::Failed(format!("create {}: {error}", target.display()))
        })?;
    file.write_all(&bytes).map_err(|error| {
        PrepareImportError::Failed(format!("write {}: {error}", target.display()))
    })?;
    file.sync_all()
        .map_err(|error| PrepareImportError::Failed(format!("sync {}: {error}", target.display())))
}

fn verify_imported_adapter(
    root: &Path,
    expected_name: &str,
    model_config: &kiln_core::config::ModelConfig,
) -> Result<
    (
        HfTrlExportManifestV1,
        HfTrlTrainingResultV1,
        HfTrlImportReceiptV1,
        LoraSourceIdentity,
        u64,
    ),
    PrepareImportError,
> {
    let mut actual = BTreeSet::new();
    let mut size_bytes = 0u64;
    for entry in fs::read_dir(root)
        .map_err(|error| PrepareImportError::Failed(format!("read adapter staging: {error}")))?
    {
        let entry = entry
            .map_err(|error| PrepareImportError::Failed(format!("read adapter entry: {error}")))?;
        let metadata = fs::symlink_metadata(entry.path())
            .map_err(|error| PrepareImportError::Failed(format!("stat adapter entry: {error}")))?;
        if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
            return Err(PrepareImportError::Invalid(format!(
                "imported adapter contains non-file entry {}",
                entry.path().display()
            )));
        }
        let name = entry.file_name().into_string().map_err(|_| {
            PrepareImportError::Invalid("adapter filename is not UTF-8".to_string())
        })?;
        actual.insert(name);
        size_bytes = size_bytes
            .checked_add(metadata.len())
            .ok_or_else(|| PrepareImportError::Invalid("adapter size overflow".to_string()))?;
    }
    let expected = IMPORTED_ADAPTER_FILES
        .iter()
        .map(|value| (*value).to_string())
        .collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(PrepareImportError::Invalid(format!(
            "imported adapter file set differs: expected {expected:?}, found {actual:?}"
        )));
    }
    let export = read_hf_trl_export_manifest(root)
        .map_err(|error| PrepareImportError::Invalid(format!("{error:#}")))?;
    let result = read_hf_trl_training_result(root)
        .map_err(|error| PrepareImportError::Invalid(format!("{error:#}")))?;
    let receipt = read_hf_trl_import_receipt(root)
        .map_err(|error| PrepareImportError::Invalid(format!("{error:#}")))?;
    receipt
        .validate_against(&export, &result)
        .map_err(|error| PrepareImportError::Invalid(format!("{error:#}")))?;
    if receipt.adapter_name != expected_name {
        return Err(PrepareImportError::Invalid(
            "import receipt adapter name differs from request".to_string(),
        ));
    }
    result
        .verify_files(root)
        .map_err(|error| PrepareImportError::Invalid(format!("{error:#}")))?;
    validate_safetensors_header(&root.join(HF_TRL_ADAPTER_MODEL_FILENAME))?;
    // SAFETY: `root` is a private Kiln-owned staging directory. Every file was
    // copied create-new, no path escapes or links are admitted, and no handle
    // mutates the PEFT files during this blocking validation.
    let source_identity =
        unsafe { LoraSourceIdentity::from_immutable_adapter_dir_for_model(root, model_config) }
            .map_err(|error| {
                PrepareImportError::Invalid(format!("invalid PEFT adapter: {error:#}"))
            })?;
    Ok((export, result, receipt, source_identity, size_bytes))
}

fn validate_safetensors_header(path: &Path) -> Result<(), PrepareImportError> {
    let mut file = File::open(path)
        .map_err(|error| PrepareImportError::Failed(format!("open {}: {error}", path.display())))?;
    let mut encoded = [0u8; 8];
    file.read_exact(&mut encoded).map_err(|error| {
        PrepareImportError::Invalid(format!("read safetensors header length: {error}"))
    })?;
    let header_bytes = u64::from_le_bytes(encoded);
    if header_bytes > MAX_IMPORT_SAFETENSORS_HEADER_BYTES {
        return Err(PrepareImportError::Invalid(format!(
            "safetensors header declares {header_bytes} bytes; import limit is {MAX_IMPORT_SAFETENSORS_HEADER_BYTES}"
        )));
    }
    let file_bytes = file
        .metadata()
        .map_err(|error| PrepareImportError::Failed(format!("stat {}: {error}", path.display())))?
        .len();
    let minimum = header_bytes.checked_add(8).ok_or_else(|| {
        PrepareImportError::Invalid("safetensors header length overflow".to_string())
    })?;
    if minimum > file_bytes {
        return Err(PrepareImportError::Invalid(format!(
            "safetensors header requires {minimum} bytes but file contains {file_bytes}"
        )));
    }
    Ok(())
}

fn measure_finalized_adapter_dir_bytes_strict(adapter_dir: &Path) -> Result<u64, String> {
    let entries = fs::read_dir(adapter_dir)
        .map_err(|error| format!("read adapter registry {}: {error}", adapter_dir.display()))?;
    let mut total = 0u64;
    for entry in entries {
        let entry = entry.map_err(|error| format!("read adapter registry entry: {error}"))?;
        if entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        total = total
            .checked_add(measure_real_tree_bytes(&entry.path())?)
            .ok_or_else(|| "adapter registry byte count overflow".to_string())?;
    }
    Ok(total)
}

fn measure_real_tree_bytes(path: &Path) -> Result<u64, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("stat adapter registry entry {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() {
        return Err(format!(
            "adapter registry contains a symlink at {}",
            path.display()
        ));
    }
    if metadata.file_type().is_file() {
        return Ok(metadata.len());
    }
    if !metadata.file_type().is_dir() {
        return Err(format!(
            "adapter registry contains a special entry at {}",
            path.display()
        ));
    }
    let entries = fs::read_dir(path)
        .map_err(|error| format!("read adapter directory {}: {error}", path.display()))?;
    let mut total = 0u64;
    for entry in entries {
        let entry = entry.map_err(|error| format!("read adapter directory entry: {error}"))?;
        total = total
            .checked_add(measure_real_tree_bytes(&entry.path())?)
            .ok_or_else(|| "adapter registry byte count overflow".to_string())?;
    }
    Ok(total)
}

enum PublishError {
    AlreadyExists,
    Failed(String),
}

fn publish_staging(staging: &Path, target: &Path, parent: &Path) -> Result<(), PublishError> {
    kiln_resource::atomic_rename_noreplace(staging, target).map_err(|error| {
        if error.kind() == io::ErrorKind::AlreadyExists {
            PublishError::AlreadyExists
        } else {
            PublishError::Failed(format!(
                "publish adapter {} -> {} without replacement: {error}",
                staging.display(),
                target.display()
            ))
        }
    })?;
    if let Err(sync_error) = sync_directory(parent) {
        let rollback = fs::rename(target, staging);
        let rollback_sync = sync_directory(parent);
        return Err(PublishError::Failed(format!(
            "sync adapter registry after publication: {sync_error}; rollback={rollback:?}; rollback_sync={rollback_sync:?}"
        )));
    }
    Ok(())
}

fn sync_directory(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use axum::body::to_bytes;
    use axum::http::Request;
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use kiln_train::{
        HfTrlConfigValue, HfTrlFileIdentity, HfTrlOutputAdapter, HfTrlTask, HfTrlTrainerIdentity,
        HfTrlTrainerKind, HfTrlTrainingResultV1, read_hf_trl_export_manifest,
        verify_hf_trl_completed_bundle, write_hf_trl_import_envelope,
    };
    use tower::ServiceExt;

    use super::*;

    async fn response_json(response: Response) -> serde_json::Value {
        serde_json::from_slice(
            &to_bytes(response.into_body(), 8 * 1024 * 1024)
                .await
                .unwrap(),
        )
        .unwrap()
    }

    fn result_file(root: &Path, relative: &str) -> HfTrlFileIdentity {
        HfTrlFileIdentity::from_file(root, relative).unwrap()
    }

    #[test]
    fn import_names_match_portable_archive_root_contract() {
        for valid in ["a", "run-01", "run_01", "run.v2", "A9"] {
            validate_import_name(valid).unwrap();
        }
        for invalid in ["", "-run", ".run", "run..v2", "run path", "run/path", "é"] {
            assert!(
                validate_import_name(invalid).is_err(),
                "accepted {invalid:?}"
            );
        }
        assert!(validate_import_name(&"a".repeat(129)).is_err());
    }

    fn write_peft_result(
        root: &Path,
        export: &HfTrlExportManifestV1,
        resident_shape_compatible: bool,
    ) {
        fs::copy(
            root.join(kiln_train::HF_TRL_REFERENCE_SCRIPT_FILENAME),
            root.join(HF_TRL_EXECUTED_SCRIPT_FILENAME),
        )
        .unwrap();
        fs::write(
            root.join(HF_TRL_ADAPTER_CONFIG_FILENAME),
            br#"{"r":1,"lora_alpha":1.0,"target_modules":["q_proj"],"task_type":"CAUSAL_LM"}"#,
        )
        .unwrap();
        let model_config: kiln_core::config::ModelConfig = serde_json::from_slice(
            &fs::read(root.join(&export.model.model_config.relative_path)).unwrap(),
        )
        .unwrap();
        let layer = (0..model_config.num_layers)
            .find(|layer| model_config.is_full_attention_layer(*layer))
            .unwrap();
        let input = model_config.hidden_size;
        let output = if resident_shape_compatible {
            model_config.full_attn_q_proj_dim()
        } else {
            model_config.full_attn_q_proj_dim() - 1
        };
        let a_bytes = (0..input)
            .flat_map(|_| 0.25f32.to_le_bytes())
            .collect::<Vec<_>>();
        let b_bytes = (0..output)
            .flat_map(|_| 0.5f32.to_le_bytes())
            .collect::<Vec<_>>();
        let a =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![1, input], &a_bytes)
                .unwrap();
        let b = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            vec![output, 1],
            &b_bytes,
        )
        .unwrap();
        let a_name =
            format!("base_model.model.model.layers.{layer}.self_attn.q_proj.lora_A.weight");
        let b_name =
            format!("base_model.model.model.layers.{layer}.self_attn.q_proj.lora_B.weight");
        let weights =
            safetensors::tensor::serialize([(a_name.as_str(), a), (b_name.as_str(), b)], None)
                .unwrap();
        fs::write(root.join(HF_TRL_ADAPTER_MODEL_FILENAME), weights).unwrap();
        let result = HfTrlTrainingResultV1::new(
            export.export_sha256.clone(),
            HfTrlTask::Sft,
            HfTrlTrainerIdentity {
                kind: HfTrlTrainerKind::TrlSftTrainer,
                python_version: "3.13.5".to_string(),
                torch_version: "2.13.0".to_string(),
                transformers_version: "5.13.1".to_string(),
                trl_version: "1.8.0".to_string(),
                peft_version: "0.19.1".to_string(),
                script: result_file(root, HF_TRL_EXECUTED_SCRIPT_FILENAME),
            },
            BTreeMap::from([("seed".to_string(), HfTrlConfigValue::Unsigned(42))]),
            HfTrlOutputAdapter {
                config: result_file(root, HF_TRL_ADAPTER_CONFIG_FILENAME),
                model: result_file(root, HF_TRL_ADAPTER_MODEL_FILENAME),
            },
        )
        .unwrap();
        fs::write(
            root.join(HF_TRL_RESULT_MANIFEST_FILENAME),
            format!("{}\n", serde_json::to_string_pretty(&result).unwrap()),
        )
        .unwrap();
    }

    async fn completed_envelope(
        state: &AppState,
        export_name: &str,
    ) -> (tempfile::TempDir, PathBuf) {
        completed_envelope_with_compatibility(state, export_name, true).await
    }

    async fn completed_envelope_with_compatibility(
        state: &AppState,
        export_name: &str,
        resident_shape_compatible: bool,
    ) -> (tempfile::TempDir, PathBuf) {
        let app = crate::api::router(state.clone());
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/train/hf/sft/exports")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&serde_json::json!({
                            "name": export_name,
                            "examples": [{
                                "messages": [
                                    {"role": "user", "content": "a"},
                                    {"role": "assistant", "content": "b"}
                                ]
                            }],
                            "invalid_row_policy": "fail"
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CREATED);
        let completed = state
            .adapter_dir
            .join(".hf_trl_exports")
            .join(format!("{export_name}.kiln-hf"));
        let export = read_hf_trl_export_manifest(&completed).unwrap();
        write_peft_result(&completed, &export, resident_shape_compatible);
        verify_hf_trl_completed_bundle(&completed).unwrap();
        let temporary = tempfile::tempdir().unwrap();
        let envelope = temporary
            .path()
            .join(format!("{export_name}{HF_TRL_IMPORT_ENVELOPE_SUFFIX}"));
        write_hf_trl_import_envelope(&completed, &envelope).unwrap();
        (temporary, envelope)
    }

    fn archive_envelope(envelope: &Path, import_name: &str) -> Vec<u8> {
        let encoder = GzEncoder::new(Vec::new(), Compression::default());
        let mut archive = tar::Builder::new(encoder);
        archive
            .append_dir_all(
                format!("{import_name}{HF_TRL_IMPORT_ENVELOPE_SUFFIX}"),
                envelope,
            )
            .unwrap();
        archive.into_inner().unwrap().finish().unwrap()
    }

    fn import_request(name: &str, bytes: Vec<u8>) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(format!("/v1/train/hf/peft/imports/{name}"))
            .header(header::CONTENT_TYPE, "application/gzip")
            .header(header::CONTENT_LENGTH, bytes.len())
            .body(Body::from(bytes))
            .unwrap()
    }

    #[tokio::test]
    async fn peft_import_publishes_exact_verified_adapter_and_receipt() {
        let fixture = super::super::hf_trl::tests::test_state(true);
        let (_temporary, envelope) = completed_envelope(&fixture.state, "trained").await;
        let archive = archive_envelope(&envelope, "imported");
        let app = crate::api::router(fixture.state.clone());
        let response = app
            .clone()
            .oneshot(import_request("imported", archive.clone()))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CREATED);
        let etag = response.headers()[header::ETAG].clone();
        let response = response_json(response).await;
        assert_eq!(response["status"], "imported");
        assert_eq!(response["name"], "imported");
        assert_eq!(response["task"], "sft");
        assert_eq!(response["files"], IMPORTED_ADAPTER_FILES.len());
        assert_eq!(
            etag,
            format!("\"{}\"", response["import_sha256"].as_str().unwrap())
        );

        let installed = fixture.state.adapter_dir.join("imported");
        let names = fs::read_dir(&installed)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            names,
            IMPORTED_ADAPTER_FILES
                .iter()
                .map(|value| (*value).to_string())
                .collect()
        );
        assert!(!installed.join(kiln_train::HF_TRL_DATASET_FILENAME).exists());
        let export = read_hf_trl_export_manifest(&installed).unwrap();
        let result = read_hf_trl_training_result(&installed).unwrap();
        let receipt = read_hf_trl_import_receipt(&installed).unwrap();
        receipt.validate_against(&export, &result).unwrap();
        result.verify_files(&installed).unwrap();
        assert_eq!(receipt.adapter_name, "imported");
        assert_eq!(receipt.import_sha256, response["import_sha256"]);

        let collision = app
            .oneshot(import_request("imported", archive))
            .await
            .unwrap();
        assert_eq!(collision.status(), StatusCode::CONFLICT);
        assert_eq!(
            response_json(collision).await["error"]["code"],
            "adapter_already_exists"
        );
    }

    #[tokio::test]
    async fn peft_import_cli_streams_completed_bundle_and_retains_source() {
        let fixture = super::super::hf_trl::tests::test_state(true);
        let (_temporary, _envelope) = completed_envelope(&fixture.state, "cli-source").await;
        let completed = fixture
            .state
            .adapter_dir
            .join(".hf_trl_exports")
            .join("cli-source.kiln-hf");
        let expected = verify_hf_trl_completed_bundle(&completed).unwrap();

        let app = crate::api::router(fixture.state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let options = crate::hf_train_cli::ImportPeftOptions {
            url: format!("http://{address}"),
            bundle: completed.clone(),
            name: "cli.imported-01".to_string(),
        };
        crate::hf_train_cli::run_import_peft(options.clone())
            .await
            .unwrap();

        assert_eq!(
            verify_hf_trl_completed_bundle(&completed).unwrap(),
            expected,
            "the CLI must not modify or remove its completed source bundle"
        );
        let installed = fixture.state.adapter_dir.join("cli.imported-01");
        let export = read_hf_trl_export_manifest(&installed).unwrap();
        let result = read_hf_trl_training_result(&installed).unwrap();
        let receipt = read_hf_trl_import_receipt(&installed).unwrap();
        receipt.validate_against(&export, &result).unwrap();
        assert_eq!(receipt.adapter_name, "cli.imported-01");

        let collision = crate::hf_train_cli::run_import_peft(options)
            .await
            .unwrap_err();
        assert!(
            format!("{collision:#}").contains("adapter_already_exists"),
            "{collision:#}"
        );
        assert_eq!(
            verify_hf_trl_completed_bundle(&completed).unwrap(),
            expected
        );
        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn peft_import_cli_treats_non_created_success_as_ambiguous() {
        let fixture = super::super::hf_trl::tests::test_state(true);
        let (_temporary, _envelope) = completed_envelope(&fixture.state, "status-source").await;
        let completed = fixture
            .state
            .adapter_dir
            .join(".hf_trl_exports")
            .join("status-source.kiln-hf");
        let expected = verify_hf_trl_completed_bundle(&completed).unwrap();

        let app = axum::Router::new().route(
            "/v1/train/hf/peft/imports/{name}",
            axum::routing::post(|| async { StatusCode::NO_CONTENT }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let error = crate::hf_train_cli::run_import_peft(crate::hf_train_cli::ImportPeftOptions {
            url: format!("http://{address}"),
            bundle: completed.clone(),
            name: "ambiguous-status".to_string(),
        })
        .await
        .unwrap_err();
        let detail = format!("{error:#}");
        assert!(detail.contains("instead of HTTP 201"), "{detail}");
        assert!(detail.contains("may already be installed"), "{detail}");
        assert_eq!(
            verify_hf_trl_completed_bundle(&completed).unwrap(),
            expected
        );
        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn peft_import_rejects_resident_identity_drift_without_publication() {
        let mut fixture = super::super::hf_trl::tests::test_state(true);
        let (_temporary, envelope) = completed_envelope(&fixture.state, "source").await;
        let archive = archive_envelope(&envelope, "wrong-resident");
        fixture.state.served_model_id = "different/model".to_string();
        let response = crate::api::router(fixture.state.clone())
            .oneshot(import_request("wrong-resident", archive))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CONFLICT);
        assert_eq!(
            response_json(response).await["error"]["code"],
            "hf_trl_import_identity_mismatch"
        );
        assert!(!fixture.state.adapter_dir.join("wrong-resident").exists());
    }

    #[tokio::test]
    async fn peft_import_rejects_self_verified_but_incompatible_tensor_shapes() {
        let fixture = super::super::hf_trl::tests::test_state(true);
        let (_temporary, envelope) =
            completed_envelope_with_compatibility(&fixture.state, "bad-shape-source", false).await;
        let archive = archive_envelope(&envelope, "bad-shape");
        let response = crate::api::router(fixture.state.clone())
            .oneshot(import_request("bad-shape", archive))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            response_json(response).await["error"]["code"],
            "hf_trl_import_invalid"
        );
        assert!(!fixture.state.adapter_dir.join("bad-shape").exists());
    }

    #[tokio::test]
    async fn stalled_import_body_times_out_without_publishing_bytes() {
        let temporary = tempfile::tempdir().unwrap();
        let target = temporary.path().join("stalled.tar.gz");
        let pending = futures::stream::pending::<Result<axum::body::Bytes, io::Error>>();
        let error = stream_archive_with_idle_timeout(
            Body::from_stream(pending),
            &target,
            Duration::from_millis(1),
        )
        .await
        .unwrap_err();
        assert_eq!(error.status, StatusCode::REQUEST_TIMEOUT);
        assert_eq!(error.code, "hf_trl_import_timeout");
        assert_eq!(fs::metadata(target).unwrap().len(), 0);
    }

    #[test]
    fn import_headers_and_archive_paths_fail_closed() {
        let mut headers = HeaderMap::new();
        assert!(validate_import_headers(&headers).is_err());
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/gzip"),
        );
        validate_import_headers(&headers).unwrap();
        headers.append(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/gzip"),
        );
        assert!(validate_import_headers(&headers).is_err());
        headers.remove(header::CONTENT_TYPE);
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/gzip"),
        );
        headers.insert(header::CONTENT_ENCODING, HeaderValue::from_static("gzip"));
        assert!(validate_import_headers(&headers).is_err());

        assert!(validate_path_bytes(b"run.kiln-hf-import/file", "run.kiln-hf-import").is_ok());
        for invalid in [
            b"other/file".as_slice(),
            b"run.kiln-hf-import/../file",
            b"run.kiln-hf-import//file",
            b"run.kiln-hf-import/file\\alias",
            b"run.kiln-hf-import/\xff",
        ] {
            assert!(validate_path_bytes(invalid, "run.kiln-hf-import").is_err());
        }
    }

    #[test]
    fn archive_trailing_data_and_publication_collisions_fail_closed() {
        let temporary = tempfile::tempdir().unwrap();
        let archive_path = temporary.path().join("trailing.tar.gz");
        let encoder = GzEncoder::new(Vec::new(), Compression::default());
        let mut archive = tar::Builder::new(encoder);
        let mut header = tar::Header::new_gnu();
        header.set_size(1);
        header.set_mode(0o600);
        header.set_cksum();
        archive
            .append_data(&mut header, "run.kiln-hf-import/file", &b"x"[..])
            .unwrap();
        let mut encoder = archive.into_inner().unwrap();
        encoder.write_all(b"decoded tail").unwrap();
        fs::write(&archive_path, encoder.finish().unwrap()).unwrap();
        let extraction = temporary.path().join("extraction");
        fs::create_dir(&extraction).unwrap();
        match extract_archive(&archive_path, &extraction, "run") {
            Err(PrepareImportError::Invalid(error)) => {
                assert!(error.contains("non-zero decoded bytes after the tar terminator"));
            }
            _ => panic!("decoded trailing data must be rejected"),
        }

        let staging = temporary.path().join("staging");
        let target = temporary.path().join("target");
        fs::create_dir(&staging).unwrap();
        fs::write(staging.join("payload"), b"new").unwrap();
        fs::create_dir(&target).unwrap();
        assert!(matches!(
            publish_staging(&staging, &target, temporary.path()),
            Err(PublishError::AlreadyExists)
        ));
        assert!(target.read_dir().unwrap().next().is_none());
        assert_eq!(fs::read(staging.join("payload")).unwrap(), b"new");

        let oversized_header = temporary.path().join("oversized.safetensors");
        fs::write(
            &oversized_header,
            (MAX_IMPORT_SAFETENSORS_HEADER_BYTES + 1).to_le_bytes(),
        )
        .unwrap();
        assert!(matches!(
            validate_safetensors_header(&oversized_header),
            Err(PrepareImportError::Invalid(_))
        ));

        let registry = temporary.path().join("registry");
        let adapter = registry.join("adapter");
        fs::create_dir_all(&adapter).unwrap();
        fs::write(adapter.join("weights"), b"123").unwrap();
        let hidden = registry.join(".internal");
        fs::create_dir(&hidden).unwrap();
        fs::write(hidden.join("ignored"), b"not finalized").unwrap();
        assert_eq!(
            measure_finalized_adapter_dir_bytes_strict(&registry).unwrap(),
            3
        );
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(adapter.join("weights"), adapter.join("alias")).unwrap();
            assert!(measure_finalized_adapter_dir_bytes_strict(&registry).is_err());
        }
    }
}
