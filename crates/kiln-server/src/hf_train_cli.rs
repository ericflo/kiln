//! Verified CLI transport for immutable Hugging Face TRL/PEFT exports.

use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use console::style;
use flate2::bufread::GzDecoder;
use flate2::{Compression, GzBuilder};
use kiln_model::lora_loader::LoraSourceIdentity;
use kiln_train::{
    HF_TRL_ADAPTER_CONFIG_FILENAME, HF_TRL_ADAPTER_MODEL_FILENAME, HF_TRL_BUNDLE_SUFFIX,
    HF_TRL_EXECUTED_SCRIPT_FILENAME, HF_TRL_EXPORT_MANIFEST_FILENAME,
    HF_TRL_IMPORT_ENVELOPE_SUFFIX,
    HF_TRL_IMPORT_MAX_ADAPTER_CONFIG_BYTES as MAX_IMPORT_ADAPTER_CONFIG_BYTES,
    HF_TRL_IMPORT_MAX_ARCHIVE_BYTES as MAX_IMPORT_ARCHIVE_BYTES,
    HF_TRL_IMPORT_MAX_AUXILIARY_BYTES as MAX_IMPORT_AUXILIARY_BYTES,
    HF_TRL_IMPORT_MAX_EXPANDED_BYTES as MAX_IMPORT_EXPANDED_BYTES,
    HF_TRL_IMPORT_MAX_MANIFEST_BYTES as MAX_IMPORT_MANIFEST_BYTES,
    HF_TRL_IMPORT_MAX_SCRIPT_BYTES as MAX_IMPORT_SCRIPT_BYTES, HF_TRL_IMPORT_RECEIPT_FILENAME,
    HF_TRL_IMPORTED_ADAPTER_FILES as IMPORTED_ADAPTER_FILES, HF_TRL_RESULT_MANIFEST_FILENAME,
    HfTrlExportManifestV1, HfTrlImportReceiptV1, HfTrlResidentModelIdentity, HfTrlTask,
    HfTrlTrainingResultV1, hf_trl_import_envelope_files,
    validate_hf_trl_import_name as validate_import_name,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::teacher_identity::is_lower_sha256_identity;

const MAX_JSON_RESPONSE_BYTES: usize = 4 * 1024 * 1024;
const MAX_ARCHIVE_ENTRIES: usize = 4096;
const MAX_ARCHIVE_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const MAX_EXPANDED_ARCHIVE_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const HTTP_READ_IDLE_TIMEOUT: Duration = Duration::from_secs(120);
const IMPORT_UPLOAD_CHUNK_BYTES: usize = 256 * 1024;
const IMPORT_UPLOAD_CHANNEL_DEPTH: usize = 4;

#[derive(Debug, Clone)]
pub struct ExportSftOptions {
    pub url: String,
    pub file: Option<String>,
    pub dataset: Option<String>,
    pub name: String,
    pub output: Option<PathBuf>,
    pub invalid_row_policy: String,
    pub input_adapter: Option<String>,
    pub split_manifest: Option<PathBuf>,
    pub keep_server_copy: bool,
}

#[derive(Debug, Clone)]
pub struct ExportGrpoOptions {
    pub url: String,
    pub file: String,
    pub name: String,
    pub output: Option<PathBuf>,
    pub input_adapter: Option<String>,
    pub split_manifest: Option<PathBuf>,
    pub keep_server_copy: bool,
}

#[derive(Debug, Clone)]
pub struct ImportPeftOptions {
    pub url: String,
    pub bundle: PathBuf,
    pub name: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ExportSummary {
    name: String,
    task: HfTrlTask,
    export_sha256: String,
    source_name: String,
    row_count: u64,
    ordered_corpus_sha256: String,
    input_adapter: Option<String>,
    bundle_filename: String,
    download_url: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct ExportList {
    data: Vec<ExportSummary>,
}

#[derive(Debug, Deserialize)]
struct DeleteExportResponse {
    status: String,
    name: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ImportPeftResponse {
    status: String,
    name: String,
    task: HfTrlTask,
    export_sha256: String,
    result_sha256: String,
    import_sha256: String,
    content_revision: String,
    used_exported_reference_script: bool,
    size_bytes: u64,
    files: usize,
}

struct PreparedImportUpload {
    _temporary: tempfile::TempDir,
    envelope: PathBuf,
    export: HfTrlExportManifestV1,
    result: HfTrlTrainingResultV1,
    receipt: HfTrlImportReceiptV1,
    content_revision: String,
    expanded_bytes: u64,
    installed_bytes: u64,
}

struct ImportArchiveWriter {
    sender: mpsc::Sender<std::result::Result<Vec<u8>, io::Error>>,
    buffer: Vec<u8>,
    written: u64,
}

struct DownloadedBundle {
    manifest: HfTrlExportManifestV1,
    archive_sha256: String,
    archive_bytes: u64,
}

fn build_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(30))
        .read_timeout(HTTP_READ_IDLE_TIMEOUT)
        .build()
        .context("building HF/TRL export HTTP client")
}

fn endpoint(base_url: &str, path: &str) -> Result<reqwest::Url> {
    let base_url = base_url.trim();
    ensure!(!base_url.is_empty(), "server URL must not be empty");
    let parsed = reqwest::Url::parse(base_url).context("parsing server URL")?;
    ensure!(
        matches!(parsed.scheme(), "http" | "https"),
        "server URL must use http or https"
    );
    ensure!(
        parsed.query().is_none() && parsed.fragment().is_none(),
        "server URL must not contain a query or fragment"
    );
    ensure!(
        parsed.username().is_empty() && parsed.password().is_none(),
        "server URL must not contain embedded credentials"
    );
    reqwest::Url::parse(&format!("{}{path}", base_url.trim_end_matches('/')))
        .context("building HF/TRL export endpoint URL")
}

fn validate_export_name(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty()
            && name.len() <= 128
            && name.bytes().enumerate().all(|(index, byte)| {
                byte.is_ascii_alphanumeric() || (index > 0 && matches!(byte, b'-' | b'_'))
            }),
        "export name must be 1..=128 ASCII bytes, start with an alphanumeric character, and contain only alphanumerics, '-' or '_'"
    );
    Ok(())
}

fn validate_export_sha256(digest: &str) -> Result<()> {
    ensure!(
        is_lower_sha256_identity(digest),
        "export identity must be lowercase sha256:<64-hex>"
    );
    Ok(())
}

fn validate_export_etag(
    headers: &reqwest::header::HeaderMap,
    expected_export_sha256: &str,
) -> Result<()> {
    let mut values = headers.get_all(reqwest::header::ETAG).iter();
    let value = values
        .next()
        .context("HF/TRL export response is missing its ETag identity")?;
    ensure!(
        values.next().is_none(),
        "HF/TRL export response contains multiple ETag identities"
    );
    let value = value
        .to_str()
        .context("HF/TRL export ETag must be visible ASCII")?;
    ensure!(
        value == format!("\"{expected_export_sha256}\""),
        "HF/TRL export ETag {value:?} differs from identity {expected_export_sha256}"
    );
    Ok(())
}

fn validate_import_etag(
    headers: &reqwest::header::HeaderMap,
    expected_import_sha256: &str,
) -> Result<()> {
    let mut values = headers.get_all(reqwest::header::ETAG).iter();
    let value = values
        .next()
        .context("HF/TRL import response is missing its ETag identity")?;
    ensure!(
        values.next().is_none(),
        "HF/TRL import response contains multiple ETag identities"
    );
    let value = value
        .to_str()
        .context("HF/TRL import ETag must be visible ASCII")?;
    ensure!(
        value == format!("\"{expected_import_sha256}\""),
        "HF/TRL import ETag {value:?} differs from expected identity {expected_import_sha256}"
    );
    Ok(())
}

fn validate_json_content_type(headers: &reqwest::header::HeaderMap) -> Result<()> {
    let mut values = headers.get_all(reqwest::header::CONTENT_TYPE).iter();
    let value = values
        .next()
        .context("HF/TRL import response is missing Content-Type")?;
    ensure!(
        values.next().is_none(),
        "HF/TRL import response contains multiple Content-Type values"
    );
    let value = value
        .to_str()
        .context("HF/TRL import Content-Type must be visible ASCII")?;
    ensure!(
        value
            .split(';')
            .next()
            .is_some_and(|media_type| media_type.trim().eq_ignore_ascii_case("application/json")),
        "HF/TRL import response has unexpected Content-Type {value:?}"
    );
    Ok(())
}

fn destination_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn ensure_available_destination(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(_) => bail!(
            "refusing to overwrite existing HF/TRL export output {}",
            path.display()
        ),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).with_context(|| format!("inspecting {}", path.display())),
    }
    let parent = destination_parent(path);
    let metadata = fs::metadata(parent)
        .with_context(|| format!("inspecting output directory {}", parent.display()))?;
    ensure!(
        metadata.is_dir(),
        "output parent is not a directory: {}",
        parent.display()
    );
    Ok(())
}

fn read_split_manifest(path: &Path) -> Result<serde_json::Value> {
    let bytes =
        fs::read(path).with_context(|| format!("reading split manifest {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing split manifest {} as JSON", path.display()))?;
    ensure!(
        value.is_object(),
        "split manifest must be a JSON object: {}",
        path.display()
    );
    Ok(value)
}

async fn read_bounded_body(mut response: reqwest::Response, limit: usize) -> Result<Vec<u8>> {
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .context("reading HTTP response body")?
    {
        ensure!(
            body.len().saturating_add(chunk.len()) <= limit,
            "HTTP response body exceeds {limit} bytes"
        );
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn render_api_error(status: reqwest::StatusCode, body: &[u8]) -> String {
    let parsed = serde_json::from_slice::<serde_json::Value>(body).ok();
    if let Some(error) = parsed
        .as_ref()
        .and_then(|value| value.get("error"))
        .and_then(serde_json::Value::as_object)
    {
        let message = error
            .get("message")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("request failed");
        let code = error
            .get("code")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown_error");
        let hint = error
            .get("hint")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        return if hint.is_empty() {
            format!("{message} ({code}; HTTP {status})")
        } else {
            format!("{message} ({code}; HTTP {status})\n  hint: {hint}")
        };
    }
    let text = String::from_utf8_lossy(body).trim().to_string();
    if text.is_empty() {
        format!("HTTP {status}")
    } else {
        format!("HTTP {status}: {text}")
    }
}

async fn decode_json_response<T: DeserializeOwned>(response: reqwest::Response) -> Result<T> {
    let status = response.status();
    let body = read_bounded_body(response, MAX_JSON_RESPONSE_BYTES).await?;
    if !status.is_success() {
        bail!("{}", render_api_error(status, &body));
    }
    serde_json::from_slice(&body).context("decoding HF/TRL export server response")
}

impl ImportArchiveWriter {
    fn new(sender: mpsc::Sender<std::result::Result<Vec<u8>, io::Error>>) -> Self {
        Self {
            sender,
            buffer: Vec::with_capacity(IMPORT_UPLOAD_CHUNK_BYTES),
            written: 0,
        }
    }

    fn emit(&mut self) -> io::Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let chunk = std::mem::replace(
            &mut self.buffer,
            Vec::with_capacity(IMPORT_UPLOAD_CHUNK_BYTES),
        );
        self.sender.blocking_send(Ok(chunk)).map_err(|_| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                "HF/TRL import request stopped consuming the upload",
            )
        })
    }

    fn finish(mut self) -> io::Result<u64> {
        self.emit()?;
        Ok(self.written)
    }
}

impl Write for ImportArchiveWriter {
    fn write(&mut self, mut bytes: &[u8]) -> io::Result<usize> {
        let accepted = bytes.len();
        let next = self
            .written
            .checked_add(u64::try_from(accepted).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "upload write length exceeds u64",
                )
            })?)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "upload size overflow"))?;
        if next > MAX_IMPORT_ARCHIVE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("compressed import archive exceeds {MAX_IMPORT_ARCHIVE_BYTES} bytes"),
            ));
        }
        self.written = next;
        while !bytes.is_empty() {
            let available = IMPORT_UPLOAD_CHUNK_BYTES - self.buffer.len();
            let take = available.min(bytes.len());
            self.buffer.extend_from_slice(&bytes[..take]);
            bytes = &bytes[take..];
            if self.buffer.len() == IMPORT_UPLOAD_CHUNK_BYTES {
                self.emit()?;
            }
        }
        Ok(accepted)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.emit()
    }
}

fn import_file_limit(relative: &str) -> u64 {
    match relative {
        HF_TRL_EXPORT_MANIFEST_FILENAME | HF_TRL_RESULT_MANIFEST_FILENAME => {
            MAX_IMPORT_MANIFEST_BYTES
        }
        HF_TRL_EXECUTED_SCRIPT_FILENAME => MAX_IMPORT_SCRIPT_BYTES,
        HF_TRL_ADAPTER_CONFIG_FILENAME => MAX_IMPORT_ADAPTER_CONFIG_BYTES,
        HF_TRL_ADAPTER_MODEL_FILENAME => MAX_IMPORT_EXPANDED_BYTES,
        _ => MAX_IMPORT_AUXILIARY_BYTES,
    }
}

fn regular_file_size(root: &Path, relative: &str) -> Result<u64> {
    let path = root.join(relative);
    let metadata = fs::symlink_metadata(&path)
        .with_context(|| format!("inspecting verified import artifact {}", path.display()))?;
    ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "verified import artifact is not a regular file: {}",
        path.display()
    );
    ensure!(
        metadata.len() <= import_file_limit(relative),
        "import artifact {relative:?} is {} bytes; server limit is {} bytes",
        metadata.len(),
        import_file_limit(relative)
    );
    Ok(metadata.len())
}

fn expected_resident_identity(export: &HfTrlExportManifestV1) -> HfTrlResidentModelIdentity {
    HfTrlResidentModelIdentity {
        served_model_id: export.model.served_model_id.clone(),
        base_weight_shard_manifest: export.model.base_weight_shard_manifest.clone(),
        model_config_sha256: export.model.model_config.sha256.clone(),
        tokenizer_vocab_sha256: export.model.tokenizer_vocab_sha256.clone(),
        tokenizer_config_sha256: export.model.tokenizer.sha256.clone(),
        chat_template_sha256: export.model.chat_template.sha256.clone(),
        native_training_chat_template_sha256: export
            .model
            .native_training_chat_template
            .sha256
            .clone(),
        trl_training_chat_template_sha256: export.model.trl_training_chat_template.sha256.clone(),
    }
}

fn preflight_import_bundle(
    bundle: &Path,
) -> Result<(HfTrlExportManifestV1, HfTrlTrainingResultV1, u64)> {
    let root_metadata = fs::symlink_metadata(bundle)
        .with_context(|| format!("inspecting completed HF/TRL bundle {}", bundle.display()))?;
    ensure!(
        root_metadata.file_type().is_dir() && !root_metadata.file_type().is_symlink(),
        "completed HF/TRL bundle is not a real directory: {}",
        bundle.display()
    );
    regular_file_size(bundle, HF_TRL_EXPORT_MANIFEST_FILENAME)?;
    regular_file_size(bundle, HF_TRL_RESULT_MANIFEST_FILENAME)?;
    let export = kiln_train::read_hf_trl_export_manifest(bundle)
        .context("reading completed-bundle export manifest")?;
    let result = kiln_train::read_hf_trl_training_result(bundle)
        .context("reading completed-bundle result manifest")?;
    result
        .validate_against_export(&export)
        .context("validating completed-bundle manifest linkage")?;
    let files = hf_trl_import_envelope_files(&export);
    ensure!(
        files.len() == 10,
        "HF/TRL import envelope contract must contain exactly ten files"
    );
    let mut expanded_bytes = 0u64;
    for relative in files {
        expanded_bytes = expanded_bytes
            .checked_add(regular_file_size(bundle, &relative)?)
            .context("HF/TRL import expanded-size overflow")?;
    }
    ensure!(
        expanded_bytes <= MAX_IMPORT_EXPANDED_BYTES,
        "HF/TRL import envelope contains {expanded_bytes} bytes; server limit is {MAX_IMPORT_EXPANDED_BYTES} bytes"
    );
    Ok((export, result, expanded_bytes))
}

fn prepare_import_upload(bundle: &Path, name: &str) -> Result<PreparedImportUpload> {
    let (preflight_export, preflight_result, expanded_bytes) = preflight_import_bundle(bundle)?;
    let temporary = tempfile::Builder::new()
        .prefix("kiln-hf-import-")
        .tempdir()
        .context("creating private HF/TRL import staging directory")?;
    let envelope = temporary
        .path()
        .join(format!("{name}{HF_TRL_IMPORT_ENVELOPE_SUFFIX}"));
    let (export, result) = kiln_train::write_hf_trl_import_envelope(bundle, &envelope)
        .with_context(|| format!("verifying completed HF/TRL bundle {}", bundle.display()))?;
    ensure!(
        export == preflight_export && result == preflight_result,
        "HF/TRL completed-bundle identities changed after resource preflight"
    );

    let receipt = HfTrlImportReceiptV1::new(
        name.to_string(),
        &export,
        &result,
        expected_resident_identity(&export),
    )
    .context("deriving expected HF/TRL import receipt")?;
    let mut receipt_bytes = serde_json::to_vec_pretty(&receipt)
        .context("serializing expected HF/TRL import receipt")?;
    receipt_bytes.push(b'\n');
    let mut installed_bytes =
        u64::try_from(receipt_bytes.len()).context("import receipt size exceeds u64")?;
    for relative in IMPORTED_ADAPTER_FILES
        .into_iter()
        .filter(|relative| *relative != HF_TRL_IMPORT_RECEIPT_FILENAME)
    {
        installed_bytes = installed_bytes
            .checked_add(regular_file_size(&envelope, relative)?)
            .context("installed HF/TRL adapter size overflow")?;
    }
    let weights_sha256 = result
        .output_adapter
        .model
        .sha256
        .strip_prefix("sha256:")
        .context("verified PEFT weight identity is not sha256-prefixed")?;
    let config_sha256 = result
        .output_adapter
        .config
        .sha256
        .strip_prefix("sha256:")
        .context("verified PEFT configuration identity is not sha256-prefixed")?;
    let content_revision = LoraSourceIdentity::from_verified_peft_digests(
        result.output_adapter.model.size_bytes,
        weights_sha256,
        config_sha256,
    )
    .context("deriving expected imported adapter content revision")?
    .content_revision();

    Ok(PreparedImportUpload {
        _temporary: temporary,
        envelope,
        export,
        result,
        receipt,
        content_revision,
        expanded_bytes,
        installed_bytes,
    })
}

fn write_import_archive(
    envelope: &Path,
    root: &str,
    sender: mpsc::Sender<std::result::Result<Vec<u8>, io::Error>>,
) -> Result<u64> {
    let writer = ImportArchiveWriter::new(sender);
    let encoder = GzBuilder::new()
        .mtime(0)
        .write(writer, Compression::default());
    let mut archive = tar::Builder::new(encoder);
    let export = kiln_train::read_hf_trl_export_manifest(envelope)
        .context("reading verified import-envelope manifest")?;
    for relative in hf_trl_import_envelope_files(&export) {
        let path = envelope.join(&relative);
        let size = regular_file_size(envelope, &relative)?;
        let file = File::open(&path)
            .with_context(|| format!("opening import artifact {}", path.display()))?;
        let mut header = tar::Header::new_ustar();
        header.set_entry_type(tar::EntryType::Regular);
        header.set_mode(0o600);
        header.set_uid(0);
        header.set_gid(0);
        header.set_mtime(0);
        header.set_size(size);
        header.set_cksum();
        archive
            .append_data(&mut header, format!("{root}/{relative}"), file)
            .with_context(|| format!("archiving import artifact {relative}"))?;
    }
    let encoder = archive
        .into_inner()
        .context("finishing HF/TRL import tar stream")?;
    let writer = encoder
        .finish()
        .context("finishing HF/TRL import gzip stream")?;
    writer
        .finish()
        .context("flushing HF/TRL import upload stream")
}

// Judgment keep (round 74): the stream + JoinHandle pair is the two ends of
// one upload task; a named struct would add a type for a single call site.
#[allow(clippy::type_complexity)]
fn spawn_import_archive(
    envelope: PathBuf,
    root: String,
) -> (
    ReceiverStream<std::result::Result<Vec<u8>, io::Error>>,
    tokio::task::JoinHandle<Result<u64>>,
) {
    let (sender, receiver) = mpsc::channel(IMPORT_UPLOAD_CHANNEL_DEPTH);
    let error_sender = sender.clone();
    let worker = tokio::task::spawn_blocking(move || {
        let result = write_import_archive(&envelope, &root, sender);
        if let Err(error) = &result {
            let _ = error_sender.blocking_send(Err(io::Error::other(format!("{error:#}"))));
        }
        result
    });
    (ReceiverStream::new(receiver), worker)
}

fn validate_import_response(
    response: &ImportPeftResponse,
    prepared: &PreparedImportUpload,
    name: &str,
) -> Result<()> {
    let expected_reference_script = prepared
        .result
        .uses_exported_reference_script(&prepared.export)
        .context("checking expected reference-script identity")?;
    ensure!(
        response.status == "imported",
        "server import status is not imported"
    );
    ensure!(
        response.name == name,
        "server imported a different adapter name"
    );
    ensure!(
        response.task == prepared.result.task,
        "server import task differs from the verified result"
    );
    ensure!(
        response.export_sha256 == prepared.export.export_sha256,
        "server import export identity differs from the verified bundle"
    );
    ensure!(
        response.result_sha256 == prepared.result.result_sha256,
        "server import result identity differs from the verified bundle"
    );
    ensure!(
        response.import_sha256 == prepared.receipt.import_sha256,
        "server import receipt identity differs from the locally derived identity"
    );
    ensure!(
        response.content_revision == prepared.content_revision,
        "server adapter content revision differs from the verified PEFT bytes"
    );
    ensure!(
        response.used_exported_reference_script == expected_reference_script,
        "server import reference-script status differs from the verified result"
    );
    ensure!(
        response.size_bytes == prepared.installed_bytes,
        "server installed byte count {} differs from locally derived {}",
        response.size_bytes,
        prepared.installed_bytes
    );
    ensure!(
        response.files == IMPORTED_ADAPTER_FILES.len(),
        "server installed file count {} differs from contract value {}",
        response.files,
        IMPORTED_ADAPTER_FILES.len()
    );
    Ok(())
}

fn validate_archive_path(path: &Path, expected_root: &OsStr) -> Result<()> {
    let mut components = path.components();
    ensure!(
        matches!(components.next(), Some(Component::Normal(root)) if root == expected_root),
        "archive entry must be rooted at {}: {}",
        expected_root.to_string_lossy(),
        path.display()
    );
    ensure!(
        components.all(|component| matches!(component, Component::Normal(_))),
        "archive entry contains an unsafe path component: {}",
        path.display()
    );
    Ok(())
}

fn validate_archive_path_bytes(bytes: &[u8], expected_root: &str) -> Result<String> {
    ensure!(
        bytes.is_ascii(),
        "archive entry path must contain only ASCII bytes"
    );
    let bytes = bytes.strip_suffix(b"/").unwrap_or(bytes);
    let mut components = bytes.split(|byte| *byte == b'/');
    ensure!(
        components.next() == Some(expected_root.as_bytes()),
        "archive entry must use the exact root {expected_root}"
    );
    for component in components {
        ensure!(
            !component.is_empty() && component != b"." && component != b"..",
            "archive entry contains an empty or relative path component"
        );
        ensure!(
            !component.contains(&b'\\'),
            "archive entry contains a platform-dependent backslash"
        );
    }
    Ok(String::from_utf8(bytes.to_ascii_lowercase())
        .expect("ASCII archive path has a UTF-8 lowercase representation"))
}

pub(crate) fn verify_downloaded_archive(
    archive_path: &Path,
    output_parent: &Path,
    name: &str,
    expected_export_sha256: &str,
) -> Result<HfTrlExportManifestV1> {
    let extraction = tempfile::Builder::new()
        .prefix(".kiln-hf-verify-")
        .tempdir_in(output_parent)
        .with_context(|| {
            format!(
                "creating archive verification directory in {}",
                output_parent.display()
            )
        })?;
    let file = File::open(archive_path)
        .with_context(|| format!("opening staged archive {}", archive_path.display()))?;
    let decoder = GzDecoder::new(BufReader::new(file));
    let mut archive = tar::Archive::new(decoder);
    let expected_root = format!("{name}{HF_TRL_BUNDLE_SUFFIX}");
    let mut seen = BTreeSet::new();
    let mut portable_seen = BTreeSet::new();
    let mut entry_count = 0usize;
    let mut expanded_bytes = 0u64;
    let mut regular_files = 0usize;

    for entry in archive.entries().context("reading gzip tar entries")? {
        let mut entry = entry.context("reading gzip tar entry")?;
        entry_count = entry_count.saturating_add(1);
        ensure!(
            entry_count <= MAX_ARCHIVE_ENTRIES,
            "archive exceeds {MAX_ARCHIVE_ENTRIES} entries"
        );
        let portable_path = validate_archive_path_bytes(&entry.path_bytes(), &expected_root)?;
        let path = entry
            .path()
            .context("reading archive entry path")?
            .into_owned();
        validate_archive_path(&path, OsStr::new(&expected_root))?;
        ensure!(
            seen.insert(path.clone()),
            "archive contains a duplicate entry: {}",
            path.display()
        );
        ensure!(
            portable_seen.insert(portable_path),
            "archive contains a case-insensitive or platform path alias: {}",
            path.display()
        );

        let entry_type = entry.header().entry_type();
        if entry_type.is_file() {
            regular_files = regular_files.saturating_add(1);
            expanded_bytes = expanded_bytes
                .checked_add(
                    entry
                        .header()
                        .size()
                        .context("reading archive entry size")?,
                )
                .context("archive expanded-size overflow")?;
            ensure!(
                expanded_bytes <= MAX_EXPANDED_ARCHIVE_BYTES,
                "archive expands beyond {} bytes",
                MAX_EXPANDED_ARCHIVE_BYTES
            );
        } else {
            ensure!(
                entry_type.is_dir(),
                "archive contains a link or special entry at {}",
                path.display()
            );
        }
        ensure!(
            entry
                .unpack_in(extraction.path())
                .with_context(|| format!("extracting archive entry {}", path.display()))?,
            "archive entry escaped the verification directory: {}",
            path.display()
        );
    }
    ensure!(regular_files > 0, "archive contains no regular files");

    // tar stops at its end marker. Read the decoder to EOF as well so gzip
    // trailer/checksum failures cannot be hidden behind an otherwise valid tar.
    let mut decoder = archive.into_inner();
    io::copy(&mut decoder, &mut io::sink()).context("validating gzip trailer")?;
    let mut compressed = decoder.into_inner();
    ensure!(
        compressed
            .fill_buf()
            .context("checking for trailing archive bytes")?
            .is_empty(),
        "archive contains bytes or another gzip member after its gzip trailer"
    );

    let root = extraction.path().join(&expected_root);
    let metadata = fs::symlink_metadata(&root)
        .with_context(|| format!("archive is missing top directory {expected_root}"))?;
    ensure!(
        metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
        "archive top path is not a real directory: {expected_root}"
    );
    let manifest = kiln_train::verify_hf_trl_export_bundle(&root)
        .context("verifying downloaded pristine HF/TRL bundle")?;
    ensure!(
        manifest.export_sha256 == expected_export_sha256,
        "downloaded export identity {} differs from create response {}",
        manifest.export_sha256,
        expected_export_sha256
    );
    Ok(manifest)
}

fn sync_directory(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

async fn download_and_publish(
    client: &reqwest::Client,
    base_url: &str,
    summary: &ExportSummary,
    output: &Path,
) -> Result<DownloadedBundle> {
    ensure_available_destination(output)?;
    let parent = destination_parent(output);
    let mut staged = tempfile::Builder::new()
        .prefix(".kiln-hf-download-")
        .tempfile_in(parent)
        .with_context(|| format!("creating staged download in {}", parent.display()))?;
    let staged_path = staged.path().to_path_buf();
    let mut async_file = tokio::fs::File::from_std(
        staged
            .as_file()
            .try_clone()
            .context("cloning staged download handle")?,
    );

    let download_url = endpoint(
        base_url,
        &format!("/v1/train/hf/exports/{}/download", summary.name),
    )?;
    let mut response = client
        .get(download_url)
        .send()
        .await
        .context("requesting HF/TRL export download")?;
    let status = response.status();
    if !status.is_success() {
        let body = read_bounded_body(response, MAX_JSON_RESPONSE_BYTES).await?;
        bail!("{}", render_api_error(status, &body));
    }
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    ensure!(
        content_type
            .split(';')
            .next()
            .is_some_and(|value| value.trim().eq_ignore_ascii_case("application/gzip")),
        "download returned unexpected content type {content_type:?}"
    );
    validate_export_etag(response.headers(), &summary.export_sha256)?;
    if let Some(content_length) = response.content_length() {
        ensure!(
            content_length <= MAX_ARCHIVE_BYTES,
            "download declares {content_length} bytes; maximum archive size is {MAX_ARCHIVE_BYTES} bytes"
        );
    }

    let mut archive_hasher = Sha256::new();
    let mut archive_bytes = 0u64;
    while let Some(chunk) = response.chunk().await.context("streaming HF/TRL export")? {
        archive_bytes = archive_bytes
            .checked_add(u64::try_from(chunk.len()).context("download chunk length overflow")?)
            .context("download size overflow")?;
        ensure!(
            archive_bytes <= MAX_ARCHIVE_BYTES,
            "download exceeds maximum archive size of {MAX_ARCHIVE_BYTES} bytes"
        );
        archive_hasher.update(&chunk);
        async_file
            .write_all(&chunk)
            .await
            .context("writing staged HF/TRL export")?;
    }
    ensure!(archive_bytes > 0, "downloaded HF/TRL archive is empty");
    async_file
        .flush()
        .await
        .context("flushing staged HF/TRL export")?;
    async_file
        .sync_all()
        .await
        .context("syncing staged HF/TRL export")?;
    drop(async_file);

    let expected_sha256 = summary.export_sha256.clone();
    let name = summary.name.clone();
    let verification_parent = parent.to_path_buf();
    let manifest = tokio::task::spawn_blocking(move || {
        verify_downloaded_archive(&staged_path, &verification_parent, &name, &expected_sha256)
    })
    .await
    .context("archive verification worker panicked")??;
    validate_downloaded_manifest(summary, &manifest)?;

    let digest = archive_hasher.finalize();
    let archive_sha256 = format!("sha256:{}", hex_digest(digest.as_slice()));
    staged
        .as_file_mut()
        .sync_all()
        .context("syncing staged archive before publication")?;
    staged.persist_noclobber(output).map_err(|error| {
        anyhow::anyhow!(
            "publishing verified archive {} without overwrite: {}",
            output.display(),
            error.error
        )
    })?;
    sync_directory(parent)
        .with_context(|| format!("syncing output directory {}", parent.display()))?;

    Ok(DownloadedBundle {
        manifest,
        archive_sha256,
        archive_bytes,
    })
}

async fn delete_export(
    client: &reqwest::Client,
    base_url: &str,
    name: &str,
    expected_export_sha256: Option<&str>,
    missing_is_success: bool,
) -> Result<()> {
    let mut request = client.delete(endpoint(base_url, &format!("/v1/train/hf/exports/{name}"))?);
    if let Some(expected) = expected_export_sha256 {
        request = request.header(reqwest::header::IF_MATCH, format!("\"{expected}\""));
    }
    let response = request
        .send()
        .await
        .with_context(|| format!("deleting server HF/TRL export {name}"))?;
    let status = response.status();
    let body = read_bounded_body(response, MAX_JSON_RESPONSE_BYTES).await?;
    if missing_is_success && status == reqwest::StatusCode::NOT_FOUND {
        return Ok(());
    }
    if !status.is_success() {
        bail!("{}", render_api_error(status, &body));
    }
    let deleted: DeleteExportResponse =
        serde_json::from_slice(&body).context("decoding HF/TRL delete response")?;
    ensure!(
        deleted.status == "deleted" && deleted.name == name,
        "server returned an inconsistent HF/TRL delete receipt"
    );
    Ok(())
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn shell_quote(value: &Path) -> String {
    shell_quote_text(&value.to_string_lossy())
}

fn shell_quote_text(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn default_export_output(name: &str) -> PathBuf {
    PathBuf::from(format!("{name}{HF_TRL_BUNDLE_SUFFIX}.tar.gz"))
}

fn add_common_export_fields(
    request: &mut serde_json::Map<String, serde_json::Value>,
    input_adapter: Option<&str>,
    split_manifest: Option<serde_json::Value>,
) -> Result<()> {
    if let Some(input_adapter) = input_adapter {
        ensure!(
            !input_adapter.is_empty() && input_adapter.trim() == input_adapter,
            "input adapter must be non-blank and contain no leading or trailing whitespace"
        );
        request.insert("input_adapter".to_string(), input_adapter.into());
    }
    if let Some(split_manifest) = split_manifest {
        request.insert("split_manifest".to_string(), split_manifest);
    }
    Ok(())
}

fn validate_created_export(
    summary: &ExportSummary,
    headers: &reqwest::header::HeaderMap,
    name: &str,
    task: HfTrlTask,
) -> Result<()> {
    validate_export_sha256(&summary.export_sha256)?;
    validate_export_etag(headers, &summary.export_sha256)?;
    ensure!(
        summary.name == name
            && summary.task == task
            && summary.bundle_filename == format!("{name}{HF_TRL_BUNDLE_SUFFIX}")
            && summary.download_url == format!("/v1/train/hf/exports/{name}/download"),
        "server returned an inconsistent HF/TRL export summary"
    );
    Ok(())
}

fn validate_downloaded_manifest(
    summary: &ExportSummary,
    manifest: &HfTrlExportManifestV1,
) -> Result<()> {
    ensure!(
        manifest.task == summary.task
            && manifest.data.source_name == summary.source_name
            && manifest.data.row_count == summary.row_count
            && manifest.data.ordered_corpus_sha256 == summary.ordered_corpus_sha256
            && manifest
                .input_adapter
                .as_ref()
                .map(|adapter| adapter.name.as_str())
                == summary.input_adapter.as_deref(),
        "downloaded HF/TRL manifest contradicts its creation summary"
    );
    Ok(())
}

async fn download_verify_and_finish_export(
    client: &reqwest::Client,
    base_url: &str,
    name: &str,
    output: &Path,
    keep_server_copy: bool,
    summary: ExportSummary,
) -> Result<()> {
    println!(
        "{} Streaming and verifying {} row(s) into {}",
        style("→").cyan().bold(),
        summary.row_count,
        output.display()
    );
    let downloaded = download_and_publish(client, base_url, &summary, output)
        .await
        .with_context(|| {
            format!(
                "server export '{name}' remains available; retry the download or remove only identity {} with: kiln train hf delete --name {name} --export-sha256 {} --url {}",
                summary.export_sha256,
                summary.export_sha256,
                shell_quote_text(base_url),
            )
        })?;
    let server_copy_removed = if keep_server_copy {
        false
    } else {
        match delete_export(client, base_url, name, Some(&summary.export_sha256), true).await {
            Ok(()) => true,
            Err(error) => {
                eprintln!(
                    "{} verified local archive is safe, but server cleanup failed: {error:#}",
                    style("warning:").yellow().bold()
                );
                eprintln!(
                    "  retry safely with: kiln train hf delete --name {name} --export-sha256 {} --url {}",
                    summary.export_sha256,
                    shell_quote_text(base_url)
                );
                false
            }
        }
    };

    println!(
        "{} Published verified HF/TRL export {}",
        style("✓").green().bold(),
        output.display()
    );
    println!("export_sha256: {}", downloaded.manifest.export_sha256);
    println!("archive_sha256: {}", downloaded.archive_sha256);
    println!("archive_bytes: {}", downloaded.archive_bytes);
    println!("rows: {}", downloaded.manifest.data.row_count);
    println!(
        "server_copy: {}",
        if keep_server_copy {
            "retained"
        } else if server_copy_removed {
            "removed"
        } else {
            "cleanup required"
        }
    );

    let parent = destination_parent(output);
    let bundle = parent.join(format!("{name}{HF_TRL_BUNDLE_SUFFIX}"));
    println!("next:");
    println!(
        "  tar -xzf {} -C {}",
        shell_quote(output),
        shell_quote(parent)
    );
    println!(
        "  python {} {} --base-model /absolute/path/to/hf-model",
        shell_quote(&bundle.join("train.py")),
        shell_quote(&bundle)
    );
    Ok(())
}

pub async fn run_export_sft(options: ExportSftOptions) -> Result<()> {
    validate_export_name(&options.name)?;
    ensure!(
        matches!(options.invalid_row_policy.as_str(), "fail" | "skip"),
        "invalid row policy must be fail or skip"
    );
    let output = options
        .output
        .clone()
        .unwrap_or_else(|| default_export_output(&options.name));
    ensure_available_destination(&output)?;

    let file = options
        .file
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let dataset = options
        .dataset
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    ensure!(
        matches!((file, dataset), (Some(_), None) | (None, Some(_))),
        "exactly one non-blank --file or --dataset is required"
    );
    let split_manifest = options
        .split_manifest
        .as_deref()
        .map(read_split_manifest)
        .transpose()?;

    let mut request = serde_json::Map::new();
    request.insert("name".to_string(), options.name.clone().into());
    request.insert(
        "invalid_row_policy".to_string(),
        options.invalid_row_policy.clone().into(),
    );
    if let Some(file) = file {
        request.insert("dataset_path".to_string(), file.into());
    }
    if let Some(dataset) = dataset {
        request.insert("dataset".to_string(), dataset.into());
    }
    add_common_export_fields(
        &mut request,
        options.input_adapter.as_deref(),
        split_manifest,
    )?;

    let client = build_client()?;
    println!(
        "{} Creating immutable HF/TRL SFT export '{}'",
        style("→").cyan().bold(),
        style(&options.name).white().bold()
    );
    let response = client
        .post(endpoint(&options.url, "/v1/train/hf/sft/exports")?)
        .json(&serde_json::Value::Object(request))
        .send()
        .await
        .context("creating server HF/TRL SFT export")?;
    let response_headers = response.headers().clone();
    let summary: ExportSummary = decode_json_response(response).await?;
    validate_created_export(&summary, &response_headers, &options.name, HfTrlTask::Sft)?;
    download_verify_and_finish_export(
        &client,
        &options.url,
        &options.name,
        &output,
        options.keep_server_copy,
        summary,
    )
    .await
}

pub async fn run_export_grpo(options: ExportGrpoOptions) -> Result<()> {
    validate_export_name(&options.name)?;
    let file = options.file.trim();
    ensure!(!file.is_empty(), "a non-blank --file is required");
    let output = options
        .output
        .clone()
        .unwrap_or_else(|| default_export_output(&options.name));
    ensure_available_destination(&output)?;
    let split_manifest = options
        .split_manifest
        .as_deref()
        .map(read_split_manifest)
        .transpose()?;

    let mut request = serde_json::Map::new();
    request.insert("name".to_string(), options.name.clone().into());
    request.insert("dataset_path".to_string(), file.into());
    add_common_export_fields(
        &mut request,
        options.input_adapter.as_deref(),
        split_manifest,
    )?;

    let client = build_client()?;
    println!(
        "{} Creating immutable HF/TRL GRPO export '{}'",
        style("→").cyan().bold(),
        style(&options.name).white().bold()
    );
    let response = client
        .post(endpoint(&options.url, "/v1/train/hf/grpo/exports")?)
        .json(&serde_json::Value::Object(request))
        .send()
        .await
        .context("creating server HF/TRL GRPO export")?;
    let response_headers = response.headers().clone();
    let summary: ExportSummary = decode_json_response(response).await?;
    validate_created_export(&summary, &response_headers, &options.name, HfTrlTask::Grpo)?;
    download_verify_and_finish_export(
        &client,
        &options.url,
        &options.name,
        &output,
        options.keep_server_copy,
        summary,
    )
    .await
}

pub async fn run_import_peft(options: ImportPeftOptions) -> Result<()> {
    validate_import_name(&options.name)?;
    let import_url = endpoint(
        &options.url,
        &format!("/v1/train/hf/peft/imports/{}", options.name),
    )?;
    let client = build_client()?;

    println!(
        "{} Verifying completed HF/TRL bundle {}",
        style("→").cyan().bold(),
        options.bundle.display()
    );
    let bundle = options.bundle.clone();
    let name = options.name.clone();
    let prepared = tokio::task::spawn_blocking(move || prepare_import_upload(&bundle, &name))
        .await
        .context("HF/TRL import preparation worker panicked")??;

    println!(
        "{} Streaming {} verified envelope bytes for adapter '{}'",
        style("→").cyan().bold(),
        prepared.expanded_bytes,
        style(&options.name).white().bold()
    );
    let root = format!("{}{}", options.name, HF_TRL_IMPORT_ENVELOPE_SUFFIX);
    let (stream, producer) = spawn_import_archive(prepared.envelope.clone(), root);
    let request_result = client
        .post(import_url)
        .header(reqwest::header::CONTENT_TYPE, "application/gzip")
        .header(reqwest::header::CONTENT_ENCODING, "identity")
        .body(reqwest::Body::wrap_stream(stream))
        .send()
        .await;
    let producer_result = match producer.await {
        Ok(result) => result,
        Err(worker_error) => {
            if request_result
                .as_ref()
                .is_ok_and(|response| response.status() == reqwest::StatusCode::CREATED)
            {
                bail!(
                    "server returned HTTP 201 but the HF/TRL import archive worker failed: {worker_error}; adapter '{}' may already be installed",
                    options.name
                );
            }
            bail!("HF/TRL import archive worker failed: {worker_error}");
        }
    };

    let response = match request_result {
        Ok(response) => response,
        Err(request_error) => match producer_result {
            Ok(_) => {
                bail!(
                    "HF/TRL import request failed and its server outcome is unknown: {request_error}; inspect adapter '{}' with `kiln adapters list --url {}` before retrying",
                    options.name,
                    shell_quote_text(&options.url)
                )
            }
            Err(producer_error) => {
                bail!(
                    "HF/TRL import request and upload producer failed, so the server outcome is unknown: request={request_error}; producer={producer_error:#}; inspect adapter '{}' with `kiln adapters list --url {}` before retrying",
                    options.name,
                    shell_quote_text(&options.url)
                )
            }
        },
    };
    let status = response.status();
    let headers = response.headers().clone();
    let body = read_bounded_body(response, MAX_JSON_RESPONSE_BYTES)
        .await
        .with_context(|| {
            if status == reqwest::StatusCode::CREATED {
                format!(
                    "server returned HTTP 201 but its bounded response body could not be read; adapter '{}' may already be installed",
                    options.name
                )
            } else {
                "reading HF/TRL import response body".to_string()
            }
        })?;
    if status != reqwest::StatusCode::CREATED {
        if status.is_success() {
            bail!(
                "server returned HTTP {status} instead of HTTP 201; adapter '{}' may already be installed, so inspect `kiln adapters list --url {}` before retrying",
                options.name,
                shell_quote_text(&options.url)
            );
        }
        let api_error = render_api_error(status, &body);
        if let Err(producer_error) = producer_result {
            let producer_detail = format!("{producer_error:#}");
            if !producer_detail.contains("stopped consuming the upload") {
                bail!("{api_error}\n  upload: {producer_detail}");
            }
        }
        bail!("{api_error}");
    }
    let archive_bytes = producer_result.with_context(|| {
        format!(
            "server returned HTTP 201 but the HF/TRL import archive did not finish; adapter '{}' may already be installed",
            options.name
        )
    })?;
    ensure!(
        archive_bytes > 0,
        "HF/TRL import archive producer emitted no bytes"
    );

    validate_json_content_type(&headers).with_context(|| {
        format!(
            "server returned HTTP 201 with an invalid content type; adapter '{}' may already be installed",
            options.name
        )
    })?;
    let imported: ImportPeftResponse = serde_json::from_slice(&body).with_context(|| {
        format!(
            "server returned HTTP 201 with an invalid import receipt; adapter '{}' may already be installed",
            options.name
        )
    })?;
    validate_import_etag(&headers, &prepared.receipt.import_sha256).with_context(|| {
        format!(
            "server returned HTTP 201 but its response identity is invalid; adapter '{}' may already be installed",
            options.name
        )
    })?;
    validate_import_response(&imported, &prepared, &options.name).with_context(|| {
        format!(
            "server returned HTTP 201 but its import receipt is inconsistent; adapter '{}' may already be installed",
            options.name
        )
    })?;

    println!(
        "{} Imported verified PEFT adapter '{}'",
        style("✓").green().bold(),
        style(&options.name).white().bold()
    );
    println!("import_sha256: {}", imported.import_sha256);
    println!("export_sha256: {}", imported.export_sha256);
    println!("result_sha256: {}", imported.result_sha256);
    println!("content_revision: {}", imported.content_revision);
    println!("uploaded_archive_bytes: {archive_bytes}");
    println!("installed_bytes: {}", imported.size_bytes);
    println!("source_bundle: retained at {}", options.bundle.display());
    Ok(())
}

pub async fn run_list(base_url: &str, json: bool) -> Result<()> {
    let client = build_client()?;
    let response = client
        .get(endpoint(base_url, "/v1/train/hf/exports")?)
        .send()
        .await
        .context("listing server HF/TRL exports")?;
    let exports: ExportList = decode_json_response(response).await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&exports)?);
        return Ok(());
    }
    if exports.data.is_empty() {
        println!("No server HF/TRL exports.");
        return Ok(());
    }
    for export in exports.data {
        let task = match export.task {
            HfTrlTask::Sft => "sft",
            HfTrlTask::Grpo => "grpo",
        };
        let adapter = export.input_adapter.as_deref().unwrap_or("new adapter");
        println!(
            "{}  task={} rows={} source={} input={}\n  {}",
            style(export.name).white().bold(),
            task,
            export.row_count,
            export.source_name,
            adapter,
            export.export_sha256
        );
    }
    Ok(())
}

pub async fn run_delete(
    base_url: &str,
    name: &str,
    expected_export_sha256: Option<&str>,
) -> Result<()> {
    validate_export_name(name)?;
    if let Some(expected) = expected_export_sha256 {
        validate_export_sha256(expected)?;
    }
    let client = build_client()?;
    delete_export(&client, base_url, name, expected_export_sha256, false).await?;
    println!(
        "{} Deleted server HF/TRL export '{}'",
        style("✓").green().bold(),
        style(name).white().bold()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::Compression;
    use flate2::write::GzEncoder;

    fn write_test_archive(path: &Path, entries: &[(&str, tar::EntryType, &[u8])]) {
        let file = File::create(path).unwrap();
        let encoder = GzEncoder::new(file, Compression::default());
        let mut archive = tar::Builder::new(encoder);
        for (path, entry_type, bytes) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_entry_type(*entry_type);
            header.set_mode(0o600);
            header.set_size(u64::try_from(bytes.len()).unwrap());
            header.set_cksum();
            archive.append_data(&mut header, path, *bytes).unwrap();
        }
        archive.into_inner().unwrap().finish().unwrap();
    }

    #[test]
    fn export_name_validation_matches_server_contract() {
        for valid in ["a", "run-01", "run_01", "A9"] {
            validate_export_name(valid).unwrap();
        }
        for invalid in ["", "-run", "_run", "run.x", "run/path", "é"] {
            assert!(
                validate_export_name(invalid).is_err(),
                "accepted {invalid:?}"
            );
        }
        assert!(validate_export_name(&"a".repeat(129)).is_err());
        validate_export_sha256(&format!("sha256:{}", "a".repeat(64))).unwrap();
        assert!(validate_export_sha256(&format!("sha256:{}", "A".repeat(64))).is_err());
        assert!(validate_export_sha256("sha256:short").is_err());
    }

    #[test]
    fn import_name_validation_is_portable_and_bounded() {
        for valid in ["a", "run-01", "run_01", "run.v2", "A9"] {
            validate_import_name(valid).unwrap();
        }
        for invalid in ["", "-run", ".run", "run..v2", "run/path", "run path", "é"] {
            assert!(
                validate_import_name(invalid).is_err(),
                "accepted {invalid:?}"
            );
        }
        let boundary = "a".repeat(128);
        validate_import_name(&boundary).unwrap();
        let archive_path =
            format!("{boundary}{HF_TRL_IMPORT_ENVELOPE_SUFFIX}/kiln_training_chat_template.jinja");
        let mut header = tar::Header::new_ustar();
        header.set_path(&archive_path).unwrap();
        assert_eq!(header.path().unwrap(), Path::new(&archive_path));
        assert!(validate_import_name(&"a".repeat(129)).is_err());
    }

    #[test]
    fn import_archive_writer_chunks_and_enforces_compressed_limit() {
        let (sender, mut receiver) = mpsc::channel(IMPORT_UPLOAD_CHANNEL_DEPTH);
        let mut writer = ImportArchiveWriter::new(sender);
        let bytes = vec![7u8; IMPORT_UPLOAD_CHUNK_BYTES + 1];
        writer.write_all(&bytes).unwrap();
        assert_eq!(writer.finish().unwrap(), bytes.len() as u64);
        let first = receiver.try_recv().unwrap().unwrap();
        let second = receiver.try_recv().unwrap().unwrap();
        assert_eq!(first.len(), IMPORT_UPLOAD_CHUNK_BYTES);
        assert_eq!(second, vec![7u8]);

        let (sender, _receiver) = mpsc::channel(1);
        let mut writer = ImportArchiveWriter::new(sender);
        writer.written = MAX_IMPORT_ARCHIVE_BYTES;
        let error = writer.write_all(&[1]).unwrap_err();
        assert!(error.to_string().contains("compressed import archive"));
    }

    #[tokio::test]
    async fn import_rejects_incomplete_bundle_before_connecting() {
        let directory = tempfile::tempdir().unwrap();
        let error = run_import_peft(ImportPeftOptions {
            url: "http://127.0.0.1:1".to_string(),
            bundle: directory.path().to_path_buf(),
            name: "not-contacted".to_string(),
        })
        .await
        .unwrap_err();
        let detail = format!("{error:#}");
        assert!(
            detail.contains("inspecting verified import artifact"),
            "{detail}"
        );
        assert!(!detail.contains("connection refused"), "{detail}");
    }

    #[test]
    fn split_manifest_requires_an_object() {
        let directory = tempfile::tempdir().unwrap();
        let object = directory.path().join("object.json");
        fs::write(&object, br#"{"train":[0]}"#).unwrap();
        assert!(read_split_manifest(&object).unwrap().is_object());
        let array = directory.path().join("array.json");
        fs::write(&array, b"[]").unwrap();
        assert!(read_split_manifest(&array).is_err());
    }

    #[test]
    fn destination_check_rejects_files_and_dangling_symlinks() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("run.tar.gz");
        ensure_available_destination(&output).unwrap();
        fs::write(&output, b"existing").unwrap();
        assert!(ensure_available_destination(&output).is_err());

        #[cfg(unix)]
        {
            let dangling = directory.path().join("dangling.tar.gz");
            std::os::unix::fs::symlink("missing", &dangling).unwrap();
            assert!(ensure_available_destination(&dangling).is_err());
        }
    }

    #[test]
    fn archive_paths_require_one_exact_safe_root() {
        let root = OsStr::new("run.kiln-hf");
        validate_archive_path(Path::new("run.kiln-hf/manifest.json"), root).unwrap();
        assert!(validate_archive_path(Path::new("other/manifest.json"), root).is_err());
        assert!(validate_archive_path(Path::new("../run.kiln-hf/manifest.json"), root).is_err());
        assert!(validate_archive_path(Path::new("/run.kiln-hf/manifest.json"), root).is_err());
        assert!(validate_archive_path_bytes(b"run.kiln-hf/manifest.json", "run.kiln-hf").is_ok());
        for invalid in [
            b"run.kiln-hf//manifest.json".as_slice(),
            b"run.kiln-hf/./manifest.json",
            b"run.kiln-hf/../manifest.json",
            b"run.kiln-hf\\manifest.json",
            b"run.kiln-hf/\xff",
        ] {
            assert!(
                validate_archive_path_bytes(invalid, "run.kiln-hf").is_err(),
                "accepted raw archive path {invalid:?}"
            );
        }
    }

    #[test]
    fn endpoint_and_etag_validation_fail_closed() {
        assert!(endpoint("file:///tmp/server", "/v1/test").is_err());
        assert!(endpoint("http://localhost:8420?x=1", "/v1/test").is_err());
        assert!(endpoint("http://user:secret@localhost:8420", "/v1/test").is_err());
        assert_eq!(
            endpoint("http://localhost:8420/", "/v1/test")
                .unwrap()
                .as_str(),
            "http://localhost:8420/v1/test"
        );

        let digest = format!("sha256:{}", "a".repeat(64));
        let mut headers = reqwest::header::HeaderMap::new();
        assert!(validate_export_etag(&headers, &digest).is_err());
        headers.insert(
            reqwest::header::ETAG,
            reqwest::header::HeaderValue::from_str(&format!("\"{digest}\"")).unwrap(),
        );
        validate_export_etag(&headers, &digest).unwrap();
        assert!(validate_export_etag(&headers, &format!("sha256:{}", "b".repeat(64))).is_err());

        let mut content_headers = reqwest::header::HeaderMap::new();
        assert!(validate_json_content_type(&content_headers).is_err());
        content_headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json; charset=utf-8"),
        );
        validate_json_content_type(&content_headers).unwrap();
        content_headers.append(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json"),
        );
        assert!(validate_json_content_type(&content_headers).is_err());
    }

    #[test]
    fn archive_verifier_rejects_links_before_unpacking() {
        let directory = tempfile::tempdir().unwrap();
        let archive = directory.path().join("link.tar.gz");
        write_test_archive(
            &archive,
            &[("run.kiln-hf/link", tar::EntryType::new(b'2'), b"")],
        );
        let error = verify_downloaded_archive(&archive, directory.path(), "run", "sha256:unused")
            .unwrap_err();
        assert!(error.to_string().contains("link or special"), "{error:#}");
        assert!(!directory.path().join("run.kiln-hf").exists());
    }

    #[test]
    fn archive_verifier_rejects_duplicate_paths() {
        let directory = tempfile::tempdir().unwrap();
        let archive = directory.path().join("duplicate.tar.gz");
        write_test_archive(
            &archive,
            &[
                ("run.kiln-hf/manifest.json", tar::EntryType::Regular, b"a"),
                ("run.kiln-hf/manifest.json", tar::EntryType::Regular, b"b"),
            ],
        );
        let error = verify_downloaded_archive(&archive, directory.path(), "run", "sha256:unused")
            .unwrap_err();
        assert!(error.to_string().contains("duplicate entry"), "{error:#}");
    }

    #[test]
    fn archive_verifier_rejects_case_insensitive_aliases() {
        let directory = tempfile::tempdir().unwrap();
        let archive = directory.path().join("alias.tar.gz");
        write_test_archive(
            &archive,
            &[
                ("run.kiln-hf/manifest.json", tar::EntryType::Regular, b"a"),
                ("run.kiln-hf/MANIFEST.JSON", tar::EntryType::Regular, b"b"),
            ],
        );
        let error = verify_downloaded_archive(&archive, directory.path(), "run", "sha256:unused")
            .unwrap_err();
        assert!(
            error.to_string().contains("platform path alias"),
            "{error:#}"
        );
    }

    #[test]
    fn archive_verifier_rejects_trailing_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let archive = directory.path().join("trailing.tar.gz");
        write_test_archive(
            &archive,
            &[(
                "run.kiln-hf/manifest.json",
                tar::EntryType::Regular,
                b"not reached",
            )],
        );
        use std::io::Write as _;
        let mut file = fs::OpenOptions::new().append(true).open(&archive).unwrap();
        file.write_all(b"trailing").unwrap();
        let error = verify_downloaded_archive(&archive, directory.path(), "run", "sha256:unused")
            .unwrap_err();
        assert!(
            error.to_string().contains("after its gzip trailer"),
            "{error:#}"
        );
    }
}
