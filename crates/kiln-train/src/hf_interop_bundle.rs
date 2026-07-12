//! Atomic construction of immutable HF/TRL handoff directories.

use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use uuid::Uuid;

use crate::adapter_output::{ADAPTER_MANIFEST_FILENAME, read_adapter_manifest};
use crate::{
    HF_TRL_CHAT_TEMPLATE_FILENAME, HF_TRL_DATASET_FILENAME, HF_TRL_ENVIRONMENT_LOCK_FILENAME,
    HF_TRL_EXPORT_MANIFEST_FILENAME, HF_TRL_MODEL_CONFIG_FILENAME,
    HF_TRL_NATIVE_TRAINING_TEMPLATE_FILENAME, HF_TRL_REFERENCE_SCRIPT_FILENAME,
    HF_TRL_SFT_INGESTION_FILENAME, HF_TRL_SPLIT_MANIFEST_FILENAME, HF_TRL_TOKENIZER_FILENAME,
    HF_TRL_TRAINING_TEMPLATE_FILENAME, HfTrlDataExport, HfTrlDatasetFormat, HfTrlExportManifestV1,
    HfTrlFileIdentity, HfTrlInputAdapter, HfTrlModelIdentity, HfTrlSftSelection, HfTrlTask,
    SftPreparedDataset,
};

pub const HF_TRL_BUNDLE_SUFFIX: &str = ".kiln-hf";

/// Optional Kiln PEFT adapter to copy into an SFT handoff.
#[derive(Debug, Clone, Copy)]
pub struct HfTrlInputAdapterSource<'a> {
    pub name: &'a str,
    pub directory: &'a Path,
}

/// Already-admitted SFT data and immutable resident identities needed to
/// construct one handoff. Callers retain ownership of every input while the
/// synchronous snapshot is written.
pub struct HfTrlSftBundleInput<'a> {
    pub served_model_id: &'a str,
    pub model_config: &'a kiln_core::config::ModelConfig,
    pub tokenizer: &'a kiln_core::tokenizer::KilnTokenizer,
    pub base_weight_shard_manifest: &'a kiln_core::model_provenance::BaseWeightShardManifest,
    pub source_execution_provenance: &'a kiln_core::execution_provenance::ExecutionProvenanceV1,
    pub prepared: &'a SftPreparedDataset,
    pub reference_script: &'a [u8],
    pub environment_lock: &'a [u8],
    pub split_manifest: Option<&'a [u8]>,
    pub input_adapter: Option<HfTrlInputAdapterSource<'a>>,
}

/// Build and atomically publish one immutable SFT handoff directory.
///
/// `target` must end in `.kiln-hf` and must not already exist. The function
/// writes a sibling staging directory, fsyncs and validates the complete
/// bundle, renames it into place, and fsyncs the parent directory.
pub fn write_hf_trl_sft_bundle(
    target: &Path,
    input: HfTrlSftBundleInput<'_>,
) -> Result<HfTrlExportManifestV1> {
    let (parent, target, basename) = prepare_target(target)?;
    validate_input(&input)?;

    let staging = parent.join(format!(".{basename}.incomplete-{}", Uuid::new_v4()));
    fs::create_dir(&staging)
        .with_context(|| format!("create HF/TRL staging directory {}", staging.display()))?;
    let mut guard = StagingGuard::new(staging.clone());

    let model_config_value =
        serde_json::to_value(input.model_config).context("serialize HF/TRL model config")?;
    let model_config_bytes =
        serde_json::to_vec(&model_config_value).context("encode canonical HF/TRL model config")?;
    write_new_synced_file(
        &staging.join(HF_TRL_MODEL_CONFIG_FILENAME),
        &model_config_bytes,
    )?;
    let tokenizer_bytes = input
        .tokenizer
        .tokenizer_config_json()
        .map_err(|error| anyhow::anyhow!("serialize HF/TRL tokenizer: {error}"))?;
    write_new_synced_file(
        &staging.join(HF_TRL_TOKENIZER_FILENAME),
        tokenizer_bytes.as_bytes(),
    )?;
    write_new_synced_file(
        &staging.join(HF_TRL_CHAT_TEMPLATE_FILENAME),
        input
            .tokenizer
            .chat_template()
            .context("HF/TRL SFT export requires a configured chat template")?
            .as_bytes(),
    )?;
    write_new_synced_file(
        &staging.join(HF_TRL_NATIVE_TRAINING_TEMPLATE_FILENAME),
        input
            .tokenizer
            .training_chat_template()
            .context("HF/TRL SFT export requires a native training template")?
            .as_bytes(),
    )?;
    write_new_synced_file(
        &staging.join(HF_TRL_TRAINING_TEMPLATE_FILENAME),
        input
            .tokenizer
            .trl_training_chat_template()
            .context("HF/TRL SFT export requires a TRL training template")?
            .as_bytes(),
    )?;
    write_sft_dataset(&staging.join(HF_TRL_DATASET_FILENAME), input.prepared)?;
    write_pretty_json(
        &staging.join(HF_TRL_SFT_INGESTION_FILENAME),
        &input.prepared.ingestion,
    )?;
    write_new_synced_file(
        &staging.join(HF_TRL_REFERENCE_SCRIPT_FILENAME),
        input.reference_script,
    )?;
    write_new_synced_file(
        &staging.join(HF_TRL_ENVIRONMENT_LOCK_FILENAME),
        input.environment_lock,
    )?;
    let split_manifest = if let Some(bytes) = input.split_manifest {
        write_new_synced_file(&staging.join(HF_TRL_SPLIT_MANIFEST_FILENAME), bytes)?;
        Some(HfTrlFileIdentity::from_file(
            &staging,
            HF_TRL_SPLIT_MANIFEST_FILENAME,
        )?)
    } else {
        None
    };
    let input_adapter = input
        .input_adapter
        .map(|source| copy_input_adapter(&staging, source))
        .transpose()?;

    let model = HfTrlModelIdentity {
        served_model_id: input.served_model_id.to_string(),
        base_weight_shard_manifest: input.base_weight_shard_manifest.clone(),
        tokenizer_vocab_sha256: input.tokenizer.vocab_identity_sha256(),
        model_config: HfTrlFileIdentity::from_file(&staging, HF_TRL_MODEL_CONFIG_FILENAME)?,
        tokenizer: HfTrlFileIdentity::from_file(&staging, HF_TRL_TOKENIZER_FILENAME)?,
        chat_template: HfTrlFileIdentity::from_file(&staging, HF_TRL_CHAT_TEMPLATE_FILENAME)?,
        native_training_chat_template: HfTrlFileIdentity::from_file(
            &staging,
            HF_TRL_NATIVE_TRAINING_TEMPLATE_FILENAME,
        )?,
        trl_training_chat_template: HfTrlFileIdentity::from_file(
            &staging,
            HF_TRL_TRAINING_TEMPLATE_FILENAME,
        )?,
    };
    let selection = HfTrlSftSelection::from_ingestion(
        &input.prepared.ingestion,
        HfTrlFileIdentity::from_file(&staging, HF_TRL_SFT_INGESTION_FILENAME)?,
    )?;
    let data = HfTrlDataExport {
        source_name: input.prepared.ingestion.source.clone(),
        format: HfTrlDatasetFormat::SftMessagesJsonl,
        row_count: u64::try_from(input.prepared.examples.len())
            .context("HF/TRL SFT row count exceeds u64")?,
        ordered_corpus_sha256: input.prepared.ingestion.kept_corpus_sha256.clone(),
        dataset: HfTrlFileIdentity::from_file(&staging, HF_TRL_DATASET_FILENAME)?,
        sft_selection: Some(selection),
        rollout_provenance_schema: None,
        split_manifest,
    };
    let manifest = HfTrlExportManifestV1::new(
        HfTrlTask::Sft,
        input.source_execution_provenance.clone(),
        model,
        data,
        HfTrlFileIdentity::from_file(&staging, HF_TRL_REFERENCE_SCRIPT_FILENAME)?,
        HfTrlFileIdentity::from_file(&staging, HF_TRL_ENVIRONMENT_LOCK_FILENAME)?,
        input_adapter,
    )?;
    write_pretty_json(&staging.join(HF_TRL_EXPORT_MANIFEST_FILENAME), &manifest)?;
    ensure_exact_file_set(&staging, &manifest)?;
    manifest.verify_files(&staging)?;
    let loaded = crate::read_hf_trl_export_manifest(&staging)?;
    ensure!(
        loaded == manifest,
        "published HF/TRL manifest differs after serialization"
    );
    sync_directory_tree(&staging)?;

    fs::rename(&staging, &target).with_context(|| {
        format!(
            "publish HF/TRL bundle {} -> {}",
            staging.display(),
            target.display()
        )
    })?;
    sync_directory(&parent)?;
    guard.disarm();
    manifest.verify_files(&target)?;
    Ok(manifest)
}

fn validate_input(input: &HfTrlSftBundleInput<'_>) -> Result<()> {
    input
        .base_weight_shard_manifest
        .validate()
        .context("validate HF/TRL base-weight source")?;
    input
        .source_execution_provenance
        .validate()
        .context("validate HF/TRL source execution provenance")?;
    input
        .prepared
        .ingestion
        .validate()
        .context("validate HF/TRL SFT ingestion receipt")?;
    let (max_seq_len, max_supervised_tokens) = crate::verify_prepared_sft_examples(
        &input.prepared.examples,
        input.tokenizer,
        &input.prepared.ingestion,
    )?;
    ensure!(
        max_seq_len == input.prepared.max_seq_len
            && max_supervised_tokens == input.prepared.max_supervised_tokens,
        "HF/TRL prepared SFT sizing differs from revalidation"
    );
    ensure!(
        !input.reference_script.is_empty(),
        "HF/TRL reference script must not be empty"
    );
    let reference_script = std::str::from_utf8(input.reference_script)
        .context("HF/TRL reference script must be UTF-8 text")?;
    ensure!(
        !reference_script.contains('\0'),
        "HF/TRL reference script must not contain NUL"
    );
    ensure!(
        !input.environment_lock.is_empty(),
        "HF/TRL environment lock must not be empty"
    );
    let environment_lock = std::str::from_utf8(input.environment_lock)
        .context("HF/TRL environment lock must be UTF-8 text")?;
    ensure!(
        !environment_lock.contains('\0'),
        "HF/TRL environment lock must not contain NUL"
    );
    if let Some(split) = input.split_manifest {
        ensure!(!split.is_empty(), "HF/TRL split manifest must not be empty");
        let split = serde_json::from_slice::<serde_json::Value>(split)
            .context("parse HF/TRL split manifest as JSON")?;
        ensure!(
            split.is_object(),
            "HF/TRL split manifest must be a JSON object"
        );
    }
    let trl_template = input
        .tokenizer
        .trl_training_chat_template()
        .context("HF/TRL SFT export requires a TRL training template")?;
    ensure!(
        contains_generation_tag(trl_template, false) && contains_generation_tag(trl_template, true),
        "HF/TRL SFT export requires generation/endgeneration spans in its TRL template"
    );
    Ok(())
}

fn contains_generation_tag(template: &str, end: bool) -> bool {
    let name = if end { "endgeneration" } else { "generation" };
    [
        format!("{{% {name} %}}"),
        format!("{{%- {name} %}}"),
        format!("{{% {name} -%}}"),
        format!("{{%- {name} -%}}"),
    ]
    .iter()
    .any(|tag| template.contains(tag))
}

fn prepare_target(target: &Path) -> Result<(PathBuf, PathBuf, String)> {
    let basename = target
        .file_name()
        .and_then(|name| name.to_str())
        .context("HF/TRL target must have a UTF-8 basename")?;
    ensure!(
        basename.ends_with(HF_TRL_BUNDLE_SUFFIX)
            && basename.len() > HF_TRL_BUNDLE_SUFFIX.len()
            && !basename.chars().any(char::is_control)
            && !basename.contains('\\'),
        "HF/TRL target basename must be non-empty and end in {HF_TRL_BUNDLE_SUFFIX}"
    );
    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("create HF/TRL target parent {}", parent.display()))?;
    let parent = parent
        .canonicalize()
        .with_context(|| format!("resolve HF/TRL target parent {}", parent.display()))?;
    let target = parent.join(basename);
    match fs::symlink_metadata(&target) {
        Ok(_) => bail!("refusing to overwrite HF/TRL bundle {}", target.display()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("stat HF/TRL bundle target {}", target.display()));
        }
    }
    Ok((parent, target, basename.to_string()))
}

fn copy_input_adapter(
    staging: &Path,
    source: HfTrlInputAdapterSource<'_>,
) -> Result<HfTrlInputAdapter> {
    let root_meta = fs::symlink_metadata(source.directory).with_context(|| {
        format!(
            "stat HF/TRL input adapter directory {}",
            source.directory.display()
        )
    })?;
    ensure!(
        root_meta.file_type().is_dir() && !root_meta.file_type().is_symlink(),
        "HF/TRL input adapter source must be a real directory"
    );
    let destination = staging.join("input_adapter");
    fs::create_dir(&destination)
        .with_context(|| format!("create HF/TRL input adapter dir {}", destination.display()))?;
    copy_regular_file(
        &source.directory.join("adapter_config.json"),
        &destination.join("adapter_config.json"),
    )?;
    copy_regular_file(
        &source.directory.join("adapter_model.safetensors"),
        &destination.join("adapter_model.safetensors"),
    )?;
    let source_manifest = source.directory.join(ADAPTER_MANIFEST_FILENAME);
    let kiln_manifest = match fs::symlink_metadata(&source_manifest) {
        Ok(metadata) => {
            ensure!(
                metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
                "HF/TRL input adapter manifest must be a regular file"
            );
            let parsed = read_adapter_manifest(&source_manifest)?;
            ensure!(
                parsed.adapter_name == source.name,
                "HF/TRL input adapter name differs from its Kiln manifest"
            );
            copy_regular_file(
                &source_manifest,
                &destination.join(ADAPTER_MANIFEST_FILENAME),
            )?;
            Some(HfTrlFileIdentity::from_file(
                staging,
                "input_adapter/adapter_manifest.json",
            )?)
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "stat HF/TRL input adapter manifest {}",
                    source_manifest.display()
                )
            });
        }
    };
    sync_directory(&destination)?;
    Ok(HfTrlInputAdapter {
        name: source.name.to_string(),
        config: HfTrlFileIdentity::from_file(staging, "input_adapter/adapter_config.json")?,
        model: HfTrlFileIdentity::from_file(staging, "input_adapter/adapter_model.safetensors")?,
        kiln_manifest,
    })
}

fn copy_regular_file(source: &Path, target: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(source)
        .with_context(|| format!("stat HF/TRL source artifact {}", source.display()))?;
    ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "HF/TRL source artifact must be a regular file: {}",
        source.display()
    );
    let mut reader = File::open(source)
        .with_context(|| format!("open HF/TRL source artifact {}", source.display()))?;
    let mut writer = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(target)
        .with_context(|| format!("create HF/TRL artifact {}", target.display()))?;
    io::copy(&mut reader, &mut writer).with_context(|| {
        format!(
            "copy HF/TRL artifact {} -> {}",
            source.display(),
            target.display()
        )
    })?;
    writer
        .sync_all()
        .with_context(|| format!("sync HF/TRL artifact {}", target.display()))
}

fn write_sft_dataset(path: &Path, prepared: &SftPreparedDataset) -> Result<()> {
    let mut file = create_new_file(path)?;
    for (index, example) in prepared.examples.iter().enumerate() {
        serde_json::to_writer(&mut file, example)
            .with_context(|| format!("serialize HF/TRL SFT row {}", index + 1))?;
        file.write_all(b"\n")
            .with_context(|| format!("write HF/TRL SFT row {}", index + 1))?;
    }
    file.sync_all()
        .with_context(|| format!("sync HF/TRL dataset {}", path.display()))
}

fn write_pretty_json<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)
        .with_context(|| format!("serialize HF/TRL JSON artifact {}", path.display()))?;
    write_new_synced_file(path, &bytes)
}

fn write_new_synced_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = create_new_file(path)?;
    file.write_all(bytes)
        .with_context(|| format!("write HF/TRL artifact {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("sync HF/TRL artifact {}", path.display()))
}

fn create_new_file(path: &Path) -> Result<File> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create HF/TRL artifact parent {}", parent.display()))?;
    }
    OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .with_context(|| format!("create HF/TRL artifact {}", path.display()))
}

fn ensure_exact_file_set(root: &Path, manifest: &HfTrlExportManifestV1) -> Result<()> {
    let mut expected = BTreeSet::from([
        HF_TRL_EXPORT_MANIFEST_FILENAME.to_string(),
        manifest.model.model_config.relative_path.clone(),
        manifest.model.tokenizer.relative_path.clone(),
        manifest.model.chat_template.relative_path.clone(),
        manifest
            .model
            .native_training_chat_template
            .relative_path
            .clone(),
        manifest
            .model
            .trl_training_chat_template
            .relative_path
            .clone(),
        manifest.data.dataset.relative_path.clone(),
        manifest.reference_script.relative_path.clone(),
        manifest.environment_lock.relative_path.clone(),
    ]);
    if let Some(selection) = manifest.data.sft_selection.as_ref() {
        expected.insert(selection.ingestion_receipt.relative_path.clone());
    }
    if let Some(split) = manifest.data.split_manifest.as_ref() {
        expected.insert(split.relative_path.clone());
    }
    if let Some(adapter) = manifest.input_adapter.as_ref() {
        expected.insert(adapter.config.relative_path.clone());
        expected.insert(adapter.model.relative_path.clone());
        if let Some(kiln_manifest) = adapter.kiln_manifest.as_ref() {
            expected.insert(kiln_manifest.relative_path.clone());
        }
    }
    let actual = collect_relative_files(root)?;
    ensure!(
        actual == expected,
        "HF/TRL staging file set differs: expected {expected:?}, found {actual:?}"
    );
    Ok(())
}

fn collect_relative_files(root: &Path) -> Result<BTreeSet<String>> {
    fn visit(root: &Path, directory: &Path, files: &mut BTreeSet<String>) -> Result<()> {
        for entry in fs::read_dir(directory)
            .with_context(|| format!("read HF/TRL directory {}", directory.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)?;
            ensure!(
                !metadata.file_type().is_symlink(),
                "HF/TRL staging tree contains a symlink: {}",
                path.display()
            );
            if metadata.file_type().is_dir() {
                visit(root, &path, files)?;
            } else if metadata.file_type().is_file() {
                let relative = path
                    .strip_prefix(root)
                    .expect("visited path must remain below root")
                    .to_str()
                    .context("HF/TRL artifact path must be UTF-8")?
                    .replace(std::path::MAIN_SEPARATOR, "/");
                ensure!(
                    files.insert(relative.clone()),
                    "duplicate HF/TRL artifact path {relative:?}"
                );
            } else {
                bail!("unsupported HF/TRL staging entry {}", path.display());
            }
        }
        Ok(())
    }

    let mut files = BTreeSet::new();
    visit(root, root, &mut files)?;
    Ok(files)
}

fn sync_directory_tree(root: &Path) -> Result<()> {
    let mut directories = vec![root.to_path_buf()];
    let mut cursor = 0;
    while cursor < directories.len() {
        let current = directories[cursor].clone();
        cursor += 1;
        for entry in fs::read_dir(&current)
            .with_context(|| format!("read HF/TRL directory {} for sync", current.display()))?
        {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                directories.push(entry.path());
            }
        }
    }
    for directory in directories.into_iter().rev() {
        sync_directory(&directory)?;
    }
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .with_context(|| format!("open HF/TRL directory {} for sync", path.display()))?
        .sync_all()
        .with_context(|| format!("sync HF/TRL directory {}", path.display()))
}

struct StagingGuard {
    path: PathBuf,
    armed: bool,
}

impl StagingGuard {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for StagingGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use kiln_core::config::ModelConfig;
    use kiln_core::execution_provenance::{
        ExecutionBackendIdentity, ExecutionBuildIdentity, ExecutionConfigurationIdentity,
        ExecutionKernelIdentity, ExecutionModelIdentity, ExecutionPrecisionIdentity,
        ExecutionProvenanceV1,
    };
    use kiln_core::model_provenance::{BaseWeightShardIdentity, BaseWeightShardManifest};
    use kiln_core::tokenizer::KilnTokenizer;

    use super::*;
    use crate::{ChatMessage, SftExample, SftInvalidRowPolicy, prepare_sft_examples};

    struct Fixture {
        model_config: ModelConfig,
        tokenizer: KilnTokenizer,
        base_weights: BaseWeightShardManifest,
        provenance: ExecutionProvenanceV1,
        prepared: SftPreparedDataset,
    }

    impl Fixture {
        fn input(&self) -> HfTrlSftBundleInput<'_> {
            HfTrlSftBundleInput {
                served_model_id: "test/qwen",
                model_config: &self.model_config,
                tokenizer: &self.tokenizer,
                base_weight_shard_manifest: &self.base_weights,
                source_execution_provenance: &self.provenance,
                prepared: &self.prepared,
                reference_script: b"print('train')\n",
                environment_lock: b"transformers==5.13.1\ntrl==1.8.0\n",
                split_manifest: Some(b"{\"schema\":\"test.split.v1\"}"),
                input_adapter: None,
            }
        }
    }

    fn fixture() -> Fixture {
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
            include_str!("../../kiln-core/test_fixtures/qwen35_4b_chat_template.jinja").to_string(),
        );
        let prepared = prepare_sft_examples(
            [SftExample {
                messages: vec![
                    ChatMessage::new("user", "a"),
                    ChatMessage::new("assistant", "b"),
                ],
            }],
            &tokenizer,
            SftInvalidRowPolicy::Fail,
            "inline",
            None,
        )
        .unwrap();
        let model_config = ModelConfig::qwen3_5_4b();
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
        let provenance = ExecutionProvenanceV1::new(
            ExecutionBackendIdentity {
                name: "rocm".to_string(),
                device: "test-gpu".to_string(),
                numerical_runtime_sha256: hash(b"runtime"),
            },
            ExecutionBuildIdentity {
                package_version: "0.4.1".to_string(),
                target: "test-target".to_string(),
                executable_sha256: hash(b"executable"),
                git_commit: Some("a".repeat(40)),
                source_tree_sha256: Some(hash(b"source")),
                source_dirty: Some(false),
            },
            ExecutionModelIdentity {
                model_config_sha256: kiln_core::config_hashes::sha256_json_serializable(
                    &model_config,
                )
                .unwrap(),
                tokenizer_vocab_sha256: tokenizer.vocab_identity_sha256(),
                tokenizer_config_sha256: tokenizer.tokenizer_config_sha256().unwrap(),
                chat_template_sha256: tokenizer.chat_template_sha256(),
                training_chat_template_sha256: tokenizer.training_chat_template_sha256(),
            },
            ExecutionPrecisionIdentity {
                inference_dtype: "bf16".to_string(),
                training_policy: "bf16".to_string(),
            },
            ExecutionKernelIdentity::new(
                BTreeMap::from([("test_kernel".to_string(), "v1".to_string())]),
                vec!["rocm".to_string()],
            )
            .unwrap(),
            ExecutionConfigurationIdentity {
                effective_server_config_sha256: hash(b"server-config"),
                effective_environment_sha256: hash(b"environment"),
            },
        )
        .unwrap();
        Fixture {
            model_config,
            tokenizer,
            base_weights,
            provenance,
            prepared,
        }
    }

    fn incomplete_entries(parent: &Path) -> Vec<String> {
        fs::read_dir(parent)
            .unwrap()
            .filter_map(|entry| {
                let name = entry.ok()?.file_name().into_string().ok()?;
                name.contains(".incomplete-").then_some(name)
            })
            .collect()
    }

    #[test]
    fn publishes_a_revalidated_immutable_sft_bundle() {
        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("export.kiln-hf");
        let fixture = fixture();
        let manifest = write_hf_trl_sft_bundle(&target, fixture.input()).unwrap();

        assert_eq!(manifest.task, HfTrlTask::Sft);
        assert_eq!(manifest.data.row_count, 1);
        assert!(target.is_dir());
        assert!(incomplete_entries(directory.path()).is_empty());
        assert_eq!(
            crate::read_hf_trl_export_manifest(&target).unwrap(),
            manifest
        );
        manifest.verify_files(&target).unwrap();
        let dataset = fs::read_to_string(target.join(HF_TRL_DATASET_FILENAME)).unwrap();
        assert_eq!(dataset.lines().count(), 1);
        assert_eq!(
            serde_json::from_str::<SftExample>(dataset.trim()).unwrap(),
            fixture.prepared.examples[0]
        );

        let error = write_hf_trl_sft_bundle(&target, fixture.input()).unwrap_err();
        assert!(error.to_string().contains("overwrite"), "{error:#}");

        fs::write(target.join(HF_TRL_DATASET_FILENAME), b"tampered\n").unwrap();
        let error = manifest.verify_files(&target).unwrap_err();
        assert!(error.to_string().contains("differs"), "{error:#}");
    }

    #[test]
    fn input_drift_fails_without_publishing_or_leaving_staging() {
        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("drift.kiln-hf");
        let mut fixture = fixture();
        fixture.prepared.examples[0].messages[1].content = "changed".to_string();

        let error = write_hf_trl_sft_bundle(&target, fixture.input()).unwrap_err();
        assert!(error.to_string().contains("hash differs"), "{error:#}");
        assert!(!target.exists());
        assert!(incomplete_entries(directory.path()).is_empty());
    }

    #[test]
    fn snapshots_optional_input_adapter_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let adapter = directory.path().join("adapter");
        fs::create_dir(&adapter).unwrap();
        fs::write(
            adapter.join("adapter_config.json"),
            b"{\"peft_type\":\"LORA\"}",
        )
        .unwrap();
        fs::write(adapter.join("adapter_model.safetensors"), b"adapter").unwrap();
        let target = directory.path().join("adapter-export.kiln-hf");
        let fixture = fixture();
        let mut input = fixture.input();
        input.split_manifest = None;
        input.input_adapter = Some(HfTrlInputAdapterSource {
            name: "adapter",
            directory: &adapter,
        });

        let manifest = write_hf_trl_sft_bundle(&target, input).unwrap();
        let exported = manifest.input_adapter.as_ref().unwrap();
        assert_eq!(exported.name, "adapter");
        assert!(exported.kiln_manifest.is_none());
        assert_eq!(
            fs::read(target.join(&exported.config.relative_path)).unwrap(),
            b"{\"peft_type\":\"LORA\"}"
        );
        assert_eq!(
            fs::read(target.join(&exported.model.relative_path)).unwrap(),
            b"adapter"
        );
        manifest.verify_files(&target).unwrap();
    }

    #[test]
    fn rejects_bad_suffix_split_and_symlinked_adapter_artifact() {
        let directory = tempfile::tempdir().unwrap();
        let fixture = fixture();
        assert!(
            write_hf_trl_sft_bundle(&directory.path().join("export"), fixture.input()).is_err()
        );

        let target = directory.path().join("split.kiln-hf");
        let mut bad_split = fixture.input();
        bad_split.split_manifest = Some(b"not-json");
        assert!(write_hf_trl_sft_bundle(&target, bad_split).is_err());
        assert!(!target.exists());
        assert!(incomplete_entries(directory.path()).is_empty());

        #[cfg(unix)]
        {
            let adapter = directory.path().join("adapter");
            fs::create_dir(&adapter).unwrap();
            fs::write(adapter.join("real-config.json"), b"{}").unwrap();
            std::os::unix::fs::symlink(
                adapter.join("real-config.json"),
                adapter.join("adapter_config.json"),
            )
            .unwrap();
            fs::write(adapter.join("adapter_model.safetensors"), b"adapter").unwrap();
            let mut with_adapter = fixture.input();
            with_adapter.split_manifest = None;
            with_adapter.input_adapter = Some(HfTrlInputAdapterSource {
                name: "adapter",
                directory: &adapter,
            });
            let error = write_hf_trl_sft_bundle(&target, with_adapter).unwrap_err();
            assert!(error.to_string().contains("regular file"), "{error:#}");
            assert!(!target.exists());
            assert!(incomplete_entries(directory.path()).is_empty());
        }
    }
}
